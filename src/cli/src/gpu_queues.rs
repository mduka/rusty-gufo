//! The device-wide budget of hardware compute queues
//! (`src/core/diagnostics/gpu_queues.cpp`): counts the queues every process
//! holds and chooses the `GPU_MAX_HW_QUEUES` a server asks for.

use std::path::Path;

/// Hardware compute queues the gfx1151 command processor keeps mapped
/// without time-slicing. Past this total, counted across every process on
/// the device, the firmware cycles the queues even when all are empty: the
/// GPU reports a busy engine at its top shader clock and draws about 26 W
/// above idle until one of the processes exits.
pub const COMPUTE_QUEUE_BUDGET: u32 = 8;
/// Compute queues any process holds once it dispatches at all: the first
/// dispatch claims two whatever the cap, so a device fits four HIP
/// processes at most.
pub const MINIMUM_PROCESS_QUEUES: u32 = 2;
/// Queues the HIP runtime pools by default; decides whether an unmeasured
/// modality has room to keep the runtime's own default.
pub const RUNTIME_DEFAULT_CAP: u32 = 4;

/// Compute and SDMA queues open across every process on the device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueueCensus {
    pub compute_queues: u32,
    pub sdma_queues: u32,
    pub processes: u32,
}

/// Counts the queues every process holds, including other users'
/// processes: the KFD nodes are world-readable, so a llama.cpp server shows
/// up too. Reads no HIP state; safe before the runtime starts.
pub fn query_queue_census(sys_root: &Path) -> QueueCensus {
    let mut census = QueueCensus::default();
    let Ok(processes) = std::fs::read_dir(sys_root.join("class/kfd/kfd/proc")) else {
        return census;
    };
    for process in processes.flatten() {
        let Ok(queues) = std::fs::read_dir(process.path().join("queues")) else {
            continue;
        };
        let mut counted = false;
        for queue in queues.flatten() {
            // KFD writes a bare digit, without a newline.
            match std::fs::read_to_string(queue.path().join("type"))
                .unwrap_or_default()
                .trim_end()
            {
                "0" => census.compute_queues += 1,
                "1" => census.sdma_queues += 1,
                _ => continue,
            }
            counted = true;
        }
        census.processes += u32::from(counted);
    }
    census
}

/// A server's modality, which decides how many queues to ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueProfile {
    /// `serve llm`.
    Text,
    /// `serve tts` and `serve asr`: stages run in sequence, each ends in a
    /// stream synchronise, so extra queues have nothing to overlap.
    Audio,
    /// `serve image` and `serve video`: never measured, so the runtime
    /// default stands unless the budget is too tight for it.
    Unmeasured,
}

impl QueueProfile {
    fn preferred_cap(self) -> u32 {
        match self {
            Self::Text => 2,
            Self::Audio => 1,
            Self::Unmeasured => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueuePlan {
    /// Compute queues already held when this process started.
    pub observed_queues: u32,
    /// The `GPU_MAX_HW_QUEUES` to request; zero leaves it alone.
    pub cap: u32,
    /// Upper bound on the queues this process adds.
    pub expected_queues: u32,
    /// The operator set `GPU_MAX_HW_QUEUES`, which is never overridden.
    pub operator_supplied: bool,
    /// The operator's value when it parses as a queue count, else zero.
    pub operator_cap: u32,
    /// Even the two-queue floor does not fit: the device will idle busy.
    pub exceeds_budget: bool,
    /// The upper bound does not fit, so the device may idle busy.
    pub may_exceed_budget: bool,
}

/// The cap for `profile`, lowered to what the device has left but never
/// raised to fill free slots, so a server's throughput does not depend on
/// the order servers started (`PlanQueues`).
pub fn plan_queues(
    profile: QueueProfile,
    census: &QueueCensus,
    operator_value: Option<&str>,
) -> QueuePlan {
    let mut plan = QueuePlan {
        observed_queues: census.compute_queues,
        ..QueuePlan::default()
    };
    if let Some(value) = operator_value.filter(|v| !v.is_empty()) {
        // An operator's cap of N resolves to at most N + 1 queues, enough to
        // tell them when it will not fit.
        plan.operator_supplied = true;
        match value.parse::<u32>() {
            Ok(parsed) if parsed > 0 && !value.starts_with('+') => {
                plan.operator_cap = parsed;
                plan.expected_queues = parsed + 1;
            }
            _ => plan.expected_queues = MINIMUM_PROCESS_QUEUES,
        }
        plan.exceeds_budget = census.compute_queues + MINIMUM_PROCESS_QUEUES > COMPUTE_QUEUE_BUDGET;
        plan.may_exceed_budget =
            census.compute_queues + plan.expected_queues > COMPUTE_QUEUE_BUDGET;
        return plan;
    }
    let room = COMPUTE_QUEUE_BUDGET.saturating_sub(census.compute_queues);
    // A cap of N resolves to at most N + 1 queues, so N + 1 has to fit.
    let fits = if room > MINIMUM_PROCESS_QUEUES {
        room - 1
    } else {
        1
    };
    let preferred = profile.preferred_cap();
    if preferred == 0 {
        if room > RUNTIME_DEFAULT_CAP {
            plan.expected_queues = RUNTIME_DEFAULT_CAP + 1;
            return plan;
        }
        plan.cap = fits;
    } else {
        plan.cap = preferred.min(fits);
    }
    plan.cap = plan.cap.max(1);
    plan.expected_queues = plan.cap + 1;
    // The cap was chosen to fit, so only the floor can fail.
    plan.exceeds_budget = census.compute_queues + plan.expected_queues > COMPUTE_QUEUE_BUDGET;
    plan.may_exceed_budget = plan.exceeds_budget;
    plan
}

/// The `queue_budget` log line.
pub fn describe_queue_plan(kind: &str, plan: &QueuePlan) -> String {
    let cap = if plan.operator_supplied && plan.operator_cap == 0 {
        return format!(
            "event=queue_budget kind={kind} observed={} budget={COMPUTE_QUEUE_BUDGET} \
             expected_max=unknown cap=operator",
            plan.observed_queues
        );
    } else if plan.operator_supplied {
        format!("operator:{}", plan.operator_cap)
    } else if plan.cap == 0 {
        "runtime".to_owned()
    } else {
        plan.cap.to_string()
    };
    format!(
        "event=queue_budget kind={kind} observed={} budget={COMPUTE_QUEUE_BUDGET} \
         expected_max={} cap={cap}",
        plan.observed_queues, plan.expected_queues
    )
}

/// The `queue_budget_exceeded` warning.
pub fn describe_queue_pressure(plan: &QueuePlan) -> String {
    let adds = if plan.exceeds_budget {
        // Even the floor every process pays does not fit: certain.
        format!("adds_at_least={MINIMUM_PROCESS_QUEUES}")
    } else {
        format!(
            "adds_up_to={} cap=operator:{}",
            plan.expected_queues, plan.operator_cap
        )
    };
    format!(
        "event=queue_budget_exceeded observed={} budget={COMPUTE_QUEUE_BUDGET} {adds} \
         detail=the GPU reports a busy engine at its top clock while idle, about 26 W above \
         idle, until a process holding queues exits",
        plan.observed_queues
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The KFD layout: a directory per process and per queue, each queue
    /// with a `type` file holding a bare digit.
    fn fake_kfd(processes: &[(u32, u32, u32)]) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("temporary directory");
        for &(pid, compute, sdma) in processes {
            let queues = root.path().join(format!("class/kfd/kfd/proc/{pid}/queues"));
            for index in 0..compute + sdma {
                let queue = queues.join(index.to_string());
                std::fs::create_dir_all(&queue).expect("queue");
                std::fs::write(queue.join("type"), if index < compute { "0" } else { "1" })
                    .expect("type");
            }
        }
        root
    }

    fn census(compute_queues: u32) -> QueueCensus {
        QueueCensus {
            compute_queues,
            ..QueueCensus::default()
        }
    }

    #[test]
    fn census_counts_queues_across_processes() {
        let kfd = fake_kfd(&[(101, 5, 1), (202, 2, 1)]);
        assert_eq!(
            query_queue_census(kfd.path()),
            QueueCensus {
                compute_queues: 7,
                sdma_queues: 2,
                processes: 2
            }
        );
        let absent = tempfile::tempdir().expect("temporary directory");
        assert_eq!(query_queue_census(absent.path()), QueueCensus::default());
    }

    #[test]
    fn profiles_pick_their_measured_cap_clamped_to_free_slots() {
        let empty = QueueCensus::default();
        let text = plan_queues(QueueProfile::Text, &empty, None);
        assert!(text.cap == 2 && text.expected_queues == 3 && !text.exceeds_budget);
        let audio = plan_queues(QueueProfile::Audio, &empty, None);
        assert!(
            audio.cap == 1 && audio.expected_queues == 2,
            "never raised to fill slots"
        );
        let unmeasured = plan_queues(QueueProfile::Unmeasured, &empty, None);
        assert!(
            unmeasured.cap == 0 && !unmeasured.exceeds_budget,
            "runtime default"
        );
        let clamped = plan_queues(QueueProfile::Unmeasured, &census(5), None);
        assert!(clamped.cap == 2 && clamped.observed_queues + clamped.expected_queues <= 8);
        let beside_text = plan_queues(QueueProfile::Audio, &census(5), None);
        assert!(beside_text.cap == 1 && !beside_text.exceeds_budget);
        let tight = plan_queues(QueueProfile::Text, &census(6), None);
        assert!(tight.cap == 1 && tight.expected_queues == 2 && !tight.exceeds_budget);
        let full = plan_queues(QueueProfile::Audio, &census(COMPUTE_QUEUE_BUDGET), None);
        assert!(full.cap == 1 && full.expected_queues == 2 && full.exceeds_budget);
        let chosen = plan_queues(QueueProfile::Audio, &census(5), None);
        assert!(!chosen.may_exceed_budget && chosen.exceeds_budget == chosen.may_exceed_budget);
    }

    #[test]
    fn operator_values_are_kept_and_checked() {
        let plan = plan_queues(QueueProfile::Text, &census(3), Some("4"));
        assert!(plan.operator_supplied && plan.cap == 0 && plan.operator_cap == 4);
        assert!(!plan.may_exceed_budget, "three queues plus five fits");
        let pressured = plan_queues(QueueProfile::Text, &census(7), Some("4"));
        assert!(pressured.operator_supplied && pressured.exceeds_budget);
        // The floor alone would fit beside five queues; the operator's cap
        // of four (five queues) does not.
        let plan = plan_queues(QueueProfile::Audio, &census(5), Some("4"));
        assert!(plan.expected_queues == 5 && !plan.exceeds_budget && plan.may_exceed_budget);
        let line = describe_queue_pressure(&plan);
        assert!(line.contains("adds_up_to=5") && line.contains("cap=operator:4"));
        let unparsed = plan_queues(QueueProfile::Audio, &census(5), Some("yes"));
        assert!(unparsed.operator_supplied && unparsed.operator_cap == 0);
        assert!(unparsed.expected_queues == 2 && !unparsed.may_exceed_budget);
        assert!(describe_queue_plan("tts", &unparsed).contains("expected_max=unknown"));
        let empty = plan_queues(QueueProfile::Text, &QueueCensus::default(), Some(""));
        assert!(
            !empty.operator_supplied && empty.cap == 2,
            "an empty value is ignored"
        );
    }

    #[test]
    fn plans_are_described_for_the_log() {
        let plan = plan_queues(QueueProfile::Text, &QueueCensus::default(), None);
        assert_eq!(
            describe_queue_plan("llm", &plan),
            "event=queue_budget kind=llm observed=0 budget=8 expected_max=3 cap=2"
        );
        let operator = plan_queues(QueueProfile::Text, &QueueCensus::default(), Some("4"));
        let line = describe_queue_plan("llm", &operator);
        assert!(line.contains("cap=operator:4") && line.contains("expected_max=5"));
    }
}
