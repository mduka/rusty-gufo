//! The per-request DSpark controller state (`ds4_dspark_request_state`),
//! which snapshots persist for an exact continuation.

use crate::dspark_model::{MAX_BLOCK, RETRY_TOKENS};

/// Statistics and policy of the speculative cycles of one request. A new
/// request that reuses the prefix resets it; sessions without a support
/// model keep it in the plain state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestState {
    pub drafted: u64,
    pub accepted: u64,
    pub support_drafted: u64,
    pub support_accepted: u64,
    pub positional_accepted: u64,
    pub anchors: u64,
    pub full_blocks: u64,
    pub steps: u64,
    pub skipped: u64,
    pub policy_start_drafted: u64,
    pub policy_start_accepted: u64,
    pub policy_start_steps: u64,
    pub skip_remaining: u32,
    pub concurrent_width: u32,
    pub policy_concurrency: u32,
    pub probe_cycles: u32,
    pub probe_emitted: u32,
    pub probe_cost: u32,
    pub plain_only: bool,
    pub force_plain_request: bool,
    /// The sampled token the last cycle rejected, the next anchor; never
    /// persisted, a restored prefix starts a new request.
    pub pending_anchor: i32,
}

impl Default for RequestState {
    fn default() -> Self {
        Self {
            drafted: 0,
            accepted: 0,
            support_drafted: 0,
            support_accepted: 0,
            positional_accepted: 0,
            anchors: 0,
            full_blocks: 0,
            steps: 0,
            skipped: 0,
            policy_start_drafted: 0,
            policy_start_accepted: 0,
            policy_start_steps: 0,
            skip_remaining: 0,
            concurrent_width: 3,
            policy_concurrency: 1,
            probe_cycles: 0,
            probe_emitted: 0,
            probe_cost: 0,
            plain_only: false,
            force_plain_request: false,
            pending_anchor: -1,
        }
    }
}

impl RequestState {
    /// The counters a snapshot stores, in payload order.
    pub(crate) fn counters(&self) -> [u64; 12] {
        [
            self.drafted,
            self.accepted,
            self.support_drafted,
            self.support_accepted,
            self.positional_accepted,
            self.anchors,
            self.full_blocks,
            self.steps,
            self.skipped,
            self.policy_start_drafted,
            self.policy_start_accepted,
            self.policy_start_steps,
        ]
    }

    /// The flags a snapshot stores, in payload order.
    pub(crate) fn flags(&self) -> [u32; 8] {
        [
            self.skip_remaining,
            self.concurrent_width,
            u32::from(self.plain_only),
            u32::from(self.force_plain_request),
            self.policy_concurrency,
            self.probe_cycles,
            self.probe_emitted,
            self.probe_cost,
        ]
    }

    /// The state a snapshot stored, if consistent
    /// (`payload_read_dspark_state`).
    pub(crate) fn from_payload(counters: [u64; 12], flags: [u32; 8]) -> Option<Self> {
        let [
            drafted,
            accepted,
            support_drafted,
            support_accepted,
            positional_accepted,
            anchors,
            full_blocks,
            steps,
            skipped,
            policy_start_drafted,
            policy_start_accepted,
            policy_start_steps,
        ] = counters;
        let consistent = flags[0] <= RETRY_TOKENS
            && flags[1] != 0
            && flags[1] <= MAX_BLOCK
            && flags[2] <= 1
            && flags[3] <= flags[2]
            && accepted <= drafted
            && support_accepted <= support_drafted
            && positional_accepted <= support_drafted
            && full_blocks <= steps
            && anchors == steps
            && policy_start_drafted <= support_drafted
            && policy_start_accepted <= support_accepted
            && policy_start_steps <= steps
            && (1..=8).contains(&flags[4])
            && flags[5] <= 3
            && u64::from(flags[6]) <= u64::from(flags[5]) * u64::from(MAX_BLOCK + 1)
            && u64::from(flags[7]) <= u64::from(flags[5]) * 1000;
        consistent.then_some(Self {
            drafted,
            accepted,
            support_drafted,
            support_accepted,
            positional_accepted,
            anchors,
            full_blocks,
            steps,
            skipped,
            policy_start_drafted,
            policy_start_accepted,
            policy_start_steps,
            skip_remaining: flags[0],
            concurrent_width: flags[1],
            plain_only: flags[2] != 0,
            force_plain_request: flags[3] != 0,
            policy_concurrency: flags[4],
            probe_cycles: flags[5],
            probe_emitted: flags[6],
            probe_cost: flags[7],
            pending_anchor: -1,
        })
    }
}

/// `ds4_dspark_cycle_cost`: a speculative cycle's cost with `tail` drafted
/// tokens at context `depth` among `concurrency` requests, in hundredths of
/// an ordinary batch decode step, calibrated offline on gfx1151 (see
/// docs/models/deepseek-v4-flash/artifacts/cost-calibration.json); wall
/// clocks never steer token decisions.
pub fn cycle_cost(depth: u32, tail: u32, concurrency: usize) -> u32 {
    type Costs = [u32; 3];
    const SHORT: [Costs; 4] = [
        [165, 185, 215],
        [195, 260, 430],
        [225, 275, 375],
        [255, 330, 445],
    ];
    const MEDIUM: [Costs; 4] = [
        [160, 190, 220],
        [185, 245, 295],
        [215, 275, 360],
        [230, 305, 410],
    ];
    const LONG: [Costs; 4] = [
        [160, 190, 225],
        [185, 240, 300],
        [210, 270, 370],
        [220, 305, 390],
    ];
    let group = (concurrency.saturating_sub(1) / 2).min(3);
    let column = tail.clamp(1, 3) as usize - 1;
    let deep = depth > 4096;
    let span = if deep { 12288 } else { 3968 };
    let weight = if deep {
        (depth - 4096).min(span)
    } else {
        depth.saturating_sub(128)
    };
    let low = if deep { MEDIUM } else { SHORT }[group][column];
    let high = if deep { LONG } else { MEDIUM }[group][column];
    (low * (span - weight) + high * weight) / span
}

/// `ds4_dspark_confidence_policy`: whether a sampled block keeps drafting.
/// The current position's confidence is inspected before it is sampled and
/// stands in for the later positions too; an earlier proposal is never
/// dropped for a later one.
#[derive(Debug, Clone, Copy)]
pub struct ConfidencePolicy {
    depth: u32,
    concurrency: usize,
    maximum: u32,
    included: u32,
    survival: f64,
    /// The target-known anchor is always emitted.
    expected: f64,
}

impl ConfidencePolicy {
    pub fn new(depth: u32, concurrency: usize, maximum: u32) -> Self {
        Self {
            depth,
            concurrency,
            maximum,
            included: 0,
            survival: 1.0,
            expected: 1.0,
        }
    }

    fn cost(&self, width: u32) -> u32 {
        cycle_cost(self.depth, width, self.concurrency)
    }

    /// `Include`: drafts the next position when some wider block repays its
    /// cost under this confidence.
    pub fn include(&mut self, confidence: f32) -> bool {
        if !confidence.is_finite()
            || !(0.0..=1.0).contains(&confidence)
            || self.included >= self.maximum
        {
            return false;
        }
        let current_cost = if self.included != 0 {
            self.cost(self.included)
        } else {
            100
        };
        let (mut survival, mut expected) = (self.survival, self.expected);
        let mut profitable = false;
        for width in self.included + 1..=self.maximum {
            survival *= f64::from(confidence);
            expected += survival;
            profitable |=
                expected * f64::from(current_cost) > self.expected * f64::from(self.cost(width));
        }
        if profitable {
            self.survival *= f64::from(confidence);
            self.expected += self.survival;
            self.included += 1;
        }
        profitable
    }
}

#[cfg(test)]
mod tests {
    use super::ConfidencePolicy;

    const CONCURRENCIES: [usize; 4] = [1, 2, 4, 8];

    /// The C++ test's policy: depth 4096 and the default three positions.
    fn policy(concurrency: usize) -> ConfidencePolicy {
        ConfidencePolicy::new(4096, concurrency, 3)
    }

    #[test]
    fn low_predicted_survival_selects_autoregression_before_drawing() {
        for concurrency in CONCURRENCIES {
            let mut cold = policy(concurrency);
            assert!(!cold.include(0.01) && cold.included == 0);
        }
    }

    #[test]
    fn high_confidence_respects_measured_hardware_cost() {
        for concurrency in CONCURRENCIES {
            let mut hot = policy(concurrency);
            let profitable = (1..=3).any(|width| hot.cost(width) < 100 * (width + 1));
            assert_eq!(hot.include(1.0), profitable, "concurrency {concurrency}");
            let included = hot.included;
            assert!(
                !hot.include(0.0) && hot.included == included,
                "stopping never retracts an earlier proposal"
            );
            assert!(
                !hot.include(f32::NAN) && !hot.include(f32::INFINITY),
                "non-finite confidence fails closed"
            );
        }
    }
}
