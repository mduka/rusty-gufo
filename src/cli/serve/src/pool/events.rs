//! Runner validation and the cache event log lines.

use super::*;

pub(super) fn validate_runner<R: TextModelRunner>(
    runner: Arc<R>,
    state_count: usize,
) -> Result<Validated<R>> {
    if state_count == 0 {
        return Err(Error::invalid(
            "text runner state count must be at least one",
        ));
    }
    let descriptor = runner.descriptor();
    if descriptor.model_id.is_empty() {
        return Err(Error::invalid("text runner model ID must not be empty"));
    }
    if descriptor.state_abi.is_empty() {
        return Err(Error::invalid("text runner state ABI must not be empty"));
    }
    if descriptor.max_context == 0 {
        return Err(Error::invalid(
            "text runner maximum context must be at least one token",
        ));
    }
    let capabilities = &descriptor.capabilities;
    if capabilities.fork && !capabilities.snapshot {
        return Err(Error::invalid(
            "text runner fork capability requires snapshot support",
        ));
    }
    if let Some(persistence) = &descriptor.persistence {
        if !capabilities.snapshot || !capabilities.fork {
            return Err(Error::invalid(
                "text runner persistence requires snapshot and fork support",
            ));
        }
        if persistence.compatibility_identity.is_empty() {
            return Err(Error::invalid(
                "text runner persistence identity must not be empty",
            ));
        }
        if persistence.payload_version == 0 {
            return Err(Error::invalid(
                "text runner persistence payload version must be nonzero",
            ));
        }
    }

    let resources = runner.resource_claim();
    if let (Some(capacity), Some(per_request)) = (
        resources.state_capacity_bytes,
        per_state_reservation_bytes(&resources)?,
    ) && per_request != 0
        && state_count > capacity / per_request
    {
        return Err(Error::invalid(
            "text runner request-state claim exceeds state capacity",
        ));
    }

    let plans = runner.supported_plans();
    if plans.is_empty() {
        return Err(Error::invalid(
            "text runner must expose at least one execution plan",
        ));
    }
    for (index, plan) in plans.iter().enumerate() {
        if plan.physical_width == 0 {
            return Err(Error::invalid(
                "text runner execution plan width must be at least one",
            ));
        }
        let consistent = match plan.kind {
            TextExecutionPlanKind::Serial => plan.physical_width == 1,
            TextExecutionPlanKind::Batched => plan.physical_width >= 2,
        };
        if !consistent {
            return Err(Error::invalid(
                "text runner execution plan kind and width are inconsistent",
            ));
        }
        if plans[..index].contains(plan) {
            return Err(Error::invalid("text runner execution plans must be unique"));
        }
    }
    if !plans.contains(&TextExecutionPlan::SERIAL) {
        return Err(Error::invalid(
            "text runner must support the serial single-request plan",
        ));
    }
    Ok(Validated {
        runner,
        descriptor,
        resources,
        plans,
    })
}

pub(super) fn reconcile_state_bytes<S: TextRunnerState>(
    resources: &TextRunnerResourceClaim,
    state: &S,
) -> Result<()> {
    let measured = state.measured_resources();
    if let (Some(measured), Some(claimed)) = (
        measured.per_request_state_bytes,
        resources.per_request_state_bytes,
    ) && measured > claimed
    {
        return Err(Error::runtime(
            "text runner measured state exceeds its resource claim",
        ));
    }
    if let (Some(measured), Some(claimed)) = (
        measured.temporary_scratch_bytes,
        resources.temporary_scratch_bytes,
    ) && measured > claimed
    {
        return Err(Error::runtime(
            "text runner measured scratch exceeds its resource claim",
        ));
    }
    Ok(())
}

/// Replacing a retained prefix, or dropping one to fit the byte budget, is
/// routine (request summaries already report reuse). Losing one because
/// every record is taken is not: the server holds fewer conversations than
/// the workload rotates through. Skipped captures need attention too.
pub(super) fn emit_snapshot_event(event: &SnapshotEvent) {
    if event.action == SnapshotEventAction::Removed
        && event.reason != SnapshotEventReason::EntryCapacity
    {
        return;
    }
    logging::warn(
        "cache",
        &format!(
            "event=snapshot action={} reason={} bytes={} tokens={} retained_bytes={} \
             reserved_bytes={} capacity_bytes={}",
            event.action.name(),
            event.reason.name(),
            event.snapshot_bytes,
            event.token_count,
            event.retained_snapshot_bytes,
            event.reserved_snapshot_bytes,
            event.capacity_bytes
        ),
    );
}

pub(super) fn disk_action_name(action: ContinuationDiskEventAction) -> &'static str {
    match action {
        ContinuationDiskEventAction::Stored => "stored",
        ContinuationDiskEventAction::Restored => "restored",
        ContinuationDiskEventAction::Miss => "miss",
        ContinuationDiskEventAction::Removed => "removed",
        ContinuationDiskEventAction::Skipped => "skipped",
    }
}

pub(super) fn disk_reason_name(reason: ContinuationDiskEventReason) -> &'static str {
    use ContinuationDiskEventReason as R;
    match reason {
        R::Saved => "saved",
        R::Hit => "hit",
        R::NotFound => "not_found",
        R::Unsupported => "unsupported",
        R::ByteCapacity => "byte_capacity",
        R::StagingCapacity => "staging_capacity",
        R::Lru => "lru",
        R::ExactReplacement => "exact_replacement",
        R::MinStep => "min_step",
        R::Corrupt => "corrupt",
        R::ChecksumMismatch => "checksum_mismatch",
        R::UnsafeFile => "unsafe_file",
        R::IoFailure => "io_failure",
        R::SerializationFailure => "serialization_failure",
        R::RestoreFailure => "restore_failure",
        R::Busy => "busy",
    }
}

pub(super) fn emit_disk_event(event: &ContinuationDiskEvent) {
    use ContinuationDiskEventReason as R;
    match event.reason {
        R::Hit | R::NotFound | R::ExactReplacement | R::Busy => return,
        R::ByteCapacity if event.action == ContinuationDiskEventAction::Removed => return,
        _ => {}
    }
    let mut line = format!(
        "event=disk_cache action={} reason={} file_bytes={} payload_bytes={} tokens={} \
         retained_bytes={} capacity_bytes={} staging_capacity_bytes={} staging_used_bytes={}",
        disk_action_name(event.action),
        disk_reason_name(event.reason),
        event.file_bytes,
        event.payload_bytes,
        event.token_count,
        event.retained_bytes,
        event.capacity_bytes,
        event.staging_capacity_bytes,
        event.staging_used_bytes
    );
    if event.reason == R::Saved {
        let _ = write!(line, " write_ms={}", format_general(event.elapsed_ms));
        logging::info("cache", &line);
    } else if event.reason == R::Lru || event.reason == R::MinStep {
        // Staying inside the budget or skipping a checkpoint that barely
        // advances is expected, but it has to be visible: otherwise a run
        // that evicts every other conversation looks like one that never
        // cached anything.
        logging::info("cache", &line);
    } else {
        logging::warn("cache", &line);
    }
}
