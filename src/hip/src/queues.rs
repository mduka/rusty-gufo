//! The HIP runtime's hardware compute queue pool, sized by
//! `GPU_MAX_HW_QUEUES` before the runtime starts.

use crate::error::HipError;

const QUEUE_CAP_ENVIRONMENT: &str = "GPU_MAX_HW_QUEUES";

/// Threads of this process (`/proc/self/status`), when readable.
fn thread_count() -> Option<usize> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("Threads:"))?
        .trim()
        .parse()
        .ok()
}

/// Exports `GPU_MAX_HW_QUEUES=cap` unless the environment already sets it
/// (an operator's value is never replaced); true when it was exported. The
/// runtime reads the variable once, grows its queue pool to the cap and
/// releases nothing for the life of the process, so this must run before
/// the first HIP call. Environment writes race with readers on other
/// threads, so it is refused unless this is the process's only thread.
pub fn cap_hardware_queues(cap: u32) -> Result<bool, HipError> {
    if std::env::var_os(QUEUE_CAP_ENVIRONMENT).is_some() {
        return Ok(false);
    }
    if thread_count() != Some(1) {
        return Err(HipError::refused(
            "GPU_MAX_HW_QUEUES can only be set while the process has one thread",
        ));
    }
    // SAFETY: this is the process's only thread (checked above), so no other
    // thread can read or write the environment concurrently.
    unsafe { std::env::set_var(QUEUE_CAP_ENVIRONMENT, cap.to_string()) };
    Ok(true)
}
