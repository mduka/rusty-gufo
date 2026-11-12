//! The GPU share an arbiter grants a background server (`--arbiter`).
//!
//! Background work (text generation) runs for at most `share` of the time
//! while realtime servers need the GPU: after each step it pauses for the
//! rest. A grant carries a lifetime and lapses to the full share when the
//! arbiter stops renewing it, so a lost arbiter never leaves a server slowed.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// The smallest share a grant can set: background work keeps progressing.
pub const MIN_SHARE: f32 = 0.05;

/// The granted share and its expiry, readable from any thread without a lock.
#[derive(Debug)]
pub struct GpuShare {
    origin: Instant,
    /// The share's `f32` bits.
    share: AtomicU32,
    /// Nanoseconds after `origin` at which the grant lapses.
    expires: AtomicU64,
    /// Nanoseconds spent pausing for the grant.
    yielded: AtomicU64,
}

impl Default for GpuShare {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
            share: AtomicU32::new(1.0f32.to_bits()),
            expires: AtomicU64::new(0),
            yielded: AtomicU64::new(0),
        }
    }
}

impl GpuShare {
    /// Grants `share` (clamped to `MIN_SHARE..=1`) for `lifetime`.
    pub fn grant(&self, share: f32, lifetime: Duration) {
        let share = if share.is_nan() {
            1.0
        } else {
            share.clamp(MIN_SHARE, 1.0)
        };
        let expires = self
            .nanos(Instant::now())
            .saturating_add(saturating_nanos(lifetime));
        self.share.store(share.to_bits(), Ordering::Relaxed);
        self.expires.store(expires, Ordering::Release);
    }

    /// Ends the grant: the full share applies again.
    pub fn release(&self) {
        self.expires.store(0, Ordering::Release);
    }

    /// The share in force now; 1 once the grant lapsed.
    pub fn current(&self) -> f32 {
        if self.nanos(Instant::now()) >= self.expires.load(Ordering::Acquire) {
            return 1.0;
        }
        f32::from_bits(self.share.load(Ordering::Relaxed))
    }

    /// Time spent pausing for grants so far.
    pub fn yielded(&self) -> Duration {
        Duration::from_nanos(self.yielded.load(Ordering::Relaxed))
    }

    /// After a step that took `step`, pauses so that steps take the share
    /// in force of the time. The pause runs in slices of at most `slice`,
    /// rereading the share, and ends early once the share rises or `stop`
    /// holds. Returns the time paused.
    pub fn pause_after(
        &self,
        step: Duration,
        slice: Duration,
        stop: &dyn Fn() -> bool,
    ) -> Duration {
        if self.current() >= 1.0 {
            return Duration::ZERO;
        }
        let started = Instant::now();
        loop {
            let share = self.current();
            if share >= 1.0 || stop() {
                break;
            }
            let due = step.mul_f32((1.0 - share) / share);
            let paused = started.elapsed();
            if paused >= due {
                break;
            }
            std::thread::sleep((due - paused).min(slice));
        }
        let paused = started.elapsed();
        self.yielded
            .fetch_add(saturating_nanos(paused), Ordering::Relaxed);
        paused
    }

    fn nanos(&self, at: Instant) -> u64 {
        saturating_nanos(at.saturating_duration_since(self.origin))
    }
}

fn saturating_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grant_lapses_to_the_full_share() {
        let share = GpuShare::default();
        assert_eq!(share.current(), 1.0);
        share.grant(0.25, Duration::from_secs(60));
        assert_eq!(share.current(), 0.25);
        share.release();
        assert_eq!(share.current(), 1.0);
        share.grant(0.25, Duration::ZERO);
        assert_eq!(share.current(), 1.0, "an expired grant no longer applies");
    }

    #[test]
    fn grants_are_clamped() {
        let share = GpuShare::default();
        share.grant(0.0, Duration::from_secs(60));
        assert_eq!(share.current(), MIN_SHARE);
        share.grant(7.0, Duration::from_secs(60));
        assert_eq!(share.current(), 1.0);
        share.grant(f32::NAN, Duration::from_secs(60));
        assert_eq!(share.current(), 1.0);
    }

    #[test]
    fn the_pause_fills_the_rest_of_the_share() {
        let share = GpuShare::default();
        let slice = Duration::from_millis(5);
        assert_eq!(
            share.pause_after(Duration::from_millis(10), slice, &|| false),
            Duration::ZERO
        );
        share.grant(0.5, Duration::from_secs(60));
        let paused = share.pause_after(Duration::from_millis(20), slice, &|| false);
        assert!(paused >= Duration::from_millis(20), "{paused:?}");
        assert!(paused < Duration::from_millis(200), "{paused:?}");
        assert!(share.yielded() >= Duration::from_millis(20));
        let stopped = share.pause_after(Duration::from_secs(10), slice, &|| true);
        assert!(stopped < Duration::from_millis(100), "{stopped:?}");
    }
}
