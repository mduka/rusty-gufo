//! How long a request may wait for a checkpoint still being written to
//! disk (`text_publication_wait.hpp`): no longer than
//! prefilling the tokens it would save takes, as recent prefill steps
//! measure it, and at most two minutes.

use std::time::Duration;

/// A checkpoint queued or being written that a request's prompt extends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingPublication {
    pub checkpoint_id: u64,
    /// The prefix a lookup would reuse now, from RAM or disk.
    pub available_tokens: usize,
    pub pending_tokens: usize,
}

impl PendingPublication {
    /// Waiting pays off only for a large saving.
    pub const MINIMUM_GAIN_TOKENS: usize = 4096;

    /// The prompt tokens the pending checkpoint would save.
    pub fn gain(&self) -> usize {
        self.pending_tokens - self.pending_tokens.min(self.available_tokens)
    }
}

/// The prefill cost of recent steps (`PublicationWaitEstimate`): an estimate
/// of the useful work a wait saves, not a lifetime average that an old
/// warmup or tiny cached continuations would dominate.
#[derive(Debug, Default)]
pub struct PublicationWaitEstimate {
    /// Tokens and milliseconds of the last eight qualifying steps.
    samples: [(f64, f64); 8],
    next: usize,
}

impl PublicationWaitEstimate {
    pub const MAXIMUM_WAIT: Duration = Duration::from_secs(120);

    /// Records one prefill step that ran alone; short or untimed steps say
    /// little about the cost per token.
    pub fn observe(&mut self, tokens: usize, milliseconds: f64) {
        if tokens < 256 || !milliseconds.is_finite() || milliseconds < 1.0 {
            return;
        }
        self.samples[self.next] = (tokens as f64, milliseconds);
        self.next = (self.next + 1) % self.samples.len();
    }

    /// How long prefilling `gain` tokens takes at the recent rate, capped at
    /// [`MAXIMUM_WAIT`](Self::MAXIMUM_WAIT); `None` before any sample.
    pub fn limit(&self, gain: usize) -> Option<Duration> {
        let (tokens, milliseconds) = self
            .samples
            .iter()
            .fold((0.0, 0.0), |(t, m), &(tokens, ms)| (t + tokens, m + ms));
        if tokens == 0.0 {
            return None;
        }
        let cost = gain as f64 * (milliseconds / tokens);
        Some(Duration::from_secs_f64(
            (cost / 1e3).min(Self::MAXIMUM_WAIT.as_secs_f64()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_limit_follows_recent_prefill_cost() {
        let mut estimate = PublicationWaitEstimate::default();
        assert_eq!(estimate.limit(8192), None, "no estimate without samples");
        estimate.observe(255, 100.0);
        estimate.observe(1024, 0.5);
        estimate.observe(1024, f64::NAN);
        assert_eq!(
            estimate.limit(8192),
            None,
            "short and untimed steps are ignored"
        );
        estimate.observe(2048, 1024.0);
        assert_eq!(estimate.limit(4096), Some(Duration::from_millis(2048)));
        // Eight newer samples replace it.
        for _ in 0..8 {
            estimate.observe(1000, 100.0);
        }
        assert_eq!(estimate.limit(5000), Some(Duration::from_millis(500)));
        assert_eq!(
            estimate.limit(usize::MAX / 2),
            Some(PublicationWaitEstimate::MAXIMUM_WAIT),
            "the wait is capped"
        );
    }

    #[test]
    fn gain_counts_only_tokens_beyond_the_available_prefix() {
        let pending = |available, pending| PendingPublication {
            checkpoint_id: 1,
            available_tokens: available,
            pending_tokens: pending,
        };
        assert_eq!(pending(1000, 9000).gain(), 8000);
        assert_eq!(pending(9000, 1000).gain(), 0);
    }
}
