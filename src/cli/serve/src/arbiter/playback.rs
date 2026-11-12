//! The playback buffer a streamed TTS reports to the arbiter.
//!
//! A client plays streamed audio as it arrives: every chunk fills its
//! buffer by the chunk's duration, playback drains it at one second per
//! second, and an empty buffer is a gap. The model follows that buffer,
//! assuming playback starts with the first chunk; it says nothing about
//! pressure, which the arbiter derives from it.

use std::sync::Mutex;
use std::time::Instant;

#[derive(Debug, Default)]
struct State {
    /// Seconds buffered at `updated`.
    buffered: f64,
    updated: Option<Instant>,
    /// Streamed syntheses running.
    active: usize,
    /// Seconds of audio streamed so far.
    audio: f64,
}

/// What a streamed TTS reports: whether it synthesizes, the modelled
/// buffer and the audio streamed so far.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaybackSnapshot {
    pub active: bool,
    pub buffer_s: f64,
    pub audio_s: f64,
}

#[derive(Debug, Default)]
pub struct PlaybackBuffer {
    state: Mutex<State>,
}

impl PlaybackBuffer {
    /// A streamed synthesis starts.
    pub fn begin(&self) {
        self.lock().active += 1;
    }

    /// A streamed synthesis ends, successfully or not.
    pub fn end(&self) {
        let mut state = self.lock();
        state.active = state.active.saturating_sub(1);
    }

    /// `seconds` of audio went to a client at `now`.
    pub fn add(&self, seconds: f64, now: Instant) {
        let mut state = self.lock();
        let buffered = drained(&state, now);
        state.buffered = buffered + seconds;
        state.updated = Some(now);
        state.audio += seconds;
    }

    pub fn snapshot(&self, now: Instant) -> PlaybackSnapshot {
        let state = self.lock();
        PlaybackSnapshot {
            active: state.active > 0,
            buffer_s: drained(&state, now),
            audio_s: state.audio,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The buffer at `now`: what was buffered less the playback since.
fn drained(state: &State, now: Instant) -> f64 {
    state.updated.map_or(0.0, |updated| {
        (state.buffered - now.saturating_duration_since(updated).as_secs_f64()).max(0.0)
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn the_buffer_fills_with_audio_and_drains_with_time() {
        let buffer = PlaybackBuffer::default();
        let start = Instant::now();
        assert_eq!(buffer.snapshot(start).buffer_s, 0.0);
        buffer.begin();
        buffer.add(1.0, start);
        let half = start + Duration::from_millis(500);
        let snapshot = buffer.snapshot(half);
        assert!(snapshot.active);
        assert!((snapshot.buffer_s - 0.5).abs() < 1e-9);
        buffer.add(2.0, half);
        assert!((buffer.snapshot(half).buffer_s - 2.5).abs() < 1e-9);
        // Playback ran dry: the buffer stays empty instead of going negative.
        assert_eq!(
            buffer.snapshot(half + Duration::from_secs(10)).buffer_s,
            0.0
        );
        assert!((buffer.snapshot(half).audio_s - 3.0).abs() < 1e-9);
        buffer.end();
        assert!(!buffer.snapshot(half).active);
        buffer.end();
        assert!(!buffer.snapshot(half).active, "ends never underflow");
    }
}
