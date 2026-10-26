//! FIFO admission to a runtime with mutable device state
//! (`src/core/cancellable_gate.hpp`): waiting clients can cancel without
//! waiting for the current inference to finish.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

use crate::text::{Error, Result};

/// Waiting clients beyond this are refused.
const MAXIMUM_QUEUE: usize = 16;
const POLL_INTERVAL: Duration = Duration::from_millis(10);

struct State {
    queue: VecDeque<u64>,
    next: u64,
    active: usize,
}

pub struct CancellableGate {
    state: Mutex<State>,
    ready: Condvar,
    capacity: usize,
}

/// One admitted client; dropping it admits the next.
pub struct GateLease<'a> {
    gate: &'a CancellableGate,
}

impl Drop for GateLease<'_> {
    fn drop(&mut self) {
        self.gate.lock().active -= 1;
        self.gate.ready.notify_all();
    }
}

impl CancellableGate {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::invalid("inference capacity must be positive"));
        }
        Ok(Self {
            state: Mutex::new(State {
                queue: VecDeque::new(),
                next: 0,
                active: 0,
            }),
            ready: Condvar::new(),
            capacity,
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Waits in FIFO order for a free slot, polling `cancelled` every 10 ms.
    /// The error is the C++ message: a full queue or a cancelled wait.
    pub fn acquire(
        &self,
        cancelled: &dyn Fn() -> bool,
    ) -> std::result::Result<GateLease<'_>, String> {
        let mut state = self.lock();
        if state.queue.len() >= MAXIMUM_QUEUE {
            return Err("audio request queue is full".into());
        }
        state.next += 1;
        let ticket = state.next;
        state.queue.push_back(ticket);
        loop {
            if cancelled() {
                state.queue.retain(|&t| t != ticket);
                drop(state);
                self.ready.notify_all();
                return Err("audio request cancelled while queued".into());
            }
            if state.active < self.capacity && state.queue.front() == Some(&ticket) {
                state.queue.pop_front();
                state.active += 1;
                drop(state);
                self.ready.notify_all();
                return Ok(GateLease { gate: self });
            }
            state = self
                .ready
                .wait_timeout(state, POLL_INTERVAL)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|e| e.into_inner().0);
        }
    }
}

impl Default for CancellableGate {
    /// Capacity one.
    fn default() -> Self {
        Self {
            state: Mutex::new(State {
                queue: VecDeque::new(),
                next: 0,
                active: 0,
            }),
            ready: Condvar::new(),
            capacity: 1,
        }
    }
}
