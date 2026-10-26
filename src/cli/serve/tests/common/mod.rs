//! Helpers shared by the serving tests.

use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// `std::counting_semaphore` with a timed acquire.
#[derive(Default)]
pub struct Semaphore {
    count: Mutex<usize>,
    signal: Condvar,
}

#[allow(dead_code)]
impl Semaphore {
    pub fn release(&self) {
        self.release_n(1);
    }

    pub fn release_n(&self, n: usize) {
        *self.count.lock().expect("semaphore") += n;
        self.signal.notify_all();
    }

    pub fn acquire(&self) {
        let mut count = self.count.lock().expect("semaphore");
        while *count == 0 {
            count = self.signal.wait(count).expect("semaphore");
        }
        *count -= 1;
    }

    pub fn try_acquire_for(&self, timeout: Duration) -> bool {
        let count = self.count.lock().expect("semaphore");
        let (mut count, _) = self
            .signal
            .wait_timeout_while(count, timeout, |c| *c == 0)
            .expect("semaphore");
        if *count == 0 {
            return false;
        }
        *count -= 1;
        true
    }
}
