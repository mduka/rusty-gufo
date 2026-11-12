//! Test-only fault injection: the next `check` of a named operation reports
//! the planned error code after the real call. Per thread, so parallel tests
//! do not interfere. Not compiled outside `cfg(test)`.

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static PLANNED: RefCell<HashMap<&'static str, i32>> = RefCell::new(HashMap::new());
}

/// Makes the next check of `operation` fail with `code`.
pub(crate) fn inject(operation: &'static str, code: i32) {
    PLANNED.with(|planned| planned.borrow_mut().insert(operation, code));
}

pub(crate) fn take(operation: &str) -> Option<i32> {
    PLANNED.with(|planned| planned.borrow_mut().remove(operation))
}
