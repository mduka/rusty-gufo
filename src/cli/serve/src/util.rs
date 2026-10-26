//! Small helpers shared by the HTTP APIs.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const ID_CHARACTERS: &[u8; 36] = b"abcdefghijklmnopqrstuvwxyz0123456789";

/// Seconds since the epoch (`std::time(nullptr)`).
pub fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `prefix` plus `length` random characters from `[a-z0-9]`, for response
/// and tool-call IDs (C++ uses `std::mt19937` seeded from `random_device`).
pub fn random_id(prefix: &str, length: usize) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let state = RandomState::new();
    let mut id = String::with_capacity(prefix.len() + length);
    id.push_str(prefix);
    let mut word = 0u64;
    for index in 0..length {
        if index % 8 == 0 {
            let mut hasher = state.build_hasher();
            hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
            hasher.write_usize(index);
            word = hasher.finish();
        }
        id.push(char::from(ID_CHARACTERS[(word % 36) as usize]));
        word /= 36;
    }
    id
}

#[cfg(test)]
mod tests {
    use super::random_id;

    #[test]
    fn ids_use_the_alphabet_and_differ() {
        let a = random_id("call_", 20);
        let b = random_id("call_", 20);
        assert_eq!(a.len(), 25);
        assert!(
            a[5..]
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        );
        assert_ne!(a, b);
    }
}
