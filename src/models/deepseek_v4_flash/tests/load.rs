//! Invalid artifacts against `Engine::open` (the C++ `ds4.load` test): a
//! truncated file, a GGUF header without tensors or metadata, and one whose
//! tokenizer lacks the required special tokens. Each must fail three times
//! in a row before the device is touched, leaving no mapping, no extra
//! descriptor and no process lock behind, and a lock held elsewhere must
//! make another open fail. Needs neither the model nor the device.

#![allow(clippy::unwrap_used)]

use std::fs::File;
use std::os::fd::AsFd;
use std::path::Path;
use std::process::Command;

use gufo_deepseek_v4_flash::engine::Engine;
use rustix::fs::{FlockOperation, flock};

fn integer(out: &mut Vec<u8>, value: u64, bytes: usize) {
    out.extend_from_slice(&value.to_le_bytes()[..bytes]);
}

fn string(out: &mut Vec<u8>, text: &str) {
    integer(out, text.len() as u64, 8);
    out.extend_from_slice(text.as_bytes());
}

/// The bytes of fixture `index` and the error `Engine::open` must return:
/// each from a stage before `Backend::init` (GGUF parse, tokenizer).
fn fixture(index: usize) -> (Vec<u8>, &'static str) {
    if index == 0 {
        return (b"truncated".to_vec(), "model file is too small to be GGUF");
    }
    let mut out = Vec::new();
    integer(&mut out, 0x4655_4747, 4);
    integer(&mut out, 3, 4);
    integer(&mut out, 0, 8); // No tensors.
    integer(&mut out, if index == 1 { 0 } else { 2 }, 8);
    if index == 2 {
        string(&mut out, "tokenizer.ggml.tokens");
        integer(&mut out, 9, 4); // Array.
        integer(&mut out, 8, 4); // Strings.
        integer(&mut out, 1, 8);
        string(&mut out, "hello"); // Required special tokens are missing.
        string(&mut out, "tokenizer.ggml.merges");
        integer(&mut out, 9, 4);
        integer(&mut out, 8, 4);
        integer(&mut out, 0, 8);
    }
    while out.len() < 32 || !out.len().is_multiple_of(32) {
        out.push(0);
    }
    let error = if index == 1 {
        "GGUF tokenizer token table is missing or invalid"
    } else {
        "required tokenizer token is missing"
    };
    (out, error)
}

fn open_descriptors() -> usize {
    std::fs::read_dir("/proc/self/fd").unwrap().count()
}

fn mapped(path: &Path) -> bool {
    std::fs::read_to_string("/proc/self/maps")
        .unwrap()
        .contains(path.to_str().unwrap())
}

fn open_error(path: &Path) -> Option<String> {
    Engine::open(path, None).err()
}

/// The engine's lock file comes from `DS4_LOCK_FILE`, which a process sets
/// before it starts: the test runs itself again as a child with the
/// variable pointing into a temporary directory.
#[test]
fn invalid_artifacts_fail_and_release_files_and_lock() {
    let Some(root) = std::env::var_os("GUFO_DS4_LOAD_DIRECTORY") else {
        let root = tempfile::Builder::new()
            .prefix("gufo-ds4-load-")
            .tempdir()
            .unwrap();
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "invalid_artifacts_fail_and_release_files_and_lock",
                "--nocapture",
            ])
            .env("GUFO_DS4_LOAD_DIRECTORY", root.path())
            .env("DS4_LOCK_FILE", root.path().join("instance.lock"))
            .status()
            .unwrap();
        assert!(
            status.success(),
            "the load checks failed in the child process"
        );
        return;
    };
    let root = Path::new(&root);
    let path = root.join("invalid.gguf");
    let lock_path = root.join("instance.lock");
    let descriptors = open_descriptors();
    for index in 0..3 {
        let (bytes, error) = fixture(index);
        std::fs::write(&path, bytes).unwrap();
        for repeat in 0..3 {
            let case = format!("fixture {index} repeat {repeat}");
            // An error result carries no engine.
            assert_eq!(open_error(&path).as_deref(), Some(error), "{case}");
            // Failure must release both the mapped file and the process lock.
            assert!(!mapped(&path), "{case}");
            let lock = File::options()
                .read(true)
                .write(true)
                .open(&lock_path)
                .unwrap();
            flock(lock.as_fd(), FlockOperation::NonBlockingLockExclusive).unwrap();
            // A separately held lock must cause another ordinary load failure.
            assert_eq!(
                open_error(&path).as_deref(),
                Some("DeepSeek instance lock unavailable"),
                "{case}"
            );
            drop(lock);
            assert_eq!(open_descriptors(), descriptors, "{case}");
        }
    }
}
