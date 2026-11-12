//! `gufo eval` against the HTTP fixture server of
//! `tests/tools/test_eval_http.py`: requests, grading, checkpoints, failure
//! classes, model discovery and redaction, as the C++ command passed them.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn eval_passes_the_http_fixture_suite() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new("python3")
        .arg(root.join("tests/tools/test_eval_http.py"))
        .arg("--gufo")
        .arg(env!("CARGO_BIN_EXE_gufo"))
        .arg("--data")
        .arg(root.join("tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json"))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("python3 runs the fixture suite");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
