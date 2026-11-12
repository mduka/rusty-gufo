//! `gufo serve` startup as a process: cargo test -p gufo --test serve

#![allow(clippy::unwrap_used)]

use std::process::Command;

/// The stderr of a `serve llm` that fails at its missing model, after
/// everything the log level shapes was logged.
fn failed_load(extra: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_gufo"))
        .args(["serve", "llm", "--model", "/nonexistent.gguf"])
        .args(extra)
        .env("NO_COLOR", "1")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1), "{extra:?}");
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn log_level_changes_startup_output() {
    assert!(!failed_load(&[]).contains("event=options"));
    for spelling in [&["-v"][..], &["--verbose"], &["--log-level", "debug"]] {
        let stderr = failed_load(spelling);
        assert!(
            stderr.contains("[DEBUG] [server] event=options") && stderr.contains("log_level=debug"),
            "{spelling:?}: {stderr}"
        );
    }
    for level in ["info", "warn", "error"] {
        assert!(
            !failed_load(&["--log-level", level]).contains("event=options"),
            "{level}"
        );
    }
    // The queue budget is the first startup line, logged after the level is
    // armed, so the quiet tiers drop it with the rest of the boot sequence.
    for (extra, shown) in [
        (&[][..], true),
        (&["--log-level", "info"], true),
        (&["--log-level", "debug"], true),
        (&["--log-level", "warn"], false),
        (&["--log-level", "error"], false),
    ] {
        assert_eq!(
            failed_load(extra).contains("event=queue_budget kind=llm"),
            shown,
            "{extra:?}"
        );
    }
    // Debug lines report the API key as set without leaking it.
    let keyed = failed_load(&["-v", "--api-key", "super-secret-value"]);
    assert!(keyed.contains("api_key=set"), "{keyed}");
    assert!(!keyed.contains("super-secret-value"), "{keyed}");
    // The load error itself is an error at every tier.
    assert!(failed_load(&["--log-level", "error"]).contains("Error loading model"));
}
