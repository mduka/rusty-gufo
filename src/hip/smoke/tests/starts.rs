//! Runs the smoke binary directly, without the test harness or
//! LD_LIBRARY_PATH: only its own rpath can find the right runtime.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::process::Command;

#[test]
fn a_consuming_binary_loads_the_runtime_it_was_built_against() {
    let binary = env!("CARGO_BIN_EXE_gufo-hip-smoke");
    let output = Command::new(binary)
        .env_remove("LD_LIBRARY_PATH")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}{stderr}");
    assert!(
        stdout.contains(&format!("loaded HIP {}", gufo_hip::BUILD_HIP_VERSION)),
        "{stdout}"
    );

    let ldd = Command::new("ldd").arg(binary).output().unwrap();
    let ldd = String::from_utf8_lossy(&ldd.stdout);
    let runtime = ldd
        .lines()
        .find(|line| line.contains("libamdhip64"))
        .unwrap();
    assert!(runtime.contains(gufo_hip::BUILD_ROCM_ROOT), "{runtime}");
}
