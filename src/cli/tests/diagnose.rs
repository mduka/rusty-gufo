//! `gufo diagnose` and `gufo --version` as processes: cargo test -p gufo [--features gpu]

#![allow(clippy::unwrap_used)]

use std::process::{Command, Output};

fn gufo(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gufo"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn allocation_options_are_checked_before_the_run() {
    for (args, message) in [
        (
            &["--repetitions", "0"][..],
            "Error: --repetitions must be at least 1 for the allocation diagnostic\n",
        ),
        (
            &["--working-set-mib", "1,,2"],
            "Error: working-set sizes must be positive comma-separated MiB\n",
        ),
        (
            &["--working-set-mib", "0"],
            "Error: working-set sizes must be positive comma-separated MiB\n",
        ),
    ] {
        let output = gufo(&[&["diagnose", "--benchmark", "allocation"][..], args].concat());
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert_eq!(String::from_utf8_lossy(&output.stderr), message);
    }
}

/// Every path at two working sets: completed with separated phases and
/// verified CPU and GPU checksums.
#[cfg(feature = "gpu")]
#[test]
fn every_allocation_path_completes_with_verified_checksums() {
    let output = gufo(&[
        "diagnose",
        "--benchmark",
        "allocation",
        "--json",
        "--repetitions",
        "2",
        "--warmup",
        "0",
        "--working-set-mib",
        "1,3",
    ]);
    assert!(output.status.success(), "{output:?}");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["summary"]["status"], "completed");
    let paths = report["paths"].as_array().unwrap();
    let names: Vec<&str> = paths
        .iter()
        .map(|p| p["pathName"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "hip_malloc",
            "hip_malloc_managed",
            "hip_host_malloc_mapped",
            "file_mmap_host_register",
            "hip_malloc_async",
        ]
        .repeat(2)
    );
    for path in paths {
        assert_eq!(path["status"], "completed", "{path}");
        assert_eq!(path["checksumVerified"], true);
        assert_eq!(path["cpuChecksum"], path["expectedChecksum"]);
        let phases: Vec<&str> = path["phases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["name"].as_str().unwrap())
            .collect();
        for name in [
            "allocation",
            "first_gpu_touch_read",
            "warm_gpu_write",
            "teardown",
        ] {
            assert!(phases.contains(&name), "{phases:?}");
        }
        // Two repetitions: one sample resource beside the measured one.
        let allocation = &path["phases"][0];
        assert_eq!(allocation["rawRepetitions"].as_array().unwrap().len(), 2);
    }
    let pointer = |index: usize| paths[index]["pointerMemoryType"].as_str().unwrap();
    assert_eq!(
        (pointer(0), pointer(1), pointer(2), pointer(3)),
        ("device", "managed", "host", "host")
    );
    // Registration pins the memory file's pages; residency counts them.
    let evidence = paths[3]["residencyEvidence"].as_str().unwrap();
    assert_eq!(
        evidence,
        "before first touch: 256/256 pages resident; after measured phases: 256/256 pages resident"
    );
    assert!(
        paths[0]["residencyEvidence"]
            .as_str()
            .unwrap()
            .starts_with("before first touch: mincore unavailable: no host mapping;")
    );
}

#[test]
fn the_version_names_the_release_and_the_revision() {
    let output = gufo(&["--version"]);
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    let (release, revision) = text
        .trim_end()
        .strip_prefix("gufo version ")
        .and_then(|rest| rest.split_once(" ("))
        .expect("gufo version <release> (<revision>)");
    assert!(!release.is_empty() && revision.ends_with(')'), "{text}");
}
