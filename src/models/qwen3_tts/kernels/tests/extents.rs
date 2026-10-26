//! Runs the extent probes generated from launch.toml:
//! cargo test -p gufo-qwen3-tts-kernels --features gpu

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::Stream;
use gufo_qwen3_tts_kernels as k;

#[test]
fn every_launch_stays_within_its_declared_extents() {
    let stream = Stream::new().unwrap();
    let violations = k::probes::all(&stream).unwrap();
    assert!(violations.is_empty(), "{violations:#?}");
}
