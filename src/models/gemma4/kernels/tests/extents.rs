//! Runs the extent probes generated from launch.toml:
//! cargo test -p gufo-gemma4-kernels --features gpu

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_gemma4_kernels as k;
use gufo_hip::Stream;

#[test]
fn every_launch_stays_within_its_declared_extents() {
    let stream = Stream::new().unwrap();
    let violations = k::probes::all(&stream).unwrap();
    assert!(violations.is_empty(), "{violations:#?}");
}
