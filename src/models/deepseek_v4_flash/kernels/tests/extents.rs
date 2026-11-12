//! Runs the extent probes generated from launch.toml:
//! cargo test --profile gpu-test -p gufo-deepseek-v4-flash-kernels --features gpu --test extents

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_deepseek_v4_flash_kernels as k;
use gufo_hip::Stream;

#[test]
fn every_launch_stays_within_its_declared_extents() {
    let stream = Stream::new().unwrap();
    let violations = k::probes::all(&stream).unwrap();
    assert!(violations.is_empty(), "{violations:#?}");
}
