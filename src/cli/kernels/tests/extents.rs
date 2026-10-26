//! Runs the extent probes generated from launch.toml and checks the kernels
//! against the host: cargo test -p gufo-cli-kernels --features gpu

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_cli_kernels as k;
use gufo_hip::Stream;

#[test]
fn every_launch_stays_within_its_declared_extents() {
    let stream = Stream::new().unwrap();
    let violations = k::probes::all(&stream).unwrap();
    assert!(violations.is_empty(), "{violations:#?}");
}

#[test]
fn the_pattern_and_its_checksum_match_the_host() {
    let stream = Stream::new().unwrap();
    for (size, blocks) in [(1usize, 1u32), (1000, 7), (3 << 20, 160)] {
        let mut data = stream.alloc::<u8>(size).unwrap();
        let mut sum = stream.alloc::<u64>(1).unwrap();
        k::fill_pattern(&stream, data.as_mut_slice(), size, 37, blocks).unwrap();
        stream.fill_bytes(sum.as_mut_slice(), 0).unwrap();
        k::checksum(&stream, data.as_slice(), sum.as_mut_slice(), size, blocks).unwrap();
        let mut bytes = vec![0u8; size];
        stream.download(data.as_slice(), &mut bytes).unwrap();
        let mut total = [0u64];
        stream.download(sum.as_slice(), &mut total).unwrap();
        assert!(
            bytes
                .iter()
                .enumerate()
                .all(|(i, &b)| b == (i as u64 + 37) as u8)
        );
        assert_eq!(total[0], bytes.iter().map(|&b| u64::from(b)).sum::<u64>());
    }
}
