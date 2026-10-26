//! Starts the HIP runtime like a gufo executable: proves that a binary which
//! depends on gufo-hip loads the ROCm runtime it was built against. Template
//! for the build script of every Rust executable (see build.rs).

#![forbid(unsafe_code)]

use std::process::ExitCode;

fn main() -> ExitCode {
    println!(
        "built against HIP {} from {}",
        gufo_hip::BUILD_HIP_VERSION,
        gufo_hip::BUILD_ROCM_ROOT
    );
    match gufo_hip::runtime_version() {
        Ok(version) => println!("loaded HIP {version}"),
        Err(error) => println!("loaded HIP unknown: {error}"),
    }
    match gufo_hip::Stream::new() {
        Ok(_) => {
            println!("stream ok");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
