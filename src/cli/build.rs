use std::env;
use std::process::Command;

fn main() {
    gufo_kernel_build::link_hip_runtime();
    // The release version and source revision `gufo --version` prints: the
    // package version unless GUFO_RELEASE_VERSION is set; an empty value (a
    // container build argument left out) counts as unset.
    let package_version = env::var("CARGO_PKG_VERSION").unwrap_or_default();
    for (name, default) in [
        ("GUFO_RELEASE_VERSION", package_version.as_str()),
        ("GUFO_REVISION", "unknown"),
    ] {
        println!("cargo:rerun-if-env-changed={name}");
        let value = env::var(name)
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| default.into());
        println!("cargo:rustc-env={name}={value}");
    }
    // The compiler `gufo diagnose` reports, e.g. "rustc 1.98.1".
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let compiler = Command::new(&rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|text| {
            text.split_whitespace()
                .take(2)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "rustc".into());
    println!("cargo:rustc-env=GUFO_RUSTC_VERSION={compiler}");
}
