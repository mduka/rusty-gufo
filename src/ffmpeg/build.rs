//! Compiles the field shim against the build's FFmpeg headers and records
//! where the matching FFmpeg libraries are, without linking them.
//!
//! `pkg-config` names the headers and the library directory of
//! libavformat, libavcodec, libavutil and libswscale. Each library's
//! versioned file (the target of its development symlink) is recorded as
//! `GUFO_<NAME>_LIBRARY`; `gufo-ffmpeg` opens that file on the first video,
//! falling back to its file name for relocated packages (as upstream's
//! `GUFO_*_LIBRARY` definitions do).

use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

const COMPONENTS: [&str; 4] = ["libavformat", "libavcodec", "libavutil", "libswscale"];

fn pkg_config(args: &[&str]) -> String {
    let output = Command::new(env::var("PKG_CONFIG").unwrap_or_else(|_| "pkg-config".into()))
        .args(args)
        .args(COMPONENTS)
        .output()
        .unwrap_or_else(|e| panic!("gufo-ffmpeg: cannot run pkg-config: {e}"));
    if !output.status.success() {
        panic!(
            "gufo-ffmpeg: FFmpeg development files not found ({}); install \
             libavformat, libavcodec, libavutil and libswscale headers",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

fn main() {
    println!("cargo:rerun-if-changed=shim/fields.c");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=CC");
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let cflags = pkg_config(&["--cflags"]);

    let object = out.join("fields.o");
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(&cc)
        .args([
            "-std=c11",
            "-O2",
            "-fPIC",
            "-Wall",
            "-Werror",
            "-c",
            "shim/fields.c",
            "-o",
        ])
        .arg(&object)
        .args(cflags.split_whitespace())
        .status()
        .unwrap_or_else(|e| panic!("gufo-ffmpeg: cannot run {cc}: {e}"));
    assert!(
        status.success(),
        "gufo-ffmpeg: {cc} failed on shim/fields.c"
    );
    let archive = out.join("libgufo_ffmpeg_fields.a");
    let _ = fs::remove_file(&archive);
    let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
    let status = Command::new(&ar)
        .arg("crs")
        .arg(&archive)
        .arg(&object)
        .status()
        .unwrap_or_else(|e| panic!("gufo-ffmpeg: cannot run {ar}: {e}"));
    assert!(status.success(), "gufo-ffmpeg: {ar} failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=gufo_ffmpeg_fields");

    for component in COMPONENTS {
        let output = Command::new(env::var("PKG_CONFIG").unwrap_or_else(|_| "pkg-config".into()))
            .args(["--variable=libdir", component])
            .output()
            .unwrap_or_else(|e| panic!("gufo-ffmpeg: cannot run pkg-config: {e}"));
        let libdir = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
        let name = component.trim_start_matches("lib");
        let link = libdir.join(format!("lib{name}.so"));
        println!("cargo:rerun-if-changed={}", link.display());
        // The versioned file keeps loading the library the headers describe.
        let runtime = fs::canonicalize(&link)
            .unwrap_or_else(|e| panic!("gufo-ffmpeg: cannot resolve {}: {e}", link.display()));
        println!(
            "cargo:rustc-env=GUFO_{}_LIBRARY={}",
            name.to_ascii_uppercase(),
            runtime.display()
        );
    }
}
