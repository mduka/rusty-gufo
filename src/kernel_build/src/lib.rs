//! Build-script support for kernel crates: compiles a model's HIP sources
//! with hipcc and generates both sides of its launcher ABI from `launch.toml`.
//!
//! The C++ shim `static_assert`s each launcher's exact signature, so a drifted
//! declaration fails the build instead of passing mismatched arguments. The
//! Rust wrappers check stream health, buffer ownership and declared extents
//! before anything is enqueued. See docs/RUST.md.
//!
//! The only `unsafe` here is the jobserver handoff in `compile.rs`, which runs
//! at build time.

#![deny(unsafe_code)]

mod codegen;
mod compile;
mod extent;
mod manifest;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

use manifest::{Manifest, Source, default_std};

/// Entry point for a kernel crate's `build.rs`:
/// `gufo_kernel_build::compile_kernels("launch.toml")`.
pub fn compile_kernels(manifest_path: &str) {
    if let Err(message) = try_compile_kernels(Path::new(manifest_path)) {
        panic!("gufo-kernel-build: {message}");
    }
}

fn try_compile_kernels(manifest_path: &Path) -> Result<(), String> {
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    for var in ["HIPCC", "ROCM_PATH", "AR", "CC"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let text = fs::read_to_string(manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest =
        Manifest::parse(&text).map_err(|e| format!("{}: {e}", manifest_path.display()))?;

    let out = PathBuf::from(env::var("OUT_DIR").map_err(|_| "OUT_DIR is not set")?);
    let root = manifest_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(&manifest.library.root);
    let root = root
        .canonicalize()
        .map_err(|e| format!("library root {}: {e}", root.display()))?;

    write_if_changed(&out.join("launch.rs"), &codegen::rust_bindings(&manifest))?;
    let shim = out.join("launch_shim.hip");
    write_if_changed(&shim, &codegen::cpp_shim(&manifest))?;

    let mut units: Vec<Source> = manifest
        .sources
        .iter()
        .map(|s| Source {
            path: root.join(&s.path),
            ..s.clone()
        })
        .collect();
    units.push(Source {
        path: shim,
        std: default_std(),
        flags: Vec::new(),
        defines: Vec::new(),
    });
    // The prefix gufo-hip links against, so compiler, headers and runtime
    // come from the same installation.
    let rocm = env::var_os("DEP_AMDHIP64_ROOT").map(PathBuf::from);
    let objects = compile::compile_all(&units, &root, &out.join("obj"), rocm.as_deref())?;
    link_static(
        &out,
        &format!("gufo_kernels_{}", manifest.library.name),
        &objects,
    )?;
    link_hip_runtime();
    Ok(())
}

/// Compiles hand-written host C++ of the calling crate (paths relative to
/// its manifest) against the ROCm prefix `rocm` and links it statically. For
/// gufo-hip's BLAS shim, whose build script knows the prefix itself.
pub fn compile_host_cpp(name: &str, sources: &[&str], rocm: &Path) {
    let result = (|| {
        let crate_dir = PathBuf::from(
            env::var("CARGO_MANIFEST_DIR").map_err(|_| "CARGO_MANIFEST_DIR is not set")?,
        );
        let out = PathBuf::from(env::var("OUT_DIR").map_err(|_| "OUT_DIR is not set")?);
        let units: Vec<Source> = sources
            .iter()
            .map(|source| Source {
                path: crate_dir.join(source),
                std: default_std(),
                // After `-x hip`: compile these inputs as host C++.
                flags: vec!["-x".into(), "c++".into()],
                defines: vec!["__HIP_PLATFORM_AMD__=1".into()],
            })
            .collect();
        println!("cargo:rerun-if-env-changed=HIPCC");
        let objects = compile::compile_all(
            &units,
            &crate_dir,
            &out.join(format!("obj-{name}")),
            Some(rocm),
        )?;
        link_static(&out, name, &objects)
    })();
    if let Err(message) = result {
        panic!("gufo-kernel-build: {message}");
    }
}

fn link_static(out: &Path, name: &str, objects: &[PathBuf]) -> Result<(), String> {
    let archive = out.join(format!("lib{name}.a"));
    let _ = fs::remove_file(&archive);
    let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
    let status = Command::new(&ar)
        .arg("crs")
        .arg(&archive)
        .args(objects)
        .status()
        .map_err(|e| format!("cannot run {ar}: {e}"))?;
    if !status.success() {
        return Err(format!("{ar} failed ({status})"));
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static={name}");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    Ok(())
}

/// Embeds the HIP runtime directory of gufo-hip's ROCm prefix as rpath in
/// this package's executables (binaries, tests, examples).
///
/// `cargo:rustc-link-arg` only reaches the package that prints it, so every
/// crate that produces an executable calls this from its build script and
/// depends on `gufo-hip` directly (Cargo passes `DEP_AMDHIP64_LIB` only to
/// direct dependents). Without it the loader may pick another installation's
/// `libamdhip64.so.7`, and `Stream::new` refuses to run.
pub fn link_hip_runtime() {
    match env::var("DEP_AMDHIP64_LIB") {
        Ok(lib) => println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}"),
        Err(_) => {
            panic!("gufo-kernel-build: DEP_AMDHIP64_LIB is not set; depend on gufo-hip directly")
        }
    }
}

fn write_if_changed(path: &Path, contents: &str) -> Result<(), String> {
    if fs::read_to_string(path).ok().as_deref() == Some(contents) {
        return Ok(());
    }
    fs::write(path, contents).map_err(|e| format!("{}: {e}", path.display()))
}
