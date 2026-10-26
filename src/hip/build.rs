//! Resolves the single ROCm prefix of the build and links its HIP runtime.
//!
//! Order: `$ROCM_PATH`, `/opt/rocm/core` (AMD packages), `/usr` (distribution
//! packages). Both runtimes use the soname `libamdhip64.so.7`, so the prefix is
//! embedded as rpath; without it the loader may pick the other installation.
//! Build scripts of direct dependents receive the prefix as
//! `DEP_AMDHIP64_ROOT` and its library directory as `DEP_AMDHIP64_LIB`; the
//! rpath printed here only covers this crate's own tests, so executables call
//! `gufo_kernel_build::link_hip_runtime()`. The runtime checks the loaded HIP
//! version against the one recorded here.

use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    let root = env::var_os("ROCM_PATH")
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("/opt/rocm/core")).filter(|p| p.join("bin/hipcc").exists()))
        .unwrap_or_else(|| PathBuf::from("/usr"));
    let root = root.canonicalize().unwrap_or(root);
    let header = root.join("include/hip/hip_version.h");
    println!("cargo:rerun-if-changed={}", header.display());
    let version = fs::read_to_string(&header)
        .ok()
        .and_then(|text| hip_version(&text))
        .unwrap_or_else(|| {
            panic!(
                "gufo-hip: cannot read HIP version from {}",
                header.display()
            )
        });

    let lib = library_dir(&root);
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    // The BLAS and device-property shims and the libraries they call.
    gufo_kernel_build::compile_host_cpp(
        "gufo_hip_shim",
        &["shim/blas.cpp", "shim/device.cpp"],
        &root,
    );
    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!("cargo:rustc-link-lib=dylib=hipblaslt");
    println!("cargo:rustc-link-lib=dylib=rocblas");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo:root={}", root.display());
    println!("cargo:lib={}", lib.display());

    let rocm_header = root.join("include/rocm-core/rocm_version.h");
    let rocm_info = root.join(".info/version");
    println!("cargo:rerun-if-changed={}", rocm_header.display());
    println!("cargo:rerun-if-changed={}", rocm_info.display());
    let rocm_version = fs::read_to_string(&rocm_header)
        .ok()
        .and_then(|text| rocm_version(&text))
        .or_else(|| {
            fs::read_to_string(&rocm_info)
                .ok()
                .map(|text| text.trim().to_string())
                .filter(|text| !text.is_empty())
        })
        .unwrap_or_else(|| "unknown".into());

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let generated = format!(
        "/// ROCm prefix this crate was built against.\npub const BUILD_ROCM_ROOT: &str = {:?};\n\
         /// `HIP_VERSION` of that prefix (major * 10^7 + minor * 10^5 + patch).\n\
         pub const BUILD_HIP_VERSION: i32 = {version};\n\
         /// ROCm release of that prefix (`rocm-core/rocm_version.h`, else\n\
         /// `.info/version`), or \"unknown\".\n\
         pub const BUILD_ROCM_VERSION: &str = {rocm_version:?};\n",
        root.display().to_string()
    );
    fs::write(out.join("build_info.rs"), generated).expect("write build_info.rs");
}

fn library_dir(root: &Path) -> PathBuf {
    ["lib", "lib64", "lib/x86_64-linux-gnu"]
        .iter()
        .map(|dir| root.join(dir))
        .find(|dir| dir.join("libamdhip64.so").exists())
        .unwrap_or_else(|| panic!("gufo-hip: no libamdhip64.so under {}", root.display()))
}

/// `MAJOR.MINOR.PATCH` of `rocm_version.h`.
fn rocm_version(header: &str) -> Option<String> {
    let field = |name: &str| {
        header.lines().find_map(|line| {
            let rest = line.trim().strip_prefix("#define ")?.strip_prefix(name)?;
            rest.trim().parse::<u32>().ok()
        })
    };
    Some(format!(
        "{}.{}.{}",
        field("ROCM_VERSION_MAJOR ")?,
        field("ROCM_VERSION_MINOR ")?,
        field("ROCM_VERSION_PATCH ")?
    ))
}

fn hip_version(header: &str) -> Option<i32> {
    let field = |name: &str| {
        header.lines().find_map(|line| {
            let rest = line.trim().strip_prefix("#define ")?.strip_prefix(name)?;
            rest.trim().parse::<i32>().ok()
        })
    };
    Some(
        field("HIP_VERSION_MAJOR ")? * 10_000_000
            + field("HIP_VERSION_MINOR ")? * 100_000
            + field("HIP_VERSION_PATCH ")?,
    )
}
