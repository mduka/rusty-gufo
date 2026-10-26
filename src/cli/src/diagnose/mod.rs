//! `gufo diagnose` and `gufo probe` (`src/cli/diagnose/diagnose.cpp`) with
//! the diagnostics of `src/core/diagnostics`: inventory, compatibility,
//! machine fingerprint, artifact validation and the bandwidth benchmark.
//!
//! The toolchain fields and labels describe this build (rustc, the Rust
//! edition, the ROCm release of `gufo-hip`, the gufo revision), where C++
//! prints its fixed Nix pins; fingerprint IDs therefore differ from the C++
//! build's. Integer arguments C++ reads with `std::stoul` and whose value it
//! cannot convert end the command with a usage error instead of an uncaught
//! exception.

mod allocation;
mod bandwidth;
mod compatibility;
mod fingerprint;
mod inventory;
mod report;
mod sysfs;
mod validator;

use std::path::Path;
use std::process::ExitCode;

use crate::arg_parser::ArgParser;
use report::{Check, DiagnosticReport, Status};
use sysfs::LinuxSysfs;

const HELP: &str = "Usage: gufo diagnose [OPTIONS]\n\n\
Run non-interactive system, hardware, and benchmark diagnostics for Strix Halo.\n\n\
Options:\n  \
--json                       Emit structured JSON output conforming to diagnostics schema v1.0.0\n  \
--fingerprint                Generate canonical machine fingerprint and SHA-256 ID\n  \
--validate-artifact <path>   Validate a diagnostic/benchmark artifact against schema & fingerprint\n  \
--benchmark <name>           Run benchmark suite: bandwidth, allocation\n  \
--backends <csv>             Backends to benchmark (default: cpu,hip)\n  \
--warmup <n>                 Number of warmup iterations (default: 3)\n  \
--repetitions <n>            Minimum benchmark repetitions (default: 10)\n  \
--duration-ms <ms>           Minimum duration per bandwidth path in ms (default: 2000)\n  \
--working-set-mib <csv>      Allocation diagnostic sizes in MiB (default: 64)\n  \
--output <path>              Write command output to specified file\n  \
--section <name>             Diagnostic section to run: all, inventory, platform, gpu\n  \
-h, --help                   Print help information\n";

fn output_content(content: &str, output_file: &str) {
    print!("{content}");
    if !output_file.is_empty() && std::fs::write(output_file, content).is_err() {
        eprintln!("Warning: could not write output to file: {output_file}");
    }
}

fn split_comma_separated(csv: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut rest = csv;
    while !rest.is_empty() {
        match rest.split_once(',') {
            Some((item, tail)) => {
                result.push(item.to_string());
                rest = tail;
            }
            None => {
                result.push(rest.to_string());
                break;
            }
        }
    }
    result
}

/// `static_cast<std::uint32_t>(std::stoul(text))`: leading whitespace, an
/// optional sign (a negative value wraps as in `strtoul`) and decimal
/// digits, then anything; `None` where `std::stoul` throws.
fn stoul_u32(text: &str) -> Option<u32> {
    let text = text.trim_start_matches([' ', '\t', '\n', '\u{b}', '\u{c}', '\r']);
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let end = digits
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(digits.len());
    let value: u64 = digits[..end].parse().ok()?;
    Some(if negative {
        value.wrapping_neg()
    } else {
        value
    } as u32)
}

/// The report of `section`: inventory, compatibility, fingerprint and the
/// section's checks.
fn collect_diagnostics(sysfs: &LinuxSysfs, section: &str) -> DiagnosticReport {
    let inventory = inventory::collect(sysfs);
    let mut report = DiagnosticReport::new(crate::REVISION);
    report.set_compatibility(compatibility::evaluate(&inventory, section));
    report.set_fingerprint(fingerprint::generate(&inventory));

    if matches!(section, "all" | "platform" | "inventory") {
        let check = if inventory.cpu.architecture == "x86_64" {
            Check::new(
                "platform",
                Status::Pass,
                "Supported architecture: Linux x86-64",
            )
            .detail("targetArchitecture", "x86_64-linux")
        } else {
            report.add_error("Unsupported platform architecture detected");
            Check::new(
                "platform",
                Status::Fail,
                "Unsupported target platform: only Linux x86-64 is supported",
            )
        };
        report.add_check(check);
    }
    if matches!(section, "all" | "inventory") {
        report.add_check(
            Check::new(
                "system",
                Status::Pass,
                "Host operating system and kernel identified",
            )
            .detail("kernelRelease", &inventory.toolchain.kernel_release),
        );
        report.add_check(
            Check::new("cpu", Status::Pass, "CPU processor topology queried")
                .detail("modelName", &inventory.cpu.model_name)
                .detail("logicalCores", inventory.cpu.logical_cores)
                .detail("physicalCores", inventory.cpu.physical_cores),
        );
        let memory = if inventory.memory.is_unified {
            "Host unified memory pool identified"
        } else {
            "Host memory identified"
        };
        report.add_check(
            Check::new("memory", Status::Pass, memory)
                .detail("totalBytes", inventory.memory.total_bytes)
                .detail("availableBytes", inventory.memory.available_bytes)
                .detail("memoryType", &inventory.memory.memory_type),
        );
    }
    if matches!(section, "all" | "gpu" | "inventory") {
        let gpu = &inventory.gpu;
        let (status, message) = if gpu.architecture == "gfx1151" {
            (
                Status::Pass,
                "Detected AMD Strix Halo GPU (gfx1151)".to_string(),
            )
        } else {
            report.add_error(format!(
                "Unsupported GPU architecture: {}",
                gpu.architecture
            ));
            (
                Status::Fail,
                format!(
                    "Unsupported GPU architecture: expected gfx1151, detected {}",
                    gpu.architecture
                ),
            )
        };
        report.add_check(
            Check::new("gpu", status, message)
                .detail("name", &gpu.name)
                .detail("architecture", &gpu.architecture)
                .detail("computeUnits", gpu.compute_units)
                .detail("driverName", &gpu.driver_name)
                .detail("pciId", &gpu.pci_id)
                .detail("powerMode", &gpu.power_mode),
        );
    }
    if matches!(section, "all" | "inventory") {
        let toolchain = &inventory.toolchain;
        let (status, message) = if matches!(toolchain.amdgpu_status.as_str(), "loaded" | "built-in")
        {
            (
                Status::Pass,
                "Toolchain and driver stack verified".to_string(),
            )
        } else {
            let message = format!("amdgpu kernel driver: {}", toolchain.amdgpu_status);
            report.add_error(message.clone());
            (Status::Fail, message)
        };
        report.add_check(
            Check::new("toolchain", status, message)
                .detail("kernelRelease", &toolchain.kernel_release)
                .detail("amdgpuStatus", &toolchain.amdgpu_status)
                .detail("rocmVersion", &toolchain.rocm_version),
        );
    }
    report.set_inventory(inventory);
    report
}

fn usage_error(message: &str) -> ExitCode {
    eprintln!("Error: {message}");
    print!("{HELP}");
    ExitCode::from(2)
}

/// `gufo diagnose`; `args` starts at `diagnose` (or at the options for
/// `gufo help diagnose`).
pub fn run(args: &[String]) -> ExitCode {
    let mut json = false;
    let mut fingerprint_mode = false;
    let mut validate_artifact = String::new();
    let mut benchmark = String::new();
    let mut backends = "cpu,hip".to_string();
    let (mut warmup, mut repetitions, mut duration_ms) = (3u32, 10u32, 2000u32);
    let mut output = String::new();
    let mut section = "all".to_string();
    let mut working_set_mib = "64".to_string();

    let mut start = 0;
    if args
        .first()
        .is_some_and(|first| first == "gufo" || first.ends_with("/gufo"))
    {
        start = 1;
    }
    if args
        .get(start)
        .is_some_and(|command| command == "diagnose" || command == "probe")
    {
        start += 1;
    }
    let mut index = start;
    while index < args.len() {
        let arg = args[index].as_str();
        // An option's value from the next argument or after `=`.
        let mut value = |name: &str, missing: &str| -> Result<Option<String>, ExitCode> {
            if arg == name {
                index += 1;
                return args
                    .get(index)
                    .cloned()
                    .map(Some)
                    .ok_or_else(|| usage_error(missing));
            }
            Ok(arg
                .strip_prefix(name)
                .and_then(|rest| rest.strip_prefix('='))
                .map(str::to_string))
        };
        let integer =
            |text: String, missing: &str| stoul_u32(&text).ok_or_else(|| usage_error(missing));
        let parsed: Result<bool, ExitCode> = (|| {
            if arg == "--json" {
                json = true;
            } else if arg == "--fingerprint" {
                fingerprint_mode = true;
            } else if let Some(path) = value(
                "--validate-artifact",
                "--validate-artifact requires a path argument",
            )? {
                validate_artifact = path;
            } else if let Some(name) = value("--benchmark", "--benchmark requires an argument")? {
                benchmark = name;
            } else if let Some(csv) = value("--backends", "--backends requires an argument")? {
                backends = csv;
            } else if let Some(text) = value("--warmup", "--warmup requires an integer argument")? {
                warmup = integer(text, "--warmup requires an integer argument")?;
            } else if let Some(text) = value(
                "--repetitions",
                "--repetitions requires an integer argument",
            )? {
                repetitions = integer(text, "--repetitions requires an integer argument")?;
            } else if let Some(text) = value(
                "--duration-ms",
                "--duration-ms requires an integer argument",
            )? {
                duration_ms = integer(text, "--duration-ms requires an integer argument")?;
            } else if let Some(csv) = value(
                "--working-set-mib",
                "--working-set-mib requires a comma-separated size list",
            )? {
                working_set_mib = csv;
            } else if let Some(path) = value("--output", "--output requires a file path argument")?
            {
                output = path;
            } else if arg == "--help" || arg == "-h" {
                print!("{HELP}");
                return Ok(false);
            } else if let Some(name) = value("--section", "--section requires an argument")? {
                section = name;
            } else {
                eprintln!("Error: unknown diagnose option '{arg}'");
                print!("{HELP}");
                return Err(ExitCode::from(2));
            }
            Ok(true)
        })();
        match parsed {
            Ok(true) => index += 1,
            Ok(false) => return ExitCode::SUCCESS,
            Err(code) => return code,
        }
    }

    if !matches!(section.as_str(), "all" | "inventory" | "platform" | "gpu") {
        eprintln!("Error: unknown diagnostic section '{section}'");
        return ExitCode::from(2);
    }
    let render = |json_text: String, human: String| if json { json_text } else { human };

    if !validate_artifact.is_empty() {
        let result = validator::validate_file(Path::new(&validate_artifact));
        output_content(&render(result.to_json(), result.to_human()), &output);
        return if result.is_valid {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        };
    }

    if !benchmark.is_empty() {
        if !matches!(
            benchmark.as_str(),
            "bandwidth" | "allocation" | "hip-allocation"
        ) {
            eprintln!(
                "Error: unsupported benchmark '{benchmark}'. Supported: bandwidth, allocation"
            );
            return ExitCode::from(2);
        }
        if benchmark != "bandwidth" {
            return run_allocation(warmup, repetitions, &working_set_mib, json, &output);
        }
        let options = bandwidth::BandwidthOptions {
            backends: split_comma_separated(&backends),
            warmup,
            repetitions,
            duration_ms,
            ..bandwidth::BandwidthOptions::default()
        };
        if options.backends.is_empty() {
            eprintln!("Error: --backends requires at least one backend");
            return ExitCode::from(2);
        }
        if let Some(backend) = options.backends.iter().find(|b| *b != "cpu" && *b != "hip") {
            eprintln!("Error: unknown bandwidth backend '{backend}'");
            return ExitCode::from(2);
        }
        let inventory = inventory::collect(&LinuxSysfs::default());
        let fingerprint = fingerprint::generate(&inventory);
        return match bandwidth::run(&options, &fingerprint, crate::REVISION) {
            Ok(report) => {
                output_content(&render(report.to_json(), report.to_human()), &output);
                ExitCode::SUCCESS
            }
            Err(message) => {
                eprintln!("Error: {message}");
                ExitCode::from(2)
            }
        };
    }

    if fingerprint_mode {
        let fingerprint = fingerprint::generate(&inventory::collect(&LinuxSysfs::default()));
        output_content(
            &render(fingerprint.to_json(), fingerprint.to_human()),
            &output,
        );
        return ExitCode::SUCCESS;
    }

    let report = collect_diagnostics(&LinuxSysfs::default(), &section);
    output_content(&render(report.to_json(), report.to_human()), &output);
    if report.status() == Status::Fail {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

/// `gufo diagnose --benchmark allocation`: exit status 1 when a path fails.
fn run_allocation(
    warmup: u32,
    repetitions: u32,
    working_set_mib: &str,
    json: bool,
    output: &str,
) -> ExitCode {
    let inventory = inventory::collect(&LinuxSysfs::default());
    let fingerprint = fingerprint::generate(&inventory);
    if repetitions == 0 {
        eprintln!("Error: --repetitions must be at least 1 for the allocation diagnostic");
        return ExitCode::from(2);
    }
    let working_set_bytes = match allocation::parse_working_set_sizes(working_set_mib) {
        Ok(sizes) => sizes,
        Err(message) => {
            eprintln!("Error: {message}");
            return ExitCode::from(2);
        }
    };
    let options = allocation::AllocationOptions {
        working_set_bytes,
        warmup,
        repetitions,
        ..allocation::AllocationOptions::default()
    };
    let report = allocation::run(&options, &inventory, &fingerprint, crate::REVISION);
    let content = if json {
        report.to_json()
    } else {
        report.to_human()
    };
    output_content(&content, output);
    if report.success() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// `GUFO_REQUIRE_HIP`, else `GUFO_REQUIRE_GPU`, else `GUFO_REQUIRE_GFX1151`,
/// set to anything but `0` or `false`.
fn hip_required() -> bool {
    [
        "GUFO_REQUIRE_HIP",
        "GUFO_REQUIRE_GPU",
        "GUFO_REQUIRE_GFX1151",
    ]
    .iter()
    .find_map(std::env::var_os)
    .is_some_and(|value| value != "0" && value != "false")
}

/// Prints the HIP devices; `false` when a required device is missing.
fn check_gpu() -> bool {
    println!("ROCm version: {}", gufo_hip::BUILD_ROCM_VERSION);
    let hip = gufo_hip::runtime_version().unwrap_or(gufo_hip::BUILD_HIP_VERSION);
    println!(
        "HIP version: {}.{}.{}",
        hip / 10_000_000,
        hip / 100_000 % 100,
        hip % 100_000
    );
    let count = match gufo_hip::device_count() {
        Ok(count) => count,
        Err(error) => {
            eprintln!(
                "HIP error on hipGetDeviceCount: {}",
                gufo_hip::error_name(error.code)
            );
            if hip_required() {
                return false;
            }
            println!("SKIP: HIP device discovery failed");
            return true;
        }
    };
    println!("HIP device count: {count}");
    if count == 0 {
        println!("No HIP device found — is kfd loaded / device in render group?");
        return !hip_required();
    }
    for device in 0..count {
        let Ok(properties) = gufo_hip::device_properties(device) else {
            continue;
        };
        println!("device[{device}]: {}", properties.name);
        println!("  arch:        {}", properties.arch_name);
        println!("  computeCap:  {}.{}", properties.major, properties.minor);
        println!(
            "  totalMem:    {:.2} GiB (unified)",
            properties.total_global_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    true
}

/// `gufo probe`.
pub fn run_probe(args: &[String]) -> ExitCode {
    let mut parser = ArgParser::new("gufo probe", "Probe GPU availability.");
    if let Err(error) = parser.parse(args) {
        eprintln!("Error: {error}");
        return ExitCode::from(2);
    }
    if parser.is_help_requested() {
        parser.print_help();
        return ExitCode::SUCCESS;
    }
    println!("Hello, Strix Halo!");
    println!(
        "Build: Cargo, {}, ROCm/HIP enabled",
        inventory::LANGUAGE_STANDARD
    );
    if !check_gpu() {
        return ExitCode::from(1);
    }
    println!("OK");
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_follow_stoul() {
        for (text, expected) in [
            ("12", Some(12)),
            ("  7x", Some(7)),
            ("+3", Some(3)),
            ("-1", Some(u32::MAX)),
            ("4294967296", Some(0)),
            ("abc", None),
            ("", None),
            ("-", None),
            ("18446744073709551616", None),
        ] {
            assert_eq!(stoul_u32(text), expected, "{text:?}");
        }
    }

    #[test]
    fn comma_lists_split_like_cpp() {
        assert_eq!(split_comma_separated("cpu,hip"), ["cpu", "hip"]);
        assert_eq!(split_comma_separated("cpu,"), ["cpu"]);
        assert_eq!(split_comma_separated(",cpu"), ["", "cpu"]);
        assert!(split_comma_separated("").is_empty());
    }
}
