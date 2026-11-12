//! Sustained memory bandwidth (`bandwidth.cpp`, `hip/bandwidth.cpp`): CPU
//! copy, read and write, and HIP host-to-device, device copy and
//! device-to-host transfers.

use std::fmt::Write;
use std::hint::black_box;
use std::time::Instant;

use gufo_hip::{Event, Stream};

use super::fingerprint::MachineFingerprint;
use super::inventory::escape;
use super::report::iso8601_utc;
use crate::bench::cpp_double;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandwidthOptions {
    pub backends: Vec<String>,
    pub warmup: u32,
    /// Minimum measured iterations per path.
    pub repetitions: u32,
    pub duration_ms: u32,
    pub working_set_bytes: usize,
}

impl Default for BandwidthOptions {
    fn default() -> Self {
        Self {
            backends: vec!["cpu".into(), "hip".into()],
            warmup: 3,
            repetitions: 10,
            duration_ms: 2000,
            working_set_bytes: 256 << 20,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PathResult {
    pub backend: String,
    pub path_name: String,
    pub allocation_type: String,
    pub working_set_bytes: usize,
    pub warmup_runs: u32,
    /// Actual measured iterations.
    pub repetitions: u64,
    /// Measured loop wall time, excluding warmup.
    pub elapsed_ms: f64,
    pub median_gbps: f64,
    pub p95_gbps: f64,
    pub min_gbps: f64,
    pub max_gbps: f64,
    pub raw_repetitions_gbps: Vec<f64>,
    pub sentinel_verified: bool,
    pub status: String,
    pub reason: String,
}

impl PathResult {
    fn new(backend: &str, path_name: &str, allocation_type: &str) -> Self {
        Self {
            backend: backend.into(),
            path_name: path_name.into(),
            allocation_type: allocation_type.into(),
            working_set_bytes: 0,
            warmup_runs: 0,
            repetitions: 0,
            elapsed_ms: 0.0,
            median_gbps: 0.0,
            p95_gbps: 0.0,
            min_gbps: 0.0,
            max_gbps: 0.0,
            raw_repetitions_gbps: Vec::new(),
            sentinel_verified: true,
            status: "completed".into(),
            reason: String::new(),
        }
    }

    fn measured(
        backend: &str,
        path_name: &str,
        allocation_type: &str,
        options: &BandwidthOptions,
    ) -> Self {
        Self {
            working_set_bytes: options.working_set_bytes,
            warmup_runs: options.warmup,
            ..Self::new(backend, path_name, allocation_type)
        }
    }

    fn unavailable(path_name: &str, allocation_type: &str, reason: String) -> Self {
        Self {
            status: "unavailable".into(),
            reason,
            ..Self::new("hip", path_name, allocation_type)
        }
    }

    fn compute_statistics(&mut self) {
        if self.raw_repetitions_gbps.is_empty() {
            return;
        }
        let mut sorted = self.raw_repetitions_gbps.clone();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        self.min_gbps = sorted[0];
        self.max_gbps = sorted[n - 1];
        self.median_gbps = if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let p95 = (0.95 * n as f64).ceil() as usize - 1;
        self.p95_gbps = sorted[p95.min(n - 1)];
    }
}

/// One path's measured loop (`BandwidthRun`), started after warmup: it runs
/// for both the requested duration and the minimum iteration count.
struct Run<'a> {
    options: &'a BandwidthOptions,
    started: Instant,
}

impl<'a> Run<'a> {
    fn new(options: &'a BandwidthOptions) -> Self {
        Self {
            options,
            started: Instant::now(),
        }
    }

    fn should_continue(&self, path: &mut PathResult) -> bool {
        path.elapsed_ms = self.started.elapsed().as_secs_f64() * 1e3;
        path.repetitions < u64::from(self.options.repetitions)
            || path.elapsed_ms < f64::from(self.options.duration_ms)
    }

    fn record(&self, path: &mut PathResult, seconds: f64, bytes: f64) {
        path.raw_repetitions_gbps
            .push(bytes / 1e9 / seconds.max(1e-9));
        path.repetitions += 1;
        path.elapsed_ms = self.started.elapsed().as_secs_f64() * 1e3;
    }
}

fn check_options(options: &BandwidthOptions) -> Result<(), String> {
    if options.repetitions == 0 || options.working_set_bytes == 0 {
        return Err("bandwidth requires nonzero repetitions and working set".into());
    }
    Ok(())
}

pub fn measure_cpu(options: &BandwidthOptions) -> Result<Vec<PathResult>, String> {
    check_options(options)?;
    let size = options.working_set_bytes;
    let source: Vec<u8> = (0..size).map(|i| (i.wrapping_mul(17) + 31) as u8).collect();
    let mut destination = vec![0u8; size];
    let mut results = Vec::new();

    let mut copy = PathResult::measured("cpu", "cpu_copy", "host_pageable", options);
    for _ in 0..options.warmup {
        destination.copy_from_slice(&source);
        black_box(&mut destination);
    }
    let mut sentinel = true;
    let run = Run::new(options);
    while run.should_continue(&mut copy) {
        destination.fill(0);
        black_box(&mut destination);
        let start = Instant::now();
        destination.copy_from_slice(black_box(&source));
        black_box(&mut destination);
        let seconds = start.elapsed().as_secs_f64();
        // A copy reads and writes the working set.
        run.record(&mut copy, seconds, size as f64 * 2.0);
        if destination[0] != source[0]
            || destination[size / 2] != source[size / 2]
            || destination[size - 1] != source[size - 1]
        {
            sentinel = false;
        }
    }
    copy.sentinel_verified = sentinel;
    copy.compute_statistics();
    results.push(copy);

    let mut read = PathResult::measured("cpu", "cpu_read", "host_pageable", options);
    let checksum = |bytes: &[u8]| bytes.iter().map(|&b| u64::from(b)).sum::<u64>();
    let mut sum = 0;
    for _ in 0..options.warmup {
        sum = black_box(checksum(black_box(&source)));
    }
    let run = Run::new(options);
    while run.should_continue(&mut read) {
        let start = Instant::now();
        sum = black_box(checksum(black_box(&source)));
        let seconds = start.elapsed().as_secs_f64();
        run.record(&mut read, seconds, size as f64);
    }
    read.sentinel_verified = sum != 0;
    read.compute_statistics();
    results.push(read);

    let mut write = PathResult::measured("cpu", "cpu_write", "host_pageable", options);
    for warmup in 0..options.warmup {
        destination.fill(warmup as u8);
        black_box(&mut destination);
    }
    let mut sentinel = true;
    let run = Run::new(options);
    while run.should_continue(&mut write) {
        let value = (write.repetitions + 0x5A) as u8;
        let start = Instant::now();
        destination.fill(value);
        black_box(&mut destination);
        let seconds = start.elapsed().as_secs_f64();
        run.record(&mut write, seconds, size as f64);
        if destination[0] != value || destination[size - 1] != value {
            sentinel = false;
        }
    }
    write.sentinel_verified = sentinel;
    write.compute_statistics();
    results.push(write);
    Ok(results)
}

/// Times `transfer` between two events recorded on `stream`.
fn timed(stream: &Stream, events: &(Event, Event), transfer: impl FnOnce()) -> f64 {
    let _ = stream.record(&events.0);
    transfer();
    let _ = stream.record(&events.1);
    let _ = events.1.synchronize();
    f64::from(Event::elapsed_ms(&events.0, &events.1).unwrap_or(0.0)) / 1000.0
}

pub fn measure_hip(options: &BandwidthOptions) -> Result<Vec<PathResult>, String> {
    check_options(options)?;
    match gufo_hip::device_count() {
        Ok(count) if count > 0 => {}
        result => {
            let reason = match result {
                Err(error) => format!(
                    "hipGetDeviceCount failed: {}",
                    gufo_hip::error_name(error.code)
                ),
                Ok(_) => "No HIP GPU device found".into(),
            };
            return Ok(vec![PathResult::unavailable(
                "hip_device_visible",
                "hip_device_memory",
                reason,
            )]);
        }
    }
    let size = options.working_set_bytes;
    let source: Vec<u8> = (0..size).map(|i| (i.wrapping_mul(31) + 17) as u8).collect();
    let mut destination = vec![0u8; size];
    let allocated = Stream::new().and_then(|stream| {
        let device_source = stream.alloc_persistent::<u8>(size)?;
        let device_destination = stream.alloc_persistent::<u8>(size)?;
        let events = (Event::timed()?, Event::timed()?);
        Ok((stream, device_source, device_destination, events))
    });
    let Ok((stream, mut device_source, mut device_destination, events)) = allocated else {
        return Ok(vec![PathResult::unavailable(
            "hip_device_memory",
            "",
            "hipMalloc failed for working set size".into(),
        )]);
    };
    let mut results = Vec::new();

    let mut h2d = PathResult::measured("hip", "hip_h2d", "hip_device_memory", options);
    for _ in 0..options.warmup {
        let _ = stream.upload(&source, device_source.as_mut_slice());
    }
    let run = Run::new(options);
    while run.should_continue(&mut h2d) {
        let seconds = timed(&stream, &events, || {
            let _ = stream.upload(&source, device_source.as_mut_slice());
        });
        run.record(&mut h2d, seconds, size as f64);
    }
    h2d.compute_statistics();
    results.push(h2d);

    let mut d2d = PathResult::measured("hip", "hip_device_copy", "hip_device_memory", options);
    for _ in 0..options.warmup {
        let _ = stream.copy(device_source.as_slice(), device_destination.as_mut_slice());
        let _ = stream.synchronize();
    }
    let run = Run::new(options);
    while run.should_continue(&mut d2d) {
        let seconds = timed(&stream, &events, || {
            let _ = stream.copy(device_source.as_slice(), device_destination.as_mut_slice());
        });
        // A device copy reads and writes the working set.
        run.record(&mut d2d, seconds, size as f64 * 2.0);
    }
    d2d.compute_statistics();
    results.push(d2d);

    let mut d2h = PathResult::measured("hip", "hip_d2h", "hip_device_memory", options);
    for _ in 0..options.warmup {
        let _ = stream.download(device_destination.as_slice(), &mut destination);
    }
    let mut sentinel = true;
    let run = Run::new(options);
    while run.should_continue(&mut d2h) {
        destination.fill(0);
        let seconds = timed(&stream, &events, || {
            let _ = stream.download(device_destination.as_slice(), &mut destination);
        });
        run.record(&mut d2h, seconds, size as f64);
        if destination[0] != source[0]
            || destination[size / 2] != source[size / 2]
            || destination[size - 1] != source[size - 1]
        {
            sentinel = false;
        }
    }
    d2h.sentinel_verified = sentinel;
    d2h.compute_statistics();
    results.push(d2h);
    Ok(results)
}

pub struct BandwidthReport {
    pub schema_version: String,
    pub fingerprint_id: String,
    pub engine_revision: String,
    pub timestamp: String,
    pub options: BandwidthOptions,
    pub paths: Vec<PathResult>,
}

pub fn run(
    options: &BandwidthOptions,
    fingerprint: &MachineFingerprint,
    engine_revision: &str,
) -> Result<BandwidthReport, String> {
    check_options(options)?;
    let mut report = BandwidthReport {
        schema_version: "1.0.0".into(),
        fingerprint_id: fingerprint.fingerprint_id.clone(),
        engine_revision: engine_revision.into(),
        timestamp: iso8601_utc(),
        options: options.clone(),
        paths: Vec::new(),
    };
    for backend in &options.backends {
        match backend.as_str() {
            "cpu" => report.paths.extend(measure_cpu(options)?),
            "hip" => report.paths.extend(measure_hip(options)?),
            _ => {}
        }
    }
    Ok(report)
}

/// `std::to_string(int(v)) + "." + std::to_string(int(v * 100) % 100)`.
fn truncated(value: f64) -> String {
    format!("{}.{}", value as i32, (value * 100.0) as i32 % 100)
}

impl BandwidthReport {
    pub fn to_json(&self) -> String {
        let mut out = format!(
            "{{\n  \"schemaVersion\": \"{}\",\n  \"fingerprintId\": \"{}\",\n  \
             \"engineRevision\": \"{}\",\n  \"timestamp\": \"{}\",\n  \"options\": {{\n    \
             \"warmup\": {},\n    \"repetitions\": {},\n    \"durationMs\": {},\n    \
             \"workingSetBytes\": {}\n  }},\n  \"paths\": [\n",
            escape(&self.schema_version),
            escape(&self.fingerprint_id),
            escape(&self.engine_revision),
            escape(&self.timestamp),
            self.options.warmup,
            self.options.repetitions,
            self.options.duration_ms,
            self.options.working_set_bytes
        );
        for (index, path) in self.paths.iter().enumerate() {
            // The C++ stream switches to fixed two-digit output at the first
            // path's median and stays there.
            let elapsed = if index == 0 {
                cpp_double(path.elapsed_ms)
            } else {
                format!("{:.2}", path.elapsed_ms)
            };
            let samples: Vec<String> = path
                .raw_repetitions_gbps
                .iter()
                .map(|sample| format!("{sample:.2}"))
                .collect();
            let _ = write!(
                out,
                "    {{\n      \"backend\": \"{}\",\n      \"pathName\": \"{}\",\n      \
                 \"allocationType\": \"{}\",\n      \"workingSetBytes\": {},\n      \
                 \"warmupRuns\": {},\n      \"repetitions\": {},\n      \"elapsedMs\": {},\n      \
                 \"medianGbps\": {:.2},\n      \"p95Gbps\": {:.2},\n      \"minGbps\": {:.2},\n      \
                 \"maxGbps\": {:.2},\n      \"sentinelVerified\": {},\n      \"status\": \"{}\",\n      \
                 \"reason\": \"{}\",\n      \"rawRepetitionsGbps\": [{}]\n    }}",
                escape(&path.backend),
                escape(&path.path_name),
                escape(&path.allocation_type),
                path.working_set_bytes,
                path.warmup_runs,
                path.repetitions,
                elapsed,
                path.median_gbps,
                path.p95_gbps,
                path.min_gbps,
                path.max_gbps,
                path.sentinel_verified,
                escape(&path.status),
                escape(&path.reason),
                samples.join(", ")
            );
            if index + 1 < self.paths.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ]\n}\n");
        out
    }

    pub fn to_human(&self) -> String {
        let mut out = format!(
            "=== Sustained Memory Bandwidth Benchmark ===\nSchema Version  : {}\n\
             Fingerprint ID  : {}\nEngine Revision : {}\nTimestamp       : {}\n\
             Working Set     : {} MiB\n\n",
            self.schema_version,
            self.fingerprint_id,
            self.engine_revision,
            self.timestamp,
            self.options.working_set_bytes / (1024 * 1024)
        );
        let _ = writeln!(
            out,
            "{:<8}{:<26}{:<14}{:<12}{:<12}{:<12}",
            "Backend", "Path", "Median GB/s", "P95 GB/s", "Max GB/s", "Status"
        );
        let _ = writeln!(out, "{}", "-".repeat(84));
        for path in &self.paths {
            let _ = write!(
                out,
                "{:<8}{:<26}{:<14}{:<12}{:<12}{:<12}",
                path.backend,
                path.path_name,
                truncated(path.median_gbps),
                truncated(path.p95_gbps),
                truncated(path.max_gbps),
                path.status
            );
            if !path.reason.is_empty() {
                let _ = write!(out, " ({})", path.reason);
            }
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::{fingerprint, validator};
    use super::*;

    fn options(repetitions: u32, duration_ms: u32, working_set_bytes: usize) -> BandwidthOptions {
        BandwidthOptions {
            backends: vec!["cpu".into()],
            warmup: 1,
            repetitions,
            duration_ms,
            working_set_bytes,
        }
    }

    /// `TestCpuBandwidthCorrectness` of the C++ bandwidth test.
    #[test]
    fn cpu_paths_are_correct() {
        let results = measure_cpu(&options(3, 0, 16 << 20)).expect("measured");
        assert_eq!(results.len(), 3);
        for path in &results {
            assert_eq!(
                (path.backend.as_str(), path.status.as_str()),
                ("cpu", "completed")
            );
            assert!(path.sentinel_verified);
            assert_eq!(path.raw_repetitions_gbps.len(), 3);
            assert!(path.median_gbps > 0.0);
            assert!(path.min_gbps <= path.median_gbps && path.median_gbps <= path.max_gbps);
            assert!(path.p95_gbps >= path.median_gbps);
        }
    }

    /// `TestBandwidthDuration`.
    #[test]
    fn paths_run_for_the_requested_duration() {
        for duration_ms in [1, 30] {
            let options = options(2, duration_ms, 1 << 20);
            for path in measure_cpu(&options).expect("measured") {
                assert!(path.elapsed_ms >= f64::from(duration_ms));
                assert!(path.repetitions >= 2);
                assert_eq!(path.raw_repetitions_gbps.len() as u64, path.repetitions);
                assert!(path.sentinel_verified && path.median_gbps > 0.0);
            }
        }
        assert!(measure_cpu(&options(0, 0, 1 << 20)).is_err());
    }

    /// `TestBandwidthReportJsonStructure` and `TestSchemaFileExists`.
    #[test]
    fn reports_are_valid_artifacts() {
        let fingerprint = fingerprint::generate(&Default::default());
        let report = run(&options(2, 0, 8 << 20), &fingerprint, "dev").expect("report");
        let json = report.to_json();
        assert!(json.contains("\"schemaVersion\": \"1.0.0\""));
        assert!(json.contains(&format!(
            "\"fingerprintId\": \"{}\"",
            fingerprint.fingerprint_id
        )));
        assert!(json.contains("\"paths\":") && json.contains("\"cpu_copy\""));
        assert!(json.contains("\"rawRepetitionsGbps\":"));
        assert!(
            validator::validate_content(json.as_bytes()).is_valid,
            "{json}"
        );
        let schema = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/diagnostics/bandwidth_schema.json");
        assert!(!std::fs::read_to_string(schema).expect("schema").is_empty());
    }

    #[test]
    fn numbers_render_like_the_cpp_stream() {
        let mut first = PathResult::new("cpu", "a", "host_pageable");
        first.elapsed_ms = 12.345_678_9;
        first.median_gbps = 12.056;
        let mut second = first.clone();
        second.raw_repetitions_gbps = vec![1.0, 2.346];
        let report = BandwidthReport {
            schema_version: "1.0.0".into(),
            fingerprint_id: "f".into(),
            engine_revision: "dev".into(),
            timestamp: "t".into(),
            options: options(1, 0, 1 << 20),
            paths: vec![first, second],
        };
        let json = report.to_json();
        // Six significant digits before the first fixed output, then two
        // decimals.
        assert!(
            json.contains("\"elapsedMs\": 12.3457,\n      \"medianGbps\": 12.06,"),
            "{json}"
        );
        assert!(
            json.contains("\"elapsedMs\": 12.35,\n      \"medianGbps\": 12.06,"),
            "{json}"
        );
        assert!(
            json.contains("\"rawRepetitionsGbps\": [1.00, 2.35]"),
            "{json}"
        );
        // No zero padding after the point, as in C++.
        let row = format!("{:<8}{:<26}{:<14}{:<12}", "cpu", "a", "12.5", "0.0");
        assert!(report.to_human().contains(&row), "{}", report.to_human());
    }
}
