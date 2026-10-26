//! The HIP allocation and placement diagnostic (`allocation_benchmark.cpp`,
//! `hip/allocation_benchmark.hip`): five allocation paths per working set,
//! each with separated phases (allocation, first touch, warm reads and
//! writes, explicit copies, managed advice, pool reuse, teardown), checksums
//! on the CPU and the GPU, and page residency evidence.

use std::fmt::Write;
use std::fs;
use std::time::Instant;

use gufo_hip::{DeviceBuffer, ErrorKind, Event, HipError, HostMemory, LaunchError, Stream};

use super::fingerprint::MachineFingerprint;
use super::inventory::SystemInventory;
use super::report::iso8601_utc;

const SAFETY_HEADROOM: usize = 4 << 30;
const PATTERN_SEED: u8 = 37;
const PATH_NAMES: [&str; 5] = [
    "hip_malloc",
    "hip_malloc_managed",
    "hip_host_malloc_mapped",
    "file_mmap_host_register",
    "hip_malloc_async",
];
const HIP_ERROR_INVALID_VALUE: i32 = 1;
const HIP_ERROR_OUT_OF_MEMORY: i32 = 2;
const HIP_ERROR_NOT_SUPPORTED: i32 = 801;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocationOptions {
    pub working_set_bytes: Vec<usize>,
    pub warmup: u32,
    pub repetitions: u32,
    pub transfer_chunk_bytes: usize,
}

impl Default for AllocationOptions {
    fn default() -> Self {
        Self {
            working_set_bytes: vec![64 << 20],
            warmup: 1,
            repetitions: 5,
            transfer_chunk_bytes: 64 << 20,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Faults {
    pub minor: u64,
    pub major: u64,
}

impl Faults {
    fn add(&mut self, other: Faults) {
        self.minor += other.minor;
        self.major += other.major;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Phase {
    pub name: String,
    pub operation: String,
    pub unit: String,
    pub status: String,
    pub reason: String,
    pub raw_repetitions: Vec<f64>,
    pub median: f64,
    pub p95: f64,
    pub minimum: f64,
    pub maximum: f64,
    pub faults: Faults,
}

impl Phase {
    fn new(name: &str, operation: &str, unit: &str) -> Self {
        Self {
            name: name.into(),
            operation: operation.into(),
            unit: unit.into(),
            status: "completed".into(),
            reason: String::new(),
            raw_repetitions: Vec::new(),
            median: 0.0,
            p95: 0.0,
            minimum: 0.0,
            maximum: 0.0,
            faults: Faults::default(),
        }
    }

    /// `ComputeAllocationPhaseStatistics`.
    pub fn compute_statistics(&mut self) {
        if self.raw_repetitions.is_empty() {
            return;
        }
        let mut sorted = self.raw_repetitions.clone();
        sorted.sort_by(f64::total_cmp);
        let count = sorted.len();
        self.minimum = sorted[0];
        self.maximum = sorted[count - 1];
        self.median = if count.is_multiple_of(2) {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };
        let index = ((0.95 * count as f64).ceil() as usize).saturating_sub(1);
        self.p95 = sorted[index.min(count - 1)];
    }

    fn fail(&mut self, error: &HipError) {
        self.status = failure_status(error).into();
        self.reason = hip_reason(error);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PathResult {
    pub path_name: String,
    pub allocation_api: String,
    pub flags: String,
    pub status: String,
    pub reason: String,
    pub pointer_memory_type: String,
    pub residency_evidence: String,
    pub placement_inference: String,
    pub working_set_bytes: usize,
    pub expected_checksum: u64,
    pub cpu_checksum: u64,
    pub gpu_checksum: u64,
    pub checksum_verified: bool,
    pub managed: bool,
    pub phases: Vec<Phase>,
}

impl PathResult {
    fn new(path_name: &str, working_set_bytes: usize) -> Self {
        Self {
            path_name: path_name.into(),
            allocation_api: String::new(),
            flags: String::new(),
            status: "completed".into(),
            reason: String::new(),
            pointer_memory_type: String::new(),
            residency_evidence: String::new(),
            placement_inference: String::new(),
            working_set_bytes,
            expected_checksum: 0,
            cpu_checksum: 0,
            gpu_checksum: 0,
            checksum_verified: false,
            managed: false,
            phases: Vec::new(),
        }
    }

    fn has_phase(&self, name: &str) -> bool {
        self.phases.iter().any(|phase| phase.name == name)
    }

    /// `MergePhaseSample`.
    fn merge(&mut self, sample: &Phase) {
        let Some(phase) = self.phases.iter_mut().find(|p| p.name == sample.name) else {
            self.phases.push(sample.clone());
            return;
        };
        phase
            .raw_repetitions
            .extend_from_slice(&sample.raw_repetitions);
        phase.faults.add(sample.faults);
        if sample.status != "completed" {
            phase.status.clone_from(&sample.status);
            phase.reason.clone_from(&sample.reason);
        }
        phase.compute_statistics();
    }
}

#[derive(Debug, Clone)]
pub struct AllocationReport {
    pub schema_version: String,
    pub artifact_type: String,
    pub fingerprint_id: String,
    pub engine_revision: String,
    pub timestamp: String,
    pub options: AllocationOptions,
    pub inventory: SystemInventory,
    pub fingerprint: MachineFingerprint,
    pub page_size_bytes: u64,
    pub huge_pages_total: u64,
    pub huge_page_size_bytes: u64,
    pub paths: Vec<PathResult>,
    pub limitations: Vec<String>,
    pub all_requested_paths_reported: bool,
    pub checksums_verified: bool,
    pub phases_separated: bool,
}

/// `EscapeJson` of the allocation report: quotes, backslashes, newlines,
/// carriage returns and tabs.
fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(character),
        }
    }
    escaped
}

/// A double as `std::ostream` writes it after `std::setprecision(12)`.
fn number(value: f64) -> String {
    crate::bench::cpp_general(value, 12)
}

fn list<T>(values: &[T], format: impl Fn(&T) -> String) -> String {
    values.iter().map(format).collect::<Vec<_>>().join(", ")
}

impl AllocationReport {
    pub fn success(&self) -> bool {
        self.all_requested_paths_reported && self.checksums_verified && self.phases_separated
    }

    pub fn to_json(&self) -> String {
        let flag = |value: bool| if value { "true" } else { "false" };
        let (inventory, options) = (&self.inventory, &self.options);
        let mut out = String::new();
        let _ = write!(
            out,
            "{{\n  \"schemaVersion\": \"{}\",\n  \"artifactType\": \"{}\",\n  \
             \"fingerprintId\": \"{}\",\n  \"engineRevision\": \"{}\",\n  \
             \"timestamp\": \"{}\",\n  \"summary\": {{\n    \
             \"allRequestedPathsReported\": {},\n    \"checksumsVerified\": {},\n    \
             \"phasesSeparated\": {},\n    \"status\": \"{}\"\n  }},\n  \"options\": {{\n    \
             \"warmup\": {},\n    \"repetitions\": {},\n    \"transferChunkBytes\": {},\n    \
             \"workingSetBytes\": [{}]\n  }},\n",
            escape(&self.schema_version),
            escape(&self.artifact_type),
            escape(&self.fingerprint_id),
            escape(&self.engine_revision),
            escape(&self.timestamp),
            flag(self.all_requested_paths_reported),
            flag(self.checksums_verified),
            flag(self.phases_separated),
            if self.success() {
                "completed"
            } else {
                "failed"
            },
            options.warmup,
            options.repetitions,
            options.transfer_chunk_bytes,
            list(&options.working_set_bytes, ToString::to_string)
        );
        let _ = write!(
            out,
            "  \"machine\": {{\n    \"kernelRelease\": \"{}\",\n    \"rocmVersion\": \"{}\",\n    \
             \"gpuArchitecture\": \"{}\",\n    \"gpuComputeUnits\": {},\n    \
             \"gpuPowerMode\": \"{}\",\n    \"gpuClockState\": \"{}\",\n    \
             \"memoryTotalBytes\": {},\n    \"memoryAvailableBytes\": {},\n    \
             \"memoryType\": \"{}\",\n    \"pageSizeBytes\": {},\n    \"hugePagesTotal\": {},\n    \
             \"hugePageSizeBytes\": {}\n  }},\n  \"fingerprint\": {},\n  \"paths\": [\n",
            escape(&inventory.toolchain.kernel_release),
            escape(&inventory.toolchain.rocm_version),
            escape(&inventory.gpu.architecture),
            inventory.gpu.compute_units,
            escape(&inventory.gpu.power_mode),
            escape(&inventory.gpu.clock_state),
            inventory.memory.total_bytes,
            inventory.memory.available_bytes,
            escape(&inventory.memory.memory_type),
            self.page_size_bytes,
            self.huge_pages_total,
            self.huge_page_size_bytes,
            self.fingerprint.to_json()
        );
        for (path_index, path) in self.paths.iter().enumerate() {
            let _ = write!(
                out,
                "    {{\n      \"pathName\": \"{}\",\n      \"allocationApi\": \"{}\",\n      \
                 \"flags\": \"{}\",\n      \"workingSetBytes\": {},\n      \"status\": \"{}\",\n      \
                 \"reason\": \"{}\",\n      \"pointerMemoryType\": \"{}\",\n      \
                 \"managed\": {},\n      \"residencyEvidence\": \"{}\",\n      \
                 \"placementInference\": \"{}\",\n      \"expectedChecksum\": {},\n      \
                 \"cpuChecksum\": {},\n      \"gpuChecksum\": {},\n      \
                 \"checksumVerified\": {},\n      \"phases\": [\n",
                escape(&path.path_name),
                escape(&path.allocation_api),
                escape(&path.flags),
                path.working_set_bytes,
                escape(&path.status),
                escape(&path.reason),
                escape(&path.pointer_memory_type),
                flag(path.managed),
                escape(&path.residency_evidence),
                escape(&path.placement_inference),
                path.expected_checksum,
                path.cpu_checksum,
                path.gpu_checksum,
                flag(path.checksum_verified)
            );
            for (phase_index, phase) in path.phases.iter().enumerate() {
                let _ = write!(
                    out,
                    "          {{\n            \"name\": \"{}\",\n            \
                     \"operation\": \"{}\",\n            \"unit\": \"{}\",\n            \
                     \"status\": \"{}\",\n            \"reason\": \"{}\",\n            \
                     \"median\": {},\n            \"p95\": {},\n            \"minimum\": {},\n            \
                     \"maximum\": {},\n            \"minorFaults\": {},\n            \
                     \"majorFaults\": {},\n            \"rawRepetitions\": [{}]\n          }}",
                    escape(&phase.name),
                    escape(&phase.operation),
                    escape(&phase.unit),
                    escape(&phase.status),
                    escape(&phase.reason),
                    number(phase.median),
                    number(phase.p95),
                    number(phase.minimum),
                    number(phase.maximum),
                    phase.faults.minor,
                    phase.faults.major,
                    list(&phase.raw_repetitions, |value| number(*value))
                );
                if phase_index + 1 != path.phases.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str("      ]\n    }");
            if path_index + 1 != self.paths.len() {
                out.push(',');
            }
            out.push('\n');
        }
        let _ = write!(
            out,
            "  ],\n  \"limitations\": [{}]\n}}\n",
            list(&self.limitations, |text| format!("\"{}\"", escape(text)))
        );
        out
    }

    pub fn to_human(&self) -> String {
        let inventory = &self.inventory;
        let mut out = String::new();
        let _ = write!(
            out,
            "=== HIP Allocation and Placement Diagnostic ===\n\
             Status          : {}\n\
             Fingerprint ID  : {}\n\
             Engine Revision : {}\n\
             ROCm / Kernel   : {} / {}\n\
             GPU / Power     : {} / {}\n\
             Pages           : {} B base, {} huge pages @ {} B\n\n",
            if self.success() {
                "completed"
            } else {
                "failed"
            },
            self.fingerprint_id,
            self.engine_revision,
            inventory.toolchain.rocm_version,
            inventory.toolchain.kernel_release,
            inventory.gpu.architecture,
            inventory.gpu.power_mode,
            self.page_size_bytes,
            self.huge_pages_total,
            self.huge_page_size_bytes
        );
        for path in &self.paths {
            let _ = write!(
                out,
                "{} @ {} MiB: {}",
                path.path_name,
                path.working_set_bytes >> 20,
                path.status
            );
            if !path.reason.is_empty() {
                let _ = write!(out, " ({})", path.reason);
            }
            let _ = writeln!(
                out,
                "\n  api={} flags={} pointer={} checksum={}",
                path.allocation_api,
                path.flags,
                path.pointer_memory_type,
                if path.checksum_verified {
                    "verified"
                } else {
                    "n/a"
                }
            );
            for phase in &path.phases {
                let _ = write!(
                    out,
                    "    {:<26}{:>12.3} {}  {}",
                    phase.name, phase.median, phase.unit, phase.status
                );
                if !phase.reason.is_empty() {
                    let _ = write!(out, " ({})", phase.reason);
                }
                out.push('\n');
            }
        }
        if !self.limitations.is_empty() {
            out.push_str("\nInference limits:\n");
            for limitation in &self.limitations {
                let _ = writeln!(out, "  - {limitation}");
            }
        }
        out
    }
}

/// `ParseWorkingSetSizes`: positive comma-separated MiB counts.
pub fn parse_working_set_sizes(text: &str) -> Result<Vec<usize>, String> {
    let invalid = || "working-set sizes must be positive comma-separated MiB".to_string();
    if text.is_empty() || text.starts_with(',') || text.ends_with(',') {
        return Err(invalid());
    }
    let mut sizes = Vec::new();
    for field in text.split(',') {
        // `std::from_chars` into `uint64_t`: digits only, the whole field.
        let mib: usize = if !field.is_empty() && field.bytes().all(|b| b.is_ascii_digit()) {
            field.parse().map_err(|_| invalid())?
        } else {
            return Err(invalid());
        };
        if mib == 0 || mib > usize::MAX / (1 << 20) {
            return Err(invalid());
        }
        sizes.push(mib << 20);
    }
    Ok(sizes)
}

/// `hipGetErrorName: hipGetErrorString` of a HIP failure; gufo's own
/// refusals keep their message.
fn hip_reason(error: &HipError) -> String {
    match error.kind {
        ErrorKind::Runtime | ErrorKind::Poisoned => format!(
            "{}: {}",
            gufo_hip::error_name(error.code),
            gufo_hip::error_string(error.code)
        ),
        _ => error.to_string(),
    }
}

fn launch_error(error: LaunchError) -> HipError {
    match error {
        LaunchError::Hip(error) => error,
        _ => HipError {
            code: 0,
            operation: "launch outside the kernel contract",
            kind: ErrorKind::Refused,
        },
    }
}

/// `IsUnsupported`: not supported or an invalid value.
fn failure_status(error: &HipError) -> &'static str {
    let unsupported = matches!(error.kind, ErrorKind::Runtime)
        && matches!(
            error.code,
            HIP_ERROR_NOT_SUPPORTED | HIP_ERROR_INVALID_VALUE
        );
    if unsupported {
        "unsupported"
    } else {
        "unavailable"
    }
}

/// Minor and major page faults of the process (`getrusage` in C++; the
/// same process-wide counters from `/proc/self/stat`).
fn read_faults() -> Faults {
    let Ok(stat) = fs::read_to_string("/proc/self/stat") else {
        return Faults::default();
    };
    // Fields after the command name, which may hold spaces: state is the
    // first, minflt the seventh and majflt the ninth.
    let Some((_, rest)) = stat.rsplit_once(')') else {
        return Faults::default();
    };
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let field = |index: usize| fields.get(index).and_then(|v| v.parse().ok()).unwrap_or(0);
    Faults {
        minor: field(7),
        major: field(9),
    }
}

fn fault_delta(before: Faults, after: Faults) -> Faults {
    Faults {
        minor: after.minor.saturating_sub(before.minor),
        major: after.major.saturating_sub(before.major),
    }
}

/// `ReadMeminfoValue`: bytes for a `kB` value, else the plain number; 0 when
/// the key is missing.
fn meminfo_value(key: &str) -> u64 {
    let Ok(meminfo) = fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in meminfo.lines() {
        let mut fields = line.split_whitespace();
        if fields.next() != Some(key) {
            continue;
        }
        let Some(value) = fields.next().and_then(|v| v.parse::<u64>().ok()) else {
            continue;
        };
        return if fields.next() == Some("kB") {
            value.wrapping_mul(1024)
        } else {
            value
        };
    }
    0
}

fn expected_checksum(size: usize, seed: u8) -> u64 {
    let cycle: u64 = (0..256u32)
        .map(|i| u64::from((i as u8).wrapping_add(seed)))
        .sum();
    let mut total = (size / 256) as u64 * cycle;
    for index in 0..size % 256 {
        total += u64::from((index as u8).wrapping_add(seed));
    }
    total
}

fn fill_cpu(bytes: &mut [u8], seed: u8) {
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_add(seed);
    }
}

fn checksum_cpu(bytes: &[u8]) -> u64 {
    bytes.iter().map(|&byte| u64::from(byte)).sum()
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// `MeasureHipCall`: one timed call with its faults.
fn measure_call<T>(
    phase: &mut Phase,
    call: impl FnOnce() -> Result<T, HipError>,
) -> Result<T, HipError> {
    let before = read_faults();
    let start = Instant::now();
    let result = call();
    phase.raw_repetitions.push(milliseconds(start));
    phase.faults = fault_delta(before, read_faults());
    if let Err(error) = &result {
        phase.fail(error);
    }
    phase.compute_statistics();
    result
}

/// `MeasureCpuCall`.
fn measure_cpu(phase: &mut Phase, call: impl FnOnce()) {
    let before = read_faults();
    let start = Instant::now();
    call();
    phase.raw_repetitions.push(milliseconds(start));
    phase.faults = fault_delta(before, read_faults());
    phase.compute_statistics();
}

/// `MeasureGpuOperation`: event-timed throughput of `launch` over `bytes`
/// and the wait for it.
fn measure_gpu(
    throughput: &mut Phase,
    synchronization: &mut Phase,
    bytes: usize,
    stream: &Stream,
    launch: impl FnOnce() -> Result<(), HipError>,
) -> bool {
    let (Ok(start), Ok(stop)) = (Event::timed(), Event::timed()) else {
        throughput.status = "unavailable".into();
        throughput.reason = "hipEventCreate failed".into();
        synchronization.status = "unavailable".into();
        synchronization.reason.clone_from(&throughput.reason);
        return false;
    };
    let enqueued = stream
        .record(&start)
        .and_then(|()| launch())
        .and_then(|()| stream.record(&stop));
    let before = read_faults();
    let wait = Instant::now();
    let waited = enqueued.and_then(|()| stop.synchronize());
    synchronization.raw_repetitions.push(milliseconds(wait));
    synchronization
        .faults
        .add(fault_delta(before, read_faults()));
    if let Err(error) = waited {
        throughput.fail(&error);
        synchronization.status.clone_from(&throughput.status);
        synchronization.reason.clone_from(&throughput.reason);
        synchronization.compute_statistics();
        return false;
    }
    let elapsed = Event::elapsed_ms(&start, &stop);
    match elapsed {
        Ok(ms) if ms > 0.0 => {
            let seconds = f64::from(ms) / 1000.0;
            throughput
                .raw_repetitions
                .push(bytes as f64 / 1.0e9 / seconds);
            throughput.compute_statistics();
            synchronization.compute_statistics();
            true
        }
        other => {
            throughput.status = "unavailable".into();
            throughput.reason = match other {
                Ok(_) => "HIP event reported a non-positive duration".into(),
                Err(error) => hip_reason(&error),
            };
            synchronization.compute_statistics();
            false
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Malloc,
    Managed,
    HostMapped,
    FileMapped,
    Async,
}

const KINDS: [Kind; 5] = [
    Kind::Malloc,
    Kind::Managed,
    Kind::HostMapped,
    Kind::FileMapped,
    Kind::Async,
];

/// One allocation under test (`AllocationResource`) with the GPU checksum
/// scalar of its stream.
struct Resource {
    buffer: Option<DeviceBuffer<u8>>,
    sum: Option<DeviceBuffer<u64>>,
    cpu_accessible: bool,
    managed: bool,
    stream: Stream,
}

impl Resource {
    fn buffer(&self) -> &DeviceBuffer<u8> {
        self.buffer
            .as_ref()
            .expect("the buffer exists until cleanup")
    }

    fn buffer_mut(&mut self) -> &mut DeviceBuffer<u8> {
        self.buffer
            .as_mut()
            .expect("the buffer exists until cleanup")
    }

    /// `Cleanup`: releases the memory (the free queue synchronises the
    /// device) and the stream. The checksum scalar goes too, because the
    /// stream lives as long as a buffer of it (C++ frees it afterwards).
    fn cleanup(self) {
        let Resource {
            buffer,
            sum,
            stream,
            ..
        } = self;
        drop(buffer);
        let _ = stream.synchronize();
        drop(sum);
        gufo_hip::collect_deferred_frees();
        drop(stream);
    }
}

/// The device kernels of the diagnostic at the device's width
/// (`KernelBlocks`: eight blocks per multiprocessor, else 160).
struct Kernels {
    blocks: u32,
}

impl Kernels {
    fn new() -> Self {
        let blocks = gufo_hip::device_properties(0)
            .ok()
            .filter(|p| p.multiprocessor_count > 0)
            .map_or(160, |p| {
                (p.multiprocessor_count as u32).saturating_mul(8).max(1)
            });
        Self { blocks }
    }

    fn fill(&self, stream: &Stream, buffer: &mut DeviceBuffer<u8>) -> Result<(), HipError> {
        let size = buffer.len();
        gufo_cli_kernels::fill_pattern(
            stream,
            buffer.as_mut_slice(),
            size,
            u32::from(PATTERN_SEED),
            self.blocks,
        )
        .map_err(launch_error)
    }

    /// Clears `sum` and adds every byte of `buffer` into it.
    fn checksum(
        &self,
        stream: &Stream,
        buffer: &DeviceBuffer<u8>,
        sum: &mut DeviceBuffer<u64>,
    ) -> Result<(), HipError> {
        stream.fill_bytes(sum.as_mut_slice(), 0)?;
        gufo_cli_kernels::checksum(
            stream,
            buffer.as_slice(),
            sum.as_mut_slice(),
            buffer.len(),
            self.blocks,
        )
        .map_err(launch_error)
    }

    /// `VerifyGpuChecksum`.
    fn verify(&self, resource: &Resource, sum: &mut DeviceBuffer<u64>) -> Option<u64> {
        self.checksum(&resource.stream, resource.buffer(), sum)
            .ok()?;
        resource.stream.synchronize().ok()?;
        let mut value = [0u64];
        resource.stream.download(sum.as_slice(), &mut value).ok()?;
        Some(value[0])
    }
}

fn mark_creation_failure(result: &mut PathResult, error: &HipError) {
    result.status = failure_status(error).into();
    result.reason = hip_reason(error);
    result.checksum_verified = false;
}

/// `CreateResource`: allocates `size` bytes the `kind` way and records its
/// setup phases; `None` after a failure recorded in `result`.
fn create_resource(kind: Kind, size: usize, result: &mut PathResult) -> Option<Resource> {
    let mut allocation = Phase::new("allocation", "allocate untouched working set", "ms");
    let mut setup = Vec::new();
    let stream = match Stream::new() {
        Ok(stream) => stream,
        Err(error) => {
            allocation.status = "unavailable".into();
            allocation.reason = hip_reason(&error);
            result.phases.insert(0, allocation);
            mark_creation_failure(result, &error);
            return None;
        }
    };
    let (mut cpu_accessible, mut managed) = (false, false);
    let created: Result<DeviceBuffer<u8>, HipError> = match kind {
        Kind::Malloc => {
            result.allocation_api = "hipMalloc".into();
            result.flags = "none".into();
            result.placement_inference = "HIP reports device memory. On this integrated APU that \
                 does not identify a distinct physical memory pool or prove residency."
                .into();
            measure_call(&mut allocation, || stream.alloc_persistent::<u8>(size))
        }
        Kind::Managed => {
            result.allocation_api = "hipMallocManaged".into();
            result.flags = "hipMemAttachGlobal".into();
            result.placement_inference = "HIP reports managed memory. Pointer attributes and \
                 successful prefetch do not reveal physical page placement."
                .into();
            let buffer = measure_call(&mut allocation, || stream.alloc_managed::<u8>(size));
            (cpu_accessible, managed) = (buffer.is_ok(), buffer.is_ok());
            buffer
        }
        Kind::HostMapped => {
            result.allocation_api = "hipHostMalloc + hipHostGetDevicePointer".into();
            result.flags = "hipHostMallocMapped".into();
            result.placement_inference = "HIP reports host memory mapped into GPU virtual address \
                 space. This does not expose DRAM bank or cache residency."
                .into();
            match measure_call(&mut allocation, || HostMemory::pinned(size, true)) {
                Ok(memory) => {
                    cpu_accessible = true;
                    let mut alias = Phase::new("device_alias", "hipHostGetDevicePointer", "ms");
                    let buffer = measure_call(&mut alias, || {
                        memory.into_device(&stream).map_err(|(_, error)| error)
                    });
                    setup.push(alias);
                    buffer
                }
                Err(error) => Err(error),
            }
        }
        Kind::FileMapped => {
            result.allocation_api = "memfd_create + mmap(MAP_SHARED) + hipHostRegister".into();
            result.flags = "PROT_READ|PROT_WRITE, MAP_SHARED, hipHostRegisterMapped".into();
            result.placement_inference =
                "mincore reports Linux page-cache residency for the memfd \
                 mapping; HIP exposes only a mapped host alias and no physical GPU residency."
                    .into();
            let before = read_faults();
            let start = Instant::now();
            let memory = HostMemory::memfd("gufo-allocation-benchmark", size);
            allocation.raw_repetitions.push(milliseconds(start));
            allocation.faults = fault_delta(before, read_faults());
            allocation.compute_statistics();
            match memory {
                Ok(mut memory) => {
                    cpu_accessible = true;
                    let mut registration = Phase::new(
                        "registration",
                        "hipHostRegister(mapped) + hipHostGetDevicePointer",
                        "ms",
                    );
                    let buffer = measure_call(&mut registration, || {
                        memory.register()?;
                        memory.into_device(&stream).map_err(|(_, error)| error)
                    });
                    setup.push(registration);
                    buffer
                }
                Err(_) => {
                    allocation.status = "unavailable".into();
                    allocation.reason = "memfd_create/ftruncate/mmap failed".into();
                    Err(HipError {
                        code: HIP_ERROR_OUT_OF_MEMORY,
                        operation: "memfd_create",
                        kind: ErrorKind::Runtime,
                    })
                }
            }
        }
        Kind::Async => {
            result.allocation_api = "hipMallocAsync".into();
            result.flags = "default device memory pool".into();
            result.placement_inference = "hipMallocAsync is measured as allocation/reuse policy \
                 only; it is not claimed as a distinct physical placement from hipMalloc."
                .into();
            measure_call(&mut allocation, || {
                let buffer = stream.alloc::<u8>(size)?;
                stream.synchronize()?;
                Ok(buffer)
            })
        }
    };
    result.phases.insert(0, allocation);
    result.phases.extend(setup);
    let buffer = match created {
        Ok(buffer) => buffer,
        Err(error) => {
            mark_creation_failure(result, &error);
            return None;
        }
    };
    match buffer.pointer_attributes() {
        Ok(attributes) => {
            result.pointer_memory_type = attributes.memory_type.name().into();
            result.managed = attributes.managed;
        }
        Err(error) => result.pointer_memory_type = format!("unavailable: {}", hip_reason(&error)),
    }
    Some(Resource {
        buffer: Some(buffer),
        sum: None,
        cpu_accessible,
        managed,
        stream,
    })
}

fn is_setup_phase(name: &str) -> bool {
    matches!(
        name,
        "allocation" | "device_alias" | "registration" | "teardown"
    )
}

/// `MincoreEvidence` of the resource's host mapping.
fn residency_evidence(resource: &Resource) -> String {
    match resource.buffer().host_residency() {
        None => "mincore unavailable: no host mapping".into(),
        Some(Err(message)) => message,
        Some(Ok(residency)) => {
            format!("{}/{} pages resident", residency.resident, residency.pages)
        }
    }
}

/// `AddExplicitCopyPhases`: chunked host-to-device and device-to-host
/// copies through bounded staging buffers.
fn add_explicit_copies(
    resource: &mut Resource,
    size: usize,
    options: &AllocationOptions,
    result: &mut PathResult,
) {
    let chunk = size.min(options.transfer_chunk_bytes).max(1);
    let stream = &resource.stream;
    let staging = stream.alloc_persistent::<u8>(chunk).and_then(|device| {
        let host = HostMemory::pinned(chunk, false)?
            .into_device(stream)
            .map_err(|(_, error)| error)?;
        Ok((device, host))
    });
    let Ok((mut device_staging, mut host_staging)) = staging else {
        let mut copy = Phase::new("explicit_copy", "bounded chunked hipMemcpyAsync", "GB/s");
        copy.status = "unavailable".into();
        copy.reason = "failed to allocate bounded transfer staging buffers".into();
        result.phases.push(copy);
        return;
    };
    if let Ok(host) = host_staging.host_mut() {
        host.fill(0);
    }
    let offsets = || {
        (0..size)
            .step_by(chunk)
            .map(move |o| (o, chunk.min(size - o)))
    };

    let mut h2d = Phase::new(
        "explicit_h2d_copy",
        "chunked hipMemcpyAsync(host to device)",
        "GB/s",
    );
    let mut h2d_sync = Phase::new("explicit_h2d_sync", "hipEventSynchronize after H2D", "ms");
    for _ in 0..options.repetitions {
        measure_gpu(&mut h2d, &mut h2d_sync, size, stream, || {
            for (offset, bytes) in offsets() {
                let source = if resource.cpu_accessible {
                    resource.buffer().slice(offset..offset + bytes)
                } else {
                    host_staging.slice(0..bytes)
                };
                stream.copy_from_host_view(source, device_staging.slice_mut(0..bytes))?;
            }
            Ok(())
        });
    }
    result.phases.push(h2d);
    result.phases.push(h2d_sync);

    let mut d2h = Phase::new(
        "explicit_d2h_copy",
        "chunked hipMemcpyAsync(device to host)",
        "GB/s",
    );
    let mut d2h_sync = Phase::new("explicit_d2h_sync", "hipEventSynchronize after D2H", "ms");
    for _ in 0..options.repetitions {
        measure_gpu(&mut d2h, &mut d2h_sync, size, stream, || {
            for (offset, bytes) in offsets() {
                stream.copy_to_host_view(
                    resource.buffer().slice(offset..offset + bytes),
                    host_staging.slice_mut(0..bytes),
                )?;
            }
            Ok(())
        });
    }
    result.phases.push(d2h);
    result.phases.push(d2h_sync);
}

/// `AddCpuViaCopyPhase`: device-to-host copies through a bounded staging
/// buffer and a CPU checksum of each chunk.
fn add_cpu_via_copy(
    resource: &Resource,
    size: usize,
    options: &AllocationOptions,
    result: &mut PathResult,
) {
    let chunk = size.min(options.transfer_chunk_bytes).max(1);
    let stream = &resource.stream;
    let mut phase = Phase::new(
        "warm_cpu_read_via_d2h",
        "hipMemcpy D2H plus CPU checksum",
        "GB/s",
    );
    let staging = HostMemory::pinned(chunk, false)
        .and_then(|memory| memory.into_device(stream).map_err(|(_, error)| error));
    let mut staging = match staging {
        Ok(staging) => staging,
        Err(error) => {
            phase.status = "unavailable".into();
            phase.reason = hip_reason(&error);
            result.phases.push(phase);
            return;
        }
    };
    for _ in 0..options.repetitions {
        let before = read_faults();
        let start = Instant::now();
        let mut checksum = 0;
        for offset in (0..size).step_by(chunk) {
            let bytes = chunk.min(size - offset);
            let copied = stream
                .copy_to_host_view(
                    resource.buffer().slice(offset..offset + bytes),
                    staging.slice_mut(0..bytes),
                )
                .and_then(|()| staging.host().map(|host| checksum_cpu(&host[..bytes])));
            match copied {
                Ok(sum) => checksum += sum,
                Err(error) => {
                    phase.status = "unavailable".into();
                    phase.reason = hip_reason(&error);
                    break;
                }
            }
        }
        let seconds = start.elapsed().as_secs_f64();
        phase.faults.add(fault_delta(before, read_faults()));
        if phase.status != "completed" {
            break;
        }
        result.cpu_checksum = checksum;
        phase.raw_repetitions.push(size as f64 / 1.0e9 / seconds);
    }
    phase.compute_statistics();
    result.phases.push(phase);
}

/// `AddManagedAdvicePhases`.
fn add_managed_advice(resource: &Resource, result: &mut PathResult) {
    let (buffer, stream) = (resource.buffer(), &resource.stream);
    let mut advice = Phase::new(
        "memory_advice",
        "hipMemAdviseSetPreferredLocation(device 0)",
        "ms",
    );
    let _ = measure_call(&mut advice, || buffer.advise_preferred_location(0));
    result.phases.push(advice);
    for (name, operation, device) in [
        ("prefetch_to_gpu", "hipMemPrefetchAsync(device 0)", Some(0)),
        (
            "prefetch_to_cpu",
            "hipMemPrefetchAsync(hipCpuDeviceId)",
            None,
        ),
    ] {
        let mut prefetch = Phase::new(name, operation, "ms");
        let _ = measure_call(&mut prefetch, || {
            buffer.prefetch(stream, device)?;
            stream.synchronize()
        });
        result.phases.push(prefetch);
    }
}

/// `AddAsyncReusePhase`: frees the pool allocation and takes another one of
/// the same size, in stream order.
fn add_async_reuse(
    resource: &mut Resource,
    size: usize,
    options: &AllocationOptions,
    result: &mut PathResult,
) {
    let mut reuse = Phase::new("allocation_reuse", "hipMallocAsync + hipFreeAsync", "ms");
    for _ in 0..options.repetitions {
        let start = Instant::now();
        drop(resource.buffer.take());
        let replaced = resource.stream.alloc::<u8>(size).and_then(|buffer| {
            resource.buffer = Some(buffer);
            resource.stream.synchronize()
        });
        let elapsed = milliseconds(start);
        if let Err(error) = replaced {
            reuse.fail(&error);
            break;
        }
        reuse.raw_repetitions.push(elapsed);
    }
    reuse.compute_statistics();
    result.phases.push(reuse);
}

/// `RunPath`: every phase of one allocation path at one working set.
fn run_path(kind: Kind, path_name: &str, size: usize, options: &AllocationOptions) -> PathResult {
    let mut result = PathResult::new(path_name, size);
    result.expected_checksum = expected_checksum(size, PATTERN_SEED);

    let available = meminfo_value("MemAvailable:");
    if available != 0
        && (size > usize::MAX - SAFETY_HEADROOM || available < (size + SAFETY_HEADROOM) as u64)
    {
        result.status = "unavailable".into();
        result.reason = "insufficient MemAvailable for requested working set plus 4 GiB safety \
             headroom; size was not reduced"
            .into();
        result.allocation_api = path_name.into();
        result.placement_inference =
            "No allocation was attempted after the bounded memory safety check.".into();
        return result;
    }

    let mut setup_samples = Vec::new();
    for _ in 1..options.repetitions {
        let mut sample = PathResult::new(path_name, size);
        let Some(resource) = create_resource(kind, size, &mut sample) else {
            sample.expected_checksum = result.expected_checksum;
            return sample;
        };
        let mut teardown = Phase::new("teardown", "release allocation and mapping", "ms");
        measure_cpu(&mut teardown, || resource.cleanup());
        sample.phases.push(teardown);
        setup_samples.extend(
            sample
                .phases
                .into_iter()
                .filter(|phase| is_setup_phase(&phase.name)),
        );
    }

    let Some(mut resource) = create_resource(kind, size, &mut result) else {
        return result;
    };
    for phase in setup_samples.iter().filter(|p| p.name != "teardown") {
        result.merge(phase);
    }
    let before_residency = residency_evidence(&resource);
    let kernels = Kernels::new();
    let Ok(mut sum) = resource.stream.alloc_persistent::<u64>(1) else {
        result.status = "unavailable".into();
        result.reason = "failed to allocate GPU checksum scalar".into();
        return result;
    };

    if resource.cpu_accessible {
        let mut first = Phase::new("first_cpu_touch", "seed every byte from CPU", "ms");
        let host = resource.buffer_mut().host_mut();
        match host {
            Ok(host) => {
                measure_cpu(&mut first, || fill_cpu(host, PATTERN_SEED));
                result.phases.push(first);
            }
            Err(error) => {
                first.fail(&error);
                result.phases.push(first);
            }
        }
        result.cpu_checksum = resource.buffer().host().map_or(0, checksum_cpu);
    } else {
        let mut touch = Phase::new("first_cpu_touch", "direct CPU write to allocation", "ms");
        touch.status = "unsupported".into();
        touch.reason = "device allocation has no supported direct CPU pointer; explicit \
             hipMemcpy phases are reported separately"
            .into();
        result.phases.push(touch);
        let mut write = Phase::new("first_gpu_touch_write", "seed every byte from GPU", "GB/s");
        let mut sync = Phase::new("first_gpu_touch_sync", "hipEventSynchronize", "ms");
        let Resource { buffer, stream, .. } = &mut resource;
        let buffer = buffer.as_mut().expect("the buffer exists until cleanup");
        measure_gpu(&mut write, &mut sync, size, stream, || {
            kernels.fill(stream, buffer)
        });
        result.phases.push(write);
        result.phases.push(sync);
    }

    let mut read = Phase::new(
        "first_gpu_touch_read",
        "GPU checksum over every byte",
        "GB/s",
    );
    let mut read_sync = Phase::new("first_gpu_touch_read_sync", "hipEventSynchronize", "ms");
    measure_gpu(&mut read, &mut read_sync, size, &resource.stream, || {
        kernels.checksum(&resource.stream, resource.buffer(), &mut sum)
    });
    result.phases.push(read);
    result.phases.push(read_sync);
    if let Some(checksum) = kernels.verify(&resource, &mut sum) {
        result.gpu_checksum = checksum;
    }

    if resource.cpu_accessible {
        let mut warm_read = Phase::new("warm_cpu_read", "CPU checksum over every byte", "GB/s");
        if let Ok(host) = resource.buffer().host() {
            for _ in 0..options.warmup {
                result.cpu_checksum = checksum_cpu(host);
            }
            for _ in 0..options.repetitions {
                let before = read_faults();
                let start = Instant::now();
                result.cpu_checksum = checksum_cpu(host);
                let seconds = start.elapsed().as_secs_f64();
                warm_read.faults.add(fault_delta(before, read_faults()));
                warm_read
                    .raw_repetitions
                    .push(size as f64 / 1.0e9 / seconds);
            }
        }
        warm_read.compute_statistics();
        result.phases.push(warm_read);

        let mut warm_write = Phase::new("warm_cpu_write", "CPU seed write over every byte", "GB/s");
        if let Ok(host) = resource.buffer_mut().host_mut() {
            for _ in 0..options.repetitions {
                let before = read_faults();
                let start = Instant::now();
                fill_cpu(host, PATTERN_SEED);
                let seconds = start.elapsed().as_secs_f64();
                warm_write.faults.add(fault_delta(before, read_faults()));
                warm_write
                    .raw_repetitions
                    .push(size as f64 / 1.0e9 / seconds);
            }
        }
        warm_write.compute_statistics();
        result.phases.push(warm_write);
    } else {
        add_cpu_via_copy(&resource, size, options, &mut result);
    }

    let mut warm_read = Phase::new("warm_gpu_read", "GPU checksum over every byte", "GB/s");
    let mut warm_read_sync = Phase::new("warm_gpu_read_sync", "hipEventSynchronize", "ms");
    for _ in 0..options.warmup {
        let _ = kernels.checksum(&resource.stream, resource.buffer(), &mut sum);
        let _ = resource.stream.synchronize();
    }
    for _ in 0..options.repetitions {
        measure_gpu(
            &mut warm_read,
            &mut warm_read_sync,
            size,
            &resource.stream,
            || kernels.checksum(&resource.stream, resource.buffer(), &mut sum),
        );
    }
    result.phases.push(warm_read);
    result.phases.push(warm_read_sync);

    let mut warm_write = Phase::new("warm_gpu_write", "GPU seed write over every byte", "GB/s");
    let mut warm_write_sync = Phase::new("warm_gpu_write_sync", "hipEventSynchronize", "ms");
    for _ in 0..options.repetitions {
        let Resource { buffer, stream, .. } = &mut resource;
        let buffer = buffer.as_mut().expect("the buffer exists until cleanup");
        measure_gpu(&mut warm_write, &mut warm_write_sync, size, stream, || {
            kernels.fill(stream, buffer)
        });
    }
    result.phases.push(warm_write);
    result.phases.push(warm_write_sync);

    add_explicit_copies(&mut resource, size, options, &mut result);
    if resource.managed {
        add_managed_advice(&resource, &mut result);
    }
    if kind == Kind::Async {
        add_async_reuse(&mut resource, size, options, &mut result);
    }

    if resource.buffer.is_some() {
        let Resource { buffer, stream, .. } = &mut resource;
        let buffer = buffer.as_mut().expect("the buffer exists until cleanup");
        let _ = kernels.fill(stream, buffer);
        let _ = stream.synchronize();
        if resource.cpu_accessible {
            result.cpu_checksum = resource.buffer().host().map_or(0, checksum_cpu);
        } else {
            let single = AllocationOptions {
                working_set_bytes: vec![size],
                warmup: 0,
                repetitions: 1,
                transfer_chunk_bytes: options.transfer_chunk_bytes,
            };
            add_cpu_via_copy(&resource, size, &single, &mut result);
        }
        if let Some(checksum) = kernels.verify(&resource, &mut sum) {
            result.gpu_checksum = checksum;
        }
    }
    result.checksum_verified = result.cpu_checksum == result.expected_checksum
        && result.gpu_checksum == result.expected_checksum;
    if !result.checksum_verified {
        result.status = "unavailable".into();
        result.reason = "CPU/GPU deterministic checksum mismatch".into();
    }
    let after_residency = if resource.buffer.is_some() {
        residency_evidence(&resource)
    } else {
        "mincore unavailable: no host mapping".into()
    };
    result.residency_evidence =
        format!("before first touch: {before_residency}; after measured phases: {after_residency}");

    resource.sum = Some(sum);
    let mut teardown = Phase::new("teardown", "release allocation and mapping", "ms");
    measure_cpu(&mut teardown, || resource.cleanup());
    result.phases.push(teardown);
    for phase in setup_samples.iter().filter(|p| p.name == "teardown") {
        result.merge(phase);
    }
    result
}

/// `RunHipAllocationBenchmark`.
pub fn run(
    options: &AllocationOptions,
    inventory: &SystemInventory,
    fingerprint: &MachineFingerprint,
    engine_revision: &str,
) -> AllocationReport {
    let mut report = AllocationReport {
        schema_version: "1.0.0".into(),
        artifact_type: "hipAllocation".into(),
        fingerprint_id: fingerprint.fingerprint_id.clone(),
        engine_revision: engine_revision.into(),
        timestamp: iso8601_utc(),
        options: options.clone(),
        inventory: inventory.clone(),
        fingerprint: fingerprint.clone(),
        page_size_bytes: rustix::param::page_size() as u64,
        huge_pages_total: meminfo_value("HugePages_Total:"),
        huge_page_size_bytes: meminfo_value("Hugepagesize:"),
        paths: Vec::new(),
        limitations: [
            "HIP pointer attributes identify API-visible memory types, not physical DRAM banks or \
             cache residency on an integrated APU.",
            "Linux mincore reports whether host-mapped pages are resident in the process mapping; \
             it does not identify GPU cache residency.",
            "Minor and major faults are process-level getrusage deltas and may include unrelated \
             runtime activity.",
            "No privileged performance counters, page migration counters, persistent huge-page \
             settings, NUMA changes, or power-policy changes are used.",
            "The file-backed path uses an unnamed memfd mapping so artifacts contain no host path; \
             end-to-end #125 separately measures the real GGUF files.",
        ]
        .map(String::from)
        .to_vec(),
        all_requested_paths_reported: false,
        checksums_verified: false,
        phases_separated: false,
    };

    match gufo_hip::device_count() {
        Ok(count) if count > 0 => {
            for &size in &options.working_set_bytes {
                for (kind, name) in KINDS.iter().zip(PATH_NAMES) {
                    report.paths.push(run_path(*kind, name, size, options));
                }
            }
        }
        status => {
            let reason = match status {
                Ok(_) => "no HIP device found".to_string(),
                Err(error) => hip_reason(&error),
            };
            for &size in &options.working_set_bytes {
                for name in PATH_NAMES {
                    let mut path = PathResult::new(name, size);
                    path.allocation_api = "HIP runtime".into();
                    path.status = "unavailable".into();
                    path.reason.clone_from(&reason);
                    path.placement_inference = "No placement evidence is available.".into();
                    report.paths.push(path);
                }
            }
        }
    }

    report.all_requested_paths_reported =
        report.paths.len() == options.working_set_bytes.len() * PATH_NAMES.len();
    report.checksums_verified = report.paths.iter().all(|path| {
        if path.status == "completed" {
            path.checksum_verified
        } else {
            !path.reason.is_empty()
        }
    });
    report.phases_separated = report.paths.iter().all(|path| {
        if path.status != "completed" {
            return !path.reason.is_empty();
        }
        [
            "allocation",
            "first_gpu_touch_read",
            "warm_gpu_read",
            "warm_gpu_write",
            "teardown",
        ]
        .iter()
        .all(|name| path.has_phase(name))
    });
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn working_set_sizes_parse_like_cpp() {
        assert_eq!(parse_working_set_sizes("64"), Ok(vec![64 << 20]));
        assert_eq!(
            parse_working_set_sizes("1,2048"),
            Ok(vec![1 << 20, 2048 << 20])
        );
        for invalid in ["", ",1", "1,", "0", "1,,2", " 1", "+1", "-1", "1x", "1.5"] {
            assert!(parse_working_set_sizes(invalid).is_err(), "{invalid:?}");
        }
        assert!(parse_working_set_sizes("18446744073709551615").is_err());
    }

    #[test]
    fn statistics_follow_the_cpp_rounding() {
        let mut phase = Phase::new("p", "op", "ms");
        phase.compute_statistics();
        assert_eq!(phase.median, 0.0);
        phase.raw_repetitions = vec![4.0, 1.0, 3.0, 2.0];
        phase.compute_statistics();
        assert_eq!(
            (phase.minimum, phase.maximum, phase.median, phase.p95),
            (1.0, 4.0, 2.5, 4.0)
        );
        phase.raw_repetitions = (1..=20).map(f64::from).collect();
        phase.compute_statistics();
        // ceil(0.95 * 20) - 1 = 18.
        assert_eq!((phase.median, phase.p95), (10.5, 19.0));
    }

    #[test]
    fn the_expected_checksum_sums_the_pattern() {
        for size in [0usize, 1, 255, 256, 257, 1 << 20] {
            let mut bytes = vec![0u8; size];
            fill_cpu(&mut bytes, PATTERN_SEED);
            assert_eq!(
                expected_checksum(size, PATTERN_SEED),
                checksum_cpu(&bytes),
                "{size}"
            );
        }
    }

    #[test]
    fn page_faults_come_from_proc() {
        let before = read_faults();
        let mut touched = vec![0u8; 16 << 20];
        fill_cpu(&mut touched, 1);
        assert!(checksum_cpu(&touched) > 0);
        assert!(fault_delta(before, read_faults()).minor > 0);
    }
}
