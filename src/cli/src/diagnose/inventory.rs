//! The hardware and software inventory (`system_inventory.cpp`). The
//! toolchain fields describe this build: its compiler, language edition and
//! the ROCm release it was built against and runs on. Every field comes from
//! the host or is "unknown"; unlike C++, nothing defaults to a Strix Halo.

use std::collections::BTreeMap;
use std::fmt::Write;

use super::sysfs::LinuxSysfs;

/// The compiler of this binary, e.g. "rustc 1.98.1".
pub const COMPILER: &str = env!("GUFO_RUSTC_VERSION");
/// The language standard of the host code.
pub const LANGUAGE_STANDARD: &str = "Rust 2024";
/// A field no source reported.
pub const UNKNOWN: &str = "unknown";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuInventory {
    pub model_name: String,
    pub vendor: String,
    pub architecture: String,
    pub logical_cores: u32,
    pub physical_cores: u32,
    pub numa_nodes: u32,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryInventory {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub free_bytes: u64,
    pub memory_type: String,
    pub is_unified: bool,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInventory {
    pub name: String,
    pub architecture: String,
    pub raw_arch_name: String,
    pub compute_units: u32,
    pub total_memory_bytes: u64,
    pub driver_name: String,
    pub pci_id: String,
    pub pci_slot: String,
    pub power_mode: String,
    pub clock_state: String,
    pub temperatures: BTreeMap<String, String>,
    pub availability: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolchainInventory {
    pub kernel_release: String,
    pub amdgpu_status: String,
    pub rocm_version: String,
    pub compiler: String,
    pub language_standard: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemInventory {
    pub cpu: CpuInventory,
    pub memory: MemoryInventory,
    pub gpu: GpuInventory,
    pub toolchain: ToolchainInventory,
}

impl Default for SystemInventory {
    fn default() -> Self {
        Self {
            cpu: CpuInventory {
                model_name: String::new(),
                vendor: String::new(),
                architecture: "x86_64".into(),
                logical_cores: 0,
                physical_cores: 0,
                numa_nodes: 1,
                source: "/proc/cpuinfo".into(),
            },
            memory: MemoryInventory {
                total_bytes: 0,
                available_bytes: 0,
                free_bytes: 0,
                memory_type: UNKNOWN.into(),
                is_unified: false,
                source: "/proc/meminfo".into(),
            },
            gpu: GpuInventory {
                name: UNKNOWN.into(),
                architecture: UNKNOWN.into(),
                raw_arch_name: String::new(),
                compute_units: 0,
                total_memory_bytes: 0,
                driver_name: UNKNOWN.into(),
                pci_id: UNKNOWN.into(),
                pci_slot: String::new(),
                power_mode: UNKNOWN.into(),
                clock_state: "unavailable (sysfs unreadable or power-managed)".into(),
                temperatures: BTreeMap::new(),
                availability: UNKNOWN.into(),
                source: "HIP + /sys/class/drm".into(),
            },
            toolchain: ToolchainInventory {
                kernel_release: String::new(),
                amdgpu_status: UNKNOWN.into(),
                rocm_version: gufo_hip::BUILD_ROCM_VERSION.into(),
                compiler: COMPILER.into(),
                language_standard: LANGUAGE_STANDARD.into(),
                source: "Build Toolchain + Linux UAPI".into(),
            },
        }
    }
}

/// `EscapeJsonStr` of the inventory: quotes, backslashes, newlines, tabs.
pub(super) fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(character),
        }
    }
    escaped
}

/// Collects the inventory from `sysfs` and the first HIP device.
pub fn collect(sysfs: &LinuxSysfs) -> SystemInventory {
    let mut inventory = SystemInventory::default();
    if let Some(cpu) = sysfs.query_cpu_info() {
        inventory.cpu.model_name = cpu.model_name;
        inventory.cpu.vendor = cpu.vendor_id;
        inventory.cpu.architecture = cpu.architecture;
        inventory.cpu.logical_cores = cpu.logical_cores;
        inventory.cpu.physical_cores = cpu.physical_cores;
        inventory.cpu.numa_nodes = cpu.numa_nodes;
    } else {
        inventory.cpu.model_name = "unavailable (could not read /proc/cpuinfo)".into();
    }
    if let Some(memory) = sysfs.query_mem_info() {
        inventory.memory.total_bytes = memory.total_bytes;
        inventory.memory.available_bytes = memory.available_bytes;
        inventory.memory.free_bytes = memory.free_bytes;
    }
    inventory.toolchain.kernel_release = sysfs.kernel_release().unwrap_or_else(|| "unknown".into());
    inventory.toolchain.amdgpu_status = sysfs.module_state("amdgpu");

    if let Some(primary) = sysfs.query_drm_gpu_devices().into_iter().next() {
        let gpu = &mut inventory.gpu;
        gpu.driver_name = if primary.driver.is_empty() {
            "amdgpu".into()
        } else {
            primary.driver.clone()
        };
        gpu.pci_id = primary.pci_id.clone();
        gpu.pci_slot = primary.pci_slot;
        gpu.power_mode = primary.power_mode;
        if let Some(clock_state) = primary.clock_state {
            gpu.clock_state = clock_state;
        }
        gpu.temperatures = primary.hwmon_sensors;
        if ["1002:1586", "1002:1587", "1002:1588"]
            .iter()
            .any(|id| primary.pci_id.contains(id))
        {
            gpu.architecture = "gfx1151".into();
            gpu.availability = "available".into();
        } else {
            gpu.architecture = primary.pci_id.clone();
            gpu.availability = format!("unsupported GPU ({})", primary.pci_id);
        }
    }

    let mut integrated = None;
    match gufo_hip::device_count() {
        Ok(count) if count > 0 => {
            if let Ok(properties) = gufo_hip::device_properties(0) {
                integrated = Some(properties.integrated);
                let gpu = &mut inventory.gpu;
                gpu.name = properties.name;
                gpu.total_memory_bytes = properties.total_global_memory;
                gpu.compute_units = properties.multiprocessor_count as u32;
                if properties.arch_name.contains("gfx1151") {
                    gpu.architecture = "gfx1151".into();
                    gpu.availability = "available".into();
                } else {
                    gpu.architecture = properties.arch_name.clone();
                    gpu.availability = "unsupported architecture".into();
                }
                gpu.raw_arch_name = properties.arch_name;
            }
        }
        result => {
            let code = result.err().map_or(0, |error| error.code);
            inventory.gpu.availability = format!("unavailable ({})", gufo_hip::error_name(code));
        }
    }
    // Strix Halo is an APU and takes only LPDDR5X; other memory types are
    // not readable without privileges (DMI).
    let strix_halo = inventory.gpu.architecture == "gfx1151";
    inventory.memory.is_unified = integrated.unwrap_or(strix_halo);
    inventory.memory.memory_type = match (inventory.memory.is_unified, strix_halo) {
        (true, true) => "LPDDR5X (Unified)",
        (true, false) => "unknown (Unified)",
        (false, _) => UNKNOWN,
    }
    .into();
    inventory
}

impl SystemInventory {
    pub fn to_json(&self) -> String {
        let (cpu, memory, gpu, toolchain) = (&self.cpu, &self.memory, &self.gpu, &self.toolchain);
        let mut out = String::from("{\n");
        let _ = write!(
            out,
            "  \"cpu\": {{\n    \"modelName\": \"{}\",\n    \"vendor\": \"{}\",\n    \
             \"architecture\": \"{}\",\n    \"logicalCores\": {},\n    \
             \"physicalCores\": {},\n    \"numaNodes\": {},\n    \"source\": \"{}\"\n  }},\n",
            escape(&cpu.model_name),
            escape(&cpu.vendor),
            escape(&cpu.architecture),
            cpu.logical_cores,
            cpu.physical_cores,
            cpu.numa_nodes,
            escape(&cpu.source)
        );
        let _ = write!(
            out,
            "  \"memory\": {{\n    \"totalBytes\": {},\n    \"availableBytes\": {},\n    \
             \"freeBytes\": {},\n    \"memoryType\": \"{}\",\n    \"isUnified\": {},\n    \
             \"source\": \"{}\"\n  }},\n",
            memory.total_bytes,
            memory.available_bytes,
            memory.free_bytes,
            escape(&memory.memory_type),
            memory.is_unified,
            escape(&memory.source)
        );
        let _ = write!(
            out,
            "  \"gpu\": {{\n    \"name\": \"{}\",\n    \"architecture\": \"{}\",\n    \
             \"rawArchName\": \"{}\",\n    \"computeUnits\": {},\n    \
             \"totalMemoryBytes\": {},\n    \"driverName\": \"{}\",\n    \"pciId\": \"{}\",\n    \
             \"pciSlot\": \"{}\",\n    \"powerMode\": \"{}\",\n    \"clockState\": \"{}\",\n    \
             \"availability\": \"{}\",\n    \"temperatures\": {{",
            escape(&gpu.name),
            escape(&gpu.architecture),
            escape(&gpu.raw_arch_name),
            gpu.compute_units,
            gpu.total_memory_bytes,
            escape(&gpu.driver_name),
            escape(&gpu.pci_id),
            escape(&gpu.pci_slot),
            escape(&gpu.power_mode),
            escape(&gpu.clock_state),
            escape(&gpu.availability)
        );
        if gpu.temperatures.is_empty() {
            out.push_str("},\n");
        } else {
            out.push('\n');
            for (index, (key, value)) in gpu.temperatures.iter().enumerate() {
                let _ = write!(out, "      \"{}\": \"{}\"", escape(key), escape(value));
                if index + 1 < gpu.temperatures.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str("    },\n");
        }
        let _ = write!(out, "    \"source\": \"{}\"\n  }},\n", escape(&gpu.source));
        let _ = write!(
            out,
            "  \"toolchain\": {{\n    \"kernelRelease\": \"{}\",\n    \
             \"amdgpuStatus\": \"{}\",\n    \"rocmVersion\": \"{}\",\n    \
             \"cxxCompiler\": \"{}\",\n    \"cppStandard\": \"{}\",\n    \
             \"source\": \"{}\"\n  }}\n}}\n",
            escape(&toolchain.kernel_release),
            escape(&toolchain.amdgpu_status),
            escape(&toolchain.rocm_version),
            escape(&toolchain.compiler),
            escape(&toolchain.language_standard),
            escape(&toolchain.source)
        );
        out
    }

    pub fn to_human(&self) -> String {
        format!(
            "=== Hardware and Software Inventory ===\n\n\
             CPU Model           : {} ({} threads, {})\n\
             System Memory       : {} GiB ({})\n\
             GPU Model           : {} [{}, {} CUs] ({})\n\
             Linux Kernel        : {}\n\
             ROCm Version        : {}\n",
            self.cpu.model_name,
            self.cpu.logical_cores,
            self.cpu.architecture,
            self.memory.total_bytes / (1024 * 1024 * 1024),
            self.memory.memory_type,
            self.gpu.name,
            self.gpu.architecture,
            self.gpu.compute_units,
            self.gpu.availability,
            self.toolchain.kernel_release,
            self.toolchain.rocm_version
        )
    }
}
