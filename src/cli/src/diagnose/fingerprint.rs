//! The canonical machine fingerprint and its SHA-256 ID (`fingerprint.cpp`).

use std::fmt::Write;

use sha2::{Digest, Sha256};

use super::inventory::{self, SystemInventory, escape};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalFingerprint {
    pub schema_version: String,
    pub cpu_model: String,
    pub cpu_architecture: String,
    pub cpu_logical_cores: u32,
    pub cpu_physical_cores: u32,
    pub memory_total_bytes: u64,
    pub memory_type: String,
    pub gpu_name: String,
    pub gpu_architecture: String,
    pub gpu_compute_units: u32,
    pub gpu_driver: String,
    pub gpu_pci_id: String,
    pub kernel_release: String,
    pub rocm_version: String,
    pub compiler: String,
    pub language_standard: String,
}

/// The initializers of the C++ struct, for artifacts built field by field
/// (the validator and its tests); reported fingerprints come from the
/// inventory (`generate`).
impl Default for CanonicalFingerprint {
    fn default() -> Self {
        Self {
            schema_version: "1.0.0".into(),
            cpu_model: String::new(),
            cpu_architecture: "x86_64".into(),
            cpu_logical_cores: 0,
            cpu_physical_cores: 0,
            memory_total_bytes: 0,
            memory_type: "LPDDR5X (Unified)".into(),
            gpu_name: String::new(),
            gpu_architecture: "gfx1151".into(),
            gpu_compute_units: 0,
            gpu_driver: "amdgpu".into(),
            gpu_pci_id: "1002:1586".into(),
            kernel_release: String::new(),
            rocm_version: gufo_hip::BUILD_ROCM_VERSION.into(),
            compiler: inventory::COMPILER.into(),
            language_standard: inventory::LANGUAGE_STANDARD.into(),
        }
    }
}

/// Lowercase hexadecimal SHA-256 of `data`.
pub fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .fold(String::new(), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

impl CanonicalFingerprint {
    /// The hashed text: keys in lexicographic order, two-space indent.
    pub fn canonical_json(&self) -> String {
        format!(
            "{{\n  \"cppStandard\": \"{}\",\n  \"cpuArchitecture\": \"{}\",\n  \
             \"cpuLogicalCores\": {},\n  \"cpuModel\": \"{}\",\n  \"cpuPhysicalCores\": {},\n  \
             \"cxxCompiler\": \"{}\",\n  \"gpuArchitecture\": \"{}\",\n  \
             \"gpuComputeUnits\": {},\n  \"gpuDriver\": \"{}\",\n  \"gpuName\": \"{}\",\n  \
             \"gpuPciId\": \"{}\",\n  \"kernelRelease\": \"{}\",\n  \
             \"memoryTotalBytes\": {},\n  \"memoryType\": \"{}\",\n  \"rocmVersion\": \"{}\",\n  \
             \"schemaVersion\": \"{}\"\n}}",
            escape(&self.language_standard),
            escape(&self.cpu_architecture),
            self.cpu_logical_cores,
            escape(&self.cpu_model),
            self.cpu_physical_cores,
            escape(&self.compiler),
            escape(&self.gpu_architecture),
            self.gpu_compute_units,
            escape(&self.gpu_driver),
            escape(&self.gpu_name),
            escape(&self.gpu_pci_id),
            escape(&self.kernel_release),
            self.memory_total_bytes,
            escape(&self.memory_type),
            escape(&self.rocm_version),
            escape(&self.schema_version)
        )
    }

    pub fn fingerprint_id(&self) -> String {
        sha256_hex(self.canonical_json().as_bytes())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MachineFingerprint {
    pub fingerprint_id: String,
    pub canonical: CanonicalFingerprint,
}

pub fn generate(inventory: &SystemInventory) -> MachineFingerprint {
    let canonical = CanonicalFingerprint {
        cpu_model: inventory.cpu.model_name.clone(),
        cpu_architecture: inventory.cpu.architecture.clone(),
        cpu_logical_cores: inventory.cpu.logical_cores,
        cpu_physical_cores: inventory.cpu.physical_cores,
        memory_total_bytes: inventory.memory.total_bytes,
        memory_type: inventory.memory.memory_type.clone(),
        gpu_name: inventory.gpu.name.clone(),
        gpu_architecture: inventory.gpu.architecture.clone(),
        gpu_compute_units: inventory.gpu.compute_units,
        gpu_driver: inventory.gpu.driver_name.clone(),
        gpu_pci_id: inventory.gpu.pci_id.clone(),
        kernel_release: inventory.toolchain.kernel_release.clone(),
        rocm_version: inventory.toolchain.rocm_version.clone(),
        compiler: inventory.toolchain.compiler.clone(),
        language_standard: inventory.toolchain.language_standard.clone(),
        ..CanonicalFingerprint::default()
    };
    MachineFingerprint {
        fingerprint_id: canonical.fingerprint_id(),
        canonical,
    }
}

impl MachineFingerprint {
    pub fn to_json(&self) -> String {
        format!(
            "{{\n  \"schemaVersion\": \"{}\",\n  \"fingerprintId\": \"{}\",\n  \
             \"canonical\": {}\n}}\n",
            escape(&self.canonical.schema_version),
            escape(&self.fingerprint_id),
            self.canonical.canonical_json()
        )
    }

    pub fn to_human(&self) -> String {
        let c = &self.canonical;
        format!(
            "=== Machine Fingerprint ===\n\
             Fingerprint ID      : {}\n\
             Target Platform     : {}-linux (Linux {})\n\
             CPU Model           : {} ({} threads)\n\
             GPU Architecture    : {} [{}, {} CUs]\n\
             Build Toolchain     : ROCm {}, {}\n",
            self.fingerprint_id,
            c.cpu_architecture,
            c.kernel_release,
            c.cpu_model,
            c.cpu_logical_cores,
            c.gpu_name,
            c.gpu_architecture,
            c.gpu_compute_units,
            c.rocm_version,
            c.compiler
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_digests() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
