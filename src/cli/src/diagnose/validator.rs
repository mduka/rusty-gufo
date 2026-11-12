//! Validation of diagnostic and benchmark artifacts
//! (`artifact_validator.cpp`).

use std::fmt::Write;
use std::io::Read;
use std::path::Path;

use gufo_core::json::{self, Value};

use super::fingerprint::CanonicalFingerprint;

const MAXIMUM_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
const MAXIMUM_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub fingerprint_id: String,
    pub schema_version: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self {
            is_valid: true,
            fingerprint_id: String::new(),
            schema_version: "1.0.0".into(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

impl ValidationResult {
    fn fail(&mut self, error: impl Into<String>) {
        self.is_valid = false;
        self.errors.push(error.into());
    }

    pub fn to_json(&self) -> String {
        let strings = |list: &[String]| {
            list.iter().fold(Value::array(), |mut array, entry| {
                array.push(entry.as_str());
                array
            })
        };
        let value = Value::object()
            .with("isValid", self.is_valid)
            .with("schemaVersion", self.schema_version.as_str())
            .with("fingerprintId", self.fingerprint_id.as_str())
            .with("errors", strings(&self.errors))
            .with("warnings", strings(&self.warnings));
        value.dump().unwrap_or_default() + "\n"
    }

    pub fn to_human(&self) -> String {
        let mut out = format!(
            "=== Artifact Validation Result: [{}] ===\nSchema Version  : {}\n\
             Fingerprint ID  : {}\n",
            if self.is_valid { "VALID" } else { "INVALID" },
            self.schema_version,
            self.fingerprint_id
        );
        if !self.errors.is_empty() {
            out.push_str("\n--- Validation Errors ---\n");
            for error in &self.errors {
                let _ = writeln!(out, "  [ERROR] {error}");
            }
        }
        if !self.warnings.is_empty() {
            out.push_str("\n--- Validation Warnings ---\n");
            for warning in &self.warnings {
                let _ = writeln!(out, "  [WARN] {warning}");
            }
        }
        out
    }
}

fn object<'a>(parent: &'a Value, name: &str) -> Result<&'a Value, String> {
    parent
        .find(name)
        .filter(|value| value.is_object())
        .ok_or_else(|| format!("Expected object '{name}'"))
}

fn string(object: &Value, name: &str) -> Result<String, String> {
    object
        .find(name)
        .filter(|value| value.is_string())
        .map(|value| value.str().to_string())
        .ok_or_else(|| format!("Expected string '{name}'"))
}

fn unsigned(object: &Value, name: &str, maximum: u64) -> Result<u64, String> {
    object
        .find(name)
        .filter(|value| value.is_number())
        .map(|value| value.as_double(0.0))
        .filter(|&number| number >= 0.0 && number <= maximum as f64 && number.floor() == number)
        .map(|number| number as u64)
        .ok_or_else(|| format!("Expected bounded unsigned integer '{name}'"))
}

fn boolean(object: &Value, name: &str) -> Result<bool, String> {
    object
        .find(name)
        .filter(|value| value.is_bool())
        .map(|value| value.as_bool(false))
        .ok_or_else(|| format!("Expected boolean '{name}'"))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

fn contains_local_metadata(value: &Value) -> bool {
    if value.is_string() {
        return value.str().contains("/home/");
    }
    value.members().iter().any(|(name, member)| {
        matches!(name.as_str(), "hostname" | "username" | "secret")
            || contains_local_metadata(member)
    }) || value.items().iter().any(contains_local_metadata)
}

fn validate(content: &[u8], result: &mut ValidationResult) -> Result<(), String> {
    if content.len() > MAXIMUM_ARTIFACT_BYTES {
        return Err("Artifact exceeds the 16 MiB limit".into());
    }
    let root = json::parse(content).map_err(|e| e.message().to_string())?;
    if !root.is_object() {
        return Err("Artifact must be a JSON object".into());
    }
    result.schema_version = string(&root, "schemaVersion")?;
    if result.schema_version != "1.0.0" {
        let message = format!("Unsupported schema version: {}", result.schema_version);
        result.fail(message);
    }
    result.fingerprint_id = string(&root, "fingerprintId")?;
    if !is_sha256(&result.fingerprint_id) {
        result.fail("Invalid fingerprintId: expected a lowercase hexadecimal SHA-256");
    }
    let check_architecture = |object: &Value, result: &mut ValidationResult| {
        if object.contains("gpuArchitecture") {
            let architecture = string(object, "gpuArchitecture")?;
            if !architecture.is_empty() && architecture != "gfx1151" {
                result.fail(format!("Incompatible GPU architecture: {architecture}"));
            }
        }
        Ok::<(), String>(())
    };
    check_architecture(&root, result)?;
    if root.contains("machine") {
        check_architecture(object(&root, "machine")?, result)?;
    }
    if root.contains("canonical") {
        let block = object(&root, "canonical")?;
        let canonical = CanonicalFingerprint {
            schema_version: string(block, "schemaVersion")?,
            language_standard: string(block, "cppStandard")?,
            cpu_architecture: string(block, "cpuArchitecture")?,
            cpu_logical_cores: unsigned(block, "cpuLogicalCores", u64::from(u32::MAX))? as u32,
            cpu_model: string(block, "cpuModel")?,
            cpu_physical_cores: unsigned(block, "cpuPhysicalCores", u64::from(u32::MAX))? as u32,
            compiler: string(block, "cxxCompiler")?,
            gpu_architecture: string(block, "gpuArchitecture")?,
            gpu_compute_units: unsigned(block, "gpuComputeUnits", u64::from(u32::MAX))? as u32,
            gpu_driver: string(block, "gpuDriver")?,
            gpu_name: string(block, "gpuName")?,
            gpu_pci_id: string(block, "gpuPciId")?,
            kernel_release: string(block, "kernelRelease")?,
            memory_total_bytes: unsigned(block, "memoryTotalBytes", MAXIMUM_SAFE_INTEGER)?,
            memory_type: string(block, "memoryType")?,
            rocm_version: string(block, "rocmVersion")?,
        };
        if canonical.schema_version != result.schema_version {
            result.fail("Canonical schemaVersion does not match the artifact");
        }
        if canonical.fingerprint_id() != result.fingerprint_id {
            result.fail("Fingerprint ID mismatch over canonical fields");
        }
        check_architecture(block, result)?;
    }
    let artifact_type = if root.contains("artifactType") {
        string(&root, "artifactType")?
    } else {
        String::new()
    };
    if artifact_type == "hipAllocation" {
        let summary = object(&root, "summary")?;
        if !boolean(summary, "allRequestedPathsReported")?
            || !boolean(summary, "checksumsVerified")?
            || !boolean(summary, "phasesSeparated")?
            || string(summary, "status")? != "completed"
        {
            result.fail("HIP allocation diagnostic did not complete all paths and checks");
        }
    }
    if !artifact_type.is_empty() && artifact_type != "hipAllocation" {
        result.fail(format!("Unknown diagnostic artifactType: {artifact_type}"));
    }
    if artifact_type.is_empty() && !root.contains("canonical") {
        // Bandwidth artifacts reference a fingerprint rather than embedding
        // it; a schema version and a hash alone are no artifact.
        let options = object(&root, "options")?;
        for name in ["warmup", "repetitions", "durationMs", "workingSetBytes"] {
            unsigned(options, name, MAXIMUM_SAFE_INTEGER)?;
        }
        let paths = root
            .find("paths")
            .filter(|paths| paths.is_array())
            .ok_or("Expected bandwidth paths array")?;
        for path in paths.items() {
            if !path.is_object() {
                return Err("Expected bandwidth path object".into());
            }
            string(path, "backend")?;
            string(path, "pathName")?;
            boolean(path, "sentinelVerified")?;
            unsigned(path, "workingSetBytes", MAXIMUM_SAFE_INTEGER)?;
            let samples = path
                .find("rawRepetitionsGbps")
                .filter(|samples| samples.is_array())
                .ok_or("Expected bandwidth samples array")?;
            if samples
                .items()
                .iter()
                .any(|sample| !sample.is_number() || sample.as_double(0.0) < 0.0)
            {
                return Err("Invalid bandwidth sample".into());
            }
        }
    }
    if contains_local_metadata(&root) {
        result
            .warnings
            .push("Artifact contains unredacted local host metadata or paths".into());
    }
    Ok(())
}

pub fn validate_content(content: &[u8]) -> ValidationResult {
    let mut result = ValidationResult::default();
    if let Err(error) = validate(content, &mut result) {
        result.fail(error);
    }
    result
}

pub fn validate_file(path: &Path) -> ValidationResult {
    let Ok(file) = std::fs::File::open(path) else {
        let mut result = ValidationResult::default();
        result.fail(format!("Could not open artifact file: {}", path.display()));
        return result;
    };
    let mut content = Vec::new();
    if file
        .take(MAXIMUM_ARTIFACT_BYTES as u64 + 1)
        .read_to_end(&mut content)
        .is_err()
    {
        let mut result = ValidationResult::default();
        result.fail("Could not read artifact file");
        return result;
    }
    validate_content(&content)
}

#[cfg(test)]
mod tests {
    use super::super::fingerprint::MachineFingerprint;
    use super::*;

    fn parse(text: &str) -> Value {
        json::parse_str(text).expect("json")
    }

    fn valid(value: &Value) -> bool {
        validate_content(value.dump().expect("dump").as_bytes()).is_valid
    }

    /// The cases of `TestArtifactValidation` in the C++ bandwidth test.
    #[test]
    fn artifacts_are_validated_like_cpp() {
        let mut fingerprint = MachineFingerprint::default();
        fingerprint.canonical.cpu_model = "a \"quoted\" CPU".into();
        fingerprint.fingerprint_id = fingerprint.canonical.fingerprint_id();
        let text = fingerprint.to_json();
        assert!(
            validate_content(text.as_bytes()).is_valid,
            "escaped canonical fields"
        );
        for malformed in [
            format!("{text}garbage"),
            text[..text.len() - 3].to_string(),
            "[]".into(),
            "null".into(),
        ] {
            assert!(
                !validate_content(malformed.as_bytes()).is_valid,
                "{malformed}"
            );
        }
        let mut root = parse(&text);
        root.set("fingerprintId", "z".repeat(64));
        assert!(!valid(&root), "non-hex SHA");
        let mut root = parse(&text);
        root.set("schemaVersion", 1);
        assert!(!valid(&root), "wrong type");
        let mut root = parse(&text);
        root.member_mut("canonical").set("cpuLogicalCores", -1);
        assert!(!valid(&root), "negative core count");
        let mut root = parse(&text);
        root.member_mut("canonical")
            .set("cpuLogicalCores", 4_294_967_296.0);
        assert!(!valid(&root), "overflowing core count");
        let mut root = parse(&text);
        root.set("cpuModel", "spoofed value outside canonical");
        assert!(valid(&root), "only canonical fields are hashed");
        root.member_mut("canonical").set("cpuModel", "tampered");
        assert!(!valid(&root), "tampered canonical field");
        let root = Value::object().with("nested", parse(&text));
        assert!(!valid(&root), "nested required fields");
        let mut root = parse(&text);
        root.append_member("fingerprintId", fingerprint.fingerprint_id.as_str());
        assert!(!valid(&root), "duplicate key");
        let mut root = parse(&text);
        root.set("canonical", "not an object");
        assert!(!valid(&root), "canonical must be an object");
        let root = Value::object()
            .with("schemaVersion", "1.0.0")
            .with("fingerprintId", fingerprint.fingerprint_id.as_str());
        assert!(!valid(&root), "a schema and hash alone are no artifact");
        let mut root = parse(&text);
        root.set("artifactType", "unknown");
        assert!(!valid(&root), "unknown artifact type");
    }

    #[test]
    fn results_render_like_cpp() {
        let mut result = validate_content(b"{\"schemaVersion\": \"2\", \"fingerprintId\": \"x\"}");
        assert_eq!(
            result.errors,
            [
                "Unsupported schema version: 2",
                "Invalid fingerprintId: expected a lowercase hexadecimal SHA-256",
                "Expected object 'options'"
            ]
        );
        result.warnings.push("w".into());
        assert_eq!(
            result.to_json(),
            "{\"isValid\":false,\"schemaVersion\":\"2\",\"fingerprintId\":\"x\",\"errors\":\
             [\"Unsupported schema version: 2\",\"Invalid fingerprintId: expected a lowercase \
             hexadecimal SHA-256\",\"Expected object 'options'\"],\"warnings\":[\"w\"]}\n"
        );
    }
}
