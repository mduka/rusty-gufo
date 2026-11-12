//! The compatibility verdict (`compatibility.cpp`).

use std::fmt::Write;

use super::inventory::SystemInventory;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Supported,
    Unsupported,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Supported => "supported",
            Verdict::Unsupported => "unsupported",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityItem {
    pub component: String,
    pub verdict: Verdict,
    pub detected_value: String,
    pub required_value: String,
    pub evidence: String,
    pub remediation_hint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityReport {
    pub overall_verdict: Verdict,
    pub items: Vec<CompatibilityItem>,
}

/// `EscapeJsonString` of this file: quotes, backslashes and newlines.
fn escape(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn item(
    component: &str,
    supported: bool,
    detected: String,
    required: &str,
    evidence: &str,
    remediation: String,
) -> CompatibilityItem {
    CompatibilityItem {
        component: component.into(),
        verdict: if supported {
            Verdict::Supported
        } else {
            Verdict::Unsupported
        },
        detected_value: detected,
        required_value: required.into(),
        evidence: evidence.into(),
        remediation_hint: if supported {
            "None".into()
        } else {
            remediation
        },
    }
}

pub fn evaluate(inventory: &SystemInventory, section: &str) -> CompatibilityReport {
    let mut items = Vec::new();
    if matches!(section, "all" | "platform" | "inventory") {
        items.push(item(
            "platform",
            inventory.cpu.architecture == "x86_64",
            format!("{}-linux", inventory.cpu.architecture),
            "x86_64-linux",
            "CPU architecture / host OS",
            "gufo only supports Linux x86-64".into(),
        ));
    }
    if matches!(section, "all" | "gpu" | "inventory") {
        items.push(item(
            "gpu",
            inventory.gpu.architecture == "gfx1151",
            inventory.gpu.architecture.clone(),
            "gfx1151",
            &inventory.gpu.source,
            format!(
                "gufo requires AMD Strix Halo gfx1151 GPU (Radeon 8060S / 8050S). Detected: {}",
                inventory.gpu.architecture
            ),
        ));
    }
    if matches!(section, "all" | "inventory") {
        items.push(item(
            "kernel_driver",
            matches!(
                inventory.toolchain.amdgpu_status.as_str(),
                "loaded" | "built-in"
            ),
            format!("amdgpu: {}", inventory.toolchain.amdgpu_status),
            "amdgpu: loaded or built-in",
            "/sys/module/amdgpu",
            "Ensure Linux kernel >= 6.10 with the amdgpu driver loaded".into(),
        ));
    }
    let overall_verdict = if items.iter().any(|i| i.verdict == Verdict::Unsupported) {
        Verdict::Unsupported
    } else {
        Verdict::Supported
    };
    CompatibilityReport {
        overall_verdict,
        items,
    }
}

impl CompatibilityReport {
    pub fn to_json(&self) -> String {
        let mut out = format!(
            "{{\n  \"overallVerdict\": \"{}\",\n  \"items\": [\n",
            self.overall_verdict.as_str()
        );
        for (index, item) in self.items.iter().enumerate() {
            let _ = write!(
                out,
                "    {{\n      \"component\": \"{}\",\n      \"verdict\": \"{}\",\n      \
                 \"detectedValue\": \"{}\",\n      \"requiredValue\": \"{}\",\n      \
                 \"evidence\": \"{}\",\n      \"remediationHint\": \"{}\"\n    }}",
                escape(&item.component),
                item.verdict.as_str(),
                escape(&item.detected_value),
                escape(&item.required_value),
                escape(&item.evidence),
                escape(&item.remediation_hint)
            );
            if index + 1 < self.items.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ]\n}\n");
        out
    }

    pub fn to_human(&self) -> String {
        let mut out = format!(
            "=== Compatibility Verdict: [{}] ===\n\n",
            self.overall_verdict.as_str()
        );
        for item in &self.items {
            let _ = write!(
                out,
                "  - {} [{}]\n      Detected   : {}\n      Required   : {}\n",
                item.component,
                item.verdict.as_str(),
                item.detected_value,
                item.required_value
            );
            if item.verdict != Verdict::Supported
                && !item.remediation_hint.is_empty()
                && item.remediation_hint != "None"
            {
                let _ = writeln!(out, "      Remediation: {}", item.remediation_hint);
            }
        }
        out
    }
}
