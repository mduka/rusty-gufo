//! The diagnostic report (`report.cpp`).

use std::collections::BTreeMap;
use std::fmt::Write;

use super::compatibility::{CompatibilityReport, Verdict};
use super::fingerprint::MachineFingerprint;
use super::inventory::SystemInventory;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pass,
    Warn,
    Fail,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Warn => "WARN",
            Status::Fail => "FAIL",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub name: String,
    pub status: Status,
    pub message: String,
    pub details: BTreeMap<String, String>,
}

impl Check {
    pub fn new(name: &str, status: Status, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status,
            message: message.into(),
            details: BTreeMap::new(),
        }
    }

    pub fn detail(mut self, key: &str, value: impl ToString) -> Self {
        self.details.insert(key.into(), value.to_string());
        self
    }
}

/// UTC now as `%Y-%m-%dT%H:%M:%SZ`.
pub fn iso8601_utc() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// `EscapeJson` of the report: JSON escapes with `\u00XX` for other
/// control characters.
fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\u{8}' => escaped.push_str("\\b"),
            '\u{c}' => escaped.push_str("\\f"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(escaped, "\\u{:04x}", c as u32);
            }
            c => escaped.push(c),
        }
    }
    escaped
}

pub struct DiagnosticReport {
    schema_version: String,
    engine_revision: String,
    timestamp: String,
    status: Status,
    checks: Vec<Check>,
    warnings: Vec<String>,
    errors: Vec<String>,
    inventory: Option<SystemInventory>,
    compatibility: Option<CompatibilityReport>,
    fingerprint: Option<MachineFingerprint>,
}

impl DiagnosticReport {
    pub fn new(engine_revision: &str) -> Self {
        Self {
            schema_version: "1.0.0".into(),
            engine_revision: engine_revision.into(),
            timestamp: iso8601_utc(),
            status: Status::Pass,
            checks: Vec::new(),
            warnings: Vec::new(),
            errors: Vec::new(),
            inventory: None,
            compatibility: None,
            fingerprint: None,
        }
    }

    pub fn status(&self) -> Status {
        self.status
    }

    pub fn add_check(&mut self, check: Check) {
        self.checks.push(check);
        self.recompute_status();
    }

    pub fn add_error(&mut self, error: impl Into<String>) {
        self.errors.push(error.into());
        self.status = Status::Fail;
    }

    pub fn set_inventory(&mut self, inventory: SystemInventory) {
        self.inventory = Some(inventory);
    }

    pub fn set_compatibility(&mut self, compatibility: CompatibilityReport) {
        if compatibility.overall_verdict == Verdict::Unsupported {
            self.status = Status::Fail;
        }
        self.compatibility = Some(compatibility);
    }

    pub fn set_fingerprint(&mut self, fingerprint: MachineFingerprint) {
        self.fingerprint = Some(fingerprint);
    }

    fn recompute_status(&mut self) {
        let mut status = Status::Pass;
        for check in &self.checks {
            if check.status == Status::Fail {
                status = Status::Fail;
                break;
            }
            if check.status == Status::Warn {
                status = Status::Warn;
            }
        }
        let unsupported = self
            .compatibility
            .as_ref()
            .is_some_and(|c| c.overall_verdict == Verdict::Unsupported);
        if !self.errors.is_empty() || unsupported {
            status = Status::Fail;
        } else if !self.warnings.is_empty() && status == Status::Pass {
            status = Status::Warn;
        }
        self.status = status;
    }

    pub fn to_json(&self) -> String {
        let mut out = format!(
            "{{\n  \"schemaVersion\": \"{}\",\n  \"engineRevision\": \"{}\",\n  \
             \"timestamp\": \"{}\",\n  \"status\": \"{}\",\n",
            escape(&self.schema_version),
            escape(&self.engine_revision),
            escape(&self.timestamp),
            self.status.as_str()
        );
        if let Some(fingerprint) = &self.fingerprint {
            let _ = write!(
                out,
                "  \"fingerprintId\": \"{}\",\n  \"fingerprint\": {},\n",
                escape(&fingerprint.fingerprint_id),
                fingerprint.to_json()
            );
        }
        if let Some(inventory) = &self.inventory {
            let _ = writeln!(out, "  \"inventory\": {},", inventory.to_json());
        }
        if let Some(compatibility) = &self.compatibility {
            let _ = writeln!(out, "  \"compatibility\": {},", compatibility.to_json());
        }
        out.push_str("  \"checks\": [\n");
        for (index, check) in self.checks.iter().enumerate() {
            let _ = write!(
                out,
                "    {{\n      \"name\": \"{}\",\n      \"status\": \"{}\",\n      \
                 \"message\": \"{}\",\n      \"details\": {{",
                escape(&check.name),
                check.status.as_str(),
                escape(&check.message)
            );
            if check.details.is_empty() {
                out.push_str("}\n");
            } else {
                out.push('\n');
                for (detail, (key, value)) in check.details.iter().enumerate() {
                    let _ = write!(out, "        \"{}\": \"{}\"", escape(key), escape(value));
                    if detail + 1 < check.details.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str("      }\n");
            }
            out.push_str("    }");
            if index + 1 < self.checks.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ],\n");
        for (name, list, last) in [
            ("warnings", &self.warnings, false),
            ("errors", &self.errors, true),
        ] {
            let _ = write!(out, "  \"{name}\": [");
            if list.is_empty() {
                out.push(']');
            } else {
                out.push('\n');
                for (index, entry) in list.iter().enumerate() {
                    let _ = write!(out, "    \"{}\"", escape(entry));
                    if index + 1 < list.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str("  ]");
            }
            out.push_str(if last { "\n" } else { ",\n" });
        }
        out.push_str("}\n");
        out
    }

    pub fn to_human(&self) -> String {
        let mut out = format!(
            "=== Gufo Diagnostics ===\nEngine Revision : {}\nSchema Version  : {}\n\
             Timestamp       : {}\nOverall Status  : [{}]\n\n",
            self.engine_revision,
            self.schema_version,
            self.timestamp,
            self.status.as_str()
        );
        if let Some(fingerprint) = &self.fingerprint {
            let _ = writeln!(out, "{}", fingerprint.to_human());
        }
        if let Some(inventory) = &self.inventory {
            let _ = writeln!(out, "{}", inventory.to_human());
        }
        if let Some(compatibility) = &self.compatibility {
            let _ = writeln!(out, "{}", compatibility.to_human());
        }
        out.push_str("--- Checks ---\n");
        for check in &self.checks {
            let _ = writeln!(
                out,
                "  [{}] {}: {}",
                check.status.as_str(),
                check.name,
                check.message
            );
            for (key, value) in &check.details {
                let _ = writeln!(out, "      - {key}: {value}");
            }
        }
        if !self.warnings.is_empty() {
            out.push_str("\n--- Warnings ---\n");
            for warning in &self.warnings {
                let _ = writeln!(out, "  [WARN] {warning}");
            }
        }
        if !self.errors.is_empty() {
            out.push_str("\n--- Errors ---\n");
            for error in &self.errors {
                let _ = writeln!(out, "  [ERROR] {error}");
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_follows_checks_errors_and_compatibility() {
        let mut report = DiagnosticReport::new("dev");
        report.add_check(Check::new("a", Status::Warn, "w"));
        assert_eq!(report.status(), Status::Warn);
        report.add_error("broken");
        assert_eq!(report.status(), Status::Fail);
        report.add_check(Check::new("b", Status::Pass, "p"));
        assert_eq!(
            report.status(),
            Status::Fail,
            "errors keep the report failed"
        );

        let mut unsupported = DiagnosticReport::new("dev");
        unsupported.set_compatibility(CompatibilityReport {
            overall_verdict: Verdict::Unsupported,
            items: Vec::new(),
        });
        assert_eq!(unsupported.status(), Status::Fail);
        unsupported.add_check(Check::new("c", Status::Pass, "p"));
        assert_eq!(unsupported.status(), Status::Fail);
    }

    #[test]
    fn json_escapes_control_characters() {
        let mut report = DiagnosticReport::new("dev");
        report.add_check(Check::new("x", Status::Pass, "a\"b\u{1}\tc").detail("k", 1));
        let json = report.to_json();
        assert!(json.contains(r#""message": "a\"b\u0001\tc""#), "{json}");
        assert!(json.contains("\"details\": {\n        \"k\": \"1\"\n      }\n"));
        assert!(json.ends_with("  \"warnings\": [],\n  \"errors\": []\n}\n"));
    }
}
