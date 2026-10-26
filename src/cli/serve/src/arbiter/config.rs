//! The arbiter's rules (`gufo arbiter --config FILE`, TOML).
//!
//! ```toml
//! [background]           # text generation
//! min_share = 0.25       # the least GPU share under full pressure
//! lease_ms = 500         # a grant lapses unless renewed within this
//!
//! [tts]                  # streamed speech
//! buffer_target_s = 2.5  # pressure while the playback buffer is below this
//!
//! [[instances]]          # rules for matching servers
//! modality = "llm"
//! model = "qwen3.8-flash-next"
//! min_share = 0.4
//! ```
//!
//! Every key is optional; unknown keys are errors, so a misspelt rule never
//! silently does nothing.

use std::time::Duration;

use gufo_core::json::Value;

use super::Identity;
use crate::gpu_share::MIN_SHARE;

#[derive(Debug, Clone, PartialEq)]
pub struct ArbiterConfig {
    /// The least share a background server gets under full pressure.
    pub min_share: f32,
    /// How long a grant holds without renewal.
    pub lease: Duration,
    /// Streamed TTS raises pressure while its playback buffer holds less.
    pub tts_buffer_target_s: f64,
    /// Rules for matching servers; the first match applies.
    pub instances: Vec<InstanceRule>,
}

/// A rule for the servers whose modality and model match.
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceRule {
    pub modality: Option<String>,
    pub model: Option<String>,
    pub min_share: Option<f32>,
}

impl Default for ArbiterConfig {
    fn default() -> Self {
        Self {
            min_share: 0.25,
            lease: Duration::from_millis(500),
            tts_buffer_target_s: 2.5,
            instances: Vec::new(),
        }
    }
}

impl InstanceRule {
    fn matches(&self, identity: &Identity) -> bool {
        self.modality
            .as_ref()
            .is_none_or(|m| *m == identity.modality)
            && self.model.as_ref().is_none_or(|m| *m == identity.model)
    }
}

impl ArbiterConfig {
    pub fn parse(text: &str) -> Result<Self, String> {
        let table: toml::Table = text.parse().map_err(|e: toml::de::Error| e.to_string())?;
        let mut config = Self::default();
        for (key, value) in &table {
            match key.as_str() {
                "background" => {
                    let section = as_table(value, "background")?;
                    for (name, value) in section {
                        match name.as_str() {
                            "min_share" => config.min_share = share(value, "background.min_share")?,
                            "lease_ms" => {
                                let ms = integer(value, "background.lease_ms", 100, 10_000)?;
                                config.lease = Duration::from_millis(ms);
                            }
                            other => return Err(format!("unknown key 'background.{other}'")),
                        }
                    }
                }
                "tts" => {
                    let section = as_table(value, "tts")?;
                    for (name, value) in section {
                        match name.as_str() {
                            "buffer_target_s" => {
                                let seconds = number(value, "tts.buffer_target_s")?;
                                if !(seconds > 0.0 && seconds <= 30.0) {
                                    return Err(
                                        "'tts.buffer_target_s' must be above 0 and at most 30"
                                            .into(),
                                    );
                                }
                                config.tts_buffer_target_s = seconds;
                            }
                            other => return Err(format!("unknown key 'tts.{other}'")),
                        }
                    }
                }
                "instances" => {
                    let rules = value
                        .as_array()
                        .ok_or("'instances' must be an array of tables ([[instances]])")?;
                    for (index, rule) in rules.iter().enumerate() {
                        config.instances.push(instance_rule(rule, index)?);
                    }
                }
                other => return Err(format!("unknown section '{other}'")),
            }
        }
        Ok(config)
    }

    /// The least share for `identity`: its first matching rule's, else the
    /// default.
    pub fn min_share_for(&self, identity: &Identity) -> f32 {
        self.instances
            .iter()
            .find(|rule| rule.matches(identity))
            .and_then(|rule| rule.min_share)
            .unwrap_or(self.min_share)
    }

    pub fn to_json(&self) -> Value {
        let mut instances = Value::array();
        for rule in &self.instances {
            let mut entry = Value::object();
            if let Some(modality) = &rule.modality {
                entry.set("modality", modality.as_str());
            }
            if let Some(model) = &rule.model {
                entry.set("model", model.as_str());
            }
            if let Some(min_share) = rule.min_share {
                entry.set("min_share", f64::from(min_share));
            }
            instances.push(entry);
        }
        Value::object()
            .with(
                "background",
                Value::object()
                    .with("min_share", f64::from(self.min_share))
                    .with("lease_ms", self.lease.as_millis() as u64),
            )
            .with(
                "tts",
                Value::object().with("buffer_target_s", self.tts_buffer_target_s),
            )
            .with("instances", instances)
    }
}

fn instance_rule(value: &toml::Value, index: usize) -> Result<InstanceRule, String> {
    let at = format!("instances[{index}]");
    let table = as_table(value, &at)?;
    let mut rule = InstanceRule {
        modality: None,
        model: None,
        min_share: None,
    };
    for (name, value) in table {
        match name.as_str() {
            "modality" => rule.modality = Some(string(value, &format!("{at}.modality"))?),
            "model" => rule.model = Some(string(value, &format!("{at}.model"))?),
            "min_share" => rule.min_share = Some(share(value, &format!("{at}.min_share"))?),
            other => return Err(format!("unknown key '{at}.{other}'")),
        }
    }
    if rule.modality.is_none() && rule.model.is_none() {
        return Err(format!("'{at}' needs a 'modality' or a 'model' to match"));
    }
    Ok(rule)
}

fn as_table<'a>(value: &'a toml::Value, name: &str) -> Result<&'a toml::Table, String> {
    value
        .as_table()
        .ok_or_else(|| format!("'{name}' must be a table"))
}

fn string(value: &toml::Value, name: &str) -> Result<String, String> {
    value
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("'{name}' must be a non-empty string"))
}

fn number(value: &toml::Value, name: &str) -> Result<f64, String> {
    value
        .as_float()
        .or_else(|| value.as_integer().map(|i| i as f64))
        .ok_or_else(|| format!("'{name}' must be a number"))
}

fn integer(value: &toml::Value, name: &str, low: u64, high: u64) -> Result<u64, String> {
    value
        .as_integer()
        .and_then(|i| u64::try_from(i).ok())
        .filter(|i| (low..=high).contains(i))
        .ok_or_else(|| format!("'{name}' must be an integer from {low} to {high}"))
}

fn share(value: &toml::Value, name: &str) -> Result<f32, String> {
    let share = number(value, name)?;
    if !(f64::from(MIN_SHARE)..=1.0).contains(&share) {
        return Err(format!("'{name}' must be from {MIN_SHARE} to 1"));
    }
    Ok(share as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(modality: &str, model: &str) -> Identity {
        Identity {
            modality: modality.into(),
            model: model.into(),
            address: "127.0.0.1:8080".into(),
            pid: 1,
        }
    }

    #[test]
    fn an_empty_file_keeps_the_defaults() {
        assert_eq!(ArbiterConfig::parse("").unwrap(), ArbiterConfig::default());
    }

    #[test]
    fn every_rule_parses() {
        let config = ArbiterConfig::parse(
            "[background]\nmin_share = 0.3\nlease_ms = 800\n\n[tts]\nbuffer_target_s = 2\n\n\
             [[instances]]\nmodality = \"llm\"\nmodel = \"big\"\nmin_share = 0.5\n",
        )
        .unwrap();
        assert_eq!(config.min_share, 0.3);
        assert_eq!(config.lease, Duration::from_millis(800));
        assert_eq!(config.tts_buffer_target_s, 2.0);
        assert_eq!(config.min_share_for(&identity("llm", "big")), 0.5);
        assert_eq!(config.min_share_for(&identity("llm", "small")), 0.3);
    }

    #[test]
    fn mistakes_are_errors() {
        for (text, needle) in [
            ("[backgrund]\n", "unknown section 'backgrund'"),
            (
                "[background]\nmin_shar = 0.3\n",
                "unknown key 'background.min_shar'",
            ),
            ("[background]\nmin_share = 0\n", "must be from"),
            ("[background]\nmin_share = 1.5\n", "must be from"),
            ("[background]\nlease_ms = 5\n", "integer from 100"),
            ("[tts]\nbuffer_target_s = -1\n", "above 0"),
            (
                "[[instances]]\nmin_share = 0.5\n",
                "needs a 'modality' or a 'model'",
            ),
            (
                "[[instances]]\nmodality = \"llm\"\nshare = 1\n",
                "unknown key 'instances[0].share'",
            ),
            ("background = 1\n", "must be a table"),
            ("[background\n", ""),
        ] {
            let error = ArbiterConfig::parse(text).unwrap_err();
            assert!(error.contains(needle), "{text:?}: {error}");
        }
    }

    #[test]
    fn the_effective_rules_print_as_json() {
        let json = ArbiterConfig::default().to_json().dump().unwrap();
        assert!(json.contains("\"min_share\":0.25"), "{json}");
        assert!(json.contains("\"lease_ms\":500"), "{json}");
        assert!(json.contains("\"buffer_target_s\":2.5"), "{json}");
    }
}
