//! GPU sharing between `gufo serve` processes through an arbiter
//! (`gufo arbiter`).
//!
//! A server started with `--arbiter HOST:PORT` connects to the arbiter's
//! WebSocket (`/v1/connect`), introduces itself (`hello`) and reports its
//! state (`status`). The arbiter turns the reports of realtime servers
//! (streamed TTS) into pressure and grants background servers (text
//! generation) a GPU share for a lifetime (`share`), which they keep to by
//! pausing between steps. The rules live in the arbiter
//! ([`ArbiterConfig`]); a server only measures and obeys. A grant lapses to
//! the full share when it is not renewed, so a lost arbiter never leaves a
//! server slowed. Without `--arbiter` none of this runs.
//!
//! Messages are JSON objects with a `type`, one per WebSocket text frame:
//!
//! | From | `type` | Fields |
//! | --- | --- | --- |
//! | server | `hello` | `protocol`, `modality`, `model`, `address`, `pid` |
//! | server | `status` | text: `share`, `yielded_s`; TTS: `active`, `buffer_s`, `audio_s` |
//! | arbiter | `welcome` | `protocol`, `id` |
//! | arbiter | `share` | `share`, `ttl_ms` |
//! | arbiter | `error` | `message` (then the arbiter closes) |

mod client;
mod config;
mod playback;
mod service;
mod websocket_client;

pub use client::{ArbiterClient, Presence, StatusSource};
pub use config::{ArbiterConfig, InstanceRule};
pub use playback::{PlaybackBuffer, PlaybackSnapshot};
pub use service::ArbiterService;

use gufo_core::json::Value;

/// The protocol version servers and the arbiter exchange in `hello`.
pub const PROTOCOL_VERSION: u32 = 1;
/// The arbiter's default port.
pub const DEFAULT_PORT: u16 = 7071;
/// The WebSocket servers connect to.
pub const CONNECT_PATH: &str = "/v1/connect";

/// Who a server is: its modality, served model and address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    /// `llm`, `tts`, `asr`, `embedding` or `image`.
    pub modality: String,
    /// The model ID the server serves.
    pub model: String,
    /// The server's own HTTP address, `host:port`.
    pub address: String,
    pub pid: u32,
}

impl Identity {
    /// The arbiter's name for the server: `modality:model@address`.
    pub fn id(&self) -> String {
        format!("{}:{}@{}", self.modality, self.model, self.address)
    }

    pub fn hello(&self) -> Value {
        Value::object()
            .with("type", "hello")
            .with("protocol", PROTOCOL_VERSION)
            .with("modality", self.modality.as_str())
            .with("model", self.model.as_str())
            .with("address", self.address.as_str())
            .with("pid", self.pid)
    }

    /// The identity of a `hello` message.
    pub fn from_hello(message: &Value) -> Result<Self, String> {
        if message.member_str("type", "") != "hello" {
            return Err("the first message must be a hello".into());
        }
        let protocol = message.find("protocol").map_or(0.0, |v| v.as_double(0.0));
        if protocol != f64::from(PROTOCOL_VERSION) {
            return Err(format!(
                "protocol {protocol} is not supported; this arbiter speaks {PROTOCOL_VERSION}"
            ));
        }
        let text = |name: &str| -> Result<String, String> {
            match message.find(name) {
                Some(value) if value.is_string() && !value.str().is_empty() => {
                    Ok(value.str().to_owned())
                }
                _ => Err(format!("hello needs a non-empty string '{name}'")),
            }
        };
        let modality = text("modality")?;
        if !Role::KNOWN_MODALITIES.contains(&modality.as_str()) {
            return Err(format!("unknown modality '{modality}'"));
        }
        let pid = message
            .find("pid")
            .filter(|v| v.is_number())
            .map(|v| v.as_double(0.0))
            .filter(|&v| v >= 0.0 && v <= f64::from(u32::MAX) && v.fract() == 0.0)
            .ok_or("hello needs an integer 'pid'")?;
        Ok(Self {
            modality,
            model: text("model")?,
            address: text("address")?,
            pid: pid as u32,
        })
    }
}

/// How the arbiter treats a server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Its reports raise pressure (streamed TTS).
    Realtime,
    /// It yields under pressure (text generation).
    Background,
    /// Reported only.
    Observed,
}

impl Role {
    const KNOWN_MODALITIES: [&'static str; 5] = ["llm", "tts", "asr", "embedding", "image"];

    pub fn of(modality: &str) -> Self {
        match modality {
            "tts" => Self::Realtime,
            "llm" => Self::Background,
            _ => Self::Observed,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Realtime => "realtime",
            Self::Background => "background",
            Self::Observed => "observed",
        }
    }
}

/// A `share` message.
pub fn share_message(share: f32, ttl_ms: u64) -> Value {
    Value::object()
        .with("type", "share")
        .with("share", f64::from(share))
        .with("ttl_ms", ttl_ms)
}

/// An `error` message.
pub fn error_message(message: &str) -> Value {
    Value::object()
        .with("type", "error")
        .with("message", message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Identity {
        Identity {
            modality: "llm".into(),
            model: "qwen3.8-flash-next".into(),
            address: "127.0.0.1:8080".into(),
            pid: 4242,
        }
    }

    #[test]
    fn a_hello_round_trips() {
        let identity = identity();
        assert_eq!(
            Identity::from_hello(&identity.hello()),
            Ok(identity.clone())
        );
        assert_eq!(identity.id(), "llm:qwen3.8-flash-next@127.0.0.1:8080");
    }

    #[test]
    fn invalid_hellos_are_refused() {
        let hello = identity().hello();
        let mut wrong_protocol = hello.clone();
        wrong_protocol.set("protocol", 2);
        assert!(
            Identity::from_hello(&wrong_protocol)
                .unwrap_err()
                .contains("protocol 2")
        );
        let mut unknown = hello.clone();
        unknown.set("modality", "video");
        assert!(Identity::from_hello(&unknown).is_err());
        let mut empty_model = hello.clone();
        empty_model.set("model", "");
        assert!(Identity::from_hello(&empty_model).is_err());
        let mut status = hello;
        status.set("type", "status");
        assert!(Identity::from_hello(&status).is_err());
    }

    #[test]
    fn roles_follow_the_modality() {
        assert_eq!(Role::of("tts"), Role::Realtime);
        assert_eq!(Role::of("llm"), Role::Background);
        assert_eq!(Role::of("asr"), Role::Observed);
    }
}
