//! The arbiter (`gufo arbiter`): the registry of connected servers, the
//! policy that turns realtime pressure into background grants, and the
//! HTTP interface for monitoring and control.
//!
//! | Route | Use |
//! | --- | --- |
//! | `GET /v1/connect` | WebSocket for `gufo serve --arbiter` |
//! | `GET /v1/events` | WebSocket of state snapshots |
//! | `GET /v1/instances` | Servers, their last status and grant |
//! | `GET /v1/config` | The rules in force |
//! | `GET /metrics` | Prometheus text |
//! | `POST /v1/pressure` | `{"source", "pressure", "ttl_ms"}` from any client |
//! | `POST /v1/instances/{id}/share` | `{"share", "ttl_ms"}`; `"share": null` ends it |

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use gufo_core::json::{self, Value};

use super::config::ArbiterConfig;
use super::{Identity, PROTOCOL_VERSION, Role, error_message, share_message};
use crate::gpu_share::MIN_SHARE;
use crate::http::websocket::{WebSocket, is_websocket_upgrade, upgrade_websocket};
use crate::http::{HttpRequest, HttpResponse};
use crate::logging;

/// How often the policy runs.
const TICK: Duration = Duration::from_millis(50);
/// A realtime status older than this no longer raises pressure.
const STALE_STATUS: Duration = Duration::from_secs(1);
/// The least interval between two snapshots to `/v1/events`.
const EVENT_INTERVAL: Duration = Duration::from_millis(250);
/// The longest lifetime a client may give pressure or a manual share.
const MAXIMUM_TTL: Duration = Duration::from_secs(3600);

struct Instance {
    identity: Identity,
    role: Role,
    connected: Instant,
    status: Value,
    status_at: Option<Instant>,
    /// The share last sent; 1 when none limits the server.
    share: f32,
    share_sent: Option<Instant>,
    /// A share set through `POST /v1/instances/{id}/share`, until it expires.
    manual: Option<(f32, Instant)>,
    outbox: Sender<String>,
}

#[derive(Default)]
struct State {
    instances: BTreeMap<String, Instance>,
    /// Pressure from `POST /v1/pressure`, by source, until it expires.
    external: BTreeMap<String, (f32, Instant)>,
    pressure: f32,
    subscribers: Vec<Sender<String>>,
    last_event: Option<Instant>,
    event_due: bool,
}

pub struct ArbiterService {
    config: ArbiterConfig,
    state: Mutex<State>,
    stop: AtomicBool,
}

impl ArbiterService {
    /// The arbiter with `config`; its policy runs on a thread until the
    /// service is dropped.
    pub fn start(config: ArbiterConfig) -> std::io::Result<Arc<Self>> {
        let service = Arc::new(Self {
            config,
            state: Mutex::new(State::default()),
            stop: AtomicBool::new(false),
        });
        let weak: Weak<Self> = Arc::downgrade(&service);
        std::thread::Builder::new()
            .name("gufo-arbiter-policy".into())
            .spawn(move || {
                while let Some(service) = weak.upgrade() {
                    if service.stop.load(Ordering::Acquire) {
                        break;
                    }
                    service.tick(Instant::now());
                    drop(service);
                    std::thread::sleep(TICK);
                }
            })?;
        Ok(service)
    }

    pub fn config(&self) -> &ArbiterConfig {
        &self.config
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn handle(self: &Arc<Self>, request: &HttpRequest) -> HttpResponse {
        let method = request.method.as_str();
        let path = request.path.as_str();
        match (method, path) {
            ("GET", super::CONNECT_PATH) if is_websocket_upgrade(request) => {
                let service = self.clone();
                upgrade_websocket(
                    request,
                    Box::new(move |socket: &WebSocket| service.serve_instance(socket)),
                )
            }
            ("GET", "/v1/events") if is_websocket_upgrade(request) => {
                let service = self.clone();
                upgrade_websocket(
                    request,
                    Box::new(move |socket: &WebSocket| service.serve_events(socket)),
                )
            }
            ("GET", "/v1/instances") => json_response(&self.snapshot(Instant::now())),
            ("GET", "/v1/config") => json_response(&self.config.to_json()),
            ("GET", "/metrics") => HttpResponse::new(200, "OK", self.metrics())
                .with_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8"),
            ("POST", "/v1/pressure") => self.post_pressure(request),
            ("POST", _) if path.starts_with("/v1/instances/") && path.ends_with("/share") => {
                let segment = &path["/v1/instances/".len()..path.len() - "/share".len()];
                match percent_decode(segment) {
                    Some(id) => self.post_share(&id, request),
                    None => error_response(
                        400,
                        "Bad Request",
                        "invalid instance id escape",
                        "invalid_id",
                    ),
                }
            }
            _ => error_response(404, "Not Found", "no such arbiter route", "not_found"),
        }
    }

    /// A `gufo serve` connection: `hello`, then `status` reports until it
    /// closes; grants go out through the instance's outbox.
    fn serve_instance(self: &Arc<Self>, socket: &WebSocket) {
        let refuse = |message: &str| {
            if let Ok(text) = error_message(message).dump() {
                socket.send_text(&text);
            }
            socket.close(1008);
        };
        let Some(first) = socket.receive() else {
            return;
        };
        let identity = match json::parse_str(&first)
            .map_err(|e| e.to_string())
            .and_then(|hello| Identity::from_hello(&hello))
        {
            Ok(identity) => identity,
            Err(message) => return refuse(&message),
        };
        let id = identity.id();
        let (outbox, mail) = mpsc::channel::<String>();
        {
            let mut state = self.state();
            if state.instances.contains_key(&id) {
                drop(state);
                return refuse(&format!("{id} is already connected"));
            }
            state.instances.insert(
                id.clone(),
                Instance {
                    role: Role::of(&identity.modality),
                    identity,
                    connected: Instant::now(),
                    status: Value::object(),
                    status_at: None,
                    share: 1.0,
                    share_sent: None,
                    manual: None,
                    outbox,
                },
            );
            state.event_due = true;
        }
        logging::info("arbiter", &format!("{id} connected"));
        let welcome = Value::object()
            .with("type", "welcome")
            .with("protocol", PROTOCOL_VERSION)
            .with("id", id.as_str());
        if let Ok(text) = welcome.dump() {
            socket.send_text(&text);
        }
        std::thread::scope(|scope| {
            scope.spawn(move || forward(&mail, socket));
            while let Some(text) = socket.receive() {
                let Ok(message) = json::parse_str(&text) else {
                    continue;
                };
                if message.member_str("type", "") == "status" {
                    let mut state = self.state();
                    if let Some(instance) = state.instances.get_mut(&id) {
                        instance.status = message;
                        instance.status_at = Some(Instant::now());
                    }
                }
            }
            // Dropping the instance closes its outbox, which ends `forward`.
            let mut state = self.state();
            state.instances.remove(&id);
            state.event_due = true;
        });
        logging::info("arbiter", &format!("{id} disconnected"));
    }

    /// A `/v1/events` subscriber: a snapshot now and on every change.
    fn serve_events(self: &Arc<Self>, socket: &WebSocket) {
        let (outbox, mail) = mpsc::channel::<String>();
        if let Ok(text) = self.snapshot(Instant::now()).dump() {
            let _ = outbox.send(text);
        }
        self.state().subscribers.push(outbox);
        forward(&mail, socket);
    }

    fn post_pressure(&self, request: &HttpRequest) -> HttpResponse {
        let body = match json::parse(&request.body) {
            Ok(body) if body.is_object() => body,
            _ => {
                return error_response(
                    400,
                    "Bad Request",
                    "body must be a JSON object",
                    "invalid_body",
                );
            }
        };
        let source = body.member_str("source", "");
        let pressure = body
            .find("pressure")
            .filter(|v| v.is_number())
            .map(|v| v.as_double(0.0));
        let Some(ttl) = ttl(&body) else {
            return error_response(
                400,
                "Bad Request",
                "'ttl_ms' must be from 1 to 3600000",
                "invalid_ttl",
            );
        };
        let Some(pressure) = pressure.filter(|p| (0.0..=1.0).contains(p)) else {
            return error_response(
                400,
                "Bad Request",
                "'pressure' must be a number from 0 to 1",
                "invalid_pressure",
            );
        };
        if source.is_empty() {
            return error_response(
                400,
                "Bad Request",
                "'source' must be a non-empty string",
                "invalid_source",
            );
        }
        let mut state = self.state();
        state
            .external
            .insert(source, (pressure as f32, Instant::now() + ttl));
        state.event_due = true;
        json_response(&Value::object().with("status", "ok"))
    }

    fn post_share(&self, id: &str, request: &HttpRequest) -> HttpResponse {
        let body = match json::parse(&request.body) {
            Ok(body) if body.is_object() => body,
            _ => {
                return error_response(
                    400,
                    "Bad Request",
                    "body must be a JSON object",
                    "invalid_body",
                );
            }
        };
        let share = match body.find("share") {
            Some(value) if value.is_null() => None,
            Some(value) if value.is_number() => {
                let share = value.as_double(1.0);
                if !(f64::from(MIN_SHARE)..=1.0).contains(&share) {
                    return error_response(
                        400,
                        "Bad Request",
                        &format!("'share' must be from {MIN_SHARE} to 1 or null"),
                        "invalid_share",
                    );
                }
                Some(share as f32)
            }
            _ => {
                return error_response(
                    400,
                    "Bad Request",
                    "'share' must be a number or null",
                    "invalid_share",
                );
            }
        };
        let ttl = if share.is_some() {
            ttl(&body)
        } else {
            Some(Duration::ZERO)
        };
        let Some(ttl) = ttl else {
            return error_response(
                400,
                "Bad Request",
                "'ttl_ms' must be from 1 to 3600000",
                "invalid_ttl",
            );
        };
        let mut state = self.state();
        let Some(instance) = state.instances.get_mut(id) else {
            return error_response(
                404,
                "Not Found",
                &format!("no instance {id}"),
                "unknown_instance",
            );
        };
        if instance.role != Role::Background {
            return error_response(
                400,
                "Bad Request",
                &format!("{id} does not take GPU grants"),
                "not_background",
            );
        }
        instance.manual = share.map(|share| (share, Instant::now() + ttl));
        state.event_due = true;
        json_response(&Value::object().with("status", "ok"))
    }

    /// One policy round: pressure from realtime reports and client
    /// pressure, then the grant of every background server.
    fn tick(&self, now: Instant) {
        let mut state = self.state();
        state.external.retain(|_, (_, expires)| *expires > now);
        let mut pressure = state
            .external
            .values()
            .map(|(pressure, _)| *pressure)
            .fold(0.0f32, f32::max);
        for instance in state.instances.values() {
            if instance.role == Role::Realtime {
                pressure = pressure.max(self.realtime_pressure(instance, now));
            }
        }
        if (pressure - state.pressure).abs() >= 0.01 {
            state.event_due = true;
        }
        state.pressure = pressure;
        let lease = self.config.lease;
        let mut changed = false;
        for instance in state.instances.values_mut() {
            if instance.role != Role::Background {
                continue;
            }
            if instance.manual.is_some_and(|(_, expires)| expires <= now) {
                instance.manual = None;
            }
            let share = instance.manual.map_or_else(
                || {
                    let min_share = self.config.min_share_for(&instance.identity);
                    1.0 - pressure * (1.0 - min_share)
                },
                |(share, _)| share,
            );
            let renew = instance
                .share_sent
                .is_none_or(|sent| now.saturating_duration_since(sent) >= lease / 2);
            let message = if share < 1.0 {
                ((share - instance.share).abs() >= 0.01 || renew)
                    .then(|| share_message(share, lease.as_millis() as u64))
            } else {
                (instance.share < 1.0).then(|| share_message(1.0, 0))
            };
            if let Some(message) = message {
                if let Ok(text) = message.dump() {
                    let _ = instance.outbox.send(text);
                }
                changed |= (share - instance.share).abs() >= 0.01;
                instance.share = share;
                instance.share_sent = Some(now);
            }
        }
        if changed {
            state.event_due = true;
        }
        let due = state.event_due
            && state
                .last_event
                .is_none_or(|last| now.saturating_duration_since(last) >= EVENT_INTERVAL);
        if due && !state.subscribers.is_empty() {
            let snapshot = snapshot_of(&state, now, &self.config);
            if let Ok(text) = snapshot.dump() {
                state
                    .subscribers
                    .retain(|subscriber| subscriber.send(text.clone()).is_ok());
            }
            state.last_event = Some(now);
            state.event_due = false;
        } else if due {
            state.event_due = false;
        }
    }

    /// A streamed TTS presses while it synthesizes and its playback buffer
    /// holds less than the target.
    fn realtime_pressure(&self, instance: &Instance, now: Instant) -> f32 {
        let fresh = instance
            .status_at
            .is_some_and(|at| now.saturating_duration_since(at) <= STALE_STATUS);
        if !fresh
            || !instance
                .status
                .find("active")
                .is_some_and(|a| a.as_bool(false))
        {
            return 0.0;
        }
        let buffer = instance.status.member_double("buffer_s", 0.0).max(0.0);
        (1.0 - buffer / self.config.tts_buffer_target_s).clamp(0.0, 1.0) as f32
    }

    fn snapshot(&self, now: Instant) -> Value {
        snapshot_of(&self.state(), now, &self.config)
    }

    fn metrics(&self) -> String {
        let state = self.state();
        let mut out = String::new();
        let _ = writeln!(
            out,
            "# HELP gufo_arbiter_pressure Pressure from realtime servers and clients, 0 to 1."
        );
        let _ = writeln!(out, "# TYPE gufo_arbiter_pressure gauge");
        let _ = writeln!(out, "gufo_arbiter_pressure {}", state.pressure);
        let _ = writeln!(out, "# HELP gufo_arbiter_instances Connected servers.");
        let _ = writeln!(out, "# TYPE gufo_arbiter_instances gauge");
        for role in [Role::Realtime, Role::Background, Role::Observed] {
            let count = state.instances.values().filter(|i| i.role == role).count();
            let _ = writeln!(
                out,
                "gufo_arbiter_instances{{role=\"{}\"}} {count}",
                role.name()
            );
        }
        let _ = writeln!(
            out,
            "# HELP gufo_arbiter_share The GPU share granted to a background server."
        );
        let _ = writeln!(out, "# TYPE gufo_arbiter_share gauge");
        for (id, instance) in &state.instances {
            if instance.role == Role::Background {
                let _ = writeln!(
                    out,
                    "gufo_arbiter_share{{{}}} {}",
                    labels(id, instance),
                    instance.share
                );
            }
        }
        let _ = writeln!(
            out,
            "# HELP gufo_arbiter_yielded_seconds_total Time a background server paused for grants."
        );
        let _ = writeln!(out, "# TYPE gufo_arbiter_yielded_seconds_total counter");
        for (id, instance) in &state.instances {
            if let Some(yielded) = instance.status.find("yielded_s") {
                let _ = writeln!(
                    out,
                    "gufo_arbiter_yielded_seconds_total{{{}}} {}",
                    labels(id, instance),
                    yielded.as_double(0.0)
                );
            }
        }
        let _ = writeln!(
            out,
            "# HELP gufo_arbiter_tts_buffer_seconds The modelled playback buffer of a streamed TTS."
        );
        let _ = writeln!(out, "# TYPE gufo_arbiter_tts_buffer_seconds gauge");
        for (id, instance) in &state.instances {
            if let Some(buffer) = instance.status.find("buffer_s") {
                let _ = writeln!(
                    out,
                    "gufo_arbiter_tts_buffer_seconds{{{}}} {}",
                    labels(id, instance),
                    buffer.as_double(0.0)
                );
            }
        }
        let _ = writeln!(
            out,
            "# HELP gufo_arbiter_tts_audio_seconds_total Audio a streamed TTS sent."
        );
        let _ = writeln!(out, "# TYPE gufo_arbiter_tts_audio_seconds_total counter");
        for (id, instance) in &state.instances {
            if let Some(audio) = instance.status.find("audio_s") {
                let _ = writeln!(
                    out,
                    "gufo_arbiter_tts_audio_seconds_total{{{}}} {}",
                    labels(id, instance),
                    audio.as_double(0.0)
                );
            }
        }
        out
    }
}

impl Drop for ArbiterService {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

/// Sends what arrives in `mail` until the outbox closes or the socket does.
fn forward(mail: &Receiver<String>, socket: &WebSocket) {
    loop {
        match mail.recv_timeout(Duration::from_millis(500)) {
            Ok(text) => {
                if !socket.send_text(&text) {
                    return;
                }
            }
            Err(RecvTimeoutError::Timeout) if !socket.cancelled() => {}
            Err(_) => return,
        }
    }
}

fn snapshot_of(state: &State, now: Instant, config: &ArbiterConfig) -> Value {
    let mut instances = Value::array();
    for (id, instance) in &state.instances {
        let mut entry = Value::object()
            .with("id", id.as_str())
            .with("modality", instance.identity.modality.as_str())
            .with("model", instance.identity.model.as_str())
            .with("address", instance.identity.address.as_str())
            .with("pid", instance.identity.pid)
            .with("role", instance.role.name())
            .with(
                "connected_s",
                now.saturating_duration_since(instance.connected)
                    .as_secs_f64(),
            );
        if instance.role == Role::Background {
            entry.set("share", f64::from(instance.share));
            entry.set(
                "min_share",
                f64::from(config.min_share_for(&instance.identity)),
            );
            if let Some((share, expires)) = instance.manual {
                entry.set(
                    "manual",
                    Value::object().with("share", f64::from(share)).with(
                        "ttl_ms",
                        expires.saturating_duration_since(now).as_millis() as u64,
                    ),
                );
            }
        }
        let mut status = Value::object();
        for (name, value) in instance.status.members() {
            if name != "type" {
                status.set(name, value.clone());
            }
        }
        entry.set("status", status);
        if let Some(at) = instance.status_at {
            entry.set(
                "status_age_ms",
                now.saturating_duration_since(at).as_millis() as u64,
            );
        }
        instances.push(entry);
    }
    let mut external = Value::array();
    for (source, (pressure, expires)) in &state.external {
        external.push(
            Value::object()
                .with("source", source.as_str())
                .with("pressure", f64::from(*pressure))
                .with(
                    "ttl_ms",
                    expires.saturating_duration_since(now).as_millis() as u64,
                ),
        );
    }
    Value::object()
        .with("type", "state")
        .with("pressure", f64::from(state.pressure))
        .with("instances", instances)
        .with("pressure_sources", external)
}

/// A path segment with its `%XX` escapes decoded (instance ids may hold
/// spaces); `None` for a broken escape or invalid UTF-8.
fn percent_decode(segment: &str) -> Option<String> {
    let bytes = segment.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = segment.get(i + 1..i + 3)?;
            decoded.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            decoded.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(decoded).ok()
}

fn ttl(body: &Value) -> Option<Duration> {
    body.find("ttl_ms")
        .filter(|v| v.is_number())
        .map(|v| v.as_double(0.0))
        .filter(|ms| *ms >= 1.0 && *ms <= MAXIMUM_TTL.as_millis() as f64)
        .map(|ms| Duration::from_millis(ms as u64))
}

fn labels(id: &str, instance: &Instance) -> String {
    format!(
        "instance=\"{}\",modality=\"{}\",model=\"{}\"",
        escape_label(id),
        escape_label(&instance.identity.modality),
        escape_label(&instance.identity.model)
    )
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn json_response(value: &Value) -> HttpResponse {
    HttpResponse::new(200, "OK", value.dump().unwrap_or_default())
}

fn error_response(status: u16, reason: &str, message: &str, code: &str) -> HttpResponse {
    let body = Value::object().with(
        "error",
        Value::object()
            .with("message", message)
            .with("type", "invalid_request_error")
            .with("code", code),
    );
    HttpResponse::new(status, reason, body.dump().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::percent_decode;

    #[test]
    fn instance_ids_decode() {
        assert_eq!(
            percent_decode("llm:Qwen3.8%20Flash%20Next@127.0.0.1:8080").as_deref(),
            Some("llm:Qwen3.8 Flash Next@127.0.0.1:8080")
        );
        assert_eq!(percent_decode("plain").as_deref(), Some("plain"));
        assert_eq!(percent_decode("broken%2"), None);
        assert_eq!(percent_decode("bad%zz"), None);
        assert_eq!(percent_decode("%ff"), None, "invalid UTF-8");
    }
}
