//! Server log lines (`src/cli/serve/logging.cpp`):
//! `YYYY-MM-DD HH:MM:SS [INFO] [component] message` on stderr, with control
//! bytes escaped and the level tinted on a terminal unless `NO_COLOR` is set.

use std::cell::{Cell, RefCell};
use std::fmt::Write as _;
use std::io::{IsTerminal, Write as _};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

/// Log verbosity from chattiest to quietest, so a threshold test is one
/// comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    const ALL: [LogLevel; 4] = [Self::Debug, Self::Info, Self::Warn, Self::Error];

    /// Parses `error`, `warn`, `info` or `debug`.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|level| level.name() == name)
    }

    /// The spelling [`Self::from_name`] accepts.
    pub fn name(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

/// The process-wide threshold: every modality, connection and scheduler
/// thread logs through it.
static LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Info as u8);

thread_local! {
    /// Lines logged by this thread while `capture` runs.
    static CAPTURE: RefCell<Option<String>> = const { RefCell::new(None) };
    /// This thread's threshold while `capture_at` runs.
    static LEVEL_OVERRIDE: Cell<Option<LogLevel>> = const { Cell::new(None) };
}

/// Sets the threshold once at startup, before any loader, scheduler or HTTP
/// work logs. The default is `Info`.
pub fn set_level(level: LogLevel) {
    LEVEL.store(level as u8, Ordering::Relaxed);
}

pub fn level() -> LogLevel {
    LogLevel::ALL[usize::from(LEVEL.load(Ordering::Relaxed))]
}

/// False when a line at `level` would be discarded; costly lines check it
/// before formatting.
pub fn enabled(level: LogLevel) -> bool {
    level >= LEVEL_OVERRIDE.with(Cell::get).unwrap_or_else(self::level)
}

/// Lines logged by any thread while `capture_all` runs.
static GLOBAL_CAPTURE: Mutex<Option<String>> = Mutex::new(None);

fn log_mutex() -> &'static Mutex<()> {
    static MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
    MUTEX.get_or_init(|| Mutex::new(()))
}

/// Escapes bytes below 0x20 and DEL as `\xNN` and truncates after `limit`
/// bytes (on a character boundary) with `...`.
fn safe_line(input: &str, limit: usize) -> String {
    let mut end = input.len().min(limit);
    while !input.is_char_boundary(end) {
        end -= 1;
    }
    let mut output = String::with_capacity(end);
    for c in input[..end].chars() {
        if (c as u32) < 0x20 || c == '\u{7f}' {
            let _ = write!(output, "\\x{:02x}", c as u32);
        } else {
            output.push(c);
        }
    }
    if input.len() > limit {
        output.push_str("...");
    }
    output
}

/// Local time as `YYYY-MM-DD HH:MM:SS` (`localtime_r` in C++).
fn timestamp() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

pub fn log(level: LogLevel, component: &str, message: &str) {
    if !enabled(level) {
        return;
    }
    static COLOR: OnceLock<bool> = OnceLock::new();
    let color = *COLOR
        .get_or_init(|| std::env::var_os("NO_COLOR").is_none() && std::io::stderr().is_terminal());
    let (tag, tint) = match level {
        LogLevel::Error => ("ERROR", "\x1b[31m"),
        LogLevel::Warn => ("WARN", "\x1b[33m"),
        LogLevel::Info => ("INFO", "\x1b[32m"),
        LogLevel::Debug => ("DEBUG", "\x1b[2;36m"),
    };
    let captured = CAPTURE.with(|capture| {
        capture.borrow_mut().as_mut().map(|lines| {
            let _ = writeln!(
                lines,
                "[{tag}] [{}] {}",
                safe_line(component, 64),
                safe_line(message, 8192)
            );
        })
    });
    if captured.is_some() {
        return;
    }
    if let Some(lines) = GLOBAL_CAPTURE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
    {
        let _ = writeln!(
            lines,
            "[{tag}] [{}] {}",
            safe_line(component, 64),
            safe_line(message, 8192)
        );
        return;
    }
    let mut line = timestamp();
    line.push(' ');
    if color {
        line.push_str(tint);
    }
    let _ = write!(line, "[{tag}]");
    if color {
        line.push_str("\x1b[0m");
    }
    let _ = writeln!(
        line,
        " [{}] {}",
        safe_line(component, 64),
        safe_line(message, 8192)
    );
    let _guard = log_mutex().lock();
    let _ = std::io::stderr().write_all(line.as_bytes());
}

pub fn debug(component: &str, message: &str) {
    log(LogLevel::Debug, component, message);
}

pub fn info(component: &str, message: &str) {
    log(LogLevel::Info, component, message);
}

pub fn warn(component: &str, message: &str) {
    log(LogLevel::Warn, component, message);
}

pub fn error(component: &str, message: &str) {
    log(LogLevel::Error, component, message);
}

fn read_memory_kib(path: &str, field: &str) -> usize {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix(field)
                    .and_then(|rest| rest.split_whitespace().next())
                    .and_then(|value| value.parse().ok())
            })
        })
        .unwrap_or(0)
}

/// Process RSS and host available memory; these are not additive GPU totals.
pub fn memory_status() -> String {
    format!(
        "rss_mib={} host_available_mib={}",
        read_memory_kib("/proc/self/status", "VmRSS:") / 1024,
        read_memory_kib("/proc/meminfo", "MemAvailable:") / 1024
    )
}

/// Logs completion after the response body, including streamed generation.
/// An unremarkable request logs at `success_level`, so polls quiet by
/// default appear only at debug; 4xx, 5xx and disconnects still escalate.
#[allow(clippy::too_many_arguments)]
pub fn log_request(
    id: &str,
    method: &str,
    path: &str,
    status_code: u16,
    duration_ms: f64,
    details: &str,
    outcome: &str,
    success_level: LogLevel,
) {
    let level = if status_code >= 500 || outcome == "stream_error" {
        LogLevel::Error
    } else if status_code >= 400 || outcome == "disconnected" {
        LogLevel::Warn
    } else {
        success_level
    };
    // A filtered line must not pay for the two /proc reads.
    if !enabled(level) {
        return;
    }
    let mut message = format!(
        "request={id} event=completed method={method} path={path} status={status_code} \
         duration_ms={duration_ms:.1} outcome={outcome}"
    );
    if !details.is_empty() {
        message.push(' ');
        message.push_str(details);
    }
    message.push(' ');
    message.push_str(&memory_status());
    log(level, "http", &message);
}

/// Runs `f` and returns what this thread logged meanwhile, as
/// `[LEVEL] [component] message` lines, instead of writing it to stderr.
pub fn capture<T>(f: impl FnOnce() -> T) -> (T, String) {
    let previous = CAPTURE.with(|capture| capture.borrow_mut().replace(String::new()));
    let value = f();
    let lines = CAPTURE
        .with(|capture| std::mem::replace(&mut *capture.borrow_mut(), previous))
        .unwrap_or_default();
    (value, lines)
}

/// [`capture`] with this thread's threshold at `level`; other threads keep
/// the process-wide one.
pub fn capture_at<T>(level: LogLevel, f: impl FnOnce() -> T) -> (T, String) {
    let previous = LEVEL_OVERRIDE.with(|cell| cell.replace(Some(level)));
    let captured = capture(f);
    LEVEL_OVERRIDE.with(|cell| cell.set(previous));
    captured
}

/// Runs `f` and returns what every thread logged meanwhile. Meant for a
/// test binary of its own, since concurrent tests' lines are captured too.
pub fn capture_all<T>(f: impl FnOnce() -> T) -> (T, String) {
    *GLOBAL_CAPTURE.lock().unwrap_or_else(|e| e.into_inner()) = Some(String::new());
    let value = f();
    let lines = GLOBAL_CAPTURE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_default();
    (value, lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_bytes_are_escaped_and_long_lines_truncated() {
        assert_eq!(safe_line("a\nb\x7f", 64), "a\\x0ab\\x7f");
        assert_eq!(safe_line("abcdef", 3), "abc...");
        assert_eq!(safe_line("äöü", 3), "ä...");
    }

    #[test]
    fn level_names_round_trip() {
        for (name, level) in [
            ("debug", LogLevel::Debug),
            ("info", LogLevel::Info),
            ("warn", LogLevel::Warn),
            ("error", LogLevel::Error),
        ] {
            assert_eq!(LogLevel::from_name(name), Some(level));
            assert_eq!(level.name(), name);
        }
        for name in ["trace", "DEBUG", "", "info,debug"] {
            assert_eq!(LogLevel::from_name(name), None, "{name}");
        }
    }

    #[test]
    fn threshold_filters_lines_and_requests() {
        let ((), quiet) = capture_at(LogLevel::Error, || {
            debug("probe", "hidden-debug");
            info("probe", "hidden-info");
            warn("probe", "hidden-warn");
            error("probe", "visible-error");
            let request = |id: &str, status: u16, success: LogLevel| {
                log_request(id, "GET", "/x", status, 1.0, "", "completed", success);
            };
            request("r-quiet", 200, LogLevel::Debug);
            request("r-404", 404, LogLevel::Info);
            request("r-500", 500, LogLevel::Info);
        });
        assert!(quiet.contains("visible-error"), "{quiet}");
        assert!(quiet.contains("request=r-500 event=completed"), "{quiet}");
        for hidden in ["hidden-", "r-quiet", "r-404", "[DEBUG]"] {
            assert!(!quiet.contains(hidden), "{hidden}: {quiet}");
        }
        let ((), chatty) = capture_at(LogLevel::Debug, || {
            debug("probe", "visible-debug");
            log_request(
                "r-poll",
                "GET",
                "/health",
                200,
                1.0,
                "",
                "completed",
                LogLevel::Debug,
            );
        });
        assert!(chatty.contains("[DEBUG] [probe] visible-debug"), "{chatty}");
        assert!(
            chatty.contains("[DEBUG] [http] request=r-poll event=completed"),
            "{chatty}"
        );
        assert_eq!(
            level(),
            LogLevel::Info,
            "the process threshold is untouched"
        );
    }

    #[test]
    fn capture_collects_this_threads_lines() {
        let ((), lines) = capture(|| warn("cache", "event=snapshot reason=capture_failure"));
        assert_eq!(
            lines,
            "[WARN] [cache] event=snapshot reason=capture_failure\n"
        );
    }
}
