//! Errors of the text-generation stack: model runners, the continuation
//! caches, the scheduler and the serving backends. The kinds mirror the C++
//! exception types (`std::invalid_argument`, `std::logic_error`,
//! `std::length_error`, `std::runtime_error`) plus the stable
//! `TextGenerationError` codes the HTTP layer maps to status codes.

use std::fmt;

/// `TextGenerationErrorCode`: failures with a stable wire code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationErrorCode {
    QueueFull,
    ClientQueueFull,
    DeadlineExceeded,
    OutputLimit,
    OutputBackpressure,
    SchedulerStopping,
    ToolChoiceUnsatisfied,
    /// The GPU context can no longer run work, for example after a reset;
    /// only a new process recovers.
    DeviceLost,
    /// The scheduler thread panicked: no work runs any more, and only a new
    /// process recovers (the C++ server terminates in this case).
    SchedulerFailed,
}

/// The client-facing message of [`GenerationErrorCode::DeviceLost`].
pub const DEVICE_LOST_MESSAGE: &str = "GPU context lost; restart required";

/// The client-facing message of [`GenerationErrorCode::SchedulerFailed`].
pub const SCHEDULER_FAILED_MESSAGE: &str = "text generation scheduler failed; restart required";

impl GenerationErrorCode {
    pub fn http_status(self) -> u16 {
        match self {
            Self::QueueFull | Self::ClientQueueFull => 429,
            Self::DeadlineExceeded => 408,
            Self::OutputLimit
            | Self::OutputBackpressure
            | Self::SchedulerStopping
            | Self::DeviceLost
            | Self::SchedulerFailed => 503,
            Self::ToolChoiceUnsatisfied => 502,
        }
    }

    pub fn stable_code(self) -> &'static str {
        match self {
            Self::QueueFull => "queue_full",
            Self::ClientQueueFull => "client_queue_full",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::OutputLimit => "output_limit",
            Self::OutputBackpressure => "output_backpressure",
            Self::SchedulerStopping => "scheduler_stopping",
            Self::ToolChoiceUnsatisfied => "tool_choice_unsatisfied",
            Self::DeviceLost => "device_lost",
            Self::SchedulerFailed => "scheduler_failed",
        }
    }

    pub fn retryable(self) -> bool {
        !matches!(
            self,
            Self::OutputLimit
                | Self::ToolChoiceUnsatisfied
                | Self::DeviceLost
                | Self::SchedulerFailed
        )
    }

    /// Whether only a new process recovers: the server then reports the
    /// failure on every generation and health route and shuts down.
    pub fn requires_restart(self) -> bool {
        matches!(self, Self::DeviceLost | Self::SchedulerFailed)
    }

    /// The client-facing message of a failure that requires a restart.
    pub fn restart_message(self) -> Option<&'static str> {
        match self {
            Self::DeviceLost => Some(DEVICE_LOST_MESSAGE),
            Self::SchedulerFailed => Some(SCHEDULER_FAILED_MESSAGE),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// `std::invalid_argument`: the caller passed something unusable.
    InvalidArgument,
    /// `std::logic_error`: an operation in the wrong state, or unsupported.
    Logic,
    /// `std::length_error`: a sequence exceeds a context or capacity.
    Length,
    /// `std::runtime_error`: the model or the system failed.
    Runtime,
    /// `TextGenerationError`.
    Generation(GenerationErrorCode),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidArgument, message)
    }

    pub fn logic(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Logic, message)
    }

    pub fn length(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Length, message)
    }

    pub fn runtime(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Runtime, message)
    }

    pub fn generation(code: GenerationErrorCode, message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Generation(code), message)
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The generation code, for errors that carry one.
    pub fn generation_code(&self) -> Option<GenerationErrorCode> {
        match self.kind {
            ErrorKind::Generation(code) => Some(code),
            _ => None,
        }
    }

    /// The message, as C++ `what()`.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T, E = Error> = std::result::Result<T, E>;
