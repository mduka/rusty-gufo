//! The model-independent text-generation interface: chat input, errors and
//! the runner trait that serving drives.

mod chat;
mod error;
mod runner;

pub use chat::{
    ChatMessage, ChatRequest, ChatRole, ChatTool, ImagePart, OutputGrammar, ReasoningEffort,
    ReasoningOptions, ToolArgument, ToolCall, ToolChoice,
};
pub use error::{
    DEVICE_LOST_MESSAGE, Error, ErrorKind, GenerationErrorCode, Result, SCHEDULER_FAILED_MESSAGE,
};
pub use runner::{
    CancellationCheck, ContinuationInputPrefix, DeviceStatus, HOST_SNAPSHOT_HEADROOM_BYTES,
    InitialOutputState, NoPromptContext, PromptContext, ReasoningTags, Sampler,
    TextDecodeSelection, TextDecodeStep, TextExecutionPlan, TextExecutionPlanKind, TextModelRunner,
    TextPrefillStep, TextPrefillThrough, TextPreparedPrompt, TextRunnerAdvance,
    TextRunnerCapabilities, TextRunnerDecode, TextRunnerDescriptor, TextRunnerMeasuredResources,
    TextRunnerPersistenceDescriptor, TextRunnerResourceClaim, TextRunnerSnapshot, TextRunnerState,
    Token, cgroup_available_bytes, default_advance_batch, default_decode_batch,
    default_decode_step, host_snapshot_budget_bytes, host_snapshot_ceiling_bytes,
    meminfo_available_bytes, prefix_input_identity,
};
