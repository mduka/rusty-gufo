//! The Rust side of the disk-cache cross-check with the C++ store
//! (`tests/reference/cpp_disk_store_interop.cpp`, same runner, entries and queries):
//!
//!   disk_store_interop write <dir>   stores the three fixed entries
//!   disk_store_interop read <dir>    restores the four fixed queries

use std::path::PathBuf;
use std::process::ExitCode;

use gufo_core::sampling::SamplingConfig;
use gufo_core::text::{
    ChatRequest, NoPromptContext, Result, Sampler, TextDecodeSelection, TextExecutionPlan,
    TextModelRunner, TextPrefillStep, TextRunnerCapabilities, TextRunnerDescriptor,
    TextRunnerPersistenceDescriptor, TextRunnerResourceClaim, TextRunnerSnapshot, TextRunnerState,
    Token,
};
use gufo_serve::disk_store::{ContinuationDiskStore, ContinuationDiskStoreOptions};

struct Snapshot(Vec<u8>);

impl TextRunnerSnapshot for Snapshot {
    fn payload_bytes(&self) -> usize {
        self.0.len()
    }
}

#[derive(Default)]
struct State(Vec<u8>);

impl TextRunnerState for State {
    fn invalidate(&mut self) {
        self.0.clear();
    }
}

#[derive(Clone)]
struct NoSampler;

impl Sampler for NoSampler {
    fn accept(&mut self, _token: Token) {}
}

struct Runner;

impl TextModelRunner for Runner {
    type State = State;
    type Snapshot = Snapshot;
    type Sampler = NoSampler;
    type PromptContext = NoPromptContext;

    fn descriptor(&self) -> TextRunnerDescriptor {
        TextRunnerDescriptor {
            model_id: "interop".into(),
            state_abi: "interop-state".into(),
            max_context: 64,
            capabilities: TextRunnerCapabilities {
                snapshot: true,
                fork: true,
                ..TextRunnerCapabilities::default()
            },
            persistence: Some(TextRunnerPersistenceDescriptor {
                compatibility_identity: b"interop-v1".to_vec(),
                payload_version: 7,
            }),
        }
    }

    fn resource_claim(&self) -> TextRunnerResourceClaim {
        TextRunnerResourceClaim::default()
    }

    fn supported_plans(&self) -> Vec<TextExecutionPlan> {
        vec![TextExecutionPlan::SERIAL]
    }

    fn tokenize(&self, _text: &str) -> Result<Vec<Token>> {
        Ok(Vec::new())
    }

    fn render_and_tokenize(&self, _request: &ChatRequest) -> Result<Option<Vec<Token>>> {
        Ok(None)
    }

    fn decode(&self, _tokens: &[Token]) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }

    fn create_state(&self) -> Result<State> {
        Ok(State::default())
    }

    fn create_sampler(&self, _config: &SamplingConfig, _prompt: &[Token]) -> Result<NoSampler> {
        Ok(NoSampler)
    }

    fn prefill(&self, _: &mut State, _: &[Token], _: usize, _: usize) -> Result<TextPrefillStep> {
        Ok(TextPrefillStep::default())
    }

    fn select_next(&self, _: &mut State, _: &mut NoSampler) -> Result<TextDecodeSelection> {
        Ok(TextDecodeSelection::default())
    }

    fn advance(&self, _: &mut State, _: Token) -> Result<()> {
        Ok(())
    }

    fn checkpoint_position(&self, _: &State) -> Result<usize> {
        Ok(0)
    }

    fn persistent_snapshot_payload_bytes(&self, snapshot: &Snapshot) -> Result<usize> {
        Ok(snapshot.0.len())
    }

    fn serialize_persistent_snapshot(
        &self,
        snapshot: &Snapshot,
        destination: &mut [u8],
    ) -> Result<usize> {
        destination.copy_from_slice(&snapshot.0);
        Ok(snapshot.0.len())
    }

    fn restore_persistent_snapshot(&self, state: &mut State, payload: &[u8]) -> Result<()> {
        state.0 = payload.to_vec();
        Ok(())
    }
}

fn run(mode: &str, directory: PathBuf) -> Result<()> {
    let store = ContinuationDiskStore::new(
        ContinuationDiskStoreOptions {
            directory,
            capacity_bytes: 1 << 20,
            staging_capacity_bytes: 1 << 20,
            min_checkpoint_step_tokens: 0,
        },
        None,
        None,
    )?;
    let runner = Runner;
    match mode {
        "write" => {
            let entries: [(&[Token], &str, &[u8]); 3] = [
                (&[1, 2, 3], "alpha", &[]),
                (&[1, 2, 3, 4, 5], "bravo-bravo", &[]),
                (&[9, 8], "image", &[0xAA, 0xBB]),
            ];
            for (tokens, payload, identity) in entries {
                let saved = store.save(
                    &runner,
                    tokens,
                    &Snapshot(payload.as_bytes().to_vec()),
                    identity,
                )?;
                println!(
                    "stored={} file_bytes={} payload_bytes={}",
                    u8::from(saved.stored),
                    saved.file_bytes,
                    saved.payload_bytes
                );
            }
        }
        _ => {
            let queries: [(&[Token], &[u8]); 4] = [
                (&[1, 2, 3, 4, 5, 6], &[]),
                (&[1, 2, 3, 9], &[]),
                (&[9, 8, 7], &[0xAA, 0xBB]),
                (&[9, 8, 7], &[]),
            ];
            for (tokens, identity) in queries {
                let mut state = State::default();
                let restored = store.restore_longest_prefix(
                    &runner,
                    &mut state,
                    tokens,
                    identity.into(),
                    0,
                    0,
                )?;
                println!(
                    "restored={} tokens={} file_bytes={} payload={}",
                    u8::from(restored.restored),
                    restored.token_count,
                    restored.file_bytes,
                    String::from_utf8_lossy(&state.0)
                );
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(mode), Some(directory)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: disk_store_interop write|read <dir>");
        return ExitCode::from(2);
    };
    if mode != "write" && mode != "read" {
        eprintln!("unknown mode {mode}");
        return ExitCode::from(2);
    }
    match run(mode, PathBuf::from(directory)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
