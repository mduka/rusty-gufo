//! Graph capture and replay with real kernels:
//! cargo test -p gufo-qwen3-asr-kernels --features gpu

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use gufo_hip::{
    Bf16, DeviceBuffer, DeviceCopy, DeviceTensor, Graph, GraphCache, GraphRun, Launch, LaunchError,
    MismatchPolicy, PinnedBuffer, Stream,
};
use gufo_qwen3_asr_kernels as k;

const BATCH: usize = 2;
const DIM: usize = 1024;
const N: usize = BATCH * DIM;

fn device<T: DeviceCopy>(stream: &Stream, host: &[T]) -> DeviceBuffer<T> {
    let mut buffer = stream.alloc::<T>(host.len()).unwrap();
    stream.upload(host, buffer.as_mut_slice()).unwrap();
    buffer
}

fn host<T: DeviceCopy + Default>(stream: &Stream, buffer: &DeviceBuffer<T>) -> Vec<T> {
    let mut values = vec![T::default(); buffer.len()];
    stream.download(buffer.as_slice(), &mut values).unwrap();
    values
}

fn input(step: usize) -> Vec<f32> {
    (0..N)
        .map(|i| Bf16::from_f32(((i * 7 + step * 13) % 97) as f32 / 48.0 - 1.0).to_f32())
        .collect()
}

/// rms_norm → bf16_to_f32 → swiglu, a small slice of a decoder layer.
struct Chain {
    input: DeviceBuffer<f32>,
    weight: DeviceTensor<f32>,
    normed: DeviceBuffer<Bf16>,
    widened: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    output: DeviceBuffer<Bf16>,
}

impl Chain {
    fn new(stream: &Stream) -> Self {
        let weight: Vec<f32> = (0..DIM)
            .map(|i| Bf16::from_f32(1.0 + (i % 5) as f32 / 8.0).to_f32())
            .collect();
        let up: Vec<f32> = (0..N).map(|i| (i % 11) as f32 / 11.0).collect();
        Self {
            input: device(stream, &input(0)),
            weight: device(stream, &weight).freeze(stream).unwrap(),
            normed: stream.alloc(N).unwrap(),
            widened: stream.alloc(N).unwrap(),
            up: device(stream, &up),
            output: stream.alloc(N).unwrap(),
        }
    }

    fn step(&mut self, s: &Stream) -> Result<(), LaunchError> {
        k::rms_norm(
            s,
            self.input.as_slice(),
            self.weight.view(),
            self.normed.as_mut_slice(),
            BATCH,
            DIM,
            1e-6,
        )?;
        k::bf16_to_f32(s, self.normed.as_slice(), self.widened.as_mut_slice(), N)?;
        k::swiglu(
            s,
            self.widened.as_slice(),
            self.up.as_slice(),
            self.output.as_mut_slice(),
            N,
        )
    }

    fn eager(stream: &Stream, step: usize) -> Vec<Bf16> {
        let mut chain = Self::new(stream);
        stream
            .upload(&input(step), chain.input.as_mut_slice())
            .unwrap();
        chain.step(stream).unwrap();
        host(stream, &chain.output)
    }
}

#[test]
fn graph_replays_match_eager_results_step_by_step() {
    let stream = Stream::new().unwrap();
    let mut chain = Chain::new(&stream);
    let mut graphs = GraphCache::new().with_validation(true);
    let mut runs = Vec::new();
    for step in 0..5 {
        stream
            .upload(&input(step), chain.input.as_mut_slice())
            .unwrap();
        runs.push(graphs.run(&stream, BATCH, |s| chain.step(s)).unwrap());
        assert_eq!(
            host(&stream, &chain.output),
            Chain::eager(&stream, step),
            "step {step}"
        );
    }
    use GraphRun::*;
    assert_eq!(runs, [Eager, Captured, Replayed, Replayed, Replayed]);
}

#[test]
fn a_replaced_buffer_makes_the_graph_stale_and_it_is_recaptured() {
    let stream = Stream::new().unwrap();
    let mut chain = Chain::new(&stream);
    let mut graphs = GraphCache::new();
    graphs.run(&stream, BATCH, |s| chain.step(s)).unwrap();
    graphs.run(&stream, BATCH, |s| chain.step(s)).unwrap();
    let replaced = Arc::downgrade(chain.output.allocation());
    assert!(graphs.retained_bytes() >= N * size_of::<Bf16>());

    chain.output = stream.alloc(N).unwrap();
    assert!(
        replaced.upgrade().is_some(),
        "the graph still holds the old buffer"
    );
    stream
        .upload(&input(3), chain.input.as_mut_slice())
        .unwrap();
    assert_eq!(
        graphs.run(&stream, BATCH, |s| chain.step(s)).unwrap(),
        GraphRun::Recaptured
    );
    assert!(
        replaced.upgrade().is_none(),
        "the stale graph and its memory are gone"
    );
    assert_eq!(host(&stream, &chain.output), Chain::eager(&stream, 3));
}

#[test]
fn a_per_step_scalar_is_reported_by_validation() {
    let stream = Stream::new().unwrap();
    let mut chain = Chain::new(&stream);
    let mut body = |s: &Stream, count: usize| {
        k::swiglu(
            s,
            chain.widened.as_slice(),
            chain.up.as_slice(),
            chain.output.as_mut_slice(),
            count,
        )
    };

    let mut validated = GraphCache::new()
        .with_validation(true)
        .with_mismatch_policy(MismatchPolicy::Error);
    validated.run(&stream, 0, |s| body(s, N)).unwrap();
    validated.run(&stream, 0, |s| body(s, N - 1)).unwrap();
    let error = validated.run(&stream, 0, |s| body(s, N - 2)).unwrap_err();
    assert_eq!(
        error,
        LaunchError::GraphMismatch {
            launch: 0,
            launcher: "LaunchTextSwiGLU",
            argument: "count"
        }
    );
    assert_eq!(validated.mismatches(), 1);
    // The mismatch was found after the replay; the key never replays again.
    assert_eq!(
        validated.run(&stream, 0, |s| body(s, N - 2)).unwrap(),
        GraphRun::Eager
    );

    // Without validation the graph silently replays the frozen count.
    let mut unchecked = GraphCache::new().with_validation(false);
    unchecked.run(&stream, 0, |s| body(s, N)).unwrap();
    unchecked.run(&stream, 0, |s| body(s, N - 1)).unwrap();
    assert_eq!(
        unchecked.run(&stream, 0, |s| body(s, N - 2)).unwrap(),
        GraphRun::Replayed
    );
}

/// The review's case: two live output buffers used alternately under one key.
/// Ownership checks cannot tell them apart; validation can.
#[test]
fn alternating_live_buffers_are_detected_in_release_policy() {
    let stream = Stream::new().unwrap();
    let mut chain = Chain::new(&stream);
    let mut outputs = [
        stream.alloc::<Bf16>(N).unwrap(),
        stream.alloc::<Bf16>(N).unwrap(),
    ];
    let mut run_step = |graphs: &mut GraphCache<usize>, step: usize, chain: &mut Chain| {
        stream
            .upload(&input(step), chain.input.as_mut_slice())
            .unwrap();
        let output = &mut outputs[step % 2];
        let run = graphs
            .run(&stream, BATCH, |s| {
                k::rms_norm(
                    s,
                    chain.input.as_slice(),
                    chain.weight.view(),
                    chain.normed.as_mut_slice(),
                    BATCH,
                    DIM,
                    1e-6,
                )?;
                k::bf16_to_f32(s, chain.normed.as_slice(), chain.widened.as_mut_slice(), N)?;
                k::swiglu(
                    s,
                    chain.widened.as_slice(),
                    chain.up.as_slice(),
                    output.as_mut_slice(),
                    N,
                )
            })
            .unwrap();
        (run, host(&stream, output))
    };

    let mut graphs = GraphCache::new()
        .with_validation(true)
        .with_mismatch_policy(MismatchPolicy::Recapture);
    let mut runs = Vec::new();
    for step in 0..5 {
        let (run, output) = run_step(&mut graphs, step, &mut chain);
        assert_eq!(
            output,
            Chain::eager(&stream, step),
            "step {step} writes the right buffer"
        );
        runs.push(run);
    }
    use GraphRun::*;
    // Step 2 mismatches and recaptures; step 3 mismatches again, so the key
    // runs eagerly from then on.
    assert_eq!(runs, [Eager, Captured, Recaptured, Eager, Eager]);
    assert_eq!(graphs.mismatches(), 2);
    assert!(matches!(
        graphs.last_mismatch(),
        Some(LaunchError::GraphMismatch {
            launcher: "LaunchTextSwiGLU",
            argument: "output",
            ..
        })
    ));

    // Without validation the third step silently replays into the other buffer.
    let mut unchecked = GraphCache::new().with_validation(false);
    let mut chain = Chain::new(&stream);
    run_step(&mut unchecked, 0, &mut chain);
    run_step(&mut unchecked, 1, &mut chain);
    let (run, output) = run_step(&mut unchecked, 2, &mut chain);
    assert_eq!(run, Replayed);
    assert_ne!(
        output,
        Chain::eager(&stream, 2),
        "the frozen graph wrote the wrong buffer"
    );
}

#[test]
fn allocating_in_the_body_falls_back_to_eager_execution() {
    let stream = Stream::new().unwrap();
    let mut chain = Chain::new(&stream);
    let mut graphs = GraphCache::new();
    let mut runs = Vec::new();
    for step in 0..3 {
        stream
            .upload(&input(step), chain.input.as_mut_slice())
            .unwrap();
        let run = graphs.run(&stream, BATCH, |s| {
            let mut scratch = s.alloc::<f32>(N)?;
            k::rms_norm(
                s,
                chain.input.as_slice(),
                chain.weight.view(),
                chain.normed.as_mut_slice(),
                BATCH,
                DIM,
                1e-6,
            )?;
            k::bf16_to_f32(s, chain.normed.as_slice(), scratch.as_mut_slice(), N)?;
            k::swiglu(
                s,
                scratch.as_slice(),
                chain.up.as_slice(),
                chain.output.as_mut_slice(),
                N,
            )
        });
        runs.push(run.unwrap());
        assert_eq!(
            host(&stream, &chain.output),
            Chain::eager(&stream, step),
            "step {step}"
        );
    }
    assert_eq!(runs, [GraphRun::Eager; 3]);
}

#[test]
fn unsupported_launch_results_are_replayed_by_validation() {
    let stream = Stream::new().unwrap();
    let (heads, head_dim) = (4u32, 63u32); // odd head_dim: the launcher declines
    let len = (heads * head_dim) as usize;
    let mut query = device(&stream, &vec![1.5f32; len]);
    let mut key = device(&stream, &vec![0.5f32; len]);
    let mut value = device(&stream, &vec![0.25f32; len]);
    let norm = device(&stream, &vec![1.0f32; head_dim as usize]);
    let mut graphs = GraphCache::new().with_validation(true);
    let mut runs = Vec::new();
    for _ in 0..3 {
        let run = graphs.run(&stream, 0, |s| {
            let launched = k::qk_norm_rope(
                s,
                query.as_mut_slice(),
                key.as_mut_slice(),
                value.as_mut_slice(),
                norm.as_slice(),
                norm.as_slice(),
                1,
                heads,
                heads,
                head_dim,
                0,
                1e6,
                1e-6,
                None,
                None,
                0,
                16,
            )?;
            if launched == Launch::Unsupported {
                k::round_bf16(s, query.as_mut_slice(), len)?;
            }
            Ok(())
        });
        runs.push(run.unwrap());
    }
    assert_eq!(
        runs,
        [GraphRun::Eager, GraphRun::Captured, GraphRun::Replayed]
    );
}

#[test]
fn graphs_hold_captured_memory_and_refuse_other_streams() {
    let (stream, other) = (Stream::new().unwrap(), Stream::new().unwrap());
    let gate = device(&stream, &vec![1.0f32; N]);
    let mut output = stream.alloc::<Bf16>(N).unwrap();
    let graph = Graph::capture(&stream, |s| {
        k::swiglu(
            s,
            gate.as_slice(),
            gate.as_slice(),
            output.as_mut_slice(),
            N,
        )
    })
    .unwrap();
    assert_eq!(graph.launch_count(), 1);
    assert_eq!(graph.retained().len(), 2, "gate is retained once");

    let error = graph.launch(&other).unwrap_err();
    assert!(error.to_string().contains("another stream"), "{error}");

    let held = Arc::downgrade(output.allocation());
    drop(output);
    assert!(graph.is_stale());
    assert!(held.upgrade().is_some());
    drop(graph);
    assert!(held.upgrade().is_none());
}

/// Regression: freeing shared tensors or pinned memory synchronises the whole
/// device, which invalidated captures on other threads (HIP error 901, seen as
/// intermittent failures of parallel tests). Such frees now wait until no
/// capture runs.
#[test]
fn captures_survive_frees_on_other_threads() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let stop = Arc::new(AtomicBool::new(false));
    let churn = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let stream = Stream::new().unwrap();
            let mut frees = 0usize;
            while !stop.load(Ordering::Relaxed) {
                let tensor = stream.alloc::<f32>(4096).unwrap().freeze(&stream).unwrap();
                drop(tensor);
                drop(PinnedBuffer::<f32>::zeroed(4096).unwrap());
                frees += 2;
            }
            frees
        })
    };

    let stream = Stream::new().unwrap();
    let mut chain = Chain::new(&stream);
    for round in 0..200 {
        let mut graphs = GraphCache::new().with_validation(true);
        for step in 0..3 {
            stream
                .upload(&input(step), chain.input.as_mut_slice())
                .unwrap();
            let run = graphs.run(&stream, BATCH, |s| chain.step(s)).unwrap();
            let expected = [GraphRun::Eager, GraphRun::Captured, GraphRun::Replayed][step];
            assert_eq!(run, expected, "round {round} step {step}");
        }
        assert_eq!(
            host(&stream, &chain.output),
            Chain::eager(&stream, 2),
            "round {round}"
        );
    }
    stop.store(true, Ordering::Relaxed);
    assert!(churn.join().unwrap() > 0);
    gufo_hip::collect_deferred_frees();
}

#[test]
fn device_copies_are_captured_validated_and_retained() {
    let stream = Stream::new().unwrap();
    let mut chain = Chain::new(&stream);
    let mut copy: DeviceBuffer<Bf16> = stream.alloc(N).unwrap();
    let mut graphs = GraphCache::new()
        .with_validation(true)
        .with_mismatch_policy(MismatchPolicy::Error);
    let mut runs = Vec::new();
    for step in 0..4 {
        stream
            .upload(&input(step), chain.input.as_mut_slice())
            .unwrap();
        let run = graphs.run(&stream, BATCH, |s| {
            chain.step(s)?;
            s.graph_copy(chain.output.as_slice(), copy.as_mut_slice())
        });
        runs.push(run.unwrap());
        assert_eq!(
            host(&stream, &copy),
            Chain::eager(&stream, step),
            "step {step}"
        );
    }
    use GraphRun::*;
    assert_eq!(runs, [Eager, Captured, Replayed, Replayed]);
    assert!(graphs.retained_bytes() >= 2 * N * size_of::<Bf16>());
    // Another destination is another body: a mismatch, never a silent replay.
    let mut other: DeviceBuffer<Bf16> = stream.alloc(N).unwrap();
    let error = graphs
        .run(&stream, BATCH, |s| {
            chain.step(s)?;
            s.graph_copy(chain.output.as_slice(), other.as_mut_slice())
        })
        .err()
        .unwrap();
    assert!(
        matches!(error, LaunchError::GraphMismatch { .. }),
        "{error}"
    );
}
