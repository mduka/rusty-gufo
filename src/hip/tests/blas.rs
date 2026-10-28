//! hipBLAS / hipBLASLt through the shim: cargo test -p gufo-hip --features gpu

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::blas::{
    AlgoPolicy, Blas, BlasLt, FirstUsable, Heuristic, Layout, MatmulShape, Op, Strides,
};
use gufo_hip::{
    Bf16, DeviceBuffer, DeviceCopy, ErrorKind, Graph, GraphCache, GraphRun, HipError, Launch,
    LaunchError, MismatchPolicy, Stream,
};

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

fn bf16_values(len: usize, seed: u32) -> Vec<Bf16> {
    (0..len as u32)
        .map(|i| {
            Bf16::from_f32(
                ((i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 20) as f32 / 2048.0) - 1.0,
            )
        })
        .collect()
}

/// The Qwen3-ASR projection: `Y[batch, m] = X[batch, k] · A[m, k]^T`, i.e.
/// column-major `C (m × batch) = A^T (A stored k × m) · X (stored k × batch)`.
struct Projection {
    batch: usize,
    m: usize,
    k: usize,
    a: Vec<Bf16>,
    x: Vec<Bf16>,
    weights: DeviceBuffer<Bf16>,
    input: DeviceBuffer<Bf16>,
}

impl Projection {
    fn new(stream: &Stream, batch: usize, m: usize, k: usize) -> Self {
        let (a, x) = (bf16_values(m * k, 1), bf16_values(batch * k, 7));
        let (weights, input) = (device(stream, &a), device(stream, &x));
        Self {
            batch,
            m,
            k,
            a,
            x,
            weights,
            input,
        }
    }

    /// Replaces the input on the device and in the reference.
    fn set_input(&mut self, stream: &Stream, seed: u32) {
        self.x = bf16_values(self.batch * self.k, seed);
        stream.upload(&self.x, self.input.as_mut_slice()).unwrap();
    }

    fn shape(&self) -> MatmulShape {
        MatmulShape {
            trans_a: Some(Op::T),
            trans_b: None,
            a: Layout::new(self.k, self.m, self.k),
            b: Layout::new(self.k, self.batch, self.k),
            c: Layout::new(self.m, self.batch, self.m),
            batch: None,
        }
    }

    /// f64 reference of one output element and the sum of |terms|.
    fn expected(&self, row: usize, col: usize) -> (f64, f64) {
        (0..self.k).fold((0.0, 0.0), |(sum, scale), i| {
            let term = f64::from(self.x[row * self.k + i].to_f32())
                * f64::from(self.a[col * self.k + i].to_f32());
            (sum + term, scale + term.abs())
        })
    }

    /// Checks every element, or 256 spread over the output for large shapes.
    fn check(&self, output: &[f32]) {
        let total = self.batch * self.m;
        let step = (total / 256).max(1);
        for index in (0..total).step_by(step) {
            let (want, scale) = self.expected(index / self.m, index % self.m);
            let got = output[index];
            assert!(
                (f64::from(got) - want).abs() <= 1e-5 * scale + 1e-6,
                "element {index}: {got} against {want}"
            );
        }
    }
}

#[test]
fn gemm_and_matmul_match_the_reference() {
    let stream = Stream::new().unwrap();
    let p = Projection::new(&stream, 7, 96, 256);
    let blas = Blas::new(&stream).unwrap();
    let mut output = stream.alloc::<f32>(p.batch * p.m).unwrap();
    blas.gemm(
        &stream,
        Op::T,
        Op::N,
        p.m,
        p.batch,
        p.k,
        p.weights.as_slice(),
        p.k,
        p.input.as_slice(),
        p.k,
        output.as_mut_slice(),
        p.m,
    )
    .unwrap();
    p.check(&host(&stream, &output));

    let mut lt = BlasLt::new(&stream, 0).unwrap();
    let mut lt_output = stream.alloc::<f32>(p.batch * p.m).unwrap();
    let launched = lt
        .matmul(
            &stream,
            &p.shape(),
            p.weights.as_slice(),
            p.input.as_slice(),
            lt_output.as_mut_slice(),
            &FirstUsable,
        )
        .unwrap();
    assert_eq!(launched, Launch::Launched);
    p.check(&host(&stream, &lt_output));
    assert!(lt.installed_index::<Bf16, Bf16, f32>(&p.shape()).is_some());
}

fn f32_values(len: usize, seed: u32) -> Vec<f32> {
    bf16_values(len, seed).iter().map(|v| v.to_f32()).collect()
}

/// `alpha · A^T · X` of a stored `k × m` A and `k × n` X, column-major.
fn scaled_product(a: &[f32], x: &[f32], m: usize, n: usize, k: usize, alpha: f32) -> Vec<f64> {
    (0..n * m)
        .map(|index| {
            let (col, row) = (index / m, index % m);
            let dot: f64 = (0..k)
                .map(|i| f64::from(a[row * k + i]) * f64::from(x[col * k + i]))
                .sum();
            f64::from(alpha) * dot
        })
        .collect()
}

fn assert_close(got: &[f32], want: &[f64]) {
    for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
        assert!(
            (f64::from(got) - want).abs() <= 1e-4 * want.abs().max(1.0),
            "element {index}: {got} against {want}"
        );
    }
}

#[test]
fn sgemm_and_strided_sgemm_scale_the_product() {
    let stream = Stream::new().unwrap();
    let blas = Blas::with_default_atomics(&stream).unwrap();
    let (m, n, k, alpha) = (24, 5, 64, 0.125f32);
    let (a, x) = (f32_values(m * k, 3), f32_values(n * k, 11));
    let (a_dev, x_dev) = (device(&stream, &a), device(&stream, &x));
    let mut out = stream.alloc::<f32>(m * n).unwrap();
    blas.sgemm(
        &stream,
        Op::T,
        Op::N,
        m,
        n,
        k,
        alpha,
        a_dev.as_slice(),
        k,
        x_dev.as_slice(),
        k,
        out.as_mut_slice(),
        m,
    )
    .unwrap();
    assert_close(
        &host(&stream, &out),
        &scaled_product(&a, &x, m, n, k, alpha),
    );

    // Three problems share A (stride 0), each with its own X and C.
    let batches = 3;
    let xs = f32_values(batches * n * k, 17);
    let xs_dev = device(&stream, &xs);
    let mut outs = stream.alloc::<f32>(batches * m * n).unwrap();
    blas.sgemm_strided_batched(
        &stream,
        Op::T,
        Op::N,
        m,
        n,
        k,
        alpha,
        a_dev.as_slice(),
        k,
        xs_dev.as_slice(),
        k,
        outs.as_mut_slice(),
        m,
        Strides {
            count: batches,
            a: 0,
            b: n * k,
            c: m * n,
        },
    )
    .unwrap();
    let got = host(&stream, &outs);
    for batch in 0..batches {
        let want = scaled_product(&a, &xs[batch * n * k..][..n * k], m, n, k, alpha);
        assert_close(&got[batch * m * n..][..m * n], &want);
    }
}

#[test]
fn interleaved_strided_outputs_are_disjoint() {
    // Three problems write rows `i * m .. (i + 1) * m` of every column of a
    // `3 * m`-row output, as the attention output's grouped projection does.
    let stream = Stream::new().unwrap();
    let blas = Blas::with_default_atomics(&stream).unwrap();
    let (m, n, k, count) = (8, 4, 32, 3);
    let (a, x) = (f32_values(count * m * k, 5), f32_values(count * n * k, 9));
    let (a_dev, x_dev) = (device(&stream, &a), device(&stream, &x));
    let ldc = count * m;
    let mut out = stream.alloc::<f32>(n * ldc).unwrap();
    let strides = Strides {
        count,
        a: m * k,
        b: n * k,
        c: m,
    };
    blas.sgemm_strided_batched(
        &stream,
        Op::T,
        Op::N,
        m,
        n,
        k,
        1.0,
        a_dev.as_slice(),
        k,
        x_dev.as_slice(),
        k,
        out.as_mut_slice(),
        ldc,
        strides,
    )
    .unwrap();
    let got = host(&stream, &out);
    for problem in 0..count {
        let want = scaled_product(
            &a[problem * m * k..][..m * k],
            &x[problem * n * k..][..n * k],
            m,
            n,
            k,
            1.0,
        );
        for col in 0..n {
            assert_close(&got[col * ldc + problem * m..][..m], &want[col * m..][..m]);
        }
    }
    // A stride below the rows overlaps, and so does a column too short for
    // every problem's rows.
    for (c, ldc) in [(m - 1, count * m), (m, count * m - 1)] {
        let mut out = stream.alloc::<f32>(n * count * m).unwrap();
        let refused = blas.sgemm_strided_batched(
            &stream,
            Op::T,
            Op::N,
            m,
            n,
            k,
            1.0,
            a_dev.as_slice(),
            k,
            x_dev.as_slice(),
            k,
            out.as_mut_slice(),
            ldc,
            Strides { c, ..strides },
        );
        assert!(refused.is_err(), "c stride {c}, ldc {ldc}");
    }
}

/// Launches candidate `first`, then the rest in ranked order.
struct TrialFrom(usize);

impl AlgoPolicy for TrialFrom {
    fn choose(&self, _: &[Heuristic], _: usize) -> Option<usize> {
        None
    }

    fn resolve_names(&self) -> bool {
        false
    }

    fn trial_order(&self, ranked: &[Heuristic], _: usize) -> Option<Vec<usize>> {
        Some(std::iter::once(self.0).chain(0..ranked.len()).collect())
    }
}

#[test]
fn a_trial_policy_installs_the_first_candidate_that_launches() {
    let stream = Stream::new().unwrap();
    let p = Projection::new(&stream, 6, 128, 256);
    let mut lt = BlasLt::new(&stream, 0).unwrap();
    assert!(
        lt.prepare::<Bf16, Bf16, f32>(&stream, &p.shape(), &TrialFrom(0))
            .is_err()
    );
    assert_eq!(lt.plans(), 0);
    let mut output = stream.alloc::<f32>(p.batch * p.m).unwrap();
    for _ in 0..2 {
        let launched = lt
            .matmul(
                &stream,
                &p.shape(),
                p.weights.as_slice(),
                p.input.as_slice(),
                output.as_mut_slice(),
                &TrialFrom(1),
            )
            .unwrap();
        assert_eq!(launched, Launch::Launched);
        p.check(&host(&stream, &output));
    }
    assert_eq!(lt.plans(), 1);
    assert!(lt.installed_index::<Bf16, Bf16, f32>(&p.shape()).is_some());
}

/// Times every usable candidate and keeps the fastest.
struct Fastest;

impl AlgoPolicy for Fastest {
    fn requested(&self) -> usize {
        32
    }

    fn choose(&self, _: &[Heuristic], _: usize) -> Option<usize> {
        None
    }

    fn resolve_names(&self) -> bool {
        false
    }

    fn timed_runs(&self) -> Option<usize> {
        Some(3)
    }
}

#[test]
fn a_timed_policy_installs_a_usable_candidate_and_returns_its_result() {
    let stream = Stream::new().unwrap();
    let p = Projection::new(&stream, 32, 256, 512);
    let mut lt = BlasLt::new(&stream, 0).unwrap();
    assert!(
        lt.prepare::<Bf16, Bf16, f32>(&stream, &p.shape(), &Fastest)
            .is_err(),
        "a timed policy plans on its first matmul"
    );
    assert_eq!(lt.plans(), 0);
    let mut output = stream.alloc::<f32>(p.batch * p.m).unwrap();
    for _ in 0..2 {
        stream.fill_bytes(output.as_mut_slice(), 0xff).unwrap();
        let launched = lt
            .matmul(
                &stream,
                &p.shape(),
                p.weights.as_slice(),
                p.input.as_slice(),
                output.as_mut_slice(),
                &Fastest,
            )
            .unwrap();
        assert_eq!(launched, Launch::Launched);
        p.check(&host(&stream, &output));
    }
    assert_eq!(lt.plans(), 1);
    assert!(lt.installed_index::<Bf16, Bf16, f32>(&p.shape()).is_some());
}

struct Decline;

impl AlgoPolicy for Decline {
    fn choose(&self, _: &[Heuristic], _: usize) -> Option<usize> {
        None
    }
}

#[test]
fn a_declined_shape_is_cached_and_reported_unsupported() {
    let stream = Stream::new().unwrap();
    let p = Projection::new(&stream, 2, 32, 64);
    let mut lt = BlasLt::new(&stream, 0).unwrap();
    let mut output = stream.alloc::<f32>(p.batch * p.m).unwrap();
    for _ in 0..2 {
        let launched = lt
            .matmul(
                &stream,
                &p.shape(),
                p.weights.as_slice(),
                p.input.as_slice(),
                output.as_mut_slice(),
                &Decline,
            )
            .unwrap();
        assert_eq!(launched, Launch::Unsupported);
    }
    assert_eq!(lt.plans(), 1);
}

#[test]
fn extents_layouts_and_streams_are_checked() {
    let stream = Stream::new().unwrap();
    let p = Projection::new(&stream, 3, 32, 64);
    let blas = Blas::new(&stream).unwrap();
    let mut short = stream.alloc::<f32>(p.batch * p.m - 1).unwrap();
    let error = blas
        .gemm(
            &stream,
            Op::T,
            Op::N,
            p.m,
            p.batch,
            p.k,
            p.weights.as_slice(),
            p.k,
            p.input.as_slice(),
            p.k,
            short.as_mut_slice(),
            p.m,
        )
        .unwrap_err();
    assert!(
        matches!(error, LaunchError::ExtentTooShort { param: "c", .. }),
        "{error}"
    );
    let mut output = stream.alloc::<f32>(p.batch * p.m).unwrap();
    let error = blas
        .gemm(
            &stream,
            Op::T,
            Op::N,
            p.m,
            p.batch,
            p.k,
            p.weights.as_slice(),
            p.k - 1,
            p.input.as_slice(),
            p.k,
            output.as_mut_slice(),
            p.m,
        )
        .unwrap_err();
    assert!(
        matches!(error, LaunchError::InvalidArgument { param: "a", .. }),
        "{error}"
    );
    let other = Stream::new().unwrap();
    let error = blas
        .gemm(
            &other,
            Op::T,
            Op::N,
            p.m,
            p.batch,
            p.k,
            p.weights.as_slice(),
            p.k,
            p.input.as_slice(),
            p.k,
            output.as_mut_slice(),
            p.m,
        )
        .unwrap_err();
    assert!(
        matches!(
            error,
            LaunchError::Hip(HipError {
                kind: ErrorKind::Refused,
                ..
            })
        ),
        "{error}"
    );
}

#[test]
fn blas_calls_replay_in_graphs() {
    let stream = Stream::new().unwrap();
    let mut p = Projection::new(&stream, 5, 64, 128);
    let blas = Blas::new(&stream).unwrap();
    let mut lt = BlasLt::new(&stream, 0).unwrap();
    let mut via_gemm = stream.alloc::<f32>(p.batch * p.m).unwrap();
    let mut via_lt = stream.alloc::<f32>(p.batch * p.m).unwrap();
    let shape = p.shape();
    let mut graphs = GraphCache::new().with_validation(true);
    let mut runs = Vec::new();
    for step in 0..4 {
        p.set_input(&stream, 100 + step);
        let (weights, input, m, n, k) = (&p.weights, &p.input, p.m, p.batch, p.k);
        let run = graphs
            .run(&stream, 0, |s| {
                blas.gemm(
                    s,
                    Op::T,
                    Op::N,
                    m,
                    n,
                    k,
                    weights.as_slice(),
                    k,
                    input.as_slice(),
                    k,
                    via_gemm.as_mut_slice(),
                    m,
                )?;
                let launched = lt.matmul(
                    s,
                    &shape,
                    weights.as_slice(),
                    input.as_slice(),
                    via_lt.as_mut_slice(),
                    &FirstUsable,
                )?;
                assert_eq!(launched, Launch::Launched);
                Ok(())
            })
            .unwrap();
        runs.push(run);
        p.check(&host(&stream, &via_gemm));
        p.check(&host(&stream, &via_lt));
    }
    use GraphRun::*;
    assert_eq!(runs, [Eager, Captured, Replayed, Replayed]);
}

/// The pre-launch work runs before every launch; a replay validates after it
/// (`MismatchPolicy::Error`, the default) or before it (`Recapture`).
#[test]
fn pre_launch_work_precedes_the_graph() {
    for (policy, replay) in [
        (MismatchPolicy::Error, ["before", "body"]),
        (MismatchPolicy::Recapture, ["body", "before"]),
    ] {
        pre_launch_order(policy, replay);
    }
}

fn pre_launch_order(policy: MismatchPolicy, replay: [&str; 2]) {
    let stream = Stream::new().unwrap();
    let blas = Blas::new(&stream).unwrap();
    let mut out = stream.alloc::<f32>(5 * 64).unwrap();
    let mut graphs = GraphCache::new()
        .with_validation(true)
        .with_mismatch_policy(policy);
    let mut context = (Projection::new(&stream, 5, 64, 128), Vec::new());
    let mut runs = Vec::new();
    for step in 0..4 {
        let run = graphs
            .run_with(
                &stream,
                0,
                &mut context,
                |(p, log), s| {
                    log.push("body");
                    blas.gemm(
                        s,
                        Op::T,
                        Op::N,
                        p.m,
                        p.batch,
                        p.k,
                        p.weights.as_slice(),
                        p.k,
                        p.input.as_slice(),
                        p.k,
                        out.as_mut_slice(),
                        p.m,
                    )
                },
                |(p, log)| {
                    // Input the graph reads, uploaded before its launch.
                    log.push("before");
                    p.set_input(&stream, 100 + step);
                    Ok(())
                },
            )
            .unwrap();
        runs.push(run);
        context.0.check(&host(&stream, &out));
    }
    use GraphRun::*;
    assert_eq!(runs, [Eager, Captured, Replayed, Replayed]);
    // A replay's body is its validation (record-only).
    let order: Vec<_> = context.1.chunks(2).map(|pair| pair.to_vec()).collect();
    assert_eq!(
        order,
        [
            vec!["before", "body"],
            vec!["before", "body"],
            replay.to_vec(),
            replay.to_vec()
        ],
        "{policy:?}"
    );
}

/// An overlapped capture enqueues the pass, restores the dispatch state the
/// pass started from and records the same launches without launching them;
/// later runs replay the graph.
#[test]
fn an_overlapped_capture_records_after_the_pass() {
    let stream = Stream::new().unwrap();
    let blas = Blas::new(&stream).unwrap();
    let mut out = stream.alloc::<f32>(5 * 64).unwrap();
    let mut graphs = GraphCache::new().with_validation(true);
    // The projection, the log, and a dispatch counter the body advances.
    let mut context = (Projection::new(&stream, 5, 64, 128), Vec::new(), 0u32);
    let mut runs = Vec::new();
    for step in 0..4 {
        let run = graphs
            .run_overlapped(
                &stream,
                0,
                &mut context,
                |(p, log, dispatch), s| {
                    log.push(format!("body {dispatch}"));
                    *dispatch += 1;
                    blas.gemm(
                        s,
                        Op::T,
                        Op::N,
                        p.m,
                        p.batch,
                        p.k,
                        p.weights.as_slice(),
                        p.k,
                        p.input.as_slice(),
                        p.k,
                        out.as_mut_slice(),
                        p.m,
                    )
                },
                |(p, log, _)| {
                    log.push("before".to_string());
                    p.set_input(&stream, 100 + step);
                    Ok(())
                },
                |(_, _, dispatch)| *dispatch,
                |(_, log, dispatch), saved| {
                    log.push("restore".to_string());
                    *dispatch = saved;
                },
            )
            .unwrap();
        runs.push(run);
        context.0.check(&host(&stream, &out));
        context.2 = 0;
    }
    use GraphRun::*;
    assert_eq!(runs, [Eager, Captured, Replayed, Replayed]);
    assert_eq!(
        context.1,
        [
            "before", "body 0", // eager first run
            "before", "body 0", "restore", "body 0", // pass, then its recording
            "before", "body 0", // replay, validated after its launch
            "before", "body 0",
        ]
    );
}

#[test]
fn planning_is_refused_while_a_graph_is_recorded() {
    let stream = Stream::new().unwrap();
    let p = Projection::new(&stream, 2, 32, 64);
    let mut lt = BlasLt::new(&stream, 0).unwrap();
    let mut output = stream.alloc::<f32>(p.batch * p.m).unwrap();
    let error = Graph::capture(&stream, |s| {
        lt.matmul(
            s,
            &p.shape(),
            p.weights.as_slice(),
            p.input.as_slice(),
            output.as_mut_slice(),
            &FirstUsable,
        )
        .map(|_| ())
    })
    .err()
    .unwrap();
    assert!(
        matches!(
            error,
            LaunchError::Hip(HipError {
                kind: ErrorKind::CaptureViolation,
                ..
            })
        ),
        "{error}"
    );
}

/// The Qwen3-ASR choice (src/models/qwen3_asr/hip/blas.cpp): hipBLASLt's
/// rank 6 without workspace, else the first usable result.
struct AsrRank;

impl AlgoPolicy for AsrRank {
    fn choose(&self, ranked: &[Heuristic], capacity: usize) -> Option<usize> {
        let rank = 6.min(ranked.len().checked_sub(1)?);
        if ranked[rank].is_usable(capacity) {
            return Some(rank);
        }
        ranked.iter().position(|h| h.is_usable(capacity))
    }
}

#[test]
fn the_asr_rank_policy_serves_asr_shapes() {
    let stream = Stream::new().unwrap();
    let mut lt = BlasLt::new(&stream, 0).unwrap();
    // Text decoder projections at a 211-token prompt (hidden 2048).
    for (batch, m, k) in [
        (211, 2048, 2048),
        (211, 1024, 2048),
        (211, 6144, 2048),
        (211, 2048, 6144),
    ] {
        let p = Projection::new(&stream, batch, m, k);
        let mut output = stream.alloc::<f32>(batch * m).unwrap();
        let launched = lt
            .matmul(
                &stream,
                &p.shape(),
                p.weights.as_slice(),
                p.input.as_slice(),
                output.as_mut_slice(),
                &AsrRank,
            )
            .unwrap();
        assert_eq!(launched, Launch::Launched, "{batch}x{m}x{k}");
        p.check(&host(&stream, &output));
        // FNV-1a over the output bits, comparable with the C++ GemmLt.
        let hash = host(&stream, &output)
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, value| {
                value.to_bits().to_le_bytes().iter().fold(h, |h, &byte| {
                    (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
                })
            });
        println!(
            "{batch}x{m}x{k}: solution {:?} hash={hash:016x}",
            lt.installed_index::<Bf16, Bf16, f32>(&p.shape())
        );
    }
}
