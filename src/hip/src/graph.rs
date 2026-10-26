//! HIP graphs whose captured memory cannot disappear.
//!
//! A [`Graph`] holds every allocation its launches captured, so replaying it
//! can never touch freed memory. Before each replay [`GraphCache`] checks that
//! all of them still have their owner; a graph that captured a buffer the
//! program has since replaced is stale and captured again. Validation runs the
//! body record-only before each replay; its launches, buffers, lengths and
//! scalars must equal the capture. This catches per-step values passed as
//! scalars (which a graph would freeze) and switches between live buffers
//! that ownership checks cannot see.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

use crate::deferred;
use crate::error::{ErrorKind, HipError, LaunchError, check};
use crate::memory::Allocation;
use crate::record::{LaunchRecord, Recorder};
use crate::stream::Stream;
use crate::sys;

/// An instantiated capture of one stream's launches.
pub struct Graph {
    exec: sys::HipGraphExec,
    stream_id: u64,
    launches: Arc<[LaunchRecord]>,
    retained: Vec<Arc<Allocation>>,
}

// SAFETY: the executable graph is a HIP handle usable from any thread; the
// retained allocations are Send.
unsafe impl Send for Graph {}

/// Ends an interrupted capture or validation (error or panic in the body).
struct RecordingGuard<'s> {
    stream: &'s Stream,
    capturing: bool,
}

impl Drop for RecordingGuard<'_> {
    fn drop(&mut self) {
        let _ = self.stream.take_recorder();
        if self.capturing {
            let mut graph = std::ptr::null_mut();
            // SAFETY: the stream is capturing; ending it returns the partial
            // graph, which is destroyed unused.
            unsafe {
                let _ = sys::hipStreamEndCapture(self.stream.as_raw(), &mut graph);
                if !graph.is_null() {
                    let _ = sys::hipGraphDestroy(graph);
                }
            }
            self.stream.set_capturing(false);
        }
    }
}

impl Graph {
    /// Captures the launches of `body` on `stream`. Nothing executes; the
    /// body must not allocate, synchronise or copy (see [`ErrorKind`]).
    pub fn capture<F>(stream: &Stream, body: F) -> Result<Self, LaunchError>
    where
        F: FnOnce(&Stream) -> Result<(), LaunchError>,
    {
        let captured = Self::capture_exclusive(stream, body);
        // Frees queued while this (or any) capture ran.
        deferred::collect();
        captured
    }

    fn capture_exclusive<F>(stream: &Stream, body: F) -> Result<Self, LaunchError>
    where
        F: FnOnce(&Stream) -> Result<(), LaunchError>,
    {
        stream.ensure_healthy()?;
        if stream.is_recording() {
            return Err(HipError::capture_violation("nested graph capture").into());
        }
        // Device-wide synchronisations wait while this guard is held; they
        // would invalidate the capture.
        let _gate = deferred::capture_guard();
        // SAFETY: live stream; thread-local mode only restricts this thread.
        let status = unsafe {
            sys::hipStreamBeginCapture(stream.as_raw(), sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        };
        check(status, "hipStreamBeginCapture")?;
        stream.set_capturing(true);
        stream.begin_recording(Recorder::capture());
        let mut guard = RecordingGuard {
            stream,
            capturing: true,
        };
        body(stream)?;

        guard.capturing = false;
        let captured = stream.take_recorder().and_then(Recorder::into_captured);
        let mut graph = std::ptr::null_mut();
        // SAFETY: the stream is capturing and `graph` is a valid out-pointer.
        let status = unsafe { sys::hipStreamEndCapture(stream.as_raw(), &mut graph) };
        stream.set_capturing(false);
        drop(guard);
        check(status, "hipStreamEndCapture")?;
        let captured = captured.ok_or(HipError::refused("graph capture lost its recorder"))?;

        let mut exec = std::ptr::null_mut();
        // SAFETY: `graph` is the captured graph; no error node or log buffer.
        let status = unsafe {
            sys::hipGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        // SAFETY: the executable graph does not depend on the template.
        unsafe {
            let _ = sys::hipGraphDestroy(graph);
        }
        check(status, "hipGraphInstantiate")?;
        Ok(Self {
            exec,
            stream_id: stream.id(),
            launches: captured.launches,
            retained: captured.retained,
        })
    }

    /// Whether any captured allocation lost its owner since the capture.
    pub fn is_stale(&self) -> bool {
        self.retained
            .iter()
            .any(|allocation| !allocation.is_owned())
    }

    /// Enqueues the graph on the stream it was captured on.
    pub fn launch(&self, stream: &Stream) -> Result<(), LaunchError> {
        stream.ensure_healthy()?;
        if stream.id() != self.stream_id {
            return Err(
                HipError::refused("graph replay on another stream than its capture").into(),
            );
        }
        if stream.is_recording() {
            return Err(HipError::capture_violation("graph replay").into());
        }
        // SAFETY: live executable graph and stream; every captured allocation
        // is retained by `self` and was checked for its owner by the caller.
        let status = unsafe { sys::hipGraphLaunch(self.exec, stream.as_raw()) };
        Ok(stream.observe(check(status, "hipGraphLaunch"))?)
    }

    /// Runs `body` record-only and compares its launches with the capture.
    pub fn validate<F>(&self, stream: &Stream, body: F) -> Result<(), LaunchError>
    where
        F: FnOnce(&Stream) -> Result<(), LaunchError>,
    {
        if stream.is_recording() {
            return Err(HipError::capture_violation("nested graph validation").into());
        }
        stream.begin_recording(Recorder::validate(Arc::clone(&self.launches)));
        let guard = RecordingGuard {
            stream,
            capturing: false,
        };
        body(stream)?;
        let recorder = stream.take_recorder();
        drop(guard);
        match recorder {
            Some(recorder) => recorder.finish_validation(),
            None => Err(HipError::refused("graph validation lost its recorder").into()),
        }
    }

    pub fn launch_count(&self) -> usize {
        self.launches.len()
    }

    pub fn retained(&self) -> &[Arc<Allocation>] {
        &self.retained
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: instantiated by hipGraphInstantiate and destroyed once. HIP
        // defers the release until enqueued replays have completed; the
        // retained allocations are freed after this, in their streams' order.
        unsafe {
            let _ = sys::hipGraphExecDestroy(self.exec);
        }
    }
}

/// How [`GraphCache::run`] executed the body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphRun {
    /// Executed directly: the warm-up run of a new key, or a key whose body
    /// cannot be captured.
    Eager,
    /// Captured for the first time and replayed.
    Captured,
    /// Replayed an existing capture.
    Replayed,
    /// The previous capture was stale; captured again and replayed.
    Recaptured,
}

/// When a replay is validated and what happens when the body no longer
/// matches its capture: a scalar that changes per step, another buffer or
/// slice, or a different launch sequence under the same key. Such a body is a
/// bug; per-step values belong in device buffers. Memory stays sound either
/// way: a graph retains every allocation it captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MismatchPolicy {
    /// Validate after the launch, while the device runs the graph, and return
    /// [`LaunchError::GraphMismatch`]: the replayed step has already run with
    /// the captured arguments, so what it wrote is wrong and the caller must
    /// treat its state as lost. The key runs eagerly from then on. Default.
    Error,
    /// Validate before the launch, capture again and count the mismatch;
    /// results stay correct, at the cost of the validation on the critical
    /// path. A key that mismatches on two replays in a row runs eagerly from
    /// then on.
    Recapture,
}

enum Entry {
    Warm,
    Captured { graph: Graph, after_mismatch: bool },
    EagerOnly,
}

enum Action {
    FirstRun,
    Eager,
    Replay,
    Capture { run: GraphRun, after_mismatch: bool },
    Mismatch { error: LaunchError, repeated: bool },
}

/// One graph per key (typically per batch shape), owned by a session.
///
/// A new key first runs eagerly, because warm-up may grow arenas; the second
/// run captures, later runs replay. A body that allocates, synchronises or
/// copies during capture, or that HIP cannot capture, stays eager for that key.
///
/// With every replay the body runs record-only and must reproduce the
/// captured launches exactly (validation, about 160 ns per launch of a
/// generated launcher), after the launch by default; see [`MismatchPolicy`].
/// `GUFO_GRAPH_VALIDATE=0` disables it for measurements.
pub struct GraphCache<K> {
    entries: HashMap<K, Entry>,
    validate: bool,
    policy: MismatchPolicy,
    mismatches: u64,
    last_mismatch: Option<LaunchError>,
}

impl<K: Eq + Hash> Default for GraphCache<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash> GraphCache<K> {
    pub fn new() -> Self {
        let disabled = std::env::var("GUFO_GRAPH_VALIDATE").is_ok_and(|v| v == "0");
        Self {
            entries: HashMap::new(),
            validate: !disabled,
            policy: MismatchPolicy::Error,
            mismatches: 0,
            last_mismatch: None,
        }
    }

    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }

    pub fn with_mismatch_policy(mut self, policy: MismatchPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn validates(&self) -> bool {
        self.validate
    }

    /// Validation mismatches, reported or recaptured.
    pub fn mismatches(&self) -> u64 {
        self.mismatches
    }

    pub fn last_mismatch(&self) -> Option<&LaunchError> {
        self.last_mismatch.as_ref()
    }

    pub fn run<F>(&mut self, stream: &Stream, key: K, mut body: F) -> Result<GraphRun, LaunchError>
    where
        F: FnMut(&Stream) -> Result<(), LaunchError>,
    {
        self.run_with(stream, key, &mut (), |_, s| body(s), |_| Ok(()))
    }

    /// [`run`](Self::run) with `before_launch`, called once before any of
    /// the body's work is enqueued (after the validation of a
    /// [`MismatchPolicy::Recapture`] replay, before that of an
    /// [`MismatchPolicy::Error`] one): host work the graph's device work
    /// depends on, such as waiting for input it uploads. `context` is lent to
    /// both closures. `before_launch` may enqueue work and wait, but must not
    /// change what the body launches; a buffer it replaces still makes the
    /// graph stale, which captures it again instead of replaying.
    pub fn run_with<C, F, P>(
        &mut self,
        stream: &Stream,
        key: K,
        context: &mut C,
        mut body: F,
        before_launch: P,
    ) -> Result<GraphRun, LaunchError>
    where
        F: FnMut(&mut C, &Stream) -> Result<(), LaunchError>,
        P: FnOnce(&mut C) -> Result<(), LaunchError>,
    {
        let mut bound = |s: &Stream| body(context, s);
        let action = match self.entries.get(&key) {
            None => Action::FirstRun,
            Some(Entry::EagerOnly) => Action::Eager,
            Some(Entry::Captured {
                graph,
                after_mismatch,
            }) if !graph.is_stale() => {
                let verdict = if self.validate && self.policy == MismatchPolicy::Recapture {
                    graph.validate(stream, &mut bound)
                } else {
                    Ok(())
                };
                match verdict {
                    Ok(()) => Action::Replay,
                    Err(error @ LaunchError::GraphMismatch { .. })
                        if self.policy == MismatchPolicy::Recapture =>
                    {
                        Action::Mismatch {
                            error,
                            repeated: *after_mismatch,
                        }
                    }
                    Err(error) => return Err(error),
                }
            }
            Some(Entry::Captured { .. }) => Action::Capture {
                run: GraphRun::Recaptured,
                after_mismatch: false,
            },
            Some(Entry::Warm) => Action::Capture {
                run: GraphRun::Captured,
                after_mismatch: false,
            },
        };
        before_launch(context)?;
        let mut bound = |s: &Stream| body(context, s);
        match action {
            Action::FirstRun => {
                bound(stream)?;
                self.entries.insert(key, Entry::Warm);
                Ok(GraphRun::Eager)
            }
            Action::Eager => {
                bound(stream)?;
                Ok(GraphRun::Eager)
            }
            Action::Replay => {
                let validate_after = self.validate && self.policy == MismatchPolicy::Error;
                let verdict = match self.entries.get_mut(&key) {
                    Some(Entry::Captured {
                        graph,
                        after_mismatch,
                    }) if !graph.is_stale() => {
                        graph.launch(stream)?;
                        *after_mismatch = false;
                        // The device runs the graph meanwhile.
                        Some(if validate_after {
                            graph.validate(stream, &mut bound)
                        } else {
                            Ok(())
                        })
                    }
                    _ => None,
                };
                match verdict {
                    Some(Ok(())) => Ok(GraphRun::Replayed),
                    Some(Err(error)) => {
                        // The body no longer reproduces its capture: never
                        // replay this key again.
                        if matches!(error, LaunchError::GraphMismatch { .. }) {
                            self.mismatches += 1;
                            self.last_mismatch = Some(error.clone());
                        }
                        self.entries.insert(key, Entry::EagerOnly);
                        Err(error)
                    }
                    None => self.capture(stream, key, &mut bound, GraphRun::Recaptured, false),
                }
            }
            Action::Capture {
                run,
                after_mismatch,
            } => self.capture(stream, key, &mut bound, run, after_mismatch),
            Action::Mismatch { error, repeated } => {
                self.mismatches += 1;
                self.last_mismatch = Some(error);
                if repeated {
                    self.entries.insert(key, Entry::EagerOnly);
                    bound(stream)?;
                    Ok(GraphRun::Eager)
                } else {
                    self.capture(stream, key, &mut bound, GraphRun::Recaptured, true)
                }
            }
        }
    }

    fn capture<F>(
        &mut self,
        stream: &Stream,
        key: K,
        body: &mut F,
        run: GraphRun,
        after_mismatch: bool,
    ) -> Result<GraphRun, LaunchError>
    where
        F: FnMut(&Stream) -> Result<(), LaunchError>,
    {
        // Drop a stale graph first so its retained memory is released.
        self.entries.remove(&key);
        match Graph::capture(stream, &mut *body) {
            Ok(graph) => {
                graph.launch(stream)?;
                self.entries.insert(
                    key,
                    Entry::Captured {
                        graph,
                        after_mismatch,
                    },
                );
                Ok(run)
            }
            Err(error) if capture_unsupported(&error) => {
                body(stream)?;
                self.entries.insert(key, Entry::EagerOnly);
                Ok(GraphRun::Eager)
            }
            Err(error) => Err(error),
        }
    }

    /// Bytes kept alive by captured graphs (each allocation counted once),
    /// including memory whose owner is already gone.
    pub fn retained_bytes(&self) -> usize {
        let mut seen = HashSet::new();
        self.entries
            .values()
            .filter_map(|entry| match entry {
                Entry::Captured { graph, .. } => Some(graph.retained()),
                _ => None,
            })
            .flatten()
            .filter(|allocation| seen.insert(allocation.id()))
            .map(|allocation| allocation.bytes())
            .sum()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Capture failures that mean "run this body eagerly", not "the body is wrong".
fn capture_unsupported(error: &LaunchError) -> bool {
    matches!(
        error,
        LaunchError::Hip(HipError {
            kind: ErrorKind::CaptureViolation,
            ..
        }) | LaunchError::Hip(HipError {
            code: 900..=908,
            kind: ErrorKind::Runtime,
            ..
        })
    )
}
