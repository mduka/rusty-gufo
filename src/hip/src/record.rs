//! Launch recording for graph capture and validation.
//!
//! While a stream records, every generated launch wrapper reports its
//! arguments here after its checks. During capture the recorder retains the
//! allocation behind every buffer argument and stores the launch sequence.
//! During validation the wrapper does not launch; its arguments are compared
//! with the captured sequence instead.

use std::collections::HashSet;
use std::sync::Arc;

use crate::error::LaunchError;
use crate::launch::Launch;
use crate::memory::Allocation;

/// One launch argument as the recorder sees it.
pub struct Arg<'a> {
    name: &'static str,
    value: u64,
    extent: u64,
    allocation: Option<&'a Arc<Allocation>>,
}

impl<'a> Arg<'a> {
    /// A buffer argument: its allocation (retained by capture), device address
    /// and length. `None` for an absent optional buffer.
    pub fn buffer(
        name: &'static str,
        allocation: Option<&'a Arc<Allocation>>,
        address: usize,
        len: usize,
    ) -> Self {
        Self {
            name,
            value: address as u64,
            extent: len as u64,
            allocation,
        }
    }

    /// A scalar argument by its bit pattern.
    pub fn scalar(name: &'static str, bits: u64) -> Self {
        Self {
            name,
            value: bits,
            extent: 0,
            allocation: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LaunchRecord {
    launcher: &'static str,
    arguments: Vec<(&'static str, u64, u64)>,
    result: Launch,
}

pub(crate) enum Recorder {
    Capture {
        launches: Vec<LaunchRecord>,
        retained: Vec<Arc<Allocation>>,
        seen: HashSet<u64>,
    },
    Validate {
        expected: Arc<[LaunchRecord]>,
        cursor: usize,
    },
}

/// What a finished capture keeps.
pub(crate) struct Captured {
    pub(crate) launches: Arc<[LaunchRecord]>,
    pub(crate) retained: Vec<Arc<Allocation>>,
}

impl Recorder {
    pub(crate) fn capture() -> Self {
        Self::Capture {
            launches: Vec::new(),
            retained: Vec::new(),
            seen: HashSet::new(),
        }
    }

    pub(crate) fn validate(expected: Arc<[LaunchRecord]>) -> Self {
        Self::Validate {
            expected,
            cursor: 0,
        }
    }

    /// `Ok(None)`: launch as usual. `Ok(Some(result))`: validation, do not
    /// launch and return the result the capture saw.
    pub(crate) fn record(
        &mut self,
        launcher: &'static str,
        args: &[Arg<'_>],
    ) -> Result<Option<Launch>, LaunchError> {
        match self {
            Self::Capture {
                launches,
                retained,
                seen,
            } => {
                for allocation in args.iter().filter_map(|arg| arg.allocation) {
                    if seen.insert(allocation.id()) {
                        retained.push(Arc::clone(allocation));
                    }
                }
                launches.push(LaunchRecord {
                    launcher,
                    arguments: args.iter().map(|a| (a.name, a.value, a.extent)).collect(),
                    result: Launch::Launched,
                });
                Ok(None)
            }
            Self::Validate { expected, cursor } => {
                let launch = *cursor;
                let mismatch = |argument| LaunchError::GraphMismatch {
                    launch,
                    launcher,
                    argument,
                };
                let record = expected.get(launch).ok_or(mismatch("<extra launch>"))?;
                if record.launcher != launcher || record.arguments.len() != args.len() {
                    return Err(mismatch("<different launcher>"));
                }
                for (&(name, value, extent), arg) in record.arguments.iter().zip(args) {
                    if name != arg.name || value != arg.value || extent != arg.extent {
                        return Err(mismatch(arg.name));
                    }
                }
                *cursor += 1;
                Ok(Some(record.result))
            }
        }
    }

    /// The launcher's actual result for the launch recorded last.
    pub(crate) fn set_result(&mut self, result: Launch) {
        if let Self::Capture { launches, .. } = self
            && let Some(last) = launches.last_mut()
        {
            last.result = result;
        }
    }

    pub(crate) fn into_captured(self) -> Option<Captured> {
        match self {
            Self::Capture {
                launches, retained, ..
            } => Some(Captured {
                launches: launches.into(),
                retained,
            }),
            Self::Validate { .. } => None,
        }
    }

    /// Validation must have replayed exactly the captured launches.
    pub(crate) fn finish_validation(&self) -> Result<(), LaunchError> {
        if let Self::Validate { expected, cursor } = self
            && let Some(missing) = expected.get(*cursor)
        {
            return Err(LaunchError::GraphMismatch {
                launch: *cursor,
                launcher: missing.launcher,
                argument: "<missing launch>",
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(args: &[Arg<'_>]) -> Arc<[LaunchRecord]> {
        let mut recorder = Recorder::capture();
        assert_eq!(recorder.record("LaunchOp", args).unwrap(), None);
        recorder.set_result(Launch::Unsupported);
        recorder.into_captured().unwrap().launches
    }

    #[test]
    fn validation_replays_results_and_names_the_first_changed_argument() {
        let captured = capture(&[Arg::scalar("pos", 7), Arg::scalar("dim", 128)]);
        let mut same = Recorder::validate(Arc::clone(&captured));
        let replayed = same.record(
            "LaunchOp",
            &[Arg::scalar("pos", 7), Arg::scalar("dim", 128)],
        );
        assert_eq!(replayed.unwrap(), Some(Launch::Unsupported));
        assert!(same.finish_validation().is_ok());

        let mut frozen = Recorder::validate(Arc::clone(&captured));
        let error = frozen
            .record(
                "LaunchOp",
                &[Arg::scalar("pos", 8), Arg::scalar("dim", 128)],
            )
            .unwrap_err();
        assert_eq!(
            error,
            LaunchError::GraphMismatch {
                launch: 0,
                launcher: "LaunchOp",
                argument: "pos"
            }
        );
    }

    #[test]
    fn validation_rejects_extra_and_missing_launches() {
        let captured = capture(&[Arg::scalar("n", 1)]);
        let mut extra = Recorder::validate(Arc::clone(&captured));
        extra.record("LaunchOp", &[Arg::scalar("n", 1)]).unwrap();
        let error = extra
            .record("LaunchOp", &[Arg::scalar("n", 1)])
            .unwrap_err();
        assert!(matches!(
            error,
            LaunchError::GraphMismatch {
                argument: "<extra launch>",
                ..
            }
        ));

        let missing = Recorder::validate(captured);
        assert!(matches!(
            missing.finish_validation(),
            Err(LaunchError::GraphMismatch {
                argument: "<missing launch>",
                ..
            })
        ));
    }
}
