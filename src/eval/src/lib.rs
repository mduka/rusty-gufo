//! The capability evaluation behind `gufo eval`, a port of `src/eval`: the
//! pinned cases ([`dataset`]), answer extraction and grading ([`extract`]),
//! the offline regrade of a captured trace ([`regrade`]) and the client of
//! the OpenAI-compatible endpoint ([`http`]).

#![forbid(unsafe_code)]

pub mod dataset;
pub mod extract;
pub mod http;
pub mod regrade;
