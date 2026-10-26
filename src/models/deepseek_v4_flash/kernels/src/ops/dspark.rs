//! DSpark candidate selection, through the generated launchers of
//! `launch.toml` (formerly the DSpark part of `ds4_rocm_indexer.hip.hpp`).
//! The kernels read the previous tokens from the device as rows of
//! the Markov table without checking them, so the wrappers take them from
//! the host, check them and write them, as the C++ graph does before every
//! launch. The row-span checks of the C++ launchers can wrap; the wrappers
//! check without.

use crate::backend::{Backend, Error, Result};
use crate::generated;
use crate::tensor::{Input, Output};

use super::{I32, Weight, ensure, launched, need, product, typed, typed_mut};

/// `ds4_dspark_candidates`: eight IDs, eight logits and a confidence.
pub const CANDIDATES_BYTES: u64 = 68;
impl Backend {
    /// Checks `tokens` below `vocab` and writes them to the start of
    /// `buffer` as I32.
    fn write_previous(
        &self,
        op: &'static str,
        buffer: &mut impl Output,
        tokens: &[i32],
        vocab: u32,
    ) -> Result<()> {
        ensure(op, tokens.iter().all(|&t| t >= 0 && (t as u32) < vocab))?;
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        self.write(buffer, 0, &bytes)
    }

    /// The Markov-coupled candidates and confidence of one drafted position
    /// for each request, after the request's `previous` token (written to
    /// `previous_buffer`).
    pub fn dspark_candidates(
        &self,
        out: &mut impl Output,
        scratch: &mut impl Output,
        logits: &impl Input,
        logits_stride: u64,
        hidden: &impl Input,
        hidden_stride: u64,
        previous: &[i32],
        previous_buffer: &mut (impl Output + Input),
        w1: Weight,
        w2: Weight,
        confidence: Weight,
        vocab: u32,
        rank: u32,
        hidden_width: u32,
    ) -> Result<()> {
        const OP: &str = "dspark candidates";
        let requests = previous.len() as u64;
        ensure(
            OP,
            (1..=8).contains(&requests) && vocab != 0 && rank.is_multiple_of(32),
        )?;
        ensure(OP, w1.model == w2.model && w1.model == confidence.model)?;
        // The kernels read the blocks' F16 scales in place.
        ensure(
            OP,
            w1.offset.is_multiple_of(2)
                && w2.offset.is_multiple_of(2)
                && confidence.offset.is_multiple_of(2),
        )?;
        need(OP, previous_buffer, &[requests, I32])?;
        self.write_previous(OP, previous_buffer, previous, vocab)?;
        let table = product(OP, &[u64::from(vocab), u64::from(rank / 32), 34])?;
        let confidence_bytes = (u64::from(hidden_width) + u64::from(rank)) / 32 * 34;
        let w1 = self.weight_bytes(w1, table)?;
        let w2 = self.weight_bytes(w2, table)?;
        let confidence = self.weight_bytes(confidence, confidence_bytes)?;
        let to_usize = |v: u64| usize::try_from(v).map_err(|_| Error { op: OP });
        launched(
            OP,
            generated::dspark_candidates(
                self.stream(),
                out.output(),
                typed_mut(OP, scratch.output())?,
                typed(OP, logits.input())?,
                to_usize(logits_stride)?,
                typed(OP, hidden.input())?,
                to_usize(hidden_stride)?,
                typed(OP, previous_buffer.input())?,
                w1,
                w2,
                confidence,
                vocab,
                rank,
                hidden_width,
                requests as u32,
            ),
        )
    }

    /// The best token of one logit row after `previous_token` into
    /// `out_index` (I32).
    pub fn dspark_markov_argmax(
        &self,
        out_index: &mut impl Output,
        scratch_key: &mut impl Output,
        logits_row: &impl Input,
        w1: Weight,
        w2: Weight,
        vocab: u32,
        markov_rank: u32,
        previous_token: u32,
    ) -> Result<()> {
        const OP: &str = "dspark markov argmax";
        ensure(OP, w1.model == w2.model && markov_rank.is_multiple_of(32))?;
        ensure(
            OP,
            w1.offset.is_multiple_of(2) && w2.offset.is_multiple_of(2),
        )?;
        ensure(OP, previous_token < vocab)?;
        let row_bytes = u64::from(markov_rank / 32) * 34;
        let w1_row = Weight {
            model: w1.model,
            offset: w1
                .offset
                .checked_add(product(OP, &[u64::from(previous_token), row_bytes])?)
                .ok_or(Error { op: OP })?,
        };
        let w1_row = self.weight_bytes(w1_row, row_bytes)?;
        let w2 = self.weight_bytes(w2, product(OP, &[u64::from(vocab), row_bytes])?)?;
        launched(
            OP,
            generated::dspark_markov_argmax(
                self.stream(),
                typed_mut(OP, out_index.output())?,
                typed_mut(OP, scratch_key.output())?,
                typed(OP, logits_row.input())?,
                w1_row,
                w2,
                vocab,
                markov_rank,
            ),
        )
    }

    /// One row per request (2, 4, 6 or 8), `logits_row_stride` floats apart,
    /// after each request's `previous` token (written to
    /// `previous_buffer`).
    pub fn dspark_markov_argmax_batch(
        &self,
        out_index: &mut impl Output,
        scratch_key: &mut impl Output,
        logits_rows: &impl Input,
        previous: &[i32],
        previous_buffer: &mut (impl Output + Input),
        w1: Weight,
        w2: Weight,
        vocab: u32,
        markov_rank: u32,
        logits_row_stride: u64,
    ) -> Result<()> {
        const OP: &str = "dspark markov argmax batch";
        let n_rows = previous.len() as u64;
        ensure(OP, matches!(n_rows, 2 | 4 | 6 | 8))?;
        ensure(OP, w1.model == w2.model && markov_rank.is_multiple_of(32))?;
        ensure(
            OP,
            w1.offset.is_multiple_of(2) && w2.offset.is_multiple_of(2),
        )?;
        need(OP, previous_buffer, &[n_rows, I32])?;
        self.write_previous(OP, previous_buffer, previous, vocab)?;
        let table = product(OP, &[u64::from(vocab), u64::from(markov_rank / 32), 34])?;
        let w1 = self.weight_bytes(w1, table)?;
        let w2 = self.weight_bytes(w2, table)?;
        let launch = launched(
            OP,
            generated::dspark_markov_argmax_batch(
                self.stream(),
                typed_mut(OP, out_index.output())?,
                typed_mut(OP, scratch_key.output())?,
                typed(OP, logits_rows.input())?,
                usize::try_from(logits_row_stride).map_err(|_| Error { op: OP })?,
                typed(OP, previous_buffer.input())?,
                w1,
                w2,
                vocab,
                markov_rank,
                n_rows as u32,
            ),
        )?;
        ensure(OP, launch == gufo_hip::Launch::Launched)
    }
}
