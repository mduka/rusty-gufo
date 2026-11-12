//! One draft-and-verify round (`speculative.cpp`).

use std::cell::Cell;

use gufo_core::sampling::SamplerState;

use crate::engine::Session;
use crate::logit_head::{LogitHead, head_filter_for, sample_with_head};

/// The distribution the next token is drawn from: its logits row and, when
/// the session summarized it on the device, its head (empty otherwise).
#[derive(Debug, Clone, Default)]
pub struct Frontier {
    pub logits: Vec<f32>,
    pub head: LogitHead,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeculativeRound {
    /// Emitted tokens; the first was drawn from the incoming row.
    pub tokens: Vec<i32>,
    /// With `keep_rows`, the row each token was drawn from, in order.
    pub rows: Vec<f32>,
    pub drafted: u32,
    pub accepted: u32,
    /// The last token ends the response and was not run.
    pub stop: bool,
}

/// How the trunk has accepted the drafts the lattice was unsure about, in
/// one request: decayed sums of the kept drafts and of the lattice's
/// probabilities for them. The lattice's probabilities track acceptance on
/// average but not per text (code keeps uncertain drafts about twice as
/// often as prose), so rounds scale them by this ratio.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DraftHistory {
    pub kept: f64,
    pub predicted: f64,
}

/// A round verifies drafts while the walk's transition probabilities,
/// scaled by the request's `DraftHistory` and multiplied from the first slot
/// on, stay at or above this; a slot the trunk is less likely to reach costs
/// more verification time than it saves on average. The extension measured
/// +3.5 to +15 % tokens per second on an essay, an explanation, JSON and
/// sampled prose, and -1.8 % on code, whose rounds already keep 6 of 7
/// drafts.
const MIN_DRAFT_REACH: f32 = 0.15;
/// Drafts at or above this probability are nearly always kept and say
/// nothing about the lattice's calibration.
const CERTAIN_DRAFT: f32 = 0.99;
/// Per recorded draft, so the history follows the text within ~30 drafts.
const HISTORY_DECAY: f64 = 0.97;
/// Pseudo-counts that start the ratio at 1.
const HISTORY_PRIOR: f64 = 2.0;

/// Drafts kept for verification: at least one, then while the cumulative
/// scaled probability stays at or above `MIN_DRAFT_REACH`.
fn drafts_worth_verifying(probabilities: &[f32], history: &DraftHistory) -> usize {
    let scale = ((history.kept + HISTORY_PRIOR) / (history.predicted + HISTORY_PRIOR)) as f32;
    let mut reach = 1.0f32;
    let mut keep = 0;
    for &p in probabilities {
        // std::min(1.0F, p * scale)
        let scaled = p * scale;
        reach *= if scaled < 1.0 { scaled } else { 1.0 };
        if keep > 0 && reach < MIN_DRAFT_REACH {
            break;
        }
        keep += 1;
    }
    keep
}

thread_local! {
    /// A round's verified rows and their heads, kept for the thread's next
    /// round as the C++ keeps them (`static thread_local`): a round neither
    /// allocates nor clears megabytes of logits.
    static ROUND_ROWS: Cell<(Vec<f32>, Vec<LogitHead>)> = const { Cell::new((Vec::new(), Vec::new())) };
}

fn token(id: u32) -> Result<i32, String> {
    i32::try_from(id).map_err(|_| "token outside the vocabulary".to_string())
}

/// One draft-and-verify round. `frontier` holds the session's next-token
/// distribution and is replaced by the one the next token will be drawn
/// from; for random sampling the session summarizes the rows on the device,
/// so draws read a few candidates instead of the vocabulary. The first token
/// comes from `sampler`; the drafter proposes up to `max_tokens - 1` more,
/// of which the round keeps the leading ones likely enough to be accepted
/// (at least one) by the lattice's probabilities scaled with `history`, and
/// the trunk verifies them in one pass; `history` then records the verified
/// drafts. Each draft is kept while the
/// sampler, given the trunk's row before it, draws that same draft, so the
/// emitted tokens follow the trunk's own distribution. On the first mismatch
/// the trunk's draw stays pending in the sampler (`defer_sample`) and opens
/// the next round. Emitted tokens are run and accepted into `sampler`,
/// except a stop token: it ends the round as the last token, is not run, and
/// `frontier` keeps the row it was drawn from. Without a drafter, or with
/// `max_tokens == 1`, the round is one ordinary decode step.
pub fn speculative_step(
    session: &mut Session,
    sampler: &mut SamplerState,
    frontier: &mut Frontier,
    history: &mut DraftHistory,
    max_tokens: u32,
    is_stop: &dyn Fn(i32) -> bool,
    keep_rows: bool,
) -> Result<SpeculativeRound, String> {
    let mut out = SpeculativeRound::default();
    if max_tokens == 0 {
        return Err("a decode round emits at least one token".into());
    }
    let vocab = frontier.logits.len();
    let filter = head_filter_for(sampler.config());
    let first = token(
        sample_with_head(sampler, &frontier.logits, &frontier.head).map_err(|e| e.to_string())?,
    )?;
    out.tokens.push(first);
    if keep_rows {
        out.rows.extend_from_slice(&frontier.logits);
    }
    if is_stop(first) {
        out.stop = true;
        return Ok(out);
    }
    sampler.accept(first as u32).map_err(|e| e.to_string())?;
    if session.position() >= session.max_context() {
        return Err("the session context is full".into());
    }

    let mut drafts = Vec::new();
    let mut probabilities = Vec::new();
    let room = session.max_context() - session.position();
    let position = session.position();
    if max_tokens > 1
        && room > 1
        && position > 0
        && let Some(lattice) = session.draft_block(first, position)?
    {
        drafts =
            lattice.walk_with_probabilities((max_tokens - 1).min(room - 1), &mut probabilities);
        drafts.truncate(drafts_worth_verifying(&probabilities, history));
    }
    if drafts.is_empty() {
        let one = [first];
        return match filter {
            Some(filter) => session
                .append_with_head(&one, &mut frontier.logits, &filter, &mut frontier.head)
                .map(|()| out),
            None => {
                frontier.head.candidates.clear();
                session.append(&one, &mut frontier.logits).map(|()| out)
            }
        };
    }

    let mut block = Vec::with_capacity(drafts.len() + 1);
    block.push(first);
    block.extend_from_slice(&drafts);
    let (mut rows, mut heads) = ROUND_ROWS.take();
    rows.resize(block.len() * vocab, 0.0);
    heads.resize_with(block.len(), LogitHead::default);
    for head in &mut heads {
        head.candidates.clear();
    }
    match filter {
        Some(filter) => session.verify_with_heads(&block, &mut rows, &filter, &mut heads)?,
        None => session.verify(&block, &mut rows)?,
    }
    let mut kept = 1usize;
    for (i, &draft) in drafts.iter().enumerate() {
        let row = &rows[i * vocab..(i + 1) * vocab];
        let drawn = token(sample_with_head(sampler, row, &heads[i]).map_err(|e| e.to_string())?)?;
        if drawn != draft {
            sampler
                .defer_sample(drawn as u32)
                .map_err(|e| e.to_string())?;
            break;
        }
        out.tokens.push(drawn);
        if keep_rows {
            out.rows.extend_from_slice(row);
        }
        if is_stop(drawn) {
            // Like a first token that stops, it is reported but never run.
            out.stop = true;
            break;
        }
        sampler.accept(drawn as u32).map_err(|e| e.to_string())?;
        kept += 1;
    }
    out.drafted = drafts.len() as u32;
    out.accepted = out.tokens.len() as u32 - 1;
    // The verified drafts: every accepted one and the first rejected one; a
    // stop token ends the round without a rejection.
    let accepted = out.accepted as usize;
    for (i, &p) in probabilities.iter().enumerate().take(drafts.len()) {
        if i > accepted || (i == accepted && out.stop) {
            break;
        }
        if p >= CERTAIN_DRAFT {
            continue;
        }
        history.kept = HISTORY_DECAY * history.kept + if i < accepted { 1.0 } else { 0.0 };
        history.predicted = HISTORY_DECAY * history.predicted + f64::from(p);
    }
    session.commit(kept as u32)?;
    frontier
        .logits
        .copy_from_slice(&rows[(kept - 1) * vocab..kept * vocab]);
    std::mem::swap(&mut frontier.head, &mut heads[kept - 1]);
    ROUND_ROWS.set((rows, heads));
    Ok(out)
}
