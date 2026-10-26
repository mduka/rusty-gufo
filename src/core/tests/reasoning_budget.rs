//! Port of the reasoning budget cases of `tests/core/sampling_test.cpp`
//!: an exhausted budget forces the end of thinking on
//! every sampling path, as llama.cpp's reasoning-budget sampler does.

use std::sync::Arc;

use gufo_core::json_constraint::{ConstraintVocabulary, JsonConstraint, Piece, TokenConstraint};
use gufo_core::sampling::{
    ReasoningBudget, SamplerState, SamplingConfig, TokenId, build_distribution,
};

/// Greedy logits whose argmax is `winner`; token 3 stands for `</think>`.
fn logits(winner: TokenId) -> Vec<f32> {
    let mut logits = vec![0.0f32; 8];
    logits[winner as usize] = 5.0;
    logits
}

fn budget(tokens: usize, forced: &[TokenId], prompt: &[TokenId]) -> ReasoningBudget {
    ReasoningBudget {
        tokens,
        forced: forced.to_vec(),
        ends: vec![3],
        prompt: prompt.to_vec(),
        partial_utf8: None,
    }
}

fn with_budget(budget: ReasoningBudget) -> SamplingConfig {
    SamplingConfig {
        reasoning_budget: Some(Arc::new(budget)),
        ..SamplingConfig::default()
    }
}

#[test]
fn reasoning_budget_forces_the_end() {
    let config = with_budget(budget(2, &[3], &[]));
    let mut sampler = SamplerState::new(config.clone(), &[]).unwrap();
    assert!(
        !config.can_use_unmodified_argmax(),
        "a budget keeps requests off the unmodified argmax path"
    );
    assert!(
        sampler.forced_token().is_none()
            && sampler.can_select_argmax(1, false).unwrap()
            && !sampler.needs_constraint_mask().unwrap(),
        "the budget does not restrict reasoning before it is spent"
    );
    let token = sampler.sample(&logits(1)).unwrap();
    sampler.accept(token).unwrap();
    let token = sampler.sample(&logits(2)).unwrap();
    sampler.accept(token).unwrap();
    assert_eq!(
        sampler.forced_token(),
        Some(3),
        "two reasoning tokens exhaust a budget of two"
    );
    assert!(
        sampler.needs_constraint_mask().unwrap(),
        "device sampling defers the forced end to the host sampler"
    );
    assert!(
        !sampler.can_select_argmax(1, false).unwrap()
            && sampler.can_select_argmax(3, false).unwrap(),
        "only the reasoning end is a selectable argmax"
    );
    assert_eq!(
        sampler
            .distribution(&logits(1))
            .unwrap()
            .best_token()
            .unwrap(),
        3
    );
    let compact = [1.0f32, 0.0, 2.0];
    assert_eq!(
        sampler
            .distribution_compact(&compact, &[5, 3, 1])
            .unwrap()
            .best_token()
            .unwrap(),
        1,
        "compact distributions index the forced end among their IDs"
    );
    // An MTP draft shortlist may miss the forced end: its proposal follows
    // the shortlist, and target verification rejects it and emits the end.
    assert_eq!(
        sampler
            .distribution_compact(&compact, &[5, 4, 1])
            .unwrap()
            .best_token()
            .unwrap(),
        2,
        "a shortlist without the forced end keeps its own distribution"
    );
    let mut copy = sampler.clone();
    assert_eq!(
        sampler.sample(&logits(1)).unwrap(),
        3,
        "sampling returns the forced end"
    );
    copy.accept(3).unwrap();
    assert!(
        copy.forced_token().is_none()
            && !copy.needs_constraint_mask().unwrap()
            && sampler.forced_token().is_some(),
        "copies advance independently, as speculative verification needs"
    );
    assert_eq!(
        copy.sample(&logits(6)).unwrap(),
        6,
        "the answer samples normally"
    );
    copy.reset_history(&[]);
    assert!(
        copy.forced_token().is_none(),
        "a restart counts the budget again"
    );
}

#[test]
fn early_end_and_no_budget() {
    let mut sampler = SamplerState::new(with_budget(budget(4, &[3], &[])), &[]).unwrap();
    sampler.accept_tokens(&[1, 3, 1, 1, 1, 1]).unwrap();
    assert!(
        sampler.forced_token().is_none(),
        "a reasoning end within the budget closes the phase for good"
    );
    let mut unlimited = SamplerState::new(SamplingConfig::default(), &[]).unwrap();
    unlimited.accept_tokens(&[1; 100]).unwrap();
    assert!(
        unlimited.forced_token().is_none(),
        "requests without a budget are unchanged"
    );
    let zero = SamplerState::new(with_budget(budget(0, &[3], &[])), &[]).unwrap();
    assert_eq!(
        zero.forced_token(),
        Some(3),
        "a zero budget closes reasoning at once"
    );
}

/// Qwen as llama.cpp runs it: the prompt ends with "<think>\n", the forced
/// end is "\n</think>" (tokens 4, 3), and token 7 (a tool call) also ends
/// reasoning.
#[test]
fn reasoning_budget_matches_llama_cpp() {
    for tokens in [1usize, 2, 4] {
        let mut sampler =
            SamplerState::new(with_budget(budget(tokens, &[4, 3], &[4])), &[]).unwrap();
        let mut emitted = Vec::new();
        for _ in 0..8 {
            let token = sampler.sample(&logits(1)).unwrap();
            sampler.accept(token).unwrap();
            emitted.push(token);
        }
        let mut expected = vec![1; tokens - 1];
        expected.extend([4, 3]);
        expected.resize(8, 1);
        assert_eq!(
            emitted, expected,
            "the prompt newline counts, the end is forced"
        );
    }
    let mut zero = SamplerState::new(with_budget(budget(0, &[4, 3], &[4])), &[]).unwrap();
    assert_eq!(
        zero.forced_token(),
        Some(3),
        "a zero budget's prompt newline stands for the forced newline"
    );
    zero.accept(3).unwrap();
    assert!(
        zero.forced_token().is_none(),
        "the forced end completes reasoning"
    );

    let mut ending = budget(2, &[4, 3], &[]);
    ending.ends.push(7);
    let mut tool = SamplerState::new(with_budget(ending), &[]).unwrap();
    tool.accept(7).unwrap();
    tool.accept_tokens(&[1; 4]).unwrap();
    assert!(
        tool.forced_token().is_none(),
        "a tool call ends reasoning before the budget"
    );

    // Token 5 ends inside a UTF-8 character, token 6 completes it.
    let mut utf8 = budget(2, &[3], &[]);
    utf8.partial_utf8 = Some(Arc::new(vec![
        false, false, false, false, false, true, false, false,
    ]));
    let mut split = SamplerState::new(with_budget(utf8), &[]).unwrap();
    split.accept_tokens(&[1, 5, 5]).unwrap();
    assert!(
        split.forced_token().is_none(),
        "a spent budget waits for a whole character"
    );
    split.accept(6).unwrap();
    assert_eq!(
        split.forced_token(),
        Some(3),
        "the end follows the completed character"
    );
}

#[test]
fn reasoning_budget_keeps_literal_answer() {
    let config = with_budget(budget(2, &[4, 3], &[4]));
    for natural_end in [false, true] {
        let mut sampler = SamplerState::new(config.clone(), &[]).unwrap();
        if natural_end {
            sampler.accept(3).unwrap();
        } else {
            sampler.accept_tokens(&[1, 4, 3]).unwrap();
        }
        // Token 0 stands for a literal <think> in answer or tool data: the
        // response framing reads reasoning markup only in the first phase.
        sampler.accept_tokens(&[6, 0, 1, 2, 1, 2, 1, 2]).unwrap();
        assert!(
            sampler.forced_token().is_none() && sampler.sample(&logits(6)).unwrap() == 6,
            "literal answer tokens never restart a completed budget"
        );
    }
    assert_eq!(
        build_distribution(&logits(1), &config, &[], &[1])
            .unwrap()
            .best_token()
            .unwrap(),
        4,
        "distribution construction enforces an exhausted budget"
    );
    assert_eq!(
        build_distribution(&logits(1), &config, &[], &[1, 4, 3, 6, 0, 1, 2])
            .unwrap()
            .best_token()
            .unwrap(),
        1,
        "distribution construction keeps literal answer content"
    );
}

/// With a tool or format grammar, the forced end is a legal grammar step
/// and the answer after it stays constrained.
#[test]
fn reasoning_budget_with_grammar() {
    let pieces: [&[u8]; 6] = [
        b"thinking",
        b"</think>",
        b"{\"x\":",
        b"true}",
        b"prose",
        b"",
    ];
    let vocabulary = Arc::new(
        ConstraintVocabulary::new(pieces.len() as u32, |i| {
            Ok(Piece {
                text: pieces[i as usize].to_vec(),
                stop: i == 5,
            })
        })
        .unwrap(),
    );
    let constraint = Arc::new(TokenConstraint::new(
        JsonConstraint::with_reasoning(JsonConstraint::object().unwrap()),
        vocabulary,
    ));
    let config = SamplingConfig {
        constraint: Some(constraint),
        reasoning_budget: Some(Arc::new(ReasoningBudget {
            tokens: 1,
            forced: vec![1],
            ends: vec![1],
            ..ReasoningBudget::default()
        })),
        ..SamplingConfig::default()
    };
    let mut sampler = SamplerState::new(config, &[]).unwrap();
    let token = sampler.sample(&[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    sampler.accept(token).unwrap();
    let forced = sampler.sample(&[5.0, 0.0, 0.0, 0.0, 4.0, 0.0]).unwrap();
    assert_eq!(forced, 1, "the grammar request also closes at its budget");
    sampler.accept(forced).unwrap();
    assert!(
        sampler.forced_token().is_none()
            && sampler.sample(&[5.0, 0.0, 1.0, 0.0, 4.0, 0.0]).unwrap() == 2,
        "after the forced end the answer still follows the grammar"
    );
}

#[test]
fn partial_utf8_tokens() {
    let pieces: [&[u8]; 7] = [
        b"a",
        "é".as_bytes(),
        &[0xC3],
        &[b'x', 0xE2, 0x82],
        "€".as_bytes(),
        b"",
        &[0x80, 0x80, 0x80, 0x80],
    ];
    let vocabulary = ConstraintVocabulary::new(pieces.len() as u32, |i| {
        Ok(Piece {
            text: pieces[i as usize].to_vec(),
            stop: false,
        })
    })
    .unwrap();
    assert_eq!(
        vocabulary.partial_utf8(),
        [false, false, true, true, false, false, true],
        "llama.cpp's common_utf8_is_complete, negated"
    );
}
