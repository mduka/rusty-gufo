//! The extension's `draft_policy_test.cpp`: option names, the drafter
//! signal, calibration learning, the cost table and the calibrated chain
//! decision (and how it is reported), and sampled verification with a
//! sibling.

#![allow(clippy::unwrap_used)]

use gufo_core::sampling::{self, SamplerState, SamplingConfig, TokenId};
use gufo_gemma4::draft_policy::{
    CalibratedChain, DraftBatch, DraftCalibration, DraftCalibrationScope, DraftPolicy,
    draft_costs_at, draft_signal,
};
use gufo_gemma4::mtp_sampling::{MtpCandidateLogits, sample_mtp_proposal, verify_mtp_proposal};
use gufo_gemma4::proposal_sampler::without_top_p;
use gufo_gemma4::sibling_verification::verify_with_sibling;

fn candidates(spread: f32) -> MtpCandidateLogits {
    let mut c = MtpCandidateLogits {
        size: 64,
        ..MtpCandidateLogits::default()
    };
    for i in 0..c.size {
        c.ids[i] = i as TokenId;
        c.logits[i] = if i == 0 { spread } else { 0.0 };
    }
    c
}

#[test]
fn names() {
    assert_eq!(DraftPolicy::parse("").unwrap(), DraftPolicy::Calibrated);
    for policy in [
        DraftPolicy::Calibrated,
        DraftPolicy::Confidence,
        DraftPolicy::Fixed,
    ] {
        assert_eq!(DraftPolicy::parse(policy.name()).unwrap(), policy);
    }
    assert!(
        DraftPolicy::parse("adaptive").is_err(),
        "unknown policy accepted"
    );
    assert_eq!(
        DraftCalibrationScope::parse("").unwrap(),
        DraftCalibrationScope::Shared
    );
    assert_eq!(
        DraftCalibrationScope::parse("request").unwrap(),
        DraftCalibrationScope::Request
    );
}

#[test]
fn signal() {
    // A flat top-64 has entropy ln 64: 1 - sqrt(0.2 ln 64) = 0.088.
    let flat = draft_signal(&candidates(0.0));
    assert!((flat - 0.0880).abs() < 1e-3, "flat signal {flat}");
    let peaked = draft_signal(&candidates(30.0));
    assert!(peaked > 0.99 && peaked <= 1.0, "peaked signal {peaked}");
    assert!(
        draft_signal(&candidates(4.0)) > draft_signal(&candidates(2.0)),
        "signal does not grow with confidence"
    );
}

#[test]
fn calibration() {
    let mut calibration = DraftCalibration::default();
    assert!(
        (calibration.estimate(0.95) - 0.9375).abs() < 1e-6,
        "prior is not the bin midpoint"
    );
    for i in 0..1000 {
        calibration.observe(0.95, i % 4 != 0); // 75% accepted
    }
    let learned = calibration.estimate(0.9);
    assert!(
        (learned - 0.75).abs() < 0.01,
        "calibration did not learn: {learned}"
    );
    assert!(
        (calibration.estimate(0.3) - 0.3125).abs() < 1e-6,
        "another bin moved"
    );
    calibration.reset();
    assert!(
        (calibration.estimate(0.95) - 0.9375).abs() < 1e-6,
        "reset kept observations"
    );
}

#[test]
fn costs() {
    let shallow = draft_costs_at(0, false);
    assert!(
        shallow.verify[1] == 99.5 && shallow.verify[8] == 103.9,
        "d0 verification costs"
    );
    assert!(
        (shallow.draft[7] - 7.0 * 1.93).abs() < 1e-4,
        "d0 draft steps"
    );
    let mid = draft_costs_at(2048, false);
    assert!(
        (mid.verify[3] - 0.5 * (100.6 + 106.3)).abs() < 1e-3,
        "interpolation between depths"
    );
    let deep = draft_costs_at(131_072, false);
    let measured = draft_costs_at(65_536, false);
    for r in 1..deep.verify.len() {
        assert!(
            deep.verify[r] > measured.verify[r],
            "extrapolation past 64K"
        );
        assert!(
            deep.verify[r] >= deep.verify[r - 1],
            "rows are not monotone"
        );
    }
    // The 26B-A4B's own table: cheaper cycles whose rows cost relatively more.
    let moe = draft_costs_at(0, true);
    assert!(
        moe.verify[1] == 18.3 && moe.verify[8] == 31.8 && (moe.draft[1] - 1.25).abs() < 1e-4,
        "expert-model d0 costs"
    );
    assert!(
        moe.verify[8] / moe.verify[1] > shallow.verify[8] / shallow.verify[1],
        "expert-model rows are not relatively dearer"
    );
}

fn learned(accepted: bool) -> DraftCalibration {
    let mut calibration = DraftCalibration::default();
    for _ in 0..10_000 {
        let mut s = 0.05f32;
        while s < 1.0 {
            calibration.observe(s, accepted);
            s += 0.125;
        }
    }
    calibration
}

#[test]
fn chain() {
    let costs = draft_costs_at(0, false);
    let sure = learned(true);
    let hopeless = learned(false);
    {
        let mut chain = CalibratedChain::new(&sure, &costs, 1, 7, DraftBatch::default());
        for _ in 0..7 {
            assert!(chain.include(0.5), "a sure chain stopped");
        }
        assert!(!chain.include(0.5), "the cap was exceeded");
        assert!(!chain.last_decision().compared, "the cap compared costs");
    }
    {
        let mut chain = CalibratedChain::new(&hopeless, &costs, 1, 7, DraftBatch::default());
        assert!(chain.include(0.5), "the first draft was not verified");
        assert!(
            !chain.last_decision().compared,
            "the minimum compared costs"
        );
        assert!(!chain.include(0.5), "a hopeless chain continued");
        let stop = chain.last_decision();
        assert!(
            stop.compared && stop.gain < stop.cost,
            "the stop decision was not reported"
        );
    }
    {
        let mut chain = CalibratedChain::new(&hopeless, &costs, 3, 7, DraftBatch::default());
        for _ in 0..3 {
            assert!(chain.include(0.5), "the minimum was not verified");
        }
        assert!(!chain.include(0.5), "a hopeless chain passed its minimum");
    }
    {
        // Prior only: signal s is taken as the acceptance probability of its
        // bin midpoint. Draft two is worth it while survival * T >= E * dT
        // (at 64K keys, where a row costs enough to put the threshold between
        // bins).
        let deep = draft_costs_at(65_536, false);
        let prior = DraftCalibration::default();
        let mut chain = CalibratedChain::new(&prior, &deep, 1, 7, DraftBatch::default());
        assert!(chain.include(0.95), "first draft"); // survival 0.9375
        let current = deep.draft[2] + deep.verify[2];
        let next = deep.draft[3] + deep.verify[3];
        let needed = (1.0 + 0.9375) * (next - current) / current;
        // Bin midpoints 0.0625 (0.0-0.125) and 0.1875: survival 0.059 / 0.176.
        assert!(
            0.9375 * 0.0625 < needed && 0.9375 * 0.1875 > needed,
            "test premise: threshold between two bins"
        );
        let mut again = CalibratedChain::new(&prior, &deep, 1, 7, DraftBatch::default());
        assert!(again.include(0.95), "first draft");
        assert!(!again.include(0.05), "draft below the cost threshold kept");
        assert!(
            chain.include(0.15),
            "draft above the cost threshold dropped"
        );
    }
}

/// Every signal accepted with the given pattern.
fn accepted_with(rate: impl Fn(i32) -> bool) -> DraftCalibration {
    let mut calibration = DraftCalibration::default();
    for i in 0..10_000 {
        calibration.observe(0.9, rate(i));
    }
    calibration
}

/// In a batch the other sessions' rows make each extra row dearer and their
/// tokens raise the rate a draft must match: a chain that goes on alone
/// stops beside seven sessions on the expert model's costs.
#[test]
fn batch() {
    let costs = draft_costs_at(0, true);
    // Every signal accepted three times in five.
    let likely = accepted_with(|i| i % 5 < 3);
    let mut alone = CalibratedChain::new(&likely, &costs, 1, 7, DraftBatch::default());
    assert!(
        alone.include(0.9) && alone.include(0.9),
        "a likely chain stopped alone"
    );
    assert!(
        (alone.expected() - (1.0 + 0.6 + 0.36)).abs() < 1e-2
            && (alone.draft_ms() - costs.draft[2]).abs() < 1e-5,
        "chain expectation and drafter time"
    );
    // Beside three sessions (a C4 batch) the second draft no longer pays.
    let three = DraftBatch {
        rows: 3,
        expected: 3.0,
        ..DraftBatch::default()
    };
    let mut batched = CalibratedChain::new(&likely, &costs, 1, 7, three);
    assert!(batched.include(0.9), "a likely first draft was dropped");
    assert!(!batched.include(0.9), "a batched chain ignored its peers");
    // Beside seven sessions a doubtful first draft is not verified at all.
    let doubtful = accepted_with(|i| i % 10 == 0);
    let mut lone = CalibratedChain::new(&doubtful, &costs, 1, 7, DraftBatch::default());
    assert!(lone.include(0.9), "a lone cycle skipped its first draft");
    let seven = DraftBatch {
        rows: 7,
        expected: 7.0,
        ..DraftBatch::default()
    };
    let mut crowded = CalibratedChain::new(&doubtful, &costs, 1, 7, seven);
    assert!(
        !crowded.include(0.9),
        "a doubtful batched draft was verified"
    );
    // Beside ten rows even a certain draft cannot pay for its row; beside one
    // it can, and a lone cycle always drafts.
    let ten = DraftBatch {
        rows: 10,
        expected: 10.0,
        ..DraftBatch::default()
    };
    assert!(
        !CalibratedChain::new(&doubtful, &costs, 1, 1, ten).first_draft_can_pay(),
        "an unpayable draft would run the drafter"
    );
    let one = DraftBatch {
        rows: 1,
        expected: 1.0,
        ..DraftBatch::default()
    };
    assert!(
        CalibratedChain::new(&doubtful, &costs, 1, 7, one).first_draft_can_pay()
            && lone.first_draft_can_pay(),
        "a payable draft was skipped"
    );
}

/// A sibling adds a token where the chain reaches its draft, the draft is
/// rejected and the sibling accepted; it pays for its row like a draft.
#[test]
fn siblings() {
    let costs = draft_costs_at(0, false);
    let likely = accepted_with(|i| i % 5 < 3);
    let mut chain = CalibratedChain::new(&likely, &costs, 1, 7, DraftBatch::default());
    assert!(chain.include(0.9) && chain.include(0.9), "a likely chain");
    let expected = chain.expected();
    // Rows: pending and two drafts. The first sibling's gain is
    // reach 1 x rejection 0.4 x hit 0.5.
    assert!(
        chain.include_sibling(0, 3, 0.5),
        "a paying sibling was dropped"
    );
    assert!(
        (chain.expected() - (expected + 0.2)).abs() < 1e-3,
        "sibling expectation"
    );
    // A sibling that is never accepted, or past the chain, does not pay.
    assert!(
        !chain.include_sibling(1, 4, 0.0),
        "a hopeless sibling was kept"
    );
    assert!(
        !chain.include_sibling(2, 4, 1.0),
        "a sibling beyond the chain"
    );
    // Beside three sessions on the expert model a doubtful sibling's row
    // costs more than it adds.
    let moe = draft_costs_at(0, true);
    let three = DraftBatch {
        rows: 3,
        expected: 3.0,
        ..DraftBatch::default()
    };
    let mut crowded = CalibratedChain::new(&likely, &moe, 1, 7, three);
    assert!(crowded.include(0.9), "first draft");
    assert!(
        !crowded.include_sibling(0, 2, 0.05),
        "a doubtful batched sibling was verified"
    );
}

/// Sampled verification with a sibling (recursive rejection sampling) keeps
/// the target's token frequencies, for proposals drawn with the target's own
/// top-p and without it (Gemma 4 drafts skip top-p): the sibling is a draw
/// from q without the draft, as the engine draws it.
#[test]
fn sibling_frequencies() {
    let logits = [0.3f32, -0.2, 1.1, 0.7, -1.0, 0.9];
    let mut c = MtpCandidateLogits {
        size: 5,
        ..MtpCandidateLogits::default()
    };
    c.ids[..5].copy_from_slice(&[3, 1, 5, 2, 0]);
    c.logits[..5].copy_from_slice(&[1.2, 0.1, 0.8, 0.9, -0.7]);
    const TRIALS: u32 = 120_000;
    for top_p in [0.6f32, 0.9] {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 5,
            top_p,
            seed: 53,
            ..SamplingConfig::default()
        };
        let mut sampler = SamplerState::new(config, &[]).unwrap();
        let p = sampler.distribution(&logits).unwrap();
        for unfiltered in [false, true] {
            let proposals = if unfiltered {
                without_top_p(&sampler).unwrap()
            } else {
                sampler.clone()
            };
            let mut counts = [0u32; 6];
            let mut sibling_rng = 7u64;
            let mut siblings = 0;
            for _ in 0..TRIALS {
                let mut draft_rng = sampling::next_random(sampler.mutable_rng_state());
                let q = sample_mtp_proposal(&c, &proposals, &mut draft_rng).unwrap();
                let rest = 1.0 - f64::from(q.probability);
                if rest <= 0.0 {
                    let token = verify_mtp_proposal(&logits, &q, &mut sampler)
                        .unwrap()
                        .token;
                    counts[token as usize] += 1;
                    continue;
                }
                let draw = sampling::uniform(&mut sibling_rng) * rest;
                let mut cumulative = 0.0f64;
                let mut sibling = q.ids[0];
                for i in 0..q.size {
                    if q.ids[i] == q.token || q.probabilities[i] <= 0.0 {
                        continue;
                    }
                    sibling = q.ids[i];
                    cumulative += f64::from(q.probabilities[i]);
                    if draw < cumulative {
                        break;
                    }
                }
                siblings += 1;
                let token = verify_with_sibling(&p, &q, sibling, &mut sampler)
                    .unwrap()
                    .token;
                counts[token as usize] += 1;
            }
            assert!(siblings > TRIALS / 4, "too few sibling cycles to check");
            for (token, &count) in counts.iter().enumerate() {
                let frequency = f64::from(count) / f64::from(TRIALS);
                assert!(
                    (frequency - p.probability(token as TokenId)).abs() < 0.008,
                    "sibling verification distorts target frequencies"
                );
            }
        }
    }
}
