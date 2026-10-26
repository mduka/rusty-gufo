//! Compact sampled verification on the device against the full rows: the
//! candidate union holds the exact unpenalized top 64 (an independent sort)
//! and every penalized token with its original logit bits, and the target
//! distribution, acceptance, residual correction and RNG equal those of the
//! complete row, over ragged vocabularies, widths 1 to 7, penalties, ties,
//! signed zeros and nonfinite logits (`verification_sampling_test.cpp`):
//! cargo test --profile gpu-test -p gufo-qwen38-flash-next --features gpu --test verification

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_core::sampling::{LogitCandidate, SamplerState, SamplingConfig, TokenId};
use gufo_hip::Stream;
use gufo_qwen38_flash_next::mtp::{MTP_CANDIDATES, MtpCandidates};
use gufo_qwen38_flash_next::policy::{
    sample_proposal, verify_proposal, verify_proposal_candidates,
};
use gufo_qwen38_flash_next_kernels as k;

/// The full row, the row rebuilt from the candidates and the candidates give
/// the same distribution bit for bit, and verification draws alike.
fn compare(full: &[f32], compact: &[f32], candidates: &[LogitCandidate], state: &SamplerState) {
    let a = state.distribution(full).unwrap();
    let b = state.distribution(compact).unwrap();
    let c = state.distribution_from_candidates(candidates).unwrap();
    assert_eq!(
        a.entries().len(),
        b.entries().len(),
        "target support differs"
    );
    assert_eq!(
        a.entries().len(),
        c.entries().len(),
        "compact support differs"
    );
    for ((a, b), c) in a.entries().iter().zip(b.entries()).zip(c.entries()) {
        assert!(
            a.token == b.token
                && a.token == c.token
                && a.value.to_bits() == b.value.to_bits()
                && a.value.to_bits() == c.value.to_bits(),
            "target probabilities are not bit exact"
        );
    }
    // Accepted and rejected point-mass and neural proposals.
    let mut draft = MtpCandidates {
        size: MTP_CANDIDATES.min(full.len()),
        ..MtpCandidates::default()
    };
    for i in 0..draft.size {
        draft.ids[i] = ((i * 37) % full.len()) as u32;
        draft.logits[i] = ((i % 13) as i32 - 6) as f32 / 10.0;
    }
    let mut proposal_rng = 987u64;
    let best = a.best_token().unwrap();
    for seed in 0..16u64 {
        let (mut left, mut right) = (state.clone(), state.clone());
        left.set_rng_state(seed);
        right.set_rng_state(seed);
        let mut proposal = sample_proposal(&draft, state, &mut proposal_rng).unwrap();
        if seed % 2 == 0 {
            proposal.size = 1;
            proposal.ids[0] = if seed % 4 != 0 {
                best
            } else {
                full.len() as TokenId - 1
            };
            proposal.probabilities[0] = 1.0;
            proposal.token = proposal.ids[0];
            proposal.probability = 1.0;
        }
        let x = verify_proposal(full, &proposal, &mut left).unwrap();
        let y = verify_proposal_candidates(candidates, full.len(), &proposal, &mut right).unwrap();
        assert!(
            x == y && left.rng_state() == right.rng_state(),
            "verification or correction RNG differs"
        );
    }
}

fn run(stream: &Stream, vocab: u32, rows: usize, mode: u32) {
    let v = vocab as usize;
    let workspace = k::mtp_candidate_workspace(vocab) as usize;
    let mut config = SamplingConfig {
        temperature: 0.8,
        top_k: match mode % 3 {
            0 => 1,
            1 => 20,
            _ => 64,
        },
        top_p: 0.95,
        min_p: if !mode.is_multiple_of(2) { 0.1 } else { 0.0 },
        min_keep: if mode == 3 { 64 } else { 1 },
        seed: 83,
        repeat_penalty: if !mode.is_multiple_of(2) { 0.7 } else { 1.3 },
        repeat_last_n: 512,
        frequency_penalty: if !mode.is_multiple_of(2) { -0.4 } else { 0.3 },
        presence_penalty: if !mode.is_multiple_of(2) { -1.5 } else { 1.0 },
        ..SamplingConfig::default()
    };
    if mode == 6 {
        config.repeat_penalty = 1.0;
        config.frequency_penalty = 0.0;
        config.presence_penalty = 0.0;
    }
    let history: Vec<TokenId> = (0..vocab.min(300)).map(|i| (i * 17) % vocab).collect();
    let mut state = SamplerState::new(config, &history).unwrap();
    state.accept_tokens(&history).unwrap();
    let mut logits = vec![0f32; rows * v];
    let (mut states, mut triples, mut offsets) = (Vec::new(), Vec::new(), vec![0u32]);
    for row in 0..rows {
        states.push(state.clone());
        for p in state.penalties() {
            triples.extend([p.token, p.generated_count, p.repeated]);
        }
        offsets.push((triples.len() / 3) as u32);
        let values = &mut logits[row * v..(row + 1) * v];
        for (i, value) in values.iter_mut().enumerate() {
            let mixed = (i as u32).wrapping_mul(7919).wrapping_add(row as u32 * 71) % 65521;
            *value = (mixed as i32 - 32768) as f32 / 4096.0;
        }
        match mode {
            3 => {
                for (i, value) in values.iter_mut().enumerate() {
                    *value = if i % 2 != 0 { 0.0 } else { -0.0 };
                }
            }
            4 => {
                values.fill(f32::NEG_INFINITY);
                for (i, value) in values.iter_mut().take(17).enumerate() {
                    *value = i as f32;
                }
            }
            5 => {
                values[0] = f32::INFINITY;
                values[v - 1] = f32::NAN;
            }
            _ => {}
        }
        state.accept((row as TokenId * 13 + 3) % vocab).unwrap();
    }
    let penalized = triples.len() / 3;
    let capacity = penalized.max(1);
    triples.resize(3 * capacity, 0);
    let count = rows * MTP_CANDIDATES + penalized;
    let mut device_logits = stream.alloc::<f32>(logits.len()).unwrap();
    stream
        .upload(&logits, device_logits.as_mut_slice())
        .unwrap();
    let mut penalties = stream.alloc::<u32>(triples.len()).unwrap();
    stream.upload(&triples, penalties.as_mut_slice()).unwrap();
    // Row r ends at offsets[r + 1]; the ends past the last row repeat it.
    let last = *offsets.last().unwrap();
    let ends: [u32; 7] = std::array::from_fn(|r| offsets.get(r + 1).copied().unwrap_or(last));
    let mut ids = stream.alloc::<u32>(rows * workspace).unwrap();
    let mut scratch = stream.alloc::<u32>(rows * workspace).unwrap();
    let mut out = stream
        .alloc::<u8>(8 * (rows * MTP_CANDIDATES + capacity))
        .unwrap();
    let launched = k::verification_candidates(
        stream,
        device_logits.as_slice(),
        penalties.as_slice(),
        ends[0],
        ends[1],
        ends[2],
        ends[3],
        ends[4],
        ends[5],
        ends[6],
        ids.as_mut_slice(),
        scratch.as_mut_slice(),
        out.as_mut_slice(),
        rows as u32,
        vocab,
        capacity as u32,
        workspace as u32,
    )
    .unwrap();
    assert_eq!(launched, gufo_hip::Launch::Launched);
    let mut bytes = vec![0u8; 8 * count];
    stream
        .download(out.slice(0..8 * count), &mut bytes)
        .unwrap();
    let output: Vec<LogitCandidate> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| LogitCandidate {
            token: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            value: f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        })
        .collect();
    for row in 0..rows {
        let full = &logits[row * v..(row + 1) * v];
        let begin = row * MTP_CANDIDATES + offsets[row] as usize;
        let end = (row + 1) * MTP_CANDIDATES + offsets[row + 1] as usize;
        let result = &output[begin..end];
        // An independent full sort, without the production selector.
        let mut penalized = vec![false; v];
        for p in states[row].penalties() {
            penalized[p.token as usize] = true;
        }
        let mut expected: Vec<u32> = (0..vocab)
            .filter(|&id| full[id as usize].is_finite() && !penalized[id as usize])
            .collect();
        expected.sort_by(|&a, &b| {
            full[b as usize]
                .partial_cmp(&full[a as usize])
                .unwrap()
                .then(a.cmp(&b))
        });
        expected.truncate(MTP_CANDIDATES);
        for (i, candidate) in result.iter().take(MTP_CANDIDATES).enumerate() {
            assert_eq!(
                candidate.token,
                expected.get(i).copied().unwrap_or(u32::MAX),
                "unpenalized top-k differs from an independent sort: \
                 vocab={vocab} rows={rows} mode={mode} row={row} rank={i}"
            );
        }
        let mut reconstructed = vec![f32::NEG_INFINITY; v];
        for candidate in result.iter().filter(|c| c.token < vocab) {
            assert_eq!(
                candidate.value.to_bits(),
                full[candidate.token as usize].to_bits(),
                "candidate logit bits changed"
            );
            reconstructed[candidate.token as usize] = candidate.value;
        }
        compare(full, &reconstructed, result, &states[row]);
    }
}

#[test]
fn compact_candidates_verify_like_full_rows() {
    let stream = Stream::new().unwrap();
    for vocab in [67, 1023, 1025, 248_320] {
        for rows in 1..=7 {
            for mode in 0..7 {
                run(&stream, vocab, rows, mode);
            }
        }
    }
}
