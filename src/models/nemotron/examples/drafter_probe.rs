//! Compares the LiLiCorr drafter with llama.cpp on llama.cpp's own trunk
//! features (`tests/drafter_probe.cpp` of the C++ extension): reads a dump
//! of `tools/llama_parity/llama_dflash.cpp`, injects the same features,
//! drafts at the same anchors and reports, per drafted slot, top-1 agreement
//! and KL of the drafter logits, candidate agreement, the largest lattice
//! score difference, and the greedy walks.
//!
//! cargo run --profile gpu-test -p gufo-nemotron --example drafter_probe -- \
//!   --model MODEL.gguf --drafter DRAFT.gguf --dump llama_dflash.bin
//!
//! The C++ probe drafts through a trunk session; this one uploads the trunk
//! the same way (all of it, as `Model::Load` does) and drives a draft
//! context on its own stream, which is what the session's context does.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::process::ExitCode;

use gufo_hip::blas::Blas;
use gufo_hip::{MappedFile, Stream};
use gufo_nemotron::device_model::DeviceModel;
use gufo_nemotron::drafter::{DraftContext, DraftLattice, Drafter};
use gufo_nemotron::weights::{GgufShard, ModelWeights};
use gufo_nemotron_kernels as k;

/// The session's draft context budget (`kPrefillChunk`).
const MAX_ROWS: usize = 2048;

fn read_f32(input: &mut impl Read, count: usize) -> Option<Vec<f32>> {
    let mut bytes = vec![0u8; count * 4];
    input.read_exact(&mut bytes).ok()?;
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|raw| f32::from_le_bytes(*raw))
            .collect(),
    )
}

fn read_u32(input: &mut impl Read, count: usize) -> Option<Vec<u32>> {
    let mut bytes = vec![0u8; count * 4];
    input.read_exact(&mut bytes).ok()?;
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|raw| u32::from_le_bytes(*raw))
            .collect(),
    )
}

fn kl(p_logits: &[f32], q_logits: &[f32]) -> f64 {
    let lse = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f64 = v.iter().map(|&x| (f64::from(x) - f64::from(m)).exp()).sum();
        f64::from(m) + z.ln()
    };
    let lp = lse(p_logits);
    let lq = lse(q_logits);
    p_logits
        .iter()
        .zip(q_logits)
        .map(|(&p, &q)| {
            let a = f64::from(p) - lp;
            a.exp() * (a - (f64::from(q) - lq))
        })
        .sum()
}

/// `std::max_element`: the first of equal maxima.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (mut model_path, mut drafter_path, mut dump) =
        (String::new(), String::new(), String::new());
    let mut i = 1;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--model" => model_path = args[i + 1].clone(),
            "--drafter" => drafter_path = args[i + 1].clone(),
            "--dump" => dump = args[i + 1].clone(),
            _ => {}
        }
        i += 2;
    }
    if model_path.is_empty() || drafter_path.is_empty() || dump.is_empty() {
        eprintln!(
            "usage: {} --model MODEL.gguf --drafter DRAFT.gguf --dump llama_dflash.bin",
            args[0]
        );
        return ExitCode::from(2);
    }
    match run(&model_path, &drafter_path, &dump) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run(model_path: &str, drafter_path: &str, dump: &str) -> Result<(), String> {
    let mut input = BufReader::new(File::open(dump).map_err(|e| format!("{dump}: {e}"))?);
    let mut magic = [0u8; 4];
    let header = input
        .read_exact(&mut magic)
        .ok()
        .and_then(|()| read_u32(&mut input, 7))
        .filter(|_| &magic == b"DFL1")
        .ok_or("not a llama_dflash dump")?;
    let (n_tokens, n_feat, n_anchor, block, vocab, n_embd) = (
        header[0] as usize,
        (header[1] * header[2]) as usize,
        header[3],
        header[4] as usize,
        header[5] as usize,
        header[6] as usize,
    );
    let features = read_f32(&mut input, n_tokens * n_feat).ok_or("truncated dump")?;
    read_f32(&mut input, n_tokens * n_embd).ok_or("truncated dump")?;

    let load = |e: String| format!("load: {e}");
    let file = MappedFile::open(model_path).map_err(|e| load(format!("open {model_path}: {e}")))?;
    let shard = GgufShard::parse(file.bytes()).map_err(load)?;
    let weights = ModelWeights::bind(&[shard]).map_err(load)?;
    let stream = Stream::new().map_err(|e| load(e.to_string()))?;
    let trunk =
        DeviceModel::upload(&weights, std::slice::from_ref(&file), &stream).map_err(load)?;
    let drafter = Drafter::load(Path::new(drafter_path), &weights.config).map_err(load)?;
    let dc = drafter.config().clone();
    if dc.feature_size() as usize != n_feat
        || dc.block_size as usize != block
        || weights.config.vocab_size as usize != vocab
    {
        return Err("dump does not match the models".into());
    }
    // The session's stream and handle: hipBLAS in its default atomics mode
    // and the quantized products bound to the stream.
    let blas = Blas::with_default_atomics(&stream).map_err(|e| format!("session: {e}"))?;
    if !matches!(k::mmq_bind_stream(&stream), Ok(gufo_hip::Launch::Launched)) {
        return Err("session: could not bind the quantized kernels".into());
    }
    let mut draft = DraftContext::create(&stream, &blas, &drafter, &trunk, MAX_ROWS)
        .map_err(|e| format!("session: {e}"))?;
    let slots = dc.slots() as usize;
    let top_k = dc.lc_top_k as usize;
    let mut logits = vec![0f32; slots * vocab];
    let mut kls = Vec::new();
    let (mut top1, mut rows, mut cand_same, mut walk_same) = (0usize, 0usize, 0usize, 0usize);
    let (mut slot_lists, mut slot_sets) = (0usize, 0usize);
    let mut score_diffs = Vec::new();
    let mut max_score_diff = 0f64;
    let mut injected = 0usize;
    for a in 0..n_anchor {
        let meta = read_u32(&mut input, 2).ok_or_else(|| format!("truncated anchor {a}"))?;
        let llama_logits =
            read_f32(&mut input, block * vocab).ok_or_else(|| format!("truncated anchor {a}"))?;
        let llama_lattice =
            read_f32(&mut input, block * n_embd).ok_or_else(|| format!("truncated anchor {a}"))?;
        let anchor = meta[0] as usize;
        if anchor > injected {
            let rows_in = &features[injected * n_feat..anchor * n_feat];
            let mut offset = 0;
            while offset < rows_in.len() {
                let len = (rows_in.len() - offset).min(512 * n_feat);
                draft
                    .inject_host(
                        &stream,
                        &blas,
                        &rows_in[offset..offset + len],
                        (injected + offset / n_feat) as u32,
                    )
                    .map_err(|e| format!("inject: {e}"))?;
                offset += len;
            }
            injected = anchor;
        }
        let lattice = draft
            .draft(
                &stream,
                &blas,
                meta[1] as i32,
                anchor as u32,
                Some(&mut logits),
            )
            .map_err(|e| format!("draft: {e}"))?;
        // llama.cpp's lattice row s + 1 holds slot s: candidates, then scores.
        let mut reference = DraftLattice {
            slots: slots as u32,
            top_k: top_k as u32,
            candidates: Vec::with_capacity(slots * top_k),
            scores: vec![0.0; lattice.scores.len()],
        };
        let mut same_candidates = true;
        for s in 0..slots {
            let row = &llama_lattice[(s + 1) * n_embd..(s + 2) * n_embd];
            for (j, &id) in row[..top_k].iter().enumerate() {
                reference.candidates.push(id as i32);
                same_candidates &= lattice.candidates[s * top_k + j] == id as i32;
            }
            let mut mine = lattice.candidates[s * top_k..(s + 1) * top_k].to_vec();
            let mut theirs = reference.candidates[s * top_k..(s + 1) * top_k].to_vec();
            slot_lists += usize::from(mine == theirs);
            mine.sort_unstable();
            theirs.sort_unstable();
            slot_sets += usize::from(mine == theirs);
            let (count, base) = if s == 0 {
                (top_k, 0)
            } else {
                (top_k * top_k, top_k + (s - 1) * top_k * top_k)
            };
            reference.scores[base..base + count].copy_from_slice(&row[top_k..top_k + count]);
            let ours = &logits[s * vocab..(s + 1) * vocab];
            let theirs = &llama_logits[(s + 1) * vocab..(s + 2) * vocab];
            kls.push(kl(theirs, ours));
            top1 += usize::from(argmax(theirs) == argmax(ours));
            rows += 1;
        }
        // Scores of slots whose own and previous candidate lists agree.
        for s in 0..slots {
            let same = (s.saturating_sub(1)..=s).all(|q| {
                lattice.candidates[q * top_k..(q + 1) * top_k]
                    == reference.candidates[q * top_k..(q + 1) * top_k]
            });
            if !same {
                continue;
            }
            let (count, base) = if s == 0 {
                (top_k, 0)
            } else {
                (top_k * top_k, top_k + (s - 1) * top_k * top_k)
            };
            for j in base..base + count {
                score_diffs.push(f64::from((lattice.scores[j] - reference.scores[j]).abs()));
            }
        }
        if same_candidates {
            cand_same += 1;
            for (a, b) in lattice.scores.iter().zip(&reference.scores) {
                max_score_diff = max_score_diff.max(f64::from((a - b).abs()));
            }
        }
        let ours = lattice.walk(slots as u32);
        let theirs = reference.walk(slots as u32);
        walk_same += usize::from(ours == theirs);
        let walk: String = ours
            .iter()
            .zip(&theirs)
            .map(|(o, t)| format!(" {o}{}", if o == t { "" } else { "!" }))
            .collect();
        println!(
            "anchor {anchor:4}:{walk}{}",
            if same_candidates {
                ""
            } else {
                "  (candidates differ)"
            }
        );
    }
    kls.sort_by(f64::total_cmp);
    score_diffs.sort_by(f64::total_cmp);
    if let Some(&max) = score_diffs.last() {
        println!(
            "lattice scores on agreeing slots: {}, |diff| median {:.3e} p90 {:.3e} p99 {:.3e} max {:.3e}",
            score_diffs.len(),
            score_diffs[score_diffs.len() / 2],
            score_diffs[score_diffs.len() * 9 / 10],
            score_diffs[score_diffs.len() * 99 / 100],
            max
        );
    }
    if let Some(&max) = kls.last() {
        println!(
            "slots {rows}: top-1 {top1}/{rows}, KL median {:.2e} p90 {:.2e} max {:.2e}\n\
             slot candidates: same order {slot_lists}, same set {slot_sets} of {rows}\n\
             anchors {n_anchor}: same candidates {cand_same}, same walk {walk_same}, max score diff {max_score_diff:.3}",
            kls[kls.len() / 2],
            kls[kls.len() * 9 / 10],
            max
        );
    }
    drop(draft);
    let _ = k::mmq_release_stream(&stream);
    Ok(())
}
