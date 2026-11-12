//! Batched decode timing on the workload of the C++ probe's `batched` mode
//! (`tests/reference/q35moe_probe.cpp` `RunBatched`): four
//! sessions of one model (haiku, and the synthetic bodies 7, 100 and 341),
//! each alone a prefill and twelve `forward_token` steps on session 0, then
//! the first two, three and four together, each prefilled, and twelve
//! `forward_token_batch` steps. Every trace entry downloads the session's
//! logits and digests them (FNV-1a and the top five), as the probe records
//! them. The mode runs `ROUNDS` times on the same sessions; the first round
//! includes the warm-up, as the probe's first run of a mode includes
//! creating its extra executors. Compare a later round with the probe's
//! second `batched` timing of `--modes batched,batched`:
//! cargo run --profile gpu-test -p gufo-qwen35moe --example batched -- MODEL [ROUNDS]

use std::time::{Duration, Instant};

use gufo_qwen35moe::{Model, Session};

const HAIKU: [u32; 20] = [
    248045, 846, 198, 7734, 264, 6185, 36974, 883, 279, 9117, 13, 248046, 198, 248045, 74455, 198,
    248068, 271, 248069, 271,
];
const HEAD: [u32; 3] = [248045, 846, 198];
const TAIL: [u32; 9] = [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271];
const STEPS: usize = 12;

fn synthetic(body: u32) -> Vec<u32> {
    HEAD.iter()
        .copied()
        .chain((0..body).map(|i| 1000 + (i * 7919 + 13) % 50_000))
        .chain(TAIL)
        .collect()
}

/// The probe's `Trace::Record`: the digest and the top five of a row.
fn record(model: &Model, st: &Session, out: &mut Vec<String>) -> Result<(), String> {
    let logits = model.logits(st)?;
    let hash = logits
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    let before = |a: &u32, b: &u32| {
        let (x, y) = (logits[*a as usize], logits[*b as usize]);
        y.partial_cmp(&x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    order.select_nth_unstable_by(4, before);
    order[..5].sort_by(before);
    let top: Vec<String> = order[..5]
        .iter()
        .map(|&id| format!("[{id}, {:e}]", logits[id as usize]))
        .collect();
    out.push(format!("{hash:016x} [{}]", top.join(", ")));
    Ok(())
}

/// One run of the mode; the time its batched steps took.
fn mode(
    model: &mut Model,
    sessions: &mut [Session],
    prompts: &[Vec<u32>],
) -> Result<Duration, String> {
    let mut trace = Vec::new();
    for prompt in prompts {
        let st = &mut sessions[0];
        model.reset(st)?;
        let mut next = model
            .forward_prompt(st, prompt, 0, true)?
            .ok_or("prefill returned no token")?;
        for step in 0..=STEPS {
            record(model, st, &mut trace)?;
            if step < STEPS {
                next = model
                    .forward_token(st, next, (prompt.len() + step) as u32, true)?
                    .ok_or("decode returned no token")?;
            }
        }
    }
    let mut batched = Duration::ZERO;
    for width in 2..=4 {
        let mut next = Vec::new();
        let mut positions = Vec::new();
        for (st, prompt) in sessions[..width].iter_mut().zip(prompts) {
            model.reset(st)?;
            next.push(
                model
                    .forward_prompt(st, prompt, 0, true)?
                    .ok_or("prefill returned no token")?,
            );
            positions.push(prompt.len() as u32);
            record(model, st, &mut trace)?;
        }
        for _ in 0..STEPS {
            let start = Instant::now();
            let mut items: Vec<(&mut Session, u32, u32)> = sessions[..width]
                .iter_mut()
                .zip(next.iter().zip(&positions))
                .map(|(st, (&token, &position))| (st, token, position))
                .collect();
            let out = model.forward_token_batch(&mut items)?;
            drop(items);
            batched += start.elapsed();
            for (index, st) in sessions[..width].iter().enumerate() {
                next[index] = out[index];
                positions[index] += 1;
                record(model, st, &mut trace)?;
            }
        }
    }
    std::hint::black_box(trace);
    Ok(batched)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: batched MODEL [ROUNDS]")?;
    let rounds = args
        .get(2)
        .map_or(Ok(3), |value| value.parse::<usize>())
        .map_err(|e| e.to_string())?;
    let mut model = Model::load(std::path::Path::new(path), 6144)?;
    let mut sessions = (0..4)
        .map(|_| model.new_session())
        .collect::<Result<Vec<_>, _>>()?;
    let prompts = [HAIKU.to_vec(), synthetic(7), synthetic(100), synthetic(341)];
    for round in 0..rounds {
        let start = Instant::now();
        let batched = mode(&mut model, &mut sessions, &prompts)?;
        println!(
            "round {round}: mode {:.6} s, of it 36 batched steps {:.6} s",
            start.elapsed().as_secs_f64(),
            batched.as_secs_f64()
        );
    }
    Ok(())
}
