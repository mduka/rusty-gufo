//! Prompt processing and greedy generation speed of the target, measured as
//! the extension's `gufo bench` driver does (`drivers.cpp`
//! `RunQwenBenchmarkImpl`, depth 0): a 4096-position context, synthetic
//! tokens `(i % 1000) + 100`, `pp<N>` as one prefill of `N` tokens into a
//! fresh sequence, `tg<N>` as `N` greedy decode steps after a 16-token
//! prompt, each repeated and reported as mean ± standard deviation in
//! tokens per second:
//! cargo run --profile gpu-test -p gufo-qwen35moe --example bench -- MODEL [PP] [TG] [REPS]

use std::time::Instant;

use gufo_qwen35moe::Engine;

fn tokens(count: usize) -> Vec<u32> {
    (0..count as u32).map(|i| i % 1000 + 100).collect()
}

fn stats(runs: &[f64]) -> (f64, f64) {
    let mean = runs.iter().sum::<f64>() / runs.len() as f64;
    let variance = if runs.len() > 1 {
        runs.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (runs.len() - 1) as f64
    } else {
        0.0
    };
    (mean, variance.sqrt())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).ok_or("usage: bench MODEL [PP] [TG] [REPS]")?;
    let number = |index: usize, default: usize| {
        args.get(index)
            .map_or(Ok(default), |value| value.parse::<usize>())
            .map_err(|e| e.to_string())
    };
    let (pp, tg, reps) = (number(2, 512)?, number(3, 128)?, number(4, 3)?);
    let mut engine = Engine::load(std::path::Path::new(model), 4096)?;
    // Warmup.
    engine.reset()?;
    engine.forward_prompt(&tokens(32), 0, true)?;

    let prompt = tokens(pp);
    engine.reset()?;
    engine.forward_prompt(&prompt, 0, true)?;
    let mut runs = Vec::new();
    for _ in 0..reps {
        engine.reset()?;
        let start = Instant::now();
        engine.forward_prompt(&prompt, 0, true)?;
        runs.push(pp as f64 / start.elapsed().as_secs_f64());
    }
    let (mean, sd) = stats(&runs);
    println!("pp{pp}: {mean:.2} ± {sd:.2} tokens/s");

    engine.reset()?;
    engine.forward_token(100, 0, true)?;
    let mut runs = Vec::new();
    for _ in 0..reps {
        engine.reset()?;
        let start_prompt = tokens(16);
        let mut token = engine
            .forward_prompt(&start_prompt, 0, true)?
            .ok_or("prefill returned no token")?;
        let start = Instant::now();
        for pos in 16..16 + tg as u32 {
            token = engine
                .forward_token(token, pos, true)?
                .ok_or("decode returned no token")?;
        }
        runs.push(tg as f64 / start.elapsed().as_secs_f64());
    }
    let (mean, sd) = stats(&runs);
    println!("tg{tg}: {mean:.2} ± {sd:.2} tokens/s");
    Ok(())
}
