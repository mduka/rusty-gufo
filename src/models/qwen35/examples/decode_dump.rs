//! Token IDs and logits of the Rust engine in the file layout of
//! `tests/reference/cpp_decode_dump.cpp`, for bit-exact comparison (`decode`: the prompt
//! token by token; `prefill`: the prompt as one batch):
//! cargo run --release -p gufo-qwen35 --example decode_dump -- MODEL PROMPT OUTDIR decode|prefill N [MAX_CONTEXT]

use std::path::PathBuf;

use gufo_qwen35::Engine;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, prompt, out, mode, steps, rest @ ..] = args.as_slice() else {
        return Err("usage: decode_dump MODEL PROMPT OUTDIR decode|prefill N [MAX_CONTEXT]".into());
    };
    let steps: usize = steps.parse().map_err(|_| "N must be a count")?;
    let max_context: u32 = match rest.first() {
        Some(v) => v.parse().map_err(|_| "MAX_CONTEXT must be a count")?,
        None => 4096,
    };
    let bytes = std::fs::read(prompt).map_err(|e| format!("{prompt}: {e}"))?;
    let prompt: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| u32::from_le_bytes(b))
        .collect();
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let write = |name: String, bytes: Vec<u8>| {
        std::fs::write(out.join(&name), bytes).map_err(|e| format!("{name}: {e}"))
    };

    let mut engine = Engine::load(model.as_ref(), max_context)?;
    let mut next = 0;
    let mut pos = 0u32;
    match mode.as_str() {
        "prefill" => {
            next = engine.forward_prompt(&prompt, 0, true)?.ok_or("no token")?;
            pos = prompt.len() as u32;
        }
        "decode" => {
            for (i, &token) in prompt.iter().enumerate() {
                let last = i + 1 == prompt.len();
                if let Some(token) = engine.forward_token(token, pos, last)? {
                    next = token;
                }
                pos += 1;
            }
        }
        _ => return Err("mode must be decode or prefill".into()),
    }
    let mut ids = Vec::new();
    for step in 0..=steps {
        ids.push(next);
        let logits = engine.logits()?;
        write(
            format!("logits_{step}.bin"),
            logits.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )?;
        if step == steps {
            break;
        }
        next = engine.forward_token(next, pos, true)?.ok_or("no token")?;
        pos += 1;
    }
    write(
        "ids.bin".into(),
        ids.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )?;
    let text: Vec<String> = ids.iter().map(u32::to_string).collect();
    println!("{}", text.join(" "));
    Ok(())
}
