//! Writes every stage of one transcription as raw little-endian files, named
//! like the C++ engine's dump harness, for bit-exact comparisons:
//! cargo run --release -p gufo-qwen3-asr --example stage_dump -- MODEL WAV OUT

use std::path::{Path, PathBuf};
use std::time::Instant;

use gufo_qwen3_asr::{Audio, Request, Transcriber, audio, prompt};

fn write<T: Copy, const N: usize>(dir: &Path, name: &str, values: &[T], bytes: fn(T) -> [u8; N]) {
    let data: Vec<u8> = values.iter().flat_map(|&v| bytes(v)).collect();
    if let Err(e) = std::fs::write(dir.join(name), data) {
        eprintln!("{name}: {e}");
    }
}

fn main() -> Result<(), String> {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let [model, wav, out] = args.as_slice() else {
        return Err("usage: stage_dump MODEL WAV OUT".into());
    };
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let bytes = std::fs::read(wav).map_err(|e| e.to_string())?;
    let waveform = audio::resample_mono_16k(&audio::decode_wav(&bytes)?)?;
    let features = audio::log_mel_features(&waveform)?;
    write(out, "waveform.f32", &waveform, f32::to_le_bytes);
    write(out, "features.f32", &features.values, f32::to_le_bytes);

    let begin = Instant::now();
    let transcriber = Transcriber::load(model, 1024)?;
    eprintln!("load: {:.1} ms", begin.elapsed().as_secs_f64() * 1e3);
    let tokens = audio::audio_embedding_token_count(features.frames)?;
    let ids = prompt::build_prompt(transcriber.tokenizer(), "", None, tokens)?;
    write(out, "prompt.u32", &ids, u32::to_le_bytes);
    let (encoder, decoder) = transcriber.trace(&features, &ids)?;
    write(
        out,
        "encoder_frontend.f32",
        &encoder.frontend,
        f32::to_le_bytes,
    );
    write(out, "encoder_layer0.f32", &encoder.layer0, f32::to_le_bytes);
    write(out, "encoder_final.f32", &encoder.output, f32::to_le_bytes);
    write(out, "decoder_layer0.f32", &decoder.layer0, f32::to_le_bytes);
    write(out, "logits.f32", &decoder.logits, f32::to_le_bytes);

    let request = Request {
        audio: Audio::Wav(&bytes),
        context: "",
        language: None,
        max_new_tokens: 256,
    };
    for run in 0..3 {
        let result = transcriber.transcribe(&request, &|| false, None)?;
        eprintln!("run {run}: {:?}", result.timings);
        if run == 0 {
            write(
                out,
                "generated.u32",
                &result.generated_ids,
                u32::to_le_bytes,
            );
            println!("{}\n{}", result.language, result.text);
        }
    }
    Ok(())
}
