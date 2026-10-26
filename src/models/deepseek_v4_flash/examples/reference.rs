//! The reference measurements of `tools/ds4/reference.py`, a port of
//! `tests/models/deepseek_v4_flash/official_reference.cpp` with its output
//! files. `--official OUT.json` scores the 105 official 0731 continuations
//! (`tests/models/deepseek_v4_flash/fixtures/official-0731.json`) by forced
//! decoding: per step the target, the greedy token and the target's log
//! probability. `--reference-frontiers OUT-DIR [--prefill-only]` writes the
//! full logits before (and after 128 forced tokens) of chat prompts of 16,
//! 4K, 8K, 12K and 16K tokens prefilled in 2K and 4K calls, with their
//! prompts, tokens and a manifest for the upstream run. The model is
//! `GUFO_DEEPSEEK_V4_FLASH_MODEL`:
//! cargo run --release -p gufo-deepseek-v4-flash --example reference -- --official OUT.json

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use gufo_core::json::{self, Value};
use gufo_deepseek_v4_flash::chat_template::{ChatMessage, ChatTemplateOptions, render_chat};
use gufo_deepseek_v4_flash::engine::Engine;
use gufo_deepseek_v4_flash::session::Session;
use sha2::{Digest, Sha256};

const OFFICIAL_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../tests/models/deepseek_v4_flash/fixtures/official-0731.json"
);

type Result<T> = std::result::Result<T, String>;

fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// A double as `std::ostream` writes it with `std::setprecision(17)`
/// (`%.17g`).
fn number(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let text = format!("{value:.16e}");
    let (mantissa, power) = text.split_once('e').unwrap_or((&text, "0"));
    let exponent: i32 = power.parse().unwrap_or(0);
    if !(-4..17).contains(&exponent) {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        return format!(
            "{mantissa}e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    }
    let decimals = (16 - exponent).max(0) as usize;
    let text = format!("{value:.decimals$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// `WriteScoreJson`: every scoring digit, where the HTTP serializer keeps
/// six.
fn write_score(value: &Value, out: &mut String) -> Result<()> {
    match value {
        Value::Number(v) => {
            require(v.is_finite(), "non-finite score output")?;
            out.push_str(&number(*v));
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                write_score(item, out)?;
            }
            out.push(']');
        }
        Value::Object(members) => {
            out.push('{');
            for (i, (key, item)) in members.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                out.push_str(
                    &Value::String(key.clone())
                        .dump()
                        .map_err(|e| e.to_string())?,
                );
                out.push(':');
                write_score(item, out)?;
            }
            out.push('}');
        }
        other => out.push_str(&other.dump().map_err(|e| e.to_string())?),
    }
    Ok(())
}

fn score_json(value: &Value) -> Result<String> {
    let mut out = String::new();
    write_score(value, &mut out)?;
    out.push('\n');
    Ok(out)
}

fn tokenize(engine: &Engine, text: &str) -> Vec<i32> {
    let mut tokens = Vec::new();
    engine.vocab.tokenize(text.as_bytes(), &mut tokens);
    tokens
}

fn finite_logits(session: &Session, vocabulary: usize, what: &str) -> Result<Vec<f32>> {
    let logits = session.logits().ok_or(format!("{what}: no logits"))?;
    require(
        logits.len() == vocabulary && logits.iter().all(|v| v.is_finite()),
        format!("{what}: invalid target logits"),
    )?;
    Ok(logits.to_vec())
}

/// The first maximum (`std::max_element`) and the log probability of
/// `target` under the logits.
fn first_max_and_logprob(logits: &[f32], target: i32) -> (usize, f64) {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    let maximum = f64::from(logits[best]);
    let normalizer: f64 = logits.iter().map(|&v| (f64::from(v) - maximum).exp()).sum();
    (
        best,
        f64::from(logits[target as usize]) - (maximum + normalizer.ln()),
    )
}

#[derive(Default)]
struct Totals {
    cases: usize,
    tokens: usize,
    first_matches: usize,
    matching_prefix: usize,
    matches: usize,
    nll: f64,
}

/// `ScoreOfficialReference`: external continuations of the matching hosted
/// checkpoint, deliberately separate from comparing two routes through this
/// decoder.
fn score_official(engine: &mut Engine, output: &Path) -> Result<()> {
    let bytes = std::fs::read(OFFICIAL_FIXTURE)
        .map_err(|e| format!("cannot open official continuation fixture: {e}"))?;
    let fixture = json::parse(&bytes).map_err(|e| e.to_string())?;
    require(
        fixture.member_str("schema", "") == "gufo.ds4-official-continuations.v1",
        "unexpected official fixture schema",
    )?;
    let cases = fixture
        .find("cases")
        .ok_or("missing official fixture field: cases")?
        .items();
    require(cases.len() == 105, "incomplete official fixture")?;
    let vocabulary = engine.vocab.len();
    let mut records = Vec::new();
    let mut groups: BTreeMap<String, Totals> = BTreeMap::new();
    for item in cases {
        let id = item.member_str("id", "");
        let group = item.member_str("group", "");
        require(
            !id.is_empty() && !group.is_empty(),
            "missing official case identity",
        )?;
        let mut prompt = Vec::new();
        engine
            .vocab
            .encode_chat(b"", item.member_str("prompt", "").as_bytes(), &mut prompt);
        let target = tokenize(engine, &item.member_str("continuation", ""));
        let official = item
            .find("token_bytes_hex")
            .ok_or("missing official fixture field: token_bytes_hex")?
            .items();
        require(
            !prompt.is_empty() && !target.is_empty() && target.len() == official.len(),
            format!("{id}: tokenization differs from official continuation"),
        )?;
        for (&token, expected) in target.iter().zip(official) {
            require(
                hex(&engine.vocab.token_text(token)) == expected.str(),
                format!("{id}: official token boundaries differ"),
            )?;
        }
        let context: u32 = if group == "continuation-100" {
            4096
        } else {
            16384
        };
        require(
            prompt.len() + target.len() + 1 < context as usize,
            format!("{id}: official case exceeds context"),
        )?;
        let mut session = engine
            .create_session(context, false)
            .map_err(|e| e.to_string())?;
        session.sync(engine, &prompt).map_err(|e| e.to_string())?;
        let mut result = Totals {
            cases: 1,
            tokens: target.len(),
            ..Totals::default()
        };
        let mut prefix_matches = true;
        let mut steps = Vec::new();
        for (index, &token) in target.iter().enumerate() {
            let logits = finite_logits(&session, vocabulary, &id)?;
            let greedy = session.argmax();
            let (best, logprob) = first_max_and_logprob(&logits, token);
            require(
                greedy == best as i32,
                format!("{id}: decoder argmax differs from its full logits"),
            )?;
            let matches = greedy == token;
            result.first_matches += usize::from(index == 0 && matches);
            result.matches += usize::from(matches);
            prefix_matches &= matches;
            result.matching_prefix += usize::from(prefix_matches);
            require(
                logprob.is_finite(),
                format!("{id}: invalid target log probability"),
            )?;
            result.nll -= logprob;
            steps.push(
                Value::object()
                    .with("target", f64::from(token))
                    .with("greedy", f64::from(greedy))
                    .with("logprob", logprob),
            );
            session.eval(engine, token).map_err(|e| e.to_string())?;
        }
        let total = groups.entry(group.clone()).or_default();
        total.cases += result.cases;
        total.tokens += result.tokens;
        total.first_matches += result.first_matches;
        total.matching_prefix += result.matching_prefix;
        total.matches += result.matches;
        total.nll += result.nll;
        records.push(
            Value::object()
                .with("id", id.as_str())
                .with("group", group.as_str())
                .with("context_tokens", f64::from(context))
                .with("prefill_capacity", f64::from(session.prefill_capacity()))
                .with("prompt_tokens", prompt.len() as f64)
                .with("target_tokens", target.len() as f64)
                .with("nll", result.nll)
                .with("first_match", result.first_matches as f64)
                .with("greedy_lcp", result.matching_prefix as f64)
                .with("steps", Value::Array(steps)),
        );
        eprintln!(
            "Official {id}: prompt={} tokens={} NLL={} prefix={}",
            prompt.len(),
            target.len(),
            result.nll / target.len() as f64,
            result.matching_prefix
        );
    }
    let coverage = |name: &str| groups.get(name).map_or((0, 0), |t| (t.cases, t.tokens));
    require(
        coverage("continuation-100") == (100, 2313) && coverage("smoke-5") == (5, 14),
        "official scoring coverage changed",
    )?;
    let summary = Value::Object(
        groups
            .iter()
            .map(|(name, total)| {
                (
                    name.clone(),
                    Value::object()
                        .with("cases", total.cases as f64)
                        .with("tokens", total.tokens as f64)
                        .with("nll", total.nll / total.tokens as f64)
                        .with("first_matches", total.first_matches as f64)
                        .with(
                            "mean_greedy_prefix",
                            total.matching_prefix as f64 / total.cases as f64,
                        )
                        .with("greedy_matches", total.matches as f64),
                )
            })
            .collect(),
    );
    let report = Value::object()
        .with("schema", "gufo.ds4-official-score.v1")
        .with("source_revision", fixture.member_str("source_revision", ""))
        .with("fixture_sha256", hex(&Sha256::digest(&bytes)))
        .with("prefill_policy", "native automatic")
        .with("cases", Value::Array(records))
        .with("summary", summary.clone());
    std::fs::write(output, score_json(&report)?)
        .map_err(|e| format!("cannot write official score output: {e}"))?;
    print!("{}", score_json(&summary)?);
    Ok(())
}

fn write(path: &Path, bytes: impl AsRef<[u8]>, what: &str) -> Result<()> {
    std::fs::write(path, bytes).map_err(|e| format!("cannot write {what}: {e}"))
}

fn lines(tokens: &[i32]) -> String {
    tokens.iter().map(|t| format!("{t}\n")).collect()
}

/// `DumpReferenceFrontiers`.
fn dump_frontiers(engine: &mut Engine, directory: &Path, prefill_only: bool) -> Result<()> {
    require(
        !directory.exists(),
        "frontier output directory already exists",
    )?;
    std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let mut continuation = String::new();
    let mut targets = Vec::new();
    while targets.len() < 128 {
        continuation.push_str(
            " Virtual memory maps pages to physical memory. \
             The operating system tracks permissions and page faults.",
        );
        targets = tokenize(engine, &continuation);
    }
    targets.truncate(128);
    write(
        &directory.join("targets.txt"),
        lines(&targets),
        "frontier target tokens",
    )?;

    let messages = |body: String| {
        vec![ChatMessage {
            role: "user".into(),
            content: body,
            ..ChatMessage::default()
        }]
    };
    let options = ChatTemplateOptions::default();
    require(
        tokenize(engine, "x").len() == 1 && tokenize(engine, " x").len() == 1,
        "frontier prefix requires single-token words",
    )?;
    let overhead = engine
        .encode_chat_messages(&messages("x".into()), &[], &options)
        .len()
        - 1;
    let decode_tokens = if prefill_only { 0 } else { targets.len() };
    let vocabulary = engine.vocab.len();
    let mut points = Vec::new();
    let mut manifest = String::new();
    for depth in [0usize, 4096, 8192, 12288, 16384] {
        let prefix = if depth == 0 { 16 } else { depth };
        require(
            prefix > overhead,
            "frontier prefix is shorter than chat framing",
        )?;
        let mut body = String::from("x");
        for _ in 1..prefix - overhead {
            body.push_str(" x");
        }
        let chat = messages(body);
        let prompt = engine.encode_chat_messages(&chat, &[], &options);
        require(prompt.len() == prefix, "frontier prefix depth changed")?;
        let stem = depth.to_string();
        write(
            &directory.join(format!("{stem}.tokens.txt")),
            lines(&prompt),
            "frontier prompt tokens",
        )?;
        let prompt_path = directory.join(format!("{stem}.prompt.txt"));
        write(
            &prompt_path,
            render_chat(&chat, &[], &options),
            "frontier prompt",
        )?;
        let absolute = std::path::absolute(&prompt_path).map_err(|e| e.to_string())?;
        for prefill_step in [2048usize, 4096] {
            let mut session = engine
                .create_session(20480, false)
                .map_err(|e| e.to_string())?;
            let _ = writeln!(
                manifest,
                "{prefill_step}\t{depth}\t{prefix}\t{}\t{}",
                session.prefill_capacity(),
                absolute.display()
            );
            let mut end = 0;
            while end < prompt.len() {
                end = prompt.len().min(end + prefill_step);
                session
                    .sync(engine, &prompt[..end])
                    .map_err(|e| e.to_string())?;
            }
            let dump = |suffix: &str, logits: &[f32]| -> Result<String> {
                let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
                let path = directory.join(format!("{prefill_step}-{stem}{suffix}.f32"));
                write(&path, &bytes, "frontier logits")?;
                Ok(hex(&Sha256::digest(&bytes)))
            };
            let mut point = Value::object()
                .with("depth", depth as f64)
                .with("prefill_step", prefill_step as f64)
                .with("prefill_capacity", f64::from(session.prefill_capacity()))
                .with("prompt_tokens", prefix as f64);
            let (mut greedy_tokens, mut logprobs) = (Vec::new(), Vec::new());
            let (mut before, mut after) = (None, None);
            for step in 0..=targets.len() {
                let logits = finite_logits(&session, vocabulary, "invalid frontier logits")?;
                let greedy = session.argmax();
                let target = targets.get(step).copied().unwrap_or(0);
                let (best, logprob) = first_max_and_logprob(&logits, target);
                require(
                    greedy == best as i32,
                    "frontier argmax differs from its full logits",
                )?;
                if step == 0 {
                    before = Some(dump("-before", &logits)?);
                }
                if step == decode_tokens {
                    if !prefill_only {
                        after = Some(dump("-after", &logits)?);
                    }
                    break;
                }
                require(logprob.is_finite(), "invalid frontier target likelihood")?;
                greedy_tokens.push(Value::from(f64::from(greedy)));
                logprobs.push(Value::from(logprob));
                session.eval(engine, target).map_err(|e| e.to_string())?;
            }
            point = point
                .with("greedy", Value::Array(greedy_tokens))
                .with("target_logprobs", Value::Array(logprobs));
            if let Some(before) = before {
                point = point.with("before_sha256", before);
            }
            if let Some(after) = after {
                point = point.with("after_sha256", after);
            }
            points.push(point);
            eprintln!(
                "AR frontier prefill={prefill_step} depth={depth} tokens={decode_tokens} complete"
            );
        }
    }
    write(
        &directory.join("manifest.tsv"),
        manifest,
        "frontier manifest",
    )?;
    let report = Value::object()
        .with("vocab", vocabulary as f64)
        .with("dtype", "float32-le")
        .with("prefill_policy", "explicit 2048/4096-token calls")
        .with("context_capacity", 20480.0)
        .with(
            "prefix",
            "repeated x words inside the native no-thinking chat template",
        )
        .with("decode_tokens", decode_tokens as f64)
        .with("points", Value::Array(points));
    write(
        &directory.join("summary.json"),
        score_json(&report)?,
        "frontier summary",
    )
}

fn main() -> std::result::Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = std::env::var_os("GUFO_DEEPSEEK_V4_FLASH_MODEL")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .ok_or("GUFO_DEEPSEEK_V4_FLASH_MODEL is not set")?;
    let usage =
        "usage: reference --official OUT.json | --reference-frontiers OUT-DIR [--prefill-only]";
    let mut engine = Engine::open(&model, None)?;
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["--official", output] => score_official(&mut engine, Path::new(output)),
        ["--reference-frontiers", directory] => {
            dump_frontiers(&mut engine, Path::new(directory), false)
        }
        ["--reference-frontiers", directory, "--prefill-only"] => {
            dump_frontiers(&mut engine, Path::new(directory), true)
        }
        _ => Err(usage.into()),
    }
}
