//! Sampled DeepSeek V4 Flash `prompt` and `chat` as processes (the C++
//! `ds4.chat` check, `cli_test.cpp --sampled-chat`): three seeded chat turns
//! with DSpark repeat the autoregressive token traces after length and EOS
//! stops, and a prompt matches the first turn. Needs the model and its
//! support artifact:
//! cargo test -p gufo --features gpu --test deepseek -- --ignored

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::io::Write;
use std::process::{Command, Stdio};

const PROMPT: &str =
    "Continue this pattern for twenty more terms: red, blue, blue, red, blue, blue,";

/// The artifact `variable` names; unset, the variable itself, so the failure
/// says what to set.
fn artifact(variable: &str) -> String {
    std::env::var(variable)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| format!("${variable}"))
}

/// The `[TokenTrace]` lines of one run.
fn traces(chat: bool, dspark: bool) -> Vec<String> {
    let model = artifact("GUFO_DEEPSEEK_V4_FLASH_MODEL");
    let support = artifact("GUFO_DEEPSEEK_V4_FLASH_DSPARK_MODEL");
    let mut args = vec![
        if chat { "chat" } else { "prompt" },
        "--model",
        &model,
        "--system",
        "You are a concise assistant.",
        "--max-tokens",
        "32",
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--seed",
        "7",
        "--repeat-penalty",
        "1.1",
        "--verbose",
    ];
    if dspark {
        args.extend(["--dspark-model", &support, "--speculative", "dspark"]);
    }
    if !chat {
        args.extend(["--prompt", PROMPT, "--no-display-prompt"]);
    }
    let mut child = Command::new(env!("CARGO_BIN_EXE_gufo"))
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let input = format!(
        "{PROMPT}\nExplain the rule in one short sentence.\nRepeat the rule again.\nexit\n"
    );
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "CLI generation failed: {diagnostics}"
    );
    if dspark {
        assert!(
            diagnostics.contains("[Speculative]: acceptance="),
            "sampled CLI did not execute DSpark"
        );
    }
    diagnostics
        .lines()
        .filter(|line| line.starts_with("[TokenTrace]:"))
        .map(str::to_owned)
        .collect()
}

#[test]
#[ignore = "QUALITY check on the DeepSeek V4 Flash artifacts"]
fn sampled_chat_keeps_autoregressive_token_identity() {
    let autoregressive = traces(true, false);
    let speculative = traces(true, true);
    assert!(
        autoregressive.len() == 3 && speculative == autoregressive,
        "sampled chat retains AR token identity after length and EOS stops"
    );
    let single = traces(false, true);
    assert!(
        single.len() == 1 && single[0] == autoregressive[0],
        "sampled prompt and first chat turn share the same token trace"
    );
}
