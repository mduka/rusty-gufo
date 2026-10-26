//! The `gufo` command line (`RunGufo` of the C++ CLI): the dispatch, the
//! help, the version, `serve` (every modality), `prompt`, `chat`, `bench`,
//! `eval`, `transcribe` (Qwen3-ASR-1.7B), `probe` and `diagnose`.

#![forbid(unsafe_code)]

mod arbiter;
mod arg_parser;
mod banner;
mod bench;
mod diagnose;
mod eval;
mod gpu_queues;
mod prompt;
mod sampling_options;
mod serve;
mod serve_cli;
mod transcribe;

use std::ffi::OsString;
use std::process::ExitCode;

/// The semantic release version (`GUFO_RELEASE_VERSION` at build time).
const RELEASE_VERSION: &str = env!("GUFO_RELEASE_VERSION");
/// The source revision and provenance identity (`GUFO_REVISION` at build
/// time).
pub(crate) const REVISION: &str = env!("GUFO_REVISION");

fn print_help(program: &str) {
    print!(
        "Usage: {program} <COMMAND> [OPTIONS]\n\n\
         Fast LLM & DiT inference engine for AMD Strix Halo (gfx1151).\n\n\
         Commands:\n  \
         serve          Start the OpenAI-compatible HTTP server\n  \
         arbiter        Share the GPU between serve processes\n  \
         prompt         Execute one prompt request and exit\n  \
         chat           Start an interactive terminal conversation\n  \
         bench          Benchmark prompt processing and token generation\n  \
         eval           Evaluate an OpenAI-compatible text server\n  \
         transcribe     Transcribe WAV audio with Qwen3-ASR-1.7B\n  \
         diagnose       Inspect system hardware, memory, and drivers\n  \
         help           Print help for a specific command\n\n\
         Run '{program} help <COMMAND>' or '{program} <COMMAND> --help' for details on a command\n\n\
         Options:\n  \
         -h, --help     Print help\n  \
         -v, --version  Print version\n"
    );
}

fn help(program: &str, args: &[String]) -> ExitCode {
    let Some(topic) = args.get(1) else {
        print_help(program);
        return ExitCode::SUCCESS;
    };
    match topic.as_str() {
        "serve" => {
            let mut topic: Vec<String> = args.get(2).cloned().into_iter().collect();
            topic.push("--help".into());
            ExitCode::from(serve::run(&topic))
        }
        command @ ("prompt" | "chat") => {
            prompt::print_help(command);
            ExitCode::SUCCESS
        }
        "bench" => {
            bench::print_help();
            ExitCode::SUCCESS
        }
        "transcribe" => {
            transcribe::print_help();
            ExitCode::SUCCESS
        }
        "arbiter" => {
            arbiter::print_help();
            ExitCode::SUCCESS
        }
        "eval" => {
            eval::print_help();
            ExitCode::SUCCESS
        }
        "probe" => diagnose::run_probe(&["--help".to_string()]),
        "diagnose" => diagnose::run(&["--help".to_string()]),
        _ => {
            eprintln!("Error: unknown help topic '{topic}'");
            print_help(program);
            ExitCode::from(2)
        }
    }
}

fn main() -> ExitCode {
    let raw: Vec<OsString> = std::env::args_os().collect();
    let Some(program) = raw.first().map(|arg| arg.to_string_lossy().into_owned()) else {
        eprintln!("Error: missing command or option");
        print_help("gufo");
        return ExitCode::from(2);
    };
    let args: Vec<String> = raw
        .iter()
        .skip(1)
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let Some(first) = args.first() else {
        print_help(&program);
        return ExitCode::SUCCESS;
    };
    match first.as_str() {
        "--version" | "-v" => {
            println!("gufo version {RELEASE_VERSION} ({REVISION})");
            ExitCode::SUCCESS
        }
        "--help" | "-h" => {
            print_help(&program);
            ExitCode::SUCCESS
        }
        "help" => help(&program, &args),
        _ => {
            banner::print_startup_banner(RELEASE_VERSION, &args);
            run_command(&program, &args)
        }
    }
}

/// The commands after the startup banner (`RunGufo` from `probe` on).
fn run_command(program: &str, args: &[String]) -> ExitCode {
    let first = args[0].as_str();
    match first {
        "probe" => diagnose::run_probe(&args[1..]),
        "diagnose" => diagnose::run(args),
        "bench" => bench::run(&args[1..]),
        "eval" => eval::run(&args[1..]),
        "prompt" => prompt::run_prompt(&args[1..]),
        "transcribe" => transcribe::run(&args[1..]),
        "chat" => prompt::run_chat(&args[1..]),
        "serve" => ExitCode::from(serve::run(&args[1..])),
        "arbiter" => arbiter::run(&args[1..]),
        // Invoked as the legacy server binary: every argument goes to serve.
        _ if program.ends_with("gufo-server") => ExitCode::from(serve::run(args)),
        _ => {
            eprintln!("Error: unknown command '{first}'");
            print_help(program);
            ExitCode::from(2)
        }
    }
}
