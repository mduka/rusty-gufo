//! `gufo arbiter`: shares the GPU between `gufo serve` processes. Servers
//! started with `--arbiter HOST:PORT` report to it; it throttles text
//! generation while a streamed TTS falls behind real time
//! (`gufo_serve::arbiter`).

use std::process::ExitCode;

use gufo_serve::arbiter::{ArbiterConfig, ArbiterService, DEFAULT_PORT};
use gufo_serve::http::HttpServerOptions;
use gufo_serve::http::server::{HttpServer, HttpServices};
use gufo_serve::logging::{self, LogLevel};

use crate::arg_parser::ArgParser;

struct Options {
    host: String,
    port: i32,
    config: String,
    log_level: Option<LogLevel>,
    verbose: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: i32::from(DEFAULT_PORT),
            config: String::new(),
            log_level: None,
            verbose: false,
        }
    }
}

fn parser(options: &mut Options) -> ArgParser<'_> {
    let Options {
        host,
        port,
        config,
        log_level,
        verbose,
    } = options;
    let mut parser = ArgParser::new(
        "gufo arbiter",
        "Share the GPU between gufo serve processes started with --arbiter: text generation \
         yields while streamed speech falls behind real time.",
    );
    parser
        .add_option("-i", "--host", "IP", "Bind address", "Server", host)
        .add_option("-p", "--port", "N", "Port to listen on", "Server", port)
        .add_option(
            "",
            "--config",
            "FILE",
            "Rules (TOML); built-in defaults without one",
            "Rules",
            config,
        )
        .add_custom_option(
            "",
            "--log-level",
            "LEVEL",
            "Log verbosity: error, warn, info or debug (default: info)",
            "Logging",
            move |flag, value| {
                *log_level = Some(
                    LogLevel::from_name(value)
                        .ok_or_else(|| format!("{flag} must be error, warn, info or debug"))?,
                );
                Ok(())
            },
        )
        .add_flag(
            "-v",
            "--verbose",
            "Shorthand for --log-level=debug",
            "Logging",
            verbose,
        );
    parser
}

pub fn print_help() {
    parser(&mut Options::default()).print_help();
}

pub fn run(args: &[String]) -> ExitCode {
    let mut options = Options::default();
    let (parsed, help) = {
        let mut parser = parser(&mut options);
        (parser.parse(args), parser.is_help_requested())
    };
    if let Err(error) = parsed {
        eprintln!("Error: {error}");
        print_help();
        return ExitCode::from(2);
    }
    if help {
        print_help();
        return ExitCode::SUCCESS;
    }
    let config = if options.config.is_empty() {
        ArbiterConfig::default()
    } else {
        match std::fs::read_to_string(&options.config)
            .map_err(|e| e.to_string())
            .and_then(|text| ArbiterConfig::parse(&text))
        {
            Ok(config) => config,
            Err(error) => {
                eprintln!("Error: {}: {error}", options.config);
                return ExitCode::from(1);
            }
        }
    };
    logging::set_level(if options.verbose {
        LogLevel::Debug
    } else {
        options.log_level.unwrap_or(LogLevel::Info)
    });
    let rules = format!(
        "min_share={} lease_ms={} tts_buffer_target_s={} instance_rules={}",
        config.min_share,
        config.lease.as_millis(),
        config.tts_buffer_target_s,
        config.instances.len()
    );
    let service = match ArbiterService::start(config) {
        Ok(service) => service,
        Err(error) => {
            eprintln!("Error starting the arbiter: {error}");
            return ExitCode::from(1);
        }
    };
    let services = HttpServices {
        arbiter: Some(service),
        ..HttpServices::default()
    };
    let server = match HttpServer::new(
        &options.host,
        options.port,
        None,
        services,
        HttpServerOptions::default(),
    ) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("Error starting the arbiter: {error}");
            return ExitCode::from(1);
        }
    };
    if let Err(error) = server.start() {
        eprintln!("Error starting the arbiter: {error}");
        return ExitCode::from(1);
    }
    logging::info(
        "arbiter",
        &format!(
            "event=arbiter_started address={}:{} {rules}",
            options.host,
            server.port()
        ),
    );
    server.run(true);
    ExitCode::SUCCESS
}
