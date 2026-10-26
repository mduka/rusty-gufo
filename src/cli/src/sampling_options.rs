//! The model-independent sampling options (`src/cli/sampling_options.hpp`).

use gufo_core::sampling::{SamplingConfig, SamplingOverrides};

use crate::arg_parser::ArgParser;

/// `SamplingOptionsSupplied`: the sampling options the last parse saw.
pub fn sampling_options_supplied(parser: &ArgParser<'_>) -> SamplingOverrides {
    SamplingOverrides {
        temperature: parser.was_supplied("--temperature"),
        top_k: parser.was_supplied("--top-k"),
        top_p: parser.was_supplied("--top-p"),
        min_p: parser.was_supplied("--min-p"),
        min_keep: parser.was_supplied("--min-keep"),
        seed: parser.was_supplied("--seed"),
        repeat_penalty: parser.was_supplied("--repeat-penalty"),
        repeat_last_n: parser.was_supplied("--repeat-last-n"),
        frequency_penalty: parser.was_supplied("--frequency-penalty"),
        presence_penalty: parser.was_supplied("--presence-penalty"),
    }
}

/// `RegisterSamplingOptions`: the complete sampling control surface. Short
/// aliases (`-t`, `-s`) can be left out when the command owns them;
/// `model_defaults` describes the model's recommended defaults, which fill
/// the options left unset.
pub fn register_sampling_options<'a>(
    parser: &mut ArgParser<'a>,
    config: &'a mut SamplingConfig,
    group: &str,
    register_short_aliases: bool,
    model_defaults: bool,
) {
    let SamplingConfig {
        temperature,
        top_k,
        top_p,
        min_p,
        min_keep,
        seed,
        repeat_penalty,
        repeat_last_n,
        frequency_penalty,
        presence_penalty,
        constraint: _,
        reasoning_budget: _,
    } = config;
    let alias = |name: &'static str| if register_short_aliases { name } else { "" };
    parser
        .add_option(
            alias("-t"),
            "--temperature",
            "T",
            if model_defaults {
                "Randomness scale; 0 selects greedy (default: model/thinking preset)"
            } else {
                "Randomness scale; 0.0 selects greedy argmax (default: 0.0)"
            },
            group,
            temperature,
        )
        .add_option(
            "",
            "--top-k",
            "K",
            if model_defaults {
                "Highest-logit token limit; 0 disables (default: model preset)"
            } else {
                "Keep only the K highest-logit tokens (default: 0 = disabled)"
            },
            group,
            top_k,
        )
        .add_option(
            "",
            "--top-p",
            "P",
            if model_defaults {
                "Cumulative probability cutoff (default: model/thinking preset)"
            } else {
                "Keep the smallest token set whose cumulative probability reaches P \
                 (default: 1.0 = disabled)"
            },
            group,
            top_p,
        )
        .add_option(
            "",
            "--min-p",
            "P",
            "Keep tokens with probability at least P times the best token \
             (default: 0.0 = disabled)",
            group,
            min_p,
        )
        .add_option(
            "",
            "--min-keep",
            "N",
            "Minimum candidates retained by enabled sampling filters (default: 0)",
            group,
            min_keep,
        )
        .add_option(
            alias("-s"),
            "--seed",
            "N",
            "RNG seed for reproducible generation (default: -1 = random)",
            group,
            seed,
        )
        .add_option(
            "",
            "--repeat-penalty",
            "N",
            "Multiplicative penalty for recently used tokens (default: 1.0 = disabled)",
            group,
            repeat_penalty,
        )
        .add_option(
            "",
            "--repeat-last-n",
            "N",
            "Trailing token window used by repetition penalties (default: 64; 0 = disabled)",
            group,
            repeat_last_n,
        )
        .add_option(
            "",
            "--frequency-penalty",
            "N",
            "Penalty multiplied by each token's generated occurrence count \
             (default: 0.0 = disabled)",
            group,
            frequency_penalty,
        )
        .add_option(
            "",
            "--presence-penalty",
            "N",
            if model_defaults {
                "Generated-token presence penalty (default: model/thinking preset)"
            } else {
                "One-time penalty for tokens already generated in this response \
                 (default: 0.0 = disabled)"
            },
            group,
            presence_penalty,
        );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_control_parses_into_the_config() {
        let mut config = SamplingConfig::default();
        {
            let mut parser = ArgParser::new("t", "");
            register_sampling_options(&mut parser, &mut config, "Sampling", true, false);
            parser
                .parse(&[
                    "-t",
                    "0.7",
                    "--top-k",
                    "40",
                    "--top-p",
                    "0.9",
                    "--min-p",
                    "0.05",
                    "--min-keep",
                    "2",
                    "-s",
                    "-5",
                    "--repeat-penalty",
                    "1.1",
                    "--repeat-last-n",
                    "32",
                    "--frequency-penalty",
                    "0.2",
                    "--presence-penalty",
                    "0.3",
                ])
                .unwrap_or_else(|e| panic!("{e}"));
        }
        assert_eq!(
            config,
            SamplingConfig {
                temperature: 0.7,
                top_k: 40,
                top_p: 0.9,
                min_p: 0.05,
                min_keep: 2,
                seed: -5,
                repeat_penalty: 1.1,
                repeat_last_n: 32,
                frequency_penalty: 0.2,
                presence_penalty: 0.3,
                constraint: None,
                reasoning_budget: None,
            }
        );
        let mut other = SamplingConfig::default();
        let mut parser = ArgParser::new("t", "");
        register_sampling_options(&mut parser, &mut other, "Sampling", false, false);
        assert!(parser.find_option("-t").is_none());
        assert!(parser.find_option("--seed").is_some());
    }
}
