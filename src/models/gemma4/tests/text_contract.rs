//! The non-tool checks of the extension's `text_contract_test.cpp`: the
//! sampling defaults and their resolution against explicit settings. The
//! image prefill stops are checked in `vision_prompt.rs`; the tool dialect
//! and reasoning delimiters belong to the serving layer.

use gufo_core::sampling::{SamplingConfig, SamplingOverrides};
use gufo_gemma4::text_contract::SAMPLING_DEFAULTS;

#[test]
fn sampling_defaults() {
    let defaults = SAMPLING_DEFAULTS;
    assert!(
        defaults.temperature == 1.0 && defaults.top_k == 64 && defaults.top_p == 0.95,
        "generation_config defaults"
    );
    let resolved = defaults.resolve(&SamplingConfig::default(), &SamplingOverrides::default());
    let base = SamplingConfig::default();
    assert!(
        resolved.temperature == 1.0 && resolved.top_k == 64 && resolved.top_p == 0.95,
        "resolved defaults"
    );
    assert!(
        resolved.min_p == base.min_p
            && resolved.min_keep == base.min_keep
            && resolved.repeat_penalty == base.repeat_penalty
            && resolved.repeat_last_n == base.repeat_last_n
            && resolved.frequency_penalty == base.frequency_penalty
            && resolved.presence_penalty == base.presence_penalty,
        "no other defaults"
    );
    let configured = SamplingConfig {
        top_k: 5,
        min_p: 0.2,
        ..SamplingConfig::default()
    };
    let supplied = SamplingOverrides {
        top_k: true,
        ..SamplingOverrides::default()
    };
    let resolved = defaults.resolve(&configured, &supplied);
    assert!(
        resolved.top_k == 5 && resolved.temperature == 1.0 && resolved.min_p == base.min_p,
        "explicit settings override the defaults; unsupplied ones do not leak"
    );
}
