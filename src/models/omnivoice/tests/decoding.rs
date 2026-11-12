//! The C++ extension's `decoding_test.cpp`: the unmasking schedule and the
//! conditional input against the reference's generator fixture, and the
//! greedy position choice.

#![allow(clippy::unwrap_used)]

mod config_samples;

use gufo_core::json::Value;
use gufo_omnivoice::decoding::{
    ConditionalInput, DecodingConfig, PositionSelector, unmask_schedule,
};

fn flatten(rows: &Value) -> Vec<i32> {
    rows.items()
        .iter()
        .flat_map(|row| row.items().iter().map(|value| value.as_double(0.0) as i32))
        .collect()
}

#[test]
fn schedule_and_conditional_input() {
    let golden = config_samples::golden("generator_golden.json");
    let config = golden.find("config").unwrap();
    let steps = config.find("num_step").unwrap().as_size(0);
    let t_shift = config.find("t_shift").unwrap().as_double(0.0) as f32;
    let entries = golden.find("generate").unwrap().items();
    assert!(!entries.is_empty());
    for entry in entries {
        let frames = entry.find("target").unwrap().as_size(0);
        let expected: Vec<usize> = entry
            .find("schedule")
            .unwrap()
            .items()
            .iter()
            .map(|count| count.as_size(0))
            .collect();
        assert_eq!(
            unmask_schedule(8 * frames, steps, t_shift),
            expected,
            "schedule for {frames} frames"
        );
        // The conditional input from its text IDs and reference codes.
        let ids = flatten(entry.find("input_ids").unwrap());
        let length = ids.len() / 8;
        let audio_start = entry.find("audio_start").unwrap().as_size(0);
        let text: Vec<u32> = ids[..audio_start].iter().map(|&id| id as u32).collect();
        let reference_value = entry.find("reference_codes").unwrap();
        let reference = if reference_value.is_null() {
            Vec::new()
        } else {
            flatten(reference_value)
        };
        let input =
            ConditionalInput::build(&text, &reference, reference.len() / 8, frames, 8, 1024);
        assert!(
            input.length == length && input.ids == ids,
            "conditional input for {}",
            entry.find("text").unwrap().str()
        );
    }
}

#[test]
fn greedy_position_choice() {
    // The best masked scores after the layer penalty.
    let decoding = DecodingConfig {
        position_temperature: 0.0,
        ..DecodingConfig::default()
    };
    let mut selector = PositionSelector::new(&decoding, 0);
    let mut codes = vec![1024, 7, 1024, 1024, 1024, 1024];
    let prediction = [10, 11, 12, 13, 14, 15];
    // Two codebooks of three frames; codebook 1 pays a penalty of 5.
    let confidence = [-0.5f32, -0.1, -2.0, -0.2, -0.3, -9.0];
    let chosen = selector.fill(2, 2, 3, &prediction, &confidence, 1024, &mut codes);
    assert_eq!(chosen, [0, 2]);
    assert_eq!(codes, [10, 7, 12, 1024, 1024, 1024]);
}
