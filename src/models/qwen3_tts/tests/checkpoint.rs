//! The checkpoint cases of the C++ unit tests (`loader_test.cpp`,
//! `tokenizer_test.cpp`) on the CustomVoice checkpoint, without a device.
//! cargo test -p gufo-qwen3-tts --features models --test checkpoint

#![cfg(feature = "models")]
#![allow(clippy::unwrap_used)]

mod common;

use gufo_qwen3_tts::checkpoint::{Checkpoint, Region, Tensor};
use gufo_qwen3_tts::config::ModelVariant;
use gufo_qwen3_tts::tokenizer::Tokenizer;

#[test]
fn checkpoint_loads_and_selects_component_regions() {
    let root = common::model_root(ModelVariant::CustomVoice);
    let checkpoint = Checkpoint::load(&root).unwrap();
    let config = &checkpoint.config;
    assert!(config.is_supported());
    let talker = &config.talker;
    assert_eq!(
        (
            talker.hidden_size,
            talker.num_hidden_layers,
            talker.num_attention_heads,
            talker.num_key_value_heads,
            talker.intermediate_size,
            talker.codec_eos_token_id,
        ),
        (2048, 28, 16, 8, 6144, 2150)
    );
    assert_eq!(
        (
            config.code_predictor.num_hidden_layers,
            config.code_predictor.hidden_size
        ),
        (5, 1024)
    );
    let speech = &config.speech_tokenizer;
    assert_eq!(
        (
            speech.output_sample_rate,
            speech.num_quantizers,
            speech.latent_dim,
            speech.codebook_dim,
            speech.decoder_dim,
            speech.hidden_size,
            speech.intermediate_size,
            speech.num_hidden_layers,
            speech.num_attention_heads,
            speech.head_dim,
        ),
        (24000, 16, 1024, 512, 1536, 512, 1024, 8, 16, 64)
    );
    assert_eq!(speech.upsample_rates, [8, 5, 4, 3]);
    assert_eq!(speech.upsampling_ratios, [2, 2]);
    assert_eq!(talker.spk_id["vivian"], 3065);
    assert_eq!(
        talker.spk_dialect["eric"].as_deref(),
        Some("sichuan_dialect")
    );

    let text_embedding = checkpoint
        .find("talker.model.text_embedding.weight")
        .unwrap();
    assert_eq!(text_embedding.shape[0], 151_936);
    let waveform = checkpoint.find("decoder.decoder.6.conv.weight").unwrap();
    assert_eq!(waveform.shape.len(), 3);
    let talker_regions = checkpoint.regions("talker.").unwrap();
    let decoder_regions = checkpoint.regions("decoder.").unwrap();
    let contains = |regions: &[Region], tensor: &Tensor| {
        regions.iter().any(|r| {
            r.file == tensor.file
                && tensor.offset >= r.begin
                && tensor.offset + tensor.bytes <= r.begin + r.size
        })
    };
    assert!(talker_regions.len() == 1 && decoder_regions.len() == 1);
    assert!(contains(&talker_regions, text_embedding) && contains(&decoder_regions, waveform));
    let encoder = checkpoint
        .find("encoder.encoder.layers.0.conv.weight")
        .unwrap();
    assert!(
        !contains(&decoder_regions, encoder) && !contains(&talker_regions, waveform),
        "synthesis uploads exclude unused component weights"
    );
    let all: usize = (0..2).map(|file| checkpoint.file(file).len()).sum();
    let selected = talker_regions[0].size + decoder_regions[0].size;
    assert!(selected < all, "unused weight bytes are not uploaded");
}

#[test]
fn tokenizer_matches_the_official_assistant_prompt() {
    let tokenizer = Tokenizer::load(&common::model_root(ModelVariant::CustomVoice)).unwrap();
    let ids = tokenizer
        .encode_assistant_prompt(
            "The boy who lived. Mr. and Mrs. Dursley, of number four, Privet Drive, were \
             proud to say that they were perfectly normal, thank you very much. They were \
             the last people you'd expect to be involved in anything strange or mysterious, \
             because they just didn't hold with such nonsense.",
        )
        .unwrap();
    let expected: &[u32] = &[
        151644, 77091, 198, 785, 8171, 879, 12163, 13, 4392, 13, 323, 17618, 13, 422, 1723, 3179,
        11, 315, 1372, 3040, 11, 15438, 295, 16150, 11, 1033, 12409, 311, 1977, 429, 807, 1033,
        13942, 4622, 11, 9702, 498, 1602, 1753, 13, 2379, 1033, 279, 1537, 1251, 498, 4172, 1720,
        311, 387, 6398, 304, 4113, 14888, 476, 25382, 11, 1576, 807, 1101, 3207, 944, 3331, 448,
        1741, 40802, 13, 151645, 198, 151644, 77091, 198,
    ];
    assert_eq!(
        ids, expected,
        "assistant prompt differs from the official tokenizer"
    );
}
