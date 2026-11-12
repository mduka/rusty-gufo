//! The Qwen3-VL tokenizer of `processor/tokenizer.json` (`tokenizer.cpp`):
//! the vocabulary, merges and added tokens it declares, with the settings
//! the C++ loader accepts. Text between added tokens is normalised to NFC
//! and split by the Qwen2 regex before byte-level BPE, which the core
//! tokenizer's `qwen2` pre-tokenizer reproduces (the C++ code runs the regex
//! through ICU).

use std::path::Path;

use gufo_core::bpe::{BpeTokenizer, PreTokenizer, TokenId};
use gufo_core::json::Value;
use gufo_core::text::{Error, Result};

use crate::weights::read_json;

/// Vocabulary entries the C++ loader reserves before trimming.
const VOCABULARY: usize = 151_669;
/// Token IDs must stay below the text encoder's vocabulary.
const VOCAB_SIZE: f64 = 151_936.0;
/// The Qwen2 split regex the C++ tokenizer accepts.
const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Tokenizer {
    bpe: BpeTokenizer,
}

fn require_false(object: &Value, key: &str) -> Result<()> {
    match object.find(key) {
        Some(value) if value.is_bool() && !value.as_bool(true) => Ok(()),
        _ => Err(Error::runtime(format!(
            "unsupported tokenizer setting: {key}"
        ))),
    }
}

fn token_id(value: Option<&Value>) -> Result<usize> {
    let number = value.map_or(f64::NAN, |v| v.as_double(f64::NAN));
    if !value.is_some_and(Value::is_number)
        || !number.is_finite()
        || !(0.0..VOCAB_SIZE).contains(&number)
        || number.fract() != 0.0
    {
        return Err(Error::runtime("invalid tokenizer ID"));
    }
    Ok(number as usize)
}

impl Tokenizer {
    /// `Tokenizer::Tokenizer`.
    pub fn load(root: &Path) -> Result<Self> {
        let config = read_json(&root.join("processor/tokenizer.json"))?;
        let model = config.find("model");
        let vocab = model.and_then(|m| m.find("vocab"));
        let merges = model.and_then(|m| m.find("merges"));
        let added = config.find("added_tokens");
        let sequence = config
            .find("pre_tokenizer")
            .and_then(|p| p.find("pretokenizers"));
        let (Some(model), Some(vocab), Some(merges), Some(added), Some(sequence)) =
            (model, vocab, merges, added, sequence)
        else {
            return Err(Error::runtime("invalid Qwen-Image tokenizer"));
        };
        if !vocab.is_object()
            || !merges.is_array()
            || !added.is_array()
            || !sequence.is_array()
            || sequence.len() != 2
        {
            return Err(Error::runtime("invalid Qwen-Image tokenizer"));
        }
        let split = &sequence.items()[0];
        let byte_level = &sequence.items()[1];
        let regex = split.find("pattern");
        if split.member_str("type", "") != "Split"
            || regex.is_none()
            || byte_level.member_str("type", "") != "ByteLevel"
        {
            return Err(Error::runtime("unsupported Qwen-Image pre-tokenizer"));
        }
        if model.member_str("type", "") != "BPE"
            || split.member_str("behavior", "") != "Isolated"
            || !model.find("dropout").is_some_and(Value::is_null)
            || !model.member_str("continuing_subword_prefix", "").is_empty()
            || !model.member_str("end_of_word_suffix", "").is_empty()
        {
            return Err(Error::runtime("unsupported Qwen-Image BPE configuration"));
        }
        require_false(split, "invert")?;
        require_false(byte_level, "add_prefix_space")?;
        require_false(byte_level, "use_regex")?;
        require_false(model, "byte_fallback")?;
        require_false(model, "ignore_merges")?;
        let pattern = regex.map(|r| r.member_str("Regex", "")).unwrap_or_default();
        if pattern.is_empty() {
            return Err(Error::runtime("missing tokenizer regex"));
        }
        // The C++ tokenizer compiles whatever regex the file names; the
        // pre-tokenizer here is the Qwen2 one.
        if pattern != PATTERN {
            return Err(Error::runtime("unsupported Qwen-Image pre-tokenizer"));
        }
        if config
            .find("normalizer")
            .is_none_or(|n| n.member_str("type", "") != "NFC")
        {
            return Err(Error::runtime(
                "unsupported Qwen-Image tokenizer normalizer",
            ));
        }
        let mut tokens: Vec<Vec<u8>> = vec![Vec::new(); VOCABULARY];
        let mut install = |id: usize, word: &str| -> Result<()> {
            if id >= VOCAB_SIZE as usize || word.is_empty() {
                return Err(Error::runtime("invalid tokenizer entry"));
            }
            if id >= tokens.len() {
                tokens.resize(id + 1, Vec::new());
            }
            if !tokens[id].is_empty() && tokens[id] != word.as_bytes() {
                return Err(Error::runtime("duplicate tokenizer entry"));
            }
            tokens[id] = word.as_bytes().to_vec();
            Ok(())
        };
        for (word, id) in vocab.members() {
            install(token_id(Some(id))?, word)?;
        }
        let mut specials: Vec<(Vec<u8>, TokenId)> = Vec::new();
        for entry in added.items() {
            for key in ["single_word", "lstrip", "rstrip", "normalized"] {
                require_false(entry, key)?;
            }
            let id = token_id(entry.find("id"))?;
            let word = entry.member_str("content", "");
            install(id, &word)?;
            // `emplace` into a map in C++: the first entry of a name wins.
            if !specials.iter().any(|(name, _)| name == word.as_bytes()) {
                specials.push((word.into_bytes(), id as TokenId));
            }
        }
        while tokens.last().is_some_and(Vec::is_empty) {
            tokens.pop();
        }
        let mut pairs = Vec::with_capacity(merges.len());
        for merge in merges.items() {
            if merge.is_string() {
                pairs.push(merge.str().as_bytes().to_vec());
            } else if merge.is_array()
                && merge.len() == 2
                && merge.items().iter().all(Value::is_string)
            {
                pairs.push(
                    format!("{} {}", merge.items()[0].str(), merge.items()[1].str()).into_bytes(),
                );
            } else {
                return Err(Error::runtime("invalid tokenizer merge"));
            }
        }
        let bpe = BpeTokenizer::new(&tokens, &pairs, &specials)
            .map_err(Error::runtime)?
            .with_pre_tokenizer(PreTokenizer::Qwen2);
        Ok(Self { bpe })
    }

    /// `Tokenizer::Encode`: added tokens anywhere in the text, the earliest
    /// first and the longest on a tie.
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        self.bpe.encode(text)
    }
}
