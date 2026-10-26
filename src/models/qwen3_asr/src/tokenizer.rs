//! The checkpoint's tokenizer: `vocab.json`, the added tokens of
//! `tokenizer_config.json`, and `merges.txt`, which is read on the first
//! encode only (transcription without context or language never encodes).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde_json::Value;

use gufo_core::bpe::{BpeTokenizer, TokenId};

const MAXIMUM_VOCABULARY_BYTES: u64 = 16 << 20;
const MAXIMUM_TOKENIZER_CONFIG_BYTES: u64 = 1 << 20;

fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let size = std::fs::metadata(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?
        .len();
    if size > limit {
        return Err(format!("unexpected file size for {}", path.display()));
    }
    std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

pub struct Tokenizer {
    root: PathBuf,
    tokens: Vec<Vec<u8>>,
    /// Every added token: matched whole when encoding.
    added: Vec<(Vec<u8>, TokenId)>,
    /// Added tokens flagged `special`: dropped by `decode(.., true)`.
    /// `<asr_text>` is added but not special.
    special: HashSet<TokenId>,
    decoder: BpeTokenizer,
    encoder: OnceLock<Result<BpeTokenizer, String>>,
}

impl Tokenizer {
    pub fn load(root: &Path) -> Result<Self, String> {
        Self::load_inner(root).map_err(|e| format!("Qwen3-ASR tokenizer load failed: {e}"))
    }

    fn load_inner(root: &Path) -> Result<Self, String> {
        let vocabulary: Value = serde_json::from_slice(&read_limited(
            &root.join("vocab.json"),
            MAXIMUM_VOCABULARY_BYTES,
        )?)
        .map_err(|e| format!("vocab.json: {e}"))?;
        let config: Value = serde_json::from_slice(&read_limited(
            &root.join("tokenizer_config.json"),
            MAXIMUM_TOKENIZER_CONFIG_BYTES,
        )?)
        .map_err(|e| format!("tokenizer_config.json: {e}"))?;
        let (Some(vocabulary), Some(config)) = (vocabulary.as_object(), config.as_object()) else {
            return Err("tokenizer JSON root must be an object".into());
        };
        let decoder_entries = config
            .get("added_tokens_decoder")
            .and_then(Value::as_object)
            .ok_or("missing added_tokens_decoder")?;

        let mut entries = Vec::with_capacity(vocabulary.len());
        for (symbol, id) in vocabulary {
            let id = id.as_u64().ok_or("vocabulary ID must be numeric")?;
            entries.push((symbol, id));
        }
        let mut added_entries = Vec::with_capacity(decoder_entries.len());
        for (id, spec) in decoder_entries {
            let id: u64 = id.parse().map_err(|_| "invalid added-token ID")?;
            let spec = spec
                .as_object()
                .ok_or("invalid added-token specification")?;
            added_entries.push((id, spec));
        }
        let maximum = entries
            .iter()
            .map(|e| e.1)
            .chain(added_entries.iter().map(|e| e.0))
            .max()
            .unwrap_or(0);
        if maximum >= u64::from(u32::MAX) {
            return Err("tokenizer vocabulary is too large".into());
        }
        let mut tokens: Vec<Vec<u8>> = vec![Vec::new(); maximum as usize + 1];
        for (symbol, id) in entries {
            let slot = &mut tokens[id as usize];
            if !slot.is_empty() {
                return Err("duplicate vocabulary ID".into());
            }
            *slot = symbol.as_bytes().to_vec();
        }
        let mut added = Vec::with_capacity(added_entries.len());
        let mut special = HashSet::new();
        for (id, spec) in added_entries {
            let content = spec.get("content").and_then(Value::as_str).unwrap_or("");
            let slot = &mut tokens[id as usize];
            if content.is_empty() || (!slot.is_empty() && slot.as_slice() != content.as_bytes()) {
                return Err("invalid or duplicate added token".into());
            }
            *slot = content.as_bytes().to_vec();
            added.push((content.as_bytes().to_vec(), id as TokenId));
            if spec.get("special").and_then(Value::as_bool) == Some(true) {
                special.insert(id as TokenId);
            }
        }
        if tokens.iter().any(Vec::is_empty) {
            return Err("tokenizer has gaps in its ID space".into());
        }
        let decoder = BpeTokenizer::new::<_, &[u8]>(&tokens, &[], &added)?;
        Ok(Self {
            root: root.to_path_buf(),
            tokens,
            added,
            special,
            decoder,
            encoder: OnceLock::new(),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    fn encoder(&self) -> Result<&BpeTokenizer, String> {
        self.encoder
            .get_or_init(|| {
                let path = self.root.join("merges.txt");
                let text = std::fs::read(&path)
                    .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
                let merges: Vec<&[u8]> = text
                    .split(|&c| c == b'\n')
                    .filter(|line| !line.is_empty() && !line.starts_with(b"#"))
                    .collect();
                BpeTokenizer::new(&self.tokens, &merges, &self.added)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<TokenId>, String> {
        Ok(self.encoder()?.encode(text))
    }

    /// Raw bytes of `ids`; may end inside a UTF-8 sequence.
    pub fn decode(&self, ids: &[TokenId], skip_special_tokens: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &id in ids {
            if !skip_special_tokens || !self.special.contains(&id) {
                bytes.extend_from_slice(&self.decoder.decode_token(id));
            }
        }
        bytes
    }
}
