//! Model-independent sampling (`core/sampling.{hpp,cpp}`): controls, the
//! post-filter distribution and the request-owned sampler with its penalty
//! counts and xorshift64* RNG. Filters run in the order penalties,
//! temperature, top-k, top-p, min-p; the defaults are greedy decoding. Every
//! draw, sum and sort follows the C++ code, so seeded requests reproduce its
//! token sequences. A JSON grammar (`constraint`) masks the logits before
//! every filter.

use std::sync::{Arc, OnceLock};

use crate::json_constraint::{State as ConstraintState, TokenConstraint};
use crate::text::{Error, Result};

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub min_keep: usize,
    pub seed: i64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// The grammar a structured request's output follows, bound to the
    /// model vocabulary.
    pub constraint: Option<Arc<TokenConstraint>>,
    /// Set only for requests that start in their reasoning phase.
    pub reasoning_budget: Option<Arc<ReasoningBudget>>,
}

/// A reasoning token budget (`ReasoningBudget`),
/// enforced like llama.cpp's reasoning-budget sampler. Tokens count after
/// the reasoning start, reasoning text already in the prompt included. Once
/// the budget is spent and the last token completed its UTF-8 character,
/// `forced` is emitted token by token; any token of `ends` closes reasoning
/// early. Later tokens are answer content, literal reasoning tags included.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReasoningBudget {
    pub tokens: usize,
    pub forced: Vec<TokenId>,
    pub ends: Vec<TokenId>,
    pub prompt: Vec<TokenId>,
    /// Indexed by token: its text ends inside a UTF-8 character.
    pub partial_utf8: Option<Arc<Vec<bool>>>,
}

/// Equal controls, the same bound grammar and the same reasoning budget.
impl PartialEq for SamplingConfig {
    fn eq(&self, other: &Self) -> bool {
        self.temperature == other.temperature
            && self.top_k == other.top_k
            && self.top_p == other.top_p
            && self.min_p == other.min_p
            && self.min_keep == other.min_keep
            && self.seed == other.seed
            && self.repeat_penalty == other.repeat_penalty
            && self.repeat_last_n == other.repeat_last_n
            && self.frequency_penalty == other.frequency_penalty
            && self.presence_penalty == other.presence_penalty
            && match (&self.constraint, &other.constraint) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
            && self.reasoning_budget == other.reasoning_budget
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            min_keep: 0,
            seed: -1,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            constraint: None,
            reasoning_budget: None,
        }
    }
}

/// The model family whose recommended sampling a text server or CLI applies
/// (`TextModelPreset`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TextModelPreset {
    #[default]
    Unspecified,
    Qwen38,
    DeepSeekV4Flash,
}

/// Which sampling fields the caller set explicitly: presence, not value,
/// decides, so an explicit zero still overrides a preset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SamplingOverrides {
    pub temperature: bool,
    pub top_k: bool,
    pub top_p: bool,
    pub min_p: bool,
    pub min_keep: bool,
    pub seed: bool,
    pub repeat_penalty: bool,
    pub repeat_last_n: bool,
    pub frequency_penalty: bool,
    pub presence_penalty: bool,
}

impl SamplingOverrides {
    /// Every field set: `configured` is used as it is.
    pub const ALL: Self = Self {
        temperature: true,
        top_k: true,
        top_p: true,
        min_p: true,
        min_keep: true,
        seed: true,
        repeat_penalty: true,
        repeat_last_n: true,
        frequency_penalty: true,
        presence_penalty: true,
    };
}

/// The preset of a text model artifact (`TextPreset`): DeepSeek V4 Flash,
/// Qwen3.8 Flash-Next and the 27B.
pub fn text_preset(gguf: &crate::gguf::Gguf<'_>) -> TextModelPreset {
    let architecture = gguf.string("general.architecture").unwrap_or_default();
    if architecture == b"deepseek4" {
        return TextModelPreset::DeepSeekV4Flash;
    }
    let key = |name: &str| format!("{}.{name}", String::from_utf8_lossy(architecture));
    // Embedded MTP layers count towards block_count but not to the stack.
    let layers = gguf
        .u64(&key("block_count"))
        .and_then(|blocks| blocks.checked_sub(gguf.u64(&key("nextn_predict_layers")).unwrap_or(0)));
    let dense_27b = layers == Some(64)
        && gguf.u64(&key("embedding_length")) == Some(5120)
        && gguf
            .strings("tokenizer.ggml.tokens")
            .is_some_and(|tokens| tokens.len() == 248_320);
    if architecture == b"qwen4exp" || dense_27b {
        TextModelPreset::Qwen38
    } else {
        TextModelPreset::Unspecified
    }
}

/// [`text_preset`] of the GGUF at `path`; unreadable files have none.
pub fn text_preset_of(path: &std::path::Path) -> TextModelPreset {
    crate::gguf::read_header(path)
        .ok()
        .and_then(|(header, len)| {
            crate::gguf::Gguf::parse_prefix(&header, len)
                .ok()
                .map(|gguf| text_preset(&gguf))
        })
        .unwrap_or_default()
}

/// The model's recommended sampling for the thinking mode (`None`: the
/// model default, thinking on), with the `supplied` fields of `configured`
/// on top (`ResolveTextSampling`). Qwen3.8 thinks at temperature 1.0,
/// top-p 0.95 and top-k 20, answers directly at 0.7, 0.8, 20 with presence
/// penalty 1.5; DeepSeek V4 Flash samples at 1.0 and 0.95.
pub fn resolve_text_sampling(
    model: TextModelPreset,
    thinking: Option<bool>,
    configured: &SamplingConfig,
    supplied: &SamplingOverrides,
) -> SamplingConfig {
    if model == TextModelPreset::Unspecified {
        return configured.clone();
    }
    let mut result = SamplingConfig {
        temperature: 1.0,
        top_p: 0.95,
        ..SamplingConfig::default()
    };
    if model == TextModelPreset::Qwen38 {
        result.top_k = 20;
        if !thinking.unwrap_or(true) {
            result.temperature = 0.7;
            result.top_p = 0.8;
            result.presence_penalty = 1.5;
        }
    }
    fn pick<T>(set: bool, value: T, preset: T) -> T {
        if set { value } else { preset }
    }
    SamplingConfig {
        temperature: pick(
            supplied.temperature,
            configured.temperature,
            result.temperature,
        ),
        top_k: pick(supplied.top_k, configured.top_k, result.top_k),
        top_p: pick(supplied.top_p, configured.top_p, result.top_p),
        min_p: pick(supplied.min_p, configured.min_p, result.min_p),
        min_keep: pick(supplied.min_keep, configured.min_keep, result.min_keep),
        seed: pick(supplied.seed, configured.seed, result.seed),
        repeat_penalty: pick(
            supplied.repeat_penalty,
            configured.repeat_penalty,
            result.repeat_penalty,
        ),
        repeat_last_n: pick(
            supplied.repeat_last_n,
            configured.repeat_last_n,
            result.repeat_last_n,
        ),
        frequency_penalty: pick(
            supplied.frequency_penalty,
            configured.frequency_penalty,
            result.frequency_penalty,
        ),
        // Presets change sampling controls, never the caller's grammar or
        // reasoning budget.
        constraint: configured.constraint.clone(),
        reasoning_budget: configured.reasoning_budget.clone(),
        presence_penalty: pick(
            supplied.presence_penalty,
            configured.presence_penalty,
            result.presence_penalty,
        ),
    }
}

impl SamplingConfig {
    /// Greedy defaults with `temperature`.
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self
            .constraint
            .as_ref()
            .is_some_and(|constraint| constraint.vocabulary().is_empty())
        {
            return Err(Error::invalid("sampling constraint is incomplete"));
        }
        if self
            .reasoning_budget
            .as_ref()
            .is_some_and(|budget| budget.forced.is_empty())
        {
            return Err(Error::invalid("reasoning budget has no forced end"));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Error::invalid(
                "sampling temperature must be finite and nonnegative",
            ));
        }
        if self.top_k < 0 {
            return Err(Error::invalid("sampling top-k must be nonnegative"));
        }
        if !self.top_p.is_finite() || self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(Error::invalid("sampling top-p must be in [0, 1]"));
        }
        if !self.min_p.is_finite() || self.min_p < 0.0 || self.min_p > 1.0 {
            return Err(Error::invalid("sampling min-p must be in [0, 1]"));
        }
        if self.seed < -1 {
            return Err(Error::invalid("sampling seed must be -1 or nonnegative"));
        }
        if !self.repeat_penalty.is_finite() || self.repeat_penalty <= 0.0 {
            return Err(Error::invalid(
                "sampling repeat penalty must be finite and positive",
            ));
        }
        if !self.frequency_penalty.is_finite() {
            return Err(Error::invalid("sampling frequency penalty must be finite"));
        }
        if !self.presence_penalty.is_finite() {
            return Err(Error::invalid("sampling presence penalty must be finite"));
        }
        Ok(())
    }

    pub fn penalties_enabled(&self) -> bool {
        self.repeat_penalty != 1.0 || self.frequency_penalty != 0.0 || self.presence_penalty != 0.0
    }

    pub fn uses_random_sampling(&self) -> bool {
        self.temperature > 0.0 && (self.top_k != 1 || self.min_keep > 1)
    }

    /// Greedy output that the raw target argmax decides: no grammar, no
    /// penalties and no reasoning budget, which can force a token the target
    /// never predicts.
    pub fn can_use_unmodified_argmax(&self) -> bool {
        self.constraint.is_none()
            && self.reasoning_budget.is_none()
            && self.temperature == 0.0
            && !self.penalties_enabled()
    }
}

/// Token IDs of a logit row (`sampling::TokenId`).
pub type TokenId = u32;

/// Per-token penalty counts: the repetition window flag and the count over
/// the whole generated response (`TokenPenalty`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenPenalty {
    pub token: TokenId,
    pub generated_count: u32,
    pub repeated: u32,
}

/// One entry of a compact target row: a vocabulary ID and its original
/// logit (`LogitCandidate`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LogitCandidate {
    pub token: TokenId,
    pub value: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Probability {
    pub token: TokenId,
    pub value: f64,
}

/// `IsBetterProbability` as an ordering: higher value first, then lower
/// token. With finite values and unique tokens this is a strict total order,
/// so every sort below yields exactly the order of the C++ `std::sort`.
fn better_first(left: &Probability, right: &Probability) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if left.value == right.value {
        left.token.cmp(&right.token)
    } else if left.value > right.value {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

/// `IsBetterCandidate` over penalized logits, the same order.
type Candidate = Probability;

/// `sum > 0 && isfinite(sum)`: false for NaN too.
fn positive_finite(value: f64) -> bool {
    value > 0.0 && value.is_finite()
}

fn minimum_kept(config: &SamplingConfig, size: usize) -> usize {
    size.min(config.min_keep.max(1))
}

/// Sorts better-first and keeps the best `limit` (`std::partial_sort` plus
/// `resize`, or the bounded heap of `PrepareSelected`): with a total order
/// both select the same set.
fn keep_best(candidates: &mut Vec<Candidate>, limit: usize) {
    if limit < candidates.len() {
        if limit > 0 {
            candidates.select_nth_unstable_by(limit - 1, better_first);
        }
        candidates.truncate(limit);
    }
    candidates.sort_unstable_by(better_first);
}

fn softmax_weights(candidates: &[Candidate], temperature: f64) -> Result<Vec<f64>> {
    let Some(first) = candidates.first() else {
        return Err(Error::runtime("sampling filters removed every token"));
    };
    let maximum = first.value;
    let mut weights = Vec::with_capacity(candidates.len());
    let mut sum = 0.0;
    for candidate in candidates {
        let weight = ((candidate.value - maximum) / temperature).exp();
        weights.push(weight);
        sum += weight;
    }
    if !positive_finite(sum) {
        return Err(Error::runtime("logit softmax normalization failed"));
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    Ok(weights)
}

fn penalize(mut logit: f64, config: &SamplingConfig, penalty: &TokenPenalty) -> f64 {
    if penalty.repeated != 0 && config.repeat_penalty != 1.0 {
        let repeat_penalty = f64::from(config.repeat_penalty);
        logit = if logit <= 0.0 {
            logit * repeat_penalty
        } else {
            logit / repeat_penalty
        };
    }
    logit -= f64::from(config.frequency_penalty) * f64::from(penalty.generated_count);
    if penalty.generated_count != 0 {
        logit -= f64::from(config.presence_penalty);
    }
    logit
}

fn find_penalty(penalties: &[TokenPenalty], token: TokenId) -> Option<&TokenPenalty> {
    let index = penalties.partition_point(|p| p.token < token);
    penalties.get(index).filter(|p| p.token == token)
}

/// Walks the sorted sparse penalties alongside ascending vocabulary IDs,
/// instead of searching them for every entry.
struct PenaltyWalk<'a> {
    rest: &'a [TokenPenalty],
}

impl<'a> PenaltyWalk<'a> {
    fn new(penalties: &'a [TokenPenalty]) -> Self {
        Self { rest: penalties }
    }

    /// The penalty of `token`; calls come in ascending token order.
    fn at(&mut self, token: usize) -> Option<&'a TokenPenalty> {
        while let Some((first, rest)) = self.rest.split_first()
            && (first.token as usize) < token
        {
            self.rest = rest;
        }
        self.rest.first().filter(|p| p.token as usize == token)
    }

    /// Whether no penalty falls in `begin..end`.
    fn none_in(&mut self, begin: usize, end: usize) -> bool {
        self.at(begin);
        self.rest.first().is_none_or(|p| p.token as usize >= end)
    }
}

/// The vocabulary block the samplers test as a whole before visiting its
/// tokens.
const SKIP_BLOCK: usize = 64;

/// The largest value of a block, ignoring NaN; +inf counts, so such a block
/// takes the exact per-token path.
fn block_maximum(block: &[f32]) -> f32 {
    let larger = |a: f32, b: f32| if b > a { b } else { a };
    if let Ok(block) = <&[f32; SKIP_BLOCK]>::try_from(block) {
        // Independent lanes with a fixed trip count vectorize.
        let mut lanes = [f32::NEG_INFINITY; 8];
        for chunk in block.as_chunks::<8>().0 {
            for (lane, &value) in lanes.iter_mut().zip(chunk) {
                *lane = larger(*lane, value);
            }
        }
        return lanes.into_iter().fold(f32::NEG_INFINITY, larger);
    }
    block
        .iter()
        .fold(f32::NEG_INFINITY, |a, &value| larger(a, value))
}

fn apply_penalties(
    candidates: &mut [Candidate],
    config: &SamplingConfig,
    penalties: &[TokenPenalty],
) {
    if penalties.is_empty() {
        return;
    }
    for candidate in candidates {
        if let Some(penalty) = find_penalty(penalties, candidate.token) {
            candidate.value = penalize(candidate.value, config, penalty);
        }
    }
}

fn apply_top_k(candidates: &mut Vec<Candidate>, config: &SamplingConfig) {
    if config.top_k <= 0 {
        candidates.sort_unstable_by(better_first);
        return;
    }
    let configured = config.top_k as usize;
    let keep = candidates
        .len()
        .min(configured.max(minimum_kept(config, candidates.len())));
    keep_best(candidates, keep);
}

fn apply_top_p(candidates: &mut Vec<Candidate>, config: &SamplingConfig) -> Result<()> {
    if config.top_p >= 1.0 || candidates.len() <= 1 {
        return Ok(());
    }
    let probabilities = softmax_weights(candidates, f64::from(config.temperature))?;
    let minimum = minimum_kept(config, candidates.len());
    let mut cumulative = 0.0;
    let mut keep = 0;
    while keep < candidates.len() {
        cumulative += probabilities[keep];
        keep += 1;
        if keep >= minimum && cumulative >= f64::from(config.top_p) {
            break;
        }
    }
    candidates.truncate(keep);
    Ok(())
}

fn apply_min_p(candidates: &mut Vec<Candidate>, config: &SamplingConfig) {
    if config.min_p <= 0.0 || candidates.len() <= 1 {
        return;
    }
    let threshold = f64::from(config.temperature) * f64::from(config.min_p).ln();
    let maximum = candidates[0].value;
    let mut keep = 0;
    while keep < candidates.len() && candidates[keep].value - maximum >= threshold {
        keep += 1;
    }
    keep = keep.max(minimum_kept(config, candidates.len()));
    candidates.truncate(keep.min(candidates.len()));
}

/// A nondeterministic 64-bit seed for `seed == -1`. C++ combines two
/// `std::random_device` words; here the randomly keyed standard hasher
/// supplies them. Either way the value is unpredictable, never reproduced.
fn random_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let words: [u64; 2] = std::array::from_fn(|index| {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_usize(index);
        if let Ok(now) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            hasher.write_u128(now.as_nanos());
        }
        hasher.finish() & 0xffff_ffff
    });
    (words[0] << 32) ^ words[1]
}

fn initial_rng_state(seed: i64) -> u64 {
    if seed >= 0 {
        seed as u64
    } else {
        random_seed()
    }
}

/// A normalized, post-filter sampling distribution
/// (`SamplingDistribution`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingDistribution {
    entries: Vec<Probability>,
}

impl SamplingDistribution {
    /// Validates, normalizes, drops zero entries and sorts better-first.
    pub fn new(mut entries: Vec<Probability>) -> Result<Self> {
        if entries.is_empty() {
            return Err(Error::invalid("sampling distribution cannot be empty"));
        }
        let mut tokens = std::collections::HashSet::with_capacity(entries.len());
        let mut sum = 0.0;
        for entry in &entries {
            if !entry.value.is_finite() || entry.value < 0.0 || !tokens.insert(entry.token) {
                return Err(Error::invalid("sampling distribution is malformed"));
            }
            sum += entry.value;
        }
        if !positive_finite(sum) {
            return Err(Error::invalid("sampling distribution has no probability"));
        }
        for entry in &mut entries {
            entry.value /= sum;
        }
        entries.retain(|entry| entry.value != 0.0);
        entries.sort_unstable_by(better_first);
        Ok(Self { entries })
    }

    /// Internally generated entries have unique tokens: normalize by
    /// `total`, drop zeros and keep their order.
    fn with_total(mut entries: Vec<Probability>, total: f64) -> Result<Self> {
        if !positive_finite(total) {
            return Err(Error::runtime("logit softmax normalization failed"));
        }
        for entry in &mut entries {
            entry.value /= total;
        }
        entries.retain(|entry| entry.value != 0.0);
        Ok(Self { entries })
    }

    pub fn entries(&self) -> &[Probability] {
        &self.entries
    }

    pub fn best_token(&self) -> Result<TokenId> {
        self.entries
            .iter()
            .min_by(|a, b| better_first(a, b))
            .map(|entry| entry.token)
            .ok_or_else(|| Error::logic("sampling distribution is empty"))
    }

    pub fn probability(&self, token: TokenId) -> f64 {
        self.entries
            .iter()
            .find(|entry| entry.token == token)
            .map_or(0.0, |entry| entry.value)
    }

    /// Inverse-CDF draw over the entries in their order.
    pub fn sample(&self, rng_state: &mut u64) -> Result<TokenId> {
        let (Some(first), Some(last)) = (self.entries.first(), self.entries.last()) else {
            return Err(Error::logic("sampling distribution is empty"));
        };
        if self.entries.len() == 1 {
            return Ok(first.token);
        }
        let mut sample = uniform(rng_state);
        for entry in &self.entries {
            sample -= entry.value;
            if sample < 0.0 {
                return Ok(entry.token);
            }
        }
        Ok(last.token)
    }

    /// Draws from `max(p - q, 0)` for the sparse draft row `q`, or from `p`
    /// when that residual has no mass.
    pub fn sample_residual(
        &self,
        candidate_ids: &[TokenId],
        candidate_probabilities: &[f32],
        rng_state: &mut u64,
    ) -> Result<TokenId> {
        if candidate_ids.len() != candidate_probabilities.len() {
            return Err(Error::invalid("sparse probability row is malformed"));
        }
        let mut draft: std::collections::HashMap<TokenId, f64> =
            std::collections::HashMap::with_capacity(candidate_ids.len());
        for (&token, &probability) in candidate_ids.iter().zip(candidate_probabilities) {
            if !probability.is_finite() || probability < 0.0 {
                return Err(Error::invalid(
                    "sparse probability row contains an invalid value",
                ));
            }
            *draft.entry(token).or_insert(0.0) += f64::from(probability);
        }
        let mut residual = Vec::with_capacity(self.entries.len());
        let mut sum = 0.0;
        for entry in &self.entries {
            let q = draft.get(&entry.token).copied().unwrap_or(0.0);
            let value = (entry.value - q).max(0.0);
            if value > 0.0 {
                residual.push(Probability {
                    token: entry.token,
                    value,
                });
                sum += value;
            }
        }
        if !positive_finite(sum) {
            return self.sample(rng_state);
        }
        Self::with_total(residual, sum)?.sample(rng_state)
    }
}

fn check_vocabulary(logits: &[f32]) -> Result<()> {
    if logits.is_empty() {
        return Err(Error::invalid("cannot sample an empty logit distribution"));
    }
    if logits.len() > TokenId::MAX as usize {
        return Err(Error::invalid("logit distribution exceeds token ID range"));
    }
    Ok(())
}

/// The reference filter chain over sorted penalized logits
/// (`DistributionWithPenalties`).
fn distribution_with_penalties(
    logits: &[f32],
    config: &SamplingConfig,
    penalties: &[TokenPenalty],
) -> Result<SamplingDistribution> {
    config.validate()?;
    check_vocabulary(logits)?;
    let mut candidates: Vec<Candidate> = logits
        .iter()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .map(|(index, &logit)| Candidate {
            token: index as TokenId,
            value: f64::from(logit),
        })
        .collect();
    if candidates.is_empty() {
        return Err(Error::runtime(
            "logit distribution contains no finite values",
        ));
    }
    apply_penalties(&mut candidates, config, penalties);
    apply_top_k(&mut candidates, config);
    if config.temperature == 0.0 {
        return SamplingDistribution::new(vec![Probability {
            token: candidates[0].token,
            value: 1.0,
        }]);
    }
    apply_top_p(&mut candidates, config)?;
    apply_min_p(&mut candidates, config);
    let probabilities = softmax_weights(&candidates, f64::from(config.temperature))?;
    let entries = candidates
        .iter()
        .zip(probabilities)
        .map(|(candidate, value)| Probability {
            token: candidate.token,
            value,
        })
        .collect();
    SamplingDistribution::new(entries)
}

/// The exact distribution from prompt and request-generated tokens
/// (`BuildDistribution`).
pub fn build_distribution(
    logits: &[f32],
    config: &SamplingConfig,
    prompt_tokens: &[TokenId],
    generated_tokens: &[TokenId],
) -> Result<SamplingDistribution> {
    if config.constraint.is_some() || config.reasoning_budget.is_some() {
        let mut sampler = SamplerState::new(config.clone(), prompt_tokens)?;
        sampler.accept_tokens(generated_tokens)?;
        return sampler.distribution(logits);
    }
    let mut counts: std::collections::HashMap<TokenId, TokenPenalty> =
        std::collections::HashMap::new();
    if config.frequency_penalty != 0.0 || config.presence_penalty != 0.0 {
        for &token in generated_tokens {
            let penalty = counts.entry(token).or_default();
            penalty.token = token;
            penalty.generated_count += 1;
        }
    }
    if config.repeat_penalty != 1.0 && config.repeat_last_n != 0 {
        let tail = &generated_tokens
            [generated_tokens.len() - generated_tokens.len().min(config.repeat_last_n)..];
        let prompt_count = prompt_tokens.len().min(config.repeat_last_n - tail.len());
        let prompt_tail = &prompt_tokens[prompt_tokens.len() - prompt_count..];
        for &token in prompt_tail.iter().chain(tail) {
            let penalty = counts.entry(token).or_default();
            penalty.token = token;
            penalty.repeated = 1;
        }
    }
    let mut penalties: Vec<TokenPenalty> = counts.into_values().collect();
    penalties.sort_unstable_by_key(|p| p.token);
    distribution_with_penalties(logits, config, &penalties)
}

/// The RNG and deferred draw of a sampler (`SamplerState::DrawState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawState {
    pub rng: u64,
    pub pending: Option<TokenId>,
}

/// Request-owned controls, repetition window, full generated counts and RNG
/// (`SamplerState`). Call `accept` only for committed tokens; clones let
/// speculative decoding work on a tentative state and drop it on rollback.
/// The token mask of the current grammar state and whether it forbids any
/// token, computed on first use (`constraint_mask_`). It caches what the
/// state determines, so it never makes two samplers differ.
#[derive(Debug, Clone, Default)]
struct MaskCache(OnceLock<(Arc<Vec<bool>>, bool)>);

impl PartialEq for MaskCache {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// Where a reasoning budget stands (`ReasoningPhase`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ReasoningPhase {
    #[default]
    Off,
    Counting,
    /// The budget is spent; the next token that completes its UTF-8
    /// character starts the forced end.
    WaitingUtf8,
    Forcing,
}

/// A reasoning budget's progress: the tokens left and the forced tokens
/// already emitted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ReasoningProgress {
    phase: ReasoningPhase,
    remaining: usize,
    forced: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplerState {
    config: SamplingConfig,
    /// Where the grammar stands after the accepted tokens, and where each
    /// request starts; empty without a constraint.
    constraint_state: ConstraintState,
    constraint_start: ConstraintState,
    constraint_mask: MaskCache,
    history: Vec<TokenId>,
    penalty_counts: Vec<TokenPenalty>,
    rng_state: u64,
    pending_sample: Option<TokenId>,
    reasoning: ReasoningProgress,
}

impl Default for SamplerState {
    fn default() -> Self {
        // The default configuration is valid.
        Self::new(SamplingConfig::default(), &[]).unwrap_or(Self {
            config: SamplingConfig::default(),
            constraint_state: ConstraintState::new(),
            constraint_start: ConstraintState::new(),
            constraint_mask: MaskCache::default(),
            history: Vec::new(),
            penalty_counts: Vec::new(),
            rng_state: 0,
            pending_sample: None,
            reasoning: ReasoningProgress::default(),
        })
    }
}

impl crate::text::Sampler for SamplerState {
    fn accept(&mut self, token: crate::text::Token) {
        // Fails only after 2^32 - 1 generated occurrences of one token; the
        // count then stays saturated as in C++.
        let _ = SamplerState::accept(self, token);
    }
}

impl SamplerState {
    /// `initial_history` seeds the repetition window only.
    pub fn new(config: SamplingConfig, initial_history: &[TokenId]) -> Result<Self> {
        let rng_state = initial_rng_state(config.seed);
        let mut state = Self {
            config,
            constraint_state: ConstraintState::new(),
            constraint_start: ConstraintState::new(),
            constraint_mask: MaskCache::default(),
            history: initial_history.to_vec(),
            penalty_counts: Vec::new(),
            rng_state,
            pending_sample: None,
            reasoning: ReasoningProgress::default(),
        };
        state.config.validate()?;
        if let Some(constraint) = &state.config.constraint {
            state.constraint_start = constraint.grammar().start()?;
            state.constraint_state = state.constraint_start.clone();
        }
        state.start_reasoning();
        state.trim_history();
        state.rebuild_penalty_counts();
        Ok(state)
    }

    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// This sampler without its grammar: a draft's proposal distribution q
    /// may stay unconstrained, the target's p never does.
    pub fn without_constraint(&self) -> Self {
        let mut copy = self.clone();
        copy.config.constraint = None;
        copy.constraint_state.clear();
        copy.constraint_start.clear();
        copy.constraint_mask = MaskCache::default();
        copy
    }

    /// Whether the grammar forbids any token now (`NeedsConstraintMask`): a
    /// lazy tool grammar admits the whole vocabulary before a call, and
    /// ordinary CPU or device sampling then stays valid.
    /// Also true while [`forced_token`](Self::forced_token) is set, so
    /// device samplers defer to the host, which returns the forced end.
    pub fn needs_constraint_mask(&self) -> Result<bool> {
        if self.forced_token().is_some() {
            return Ok(true);
        }
        Ok(self
            .constraint_mask()?
            .is_some_and(|(_, restricts)| restricts))
    }

    /// The token every sampling path returns next, if any (`ForcedToken`):
    /// a spent reasoning budget forces its end sequence.
    pub fn forced_token(&self) -> Option<TokenId> {
        let budget = self.config.reasoning_budget.as_ref()?;
        (self.reasoning.phase == ReasoningPhase::Forcing)
            .then(|| budget.forced[self.reasoning.forced])
    }

    /// Restarts the budget, counting the reasoning already in the prompt.
    fn start_reasoning(&mut self) {
        self.reasoning = ReasoningProgress::default();
        let Some(budget) = self.config.reasoning_budget.clone() else {
            return;
        };
        self.reasoning = ReasoningProgress {
            phase: if budget.tokens == 0 {
                ReasoningPhase::Forcing
            } else {
                ReasoningPhase::Counting
            },
            remaining: budget.tokens,
            forced: 0,
        };
        for &token in &budget.prompt {
            self.advance_reasoning(token);
        }
    }

    /// One accepted token (`AdvanceReasoning`, llama.cpp's
    /// `common_reasoning_budget_accept`): forcing advances one forced token
    /// per accepted token, and a spent budget waits for a token that
    /// completes its UTF-8 character.
    fn advance_reasoning(&mut self, token: TokenId) {
        let Some(budget) = self.config.reasoning_budget.as_deref() else {
            return;
        };
        let progress = &mut self.reasoning;
        match progress.phase {
            ReasoningPhase::Off => return,
            ReasoningPhase::Forcing => {
                progress.forced += 1;
                if progress.forced >= budget.forced.len() {
                    progress.phase = ReasoningPhase::Off;
                }
                return;
            }
            ReasoningPhase::Counting | ReasoningPhase::WaitingUtf8 => {}
        }
        if budget.ends.contains(&token) {
            progress.phase = ReasoningPhase::Off;
            return;
        }
        let complete = budget
            .partial_utf8
            .as_ref()
            .is_none_or(|partial| !partial.get(token as usize).copied().unwrap_or(false));
        if progress.phase == ReasoningPhase::Counting {
            progress.remaining -= 1;
            if progress.remaining != 0 {
                return;
            }
        }
        if complete {
            progress.phase = ReasoningPhase::Forcing;
            progress.forced = 0;
        } else {
            progress.phase = ReasoningPhase::WaitingUtf8;
        }
    }

    /// The cached mask of the current state, and whether it restricts.
    fn constraint_mask(&self) -> Result<Option<(Arc<Vec<bool>>, bool)>> {
        let Some(constraint) = &self.config.constraint else {
            return Ok(None);
        };
        if let Some(cached) = self.constraint_mask.0.get() {
            return Ok(Some(cached.clone()));
        }
        let mask = constraint.allowed(&self.constraint_state)?;
        let restricts = mask.iter().any(|&allowed| !allowed);
        let _ = self.constraint_mask.0.set((mask.clone(), restricts));
        Ok(Some((mask, restricts)))
    }

    /// Whether `token`, a finite argmax of the target, can be taken without
    /// downloading its logits (`CanSelectArgmax`): greedy, nothing pending,
    /// and legal under the grammar. `penalties_applied` only for a winner
    /// computed with this sampler's exact penalty arithmetic and history;
    /// raw model argmaxes leave it false.
    pub fn can_select_argmax(&self, token: TokenId, penalties_applied: bool) -> Result<bool> {
        if let Some(forced) = self.forced_token() {
            return Ok(self.pending_sample.is_none() && token == forced);
        }
        if self.pending_sample.is_some()
            || self.config.temperature != 0.0
            || (!penalties_applied && self.config.penalties_enabled())
        {
            return Ok(false);
        }
        let Some(constraint) = &self.config.constraint else {
            return Ok(true);
        };
        if token as usize >= constraint.vocabulary().len() {
            return Err(Error::invalid(
                "target argmax exceeds the constraint vocabulary",
            ));
        }
        // Closed JSON outputs keep their full masks, which later sampled
        // requests reuse; tool and prose grammars can take a legal argmax.
        Ok(!constraint.grammar().stop_only_when_complete()
            && constraint.vocabulary().allows(
                constraint.grammar(),
                &self.constraint_state,
                token,
            )?)
    }

    /// `logits` with the tokens the grammar forbids at negative infinity;
    /// `ids` name the tokens of compact logits (`ConstrainedLogits`).
    fn constrained_logits(
        &self,
        constraint: &TokenConstraint,
        logits: &[f32],
        ids: &[TokenId],
    ) -> Result<Vec<f32>> {
        let mask = match self.constraint_mask()? {
            Some((mask, _)) => mask,
            None => constraint.allowed(&self.constraint_state)?,
        };
        if (!ids.is_empty() && ids.len() != logits.len())
            || (ids.is_empty() && logits.len() != mask.len())
        {
            return Err(Error::invalid(
                "JSON constraint vocabulary differs from logits",
            ));
        }
        let mut masked = logits.to_vec();
        for (index, value) in masked.iter_mut().enumerate() {
            let token = if ids.is_empty() {
                index
            } else {
                ids[index] as usize
            };
            match mask.get(token) {
                Some(true) => {}
                Some(false) => *value = f32::NEG_INFINITY,
                None => return Err(Error::invalid("invalid compact token ID")),
            }
        }
        Ok(masked)
    }

    pub fn history(&self) -> &[TokenId] {
        &self.history
    }

    pub fn penalties(&self) -> &[TokenPenalty] {
        &self.penalty_counts
    }

    pub fn rng_state(&self) -> u64 {
        self.rng_state
    }

    pub fn mutable_rng_state(&mut self) -> &mut u64 {
        &mut self.rng_state
    }

    /// Sets the RNG and discards any deferred draw.
    pub fn set_rng_state(&mut self, state: u64) {
        self.rng_state = state;
        self.pending_sample = None;
    }

    /// Publishes RNG and deferred draw without accepting tentative history.
    pub fn copy_draw_state_from(&mut self, other: &SamplerState) {
        self.rng_state = other.rng_state;
        self.pending_sample = other.pending_sample;
    }

    pub fn save_draw_state(&self) -> DrawState {
        DrawState {
            rng: self.rng_state,
            pending: self.pending_sample,
        }
    }

    pub fn restore_draw_state(&mut self, state: DrawState) {
        self.rng_state = state.rng;
        self.pending_sample = state.pending;
    }

    /// Starts a request: `tokens` initialize the repetition window only;
    /// generated counts reset.
    pub fn reset_history(&mut self, tokens: &[TokenId]) {
        self.constraint_state = self.constraint_start.clone();
        self.constraint_mask = MaskCache::default();
        self.pending_sample = None;
        self.start_reasoning();
        self.penalty_counts.clear();
        self.history = tokens.to_vec();
        self.trim_history();
        self.rebuild_penalty_counts();
    }

    pub fn accept(&mut self, token: TokenId) -> Result<()> {
        self.accept_tokens(&[token])
    }

    pub fn accept_tokens(&mut self, tokens: &[TokenId]) -> Result<()> {
        if self.reasoning.phase != ReasoningPhase::Off {
            for &token in tokens {
                self.advance_reasoning(token);
            }
        }
        if let Some(constraint) = &self.config.constraint {
            let mut next = self.constraint_state.clone();
            for &token in tokens {
                next = constraint
                    .vocabulary()
                    .accept(constraint.grammar(), &next, token)?;
            }
            if next != self.constraint_state {
                self.constraint_state = next;
                self.constraint_mask = MaskCache::default();
            }
        }
        if self.config.frequency_penalty != 0.0 || self.config.presence_penalty != 0.0 {
            for &token in tokens {
                let index = self.penalty_counts.partition_point(|p| p.token < token);
                if self
                    .penalty_counts
                    .get(index)
                    .is_none_or(|p| p.token != token)
                {
                    self.penalty_counts.insert(
                        index,
                        TokenPenalty {
                            token,
                            ..TokenPenalty::default()
                        },
                    );
                }
                let penalty = &mut self.penalty_counts[index];
                if penalty.generated_count == u32::MAX {
                    return Err(Error::runtime("generated token count overflow"));
                }
                penalty.generated_count += 1;
            }
        }
        if self.config.repeat_last_n != 0 {
            self.history.extend_from_slice(tokens);
            self.trim_history();
        }
        self.rebuild_penalty_counts();
        Ok(())
    }

    /// The distribution `sample` draws from.
    pub fn distribution(&self, logits: &[f32]) -> Result<SamplingDistribution> {
        if let Some(forced) = self.forced_token() {
            if forced as usize >= logits.len() {
                return Err(Error::invalid("forced token exceeds vocabulary"));
            }
            return SamplingDistribution::with_total(
                vec![Probability {
                    token: forced,
                    value: 1.0,
                }],
                1.0,
            );
        }
        if let Some(constraint) = &self.config.constraint
            && self.needs_constraint_mask()?
        {
            if self.config.temperature == 0.0 {
                self.config.validate()?;
                return SamplingDistribution::with_total(
                    vec![Probability {
                        token: self.sample_constrained_greedy(constraint, logits)?,
                        value: 1.0,
                    }],
                    1.0,
                );
            }
            let masked = self.constrained_logits(constraint, logits, &[])?;
            return self.without_constraint().distribution(&masked);
        }
        self.config.validate()?;
        if logits.is_empty() || logits.len() > TokenId::MAX as usize {
            return Err(Error::invalid("invalid sampling vocabulary size"));
        }
        if self.config.temperature == 0.0 {
            return SamplingDistribution::with_total(
                vec![Probability {
                    token: self.sample_greedy(logits)?,
                    value: 1.0,
                }],
                1.0,
            );
        }
        let needs_floor = self.config.min_p > 0.0 && self.config.min_keep > 1;
        if self.config.top_k == 0 && self.config.top_p == 1.0 && !needs_floor {
            return self.linear_distribution(logits);
        }
        let candidates = self.prepare_selected(logits)?;
        self.selected_distribution(candidates)
    }

    /// The exact target top-k from a complete candidate union
    /// (`DistributionFromCandidates`): the caller proves that omitted tokens
    /// cannot enter the post-penalty top-k. Candidates have unique vocabulary
    /// IDs, which the returned distribution keeps. Requires a bounded top-k,
    /// a positive temperature and no active constraint, and keeps the
    /// full-row sampler's FP64 filtering and normalization.
    pub fn distribution_from_candidates(
        &self,
        logits: &[LogitCandidate],
    ) -> Result<SamplingDistribution> {
        self.config.validate()?;
        if self.config.temperature <= 0.0
            || self.config.top_k <= 0
            || self.needs_constraint_mask()?
        {
            return Err(Error::invalid(
                "compact target requires unconstrained top-k",
            ));
        }
        let mut candidates = Vec::with_capacity(logits.len());
        for candidate in logits {
            if !candidate.value.is_finite() {
                continue;
            }
            let value = match find_penalty(&self.penalty_counts, candidate.token) {
                Some(penalty) => penalize(f64::from(candidate.value), &self.config, penalty),
                None => f64::from(candidate.value),
            };
            if !value.is_finite() {
                return Err(Error::runtime(
                    "sampling penalties produced a non-finite logit",
                ));
            }
            candidates.push(Candidate {
                token: candidate.token,
                value,
            });
        }
        if candidates.is_empty() {
            return Err(Error::runtime(
                "logit distribution contains no finite values",
            ));
        }
        let keep = candidates
            .len()
            .min((self.config.top_k as usize).max(self.config.min_keep.max(1)));
        candidates.sort_unstable_by(better_first);
        candidates.truncate(keep);
        self.filter_selected(&mut candidates, 0.0, false);
        self.selected_distribution(candidates)
    }

    /// The normalized distribution over filtered candidates, better-first
    /// with their penalized logits (`SelectedDistribution`).
    fn selected_distribution(
        &self,
        mut candidates: Vec<Candidate>,
    ) -> Result<SamplingDistribution> {
        let maximum = candidates[0].value;
        let temperature = f64::from(self.config.temperature);
        let mut total = 0.0;
        for candidate in &mut candidates {
            candidate.value = ((candidate.value - maximum) / temperature).exp();
            total += candidate.value;
        }
        SamplingDistribution::with_total(candidates, total)
    }

    /// Compact logits: `token_ids` select penalties, returned IDs index
    /// `logits`.
    pub fn distribution_compact(
        &self,
        logits: &[f32],
        token_ids: &[TokenId],
    ) -> Result<SamplingDistribution> {
        if let Some(forced) = self.forced_token() {
            if let Some(index) = token_ids.iter().position(|&token| token == forced) {
                return SamplingDistribution::with_total(
                    vec![Probability {
                        token: index as TokenId,
                        value: 1.0,
                    }],
                    1.0,
                );
            }
            // An MTP draft shortlist may miss the forced end: the proposal
            // then follows the shortlist, and the target's forced
            // distribution rejects it and emits the end as its correction.
            let mut unforced = self.clone();
            unforced.reasoning.phase = ReasoningPhase::Off;
            return unforced.distribution_compact(logits, token_ids);
        }
        if let Some(constraint) = &self.config.constraint
            && self.needs_constraint_mask()?
        {
            let masked = self.constrained_logits(constraint, logits, token_ids)?;
            return self
                .without_constraint()
                .distribution_compact(&masked, token_ids);
        }
        if logits.len() != token_ids.len() {
            return Err(Error::invalid(
                "compact logits and token IDs differ in size",
            ));
        }
        let penalties: Vec<TokenPenalty> = token_ids
            .iter()
            .enumerate()
            .filter_map(|(index, &token)| {
                find_penalty(&self.penalty_counts, token).map(|penalty| TokenPenalty {
                    token: index as TokenId,
                    ..*penalty
                })
            })
            .collect();
        distribution_with_penalties(logits, &self.config, &penalties)
    }

    /// Retains a speculative residual draw for the next `sample`; clones
    /// keep it with the RNG, resetting history discards it.
    pub fn defer_sample(&mut self, token: TokenId) -> Result<()> {
        if self.pending_sample.is_some() {
            return Err(Error::logic("a sampled token is already pending"));
        }
        self.pending_sample = Some(token);
        Ok(())
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<TokenId> {
        self.sample_with_argmax(logits, None)
    }

    /// [`sample`](Self::sample) with `argmax`, the first index of the largest
    /// finite logit when known (a device argmax): greedy sampling without
    /// penalties takes it instead of scanning the logits.
    pub fn sample_with_argmax(
        &mut self,
        logits: &[f32],
        argmax: Option<TokenId>,
    ) -> Result<TokenId> {
        self.config.validate()?;
        check_vocabulary(logits)?;
        if let Some(forced) = self.forced_token() {
            if forced as usize >= logits.len() {
                return Err(Error::invalid("forced token exceeds vocabulary"));
            }
            self.pending_sample = None;
            return Ok(forced);
        }
        if let Some(token) = self.pending_sample {
            if token as usize >= logits.len() {
                return Err(Error::invalid("pending sample exceeds vocabulary"));
            }
            if let Some(constraint) = &self.config.constraint
                && !constraint
                    .allowed(&self.constraint_state)?
                    .get(token as usize)
                    .copied()
                    .unwrap_or(false)
            {
                return Err(Error::runtime("pending sample violates JSON constraint"));
            }
            self.pending_sample = None;
            return Ok(token);
        }
        if self.config.constraint.is_some() {
            let distribution = self.distribution(logits)?;
            return if self.config.temperature == 0.0 {
                distribution.best_token()
            } else {
                distribution.sample(&mut self.rng_state)
            };
        }
        if self.config.temperature == 0.0 {
            if let Some(token) = argmax
                && self.penalty_counts.is_empty()
                && (token as usize) < logits.len()
            {
                return Ok(token);
            }
            return self.sample_greedy(logits);
        }
        let distribution = self.distribution(logits)?;
        distribution.sample(&mut self.rng_state)
    }

    pub fn sample_residual(
        &mut self,
        target_logits: &[f32],
        candidate_ids: &[TokenId],
        candidate_probabilities: &[f32],
    ) -> Result<TokenId> {
        let distribution = self.distribution(target_logits)?;
        distribution.sample_residual(candidate_ids, candidate_probabilities, &mut self.rng_state)
    }

    pub fn uniform(&mut self) -> f64 {
        uniform(&mut self.rng_state)
    }

    fn trim_history(&mut self) {
        if self.config.repeat_last_n == 0 {
            self.history.clear();
            return;
        }
        if self.history.len() > self.config.repeat_last_n {
            let excess = self.history.len() - self.config.repeat_last_n;
            self.history.drain(..excess);
        }
    }

    fn rebuild_penalty_counts(&mut self) {
        if !self.config.penalties_enabled() {
            return;
        }
        for penalty in &mut self.penalty_counts {
            penalty.repeated = 0;
        }
        if self.config.repeat_penalty != 1.0 {
            for &token in &self.history {
                let index = self.penalty_counts.partition_point(|p| p.token < token);
                if self
                    .penalty_counts
                    .get(index)
                    .is_none_or(|p| p.token != token)
                {
                    self.penalty_counts.insert(
                        index,
                        TokenPenalty {
                            token,
                            ..TokenPenalty::default()
                        },
                    );
                }
                self.penalty_counts[index].repeated = 1;
            }
        }
        self.penalty_counts
            .retain(|p| p.generated_count != 0 || p.repeated != 0);
    }

    /// The penalized value of the finite `logit` at `index`, the walk
    /// supplying its penalty; fails on a non-finite result.
    fn walk_adjusted(&self, walk: &mut PenaltyWalk<'_>, index: usize, logit: f32) -> Result<f64> {
        let Some(penalty) = walk.at(index) else {
            return Ok(f64::from(logit));
        };
        let adjusted = penalize(f64::from(logit), &self.config, penalty);
        if !adjusted.is_finite() {
            return Err(Error::runtime(
                "sampling penalties produced a non-finite logit",
            ));
        }
        Ok(adjusted)
    }

    /// Visits every finite logit with its penalized value, in token order.
    fn for_each_adjusted(&self, logits: &[f32], mut visit: impl FnMut(usize, f64)) -> Result<()> {
        let mut walk = PenaltyWalk::new(&self.penalty_counts);
        for (index, &logit) in logits.iter().enumerate() {
            if logit.is_finite() {
                visit(index, self.walk_adjusted(&mut walk, index, logit)?);
            }
        }
        Ok(())
    }

    /// The best `limit` finite candidates with penalties, better-first, from
    /// one pass with a bounded heap. Once the heap is full, an unpenalized
    /// block whose largest logit cannot beat the weakest kept entry is
    /// skipped: every kept entry has a lower token ID, so an equal later
    /// value loses the tie as well (`select_best` of `PrepareSelected`).
    fn select_best_penalized(&self, logits: &[f32], limit: usize) -> Result<Vec<Candidate>> {
        let mut walk = PenaltyWalk::new(&self.penalty_counts);
        let mut kept: std::collections::BinaryHeap<Kept> =
            std::collections::BinaryHeap::with_capacity(limit + 1);
        for (block_index, block) in logits.chunks(SKIP_BLOCK).enumerate() {
            let begin = block_index * SKIP_BLOCK;
            if kept.len() == limit
                && walk.none_in(begin, begin + block.len())
                && let Some(worst) = kept.peek()
                && f64::from(block_maximum(block)) <= worst.0.value
            {
                continue;
            }
            for (offset, &logit) in block.iter().enumerate() {
                if !logit.is_finite() {
                    continue;
                }
                let index = begin + offset;
                let candidate = Candidate {
                    token: index as TokenId,
                    value: self.walk_adjusted(&mut walk, index, logit)?,
                };
                if kept.len() < limit {
                    kept.push(Kept(candidate));
                } else if let Some(mut worst) = kept.peek_mut()
                    && better_first(&candidate, &worst.0) == std::cmp::Ordering::Less
                {
                    *worst = Kept(candidate);
                }
            }
        }
        Ok(into_sorted(kept))
    }

    /// The greedy token among those the grammar admits, reading the mask
    /// directly rather than masking a copy of the logits; ties keep the
    /// lowest token ID (`SampleConstrainedGreedy`).
    fn sample_constrained_greedy(
        &self,
        constraint: &TokenConstraint,
        logits: &[f32],
    ) -> Result<TokenId> {
        let mask = constraint.allowed(&self.constraint_state)?;
        if logits.len() != mask.len() {
            return Err(Error::invalid(
                "JSON constraint vocabulary differs from logits",
            ));
        }
        let mut best_logit = f64::NEG_INFINITY;
        let mut best = None;
        let mut walk = PenaltyWalk::new(&self.penalty_counts);
        for (index, (&logit, &allowed)) in logits.iter().zip(mask.iter()).enumerate() {
            if !allowed || !logit.is_finite() {
                continue;
            }
            let value = self.walk_adjusted(&mut walk, index, logit)?;
            if value > best_logit {
                best_logit = value;
                best = Some(index as TokenId);
            }
        }
        best.ok_or_else(|| Error::runtime("logit distribution contains no finite values"))
    }

    fn sample_greedy(&self, logits: &[f32]) -> Result<TokenId> {
        let no_finite = || Error::runtime("logit distribution contains no finite values");
        if self.penalty_counts.is_empty() {
            return first_finite_max(logits)
                .map(|index| index as TokenId)
                .ok_or_else(no_finite);
        }
        let mut best_logit = f64::NEG_INFINITY;
        let mut best = None;
        let mut walk = PenaltyWalk::new(&self.penalty_counts);
        for (block_index, block) in logits.chunks(SKIP_BLOCK).enumerate() {
            let begin = block_index * SKIP_BLOCK;
            // An unpenalized block that cannot exceed the best value is
            // skipped; an equal later value never replaces the earlier token.
            if best.is_some()
                && walk.none_in(begin, begin + block.len())
                && f64::from(block_maximum(block)) <= best_logit
            {
                continue;
            }
            for (offset, &logit) in block.iter().enumerate() {
                if !logit.is_finite() {
                    continue;
                }
                let index = begin + offset;
                let adjusted = self.walk_adjusted(&mut walk, index, logit)?;
                if best.is_none() || adjusted > best_logit {
                    best_logit = adjusted;
                    best = Some(index as TokenId);
                }
            }
        }
        best.ok_or_else(no_finite)
    }

    /// Full-vocabulary softmax with the min-p threshold, in token order
    /// (`LinearDistribution`).
    fn linear_distribution(&self, logits: &[f32]) -> Result<SamplingDistribution> {
        let mut candidates = Vec::with_capacity(logits.len());
        let mut maximum = f64::NEG_INFINITY;
        self.for_each_adjusted(logits, |index, value| {
            maximum = maximum.max(value);
            candidates.push(Probability {
                token: index as TokenId,
                value,
            });
        })?;
        if candidates.is_empty() {
            return Err(Error::runtime(
                "logit distribution contains no finite values",
            ));
        }
        let temperature = f64::from(self.config.temperature);
        let threshold = if self.config.min_p > 0.0 {
            maximum + temperature * f64::from(self.config.min_p).ln()
        } else {
            f64::NEG_INFINITY
        };
        let mut total = 0.0;
        for candidate in &mut candidates {
            candidate.value = if candidate.value >= threshold {
                ((candidate.value - maximum) / temperature).exp()
            } else {
                0.0
            };
            total += candidate.value;
        }
        SamplingDistribution::with_total(candidates, total)
    }

    /// The filtered candidates better-first, their values still penalized
    /// logits (`PrepareSelected`).
    fn prepare_selected(&self, logits: &[f32]) -> Result<Vec<Candidate>> {
        let temperature = f64::from(self.config.temperature);
        let select_all = || -> Result<Vec<Candidate>> {
            let mut candidates = Vec::with_capacity(logits.len());
            self.for_each_adjusted(logits, |index, value| {
                candidates.push(Candidate {
                    token: index as TokenId,
                    value,
                });
            })?;
            candidates.sort_unstable_by(better_first);
            Ok(candidates)
        };
        let select_best = |limit: usize| {
            if self.penalty_counts.is_empty() {
                Ok(select_best_plain(logits, limit))
            } else {
                self.select_best_penalized(logits, limit)
            }
        };

        let mut full_softmax_sum = 0.0;
        let mut has_full_softmax_sum = false;
        let mut candidates;
        if self.config.top_k > 0 {
            let keep = logits
                .len()
                .min((self.config.top_k as usize).max(self.config.min_keep.max(1)));
            candidates = select_best(keep)?;
        } else if self.config.top_p < 1.0 && logits.len() > 1024 {
            let mut maximum = f64::NEG_INFINITY;
            let mut found = false;
            self.for_each_adjusted(logits, |_, value| {
                maximum = maximum.max(value);
                found = true;
            })?;
            if !found {
                return Err(Error::runtime(
                    "logit distribution contains no finite values",
                ));
            }
            self.for_each_adjusted(logits, |_, value| {
                full_softmax_sum += ((value - maximum) / temperature).exp();
            })?;
            if !positive_finite(full_softmax_sum) {
                return Err(Error::runtime("logit softmax normalization failed"));
            }
            has_full_softmax_sum = true;
            const INITIAL_TOP_P_CANDIDATES: usize = 256;
            let initial_keep = logits
                .len()
                .min(INITIAL_TOP_P_CANDIDATES.max(self.config.min_keep.max(1)));
            candidates = select_best(initial_keep)?;
            let mut selected_mass = 0.0;
            for candidate in &candidates {
                selected_mass += ((candidate.value - maximum) / temperature).exp();
            }
            if selected_mass < f64::from(self.config.top_p) * full_softmax_sum {
                candidates = select_all()?;
            }
        } else {
            candidates = select_all()?;
        }
        if candidates.is_empty() {
            return Err(Error::runtime(
                "logit distribution contains no finite values",
            ));
        }
        self.filter_selected(&mut candidates, full_softmax_sum, has_full_softmax_sum);
        Ok(candidates)
    }

    /// Top-p and min-p over better-first candidates (`FilterSelected`);
    /// `full_softmax_sum` normalizes top-p when the candidates are a prefix
    /// of the vocabulary.
    fn filter_selected(
        &self,
        candidates: &mut Vec<Candidate>,
        full_softmax_sum: f64,
        has_full_softmax_sum: bool,
    ) {
        let temperature = f64::from(self.config.temperature);
        let mut keep = candidates.len();
        let minimum = minimum_kept(&self.config, candidates.len());
        if self.config.top_p < 1.0 && keep > 1 {
            let maximum = candidates[0].value;
            let sum = if has_full_softmax_sum {
                full_softmax_sum
            } else {
                let mut sum = 0.0;
                for candidate in candidates.iter() {
                    sum += ((candidate.value - maximum) / temperature).exp();
                }
                sum
            };
            let target = f64::from(self.config.top_p) * sum;
            let mut cumulative = 0.0;
            let mut top_p_keep = 0;
            while top_p_keep < keep {
                cumulative += ((candidates[top_p_keep].value - maximum) / temperature).exp();
                top_p_keep += 1;
                if top_p_keep >= minimum && cumulative >= target {
                    break;
                }
            }
            keep = top_p_keep;
        }
        if self.config.min_p > 0.0 && keep > 1 {
            let threshold = candidates[0].value + temperature * f64::from(self.config.min_p).ln();
            let mut min_p_keep = 0;
            while min_p_keep < keep && candidates[min_p_keep].value >= threshold {
                min_p_keep += 1;
            }
            keep = keep.min(min_p_keep.max(minimum));
        }
        candidates.truncate(keep);
    }
}

/// A kept candidate; the worst one is the heap's greatest.
struct Kept(Candidate);

impl PartialEq for Kept {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Kept {}

impl PartialOrd for Kept {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Kept {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        better_first(&self.0, &other.0)
    }
}

fn into_sorted(kept: std::collections::BinaryHeap<Kept>) -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = kept.into_iter().map(|k| k.0).collect();
    candidates.sort_unstable_by(better_first);
    candidates
}

/// [`SamplerState::select_best_penalized`] without penalties: the adjusted logit is the logit, so
/// one comparison with the worst kept value rejects most tokens. A later
/// token with the kept minimum's value is worse (ties go to lower IDs), so
/// a full heap only takes strictly greater logits.
fn select_best_plain(logits: &[f32], limit: usize) -> Vec<Candidate> {
    let mut kept = std::collections::BinaryHeap::with_capacity(limit + 1);
    let mut threshold = f32::NEG_INFINITY;
    for (block_index, block) in logits.chunks(SKIP_BLOCK).enumerate() {
        // A full heap skips a block that cannot beat its weakest entry.
        if kept.len() == limit && block_maximum(block) <= threshold {
            continue;
        }
        for (offset, &logit) in block.iter().enumerate() {
            if logit <= threshold || !logit.is_finite() {
                continue;
            }
            let candidate = Candidate {
                token: (block_index * SKIP_BLOCK + offset) as TokenId,
                value: f64::from(logit),
            };
            if kept.len() < limit {
                kept.push(Kept(candidate));
            } else if let Some(mut worst) = kept.peek_mut() {
                *worst = Kept(candidate);
            }
            if kept.len() == limit
                && let Some(worst) = kept.peek()
            {
                threshold = worst.0.value as f32;
            }
        }
    }
    into_sorted(kept)
}

/// The first index of the largest finite logit, as the scan `logit > best &&
/// logit.is_finite()` finds it, in two vectorizable passes: lane-wise maxima,
/// then the first block holding the maximum (`0.0` and `-0.0` compare equal
/// in both).
fn first_finite_max(logits: &[f32]) -> Option<usize> {
    const LANES: usize = 16;
    let finite = |x: f32| if x.is_finite() { x } else { f32::NEG_INFINITY };
    let larger = |a: f32, b: f32| if b > a { b } else { a };
    let (blocks, tail) = logits.as_chunks::<LANES>();
    let mut lanes = [f32::NEG_INFINITY; LANES];
    for block in blocks {
        for (lane, &x) in lanes.iter_mut().zip(block) {
            *lane = larger(*lane, finite(x));
        }
    }
    let best = tail
        .iter()
        .chain(&lanes)
        .fold(f32::NEG_INFINITY, |a, &x| larger(a, finite(x)));
    // Every finite value exceeds -inf.
    if best == f32::NEG_INFINITY {
        return None;
    }
    let block = blocks
        .iter()
        .position(|block| block.iter().fold(false, |any, &x| any | (x == best)))
        .unwrap_or(blocks.len());
    let start = block * LANES;
    logits[start..]
        .iter()
        .position(|&x| x == best)
        .map(|offset| start + offset)
}

/// One xorshift64* step (`NextRandom`); state zero is remapped.
pub fn next_random(state: &mut u64) -> u64 {
    let mut value = *state;
    if value == 0 {
        value = 0x9e37_79b9_7f4a_7c15;
    }
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    value.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

/// A uniform draw in [0, 1) with all 53 mantissa bits (`Uniform`).
pub fn uniform(state: &mut u64) -> f64 {
    let value = next_random(state);
    (value >> 11) as f64 * f64::from_bits(0x3ca0_0000_0000_0000)
}

/// Temperature-only sampling for callers without a `SamplerState`
/// (`SampleLogits`); greedy leaves the RNG untouched.
pub fn sample_logits(
    logits: &[f32],
    temperature: f32,
    rng_state: Option<&mut u64>,
) -> Result<TokenId> {
    if temperature > 0.0 && rng_state.is_none() {
        return Err(Error::invalid("sampling requires RNG state"));
    }
    let config = SamplingConfig {
        temperature,
        seed: 0,
        ..SamplingConfig::default()
    };
    let mut sampler = SamplerState::new(config, &[])?;
    if let Some(state) = rng_state.as_deref() {
        sampler.set_rng_state(*state);
    }
    let token = sampler.sample(logits)?;
    if let Some(state) = rng_state {
        *state = sampler.rng_state();
    }
    Ok(token)
}

/// The probability of `token` at `temperature` (`TokenProbability`).
pub fn token_probability(logits: &[f32], temperature: f32, token: TokenId) -> Result<f64> {
    if token as usize >= logits.len() {
        return Err(Error::logic("sampled token exceeds target vocabulary"));
    }
    let config = SamplingConfig::with_temperature(temperature);
    Ok(build_distribution(logits, &config, &[], &[])?.probability(token))
}

/// Residual sampling at `temperature` (`SampleResidual`).
pub fn sample_residual(
    target_logits: &[f32],
    temperature: f32,
    candidate_ids: &[TokenId],
    candidate_probabilities: &[f32],
    rng_state: &mut u64,
) -> Result<TokenId> {
    let config = SamplingConfig::with_temperature(temperature);
    build_distribution(target_logits, &config, &[], &[])?.sample_residual(
        candidate_ids,
        candidate_probabilities,
        rng_state,
    )
}

#[cfg(test)]
mod tests {

    #[test]
    fn model_presets_follow_the_thinking_mode_and_keep_explicit_fields() {
        let configured = SamplingConfig::default();
        let none = SamplingOverrides::default();
        let thinking = resolve_text_sampling(TextModelPreset::Qwen38, None, &configured, &none);
        assert_eq!(
            (
                thinking.temperature,
                thinking.top_p,
                thinking.top_k,
                thinking.presence_penalty
            ),
            (1.0, 0.95, 20, 0.0)
        );
        let direct =
            resolve_text_sampling(TextModelPreset::Qwen38, Some(false), &configured, &none);
        assert_eq!(
            (
                direct.temperature,
                direct.top_p,
                direct.top_k,
                direct.presence_penalty
            ),
            (0.7, 0.8, 20, 1.5)
        );
        let deepseek = resolve_text_sampling(
            TextModelPreset::DeepSeekV4Flash,
            Some(false),
            &configured,
            &none,
        );
        assert_eq!(
            (deepseek.temperature, deepseek.top_p, deepseek.top_k),
            (1.0, 0.95, 0)
        );
        // An explicit zero overrides the preset; unset fields follow it.
        let greedy = resolve_text_sampling(
            TextModelPreset::Qwen38,
            Some(false),
            &configured,
            &SamplingOverrides {
                temperature: true,
                ..SamplingOverrides::default()
            },
        );
        assert_eq!(
            (greedy.temperature, greedy.top_p, greedy.presence_penalty),
            (0.0, 0.8, 1.5)
        );
        assert_eq!(
            resolve_text_sampling(TextModelPreset::Unspecified, None, &configured, &none),
            configured
        );
        assert_eq!(
            resolve_text_sampling(
                TextModelPreset::Qwen38,
                None,
                &configured,
                &SamplingOverrides::ALL
            ),
            configured
        );
    }

    #[test]
    fn the_27b_preset_ignores_embedded_mtp_layers() {
        use crate::gguf::{
            Gguf,
            tests::{image, string_array},
        };
        let tokens: Vec<String> = (0..248_320).map(|i| i.to_string()).collect();
        let tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let preset = |blocks: u32, mtp: Option<u32>| {
            let mut metadata = vec![
                (
                    "general.architecture",
                    8,
                    [&6u64.to_le_bytes()[..], b"qwen35"].concat(),
                ),
                ("qwen35.block_count", 4, blocks.to_le_bytes().to_vec()),
                ("qwen35.embedding_length", 4, 5120u32.to_le_bytes().to_vec()),
                ("tokenizer.ggml.tokens", 9, string_array(&tokens)),
            ];
            if let Some(mtp) = mtp {
                metadata.push(("qwen35.nextn_predict_layers", 4, mtp.to_le_bytes().to_vec()));
            }
            text_preset(&Gguf::parse(&image(&metadata, &[], 0)).unwrap())
        };
        // The published 27B GGUF counts its MTP layer in block_count.
        assert_eq!(preset(65, Some(1)), TextModelPreset::Qwen38);
        assert_eq!(preset(64, None), TextModelPreset::Qwen38);
        assert_eq!(preset(65, None), TextModelPreset::Unspecified);
        assert_eq!(preset(64, Some(65)), TextModelPreset::Unspecified);
    }

    #[test]
    fn deepseek_artifacts_take_their_preset() {
        use crate::gguf::{Gguf, tests::image};
        let metadata = vec![(
            "general.architecture",
            8,
            [&9u64.to_le_bytes()[..], b"deepseek4"].concat(),
        )];
        assert_eq!(
            text_preset(&Gguf::parse(&image(&metadata, &[], 0)).unwrap()),
            TextModelPreset::DeepSeekV4Flash
        );
    }
    use super::*;

    #[test]
    fn a_known_argmax_replaces_only_the_plain_greedy_scan() {
        let logits = [0.5f32, 2.0, 1.0, 2.0];
        let mut greedy = SamplerState::new(SamplingConfig::default(), &[]).unwrap();
        // Taken as given: the caller vouches it is the first finite maximum.
        assert_eq!(greedy.sample_with_argmax(&logits, Some(1)).unwrap(), 1);
        assert_eq!(greedy.sample_with_argmax(&logits, None).unwrap(), 1);
        assert_eq!(
            greedy.sample_with_argmax(&logits, Some(9)).unwrap(),
            1,
            "out of range"
        );
        greedy.defer_sample(2).unwrap();
        assert_eq!(
            greedy.sample_with_argmax(&logits, Some(1)).unwrap(),
            2,
            "a deferred sample wins"
        );
        let penalized = SamplingConfig {
            repeat_penalty: 4.0,
            ..SamplingConfig::default()
        };
        let mut penalized = SamplerState::new(penalized, &[1]).unwrap();
        assert_eq!(
            penalized.sample_with_argmax(&logits, Some(1)).unwrap(),
            3,
            "penalties rescan"
        );
        let sampled = SamplingConfig {
            temperature: 0.8,
            seed: 5,
            ..SamplingConfig::default()
        };
        let (mut with, mut without) = (
            SamplerState::new(sampled.clone(), &[]).unwrap(),
            SamplerState::new(sampled, &[]).unwrap(),
        );
        for _ in 0..8 {
            assert_eq!(
                with.sample_with_argmax(&logits, Some(0)).unwrap(),
                without.sample(&logits).unwrap()
            );
        }
    }

    #[test]
    fn the_first_finite_maximum_matches_the_scalar_scan() {
        let scan = |logits: &[f32]| {
            let (mut best, mut at) = (f32::NEG_INFINITY, None);
            for (i, &x) in logits.iter().enumerate() {
                if x > best && x.is_finite() {
                    (best, at) = (x, Some(i));
                }
            }
            at
        };
        let specials = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MAX,
            -f32::MAX,
        ];
        let mut state = 7u64;
        for len in [0, 1, 5, 15, 16, 17, 31, 32, 33, 100, 1000, 4099] {
            for round in 0..40 {
                let logits: Vec<f32> = (0..len)
                    .map(|_| {
                        let r = next_random(&mut state);
                        match (r >> 60, round % 4) {
                            (0, _) => specials[(r % specials.len() as u64) as usize],
                            // Few distinct values force ties.
                            (_, 0) => (r % 3) as f32 - 1.0,
                            _ => (r >> 11) as f32 / (1u64 << 53) as f32 * 8.0 - 4.0,
                        }
                    })
                    .collect();
                assert_eq!(first_finite_max(&logits), scan(&logits), "{logits:?}");
            }
        }
        assert_eq!(first_finite_max(&[-0.0, 0.0, f32::NAN]), Some(0));
        assert_eq!(first_finite_max(&[f32::NAN; 40]), None);
        assert_eq!(first_finite_max(&[f32::NEG_INFINITY, -f32::MAX]), Some(1));
    }

    #[test]
    fn defaults_are_greedy_and_valid() {
        let config = SamplingConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.can_use_unmodified_argmax());
        assert!(!config.uses_random_sampling());
    }

    #[test]
    fn invalid_controls_are_rejected() {
        let invalid = [
            SamplingConfig::with_temperature(-1.0),
            SamplingConfig::with_temperature(f32::NAN),
            SamplingConfig {
                top_k: -1,
                ..SamplingConfig::default()
            },
            SamplingConfig {
                top_p: -0.01,
                ..SamplingConfig::default()
            },
            SamplingConfig {
                min_p: 1.5,
                ..SamplingConfig::default()
            },
            SamplingConfig {
                seed: -2,
                ..SamplingConfig::default()
            },
            SamplingConfig {
                repeat_penalty: 0.0,
                ..SamplingConfig::default()
            },
            SamplingConfig {
                presence_penalty: f32::INFINITY,
                ..SamplingConfig::default()
            },
        ];
        for config in invalid {
            let error = config.validate().expect_err("invalid control");
            assert_eq!(error.kind(), crate::text::ErrorKind::InvalidArgument);
        }
    }
}
