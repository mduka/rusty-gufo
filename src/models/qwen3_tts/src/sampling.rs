//! Codec token sampling (`sampling.cpp`) with the random engine and
//! distribution of the C++ build: `std::mt19937` and libstdc++'s
//! `discrete_distribution` (GCC 15), so a seed replays the same codes.

/// Sampling controls of a request (`SamplingOptions`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingOptions {
    pub sample: bool,
    pub seed: u32,
    pub top_k: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub predictor_sample: bool,
    pub predictor_top_k: usize,
    pub predictor_temperature: f32,
    pub predictor_top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            sample: true,
            seed: 42,
            top_k: 50,
            temperature: 0.9,
            top_p: 1.0,
            predictor_sample: true,
            predictor_top_k: 50,
            predictor_temperature: 0.9,
            predictor_top_p: 1.0,
            repetition_penalty: 1.05,
        }
    }
}

impl SamplingOptions {
    /// `ValidSamplingOptions`.
    pub fn is_valid(&self) -> bool {
        let temperature = |v: f32| v.is_finite() && v > 0.0;
        let top_p = |v: f32| v.is_finite() && v > 0.0 && v <= 1.0;
        temperature(self.temperature)
            && temperature(self.predictor_temperature)
            && temperature(self.repetition_penalty)
            && top_p(self.top_p)
            && top_p(self.predictor_top_p)
    }
}

/// `std::mt19937`.
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            let previous = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    fn twist(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7FFF_FFFF;
        for i in 0..624 {
            let y = (self.state[i] & UPPER) | (self.state[(i + 1) % 624] & LOWER);
            let mut next = self.state[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= 0x9908_B0DF;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^ (y >> 18)
    }

    /// libstdc++'s `generate_canonical<double, 53>`: two 32-bit draws,
    /// summed in double, scaled by 2^-64, never 1.
    pub fn canonical(&mut self) -> f64 {
        const RANGE: f64 = 4_294_967_296.0;
        let mut sum = 0.0f64;
        let mut scale = 1.0f64;
        for _ in 0..2 {
            sum += f64::from(self.next_u32()) * scale;
            scale *= RANGE;
        }
        let value = sum / scale;
        if value >= 1.0 {
            // nextafter(1, 0)
            f64::from_bits(1.0f64.to_bits() - 1)
        } else {
            value
        }
    }
}

/// libstdc++'s `discrete_distribution<size_t>(weights)(random)`: fewer than
/// two weights draw nothing and select 0.
fn discrete(weights: &[f64], random: &mut Mt19937) -> usize {
    if weights.len() < 2 {
        return 0;
    }
    let sum: f64 = weights.iter().fold(0.0, |total, &w| total + w);
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut running = 0.0f64;
    for &weight in weights {
        running += weight / sum;
        cumulative.push(running);
    }
    if let Some(last) = cumulative.last_mut() {
        *last = 1.0;
    }
    let p = random.canonical();
    cumulative.partition_point(|&c| c < p)
}

/// A candidate token and its (penalized) logit (`TokenScore`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenScore {
    pub token: u32,
    pub score: f32,
}

/// Higher score first, the lower token on a tie: a total order.
fn higher(left: &TokenScore, right: &TokenScore) -> std::cmp::Ordering {
    if left.score == right.score {
        left.token.cmp(&right.token)
    } else if left.score > right.score {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Greater
    }
}

/// `SampleCodec`: temperature, top-k (keeping boundary ties), then nucleus
/// filtering; without a random engine the greedy token. Rejects nonfinite
/// logits.
pub fn sample_codec(
    candidates: &mut Vec<TokenScore>,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    random: Option<&mut Mt19937>,
) -> Result<u32, String> {
    if candidates.is_empty()
        || !temperature.is_finite()
        || temperature <= 0.0
        || !top_p.is_finite()
        || top_p <= 0.0
        || top_p > 1.0
    {
        return Err("Qwen3-TTS sampler input is invalid".into());
    }
    if candidates.iter().any(|c| !c.score.is_finite()) {
        return Err("Qwen3-TTS sampler received nonfinite logits".into());
    }
    let Some(random) = random else {
        let best = candidates
            .iter()
            .min_by(|a, b| higher(a, b))
            .map(|c| c.token)
            .unwrap_or_default();
        return Ok(best);
    };
    if top_k != 0 && top_k < candidates.len() {
        candidates.select_nth_unstable_by(top_k - 1, higher);
        let cutoff = candidates[top_k - 1].score;
        candidates.retain(|c| c.score >= cutoff);
    }
    candidates.sort_by(higher);
    let maximum = f64::from(candidates[0].score);
    let mut weights: Vec<f64> = candidates
        .iter()
        .map(|c| ((f64::from(c.score) - maximum) / f64::from(temperature)).exp())
        .collect();
    if top_p < 1.0 {
        let threshold = weights.iter().fold(0.0f64, |total, &w| total + w) * f64::from(top_p);
        let mut cumulative = 0.0f64;
        let mut count = 0;
        loop {
            cumulative += weights[count];
            count += 1;
            if count >= weights.len() || cumulative >= threshold {
                break;
            }
        }
        weights.truncate(count);
        candidates.truncate(count);
    }
    Ok(candidates[discrete(&weights, random)].token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_matches_the_standard() {
        // The 10000th output of a default-seeded mt19937 is 4123659995.
        let mut engine = Mt19937::new(5489);
        let mut value = 0;
        for _ in 0..10000 {
            value = engine.next_u32();
        }
        assert_eq!(value, 4_123_659_995);
    }

    #[test]
    fn greedy_takes_the_lowest_token_among_equal_maxima() {
        let mut candidates = vec![
            TokenScore {
                token: 4,
                score: 2.0,
            },
            TokenScore {
                token: 1,
                score: 2.0,
            },
            TokenScore {
                token: 0,
                score: 1.0,
            },
        ];
        assert_eq!(sample_codec(&mut candidates, 50, 1.0, 0.9, None), Ok(1));
        let mut bad = vec![TokenScore {
            token: 0,
            score: f32::NAN,
        }];
        assert!(sample_codec(&mut bad, 50, 1.0, 0.9, None).is_err());
    }

    #[test]
    fn a_single_candidate_draws_nothing() {
        let mut engine = Mt19937::new(1);
        let mut reference = Mt19937::new(1);
        let mut candidates = vec![
            TokenScore {
                token: 7,
                score: 9.0,
            },
            TokenScore {
                token: 3,
                score: 1.0,
            },
        ];
        assert_eq!(
            sample_codec(&mut candidates, 1, 1.0, 1.0, Some(&mut engine)),
            Ok(7)
        );
        assert_eq!(engine.next_u32(), reference.next_u32());
    }
}
