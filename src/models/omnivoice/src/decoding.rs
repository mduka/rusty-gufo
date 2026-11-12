//! Host side of OmniVoice's iterative unmasking (`decoding.cpp`, ported from
//! k2-fsa/OmniVoice's `_generate_iterative` and `_get_time_steps`,
//! Apache-2.0): the schedule, the conditional input layout and the per-step
//! position choice.

/// OmniVoiceGenerationConfig's decoding parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodingConfig {
    pub steps: usize,
    pub guidance: f32,
    pub t_shift: f32,
    pub layer_penalty: f32,
    /// Gumbel noise on the position scores; 0 selects positions greedily.
    pub position_temperature: f32,
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            steps: 32,
            guidance: 2.0,
            t_shift: 0.1,
            layer_penalty: 5.0,
            position_temperature: 5.0,
        }
    }
}

/// Codes to unmask at each step: `ceil(total * (t[s+1] - t[s]))` of the
/// shifted linear schedule in float32, the remainder at the last step
/// (`UnmaskSchedule`).
pub fn unmask_schedule(total: usize, steps: usize, t_shift: f32) -> Vec<usize> {
    // torch.linspace(0, 1, steps + 1) in float32: the first half counts up
    // from 0, the second half down from 1.
    let step = 1.0f32 / steps as f32;
    let halfway = steps.div_ceil(2);
    let times: Vec<f32> = (0..=steps)
        .map(|index| {
            let t = if index < halfway {
                step * index as f32
            } else {
                1.0 - step * (steps - index) as f32
            };
            t_shift * t / (1.0 + (t_shift - 1.0) * t)
        })
        .collect();
    let mut schedule = vec![0usize; steps];
    let mut remaining = total;
    for index in 0..steps {
        let mut count = remaining;
        if index + 1 < steps {
            let delta = f64::from(times[index + 1]) - f64::from(times[index]);
            count = ((total as f64 * delta).ceil() as usize).min(remaining);
        }
        schedule[index] = count;
        remaining -= count;
    }
    schedule
}

/// The conditional sequence: `[text tokens | reference frames | target
/// frames]`, `codebooks` rows of `length` IDs (text positions repeat their
/// token in every row; target frames start as the mask).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConditionalInput {
    pub codebooks: usize,
    pub length: usize,
    /// First audio position: the number of text tokens.
    pub audio_start: usize,
    pub target_frames: usize,
    /// `[codebook][position]`.
    pub ids: Vec<i32>,
}

impl ConditionalInput {
    /// `Build`; `reference_codes` is `[codebook][frame]`.
    pub fn build(
        text_ids: &[u32],
        reference_codes: &[i32],
        reference_frames: usize,
        target_frames: usize,
        codebooks: usize,
        mask_id: u32,
    ) -> Self {
        let length = text_ids.len() + reference_frames + target_frames;
        let mut ids = vec![mask_id as i32; codebooks * length];
        for c in 0..codebooks {
            let row = &mut ids[c * length..(c + 1) * length];
            for (slot, &id) in row.iter_mut().zip(text_ids) {
                *slot = id as i32;
            }
            row[text_ids.len()..text_ids.len() + reference_frames].copy_from_slice(
                &reference_codes[c * reference_frames..(c + 1) * reference_frames],
            );
        }
        Self {
            codebooks,
            length,
            audio_start: text_ids.len(),
            target_frames,
            ids,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GenerationResult {
    /// `[codebook][frame]`.
    pub codes: Vec<i32>,
    /// The step that unmasked each code, `[codebook][frame]`.
    pub unmasked_at: Vec<i32>,
}

/// `std::mt19937_64`, seeded like its single-value constructor.
#[derive(Clone)]
pub struct Mt19937_64 {
    state: [u64; 312],
    index: usize,
}

impl Mt19937_64 {
    const N: usize = 312;
    const M: usize = 156;
    const MATRIX: u64 = 0xB502_6F5A_A966_19E9;
    const UPPER: u64 = 0xFFFF_FFFF_8000_0000;
    const LOWER: u64 = 0x7FFF_FFFF;

    pub fn new(seed: u64) -> Self {
        let mut state = [0u64; 312];
        state[0] = seed;
        for i in 1..Self::N {
            let previous = state[i - 1];
            state[i] = 6_364_136_223_846_793_005u64
                .wrapping_mul(previous ^ (previous >> 62))
                .wrapping_add(i as u64);
        }
        Self {
            state,
            index: Self::N,
        }
    }

    fn twist(&mut self) {
        for i in 0..Self::N {
            let y = (self.state[i] & Self::UPPER) | (self.state[(i + 1) % Self::N] & Self::LOWER);
            let mut next = self.state[(i + Self::M) % Self::N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= Self::MATRIX;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    /// The next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        if self.index >= Self::N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= (y >> 29) & 0x5555_5555_5555_5555;
        y ^= (y << 17) & 0x71D6_7FFF_EDA6_0000;
        y ^= (y << 37) & 0xFFF7_EEE0_0000_0000;
        y ^ (y >> 43)
    }
}

/// `_gumbel_sample`'s noise, with uniform noise in [0, 1) from the top 24
/// bits of one draw.
fn gumbel(random: &mut Mt19937_64) -> f32 {
    let uniform = (random.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0);
    -(-(uniform + 1e-10f32).ln() + 1e-10f32).ln()
}

/// Chooses which masked codes a step fills: scores are the guided
/// confidences minus `layer_penalty` per codebook index, divided by the
/// position temperature plus Gumbel noise when it is positive; the `count`
/// best masked positions take their predictions (`PositionSelector`).
pub struct PositionSelector {
    layer_penalty: f32,
    temperature: f32,
    random: Mt19937_64,
}

impl PositionSelector {
    pub fn new(config: &DecodingConfig, seed: u64) -> Self {
        Self {
            layer_penalty: config.layer_penalty,
            temperature: config.position_temperature,
            random: Mt19937_64::new(seed),
        }
    }

    /// `Fill`: `codes`, `prediction` and `confidence` are
    /// `[codebook][frame]`; returns the flat indices it filled, best first.
    #[allow(clippy::too_many_arguments)]
    pub fn fill(
        &mut self,
        count: usize,
        codebooks: usize,
        frames: usize,
        prediction: &[i32],
        confidence: &[f32],
        mask_id: u32,
        codes: &mut [i32],
    ) -> Vec<usize> {
        let total = codebooks * frames;
        let mut scores = vec![0.0f32; total];
        for c in 0..codebooks {
            let penalty = c as f32 * self.layer_penalty;
            for f in 0..frames {
                let index = c * frames + f;
                let mut score = confidence[index] - penalty;
                if self.temperature > 0.0 {
                    score = score / self.temperature + gumbel(&mut self.random);
                }
                if codes[index] != mask_id as i32 {
                    score = f32::NEG_INFINITY;
                }
                scores[index] = score;
            }
        }
        let count = count.min(total);
        // Higher scores first, ties by position; a NaN score ranks last.
        let order_of = |left: &usize, right: &usize| {
            let (a, b) = (scores[*left], scores[*right]);
            b.partial_cmp(&a)
                .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
                .then(left.cmp(right))
        };
        let mut order: Vec<usize> = (0..total).collect();
        if count < total && count > 0 {
            order.select_nth_unstable_by(count - 1, order_of);
        }
        order.truncate(count);
        order.sort_unstable_by(order_of);
        for &index in &order {
            codes[index] = prediction[index];
        }
        order
    }
}

#[cfg(test)]
mod tests {
    use super::{Mt19937_64, gumbel};

    /// The first draws and their Gumbel noise as libstdc++'s
    /// `std::mt19937_64` and glibc's `logf` give them (printed by a g++ 15
    /// build of the C++ expression on 2026-11-10).
    #[test]
    fn draws_and_noise_match_libstdcxx() {
        let cases: [(u64, [(u64, u32); 4]); 2] = [
            (
                0,
                [
                    (2_947_667_278_772_165_694, 0xbf1b_3f0b),
                    (18_301_848_765_998_365_067, 0x409a_f74c),
                    (729_919_693_006_235_833, 0xbf96_10f0),
                    (11_021_831_128_136_023_278, 0x3f29_dfaa),
                ],
            ),
            (
                42,
                [
                    (13_930_160_852_258_120_406, 0x3fa2_8f5e),
                    (11_788_048_577_503_494_824, 0x3f4d_abed),
                    (13_874_630_024_467_741_450, 0x3fa0_c095),
                    (2_513_787_319_205_155_662, 0xbf30_8f85),
                ],
            ),
        ];
        for (seed, draws) in cases {
            let mut raw = Mt19937_64::new(seed);
            let mut noise = Mt19937_64::new(seed);
            for (value, bits) in draws {
                assert_eq!(raw.next_u64(), value, "seed {seed}");
                assert_eq!(gumbel(&mut noise).to_bits(), bits, "seed {seed}");
            }
        }
    }

    #[test]
    fn mt19937_64_matches_the_standard() {
        // [rand.predef]: the 10000th output of a default-constructed
        // mt19937_64 (seed 5489).
        let mut random = Mt19937_64::new(5489);
        for _ in 0..9999 {
            random.next_u64();
        }
        assert_eq!(random.next_u64(), 9_981_545_732_273_789_042);
    }
}
