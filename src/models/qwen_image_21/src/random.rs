//! The initial noise of a request: `std::mt19937_64` seeded with the
//! request's seed and `std::normal_distribution<float>(0, 1)` as libstdc++ 15
//! computes them (`generate_canonical` with one 64-bit draw per float, the
//! Marsaglia polar method with its saved second value), so a seed replays the
//! C++ engine's noise bit for bit.

/// `std::mt19937_64`.
pub struct Mt19937_64 {
    state: [u64; 312],
    index: usize,
}

impl Mt19937_64 {
    const N: usize = 312;
    const M: usize = 156;
    const UPPER: u64 = 0xffff_ffff_8000_0000;
    const LOWER: u64 = 0x7fff_ffff;

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
                next ^= 0xb502_6f5a_a966_19e9;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    pub fn next_u64(&mut self) -> u64 {
        if self.index >= Self::N {
            self.twist();
        }
        let mut z = self.state[self.index];
        self.index += 1;
        z ^= (z >> 29) & 0x5555_5555_5555_5555;
        z ^= (z << 17) & 0x71d6_7fff_eda6_0000;
        z ^= (z << 37) & 0xfff7_eee0_0000_0000;
        z ^ (z >> 43)
    }

    /// `std::generate_canonical<float, 24>`: one draw scaled by 2^-64, with
    /// a result that rounds up to 1 replaced by the float below 1.
    pub fn canonical(&mut self) -> f32 {
        let value = self.next_u64() as f32 / 18_446_744_073_709_551_616.0f32;
        if value >= 1.0 {
            1.0 - f32::EPSILON / 2.0
        } else {
            value
        }
    }
}

/// `std::normal_distribution<float>` with mean 0 and deviation 1.
#[derive(Default)]
pub struct Normal {
    saved: Option<f32>,
}

impl Normal {
    pub fn sample(&mut self, rng: &mut Mt19937_64) -> f32 {
        let value = match self.saved.take() {
            Some(saved) => saved,
            None => loop {
                let x = 2.0f32 * rng.canonical() - 1.0;
                let y = 2.0f32 * rng.canonical() - 1.0;
                let r2 = x * x + y * y;
                if r2 > 1.0 || r2 == 0.0 {
                    continue;
                }
                let mult = (-2.0f32 * r2.ln() / r2).sqrt();
                self.saved = Some(x * mult);
                break y * mult;
            },
        };
        value * 1.0 + 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_64_matches_the_standard() {
        // The C++ standard's check: the 10000th draw of a default-seeded
        // engine (seed 5489).
        let mut rng = Mt19937_64::new(5489);
        let mut value = 0;
        for _ in 0..10_000 {
            value = rng.next_u64();
        }
        assert_eq!(value, 9_981_545_732_273_789_042);
    }

    #[test]
    fn normal_draws_pair_up() {
        let mut rng = Mt19937_64::new(42);
        let mut normal = Normal::default();
        let values: Vec<f32> = (0..1000).map(|_| normal.sample(&mut rng)).collect();
        let mean = values.iter().sum::<f32>() / 1000.0;
        assert!(mean.abs() < 0.1, "{mean}");
        assert!(values.iter().all(|v| v.is_finite()));
    }
}
