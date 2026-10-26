//! A request sampler lent to DSpark cycles (`DsparkSamplerBridge`,
//! `dspark_sampler.hpp`). Sampled proposals use a bounded q with exact
//! rejection and residual correction; target filters and penalties see the
//! complete vocabulary and only accepted history. Unfiltered requests and
//! greedy requests with penalties take the point-mass path, where building a
//! dense target p costs more than it saves. A rejected draw becomes the next
//! cycle's anchor. The owner copies the RNG back after each cycle.

use gufo_core::sampling::{SamplerState, SamplingConfig, TokenId, build_distribution};
use gufo_core::text::{Error, Result};

use crate::dspark_model::MAX_BLOCK;

/// `DS4_DSPARK_CANDIDATES`.
pub const CANDIDATES: usize = 8;

/// `ds4_dspark_candidates`: one drafted position's shortlist after the
/// Markov coupling, and the support model's confidence in it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Candidates {
    pub ids: [i32; CANDIDATES],
    pub logits: [f32; CANDIDATES],
    pub confidence: f32,
}

impl Candidates {
    /// The 68 bytes the candidates kernel writes per request.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 68 {
            return None;
        }
        let word = |i: usize| {
            [
                bytes[4 * i],
                bytes[4 * i + 1],
                bytes[4 * i + 2],
                bytes[4 * i + 3],
            ]
        };
        let mut candidates = Self::default();
        for i in 0..CANDIDATES {
            candidates.ids[i] = i32::from_le_bytes(word(i));
            candidates.logits[i] = f32::from_le_bytes(word(CANDIDATES + i));
        }
        candidates.confidence = f32::from_le_bytes(word(2 * CANDIDATES));
        Some(candidates)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Proposal {
    ids: [TokenId; CANDIDATES],
    probabilities: [f32; CANDIDATES],
    token: TokenId,
    probability: f32,
    size: usize,
}

/// The sampler hook of a cycle (`ds4_dspark_sampler`).
pub struct DsparkSampler {
    working: SamplerState,
    proposals: [Proposal; MAX_BLOCK as usize],
    /// `propose` and `verify` are available.
    stochastic: bool,
    draws: u32,
    verifications: u32,
    /// A hook that only samples, returning this token (a C++ test hook),
    /// and the number of draws.
    #[cfg(test)]
    fixed: Option<(i32, u32)>,
}

impl DsparkSampler {
    /// A working copy of the request sampler.
    pub fn new(sampler: &SamplerState) -> Self {
        let stochastic = Self::use_stochastic(sampler.config());
        Self {
            working: sampler.clone(),
            proposals: [Proposal::default(); MAX_BLOCK as usize],
            stochastic,
            draws: 0,
            verifications: 0,
            #[cfg(test)]
            fixed: None,
        }
    }

    /// A hook whose every draw is `token`, without proposals.
    #[cfg(test)]
    pub(crate) fn fixed(token: i32) -> Self {
        let mut sampler = Self::new(&SamplerState::new(SamplingConfig::default(), &[]).unwrap());
        sampler.fixed = Some((token, 0));
        sampler
    }

    /// The draws of a fixed hook.
    #[cfg(test)]
    pub(crate) fn fixed_draws(&self) -> u32 {
        self.fixed.map_or(0, |(_, draws)| draws)
    }

    /// Proposals drawn from a probabilistic q (not a point mass) and
    /// verified rows in this cycle.
    pub fn proposal_counts(&self) -> (u32, u32) {
        (self.draws, self.verifications)
    }

    /// `UseStochastic`.
    fn use_stochastic(config: &SamplingConfig) -> bool {
        config.uses_random_sampling()
            && (config.top_p < 1.0 || (config.top_k > 0 && config.top_k <= 256))
    }

    /// Whether the cycle may propose stochastically (`propose`/`verify`
    /// set); a single request's cycle keeps the point-mass path.
    pub fn stochastic(&self) -> bool {
        self.stochastic
    }

    /// The cycle without stochastic proposals (C1's `scalar_sampler`).
    pub fn scalar(&mut self) {
        self.stochastic = false;
    }

    /// The RNG to copy back into the request sampler.
    pub fn rng_state(&self) -> u64 {
        self.working.rng_state()
    }

    /// Accepts the target's already computed argmax where it is also this
    /// request's exact selection (`TryAcceptArgmax`): greedy without
    /// penalties and legal under the grammar. A refusal leaves the sampler
    /// unchanged.
    pub fn try_accept_argmax(&mut self, token: i32) -> Result<bool> {
        #[cfg(test)]
        if self.fixed.is_some() {
            return Ok(false);
        }
        let config = self.working.config();
        if config.temperature != 0.0 || config.penalties_enabled() {
            return Ok(false);
        }
        let Ok(token) = TokenId::try_from(token) else {
            return Ok(false);
        };
        if !self.working.can_select_argmax(token, false)? {
            return Ok(false);
        }
        self.working.accept(token)?;
        Ok(true)
    }

    /// `Sample`: draws from one row of target logits and records it.
    pub fn sample(&mut self, logits: &[f32]) -> Result<i32> {
        #[cfg(test)]
        if let Some((token, draws)) = &mut self.fixed {
            *draws += 1;
            return Ok(*token);
        }
        let token = self.working.sample(logits)?;
        self.working.accept(token)?;
        Ok(token as i32)
    }

    /// `Accept`: a token emitted without drawing it, the previous cycle's
    /// rejected draw.
    pub fn accept(&mut self, token: i32) -> Result<()> {
        #[cfg(test)]
        if self.fixed.is_some() {
            return Ok(());
        }
        self.working.accept(token as TokenId)
    }

    /// `Propose`: draws position `row` of a block from the shortlist's q
    /// (with the request temperature), keeping q for the verification.
    pub fn propose(&mut self, row: u32, candidates: &Candidates) -> Result<i32> {
        let proposal = self
            .proposals
            .get_mut(row as usize)
            .ok_or_else(|| Error::invalid("invalid DSpark proposal position"))?;
        for (i, (&id, &logit)) in candidates.ids.iter().zip(&candidates.logits).enumerate() {
            if logit.is_nan() || logit == f32::INFINITY {
                return Err(Error::invalid("non-finite DSpark proposal logit"));
            }
            if id < 0 {
                return Err(Error::invalid("invalid DSpark proposal token"));
            }
            if candidates.ids[..i].contains(&id) {
                return Err(Error::invalid("duplicate DSpark proposal token"));
            }
        }
        // Upstream q applies temperature; the request's filters and
        // penalties define p. A shortlisted q changes acceptance, never the
        // target.
        let config = SamplingConfig::with_temperature(self.working.config().temperature);
        let distribution = build_distribution(&candidates.logits, &config, &[], &[])?;
        *proposal = Proposal::default();
        // Diffuse proposals gain acceptance from p/q, concentrated ones do
        // better as a point mass; the choice depends on q alone, so both
        // routes stay exact.
        let entries = distribution.entries();
        if entries.first().is_some_and(|e| e.value >= 0.7) {
            let id = candidates.ids[distribution.best_token()? as usize];
            proposal.size = 1;
            proposal.ids[0] = id as TokenId;
            proposal.token = id as TokenId;
            proposal.probabilities[0] = 1.0;
            proposal.probability = 1.0;
            return Ok(id);
        }
        self.draws += 1;
        const UNITS: u32 = 1 << 24;
        let mut masses = [0u32; CANDIDATES];
        let mut total = 0u32;
        for entry in entries {
            let id = candidates.ids[entry.token as usize];
            proposal.ids[proposal.size] = id as TokenId;
            masses[proposal.size] = (entry.value * f64::from(UNITS)).floor() as u32;
            total += masses[proposal.size];
            proposal.size += 1;
        }
        // An F32 q sums to exactly one and matches the 24-bit draw exactly.
        masses[0] += UNITS - total;
        let draw = (self.working.uniform() * f64::from(UNITS)) as u32;
        let mut cumulative = 0u32;
        let size = proposal.size;
        for ((probability, &mass), &id) in proposal.probabilities[..size]
            .iter_mut()
            .zip(&masses)
            .zip(&proposal.ids)
        {
            *probability = mass as f32 / UNITS as f32;
            cumulative += mass;
            if proposal.probability == 0.0 && draw < cumulative {
                proposal.token = id;
                proposal.probability = *probability;
            }
        }
        Ok(proposal.token as i32)
    }

    /// `Verify`: accepts the draft of position `row` with probability
    /// p/q, else draws from the residual; p is sampled once, keeping the
    /// fast sampler, without a dense target row.
    pub fn verify(&mut self, row: u32, logits: &[f32], draft: i32) -> Result<i32> {
        let proposal = *self
            .proposals
            .get(row as usize)
            .ok_or_else(|| Error::invalid("invalid DSpark verification position"))?;
        if draft < 0
            || proposal.token != draft as TokenId
            || proposal.probability.is_nan()
            || proposal.probability <= 0.0
            || !proposal.probability.is_finite()
        {
            return Err(Error::invalid("missing DSpark proposal distribution"));
        }
        if proposal.ids[..proposal.size]
            .iter()
            .any(|&id| id as usize >= logits.len())
        {
            return Err(Error::invalid("DSpark proposal exceeds target vocabulary"));
        }
        self.verifications += 1;
        if proposal.size == 1 {
            return self.sample(logits);
        }
        let target = self.working.distribution(logits)?;
        let token = if self.working.uniform() * f64::from(proposal.probability)
            < target.probability(proposal.token)
        {
            proposal.token
        } else {
            target.sample_residual(
                &proposal.ids[..proposal.size],
                &proposal.probabilities[..proposal.size],
                self.working.mutable_rng_state(),
            )?
        };
        self.working.accept(token)?;
        Ok(token as i32)
    }
}

#[cfg(test)]
mod tests {
    use gufo_core::sampling::{SamplerState, SamplingConfig, TokenId};

    use super::{CANDIDATES, Candidates, DsparkSampler};

    const TRIALS: u32 = 120_000;
    const LOGITS: [f32; 10] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    /// Tokens 0 to 7 with descending logits.
    fn candidates() -> Candidates {
        Candidates {
            ids: std::array::from_fn(|i| i as i32),
            logits: std::array::from_fn(|i| (3 - i as i32) as f32 * 0.3),
            confidence: 0.8,
        }
    }

    /// The sampled request of the replay and malformed-row cases.
    fn seeded() -> SamplerState {
        let config = SamplingConfig {
            temperature: 0.7,
            top_k: 10,
            seed: 12,
            ..SamplingConfig::default()
        };
        SamplerState::new(config, &[]).unwrap()
    }

    /// `count` of `TRIALS` draws matches `probability` within six sigma.
    fn within_six_sigma(count: u32, probability: f64) -> bool {
        let trials = f64::from(TRIALS);
        let tolerance = 6.0 * (probability * (1.0 - probability) / trials).sqrt() + 2.0 / trials;
        (f64::from(count) / trials - probability).abs() <= tolerance
    }

    /// `CheckDistribution`: one proposed and verified position draws from
    /// the complete filtered target.
    fn check_distribution(mut config: SamplingConfig, sharp: bool) {
        const TARGET: [f32; 10] = [0.5, -0.3, 0.7, 0.1, 0.2, -0.5, -0.2, 0.4, 0.9, -0.1];
        config.seed = 731;
        let mut state = SamplerState::new(config, &[8, 2, 2, 1, 0]).unwrap();
        let expected = state.distribution(&TARGET).unwrap();
        let mut candidates = candidates();
        if sharp {
            candidates.logits[0] = 12.0;
        }
        assert!(DsparkSampler::new(&state).stochastic());
        let mut counts = [0u32; 10];
        let mut accepted = 0u32;
        for _ in 0..TRIALS {
            let mut sampler = DsparkSampler::new(&state);
            let draft = sampler.propose(0, &candidates).unwrap();
            let result = sampler.verify(0, &TARGET, draft).unwrap();
            assert!((0..10).contains(&result), "target vocabulary");
            accepted += u32::from(result == draft);
            counts[result as usize] += 1;
            state.set_rng_state(sampler.rng_state());
        }
        let mut target_mass_on_q = 0.0;
        for (token, &count) in counts.iter().enumerate() {
            let p = expected.probability(token as TokenId);
            assert!(
                within_six_sigma(count, p),
                "p/q and residual preserve the complete filtered target (token {token})"
            );
            if token < CANDIDATES {
                target_mass_on_q += p;
            }
        }
        assert!(accepted <= TRIALS, "acceptance count");
        if target_mass_on_q < 1.0 {
            assert!(
                counts[8] + counts[9] > 0,
                "residual samples outside sparse q"
            );
        }
    }

    #[test]
    fn sharp_proposals_preserve_the_target() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 10,
            ..SamplingConfig::default()
        };
        check_distribution(config, true);
    }

    #[test]
    fn proposals_preserve_filtered_targets() {
        let configs = [
            SamplingConfig {
                temperature: 0.6,
                top_k: 10,
                ..SamplingConfig::default()
            },
            SamplingConfig {
                temperature: 1.0,
                top_k: 10,
                ..SamplingConfig::default()
            },
            SamplingConfig {
                temperature: 0.6,
                top_k: 4,
                top_p: 0.95,
                ..SamplingConfig::default()
            },
            SamplingConfig {
                temperature: 1.0,
                top_p: 0.95,
                min_p: 0.2,
                min_keep: 3,
                repeat_penalty: 1.2,
                frequency_penalty: 0.3,
                presence_penalty: 0.1,
                ..SamplingConfig::default()
            },
            SamplingConfig {
                temperature: 1.0,
                top_k: 1,
                min_keep: 2,
                ..SamplingConfig::default()
            },
        ];
        for config in configs {
            check_distribution(config, false);
        }
    }

    /// `CheckConditionalProposals`: conditional q, a rejected suffix and
    /// committed penalties keep the joint target of two positions.
    #[test]
    fn conditional_proposals_preserve_the_joint_target() {
        let config = SamplingConfig {
            temperature: 0.7,
            top_k: 10,
            seed: 73,
            repeat_last_n: 3,
            frequency_penalty: 0.3,
            ..SamplingConfig::default()
        };
        let mut initial = SamplerState::new(config, &[1, 1, 4]).unwrap();
        let first: [f32; 10] = std::array::from_fn(|i| 0.1 * i as f32);
        let second: [[f32; 10]; 10] =
            std::array::from_fn(|i| std::array::from_fn(|j| 0.2 * ((i * 3 + j * 7) % 10) as f32));
        let p = initial.distribution(&first).unwrap();
        let expected: [[f64; 10]; 10] = std::array::from_fn(|i| {
            let mut conditional = initial.clone();
            conditional.accept(i as TokenId).unwrap();
            let next = conditional.distribution(&second[i]).unwrap();
            std::array::from_fn(|j| p.probability(i as TokenId) * next.probability(j as TokenId))
        });
        let mut counts = [[0u32; 10]; 10];
        for trial in 0..TRIALS {
            let mut sampler = DsparkSampler::new(&initial);
            let mut proposed = candidates();
            if trial % 2 == 0 {
                // Mixes the exact delta and a probabilistic q.
                proposed.logits[3] = 12.0;
            }
            let a = sampler.propose(0, &proposed).unwrap();
            let mut proposed = candidates();
            if a % 2 == 0 {
                proposed.logits[1] = 12.0;
            }
            // A synthetic Markov dependence on the preceding proposed token.
            for id in &mut proposed.ids {
                *id = (*id + a + 1) % 10;
            }
            let b = sampler.propose(1, &proposed).unwrap();
            let x = sampler.verify(0, &first, a).unwrap();
            // A rejection discards the unused draft and continues from the
            // residual.
            let y = if x == a {
                sampler.verify(1, &second[x as usize], b).unwrap()
            } else {
                sampler.sample(&second[x as usize]).unwrap()
            };
            counts[x as usize][y as usize] += 1;
            initial.set_rng_state(sampler.rng_state());
        }
        for (i, (counts, expected)) in counts.iter().zip(&expected).enumerate() {
            for (j, (&count, &probability)) in counts.iter().zip(expected).enumerate() {
                assert!(
                    within_six_sigma(count, probability),
                    "conditional q, rejected suffix and committed penalties preserve joint p \
                     ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn proposal_and_verification_replay_the_seed() {
        let state = seeded();
        let candidates = candidates();
        let (mut a, mut b) = (DsparkSampler::new(&state), DsparkSampler::new(&state));
        let da = a.propose(0, &candidates).unwrap();
        let db = b.propose(0, &candidates).unwrap();
        assert_eq!(da, db, "proposal seed replay");
        assert_eq!(
            a.verify(0, &LOGITS, da).unwrap(),
            b.verify(0, &LOGITS, db).unwrap(),
            "verification seed replay"
        );
        assert_eq!(a.rng_state(), b.rng_state(), "verification seed replay");
    }

    #[test]
    fn malformed_rows_fail() {
        let state = seeded();
        let candidates = candidates();
        let mut missing = DsparkSampler::new(&state);
        assert!(missing.verify(0, &LOGITS, 0).is_err(), "missing proposal");
        let mut sampler = DsparkSampler::new(&state);
        for invalid in [f32::NAN, f32::INFINITY] {
            let mut broken = candidates;
            broken.logits[2] = invalid;
            assert!(sampler.propose(0, &broken).is_err(), "non-finite logit");
        }
        let mut duplicate = candidates;
        duplicate.ids[2] = duplicate.ids[1];
        assert!(sampler.propose(0, &duplicate).is_err(), "duplicate token");
        let mut sharp = candidates;
        sharp.logits[3] = 30.0;
        sharp.ids[2] = sharp.ids[1];
        assert!(sampler.propose(0, &sharp).is_err(), "duplicate delta token");
        let mut invalid_id = candidates;
        invalid_id.ids[7] = -1;
        assert!(sampler.propose(0, &invalid_id).is_err(), "negative token");
        let mut empty = candidates;
        empty.logits = [f32::NEG_INFINITY; CANDIDATES];
        assert!(sampler.propose(0, &empty).is_err(), "no finite logit");
        let mut outside = candidates;
        outside.ids[0] = 100;
        let draft = sampler.propose(0, &outside).unwrap();
        assert!(
            sampler.verify(0, &LOGITS, draft).is_err(),
            "proposal outside the target vocabulary"
        );
    }

    #[test]
    fn concentrated_q_is_an_exact_delta_without_a_draw() {
        let mut sampler = DsparkSampler::new(&seeded());
        let mut sharp = candidates();
        sharp.logits[3] = 30.0;
        let before = sampler.rng_state();
        assert_eq!(sampler.propose(0, &sharp).unwrap(), 3);
        assert_eq!(sampler.rng_state(), before);
    }

    #[test]
    fn greedy_keeps_the_point_mass_path() {
        let config = SamplingConfig {
            repeat_penalty: 1.1,
            ..SamplingConfig::default()
        };
        let greedy = SamplerState::new(config, &[]).unwrap();
        assert!(!DsparkSampler::new(&greedy).stochastic());
    }

    /// An expensive full-vocabulary p keeps the point-mass fallback and the
    /// sampled autoregressive seed trace.
    #[test]
    fn unfiltered_point_mass_preserves_the_seed_trace() {
        let config = SamplingConfig {
            temperature: 0.7,
            seed: 12,
            ..SamplingConfig::default()
        };
        let mut unfiltered = SamplerState::new(config, &[]).unwrap();
        let mut delta = DsparkSampler::new(&unfiltered);
        assert!(!delta.stochastic());
        let expected = unfiltered.sample(&LOGITS).unwrap();
        assert_eq!(delta.sample(&LOGITS).unwrap(), expected as i32);
        assert_eq!(delta.rng_state(), unfiltered.rng_state());
    }
}
