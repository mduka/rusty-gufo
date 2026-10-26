//! Draft length per verification step, a port of the single-session
//! `DFlashLengthController` (`dflash_policy.hpp`). The adaptive policy tracks
//! a censored mean of accepted drafts and picks the length that maximises
//! expected emitted tokens per unit of measured gfx1151 cost; all arithmetic
//! is `f32` in the C++ order, so the lengths match exactly.

/// Longest DFlash2 draft block.
pub const MAX_DRAFT_TOKENS: u32 = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftPolicy {
    Fixed,
    Adaptive,
}

impl DraftPolicy {
    /// `ParseDFlashDraftPolicy`: empty means adaptive.
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "fixed" => Ok(Self::Fixed),
            "" | "adaptive" => Ok(Self::Adaptive),
            _ => Err("DFlash2 draft policy must be fixed or adaptive".into()),
        }
    }
}

/// Relative draft-plus-verification cost of 1..=7 drafts for one Q4 request.
const Q4_RELATIVE_COST: [f32; 7] = [1.0, 1.02, 1.045, 1.08, 1.14, 1.20, 1.29];
/// The same for two, four, six and eight greedy Q4 requests verified together.
const Q4_PAIRED_RELATIVE_COST: [f32; 7] = [1.0, 1.04, 1.10, 1.48, 1.45, 1.50, 1.50];
const Q4_FOUR_REQUEST_RELATIVE_COST: [f32; 7] = [1.0, 1.30, 1.33, 1.97, 2.12, 2.29, 2.32];
const Q4_SIX_REQUEST_RELATIVE_COST: [f32; 7] = [1.0, 1.44, 1.59, 1.89, 2.21, 2.43, 2.47];
const Q4_EIGHT_REQUEST_RELATIVE_COST: [f32; 7] = [1.0, 1.52, 1.68, 2.21, 2.38, 2.89, 3.08];
/// Complete Q8 cycles of two, four, six and eight requests.
const Q8_PAIRED_RELATIVE_COST: [f32; 7] = [1.0, 1.015, 1.020, 1.089, 1.127, 1.333, 1.182];
const Q8_FOUR_REQUEST_RELATIVE_COST: [f32; 7] = [1.0, 1.06, 1.153, 1.914, 1.907, 2.089, 1.853];
const Q8_SIX_REQUEST_RELATIVE_COST: [f32; 7] = [1.0, 1.760, 1.763, 2.141, 2.352, 2.505, 2.404];
const Q8_EIGHT_REQUEST_RELATIVE_COST: [f32; 7] = [1.0, 1.610, 1.561, 2.231, 2.183, 2.850, 2.813];
/// Extra cost from 2K to 32K context of the 16 full-attention layers.
const Q4_CONTEXT_COST: [f32; 7] = [0.11, 0.14, 0.21, 0.23, 0.25, 0.31, 0.35];

#[derive(Debug, Clone, PartialEq)]
pub struct LengthController {
    policy: DraftPolicy,
    limit: u32,
    q8_target: bool,
    mean: f32,
    last_full_width: u32,
}

impl LengthController {
    pub fn new(policy: DraftPolicy, limit: u32, q8_target: bool) -> Self {
        let mut controller = Self {
            policy,
            limit: limit.clamp(1, MAX_DRAFT_TOKENS),
            q8_target,
            mean: 0.0,
            last_full_width: 0,
        };
        controller.reset();
        controller
    }

    /// Drafts for the next block at `position`, at most `budget`
    /// (`Choose`); `greedy_batch_size` greedy requests verify together.
    pub fn choose(&self, budget: u32, position: u32, greedy_batch_size: usize) -> u32 {
        let cap = budget.min(self.limit);
        if self.policy == DraftPolicy::Fixed || cap == 0 {
            return cap;
        }
        let limit = self.limit as f32;
        // Preserve the full-acceptance probe at C5-C8.
        if (5..=8).contains(&greedy_batch_size) && self.mean == limit {
            return cap;
        }
        // At the limit full acceptance is censored: probe the widest block.
        let probability = if self.mean == limit {
            1.0
        } else {
            self.mean / (self.mean + 1.0)
        };
        let context_scale = position.saturating_sub(2048) as f32 / 30720.0;
        let (mut relative_costs, mut context_multiplier) = (&Q4_RELATIVE_COST, 1.0f32);
        match greedy_batch_size {
            2 => (relative_costs, context_multiplier) = (&Q4_PAIRED_RELATIVE_COST, 1.92),
            3 | 4 => (relative_costs, context_multiplier) = (&Q4_FOUR_REQUEST_RELATIVE_COST, 3.20),
            5 | 6 => (relative_costs, context_multiplier) = (&Q4_SIX_REQUEST_RELATIVE_COST, 3.50),
            7 | 8 => (relative_costs, context_multiplier) = (&Q4_EIGHT_REQUEST_RELATIVE_COST, 4.42),
            _ => {}
        }
        if self.q8_target && greedy_batch_size == 4 {
            relative_costs = &Q8_FOUR_REQUEST_RELATIVE_COST;
            context_multiplier = 4.0 * 100.0 / 170.66;
        }
        let attention_cost_scale = context_scale * context_multiplier;
        let (mut survival, mut expected, mut best_score, mut best_length) =
            (1.0f32, 1.0f32, 0.0f32, 1);
        for length in 1..=cap {
            survival *= probability;
            expected += survival;
            let index = length as usize - 1;
            let cost = if self.q8_target && greedy_batch_size != 4 {
                1.0 + 0.02 * length as f32
            } else {
                relative_costs[index] + attention_cost_scale * Q4_CONTEXT_COST[index]
            };
            let score = expected / cost;
            if score > best_score {
                best_score = score;
                best_length = length;
            }
        }
        best_length
    }

    /// One shared width for an all-adaptive Q8 cohort (`ChooseGreedyBatch`):
    /// each history stays private, their expected tokens add up; `None` when
    /// the cohort does not qualify.
    pub fn choose_greedy_batch(
        controllers: &[&LengthController],
        budgets: &[u32],
        positions: &[u32],
    ) -> Option<u32> {
        let users = controllers.len();
        if budgets.len() != users || positions.len() != users {
            return None;
        }
        let (relative_costs, base_ms, probe_width) = match users {
            2 => (&Q8_PAIRED_RELATIVE_COST, 158.26f64, 3u32),
            3 | 4 => (&Q8_FOUR_REQUEST_RELATIVE_COST, 170.66, 3),
            5 | 6 => (&Q8_SIX_REQUEST_RELATIVE_COST, 189.32, 1),
            7 | 8 => (&Q8_EIGHT_REQUEST_RELATIVE_COST, 209.20, 1),
            _ => return None,
        };
        let mut cap = MAX_DRAFT_TOKENS;
        let mut probabilities = [0f64; 8];
        let mut attention_cost_scale = 0.0f64;
        let mut full_tile = true;
        let mut neutral_or_full = !users.is_multiple_of(2);
        let mut full_histories = 0usize;
        for (index, controller) in controllers.iter().enumerate() {
            if !controller.q8_target || controller.policy != DraftPolicy::Adaptive {
                return None;
            }
            cap = cap.min(budgets[index].min(controller.limit));
            let mean = f64::from(controller.mean);
            probabilities[index] = if mean == f64::from(controller.limit) {
                1.0
            } else {
                mean / (mean + 1.0)
            };
            attention_cost_scale +=
                f64::from(positions[index].saturating_sub(2048)) / 30720.0 * 100.0 / base_ms;
            full_tile &= controller.last_full_width >= probe_width;
            let full_history = controller.last_full_width >= 3;
            full_histories += usize::from(full_history && mean >= 4.5);
            neutral_or_full &= full_history
                || (controller.last_full_width == 0
                    && controller.mean == 3.0f32.min(controller.limit as f32));
        }
        if cap == 0 {
            return None;
        }
        if full_tile || (neutral_or_full && full_histories >= 2) {
            return Some(cap);
        }
        let mut survival = [1.0f64; 8];
        let (mut expected, mut best_score, mut best_length) = (users as f64, 0.0f64, 1);
        for length in 1..=cap {
            for index in 0..users {
                survival[index] *= probabilities[index];
                expected += survival[index];
            }
            let index = length as usize - 1;
            let cost = f64::from(relative_costs[index])
                + attention_cost_scale * f64::from(Q4_CONTEXT_COST[index]);
            let score = expected / cost;
            if score > best_score {
                best_score = score;
                best_length = length;
            }
        }
        Some(best_length)
    }

    /// Feedback of one verified block.
    pub fn observe(&mut self, accepted: usize, drafted: usize) {
        if self.policy == DraftPolicy::Fixed || drafted == 0 {
            return;
        }
        let drafted = drafted.min(self.limit as usize);
        let accepted = accepted.min(drafted);
        let full = accepted == drafted;
        self.last_full_width = if full { drafted as u32 } else { 0 };
        self.mean = if full {
            (self.mean + 0.25 * accepted as f32).min(self.limit as f32)
        } else {
            0.75 * self.mean + 0.25 * accepted as f32
        };
    }

    pub fn reset(&mut self) {
        self.mean = 3.0f32.min(self.limit as f32);
        self.last_full_width = 0;
    }

    /// The state a snapshot keeps: mean and last full width.
    pub fn state(&self) -> (f32, u32) {
        (self.mean, self.last_full_width)
    }

    pub fn restore(&mut self, (mean, last_full_width): (f32, u32)) -> Result<(), String> {
        if !mean.is_finite()
            || mean < 0.0
            || mean > self.limit as f32
            || last_full_width > self.limit
        {
            return Err("DFlash2 controller state is invalid".into());
        }
        self.mean = mean;
        self.last_full_width = last_full_width;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The C++ `gufo bench`-style run of the haiku prompt (20 tokens, 64
    /// generated, context 6144): (drafted, accepted, emitted) per step, from
    /// `tests/reference/cpp_decode_dump.cpp dflash`. Budgets and positions follow the
    /// verifier; the last steps are limited by the tokens still to emit.
    const CPP_STEPS: [(u32, usize, u32); 34] = [
        (6, 2, 3),
        (5, 0, 1),
        (4, 3, 4),
        (5, 0, 1),
        (4, 0, 1),
        (4, 2, 3),
        (4, 2, 3),
        (4, 0, 1),
        (4, 0, 1),
        (3, 1, 2),
        (3, 1, 2),
        (3, 0, 1),
        (3, 0, 1),
        (3, 0, 1),
        (2, 0, 1),
        (2, 0, 1),
        (2, 0, 1),
        (1, 1, 2),
        (2, 2, 3),
        (3, 3, 4),
        (4, 4, 5),
        (5, 5, 6),
        (6, 1, 2),
        (6, 1, 2),
        (5, 0, 1),
        (4, 0, 1),
        (4, 0, 1),
        (4, 0, 1),
        (3, 0, 1),
        (3, 1, 2),
        (3, 0, 1),
        (3, 1, 2),
        (1, 0, 1),
        (0, 0, 1),
    ];

    #[test]
    fn lengths_replay_the_cpp_verifier_run() {
        let mut controller = LengthController::new(DraftPolicy::Adaptive, 7, false);
        let (mut position, mut emitted) = (20u32, 0u32);
        for (step, &(drafted, accepted, count)) in CPP_STEPS.iter().enumerate() {
            let remaining = 64 - emitted;
            let max_draft = if remaining > 1 {
                7.min(remaining - 1)
            } else {
                0
            };
            let chosen = if max_draft == 0 {
                0
            } else {
                controller.choose(max_draft.min(6144 - position - 1), position, 1)
            };
            assert_eq!(chosen, drafted, "step {step}");
            if chosen > 0 {
                controller.observe(accepted, chosen as usize);
            }
            position += count;
            emitted += count;
        }
        assert_eq!(emitted, 64);
    }

    #[test]
    fn fixed_policy_state_and_parsing() {
        let fixed = LengthController::new(DraftPolicy::Fixed, 9, false);
        assert_eq!((fixed.choose(5, 0, 1), fixed.choose(9, 0, 1)), (5, 7));
        let mut adaptive = LengthController::new(DraftPolicy::Adaptive, 7, false);
        assert_eq!(adaptive.state(), (3.0, 0));
        adaptive.observe(4, 4);
        assert_eq!(adaptive.state(), (4.0, 4));
        assert!(adaptive.restore((8.0, 0)).is_err());
        assert!(adaptive.restore((2.5, 1)).is_ok());
        assert!(LengthController::new(DraftPolicy::Adaptive, 7, false).choose(7, 30_000, 1) <= 6);
        assert_eq!(DraftPolicy::parse(""), Ok(DraftPolicy::Adaptive));
        assert!(DraftPolicy::parse("wide").is_err());
    }
}
