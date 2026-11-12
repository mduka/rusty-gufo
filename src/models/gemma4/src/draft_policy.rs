//! How many MTP drafts a cycle verifies (`draft_policy.cpp`). Every policy
//! decides from drafter outputs, verification outcomes and fixed costs,
//! never request timings, so greedy output is unchanged by the choice and
//! sampled output keeps the target distribution.

use crate::mtp_sampling::MtpCandidateLogits;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum DraftPolicy {
    /// Verify another draft while its calibrated survival probability is
    /// worth its marginal draft-plus-verification cost (the default).
    #[default]
    Calibrated,
    /// Continue while the product of raw drafter confidences stays above a
    /// fixed floor (0.5 greedy, 0.3 sampled with at most four drafts).
    Confidence,
    /// Always draft up to the configured limit.
    Fixed,
}

impl DraftPolicy {
    /// "calibrated" (also the empty string), "confidence" or "fixed"
    /// (`ParseDraftPolicy`).
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "" | "calibrated" => Ok(Self::Calibrated),
            "confidence" => Ok(Self::Confidence),
            "fixed" => Ok(Self::Fixed),
            _ => Err("Gemma 4 draft policy must be calibrated, confidence or fixed".into()),
        }
    }

    /// `DraftPolicyName`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Calibrated => "calibrated",
            Self::Confidence => "confidence",
            Self::Fixed => "fixed",
        }
    }
}

/// Where the calibrated policy keeps what it learned.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum DraftCalibrationScope {
    /// One table per model, learned across requests (default).
    #[default]
    Shared,
    /// Reset for every request: seeded sampled replay is exact.
    Request,
}

impl DraftCalibrationScope {
    /// "shared" (also the empty string) or "request"
    /// (`ParseDraftCalibrationScope`).
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "" | "shared" => Ok(Self::Shared),
            "request" => Ok(Self::Request),
            _ => Err("draft calibration must be shared or request".into()),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Request => "request",
        }
    }
}

// Costs on gfx1151 for the dense 31B family: drafter steps of the
// gemma4-assistant Q8_0 drafter measured in place 2026-10-20 (drafting time
// per drafter step over fixed-length cycles, --draft-trace), verification
// forwards (target rows with logits, median of seven) remeasured 2026-10-05
// on UD-Q4_K_XL after its projections moved to the WMMA kernels, whose cost
// barely grows with rows (a row never costs less than a narrower forward:
// the medians differ by noise there); two to four rows from 4K on lowered
// 2026-10-17 by the measured global split-attention saving of one wave per
// head (ten layers). Only cost ratios steer the decision.
const DEPTHS: [u32; 5] = [0, 4096, 16384, 32768, 65536];
const DRAFT_STEP_MS: [f32; 5] = [1.93, 2.39, 2.68, 3.13, 4.04];
// Verification rows 2-16 at 4K-64K lowered 2026-10-19 by the measured
// global split-attention savings (ten layers).
#[rustfmt::skip]
const VERIFY_MS: [[f32; 16]; 5] = [
    [99.5, 99.8, 100.6, 101.4, 102.0, 102.8, 103.3, 103.9, 109.0, 109.6, 110.1, 111.1, 111.7, 112.3, 112.7, 113.1],
    [104.1, 105.1, 106.3, 107.9, 108.5, 110.3, 111.5, 112.8, 120.5, 121.8, 122.6, 123.7, 125.2, 126.7, 126.8, 129.9],
    [108.2, 109.6, 111.2, 113.3, 116.4, 118.8, 121.4, 123.8, 134.6, 136.8, 139.5, 141.7, 144.0, 146.1, 146.9, 149.4],
    [112.5, 114.9, 117.2, 119.6, 124.6, 128.5, 132.7, 136.6, 149.7, 153.8, 157.9, 161.1, 164.9, 167.6, 169.2, 173.8],
    [121.0, 124.3, 128.3, 132.4, 140.2, 145.8, 153.8, 160.0, 180.0, 187.3, 194.8, 202.0, 208.2, 212.0, 216.0, 222.7],
];

// The 26B-A4B (UD-Q4_K_XL target, Unsloth Q8_0 drafter), measured
// 2026-09-29 on varied English prose (the rows' routing, and so the experts
// they read, follows the text) up to the 16 rows a batched forward
// verifies: every verified row adds the experts it routes to, so
// verification grows several times faster per row than on the dense family.
// Two to four rows lowered 2026-10-17 by the measured global split-attention
// saving of one wave per head (five layers). Drafter steps measured in place
// 2026-10-20 like the dense family's; the small head over frequent tokens
// drafts 95% of ~31K prose cycles (1.78 ms per step there, the full head
// 2.21 ms).
const EXPERT_DRAFT_STEP_MS: [f32; 5] = [1.25, 1.43, 1.63, 1.83, 2.17];
// Verification rows 2-16 at 4K-64K lowered 2026-10-19 by the measured
// global split-attention savings (five layers).
#[rustfmt::skip]
const EXPERT_VERIFY_MS: [[f32; 16]; 5] = [
    [18.3, 21.1, 23.1, 25.4, 27.3, 29.1, 30.8, 31.8, 33.2, 34.4, 37.1, 40.6, 42.3, 44.8, 44.6, 46.0],
    [19.8, 22.8, 25.7, 28.0, 30.0, 32.7, 34.7, 36.2, 37.1, 38.9, 42.0, 45.5, 48.6, 50.9, 50.8, 52.4],
    [20.8, 24.0, 26.6, 29.3, 31.9, 34.8, 37.5, 39.2, 40.7, 42.8, 46.2, 49.9, 51.8, 54.5, 54.7, 56.8],
    [21.7, 25.2, 28.0, 31.0, 34.3, 36.8, 39.4, 41.3, 43.6, 46.0, 49.6, 54.0, 56.6, 59.7, 59.9, 61.8],
    [23.6, 25.7, 29.1, 32.4, 36.8, 40.5, 43.9, 46.2, 49.2, 52.2, 57.0, 61.4, 66.3, 69.5, 69.9, 72.8],
];

/// The drafter's confidence in its own step: AdaEDL's entropy bound
/// 1 - sqrt(0.2 H) over the top-64 softmax, clamped to \[0, 1\]
/// (`DraftSignal`).
pub fn draft_signal(candidates: &MtpCandidateLogits) -> f32 {
    if candidates.size == 0 {
        return 0.0;
    }
    let logits = &candidates.logits[..candidates.size];
    let top = logits[1..].iter().fold(logits[0], |m, &v| m.max(v));
    let mut total = 0.0f64;
    let mut weighted = 0.0f64;
    for &logit in logits {
        let shifted = f64::from(logit - top);
        let e = shifted.exp();
        total += e;
        weighted += e * shifted;
    }
    let entropy = (total.ln() - weighted / total).max(0.0);
    let signal = 1.0 - (0.2 * entropy).sqrt();
    signal.clamp(0.0, 1.0) as f32
}

/// gfx1151 costs of one cycle in milliseconds at a context depth: `draft[n]`
/// for n drafter steps and `verify[r]` for r verified rows, for the dense 31B
/// family or the mixture-of-experts 26B-A4B (`experts`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DraftCosts {
    pub draft: [f32; 16],
    /// Up to kSplitRows rows plus the next one a decision looks at.
    pub verify: [f32; 18],
}

/// Costs from one table: linear between measured depths, past the deepest
/// one the last interval's slope continues; rows past the table keep its last
/// row step (`Interpolate`).
fn interpolate(context: u32, draft_step: &[f32; 5], verify: &[[f32; 16]; 5]) -> DraftCosts {
    const ROWS: usize = 16;
    let mut hi = 1;
    while hi + 1 < DEPTHS.len() && context > DEPTHS[hi] {
        hi += 1;
    }
    let lo = hi - 1;
    let t = (context as f32 - DEPTHS[lo] as f32) / (DEPTHS[hi] - DEPTHS[lo]) as f32;
    let lerp = |a: f32, b: f32| a.max(a + t * (b - a));
    let mut costs = DraftCosts::default();
    let step = lerp(draft_step[lo], draft_step[hi]);
    for (n, draft) in costs.draft.iter_mut().enumerate() {
        *draft = step * n as f32;
    }
    for r in 1..=ROWS.min(costs.verify.len() - 1) {
        costs.verify[r] = lerp(verify[lo][r - 1], verify[hi][r - 1]);
    }
    for r in ROWS + 1..costs.verify.len() {
        costs.verify[r] = costs.verify[r - 1] + (costs.verify[ROWS] - costs.verify[ROWS - 1]);
    }
    costs.verify[0] = costs.verify[1];
    costs
}

/// `DraftCostsAt`.
pub fn draft_costs_at(context: u32, experts: bool) -> DraftCosts {
    if experts {
        interpolate(context, &EXPERT_DRAFT_STEP_MS, &EXPERT_VERIFY_MS)
    } else {
        interpolate(context, &DRAFT_STEP_MS, &VERIFY_MS)
    }
}

/// Signal to acceptance probability, learned from verification: accepted
/// drafts and the first rejected one. Each bin starts at its midpoint with
/// the weight of [`DraftCalibration::PRIOR_WEIGHT`] observations.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DraftCalibration {
    accepted: [f32; Self::BINS],
    seen: [f32; Self::BINS],
}

impl DraftCalibration {
    pub const BINS: usize = 8;
    pub const PRIOR_WEIGHT: f32 = 8.0;

    /// The bin a signal falls in.
    pub fn bin(signal: f32) -> usize {
        let clamped = signal.clamp(0.0, 1.0);
        (Self::BINS - 1).min((clamped * Self::BINS as f32) as usize)
    }

    pub fn estimate(&self, signal: f32) -> f32 {
        let bin = Self::bin(signal);
        let prior = (bin as f32 + 0.5) / Self::BINS as f32;
        (self.accepted[bin] + Self::PRIOR_WEIGHT * prior) / (self.seen[bin] + Self::PRIOR_WEIGHT)
    }

    pub fn observe(&mut self, signal: f32, accepted: bool) {
        let bin = Self::bin(signal);
        self.accepted[bin] += if accepted { 1.0 } else { 0.0 };
        self.seen[bin] += 1.0;
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// The rest of a batched verification forward as one session sees it: the
/// other sessions' verification rows, the tokens their cycles are expected
/// to emit and their drafter time. Empty for a lone session.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DraftBatch {
    pub rows: u32,
    pub expected: f32,
    pub draft_ms: f32,
}

/// The last [`CalibratedChain::include`]'s comparison past `min_drafts`: the
/// expected gain of stopping later against the cost (the chain stops when
/// gain < cost).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Decision {
    pub compared: bool,
    pub gain: f32,
    pub cost: f32,
}

/// One cycle's calibrated decisions (`CalibratedChain`). A draft is verified
/// while the extra tokens it is expected to add per millisecond of extra
/// work (its survival over one more verification row and drafter step) at
/// least match the cycle's current rate; a lone cycle always verifies its
/// first `min_drafts` (at least one). In a batch the cycle is the whole
/// forward: the other sessions' rows price the extra row and their tokens
/// and drafter time set the rate, and a session may verify no draft at all.
#[derive(Debug, Clone)]
pub struct CalibratedChain<'c> {
    calibration: &'c DraftCalibration,
    costs: &'c DraftCosts,
    min_drafts: u32,
    cap: u32,
    others: DraftBatch,
    kept: u32,
    steps: u32,
    /// Per kept draft: the chain's survival before it and its acceptance.
    reach: [f32; 16],
    accept: [f32; 16],
    survival: f32,
    /// The target's own token is always emitted.
    expected: f32,
    decision: Decision,
}

impl<'c> CalibratedChain<'c> {
    pub fn new(
        calibration: &'c DraftCalibration,
        costs: &'c DraftCosts,
        min_drafts: u32,
        cap: u32,
        others: DraftBatch,
    ) -> Self {
        Self {
            calibration,
            costs,
            // Beside other sessions even the first draft must pay for its
            // row.
            min_drafts: if others.rows > 0 {
                0
            } else {
                min_drafts.max(1)
            },
            cap: cap.min(costs.draft.len() as u32 - 1),
            others,
            kept: 0,
            steps: 0,
            reach: [0.0; 16],
            accept: [0.0; 16],
            survival: 1.0,
            expected: 1.0,
            decision: Decision::default(),
        }
    }

    /// Price of a forward verifying `rows` of this session's rows beside the
    /// other sessions'.
    pub fn verify(&self, rows: u32) -> f32 {
        self.costs.verify[((self.others.rows + rows) as usize).min(self.costs.verify.len() - 1)]
    }

    /// Whether a first draft certain to be accepted would pay for its
    /// drafter step and row; when not, the cycle need not draft at all.
    pub fn first_draft_can_pay(&self) -> bool {
        if self.min_drafts > 0 || self.cap == 0 {
            return self.cap > 0;
        }
        // Include's rule for the first draft with survival one, against a
        // cycle that runs no drafter step.
        let without = self.others.draft_ms + self.verify(1);
        let with = self.others.draft_ms + self.costs.draft[1] + self.verify(2);
        without >= (self.others.expected + self.expected) * (with - without)
    }

    /// Decides the next draft from its signal.
    pub fn include(&mut self, signal: f32) -> bool {
        let index = self.kept; // drafter steps so far: index + 1
        self.decision = Decision::default();
        if index >= self.cap {
            return false;
        }
        self.steps = index + 1;
        let accept = self.calibration.estimate(signal);
        let survival = self.survival * accept;
        if index >= self.min_drafts {
            // Stopping here verifies `index` drafts after one more drafter
            // step than needed; verifying this one adds a row, and going on
            // adds a step.
            let current = self.others.draft_ms
                + self.costs.draft[(index + 1) as usize]
                + self.verify(index + 1);
            let next = self.others.draft_ms
                + self.costs.draft[(index + 2).min(self.cap) as usize]
                + self.verify(index + 2);
            self.decision = Decision {
                compared: true,
                gain: survival * current,
                cost: (self.others.expected + self.expected) * (next - current),
            };
            if self.decision.gain < self.decision.cost {
                return false;
            }
        }
        self.reach[self.kept as usize] = self.survival;
        self.accept[self.kept as usize] = accept;
        self.survival = survival;
        self.expected += survival;
        self.kept += 1;
        true
    }

    /// After the chain: whether a sibling of draft `depth` (0-based, decided
    /// in depth order) pays for one more verification row. It adds a token
    /// when the chain reaches that draft, the draft is rejected and the
    /// sibling accepted (`hit`); `rows` counts the cycle's rows so far.
    pub fn include_sibling(&mut self, depth: u32, rows: u32, hit: f32) -> bool {
        if depth >= self.kept {
            return false;
        }
        let gain = self.reach[depth as usize] * (1.0 - self.accept[depth as usize]) * hit;
        // The drafter does not run again: only the row is extra.
        let current =
            self.others.draft_ms + self.costs.draft[self.steps as usize] + self.verify(rows);
        let next = current - self.verify(rows) + self.verify(rows + 1);
        if gain * current < (self.others.expected + self.expected) * (next - current) {
            return false;
        }
        self.expected += gain;
        true
    }

    /// Replaces the view of the other sessions (they draft alongside).
    pub fn set_others(&mut self, others: DraftBatch) {
        self.others = others;
    }

    /// Tokens this session's cycle is expected to emit so far.
    pub fn expected(&self) -> f32 {
        self.expected
    }

    /// Drafter time of the steps decided so far (a stop still ran its step).
    pub fn draft_ms(&self) -> f32 {
        self.costs.draft[self.steps as usize]
    }

    pub fn last_decision(&self) -> Decision {
        self.decision
    }
}
