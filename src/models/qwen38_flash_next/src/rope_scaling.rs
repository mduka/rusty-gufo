//! Static YaRN context extension (`rope_scaling.{hpp,cpp}`), as Hugging Face's `rope_type: yarn`: a truncated correction
//! range, a linear ramp between the beta bounds, and the attention factor
//! 0.1 ln(factor) + 1 on cos and sin. Factor 1 is off and leaves every rope
//! site on its original instruction sequence.

/// Rope angles use float positions, which are exact below 2^24.
pub const MAX_ROPE_POSITIONS: u32 = 1 << 24;

/// A static YaRN scaling (`RopeScaling`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeScaling {
    pub factor: f32,
    /// The pretraining context (262144).
    pub original_context: u32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl Default for RopeScaling {
    fn default() -> Self {
        Self {
            factor: 1.0,
            original_context: 0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

impl RopeScaling {
    pub fn enabled(&self) -> bool {
        self.factor != 1.0
    }

    /// A disabled scaling with the default fields, whatever unused fields it
    /// carries, so two off configurations compare equal
    /// (`NormalizedRopeScaling`).
    pub fn normalized(self) -> Self {
        if self.enabled() {
            self
        } else {
            Self::default()
        }
    }
}

/// The launch constants every HIP rope site receives (`YarnRope`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnRope {
    pub enabled: u32,
    /// First ramped rotary pair.
    pub low: f32,
    /// Pair from which frequencies are divided.
    pub high: f32,
    pub inv_factor: f32,
    /// Multiplies cos and sin.
    pub mscale: f32,
}

impl Default for YarnRope {
    fn default() -> Self {
        Self {
            enabled: 0,
            low: 0.0,
            high: 0.0,
            inv_factor: 1.0,
            mscale: 1.0,
        }
    }
}

impl YarnRope {
    /// The frequency multiplier of rotary pair `pair`: 1 below `low`,
    /// 1 / factor from `high`, linear between (HF `linear_ramp_factor`).
    pub fn pair_multiplier(&self, pair: u32) -> f32 {
        let width = self.high - self.low;
        let span = if width > 0.001 { width } else { 0.001 };
        let ramp = ((pair as f32 - self.low) / span).clamp(0.0, 1.0);
        1.0 - ramp * (1.0 - self.inv_factor)
    }
}

/// Rejects non-finite or sub-unit factors, an original context outside
/// (0, native], betas that are not beta_fast > beta_slow > 0, and scaled
/// contexts past [`MAX_ROPE_POSITIONS`]; factor 1 is always valid (off).
pub fn validate_rope_scaling(scaling: &RopeScaling, native_context: u32) -> Result<(), String> {
    if !scaling.factor.is_finite() || scaling.factor < 1.0 {
        return Err("YaRN factor must be finite and at least 1".into());
    }
    if !scaling.enabled() {
        return Ok(());
    }
    if scaling.original_context == 0 || scaling.original_context > native_context {
        return Err("YaRN original context must be within the native context".into());
    }
    if !scaling.beta_fast.is_finite()
        || !scaling.beta_slow.is_finite()
        || scaling.beta_slow <= 0.0
        || scaling.beta_fast <= scaling.beta_slow
    {
        return Err("YaRN betas must satisfy beta_fast > beta_slow > 0".into());
    }
    if f64::from(scaling.original_context) * f64::from(scaling.factor)
        > f64::from(MAX_ROPE_POSITIONS)
    {
        return Err("YaRN context exceeds 16777216 exact rope positions".into());
    }
    Ok(())
}

/// The native context when off; floor(original x factor), never below the
/// native context (`ScaledContextLength`).
pub fn scaled_context_length(scaling: &RopeScaling, native_context: u32) -> u32 {
    if !scaling.enabled() || !scaling.factor.is_finite() {
        return native_context;
    }
    let scaled = (f64::from(scaling.original_context) * f64::from(scaling.factor)).floor();
    let bounded = scaled.min(f64::from(MAX_ROPE_POSITIONS)) as u32;
    native_context.max(bounded)
}

/// The static YaRN a requested serving context needs (factor = requested /
/// native, Qwen's own advice); `None` without a native context or within it
/// (`RopeScalingForContext`). The float factor is nudged up when it would
/// round the scaled context below `requested`.
pub fn rope_scaling_for_context(requested: u32, native: u32) -> Option<RopeScaling> {
    if native == 0 || requested <= native {
        return None;
    }
    let mut scaling = RopeScaling {
        factor: (f64::from(requested) / f64::from(native)) as f32,
        original_context: native,
        ..RopeScaling::default()
    };
    while scaling.factor.is_finite()
        && scaled_context_length(&scaling, native) < requested
        && scaled_context_length(&scaling, native) < MAX_ROPE_POSITIONS
    {
        scaling.factor = next_up(scaling.factor);
    }
    Some(scaling)
}

/// The next float above a finite positive `value`.
fn next_up(value: f32) -> f32 {
    f32::from_bits(value.to_bits() + 1)
}

/// Applies the `--enable-yarn` opt-in to a requested context
/// (`ResolveRopeScaling`): above `native` the flag is required, at or below
/// it the flag has no effect and is rejected. An unknown native context (0)
/// leaves YaRN off for the model's own context check, and rejects the flag.
pub fn resolve_rope_scaling(
    requested: u32,
    native: u32,
    enable_yarn: bool,
) -> Result<Option<RopeScaling>, String> {
    if native == 0 && enable_yarn {
        return Err("--enable-yarn needs the model's native context length".into());
    }
    if native == 0 {
        return Ok(None);
    }
    if requested > native && !enable_yarn {
        return Err(format!(
            "context above the native {native} tokens needs static YaRN, which applies to \
             every request; pass --enable-yarn (requested {requested})"
        ));
    }
    if requested <= native && enable_yarn {
        return Err(format!(
            "--enable-yarn has no effect: context {requested} is within the native {native} \
             tokens"
        ));
    }
    Ok(rope_scaling_for_context(requested, native))
}

/// The correction range and scale for a rotary dimension and base
/// (`MakeYarnRope`, HF `find_correction_range`).
pub fn make_yarn_rope(scaling: &RopeScaling, rotary_dim: u32, theta: f32) -> YarnRope {
    if !scaling.enabled() {
        return YarnRope::default();
    }
    // The pair index completing `rotations` turns over the original context.
    let dim = |rotations: f64| {
        f64::from(rotary_dim)
            * (f64::from(scaling.original_context) / (rotations * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * f64::from(theta).ln())
    };
    let mut low = dim(f64::from(scaling.beta_fast)).floor().max(0.0);
    let mut high = dim(f64::from(scaling.beta_slow))
        .ceil()
        .min(f64::from(rotary_dim) - 1.0);
    if high < low {
        // HF's ramp clamps every pair to 0 for a negative range, so no pair
        // interpolates: both bounds past the last pair do the same here.
        low = f64::from(rotary_dim);
        high = low;
    }
    YarnRope {
        enabled: 1,
        low: low as f32,
        high: high as f32,
        inv_factor: 1.0 / scaling.factor,
        mscale: (0.1 * f64::from(scaling.factor).ln() + 1.0) as f32,
    }
}

/// The disk-cache position policy (`RopePositionPolicy`): `absolute-v1`
/// when off, so existing entries keep their identity.
pub fn rope_position_policy(scaling: &RopeScaling) -> String {
    if !scaling.enabled() {
        return "absolute-v1".into();
    }
    format!(
        "absolute-yarn-v1;factor={};original={};beta_fast={};beta_slow={}",
        hexfloat(scaling.factor),
        scaling.original_context,
        hexfloat(scaling.beta_fast),
        hexfloat(scaling.beta_slow)
    )
}

/// `value` as C++ `std::hexfloat` prints a float (promoted to double, as
/// printf's `%a`): `0x1.9p+0` for 1.5625.
fn hexfloat(value: f32) -> String {
    let value = f64::from(value);
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0x0p+0".into()
        } else {
            "0x0p+0".into()
        };
    }
    let bits = value.to_bits();
    let sign = if bits >> 63 != 0 { "-" } else { "" };
    let exponent = ((bits >> 52) & 0x7ff) as i64;
    let mantissa = bits & ((1u64 << 52) - 1);
    let (lead, exponent) = if exponent == 0 {
        (0, -1022)
    } else {
        (1, exponent - 1023)
    };
    let mut digits = format!("{mantissa:013x}");
    while digits.ends_with('0') {
        digits.pop();
    }
    let fraction = if digits.is_empty() {
        String::new()
    } else {
        format!(".{digits}")
    };
    format!("{sign}0x{lead}{fraction}p{exponent:+}")
}

#[cfg(test)]
mod tests {
    use super::*;

    const NATIVE: u32 = 262_144;

    #[test]
    fn contexts_resolve_as_upstream() {
        assert_eq!(resolve_rope_scaling(NATIVE, NATIVE, false), Ok(None));
        assert!(
            resolve_rope_scaling(NATIVE, NATIVE, true)
                .unwrap_err()
                .contains("no effect")
        );
        assert!(
            resolve_rope_scaling(409_600, NATIVE, false)
                .unwrap_err()
                .contains("pass --enable-yarn")
        );
        let scaling = resolve_rope_scaling(409_600, NATIVE, true)
            .unwrap()
            .unwrap();
        assert_eq!(scaling.factor, 1.5625);
        assert_eq!(scaled_context_length(&scaling, NATIVE), 409_600);
        assert!(validate_rope_scaling(&scaling, NATIVE).is_ok());
        assert!(resolve_rope_scaling(1, 0, true).is_err());
        assert_eq!(resolve_rope_scaling(1, 0, false), Ok(None));
        // A non-power-of-two native context nudges the factor up.
        let odd = rope_scaling_for_context(1_000_003, 3).unwrap();
        assert!(scaled_context_length(&odd, 3) >= 1_000_003);
    }

    #[test]
    fn invalid_scalings_are_rejected() {
        let enabled = RopeScaling {
            factor: 2.0,
            original_context: NATIVE,
            ..RopeScaling::default()
        };
        for invalid in [
            RopeScaling {
                factor: 0.5,
                ..enabled
            },
            RopeScaling {
                factor: f32::NAN,
                ..enabled
            },
            RopeScaling {
                original_context: 0,
                ..enabled
            },
            RopeScaling {
                original_context: NATIVE + 1,
                ..enabled
            },
            RopeScaling {
                beta_slow: 40.0,
                ..enabled
            },
            RopeScaling {
                factor: 128.0,
                ..enabled
            },
        ] {
            assert!(
                validate_rope_scaling(&invalid, NATIVE).is_err(),
                "{invalid:?}"
            );
        }
        assert!(validate_rope_scaling(&RopeScaling::default(), NATIVE).is_ok());
    }

    #[test]
    fn yarn_constants_and_policy() {
        assert_eq!(
            make_yarn_rope(&RopeScaling::default(), 64, 1e7),
            YarnRope::default()
        );
        let scaling = RopeScaling {
            factor: 1.5625,
            original_context: NATIVE,
            ..RopeScaling::default()
        };
        let yarn = make_yarn_rope(&scaling, 64, 1e7);
        assert_eq!(yarn.enabled, 1);
        assert!(yarn.low <= yarn.high && yarn.high <= 63.0);
        assert_eq!(yarn.inv_factor, 0.64);
        assert!((yarn.mscale - (0.1 * 1.5625f64.ln() + 1.0) as f32).abs() == 0.0);
        assert_eq!(yarn.pair_multiplier(0), 1.0);
        assert_eq!(yarn.pair_multiplier(1000), yarn.inv_factor);
        assert_eq!(rope_position_policy(&RopeScaling::default()), "absolute-v1");
        assert_eq!(
            rope_position_policy(&scaling),
            "absolute-yarn-v1;factor=0x1.9p+0;original=262144;beta_fast=0x1p+5;beta_slow=0x1p+0"
        );
        assert_eq!(
            RopeScaling {
                original_context: 7,
                ..RopeScaling::default()
            }
            .normalized(),
            RopeScaling::default()
        );
    }
}
