//! The Unicode predicates of the Qwen pre-tokenizer, as ICU 78 answers them
//! for the C++ tokenizer (Unicode 17 on both sides). `tests::matches_icu`
//! pins every code point's classification to a digest taken from ICU.

use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

pub fn is_letter(c: char) -> bool {
    c.general_category_group() == GeneralCategoryGroup::Letter
}

pub fn is_letter_or_mark(c: char) -> bool {
    matches!(
        c.general_category_group(),
        GeneralCategoryGroup::Letter | GeneralCategoryGroup::Mark
    )
}

pub fn is_number(c: char) -> bool {
    c.general_category_group() == GeneralCategoryGroup::Number
}

/// Unicode `White_Space`, like ICU's `u_isUWhiteSpace`.
pub fn is_space(c: char) -> bool {
    c.is_whitespace()
}

pub fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// Whether ICU's default simple case folding maps `c` to the ASCII letter
/// `letter`: the letter itself, its capital, and U+017F LONG S for `s`.
pub fn folds_to(c: char, letter: char) -> bool {
    c == letter || c == letter.to_ascii_uppercase() || (letter == 's' && c == '\u{17F}')
}

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_normalization::UnicodeNormalization;

    struct Fnv(u64);

    impl Fnv {
        fn new() -> Self {
            Self(0xcbf2_9ce4_8422_2325)
        }
        fn byte(&mut self, b: u8) {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        fn u32(&mut self, v: u32) {
            v.to_le_bytes().iter().for_each(|&b| self.byte(b));
        }
        fn bytes(&mut self, s: &str) {
            self.u32(s.len() as u32);
            s.bytes().for_each(|b| self.byte(b));
        }
    }

    /// Digests of ICU 78.2 (Unicode 17.0), printed by
    /// `src/core/tests/reference/icu_tokenizer_check.cpp unicode`: categories and
    /// whitespace of every code point, the case folds onto the contraction
    /// letters, NFC of every single code point and NFC of every canonical
    /// decomposition.
    #[test]
    fn matches_icu() {
        let (mut flags, mut folds, mut singles, mut recomposed) =
            (Fnv::new(), Fnv::new(), Fnv::new(), Fnv::new());
        let (mut fold_count, mut single_count, mut recomposed_count) = (0, 0, 0);
        for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            flags.byte(
                u8::from(is_letter(c))
                    | u8::from(is_letter_or_mark(c)) << 1
                    | u8::from(is_number(c)) << 2
                    | u8::from(is_space(c)) << 3,
            );
            for letter in "strevmld".chars() {
                if folds_to(c, letter) {
                    folds.u32(c as u32);
                    folds.byte(letter as u8);
                    fold_count += 1;
                }
            }
            let one = c.to_string();
            let nfc: String = one.nfc().collect();
            if nfc != one {
                singles.u32(c as u32);
                singles.bytes(&nfc);
                single_count += 1;
            }
            let nfd: String = one.nfd().collect();
            if nfd != one {
                recomposed.u32(c as u32);
                recomposed.bytes(&nfd.nfc().collect::<String>());
                recomposed_count += 1;
            }
        }
        assert_eq!(
            flags.0, 0x10a1_26d9_01f2_5daf,
            "general categories and White_Space"
        );
        assert_eq!(
            (fold_count, folds.0),
            (17, 0x61d8_f497_83db_7c42),
            "case folds"
        );
        assert_eq!(
            (single_count, singles.0),
            (1120, 0xfba4_0da9_b805_5f49),
            "NFC of code points"
        );
        assert_eq!(
            (recomposed_count, recomposed.0),
            (13_253, 0x1654_bb0b_23bd_8d5b),
            "NFC of decompositions"
        );
    }
}
