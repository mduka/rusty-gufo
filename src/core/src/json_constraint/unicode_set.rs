//! The parts of `icu::UnicodeSet` the pattern compiler uses: code point sets
//! as sorted ranges, and the Unicode properties a `\p{...}` escape names.
//!
//! ICU resolves every property it knows; this port resolves the General
//! Category values and groups, the binary properties Rust's `char` answers
//! (`Alphabetic`, `Lowercase`, `Uppercase`, `White_Space`) and those with a
//! fixed code point list, plus `Any`, `ASCII` and `Assigned`. Names follow
//! ICU's loose matching. Scripts, blocks and other properties are refused
//! like an unknown name. Unicode 17 on both sides, as ICU 78.

use std::sync::OnceLock;

use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

/// The largest code point.
pub(crate) const MAX_CODE_POINT: u32 = 0x10ffff;

/// A set of code points: sorted, disjoint, non-adjacent inclusive ranges.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub(crate) struct CodepointSet {
    ranges: Vec<(u32, u32)>,
}

impl CodepointSet {
    /// `UnicodeSet(first, last)`; empty when `first > last`.
    pub(crate) fn range(first: u32, last: u32) -> Self {
        let mut set = Self::default();
        set.add(first, last);
        set
    }

    pub(crate) fn single(code_point: u32) -> Self {
        Self::range(code_point, code_point)
    }

    /// Every Unicode scalar value (`Scalars`): all code points but the
    /// surrogates.
    pub(crate) fn scalars() -> Self {
        let mut set = Self::range(0, MAX_CODE_POINT);
        set.remove(0xd800, 0xdfff);
        set
    }

    pub(crate) fn ranges(&self) -> &[(u32, u32)] {
        &self.ranges
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The number of code points (`size`).
    pub(crate) fn size(&self) -> u64 {
        self.ranges
            .iter()
            .map(|&(first, last)| u64::from(last - first) + 1)
            .sum()
    }

    /// The smallest code point (`charAt(0)`).
    pub(crate) fn first(&self) -> Option<u32> {
        self.ranges.first().map(|&(first, _)| first)
    }

    pub(crate) fn contains(&self, code_point: u32) -> bool {
        let index = self.ranges.partition_point(|&(_, last)| last < code_point);
        self.ranges
            .get(index)
            .is_some_and(|&(first, _)| first <= code_point)
    }

    /// Whether any code point of `first..=last` is in the set
    /// (`containsSome`).
    pub(crate) fn contains_some(&self, first: u32, last: u32) -> bool {
        let index = self.ranges.partition_point(|&(_, end)| end < first);
        self.ranges
            .get(index)
            .is_some_and(|&(start, _)| start <= last)
    }

    /// Adds `first..=last`, merging overlapping and adjacent ranges.
    pub(crate) fn add(&mut self, first: u32, last: u32) {
        if first > last || first > MAX_CODE_POINT {
            return;
        }
        let last = last.min(MAX_CODE_POINT);
        // Ranges ending before `first - 1` stay; those starting after
        // `last + 1` stay; everything between merges.
        let start = self
            .ranges
            .partition_point(|&(_, end)| end.saturating_add(1) < first);
        let stop = self
            .ranges
            .partition_point(|&(begin, _)| begin <= last.saturating_add(1));
        let mut merged = (first, last);
        if start < stop {
            merged.0 = merged.0.min(self.ranges[start].0);
            merged.1 = merged.1.max(self.ranges[stop - 1].1);
        }
        self.ranges.splice(start..stop, [merged]);
    }

    pub(crate) fn add_all(&mut self, other: &Self) {
        for &(first, last) in &other.ranges {
            self.add(first, last);
        }
    }

    /// Removes `first..=last`.
    pub(crate) fn remove(&mut self, first: u32, last: u32) {
        if first > last {
            return;
        }
        let mut kept = Vec::with_capacity(self.ranges.len() + 1);
        for &(begin, end) in &self.ranges {
            if end < first || begin > last {
                kept.push((begin, end));
                continue;
            }
            if begin < first {
                kept.push((begin, first - 1));
            }
            if end > last {
                kept.push((last + 1, end));
            }
        }
        self.ranges = kept;
    }

    /// The complement within all code points, surrogates included
    /// (`complement`).
    pub(crate) fn complement(&mut self) {
        let mut result = Vec::with_capacity(self.ranges.len() + 1);
        let mut next = 0u32;
        for &(first, last) in &self.ranges {
            if first > next {
                result.push((next, first - 1));
            }
            next = last + 1;
        }
        if next <= MAX_CODE_POINT {
            result.push((next, MAX_CODE_POINT));
        }
        self.ranges = result;
    }

    fn from_predicate(predicate: impl Fn(u32) -> bool) -> Self {
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        for code_point in 0..=MAX_CODE_POINT {
            if !predicate(code_point) {
                continue;
            }
            match ranges.last_mut() {
                Some((_, last)) if *last + 1 == code_point => *last = code_point,
                _ => ranges.push((code_point, code_point)),
            }
        }
        Self { ranges }
    }
}

/// ICU's `UCharCategory` index: the bit a category takes in a
/// `U_GC_*_MASK`.
fn category_index(category: GeneralCategory) -> u32 {
    use GeneralCategory as G;
    match category {
        G::Unassigned => 0,
        G::UppercaseLetter => 1,
        G::LowercaseLetter => 2,
        G::TitlecaseLetter => 3,
        G::ModifierLetter => 4,
        G::OtherLetter => 5,
        G::NonspacingMark => 6,
        G::EnclosingMark => 7,
        G::SpacingMark => 8,
        G::DecimalNumber => 9,
        G::LetterNumber => 10,
        G::OtherNumber => 11,
        G::SpaceSeparator => 12,
        G::LineSeparator => 13,
        G::ParagraphSeparator => 14,
        G::Control => 15,
        G::Format => 16,
        G::PrivateUse => 17,
        G::Surrogate => 18,
        G::DashPunctuation => 19,
        G::OpenPunctuation => 20,
        G::ClosePunctuation => 21,
        G::ConnectorPunctuation => 22,
        G::OtherPunctuation => 23,
        G::MathSymbol => 24,
        G::CurrencySymbol => 25,
        G::ModifierSymbol => 26,
        G::OtherSymbol => 27,
        G::InitialPunctuation => 28,
        G::FinalPunctuation => 29,
    }
}

/// The ICU category index of a code point (`u_charType`); surrogates are
/// `Cs`.
pub(crate) fn char_type(code_point: u32) -> u32 {
    match char::from_u32(code_point) {
        Some(c) => category_index(c.general_category()),
        None => category_index(GeneralCategory::Surrogate),
    }
}

/// The code points of each category, built once.
fn categories() -> &'static [CodepointSet; 30] {
    static CATEGORIES: OnceLock<[CodepointSet; 30]> = OnceLock::new();
    CATEGORIES.get_or_init(|| {
        let mut sets: [CodepointSet; 30] = Default::default();
        let mut current: Option<(u32, u32, u32)> = None;
        for code_point in 0..=MAX_CODE_POINT {
            let category = char_type(code_point);
            match &mut current {
                Some((_, last, kind)) if *kind == category && *last + 1 == code_point => {
                    *last = code_point;
                }
                _ => {
                    if let Some((first, last, kind)) = current {
                        sets[kind as usize].ranges.push((first, last));
                    }
                    current = Some((code_point, code_point, category));
                }
            }
        }
        if let Some((first, last, kind)) = current {
            sets[kind as usize].ranges.push((first, last));
        }
        sets
    })
}

fn category_mask(mask: u32) -> CodepointSet {
    let mut set = CodepointSet::default();
    for (index, category) in categories().iter().enumerate() {
        if mask & (1 << index) != 0 {
            set.add_all(category);
        }
    }
    set
}

/// ICU's General Category mask aliases (`u_getPropertyValueName` of
/// `UCHAR_GENERAL_CATEGORY_MASK`), with their masks.
const CATEGORY_ALIASES: &[(&str, u32)] = &[
    ("Cn", 0x0000_0001),
    ("Unassigned", 0x0000_0001),
    ("Lu", 0x0000_0002),
    ("Uppercase_Letter", 0x0000_0002),
    ("Ll", 0x0000_0004),
    ("Lowercase_Letter", 0x0000_0004),
    ("Lt", 0x0000_0008),
    ("Titlecase_Letter", 0x0000_0008),
    ("Lm", 0x0000_0010),
    ("Modifier_Letter", 0x0000_0010),
    ("Lo", 0x0000_0020),
    ("Other_Letter", 0x0000_0020),
    ("Mn", 0x0000_0040),
    ("Nonspacing_Mark", 0x0000_0040),
    ("Me", 0x0000_0080),
    ("Enclosing_Mark", 0x0000_0080),
    ("Mc", 0x0000_0100),
    ("Spacing_Mark", 0x0000_0100),
    ("Nd", 0x0000_0200),
    ("Decimal_Number", 0x0000_0200),
    ("digit", 0x0000_0200),
    ("Nl", 0x0000_0400),
    ("Letter_Number", 0x0000_0400),
    ("No", 0x0000_0800),
    ("Other_Number", 0x0000_0800),
    ("Zs", 0x0000_1000),
    ("Space_Separator", 0x0000_1000),
    ("Zl", 0x0000_2000),
    ("Line_Separator", 0x0000_2000),
    ("Zp", 0x0000_4000),
    ("Paragraph_Separator", 0x0000_4000),
    ("Cc", 0x0000_8000),
    ("Control", 0x0000_8000),
    ("cntrl", 0x0000_8000),
    ("Cf", 0x0001_0000),
    ("Format", 0x0001_0000),
    ("Co", 0x0002_0000),
    ("Private_Use", 0x0002_0000),
    ("Cs", 0x0004_0000),
    ("Surrogate", 0x0004_0000),
    ("Pd", 0x0008_0000),
    ("Dash_Punctuation", 0x0008_0000),
    ("Ps", 0x0010_0000),
    ("Open_Punctuation", 0x0010_0000),
    ("Pe", 0x0020_0000),
    ("Close_Punctuation", 0x0020_0000),
    ("Pc", 0x0040_0000),
    ("Connector_Punctuation", 0x0040_0000),
    ("Po", 0x0080_0000),
    ("Other_Punctuation", 0x0080_0000),
    ("Sm", 0x0100_0000),
    ("Math_Symbol", 0x0100_0000),
    ("Sc", 0x0200_0000),
    ("Currency_Symbol", 0x0200_0000),
    ("Sk", 0x0400_0000),
    ("Modifier_Symbol", 0x0400_0000),
    ("So", 0x0800_0000),
    ("Other_Symbol", 0x0800_0000),
    ("Pi", 0x1000_0000),
    ("Initial_Punctuation", 0x1000_0000),
    ("Pf", 0x2000_0000),
    ("Final_Punctuation", 0x2000_0000),
    ("L", 0x0000_003e),
    ("Letter", 0x0000_003e),
    ("LC", 0x0000_000e),
    ("Cased_Letter", 0x0000_000e),
    ("M", 0x0000_01c0),
    ("Mark", 0x0000_01c0),
    ("Combining_Mark", 0x0000_01c0),
    ("N", 0x0000_0e00),
    ("Number", 0x0000_0e00),
    ("Z", 0x0000_7000),
    ("Separator", 0x0000_7000),
    ("C", 0x0007_8001),
    ("Other", 0x0007_8001),
    ("P", 0x30f8_0000),
    ("Punctuation", 0x30f8_0000),
    ("punct", 0x30f8_0000),
    ("S", 0x0f00_0000),
    ("Symbol", 0x0f00_0000),
];

/// The General Category property names (`gc`, `gcm`).
const CATEGORY_PROPERTIES: &[&str] = &["gc", "General_Category", "gcm", "General_Category_Mask"];

/// The binary properties this port resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Binary {
    Alphabetic,
    Lowercase,
    Uppercase,
    WhiteSpace,
    AsciiHexDigit,
    HexDigit,
    Noncharacter,
    JoinControl,
    RegionalIndicator,
    PatternWhiteSpace,
}

/// ICU's aliases of those properties (`u_getPropertyName`).
const BINARY_ALIASES: &[(&str, Binary)] = &[
    ("Alpha", Binary::Alphabetic),
    ("Alphabetic", Binary::Alphabetic),
    ("Lower", Binary::Lowercase),
    ("Lowercase", Binary::Lowercase),
    ("Upper", Binary::Uppercase),
    ("Uppercase", Binary::Uppercase),
    ("WSpace", Binary::WhiteSpace),
    ("White_Space", Binary::WhiteSpace),
    ("space", Binary::WhiteSpace),
    ("AHex", Binary::AsciiHexDigit),
    ("ASCII_Hex_Digit", Binary::AsciiHexDigit),
    ("Hex", Binary::HexDigit),
    ("Hex_Digit", Binary::HexDigit),
    ("NChar", Binary::Noncharacter),
    ("Noncharacter_Code_Point", Binary::Noncharacter),
    ("Join_C", Binary::JoinControl),
    ("Join_Control", Binary::JoinControl),
    ("RI", Binary::RegionalIndicator),
    ("Regional_Indicator", Binary::RegionalIndicator),
    ("Pat_WS", Binary::PatternWhiteSpace),
    ("Pattern_White_Space", Binary::PatternWhiteSpace),
];

/// ICU's binary property values: false, then true.
const BINARY_VALUES: [&[&str]; 2] = [&["N", "No", "F", "False"], &["Y", "Yes", "T", "True"]];

impl Binary {
    fn contains(self, code_point: u32) -> bool {
        let is_char = |f: fn(char) -> bool| char::from_u32(code_point).is_some_and(f);
        match self {
            Self::Alphabetic => is_char(char::is_alphabetic),
            Self::Lowercase => is_char(char::is_lowercase),
            Self::Uppercase => is_char(char::is_uppercase),
            Self::WhiteSpace => is_char(char::is_whitespace),
            Self::AsciiHexDigit => is_char(|c| c.is_ascii_hexdigit()),
            Self::HexDigit => {
                is_char(|c| c.is_ascii_hexdigit())
                    || matches!(code_point, 0xff10..=0xff19 | 0xff21..=0xff26 | 0xff41..=0xff46)
            }
            Self::Noncharacter => {
                (0xfdd0..=0xfdef).contains(&code_point) || code_point & 0xfffe == 0xfffe
            }
            Self::JoinControl => (0x200c..=0x200d).contains(&code_point),
            Self::RegionalIndicator => (0x1f1e6..=0x1f1ff).contains(&code_point),
            Self::PatternWhiteSpace => matches!(
                code_point,
                0x09..=0x0d | 0x20 | 0x85 | 0x200e | 0x200f | 0x2028 | 0x2029
            ),
        }
    }

    fn set(self, value: bool) -> CodepointSet {
        CodepointSet::from_predicate(|code_point| self.contains(code_point) == value)
    }
}

/// ICU's loose name matching (`uprv_comparePropertyNames`): case, `-`, `_`
/// and ASCII white space do not count.
fn loose(name: &str) -> String {
    name.bytes()
        .filter(|&b| !matches!(b, b'-' | b'_' | b' ' | b'\t'..=b'\r'))
        .map(|b| char::from(b.to_ascii_lowercase()))
        .collect()
}

fn find<T: Copy>(aliases: &[(&str, T)], name: &str) -> Option<T> {
    let name = loose(name);
    aliases
        .iter()
        .find(|(alias, _)| loose(alias) == name)
        .map(|&(_, value)| value)
}

/// The set `[\p{name}]` denotes (`UnicodeSet::applyPropertyAlias`); `None`
/// for a property this port does not resolve. A name without a value is a
/// General Category value, else a binary property, else `Any`, `ASCII` or
/// `Assigned`; ICU would try a Script value before the binary properties,
/// and no Script alias spells a supported property.
pub(crate) fn property(name: &str) -> Option<CodepointSet> {
    // ICU reads `name=` like `name`.
    let (name, value) = match name.split_once('=') {
        Some((property, "")) => (property, None),
        Some((property, value)) => (property, Some(value)),
        None => (name, None),
    };
    if let Some(value) = value {
        let property = loose(name);
        if CATEGORY_PROPERTIES
            .iter()
            .any(|alias| loose(alias) == property)
        {
            return find(CATEGORY_ALIASES, value).map(category_mask);
        }
        let binary = find(BINARY_ALIASES, name)?;
        let value = loose(value);
        let truth = BINARY_VALUES
            .iter()
            .position(|aliases| aliases.iter().any(|alias| loose(alias) == value))?;
        return Some(binary.set(truth == 1));
    }
    if let Some(mask) = find(CATEGORY_ALIASES, name) {
        return Some(category_mask(mask));
    }
    if let Some(binary) = find(BINARY_ALIASES, name) {
        return Some(binary.set(true));
    }
    match loose(name).as_str() {
        "any" => Some(CodepointSet::range(0, MAX_CODE_POINT)),
        "ascii" => Some(CodepointSet::range(0, 0x7f)),
        "assigned" => {
            let mut set = category_mask(1);
            set.complement();
            Some(set)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fnv(u64);

    impl Fnv {
        fn byte(&mut self, b: u8) {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }

    /// Digests of ICU 78.2 (Unicode 17): `u_charType` of every code point,
    /// and the ten supported binary properties (`u_hasBinaryProperty`, one
    /// bit each in declaration order, two bytes per code point).
    #[test]
    fn properties_match_icu() {
        let (mut categories, mut binary) = (Fnv(0xcbf2_9ce4_8422_2325), Fnv(0xcbf2_9ce4_8422_2325));
        let properties = [
            Binary::Alphabetic,
            Binary::Lowercase,
            Binary::Uppercase,
            Binary::WhiteSpace,
            Binary::AsciiHexDigit,
            Binary::HexDigit,
            Binary::Noncharacter,
            Binary::JoinControl,
            Binary::RegionalIndicator,
            Binary::PatternWhiteSpace,
        ];
        for code_point in 0..=MAX_CODE_POINT {
            categories.byte(char_type(code_point) as u8);
            let bits = properties.iter().enumerate().fold(0u16, |bits, (i, p)| {
                bits | u16::from(p.contains(code_point)) << i
            });
            binary.byte(bits.to_le_bytes()[0]);
            binary.byte(bits.to_le_bytes()[1]);
        }
        assert_eq!(categories.0, 0xbd87_d081_e134_0083, "general categories");
        assert_eq!(binary.0, 0x16e9_e932_7a10_4d9c, "binary properties");
    }

    #[test]
    fn sets_merge_split_and_complement() {
        let mut set = CodepointSet::range(10, 20);
        set.add(21, 30);
        set.add(5, 8);
        assert_eq!(set.ranges(), [(5, 8), (10, 30)]);
        set.add(9, 9);
        assert_eq!(set.ranges(), [(5, 30)]);
        set.remove(10, 12);
        assert_eq!(set.ranges(), [(5, 9), (13, 30)]);
        assert!(set.contains(13) && !set.contains(11) && set.contains_some(10, 13));
        assert_eq!(set.size(), 23);
        set.complement();
        assert_eq!(set.ranges(), [(0, 4), (10, 12), (31, MAX_CODE_POINT)]);
    }

    #[test]
    fn property_names_match_loosely() {
        let letters = property("L").expect("L");
        assert_eq!(property(" letter ").as_ref(), Some(&letters));
        assert_eq!(property("General-Category = L").as_ref(), Some(&letters));
        assert_eq!(property("L=").as_ref(), Some(&letters));
        assert!(property("L= ").is_none() && property("gc=").is_none());
        assert_eq!(property("gc=Uppercase_Letter"), property("Lu"));
        assert_eq!(property("WSpace"), property("White_Space=Yes"));
        assert!(property("Script=Greek").is_none() && property("Latin").is_none());
        let assigned = property("Assigned").expect("Assigned");
        assert!(assigned.contains(u32::from('a')) && !assigned.contains(0x0378));
    }
}
