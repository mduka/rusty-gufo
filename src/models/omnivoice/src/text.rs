//! Text preprocessing of OmniVoice's reference implementation (`text.cpp`,
//! ported from k2-fsa/OmniVoice, Apache-2.0: `omnivoice/models/omnivoice.py`
//! `_resolve_language`, `_resolve_instruct`, `_estimate_target_tokens`,
//! `_combine_text`, `omnivoice/utils/voice_design.py`,
//! `omnivoice/utils/duration.py` and `omnivoice/utils/text.py`): language and
//! voice-design resolution, the rule-based duration estimator, prompt text
//! cleanup and punctuation chunking. Every function reproduces the Python
//! helper named in its comment.

use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

use crate::languages::LANGUAGES;
use crate::unicode::{
    contains_cjk_ideograph, encode, is_cjk_ideograph, is_space, strip_space, to_lower,
};

// ---- Voice design -------------------------------------------------------

struct InstructPair {
    english: &'static str,
    chinese: &'static str,
}

const fn pair(english: &'static str, chinese: &'static str) -> InstructPair {
    InstructPair { english, chinese }
}

// Each group is one mutually exclusive category.
const GENDER: [InstructPair; 2] = [pair("male", "男"), pair("female", "女")];
const AGE: [InstructPair; 5] = [
    pair("child", "儿童"),
    pair("teenager", "少年"),
    pair("young adult", "青年"),
    pair("middle-aged", "中年"),
    pair("elderly", "老年"),
];
const PITCH: [InstructPair; 5] = [
    pair("very low pitch", "极低音调"),
    pair("low pitch", "低音调"),
    pair("moderate pitch", "中音调"),
    pair("high pitch", "高音调"),
    pair("very high pitch", "极高音调"),
];
const STYLE: [InstructPair; 1] = [pair("whisper", "耳语")];
const ACCENTS: [&str; 10] = [
    "american accent",
    "british accent",
    "australian accent",
    "chinese accent",
    "canadian accent",
    "indian accent",
    "korean accent",
    "portuguese accent",
    "russian accent",
    "japanese accent",
];
const DIALECTS: [&str; 12] = [
    "河南话",
    "陕西话",
    "四川话",
    "贵州话",
    "云南话",
    "桂林话",
    "济南话",
    "石家庄话",
    "甘肃话",
    "宁夏话",
    "青岛话",
    "东北话",
];

fn find_pair<'a>(pairs: &'a [InstructPair], item: &str) -> Option<&'a InstructPair> {
    pairs
        .iter()
        .find(|pair| pair.english == item || pair.chinese == item)
}

fn find_any_pair(item: &str) -> Option<&'static InstructPair> {
    find_pair(&GENDER, item)
        .or_else(|| find_pair(&AGE, item))
        .or_else(|| find_pair(&PITCH, item))
        .or_else(|| find_pair(&STYLE, item))
}

/// Index of the item's mutually exclusive category.
fn category_of(item: &str) -> Option<usize> {
    if find_pair(&GENDER, item).is_some() {
        Some(0)
    } else if find_pair(&AGE, item).is_some() {
        Some(1)
    } else if find_pair(&PITCH, item).is_some() {
        Some(2)
    } else if find_pair(&STYLE, item).is_some() {
        Some(3)
    } else if ACCENTS.contains(&item) {
        Some(4)
    } else if DIALECTS.contains(&item) {
        Some(5)
    } else {
        None
    }
}

fn valid_items() -> String {
    let items: Vec<&str> = GENDER
        .iter()
        .chain(&AGE)
        .chain(&PITCH)
        .chain(&STYLE)
        .map(|pair| pair.english)
        .chain(ACCENTS)
        .collect();
    items.join(", ")
}

/// Splits on half- and full-width commas with surrounding whitespace
/// (`re.split(r"\s*[,，]\s*", ...)`), dropping empty items.
fn split_instruct(text: &str) -> Vec<String> {
    let points: Vec<(usize, char)> = text.char_indices().collect();
    let end_of = |index: usize| {
        let (at, c) = points[index];
        at + c.len_utf8()
    };
    let mut items = Vec::new();
    let mut start = 0;
    let mut flush = |start: usize, end: usize| {
        // Trailing whitespace before the comma belongs to the separator.
        let mut last = end;
        while last > start && is_space(points[last - 1].1 as u32) {
            last -= 1;
        }
        if last > start {
            items.push(text[points[start].0..end_of(last - 1)].to_owned());
        }
    };
    let mut index = 0;
    while index < points.len() {
        let value = points[index].1;
        if value != ',' && value != '\u{FF0C}' {
            index += 1;
            continue;
        }
        flush(start, index);
        start = index + 1;
        while start < points.len() && is_space(points[start].1 as u32) {
            start += 1;
        }
        index = start;
    }
    flush(start, points.len());
    items
}

// ---- Duration -----------------------------------------------------------

#[derive(Clone, Copy)]
enum Script {
    Cjk,
    Hangul,
    Kana,
    Ethiopic,
    Yi,
    Indic,
    ThaiLao,
    KhmerMyanmar,
    Arabic,
    Hebrew,
    Latin,
    Cyrillic,
    Greek,
    Armenian,
    Georgian,
    Default,
}

fn script_weight(script: Script) -> f64 {
    match script {
        Script::Cjk | Script::Ethiopic | Script::Yi => 3.0,
        Script::Hangul => 2.5,
        Script::Kana => 2.2,
        Script::Indic | Script::KhmerMyanmar => 1.8,
        Script::ThaiLao | Script::Arabic | Script::Hebrew => 1.5,
        Script::Latin
        | Script::Cyrillic
        | Script::Greek
        | Script::Armenian
        | Script::Georgian
        | Script::Default => 1.0,
    }
}

const PUNCTUATION_WEIGHT: f64 = 0.5;
const SPACE_WEIGHT: f64 = 0.2;
const DIGIT_WEIGHT: f64 = 3.5;
const MARK_WEIGHT: f64 = 0.0;

// Upper bounds of the script blocks, as RuleDurationEstimator.ranges.
const SCRIPT_RANGES: &[(u32, Script)] = {
    use Script::*;
    &[
        (0x02AF, Latin),
        (0x03FF, Greek),
        (0x052F, Cyrillic),
        (0x058F, Armenian),
        (0x05FF, Hebrew),
        (0x077F, Arabic),
        (0x089F, Arabic),
        (0x08FF, Arabic),
        (0x097F, Indic),
        (0x09FF, Indic),
        (0x0A7F, Indic),
        (0x0AFF, Indic),
        (0x0B7F, Indic),
        (0x0BFF, Indic),
        (0x0C7F, Indic),
        (0x0CFF, Indic),
        (0x0D7F, Indic),
        (0x0DFF, Indic),
        (0x0EFF, ThaiLao),
        (0x0FFF, Indic),
        (0x109F, KhmerMyanmar),
        (0x10FF, Georgian),
        (0x11FF, Hangul),
        (0x137F, Ethiopic),
        (0x139F, Ethiopic),
        (0x13FF, Default),
        (0x167F, Default),
        (0x169F, Default),
        (0x16FF, Default),
        (0x171F, Default),
        (0x173F, Default),
        (0x175F, Default),
        (0x177F, Default),
        (0x17FF, KhmerMyanmar),
        (0x18AF, Default),
        (0x18FF, Default),
        (0x194F, Indic),
        (0x19DF, Indic),
        (0x19FF, KhmerMyanmar),
        (0x1A1F, Indic),
        (0x1AAF, Indic),
        (0x1B7F, Indic),
        (0x1BBF, Indic),
        (0x1BFF, Indic),
        (0x1C4F, Indic),
        (0x1C7F, Indic),
        (0x1C8F, Cyrillic),
        (0x1CBF, Georgian),
        (0x1CCF, Indic),
        (0x1CFF, Indic),
        (0x1D7F, Latin),
        (0x1DBF, Latin),
        (0x1DFF, Default),
        (0x1EFF, Latin),
        (0x309F, Kana),
        (0x30FF, Kana),
        (0x312F, Cjk),
        (0x318F, Hangul),
        (0x9FFF, Cjk),
        (0xA4CF, Yi),
        (0xA4FF, Default),
        (0xA63F, Default),
        (0xA69F, Cyrillic),
        (0xA6FF, Default),
        (0xA7FF, Latin),
        (0xA82F, Indic),
        (0xA87F, Default),
        (0xA8DF, Indic),
        (0xA8FF, Indic),
        (0xA92F, Indic),
        (0xA95F, Indic),
        (0xA97F, Hangul),
        (0xA9DF, Indic),
        (0xA9FF, KhmerMyanmar),
        (0xAA5F, Indic),
        (0xAA7F, KhmerMyanmar),
        (0xAADF, Indic),
        (0xAAFF, Indic),
        (0xAB2F, Ethiopic),
        (0xAB6F, Latin),
        (0xABBF, Default),
        (0xABFF, Indic),
        (0xD7AF, Hangul),
        (0xFAFF, Cjk),
        (0xFDFF, Arabic),
        (0xFE6F, Default),
        (0xFEFF, Arabic),
        (0xFFEF, Latin),
    ]
};

fn character_weight(c: char) -> f64 {
    let code = c as u32;
    if c.is_ascii_alphabetic() {
        return script_weight(Script::Latin);
    }
    if c == ' ' {
        return SPACE_WEIGHT;
    }
    if code == 0x0640 {
        // Arabic tatweel
        return MARK_WEIGHT;
    }
    use GeneralCategory as G;
    match c.general_category() {
        G::NonspacingMark | G::EnclosingMark | G::SpacingMark => return MARK_WEIGHT,
        G::DashPunctuation
        | G::OpenPunctuation
        | G::ClosePunctuation
        | G::ConnectorPunctuation
        | G::OtherPunctuation
        | G::InitialPunctuation
        | G::FinalPunctuation
        | G::MathSymbol
        | G::CurrencySymbol
        | G::ModifierSymbol
        | G::OtherSymbol => return PUNCTUATION_WEIGHT,
        G::SpaceSeparator | G::LineSeparator | G::ParagraphSeparator => return SPACE_WEIGHT,
        G::DecimalNumber | G::LetterNumber | G::OtherNumber => return DIGIT_WEIGHT,
        _ => {}
    }
    let range = SCRIPT_RANGES.partition_point(|&(last, _)| last < code);
    if let Some(&(_, script)) = SCRIPT_RANGES.get(range) {
        return script_weight(script);
    }
    if code > 0x20000 {
        script_weight(Script::Cjk)
    } else {
        script_weight(Script::Default)
    }
}

/// RuleDurationEstimator.estimate_duration with its default low threshold
/// (50 frames) and boost strength (3).
fn estimate_duration(target: &str, reference: &str, reference_duration: f64) -> f64 {
    if reference_duration <= 0.0 || reference.is_empty() {
        return 0.0;
    }
    let reference_weight = text_weight(reference);
    if reference_weight == 0.0 {
        return 0.0;
    }
    let speed = reference_weight / reference_duration;
    let estimate = text_weight(target) / speed;
    const LOW_THRESHOLD: f64 = 50.0;
    const BOOST_STRENGTH: f64 = 3.0;
    if estimate < LOW_THRESHOLD {
        return LOW_THRESHOLD * (estimate / LOW_THRESHOLD).powf(1.0 / BOOST_STRENGTH);
    }
    estimate
}

// ---- Chunking -----------------------------------------------------------

fn is_split_punctuation(value: u32) -> bool {
    char::from_u32(value).is_some_and(|c| ".,;:!?。，；：！？".contains(c))
}

fn is_closing_mark(value: u32) -> bool {
    char::from_u32(value).is_some_and(|c| "\"'“”‘’）]》>」】".contains(c))
}

fn is_end_punctuation(value: u32) -> bool {
    char::from_u32(value).is_some_and(|c| ";:,.!?…)]}\"'“”‘’；：，。！？、）】".contains(c))
}

const ABBREVIATIONS: [&str; 53] = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.", "Pres.", "Gov.",
    "Capt.", "Gen.", "Sen.", "Rep.", "Col.", "Maj.", "Lt.", "Cmdr.", "Sgt.", "Cpl.", "Co.",
    "Corp.", "Inc.", "Ltd.", "Est.", "Dept.", "St.", "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.", "No.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "i.e.",
    "e.g.", "vs.", "Vs.", "Etc.", "approx.", "fig.", "def.",
];

/// `"".join(sentence).strip().split()[-1]`.
fn last_word(points: &[u32]) -> String {
    let mut end = points.len();
    while end > 0 && is_space(points[end - 1]) {
        end -= 1;
    }
    let mut begin = end;
    while begin > 0 && !is_space(points[begin - 1]) {
        begin -= 1;
    }
    encode(&points[begin..end])
}

fn code_points(text: &str) -> Vec<u32> {
    text.chars().map(|c| c as u32).collect()
}

/// `_resolve_language`: a language ID (`"en"`) is kept, a language name
/// (`"English"`, any case) maps to its ID; "none", unknown names and empty
/// input select the language-agnostic prompt (`None`) (`ResolveLanguage`).
pub fn resolve_language(language: &str) -> Option<String> {
    if language.is_empty() {
        return None;
    }
    let lower = to_lower(language);
    if lower == "none" {
        return None;
    }
    if LANGUAGES.iter().any(|&(_, id)| id == language) {
        return Some(language.to_owned());
    }
    let found = LANGUAGES.partition_point(|&(name, _)| name < lower.as_str());
    LANGUAGES
        .get(found)
        .filter(|&&(name, _)| name == lower)
        .map(|&(_, id)| id.to_owned())
}

/// `_resolve_instruct`: validates comma-separated voice-design items
/// (gender, age, pitch, whisper, an English accent or a Chinese dialect),
/// unifies them to Chinese when `use_chinese` or a dialect asks for it and
/// to English otherwise. `Ok(None)` means no instruction; an error means the
/// instruction is invalid (`ResolveInstruct`).
pub fn resolve_instruct(instruct: &str, use_chinese: bool) -> Result<Option<String>, String> {
    let stripped = strip_space(instruct);
    if stripped.is_empty() {
        return Ok(None);
    }
    let mut items = Vec::new();
    let mut unknown = String::new();
    for raw in split_instruct(&stripped) {
        let item = to_lower(&strip_space(&raw));
        if category_of(&item).is_none() {
            unknown += if unknown.is_empty() { "'" } else { ", '" };
            unknown += &raw;
            unknown += "'";
            continue;
        }
        items.push(item);
    }
    if !unknown.is_empty() {
        return Err(format!(
            "unsupported voice design items {unknown}; use comma-separated items from: {}, or a \
             Chinese dialect such as 四川话",
            valid_items()
        ));
    }
    let has_dialect = items.iter().any(|item| category_of(item) == Some(5));
    let has_accent = items.iter().any(|item| category_of(item) == Some(4));
    if has_dialect && has_accent {
        return Err("a voice design can't mix a Chinese dialect with an English accent".into());
    }
    let use_chinese = if has_dialect {
        true
    } else if has_accent {
        false
    } else {
        use_chinese
    };
    for item in &mut items {
        if let Some(pair) = find_any_pair(item) {
            *item = if use_chinese {
                pair.chinese
            } else {
                pair.english
            }
            .to_owned();
        }
    }
    let mut counts = [0usize; 6];
    for item in &items {
        if let Some(category) = category_of(item) {
            counts[category] += 1;
            if counts[category] > 1 {
                return Err(
                    "a voice design allows one item per category (gender, age, pitch, \
                            style, accent, dialect)"
                        .into(),
                );
            }
        }
    }
    let chinese = items.iter().any(|item| contains_cjk_ideograph(item));
    Ok(Some(items.join(if chinese { "，" } else { ", " })))
}

/// `RuleDurationEstimator.calculate_total_weight`: phonetic weight of the
/// text's characters (`TextWeight`).
pub fn text_weight(text: &str) -> f64 {
    text.chars()
        .map(character_weight)
        .fold(0.0, |weight, character| weight + character)
}

/// `OmniVoice._estimate_target_tokens`: audio frames for `text`, scaled from
/// the reference's frames per weight (or "Nice to meet you." at 25 frames
/// without a reference) and divided by `speed` (`EstimateTargetFrames`).
pub fn estimate_target_frames(
    text: &str,
    reference_text: Option<&str>,
    reference_frames: usize,
    speed: f64,
) -> usize {
    let (reference, frames) = match reference_text {
        Some(reference) if !reference.is_empty() && reference_frames > 0 => {
            (reference, reference_frames as f64)
        }
        _ => ("Nice to meet you.", 25.0),
    };
    let mut estimate = estimate_duration(text, reference, frames);
    if speed > 0.0 && speed != 1.0 {
        estimate /= speed;
    }
    (estimate as usize).max(1)
}

/// `_combine_text`: the reference transcript and the text joined by a
/// space, without line breaks, with ASCII parentheses, single spaces and no
/// spaces next to Chinese characters (`CombineText`).
pub fn combine_text(text: &str, reference_text: Option<&str>) -> String {
    let mut joined = strip_space(text);
    if let Some(reference) = reference_text.filter(|reference| !reference.is_empty()) {
        joined = format!("{} {joined}", strip_space(reference));
    }
    // Drop line breaks, use ASCII parentheses, collapse space and tab runs.
    let mut points: Vec<u32> = Vec::with_capacity(joined.len());
    for c in joined.chars() {
        let mut value = c as u32;
        if c == '\r' || c == '\n' {
            continue;
        }
        if value == 0xFF08 {
            value = u32::from(b'(');
        } else if value == 0xFF09 {
            value = u32::from(b')');
        }
        let blank = c == ' ' || c == '\t';
        if blank && points.last() == Some(&u32::from(b' ')) {
            continue;
        }
        points.push(if blank { u32::from(b' ') } else { value });
    }
    // Remove whitespace runs that touch a Chinese character.
    let mut cleaned = Vec::with_capacity(points.len());
    let mut index = 0;
    while index < points.len() {
        if !is_space(points[index]) {
            cleaned.push(points[index]);
            index += 1;
            continue;
        }
        let mut end = index;
        while end < points.len() && is_space(points[end]) {
            end += 1;
        }
        let after_chinese = index > 0 && is_cjk_ideograph(points[index - 1]);
        let before_chinese = end < points.len() && is_cjk_ideograph(points[end]);
        if !after_chinese && !before_chinese {
            cleaned.extend_from_slice(&points[index..end]);
        }
        index = end;
    }
    encode(&cleaned)
}

/// `add_punctuation`: strips the text and ends it with "." (or "。" for
/// Chinese) unless it already ends with punctuation (`AddPunctuation`).
pub fn add_punctuation(text: &str) -> String {
    let mut stripped = strip_space(text);
    let Some(last) = stripped.chars().last() else {
        return stripped;
    };
    if !is_end_punctuation(last as u32) {
        stripped += if contains_cjk_ideograph(&stripped) {
            "。"
        } else {
            "."
        };
    }
    stripped
}

/// `chunk_text_punctuation`: splits at sentence and clause punctuation
/// (not after common abbreviations), merges pieces up to `chunk_length`
/// code points and folds pieces shorter than `min_chunk_length` into a
/// neighbour (`ChunkTextPunctuation`; the reference's default minimum is 3).
pub fn chunk_text_punctuation(
    text: &str,
    chunk_length: usize,
    min_chunk_length: usize,
) -> Vec<String> {
    type Piece = Vec<u32>;
    // 1. Split after punctuation; punctuation that opens a sentence joins the
    // previous one.
    let mut sentences: Vec<Piece> = Vec::new();
    let mut current: Piece = Vec::new();
    for value in code_points(text) {
        if current.is_empty()
            && (is_split_punctuation(value) || is_closing_mark(value))
            && let Some(last) = sentences.last_mut()
        {
            last.push(value);
            continue;
        }
        current.push(value);
        if !is_split_punctuation(value) {
            continue;
        }
        if value == u32::from(b'.') && ABBREVIATIONS.contains(&last_word(&current).as_str()) {
            continue;
        }
        sentences.push(std::mem::take(&mut current));
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    // 2. Merge sentences up to the chunk length.
    let mut merged: Vec<Piece> = Vec::new();
    let mut chunk: Piece = Vec::new();
    for sentence in sentences {
        if chunk.len() + sentence.len() <= chunk_length {
            chunk.extend(sentence);
        } else {
            if !chunk.is_empty() {
                merged.push(std::mem::take(&mut chunk));
            }
            chunk = sentence;
        }
    }
    if !chunk.is_empty() {
        merged.push(chunk);
    }
    // 3. Fold undersized chunks into the previous one; a short first chunk
    // takes the second.
    let first_short = merged
        .first()
        .is_some_and(|piece| piece.len() < min_chunk_length);
    let mut final_chunks: Vec<Piece> = Vec::new();
    for (index, piece) in merged.into_iter().enumerate() {
        let fold = (index == 1 && first_short) || piece.len() < min_chunk_length;
        match final_chunks.last_mut() {
            Some(last) if fold => last.extend(piece),
            _ => final_chunks.push(piece),
        }
    }
    final_chunks
        .iter()
        .map(|piece| strip_space(&encode(piece)))
        .filter(|stripped| !stripped.is_empty())
        .collect()
}
