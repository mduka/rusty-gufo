//! Language prompts (`language.hpp`, `language.cpp`): a request's `language`
//! to the prompt index, English names of locales and the locales of the
//! language tag tokens the model emits.

use crate::config::Config;

/// English names of the languages NVIDIA lists for the model (transcription
/// ready, broad coverage and adaptation ready).
const NAMES: [(&str, &str); 40] = [
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("de", "German"),
    ("tr", "Turkish"),
    ("ru", "Russian"),
    ("ar", "Arabic"),
    ("hi", "Hindi"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("vi", "Vietnamese"),
    ("uk", "Ukrainian"),
    ("pl", "Polish"),
    ("sv", "Swedish"),
    ("cs", "Czech"),
    ("nb", "Norwegian"),
    ("da", "Danish"),
    ("bg", "Bulgarian"),
    ("fi", "Finnish"),
    ("hr", "Croatian"),
    ("sk", "Slovak"),
    ("zh", "Chinese"),
    ("hu", "Hungarian"),
    ("ro", "Romanian"),
    ("et", "Estonian"),
    ("el", "Greek"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mt", "Maltese"),
    ("sl", "Slovenian"),
    ("he", "Hebrew"),
    ("th", "Thai"),
    ("nn", "Nynorsk"),
    ("no", "Norwegian"),
    ("id", "Indonesian"),
    ("ms", "Malay"),
    ("fa", "Persian"),
];

/// ASCII lowercase with "_" as "-" (`Normalize`; bytes beyond ASCII stay).
fn normalize(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c == '_' {
                '-'
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect()
}

/// The part before the first "-" (`Base`).
fn base(locale: &str) -> &str {
    locale.split_once('-').map_or(locale, |(code, _)| code)
}

fn lookup(config: &Config, normalized: &str) -> Option<usize> {
    for (key, prompt) in &config.prompt_dictionary {
        if normalize(key) == normalized {
            return Some(*prompt);
        }
    }
    if normalized.contains('-') {
        return None;
    }
    // A bare code: its own country's locale ("zh" -> "zh-ZH"), else the one
    // prompt all its locales share.
    let own = format!("{normalized}-{normalized}");
    for (key, prompt) in &config.prompt_dictionary {
        if normalize(key) == own {
            return Some(*prompt);
        }
    }
    let mut shared = None;
    for (key, prompt) in &config.prompt_dictionary {
        let lower = normalize(key);
        if base(&lower) != normalized {
            continue;
        }
        if shared.is_some_and(|shared| shared != *prompt) {
            return None;
        }
        shared = Some(*prompt);
    }
    shared
}

/// The language prompt for a request's `language` (`ResolveLanguage`):
/// absent, empty or "auto" detects the language; otherwise a key of the
/// prompt dictionary ("de", "de-DE"; case-insensitive, "_" for "-"), a bare
/// code with its country's locale ("zh" for "zh-ZH") or whose locales share
/// one prompt ("ja" for "ja-JP"), or an English language name ("German").
pub fn resolve_language(config: &Config, language: Option<&str>) -> Result<usize, String> {
    let Some(language) = language else {
        return Ok(config.default_prompt);
    };
    let normalized = normalize(language);
    if normalized.is_empty() {
        return Ok(config.default_prompt);
    }
    if let Some(prompt) = lookup(config, &normalized) {
        return Ok(prompt);
    }
    for (code, name) in NAMES {
        if normalize(name) == normalized
            && let Some(prompt) = lookup(config, code)
        {
            return Ok(prompt);
        }
    }
    Err(format!(
        "Nemotron ASR does not support language \"{language}\"; use a code such as \"de\", \
         a locale such as \"de-DE\", an English name or \"auto\""
    ))
}

/// The English name of a locale or code ("de-DE" -> "German"); the input
/// itself for languages without one here (`LanguageName`).
pub fn language_name(locale: &str) -> String {
    let normalized = normalize(locale);
    let code = base(&normalized);
    NAMES
        .iter()
        .find(|(known, _)| *known == code)
        .map_or_else(|| locale.to_owned(), |(_, name)| (*name).to_owned())
}

/// The locale of a language tag token ("<de-DE>" -> "de-DE"), if it is one
/// (`TagLocale`).
pub fn tag_locale(token: &str) -> Option<String> {
    let bytes = token.as_bytes();
    if bytes.len() < 4 || bytes[0] != b'<' || bytes[bytes.len() - 1] != b'>' {
        return None;
    }
    let inner = &token[1..token.len() - 1];
    let dash = inner.find('-')?;
    if dash == 0 || dash + 1 == inner.len() {
        return None;
    }
    if !inner.bytes().all(|c| c.is_ascii_alphabetic() || c == b'-') {
        return None;
    }
    Some(inner.to_owned())
}
