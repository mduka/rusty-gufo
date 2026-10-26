//! The two-line startup banner of interactive commands
//! (`src/cli/banner.cpp`).

use std::io::{IsTerminal, Write};

/// Whether the environment advertises a UTF-8 locale, so the banner can use
/// the rounded eyes and the middle dot without risking mojibake.
fn supports_unicode() -> bool {
    ["LC_ALL", "LC_CTYPE", "LANG"]
        .iter()
        .filter_map(|name| std::env::var(name).ok())
        .find(|value| !value.is_empty())
        .is_some_and(|locale| locale.contains("UTF-8") || locale.contains("utf8"))
}

/// `--json` keeps structured output machine-readable, and help pages read
/// better without the banner above them.
fn suppresses_banner(options: &[String]) -> bool {
    options
        .iter()
        .any(|option| matches!(option.as_str(), "--json" | "--help" | "-h"))
}

/// Writes the banner to stderr when it is a terminal and `options` do not
/// ask for machine-readable output (`PrintStartupBanner`).
pub fn print_startup_banner(version: &str, options: &[String]) {
    if !std::io::stderr().is_terminal() || suppresses_banner(options) {
        return;
    }
    let color = std::env::var_os("NO_COLOR").is_none();
    let unicode = supports_unicode();
    let eyes = if unicode { "(◉,◉)" } else { "(o,o)" };
    let separator = if unicode { " · " } else { " - " };
    // Logo orange, matching the eyes in assets/gufo-logo.jpg.
    let (tint, bold, reset) = if color {
        ("\x1b[38;2;232;140;42m", "\x1b[1m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    let _ = write!(
        std::io::stderr().lock(),
        "  {tint},___,{reset}   {bold}gufo{reset} {version}\n  {tint}{eyes}{reset}   Fast LLM & \
         DiT inference{separator}gfx1151\n"
    );
}
