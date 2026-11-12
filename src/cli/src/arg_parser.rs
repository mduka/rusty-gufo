//! The C++ command-line parser (`src/cli/arg_parser.hpp`), ported with its
//! syntax, error texts and help layout, so ported commands print and reject
//! exactly what the C++ binary does.
//!
//! Options write into caller-owned targets, as in C++: each registration
//! borrows its target mutably for the parser's lifetime, and the values are
//! read once the parser is dropped.

use std::fmt::Write as _;
use std::path::PathBuf;

/// Receives `(name as typed, value)`; the error is the message C++ stores.
pub type ParseFn<'a> = Box<dyn FnMut(&str, &str) -> Result<(), String> + 'a>;
type PositionalFn<'a> = Box<dyn FnMut(&str) -> Result<(), String> + 'a>;

pub struct CliOption<'a> {
    pub short_name: String,
    pub long_name: String,
    pub value_hint: String,
    pub description: String,
    pub group: String,
    pub is_flag: bool,
    /// Whether the last `parse` saw this option.
    pub supplied: bool,
    parse_fn: ParseFn<'a>,
}

pub struct ArgParser<'a> {
    program_name: String,
    description: String,
    options: Vec<CliOption<'a>>,
    positional_handler: Option<PositionalFn<'a>>,
    help_requested: bool,
}

/// A typed option target (`ArgParser::ParseValue`).
pub trait OptionValue {
    fn parse_option(&mut self, value: &str, name: &str) -> Result<(), String>;
}

impl OptionValue for String {
    fn parse_option(&mut self, value: &str, _: &str) -> Result<(), String> {
        *self = value.to_owned();
        Ok(())
    }
}

impl OptionValue for PathBuf {
    fn parse_option(&mut self, value: &str, _: &str) -> Result<(), String> {
        *self = PathBuf::from(value);
        Ok(())
    }
}

/// `std::from_chars` for integers: decimal, a leading `-` only, the whole
/// value consumed, no overflow.
fn from_chars<T: std::str::FromStr>(value: &str) -> Option<T> {
    if value.starts_with('+') {
        return None;
    }
    value.parse().ok()
}

macro_rules! integer_option {
    ($($ty:ty),*) => {$(
        impl OptionValue for $ty {
            fn parse_option(&mut self, value: &str, name: &str) -> Result<(), String> {
                if value.is_empty() {
                    return Err(format!("Empty value for {name}"));
                }
                *self = from_chars(value)
                    .ok_or_else(|| format!("Invalid integer for {name}: {value}"))?;
                Ok(())
            }
        }

        impl OptionValue for Vec<$ty> {
            fn parse_option(&mut self, value: &str, name: &str) -> Result<(), String> {
                let mut result = Vec::new();
                let mut start = 0;
                while start < value.len() {
                    let comma = value[start..].find(',').map_or(value.len(), |i| start + i);
                    let token = &value[start..comma];
                    if token.is_empty() {
                        return Err(format!("Empty element in list for {name}"));
                    }
                    result.push(from_chars(token).ok_or_else(|| {
                        format!("Invalid integer in list for {name}: {token}")
                    })?);
                    start = comma + 1;
                }
                *self = result;
                Ok(())
            }
        }
    )*};
}

integer_option!(i8, i16, i32, i64, u8, u16, u32, u64, usize, isize);

impl OptionValue for f32 {
    fn parse_option(&mut self, value: &str, name: &str) -> Result<(), String> {
        *self = stof(value).ok_or_else(|| format!("Invalid float for {name}: {value}"))?;
        Ok(())
    }
}

impl OptionValue for f64 {
    fn parse_option(&mut self, value: &str, name: &str) -> Result<(), String> {
        *self = stod(value).ok_or_else(|| format!("Invalid float for {name}: {value}"))?;
        Ok(())
    }
}

/// The number `strtod` would read from the start of `text`.
enum Scanned {
    /// `[sign]digits[.digits][e[sign]digits]`, ready for Rust's parser.
    Decimal {
        literal: String,
        nonzero: bool,
    },
    Hex {
        negative: bool,
        mantissa: u128,
        exponent: i64,
        nonzero: bool,
    },
    Infinity {
        negative: bool,
    },
    NaN,
}

/// Scans like glibc `strtod`: leading C whitespace, sign, `inf`/`infinity`,
/// `nan[(chars)]`, hexadecimal and decimal forms. Returns the number and the
/// consumed length, or `None` when no conversion is possible.
fn scan_c_float(text: &str) -> Option<(Scanned, usize)> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        i += 1;
    }
    let mut negative = false;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        negative = bytes[i] == b'-';
        i += 1;
    }
    let rest = &bytes[i..];
    let starts_with =
        |word: &[u8]| rest.len() >= word.len() && rest[..word.len()].eq_ignore_ascii_case(word);
    if starts_with(b"infinity") {
        return Some((Scanned::Infinity { negative }, i + 8));
    }
    if starts_with(b"inf") {
        return Some((Scanned::Infinity { negative }, i + 3));
    }
    if starts_with(b"nan") {
        let mut end = i + 3;
        if bytes.get(end) == Some(&b'(') {
            let mut j = end + 1;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            if bytes.get(j) == Some(&b')') {
                end = j + 1;
            }
        }
        return Some((Scanned::NaN, end));
    }
    let hex_digit = |b: u8| (b as char).to_digit(16);
    if rest.len() >= 2 && rest[0] == b'0' && (rest[1] == b'x' || rest[1] == b'X') {
        let mut j = i + 2;
        let mut mantissa: u128 = 0;
        let mut exponent: i64 = 0;
        let mut digits = 0;
        let mut nonzero = false;
        let mut accumulate = |digit: u32, fraction: bool, exponent: &mut i64| {
            nonzero |= digit != 0;
            if mantissa >> 120 == 0 {
                mantissa = (mantissa << 4) | u128::from(digit);
                if fraction {
                    *exponent -= 4;
                }
            } else if !fraction {
                *exponent += 4;
            }
        };
        while let Some(digit) = bytes.get(j).and_then(|&b| hex_digit(b)) {
            accumulate(digit, false, &mut exponent);
            digits += 1;
            j += 1;
        }
        if bytes.get(j) == Some(&b'.') {
            let mut k = j + 1;
            let mut fraction_digits = 0;
            while let Some(digit) = bytes.get(k).and_then(|&b| hex_digit(b)) {
                accumulate(digit, true, &mut exponent);
                fraction_digits += 1;
                k += 1;
            }
            if digits + fraction_digits > 0 {
                digits += fraction_digits;
                j = k;
            }
        }
        if digits > 0 {
            if matches!(bytes.get(j), Some(b'p' | b'P')) {
                let mut k = j + 1;
                let mut exponent_negative = false;
                if matches!(bytes.get(k), Some(b'+' | b'-')) {
                    exponent_negative = bytes[k] == b'-';
                    k += 1;
                }
                let start = k;
                let mut value: i64 = 0;
                while let Some(&b) = bytes.get(k).filter(|b| b.is_ascii_digit()) {
                    value = value.saturating_mul(10).saturating_add(i64::from(b - b'0'));
                    k += 1;
                }
                if k > start {
                    exponent += if exponent_negative { -value } else { value };
                    j = k;
                }
            }
            return Some((
                Scanned::Hex {
                    negative,
                    mantissa,
                    exponent,
                    nonzero,
                },
                j,
            ));
        }
        // "0x" without digits reads as the zero before the x.
    }
    let mut j = i;
    let mut literal = String::from(if negative { "-" } else { "" });
    let mut digits = 0;
    let mut nonzero = false;
    while let Some(&b) = bytes.get(j).filter(|b| b.is_ascii_digit()) {
        literal.push(b as char);
        nonzero |= b != b'0';
        digits += 1;
        j += 1;
    }
    if bytes.get(j) == Some(&b'.') {
        let mut k = j + 1;
        let mut fraction = String::from(".");
        while let Some(&b) = bytes.get(k).filter(|b| b.is_ascii_digit()) {
            fraction.push(b as char);
            nonzero |= b != b'0';
            digits += 1;
            k += 1;
        }
        if digits > 0 {
            literal.push_str(&fraction);
            j = k;
        }
    }
    if digits == 0 {
        return None;
    }
    if matches!(bytes.get(j), Some(b'e' | b'E')) {
        let mut k = j + 1;
        let mut exponent = String::from("e");
        if matches!(bytes.get(k), Some(b'+' | b'-')) {
            exponent.push(bytes[k] as char);
            k += 1;
        }
        let start = k;
        while let Some(&b) = bytes.get(k).filter(|b| b.is_ascii_digit()) {
            exponent.push(b as char);
            k += 1;
        }
        if k > start {
            literal.push_str(&exponent);
            j = k;
        }
    }
    Some((Scanned::Decimal { literal, nonzero }, j))
}

fn ldexp(mantissa: u128, exponent: i64) -> f64 {
    let mut value = mantissa as f64;
    let mut exponent = exponent;
    while exponent > 0 {
        let step = exponent.min(1000);
        value *= 2f64.powi(step as i32);
        exponent -= step;
    }
    while exponent < 0 {
        let step = (-exponent).min(1000);
        value /= 2f64.powi(step as i32);
        exponent += step;
    }
    value
}

/// `std::stof`: the whole text must convert, and out-of-range results
/// (`ERANGE`: overflow, or underflow to zero or an inexact subnormal) fail.
pub fn stof(text: &str) -> Option<f32> {
    let (scanned, end) = scan_c_float(text)?;
    if end != text.len() {
        return None;
    }
    match scanned {
        Scanned::Infinity { negative } => Some(if negative {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }),
        Scanned::NaN => Some(f32::NAN),
        Scanned::Decimal { literal, nonzero } => {
            let value: f32 = literal.parse().ok()?;
            let wide: f64 = literal.parse().ok()?;
            float_in_range(value, wide, nonzero)
        }
        Scanned::Hex {
            negative,
            mantissa,
            exponent,
            nonzero,
        } => {
            let wide = ldexp(mantissa, exponent);
            let wide = if negative { -wide } else { wide };
            float_in_range(wide as f32, wide, nonzero)
        }
    }
}

fn float_in_range(value: f32, wide: f64, nonzero: bool) -> Option<f32> {
    if value.is_infinite() || (value == 0.0 && nonzero) {
        return None;
    }
    if value.is_subnormal() && f64::from(value) != wide {
        return None;
    }
    Some(value)
}

/// `std::stod`, with the same range rules as [`stof`].
pub fn stod(text: &str) -> Option<f64> {
    let (scanned, end) = scan_c_float(text)?;
    if end != text.len() {
        return None;
    }
    let (value, nonzero) = match scanned {
        Scanned::Infinity { negative } => {
            return Some(if negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            });
        }
        Scanned::NaN => return Some(f64::NAN),
        Scanned::Decimal { literal, nonzero } => (literal.parse::<f64>().ok()?, nonzero),
        Scanned::Hex {
            negative,
            mantissa,
            exponent,
            nonzero,
        } => {
            let value = ldexp(mantissa, exponent);
            (if negative { -value } else { value }, nonzero)
        }
    };
    if value.is_infinite() || (value == 0.0 && nonzero) || value.is_subnormal() {
        return None;
    }
    Some(value)
}

impl<'a> ArgParser<'a> {
    pub fn new(program_name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            program_name: program_name.into(),
            description: description.into(),
            options: Vec::new(),
            positional_handler: None,
            help_requested: false,
        }
    }

    fn push(
        &mut self,
        names: (&str, &str),
        value_hint: &str,
        description: &str,
        group: &str,
        is_flag: bool,
        parse_fn: ParseFn<'a>,
    ) -> &mut Self {
        self.options.push(CliOption {
            short_name: names.0.to_owned(),
            long_name: names.1.to_owned(),
            value_hint: value_hint.to_owned(),
            description: description.to_owned(),
            group: group.to_owned(),
            is_flag,
            supplied: false,
            parse_fn,
        });
        self
    }

    /// A switch that sets `target` (`AddFlag`).
    pub fn add_flag(
        &mut self,
        short_name: &str,
        long_name: &str,
        description: &str,
        group: &str,
        target: &'a mut bool,
    ) -> &mut Self {
        let parse_fn: ParseFn<'a> = Box::new(move |_, _| {
            *target = true;
            Ok(())
        });
        self.push(
            (short_name, long_name),
            "",
            description,
            group,
            true,
            parse_fn,
        )
    }

    /// A switch that clears `target` (`AddInverseFlag`).
    pub fn add_inverse_flag(
        &mut self,
        short_name: &str,
        long_name: &str,
        description: &str,
        group: &str,
        target: &'a mut bool,
    ) -> &mut Self {
        let parse_fn: ParseFn<'a> = Box::new(move |_, _| {
            *target = false;
            Ok(())
        });
        self.push(
            (short_name, long_name),
            "",
            description,
            group,
            true,
            parse_fn,
        )
    }

    /// A typed option (`AddOption`).
    pub fn add_option<T: OptionValue + 'a>(
        &mut self,
        short_name: &str,
        long_name: &str,
        value_hint: &str,
        description: &str,
        group: &str,
        target: &'a mut T,
    ) -> &mut Self {
        let parse_fn: ParseFn<'a> = Box::new(move |name, value| target.parse_option(value, name));
        self.push(
            (short_name, long_name),
            value_hint,
            description,
            group,
            false,
            parse_fn,
        )
    }

    /// An option with its own parser (`AddCustomOption`).
    pub fn add_custom_option(
        &mut self,
        short_name: &str,
        long_name: &str,
        value_hint: &str,
        description: &str,
        group: &str,
        parse_fn: impl FnMut(&str, &str) -> Result<(), String> + 'a,
    ) -> &mut Self {
        self.push(
            (short_name, long_name),
            value_hint,
            description,
            group,
            false,
            Box::new(parse_fn),
        )
    }

    /// `SetPositionalHandler`.
    pub fn set_positional_handler(
        &mut self,
        handler: impl FnMut(&str) -> Result<(), String> + 'a,
    ) -> &mut Self {
        self.positional_handler = Some(Box::new(handler));
        self
    }

    /// `CollectPositionals`.
    pub fn collect_positionals(&mut self, target: &'a mut Vec<String>) -> &mut Self {
        self.set_positional_handler(move |value| {
            target.push(value.to_owned());
            Ok(())
        })
    }

    /// `JoinPositionals`: positionals joined by single spaces.
    // No current command joins its positionals.
    #[allow(dead_code)]
    pub fn join_positionals(&mut self, target: &'a mut String) -> &mut Self {
        self.set_positional_handler(move |value| {
            if !target.is_empty() {
                target.push(' ');
            }
            target.push_str(value);
            Ok(())
        })
    }

    fn find_index(&self, name: &str) -> Option<usize> {
        self.options.iter().position(|option| {
            (!option.short_name.is_empty() && name == option.short_name)
                || (!option.long_name.is_empty() && name == option.long_name)
        })
    }

    /// `FindOption`: the option named `name` (short or long).
    pub fn find_option(&self, name: &str) -> Option<&CliOption<'a>> {
        self.find_index(name).map(|index| &self.options[index])
    }

    /// `Parse`: `-h`/`--help` stops parsing and succeeds; the error is the
    /// C++ message (possibly empty for a custom parser that set none).
    pub fn parse<S: AsRef<str>>(&mut self, args: &[S]) -> Result<(), String> {
        self.help_requested = false;
        for option in &mut self.options {
            option.supplied = false;
        }
        let mut i = 0;
        while i < args.len() {
            let arg = args[i].as_ref();
            if arg == "-h" || arg == "--help" {
                self.help_requested = true;
                return Ok(());
            }
            // `--name=value` / `-n=value` carry their own value.
            if let Some(equals) = arg.find('=').filter(|&e| e > 0 && arg.starts_with('-')) {
                let (name, value) = (&arg[..equals], &arg[equals + 1..]);
                if let Some(index) = self.find_index(name) {
                    let option = &mut self.options[index];
                    if option.is_flag {
                        return Err(format!("Option {name} does not take a value"));
                    }
                    (option.parse_fn)(name, value)?;
                    option.supplied = true;
                    i += 1;
                    continue;
                }
            }
            if let Some(index) = self.find_index(arg) {
                let option = &mut self.options[index];
                option.supplied = true;
                if option.is_flag {
                    (option.parse_fn)(arg, "")?;
                } else {
                    if i + 1 >= args.len() {
                        return Err(format!("Missing argument for {arg}"));
                    }
                    i += 1;
                    (option.parse_fn)(arg, args[i].as_ref())?;
                }
            } else if arg.len() > 1 && arg.starts_with('-') {
                return Err(format!("Unknown option: {arg}"));
            } else if let Some(handler) = self.positional_handler.as_mut() {
                handler(arg)?;
            } else {
                return Err(format!("Unexpected argument: {arg}"));
            }
            i += 1;
        }
        Ok(())
    }

    pub fn is_help_requested(&self) -> bool {
        self.help_requested
    }

    /// `WasSupplied`: whether the last `parse` saw the option `name`.
    pub fn was_supplied(&self, name: &str) -> bool {
        self.find_option(name).is_some_and(|option| option.supplied)
    }

    /// `FormatHelp`: groups in first-use order, "General" last and joined by
    /// the implicit `-h, --help`.
    pub fn format_help(&self) -> String {
        let mut out = format!("Usage: {} [OPTIONS]", self.program_name);
        if self.positional_handler.is_some() {
            out.push_str(" [ARGS...]");
        }
        out.push_str("\n\n");
        if !self.description.is_empty() {
            let _ = write!(out, "{}\n\n", self.description);
        }
        let mut group_order: Vec<&str> = Vec::new();
        for option in &self.options {
            let group = if option.group.is_empty() {
                "Options"
            } else {
                option.group.as_str()
            };
            if !group_order.contains(&group) {
                group_order.push(group);
            }
        }
        let has_general = group_order.contains(&"General");
        if let Some(position) = group_order.iter().position(|&g| g == "General") {
            let general = group_order.remove(position);
            group_order.push(general);
        }
        for group in &group_order {
            let _ = writeln!(out, "{group}:");
            for option in &self.options {
                let option_group = if option.group.is_empty() {
                    "Options"
                } else {
                    option.group.as_str()
                };
                if option_group != *group {
                    continue;
                }
                let mut flags = String::from("  ");
                if option.short_name.is_empty() {
                    flags.push_str("    ");
                } else {
                    flags.push_str(&option.short_name);
                    if !option.long_name.is_empty() {
                        flags.push_str(", ");
                    }
                }
                flags.push_str(&option.long_name);
                if !option.value_hint.is_empty() {
                    let _ = write!(flags, " <{}>", option.value_hint);
                }
                let _ = writeln!(out, "{flags:<28} {}", option.description);
            }
            if !(has_general && *group == "General") {
                out.push('\n');
            }
        }
        if !has_general {
            out.push_str("General:\n");
        }
        let _ = writeln!(out, "{:<28} Print help", "  -h, --help");
        out
    }

    /// `PrintHelp` to standard output.
    pub fn print_help(&self) {
        print!("{}", self.format_help());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supplied_options_are_tracked_per_parse() {
        let (mut text, mut verbose) = (String::new(), false);
        let mut parser = ArgParser::new("demo", "Demo.");
        parser
            .add_option("", "--text", "T", "Text", "Main", &mut text)
            .add_flag("-v", "--verbose", "Say more", "General", &mut verbose);
        parser.parse(&["--text=x"]).expect("parse");
        assert!(parser.was_supplied("--text") && !parser.was_supplied("-v"));
        parser.parse(&["--text", "y", "-v"]).expect("parse");
        assert!(parser.was_supplied("--text") && parser.was_supplied("--verbose"));
        parser.parse(&["--help"]).expect("parse");
        assert!(!parser.was_supplied("--text"), "each parse starts afresh");
    }

    #[test]
    fn options_flags_inline_values_and_errors() {
        let (mut count, mut name, mut path, mut verbose, mut color) =
            (3usize, String::new(), PathBuf::new(), false, true);
        let mut list: Vec<u32> = Vec::new();
        let mut rest = Vec::new();
        {
            let mut parser = ArgParser::new("demo", "Demo.");
            parser
                .add_option("-n", "--count", "N", "How many", "Main", &mut count)
                .add_option("", "--name", "TEXT", "A name", "", &mut name)
                .add_option("", "--path", "P", "A path", "Main", &mut path)
                .add_option("", "--list", "L", "Numbers", "Main", &mut list)
                .add_flag("-v", "--verbose", "Say more", "General", &mut verbose)
                .add_inverse_flag("", "--no-color", "Plain", "General", &mut color)
                .collect_positionals(&mut rest);
            parser
                .parse(&[
                    "-n",
                    "5",
                    "--name=-v",
                    "x",
                    "--path",
                    "a b",
                    "--list",
                    "1,2",
                    "-v",
                    "--no-color",
                    "-",
                ])
                .unwrap_or_else(|e| panic!("{e}"));
            assert!(!parser.is_help_requested());
        }
        assert_eq!(
            (count, name.as_str(), verbose, color),
            (5, "-v", true, false)
        );
        assert_eq!(path, PathBuf::from("a b"));
        assert_eq!(list, [1, 2]);
        assert_eq!(rest, ["x", "-"]);

        let mut value = 0i32;
        let mut flag = false;
        let mut parser = ArgParser::new("demo", "");
        parser
            .add_option("-p", "--port", "N", "", "", &mut value)
            .add_flag("-v", "", "", "", &mut flag);
        let cases: [(&[&str], &str); 7] = [
            (&["--port"], "Missing argument for --port"),
            (&["-p", ""], "Empty value for -p"),
            (&["--port=+1"], "Invalid integer for --port: +1"),
            (&["-p", "1x"], "Invalid integer for -p: 1x"),
            (&["-v=1"], "Option -v does not take a value"),
            (&["--bogus=1"], "Unknown option: --bogus=1"),
            (&["x"], "Unexpected argument: x"),
        ];
        for (args, message) in cases {
            assert_eq!(parser.parse(args), Err(message.to_owned()), "{args:?}");
        }
        assert_eq!(parser.parse(&["-p", "1", "--help", "--bogus"]), Ok(()));
        assert!(parser.is_help_requested());
    }

    #[test]
    fn lists_and_floats_follow_the_cpp_parsers() {
        let mut list: Vec<i64> = Vec::new();
        assert_eq!(
            list.parse_option("1,,2", "--l"),
            Err("Empty element in list for --l".into())
        );
        assert_eq!(
            list.parse_option("1,x", "--l"),
            Err("Invalid integer in list for --l: x".into())
        );
        assert_eq!(list.parse_option("1,", "--l"), Ok(()));
        assert_eq!(list, [1]);
        for (text, expected) in [
            ("0.5", Some(0.5)),
            (" \t0.5", Some(0.5)),
            ("+1e2", Some(100.0)),
            (".5", Some(0.5)),
            ("5.", Some(5.0)),
            ("0x1p-1", Some(0.5)),
            ("0x.8", Some(0.5)),
            ("1e", None),
            ("0x", None),
            ("1e50", None),
            ("1e-50", None),
            ("0.5 ", None),
            ("", None),
            (" ", None),
            ("abc", None),
        ] {
            assert_eq!(stof(text), expected, "{text:?}");
        }
        assert_eq!(stof("-INFINITY"), Some(f32::NEG_INFINITY));
        assert!(stof("nan(12)").is_some_and(f32::is_nan));
        assert_eq!(stof("0e999"), Some(0.0));
        assert_eq!(stod("1e300"), Some(1e300));
        assert_eq!(stod("1e400"), None);
    }

    #[test]
    fn help_layout_matches_cpp() {
        let (mut a, mut b, mut c) = (String::new(), 0u32, false);
        let mut parser = ArgParser::new("gufo demo", "Describe.");
        parser
            .add_flag("-v", "--verbose", "Verbose", "General", &mut c)
            .add_option("-m", "--model", "PATH", "Model path", "Model", &mut a)
            .add_option("", "--a-very-long-option-name", "N", "Long", "", &mut b);
        assert_eq!(
            parser.format_help(),
            "Usage: gufo demo [OPTIONS]\n\nDescribe.\n\nModel:\n  -m, --model <PATH>         Model path\n\n\
             Options:\n      --a-very-long-option-name <N> Long\n\nGeneral:\n  -v, --verbose              Verbose\n  \
             -h, --help                 Print help\n"
        );
        let mut rest = String::new();
        let mut plain = ArgParser::new("p", "");
        plain.join_positionals(&mut rest);
        assert_eq!(
            plain.format_help(),
            "Usage: p [OPTIONS] [ARGS...]\n\nGeneral:\n  -h, --help                 Print help\n"
        );
    }
}
