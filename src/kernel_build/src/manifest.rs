//! `launch.toml` schema and validation.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::Deserialize;

use crate::extent::{Expr, Relation};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub library: Library,
    #[serde(default, rename = "source")]
    pub sources: Vec<Source>,
    #[serde(default, rename = "launcher")]
    pub launchers: Vec<Launcher>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Library {
    /// Symbol prefix (`gufo_<name>_`) and archive name.
    pub name: String,
    /// Repository root, relative to the manifest.
    pub root: PathBuf,
    pub namespace: String,
    pub headers: Vec<String>,
    /// C++ type of `format` parameters (for example `gufo::core::GgmlType`,
    /// an enum over `std::uint16_t` holding GGML type IDs).
    pub format_type: Option<String>,
    /// C++ element type of `f16` buffers (default `void`, for launchers that
    /// take `void*`; for example `__half`).
    pub f16_type: Option<String>,
}

#[derive(Deserialize, Hash, Clone)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub path: PathBuf,
    #[serde(default = "default_std")]
    pub std: String,
    #[serde(default)]
    pub flags: Vec<String>,
    #[serde(default)]
    pub defines: Vec<String>,
}

pub fn default_std() -> String {
    "c++20".into()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Launcher {
    pub name: String,
    /// The C++ launcher, relative to the library namespace unless it starts
    /// with `::`.
    pub cpp: String,
    pub returns: Returns,
    #[serde(default)]
    pub doc: String,
    pub params: Vec<Param>,
    /// Relations between integer scalars that the wrapper checks before the
    /// launch, for conditions a kernel indexes by but no extent expresses
    /// (`start_position + tokens <= max_context`).
    #[serde(default)]
    pub requires: Vec<String>,
    /// Concrete scalar values (and optional buffers under `with`, and values
    /// for `u32` input buffers) for the generated extent probes; at least one
    /// per launcher.
    #[serde(default)]
    pub probes: Vec<toml::Table>,
}

/// A validated probe: a Rust literal per scalar parameter, the optional
/// buffers to pass, and the value that fills a `u32` input buffer (others hold
/// zeros) so that data-dependent writes such as a ring slot are reached.
pub struct Probe {
    pub literals: BTreeMap<String, String>,
    pub with: BTreeSet<String>,
    pub fills: BTreeMap<String, u32>,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Returns {
    /// `void` launcher: every accepted call enqueues work.
    Void,
    /// `bool` launcher: `false` means unsupported geometry, nothing enqueued.
    Supported,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Param {
    pub name: String,
    pub kind: Kind,
    pub elem: Option<Elem>,
    pub len: Option<String>,
    #[serde(rename = "type")]
    pub scalar: Option<Scalar>,
    #[serde(default)]
    pub optional: bool,
    /// Buffers only (not listed, not optional): the C++ launcher writes it
    /// from the host before it launches, through its device pointer, so it
    /// must be memory the host addresses at its device address.
    #[serde(default)]
    pub host_visible: bool,
    /// `indices` and `indices_out`: every value must be below this
    /// expression (after `mask`).
    pub below: Option<String>,
    /// `indices` and `indices_out`: the kernel skips (or writes) the padding
    /// value `u32::MAX` (`-1` as `i32`).
    #[serde(default)]
    pub padded: bool,
    /// `indices` and `indices_out`: the bound applies to `value & mask`.
    pub mask: Option<u32>,
    /// `indices` and `indices_out`: the C++ launcher takes `int32` values
    /// (`std::int32_t*`), which must stay non-negative (padding aside).
    #[serde(default)]
    pub signed: bool,
    /// `indices`: the values must have been computed from this `indices`
    /// parameter's current generation; `indices_out`: the launch computes
    /// them from it.
    pub derived_from: Option<String>,
    /// With `derived_from`: the geometry of the producing launch, an integer
    /// expression both sides must agree on.
    pub tag: Option<String>,
    /// Scalars only: part of the Rust contract (extents, bounds,
    /// requirements) but not an argument of the C++ launcher.
    #[serde(default)]
    pub contract_only: bool,
    /// `quant` only: rows and columns of the weights, and the storage formats
    /// the launcher supports (by GGML name).
    pub rows: Option<String>,
    pub cols: Option<String>,
    #[serde(default)]
    pub formats: Vec<String>,
    /// `quant` only: the kernel requires the format of this other `quant`
    /// parameter (paired gate/up weights decoded with one format).
    pub same_format: Option<String>,
    /// `quant` only: bytes past the last row the kernel may read (whole
    /// k-iterations of the quantized tier); the weights must carry them.
    pub tail: Option<String>,
    /// `format`: the `quant` parameter whose format this passes; `alias`: the
    /// written buffer whose pointer this repeats.
    pub of: Option<String>,
    /// `const` only: the C++ parameter type and the expression passed.
    pub cpp_type: Option<String>,
    pub value: Option<String>,
    /// `alias` only: the C++ parameter is a const pointer.
    #[serde(default)]
    pub read_only: bool,
    /// `table` only: values per row.
    pub width: Option<u32>,
    /// Buffers and scalars: one value per element of a list whose length is
    /// this integer scalar; C++ takes a host array (of device pointers for
    /// buffers), Rust a slice. Element extents and requirements may use the
    /// element's list scalars.
    pub list: Option<String>,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    In,
    Inout,
    Out,
    /// Read-only `u32` values a kernel indexes with (`gufo_hip::IndexSlice`).
    Indices,
    /// `u32` values a kernel writes that satisfy `below` (and `mask`,
    /// `padded`) by its contract (`gufo_hip::IndexSliceMut`).
    #[serde(rename = "indices_out")]
    IndicesOut,
    /// Read-only quantized weights (`const void*`, `gufo_hip::QuantSlice`),
    /// sized by `rows`, `cols` and the slice's GGML format.
    Quant,
    /// The GGML format of the `quant` parameter `of`, as `format_type`.
    Format,
    /// A fixed C++ argument (`cpp_type`, `value`) the Rust side never passes,
    /// for defaulted parameters such as a null vision input.
    Const,
    /// The pointer of the written buffer `of` once more, for launchers that
    /// take input and output separately but run in place
    /// (`LaunchResidualAdd(h, x, h)`).
    Alias,
    /// The launcher's `hipStream_t`, for C++ signatures that do not take it
    /// last (defaulted parameters after it); otherwise it is appended.
    Stream,
    /// Rows the kernels read through a device header of `cpp_type` (a
    /// pointer to `{ const elem* rows; u32 count; i32 tag }`), passed as
    /// `gufo_hip::RowTable<elem, width>`; kernels read at most `count` rows.
    /// Always optional (absent: a null pointer).
    Table,
    Scalar,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Elem {
    F32,
    U32,
    I32,
    U8,
    U64,
    Bf16,
    F16,
    /// Opaque bytes of a layout only the kernels know, such as the tiled
    /// Q8_1 activations: `u8` in Rust, `void*` in C++.
    Bytes,
}

impl Elem {
    pub fn rust(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::U8 => "u8",
            Self::U64 => "u64",
            Self::Bf16 => "::gufo_hip::Bf16",
            Self::F16 => "::gufo_hip::F16",
            Self::Bytes => "u8",
        }
    }

    /// Pointee type the C++ launchers use (BF16, F16 and opaque bytes travel
    /// as `void*`).
    pub fn cpp(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::U32 => "std::uint32_t",
            Self::I32 => "std::int32_t",
            Self::U8 => "std::uint8_t",
            Self::U64 => "std::uint64_t",
            Self::Bf16 | Self::F16 | Self::Bytes => "void",
        }
    }
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Scalar {
    Usize,
    U32,
    I32,
    F32,
    F64,
    Bool,
}

impl Scalar {
    pub fn rust(self) -> &'static str {
        match self {
            Self::Usize => "usize",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Bool => "bool",
        }
    }

    pub fn cpp(self) -> &'static str {
        match self {
            Self::Usize => "std::size_t",
            Self::U32 => "std::uint32_t",
            Self::I32 => "std::int32_t",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::Bool => "bool",
        }
    }

    pub fn is_integer(self) -> bool {
        matches!(self, Self::Usize | Self::U32 | Self::I32)
    }
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Self, String> {
        let manifest: Self = toml::from_str(text).map_err(|e| e.to_string())?;
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate(&self) -> Result<(), String> {
        let ident = |s: &str| {
            !s.is_empty()
                && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && !s.starts_with(|c: char| c.is_ascii_digit())
        };
        if !ident(&self.library.name) {
            return Err(format!(
                "library name `{}` is not an identifier",
                self.library.name
            ));
        }
        let mut launchers = BTreeSet::new();
        for launcher in &self.launchers {
            // `cpp` may name a nested namespace of the library's
            // (`kernels::LaunchX`) or, after a leading `::`, any namespace.
            let cpp = launcher.cpp.strip_prefix("::").unwrap_or(&launcher.cpp);
            if !ident(&launcher.name) || !cpp.split("::").all(ident) {
                return Err(format!(
                    "launcher `{}`: name and cpp must be identifiers",
                    launcher.name
                ));
            }
            if !launchers.insert(&launcher.name) {
                return Err(format!("launcher `{}` declared twice", launcher.name));
            }
            if self.library.format_type.is_none()
                && launcher.params.iter().any(|p| p.kind == Kind::Format)
            {
                return Err(format!(
                    "launcher `{}`: format parameters need `format_type` in [library]",
                    launcher.name
                ));
            }
            launcher.validate()?;
        }
        Ok(())
    }
}

/// Names the generated code keeps for itself: per-parameter locals
/// (`x_<name>`, `<name>_ptr`, `len_<name>`, ...), fixed locals and Rust
/// keywords. A parameter with such a name would shadow them or not compile.
fn reserved_name(name: &str) -> Option<&'static str> {
    const PREFIXES: [&str; 6] = ["x_", "len_", "buffer_", "below_", "spec_", "format_"];
    const SUFFIXES: [&str; 2] = ["_ptr", "_ptrs"];
    const LOCALS: [&str; 6] = ["raw", "l", "i", "len", "buffer", "violations"];
    const KEYWORDS: [&str; 38] = [
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move",
        "mut", "pub", "ref", "return", "self", "static", "struct", "super", "trait", "true",
        "type", "unsafe", "use", "where", "while", "gen",
    ];
    let valid = name.starts_with(|c: char| c.is_ascii_lowercase())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if !valid {
        Some("names are lowercase snake-case identifiers")
    } else if PREFIXES.iter().any(|p| name.starts_with(p))
        || SUFFIXES.iter().any(|s| name.ends_with(s))
        || LOCALS.contains(&name)
    {
        Some("the name collides with a local of the generated code")
    } else if KEYWORDS.contains(&name) {
        Some("the name is a Rust keyword")
    } else {
        None
    }
}

impl Launcher {
    /// The integer scalar that sizes the launcher's lists, if it has any.
    pub fn list_count(&self) -> Option<&str> {
        self.params.iter().find_map(|p| p.list.as_deref())
    }

    /// Names of the scalars that take one value per list element.
    pub fn list_scalars(&self) -> BTreeSet<&str> {
        self.params
            .iter()
            .filter(|p| p.kind == Kind::Scalar && p.list.is_some())
            .map(|p| p.name.as_str())
            .collect()
    }

    /// The probes with every scalar as a typed Rust literal.
    pub fn probes(&self) -> Result<Vec<Probe>, String> {
        if self.probes.is_empty() {
            return Err(format!(
                "launcher `{}`: declare at least one probe (scalar values for the extent test)",
                self.name
            ));
        }
        self.probes
            .iter()
            .enumerate()
            .map(|(index, table)| self.probe(index, table))
            .collect()
    }

    fn probe(&self, index: usize, table: &toml::Table) -> Result<Probe, String> {
        let at = format!("launcher `{}` probe {index}", self.name);
        let mut literals = BTreeMap::new();
        let mut with = BTreeSet::new();
        let mut fills = BTreeMap::new();
        for (key, value) in table {
            if let Some(p) = self.params.iter().find(|p| {
                p.name == *key
                    && ((p.kind == Kind::In && p.elem == Some(Elem::U32))
                        || matches!(p.kind, Kind::Indices | Kind::IndicesOut))
            }) {
                let fill = value
                    .as_integer()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or(format!("{at}: `{key}` must be a u32 fill value"))?;
                fills.insert(p.name.clone(), fill);
                continue;
            }
            if let Some(p) = self
                .params
                .iter()
                .find(|p| p.name == *key && p.kind == Kind::Quant)
            {
                let format = value
                    .as_str()
                    .filter(|name| gufo_formats::GgmlType::from_name(name).is_some())
                    .ok_or(format!("{at}: `{key}` must name a GGML format"))?;
                if !p.formats.iter().any(|f| f == format) {
                    return Err(format!(
                        "{at}: `{key}` uses a format the launcher does not list"
                    ));
                }
                literals.insert(p.name.clone(), format.to_string());
                continue;
            }
            if key == "with" {
                let names = value
                    .as_array()
                    .ok_or(format!("{at}: `with` must list buffer names"))?;
                for name in names {
                    let name = name
                        .as_str()
                        .ok_or(format!("{at}: `with` must list buffer names"))?;
                    if !self
                        .params
                        .iter()
                        .any(|p| p.name == name && p.optional && p.kind != Kind::Table)
                    {
                        return Err(format!(
                            "{at}: `{name}` in `with` is not an optional buffer"
                        ));
                    }
                    with.insert(name.to_string());
                }
                continue;
            }
            let param = self
                .params
                .iter()
                .find(|p| p.name == *key && p.scalar.is_some())
                .ok_or(format!("{at}: `{key}` is not a scalar parameter"))?;
            let scalar = param.scalar.unwrap_or(Scalar::Usize);
            if param.list.is_some() {
                // Expanded to one value per element once the count is known.
                continue;
            }
            let literal = match (scalar, value) {
                (Scalar::Usize, toml::Value::Integer(v)) if *v >= 0 => format!("{v}usize"),
                (Scalar::U32, toml::Value::Integer(v)) if u32::try_from(*v).is_ok() => {
                    format!("{v}u32")
                }
                (Scalar::I32, toml::Value::Integer(v)) if i32::try_from(*v).is_ok() => {
                    format!("{v}i32")
                }
                (Scalar::F32, toml::Value::Integer(v)) => format!("{v}.0_f32"),
                (Scalar::F32, toml::Value::Float(v)) if v.is_finite() => format!("{v:?}_f32"),
                (Scalar::F64, toml::Value::Integer(v)) => format!("{v}.0_f64"),
                (Scalar::F64, toml::Value::Float(v)) if v.is_finite() => format!("{v:?}_f64"),
                (Scalar::Bool, toml::Value::Boolean(v)) => v.to_string(),
                _ => return Err(format!("{at}: `{key}` is not a valid {}", scalar.rust())),
            };
            literals.insert(key.clone(), literal);
        }
        let integers: BTreeMap<String, u64> = table
            .iter()
            .filter_map(|(k, v)| Some((k.clone(), u64::try_from(v.as_integer()?).ok()?)))
            .filter(|(k, _)| !self.list_scalars().contains(k.as_str()))
            .collect();
        // List scalars: one integer for every element, or one per element.
        let mut elements: Vec<BTreeMap<String, u64>> = Vec::new();
        if let Some(count) = self.list_count() {
            let count = integers
                .get(count)
                .copied()
                .and_then(|c| usize::try_from(c).ok())
                .ok_or(format!("{at}: missing list count `{count}`"))?;
            elements = vec![integers.clone(); count];
            for p in self
                .params
                .iter()
                .filter(|p| p.kind == Kind::Scalar && p.list.is_some())
            {
                let values: Vec<u64> = match table.get(&p.name) {
                    Some(toml::Value::Integer(v)) => {
                        vec![u64::try_from(*v).unwrap_or(u64::MAX); count]
                    }
                    Some(toml::Value::Array(values)) => values
                        .iter()
                        .map(|v| {
                            v.as_integer()
                                .and_then(|v| u64::try_from(v).ok())
                                .unwrap_or(u64::MAX)
                        })
                        .collect(),
                    _ => return Err(format!("{at}: missing list scalar `{}`", p.name)),
                };
                let scalar = p.scalar.unwrap_or(Scalar::Usize);
                let limit = match scalar {
                    Scalar::U32 => u64::from(u32::MAX),
                    Scalar::Usize => u64::MAX - 1,
                    _ => {
                        return Err(format!(
                            "{at}: list scalar `{}` must be u32 or usize",
                            p.name
                        ));
                    }
                };
                if values.len() != count || values.iter().any(|&v| v > limit) {
                    return Err(format!(
                        "{at}: `{}` needs {count} valid {} values",
                        p.name,
                        scalar.rust()
                    ));
                }
                let suffix = scalar.rust();
                let items: Vec<String> = values.iter().map(|v| format!("{v}{suffix}")).collect();
                literals.insert(p.name.clone(), format!("vec![{}]", items.join(", ")));
                for (element, value) in elements.iter_mut().zip(values) {
                    element.insert(p.name.clone(), value);
                }
            }
        }
        for p in self
            .params
            .iter()
            .filter(|p| matches!(p.kind, Kind::Scalar | Kind::Quant))
        {
            if !literals.contains_key(&p.name) {
                let what = if p.kind == Kind::Quant {
                    "format of"
                } else {
                    "scalar"
                };
                return Err(format!("{at}: missing {what} `{}`", p.name));
            }
        }
        let list_scalars = self.list_scalars();
        for requirement in &self.requires {
            let relation = Relation::parse(requirement).map_err(|e| format!("{at}: {e}"))?;
            let per_element = relation
                .idents()
                .iter()
                .any(|name| list_scalars.contains(name.as_str()));
            let satisfied = if per_element {
                elements.iter().all(|e| relation.eval(e) == Some(true))
            } else {
                relation.eval(&integers) == Some(true)
            };
            if !satisfied {
                return Err(format!("{at}: violates `{requirement}`"));
            }
        }
        for p in self
            .params
            .iter()
            .filter(|p| matches!(p.kind, Kind::Indices | Kind::IndicesOut))
        {
            let below = p.below.as_deref().unwrap_or_default();
            if Expr::parse(below).ok().and_then(|e| e.eval(&integers)) < Some(1) {
                return Err(format!("{at}: `below` of `{}` must be at least 1", p.name));
            }
        }
        for name in fills.keys() {
            if self.params.iter().any(|p| p.name == *name && p.optional) && !with.contains(name) {
                return Err(format!("{at}: `{name}` is filled but not passed `with`"));
            }
        }
        for p in &self.params {
            if let Some(other) = &p.same_format
                && literals.get(&p.name) != literals.get(other)
            {
                return Err(format!(
                    "{at}: `{}` must use the format of `{other}`",
                    p.name
                ));
            }
        }
        Ok(Probe {
            literals,
            with,
            fills,
        })
    }

    fn validate(&self) -> Result<(), String> {
        let scalars: BTreeMap<&str, Scalar> = self
            .params
            .iter()
            .filter_map(|p| p.scalar.map(|s| (p.name.as_str(), s)))
            .collect();
        let integer_names = |at: &str, what: &str, names: Vec<String>| -> Result<(), String> {
            for name in names {
                match scalars.get(name.as_str()) {
                    Some(s) if s.is_integer() => {}
                    Some(_) => return Err(format!("{at}: `{name}` in {what} is not an integer")),
                    None => {
                        return Err(format!("{at}: `{name}` in {what} is not a scalar param"));
                    }
                }
            }
            Ok(())
        };
        let list_scalars = self.list_scalars();
        let counts: BTreeSet<&str> = self
            .params
            .iter()
            .filter_map(|p| p.list.as_deref())
            .collect();
        if counts.len() > 1 {
            return Err(format!(
                "launcher `{}`: all lists share one count",
                self.name
            ));
        }
        // Expressions of whole buffers cannot use per-element scalars.
        let whole = |at: &str, what: &str, expr: &Expr| -> Result<(), String> {
            match expr
                .idents()
                .into_iter()
                .find(|name| list_scalars.contains(name.as_str()))
            {
                Some(name) => Err(format!(
                    "{at}: {what} uses the list scalar `{name}` outside a list"
                )),
                None => Ok(()),
            }
        };
        let mut names = BTreeSet::new();
        for p in &self.params {
            let at = format!("launcher `{}` param `{}`", self.name, p.name);
            if !names.insert(&p.name) {
                return Err(format!("{at}: duplicate name"));
            }
            if let Some(reason) = reserved_name(&p.name) {
                return Err(format!("{at}: {reason}"));
            }
            if let Some(count) = &p.list {
                if !matches!(p.kind, Kind::In | Kind::Inout | Kind::Out | Kind::Scalar)
                    || p.contract_only
                {
                    return Err(format!(
                        "{at}: only buffers and passed scalars can be lists"
                    ));
                }
                match self.params.iter().find(|q| q.name == *count) {
                    Some(q)
                        if q.kind == Kind::Scalar
                            && q.list.is_none()
                            && q.scalar.is_some_and(Scalar::is_integer) => {}
                    _ => {
                        return Err(format!(
                            "{at}: the list count `{count}` is not an integer scalar"
                        ));
                    }
                }
                if p.kind == Kind::Scalar && !matches!(p.scalar, Some(Scalar::U32 | Scalar::Usize))
                {
                    return Err(format!("{at}: list scalars are u32 or usize"));
                }
            }
            if !matches!(p.kind, Kind::Indices | Kind::IndicesOut)
                && (p.padded
                    || p.signed
                    || p.mask.is_some()
                    || p.derived_from.is_some()
                    || p.tag.is_some())
            {
                return Err(format!(
                    "{at}: `padded`, `signed`, `mask`, `derived_from` and `tag` belong to indices"
                ));
            }
            if p.kind != Kind::Quant && (p.same_format.is_some() || p.tail.is_some()) {
                return Err(format!("{at}: `same_format` and `tail` belong to quant"));
            }
            if (p.name == "stream") != (p.kind == Kind::Stream) {
                return Err(format!(
                    "{at}: only the `stream` kind is named `stream` (appended when absent)"
                ));
            }
            if p.host_visible
                && (!matches!(p.kind, Kind::In | Kind::Inout | Kind::Out)
                    || p.list.is_some()
                    || p.optional)
            {
                return Err(format!(
                    "{at}: `host_visible` takes a single buffer that is not optional"
                ));
            }
            match p.kind {
                Kind::Scalar => {
                    if p.scalar.is_none()
                        || p.elem.is_some()
                        || p.len.is_some()
                        || p.optional
                        || p.below.is_some()
                    {
                        return Err(format!("{at}: a scalar takes `type` only"));
                    }
                }
                Kind::Indices | Kind::IndicesOut => {
                    let (None, Some(len), None, Some(below), false, false) = (
                        p.elem,
                        &p.len,
                        p.scalar,
                        &p.below,
                        p.optional,
                        p.contract_only,
                    ) else {
                        return Err(format!(
                            "{at}: indices take `len` and `below` (always u32, never optional)"
                        ));
                    };
                    let len = Expr::parse(len).map_err(|e| format!("{at}: {e}"))?;
                    integer_names(&at, "len", len.idents())?;
                    whole(&at, "len", &len)?;
                    let below = Expr::parse(below).map_err(|e| format!("{at}: {e}"))?;
                    integer_names(&at, "below", below.idents())?;
                    whole(&at, "below", &below)?;
                    if p.mask == Some(0) {
                        return Err(format!("{at}: a mask needs at least one bit"));
                    }
                    if let Some(source) = &p.derived_from
                        && !self
                            .params
                            .iter()
                            .any(|q| q.name == *source && q.kind == Kind::Indices)
                    {
                        return Err(format!(
                            "{at}: `derived_from` must name an `indices` parameter"
                        ));
                    }
                    match (&p.derived_from, &p.tag) {
                        (Some(_), Some(tag)) => {
                            let tag = Expr::parse(tag).map_err(|e| format!("{at}: {e}"))?;
                            integer_names(&at, "tag", tag.idents())?;
                            whole(&at, "tag", &tag)?;
                        }
                        (None, Some(_)) => {
                            return Err(format!("{at}: a `tag` needs `derived_from`"));
                        }
                        _ => {}
                    }
                }
                Kind::Quant => {
                    let (Some(rows), Some(cols), None, None, false) =
                        (&p.rows, &p.cols, p.elem, &p.len, p.optional)
                    else {
                        return Err(format!(
                            "{at}: quantized weights take `rows` and `cols` (never optional)"
                        ));
                    };
                    let tail = p.tail.as_deref().unwrap_or("0");
                    for (what, text) in [("rows", rows), ("cols", cols), ("tail", &tail.into())] {
                        let expr = Expr::parse(text).map_err(|e| format!("{at}: {e}"))?;
                        integer_names(&at, what, expr.idents())?;
                        whole(&at, what, &expr)?;
                    }
                    if let Some(other) = &p.same_format
                        && !self
                            .params
                            .iter()
                            .any(|q| q.name == *other && q.name != p.name && q.kind == Kind::Quant)
                    {
                        return Err(format!(
                            "{at}: `same_format` must name another quant parameter"
                        ));
                    }
                    if p.formats.is_empty() {
                        return Err(format!("{at}: list the supported `formats`"));
                    }
                    for name in &p.formats {
                        if gufo_formats::GgmlType::from_name(name).is_none() {
                            return Err(format!("{at}: `{name}` is not a GGML format"));
                        }
                    }
                }
                Kind::Format => {
                    let of =
                        p.of.as_deref()
                            .ok_or(format!("{at}: a format names `of`"))?;
                    if !self
                        .params
                        .iter()
                        .any(|q| q.name == of && q.kind == Kind::Quant)
                    {
                        return Err(format!("{at}: `{of}` is not a quant parameter"));
                    }
                }
                Kind::Stream => {
                    if p.elem.is_some() || p.len.is_some() || p.optional || p.contract_only {
                        return Err(format!("{at}: a `stream` takes no other fields"));
                    }
                }
                Kind::Const => {
                    if p.cpp_type.is_none() || p.value.is_none() {
                        return Err(format!("{at}: a const takes `cpp_type` and `value`"));
                    }
                }
                Kind::Table => {
                    if p.cpp_type.is_none()
                        || p.elem.is_none()
                        || !p.width.is_some_and(|w| w >= 1)
                        || p.len.is_some()
                        || !p.optional
                        || p.contract_only
                    {
                        return Err(format!(
                            "{at}: a table takes `cpp_type`, `elem` and a positive `width`, and is optional"
                        ));
                    }
                }
                Kind::Alias => {
                    let of =
                        p.of.as_deref()
                            .ok_or(format!("{at}: an alias names `of`"))?;
                    let target = self.params.iter().find(|q| q.name == of);
                    let (Some(target), Some(_)) = (target, p.elem) else {
                        return Err(format!(
                            "{at}: `{of}` is not a parameter, or `elem` is missing"
                        ));
                    };
                    if !matches!(target.kind, Kind::Inout | Kind::Out)
                        || target.optional
                        || target.list.is_some()
                        || target.elem != p.elem
                    {
                        return Err(format!(
                            "{at}: `{of}` must be a non-optional written buffer of the same element type"
                        ));
                    }
                    if p.len.is_some() || p.optional {
                        return Err(format!(
                            "{at}: an alias takes `of`, `elem` and `read_only` only"
                        ));
                    }
                }
                Kind::In | Kind::Inout | Kind::Out => {
                    let (Some(_), Some(len), None, None, false) =
                        (p.elem, &p.len, p.scalar, &p.below, p.contract_only)
                    else {
                        return Err(format!("{at}: a buffer takes `elem` and `len`"));
                    };
                    let len = Expr::parse(len).map_err(|e| format!("{at}: {e}"))?;
                    integer_names(&at, "len", len.idents())?;
                    if p.list.is_none() {
                        whole(&at, "len", &len)?;
                    }
                }
            }
        }
        for requirement in &self.requires {
            let at = format!("launcher `{}` requires `{requirement}`", self.name);
            let relation = Relation::parse(requirement).map_err(|e| format!("{at}: {e}"))?;
            integer_names(&at, "a requirement", relation.idents())?;
        }
        self.probes().map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixture with one probe that sets every declared scalar.
    fn manifest(params: &str) -> Result<Manifest, String> {
        let mut values = Vec::new();
        for part in params
            .split('{')
            .skip(1)
            .filter(|p| p.contains(r#"kind = "scalar""#))
        {
            let name = part
                .split(r#"name = ""#)
                .nth(1)
                .and_then(|r| r.split('"').next())
                .unwrap();
            let value = if part.contains(r#"type = "f32""#) {
                "1.0"
            } else {
                "1"
            };
            let entry = format!("{name} = {value}");
            if !values.contains(&entry) {
                values.push(entry);
            }
        }
        manifest_with_probes(params, &format!("{{ {} }}", values.join(", ")))
    }

    fn manifest_with_probes(params: &str, probes: &str) -> Result<Manifest, String> {
        Manifest::parse(&format!(
            r#"
            [library]
            name = "demo"
            root = "."
            namespace = "demo"
            headers = []
            [[launcher]]
            name = "op"
            cpp = "LaunchOp"
            returns = "void"
            params = [{params}]
            probes = [{probes}]
            "#
        ))
    }

    #[test]
    fn accepts_buffers_sized_by_integer_scalars() {
        let m = manifest(
            r#"{ name = "x", kind = "in", elem = "f32", len = "rows * (cols + 1)" },
               { name = "rows", kind = "scalar", type = "usize" },
               { name = "cols", kind = "scalar", type = "u32" }"#,
        );
        assert!(m.is_ok(), "{:?}", m.err());
    }

    #[test]
    fn rejects_unknown_float_and_duplicate_names() {
        let unknown = manifest(r#"{ name = "x", kind = "in", elem = "f32", len = "cnt" }"#);
        assert!(
            unknown
                .err()
                .unwrap()
                .contains("`cnt` in len is not a scalar param")
        );
        let float = manifest(
            r#"{ name = "x", kind = "in", elem = "f32", len = "eps" },
               { name = "eps", kind = "scalar", type = "f32" }"#,
        );
        assert!(float.err().unwrap().contains("not an integer"));
        let duplicate = manifest(
            r#"{ name = "n", kind = "scalar", type = "usize" },
               { name = "n", kind = "scalar", type = "usize" }"#,
        );
        assert!(duplicate.err().unwrap().contains("duplicate name"));
    }

    const PROBED: &str = r#"{ name = "x", kind = "out", elem = "f32", len = "n" },
        { name = "cache", kind = "inout", elem = "f32", len = "n", optional = true },
        { name = "slot", kind = "in", elem = "u32", len = "1", optional = true },
        { name = "n", kind = "scalar", type = "u32" },
        { name = "eps", kind = "scalar", type = "f32" }"#;

    #[test]
    fn probes_become_typed_literals() {
        let m = manifest_with_probes(PROBED, r#"{ n = 7, eps = 1e-6, with = ["cache"] }"#).unwrap();
        let probe = m.launchers[0].probes().unwrap().remove(0);
        assert_eq!(probe.literals["n"], "7u32");
        assert_eq!(probe.literals["eps"], "1e-6_f32");
        assert!(probe.with.contains("cache"));
        assert!(probe.fills.is_empty());
        let m = manifest_with_probes(
            PROBED,
            r#"{ n = 7, eps = 1e-6, slot = 15, with = ["slot"] }"#,
        )
        .unwrap();
        assert_eq!(m.launchers[0].probes().unwrap()[0].fills["slot"], 15);
    }

    #[test]
    fn rejects_missing_incomplete_and_mistyped_probes() {
        let error = |probes: &str| manifest_with_probes(PROBED, probes).err().unwrap();
        assert!(error("").contains("declare at least one probe"));
        assert!(error("{ n = 7 }").contains("missing scalar `eps`"));
        assert!(error("{ n = -1, eps = 1.0 }").contains("`n` is not a valid u32"));
        assert!(error("{ n = 1, eps = 1.0, m = 2 }").contains("`m` is not a scalar parameter"));
        assert!(
            error(r#"{ n = 1, eps = 1.0, with = ["x"] }"#)
                .contains("`x` in `with` is not an optional buffer")
        );
        assert!(
            error("{ n = 1, eps = 1.0, slot = 3 }").contains("`slot` is filled but not passed")
        );
        assert!(
            error(r#"{ n = 1, eps = 1.0, slot = -3, with = ["slot"] }"#)
                .contains("`slot` must be a u32 fill value")
        );
    }

    #[test]
    fn host_visible_takes_one_plain_buffer() {
        assert!(
            manifest(
                r#"{ name = "x", kind = "out", elem = "bytes", len = "8", host_visible = true }"#
            )
            .is_ok()
        );
        let optional = manifest(
            r#"{ name = "x", kind = "out", elem = "bytes", len = "8", optional = true, host_visible = true }"#,
        );
        assert!(
            optional
                .err()
                .unwrap()
                .contains("`host_visible` takes a single buffer")
        );
        let scalar =
            manifest(r#"{ name = "n", kind = "scalar", type = "u32", host_visible = true }"#);
        assert!(
            scalar
                .err()
                .unwrap()
                .contains("`host_visible` takes a single buffer")
        );
    }

    #[test]
    fn rejects_malformed_parameters() {
        let scalar_len = manifest(r#"{ name = "n", kind = "scalar", type = "usize", len = "1" }"#);
        assert!(
            scalar_len
                .err()
                .unwrap()
                .contains("a scalar takes `type` only")
        );
        let no_len = manifest(r#"{ name = "x", kind = "out", elem = "f32" }"#);
        assert!(
            no_len
                .err()
                .unwrap()
                .contains("a buffer takes `elem` and `len`")
        );
        let stream = manifest(r#"{ name = "stream", kind = "scalar", type = "usize" }"#);
        assert!(stream.err().unwrap().contains("only the `stream` kind"));
        let unknown_key =
            manifest(r#"{ name = "x", kind = "in", elem = "f32", len = "1", size = 3 }"#);
        assert!(unknown_key.is_err());
    }

    #[test]
    fn tables_are_optional_typed_headers() {
        const TABLE: &str = r#"{ name = "rope", kind = "table", elem = "i32", width = 3, cpp_type = "const R*", optional = true }"#;
        let m = manifest(TABLE).unwrap();
        assert_eq!(m.launchers[0].params[0].width, Some(3));
        for bad in [
            TABLE.replace(", optional = true", ""),
            TABLE.replace("width = 3", "width = 0"),
            TABLE.replace(r#", cpp_type = "const R*""#, ""),
            TABLE.replace(r#"elem = "i32", "#, ""),
        ] {
            assert!(
                manifest(&bad).err().unwrap().contains("a table takes"),
                "{bad}"
            );
        }
        // Probes pass no table.
        let with = manifest_with_probes(TABLE, r#"{ with = ["rope"] }"#);
        assert!(with.err().unwrap().contains("not an optional buffer"));
    }

    #[test]
    fn launchers_outside_the_namespace_are_fully_qualified() {
        let text = |cpp: &str| {
            format!(
                r#"
                [library]
                name = "demo"
                root = "."
                namespace = "demo"
                headers = []
                [[launcher]]
                name = "op"
                cpp = "{cpp}"
                returns = "void"
                params = [{{ name = "n", kind = "scalar", type = "u32" }}]
                probes = [{{ n = 1 }}]
                "#
            )
        };
        assert!(Manifest::parse(&text("::other::ns::LaunchOp")).is_ok());
        assert!(Manifest::parse(&text("::other::1op")).is_err());
        let shim = crate::codegen::cpp_shim(&Manifest::parse(&text("::other::LaunchOp")).unwrap());
        assert!(shim.contains("&::other::LaunchOp"));
        assert!(!shim.contains("demo::::"));
    }
}
