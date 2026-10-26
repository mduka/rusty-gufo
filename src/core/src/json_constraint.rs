//! Constrained JSON output (`src/core/json_constraint.cpp`): JSON Schemas
//! compiled into a byte grammar, advanced byte by byte over pushdown
//! stacks, and the token masks a sampler applies before sampling.
//!
//! The grammar works on bytes, so token boundaries may split UTF-8
//! characters and JSON escapes. Compiled grammars and vocabulary tries are
//! immutable and shared; a [`State`] belongs to one request and is copied
//! along with its sampler during speculative verification.

mod compiler;
mod lexeme;
mod regex;
mod unicode_set;

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use crate::json::Value;
use crate::text::{Error, Result};

use compiler::Compiler;
use lexeme::{JsonSchemaLexeme, LexemeState, ReasoningEnd, dump};

const TERMINAL: u32 = 1 << 31;
const LEXEME: u32 = 1 << 30;
/// A wide native argument set (`kArguments`): its members in any order,
/// each at most once, required ones before it completes.
const ARGUMENTS: u32 = 1 << 29;
const LEAF: u32 = TERMINAL | LEXEME;
const MAX_SCHEMA_BYTES: usize = 2 * 1024 * 1024;
const MAX_RULES: usize = 262_144;
const MAX_DEPTH: usize = 16;
const MAX_STATES: usize = 8192;
const MAX_STACK: usize = 16384;
const MAX_WORK: usize = 2_000_000;
/// Bounded grammar and mask caches keep this many entries.
const CACHE_ENTRIES: usize = 16;

/// Why a schema does not compile. `Empty` and `Invalid` are the C++
/// `std::invalid_argument`s (`JsonSchemaEmpty` and the rest); compilation
/// catches `Empty` where an alternative can still match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    /// The schema admits no finite value (`JsonSchemaEmpty`).
    Empty(String),
    /// The schema is malformed or outside the supported subset.
    Invalid(String),
    /// A grammar state or work limit (`std::runtime_error`).
    Runtime(String),
}

impl SchemaError {
    /// `Invalid`: `JSON Schema: ` and the reason.
    fn invalid(reason: &str) -> Self {
        Self::Invalid(format!("JSON Schema: {reason}"))
    }

    fn empty(message: &str) -> Self {
        Self::Empty(message.to_owned())
    }

    fn runtime(message: &str) -> Self {
        Self::Runtime(message.to_owned())
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Empty(message) | Self::Invalid(message) | Self::Runtime(message) => message,
        }
    }
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for SchemaError {}

impl From<SchemaError> for Error {
    fn from(error: SchemaError) -> Self {
        match error {
            SchemaError::Empty(message) | SchemaError::Invalid(message) => Error::invalid(message),
            SchemaError::Runtime(message) => Error::runtime(message),
        }
    }
}

type Sequence = Vec<u32>;
type Rule = Vec<Sequence>;

/// A set of bytes (`std::bitset<256>`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ByteClass([u64; 4]);

impl ByteClass {
    fn set(&mut self, byte: u8) {
        self.0[usize::from(byte >> 6)] |= 1 << (byte & 63);
    }

    fn test(&self, byte: u8) -> bool {
        self.0[usize::from(byte >> 6)] & (1 << (byte & 63)) != 0
    }

    fn none(&self) -> bool {
        self.0 == [0; 4]
    }
}

/// One pushdown stack: the symbols still to match, innermost last, and the
/// partial primitive a lexeme on top has consumed.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Stack {
    symbols: Vec<u32>,
    lexeme: LexemeState,
    /// The members a wide native argument set on this stack has taken, one
    /// bit each. Sampler copies and speculative
    /// rollback copy it with the stack.
    used_parameters: Vec<u64>,
}

/// The members of a wide native argument set and the required ones, one bit
/// per member (`ArgumentSet`). Only outer native parameter lists use sets;
/// their values stay ordinary rules, so sets never nest within a stack.
#[derive(Debug, Clone, Default)]
pub(crate) struct ArgumentSet {
    pub(crate) members: Vec<u32>,
    pub(crate) required: Vec<u64>,
}

impl ArgumentSet {
    /// Whether the set may stay empty: every member starts with a nonempty
    /// native parameter tag.
    pub(crate) fn nullable(&self) -> bool {
        self.required.iter().all(|&word| word == 0)
    }

    pub(crate) fn is_required(&self, member: usize) -> bool {
        self.required[member / 64] & (1 << (member % 64)) != 0
    }
}

/// The grammar states a request's output can be in: sorted, distinct stacks.
/// Empty when the output left the grammar.
pub type State = Vec<Stack>;

/// A tool's name and the grammar of its arguments.
pub type Tool = (String, Arc<JsonConstraint>);

/// How tool calls are spelled (`JsonConstraint::ToolFormat`): the injected
/// JSON envelope, or the native tags a model's chat template teaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ToolFormat {
    /// `<tool_call>{"name":…,"arguments":…}</tool_call>`.
    Json,
    /// `<tool_call>\n<function=…>\n<parameter=…>\n…\n</parameter>\n…`.
    Qwen,
    /// DeepSeek's DSML `<｜DSML｜tool_calls>` block.
    DeepSeek,
}

/// A compiled byte grammar (`JsonConstraint`).
#[derive(Debug, Clone)]
pub struct JsonConstraint {
    rules: Vec<Rule>,
    classes: Vec<ByteClass>,
    lexemes: Vec<Arc<JsonSchemaLexeme>>,
    argument_sets: Vec<ArgumentSet>,
    root: u32,
    prompt: String,
    stop_only_when_complete: bool,
    /// Calls the model may make or not: a natural stop (or an empty piece)
    /// stays admissible wherever the plain answer could end.
    automatic_tools: bool,
    /// A native call header that ends reasoning without `</think>`
    ///, and the rule its tool name continues with.
    reasoning_tool_prefix: String,
    reasoning_tool_root: u32,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The address a cache keys a shared grammar by.
fn address(grammar: &Arc<JsonConstraint>) -> usize {
    Arc::as_ptr(grammar).addr()
}

/// The transitions of the automaton that finds `pattern` in arbitrary
/// bytes: for each matched prefix, the bytes leading to each longest
/// matched prefix.
fn prefix_transitions(pattern: &[u8], prefix: usize) -> Vec<ByteClass> {
    let mut transitions = vec![ByteClass::default(); pattern.len() + 1];
    for byte in 0..=255u8 {
        let mut candidate = pattern[..prefix].to_vec();
        candidate.push(byte);
        let mut matched = candidate.len().min(pattern.len());
        while matched != 0 && !candidate.ends_with(&pattern[..matched]) {
            matched -= 1;
        }
        transitions[matched].set(byte);
    }
    transitions
}

impl JsonConstraint {
    fn empty() -> Self {
        Self {
            rules: Vec::new(),
            classes: Vec::new(),
            lexemes: Vec::new(),
            argument_sets: Vec::new(),
            root: 0,
            prompt: String::new(),
            stop_only_when_complete: true,
            automatic_tools: false,
            reasoning_tool_prefix: String::new(),
            reasoning_tool_root: 0,
        }
    }

    /// Whether only a complete answer may stop, as for a closed JSON output
    /// (`stop_only_when_complete_`).
    pub(crate) fn stop_only_when_complete(&self) -> bool {
        self.stop_only_when_complete
    }

    /// Resolves a local JSON pointer as schema compilation does
    /// (`ResolveReference`).
    pub fn resolve_reference<'v>(
        root: &'v Value,
        reference: &Value,
    ) -> std::result::Result<&'v Value, SchemaError> {
        compiler::resolve_reference(root, reference)
    }

    /// The grammar of a tool's arguments in `format` (`ToolParameters`), or
    /// `None` without one (a strict JSON tool that fails to compile has
    /// returned its error). Native models always keep native parameter
    /// tags: unsupported schemas take best-effort arguments, without
    /// relaxing response-format schemas. Tool choice changes the call
    /// envelope, not this cached argument grammar.
    pub fn tool_parameters(
        schema: &Value,
        strict: bool,
        format: ToolFormat,
    ) -> std::result::Result<Option<Arc<Self>>, SchemaError> {
        type Key = (String, bool, ToolFormat);
        static CACHE: Mutex<BTreeMap<Key, Option<Arc<JsonConstraint>>>> =
            Mutex::new(BTreeMap::new());
        let key: Key = (dump(schema)?, strict, format);
        if key.0.len() > MAX_SCHEMA_BYTES {
            return Err(SchemaError::invalid("maximum schema size is 2 MiB"));
        }
        if let Some(found) = lock(&CACHE).get(&key) {
            return Ok(found.clone());
        }
        let grammar = Self::build_tool_parameters(schema, strict, format)?;
        let mut cache = lock(&CACHE);
        if cache.len() >= CACHE_ENTRIES {
            cache.pop_first();
        }
        Ok(cache.entry(key).or_insert(grammar).clone())
    }

    fn build_tool_parameters(
        schema: &Value,
        strict: bool,
        format: ToolFormat,
    ) -> std::result::Result<Option<Arc<Self>>, SchemaError> {
        let catchable =
            |error: &SchemaError| matches!(error, SchemaError::Empty(_) | SchemaError::Invalid(_));
        let properties = schema.find("properties");
        let additional = schema.find("additionalProperties");
        let untyped = !strict
            && properties.is_none_or(Value::is_empty)
            && additional.is_none_or(Value::is_bool);
        let mut normalized = schema.clone();
        // A root reference already supplies the object schema: an empty,
        // closed sibling object would intersect away its arguments.
        if !strict && !normalized.contains("$ref") {
            if !normalized.contains("type") {
                normalized.set("type", "object");
            }
            // Finite object values supply their own fields.
            if !normalized.contains("const") && !normalized.contains("enum") {
                if !normalized.contains("properties") {
                    normalized.set("properties", Value::object());
                }
                if !normalized.contains("additionalProperties") {
                    normalized.set("additionalProperties", false);
                }
            }
        }
        let mut best_effort = false;
        // As in llama.cpp, a required call uses the same argument grammar as
        // an optional one; tool_choice only decides whether a call must
        // happen. Validation only on native models; JSON models keep it.
        let mut grammar = match Compiler::new(&normalized, strict, true)
            .and_then(|compiler| compiler.compile(false, false))
        {
            Ok(grammar) => Some(Arc::new(grammar)),
            Err(error) if strict || !catchable(&error) => return Err(error),
            Err(_) => {
                best_effort = true;
                // Unsupported non-strict keywords are guidance, but must not
                // discard supported nested requirements, bounds or finite
                // values.
                match Compiler::new(&normalized, strict, true)
                    .and_then(|compiler| compiler.compile(false, true))
                {
                    Ok(grammar) => Some(Arc::new(grammar)),
                    Err(error) if !catchable(&error) => return Err(error),
                    Err(_) => None,
                }
            }
        };
        let preserve_root = grammar.is_some()
            && ["const", "enum", "anyOf", "$ref"]
                .iter()
                .any(|key| schema.contains(key));
        let open = additional.is_some_and(|additional| additional.as_bool(false));
        // Both routes take the same untyped decision: a JSON-only neighbour
        // must not turn a best-effort tool into an empty-arguments grammar.
        let open_untyped = grammar.is_some()
            && untyped
            && !preserve_root
            && !compiler::closed_properties(schema, best_effort)
            && (best_effort || open)
            && !(format == ToolFormat::Qwen && compiler::has_property_schemas(schema));
        if open_untyped {
            grammar = Some(Self::open_tool_parameters(format)?);
        } else if grammar.is_some() && format != ToolFormat::Json {
            match Compiler::new(&normalized, strict, true)
                .and_then(|compiler| compiler.tool_parameters(format, best_effort))
            {
                Ok(native) => grammar = native.map(Arc::new),
                Err(error) if strict || !catchable(&error) => return Err(error),
                Err(_) => {
                    grammar = None;
                    best_effort = true;
                }
            }
        }
        if grammar.is_none() && format == ToolFormat::Json && !strict {
            grammar = Some(Self::object()?);
        }
        if grammar.is_none() && best_effort && open_untyped {
            grammar = Some(Self::open_tool_parameters(format)?);
        }
        // Never switch a native model to a JSON envelope: one tool's schema
        // would change every call's syntax and contradict the chat template
        // and history. Native tags enforce what they can carry, as in
        // llama.cpp.
        if grammar.is_none() && format != ToolFormat::Json {
            grammar = Some(
                match Compiler::new(&normalized, strict, true)
                    .and_then(|compiler| compiler.loose_tool_parameters(format))
                {
                    Ok(grammar) => Arc::new(grammar),
                    Err(error) if !catchable(&error) => return Err(error),
                    Err(_) => Self::open_tool_parameters(format)?,
                },
            );
        }
        Ok(grammar)
    }

    /// Native parameters of any name and value for a non-strict tool
    /// without declared argument types (`OpenToolParameters`); any object in
    /// JSON.
    pub fn open_tool_parameters(format: ToolFormat) -> std::result::Result<Arc<Self>, SchemaError> {
        type Cached = OnceLock<std::result::Result<Arc<JsonConstraint>, SchemaError>>;
        static QWEN: Cached = OnceLock::new();
        static DEEPSEEK: Cached = OnceLock::new();
        let cell = match format {
            ToolFormat::Json => return Self::object(),
            ToolFormat::Qwen => &QWEN,
            ToolFormat::DeepSeek => &DEEPSEEK,
        };
        cell.get_or_init(|| {
            Ok(Arc::new(
                Compiler::new(&Value::Null, false, false)?.open_tool_parameters(format)?,
            ))
        })
        .clone()
    }

    /// Compiles `schema` (`Compile`); `strict` requires every property.
    /// The last 16 schemas stay cached, so an identical schema returns the
    /// same grammar.
    pub fn compile(schema: &Value, strict: bool) -> std::result::Result<Arc<Self>, SchemaError> {
        Self::compile_cached(schema, strict, false)
    }

    /// [`JsonConstraint::compile`] for a response format, whose objects follow
    /// llama.cpp: an object admits extra keys after its declared properties
    /// when `additionalProperties` is `true` or a schema, or when it declares
    /// no properties; omitting `additionalProperties` beside `properties`
    /// still closes it. Tool parameters keep closed objects.
    pub fn compile_response(
        schema: &Value,
        strict: bool,
    ) -> std::result::Result<Arc<Self>, SchemaError> {
        Self::compile_cached(schema, strict, true)
    }

    fn compile_cached(
        schema: &Value,
        strict: bool,
        open: bool,
    ) -> std::result::Result<Arc<Self>, SchemaError> {
        let key = format!(
            "{}{}{}",
            if strict { "strict:" } else { "schema:" },
            if open { "open:" } else { "" },
            dump(schema)?
        );
        if key.len() > MAX_SCHEMA_BYTES {
            return Err(SchemaError::invalid("maximum schema size is 2 MiB"));
        }
        // Compile outside the lock so unrelated requests proceed.
        static CACHE: Mutex<BTreeMap<String, Arc<JsonConstraint>>> = Mutex::new(BTreeMap::new());
        if let Some(found) = lock(&CACHE).get(&key) {
            return Ok(found.clone());
        }
        let mut compiler = Compiler::new(schema, strict, false)?;
        if open {
            compiler = compiler.with_open_response_objects();
        }
        let grammar = Arc::new(compiler.compile(false, false)?);
        let mut cache = lock(&CACHE);
        if cache.len() >= CACHE_ENTRIES {
            cache.pop_first();
        }
        Ok(cache.entry(key).or_insert(grammar).clone())
    }

    /// Any single JSON object, nested at most 16 levels (`Object`).
    pub fn object() -> std::result::Result<Arc<Self>, SchemaError> {
        static OBJECT: OnceLock<std::result::Result<Arc<JsonConstraint>, SchemaError>> =
            OnceLock::new();
        OBJECT
            .get_or_init(|| {
                Ok(Arc::new(
                    Compiler::new(&Value::Null, false, false)?.compile(true, false)?,
                ))
            })
            .clone()
    }

    /// Unconstrained reasoning until `</think>`, then `answer`
    /// (`WithReasoning`). The phase lives in the grammar stack, so it
    /// follows sampler copies, draft rejection and request restarts.
    pub fn with_reasoning(answer: Arc<Self>) -> Arc<Self> {
        type Cache = BTreeMap<usize, (Arc<JsonConstraint>, Arc<JsonConstraint>)>;
        static CACHE: Mutex<Cache> = Mutex::new(BTreeMap::new());
        let key = address(&answer);
        if let Some((_, found)) = lock(&CACHE).get(&key) {
            return found.clone();
        }
        let mut grammar = (*answer).clone();
        if !answer.reasoning_tool_prefix.is_empty() {
            // A native call header may also end the reasoning: a lexeme
            // keeps quoted and fenced mentions of either boundary text.
            let lexeme = |grammar: &mut Self, end: ReasoningEnd| {
                let id = LEXEME | grammar.lexemes.len() as u32;
                grammar.lexemes.push(JsonSchemaLexeme::reasoning(
                    &answer.reasoning_tool_prefix,
                    end,
                ));
                id
            };
            let explicit = lexeme(&mut grammar, ReasoningEnd::Explicit);
            let tool = lexeme(&mut grammar, ReasoningEnd::Tool);
            let mut roots: Rule = vec![
                vec![explicit, answer.root],
                vec![tool, answer.reasoning_tool_root],
            ];
            if grammar.automatic_tools {
                roots.push(Vec::new());
                roots.push(vec![lexeme(&mut grammar, ReasoningEnd::Unfinished)]);
            }
            grammar.root = grammar.rules.len() as u32;
            grammar.rules.push(roots);
            let mut cache = lock(&CACHE);
            if let Some((_, found)) = cache.get(&key) {
                return found.clone();
            }
            if cache.len() >= CACHE_ENTRIES {
                cache.pop_first();
            }
            let grammar = Arc::new(grammar);
            cache.insert(key, (answer, grammar.clone()));
            return grammar;
        }
        const END: &[u8] = b"</think>";
        let base = grammar.rules.len() as u32;
        grammar
            .rules
            .resize(grammar.rules.len() + END.len(), Vec::new());
        for prefix in 0..END.len() {
            // Optional calls must not turn a natural stop after the
            // reasoning into a mandatory call; the request's output cap
            // stays independent too.
            if grammar.automatic_tools {
                grammar.rules[base as usize + prefix].push(Vec::new());
            }
            for (matched, transitions) in prefix_transitions(END, prefix).into_iter().enumerate() {
                if transitions.none() {
                    continue;
                }
                let terminal = TERMINAL | grammar.classes.len() as u32;
                grammar.classes.push(transitions);
                let next = if matched == END.len() {
                    answer.root
                } else {
                    base + matched as u32
                };
                grammar.rules[base as usize + prefix].push(vec![terminal, next]);
            }
        }
        grammar.root = base;
        let mut cache = lock(&CACHE);
        if let Some((_, found)) = cache.get(&key) {
            return found.clone();
        }
        if cache.len() >= CACHE_ENTRIES {
            cache.pop_first();
        }
        let grammar = Arc::new(grammar);
        cache.insert(key, (answer, grammar.clone()));
        grammar
    }

    /// Adds calls of `tools` in `format` to `answer` (`WithTools`): the
    /// JSON envelope `<tool_call>{"name":…,"arguments":…}</tool_call>`, or
    /// the model's native tags. Without an answer, ordinary text stays
    /// unconstrained until a call marker; `required` demands a call,
    /// `parallel` allows several. No tools leave `answer` unchanged.
    pub fn with_tools(
        answer: Option<Arc<Self>>,
        tools: Vec<Tool>,
        required: bool,
        parallel: bool,
        format: ToolFormat,
    ) -> std::result::Result<Option<Arc<Self>>, SchemaError> {
        if tools.is_empty() {
            return Ok(answer);
        }
        type Key = (usize, Vec<(String, usize)>, bool, bool, ToolFormat);
        type Entry = (Option<Arc<JsonConstraint>>, Vec<Tool>, Arc<JsonConstraint>);
        static CACHE: Mutex<BTreeMap<Key, Entry>> = Mutex::new(BTreeMap::new());
        let key: Key = (
            answer.as_ref().map_or(0, address),
            tools
                .iter()
                .map(|(name, grammar)| (name.clone(), address(grammar)))
                .collect(),
            required,
            parallel,
            format,
        );
        if let Some((_, _, found)) = lock(&CACHE).get(&key) {
            return Ok(Some(found.clone()));
        }
        let grammar = Self::build_tools(answer.as_ref(), &tools, required, parallel, format)?;
        let mut cache = lock(&CACHE);
        if let Some((_, _, found)) = cache.get(&key) {
            return Ok(Some(found.clone()));
        }
        if cache.len() >= CACHE_ENTRIES {
            cache.pop_first();
        }
        let grammar = Arc::new(grammar);
        cache.insert(key, (answer, tools, grammar.clone()));
        Ok(Some(grammar))
    }

    fn build_tools(
        answer: Option<&Arc<Self>>,
        tools: &[Tool],
        required: bool,
        parallel: bool,
        format: ToolFormat,
    ) -> std::result::Result<Self, SchemaError> {
        let deepseek = format == ToolFormat::DeepSeek;
        let marker: &[u8] = if deepseek {
            "<｜DSML｜tool_calls>".as_bytes()
        } else {
            b"<tool_call>"
        };
        let plain_answer = answer.is_none();
        let mut grammar = match answer {
            Some(answer) => (**answer).clone(),
            None => (*Self::object()?).clone(),
        };
        grammar.stop_only_when_complete = !plain_answer && !parallel;
        grammar.automatic_tools = plain_answer && !required;
        let mut alternatives: Rule = Vec::new();
        if let Some(answer) = answer
            && !required
        {
            alternatives.push(vec![answer.root]);
        }
        let end = grammar.tool_literal(match format {
            ToolFormat::Json => b"}</tool_call>".as_slice(),
            ToolFormat::DeepSeek => "</｜DSML｜invoke>".as_bytes(),
            ToolFormat::Qwen => b"</function>\n</tool_call>".as_slice(),
        })?;
        let calls = grammar.rules.len() as u32;
        grammar.rules.push(Vec::new());
        // In tool-only mode ordinary text remains unconstrained until a
        // canonical call marker. The small prefix automaton is carried by the
        // request grammar, across tokens, speculative rollback and sampler
        // copies.
        let prose = if plain_answer {
            grammar.tool_text(marker, Some(calls))?
        } else {
            u32::MAX
        };
        // As in llama.cpp's DeepSeek V4 and Qwen3-Coder parsers, calls end
        // the output: parallel calls share one DeepSeek block, Qwen calls may
        // follow each other, and no text may follow them.
        let qwen = format == ToolFormat::Qwen;
        let ends_output = deepseek || qwen || !plain_answer;
        let mut after = if ends_output {
            grammar.rules.len() as u32
        } else if parallel {
            prose
        } else {
            grammar.tool_text(marker, None)?
        };
        if ends_output {
            grammar.rules.push(vec![Vec::new()]);
            if qwen {
                // llama.cpp's SPACE_RULE: empty, one space, or one or two
                // newlines followed by up to 20 spaces or tabs
                // (common/json-schema-to-grammar.cpp).
                grammar.check_capacity(23, 1, 0, 0)?;
                let space = grammar.rules.len() as u32;
                grammar.rules.push(vec![Vec::new()]);
                let single_space = grammar.tool_literal(b" ")?;
                let newline = grammar.tool_literal(b"\n")?;
                let double_newline = grammar.tool_literal(b"\n\n")?;
                let mut indentation = ByteClass::default();
                indentation.set(b' ');
                indentation.set(b'\t');
                let indent = TERMINAL | grammar.classes.len() as u32;
                grammar.classes.push(indentation);
                let mut padding = grammar.rules.len() as u32;
                grammar.rules.push(vec![Vec::new()]);
                for _ in 0..20 {
                    let next = grammar.rules.len() as u32;
                    grammar.rules.push(vec![Vec::new(), vec![indent, padding]]);
                    padding = next;
                }
                let alternatives = &mut grammar.rules[space as usize];
                alternatives.push(vec![single_space]);
                alternatives.push(vec![newline, padding]);
                alternatives.push(vec![double_newline, padding]);
                grammar.rules[after as usize] = vec![vec![space]];
                if parallel {
                    let begin = grammar.tool_literal(marker)?;
                    grammar.rules[after as usize].push(vec![space, begin, calls]);
                }
            } else if parallel && !deepseek {
                let begin = grammar.tool_literal(marker)?;
                grammar.rules[after as usize].push(vec![begin, calls]);
            }
        }
        if deepseek {
            let finish = grammar.tool_literal("\n</｜DSML｜tool_calls>".as_bytes())?;
            grammar.check_capacity(1, 0, 0, 0)?;
            let tail = grammar.rules.len() as u32;
            grammar.rules.push(vec![vec![finish, after]]);
            if parallel {
                grammar.rules[tail as usize].push(vec![calls]);
            }
            after = tail;
        }
        let mut imported: HashMap<usize, u32> = HashMap::new();
        if let Some(answer) = answer {
            imported.insert(address(answer), answer.root);
        }
        // Reasoning can end in a native call header: its rule continues with
        // the tool name (`with_reasoning`).
        if format != ToolFormat::Json {
            grammar.reasoning_tool_prefix = if deepseek {
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"".to_owned()
            } else {
                "<tool_call>\n<function=".to_owned()
            };
            grammar.check_capacity(1, 0, 0, 0)?;
            grammar.reasoning_tool_root = grammar.rules.len() as u32;
            grammar.rules.push(Vec::new());
        }
        let named = |grammar: &mut Self, name: &str| {
            grammar.tool_literal(
                format!("{name}{}", if deepseek { "\">\n" } else { ">\n" }).as_bytes(),
            )
        };
        for (name, arguments) in tools {
            let begin = match format {
                ToolFormat::Json => format!(
                    "{{\"name\":{},\"arguments\":",
                    dump(&Value::from(name.as_str()))?
                ),
                ToolFormat::DeepSeek => format!("\n<｜DSML｜invoke name=\"{name}\">\n"),
                ToolFormat::Qwen => format!("\n<function={name}>\n"),
            };
            let begin = grammar.tool_literal(begin.as_bytes())?;
            if let Some(&root) = imported.get(&address(arguments)) {
                grammar.rules[calls as usize].push(vec![begin, root, end, after]);
                if format != ToolFormat::Json {
                    let named = named(&mut grammar, name)?;
                    let reasoning_root = grammar.reasoning_tool_root as usize;
                    grammar.rules[reasoning_root].push(vec![named, root, end, after]);
                }
                continue;
            }
            grammar.check_capacity(
                arguments.rules.len(),
                arguments.classes.len(),
                arguments.lexemes.len(),
                arguments.argument_sets.len(),
            )?;
            let rules = grammar.rules.len() as u32;
            let classes = grammar.classes.len() as u32;
            let lexemes = grammar.lexemes.len() as u32;
            let argument_sets = grammar.argument_sets.len() as u32;
            grammar.classes.extend_from_slice(&arguments.classes);
            grammar.lexemes.extend(arguments.lexemes.iter().cloned());
            for set in &arguments.argument_sets {
                let mut set = set.clone();
                for member in &mut set.members {
                    *member += rules;
                }
                grammar.argument_sets.push(set);
            }
            for rule in &arguments.rules {
                let mut rule = rule.clone();
                for symbol in rule.iter_mut().flatten() {
                    *symbol = if *symbol & TERMINAL != 0 {
                        TERMINAL | ((*symbol & !TERMINAL) + classes)
                    } else if *symbol & LEXEME != 0 {
                        LEXEME | ((*symbol & !LEXEME) + lexemes)
                    } else if *symbol & ARGUMENTS != 0 {
                        ARGUMENTS | ((*symbol & !ARGUMENTS) + argument_sets)
                    } else {
                        *symbol + rules
                    };
                }
                grammar.rules.push(rule);
            }
            let root = arguments.root + rules;
            imported.insert(address(arguments), root);
            grammar.rules[calls as usize].push(vec![begin, root, end, after]);
            if format != ToolFormat::Json {
                let named = named(&mut grammar, name)?;
                let reasoning_root = grammar.reasoning_tool_root as usize;
                grammar.rules[reasoning_root].push(vec![named, root, end, after]);
            }
        }
        if plain_answer && !required {
            alternatives.push(vec![prose]);
        } else if format == ToolFormat::Json {
            let begin = grammar.tool_literal(marker)?;
            alternatives.push(vec![begin, calls]);
        } else {
            // Native calls may follow whitespace the template leaves.
            let mut whitespace = ByteClass::default();
            for &byte in b" \t\r\n" {
                whitespace.set(byte);
            }
            grammar.check_capacity(1, 1, 0, 0)?;
            let terminal = TERMINAL | grammar.classes.len() as u32;
            grammar.classes.push(whitespace);
            let ws = grammar.rules.len() as u32;
            grammar.rules.push(vec![Vec::new(), vec![terminal, ws]]);
            let begin = grammar.tool_literal(marker)?;
            alternatives.push(vec![ws, begin, calls]);
        }
        grammar.check_capacity(1, 0, 0, 0)?;
        grammar.root = grammar.rules.len() as u32;
        grammar.rules.push(alternatives);
        Ok(grammar)
    }

    fn check_capacity(
        &self,
        rules: usize,
        classes: usize,
        lexemes: usize,
        argument_sets: usize,
    ) -> std::result::Result<(), SchemaError> {
        let fits =
            |current: usize, extra: usize| current <= MAX_RULES && extra <= MAX_RULES - current;
        if !fits(self.rules.len(), rules)
            || !fits(self.classes.len(), classes)
            || !fits(self.lexemes.len(), lexemes)
            || !fits(self.argument_sets.len(), argument_sets)
        {
            return Err(SchemaError::invalid(
                "combined tool grammar exceeds its resource budget",
            ));
        }
        Ok(())
    }

    fn tool_literal(&mut self, text: &[u8]) -> std::result::Result<u32, SchemaError> {
        self.check_capacity(1, 0, 0, 0)?;
        let id = self.rules.len() as u32;
        self.rules.push(vec![
            text.iter().map(|&b| TERMINAL | u32::from(b)).collect(),
        ]);
        Ok(id)
    }

    /// Unconstrained text that continues at `target` after `marker`, or
    /// never contains it when `target` is `None`.
    fn tool_text(
        &mut self,
        marker: &[u8],
        target: Option<u32>,
    ) -> std::result::Result<u32, SchemaError> {
        self.check_capacity(marker.len(), marker.len() * 256, 0, 0)?;
        let base = self.rules.len() as u32;
        self.rules
            .resize(self.rules.len() + marker.len(), Vec::new());
        for prefix in 0..marker.len() {
            self.rules[base as usize + prefix].push(Vec::new());
            for (matched, transitions) in prefix_transitions(marker, prefix).into_iter().enumerate()
            {
                if transitions.none() || (matched == marker.len() && target.is_none()) {
                    continue;
                }
                let terminal = TERMINAL | self.classes.len() as u32;
                self.classes.push(transitions);
                let next = match target {
                    Some(target) if matched == marker.len() => target,
                    _ => base + matched as u32,
                };
                self.rules[base as usize + prefix].push(vec![terminal, next]);
            }
        }
        Ok(base)
    }

    /// Expands rule symbols until every stack is empty or has a terminal or
    /// lexeme on top (`Expand`).
    fn expand(&self, mut pending: State) -> Result<State> {
        let mut output = State::new();
        let mut work = 0usize;
        while let Some(mut stack) = pending.pop() {
            work += 1;
            if work > MAX_WORK || pending.len() + 1 + output.len() > MAX_STATES {
                return Err(Error::runtime("JSON grammar state limit exceeded"));
            }
            let Some(&rule) = stack.symbols.last().filter(|&&symbol| symbol & LEAF == 0) else {
                output.push(stack);
                continue;
            };
            stack.symbols.pop();
            if rule & ARGUMENTS != 0 {
                let arguments = &self.argument_sets[(rule & !ARGUMENTS) as usize];
                if stack.used_parameters.is_empty() {
                    stack.used_parameters = vec![0; arguments.required.len()];
                }
                if stack.symbols.len() + 2 > MAX_STACK {
                    return Err(Error::runtime("JSON grammar stack limit exceeded"));
                }
                let complete = arguments
                    .required
                    .iter()
                    .zip(&stack.used_parameters)
                    .all(|(&required, &used)| used & required == required);
                for (i, &member) in arguments.members.iter().enumerate() {
                    let bit = 1u64 << (i % 64);
                    if stack.used_parameters[i / 64] & bit != 0 {
                        continue;
                    }
                    if pending.len() + output.len() >= MAX_STATES {
                        return Err(Error::runtime("JSON grammar state limit exceeded"));
                    }
                    let mut next = stack.clone();
                    next.used_parameters[i / 64] |= bit;
                    next.symbols.push(rule);
                    next.symbols.push(member);
                    pending.push(next);
                }
                if complete {
                    stack.used_parameters.clear();
                    pending.push(stack);
                }
                continue;
            }
            let sequences = &self.rules[rule as usize];
            for (index, sequence) in sequences.iter().enumerate() {
                if stack.symbols.len() + sequence.len() > MAX_STACK {
                    return Err(Error::runtime("JSON grammar stack limit exceeded"));
                }
                if index + 1 == sequences.len() {
                    // The last alternative takes the stack itself.
                    stack.symbols.extend(sequence.iter().rev());
                    pending.push(stack);
                    break;
                }
                let mut symbols = Vec::with_capacity(stack.symbols.len() + sequence.len());
                symbols.extend_from_slice(&stack.symbols);
                symbols.extend(sequence.iter().rev());
                pending.push(Stack {
                    symbols,
                    lexeme: stack.lexeme.clone(),
                    used_parameters: stack.used_parameters.clone(),
                });
            }
        }
        output.sort_unstable();
        output.dedup();
        Ok(output)
    }

    /// The states before any output (`Start`).
    pub fn start(&self) -> Result<State> {
        self.expand(vec![Stack {
            symbols: vec![self.root],
            lexeme: LexemeState::default(),
            used_parameters: Vec::new(),
        }])
    }

    /// The states after `byte` (`Advance`); empty when no stack admits it.
    pub fn advance(&self, state: &State, byte: u8) -> Result<State> {
        let mut next = State::new();
        for stack in state {
            let Some(&symbol) = stack.symbols.last() else {
                continue;
            };
            if symbol & LEXEME != 0 {
                let lexeme = &self.lexemes[(symbol & !LEXEME) as usize];
                if !lexeme.allows_byte(byte) {
                    continue;
                }
                let mut candidate = stack.clone();
                let result = lexeme.advance(&mut candidate.lexeme, byte);
                if result.prefix && result.complete {
                    next.push(candidate.clone());
                } else if result.prefix {
                    next.push(candidate);
                    continue;
                }
                if result.complete {
                    candidate.symbols.pop();
                    candidate.lexeme.clear();
                    next.push(candidate);
                }
            } else if self.classes[(symbol & !TERMINAL) as usize].test(byte) {
                let mut candidate = stack.clone();
                candidate.symbols.pop();
                next.push(candidate);
            }
        }
        self.expand(next)
    }

    /// Whether the output so far is a complete value (`Complete`).
    pub fn complete(&self, state: &State) -> bool {
        state.iter().any(|stack| stack.symbols.is_empty())
    }

    /// The instruction that describes the constraint to the model.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// `state` as a mask cache key (`CanonicalMaskState`).
    fn canonical_mask_state(&self, state: &State, token_bytes: usize) -> State {
        let mut canonical = state.clone();
        for stack in &mut canonical {
            if let Some(&symbol) = stack.symbols.last()
                && symbol & LEXEME != 0
            {
                self.lexemes[(symbol & !LEXEME) as usize]
                    .canonical_mask_state(&mut stack.lexeme, token_bytes);
            }
        }
        canonical.sort_unstable();
        canonical.dedup();
        canonical
    }

    /// Whether a lexeme on top of a stack keeps text a trie walk may not
    /// intern.
    fn uncacheable(&self, state: &State) -> bool {
        state.iter().any(|stack| {
            stack.symbols.last().is_some_and(|&symbol| {
                symbol & LEXEME != 0
                    && !self.lexemes[(symbol & !LEXEME) as usize].cache_transitions()
            })
        })
    }
}

/// A token's text and whether it stops generation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Piece {
    pub text: Vec<u8>,
    pub stop: bool,
}

/// A trie node while the vocabulary is read.
#[derive(Debug, Clone, Default)]
struct Node {
    /// Child and byte.
    edges: Vec<(u32, u8)>,
    tokens: Vec<u32>,
}

/// The token trie in contiguous arrays: node `n` has the edges
/// `edges[first_edge[n]..first_edge[n + 1]]` in insertion order and the
/// tokens `tokens[first_token[n]..first_token[n + 1]]`.
#[derive(Debug, Default)]
struct Trie {
    first_edge: Vec<u32>,
    edges: Vec<(u32, u8)>,
    first_token: Vec<u32>,
    tokens: Vec<u32>,
}

impl Trie {
    fn new(nodes: Vec<Node>) -> Self {
        let mut trie = Self::default();
        for node in nodes {
            trie.first_edge.push(trie.edges.len() as u32);
            trie.edges.extend(node.edges);
            trie.first_token.push(trie.tokens.len() as u32);
            trie.tokens.extend(node.tokens);
        }
        trie.first_edge.push(trie.edges.len() as u32);
        trie.first_token.push(trie.tokens.len() as u32);
        trie
    }

    fn edges(&self, node: usize) -> &[(u32, u8)] {
        &self.edges[self.first_edge[node] as usize..self.first_edge[node + 1] as usize]
    }

    fn tokens(&self, node: usize) -> &[u32] {
        &self.tokens[self.first_token[node] as usize..self.first_token[node + 1] as usize]
    }
}

/// A vocabulary as a byte trie of its tokens (`ConstraintVocabulary`).
#[derive(Debug)]
pub struct ConstraintVocabulary {
    trie: Trie,
    pieces: Vec<Piece>,
    max_token_bytes: usize,
    /// Tokens that may hold a native reasoning boundary: stop and empty
    /// tokens and those containing '<' (`reasoning_boundary_tokens_`).
    reasoning_boundary_tokens: Vec<u32>,
}

/// A grammar state interned during a trie walk, with its byte transitions
/// (`u32::MAX` unknown, 0 dead).
struct CachedState {
    state: State,
    /// A lexeme on top keeps text the walk may not intern.
    uncacheable: bool,
    next: Box<[u32; 256]>,
}

impl CachedState {
    fn new(grammar: &JsonConstraint, state: State) -> Self {
        Self {
            uncacheable: grammar.uncacheable(&state),
            state,
            next: Box::new([u32::MAX; 256]),
        }
    }
}

/// The states and transitions of one mask computation.
struct Walk {
    states: Vec<CachedState>,
    intern: BTreeMap<State, u32>,
}

impl ConstraintVocabulary {
    /// Reads `size` tokens through `reader`.
    pub fn new(size: u32, mut reader: impl FnMut(u32) -> Result<Piece>) -> Result<Self> {
        if size == 0 || size > 1_048_576 {
            return Err(Error::invalid("constraint vocabulary size is unsupported"));
        }
        let mut nodes = vec![Node::default()];
        let mut pieces = Vec::with_capacity(size as usize);
        let mut max_token_bytes = 0;
        let mut total_bytes = 0usize;
        let mut reasoning_boundary_tokens = Vec::new();
        for token in 0..size {
            let piece = reader(token)?;
            if piece.stop || piece.text.is_empty() || piece.text.contains(&b'<') {
                reasoning_boundary_tokens.push(token);
            }
            total_bytes += piece.text.len();
            if total_bytes > 64 * 1024 * 1024 {
                return Err(Error::invalid("constraint vocabulary exceeds 64 MiB"));
            }
            if !piece.stop && !piece.text.is_empty() {
                if piece.text.len() > 4096 {
                    return Err(Error::invalid(
                        "constraint vocabulary token exceeds 4096 bytes",
                    ));
                }
                max_token_bytes = max_token_bytes.max(piece.text.len());
                let mut node = 0usize;
                for &byte in &piece.text {
                    let child = nodes.len();
                    let edges = &mut nodes[node].edges;
                    if let Some(&(existing, _)) = edges.iter().find(|&&(_, b)| b == byte) {
                        node = existing as usize;
                        continue;
                    }
                    edges.push((child as u32, byte));
                    if child >= 4_000_000 {
                        return Err(Error::invalid(
                            "constraint token trie exceeds its node limit",
                        ));
                    }
                    nodes.push(Node::default());
                    node = child;
                }
                nodes[node].tokens.push(token);
            }
            pieces.push(piece);
        }
        Ok(Self {
            trie: Trie::new(nodes),
            pieces,
            max_token_bytes,
            reasoning_boundary_tokens,
        })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Per token whether its text ends inside a UTF-8 character, as
    /// llama.cpp's `common_utf8_is_complete` reads it: the last lead byte
    /// within the final four bytes must have all its continuation bytes.
    pub fn partial_utf8(&self) -> Vec<bool> {
        self.pieces
            .iter()
            .map(|piece| {
                let text = &piece.text;
                let mut complete = text.is_empty();
                for i in 1..=text.len().min(4) {
                    let byte = text[text.len() - i];
                    if byte & 0xC0 != 0x80 {
                        let needed = match byte {
                            0xF0.. => 4,
                            0xE0.. => 3,
                            0xC0.. => 2,
                            _ => 1,
                        };
                        complete = i >= needed;
                        break;
                    }
                }
                !complete
            })
            .collect()
    }

    /// The tokens `grammar` admits from `state` (`Allowed`): whole tokens
    /// whose bytes it accepts, and stop tokens once the value is complete.
    pub fn allowed(&self, grammar: &JsonConstraint, state: &State) -> Result<Vec<bool>> {
        if state.iter().any(|stack| {
            stack.symbols.last().is_some_and(|&symbol| {
                symbol & LEXEME != 0
                    && grammar.lexemes[(symbol & !LEXEME) as usize].plain_reasoning(&stack.lexeme)
            })
        }) {
            // Neither native boundary can begin without '<', so most
            // vocabulary entries are valid without walking their bytes.
            // Boundary, empty and stop tokens still take the whole grammar,
            // including tokens holding both a boundary and constrained output.
            let mut mask = vec![true; self.pieces.len()];
            for &token in &self.reasoning_boundary_tokens {
                mask[token as usize] = self.allows(grammar, state, token)?;
            }
            return Ok(mask);
        }
        let mut mask = vec![false; self.pieces.len()];
        if grammar.complete(state) {
            for (allowed, piece) in mask.iter_mut().zip(&self.pieces) {
                *allowed = piece.stop || (grammar.automatic_tools && piece.text.is_empty());
            }
            if grammar.stop_only_when_complete {
                return Ok(mask);
            }
        }
        // Intern the grammar states reached while walking the token trie.
        // Long word tokens share both trie prefixes and string-body
        // transitions; expanding a pushdown state once per byte edge would
        // otherwise dominate decode time.
        let mut walk = Walk {
            states: vec![
                CachedState::new(grammar, State::new()),
                CachedState::new(grammar, state.clone()),
            ],
            intern: BTreeMap::from([(State::new(), 0), (state.clone(), 1)]),
        };
        let mut work = 0usize;
        self.walk(grammar, &mut walk, &mut mask, &mut work)?;
        if !mask.iter().any(|&allowed| allowed) {
            return Err(Error::runtime("JSON constraint has no valid token"));
        }
        Ok(mask)
    }

    /// Walks the trie depth first through interned states, in edge order.
    /// Each reachable node is visited once, whichever the order, so the
    /// mask and the work count do not depend on it.
    fn walk(
        &self,
        grammar: &JsonConstraint,
        walk: &mut Walk,
        mask: &mut [bool],
        work: &mut usize,
    ) -> Result<()> {
        // Node and interned state.
        let mut pending: Vec<(u32, u32)> = vec![(0, 1)];
        while let Some((node, current)) = pending.pop() {
            let node = node as usize;
            let cached = &walk.states[current as usize];
            // Numeric predicates keep their scalar text; walk those branches
            // with depth-bounded scratch rather than a 256-way transition
            // table per vocabulary prefix.
            if cached.uncacheable {
                self.direct(grammar, node, &cached.state, mask, work)?;
                continue;
            }
            *work += 1;
            if *work > MAX_WORK {
                return Err(Error::runtime("JSON token mask work limit exceeded"));
            }
            for &token in self.trie.tokens(node) {
                mask[token as usize] = true;
            }
            let children = pending.len();
            self.expand_edges(grammar, walk, node, current, &mut pending, mask, work)?;
            pending[children..].reverse();
        }
        Ok(())
    }

    /// Queues the children of `node` reached from interned state
    /// `current`.
    #[allow(clippy::too_many_arguments)]
    fn expand_edges(
        &self,
        grammar: &JsonConstraint,
        walk: &mut Walk,
        node: usize,
        current: u32,
        pending: &mut Vec<(u32, u32)>,
        mask: &mut [bool],
        work: &mut usize,
    ) -> Result<()> {
        for &(child, byte) in self.trie.edges(node) {
            let mut transition = walk.states[current as usize].next[usize::from(byte)];
            if transition == u32::MAX {
                let next = grammar.advance(&walk.states[current as usize].state, byte)?;
                transition = match walk.intern.get(&next) {
                    Some(&id) => id,
                    // Cache capacity is an optimization limit, not a
                    // language limit.
                    None if walk.states.len() >= 8192 => {
                        if !next.is_empty() {
                            self.direct(grammar, child as usize, &next, mask, work)?;
                        }
                        continue;
                    }
                    None => {
                        let id = walk.states.len() as u32;
                        walk.intern.insert(next.clone(), id);
                        walk.states.push(CachedState::new(grammar, next));
                        id
                    }
                };
                walk.states[current as usize].next[usize::from(byte)] = transition;
            }
            if transition != 0 {
                pending.push((child, transition));
            }
        }
        Ok(())
    }

    /// Walks the trie below `node` advancing `current` byte by byte.
    fn direct(
        &self,
        grammar: &JsonConstraint,
        node: usize,
        current: &State,
        mask: &mut [bool],
        work: &mut usize,
    ) -> Result<()> {
        *work += 1;
        if *work > MAX_WORK {
            return Err(Error::runtime("JSON token mask work limit exceeded"));
        }
        for &token in self.trie.tokens(node) {
            mask[token as usize] = true;
        }
        for &(child, byte) in self.trie.edges(node) {
            let next = grammar.advance(current, byte)?;
            if !next.is_empty() {
                self.direct(grammar, child as usize, &next, mask, work)?;
            }
        }
        Ok(())
    }

    /// The state after an admitted `token` (`Accept`); a stop token keeps a
    /// complete state.
    pub fn accept(&self, grammar: &JsonConstraint, state: &State, token: u32) -> Result<State> {
        let invalid = || Error::runtime("invalid token accepted by JSON constraint");
        let piece = self.pieces.get(token as usize).ok_or_else(invalid)?;
        if (piece.stop || (grammar.automatic_tools && piece.text.is_empty()))
            && grammar.complete(state)
        {
            return Ok(state.clone());
        }
        if piece.stop || piece.text.is_empty() {
            return Err(invalid());
        }
        let mut next = state.clone();
        for &byte in &piece.text {
            next = grammar.advance(&next, byte)?;
        }
        if next.is_empty() {
            return Err(invalid());
        }
        Ok(next)
    }

    /// Whether the mask of `state` admits `token` (`Allows`), without
    /// computing the mask: an argmax that obeys the grammar needs no
    /// download.
    pub fn allows(&self, grammar: &JsonConstraint, state: &State, token: u32) -> Result<bool> {
        let piece = self
            .pieces
            .get(token as usize)
            .ok_or_else(|| Error::invalid("token outside the constraint vocabulary"))?;
        if grammar.stop_only_when_complete && grammar.complete(state) {
            return Ok(piece.stop);
        }
        if piece.stop {
            return Ok(grammar.complete(state));
        }
        if piece.text.is_empty() {
            return Ok(grammar.automatic_tools && grammar.complete(state));
        }
        let mut next = state.clone();
        for &byte in &piece.text {
            next = grammar.advance(&next, byte)?;
            if next.is_empty() {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// A grammar bound to a vocabulary, with the last 16 token masks cached by
/// canonical state (`TokenConstraint`).
#[derive(Debug)]
pub struct TokenConstraint {
    grammar: Arc<JsonConstraint>,
    vocabulary: Arc<ConstraintVocabulary>,
    masks: Mutex<BTreeMap<State, Arc<Vec<bool>>>>,
    /// The canonical start state, whose mask every request needs.
    initial: State,
}

impl TokenConstraint {
    pub fn new(grammar: Arc<JsonConstraint>, vocabulary: Arc<ConstraintVocabulary>) -> Self {
        let initial = grammar
            .start()
            .map(|start| grammar.canonical_mask_state(&start, vocabulary.max_token_bytes))
            .unwrap_or_default();
        Self {
            grammar,
            vocabulary,
            masks: Mutex::new(BTreeMap::new()),
            initial,
        }
    }

    pub fn grammar(&self) -> &Arc<JsonConstraint> {
        &self.grammar
    }

    pub fn vocabulary(&self) -> &Arc<ConstraintVocabulary> {
        &self.vocabulary
    }

    /// The admitted tokens from `state` (`Allowed`); states that differ only
    /// in counts the mask cannot depend on share a cached mask.
    pub fn allowed(&self, state: &State) -> Result<Arc<Vec<bool>>> {
        let canonical = self
            .grammar
            .canonical_mask_state(state, self.vocabulary.max_token_bytes);
        if let Some(found) = lock(&self.masks).get(&canonical) {
            return Ok(found.clone());
        }
        let mask = Arc::new(self.vocabulary.allowed(&self.grammar, &canonical)?);
        let mut masks = lock(&self.masks);
        if let Some(found) = masks.get(&canonical) {
            return Ok(found.clone());
        }
        if masks.len() >= CACHE_ENTRIES {
            // Every request needs the initial mask, even when the model state
            // is cached: keep it within the same budget instead of
            // rebuilding the whole vocabulary mask after each native call.
            let victim = masks.keys().find(|state| **state != self.initial).cloned();
            if let Some(victim) = victim {
                masks.remove(&victim);
            }
        }
        Ok(masks.entry(canonical).or_insert(mask).clone())
    }
}
