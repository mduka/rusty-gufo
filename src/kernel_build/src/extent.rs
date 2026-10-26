//! `len` expressions: integers, integer scalar names, `+ - * / %` and
//! parentheses, and `requires` relations between two of them. The generated
//! Rust checks every operation for overflow.

use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq)]
pub enum Expr {
    Int(u64),
    Ident(String),
    Bin(char, Box<Expr>, Box<Expr>),
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum Token {
    Int(u64),
    Ident(String),
    Op(char),
}

impl Expr {
    pub fn parse(text: &str) -> Result<Self, String> {
        let tokens = tokenize(text)?;
        let mut pos = 0;
        let expr = parse_sum(&tokens, &mut pos)?;
        if pos != tokens.len() {
            return Err(format!("trailing input in len `{text}`"));
        }
        Ok(expr)
    }

    pub fn idents(&self) -> Vec<String> {
        match self {
            Self::Int(_) => Vec::new(),
            Self::Ident(name) => vec![name.clone()],
            Self::Bin(_, l, r) => [l.idents(), r.idents()].concat(),
        }
    }

    /// Value for concrete scalars; `None` on overflow, division by zero or
    /// an unknown name. Checks probes at build time.
    pub fn eval(&self, values: &BTreeMap<String, u64>) -> Option<u64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Ident(name) => values.get(name).copied(),
            Self::Bin(op, l, r) => {
                let (l, r) = (l.eval(values)?, r.eval(values)?);
                match op {
                    '+' => l.checked_add(r),
                    '-' => l.checked_sub(r),
                    '*' => l.checked_mul(r),
                    '/' => l.checked_div(r),
                    _ => l.checked_rem(r),
                }
            }
        }
    }

    /// Rust expression over `x_<name>: usize` locals, yielding
    /// `Result<usize, LaunchError>` via `?` on every operation.
    pub fn rust(&self, param: &str) -> String {
        match self {
            Self::Int(v) => format!("{v}usize"),
            Self::Ident(name) => format!("x_{name}"),
            Self::Bin(op, l, r) => {
                let f = match op {
                    '+' => "add",
                    '-' => "sub",
                    '*' => "mul",
                    '/' => "div",
                    _ => "rem",
                };
                format!("l::{f}({}, {}, \"{param}\")?", l.rust(param), r.rust(param))
            }
        }
    }
}

/// `lhs OP rhs` with OP one of `<= < >= > == !=`.
#[derive(Debug, PartialEq, Eq)]
pub struct Relation {
    pub lhs: Expr,
    pub op: &'static str,
    pub rhs: Expr,
}

impl Relation {
    pub fn parse(text: &str) -> Result<Self, String> {
        for op in ["<=", ">=", "==", "!=", "<", ">"] {
            if let Some((lhs, rhs)) = text.split_once(op) {
                return Ok(Self {
                    lhs: Expr::parse(lhs)?,
                    op,
                    rhs: Expr::parse(rhs)?,
                });
            }
        }
        Err(format!("`{text}` is not a relation (<=, <, >=, >, ==, !=)"))
    }

    pub fn idents(&self) -> Vec<String> {
        [self.lhs.idents(), self.rhs.idents()].concat()
    }

    /// Whether the relation holds; `None` when a side cannot be evaluated.
    pub fn eval(&self, values: &BTreeMap<String, u64>) -> Option<bool> {
        let (l, r) = (self.lhs.eval(values)?, self.rhs.eval(values)?);
        Some(match self.op {
            "<=" => l <= r,
            ">=" => l >= r,
            "==" => l == r,
            "!=" => l != r,
            "<" => l < r,
            _ => l > r,
        })
    }

    /// Rust `bool` expression; `?` reports overflow against `label`.
    pub fn rust(&self, label: &str) -> String {
        format!(
            "{} {} {}",
            self.lhs.rust(label),
            self.op,
            self.rhs.rust(label)
        )
    }
}

fn tokenize(text: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c.is_ascii_digit() {
            let mut value = 0u64;
            while let Some(d) = chars.peek().and_then(|c| c.to_digit(10)) {
                value = value
                    .checked_mul(10)
                    .and_then(|v| v.checked_add(u64::from(d)))
                    .ok_or_else(|| format!("integer too large in len `{text}`"))?;
                chars.next();
            }
            tokens.push(Token::Int(value));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let mut name = String::new();
            while let Some(&c) = chars
                .peek()
                .filter(|c| c.is_ascii_alphanumeric() || **c == '_')
            {
                name.push(c);
                chars.next();
            }
            tokens.push(Token::Ident(name));
        } else if "+-*/%()".contains(c) {
            tokens.push(Token::Op(c));
            chars.next();
        } else {
            return Err(format!("unexpected `{c}` in len `{text}`"));
        }
    }
    Ok(tokens)
}

fn parse_sum(t: &[Token], pos: &mut usize) -> Result<Expr, String> {
    let mut lhs = parse_product(t, pos)?;
    while let Some(Token::Op(op @ ('+' | '-'))) = t.get(*pos).cloned() {
        *pos += 1;
        lhs = Expr::Bin(op, Box::new(lhs), Box::new(parse_product(t, pos)?));
    }
    Ok(lhs)
}

fn parse_product(t: &[Token], pos: &mut usize) -> Result<Expr, String> {
    let mut lhs = parse_atom(t, pos)?;
    while let Some(Token::Op(op @ ('*' | '/' | '%'))) = t.get(*pos).cloned() {
        *pos += 1;
        lhs = Expr::Bin(op, Box::new(lhs), Box::new(parse_atom(t, pos)?));
    }
    Ok(lhs)
}

fn parse_atom(t: &[Token], pos: &mut usize) -> Result<Expr, String> {
    let token = t.get(*pos).cloned();
    *pos += 1;
    match token {
        Some(Token::Int(v)) => Ok(Expr::Int(v)),
        Some(Token::Ident(s)) => Ok(Expr::Ident(s)),
        Some(Token::Op('(')) => {
            let inner = parse_sum(t, pos)?;
            if t.get(*pos) != Some(&Token::Op(')')) {
                return Err("missing `)` in len".into());
            }
            *pos += 1;
            Ok(inner)
        }
        _ => Err("malformed len".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplication_binds_tighter_and_parentheses_group() {
        assert_eq!(
            Expr::parse("a + b * 2").unwrap().rust("p"),
            r#"l::add(x_a, l::mul(x_b, 2usize, "p")?, "p")?"#
        );
        assert_eq!(
            Expr::parse("(layer + 1) * n").unwrap().rust("p"),
            r#"l::mul(l::add(x_layer, 1usize, "p")?, x_n, "p")?"#
        );
        assert_eq!(Expr::parse("a - b - c").unwrap().idents(), ["a", "b", "c"]);
    }

    #[test]
    fn remainders_and_relations_evaluate_like_the_generated_code() {
        assert_eq!(
            Expr::parse("a % 4 * 2").unwrap().rust("p"),
            r#"l::mul(l::rem(x_a, 4usize, "p")?, 2usize, "p")?"#
        );
        let values = BTreeMap::from([("a".to_string(), 10), ("b".to_string(), 3)]);
        assert_eq!(Expr::parse("a % b + 1").unwrap().eval(&values), Some(2));
        assert_eq!(Expr::parse("b - a").unwrap().eval(&values), None);
        assert_eq!(Expr::parse("a / (b - 3)").unwrap().eval(&values), None);
        let relation = Relation::parse("a + b <= 13").unwrap();
        assert_eq!(relation.op, "<=");
        assert_eq!(relation.eval(&values), Some(true));
        assert_eq!(
            Relation::parse("a % b != 1").unwrap().eval(&values),
            Some(false)
        );
        assert_eq!(Relation::parse("b < a").unwrap().rust("f"), "x_b < x_a");
        assert!(Relation::parse("a + b").is_err());
        assert!(Relation::parse("a <= ").is_err());
    }

    #[test]
    fn rejects_malformed_expressions() {
        for bad in ["", "a +", "(a", "a b", "a ^ 2", "99999999999999999999999"] {
            assert!(Expr::parse(bad).is_err(), "{bad}");
        }
    }
}
