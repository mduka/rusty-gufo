//! The extension's `tokenizer_contract_test.cpp` and `draft_vocab_test.cpp`
//! on synthetic vocabularies shaped like Gemma 4's.

#![allow(clippy::unwrap_used)]

use gufo_gemma4::draft_vocab::draft_head_order;
use gufo_gemma4::tokenizer::{TokenId, TokenType, Tokenizer, Vocabulary};

struct Fixture {
    tokenizer: Tokenizer,
}

impl Fixture {
    fn id(&self, text: &str) -> TokenId {
        self.tokenizer
            .find_token(text.as_bytes())
            .unwrap_or_else(|| panic!("fixture token missing: {text}"))
    }
}

fn add(v: &mut Vocabulary, text: &str, ty: TokenType) {
    v.tokens.push(text.as_bytes().to_vec());
    v.types.push(ty);
}

/// A synthetic vocabulary shaped like Gemma 4's: control/user-defined
/// specials first, 256 byte-fallback tokens, then SentencePiece pieces.
fn build(add_bos: bool) -> Fixture {
    let mut v = Vocabulary::default();
    add(&mut v, "<pad>", TokenType::Control); // 0
    add(&mut v, "<eos>", TokenType::Normal); // 1, promoted to control (EOG)
    add(&mut v, "<bos>", TokenType::Control); // 2
    add(&mut v, "<unk>", TokenType::Unknown); // 3
    add(&mut v, "<|turn>", TokenType::Control); // 4
    add(&mut v, "<turn|>", TokenType::Control); // 5
    add(&mut v, "<|tool_call>", TokenType::UserDefined); // 6
    add(&mut v, "<|tool_response>", TokenType::UserDefined); // 7, promoted (EOG)
    for b in 0..256 {
        add(&mut v, &format!("<0x{b:02X}>"), TokenType::Byte);
    }
    for piece in [
        "\u{2581}",
        "h",
        "e",
        "l",
        "o",
        "u",
        "s",
        "r",
        "\n",
        "he",
        "ll",
        "llo",
        "hello",
        "\u{2581}hello",
        "\n\n",
        "\u{2581}\u{2581}",
        "us",
        "er",
        "user",
    ] {
        add(&mut v, piece, TokenType::Normal);
    }
    v.merges = [
        "h e",
        "l l",
        "ll o",
        "he llo",
        "\u{2581} hello",
        "\n \n",
        "\u{2581} \u{2581}",
        "u s",
        "e r",
        "us er",
    ]
    .iter()
    .map(|m| m.as_bytes().to_vec())
    .collect();
    v.bos = 2;
    v.eos = 5;
    v.pad = 0;
    v.unknown = 3;
    v.add_bos = add_bos;
    Fixture {
        tokenizer: Tokenizer::new(v).expect("synthetic vocabulary"),
    }
}

fn expect(f: &Fixture, text: &[u8], parse_special: bool, expected: &[TokenId], what: &str) {
    let got = f.tokenizer.encode(text, false, parse_special);
    assert_eq!(got, expected, "{what}");
}

#[test]
fn merges() {
    let f = build(true);
    expect(&f, b"hello", false, &[f.id("hello")], "full merge chain");
    expect(
        &f,
        b" hello",
        false,
        &[f.id("\u{2581}hello")],
        "space escapes to U+2581",
    );
    expect(&f, b"  ", false, &[f.id("\u{2581}\u{2581}")], "space pair");
    // Equal ranks resolve leftmost first: lll -> ll l, never l ll.
    expect(
        &f,
        b"lll",
        false,
        &[f.id("ll"), f.id("l")],
        "leftmost tie break",
    );
    expect(&f, b"user", false, &[f.id("user")], "multi-step merges");
}

#[test]
fn newlines() {
    let f = build(true);
    expect(
        &f,
        b"hello\n\nhello",
        false,
        &[f.id("hello"), f.id("\n\n"), f.id("hello")],
        "newline runs split words",
    );
    // A run that is not itself a token is merged like any other word.
    expect(
        &f,
        b"\n\n\n",
        false,
        &[f.id("\n\n"), f.id("\n")],
        "newline run outside the vocabulary",
    );
    expect(&f, b"\n", false, &[f.id("\n")], "single newline");
}

/// Long lines are cached across calls: a line encodes the same whether it is
/// new, repeated alone, or repeated inside other text.
#[test]
fn word_cache() {
    let f = build(true);
    let mut line = String::new();
    let mut expected = Vec::new();
    for i in 0..20 {
        line.push_str(if i % 3 == 0 { " user" } else { " hello" });
        expected.push(f.id(if i % 3 == 0 {
            "\u{2581}"
        } else {
            "\u{2581}hello"
        }));
        if i % 3 == 0 {
            expected.push(f.id("user"));
        }
    }
    expect(
        &f,
        line.as_bytes(),
        false,
        &expected,
        "long line, first encode",
    );
    expect(&f, line.as_bytes(), false, &expected, "long line, cached");
    let mut twice = expected.clone();
    twice.push(f.id("\n\n"));
    twice.extend_from_slice(&expected);
    expect(
        &f,
        format!("{line}\n\n{line}").as_bytes(),
        false,
        &twice,
        "cached line inside text",
    );
    // A line differing only at its end is its own entry.
    let mut longer = expected.clone();
    longer.push(f.id("\u{2581}hello"));
    expect(
        &f,
        format!("{line} hello").as_bytes(),
        false,
        &longer,
        "extended long line",
    );
}

#[test]
fn byte_fallback() {
    let f = build(true);
    expect(
        &f,
        b"\xC3\xA9",
        false,
        &[f.id("<0xC3>"), f.id("<0xA9>")],
        "UTF-8 character outside the vocabulary",
    );
    // Invalid UTF-8 is replaced by U+FFFD before merging.
    expect(
        &f,
        b"\xFF",
        false,
        &[f.id("<0xEF>"), f.id("<0xBF>"), f.id("<0xBD>")],
        "invalid byte becomes U+FFFD",
    );
    expect(
        &f,
        b"h\xE2\x82",
        false,
        &[
            f.id("h"),
            f.id("<0xEF>"),
            f.id("<0xBF>"),
            f.id("<0xBD>"),
            f.id("<0xEF>"),
            f.id("<0xBF>"),
            f.id("<0xBD>"),
        ],
        "truncated sequence",
    );
}

#[test]
fn specials() {
    let f = build(true);
    let turn = f.id("<|turn>");
    expect(
        &f,
        b"<|turn>user",
        true,
        &[turn, f.id("user")],
        "control token parsed as special",
    );
    let plain = f.tokenizer.encode(b"<|turn>", false, false);
    assert!(
        plain.len() > 1 && plain[0] != turn,
        "control token matched without parse_special"
    );
    expect(
        &f,
        b"<|tool_call>hello",
        false,
        &[f.id("<|tool_call>"), f.id("hello")],
        "user-defined token always matched",
    );
    expect(
        &f,
        b"hello<eos>",
        true,
        &[f.id("hello"), 1],
        "EOG text promoted to a control token",
    );
    // Fragments around specials are tokenized independently.
    expect(
        &f,
        b"he<turn|>llo",
        true,
        &[f.id("he"), f.id("<turn|>"), f.id("llo")],
        "special splits words",
    );
    let with_bos = f.tokenizer.encode(b"hello", true, true);
    assert!(
        with_bos.len() == 2 && with_bos[0] == 2,
        "BOS added on request"
    );
    let no_bos = build(false);
    assert_eq!(
        no_bos.tokenizer.encode(b"hello", true, true).len(),
        1,
        "BOS follows the vocabulary flag"
    );
}

#[test]
fn decode() {
    let f = build(true);
    let tokens = [
        f.id("\u{2581}hello"),
        f.id("<turn|>"),
        f.id("<|tool_call>"),
        f.id("<0xC3>"),
        f.id("<0xA9>"),
    ];
    assert_eq!(
        f.tokenizer.decode(&tokens, false),
        b" hello<|tool_call>\xC3\xA9",
        "control hidden, user-defined printed, bytes raw"
    );
    assert_eq!(
        f.tokenizer.decode(&tokens, true),
        b" hello<turn|><|tool_call>\xC3\xA9",
        "control printed with special"
    );
}

#[test]
fn end_of_generation() {
    let f = build(true);
    assert!(f.tokenizer.is_end_of_generation(1), "<eos> ends generation");
    assert!(
        f.tokenizer.is_end_of_generation(5),
        "<turn|> ends generation"
    );
    assert!(
        f.tokenizer.is_end_of_generation(7),
        "<|tool_response> ends generation"
    );
    assert!(!f.tokenizer.is_end_of_generation(4), "<|turn> does not");
    assert_eq!(
        f.tokenizer.token_type(7),
        TokenType::Control,
        "EOG user-defined token promoted to control"
    );
}

#[test]
fn malformed() {
    let mut v = Vocabulary {
        tokens: vec![b"a".to_vec(), b"b".to_vec()],
        types: vec![TokenType::Normal, TokenType::Normal],
        merges: vec![b"a b".to_vec()], // "ab" is not a token
        bos: 0,
        eos: 1,
        ..Vocabulary::default()
    };
    let error = Tokenizer::new(v.clone()).expect_err("merge outside the vocabulary accepted");
    assert!(!error.is_empty());
    v.merges = vec![b"ab".to_vec()];
    assert!(
        Tokenizer::new(v.clone()).is_err(),
        "merge without a space accepted"
    );
    v.merges.clear();
    v.types.pop();
    assert!(Tokenizer::new(v).is_err(), "type count mismatch accepted");
}

/// `draft_vocab_test.cpp`: specials first, then tokens by the merge that
/// creates them, common single symbols beside the first merges, every other
/// token after them by id.
#[test]
fn draft_head_order_ranks_merges() {
    let mut v = Vocabulary::default();
    add(&mut v, "<pad>", TokenType::Control); // 0
    add(&mut v, "<0x41>", TokenType::Byte); // 1: no merge, not a symbol
    add(&mut v, "中", TokenType::Normal); // 2: rare single symbol
    add(&mut v, "ab\u{2581}", TokenType::Normal); // 3: second merge
    add(&mut v, "a", TokenType::Normal); // 4: common symbol
    add(&mut v, "\u{2581}", TokenType::Normal); // 5: space
    add(&mut v, "—", TokenType::Normal); // 6: general punctuation
    add(&mut v, "ab", TokenType::Normal); // 7: first merge
    add(&mut v, "b", TokenType::Normal); // 8
    add(&mut v, "<|turn>", TokenType::Control); // 9
    add(&mut v, "<mask>", TokenType::UserDefined); // 10
    add(&mut v, "é", TokenType::Normal); // 11: Latin-1, common
    v.merges = vec![b"a b".to_vec(), "ab \u{2581}".as_bytes().to_vec()];
    v.bos = 0;
    v.eos = 9;
    let tokenizer = Tokenizer::new(v).unwrap();
    let ranks = tokenizer.creation_ranks();
    assert!(
        ranks[7] == 0 && ranks[3] == 1 && ranks[4] == u32::MAX,
        "creation ranks"
    );
    assert!(
        tokenizer.piece(5) == "\u{2581}".as_bytes() && tokenizer.piece(12).is_empty(),
        "pieces"
    );
    assert_eq!(
        draft_head_order(&tokenizer),
        [0, 9, 10, 4, 5, 6, 7, 8, 11, 3, 1, 2],
        "draft head order"
    );
}
