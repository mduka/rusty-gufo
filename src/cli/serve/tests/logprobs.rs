//! Port of `tests/cli/logprobs_test.cpp`: byte-fallback correction, the
//! Chat Completions, Completions and Responses JSON, the request controls,
//! and joining Responses text spans to their tokens by source position.

use gufo_core::json;
use gufo_core::sampling::{SamplingConfig, TokenLogprob, TokenLogprobs};
use gufo_serve::logprobs::{
    DecodedLogprob, DecodedLogprobs, LogprobDecoder, ResponsesLogprobs, TextSourceSpan, TokenPiece,
    chat_logprobs, completion_logprobs, parse_logprobs, parse_responses_logprobs, unicode_length,
};

fn entry(token: u32, logprob: f64) -> TokenLogprob {
    TokenLogprob { token, logprob }
}

fn span(begin: usize, size: usize) -> [TextSourceSpan; 1] {
    [TextSourceSpan { begin, size }]
}

#[test]
fn byte_fallback_tokens_decode_against_the_sampled_context() {
    // Alternatives use the preceding sampled context, never another
    // alternative at the same position.
    let vocabulary: [&[u8]; 5] = [b"\xc3", b"\xa9", b"A", b"\xaa", "🙂".as_bytes()];
    let mut decoder = LogprobDecoder::default();
    let decode = |id: u32| vocabulary[id as usize].to_vec();
    let first = decoder.decode_token(
        &TokenLogprobs {
            selected: entry(0, -1.0),
            top: vec![entry(2, -2.0), entry(4, -3.0)],
        },
        decode,
    );
    assert!(first.entries[0].token.is_empty());
    assert_eq!(first.entries[1].token, "A");
    let second = decoder.decode_token(
        &TokenLogprobs {
            selected: entry(1, -0.1),
            top: vec![entry(3, -2.0), entry(1, -0.1)],
        },
        decode,
    );
    assert_eq!(second.entries.len(), 2);
    assert_eq!(second.entries[0].token, "é");
    assert_eq!(second.entries[1].token, "ê");
    let sequence = [first, second];
    let chat = chat_logprobs(&sequence, 1);
    let content = chat.find("content").unwrap().items();
    assert_eq!(
        content[1].find("bytes").unwrap().dump().unwrap(),
        "[195,169]"
    );
    assert_eq!(content[0].find("top_logprobs").unwrap().len(), 1);
    let completion = completion_logprobs(&sequence, 7);
    assert_eq!(
        completion.find("text_offset").unwrap().dump().unwrap(),
        "[7,7]"
    );
    assert_eq!(unicode_length("aé🙂".as_bytes()), 3);
    // vLLM keeps the selected token in completion maps even with zero
    // requested alternatives; very negative values use OpenAI's JSON-safe
    // sentinel.
    let unlikely = DecodedLogprobs {
        entries: vec![DecodedLogprob {
            id: 9,
            token: "x".into(),
            logprob: f64::NEG_INFINITY,
        }],
    };
    assert_eq!(
        completion_logprobs(std::slice::from_ref(&unlikely), 0)
            .find("top_logprobs")
            .unwrap()
            .dump()
            .unwrap(),
        r#"[{"x":-9999}]"#
    );
    assert!(
        chat_logprobs(std::slice::from_ref(&unlikely), 0)
            .find("content")
            .unwrap()
            .items()[0]
            .find("top_logprobs")
            .unwrap()
            .is_empty()
    );
}

#[test]
fn request_controls_enable_logprobs_only_when_asked() {
    let parse = |body: &str| json::parse_str(body).unwrap();
    let mut config = SamplingConfig::default();
    assert!(
        parse_logprobs(
            &parse(r#"{"logprobs":null,"top_logprobs":null}"#),
            true,
            &mut config
        )
        .is_ok()
            && config.logprobs.is_none()
    );
    for body in ["{}", r#"{"logprobs":false}"#, r#"{"top_logprobs":0}"#] {
        config.logprobs = Some(20);
        assert!(parse_logprobs(&parse(body), true, &mut config).is_ok());
        assert!(config.logprobs.is_none(), "{body}");
    }
    assert!(parse_logprobs(&parse(r#"{"logprobs":true}"#), true, &mut config).is_ok());
    assert_eq!(config.logprobs, Some(0));
    for body in [
        "{}",
        r#"{"include":null}"#,
        r#"{"include":[],"top_logprobs":20}"#,
        r#"{"include":["reasoning.encrypted_content"]}"#,
    ] {
        config.logprobs = Some(20);
        assert!(parse_responses_logprobs(&parse(body), &mut config).is_ok());
        assert!(config.logprobs.is_none(), "{body}");
    }
    for body in [
        r#"{"include":"message.output_text.logprobs"}"#,
        r#"{"include":[null]}"#,
        r#"{"include":[false]}"#,
        r#"{"top_logprobs":-1}"#,
        r#"{"top_logprobs":21}"#,
        r#"{"top_logprobs":1.5}"#,
        r#"{"top_logprobs":true}"#,
        r#"{"top_logprobs":"3"}"#,
    ] {
        assert!(
            parse_responses_logprobs(&parse(body), &mut config).is_err(),
            "{body}"
        );
    }
    assert!(
        parse_responses_logprobs(
            &parse(r#"{"include":["message.output_text.logprobs"],"top_logprobs":null}"#),
            &mut config
        )
        .is_ok()
    );
    assert_eq!(config.logprobs, Some(0));
}

#[test]
fn responses_spans_take_their_tokens_by_source_position() {
    let first = DecodedLogprobs {
        entries: vec![DecodedLogprob {
            id: 0,
            token: String::new(),
            logprob: -1.0,
        }],
    };
    let second = DecodedLogprobs {
        entries: vec![DecodedLogprob {
            id: 1,
            token: "é".into(),
            logprob: -0.1,
        }],
    };
    let unlikely = DecodedLogprobs {
        entries: vec![DecodedLogprob {
            id: 9,
            token: "x".into(),
            logprob: f64::NEG_INFINITY,
        }],
    };
    // Two tokens contribute to one Unicode scalar; neither is lost or
    // repeated.
    let mut response = ResponsesLogprobs::new(0);
    response.push(&TokenPiece::with_logprobs(b"\xc3", Some(&first)), false);
    response.push(&TokenPiece::with_logprobs(b"\xa9", Some(&second)), false);
    let accent = response.take(&span(0, 2));
    assert_eq!(accent.len(), 2);
    assert!(accent.items()[0].member_str("token", "?").is_empty());
    assert_eq!(accent.items()[1].member_str("token", ""), "é");
    assert!(response.take(&span(0, 2)).is_empty());

    // A malformed prefix is replaced before the next ASCII token. Their
    // source ranges must not merge just because both decoded in one push.
    let mut malformed = ResponsesLogprobs::new(0);
    malformed.push(&TokenPiece::with_logprobs(b"\xe2", Some(&first)), false);
    malformed.push(&TokenPiece::with_logprobs(b"A", Some(&unlikely)), false);
    let ascii = malformed.take(&span(3, 1));
    assert_eq!(ascii.len(), 1);
    assert_eq!(ascii.items()[0].member_str("token", ""), "x");
    assert_eq!(malformed.take(&span(0, 3)).len(), 1);
    malformed.push(&TokenPiece::with_logprobs(b"", Some(&second)), false);
    malformed.push(&TokenPiece::new(b""), true);
    assert!(malformed.take(&span(3, 1)).is_empty());

    let mut truncated = ResponsesLogprobs::new(0);
    truncated.push(&TokenPiece::with_logprobs(b"\xe2", Some(&first)), false);
    truncated.push(&TokenPiece::new(b""), true);
    assert_eq!(truncated.take(&span(0, 3)).len(), 1);
}
