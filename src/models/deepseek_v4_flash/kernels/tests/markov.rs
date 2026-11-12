//! The device-chained Markov selection of DSpark drafts (`markov_test` of
//! the C++ tests): every chain of one to eight rows matches an independent
//! FP64 evaluation, ties included, and a replay of single-row chains; bad
//! counts, anchors, model ranges and buffers are refused:
//! cargo test -p gufo-deepseek-v4-flash-kernels --features gpu --test markov

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::io::Write;

use gufo_deepseek_v4_flash_kernels::{Backend, DeviceTensor, Input, Weight};
use gufo_hip::MappedFile;

/// Two Q8 tables of `vocab` rows of `rank` weights at `first` and `second`.
struct Fixture {
    vocab: u32,
    rank: u32,
    first: u64,
    second: u64,
}

/// Weight `column` of row `row` of the table at `offset`; fixture blocks use
/// the exactly representable scale 2^-5.
fn weight(model: &[u8], offset: u64, rank: u32, row: u32, column: u32) -> f64 {
    let block = offset as usize + (row as usize * rank as usize + column as usize) / 32 * 34;
    f64::from(model[block + 2 + column as usize % 32] as i8) / 32.0
}

fn tensor(b: &Backend, bytes: &[u8]) -> DeviceTensor {
    let mut t = b.alloc(bytes.len() as u64).unwrap();
    b.write(&mut t, 0, bytes).unwrap();
    t
}

fn indices(b: &Backend, t: &impl Input, count: usize) -> Vec<i32> {
    let mut bytes = vec![0u8; count * 4];
    b.read(t, 0, &mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| i32::from_le_bytes(c))
        .collect()
}

fn run(b: &Backend, model: &[u8], f: &Fixture) {
    let (w1, w2) = (Weight::support(f.first), Weight::support(f.second));
    let vocab = f.vocab as usize;
    for rows in 1..=8u32 {
        let logits: Vec<f32> = (0..rows as usize * vocab)
            .map(|i| (((i * 19 + i / vocab * 7) % 31) as i32 - 15) as f32 / 1024.0)
            .collect();
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input = tensor(b, &bytes);
        let mut output = b.alloc(u64::from(rows) * 4).unwrap();
        let mut key = b.alloc(8).unwrap();
        let anchor = f.vocab - 1;
        let mut select = |count: u32, previous: u32, w2: Weight| {
            b.dspark_markov_chain(
                &mut output,
                &mut key,
                &input,
                w1,
                w2,
                f.vocab,
                f.rank,
                previous,
                count,
            )
        };
        assert!(
            select(0, anchor, w2).is_err()
                && select(9, anchor, w2).is_err()
                && select(rows, f.vocab, w2).is_err()
                && select(rows, anchor, Weight::support(model.len() as u64)).is_err(),
            "reject invalid counts, anchors and model ranges"
        );
        if rows < 8 {
            assert!(
                select(rows + 1, anchor, w2).is_err(),
                "reject short input/output buffers"
            );
        }
        select(rows, anchor, w2).unwrap();
        let actual = indices(b, &output, rows as usize);
        let mut expected = Vec::with_capacity(rows as usize);
        let mut previous = anchor;
        for row in 0..rows as usize {
            let (mut best, mut chosen) = (f64::NEG_INFINITY, 0);
            for token in 0..f.vocab {
                let mut value = f64::from(logits[row * vocab + token as usize]);
                for column in 0..f.rank {
                    value += weight(model, f.first, f.rank, previous, column)
                        * weight(model, f.second, f.rank, token, column);
                }
                assert!(value.is_finite(), "finite independent scores");
                if value > best {
                    best = value;
                    chosen = token;
                }
            }
            expected.push(chosen as i32);
            previous = chosen;
        }
        assert_eq!(
            actual, expected,
            "chain matches independent FP64 formula and ties"
        );

        // Single-row chains replay the old host-mediated dependency on the
        // same rows; the full chain must choose identical IDs.
        let mut previous = anchor;
        for (row, &want) in expected.iter().enumerate() {
            // Views fall on 16-byte offsets: each row gets its own tensor.
            let row_input = tensor(b, &bytes[row * vocab * 4..(row + 1) * vocab * 4]);
            b.dspark_markov_chain(
                &mut output,
                &mut key,
                &row_input,
                w1,
                w2,
                f.vocab,
                f.rank,
                previous,
                1,
            )
            .unwrap();
            let selected = indices(b, &output, 1)[0];
            assert_eq!(selected, want, "scalar and chained choices agree");
            previous = selected as u32;
        }
    }
}

#[test]
fn chained_selection_matches_an_fp64_oracle_and_its_scalar_replay() {
    let mut b = Backend::init().unwrap();
    let mut model = Vec::new();
    let mut fixtures = Vec::new();
    let mut random = 0x251f_307bu32;
    for rank in [32u32, 96, 256] {
        for vocab in [1u32, 17, 257, 523] {
            let bytes = vocab as usize * rank as usize / 32 * 34;
            let offset = model.len();
            model.resize(offset + 2 * bytes, 0);
            fixtures.push(Fixture {
                vocab,
                rank,
                first: offset as u64,
                second: (offset + bytes) as u64,
            });
            for block in (offset..model.len()).step_by(34) {
                // IEEE FP16 2^-5.
                model[block..block + 2].copy_from_slice(&0x2800u16.to_le_bytes());
                for column in 0..32 {
                    random ^= random << 13;
                    random ^= random >> 17;
                    random ^= random << 5;
                    model[block + 2 + column] = ((random % 31) as i32 - 15) as i8 as u8;
                }
            }
        }
    }
    // A separate zero-weight table tests global equal-score tie ordering.
    let zero = model.len();
    model.resize(zero + 2 * 523 * 34, 0);
    fixtures.push(Fixture {
        vocab: 523,
        rank: 32,
        first: zero as u64,
        second: (zero + 523 * 34) as u64,
    });
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&model).unwrap();
    file.flush().unwrap();
    let map = MappedFile::open(file.path()).unwrap();
    let len = model.len() as u64;
    b.set_support(
        map,
        len.next_multiple_of(256),
        &[(0, len, "Markov fixture")],
    )
    .unwrap();
    for fixture in &fixtures {
        run(&b, &model, fixture);
    }
}
