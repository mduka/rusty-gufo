//! The C++ extension's `nvfp4_test.cpp`: NVFP4 decoding against
//! hand-computed values: halved UE4M3 scales, the NaN scale code, a
//! subnormal scale, and the nibble order within a sub-block.

use gufo_nemotron::nvfp4::{BlockNvfp4, dequantize_nvfp4};

#[test]
fn nvfp4_decodes_hand_computed_values() {
    // UE4M3 scales, halved like ggml: 0x38 = 1.0 -> 0.5, 0x40 = 2.0 -> 1.0,
    // 0x7F is the NaN code and decodes to zero, 0x01 is the subnormal 2^-9.
    let mut block = BlockNvfp4 {
        d: [0x38, 0x40, 0x7F, 0x01],
        qs: [0; 32],
    };
    block.qs[0] = 0xF2; // element 0: code 2 (1.0), element 8: code 15 (-6.0)
    block.qs[1] = 0x97; // element 1: code 7 (6.0), element 9: code 9 (-0.5)
    block.qs[8] = 0x5A; // element 16: code 10 (-1.0), element 24: code 5 (3.0)
    block.qs[16] = 0x77; // zero scale
    block.qs[24] = 0x07; // element 48: code 7 (6.0)

    let mut actual = [0.0f32; 64];
    dequantize_nvfp4(&block.to_bytes(), &mut actual);
    let mut expected = [0.0f32; 64];
    expected[0] = 1.0;
    expected[8] = -6.0;
    expected[1] = 6.0;
    expected[9] = -0.5;
    expected[16] = -2.0;
    expected[24] = 6.0;
    expected[48] = 12.0 * 0.000_976_562_5;
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(a, e, "element {i} is {a}, expected {e}");
    }
}
