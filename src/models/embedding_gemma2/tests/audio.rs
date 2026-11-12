//! Host audio preprocessing: the features of 16-kHz samples bit for bit
//! against the C++ extractor, and the decoder against librosa
//! (`audio_processor_test.cpp`, `audio_decoder_test.cpp`):
//! cargo test -p gufo-embedding-gemma2 --test audio

#![allow(clippy::unwrap_used)]

use std::io::Cursor;

use gufo_core::json::{self, Value};
use gufo_embedding_gemma2::audio_decoder::{AudioDecodeError, decode_audio};
use gufo_embedding_gemma2::audio_processor::{AudioProcessError, MAX_AUDIO_SAMPLES, process_audio};
use sha2::{Digest, Sha256};

fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    json::parse(&std::fs::read(&path).unwrap()).unwrap()
}

/// The waveforms of `audio_processor_test.cpp`.
fn waveform(count: usize, kind: &str) -> Vec<f32> {
    (0..count)
        .map(|i| match kind {
            "triangle" => {
                let phase = (i % 80) as i32;
                (phase.min(80 - phase) - 20) as f32 / 64.0
            }
            "noise" => ((((i * 73 + i * i * 19) % 1024) as i32) - 512) as f32 / 1024.0,
            _ => 0.0,
        })
        .collect()
}

/// Recorded with tests/reference/cpp_audio_features.cpp from the C++
/// release build: the features match bit for bit.
#[test]
fn features_match_the_cpp_extractor_bit_for_bit() {
    let cases = fixture("audio_features_cpp.json");
    let cases = cases.find("cases").unwrap().items();
    assert_eq!(cases.len(), 15);
    for case in cases {
        let name = case.member_str("name", "");
        let features = process_audio(
            &waveform(case.member_size("samples", 0), &case.member_str("kind", "")),
            case.member_size("padding_samples", 0),
            None,
        )
        .unwrap();
        assert_eq!(
            features.mask.len(),
            case.member_size("frames", 0),
            "{name} frames"
        );
        assert_eq!(
            features.soft_tokens(),
            case.member_size("soft_tokens", 0),
            "{name} tokens"
        );
        let mask: Vec<u8> = case
            .find("mask")
            .unwrap()
            .items()
            .iter()
            .map(|v| v.as_size(0) as u8)
            .collect();
        assert_eq!(features.mask, mask, "{name} mask");
        let bytes: Vec<u8> = features
            .values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let hash: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            hash,
            case.member_str("features_sha256", ""),
            "{name} features"
        );
    }
}

#[test]
fn invalid_waveforms_and_cancellation_are_refused() {
    for samples in [
        Vec::new(),
        vec![0.0; MAX_AUDIO_SAMPLES + 1],
        vec![f32::NAN],
        vec![f32::INFINITY],
    ] {
        assert!(matches!(
            process_audio(&samples, 0, None),
            Err(AudioProcessError::Invalid(_))
        ));
    }
    let cancelled = || true;
    assert_eq!(
        process_audio(&vec![0.0; 16000], 0, Some(&cancelled)),
        Err(AudioProcessError::Cancelled)
    );
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

/// librosa's waveforms (libsndfile, the channel mean, SOXR_HQ). 16-kHz PCM
/// WAV and FLAC decode to the same samples, MP3 and Ogg within float
/// rounding; resampled cases match only approximately (docs/development/PARITY.md).
#[test]
fn decoded_waveforms_match_librosa() {
    let cases = fixture("audio_decoder_golden.json");
    let mut failures = Vec::new();
    for case in cases.find("cases").unwrap().items() {
        let name = case.member_str("name", "");
        let bytes = hex(&case.member_str("encoded_hex", ""));
        let expected: Vec<f64> = case
            .find("waveform")
            .unwrap()
            .items()
            .iter()
            .map(|v| v.as_double(0.0))
            .collect();
        let actual = decode_audio(&bytes, None).unwrap_or_else(|e| panic!("{name}: {e}"));
        let worst = actual
            .iter()
            .zip(&expected)
            .map(|(&a, &e)| (f64::from(a) - e).abs())
            .fold(0.0f64, f64::max);
        let energy: f64 = expected.iter().map(|e| e * e).sum();
        let error: f64 = actual
            .iter()
            .zip(&expected)
            .map(|(&a, &e)| (f64::from(a) - e).powi(2))
            .sum();
        let relative = error / energy.max(1e-30);
        println!(
            "{name}: {} samples (librosa {}), worst error {worst}, relative error energy {relative}",
            actual.len(),
            expected.len()
        );
        // 16-kHz PCM WAV and FLAC decode to the same samples; the other
        // cases resample or decode a lossy codec with other libraries.
        let (exact, bound) = match name.as_str() {
            "WAV_16000_1_801" | "FLAC_16000_1_801" => (true, 0.0),
            // nanomp3 and lewton decode like mpg123 and libvorbis.
            _ if name.starts_with("MP3") || name.starts_with("OGG") => (false, 1e-9),
            // rubato against SOXR_HQ: the stereo fixtures carry content near
            // the new 8-kHz Nyquist frequency, which the two filters'
            // transition bands pass differently.
            _ => (false, 0.1),
        };
        let ok = actual.len() == expected.len()
            && if exact {
                worst < 2e-6
            } else {
                relative < bound
            };
        if !ok {
            failures.push(name.clone());
        }
        // A truncated container fails, and so does a cancelled decode.
        assert!(
            decode_audio(&bytes[..20], None).is_err(),
            "{name} truncated"
        );
        let cancelled = || true;
        assert!(
            decode_audio(&bytes, Some(&cancelled)).is_err(),
            "{name} cancelled"
        );
        assert_eq!(
            decode_audio(&bytes, None).unwrap().len(),
            actual.len(),
            "{name} recovers"
        );
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// 16-bit PCM WAV at 16 kHz is the exact path: integer samples scaled by
/// 2^-15, stereo averaged in f32 like libsndfile.
#[test]
fn pcm_wav_at_16_khz_decodes_exactly() {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut bytes = Vec::new();
    {
        let mut writer = hound::WavWriter::new(Cursor::new(&mut bytes), spec).unwrap();
        for i in 0..1000i32 {
            writer
                .write_sample((i * 37 % 65536 - 32768) as i16)
                .unwrap();
            writer
                .write_sample((i * 91 % 65536 - 32768) as i16)
                .unwrap();
        }
        writer.finalize().unwrap();
    }
    let decoded = decode_audio(&bytes, None).unwrap();
    assert_eq!(decoded.len(), 1000);
    for (i, &sample) in decoded.iter().enumerate() {
        let i = i as i32;
        let left = f32::from((i * 37 % 65536 - 32768) as i16) * (1.0 / 32768.0);
        let right = f32::from((i * 91 % 65536 - 32768) as i16) * (1.0 / 32768.0);
        assert_eq!(sample.to_bits(), ((left + right) / 2.0).to_bits());
    }
    assert_eq!(
        decode_audio(b"not audio at all", None),
        Err(AudioDecodeError::Invalid(
            "embedding audio supports WAV, FLAC, Ogg and MP3".into()
        ))
    );
    assert!(decode_audio(&[], None).is_err());
}

/// A CBR MP3 with an ID3v2 tag and no Xing/Info header (Gufo #828): libsndfile
/// estimates 9456 frames from the file size, but 9216 decode, which librosa
/// returns. The decoder keeps the frames that decode, which nanomp3 decodes
/// like mpg123.
#[test]
fn an_mp3_without_a_xing_header_keeps_its_decoded_samples() {
    let mp3 = hex(&[
        "494433040000000000325449543200000006000003636c697000545353450000000e0000034c61766636312e",
        "372e3130330000000000000000000000fff348c4001c788e6c15583000255af54ea9d53aa754ea9d77b136be",
        "b088985c84e868e0139b627389b4a68118005982f02823107221c961326030180c2c9a60e02008020e89c1fe",
        "51d39d3e7397f39cbfbba7dd9404c1f3eb0f82018940183ef93040e65dfddcb83e0f83e1f04010040100183e",
        "0f83e1f04010041c91383e2009208100681060581dbe7f9832150101fff348c41120a1f66c2d9da0009d0c80",
        "3e83a62313e6bb25460d1546c41ea6118bc70c8c2062e081a01e06185818a620a0b2700c96201a0c5d0d982d",
        "182d08523fe25110547285cc2e6ffc991cd1cd2688b116fff2648a9013222c458c7fff2e974c8bc5e44ba5d4",
        "bffff2f17912e9741a0a82acffe582a1a2dffffff55aaaa000009208a10fe7fff40ce94ca0b6f0c0fff348c4",
        "111998762637de40003c050c094190c204058c208334c5b8ef4e5d0c18fd94ff0caecb88c7585acc4d091cc0",
        "0c3f4c1c808cc0380053b958a1e8042ca1adb73339bbf62ff5bffef27757affde5efe4d6eecfb9bb7f65c43e",
        "ea8feeeefd3d55ffffd4a9414c2418c806ccd11ccddb8d0aacc0650470c0bc08b4c1d31950ca74feb8d64b21",
        "28c1840784c0d501fff348c42d23e3ba0400dfc470c0c28c0d1cc10f04dcc0ef0120c0500054c01f00003000",
        "d4ce6f24929a6dfeb5d5a19a170f8f417bff5cf409bd4d47ab547ab5f4df4f8f57ab57264c52746dfd5a1726",
        "d8debab57e0f5daa4a0d07a3d9e0b26e9ef46dabfebb63faebf0d3a8c56f76f5800011fbffe4ad610ba63c00",
        "28a002e60060088601d80fe603901b8606b83626fff348c4202103be0de8ff44a80ed099464b5de3469f10cd",
        "8613e846a6acac27076146699f465285e61700c100da0315db107f23785fff7aebab53264a8afff9687db1b7",
        "db1ebaebb7f57abd75cdae37a7fae4cfbe7daad5d5eb93e7da9ae982cf9f6fbd1aadffa7afd75f8396b53a76",
        "15a0000c0667f2eff6858c160030b8000207868550c27cc86020cf9084e34138fff348c41f1f83be15f4efca",
        "88cbb305f0d8c802a8c28b03c4c0b90504c0460ef8c00905305004b1d005080001634f5c30081a3bbd75d77c",
        "fae3bd35d31946df3fdb7affbefebabd7eb933fdfff334764cdb55a3747f7fa7a3d74ddad933ff468dd3fd7e",
        "3bd7fc4256fde9ea80047e3dffaaed2124c18031e50c92933f10c03601f4c06f0408c11a0c44c66d7ddccdbc",
        "fff348c42418e0761948d7f866100cc1050318c8459397ff8ce8b233407cc32000c00aba7568ebd60d6dbd9b",
        "aa3bfbd3726e429e9e8f713ead5a7aad4e8d0bb33bb937b76aaffdbfeefcfeee95ffffe5235f4b840e1081c0",
        "90ccc1a180c432b4c84550d2ff64c60e98b4ccd1149cc1ac06b00df8ee03cffc40d4895033f0400388605810",
        "1bf8a5066c8b9713faf53d5afff348c4431a18de1400efea80b7db3ffda259f45d53d9b53ee277d3affce15b",
        "37a1766cdf5bdab706d777aadff7fee47d3d0a8000091427f9fff71eb5b0bc4bf23c2303150ca61334f0d0ee",
        "aa531f506333428c101306d00b63208d4314e7d3040d50a04c4201a8e3d71095dbbcfead7726cfd85acf53d9",
        "fee26ad796d1f3caffede85dabb117faa95fbf7fe793f4f5fff348c45d17487621f4e7fa805547ffffd67051",
        "38c20cd054d790e6f4c04301c0c08403acc14b0b50c7527538cf9f0ec8c1440274c02101f8c112092cc06903",
        "34c0610098c0160014b7cae5faa20c57ed5d35db7db1bf4dabbd391b98bfb16f66cddbbb576eeaa4d794f925",
        "daa9e45df7fffebb91efffffe5771138c380104026220d8c0f194c302c0c69584ceefe8cfff348c4821a00da",
        "1808cfc46e4b99e50c9fa1538c16d069c0de6d103d56080d526a034381c068900881e1c609cc6608a17d06da",
        "bedeadb5d49fffd752b56df79cb7a3de4d4ea6f4ff9c55dd97b77ef4dcdbdabbff23feefd89bf727a1ffffec",
        "daf8439a3b0b01c06120c4d06cc84108cf5234e2a640cb693ba0d82608f0c29104fcc0e10534c0be0ea0c03f",
        "0500c01bfff348c49d1a71561400efea8001142800920daf074dfc02038a7d756a6bbe6c5bffe99f6d76de32",
        "d5ff793bbb8a69ea9e2f7a74dfceeedcdbc994bbef67fbbf3e8fa5800002200408fd7ffd57d94c4b94607041",
        "8382261d1118d88a66f4d1c63e6631ca93266a30512608f80982875359fc0c80a831d074040e4c16bd0eda13",
        "19ebf722edec5de8d3d5ffbc9af56bfffff348c4b61a31561400efca886a3e9bfb3d56aac4dff7b3fdffbbff",
        "5a4c414d45332e3130308000044acb3fbfff943d69ce0a000cc0040050c0120050c02a0010c06b0334c0d904",
        "bcc2510c7cc2fc5adcc15908d4c01100b8c0620098c093005cc0540114c05c0110c01800300c00c3d6d6dc71",
        "1b6fdabb77e77f693b91d4e67febd7716ff329fa6feddf55aab0addffff348c4d017286e221ce7f8807b3fdf",
        "f9e45fd3ff7e98218730571822888f9813847573027031311508d312b19b302842933a653a31302223a8436f",
        "2a2c61a6595581b8ce201551058060604ee818ec26173807a1360195db406b61c793c440be5a032c1480a13c",
        "0ca61b030e8afe4fa8ae5c580911018b81806040b008068186c0bf9a27374202416024000041c028fff348c4",
        "ed1c387626155f4800020b4100a07ff344e6eaa016dc32d00a0003bc1ea8363e1f8ffd374d4ccb74c4e41730",
        "27a13e86de2911841f20b3bffd542efb5dc62094c56a30c5902e61a02e71991a62e419affffb6fb6fc64073c",
        "708e3200388701171da3364c0ee1cc26c79ffffffffff1cf301e8730e90839e7097200892040d224c801a910",
        "21e7d5019c80404bfff348c4ff3e54163c019ea8000c90c914baa36862603998ac78bda020a6abe0a04d3990",
        "a2ac2d394b925e14c171648c3953019501204812264c2a08935a114a16688854cd2c8914ad5427815054b032",
        "581a3c0abc4a1a51e834784af586a7a223c570578883bc1576a0ef1286b50347b2a0abb11077582a75111034",
        "4c414d45332e313030aaaaaaaaaaaaaaaaaaaaaafff348c4881cf8da5c17d84800aaaaaaaaaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ]
    .concat());
    let decoded = decode_audio(&mp3, None).unwrap();
    for (index, value) in [
        (0, 0.0),
        (1500, -0.09022247046232224),
        (4600, 0.0770723968744278),
        (9215, -0.0002711241249926388),
    ] {
        println!(
            "sample {index}: {} (librosa {value})",
            decoded.get(index).copied().unwrap_or(f32::NAN)
        );
    }
    let energy: f64 = decoded.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    println!(
        "{} samples, energy {energy} (librosa 9216, 56.380433648018936)",
        decoded.len()
    );
    assert_eq!(
        decoded.len(),
        9216,
        "estimated MP3 length keeps decoded samples"
    );
    assert!(
        (energy - 56.380433648018936).abs() < 1e-4,
        "energy {energy}"
    );
    for (index, value) in [
        (0, 0.0),
        (1500, -0.09022247046232224),
        (4600, 0.0770723968744278),
        (9215, -0.0002711241249926388),
    ] {
        assert!(
            (f64::from(decoded[index]) - value).abs() < 2e-6,
            "sample {index}"
        );
    }
}
