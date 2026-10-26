//! Loads the real Qwen3-ASR checkpoint with both weight placements:
//! cargo test -p gufo-qwen3-asr-kernels --features gpu --test checkpoint -- --ignored --nocapture

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use gufo_hip::{Bf16, DeviceTensor, Event, MappedFile, Placement, Stream, WeightLoader};
use gufo_qwen3_asr_kernels as k;

const SHARDS: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];

fn model_root() -> PathBuf {
    std::env::var_os("QWEN3_ASR_MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(
                ".cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/\
             7278e1e70fe206f11671096ffdd38061171dd6e5",
            )
        })
}

/// Absolute byte ranges of every tensor, from the safetensors header (an
/// 8-byte length followed by JSON with `"data_offsets":[begin,end]` entries).
fn tensor_ranges(file: &MappedFile) -> Vec<(usize, usize)> {
    let bytes = file.bytes();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();
    let data = 8 + header_len;
    header
        .split("\"data_offsets\":[")
        .skip(1)
        .map(|rest| {
            let list = &rest[..rest.find(']').unwrap()];
            let (begin, end) = list.split_once(',').unwrap();
            let begin: usize = begin.trim().parse().unwrap();
            let end: usize = end.trim().parse().unwrap();
            (data + begin, end - begin)
        })
        .collect()
}

fn load(
    stream: &Stream,
    files: &[Arc<MappedFile>],
    placement: Placement,
) -> (Vec<DeviceTensor<Bf16>>, f64) {
    let start = Instant::now();
    let mut loader = WeightLoader::new(stream, placement);
    let mut pending = Vec::new();
    for file in files {
        for (offset, bytes) in tensor_ranges(file) {
            pending.push(loader.add::<Bf16>(file, offset, bytes / 2).unwrap());
        }
    }
    let weights = loader.finish().unwrap();
    let seconds = start.elapsed().as_secs_f64();
    (
        pending.into_iter().map(|p| weights.get(p)).collect(),
        seconds,
    )
}

#[test]
#[ignore = "needs the Qwen3-ASR checkpoint (QWEN3_ASR_MODEL_ROOT or the Hugging Face cache)"]
fn checkpoint_loads_identically_with_every_placement() {
    let root = model_root();
    let files: Vec<Arc<MappedFile>> = SHARDS
        .iter()
        .map(|s| MappedFile::open(root.join(s)).unwrap())
        .collect();
    let total: usize = files
        .iter()
        .map(|f| tensor_ranges(f).iter().map(|r| r.1).sum::<usize>())
        .sum();
    let stream = Stream::new().unwrap();

    let mut loads = Vec::new();
    for placement in [Placement::DirectCopy, Placement::CachedCopy, Placement::Map] {
        let (tensors, seconds) = load(&stream, &files, placement);
        assert_eq!(tensors.len(), 708);
        loads.push((placement, tensors, seconds));
    }

    let ranges: Vec<(usize, usize, usize)> = files
        .iter()
        .enumerate()
        .flat_map(|(index, file)| {
            tensor_ranges(file)
                .into_iter()
                .map(move |(o, b)| (index, o, b))
        })
        .collect();
    let mut loaded = Vec::new();
    for (index, (file, offset, bytes)) in ranges.into_iter().enumerate() {
        let expected = &files[file].bytes()[offset..offset + bytes];
        loaded.resize(bytes, 0u8);
        for (placement, tensors, _) in &loads {
            stream
                .download_bytes(tensors[index].view(), &mut loaded)
                .unwrap();
            assert!(
                loaded == expected,
                "{placement:?} tensor at {offset} differs from the file"
            );
        }
    }
    let gib = total as f64 / f64::from(1u32 << 30);
    for (placement, _, seconds) in &loads {
        println!(
            "{placement:?}: {gib:.2} GiB in {seconds:.3} s ({:.2} GiB/s)",
            gib / seconds
        );
    }

    // Kernels read the largest tensor (the embedding table) from each
    // placement: same result, and the read bandwidth the placement allows.
    let largest = (0..loads[0].1.len())
        .max_by_key(|&i| loads[0].1[i].len())
        .unwrap();
    let count = loads[0].1[largest].len();
    let mut widened = stream.alloc::<f32>(count).unwrap();
    let mut reference: Option<Vec<f32>> = None;
    for (placement, tensors, _) in &loads {
        let input = tensors[largest].view();
        k::bf16_to_f32(&stream, input, widened.as_mut_slice(), count).unwrap();
        let (start, end) = (Event::timed().unwrap(), Event::timed().unwrap());
        stream.record(&start).unwrap();
        for _ in 0..5 {
            k::bf16_to_f32(&stream, input, widened.as_mut_slice(), count).unwrap();
        }
        stream.record(&end).unwrap();
        end.synchronize().unwrap();
        let ms = Event::elapsed_ms(&start, &end).unwrap() / 5.0;
        let mut sample = vec![0.0f32; 1 << 20];
        stream
            .download(widened.slice(count - sample.len()..count), &mut sample)
            .unwrap();
        match &reference {
            None => reference = Some(sample),
            Some(expected) => assert!(sample == *expected, "{placement:?} kernel result differs"),
        }
        let read_gb = (count * 2) as f64 / 1e9;
        println!(
            "{placement:?}: bf16_to_f32 over {count} elements: {ms:.2} ms, reads {:.1} GB/s",
            read_gb / (f64::from(ms) / 1e3)
        );
    }
}
