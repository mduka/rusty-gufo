//! Device tests: cargo test -p gufo-hip --features gpu

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{
    Derivation, ErrorKind, Event, Graph, HipError, HostMemory, IndexSpec, LaunchError, MemoryType,
    PinnedBuffer, RowTable, Stream,
};

#[test]
fn pageable_round_trip_preserves_values() {
    let stream = Stream::new().unwrap();
    let values: Vec<u32> = (0..10_000).collect();
    let mut device = stream.alloc::<u32>(values.len()).unwrap();
    stream.upload(&values, device.as_mut_slice()).unwrap();
    let mut back = vec![0u32; values.len()];
    stream.download(device.as_slice(), &mut back).unwrap();
    assert_eq!(back, values);
}

#[test]
fn blocking_copies_complete_in_stream_order() {
    let stream = Stream::new().unwrap();
    let values: Vec<u32> = (0..1_000).collect();
    let mut device = stream.alloc::<u32>(values.len()).unwrap();
    stream
        .upload_blocking(&values, device.as_mut_slice())
        .unwrap();
    // A queued fill precedes the blocking read of its own bytes.
    stream
        .fill_bytes(device.as_mut_slice().range(0..10), 0)
        .unwrap();
    let mut back = vec![u32::MAX; values.len()];
    stream
        .download_blocking(device.as_slice(), &mut back)
        .unwrap();
    assert_eq!(back[..10], [0; 10]);
    assert_eq!(back[10..], values[10..]);
    assert!(
        stream
            .download_blocking(device.as_slice(), &mut back[..3])
            .is_err()
    );
}

#[test]
fn pinned_gathers_copy_spans_of_other_streams_in_order() {
    let owner = Stream::new().unwrap();
    let values: Vec<u8> = (0..=255).collect();
    let mut device = owner.alloc::<u8>(values.len()).unwrap();
    owner.upload(&values, device.as_mut_slice()).unwrap();
    let transfer = Stream::new().unwrap();
    let mut staging = PinnedBuffer::<u8>::zeroed(64).unwrap();
    let whole = device.as_slice();
    let spans = [whole.range(200..210), whole.range(0..0), whole.range(3..7)];
    let copied = transfer
        .download_bytes_across_pinned(&spans, &mut staging)
        .unwrap();
    assert_eq!(copied, 14);
    let host = staging.host().unwrap();
    assert_eq!(host[..10], values[200..210]);
    assert_eq!(host[10..14], values[3..7]);
    assert_eq!(host[14..], [0; 50]);
    // Spans longer than the staging buffer are refused before any copy.
    assert!(
        transfer
            .download_bytes_across_pinned(&[whole.range(0..65)], &mut staging)
            .is_err()
    );
    assert_eq!(
        transfer
            .download_bytes_across_pinned::<u8>(&[], &mut staging)
            .unwrap(),
        0
    );
}

#[test]
fn staged_uploads_copy_in_pieces_through_both_buffers() {
    let stream = Stream::new().unwrap();
    // Seven pieces of 24 bytes and a tail of 12, alternating between the
    // buffers; the source is released before the copies are confirmed.
    let values: Vec<f32> = (0..45).map(|i| i as f32 * 0.5 - 3.0).collect();
    let mut staging = [
        PinnedBuffer::<u8>::zeroed(24).unwrap(),
        PinnedBuffer::<u8>::zeroed(32).unwrap(),
    ];
    let mut device = stream.alloc::<f32>(values.len()).unwrap();
    {
        let host: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        stream
            .upload_bytes_staged(&host, device.as_mut_slice(), &mut staging)
            .unwrap();
    }
    let mut back = vec![0.0f32; values.len()];
    stream.download(device.as_slice(), &mut back).unwrap();
    assert_eq!(back, values);
    // The buffers hold the last two pieces once their copies completed.
    assert_eq!(staging[0].host().unwrap()[..24], f32_bytes(&values[36..42]));
    assert_eq!(staging[1].host().unwrap()[..12], f32_bytes(&values[42..]));
    // Lengths must match, and an empty copy needs no staging.
    assert!(
        stream
            .upload_bytes_staged(&[0u8; 8], device.as_mut_slice(), &mut staging)
            .is_err()
    );
    let mut empty = stream.alloc::<f32>(0).unwrap();
    stream
        .upload_bytes_staged(&[], empty.as_mut_slice(), &mut staging)
        .unwrap();
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn casts_reinterpret_whole_aligned_values_only() {
    let stream = Stream::new().unwrap();
    let mut device = stream.alloc::<f32>(4).unwrap();
    stream
        .upload(&[1.0f32, -2.0, 0.5, 3.0], device.as_mut_slice())
        .unwrap();
    let words = device.as_slice().cast::<u32>().unwrap();
    assert_eq!(words.len(), 4);
    let mut back = [0u32; 4];
    stream.download(words, &mut back).unwrap();
    assert_eq!(back[1], (-2.0f32).to_bits());
    let halves = device.as_mut_slice().cast::<gufo_hip::F16>().unwrap();
    assert_eq!(halves.len(), 8);
    let bytes = device.as_slice().cast::<u8>().unwrap();
    assert_eq!(bytes.len(), 16);
    // Three bytes are no whole f32, and byte 1 is no f32 boundary.
    assert!(bytes.range(0..3).cast::<f32>().is_none());
    let bytes = device.as_slice().cast::<u8>().unwrap();
    assert!(bytes.range(1..5).cast::<f32>().is_none());
}

#[test]
#[allow(clippy::single_range_in_vec_init)] // `views` takes arrays of ranges
fn views_share_overlapping_reads_and_keep_writes_exclusive() {
    let stream = Stream::new().unwrap();
    let mut device = stream.alloc::<u32>(16).unwrap();
    let values: Vec<u32> = (0..16).collect();
    stream.upload(&values, device.as_mut_slice()).unwrap();
    // Two overlapping reads beside a write; an empty view overlaps nothing.
    let ([first, second], [mut out, empty]) = device.views([0..4, 2..6], [8..12, 3..3]);
    assert_eq!(
        (first.len(), second.len(), out.len(), empty.len()),
        (4, 4, 4, 0)
    );
    stream.copy(second, out.reborrow()).unwrap();
    let mut back = [0u32; 16];
    stream.download(device.as_slice(), &mut back).unwrap();
    assert_eq!(&back[8..12], &[2, 3, 4, 5]);
    let panics = |views: &mut dyn FnMut()| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(views)).is_err()
    };
    assert!(
        panics(&mut || {
            let _ = device.views([0..4], [3..5]);
        }),
        "a write overlapping a read"
    );
    assert!(
        panics(&mut || {
            let _ = device.views([], [0..8, 7..9]);
        }),
        "overlapping writes"
    );
    assert!(
        panics(&mut || {
            let _ = device.views([], [15..17]);
        }),
        "a view past the end"
    );
}

#[test]
fn two_dimensional_copies_move_strided_rows_within_bounds() {
    let stream = Stream::new().unwrap();
    let values: Vec<u32> = (0..24).collect();
    let mut source = stream.alloc::<u32>(24).unwrap();
    stream.upload(&values, source.as_mut_slice()).unwrap();
    // Rows of 3 at pitch 6 into rows of 3 at pitch 4.
    let mut target = stream.alloc::<u32>(12).unwrap();
    stream.fill_bytes(target.as_mut_slice(), 0).unwrap();
    stream
        .copy_2d(source.as_slice(), 6, target.as_mut_slice(), 4, 3, 3)
        .unwrap();
    let mut back = [0u32; 12];
    stream.download(target.as_slice(), &mut back).unwrap();
    assert_eq!(back, [0, 1, 2, 0, 6, 7, 8, 0, 12, 13, 14, 0]);
    let mut host = vec![99u32; 8];
    stream
        .download_2d(source.as_slice().range(1..24), 6, &mut host, 4, 2, 2)
        .unwrap();
    assert_eq!(host, [1, 2, 99, 99, 7, 8, 99, 99]);
    // A fourth row would end past the target; a row wider than its pitch.
    assert!(
        stream
            .copy_2d(source.as_slice(), 6, target.as_mut_slice(), 4, 3, 4)
            .is_err()
    );
    assert!(
        stream
            .copy_2d(source.as_slice(), 2, target.as_mut_slice(), 4, 3, 1)
            .is_err()
    );
    assert!(
        stream
            .download_2d(source.as_slice(), 6, &mut host, 4, 3, 3)
            .is_err()
    );
}

#[test]
fn strided_uploads_place_host_rows_at_their_pitch() {
    let stream = Stream::new().unwrap();
    let host: Vec<u32> = (0..12).collect();
    let mut device = stream.alloc::<u32>(20).unwrap();
    stream.fill_bytes(device.as_mut_slice(), 0).unwrap();
    // Two rows of four, packed on the host, eight apart on the device.
    stream
        .upload_2d(&host[..], 4, device.slice_mut(0..20), 8, 4, 2)
        .unwrap();
    let mut back = vec![0u32; 20];
    stream.download(device.as_slice(), &mut back).unwrap();
    assert_eq!(&back[..4], &host[..4]);
    assert_eq!(&back[4..8], &[0; 4]);
    assert_eq!(&back[8..12], &host[4..8]);
    assert!(
        stream
            .upload_2d(&host[..], 4, device.slice_mut(0..19), 8, 4, 3)
            .is_err(),
        "the third row ends outside the target"
    );
}

#[test]
fn pinned_copies_are_fenced_until_complete() {
    let stream = Stream::new().unwrap();
    let count = 1 << 22;
    let values: Vec<f32> = (0..count).map(|i| i as f32).collect();
    let mut source = PinnedBuffer::from_slice(&values).unwrap();
    let mut device = stream.alloc::<f32>(count).unwrap();
    let mut target = PinnedBuffer::<f32>::zeroed(count).unwrap();

    stream
        .upload_async(&mut source, device.as_mut_slice())
        .unwrap();
    stream
        .download_async(device.as_slice(), &mut target)
        .unwrap();
    // No explicit synchronisation: host access waits for the fence.
    assert_eq!(target.host().unwrap(), values.as_slice());
    assert!(target.is_ready().unwrap());
    // The source may be rewritten once its upload has completed.
    source.host_mut().unwrap()[0] = -1.0;
}

/// Rows two "layers" leave in one scratch interleave into a pinned buffer
/// with one fence: the second layer overwrites the source in stream order,
/// after the first copy read it, and host access waits for both.
#[test]
fn pinned_downloads_stay_in_flight_together() {
    let stream = Stream::new().unwrap();
    let first: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let second: Vec<f32> = (100..112).map(|v| v as f32).collect();
    let mut scratch = stream.alloc::<f32>(12).unwrap();
    let mut next = stream.alloc::<f32>(12).unwrap();
    stream.upload(&first, scratch.as_mut_slice()).unwrap();
    stream.upload(&second, next.as_mut_slice()).unwrap();
    // [row][layer][4] behind two leading values.
    let mut host = PinnedBuffer::<f32>::zeroed(2 + 24).unwrap();
    let mut download = stream.begin_download(&mut host).unwrap();
    download.copy_2d(scratch.as_slice(), 4, 2, 8, 4, 3).unwrap();
    stream
        .copy(next.as_slice(), scratch.as_mut_slice())
        .unwrap();
    download.copy_2d(scratch.as_slice(), 4, 6, 8, 4, 3).unwrap();
    download.finish().unwrap();
    let mut expected = vec![0f32; 2];
    for row in 0..3 {
        expected.extend_from_slice(&first[row * 4..row * 4 + 4]);
        expected.extend_from_slice(&second[row * 4..row * 4 + 4]);
    }
    assert_eq!(host.host().unwrap(), expected.as_slice());

    // A download dropped without `finish` is fenced all the same.
    stream.upload(&first, scratch.as_mut_slice()).unwrap();
    {
        let mut download = stream.begin_download(&mut host).unwrap();
        download.copy_2d(scratch.as_slice(), 4, 0, 4, 4, 1).unwrap();
    }
    assert_eq!(&host.host().unwrap()[..4], &first[..4]);

    // Rows past the buffer and foreign sources are refused unenqueued.
    let other = Stream::new().unwrap();
    let foreign = other.alloc::<f32>(12).unwrap();
    let mut download = stream.begin_download(&mut host).unwrap();
    let error = download
        .copy_2d(scratch.as_slice(), 4, 20, 8, 4, 3)
        .unwrap_err();
    assert!(error.operation.contains("outside"), "{error}");
    let error = download
        .copy_2d(foreign.as_slice(), 4, 0, 8, 4, 3)
        .unwrap_err();
    assert!(error.operation.contains("another stream"), "{error}");
    download.finish().unwrap();
    assert!(host.is_ready().unwrap());
}

#[test]
fn copies_refuse_foreign_streams_and_length_mismatches() {
    let stream = Stream::new().unwrap();
    let other = Stream::new().unwrap();
    let mut device = stream.alloc::<u32>(4).unwrap();
    let error = other
        .upload(&[1, 2, 3, 4], device.as_mut_slice())
        .unwrap_err();
    assert!(error.operation.contains("another stream"), "{error}");
    let error = stream
        .upload(&[1, 2, 3], device.as_mut_slice())
        .unwrap_err();
    assert!(error.operation.contains("length mismatch"), "{error}");
}

#[test]
fn timed_events_measure_enqueued_work() {
    let stream = Stream::new().unwrap();
    let (start, end) = (Event::timed().unwrap(), Event::timed().unwrap());
    let mut device = stream.alloc::<u8>(64 << 20).unwrap();
    let host = vec![7u8; 64 << 20];
    stream.record(&start).unwrap();
    stream.upload(&host, device.as_mut_slice()).unwrap();
    stream.record(&end).unwrap();
    end.synchronize().unwrap();
    let ms = Event::elapsed_ms(&start, &end).unwrap();
    assert!(ms > 0.0, "{ms} ms");
}

#[test]
fn loaded_runtime_is_the_one_the_build_used() {
    assert_eq!(
        gufo_hip::runtime_version().unwrap(),
        gufo_hip::BUILD_HIP_VERSION
    );
    assert!(Stream::new().is_ok());
}

#[test]
fn devices_report_their_properties_and_errors_their_names() {
    assert!(gufo_hip::device_count().unwrap() >= 1);
    let properties = gufo_hip::device_properties(0).unwrap();
    assert!(
        properties.arch_name.starts_with("gfx1151"),
        "{properties:?}"
    );
    assert!(!properties.name.is_empty());
    assert!(properties.multiprocessor_count > 0 && properties.total_global_memory > 0);
    assert_eq!((properties.major, properties.minor), (11, 5));
    assert_eq!(properties.warp_size, 32);
    // Strix Halo is an APU.
    assert!(properties.integrated);
    let error = gufo_hip::device_properties(1 << 20).unwrap_err();
    assert_eq!(gufo_hip::error_name(error.code), "hipErrorInvalidDevice");
    assert_eq!(gufo_hip::error_name(0), "hipSuccess");
    assert!(gufo_hip::BUILD_ROCM_VERSION.split('.').count() == 3);
}

#[test]
fn frozen_tensors_are_readable_on_every_stream() {
    let (loader, reader) = (Stream::new().unwrap(), Stream::new().unwrap());
    let values: Vec<u32> = (0..4096).collect();
    let mut buffer = loader.alloc::<u32>(values.len()).unwrap();
    loader.upload(&values, buffer.as_mut_slice()).unwrap();
    let tensor = buffer.freeze(&loader).unwrap();
    assert_eq!(tensor.view().home_stream(), None);
    let mut back = vec![0u32; values.len()];
    reader.download(tensor.view(), &mut back).unwrap();
    assert_eq!(back, values);
}

#[test]
fn freeze_needs_the_home_stream_and_exclusive_memory() {
    let (home, other) = (Stream::new().unwrap(), Stream::new().unwrap());
    let error = home.alloc::<f32>(8).unwrap().freeze(&other).err().unwrap();
    assert!(error.operation.contains("does not own"), "{error}");

    let buffer = home.alloc::<f32>(8).unwrap();
    let held = std::sync::Arc::clone(buffer.allocation());
    let error = buffer.freeze(&home).err().unwrap();
    assert!(error.operation.contains("still holds"), "{error}");
    drop(held);
}

#[test]
fn allocations_report_when_their_owner_is_gone() {
    let stream = Stream::new().unwrap();
    let buffer = stream.alloc::<f32>(8).unwrap();
    let held = std::sync::Arc::clone(buffer.allocation());
    assert!(held.is_owned());
    drop(buffer);
    assert!(
        !held.is_owned(),
        "a holder must see that the buffer was released"
    );

    let tensor = stream.alloc::<f32>(8).unwrap().freeze(&stream).unwrap();
    let held = std::sync::Arc::clone(tensor.allocation());
    let clone = tensor.clone();
    drop(tensor);
    assert!(
        held.is_owned(),
        "a tensor stays owned while any clone exists"
    );
    drop(clone);
    assert!(!held.is_owned());
}

#[test]
fn byte_fills_and_raw_downloads_round_trip() {
    let stream = Stream::new().unwrap();
    let mut buffer = stream.alloc::<f32>(64).unwrap();
    stream.fill_bytes(buffer.as_mut_slice(), 0xA5).unwrap();
    stream.fill_bytes(buffer.slice_mut(0..16), 0).unwrap();
    let mut bytes = vec![0u8; 64 * 4];
    stream
        .download_bytes(buffer.as_slice(), &mut bytes)
        .unwrap();
    assert!(bytes[..64].iter().all(|&b| b == 0));
    assert!(bytes[64..].iter().all(|&b| b == 0xA5));
    let other = Stream::new().unwrap();
    assert!(other.fill_bytes(buffer.as_mut_slice(), 0).is_err());
    assert!(
        stream
            .download_bytes(buffer.as_slice(), &mut bytes[..8])
            .is_err()
    );
}

#[test]
fn device_copies_read_tensors_and_own_buffers_only() {
    let stream = Stream::new().unwrap();
    let values: Vec<u32> = (0..256).collect();
    let mut source = stream.alloc::<u32>(256).unwrap();
    stream.upload(&values, source.as_mut_slice()).unwrap();
    let mut target = stream.alloc::<u32>(256).unwrap();
    stream
        .copy(source.slice(0..128), target.slice_mut(128..256))
        .unwrap();
    let tensor = source.freeze(&stream).unwrap();
    stream
        .copy(tensor.slice(128..256), target.slice_mut(0..128))
        .unwrap();
    let mut back = vec![0u32; 256];
    stream.download(target.as_slice(), &mut back).unwrap();
    assert_eq!(&back[..128], &values[128..]);
    assert_eq!(&back[128..], &values[..128]);

    let other = Stream::new().unwrap();
    let mut foreign = other.alloc::<u32>(256).unwrap();
    let error = stream
        .copy(foreign.as_slice(), target.as_mut_slice())
        .unwrap_err();
    assert!(error.operation.contains("another stream"), "{error}");
    let error = stream
        .copy(tensor.view(), foreign.as_mut_slice())
        .unwrap_err();
    assert!(error.operation.contains("another stream"), "{error}");
    let error = stream
        .copy(tensor.slice(0..8), target.slice_mut(0..9))
        .unwrap_err();
    assert!(error.operation.contains("length mismatch"), "{error}");
}

/// A shared handle reads its buffer's bytes on another stream, also after
/// the owner dropped the buffer, and a batched cross-stream copy waits once.
#[test]
fn shared_bytes_outlive_their_owner_and_copy_in_one_batch() {
    let owner = Stream::new().unwrap();
    let transfer = Stream::new().unwrap();
    let values: Vec<u32> = (0..256).collect();
    let mut state = owner.alloc::<u32>(256).unwrap();
    owner.upload(&values, state.as_mut_slice()).unwrap();
    owner.synchronize().unwrap();
    let shared = state.share();
    assert!(shared.is_buffer(&state) && shared.len() == 256 * 4);
    drop(state);

    let mut first = transfer.alloc::<u8>(64).unwrap();
    let mut second = transfer.alloc::<u8>(128).unwrap();
    transfer
        .copy_across_all(&mut [
            (shared.slice(0..64), first.as_mut_slice()),
            (shared.slice(512..640), second.as_mut_slice()),
        ])
        .unwrap();
    drop(shared);
    let mut bytes = vec![0u8; 64];
    transfer.download(first.as_slice(), &mut bytes).unwrap();
    let words = |b: &[u8]| -> Vec<u32> {
        b.chunks(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    };
    assert_eq!(words(&bytes), &values[..16]);
    let mut bytes = vec![0u8; 128];
    transfer.download(second.as_slice(), &mut bytes).unwrap();
    assert_eq!(words(&bytes), &values[128..160]);

    // Lengths must match, and the destinations must be the stream's own.
    let foreign = owner.alloc::<u8>(64).unwrap();
    let shared = foreign.share();
    let error = transfer
        .copy_across_all(&mut [(shared.slice(0..8), first.slice_mut(0..9))])
        .unwrap_err();
    assert!(error.operation.contains("length mismatch"), "{error}");
    let mut foreign = foreign;
    let error = transfer
        .copy_across_all(&mut [(first.slice(0..8), foreign.slice_mut(0..8))])
        .unwrap_err();
    assert!(error.operation.contains("another stream"), "{error}");
}

#[test]
fn copies_across_streams_read_foreign_buffers_and_wait() {
    let owner = Stream::new().unwrap();
    let transfer = Stream::new().unwrap();
    let values: Vec<u32> = (0..256).collect();
    let mut state = owner.alloc::<u32>(256).unwrap();
    owner.upload(&values, state.as_mut_slice()).unwrap();

    let mut copy = transfer.alloc_persistent::<u32>(256).unwrap();
    transfer
        .copy_across(state.as_slice(), copy.as_mut_slice())
        .unwrap();
    // Rows 1 and 3 of a 4 x 64 layout, packed.
    let mut rows = transfer.alloc::<u32>(64).unwrap();
    transfer
        .copy_2d_across(state.slice(64..256), 128, rows.as_mut_slice(), 32, 32, 2)
        .unwrap();
    drop(state);
    let snapshot = copy.freeze(&transfer).unwrap();

    // The frozen copy restores on the owner like any tensor.
    let mut restored = owner.alloc::<u32>(256).unwrap();
    owner
        .copy(snapshot.view(), restored.as_mut_slice())
        .unwrap();
    let mut back = vec![0u32; 256];
    owner.download(restored.as_slice(), &mut back).unwrap();
    assert_eq!(back, values);
    let mut packed = vec![0u32; 64];
    transfer.download(rows.as_slice(), &mut packed).unwrap();
    assert_eq!(&packed[..32], &values[64..96]);
    assert_eq!(&packed[32..], &values[192..224]);

    // The destination must still be the stream's own.
    let error = transfer
        .copy_across(snapshot.view(), restored.as_mut_slice())
        .unwrap_err();
    assert!(error.operation.contains("another stream"), "{error}");
    let error = transfer
        .copy_across(restored.slice(0..8), rows.slice_mut(0..9))
        .unwrap_err();
    assert!(error.operation.contains("length mismatch"), "{error}");
}

#[test]
fn persistent_buffers_behave_like_pooled_ones() {
    let stream = Stream::new().unwrap();
    let values: Vec<u32> = (0..100_000).collect();
    let mut exact = stream.alloc_persistent::<u32>(values.len()).unwrap();
    stream.upload(&values, exact.as_mut_slice()).unwrap();
    let mut pooled = stream.alloc::<u32>(values.len()).unwrap();
    stream
        .copy(exact.as_slice(), pooled.as_mut_slice())
        .unwrap();
    let tensor = exact.freeze(&stream).unwrap();
    let other = Stream::new().unwrap();
    let mut back = vec![0u32; values.len()];
    other.download(tensor.view(), &mut back).unwrap();
    assert_eq!(back, values);
    let held = std::sync::Arc::downgrade(tensor.allocation());
    drop(tensor);
    gufo_hip::collect_deferred_frees();
    assert!(held.upgrade().is_none());
    let error = other
        .copy(
            pooled.as_slice(),
            other.alloc::<u32>(values.len()).unwrap().as_mut_slice(),
        )
        .unwrap_err();
    assert!(error.operation.contains("another stream"), "{error}");
}

#[test]
fn exclusive_operations_are_refused_inside_a_capture_body() {
    let stream = Stream::new().unwrap();
    let other = Stream::new().unwrap();
    let result = Graph::capture(&stream, |_| {
        // Would wait for this very capture to end.
        match other.alloc_persistent::<f32>(16) {
            Err(error) => Err(LaunchError::Hip(error)),
            Ok(_) => Ok(()),
        }
    });
    assert!(
        matches!(
            result.err(),
            Some(LaunchError::Hip(HipError {
                kind: ErrorKind::CaptureViolation,
                operation: "hipMalloc",
                ..
            }))
        ),
        "an exact allocation in a capture body is refused"
    );
    // Outside a capture it works.
    assert_eq!(other.alloc_persistent::<f32>(16).unwrap().len(), 16);
}

#[test]
fn indices_hold_only_checked_values() {
    let stream = Stream::new().unwrap();
    assert!(stream.alloc_indices(4, 0).is_err());
    let mut indices = stream.alloc_indices(8, 100).unwrap();
    assert_eq!(indices.bound(), 100);
    let mut back = vec![7u8; 32];
    stream
        .download_bytes(indices.view().values(), &mut back)
        .unwrap();
    assert!(back.iter().all(|&b| b == 0), "allocation zeroes the values");

    stream.upload_indices(&[3, 99, 0], &mut indices).unwrap();
    let error = stream.upload_indices(&[5, 100], &mut indices).unwrap_err();
    assert!(error.operation.contains("out of range"), "{error}");
    let error = stream.upload_indices(&[0; 9], &mut indices).unwrap_err();
    assert!(error.operation.contains("more indices"), "{error}");
    let mut values = vec![0u32; 8];
    stream
        .download(indices.view().values(), &mut values)
        .unwrap();
    assert_eq!(
        values,
        [3, 99, 0, 0, 0, 0, 0, 0],
        "rejected uploads write nothing"
    );
    assert_eq!(indices.slice(1..3).len(), 2);
}

#[test]
fn index_pairs_check_each_range_and_upload_in_one_copy() {
    let stream = Stream::new().unwrap();
    let specs = [IndexSpec::below(1000), IndexSpec::below(16)];
    assert!(
        stream
            .zeroed_index_pair(stream.alloc(2).unwrap(), 3, specs)
            .is_err()
    );
    assert!(
        stream
            .zeroed_index_pair(stream.alloc(2).unwrap(), 1, [specs[0], IndexSpec::below(0)])
            .is_err()
    );
    let mut pair = stream
        .zeroed_index_pair(stream.alloc(2).unwrap(), 1, specs)
        .unwrap();
    assert_eq!((pair.first().bound(), pair.second().bound()), (1000, 16));
    assert_eq!((pair.first().len(), pair.second().len()), (1, 1));
    assert_eq!(
        pair.second().values().address(),
        pair.first().values().address() + 4,
        "the ranges are adjacent"
    );

    let mut host = PinnedBuffer::<u32>::zeroed(2).unwrap();
    host.host_mut().unwrap().copy_from_slice(&[999, 15]);
    stream
        .upload_index_pair_async(&mut host, &mut pair)
        .unwrap();
    assert_eq!(pair.generation(), 1);
    // A token ID the position's bound would reject, and the reverse.
    for values in [[1000, 0], [20, 16]] {
        host.host_mut().unwrap().copy_from_slice(&values);
        let error = stream
            .upload_index_pair_async(&mut host, &mut pair)
            .unwrap_err();
        assert!(error.operation.contains("out of range"), "{error}");
    }
    let mut short = PinnedBuffer::<u32>::zeroed(1).unwrap();
    assert!(
        stream
            .upload_index_pair_async(&mut short, &mut pair)
            .is_err()
    );
    let mut values = vec![0u32; 2];
    stream
        .download(pair.first().values(), &mut values[..1])
        .unwrap();
    stream
        .download(pair.second().values(), &mut values[1..])
        .unwrap();
    assert_eq!(values, [999, 15], "rejected uploads write nothing");
}

#[test]
fn index_specs_mask_pad_and_track_their_source() {
    let stream = Stream::new().unwrap();
    // A tile map bounds only its low 16 bits; padding is `u32::MAX` only.
    let tiles_spec = IndexSpec::below(8).masked(0xFFFF);
    let mut tiles = stream.alloc_indices_with(4, tiles_spec).unwrap();
    stream
        .upload_indices(&[7, 7 | (3 << 16), 1 | (9 << 16)], &mut tiles)
        .unwrap();
    assert!(stream.upload_indices(&[8 | (1 << 16)], &mut tiles).is_err());
    assert!(stream.upload_indices(&[u32::MAX], &mut tiles).is_err());
    let mut rows = stream
        .alloc_indices_with(4, IndexSpec::below(5).padded())
        .unwrap();
    stream.upload_indices(&[4, u32::MAX, 0], &mut rows).unwrap();
    assert!(stream.upload_indices(&[5], &mut rows).is_err());
    assert!(stream.upload_indices(&[u32::MAX - 1], &mut rows).is_err());

    // Uploads bump the generation; a derivation names the source view.
    let mut ids = stream.alloc_indices(6, 8).unwrap();
    assert_eq!(ids.generation(), 0);
    stream.upload_indices(&[1, 2], &mut ids).unwrap();
    assert_eq!(ids.generation(), 1);
    let derivation = ids.slice(0..4).derivation(6);
    assert_eq!(
        derivation,
        Derivation {
            source: ids.view().values().allocation().id(),
            address: ids.view().values().address(),
            len: 4,
            generation: 1,
            tag: 6
        }
    );
    // Only kernel launches record derivations (unit tests in gufo-hip cover
    // them); uploaded values have none.
    let mut counts = stream.alloc_indices(8, 7).unwrap();
    stream.upload_indices(&[0], &mut counts).unwrap();
    assert_eq!(counts.derived(), None);
    let error = gufo_hip::launch::require_derived("counts", &counts.view(), "ids", &ids.view(), 6)
        .unwrap_err();
    assert!(matches!(error, LaunchError::Provenance { .. }), "{error}");

    // Consumers state the mask and whether they skip padding.
    let l = |r: Result<(), LaunchError>| r.map_err(|e| e.to_string());
    assert!(
        l(gufo_hip::launch::require_indices(
            &stream,
            "rows",
            &rows.view(),
            4,
            5
        ))
        .is_err()
    );
    l(gufo_hip::launch::require_index_spec(
        &stream,
        "rows",
        &rows.view(),
        4,
        5,
        u32::MAX,
        true,
        false,
    ))
    .unwrap();
    assert!(
        l(gufo_hip::launch::require_indices(
            &stream,
            "tiles",
            &tiles.view(),
            4,
            8
        ))
        .is_err()
    );
    l(gufo_hip::launch::require_index_spec(
        &stream,
        "tiles",
        &tiles.view(),
        4,
        8,
        0xFFFF,
        false,
        false,
    ))
    .unwrap();

    // A kernel may write only what the buffer's spec admits.
    let out = ids.view_mut();
    l(gufo_hip::launch::require_indices_out(
        &stream,
        "ids",
        &out,
        6,
        8,
        u32::MAX,
        false,
        false,
    ))
    .unwrap();
    assert!(
        l(gufo_hip::launch::require_indices_out(
            &stream,
            "ids",
            &out,
            6,
            9,
            u32::MAX,
            false,
            false,
        ))
        .is_err()
    );
    assert!(
        l(gufo_hip::launch::require_indices_out(
            &stream,
            "ids",
            &out,
            6,
            8,
            u32::MAX,
            true,
            false,
        ))
        .is_err(),
        "padding needs a padded buffer"
    );
    assert!(
        l(gufo_hip::launch::require_indices_out(
            &stream,
            "ids",
            &out,
            7,
            8,
            u32::MAX,
            false,
            false,
        ))
        .is_err()
    );
    // int32 kernels need values that stay non-negative.
    l(gufo_hip::launch::require_index_spec(
        &stream,
        "rows",
        &rows.view(),
        4,
        5,
        u32::MAX,
        true,
        true,
    ))
    .unwrap();
    assert!(
        l(gufo_hip::launch::require_index_spec(
            &stream,
            "tiles",
            &tiles.view(),
            4,
            8,
            0xFFFF,
            false,
            true,
        ))
        .is_err(),
        "a masked tile map may hold any high bits"
    );
    let signed_spec = IndexSpec::below(8).masked(0xFFFF).signed();
    let mut signed_tiles = stream.alloc_indices_with(2, signed_spec).unwrap();
    stream
        .upload_indices(&[7 | (0x7FFF << 16)], &mut signed_tiles)
        .unwrap();
    assert!(
        stream
            .upload_indices(&[7 | (0x8000 << 16)], &mut signed_tiles)
            .is_err()
    );
    l(gufo_hip::launch::require_index_spec(
        &stream,
        "tiles",
        &signed_tiles.view(),
        2,
        8,
        0xFFFF,
        false,
        true,
    ))
    .unwrap();
    assert!(
        l(gufo_hip::launch::require_indices_out(
            &stream,
            "ids",
            &ids.view_mut(),
            6,
            8,
            u32::MAX,
            false,
            true,
        ))
        .is_ok()
    );

    assert_eq!(ids.derived(), None, "checks alone record nothing");
}

mod weights {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use gufo_hip::{Bf16, MappedFile, Placement, Stream, WeightLoader};

    const FILE_BYTES: usize = 40 << 20;

    /// A weight file with a position-dependent byte pattern, removed on drop.
    struct TestFile(PathBuf);

    impl TestFile {
        fn create(directory: &Path, name: &str) -> Self {
            let path = directory.join(format!("gufo-weights-{name}-{}.bin", std::process::id()));
            let bytes: Vec<u8> = (0..FILE_BYTES as u64)
                .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
                .collect();
            std::fs::write(&path, bytes).unwrap();
            Self(path)
        }
    }

    impl Drop for TestFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn device_bytes<T: gufo_hip::DeviceCopy>(
        stream: &Stream,
        tensor: &gufo_hip::DeviceTensor<T>,
    ) -> Vec<u8> {
        let mut bytes = vec![0u8; tensor.len() * size_of::<T>()];
        stream.download_bytes(tensor.view(), &mut bytes).unwrap();
        bytes
    }

    fn round_trip(directory: &Path, name: &str, placement: Placement) {
        let test_file = TestFile::create(directory, name);
        let file = MappedFile::open(&test_file.0).unwrap();
        let (loader_stream, reader) = (Stream::new().unwrap(), Stream::new().unwrap());
        let mut loader = WeightLoader::new(&loader_stream, placement);
        // (offset, bytes): aligned start, odd offset, two chunks, file end, empty.
        let a = loader.add::<u32>(&file, 0, 1000).unwrap();
        let b = loader.add::<Bf16>(&file, 4098, 3_000_000).unwrap();
        let c = loader.add::<u8>(&file, 12_345, 20 << 20).unwrap();
        let d = loader.add::<f32>(&file, 36 << 20, (4 << 20) / 4).unwrap();
        let e = loader.add::<u32>(&file, 64, 0).unwrap();
        let weights = loader.finish().unwrap();
        let expected = |offset: usize, bytes: usize| &file.bytes()[offset..offset + bytes];

        let a = weights.get(a);
        assert_eq!(device_bytes(&reader, &a), expected(0, 4000));
        let b = weights.get(b);
        assert_eq!(device_bytes(&reader, &b), expected(4098, 6_000_000));
        let c = weights.get(c);
        assert_eq!(device_bytes(&reader, &c), expected(12_345, 20 << 20));
        let d = weights.get(d);
        assert_eq!(device_bytes(&reader, &d), expected(36 << 20, 4 << 20));
        assert!(weights.get(e).is_empty());
        match placement {
            Placement::DirectCopy | Placement::CachedCopy => assert_eq!(b.offset() % 256, 0),
            Placement::Map | Placement::HostCopy => assert_eq!(b.offset(), 4098),
        }
    }

    #[test]
    fn copies_reproduce_the_file_with_direct_io() {
        round_trip(
            Path::new(env!("CARGO_TARGET_TMPDIR")),
            "copy-direct",
            Placement::DirectCopy,
        );
    }

    #[test]
    fn copies_reproduce_the_file_on_tmpfs() {
        round_trip(Path::new("/dev/shm"), "copy-tmpfs", Placement::DirectCopy);
    }

    #[test]
    fn cached_copies_reproduce_the_file() {
        round_trip(
            Path::new(env!("CARGO_TARGET_TMPDIR")),
            "copy-cached",
            Placement::CachedCopy,
        );
    }

    #[test]
    fn mappings_reproduce_the_file() {
        round_trip(
            Path::new(env!("CARGO_TARGET_TMPDIR")),
            "map",
            Placement::Map,
        );
    }

    #[test]
    fn host_copies_reproduce_the_file() {
        round_trip(
            Path::new(env!("CARGO_TARGET_TMPDIR")),
            "host-copy",
            Placement::HostCopy,
        );
    }

    #[test]
    fn weights_must_lie_in_their_file_and_mappings_must_be_aligned() {
        let test_file = TestFile::create(Path::new(env!("CARGO_TARGET_TMPDIR")), "rules");
        let file = MappedFile::open(&test_file.0).unwrap();
        let stream = Stream::new().unwrap();
        let mut mapped = WeightLoader::new(&stream, Placement::Map);
        let error = mapped.add::<u32>(&file, 2, 4).err().unwrap();
        assert!(error.operation.contains("not aligned"), "{error}");
        let mut copied = WeightLoader::new(&stream, Placement::DirectCopy);
        assert!(copied.add::<u32>(&file, 2, 4).is_ok(), "copies realign");
        let error = copied.add::<u8>(&file, FILE_BYTES - 1, 2).err().unwrap();
        assert!(error.operation.contains("outside"), "{error}");
    }

    #[test]
    #[should_panic(expected = "another WeightLoader")]
    fn handles_of_another_loader_are_rejected() {
        let test_file = TestFile::create(Path::new(env!("CARGO_TARGET_TMPDIR")), "handles");
        let file = MappedFile::open(&test_file.0).unwrap();
        let stream = Stream::new().unwrap();
        let mut first = WeightLoader::new(&stream, Placement::DirectCopy);
        let handle = first.add::<u8>(&file, 0, 16).unwrap();
        let second = WeightLoader::new(&stream, Placement::DirectCopy)
            .finish()
            .unwrap();
        let _ = second.get(handle);
    }

    #[test]
    fn adjacent_tensors_of_one_allocation_join() {
        let test_file = TestFile::create(Path::new(env!("CARGO_TARGET_TMPDIR")), "join");
        let file = MappedFile::open(&test_file.0).unwrap();
        let stream = Stream::new().unwrap();
        let mut loader = WeightLoader::new(&stream, Placement::HostCopy);
        let a = loader.add::<u8>(&file, 256, 100).unwrap();
        let b = loader.add::<u8>(&file, 356, 50).unwrap();
        let c = loader.add::<u8>(&file, 512, 8).unwrap();
        let weights = loader.finish().unwrap();
        let (a, b, c) = (weights.get(a), weights.get(b), weights.get(c));
        let joined = a.join(&b).unwrap();
        assert_eq!((joined.offset(), joined.len()), (256, 150));
        assert_eq!(device_bytes(&stream, &joined), &file.bytes()[256..406]);
        assert!(b.join(&c).is_none(), "a gap between the tensors");
        assert!(b.join(&a).is_none(), "the wrong order");
        let mut other = WeightLoader::new(&stream, Placement::HostCopy);
        let d = other.add::<u8>(&file, 406, 8).unwrap();
        let d = other.finish().unwrap().get(d);
        assert!(joined.join(&d).is_none(), "another allocation");
        let q4 = gufo_hip::QuantTensor::new(a.clone(), gufo_hip::GgmlType::Q4K);
        let q8 = gufo_hip::QuantTensor::new(b.clone(), gufo_hip::GgmlType::Q8_0);
        assert!(q4.join(&q8).is_none(), "another format");
    }

    #[test]
    fn loaders_share_one_registration_per_file() {
        let test_file = TestFile::create(Path::new(env!("CARGO_TARGET_TMPDIR")), "shared");
        let file = MappedFile::open(&test_file.0).unwrap();
        let stream = Stream::new().unwrap();
        let load = || {
            let mut loader = WeightLoader::new(&stream, Placement::Map);
            let handle = loader.add::<u8>(&file, 0, 64).unwrap();
            loader.finish().unwrap().get(handle)
        };
        let (first, second) = (load(), load());
        assert!(Arc::ptr_eq(first.allocation(), second.allocation()));
        let held = Arc::downgrade(first.allocation());
        drop((first, second));
        gufo_hip::collect_deferred_frees();
        assert!(held.upgrade().is_none());
        // Registering again after the old registration was undone works.
        assert_eq!(load().len(), 64);
    }
}

#[test]
fn managed_memory_has_host_views_and_takes_advice() {
    let stream = Stream::new().unwrap();
    let mut managed = stream.alloc_managed::<u8>(1 << 20).unwrap();
    assert!(managed.allocation().is_host_visible());
    let attributes = managed.pointer_attributes().unwrap();
    assert!(attributes.managed, "{attributes:?}");
    managed.host_mut().unwrap().fill(7);
    managed.advise_preferred_location(0).unwrap();
    managed.prefetch(&stream, Some(0)).unwrap();
    stream.fill_bytes(managed.slice_mut(0..16), 9).unwrap();
    managed.prefetch(&stream, None).unwrap();
    let host = managed.host().unwrap();
    assert!(host[..16].iter().all(|&b| b == 9) && host[16..].iter().all(|&b| b == 7));
    let residency = managed.host_residency().unwrap().unwrap();
    assert!(residency.pages >= (1 << 20) / 4096 && residency.resident <= residency.pages);
    let other = Stream::new().unwrap();
    assert_eq!(
        managed.prefetch(&other, Some(0)).unwrap_err().kind,
        ErrorKind::Refused
    );
}

#[test]
fn pinned_host_memory_becomes_a_buffer_with_host_views() {
    let stream = Stream::new().unwrap();
    let mut pinned = HostMemory::pinned(1 << 16, true).unwrap();
    pinned.as_mut_slice().fill(3);
    let mut buffer = pinned.into_device(&stream).unwrap();
    assert!(buffer.allocation().is_host_visible());
    assert_eq!(
        buffer.pointer_attributes().unwrap().memory_type,
        MemoryType::Host
    );
    // A device write through the alias is visible to the host view.
    stream.fill_bytes(buffer.slice_mut(8..24), 5).unwrap();
    let host = buffer.host().unwrap();
    assert!(host[..8].iter().all(|&b| b == 3) && host[8..24].iter().all(|&b| b == 5));
    buffer.host_mut().unwrap()[0] = 11;
    // Host-view copies in both directions, in stream order.
    let mut device = stream.alloc::<u8>(1 << 16).unwrap();
    stream
        .copy_from_host_view(buffer.as_slice(), device.as_mut_slice())
        .unwrap();
    stream.fill_bytes(device.slice_mut(1..2), 6).unwrap();
    stream
        .copy_to_host_view(device.as_slice(), buffer.as_mut_slice())
        .unwrap();
    let host = buffer.host().unwrap();
    assert_eq!(&host[..3], &[11, 6, 3]);
    // Device memory has no host view.
    let device_only = stream.alloc::<u8>(16).unwrap();
    assert!(device_only.host().is_err() && device_only.host_residency().is_none());
    let mut target = stream.alloc::<u8>(16).unwrap();
    assert!(
        stream
            .copy_from_host_view(device_only.as_slice(), target.as_mut_slice())
            .is_err()
    );
}

#[test]
fn memory_files_register_and_report_their_residency() {
    let stream = Stream::new().unwrap();
    let bytes = 64 << 12;
    let mut file = HostMemory::memfd("gufo-hip-test", bytes).unwrap();
    assert_eq!(file.len(), bytes);
    let untouched = file.residency().unwrap();
    assert_eq!((untouched.resident, untouched.pages), (0, 64));
    file.as_mut_slice()[..4096].fill(1);
    assert_eq!(file.residency().unwrap().resident, 1);
    file.register().unwrap();
    assert_eq!(file.register().unwrap_err().kind, ErrorKind::Refused);
    let buffer = file.into_device(&stream).unwrap();
    let mut device = stream.alloc::<u8>(bytes).unwrap();
    stream
        .copy_from_host_view(buffer.as_slice(), device.as_mut_slice())
        .unwrap();
    let mut back = vec![0u8; bytes];
    stream.download(device.as_slice(), &mut back).unwrap();
    assert!(back[..4096].iter().all(|&b| b == 1) && back[4096..].iter().all(|&b| b == 0));
    assert_eq!(buffer.host_residency().unwrap().unwrap().pages, 64);
    // A pinned allocation is registered already.
    let mut pinned = HostMemory::pinned(4096, false).unwrap();
    assert_eq!(pinned.register().unwrap_err().kind, ErrorKind::Refused);
}

#[test]
fn row_tables_point_their_header_at_the_current_rows() {
    let stream = Stream::new().unwrap();
    let mut table = RowTable::<i32, 3>::new(&stream).unwrap();
    let header = |table: &RowTable<i32, 3>| {
        let mut words = [0u64; 2];
        stream.download(table.header(), &mut words).unwrap();
        words
    };
    assert_eq!(header(&table), [0, 0]);
    assert!(table.rows().is_none());
    table.set(&stream, &[[1, 2, 3], [4, 5, 6]], -7).unwrap();
    let rows = table.rows().unwrap();
    assert_eq!(
        header(&table),
        [rows.address() as u64, 2 | (u64::from(-7i32 as u32) << 32)]
    );
    let mut back = [0i32; 6];
    stream.download(rows.range(0..6), &mut back).unwrap();
    assert_eq!(back, [1, 2, 3, 4, 5, 6]);
    // Fewer rows keep the buffer; more rows move to a larger one.
    let first = rows.address();
    table.set(&stream, &[[9, 9, 9]], 0).unwrap();
    assert_eq!(table.rows().unwrap().address(), first);
    assert_eq!(header(&table)[1], 1);
    let many: Vec<[i32; 3]> = (0..100).map(|i| [i, i + 1, i + 2]).collect();
    table.set(&stream, &many, 5).unwrap();
    let grown = table.rows().unwrap();
    assert!(grown.len() >= 300);
    assert_eq!(header(&table), [grown.address() as u64, 100 | (5 << 32)]);
    table.set(&stream, &[], 0).unwrap();
    assert_eq!(header(&table)[1], 0);
    assert_eq!((table.count(), table.tag()), (0, 0));
}
