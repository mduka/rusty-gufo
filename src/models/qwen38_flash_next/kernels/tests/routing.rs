//! The MoE routing chain on the device (router top-k, expert counts, compact
//! buckets) against a host reference, and the launch contract's refusal of
//! counts that do not belong to the current expert IDs:
//! cargo test -p gufo-qwen38-flash-next-kernels --features gpu

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use gufo_hip::{DeviceIndices, IndexSpec, LaunchError, Stream};
use gufo_qwen38_flash_next_kernels as k;

const TOKENS: u32 = 37;
const EXPERTS: u32 = 64;
const USED: u32 = 6;
/// Router rows carry the shared-expert gate logit after the experts.
const STRIDE: u32 = EXPERTS + 1;

fn logits() -> Vec<f32> {
    (0..TOKENS * STRIDE)
        .map(|i| (i.wrapping_mul(2_654_435_761) % 1000) as f32 / 100.0 - 5.0)
        .collect()
}

/// Top-k of the softmax, highest probability first, lowest ID on ties.
fn reference_ids(logits: &[f32]) -> Vec<u32> {
    let mut ids = Vec::new();
    for t in 0..TOKENS as usize {
        let row = &logits[t * STRIDE as usize..][..EXPERTS as usize];
        let mut order: Vec<u32> = (0..EXPERTS).collect();
        order.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b)));
        ids.extend(&order[..USED as usize]);
    }
    ids
}

struct Routing {
    ids: DeviceIndices,
    counts: DeviceIndices,
}

fn route(stream: &Stream, logits: &[f32]) -> (Routing, gufo_hip::DeviceBuffer<f32>) {
    let slots = (TOKENS * USED) as usize;
    let mut device_logits = stream.alloc::<f32>(logits.len()).unwrap();
    stream.upload(logits, device_logits.as_mut_slice()).unwrap();
    let mut ids = stream
        .alloc_indices_with(slots, IndexSpec::below(EXPERTS).padded())
        .unwrap();
    let mut weights = stream.alloc::<f32>(slots).unwrap();
    k::router_top_k(
        stream,
        device_logits.as_slice(),
        STRIDE,
        ids.view_mut(),
        weights.as_mut_slice(),
        TOKENS,
        EXPERTS,
        USED,
    )
    .unwrap();
    let mut counts = stream
        .alloc_indices_with(EXPERTS as usize, IndexSpec::below(slots as u32 + 1))
        .unwrap();
    k::expert_counts(stream, ids.view(), counts.view_mut(), TOKENS, EXPERTS, USED).unwrap();
    (Routing { ids, counts }, weights)
}

fn download(stream: &Stream, indices: &DeviceIndices) -> Vec<u32> {
    let mut host = vec![0u32; indices.len()];
    stream.download(indices.view().values(), &mut host).unwrap();
    host
}

#[test]
fn routing_matches_the_host_reference() {
    let stream = Stream::new().unwrap();
    let logits = logits();
    let (routing, weights) = route(&stream, &logits);
    let ids = download(&stream, &routing.ids);
    assert_eq!(ids, reference_ids(&logits));
    let mut host_weights = vec![0f32; ids.len()];
    stream
        .download(weights.as_slice(), &mut host_weights)
        .unwrap();
    for token in host_weights.chunks(USED as usize) {
        assert!((token.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    let counts = download(&stream, &routing.counts);
    for e in 0..EXPERTS {
        assert_eq!(
            counts[e as usize],
            ids.iter().filter(|&&id| id == e).count() as u32
        );
    }

    // Compact buckets: every (token, slot) once in its expert's 16-padded
    // bucket, padding -1.
    let slots = (TOKENS * USED) as usize;
    let rows = k::routed_compact_rows(slots, EXPERTS as usize);
    let signed = |bound: u32| IndexSpec::below(bound).padded();
    let mut bounds = stream
        .alloc_indices_with(EXPERTS as usize + 1, IndexSpec::below(rows as u32 + 1))
        .unwrap();
    let mut cursors = stream.alloc::<i32>(EXPERTS as usize).unwrap();
    let mut rows_token = stream.alloc_indices_with(rows, signed(TOKENS)).unwrap();
    let mut rows_slot = stream
        .alloc_indices_with(rows, signed(slots as u32))
        .unwrap();
    k::routed_compact(
        &stream,
        routing.ids.view(),
        routing.counts.view(),
        bounds.view_mut(),
        cursors.as_mut_slice(),
        rows_token.view_mut(),
        rows_slot.view_mut(),
        TOKENS,
        USED,
        EXPERTS,
        TOKENS * USED,
    )
    .unwrap();
    let bounds = download(&stream, &bounds);
    let rows_token = download(&stream, &rows_token);
    let rows_slot = download(&stream, &rows_slot);
    let mut seen = vec![false; slots];
    for e in 0..EXPERTS as usize {
        let (begin, end) = (bounds[e] as usize, bounds[e + 1] as usize);
        assert_eq!(end - begin, counts[e].div_ceil(16) as usize * 16);
        for row in begin..end {
            let slot = rows_slot[row];
            if slot == u32::MAX {
                assert_eq!(rows_token[row], u32::MAX);
                continue;
            }
            assert_eq!(ids[slot as usize], e as u32);
            assert_eq!(rows_token[row], slot / USED);
            assert!(!std::mem::replace(&mut seen[slot as usize], true));
        }
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn counts_must_belong_to_the_current_ids() {
    let stream = Stream::new().unwrap();
    let logits = logits();
    let (mut routing, _) = route(&stream, &logits);
    let slots = (TOKENS * USED) as usize;
    let rows = k::routed_compact_rows(slots, EXPERTS as usize);
    let mut bounds = stream
        .alloc_indices_with(EXPERTS as usize + 1, IndexSpec::below(rows as u32 + 1))
        .unwrap();
    let mut cursors = stream.alloc::<i32>(EXPERTS as usize).unwrap();
    let mut rows_token = stream
        .alloc_indices_with(rows, IndexSpec::below(TOKENS).padded())
        .unwrap();
    let mut rows_slot = stream
        .alloc_indices_with(rows, IndexSpec::below(slots as u32).padded())
        .unwrap();
    let mut compact = |ids: &DeviceIndices, counts: &DeviceIndices| {
        k::routed_compact(
            &stream,
            ids.view(),
            counts.view(),
            bounds.view_mut(),
            cursors.as_mut_slice(),
            rows_token.view_mut(),
            rows_slot.view_mut(),
            TOKENS,
            USED,
            EXPERTS,
            TOKENS * USED,
        )
    };
    let provenance = |r: Result<(), LaunchError>| matches!(r, Err(LaunchError::Provenance { .. }));

    // Counts of other IDs.
    let (other, _) = route(&stream, &logits);
    assert!(provenance(compact(&routing.ids, &other.counts)));
    // Counts uploaded from the host, even correct ones.
    let host_counts = download(&stream, &routing.counts);
    let mut uploaded = stream
        .alloc_indices_with(EXPERTS as usize, IndexSpec::below(slots as u32 + 1))
        .unwrap();
    stream.upload_indices(&host_counts, &mut uploaded).unwrap();
    assert!(provenance(compact(&routing.ids, &uploaded)));
    // IDs routed again after the counts.
    compact(&routing.ids, &routing.counts).unwrap();
    let mut device_logits = stream.alloc::<f32>(logits.len()).unwrap();
    stream
        .upload(&logits, device_logits.as_mut_slice())
        .unwrap();
    let mut weights = stream.alloc::<f32>(slots).unwrap();
    k::router_top_k(
        &stream,
        device_logits.as_slice(),
        STRIDE,
        routing.ids.view_mut(),
        weights.as_mut_slice(),
        TOKENS,
        EXPERTS,
        USED,
    )
    .unwrap();
    assert!(provenance(compact(&routing.ids, &routing.counts)));
}

#[test]
fn non_finite_router_logits_leave_slots_inactive() {
    let stream = Stream::new().unwrap();
    let mut logits = logits();
    logits[..STRIDE as usize].fill(f32::NAN);
    logits[STRIDE as usize..2 * STRIDE as usize].fill(f32::INFINITY);
    let (routing, _) = route(&stream, &logits);
    let ids = download(&stream, &routing.ids);
    // Both rows route no slot (padding, -1); the counts skip them.
    let (inactive, routed) = ids.split_at(2 * USED as usize);
    assert!(inactive.iter().all(|&id| id == u32::MAX));
    assert!(routed.iter().all(|&id| id < EXPERTS));
    let counts = download(&stream, &routing.counts);
    assert_eq!(counts.iter().sum::<u32>() as usize, routed.len());
}
