//! Port of `tests/cli/continuation_cache_test.cpp`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use gufo_core::text::ContinuationInputPrefix;
use gufo_serve::continuation_cache::{
    ContinuationCache, ContinuationSnapshot, ContinuationState, InputIdentity, Lease,
    SnapshotEvent, SnapshotEventAction, SnapshotEventReason, SnapshotPurpose, SnapshotSupport,
};
use gufo_serve::logging::{self, LogLevel};

struct FakeState {
    id: usize,
    value: usize,
    invalidations: Arc<Mutex<Vec<usize>>>,
}

impl ContinuationState for FakeState {
    fn invalidate(&mut self) {
        self.invalidations.lock().expect("lock")[self.id] += 1;
    }

    fn snapshot_owner(&self) -> Option<u64> {
        Some(self.id as u64)
    }
}

struct FakeSnapshot {
    value: usize,
    payload_bytes: usize,
    /// The state whose rows it borrows.
    owner: Option<u64>,
}

impl FakeSnapshot {
    fn new(value: usize) -> Arc<Self> {
        Self::sized(value, std::mem::size_of::<usize>())
    }

    fn sized(value: usize, payload_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            value,
            payload_bytes,
            owner: None,
        })
    }
}

impl ContinuationSnapshot for FakeSnapshot {
    fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    fn source(&self) -> Option<u64> {
        self.owner
    }
}

type Cache = ContinuationCache<FakeState, FakeSnapshot>;
type Events = Arc<Mutex<Vec<SnapshotEvent>>>;

const WORD: usize = std::mem::size_of::<usize>();

fn counters(n: usize) -> Arc<Mutex<Vec<usize>>> {
    Arc::new(Mutex::new(vec![0; n]))
}

fn cache(
    capacity: usize,
    invalidations: &Arc<Mutex<Vec<usize>>>,
    support: SnapshotSupport<FakeState, FakeSnapshot>,
) -> Cache {
    checkpoint_cache(capacity, 0, invalidations, support)
}

/// A cache with room for `snapshot_capacity` checkpoints.
fn checkpoint_cache(
    capacity: usize,
    snapshot_capacity: usize,
    invalidations: &Arc<Mutex<Vec<usize>>>,
    support: SnapshotSupport<FakeState, FakeSnapshot>,
) -> Cache {
    let mut next_id = 0;
    let invalidations = invalidations.clone();
    ContinuationCache::new(
        capacity,
        move || {
            let id = next_id;
            next_id += 1;
            Ok(FakeState {
                id,
                value: 0,
                invalidations: invalidations.clone(),
            })
        },
        support,
        snapshot_capacity,
    )
    .expect("cache")
}

fn restoring(capacity: usize, events: Option<&Events>) -> SnapshotSupport<FakeState, FakeSnapshot> {
    SnapshotSupport {
        restore: Some(Box::new(
            |state: &mut FakeState, snapshot: &FakeSnapshot| {
                state.value = snapshot.value;
                Ok(())
            },
        )),
        capacity_bytes: Some(Box::new(move || capacity)),
        on_event: events.map(|events| {
            let events = events.clone();
            Box::new(move |event: &SnapshotEvent| events.lock().expect("lock").push(*event))
                as Box<dyn Fn(&SnapshotEvent) + Send + Sync>
        }),
    }
}

#[test]
fn appended_images_reuse_only_compatible_prefixes() {
    let invalidations = counters(2);
    let checkpoints = checkpoint_cache(2, 4, &invalidations, restoring(1024, None));
    let text = [1, 2, 3];
    let image = [1, 2, 3, 248_056, 4];
    let appended = [1, 2, 3, 248_056, 4, 248_056, 5];
    let (a, b, ab) = (vec![10u8], vec![20u8], vec![30u8]);
    let prefix = |token_count: usize, identity: &[u8]| ContinuationInputPrefix {
        token_count,
        identity: identity.to_vec(),
    };
    let first = [prefix(3, &[])];
    let second = [prefix(3, &[]), prefix(5, &a)];
    let input = |complete, prefixes| InputIdentity { complete, prefixes };
    {
        let mut lease = checkpoints
            .acquire_with(&image, None, input(&a, &first), None, true, 0)
            .expect("acquire")
            .expect("lease");
        assert!(
            lease
                .try_reserve_snapshot(WORD, text.len(), false, SnapshotPurpose::Continuation, &[])
                .expect("reserve"),
            "image request admits a checkpoint before its first image"
        );
        lease
            .publish_snapshot(text.to_vec(), FakeSnapshot::new(7), false)
            .expect("publish");
        assert!(
            lease.has_snapshot_for(&text),
            "checkpoint lookup uses the identity at the saved position"
        );
        lease.state_mut().expect("state").value = 70;
        lease
            .commit_with(Vec::new(), None, image.to_vec())
            .expect("commit");
    }
    {
        let mut lease = checkpoints
            .acquire_with(&appended, None, input(&ab, &second), None, true, 0)
            .expect("acquire")
            .expect("lease");
        assert!(
            lease.cache_hit()
                && lease.cached_tokens() == image.len()
                && lease.state().expect("state").value == 70,
            "new image reuses the live first-image state"
        );
        assert!(
            lease
                .try_reserve_snapshot(WORD, image.len(), false, SnapshotPurpose::Continuation, &[])
                .expect("reserve"),
            "first-image snapshot admitted"
        );
        lease
            .commit_with(image.to_vec(), Some(FakeSnapshot::new(70)), Vec::new())
            .expect("commit");
    }
    {
        let mut attachments = 0;
        let mut attach = |_: &mut FakeState| {
            attachments += 1;
            Ok(())
        };
        let matching = checkpoints
            .acquire_with(
                &appended,
                None,
                input(&ab, &second),
                Some(&mut attach),
                true,
                0,
            )
            .expect("acquire")
            .expect("lease");
        let changed = checkpoints
            .acquire_with(&image, None, input(&b, &first), None, true, 0)
            .expect("acquire")
            .expect("lease");
        assert!(
            matching.cached_tokens() == image.len() && changed.cached_tokens() == text.len(),
            "simultaneous different images choose independent safe frontiers"
        );
        let (matching, changed) = (
            matching.state().expect("state"),
            changed.state().expect("state"),
        );
        assert!(
            matching.value == 70 && changed.value == 7 && matching.id != changed.id,
            "branches restore separate payloads into separate sessions"
        );
        assert_eq!(
            attachments, 2,
            "request input is reattached after restoration"
        );
    }
    {
        let removed = checkpoints
            .acquire(&image)
            .expect("acquire")
            .expect("lease");
        assert_eq!(
            removed.cached_tokens(),
            text.len(),
            "removing an image cannot restore its computed embedding state"
        );
    }
    assert_eq!(
        checkpoints.retained_snapshot_bytes(),
        2 * WORD,
        "prefix identity does not duplicate snapshot payloads"
    );

    // Without an earlier checkpoint, token agreement is reported even when
    // the image identity makes the available state unusable.
    let direct = cache(1, &counters(1), SnapshotSupport::default());
    let original = direct
        .acquire_with(&image, None, input(&a, &first), None, true, 0)
        .expect("acquire")
        .expect("lease");
    original.commit(image.to_vec()).expect("commit");
    let changed = direct
        .acquire_with(&image, None, input(&b, &first), None, true, 0)
        .expect("acquire")
        .expect("lease");
    let lookup = changed.lookup();
    assert!(
        !changed.cache_hit()
            && lookup.miss_reason == "input_changed"
            && lookup.common_prefix_tokens == image.len()
            && lookup.checkpoint_tokens == image.len(),
        "image mismatch diagnostics retain the actual token agreement"
    );
}

#[test]
fn image_identity_isolation() {
    for snapshot_mode in [false, true] {
        let invalidations = counters(2);
        let support = if snapshot_mode {
            restoring(1024, None)
        } else {
            SnapshotSupport::default()
        };
        let cache = cache(2, &invalidations, support);
        let prompt = [1, 248_056, 3];
        let (a, b) = (vec![1u8, 2], vec![1u8, 3]);
        let save = |identity: &[u8], value: usize| {
            let mut lease = cache
                .acquire_with(&prompt, None, identity.into(), None, true, 0)
                .expect("acquire")
                .expect("lease");
            assert!(!lease.cache_hit(), "different image is a cold input");
            lease.state_mut().expect("state").value = value;
            if snapshot_mode {
                assert!(
                    lease
                        .try_reserve_snapshot(
                            WORD,
                            prompt.len(),
                            false,
                            SnapshotPurpose::Continuation,
                            &[]
                        )
                        .expect("reserve"),
                    "snapshot admitted"
                );
                lease
                    .commit_with(prompt.to_vec(), Some(FakeSnapshot::new(value)), Vec::new())
                    .expect("commit");
            } else {
                lease.commit(prompt.to_vec()).expect("commit");
            }
        };
        save(&a, 10);
        save(&b, 20);
        for (identity, value) in [(&a, 10), (&b, 20)] {
            let lease = cache
                .acquire_with(&prompt, None, identity.as_slice().into(), None, true, 0)
                .expect("acquire")
                .expect("lease");
            assert!(lease.cache_hit(), "same image reuses its own prefix");
            assert_eq!(
                lease.state().expect("state").value,
                value,
                "image state cannot cross requests"
            );
            if !snapshot_mode {
                lease.commit(prompt.to_vec()).expect("commit");
            }
        }
        let text = cache.acquire(&prompt).expect("acquire").expect("lease");
        assert!(
            !text.cache_hit(),
            "literal image-pad text cannot reuse image state"
        );
    }
}

#[test]
fn cold_miss_then_exact_extension_hit() {
    let invalidations = counters(1);
    let cache = cache(1, &invalidations, SnapshotSupport::default());
    {
        let lease = cache.acquire(&[1, 2, 3]).expect("acquire").expect("lease");
        assert!(!lease.cache_hit(), "first request is a cache miss");
        assert_eq!(
            lease.lookup().miss_reason,
            "no_checkpoint",
            "cold miss is distinguished from changed input"
        );
        assert_eq!(lease.cached_tokens(), 0, "cold request reuses no tokens");
        assert_eq!(
            lease.state().expect("state").id,
            0,
            "lease exposes the opaque model state"
        );
        lease.commit(vec![1, 2, 3, 4]).expect("commit");
    }
    let extension = [1, 2, 3, 4, 5, 6];
    let lease = cache.acquire(&extension).expect("acquire").expect("lease");
    assert!(lease.cache_hit(), "exact extension reuses the slot");
    assert!(
        lease.lookup().miss_reason.is_empty(),
        "hits carry no miss reason"
    );
    assert_eq!(
        lease.cached_tokens(),
        4,
        "hit reports the complete retained prefix"
    );
    assert_eq!(
        lease.state().expect("state").id,
        0,
        "hit returns the same opaque state"
    );
    assert_eq!(
        invalidations.lock().expect("lock")[0],
        0,
        "hit does not invalidate retained state"
    );
    lease.commit(extension.to_vec()).expect("commit");
}

#[test]
fn divergence_invalidates_old_state() {
    let invalidations = counters(1);
    let cache = cache(1, &invalidations, SnapshotSupport::default());
    cache
        .acquire(&[1, 2])
        .expect("acquire")
        .expect("lease")
        .commit(vec![1, 2, 3])
        .expect("commit");
    let lease = cache.acquire(&[1, 9]).expect("acquire").expect("lease");
    assert!(!lease.cache_hit(), "divergent request is a miss");
    let lookup = lease.lookup();
    assert!(
        lookup.miss_reason == "prefix_changed"
            && lookup.common_prefix_tokens == 1
            && lookup.checkpoint_tokens == 3,
        "miss identifies the first changed token without exposing it"
    );
    assert_eq!(
        invalidations.lock().expect("lock")[0],
        1,
        "divergence invalidates the previous model state"
    );
    lease.commit(vec![1, 9]).expect("commit");
}

#[test]
fn uncommitted_lease_is_invalidated() {
    let invalidations = counters(1);
    let cache = cache(1, &invalidations, SnapshotSupport::default());
    {
        let lease = cache.acquire(&[7]).expect("acquire");
        assert!(lease.is_some(), "request acquires the slot");
    }
    assert_eq!(
        invalidations.lock().expect("lock")[0],
        1,
        "abandoned request invalidates partially computed state"
    );
    let retry = cache.acquire(&[7, 8]).expect("acquire").expect("lease");
    assert!(!retry.cache_hit(), "abandoned state is never reused");
    retry.commit(vec![7, 8]).expect("commit");
}

#[test]
fn longest_available_prefix_wins() {
    let invalidations = counters(2);
    let cache = cache(2, &invalidations, SnapshotSupport::default());
    cache
        .acquire(&[1])
        .expect("acquire")
        .expect("lease")
        .commit(vec![1, 2])
        .expect("commit");
    cache
        .acquire(&[9])
        .expect("acquire")
        .expect("lease")
        .commit(vec![9, 8, 7])
        .expect("commit");
    let lease = cache
        .acquire(&[9, 8, 7, 6])
        .expect("acquire")
        .expect("lease");
    assert!(lease.cache_hit(), "one available entry matches");
    assert_eq!(lease.cached_tokens(), 3, "longest exact prefix is selected");
    assert_eq!(
        lease.state().expect("state").id,
        1,
        "matching state is selected rather than an arbitrary slot"
    );
    lease.commit(vec![9, 8, 7, 6]).expect("commit");
}

#[test]
fn waiting_acquire_can_be_cancelled() {
    let invalidations = counters(1);
    let cache = cache(1, &invalidations, SnapshotSupport::default());
    let held = cache.acquire(&[1, 2, 3]).expect("acquire").expect("lease");
    let cancelled = AtomicBool::new(true);
    let check = || cancelled.load(Ordering::Relaxed);
    let waiter = cache
        .acquire_with(
            &[1, 2, 3, 4],
            Some(&check),
            InputIdentity::default(),
            None,
            true,
            0,
        )
        .expect("acquire");
    assert!(
        waiter.is_none(),
        "cancelled waiter does not acquire a state"
    );
    held.commit(vec![1, 2, 3]).expect("commit");
}

/// A restore prefers the available state that holds the snapshot's
/// borrowed rows; a busy owner does not block a branch into another state
///.
#[test]
fn borrowed_snapshot_prefers_available_owner() {
    let invalidations = counters(2);
    let cache = cache(2, &invalidations, restoring(1024, None));
    let owner = {
        let mut root = cache.acquire(&[1, 2, 3]).expect("acquire").expect("lease");
        root.state_mut().expect("state").value = 7;
        let owner = root.state().expect("state").id;
        assert!(
            root.try_reserve_snapshot(WORD, 3, false, SnapshotPurpose::Continuation, &[])
                .expect("reserve"),
            "reserve the root"
        );
        let snapshot = Arc::new(FakeSnapshot {
            value: 7,
            payload_bytes: WORD,
            owner: Some(owner as u64),
        });
        root.commit_with(vec![1, 2, 3], Some(snapshot), Vec::new())
            .expect("commit");
        owner
    };
    let first = cache
        .acquire(&[1, 2, 3, 4])
        .expect("acquire")
        .expect("lease");
    assert!(
        first.cache_hit()
            && first.state().expect("state").id == owner
            && first.state().expect("state").value == 7,
        "restore prefers the owner over an older unused state"
    );
    let second = cache
        .acquire(&[1, 2, 3, 5])
        .expect("acquire")
        .expect("lease");
    assert!(
        second.cache_hit()
            && second.state().expect("state").id != owner
            && second.state().expect("state").value == 7,
        "a busy owner does not block a branch into another state"
    );
    first.invalidate();
    second.invalidate();
}

#[test]
fn snapshot_can_branch_into_two_independent_state_slots() {
    let invalidations = counters(2);
    let cache = cache(2, &invalidations, restoring(1024, None));
    {
        let mut root = cache.acquire(&[1, 2, 3]).expect("acquire").expect("lease");
        root.state_mut().expect("state").value = 7;
        assert!(
            root.try_reserve_snapshot(WORD, 3, false, SnapshotPurpose::Continuation, &[])
                .expect("reserve"),
            "root snapshot reserves aggregate capacity before allocation"
        );
        root.commit_with(vec![1, 2, 3], Some(FakeSnapshot::new(7)), Vec::new())
            .expect("commit");
    }
    let mut first = cache
        .acquire(&[1, 2, 3, 4])
        .expect("acquire")
        .expect("lease");
    let mut second = cache
        .acquire(&[1, 2, 3, 5])
        .expect("acquire")
        .expect("lease");
    assert!(
        first.cache_hit() && second.cache_hit(),
        "one snapshot can satisfy two simultaneous leases"
    );
    assert_ne!(
        first.state().expect("state").id,
        second.state().expect("state").id,
        "snapshot branches use different mutable state slots"
    );
    assert!(
        first.state().expect("state").value == 7 && second.state().expect("state").value == 7,
        "both mutable states restore the root payload"
    );
    first.state_mut().expect("state").value = 8;
    second.state_mut().expect("state").value = 9;
    assert!(
        first
            .try_reserve_snapshot(WORD, 4, false, SnapshotPurpose::Continuation, &[])
            .expect("reserve"),
        "first branch reserves snapshot capacity"
    );
    assert!(
        second
            .try_reserve_snapshot(WORD, 4, false, SnapshotPurpose::Continuation, &[])
            .expect("reserve"),
        "second branch reserves snapshot capacity"
    );
    first
        .commit_with(vec![1, 2, 3, 4], Some(FakeSnapshot::new(8)), Vec::new())
        .expect("commit");
    second
        .commit_with(vec![1, 2, 3, 5], Some(FakeSnapshot::new(9)), Vec::new())
        .expect("commit");
    let root_again = cache
        .acquire(&[1, 2, 3, 6])
        .expect("acquire")
        .expect("lease");
    assert!(
        root_again.cache_hit() && root_again.cached_tokens() == 3,
        "branch commits preserve the shared root snapshot"
    );
    assert_eq!(
        root_again.state().expect("state").value,
        7,
        "branch mutation never changes the immutable root payload"
    );
    root_again.invalidate();
}

#[test]
fn byte_capacity_evicts_before_snapshot_allocation() {
    let invalidations = counters(2);
    let events: Events = Arc::default();
    let cache = cache(2, &invalidations, restoring(12, Some(&events)));
    {
        let mut root = cache.acquire(&[1, 2, 3]).expect("acquire").expect("lease");
        assert!(
            root.try_reserve_snapshot(8, 3, false, SnapshotPurpose::Continuation, &[])
                .expect("reserve"),
            "initial snapshot fits the byte budget"
        );
        assert_eq!(
            root.commit_with(vec![1, 2, 3], Some(FakeSnapshot::sized(7, 8)), Vec::new())
                .expect("commit"),
            8,
            "initial snapshot is retained"
        );
    }
    assert_eq!(
        cache.retained_snapshot_bytes(),
        8,
        "retained bytes account the initial snapshot"
    );
    let mut replacement = cache.acquire(&[9, 8, 7]).expect("acquire").expect("lease");
    assert!(
        replacement
            .try_reserve_snapshot(12, 3, false, SnapshotPurpose::Continuation, &[])
            .expect("reserve"),
        "reservation evicts stale bytes before snapshot allocation"
    );
    assert!(
        cache.retained_snapshot_bytes() == 0 && cache.reserved_snapshot_bytes() == 12,
        "eviction transfers budget from retained to reserved bytes"
    );
    {
        let events = events.lock().expect("lock");
        assert!(
            events.len() == 1
                && events[0].action == SnapshotEventAction::Removed
                && events[0].reason == SnapshotEventReason::ByteCapacity
                && events[0].snapshot_bytes == 8
                && events[0].token_count == 3,
            "byte-pressure removal reports sanitized reason and dimensions"
        );
    }
    assert_eq!(
        replacement
            .commit_with(vec![9, 8, 7], Some(FakeSnapshot::sized(9, 12)), Vec::new())
            .expect("commit"),
        12,
        "reserved replacement is retained"
    );
    assert!(
        cache.retained_snapshot_bytes() == 12 && cache.reserved_snapshot_bytes() == 0,
        "commit converts the reservation into exact retained bytes"
    );
}

#[test]
fn concurrent_reservations_cannot_overcommit_budget() {
    let invalidations = counters(2);
    let events: Events = Arc::default();
    let mut support = restoring(12, Some(&events));
    support.restore = Some(Box::new(|_: &mut FakeState, _: &FakeSnapshot| Ok(())));
    let cache = cache(2, &invalidations, support);
    let mut first = cache.acquire(&[1]).expect("acquire").expect("lease");
    let mut second = cache.acquire(&[2]).expect("acquire").expect("lease");
    assert!(
        first
            .try_reserve_snapshot(8, 1, false, SnapshotPurpose::Continuation, &[])
            .expect("reserve"),
        "first in-flight snapshot reserves bytes"
    );
    assert!(
        !second
            .try_reserve_snapshot(8, 1, false, SnapshotPurpose::Continuation, &[])
            .expect("reserve"),
        "second reservation is rejected instead of overcommitting"
    );
    assert!(
        cache.retained_snapshot_bytes() == 0 && cache.reserved_snapshot_bytes() == 8,
        "only the admitted in-flight reservation is accounted"
    );
    {
        let events = events.lock().expect("lock");
        assert!(
            events.len() == 1
                && events[0].action == SnapshotEventAction::Skipped
                && events[0].reason == SnapshotEventReason::ByteCapacity
                && events[0].snapshot_bytes == 8
                && events[0].token_count == 1,
            "capacity refusal emits a sanitized skip event"
        );
    }
    second.commit(vec![2]).expect("commit");
    first
        .commit_with(vec![1], Some(FakeSnapshot::sized(1, 8)), Vec::new())
        .expect("commit");
    assert!(
        cache.retained_snapshot_bytes() == 8 && cache.reserved_snapshot_bytes() == 0,
        "completed requests leave no leaked reservation"
    );
}

#[test]
fn impossible_reservation_preserves_retained_entries() {
    let invalidations = counters(1);
    let events: Events = Arc::default();
    let cache = cache(1, &invalidations, restoring(8, Some(&events)));
    {
        let mut root = cache.acquire(&[1, 2]).expect("acquire").expect("lease");
        assert!(
            root.try_reserve_snapshot(8, 2, false, SnapshotPurpose::Continuation, &[])
                .expect("reserve"),
            "root fills the byte budget"
        );
        root.commit_with(vec![1, 2], Some(FakeSnapshot::sized(7, 8)), Vec::new())
            .expect("commit");
    }
    {
        let mut oversized = cache.acquire(&[9]).expect("acquire").expect("lease");
        assert!(
            !oversized
                .try_reserve_snapshot(9, 1, false, SnapshotPurpose::Continuation, &[])
                .expect("reserve"),
            "snapshot larger than the total budget is rejected"
        );
        oversized.commit(vec![9]).expect("commit");
    }
    assert_eq!(
        cache.retained_snapshot_bytes(),
        8,
        "impossible admission does not evict a useful retained entry"
    );
    {
        let events = events.lock().expect("lock");
        assert!(
            events.len() == 1
                && events[0].action == SnapshotEventAction::Skipped
                && events[0].reason == SnapshotEventReason::ByteCapacity
                && events[0].snapshot_bytes == 9,
            "oversized refusal emits one skip and no removal"
        );
    }
    let extension = cache.acquire(&[1, 2, 3]).expect("acquire").expect("lease");
    assert!(
        extension.cache_hit() && extension.cached_tokens() == 2,
        "retained root remains reusable after oversized refusal"
    );
    extension.invalidate();
}

#[test]
fn abandoned_reservation_is_released() {
    let invalidations = counters(1);
    let mut support = restoring(8, None);
    support.restore = Some(Box::new(|_: &mut FakeState, _: &FakeSnapshot| Ok(())));
    let cache = cache(1, &invalidations, support);
    {
        let mut abandoned = cache.acquire(&[1]).expect("acquire").expect("lease");
        assert!(
            abandoned
                .try_reserve_snapshot(8, 1, false, SnapshotPurpose::Continuation, &[])
                .expect("reserve"),
            "abandoned request owns the whole reservation"
        );
    }
    assert_eq!(
        cache.reserved_snapshot_bytes(),
        0,
        "lease destruction releases its in-flight reservation"
    );
    let mut retry = cache.acquire(&[2]).expect("acquire").expect("lease");
    assert!(
        retry
            .try_reserve_snapshot(8, 1, false, SnapshotPurpose::Continuation, &[])
            .expect("reserve"),
        "released bytes are immediately available to another request"
    );
    retry
        .commit_with(vec![2], Some(FakeSnapshot::sized(2, 8)), Vec::new())
        .expect("commit");
}

#[test]
fn reservation_mismatch_skips_retention_without_failing_commit() {
    let invalidations = counters(1);
    let events: Events = Arc::default();
    let mut support = restoring(16, Some(&events));
    support.restore = Some(Box::new(|_: &mut FakeState, _: &FakeSnapshot| Ok(())));
    let cache = cache(1, &invalidations, support);
    let mut lease = cache.acquire(&[4, 5]).expect("acquire").expect("lease");
    assert!(
        lease
            .try_reserve_snapshot(4, 2, false, SnapshotPurpose::Continuation, &[])
            .expect("reserve"),
        "snapshot estimate reserves before allocation"
    );
    assert_eq!(
        lease
            .commit_with(vec![4, 5], Some(FakeSnapshot::sized(1, 8)), Vec::new())
            .expect("commit"),
        0,
        "an underestimated snapshot is skipped without failing commit"
    );
    assert!(
        cache.retained_snapshot_bytes() == 0 && cache.reserved_snapshot_bytes() == 0,
        "mismatch releases the complete reservation"
    );
    {
        let events = events.lock().expect("lock");
        assert!(
            events.len() == 1
                && events[0].action == SnapshotEventAction::Skipped
                && events[0].reason == SnapshotEventReason::ReservationMismatch
                && events[0].snapshot_bytes == 8
                && events[0].token_count == 2,
            "reservation mismatch is observable without prompt content"
        );
    }
    let retry = cache.acquire(&[4, 5, 6]).expect("acquire").expect("lease");
    assert!(
        !retry.cache_hit(),
        "skipped snapshot is a deterministic cache miss"
    );
    retry.invalidate();
}

/// Entry pressure replaces an edited branch's old, incompatible tail before
/// a checkpoint of the shared prefix.
#[test]
fn edited_tail_replacement_preserves_shared_checkpoints() {
    let invalidations = counters(1);
    let mut support = restoring(64, None);
    support.restore = Some(Box::new(|_: &mut FakeState, _: &FakeSnapshot| Ok(())));
    let cache = checkpoint_cache(1, 3, &invalidations, support);
    for prompt in [vec![1], vec![1, 2], vec![1, 2, 3]] {
        let mut lease = cache.acquire(&prompt).expect("acquire").expect("lease");
        assert!(
            lease
                .try_reserve_snapshot(8, prompt.len(), false, SnapshotPurpose::Continuation, &[])
                .expect("reserve")
        );
        let size = prompt.len();
        lease
            .commit_with(prompt, Some(FakeSnapshot::sized(size, 8)), Vec::new())
            .expect("commit");
    }
    {
        let mut edited = cache.acquire(&[1, 2, 9]).expect("acquire").expect("lease");
        assert_eq!(
            edited.cached_tokens(),
            2,
            "edited tail reuses the shared prefix"
        );
        assert!(
            edited
                .try_reserve_snapshot(8, 3, false, SnapshotPurpose::Continuation, &[])
                .expect("reserve")
        );
        edited
            .commit_with(vec![1, 2, 9], Some(FakeSnapshot::sized(3, 8)), Vec::new())
            .expect("commit");
    }
    let earlier = cache.acquire(&[1, 8]).expect("acquire").expect("lease");
    assert_eq!(
        earlier.cached_tokens(),
        1,
        "entry pressure replaces the incompatible tail, not the older prefix"
    );
    earlier.invalidate();
    assert_eq!(
        cache.retained_snapshot_bytes(),
        24,
        "tail replacement preserves exact byte accounting"
    );
}

/// A pending branch whose source entry another request replaced cannot take
/// that conversation's tail for its own.
#[test]
fn replaced_source_does_not_evict_another_branch_tail() {
    let invalidations = counters(2);
    let mut support = restoring(64, None);
    support.restore = Some(Box::new(|_: &mut FakeState, _: &FakeSnapshot| Ok(())));
    let cache = checkpoint_cache(2, 3, &invalidations, support);
    let save = |prompt: Vec<u32>| {
        let mut lease = cache.acquire(&prompt).expect("acquire").expect("lease");
        assert!(
            lease
                .try_reserve_snapshot(8, prompt.len(), false, SnapshotPurpose::Continuation, &[])
                .expect("reserve")
        );
        let size = prompt.len();
        lease
            .commit_with(prompt, Some(FakeSnapshot::sized(size, 8)), Vec::new())
            .expect("commit");
    };
    save(vec![1]);
    save(vec![8]);
    save(vec![8, 2]);
    let mut pending = cache.acquire(&[1, 3]).expect("acquire").expect("lease");
    assert_eq!(
        pending.cached_tokens(),
        1,
        "pending branch reuses its own root"
    );
    // Another execution slot makes the pending request's source the oldest
    // entry, then replaces it; the lease still holds the original state.
    for prompt in [vec![8], vec![8, 2]] {
        let touch = cache.acquire(&prompt).expect("acquire").expect("lease");
        touch.commit(Vec::new()).expect("commit");
    }
    save(vec![7]);
    save(vec![7, 5]);
    assert!(
        pending
            .try_reserve_snapshot(8, 2, false, SnapshotPurpose::Continuation, &[])
            .expect("reserve")
    );
    pending
        .commit_with(vec![1, 3], Some(FakeSnapshot::sized(2, 8)), Vec::new())
        .expect("commit");
    let other = cache.acquire(&[7, 5, 6]).expect("acquire").expect("lease");
    assert_eq!(
        other.cached_tokens(),
        2,
        "a replaced source index cannot identify another conversation's tail"
    );
    other.invalidate();
}

#[test]
fn entry_replacement_logs_removed_snapshot() {
    let invalidations = counters(1);
    let events: Events = Arc::default();
    let mut support = restoring(32, Some(&events));
    support.restore = Some(Box::new(|_: &mut FakeState, _: &FakeSnapshot| Ok(())));
    let cache = cache(1, &invalidations, support);
    {
        let mut first = cache.acquire(&[1, 2]).expect("acquire").expect("lease");
        assert!(
            first
                .try_reserve_snapshot(8, 2, false, SnapshotPurpose::Continuation, &[])
                .expect("reserve"),
            "first entry reserves bytes"
        );
        first
            .commit_with(vec![1, 2], Some(FakeSnapshot::sized(1, 8)), Vec::new())
            .expect("commit");
    }
    {
        let mut second = cache.acquire(&[9, 8, 7]).expect("acquire").expect("lease");
        assert!(
            second
                .try_reserve_snapshot(8, 3, false, SnapshotPurpose::Continuation, &[])
                .expect("reserve"),
            "replacement reserves bytes"
        );
        second
            .commit_with(vec![9, 8, 7], Some(FakeSnapshot::sized(2, 8)), Vec::new())
            .expect("commit");
    }
    {
        let events = events.lock().expect("lock");
        assert!(
            events.len() == 1
                && events[0].action == SnapshotEventAction::Removed
                && events[0].reason == SnapshotEventReason::EntryCapacity
                && events[0].snapshot_bytes == 8
                && events[0].token_count == 2,
            "entry replacement logs the removed snapshot dimensions"
        );
    }
    assert_eq!(
        cache.retained_snapshot_bytes(),
        8,
        "entry replacement keeps exact aggregate accounting"
    );
}

fn priority_cache(
    states: usize,
    records: usize,
    capacity: usize,
    events: Option<&Events>,
) -> Cache {
    let invalidations = counters(states);
    checkpoint_cache(states, records, &invalidations, restoring(capacity, events))
}

fn lease_for(cache: &Cache, tokens: &[u32]) -> Lease<FakeState, FakeSnapshot> {
    cache.acquire(tokens).expect("acquire").expect("lease")
}

fn reserve(
    lease: &mut Lease<FakeState, FakeSnapshot>,
    bytes: usize,
    tokens: usize,
    preserve_source: bool,
    purpose: SnapshotPurpose,
    replacement: &[u32],
) -> bool {
    lease
        .try_reserve_snapshot(bytes, tokens, preserve_source, purpose, replacement)
        .expect("reserve")
}

/// Under record or byte pressure, retention drops extra copies before a
/// conversation's last useful checkpoint.
#[test]
fn retention_prefers_extra_copies_under_pressure() {
    // Three records: record pressure binds; eight: bytes bind.
    for records in [3, 8] {
        let cache = priority_cache(1, records, 24, None);
        let save = |tokens: Vec<u32>, purpose: SnapshotPurpose, stable: usize| {
            let mut lease = cache
                .acquire_with(&tokens, None, InputIdentity::default(), None, true, stable)
                .expect("acquire")
                .expect("lease");
            assert!(
                reserve(&mut lease, 8, tokens.len(), false, purpose, &[]),
                "copy is admitted"
            );
            let value = tokens[0] as usize;
            lease
                .commit_with(tokens, Some(FakeSnapshot::sized(value, 8)), Vec::new())
                .expect("commit");
            assert!(cache.retained_snapshot_bytes() <= 24 && cache.reserved_snapshot_bytes() == 0);
        };
        save(vec![1, 2], SnapshotPurpose::Continuation, 0);
        save(vec![1, 2, 3], SnapshotPurpose::Retry, 2);
        save(vec![8, 9], SnapshotPurpose::Continuation, 0);
        save(vec![6, 7], SnapshotPurpose::Continuation, 0);
        for tag in [1u32, 8, 6] {
            let returned = lease_for(&cache, &[tag, if tag == 1 { 2 } else { tag + 1 }, 0]);
            assert!(
                returned.cached_tokens() == 2
                    && returned.state().expect("state").value == tag as usize,
                "dropping a retry preserves independent conversation payloads"
            );
            returned.invalidate();
        }
        let mut optional = lease_for(&cache, &[4, 5]);
        assert!(
            !reserve(&mut optional, 8, 2, true, SnapshotPurpose::History, &[]),
            "an optional copy cannot evict a last useful continuation"
        );
        optional.invalidate();
        assert!(
            cache.retained_snapshot_bytes() == 24 && cache.entry_capacity() == records,
            "checkpoint records allocate no extra execution states"
        );
        let next = [1u32, 2, 9];
        let mut advancing = lease_for(&cache, &next);
        assert!(
            reserve(
                &mut advancing,
                8,
                3,
                false,
                SnapshotPurpose::Continuation,
                &next
            ),
            "a new boundary can replace its own old boundary under pressure"
        );
        advancing
            .commit_with(next.to_vec(), Some(FakeSnapshot::sized(1, 8)), Vec::new())
            .expect("commit");
        for tokens in [vec![1u32, 2, 9, 0], vec![8, 9, 0], vec![6, 7, 0]] {
            let returned = lease_for(&cache, &tokens);
            assert_eq!(
                returned.cached_tokens(),
                tokens.len() - 1,
                "advancing one conversation does not evict another family's last checkpoint"
            );
            returned.invalidate();
        }
    }
}

/// An optional publication rechecks record pressure: a peer that took the
/// free record meanwhile keeps its last copy.
#[test]
fn optional_publication_rechecks_record_pressure() {
    let events: Events = Arc::default();
    let cache = priority_cache(2, 2, 64, Some(&events));
    let mut initial = lease_for(&cache, &[1]);
    assert!(reserve(
        &mut initial,
        8,
        1,
        false,
        SnapshotPurpose::Continuation,
        &[]
    ));
    initial
        .commit_with(vec![1], Some(FakeSnapshot::sized(1, 8)), Vec::new())
        .expect("commit");
    let mut pending = lease_for(&cache, &[9]);
    assert!(
        reserve(&mut pending, 8, 1, true, SnapshotPurpose::History, &[]),
        "optional capture sees a free record before a peer publishes"
    );
    let mut peer = lease_for(&cache, &[8]);
    assert!(reserve(
        &mut peer,
        8,
        1,
        false,
        SnapshotPurpose::Continuation,
        &[]
    ));
    peer.commit_with(vec![8], Some(FakeSnapshot::sized(8, 8)), Vec::new())
        .expect("commit");
    assert_eq!(
        pending
            .commit_with(vec![9], Some(FakeSnapshot::sized(9, 8)), Vec::new())
            .expect("commit"),
        0,
        "optional publication refuses to evict a last copy after concurrent admission"
    );
    {
        let events = events.lock().expect("events");
        assert!(
            cache.retained_snapshot_bytes() == 16
                && cache.reserved_snapshot_bytes() == 0
                && events.len() == 1
                && events[0].reason == SnapshotEventReason::EntryCapacity,
            "skipped publication releases its reservation and reports record pressure"
        );
    }
    for tag in [1u32, 8] {
        let returned = lease_for(&cache, &[tag, 0]);
        assert!(returned.cache_hit() && returned.cached_tokens() == 1);
        returned.invalidate();
    }
}

/// A new conversation replaces its own intermediate history first, never
/// another conversation's last checkpoint.
#[test]
fn new_conversation_replaces_its_own_history_first() {
    // Record pressure and byte pressure independently.
    for records in [4, 8] {
        let cache = priority_cache(1, records, if records == 4 { 1024 } else { 32 }, None);
        let existing = [1u32, 2];
        let mut first = lease_for(&cache, &existing);
        assert!(reserve(
            &mut first,
            8,
            2,
            false,
            SnapshotPurpose::Continuation,
            &[]
        ));
        first
            .commit_with(
                existing.to_vec(),
                Some(FakeSnapshot::sized(1, 8)),
                Vec::new(),
            )
            .expect("commit");
        let prompt = [9u32, 8, 7, 6];
        let mut incoming = lease_for(&cache, &prompt);
        for count in 1..prompt.len() {
            assert!(
                reserve(&mut incoming, 8, count, true, SnapshotPurpose::History, &[]),
                "the incoming conversation's intermediate history fits"
            );
            incoming
                .publish_snapshot(prompt[..count].to_vec(), FakeSnapshot::sized(9, 8), true)
                .expect("publish");
        }
        assert!(
            reserve(
                &mut incoming,
                8,
                prompt.len(),
                false,
                SnapshotPurpose::Continuation,
                &prompt
            ),
            "the new final checkpoint can replace redundant history"
        );
        assert!(
            incoming.has_snapshot_for(&existing),
            "byte admission preserves another conversation's last checkpoint"
        );
        incoming
            .commit_with(prompt.to_vec(), Some(FakeSnapshot::sized(9, 8)), Vec::new())
            .expect("commit");
        let previous = lease_for(&cache, &existing);
        assert_eq!(previous.cached_tokens(), existing.len());
        previous.invalidate();
        let next = lease_for(&cache, &prompt);
        assert_eq!(next.cached_tokens(), prompt.len());
        next.invalidate();
    }
}

/// An exact retry copy never displaces earlier history, also when a peer's
/// publication changes the pressure after its reservation.
#[test]
fn retry_does_not_displace_earlier_history() {
    for records in [3, 8] {
        let cache = priority_cache(1, records, 24, None);
        let save = |tokens: Vec<u32>, purpose: SnapshotPurpose| {
            let mut lease = lease_for(&cache, &tokens);
            assert!(reserve(&mut lease, 8, tokens.len(), false, purpose, &[]));
            lease
                .commit_with(tokens, Some(FakeSnapshot::sized(1, 8)), Vec::new())
                .expect("commit");
        };
        save(vec![1], SnapshotPurpose::History);
        save(vec![1, 2], SnapshotPurpose::Continuation);
        save(vec![8], SnapshotPurpose::Continuation);
        let mut retry = lease_for(&cache, &[1, 2, 3]);
        assert!(
            !reserve(&mut retry, 8, 3, true, SnapshotPurpose::Retry, &[]),
            "an exact retry cannot discard the earlier edit checkpoint"
        );
        retry.invalidate();
        let edited = lease_for(&cache, &[1, 9]);
        assert_eq!(edited.cached_tokens(), 1);
        edited.invalidate();
    }
    let cache = priority_cache(2, 3, 64, None);
    for tokens in [vec![1u32], vec![1, 2]] {
        let purpose = if tokens.len() == 1 {
            SnapshotPurpose::History
        } else {
            SnapshotPurpose::Continuation
        };
        let mut lease = lease_for(&cache, &tokens);
        assert!(reserve(&mut lease, 8, tokens.len(), false, purpose, &[]));
        lease
            .commit_with(tokens, Some(FakeSnapshot::sized(1, 8)), Vec::new())
            .expect("commit");
    }
    let mut retry = lease_for(&cache, &[1, 2, 3]);
    assert!(
        reserve(&mut retry, 8, 3, true, SnapshotPurpose::Retry, &[]),
        "retry sees a free record before a peer publishes"
    );
    let mut peer = lease_for(&cache, &[8]);
    assert!(reserve(
        &mut peer,
        8,
        1,
        false,
        SnapshotPurpose::Continuation,
        &[]
    ));
    peer.commit_with(vec![8], Some(FakeSnapshot::sized(8, 8)), Vec::new())
        .expect("commit");
    assert_eq!(
        retry
            .commit_with(vec![1, 2, 3], Some(FakeSnapshot::sized(1, 8)), Vec::new())
            .expect("commit"),
        0,
        "retry publication rechecks history priority after concurrent admission"
    );
    let edited = lease_for(&cache, &[1, 9]);
    assert!(edited.cached_tokens() == 1 && cache.reserved_snapshot_bytes() == 0);
    edited.invalidate();
}

fn retain(cache: &Cache, tokens: &[u32]) {
    let mut lease = lease_for(cache, tokens);
    assert!(
        reserve(
            &mut lease,
            8,
            tokens.len(),
            false,
            SnapshotPurpose::Continuation,
            &[]
        ),
        "test snapshot is admitted"
    );
    lease
        .commit_with(
            tokens.to_vec(),
            Some(FakeSnapshot::sized(tokens.len(), 8)),
            Vec::new(),
        )
        .expect("commit");
}

/// Peeking at the longest retained prefix leases nothing.
#[test]
fn cached_prefix_tokens_peeks_without_leasing() {
    let cache = priority_cache(1, 8, 1024, None);
    assert_eq!(
        cache.cached_prefix_tokens(&[1, 2, 3], InputIdentity::default()),
        0,
        "an empty cache has no reusable prefix"
    );
    retain(&cache, &[1, 2, 3]);
    assert_eq!(
        cache.cached_prefix_tokens(&[1, 2, 3, 4], InputIdentity::default()),
        3,
        "a retained prefix of the prompt is reported"
    );
    assert!(
        cache.cached_prefix_tokens(&[1, 2], InputIdentity::default()) == 0
            && cache.cached_prefix_tokens(&[1, 9, 3, 4], InputIdentity::default()) == 0,
        "shorter or diverging prompts do not reuse the checkpoint"
    );
    let image = InputIdentity {
        complete: &[5],
        prefixes: &[],
    };
    assert_eq!(
        cache.cached_prefix_tokens(&[1, 2, 3, 4], image),
        0,
        "a different input identity does not match"
    );
    // The single state slot remains available.
    let lease = lease_for(&cache, &[1, 2, 3, 4]);
    assert!(
        lease.cache_hit() && lease.cached_tokens() == 3,
        "a peek does not consume the checkpoint or the state slot"
    );
    lease.invalidate();
}

/// A shared prefix outlives older checkpoints under pressure once two
/// retained continuations branch after it.
#[test]
fn branch_point_outlives_older_turns_under_pressure() {
    for branched in [false, true] {
        // Room for four snapshots; the fifth evicts one.
        let cache = priority_cache(1, 8, 32, None);
        retain(&cache, &[9, 9, 9]); // An older, unrelated conversation.
        retain(&cache, &[1, 2, 3]); // The shared system prompt.
        retain(&cache, &[1, 2, 3, 4, 4]); // A conversation continuing from it.
        if branched {
            retain(&cache, &[1, 2, 3, 5, 5]); // A second one diverging after it.
        } else {
            retain(&cache, &[7, 7]);
        }
        retain(&cache, &[8, 8, 8, 8]); // Pressure: one checkpoint must go.
        let prefix = cache.cached_prefix_tokens(&[1, 2, 3, 6], InputIdentity::default());
        let unrelated = cache.cached_prefix_tokens(&[9, 9, 9, 1], InputIdentity::default());
        assert_eq!(
            (prefix, unrelated),
            if branched { (3, 0) } else { (0, 3) },
            "a shared prefix outlives older checkpoints only once it branches"
        );
    }
}

/// A checkpoint of `value` for `tokens` with `purpose`, published without
/// replacing its source.
fn retain_as(cache: &Cache, tokens: &[u32], value: usize, purpose: SnapshotPurpose) {
    let mut lease = lease_for(cache, tokens);
    assert!(
        reserve(&mut lease, WORD, tokens.len(), true, purpose, &[]),
        "initial checkpoint fits without eviction"
    );
    lease
        .commit_with(
            tokens.to_vec(),
            Some(FakeSnapshot::sized(value, WORD)),
            Vec::new(),
        )
        .expect("commit");
}

/// A chat bridge replays its last user message without the metadata it
/// sent, so each request diverges where the previous user turn starts. Its
/// new boundary extends the learned point; neither a warm turn advancing
/// from it nor a cold turn publishing it evicts it before other families'
/// copies.
#[test]
fn learned_branch_point_outlives_its_older_branch() {
    for warm in [true, false] {
        let cache = priority_cache(1, 8, 4 * WORD, None);
        retain_as(&cache, &[9, 9, 9], 9003, SnapshotPurpose::Continuation);
        retain_as(&cache, &[7, 7, 7], 7003, SnapshotPurpose::Continuation);
        // The previous turn's boundary ends in the live message the next
        // request rewrites; the turn before it diverged at the learned point.
        retain_as(&cache, &[1, 2, 3], 1003, SnapshotPurpose::BranchPoint);
        retain_as(
            &cache,
            &[1, 2, 3, 4, 4],
            2005,
            SnapshotPurpose::Continuation,
        );

        let incoming = [1, 2, 3, 5, 5];
        let mut turn = cache
            .acquire_with(&incoming, None, InputIdentity::default(), None, warm, 0)
            .expect("acquire")
            .expect("lease");
        assert_eq!(
            turn.cached_tokens(),
            if warm { 3 } else { 0 },
            "warm {warm}"
        );
        assert!(
            reserve(
                &mut turn,
                WORD,
                incoming.len(),
                !warm,
                SnapshotPurpose::Continuation,
                &incoming
            ),
            "the new boundary evicts another checkpoint"
        );
        turn.commit_with(
            incoming.to_vec(),
            Some(FakeSnapshot::sized(3005, WORD)),
            Vec::new(),
        )
        .expect("commit");

        let next = lease_for(&cache, &[1, 2, 3, 6, 6]);
        assert_eq!(
            (next.cached_tokens(), next.state().expect("state").value),
            (3, 1003),
            "warm {warm}: the learned branch point outlives its own family's new boundary"
        );
        next.invalidate();
        assert_eq!(
            (
                cache.cached_prefix_tokens(&[9, 9, 9, 1], InputIdentity::default()),
                cache.cached_prefix_tokens(&[7, 7, 7, 1], InputIdentity::default()),
            ),
            (0, 3),
            "warm {warm}: the oldest other family's checkpoint yields instead"
        );
    }
}

/// Only the deepest learned branch point of a family stays protected
///.
#[test]
fn deeper_learned_branch_point_supersedes_shallower() {
    let cache = priority_cache(1, 8, 4 * WORD, None);
    retain_as(&cache, &[9, 9, 9], 3, SnapshotPurpose::Continuation);
    retain_as(&cache, &[1, 2, 3], 3, SnapshotPurpose::BranchPoint);
    retain_as(&cache, &[1, 2, 3, 4, 5], 5, SnapshotPurpose::BranchPoint);
    retain_as(
        &cache,
        &[1, 2, 3, 4, 5, 6],
        6,
        SnapshotPurpose::Continuation,
    );
    retain_as(&cache, &[8, 8, 8], 3, SnapshotPurpose::Continuation); // Pressure.
    let cached = |tokens: &[u32]| cache.cached_prefix_tokens(tokens, InputIdentity::default());
    assert_eq!(
        (
            cached(&[1, 2, 3, 7]),
            cached(&[1, 2, 3, 4, 5, 7]),
            cached(&[9, 9, 9, 1])
        ),
        (0, 5, 3)
    );
}

/// A new conversation from a shared branch point replaces an older
/// unrelated checkpoint, not the source its peers need.
#[test]
fn new_branch_preserves_shared_source_under_byte_pressure() {
    let cache = priority_cache(1, 8, 4 * WORD, None);
    let retain_replacing = |tokens: &[u32], value: usize| {
        let mut lease = lease_for(&cache, tokens);
        assert!(
            reserve(
                &mut lease,
                WORD,
                tokens.len(),
                false,
                SnapshotPurpose::Continuation,
                tokens
            ),
            "initial checkpoint fits without eviction"
        );
        lease
            .commit_with(
                tokens.to_vec(),
                Some(FakeSnapshot::sized(value, WORD)),
                Vec::new(),
            )
            .expect("commit");
    };
    retain_replacing(&[9, 9, 9], 9003);
    retain_replacing(&[1, 2, 3], 1003);
    retain_replacing(&[1, 2, 3, 4, 4], 2005);
    retain_replacing(&[1, 2, 3, 5, 5], 3005);

    let incoming = [1, 2, 3, 6, 6];
    let mut branch = lease_for(&cache, &incoming);
    assert!(
        branch.cache_hit()
            && branch.cached_tokens() == 3
            && branch.state().expect("state").value == 1003,
        "the new conversation restores the shared branch point"
    );
    assert!(
        reserve(
            &mut branch,
            WORD,
            incoming.len(),
            false,
            SnapshotPurpose::Continuation,
            &incoming
        ),
        "the new continuation can replace an older unrelated checkpoint"
    );
    branch
        .commit_with(
            incoming.to_vec(),
            Some(FakeSnapshot::sized(4005, WORD)),
            Vec::new(),
        )
        .expect("commit");
    assert_eq!(
        (
            cache.retained_snapshot_bytes(),
            cache.reserved_snapshot_bytes()
        ),
        (4 * WORD, 0)
    );
    let next = lease_for(&cache, &[1, 2, 3, 7, 7]);
    assert!(
        next.cache_hit() && next.cached_tokens() == 3 && next.state().expect("state").value == 1003,
        "a new branch must not replace the shared source needed by its peers"
    );
    next.invalidate();
    assert_eq!(
        cache.cached_prefix_tokens(&[9, 9, 9], InputIdentity::default()),
        0,
        "the oldest unrelated continuation yields to the shared branch point"
    );
}

/// Which candidate a lookup rejected, and by which guard, is logged only at
/// debug.
#[test]
fn cache_candidate_detail_is_debug_tier_only() {
    let run = |level: LogLevel| {
        let invalidations = counters(1);
        let cache = cache(1, &invalidations, SnapshotSupport::default());
        let ((), log) = logging::capture_at(level, || {
            let lease = cache.acquire(&[1, 2, 3]).expect("acquire").expect("lease");
            lease.commit(vec![1, 2, 3]).expect("commit");
            let other = cache.acquire(&[9, 8, 7]).expect("acquire").expect("lease");
            assert!(
                !other.cache_hit(),
                "a different prompt cannot reuse the slot"
            );
        });
        log
    };
    let quiet = run(LogLevel::Info);
    assert!(
        !quiet.contains("event=candidate_skip") && !quiet.contains("event=capture_evicts"),
        "{quiet}"
    );
    let verbose = run(LogLevel::Debug);
    assert!(
        verbose
            .contains("event=candidate_skip index=0 tokens=3 prompt_tokens=3 reason=token_prefix"),
        "{verbose}"
    );
    assert!(
        verbose.contains("event=capture_evicts index=0 tokens=3"),
        "{verbose}"
    );
}

/// A waiter that finds every state busy reports why each candidate was
/// rejected, once per wait.
#[test]
fn waiting_acquire_reports_candidate_skips() {
    let invalidations = counters(1);
    let cache = cache(1, &invalidations, SnapshotSupport::default());
    let seed = cache.acquire(&[1, 2, 3]).expect("acquire").expect("lease");
    seed.commit(vec![1, 2, 3]).expect("commit");
    let busy = cache
        .acquire(&[1, 2, 3, 4])
        .expect("acquire")
        .expect("lease");
    assert!(
        busy.cache_hit(),
        "the sole state is leased to a matching prompt"
    );
    // The lease clears the saved tokens without snapshots, so the waiter
    // records the busy entry with no checkpoint tokens.
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(200);
    let cancelled = || std::time::Instant::now() >= deadline;
    let (waiter, log) = logging::capture_at(LogLevel::Debug, || {
        cache
            .acquire_with(
                &[9, 8, 7],
                Some(&cancelled),
                InputIdentity::default(),
                None,
                true,
                0,
            )
            .expect("acquire")
    });
    assert!(
        waiter.is_none(),
        "the cancelled waiter never acquires the busy state"
    );
    assert!(
        log.contains("event=candidate_skip index=0 tokens=0 prompt_tokens=3 reason=unavailable"),
        "{log}"
    );
    assert_eq!(
        log.matches("event=candidate_skip").count(),
        1,
        "a long wait reports its first record set once"
    );
    busy.invalidate();
}

/// Conversations share a system prompt whose grid checkpoint is history.
/// Once a second conversation restores it and diverges, it is a branch
/// point: freezing a continued frontier under byte pressure must evict an
/// older turn's checkpoint, not the prefix every new conversation restores
///.
#[test]
fn restored_shared_prefix_outlives_frozen_frontiers() {
    for restored in [false, true] {
        let cache = priority_cache(2, 8, 4 * WORD, None);
        let unrelated = [9u32, 9, 9];
        let mut other = lease_for(&cache, &unrelated);
        assert!(
            reserve(
                &mut other,
                WORD,
                unrelated.len(),
                false,
                SnapshotPurpose::Continuation,
                &[]
            ),
            "the unrelated checkpoint fits"
        );
        other
            .commit_with(unrelated.to_vec(), Some(FakeSnapshot::new(9)), Vec::new())
            .expect("commit");

        let shared = [1u32, 2, 3];
        let first = [1u32, 2, 3, 4, 4];
        let mut a = lease_for(&cache, &first);
        assert!(
            reserve(
                &mut a,
                WORD,
                shared.len(),
                true,
                SnapshotPurpose::History,
                &[]
            ),
            "the shared grid checkpoint fits"
        );
        a.publish_snapshot(shared.to_vec(), FakeSnapshot::new(3), true)
            .expect("publish");
        assert!(
            reserve(
                &mut a,
                WORD,
                first.len(),
                false,
                SnapshotPurpose::Continuation,
                &first
            ),
            "the first prompt checkpoint fits"
        );
        a.commit_with(
            first.to_vec(),
            Some(FakeSnapshot::new(5)),
            vec![1, 2, 3, 4, 4, 8, 8],
        )
        .expect("commit");

        let second = [1u32, 2, 3, 5, 5];
        let mut b = cache
            .acquire_with(&second, None, InputIdentity::default(), None, restored, 0)
            .expect("acquire")
            .expect("lease");
        assert_eq!(
            b.cached_tokens(),
            if restored { shared.len() } else { 0 },
            "the second conversation restores the shared prefix when allowed"
        );
        assert!(
            reserve(
                &mut b,
                WORD,
                second.len(),
                false,
                SnapshotPurpose::Continuation,
                &second
            ),
            "the second prompt checkpoint fits"
        );
        b.commit_with(
            second.to_vec(),
            Some(FakeSnapshot::new(6)),
            vec![1, 2, 3, 5, 5, 8],
        )
        .expect("commit");

        // The first conversation continues from its live frontier and
        // freezes it.
        let next = [1u32, 2, 3, 4, 4, 8, 8, 6];
        let mut turn = lease_for(&cache, &next);
        assert_eq!(
            turn.cached_tokens(),
            7,
            "the next turn continues the frontier"
        );
        assert!(
            reserve(
                &mut turn,
                WORD,
                next.len(),
                false,
                SnapshotPurpose::Continuation,
                &next
            ),
            "the frozen frontier evicts one checkpoint"
        );
        turn.commit_with(next.to_vec(), Some(FakeSnapshot::new(8)), Vec::new())
            .expect("commit");

        assert_eq!(
            cache.cached_prefix_tokens(&[1, 2, 3, 7, 7], InputIdentity::default()),
            if restored { shared.len() } else { 0 },
            "a restored shared prefix outlives a frozen frontier under pressure"
        );
        assert_eq!(
            cache.cached_prefix_tokens(&first, InputIdentity::default()),
            if restored { shared.len() } else { first.len() },
            "the continued conversation's older checkpoint yields instead"
        );
    }
}
