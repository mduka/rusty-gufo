//! The runtime part of the ds4 ABI on a gfx1151: one backend per process,
//! allocations, views, transfers and their range checks.

#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use std::sync::Mutex;

use gufo_deepseek_v4_flash_kernels::{Backend, Input};

/// Tests share the process-wide runtime one at a time.
static RUNTIME: Mutex<()> = Mutex::new(());

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn one_backend_per_process() {
    let _serial = RUNTIME.lock().unwrap_or_else(|e| e.into_inner());
    let backend = Backend::init().unwrap();
    assert!(Backend::init().is_err(), "a second backend is refused");
    drop(backend);
    // The runtime initialises again after a cleanup.
    let backend = Backend::init().unwrap();
    drop(backend);
}

#[test]
fn transfers_views_and_their_ranges() {
    let _serial = RUNTIME.lock().unwrap_or_else(|e| e.into_inner());
    let backend = Backend::init().unwrap();
    let mut a = backend.alloc(64).unwrap();
    assert_eq!(a.bytes(), 64);
    let data: Vec<u8> = (0..64).collect();
    backend.write(&mut a, 0, &data).unwrap();
    let mut back = vec![0u8; 64];
    backend.read(&a, 0, &mut back).unwrap();
    assert_eq!(back, data);

    // A view addresses its range of the base.
    let view = a.view(16, 32).unwrap();
    assert_eq!(view.bytes(), 32);
    let mut part = vec![0u8; 32];
    backend.read(&view, 0, &mut part).unwrap();
    assert_eq!(part, data[16..48]);
    assert!(a.view(60, 8).is_none(), "views stay inside the base");
    assert!(a.view(65, 0).is_none());

    // Transfers outside a descriptor fail without touching memory.
    assert!(backend.read(&view, 1, &mut part).is_err());
    assert!(
        backend
            .write(&mut a.view_mut(16, 32).unwrap(), 32, &[1])
            .is_err()
    );

    // Exclusive views split their base at view offsets.
    let mut c = backend.alloc(64).unwrap();
    backend.write(&mut c, 0, &data).unwrap();
    let (head, tail) = c.split_at_mut(16).unwrap();
    assert_eq!((head.bytes(), tail.bytes()), (16, 48));
    let tail = tail.view_mut(16, 32).unwrap();
    backend.read(&tail, 0, &mut part).unwrap();
    assert_eq!(part, data[32..64]);
    assert!(c.split_at_mut(8).is_none(), "splits fall on view offsets");
    assert!(c.split_at_mut(80).is_none(), "splits stay inside the base");
    assert!(c.view_mut(60, 8).is_none());

    let mut b = backend.alloc(64).unwrap();
    backend.copy(&mut b, 8, &a, 0, 56).unwrap();
    assert!(backend.copy(&mut b, 9, &a, 0, 56).is_err());
    backend.read(&b, 8, &mut back[..56]).unwrap();
    assert_eq!(back[..56], data[..56]);

    let mut f = backend.alloc_managed(16).unwrap();
    backend.fill_f32(&mut f, 1.5, 4).unwrap();
    assert!(backend.fill_f32(&mut f, 1.5, 5).is_err());
    let mut out = vec![0u8; 16];
    backend.read(&f, 0, &mut out).unwrap();
    assert_eq!(floats(&out), [1.5; 4]);
    // Snapshot spans of several tensors append in order.
    let reader = backend.snapshot_reader();
    let mut snapshot = vec![7u8];
    reader
        .read_spans(
            &[f.input().range(4..12), a.input().range(60..64), b.input()],
            &mut snapshot,
        )
        .unwrap();
    assert_eq!(snapshot.len(), 1 + 8 + 4 + 64);
    assert_eq!(snapshot[0], 7);
    assert_eq!(floats(&snapshot[1..9]), [1.5; 2]);
    assert_eq!(snapshot[9..13], data[60..64]);
    assert_eq!(snapshot[13 + 8..], data[..56]);
    // The transfer is reused.
    reader.read_spans(&[a.input()], &mut snapshot).unwrap();
    assert_eq!(snapshot[snapshot.len() - 64..], data);
    backend.synchronize().unwrap();
}
