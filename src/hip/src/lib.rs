//! Safe HIP runtime layer: the only crate with hand-written `unsafe` host code.
//!
//! Invariants enforced for every caller:
//! * device allocations belong to a [`Stream`] and are freed in its order
//!   (`hipFreeAsync`), so dropping a buffer never races a kernel on that
//!   stream and never stalls the device;
//! * a launch or copy can only name buffers of its own stream
//!   ([`LaunchError::ForeignStream`]); only the blocking
//!   [`Stream::copy_across`] reads another stream's buffer, and returns after
//!   the copy has finished;
//! * one kernel cannot receive the same buffer as input and output: views are
//!   borrowed from their buffer, so `&` and `&mut` cannot coexist;
//! * pinned host memory cannot be touched while an asynchronous copy is in
//!   flight ([`PinnedBuffer`]);
//! * after a sticky device fault the stream refuses further work.
//!
//! Not enforced: indexing inside kernels, and ordering between streams. Those
//! stay kernel contracts, covered by numerical tests.
//!
//! One kernel cannot see the same buffer as input and output:
//!
//! ```compile_fail,E0502
//! use gufo_hip::{DeviceSlice, DeviceSliceMut, Stream};
//! fn kernel(_input: DeviceSlice<'_, f32>, _output: DeviceSliceMut<'_, f32>) {}
//! let stream = Stream::new().unwrap();
//! let mut buffer = stream.alloc::<f32>(16).unwrap();
//! kernel(buffer.as_slice(), buffer.as_mut_slice());
//! ```
//!
//! A view cannot outlive a reallocation of what it points to:
//!
//! ```compile_fail,E0506
//! use gufo_hip::Stream;
//! let stream = Stream::new().unwrap();
//! let mut buffer = stream.alloc::<f32>(16).unwrap();
//! let view = buffer.as_slice();
//! buffer = stream.alloc::<f32>(32).unwrap();
//! let _ = view.len();
//! ```
//!
//! Pinned memory cannot be read while a copy into it is being set up:
//!
//! ```compile_fail,E0502
//! use gufo_hip::{PinnedBuffer, Stream};
//! let stream = Stream::new().unwrap();
//! let device = stream.alloc::<f32>(16).unwrap();
//! let mut pinned = PinnedBuffer::<f32>::zeroed(16).unwrap();
//! let early = pinned.host().unwrap();
//! stream.download_async(device.as_slice(), &mut pinned).unwrap();
//! let _ = early[0];
//! ```

//! Immutable tensors can be read on any stream but never written:
//!
//! ```compile_fail,E0308
//! use gufo_hip::{DeviceSliceMut, Stream};
//! fn kernel(_output: DeviceSliceMut<'_, f32>) {}
//! let stream = Stream::new().unwrap();
//! let weights = stream.alloc::<f32>(16).unwrap().freeze(&stream).unwrap();
//! kernel(weights.view());
//! ```

pub mod blas;
mod deferred;
mod device;
mod error;
#[cfg(test)]
mod fault;
mod graph;
mod host;
pub mod launch;
mod mapped;
mod build_info {
    include!(concat!(env!("OUT_DIR"), "/build_info.rs"));
}
mod memory;
mod probe;
mod queues;
mod record;
mod stream;
pub mod sys;
mod table;
mod weights;

pub use build_info::{BUILD_HIP_VERSION, BUILD_ROCM_ROOT, BUILD_ROCM_VERSION};
pub use deferred::collect as collect_deferred_frees;
pub use device::{
    DeviceProperties, MemoryType, PointerAttributes, device_count, device_properties, error_name,
    error_string,
};
pub use error::{ErrorKind, HipError, LaunchError};
pub use graph::{Graph, GraphCache, GraphRun, MismatchPolicy};
pub use gufo_formats::GgmlType;
pub use host::{HostMemory, Residency};
pub use launch::Launch;
pub use mapped::{MappedFile, advise_random_access, huge_page_buffer};
pub use memory::{
    Allocation, Bf16, Derivation, DeviceBuffer, DeviceCopy, DeviceIndexPair, DeviceIndices,
    DeviceSlice, DeviceSliceMut, DeviceTensor, F16, IndexSlice, IndexSliceMut, IndexSpec,
    PinnedBuffer, QuantSlice, QuantTensor, SharedBytes,
};
pub use probe::{DeviceProbe, DeviceProbeStatus, exit_immediately};
pub use queues::cap_hardware_queues;
pub use stream::{
    Event, PinnedDownload, Stream, cap_reported_device_free, device_memory, runtime_version,
};
pub use table::RowTable;
pub use weights::{LoadedWeights, Pending, Placement, WeightLoader};
