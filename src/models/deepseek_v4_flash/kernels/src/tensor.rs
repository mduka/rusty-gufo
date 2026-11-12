//! Device memory the launchers address.
//!
//! [`DeviceTensor`] owns a gufo-hip allocation of the backend's stream; a
//! [`TensorView`] borrows a checked range of one, and a [`TensorViewMut`]
//! borrows its range exclusively, so the borrow checker proves that the
//! exclusive views a caller holds do not overlap. Each carries the gufo-hip
//! slice the generated launchers take (`input`, `output`) and dereferences
//! to its [`Tensor`] size.

use std::ops::Range;

use gufo_hip::{DeviceBuffer, DeviceSlice, DeviceSliceMut};

/// Alignment of view offsets (allocations are aligned to 256 bytes).
pub const VIEW_ALIGNMENT: u64 = 16;

/// The size of device bytes: an allocation ([`DeviceTensor`]) or part of
/// one ([`TensorView`], [`TensorViewMut`]).
pub struct Tensor {
    bytes: u64,
}

impl Tensor {
    fn new(bytes: u64) -> Self {
        Self { bytes }
    }

    /// Size in bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// The byte range `offset .. offset + bytes` of `len` bytes, when it lies
/// inside and `offset` is a multiple of `alignment`.
fn checked_range(len: u64, offset: u64, bytes: u64, alignment: u64) -> Option<Range<usize>> {
    (offset.is_multiple_of(alignment) && offset <= len && bytes <= len - offset).then(|| {
        let start = offset as usize;
        start..start + bytes as usize
    })
}

/// An allocation of the backend's stream (`Backend::alloc`: an exact
/// `hipMalloc` or managed memory, as `ds4_gpu_tensor_alloc` made), freed on
/// drop.
pub struct DeviceTensor {
    tensor: Tensor,
    buffer: DeviceBuffer<u8>,
}

impl DeviceTensor {
    /// The whole of `buffer`.
    pub(crate) fn from_buffer(buffer: DeviceBuffer<u8>) -> Self {
        Self {
            tensor: Tensor::new(buffer.len() as u64),
            buffer,
        }
    }

    /// The whole tensor as a view.
    pub fn whole(&self) -> TensorView<'_> {
        TensorView::new(self.buffer.as_slice())
    }

    /// `bytes` at `offset`, borrowing this tensor; `None` outside it or at
    /// an offset that is not a multiple of 16, since kernels access rows
    /// with 16-byte vector loads.
    pub fn view(&self, offset: u64, bytes: u64) -> Option<TensorView<'_>> {
        self.whole().view(offset, bytes)
    }

    /// The whole tensor as an exclusive view.
    pub fn whole_mut(&mut self) -> TensorViewMut<'_> {
        TensorViewMut::new(self.buffer.as_mut_slice())
    }

    /// `bytes` at `offset`, borrowing this tensor exclusively; the checks of
    /// [`view`](Self::view).
    pub fn view_mut(&mut self, offset: u64, bytes: u64) -> Option<TensorViewMut<'_>> {
        self.whole_mut().view_mut(offset, bytes)
    }

    /// The bytes before and from `bytes` as two exclusive views; `None`
    /// outside the tensor or at an offset that is not a multiple of 16.
    pub fn split_at_mut(&mut self, bytes: u64) -> Option<(TensorViewMut<'_>, TensorViewMut<'_>)> {
        self.whole_mut().split_at_mut(bytes)
    }
}

impl std::ops::Deref for DeviceTensor {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        &self.tensor
    }
}

/// A range of a tensor ([`DeviceTensor::view`], [`TensorView::view`]).
pub struct TensorView<'a> {
    tensor: Tensor,
    slice: DeviceSlice<'a, u8>,
}

impl<'a> TensorView<'a> {
    fn new(slice: DeviceSlice<'a, u8>) -> Self {
        Self {
            tensor: Tensor::new(slice.len() as u64),
            slice,
        }
    }

    /// `bytes` at `offset` of this view; the checks of
    /// [`DeviceTensor::view`].
    pub fn view(&self, offset: u64, bytes: u64) -> Option<TensorView<'a>> {
        checked_range(self.tensor.bytes(), offset, bytes, VIEW_ALIGNMENT)
            .map(|range| Self::new(self.slice.range(range)))
    }
}

impl Clone for TensorView<'_> {
    fn clone(&self) -> Self {
        Self::new(self.slice)
    }
}

impl std::ops::Deref for TensorView<'_> {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        &self.tensor
    }
}

/// An exclusive range of a tensor ([`DeviceTensor::view_mut`],
/// [`DeviceTensor::split_at_mut`]). It dereferences to a shared [`Tensor`]
/// size only.
pub struct TensorViewMut<'a> {
    tensor: Tensor,
    slice: DeviceSliceMut<'a, u8>,
}

impl<'a> TensorViewMut<'a> {
    fn new(slice: DeviceSliceMut<'a, u8>) -> Self {
        Self {
            tensor: Tensor::new(slice.len() as u64),
            slice,
        }
    }

    /// `bytes` at `offset` of this view, keeping its borrow; the checks of
    /// [`DeviceTensor::view`]. [`reborrow`](Self::reborrow) first to keep
    /// this view.
    pub fn view_mut(self, offset: u64, bytes: u64) -> Option<TensorViewMut<'a>> {
        let range = checked_range(self.tensor.bytes(), offset, bytes, VIEW_ALIGNMENT)?;
        Some(Self::new(self.slice.range(range)))
    }

    /// The bytes before and from `bytes`, keeping this view's borrow;
    /// `None` outside it or at an offset that is not a multiple of 16.
    pub fn split_at_mut(self, bytes: u64) -> Option<(TensorViewMut<'a>, TensorViewMut<'a>)> {
        checked_range(self.tensor.bytes(), bytes, 0, VIEW_ALIGNMENT)?;
        let (head, tail) = self.slice.split_at(bytes as usize);
        Some((Self::new(head), Self::new(tail)))
    }

    /// This view again, for a shorter exclusive borrow.
    pub fn reborrow(&mut self) -> TensorViewMut<'_> {
        TensorViewMut::new(self.slice.reborrow())
    }

    /// A shared view of the same bytes while this one is borrowed.
    pub fn as_view(&self) -> TensorView<'_> {
        TensorView::new(self.slice.as_const())
    }
}

/// Device bytes a generated launcher reads: an allocation or a view.
pub trait Input: std::ops::Deref<Target = Tensor> {
    /// The bytes as the launcher's input slice.
    fn input(&self) -> DeviceSlice<'_, u8>;

    /// `bytes` at `offset`; the checks of [`DeviceTensor::view`].
    fn view(&self, offset: u64, bytes: u64) -> Option<TensorView<'_>> {
        checked_range(self.input().len() as u64, offset, bytes, VIEW_ALIGNMENT)
            .map(|range| TensorView::new(self.input().range(range)))
    }
}

impl Input for DeviceTensor {
    fn input(&self) -> DeviceSlice<'_, u8> {
        self.buffer.as_slice()
    }
}

impl Input for TensorView<'_> {
    fn input(&self) -> DeviceSlice<'_, u8> {
        self.slice
    }
}

impl Input for TensorViewMut<'_> {
    fn input(&self) -> DeviceSlice<'_, u8> {
        self.slice.as_const()
    }
}

/// Device bytes a generated launcher writes: an allocation or an exclusive
/// view.
pub trait Output: std::ops::Deref<Target = Tensor> {
    /// The bytes as the launcher's output slice.
    fn output(&mut self) -> DeviceSliceMut<'_, u8>;
}

impl Output for DeviceTensor {
    fn output(&mut self) -> DeviceSliceMut<'_, u8> {
        self.buffer.as_mut_slice()
    }
}

impl Output for TensorViewMut<'_> {
    fn output(&mut self) -> DeviceSliceMut<'_, u8> {
        self.slice.reborrow()
    }
}

/// The whole tensor.
impl<'a> From<&'a mut DeviceTensor> for TensorViewMut<'a> {
    fn from(tensor: &'a mut DeviceTensor) -> Self {
        tensor.whole_mut()
    }
}

/// A reborrow ([`TensorViewMut::reborrow`]).
impl<'a, 'b> From<&'b mut TensorViewMut<'a>> for TensorViewMut<'b> {
    fn from(view: &'b mut TensorViewMut<'a>) -> Self {
        view.reborrow()
    }
}

impl std::ops::Deref for TensorViewMut<'_> {
    type Target = Tensor;
    fn deref(&self) -> &Tensor {
        &self.tensor
    }
}
