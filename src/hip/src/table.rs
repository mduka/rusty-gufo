//! Rows that kernels find through a device header instead of a launch
//! argument, so captured graphs follow new contents without a recapture.

use crate::error::HipError;
use crate::memory::{DeviceBuffer, DeviceCopy, DeviceSlice};
use crate::stream::Stream;

/// A device header `{ const T* rows; u32 count; i32 tag }` (16 bytes, the
/// layout of `qwen::vision::DeviceRope`) over `count` rows of `WIDTH` values.
///
/// Only [`RowTable::set`] writes the header, always after the rows it points
/// to, so a kernel that reads at most `count * WIDTH` values through it stays
/// inside live memory. The header keeps its address while rows change; a
/// launch records both allocations, so a graph captured before the rows moved
/// no longer validates and is not replayed with a header that points past the
/// rows it retains.
pub struct RowTable<T: DeviceCopy, const WIDTH: usize> {
    header: DeviceBuffer<u64>,
    rows: Option<DeviceBuffer<T>>,
    count: u32,
    tag: i32,
}

impl<T: DeviceCopy, const WIDTH: usize> RowTable<T, WIDTH> {
    /// An empty table (null rows, count and tag zero) of `stream`.
    pub fn new(stream: &Stream) -> Result<Self, HipError> {
        let mut header = stream.alloc::<u64>(2)?;
        stream.upload(&[0, 0], header.as_mut_slice())?;
        Ok(Self {
            header,
            rows: None,
            count: 0,
            tag: 0,
        })
    }

    /// Replaces the rows and the tag, in the order of `stream` (the home
    /// stream), and waits. The row buffer grows as needed and never shrinks;
    /// a replaced buffer is freed after the work already queued.
    pub fn set(&mut self, stream: &Stream, rows: &[[T; WIDTH]], tag: i32) -> Result<(), HipError> {
        let count = u32::try_from(rows.len())
            .map_err(|_| HipError::refused("row table exceeds u32 rows"))?;
        let values = rows.as_flattened();
        let capacity = self.rows.as_ref().map_or(0, DeviceBuffer::len);
        if values.len() > capacity {
            self.rows = Some(stream.alloc::<T>(values.len())?);
        }
        let pointer = match &mut self.rows {
            Some(buffer) if !values.is_empty() => {
                stream.upload(values, buffer.slice_mut(0..values.len()))?;
                buffer.as_slice().address() as u64
            }
            Some(buffer) => buffer.as_slice().address() as u64,
            None => 0,
        };
        let words = [pointer, u64::from(count) | (u64::from(tag as u32) << 32)];
        stream.upload(&words, self.header.as_mut_slice())?;
        self.count = count;
        self.tag = tag;
        Ok(())
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn tag(&self) -> i32 {
        self.tag
    }

    /// The header, whose address the kernels take.
    pub fn header(&self) -> DeviceSlice<'_, u64> {
        self.header.as_slice()
    }

    /// The row buffer, once rows were set.
    pub fn rows(&self) -> Option<DeviceSlice<'_, T>> {
        self.rows.as_ref().map(DeviceBuffer::as_slice)
    }
}
