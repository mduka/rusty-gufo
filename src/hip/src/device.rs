//! Device discovery: the count, the properties gufo reports, pointer
//! attributes, error names.

use std::ffi::{CStr, c_void};

use crate::error::{HipError, check};
use crate::sys;

mod ffi {
    #[repr(C)]
    pub struct Properties {
        pub name: [u8; 256],
        pub arch_name: [u8; 256],
        pub major: i32,
        pub minor: i32,
        pub total_global_memory: u64,
        pub multiprocessor_count: i32,
        pub warp_size: i32,
        pub integrated: i32,
    }

    // SAFETY: the declarations match shim/device.cpp, which the compiler
    // checks against the ROCm headers.
    unsafe extern "C" {
        pub fn gufo_hip_device_properties(device: i32, out: *mut Properties) -> i32;
        pub fn gufo_hip_pointer_attributes(
            pointer: *const std::ffi::c_void,
            memory_type: *mut i32,
            managed: *mut i32,
        ) -> i32;
    }
}

/// The memory type HIP reports for an address (`hipMemoryType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    Unregistered,
    Host,
    Device,
    Managed,
    Array,
    Unified,
    Unknown,
}

impl MemoryType {
    /// Lowercase name, e.g. "device".
    pub fn name(self) -> &'static str {
        match self {
            Self::Unregistered => "unregistered",
            Self::Host => "host",
            Self::Device => "device",
            Self::Managed => "managed",
            Self::Array => "array",
            Self::Unified => "unified",
            Self::Unknown => "unknown",
        }
    }
}

/// What `hipPointerGetAttributes` reports about an address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointerAttributes {
    pub memory_type: MemoryType,
    pub managed: bool,
}

pub(crate) fn pointer_attributes(pointer: *const c_void) -> Result<PointerAttributes, HipError> {
    let (mut memory_type, mut managed) = (0, 0);
    // SAFETY: valid out-pointers; the runtime only looks the address up.
    let status =
        unsafe { ffi::gufo_hip_pointer_attributes(pointer, &mut memory_type, &mut managed) };
    check(status, "hipPointerGetAttributes")?;
    let memory_type = match memory_type {
        0 => MemoryType::Unregistered,
        1 => MemoryType::Host,
        2 => MemoryType::Device,
        3 => MemoryType::Managed,
        4 => MemoryType::Array,
        5 => MemoryType::Unified,
        _ => MemoryType::Unknown,
    };
    Ok(PointerAttributes {
        memory_type,
        managed: managed != 0,
    })
}

/// The fields of `hipDeviceProp_t` gufo reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceProperties {
    pub name: String,
    /// `gcnArchName`, e.g. `gfx1151`.
    pub arch_name: String,
    pub major: i32,
    pub minor: i32,
    pub total_global_memory: u64,
    pub multiprocessor_count: i32,
    pub warp_size: i32,
    /// Whether the device shares the host's memory (an APU).
    pub integrated: bool,
}

/// Number of visible HIP devices (`hipGetDeviceCount`).
pub fn device_count() -> Result<i32, HipError> {
    let mut count = 0;
    // SAFETY: valid out-pointer.
    let status = unsafe { sys::hipGetDeviceCount(&mut count) };
    check(status, "hipGetDeviceCount")?;
    Ok(count)
}

/// Properties of `device` (`hipGetDeviceProperties`).
pub fn device_properties(device: i32) -> Result<DeviceProperties, HipError> {
    let mut properties = ffi::Properties {
        name: [0; 256],
        arch_name: [0; 256],
        major: 0,
        minor: 0,
        total_global_memory: 0,
        multiprocessor_count: 0,
        warp_size: 0,
        integrated: 0,
    };
    // SAFETY: valid out-pointer to a struct of the shim's layout.
    let status = unsafe { ffi::gufo_hip_device_properties(device, &mut properties) };
    check(status, "hipGetDeviceProperties")?;
    let text = |bytes: &[u8]| {
        CStr::from_bytes_until_nul(bytes)
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    Ok(DeviceProperties {
        name: text(&properties.name),
        arch_name: text(&properties.arch_name),
        major: properties.major,
        minor: properties.minor,
        total_global_memory: properties.total_global_memory,
        multiprocessor_count: properties.multiprocessor_count,
        warp_size: properties.warp_size,
        integrated: properties.integrated != 0,
    })
}

/// `hipGetErrorString` of a HIP status code, e.g. "out of memory".
pub fn error_string(code: i32) -> String {
    // SAFETY: hipGetErrorString returns a static NUL-terminated string for
    // every code, including unknown ones.
    unsafe { CStr::from_ptr(sys::hipGetErrorString(code)) }
        .to_string_lossy()
        .into_owned()
}

/// `hipGetErrorName` of a HIP status code, e.g. `hipErrorNoDevice`.
pub fn error_name(code: i32) -> String {
    // SAFETY: hipGetErrorName returns a static NUL-terminated string for
    // every code, including unknown ones.
    unsafe { CStr::from_ptr(sys::hipGetErrorName(code)) }
        .to_string_lossy()
        .into_owned()
}
