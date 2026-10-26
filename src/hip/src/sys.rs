//! Raw HIP runtime declarations: only the subset Gufo uses.

use std::ffi::{c_char, c_void};

pub type HipError = i32;
pub type HipStream = *mut c_void;
pub type HipEvent = *mut c_void;
pub type HipGraph = *mut c_void;
pub type HipGraphExec = *mut c_void;
pub type HipGraphNode = *mut c_void;
pub type HipMemPool = *mut c_void;

pub const HIP_SUCCESS: HipError = 0;
pub const HIP_ERROR_NOT_READY: HipError = 600;
pub const HIP_STREAM_NON_BLOCKING: u32 = 0x1;
pub const HIP_EVENT_DEFAULT: u32 = 0x0;
pub const HIP_EVENT_DISABLE_TIMING: u32 = 0x2;
pub const HIP_HOST_MALLOC_DEFAULT: u32 = 0x0;
pub const HIP_HOST_MALLOC_MAPPED: u32 = 0x2;
pub const HIP_MEM_ATTACH_GLOBAL: u32 = 0x1;
pub const HIP_MEM_ADVISE_SET_PREFERRED_LOCATION: i32 = 3;
pub const HIP_MEM_ADVISE_SET_COARSE_GRAIN: i32 = 100;
pub const HIP_CPU_DEVICE_ID: i32 = -1;
pub const HIP_HOST_REGISTER_MAPPED: u32 = 0x2;
pub const HIP_HOST_REGISTER_READ_ONLY: u32 = 0x08;
pub const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
pub const HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL: i32 = 1;
pub const HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD: i32 = 0x4;
pub const HIP_MEM_POOL_ATTR_RESERVED_MEM_CURRENT: i32 = 0x5;

// SAFETY: signatures match hip_runtime_api.h of the qualified ROCm release.
unsafe extern "C" {
    pub fn hipGetErrorString(error: HipError) -> *const c_char;
    pub fn hipGetErrorName(error: HipError) -> *const c_char;
    pub fn hipGetDeviceCount(count: *mut i32) -> HipError;
    pub fn hipRuntimeGetVersion(version: *mut i32) -> HipError;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> HipError;

    pub fn hipStreamCreateWithFlags(stream: *mut HipStream, flags: u32) -> HipError;
    pub fn hipStreamDestroy(stream: HipStream) -> HipError;
    pub fn hipStreamSynchronize(stream: HipStream) -> HipError;
    pub fn hipStreamQuery(stream: HipStream) -> HipError;
    pub fn hipGetLastError() -> HipError;
    pub fn hipDeviceSynchronize() -> HipError;

    pub fn hipEventCreateWithFlags(event: *mut HipEvent, flags: u32) -> HipError;
    pub fn hipEventDestroy(event: HipEvent) -> HipError;
    pub fn hipEventRecord(event: HipEvent, stream: HipStream) -> HipError;
    pub fn hipEventSynchronize(event: HipEvent) -> HipError;
    pub fn hipEventQuery(event: HipEvent) -> HipError;
    pub fn hipEventElapsedTime(ms: *mut f32, start: HipEvent, stop: HipEvent) -> HipError;

    pub fn hipStreamBeginCapture(stream: HipStream, mode: i32) -> HipError;
    pub fn hipStreamEndCapture(stream: HipStream, graph: *mut HipGraph) -> HipError;
    pub fn hipGraphInstantiate(
        exec: *mut HipGraphExec,
        graph: HipGraph,
        error_node: *mut HipGraphNode,
        log: *mut c_char,
        log_size: usize,
    ) -> HipError;
    pub fn hipGraphDestroy(graph: HipGraph) -> HipError;
    pub fn hipGraphExecDestroy(exec: HipGraphExec) -> HipError;
    pub fn hipGraphLaunch(exec: HipGraphExec, stream: HipStream) -> HipError;

    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> HipError;
    pub fn hipMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> HipError;
    pub fn hipMemAdvise(ptr: *const c_void, count: usize, advice: i32, device: i32) -> HipError;
    pub fn hipMemPrefetchAsync(
        ptr: *const c_void,
        count: usize,
        device: i32,
        stream: HipStream,
    ) -> HipError;
    pub fn hipFree(ptr: *mut c_void) -> HipError;
    pub fn hipMallocAsync(ptr: *mut *mut c_void, size: usize, stream: HipStream) -> HipError;
    pub fn hipFreeAsync(ptr: *mut c_void, stream: HipStream) -> HipError;
    pub fn hipDeviceGetDefaultMemPool(pool: *mut HipMemPool, device: i32) -> HipError;
    pub fn hipMemPoolSetAttribute(pool: HipMemPool, attr: i32, value: *mut c_void) -> HipError;
    pub fn hipMemPoolGetAttribute(pool: HipMemPool, attr: i32, value: *mut c_void) -> HipError;
    pub fn hipMemsetAsync(dst: *mut c_void, value: i32, size: usize, stream: HipStream)
    -> HipError;
    pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> HipError;
    pub fn hipHostFree(ptr: *mut c_void) -> HipError;
    pub fn hipHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> HipError;
    pub fn hipHostUnregister(ptr: *mut c_void) -> HipError;
    pub fn hipHostGetDevicePointer(
        device: *mut *mut c_void,
        host: *mut c_void,
        flags: u32,
    ) -> HipError;
    pub fn hipMemcpy2DAsync(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32,
        stream: HipStream,
    ) -> HipError;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: i32,
        stream: HipStream,
    ) -> HipError;
    pub fn hipMemcpyWithStream(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: i32,
        stream: HipStream,
    ) -> HipError;
}
