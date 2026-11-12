//! The FFmpeg functions, loaded on the first video (`FFmpegApi`), and the
//! struct fields and constants of the build's headers (`shim/fields.c`).

use std::ffi::{CStr, c_char, c_double, c_int, c_uint, c_void};
use std::sync::{Mutex, OnceLock};

use crate::Error;
use crate::library::Library;

macro_rules! opaque {
    ($($name:ident),*) => {
        $(
            #[repr(C)]
            pub(crate) struct $name {
                _private: [u8; 0],
            }
        )*
    };
}
opaque!(
    AVIOContext,
    AVFormatContext,
    AVStream,
    AVCodecParameters,
    AVCodec,
    AVCodecContext,
    AVFrame,
    AVPacket,
    AVDictionary,
    SwsContext
);

pub(crate) type ReadPacket = unsafe extern "C" fn(*mut c_void, *mut u8, c_int) -> c_int;
pub(crate) type WritePacket = unsafe extern "C" fn(*mut c_void, *const u8, c_int) -> c_int;
pub(crate) type Seek = unsafe extern "C" fn(*mut c_void, i64, c_int) -> i64;
pub(crate) type Interrupt = unsafe extern "C" fn(*mut c_void) -> c_int;

// SAFETY: the declarations match `shim/fields.c`, which is compiled into
// this crate. The constants are immutable and initialised before `main`.
unsafe extern "C" {
    pub(crate) safe static gufo_averror_eof: c_int;
    pub(crate) safe static gufo_averror_exit: c_int;
    pub(crate) safe static gufo_averror_eagain: c_int;
    pub(crate) safe static gufo_averror_einval: c_int;
    pub(crate) safe static gufo_avseek_size: c_int;
    pub(crate) safe static gufo_avseek_force: c_int;
    pub(crate) safe static gufo_av_nopts_value: i64;
    pub(crate) safe static gufo_av_time_base: i64;
    pub(crate) safe static gufo_avmedia_type_video: c_int;
    pub(crate) safe static gufo_av_pix_fmt_rgb24: c_int;
    pub(crate) safe static gufo_sws_bicubic: c_int;
    pub(crate) safe static gufo_av_ef_explode: c_int;
    safe static gufo_libavformat_version_major: c_uint;
    safe static gufo_libavcodec_version_major: c_uint;
    safe static gufo_libavutil_version_major: c_uint;
    safe static gufo_libswscale_version_major: c_uint;

    pub(crate) fn gufo_avio_buffer(io: *mut AVIOContext) -> *mut *mut u8;
    pub(crate) fn gufo_avformat_set_custom_io(format: *mut AVFormatContext, io: *mut AVIOContext);
    pub(crate) fn gufo_avformat_set_interrupt(
        format: *mut AVFormatContext,
        callback: Interrupt,
        opaque: *mut c_void,
    );
    pub(crate) fn gufo_avformat_set_limits(
        format: *mut AVFormatContext,
        max_streams: c_int,
        probesize: i64,
        max_analyze_duration: i64,
    );
    pub(crate) fn gufo_avformat_nb_streams(format: *const AVFormatContext) -> c_uint;
    pub(crate) fn gufo_avformat_duration(format: *const AVFormatContext) -> i64;
    pub(crate) fn gufo_avformat_stream(format: *const AVFormatContext, i: c_uint) -> *mut AVStream;
    pub(crate) fn gufo_avstream_codecpar(stream: *const AVStream) -> *mut AVCodecParameters;
    pub(crate) fn gufo_avstream_avg_frame_rate(stream: *const AVStream) -> c_double;
    pub(crate) fn gufo_avstream_nb_frames(stream: *const AVStream) -> i64;
    pub(crate) fn gufo_avcodecpar_codec_type(parameters: *const AVCodecParameters) -> c_int;
    pub(crate) fn gufo_avcodecpar_codec_id(parameters: *const AVCodecParameters) -> c_int;
    pub(crate) fn gufo_avcodec_set_limits(
        codec: *mut AVCodecContext,
        thread_count: c_int,
        max_pixels: i64,
        err_recognition: c_int,
    );
    pub(crate) fn gufo_avframe_width(frame: *const AVFrame) -> c_int;
    pub(crate) fn gufo_avframe_height(frame: *const AVFrame) -> c_int;
    pub(crate) fn gufo_avframe_format(frame: *const AVFrame) -> c_int;
    pub(crate) fn gufo_avframe_data(frame: *const AVFrame) -> *const *const u8;
    pub(crate) fn gufo_avframe_linesize(frame: *const AVFrame) -> *const c_int;
    pub(crate) fn gufo_avpacket_stream_index(packet: *const AVPacket) -> c_int;
}

/// The library files the build's headers describe (`build.rs`).
const AVFORMAT: &str = env!("GUFO_AVFORMAT_LIBRARY");
const AVCODEC: &str = env!("GUFO_AVCODEC_LIBRARY");
const AVUTIL: &str = env!("GUFO_AVUTIL_LIBRARY");
const SWSCALE: &str = env!("GUFO_SWSCALE_LIBRARY");

/// The FFmpeg functions gufo calls, with the libraries that hold them. The
/// libraries are fields, so every function pointer stays valid as long as
/// the table.
pub(crate) struct Api {
    pub avio_alloc_context: unsafe extern "C" fn(
        *mut u8,
        c_int,
        c_int,
        *mut c_void,
        Option<ReadPacket>,
        Option<WritePacket>,
        Option<Seek>,
    ) -> *mut AVIOContext,
    pub avio_context_free: unsafe extern "C" fn(*mut *mut AVIOContext),
    pub avformat_alloc_context: unsafe extern "C" fn() -> *mut AVFormatContext,
    pub avformat_open_input: unsafe extern "C" fn(
        *mut *mut AVFormatContext,
        *const c_char,
        *const c_void,
        *mut *mut AVDictionary,
    ) -> c_int,
    pub avformat_close_input: unsafe extern "C" fn(*mut *mut AVFormatContext),
    pub avformat_find_stream_info:
        unsafe extern "C" fn(*mut AVFormatContext, *mut *mut AVDictionary) -> c_int,
    pub av_read_frame: unsafe extern "C" fn(*mut AVFormatContext, *mut AVPacket) -> c_int,
    pub avcodec_alloc_context3: unsafe extern "C" fn(*const AVCodec) -> *mut AVCodecContext,
    pub avcodec_parameters_to_context:
        unsafe extern "C" fn(*mut AVCodecContext, *const AVCodecParameters) -> c_int,
    pub avcodec_open2:
        unsafe extern "C" fn(*mut AVCodecContext, *const AVCodec, *mut *mut AVDictionary) -> c_int,
    /// `enum AVCodecID` has the size of `int`.
    pub avcodec_find_decoder: unsafe extern "C" fn(c_int) -> *const AVCodec,
    pub avcodec_free_context: unsafe extern "C" fn(*mut *mut AVCodecContext),
    pub avcodec_send_packet: unsafe extern "C" fn(*mut AVCodecContext, *const AVPacket) -> c_int,
    pub avcodec_receive_frame: unsafe extern "C" fn(*mut AVCodecContext, *mut AVFrame) -> c_int,
    pub av_packet_alloc: unsafe extern "C" fn() -> *mut AVPacket,
    pub av_packet_free: unsafe extern "C" fn(*mut *mut AVPacket),
    pub av_packet_unref: unsafe extern "C" fn(*mut AVPacket),
    pub av_malloc: unsafe extern "C" fn(usize) -> *mut c_void,
    pub av_free: unsafe extern "C" fn(*mut c_void),
    /// Takes the address of the pointer to free and null.
    pub av_freep: unsafe extern "C" fn(*mut c_void),
    pub av_dict_set:
        unsafe extern "C" fn(*mut *mut AVDictionary, *const c_char, *const c_char, c_int) -> c_int,
    pub av_dict_free: unsafe extern "C" fn(*mut *mut AVDictionary),
    pub av_frame_alloc: unsafe extern "C" fn() -> *mut AVFrame,
    pub av_frame_free: unsafe extern "C" fn(*mut *mut AVFrame),
    pub av_frame_unref: unsafe extern "C" fn(*mut AVFrame),
    /// `enum AVPixelFormat` has the size of `int`; the filters are unused.
    #[allow(clippy::type_complexity)]
    pub sws_get_context: unsafe extern "C" fn(
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        *mut c_void,
        *mut c_void,
        *const c_double,
    ) -> *mut SwsContext,
    pub sws_free_context: unsafe extern "C" fn(*mut SwsContext),
    pub sws_scale: unsafe extern "C" fn(
        *mut SwsContext,
        *const *const u8,
        *const c_int,
        c_int,
        c_int,
        *const *mut u8,
        *const c_int,
    ) -> c_int,
    _avformat: Library,
    _avcodec: Library,
    _avutil: Library,
    _swscale: Library,
}

impl Api {
    fn load() -> Result<Self, Error> {
        let avformat = Library::open(AVFORMAT)?;
        let avcodec = Library::open(AVCODEC)?;
        let avutil = Library::open(AVUTIL)?;
        let swscale = Library::open(SWSCALE)?;
        // The shim reads struct fields at the header's offsets, which FFmpeg
        // keeps only within one major version.
        for (library, name, expected) in [
            (
                &avformat,
                c"avformat_version",
                gufo_libavformat_version_major,
            ),
            (&avcodec, c"avcodec_version", gufo_libavcodec_version_major),
            (&avutil, c"avutil_version", gufo_libavutil_version_major),
            (&swscale, c"swscale_version", gufo_libswscale_version_major),
        ] {
            // SAFETY: `unsigned <name>(void)` in every FFmpeg library; called
            // while the library is open.
            let version: unsafe extern "C" fn() -> c_uint = unsafe { library.symbol(name) }?;
            // SAFETY: as above.
            let major = unsafe { version() } >> 16;
            if major != expected {
                return Err(Error::Library(format!(
                    "FFmpeg {}() reports major version {major}; gufo was built \
                     against {expected}",
                    name.to_string_lossy()
                )));
            }
        }
        fn get<T: Copy>(library: &Library, name: &CStr) -> Result<T, Error> {
            // SAFETY: each use below names an FFmpeg function with the C
            // declaration of the field it fills (FFmpeg 8 headers); the
            // pointers live in `Api` beside their library.
            unsafe { library.symbol(name) }
        }
        Ok(Self {
            avio_alloc_context: get(&avformat, c"avio_alloc_context")?,
            avio_context_free: get(&avformat, c"avio_context_free")?,
            avformat_alloc_context: get(&avformat, c"avformat_alloc_context")?,
            avformat_open_input: get(&avformat, c"avformat_open_input")?,
            avformat_close_input: get(&avformat, c"avformat_close_input")?,
            avformat_find_stream_info: get(&avformat, c"avformat_find_stream_info")?,
            av_read_frame: get(&avformat, c"av_read_frame")?,
            avcodec_alloc_context3: get(&avcodec, c"avcodec_alloc_context3")?,
            avcodec_parameters_to_context: get(&avcodec, c"avcodec_parameters_to_context")?,
            avcodec_open2: get(&avcodec, c"avcodec_open2")?,
            avcodec_find_decoder: get(&avcodec, c"avcodec_find_decoder")?,
            avcodec_free_context: get(&avcodec, c"avcodec_free_context")?,
            avcodec_send_packet: get(&avcodec, c"avcodec_send_packet")?,
            avcodec_receive_frame: get(&avcodec, c"avcodec_receive_frame")?,
            av_packet_alloc: get(&avcodec, c"av_packet_alloc")?,
            av_packet_free: get(&avcodec, c"av_packet_free")?,
            av_packet_unref: get(&avcodec, c"av_packet_unref")?,
            av_malloc: get(&avutil, c"av_malloc")?,
            av_free: get(&avutil, c"av_free")?,
            av_freep: get(&avutil, c"av_freep")?,
            av_dict_set: get(&avutil, c"av_dict_set")?,
            av_dict_free: get(&avutil, c"av_dict_free")?,
            av_frame_alloc: get(&avutil, c"av_frame_alloc")?,
            av_frame_free: get(&avutil, c"av_frame_free")?,
            av_frame_unref: get(&avutil, c"av_frame_unref")?,
            sws_get_context: get(&swscale, c"sws_getContext")?,
            sws_free_context: get(&swscale, c"sws_freeContext")?,
            sws_scale: get(&swscale, c"sws_scale")?,
            _avformat: avformat,
            _avcodec: avcodec,
            _avutil: avutil,
            _swscale: swscale,
        })
    }
}

static API: OnceLock<Api> = OnceLock::new();
static LOADING: Mutex<()> = Mutex::new(());

/// The table, loaded by the first call. A failed load is retried by the
/// next call, as a C++ function-local static whose initialiser threw.
pub(crate) fn api() -> Result<&'static Api, Error> {
    if let Some(api) = API.get() {
        return Ok(api);
    }
    let _loading = LOADING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(api) = API.get() {
        return Ok(api);
    }
    let api = Api::load()?;
    Ok(API.get_or_init(|| api))
}
