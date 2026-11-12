//! The in-memory video decoder of `video_processor.cpp` (`Budget`,
//! `Decoder`): never opens nested URLs or files, selects the first video
//! stream and bounds the encoded and decoded work.

use std::cell::Cell;
use std::ffi::{c_int, c_void};
use std::ptr::{self, NonNull};
use std::time::{Duration, Instant};

use crate::Error;
use crate::api::{self, AVCodecContext, AVDictionary, AVFormatContext, AVFrame, AVIOContext};
use crate::api::{AVPacket, Api};

const DEADLINE: Duration = Duration::from_secs(15);
const IO_BUFFER: usize = 32768;
const MAX_STREAMS: usize = 8;
const MAX_FRAMES: usize = 4096;
const MAX_FRAME_PIXELS: usize = 4 * 1024 * 1024;
const MAX_DECODED_PIXELS: usize = 512 * 1024 * 1024;

fn bad_alloc() -> Error {
    Error::Invalid("std::bad_alloc".into())
}

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

/// Cancellation, a 15-second deadline from its creation, and the pixels
/// decoded over every pass of one video (`Budget`).
pub struct Budget<'a> {
    cancelled: Option<&'a dyn Fn() -> bool>,
    deadline: Instant,
    pixels: Cell<usize>,
}

impl<'a> Budget<'a> {
    pub fn new(cancelled: Option<&'a dyn Fn() -> bool>) -> Self {
        Self {
            cancelled,
            deadline: Instant::now() + DEADLINE,
            pixels: Cell::new(0),
        }
    }

    fn stop(&self) -> bool {
        self.cancelled.is_some_and(|cancelled| cancelled()) || Instant::now() > self.deadline
    }

    pub fn check(&self) -> Result<(), Error> {
        if self.stop() {
            return Err(invalid("embedding video decoding cancelled or timed out"));
        }
        Ok(())
    }
}

/// The encoded bytes the custom IO context reads, at a fixed address for
/// FFmpeg's callbacks.
struct Source<'a> {
    bytes: &'a [u8],
    position: usize,
    budget: &'a Budget<'a>,
}

unsafe extern "C" fn read(opaque: *mut c_void, out: *mut u8, count: c_int) -> c_int {
    // SAFETY: `opaque` is the decoder's leaked `Source`, which outlives the
    // IO context and is reached only through this address; FFmpeg calls back
    // on the decoding thread only.
    let source = unsafe { &mut *opaque.cast::<Source<'_>>() };
    if source.budget.stop() {
        return api::gufo_averror_exit;
    }
    // `std::min<std::size_t>(count, ...)`: a negative count converts to a
    // huge size.
    let n = (count as usize).min(source.bytes.len() - source.position);
    if n == 0 {
        return api::gufo_averror_eof;
    }
    // SAFETY: FFmpeg's buffer holds `count` >= `n` bytes; the source range is
    // in bounds; they do not overlap.
    unsafe { ptr::copy_nonoverlapping(source.bytes.as_ptr().add(source.position), out, n) };
    source.position += n;
    n as c_int
}

unsafe extern "C" fn seek(opaque: *mut c_void, offset: i64, whence: c_int) -> i64 {
    // SAFETY: as in `read`.
    let source = unsafe { &mut *opaque.cast::<Source<'_>>() };
    if source.budget.stop() {
        return i64::from(api::gufo_averror_exit);
    }
    let size = source.bytes.len() as i64;
    if whence == api::gufo_avseek_size {
        return size;
    }
    let base = match whence & !api::gufo_avseek_force {
        libc::SEEK_SET => 0,
        libc::SEEK_CUR => source.position as i64,
        libc::SEEK_END => size,
        _ => -1,
    };
    if base < 0 || offset < -base || offset > size - base {
        return i64::from(api::gufo_averror_einval);
    }
    source.position = (base + offset) as usize;
    source.position as i64
}

unsafe extern "C" fn interrupt(opaque: *mut c_void) -> c_int {
    // SAFETY: `opaque` is the decoder's `Budget`, which outlives the format
    // context.
    let budget = unsafe { &*opaque.cast::<Budget<'_>>() };
    c_int::from(budget.stop())
}

/// A dictionary built for one FFmpeg call and freed after it.
struct Dictionary<'a> {
    api: &'a Api,
    raw: *mut AVDictionary,
}

impl<'a> Dictionary<'a> {
    fn new(api: &'a Api, entries: &[(&std::ffi::CStr, &std::ffi::CStr)]) -> Self {
        let mut dictionary = Self {
            api,
            raw: ptr::null_mut(),
        };
        for (key, value) in entries {
            // SAFETY: NUL-terminated strings, copied by FFmpeg (flags 0).
            unsafe { (api.av_dict_set)(&mut dictionary.raw, key.as_ptr(), value.as_ptr(), 0) };
        }
        dictionary
    }
}

impl Drop for Dictionary<'_> {
    fn drop(&mut self) {
        // SAFETY: null or a dictionary from `av_dict_set`, freed once.
        unsafe { (self.api.av_dict_free)(&mut self.raw) };
    }
}

/// One decoded frame, valid during the decode callback.
pub struct Frame<'d> {
    api: &'d Api,
    raw: *const AVFrame,
}

impl Frame<'_> {
    pub fn width(&self) -> i32 {
        // SAFETY: a frame the decoder just received.
        unsafe { api::gufo_avframe_width(self.raw) }
    }

    pub fn height(&self) -> i32 {
        // SAFETY: as above.
        unsafe { api::gufo_avframe_height(self.raw) }
    }

    /// Packed RGB24 rows of the frame's size, converted by swscale's bicubic
    /// filter.
    pub fn to_rgb24(&self) -> Result<Vec<u8>, Error> {
        let failed = || invalid("embedding video RGB conversion failed");
        let (width, height) = (self.width(), self.height());
        // SAFETY: as above.
        let format = unsafe { api::gufo_avframe_format(self.raw) };
        // SAFETY: valid dimensions and pixel formats of a decoded frame; no
        // filters or parameters.
        let conversion = unsafe {
            (self.api.sws_get_context)(
                width,
                height,
                format,
                width,
                height,
                api::gufo_av_pix_fmt_rgb24,
                api::gufo_sws_bicubic,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
            )
        };
        if conversion.is_null() {
            return Err(failed());
        }
        // The decoder bounded the dimensions to positive values.
        let mut pixels = vec![0u8; width as usize * height as usize * 3];
        // swscale reads four plane pointers and strides; RGB24 uses one.
        let output = [
            pixels.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        ];
        let stride = [width * 3, 0, 0, 0];
        // SAFETY: the frame's planes and strides describe `height` rows of
        // its format; the output holds `height` rows of `width * 3` bytes.
        let scaled = unsafe {
            (self.api.sws_scale)(
                conversion,
                api::gufo_avframe_data(self.raw),
                api::gufo_avframe_linesize(self.raw),
                0,
                height,
                output.as_ptr(),
                stride.as_ptr(),
            )
        };
        // SAFETY: the context from `sws_getContext`, freed once.
        unsafe { (self.api.sws_free_context)(conversion) };
        if scaled != height {
            return Err(failed());
        }
        Ok(pixels)
    }
}

/// One pass over an encoded video held in memory (`Decoder`).
pub struct Decoder<'a> {
    api: &'static Api,
    budget: &'a Budget<'a>,
    /// A leaked box, freed by `Drop` after the IO context: FFmpeg's callbacks
    /// reach it through this address only.
    source: NonNull<Source<'a>>,
    io: *mut AVIOContext,
    format: *mut AVFormatContext,
    codec: *mut AVCodecContext,
    frame: *mut AVFrame,
    packet: *mut AVPacket,
    stream: c_int,
}

impl<'a> Decoder<'a> {
    /// Opens the container and the decoder of its first video stream
    /// (`Open`). MP4/MOV and WebM/Matroska only, without protocols or
    /// external data references; probing and decoding run single-threaded
    /// with a 4M-pixel frame bound.
    pub fn open(bytes: &'a [u8], budget: &'a Budget<'a>) -> Result<Self, Error> {
        let api = api::api()?;
        let mut decoder = Self {
            api,
            budget,
            source: NonNull::from(Box::leak(Box::new(Source {
                bytes,
                position: 0,
                budget,
            }))),
            io: ptr::null_mut(),
            format: ptr::null_mut(),
            codec: ptr::null_mut(),
            frame: ptr::null_mut(),
            packet: ptr::null_mut(),
            stream: -1,
        };
        decoder.open_streams()?;
        Ok(decoder)
    }

    fn open_streams(&mut self) -> Result<(), Error> {
        let api = self.api;
        let budget = self.budget;
        budget.check()?;
        // SAFETY: plain allocation; null on failure.
        let buffer = unsafe { (api.av_malloc)(IO_BUFFER) }.cast::<u8>();
        if buffer.is_null() {
            return Err(bad_alloc());
        }
        let source = self.source.as_ptr();
        // SAFETY: an `av_malloc` buffer of `IO_BUFFER` bytes, which the
        // context owns from here; the callbacks receive the leaked source,
        // which outlives the context (`Drop`).
        self.io = unsafe {
            (api.avio_alloc_context)(
                buffer,
                IO_BUFFER as c_int,
                0,
                source.cast(),
                Some(read),
                None,
                Some(seek),
            )
        };
        if self.io.is_null() {
            // SAFETY: the buffer the context did not take.
            unsafe { (api.av_free)(buffer.cast()) };
            return Err(bad_alloc());
        }
        // SAFETY: plain allocation; null on failure.
        self.format = unsafe { (api.avformat_alloc_context)() };
        if self.format.is_null() {
            return Err(bad_alloc());
        }
        let budget_pointer = ptr::from_ref(budget).cast_mut().cast::<c_void>();
        // SAFETY: a fresh format context; the budget outlives it and is only
        // read through the callback.
        unsafe {
            api::gufo_avformat_set_custom_io(self.format, self.io);
            api::gufo_avformat_set_interrupt(self.format, interrupt, budget_pointer);
            api::gufo_avformat_set_limits(
                self.format,
                MAX_STREAMS as c_int,
                1024 * 1024,
                2 * api::gufo_av_time_base,
            );
        }
        let mut options = Dictionary::new(
            api,
            &[
                (c"format_whitelist", c"mov,matroska,webm"),
                (c"protocol_whitelist", c""),
                // MOV external data references must never trigger
                // additional IO.
                (c"enable_drefs", c"0"),
            ],
        );
        // SAFETY: the context and options above; on failure FFmpeg frees the
        // context and nulls `self.format`, leaving the custom IO to `Drop`.
        let opened = unsafe {
            (api.avformat_open_input)(&mut self.format, ptr::null(), ptr::null(), &mut options.raw)
        };
        drop(options);
        if opened < 0 {
            return Err(invalid("embedding video container could not be opened"));
        }
        // Probing may itself decode a frame. Apply limits before that decoder
        // is created, not only to the persistent codec used for the two
        // passes.
        let mut probe_options: Vec<Dictionary<'_>> = (0..MAX_STREAMS)
            .map(|_| Dictionary::new(api, &[(c"max_pixels", c"4194304"), (c"threads", c"1")]))
            .collect();
        let mut raw: Vec<*mut AVDictionary> = probe_options.iter().map(|o| o.raw).collect();
        // SAFETY: one dictionary per possible stream (`max_streams`); FFmpeg
        // may replace entries with the unused options, which are freed below.
        let probed = unsafe { (api.avformat_find_stream_info)(self.format, raw.as_mut_ptr()) };
        for (option, raw) in probe_options.iter_mut().zip(raw) {
            option.raw = raw;
        }
        drop(probe_options);
        if probed < 0 {
            return Err(invalid(
                "embedding video supports valid MP4/MOV and WebM/Matroska containers",
            ));
        }
        // SAFETY: an opened context; streams below `nb_streams` exist.
        unsafe {
            for i in 0..api::gufo_avformat_nb_streams(self.format) {
                let parameters =
                    api::gufo_avstream_codecpar(api::gufo_avformat_stream(self.format, i));
                if api::gufo_avcodecpar_codec_type(parameters) == api::gufo_avmedia_type_video {
                    self.stream = i as c_int;
                    break;
                }
            }
        }
        // SAFETY: an opened context.
        let duration = unsafe { api::gufo_avformat_duration(self.format) };
        if duration != api::gufo_av_nopts_value && duration > 300 * api::gufo_av_time_base {
            return Err(invalid("embedding video exceeds 300 seconds"));
        }
        if self.stream < 0 {
            return Err(invalid("embedding video has no video stream"));
        }
        // SAFETY: the selected stream exists.
        let parameters = unsafe {
            api::gufo_avstream_codecpar(api::gufo_avformat_stream(self.format, self.stream as u32))
        };
        // SAFETY: a codec ID read from the stream's parameters.
        let implementation =
            unsafe { (api.avcodec_find_decoder)(api::gufo_avcodecpar_codec_id(parameters)) };
        if implementation.is_null() {
            return Err(invalid("unsupported embedding video codec"));
        }
        // SAFETY: a decoder from `avcodec_find_decoder`.
        self.codec = unsafe { (api.avcodec_alloc_context3)(implementation) };
        if self.codec.is_null() {
            return Err(bad_alloc());
        }
        // SAFETY: the context and the stream's parameters.
        if unsafe { (api.avcodec_parameters_to_context)(self.codec, parameters) } < 0 {
            return Err(invalid("invalid embedding video codec parameters"));
        }
        // SAFETY: an unopened codec context.
        unsafe {
            api::gufo_avcodec_set_limits(
                self.codec,
                1,
                MAX_FRAME_PIXELS as i64,
                api::gufo_av_ef_explode,
            );
        }
        // SAFETY: the context and the decoder it was allocated for.
        if unsafe { (api.avcodec_open2)(self.codec, implementation, ptr::null_mut()) } < 0 {
            return Err(invalid("embedding video decoder could not open"));
        }
        // SAFETY: plain allocations; null on failure.
        self.frame = unsafe { (api.av_frame_alloc)() };
        // SAFETY: as above.
        self.packet = unsafe { (api.av_packet_alloc)() };
        if self.frame.is_null() || self.packet.is_null() {
            return Err(bad_alloc());
        }
        Ok(())
    }

    /// The video stream's average frame rate (`av_q2d(avg_frame_rate)`).
    pub fn average_frame_rate(&self) -> f64 {
        // SAFETY: an opened context with a selected video stream.
        unsafe {
            api::gufo_avstream_avg_frame_rate(api::gufo_avformat_stream(
                self.format,
                self.stream as u32,
            ))
        }
    }

    /// The frame count the container declares for the video stream; zero or
    /// negative when unknown.
    pub fn declared_frames(&self) -> i64 {
        // SAFETY: as above.
        unsafe {
            api::gufo_avstream_nb_frames(api::gufo_avformat_stream(self.format, self.stream as u32))
        }
    }

    /// Decodes every frame of the video stream in order and passes each
    /// frame and its index to `consume`; returns the frame count (`Decode`).
    /// At most 4,096 frames of one size, at most 4M pixels each and 512M
    /// over every pass of the budget.
    pub fn decode(
        &mut self,
        mut consume: impl FnMut(usize, &Frame<'_>) -> Result<(), Error>,
    ) -> Result<usize, Error> {
        let api = self.api;
        let budget = self.budget;
        let mut count = 0usize;
        let (mut width, mut height) = (0, 0);
        let mut receive = |decoder: &mut Self| -> Result<(), Error> {
            loop {
                budget.check()?;
                // SAFETY: the opened codec and the decoder's frame.
                let status = unsafe { (api.avcodec_receive_frame)(decoder.codec, decoder.frame) };
                if status == api::gufo_averror_eagain || status == api::gufo_averror_eof {
                    break;
                }
                if status < 0 {
                    return Err(invalid("embedding video frame decoding failed"));
                }
                let frame = Frame {
                    api,
                    raw: decoder.frame,
                };
                let (w, h) = (frame.width(), frame.height());
                if count == 0 {
                    (width, height) = (w, h);
                }
                if w != width || h != height {
                    return Err(invalid(
                        "embedding video frame dimensions change within clip",
                    ));
                }
                // `static_cast<std::size_t>(width) * height`.
                let pixels = (w as usize).wrapping_mul(h as usize);
                count += 1;
                if count > MAX_FRAMES
                    || w < 1
                    || h < 1
                    || pixels > MAX_FRAME_PIXELS
                    || pixels > MAX_DECODED_PIXELS - budget.pixels.get()
                {
                    return Err(invalid("embedding video exceeds decoded work bounds"));
                }
                budget.pixels.set(budget.pixels.get() + pixels);
                consume(count - 1, &frame)?;
                // SAFETY: the decoder's frame, handed back for the next one.
                unsafe { (api.av_frame_unref)(decoder.frame) };
            }
            Ok(())
        };
        let status = loop {
            // SAFETY: the opened context and the decoder's packet.
            let status = unsafe { (api.av_read_frame)(self.format, self.packet) };
            if status < 0 {
                break status;
            }
            budget.check()?;
            // SAFETY: a packet `av_read_frame` filled.
            if unsafe { api::gufo_avpacket_stream_index(self.packet) } == self.stream {
                // SAFETY: the opened codec and a filled packet.
                if unsafe { (api.avcodec_send_packet)(self.codec, self.packet) } < 0 {
                    return Err(invalid("embedding video packet decoding failed"));
                }
                receive(self)?;
            }
            // SAFETY: the decoder's packet, handed back for the next one.
            unsafe { (api.av_packet_unref)(self.packet) };
        };
        // Flush only after a complete read, as `||` short-circuits in C++.
        let flushed = status == api::gufo_averror_eof
            // SAFETY: a null packet flushes the opened codec.
            && unsafe { (api.avcodec_send_packet)(self.codec, ptr::null()) } >= 0;
        if !flushed {
            return Err(invalid("embedding video is truncated or decoding failed"));
        }
        receive(self)?;
        Ok(count)
    }
}

impl Drop for Decoder<'_> {
    fn drop(&mut self) {
        let api = self.api;
        // SAFETY: each pointer is null or owned by this decoder; the free
        // functions accept null and null the pointer. The format context is
        // closed before the custom IO it reads from.
        unsafe {
            (api.av_packet_free)(&mut self.packet);
            (api.av_frame_free)(&mut self.frame);
            (api.avcodec_free_context)(&mut self.codec);
            (api.avformat_close_input)(&mut self.format);
            if !self.io.is_null() {
                (api.av_freep)(api::gufo_avio_buffer(self.io).cast());
                (api.avio_context_free)(&mut self.io);
            }
        }
        // SAFETY: the box leaked in `open`; the IO context that called back
        // into it is gone.
        drop(unsafe { Box::from_raw(self.source.as_ptr()) });
    }
}
