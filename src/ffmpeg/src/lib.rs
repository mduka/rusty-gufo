//! FFmpeg video decoding for embedding inputs, loaded on the first video
//! (`video_processor.cpp`'s `FFmpegApi` and `Decoder`).
//!
//! gufo does not link FFmpeg: libavformat, libavcodec, libavutil and
//! libswscale are opened with `dlopen` from the versioned files the build
//! found, so text, image and audio services start without FFmpeg's codec
//! initialisation and run without FFmpeg installed. The struct fields come
//! from a small C file compiled against the build's headers
//! (`shim/fields.c`); a library of another major version is refused, since
//! FFmpeg keeps struct layouts only within one.
//!
//! The only `unsafe` of this crate is the FFmpeg boundary: loading, the
//! calls, and the callbacks FFmpeg makes into the decoder.

mod api;
mod decoder;
mod library;

pub use decoder::{Budget, Decoder, Frame};

/// Why a video could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The FFmpeg libraries or one of their functions could not be loaded
    /// (`platform::LibraryError`): a server problem, not the input's.
    Library(String),
    /// The input was refused, exceeded a bound, failed to decode or was
    /// cancelled (every other exception of the C++ decoder).
    Invalid(String),
}

impl Error {
    pub fn message(&self) -> &str {
        match self {
            Self::Library(message) | Self::Invalid(message) => message,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {}

/// Loads the FFmpeg libraries unless an earlier call did; a failure is
/// retried by the next call.
pub fn load() -> Result<(), Error> {
    api::api().map(|_| ())
}
