//! Encoded audio to mono 16-kHz samples (`audio_decoder.cpp`). The C++
//! engine pins transformers' librosa backend: libsndfile float32, the
//! arithmetic channel mean, then SOXR_HQ resampling and ceil-length zero
//! padding. The port decodes in pure Rust (hound, claxon, lewton, nanomp3)
//! and resamples with rubato: integer PCM is normalised like libsndfile, so
//! 16-kHz PCM WAV gives the same samples; compressed codecs and the
//! resampler may differ from libsndfile and libsoxr (docs/development/PARITY.md). From the
//! samples on, preprocessing and the encoder are those of C++.

use std::io::Cursor;
use std::time::{Duration, Instant};

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use crate::audio_processor::AUDIO_SAMPLE_RATE;

const MAX_ENCODED_BYTES: usize = 20 * 1024 * 1024;
const MAX_SECONDS: usize = 30;
const BLOCK_FRAMES: usize = 4096;

/// Why `decode_audio` produced no samples: malformed or out-of-bounds input,
/// or a cancelled, timed-out or failed decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioDecodeError {
    Invalid(String),
    Failed(String),
}

impl std::fmt::Display for AudioDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) | Self::Failed(message) => f.write_str(message),
        }
    }
}

fn invalid(message: &str) -> AudioDecodeError {
    AudioDecodeError::Invalid(message.into())
}

const UNDECODABLE: &str = "embedding audio could not be decoded";
const BOUNDS: &str = "embedding audio exceeds 30 seconds, 8 channels or 8–192 kHz bounds";
const TOO_LONG: &str = "embedding audio exceeds 30 seconds";
const TRUNCATED: &str = "embedding audio is truncated or decoding failed";

/// The decode's cancellation and its 15-second deadline (`Reader::Stop`).
struct Stop<'a> {
    is_cancelled: Option<&'a dyn Fn() -> bool>,
    deadline: Instant,
    stopped: bool,
}

impl Stop<'_> {
    fn stop(&mut self) -> bool {
        self.stopped = self.stopped
            || self.is_cancelled.is_some_and(|cancelled| cancelled())
            || Instant::now() > self.deadline;
        self.stopped
    }
}

/// Interleaved float frames decoded so far, mixed down to mono as they
/// arrive with libsndfile's channel mean.
struct Mixer {
    channels: usize,
    rate: usize,
    mono: Vec<f32>,
}

impl Mixer {
    fn new(
        channels: usize,
        rate: usize,
        frames: Option<u64>,
        estimated: bool,
    ) -> Result<Self, AudioDecodeError> {
        let max_frames = (rate * MAX_SECONDS) as u64;
        if !(1..=8).contains(&channels)
            || !(8000..=192_000).contains(&rate)
            || frames == Some(0)
            || (!estimated && frames.is_some_and(|frames| frames > max_frames))
        {
            return Err(invalid(BOUNDS));
        }
        let capacity = frames.map_or(0, |frames| frames.min(max_frames)) as usize;
        Ok(Self {
            channels,
            rate,
            mono: Vec::with_capacity(capacity),
        })
    }

    /// Appends one block of interleaved frames.
    fn push(&mut self, block: &[f32]) -> Result<(), AudioDecodeError> {
        let frames = block.len() / self.channels;
        if self.mono.len() + frames > self.rate * MAX_SECONDS {
            return Err(invalid(TOO_LONG));
        }
        for frame in block.chunks_exact(self.channels) {
            let mut sum = 0.0f32;
            for &value in frame {
                if !value.is_finite() {
                    return Err(invalid("embedding audio has nonfinite samples"));
                }
                sum += value;
            }
            self.mono.push(sum / self.channels as f32);
        }
        Ok(())
    }
}

/// Feeds `samples` to `mixer` in blocks of 4,096 frames, polling `stop`
/// before each block like the C++ read loop.
fn push_blocks(
    mixer: &mut Mixer,
    stop: &mut Stop<'_>,
    samples: &[f32],
) -> Result<(), AudioDecodeError> {
    for block in samples.chunks(BLOCK_FRAMES * mixer.channels) {
        if stop.stop() {
            return Err(AudioDecodeError::Failed(
                "embedding audio decoding cancelled or timed out".into(),
            ));
        }
        mixer.push(block)?;
    }
    Ok(())
}

/// libsndfile's float view of `bits`-bit integer PCM: a power-of-two scale.
fn normalize(value: i32, bits: u32) -> f32 {
    value as f32 * (1.0 / (1u64 << (bits - 1)) as f32)
}

fn decode_wav(bytes: &[u8], stop: &mut Stop<'_>) -> Result<Mixer, AudioDecodeError> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes)).map_err(|_| invalid(UNDECODABLE))?;
    let spec = reader.spec();
    let frames = u64::from(reader.duration());
    let mut mixer = Mixer::new(
        usize::from(spec.channels),
        spec.sample_rate as usize,
        Some(frames),
        false,
    )?;
    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Float if spec.bits_per_sample == 32 => {
            reader.samples::<f32>().collect()
        }
        hound::SampleFormat::Int if (8..=32).contains(&spec.bits_per_sample) => {
            let bits = u32::from(spec.bits_per_sample);
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| normalize(value, bits)))
                .collect()
        }
        _ => return Err(invalid(UNDECODABLE)),
    };
    push_blocks(&mut mixer, stop, &samples.map_err(|_| invalid(TRUNCATED))?)?;
    if mixer.mono.len() as u64 != frames {
        return Err(invalid(TRUNCATED));
    }
    Ok(mixer)
}

fn decode_flac(bytes: &[u8], stop: &mut Stop<'_>) -> Result<Mixer, AudioDecodeError> {
    let mut reader =
        claxon::FlacReader::new(Cursor::new(bytes)).map_err(|_| invalid(UNDECODABLE))?;
    let info = reader.streaminfo();
    let mut mixer = Mixer::new(
        info.channels as usize,
        info.sample_rate as usize,
        info.samples,
        false,
    )?;
    let bits = info.bits_per_sample;
    let samples: Result<Vec<f32>, _> = reader
        .samples()
        .map(|sample| sample.map(|value| normalize(value, bits)))
        .collect();
    push_blocks(&mut mixer, stop, &samples.map_err(|_| invalid(TRUNCATED))?)?;
    if info
        .samples
        .is_some_and(|frames| mixer.mono.len() as u64 != frames)
    {
        return Err(invalid(TRUNCATED));
    }
    Ok(mixer)
}

fn decode_ogg(bytes: &[u8], stop: &mut Stop<'_>) -> Result<Mixer, AudioDecodeError> {
    let mut reader = lewton::inside_ogg::OggStreamReader::new(Cursor::new(bytes))
        .map_err(|_| invalid(UNDECODABLE))?;
    let channels = usize::from(reader.ident_hdr.audio_channels);
    let rate = reader.ident_hdr.audio_sample_rate as usize;
    let mut mixer = Mixer::new(channels, rate, None, false)?;
    loop {
        if stop.stop() {
            return Err(AudioDecodeError::Failed(
                "embedding audio decoding cancelled or timed out".into(),
            ));
        }
        match reader.read_dec_packet_generic::<lewton::samples::InterleavedSamples<f32>>() {
            Ok(Some(packet)) => mixer.push(&packet.samples)?,
            Ok(None) => break,
            Err(_) => return Err(invalid(TRUNCATED)),
        }
    }
    Ok(mixer)
}

fn decode_mp3(bytes: &[u8], stop: &mut Stop<'_>) -> Result<Mixer, AudioDecodeError> {
    let mut reader = nanomp3::SliceReader::<f32>::new(bytes);
    let channels = reader
        .channels()
        .map(|channels| usize::from(channels.num()))
        .ok_or_else(|| invalid(UNDECODABLE))?;
    // Without a Xing/Info header the length is only an estimate: like
    // soundfile under librosa, keep the samples that decode. The 30-second
    // bound applies while decoding. The reader trims a LAME tag's encoder
    // delay and padding, as libsndfile's mpg123 does.
    let mut mixer = Mixer::new(channels, reader.sample_rate() as usize, None, true)?;
    let mut block = vec![0.0f32; BLOCK_FRAMES * channels];
    loop {
        if stop.stop() {
            return Err(AudioDecodeError::Failed(
                "embedding audio decoding cancelled or timed out".into(),
            ));
        }
        let count = reader.read(&mut block).map_err(|_| invalid(TRUNCATED))?;
        if count == 0 {
            break;
        }
        mixer.push(&block[..count])?;
    }
    Ok(mixer)
}

enum Container {
    Wav,
    Flac,
    Ogg,
    Mp3,
}

/// The container by its leading bytes, as libsndfile probes them.
fn container(bytes: &[u8]) -> Option<Container> {
    if bytes.len() >= 12
        && (&bytes[..4] == b"RIFF" || &bytes[..4] == b"RF64")
        && &bytes[8..12] == b"WAVE"
    {
        return Some(Container::Wav);
    }
    if bytes.starts_with(b"fLaC") {
        return Some(Container::Flac);
    }
    if bytes.starts_with(b"OggS") {
        return Some(Container::Ogg);
    }
    if bytes.starts_with(b"ID3")
        || (bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0)
    {
        return Some(Container::Mp3);
    }
    None
}

/// `DecodeAudio`: mono float samples at 16 kHz. Bounded to 30 seconds, 8
/// channels, 8 to 192 kHz and 20 MiB of encoded bytes.
pub fn decode_audio(
    bytes: &[u8],
    is_cancelled: Option<&dyn Fn() -> bool>,
) -> Result<Vec<f32>, AudioDecodeError> {
    if bytes.is_empty() || bytes.len() > MAX_ENCODED_BYTES {
        return Err(invalid("embedding audio is empty or exceeds 20 MiB"));
    }
    let mut stop = Stop {
        is_cancelled,
        deadline: Instant::now() + Duration::from_secs(15),
        stopped: false,
    };
    if stop.stop() {
        return Err(invalid(UNDECODABLE));
    }
    let mixer = match container(bytes) {
        Some(Container::Wav) => decode_wav(bytes, &mut stop)?,
        Some(Container::Flac) => decode_flac(bytes, &mut stop)?,
        Some(Container::Ogg) => decode_ogg(bytes, &mut stop)?,
        Some(Container::Mp3) => decode_mp3(bytes, &mut stop)?,
        None => return Err(invalid("embedding audio supports WAV, FLAC, Ogg and MP3")),
    };
    if stop.stopped || mixer.mono.is_empty() {
        return Err(invalid(TRUNCATED));
    }
    if mixer.rate == AUDIO_SAMPLE_RATE {
        return Ok(mixer.mono);
    }
    let Some(resampled) = resample(&mixer.mono, mixer.rate).filter(|_| !stop.stop()) else {
        return Err(AudioDecodeError::Failed(
            "embedding audio resampling failed or cancelled".into(),
        ));
    };
    // librosa's fix_length uses ceil, padding with zero.
    let length = (mixer.mono.len() * AUDIO_SAMPLE_RATE).div_ceil(mixer.rate);
    let mut output = resampled;
    output.resize(length, 0.0);
    Ok(output)
}

/// `mono` at `rate` resampled to 16 kHz with the startup delay trimmed,
/// through rubato's FFT resampler. Of rubato's resamplers it is the closest
/// to SOXR_HQ on librosa's fixtures; content near the new 8-kHz Nyquist
/// frequency still passes their different transition bands differently.
fn resample(mono: &[f32], rate: usize) -> Option<Vec<f32>> {
    let mut resampler = Fft::<f32>::new(rate, AUDIO_SAMPLE_RATE, 1024, 1, FixedSync::Input).ok()?;
    let input = InterleavedSlice::new(mono, 1, mono.len()).ok()?;
    Some(
        resampler
            .process_all(&input, mono.len(), None)
            .ok()?
            .take_data(),
    )
}
