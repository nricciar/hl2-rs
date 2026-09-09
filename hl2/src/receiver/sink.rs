//! Audio sinks for the virtual receiver.
//!
//! The sink is the bottom of the pipeline — the demodulator hands it an
//! `i16` mono block and it does something *useful* with it: play it on a
//! sound card (the production route) or capture it for tests / analysis
//! (offline).
//!
//! Two sinks ship out of the box:
//!
//! * [`VecSink`] — in-memory capture, always available.
//! * [`AlsaSink`] — real ALSA playback via `cpal`, feature-gated on `alsa`.
//!
//! The trait is `i16`-only on purpose: a digital-mode demodulator that wants
//! a complex baseband (or 24-bit) output can add a *sibling* trait instead of
//! complicating the common one (PROTOCOL.md §16.3).

use alloc::boxed::Box;
use alloc::vec::Vec;

/// Sink error.
pub type SinkError = Box<dyn core::error::Error + Send + Sync>;

/// An audio sink takes demodulated `i16` mono samples and does something with
/// them (play, record, etc.).
///
/// `write` is called once per demodulator block — a few hundred to a few
/// thousand samples at the decimated rate (typically 4.8 kHz audio). Sinks
/// should keep the call fast (block-copy into a ring buffer) rather than
/// blocking synchronously.
///
/// A sink may be shared (via `Arc`) between virtual receivers; the ALSA
/// implementation is `Send + Sync` for exactly that reason.
pub trait AudioSink: Send {
    /// Write `samples` to the sink; returns the count actually accepted.
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError>;

    /// Flush any buffered audio.
    fn flush(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

/// A shared, bounded, thread-safe buffer of `i16` mono frames.
///
/// `Send + Sync`; every method is a short critical section (no `await` is
/// ever held across the lock, so `std::sync::Mutex` is correct and cheaper
/// than `tokio::sync::Mutex`). Oldest frames are dropped on overflow so a slow
/// drainer can never make the demod block.
#[cfg(feature = "std")]
#[derive(Debug)]
struct SharedAudioBuf {
    buf: std::sync::Mutex<Vec<i16>>,
    cap: usize,
}

#[cfg(feature = "std")]
impl SharedAudioBuf {
    fn new(cap: usize) -> Self {
        Self {
            buf: std::sync::Mutex::new(Vec::with_capacity(cap)),
            cap,
        }
    }

    /// Append `samples`, dropping the oldest frames if the buffer exceeds `cap`.
    fn push(&self, samples: &[i16]) {
        let mut guard = self.buf.lock().unwrap();
        guard.extend_from_slice(samples);
        if guard.len() > self.cap {
            let drop = guard.len() - self.cap;
            guard.drain(..drop);
        }
    }

    /// Consume up to `out.len()` frames into `out` (FIFO); return the count
    /// actually written (`≤ out.len()`). Frames beyond the returned count in
    /// `out` are untouched.
    fn drain_into(&self, out: &mut [i16]) -> usize {
        let mut guard = self.buf.lock().unwrap();
        let n = guard.len().min(out.len());
        out[..n].copy_from_slice(&guard[..n]);
        guard.drain(..n);
        n
    }

    fn len(&self) -> usize {
        self.buf.lock().unwrap().len()
    }

    fn is_empty(&self) -> bool {
        self.buf.lock().unwrap().is_empty()
    }
}

/// Default [`BufSink`] capacity in `i16` frames. Kept **small on purpose**
/// (~200 ms of 4 800 Hz audio) so the audio never sits queued for long
/// before it reaches the WebSocket — the fan-out task drains it in
/// ~24 ms ticks of ≤480 samples, and the demod writes ~240-sample (50 ms)
/// blocks, so 200 ms comfortably covers a handful of ticks without adding
/// perceptible end-to-end latency (a dropped/old FT8 cycle is the cost of
/// going larger). Bounded so a slow browser can't make the demod wait.
#[cfg(feature = "std")]
pub const BUF_SINK_DEFAULT_CAP: usize = 960;

/// An [`AudioSink`] that appends every `i16` block into a *shared* buffer so
/// another thread/task (the server's audio fan-out) can drain it at its own
/// pace. This is what lets the demod thread and the WebSocket broadcast be
/// decoupled the same way the `ssb` CLI decouples demod and ALSA.
///
/// `BufSink` is moved into a [`crate::receiver::VirtualReceiver`]; the paired
/// [`BufSinkHandle`] is what the API keeps for reading.
///
/// Only available under `std` (it uses `Arc<Mutex<...>>` for the shared
/// buffer).
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct BufSink {
    inner: std::sync::Arc<SharedAudioBuf>,
}

#[cfg(feature = "std")]
impl BufSink {
    /// Build a [`BufSink`] and a [`BufSinkHandle`] that share the same buffer.
    ///
    /// `cap` bounds the shared buffer (oldest dropped on overflow); pass
    /// [`BUF_SINK_DEFAULT_CAP`] (≈5 s) for the server's normal use.
    pub fn pair(cap: usize) -> (Self, BufSinkHandle) {
        let inner = std::sync::Arc::new(SharedAudioBuf::new(cap));
        (
            Self {
                inner: inner.clone(),
            },
            BufSinkHandle { inner },
        )
    }
}

#[cfg(feature = "std")]
impl Default for BufSink {
    fn default() -> Self {
        let (sink, _handle) = Self::pair(BUF_SINK_DEFAULT_CAP);
        sink
    }
}

/// A cloneable read handle for a [`BufSink`]'s shared buffer.
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct BufSinkHandle {
    inner: std::sync::Arc<SharedAudioBuf>,
}

#[cfg(feature = "std")]
impl BufSinkHandle {
    /// Frames currently queued.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Consume up to `out.len()` `i16` frames into `out` (FIFO); return the
    /// count actually written.
    pub fn drain_into(&self, out: &mut [i16]) -> usize {
        self.inner.drain_into(out)
    }

    /// Consume up to `max` frames and return them (fewer if fewer are queued).
    pub fn drain(&self, max: usize) -> Vec<i16> {
        let mut out = vec![0i16; max];
        let n = self.inner.drain_into(&mut out);
        out.truncate(n);
        out
    }
}

#[cfg(feature = "std")]
impl AudioSink for BufSink {
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError> {
        self.inner.push(samples);
        Ok(samples.len())
    }
}

/// An in-memory `AudioSink` that appends every block it receives.
///
/// Primarily a test / analysis sink, but also handy when an application wants
/// to drive its own audio path.
#[derive(Debug, Default)]
pub struct VecSink {
    buf: Vec<i16>,
    /// Optional frame cap (oldest dropped when full), to bound memory.
    max: Option<usize>,
}

impl VecSink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap the buffer at `max` frames.
    pub fn with_max(mut self, max: usize) -> Self {
        self.max = Some(max);
        self
    }

    /// The frames appended so far.
    pub fn samples(&self) -> &[i16] {
        &self.buf
    }

    /// Take the frames out, resetting the buffer.
    pub fn into_samples(self) -> Vec<i16> {
        self.buf
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

impl AudioSink for VecSink {
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError> {
        let n = samples.len();
        self.buf.extend_from_slice(samples);
        if let Some(max) = self.max {
            if self.buf.len() > max {
                self.buf.drain(..self.buf.len() - max);
            }
        }
        Ok(n)
    }
}

/// An [`AudioSink`] that discards every block it receives.
///
/// For pipelines where the audio is consumed *upstream* (a `RawSampleTap`
/// samples the pre-AGC `f32` and the AGC'd `i16` output goes nowhere) —
/// that is the auto-decode pipeline: no `BufSink` + WebSocket fan-out, no
/// buffer allocation, the demod just drops audio inline. The demod still
/// has to call `write` (the pipeline is `AudioSink`-based), so a no-op sink
/// is the cheapest possible implementation.
#[derive(Debug, Default)]
pub struct DropSink;

impl DropSink {
    pub const fn new() -> Self {
        Self
    }
}

impl AudioSink for DropSink {
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError> {
        Ok(samples.len())
    }
}

/// Bounded ring of `i16` frames, `Send + Sync`.
#[cfg(feature = "alsa")]
struct Ring {
    buf: std::sync::Mutex<Vec<i16>>,
}

#[cfg(feature = "alsa")]
impl Ring {
    fn new(capacity: usize) -> Self {
        Self {
            buf: std::sync::Mutex::new(Vec::with_capacity(capacity)),
        }
    }

    /// Append `samples`, trimming oldest if over the cap (~5 s at 48 kHz).
    fn push(&self, samples: &[i16]) {
        let mut guard = self.buf.lock().unwrap();
        guard.extend_from_slice(samples);
        let cap = 48_000 * 5;
        if guard.len() > cap {
            let drop = guard.len() - cap;
            guard.drain(..drop);
        }
    }

    /// Consume up to `out.len()` frames into `out`, zero-filling any shortfall.
    fn pop_into_i16(&self, out: &mut [i16]) -> usize {
        let mut guard = self.buf.lock().unwrap();
        let len = guard.len();
        let n = len.min(out.len());
        out[..n].copy_from_slice(&guard[..n]);
        guard.drain(..n);
        if n < out.len() {
            out[n..].fill(0);
        }
        n
    }
}

/// An ALSA-backed audio sink, feature-gated on `alsa`.
///
/// Owns a `cpal::Stream` (dropping it stops the audio thread) plus a [`Ring`]
/// that the audio thread drains.
#[cfg(feature = "alsa")]
pub struct AlsaSink {
    /// Kept alive for the lifetime of the sink so the audio thread keeps running.
    _stream: cpal::Stream,
    ring: std::sync::Arc<Ring>,
}

#[cfg(feature = "alsa")]
impl AlsaSink {
    /// Build an ALSA sink at `rate_hz` (should match the demodulator's output
    /// rate, e.g. 4 800 Hz).
    ///
    /// `device` is `None` → cpal's default output device.
    pub fn build(device: Option<cpal::Device>, rate_hz: u32) -> Result<Self, SinkError> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let dev = match device {
            Some(d) => d,
            None => host
                .default_output_device()
                .ok_or(SinkError::from(std::io::Error::other(
                    "no default ALSA output device",
                )))?,
        };

        let config = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(rate_hz),
            buffer_size: cpal::BufferSize::Default,
        };

        let ring = std::sync::Arc::new(Ring::new(4096));
        let audio_ring = ring.clone();

        let error_handler =
            Box::new(move |err: cpal::StreamError| eprintln!("[alsa] cpal stream error: {err:?}"));

        // i16 mono output; cpal invokes the callback with `&mut [i16]`.
        let stream = dev
            .build_output_stream::<i16, _, _>(
                &config,
                move |buf: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    audio_ring.pop_into_i16(buf);
                },
                error_handler,
                None,
            )
            .map_err(|e| Box::new(e) as SinkError)?;
        stream.play().map_err(|e| Box::new(e) as SinkError)?;

        Ok(Self {
            _stream: stream,
            ring,
        })
    }
}
/// Soundness: every field owned by [`AlsaSink`] is either `Send` (the
/// [`Ring`] is a `Mutex<Vec<i16>>`) or the `cpal::Stream` handle. cpal marks
/// `Stream` `!Send` only conservatively, for the Android AAudio case where a
/// stream must live on the thread that created it. For ALSA the handle is just
/// a control pointer to cpal's dedicated audio thread, and that thread's
/// callback only touches the `Arc<Ring>` (which is `Send + Sync`). Moving the
/// handle across threads is therefore safe.
#[cfg(feature = "alsa")]
unsafe impl Send for AlsaSink {}

#[cfg(feature = "alsa")]
impl AudioSink for AlsaSink {
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError> {
        self.ring.push(samples);
        Ok(samples.len())
    }
    fn flush(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vecsink_appends_frames() {
        let mut s = VecSink::new();
        s.write(&[1, 2, 3]).unwrap();
        s.write(&[4, 5, 6, 7]).unwrap();
        assert_eq!(s.samples(), [1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn vecsink_bounds_frames_when_max_set() {
        let mut s = VecSink::new().with_max(4);
        s.write(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(s.samples(), [2, 3, 4, 5]);
        s.write(&[6]).unwrap();
        assert_eq!(s.samples(), [3, 4, 5, 6]);
    }

    #[test]
    fn vecsink_into_samples_and_clear() {
        let mut s = VecSink::new();
        s.write(&[1, 2]).unwrap();
        s.clear();
        assert!(s.samples().is_empty());
        s.write(&[9]).unwrap();
        assert_eq!(s.into_samples(), [9]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn bufsink_drains_fifo() {
        let (mut s, h) = BufSink::pair(16);
        s.write(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(h.len(), 5);
        assert_eq!(h.drain_into(&mut [0, 0, 0]), 3);
        // The first 3 consumed; 2 remain (4,5).
        let rest: Vec<i16> = h.drain(8);
        assert_eq!(rest, vec![4, 5]);
        assert!(h.is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn bufsink_drops_oldest_over_cap() {
        let (mut s, h) = BufSink::pair(4);
        s.write(&[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(h.len(), 4);
        assert_eq!(h.drain(8), vec![3, 4, 5, 6]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn bufsink_shared_across_handles() {
        let (mut s, h1) = BufSink::pair(16);
        let h2 = h1.clone();
        s.write(&[9, 8, 7]).unwrap();
        assert_eq!(h2.drain(2), vec![9, 8]);
        assert_eq!(h1.drain(2), vec![7]);
    }
}
