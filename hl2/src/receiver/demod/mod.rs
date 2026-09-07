//! Demodulators for the virtual audio receiver.
//!
//! A [`Demodulator`] converts a block of complex I/Q baseband samples into
//! `i16` mono audio and writes it to an [`AudioSink`](super::sink::AudioSink).
//! The crate provides three concrete mode cores — SSB (USB/LSB voice),
//! AM (DSB-FC), and digital (FT8/JS8/FT4 12 kHz USB) — all of which are
//! [`DemodCore`] impls wrapped by the shared [`StandardDemod`] +
//! [`AudioEngine`] pair. [`make_demod`] / [`make_demod_tap`] (below) is the
//! single dispatch point for a new mode (PROTOCOL.md §16.3 / §16.3a).
//!
//! ## Layering
//!
//! ```text
//! complex I/Q ──► DemodCore::process ──► 0..=N f32s ──► AudioEngine ──► i16 → sink
//!                    (per-mode DSP)                      (AGC, tap)
//!                    ┐ SsbCore ┐ AmCore ┐ DigitalCore ┐    (shared, once)
//!                    └──────────────────────────────────────┴─► demodulator()
//! ```
//!
//! * **`DemodCore`** — the one method a new mode has to implement: per-sample
//!   DSP. See [`core`].
//! * **`AudioEngine`** — the mode-agnostic tail (DC-block + RMS AGC +
//!   pre-AGC [`RawSampleTap`] + `i16` sink write). See [`engine`].
//! * **`StandardDemod<C>`** — composes the two into a full [`Demodulator`].
//!   This is the only `Demodulator` impl in the crate; a new mode does not
//!   need one.
//! * **[`make_demod`] / [`make_demod_tap`]** (above) — the per-`Mode` dispatch
//!   that builds a mode's core, sanity-checks the rate, and hands it to the
//!   shared audio tail via [`DemodCore::demodulator`]. A new mode adds a
//!   `DemodCore` impl + a `Mode` variant + a match arm here. Shared DSP primitives (the Nco, the Kaiser-windowed windowed-sinc
//! windowed-sinc low-pass / Hilbert FIR, and the polyphase decimator) live
//! in [`dsp`] — every core builds on top of them.

pub mod am;
pub mod core;
pub mod digital;
pub mod dsp;
pub mod engine;
pub mod fm;
pub mod ssb;

use std::fmt;

use num_complex::Complex;

use super::sink::AudioSink;

// Re-export the mode cores + shared DSP + the shared types so module-root
// paths like `hl2::receiver::demod::{SsbCore, AmCore, DigitalCore, F32Fir,
// Nco, …}` (and `super::demod::RawSampleTap` from ft8/ft4/js8) resolve.
pub use am::AmCore;
pub use core::{DemodCore, StandardDemod};
pub use digital::DigitalCore;
pub use dsp::{F32Fir, F32FirState, KAISER_BETA, Nco, PolyphaseDecimator};
pub use engine::AudioEngine;
pub use fm::FmCore;
pub use ssb::SsbCore;

/// Per-mode aliases for [`StandardDemod`]: the full-tail demodulator type
/// for each mode's core.
pub type SsbDemodulator = StandardDemod<SsbCore>;
pub type AmDemodulator = StandardDemod<AmCore>;
pub type DigitalDemodulator = StandardDemod<DigitalCore>;
pub type FmDemodulator = StandardDemod<FmCore>;

/// The source rate is too close to the audio rate to decimate cleanly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateTooClose {
    pub src: u32,
    pub audio: u32,
}
impl fmt::Display for RateTooClose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "demod: source rate {} Hz must be at least 4× the audio rate {} Hz",
            self.src, self.audio
        )
    }
}
impl std::error::Error for RateTooClose {}

/// Demodulator error.
pub type DemodError = Box<dyn std::error::Error + Send + Sync>;

/// A block of complex I/Q baseband samples: `Vec<Complex<f32>>`.
pub type IqBlock = Vec<Complex<f32>>;

use crate::receiver::{AudioConfig, Mode, Sideband};

/// Guard: the polyphase decimation factor must be ≥ 4 (the cores compute
/// `m = source / audio` with plain integer division, so a too-close rate would
/// silently mis-decimate). Returns [`RateTooClose`] on violation.
fn ensure_decimable(source_rate_hz: u32, audio: AudioConfig) -> Result<(), DemodError> {
    if source_rate_hz as usize / (audio.rate_hz as usize) < 4 {
        return Err(Box::new(RateTooClose {
            src: source_rate_hz,
            audio: audio.rate_hz,
        }));
    }
    Ok(())
}

/// Build a `Box<dyn Demodulator>` for `mode`.
///
/// * `source_center_hz`: the signal's carrier offset from the source centre
///   (`Hz`); `0.0` for a baseband source.
/// * `bandwidth_hz`: the channel-select bandwidth (`Hz`); must be less than
///   the source rate.
///
/// This is the single dispatch point for a new mode (PROTOCOL.md §16.3 /
/// §16.3a): pick the `DemodCore`, sanity-check the rate, and hand it to the
/// shared audio tail via [`DemodCore::demodulator`].
pub fn make_demod(
    mode: Mode,
    source_rate_hz: u32,
    source_center_hz: f64,
    bandwidth_hz: u32,
    audio: AudioConfig,
) -> Result<Box<dyn Demodulator>, DemodError> {
    make_demod_tap(
        mode,
        source_rate_hz,
        source_center_hz,
        bandwidth_hz,
        audio,
        None,
    )
}

/// [`make_demod`] plus an optional pre-AGC [`RawSampleTap`] (`tap`). The
/// tap is the seam the FT8/JS8/FT4 slot decoders attach to: the demod still
/// emits AGC'd `i16` to the `CH_AUDIO` sink, while `tap` copies the
/// untouched pre-AGC `f32` stream to the decoder.
pub fn make_demod_tap(
    mode: Mode,
    source_rate_hz: u32,
    source_center_hz: f64,
    bandwidth_hz: u32,
    audio: AudioConfig,
    tap: Option<std::sync::Arc<dyn RawSampleTap>>,
) -> Result<Box<dyn Demodulator>, DemodError> {
    ensure_decimable(source_rate_hz, audio)?;
    Ok(match mode {
        Mode::Ft8 => digital::DigitalCore::new(source_rate_hz, source_center_hz, audio, "ft8")
            .demodulator(audio, tap),
        Mode::Js8 => digital::DigitalCore::new(source_rate_hz, source_center_hz, audio, "js8")
            .demodulator(audio, tap),
        Mode::Ft4 => digital::DigitalCore::new(source_rate_hz, source_center_hz, audio, "ft4")
            .demodulator(audio, tap),
        Mode::Am => am::AmCore::new(source_rate_hz, source_center_hz, bandwidth_hz, audio)
            .demodulator(audio, tap),
        // FM (standard) and NFM (narrow) share one demod core; they differ
        // only in the channel-select bandwidth (`bandwidth_hz`), which is
        // already resolved from `Mode::default_bandwidth_hz` (15 kHz FM /
        // 5 kHz NFM) and may be overridden via `ReceiverConfig::bandwidth_hz`.
        Mode::Fm => fm::FmCore::new(source_rate_hz, source_center_hz, bandwidth_hz, audio, "fm")
            .demodulator(audio, tap),
        Mode::FmNarrow => {
            fm::FmCore::new(source_rate_hz, source_center_hz, bandwidth_hz, audio, "nfm")
                .demodulator(audio, tap)
        }
        Mode::Ssb(side) => {
            ssb::SsbCore::new(side, source_rate_hz, source_center_hz, bandwidth_hz, audio)
                .demodulator(audio, tap)
        }
        // `SsbWide` is the wideband-complex pass-through placeholder; it runs
        // through the USB SSB pipeline today (PROTOCOL.md §16.3).
        Mode::SsbWide => ssb::SsbCore::new(
            Sideband::Usb,
            source_rate_hz,
            source_center_hz,
            bandwidth_hz,
            audio,
        )
        .demodulator(audio, tap),
    })
}

/// A demodulator: complex I/Q in, `i16` mono audio out, to a given sink.
/// Held as `Box<dyn Demodulator>` by [`crate::receiver::VirtualReceiver`],
/// fed one block at a time. `Send` so a receiver can be handed to a demod
/// thread — the pump and the demod are decoupled via the
/// [`crate::receiver::BasebandRing`].
pub trait Demodulator: Send {
    /// Demodulate one complex I/Q block into `sink`; returns audio frames
    /// written. The default implementation is a no-op (return `Ok(0)`);
    /// [`StandardDemod`] overrides this to run the core + engine.
    fn demod(&mut self, _: &IqBlock, _: &mut dyn AudioSink) -> Result<usize, DemodError> {
        Ok(0)
    }

    /// The audio format produced.
    fn audio_format(&self) -> super::AudioConfig {
        super::AudioConfig::default()
    }

    /// Flush any audio the demodulator is still buffering (e.g. a partial
    /// block that hadn't accumulated enough samples for normalisation). Safe
    /// on modes that don't buffer (no-op).
    fn flush_audio(&mut self, _sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        Ok(0)
    }

    /// Flush buffered audio into `sink` (alias for `flush_audio`, used by
    /// higher-level wrappers that call a uniform `flush`).
    fn flush(&mut self, sink: &mut dyn AudioSink) -> Result<(), DemodError> {
        self.flush_audio(sink).map(|_| ())
    }
}

/// A tap for capturing the *pre-AGC* decimated audio stream.
///
/// The SSB/digital pipeline is: NCO → SSB/digital branch → channel LPF →
/// decimate → **raw decimated f32 samples** → AGC (DC block + RMS target) →
/// i16 → sink. For the `CH_AUDIO` sink the AGC-normalised i16 output is what
/// the listener wants. For FT8 decoding, the raw decimated f32 stream (the
/// "audio the demod produced, untouched") is what `mfsk_core`'s FT8 decoder
/// should consume — the AGC's per-block RMS retargeting is a cosmetic
/// normalisation for human listening and adds nothing to (and slightly
/// perturbs) the SNR a digital decoder should see from the *relative*
/// amplitudes of the 8-GFSK tones.
///
/// Implement with [`super::ft8::Ft8Tap`] (wrapping a
/// `hl2::receiver::shared()` decoder): the API layer owns the shared
/// instance, the demod thread calls [`RawSampleTap::append`] inline while
/// demodulating, and the tokio decode task calls
/// `hl2::receiver::decode_closed_slot` on a 1 s wall-clock cadence.
///
/// The tap is consumed by the **engine** ([`AudioEngine`]), not by the
/// core — so a new-mode impl does not have to thread the tap through
/// `DemodCore`.
pub trait RawSampleTap: std::fmt::Debug + Send + Sync {
    /// Append `samples` to the tap. May be called concurrently from the
    /// demod thread; implementers should be `Send + Sync` and keep calls
    /// short (no allocation, no blocking).
    fn append(&self, samples: &[f32]);
}
