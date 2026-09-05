//! Demodulators for the virtual audio receiver.
//!
//! A [`Demodulator`] converts a block of complex I/Q baseband samples into
//! `i16` mono audio and writes it to an [`AudioSink`](super::sink::AudioSink).
//! Each radio mode (SSB, AM, FM, FT8, CW…) is a different `Demodulator`;
//! [`make_demod`] / [`make_demod_tap`] is the single dispatch point for a new
//! mode (PROTOCOL.md §16).
//!
//! This module is split by concern:
//!
//! * [`dsp`] — the shared hand-rolled DSP primitives (the phase-recurrence
//!   [`Nco`], the Kaiser-windowed windowed-sinc low-pass / Hilbert FIR
//!   ([`F32Fir`] / [`F32FirState`]), and the [`PolyphaseDecimator`] anti-alias
//!   / decimation stage). Used by every mode.
//! * [`ssb`] — the SSB (USB/LSB) **voice** demodulator ([`SsbDemodulator`]):
//!   NCO → product-discriminate (USB in-phase / LSB Hilbert) → polyphase
//!   anti-alias + decimate → AGC → i16, at the voice-rate window.
//! * [`am`] — the AM (DSB-FC, full-carrier) **voice** demodulator
//!   ([`AmDemodulator`]): NCO → in-phase arm → polyphase anti-alias +
//!   decimate → AGC → i16. No Hilbert / sideband selection — both sidebands
//!   pass through the LPF and the carrier DC is removed by the AGC
//!   DC-block.
//! * [`digital`] — the FT8/JS8/FT4 **digital** demodulator
//!   ([`DigitalDemodulator`]): USB NCO → polyphase decimation straight to the
//!   12 kHz window rate (no quadrature synthesis), with an optional
//!   pre-AGC [`RawSampleTap`] seam.
//!
//! DSP is hand-rolled (windowed-sinc FIR, Hilbert, polyphase decimator); no
//! external DSP crate. The trait is deliberately minimal — `demod(iq, sink)`
//! and `audio_format()` — so AM/FM/CW slot in without touching the receiver
//! loop or the sink API.

pub mod am;
pub mod digital;
pub mod dsp;
pub mod ssb;

use std::fmt;
use std::sync::atomic::AtomicUsize;

use num_complex::Complex;

use super::sink::AudioSink;
use super::{AudioConfig, MeterHandle, Mode, Sideband, meter_write};

// Re-export the mode demodulators + shared DSP so the module-root paths
// (`hl2::receiver::demod::{SsbDemodulator, F32Fir, Nco, …}` and
// `super::demod::RawSampleTap` from ft8/ft4/js8) keep resolving.
pub use am::AmDemodulator;
pub use digital::DigitalDemodulator;
pub use dsp::{F32Fir, F32FirState, KAISER_BETA, Nco, PolyphaseDecimator};
pub use ssb::SsbDemodulator;

/// Gated (HL2_DEBUG) instrument counter so we don't flood on every emit.
static EMIT_COUNT: AtomicUsize = AtomicUsize::new(0);

/// An SSB demodulator received a block whose length was not a multiple of two
/// (an odd number of complex pairs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidBlockLength(pub usize);
impl fmt::Display for InvalidBlockLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "demod: block length {} must be even (complex pairs)",
            self.0
        )
    }
}
impl std::error::Error for InvalidBlockLength {}

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

/// A block of complex I/Q baseband samples: `Complex<f32>` pairs.
pub type IqBlock = Vec<Complex<f32>>;

/// A demodulator: complex I/Q in, `i16` mono audio out, to a given sink.
///
/// `Send` so a [`crate::receiver::VirtualReceiver`] (which owns a
/// `Box<dyn Demodulator>`) can be handed to a dedicated demod thread — the
/// pump and the demod are decoupled via the
/// [`crate::receiver::BasebandRing`].
pub trait Demodulator: Send {
    /// Demodulate one complex I/Q block into `sink`; returns audio frames written.
    fn demod(&mut self, iq: &IqBlock, sink: &mut dyn AudioSink) -> Result<usize, DemodError>;

    /// The audio format produced.
    fn audio_format(&self) -> AudioConfig;

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

/// Minimum decimated samples to accumulate before a block is normalised and
/// emitted. A single wire chunk (63 complex samples at a 192 kHz → 4.8 kHz
/// decimation) yields only ≈ 2 samples, far too few for a meaningful
/// DC-block / AGC target (a 1-sample mean subtracts `x` from itself → the
/// output is always 0, which is what made SSB "silence"). ~50 ms of 4.8 kHz
/// audio (240 samples) is the smallest block that gives stable statistics
/// without noticeable latency.
const AUDIO_EMIN: usize = 240;

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
pub trait RawSampleTap: std::fmt::Debug + Send + Sync {
    /// Append `samples` to the tap. May be called concurrently from the
    /// demod thread; implementers should be `Send + Sync` and keep calls
    /// short (no allocation, no blocking).
    fn append(&self, samples: &[f32]);
}

/// Advance the one-pole smoothed S-meter: compute the in-band RMS of the
/// pre-AGC decimated `f32` stream, express it in dB FS (0 dBFS = full-scale
/// i16 — the natural "1.0" amplitude maps to −∞ dB, the AGC target
/// of 0.1 × 32767 maps to ≈ −20 dBFS), apply fast-attack / slow-release
/// one-pole smoothing, and store the result into the meter atomic.
///
/// The smoothing constants are shared between the SSB and digital paths
/// (one `smooth` accumulator per demod, held in the demod struct so the
/// `&mut self` borrow inside `emit` suffices — no mutex needed).
///
/// `now_db` is floored at −140 dBFS: below that the receiver is "silent"
/// (only the ADI chain's inherent quantisation noise) and it's kinder to
/// the gauge to pin the needle at its bottom instead of dropping into the
/// f64 noise floor.
pub fn meter_tick(meter: &MeterHandle, smooth: &mut f64, slice: &[f32]) {
    if slice.is_empty() {
        return;
    }
    let mut ss = 0.0f64;
    for v in slice {
        let x = *v as f64;
        ss += x * x;
    }
    let rms = (ss / slice.len() as f64).sqrt();
    let now_db = (20.0 * (rms.max(1e-7).log10())).max(-140.0).min(0.0);
    let alpha = if now_db > *smooth { 0.5 } else { 0.05 };
    *smooth += alpha * (now_db - *smooth);
    meter_write(meter, *smooth);
}

/// Normalise an `f32` audio block to `i16`. The audio comes in "natural"
/// units (the channel-filter output — typically `[-1, +1]` for a unit
/// amplitude input). This function applies:
///
///   1. **DC blocking** — subtract the block mean so the ADI chain's constant
///      I/Q offset (often ~0.3 on the in-phase arm) doesn't dominate the
///      output level for USB (which passes the in-phase arm).
///   2. **Slow RMS-targeted AGC** (replaces the old per-block peak normaliser).
///      The AGC gain is a **natural→i16 scale factor**: it maps the block's
///      RMS to the target output RMS. The target is
///      `TARGET_RMS_I16 = 0.1 × 32767` (≈ −20 dBFS). The AGC gain is
///      smoothed with a one-pole filter: fast attack (alpha=0.4) when the
///      gain needs to increase, slow release (alpha=0.08) when it decreases
///      — classic SSB AGC asymmetry that minimises "pumping" on voice gaps.
///      The gain is clamped to `[1.0, 1e7]` so a silent input doesn't
///      drive the gain to 1e9 and a saturated one doesn't crush to 0.
///   3. the user's `gain_db` (applied on top of the AGC) — a knob to add
///      headroom or trim overall level.
///
/// `agc_gain` is the running AGC scale factor, updated in-place.
fn normalize_to_i16_with_agc(
    audio: &[f32],
    out: &mut [i16],
    gain_db: f32,
    agc_gain: &mut f32,
) -> usize {
    let n = audio.len().min(out.len());
    if n == 0 {
        return 0;
    }
    // 1. DC block + compute RMS.
    let mut sum = 0.0f32;
    for v in &audio[..n] {
        sum += v;
    }
    let mean = sum / n as f32;
    let mut rms_sq = 0.0f32;
    for v in &audio[..n] {
        let x = v - mean;
        rms_sq += x * x;
    }
    let rms = (rms_sq / n as f32).sqrt();
    // 2. AGC. Target output RMS in i16 units.
    //
    // The HL2 EP6 baseband is a full-Nyquist complex stream in which SSB voice
    // occupies only ~3 kHz of 96 kHz — so the in-band voice RMS relative to
    // full-scale is typically ~1e-5–1e-4 (≈ −100…−80 dBFS). Reaching the
    // −20 dBFS target therefore requires ~1e6–1e8× of gain, well beyond a
    // "reasonable" ceiling. The reference gets this gain from the
    // *hardware* RXA AGC before digitisation (~80 dB); we do it
    // digitally, so the ceiling must cover it. 1e7 (140 dB) lands the
    // measured in-band signal right at target.
    const TARGET_RMS_I16: f32 = 0.1 * 32_767.0;
    let target_scale = (TARGET_RMS_I16 / rms.max(1e-6)).clamp(1.0, 10_000_000.0);
    let alpha = if target_scale > *agc_gain { 0.4 } else { 0.08 };
    *agc_gain += alpha * (target_scale - *agc_gain);
    // 3. Emit.
    let g = 10f32.powf(gain_db / 20.0) * *agc_gain;
    for (o, v) in out[..n].iter_mut().zip(audio.iter().take(n)) {
        let s = (v - mean) * g;
        *o = s.clamp(-32_768.0, 32_767.0) as i16;
    }
    n
}

/// Build a `Box<dyn Demodulator>` for `mode`.
///
/// * `source_center_hz`: the signal's carrier offset from the source centre
///   (`Hz`); `0.0` for a baseband source.
/// * `bandwidth_hz`: the channel-select bandwidth (`Hz`); must be less than
///   the source rate.
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
        None,
    )
}

/// `make_demod` plus an optional [`RawSampleTap`]: the tap is fed the raw,
/// pre-AGC decimated `f32` audio of every block, in arrival order, inline
/// from the demod thread. This is the seam the FT8 slot decoder hangs off:
/// the SSB/USB pipeline still emits AGC'd `i16` to the `CH_AUDIO` sink (the
/// operator hears the same audio being decoded), while the tap copies the
/// untouched `f32` stream to [`super::ft8::Ft8Decoder`].
///
/// `Mode::Ft8` / `Mode::Js8` / `Mode::Ft4` build a [`DigitalDemodulator`]
/// (see [`digital`]); `Mode::Ssb(·)` / `Mode::SsbWide` build an
/// [`SsbDemodulator`] (USB or LSB, voice-rate polyphase). `Mode::SsbWide` is
/// passed through the USB SSB pipeline today.
pub fn make_demod_tap(
    mode: Mode,
    source_rate_hz: u32,
    source_center_hz: f64,
    bandwidth_hz: u32,
    audio: AudioConfig,
    tap: Option<std::sync::Arc<dyn RawSampleTap>>,
    meter: Option<MeterHandle>,
) -> Result<Box<dyn Demodulator>, DemodError> {
    if matches!(mode, Mode::Ft8 | Mode::Js8 | Mode::Ft4) {
        let label = match mode {
            Mode::Ft8 => "ft8",
            Mode::Js8 => "js8",
            Mode::Ft4 => "ft4",
            _ => unreachable!(),
        };
        return Ok(Box::new(digital::DigitalDemodulator::new(
            source_rate_hz,
            source_center_hz,
            audio,
            tap,
            label,
            meter,
        )?));
    }
    if matches!(mode, Mode::Am) {
        return Ok(Box::new(am::AmDemodulator::new(
            source_rate_hz,
            source_center_hz,
            bandwidth_hz,
            audio,
            tap,
            meter,
        )?));
    }
    let sideband = match mode {
        Mode::Ssb(s) => s,
        Mode::SsbWide => Sideband::Usb,
        Mode::Ft8 | Mode::Js8 | Mode::Ft4 | Mode::Am => unreachable!(),
    };
    Ok(Box::new(ssb::SsbDemodulator::new(
        sideband,
        source_rate_hz,
        source_center_hz,
        bandwidth_hz,
        audio,
        tap,
        meter,
    )?))
}
