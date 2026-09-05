//! The shared **audio tail**: everything that a demodulated sample runs through
//! after the mode-specific DSP — the pre-AGC [`RawSampleTap`] seam, the S-meter,
//! the AC/DC-block + RMS-targeted AGC normalisation, and the `i16` sink write.
//!
//! Before this module this logic was copy-pasted into `ssb.rs`, `am.rs` and
//! `digital.rs` (identical `emit` / `flush_full_blocks` / `flush_audio` bodies,
//! identical `agc_gain` / `audio_buf` / `tap` / `meter` / `meter_smooth`
//! fields). It is identical for every mode — the only per-mode difference in the
//! whole pipeline is the single sample the core produces per complex input, and
//! that now flows in through [`AudioEngine::push`].
//!
//! A [`DemodCore`](super::core::DemodCore) decides *what* the decimated real
//! sample is (USB arm / LSB Hilbert / envelope / phase rate / …); the
//! [`AudioEngine`] decides how it is normalised and delivered. [`StandardDemod`]
//! ([`super::standard`]) composes the two into a full [`Demodulator`].
//!
//! The normalisation itself — DC block, slow RMS-targeted AGC, gain — is in
//! [`normalize_to_i16_with_agc`]; the S-meter smoothing is [`meter_tick`]. Both
//! are unchanged from the old per-mode copies (see PROTOCOL.md §16.3).

use std::sync::atomic::AtomicUsize;

use super::RawSampleTap;
use crate::receiver::sink::AudioSink;
use crate::receiver::{AudioConfig, MeterHandle, meter_write};

/// Minimum decimated samples to accumulate before a block is normalised and
/// emitted. A single wire chunk (63 complex samples at a 192 kHz → 4.8 kHz
/// decimation) yields only ≈ 2 samples, far too few for a meaningful
/// DC-block / AGC target (a 1-sample mean subtracts `x` from itself → the
/// output is always 0, which is what made SSB "silence"). ~50 ms of 4.8 kHz
/// audio (240 samples) is the smallest block that gives stable statistics
/// without noticeable latency.
const AUDIO_EMIN: usize = 240;

/// Gated (HL2_DEBUG) instrument counter so we don't flood on every emit.
static EMIT_COUNT: AtomicUsize = AtomicUsize::new(0);

/// The mode-agnostic audio tail (see module docs). Not `Clone` (it owns an
/// AGC state + accumulator + shared `Arc`s); built once per receiver.
pub struct AudioEngine {
    /// The audio format (rate + gain) this tail emits at.
    audio_cfg: AudioConfig,
    /// Slow RMS-targeted AGC gain (linear). See `normalize_to_i16_with_agc`.
    agc_gain: f32,
    /// Accumulated decimated `f32` audio, ready to be normalised + emitted once
    /// it reaches `AUDIO_EMIN`.
    audio_buf: Vec<f32>,
    /// Optional pre-AGC raw-sample tap (e.g. FT8 decode). `Arc` so the API
    /// layer can keep one handle for the demod and hand a clone to the decode
    /// task.
    tap: Option<std::sync::Arc<dyn RawSampleTap>>,
    /// Optional signal-level meter: a one-pole smoothed pre-AGC in-band RMS in
    /// dB FS, written to this atomic on every emit.
    meter: Option<MeterHandle>,
    /// Running smoothed level (dB FS), for the one-pole filter. Initialised to
    /// a deep floor so the first emit moves the gauge quickly.
    meter_smooth: f64,
    /// A short label for the HL2_DEBUG gated emit trace (`"ssb"` / `"am"` / …).
    mode_label: &'static str,
}

impl AudioEngine {
    /// Build an empty tail at `rate_hz` / `gain_db`. The default AGC seed
    /// (1000.0) reproduces the old per-mode initial gain so the first few emits
    /// settle identically to before.
    pub fn new(rate_hz: u32, gain_db: f32, mode_label: &'static str) -> Self {
        Self {
            audio_cfg: AudioConfig { rate_hz, gain_db },
            agc_gain: 1000.0,
            audio_buf: Vec::with_capacity(256),
            tap: None,
            meter: None,
            meter_smooth: -120.0,
            mode_label,
        }
    }

    /// Attach a pre-AGC [`RawSampleTap`] (builder; cheap — just a pointer).
    /// Pass `None` to keep no tap (the default).
    pub fn with_tap(mut self, tap: Option<std::sync::Arc<dyn RawSampleTap>>) -> Self {
        if let Some(t) = tap {
            self.tap = Some(t);
        }
        self
    }

    /// Attach a signal-level meter (builder; a cheap `Arc` clone). Pass
    /// `None` to keep no meter (the default).
    pub fn with_meter(mut self, meter: Option<MeterHandle>) -> Self {
        if let Some(m) = meter {
            self.meter = Some(m);
        }
        self
    }

    /// The audio format this tail produces.
    pub fn audio_format(&self) -> AudioConfig {
        self.audio_cfg
    }

    /// Accumulate one decimated `f32` sample. Emits nothing here on its own —
    /// see [`emit_full_blocks`](Self::emit_full_blocks). A core calls this once
    /// per decimator output it produces, then calls
    /// [`emit_full_blocks`](Self::emit_full_blocks) per input block.
    #[inline]
    pub fn push(&mut self, sample: f32) {
        self.audio_buf.push(sample);
    }

    /// Normalise + emit every `AUDIO_EMIN`-complete block currently buffered,
    /// up to 1024 samples per `sink.write` (matching the previous
    /// `flush_full_blocks`). Returns the total `i16` frames written.
    pub fn emit_full_blocks(
        &mut self,
        sink: &mut dyn AudioSink,
    ) -> Result<usize, super::DemodError> {
        let mut frames_written = 0usize;
        while self.audio_buf.len() >= AUDIO_EMIN {
            let block_len = self.audio_buf.len().min(1024);
            let slice: Vec<f32> = self.audio_buf.drain(..block_len).collect();
            frames_written = frames_written.saturating_add(self.emit(&slice, sink)?);
        }
        Ok(frames_written)
    }

    /// Emit the residual partial block (sub-`AUDIO_EMIN` tail) so the final
    /// ~50 ms isn't silently dropped at shutdown.
    pub fn flush_residue(&mut self, sink: &mut dyn AudioSink) -> Result<usize, super::DemodError> {
        if self.audio_buf.is_empty() {
            return Ok(0);
        }
        let n = std::mem::take(&mut self.audio_buf);
        self.emit(&n, sink)
    }

    /// Normalise + write one slice to `sink`, advancing the AGC. Returns the
    /// number of `i16` frames written.
    fn emit(
        &mut self,
        slice: &[f32],
        sink: &mut dyn AudioSink,
    ) -> Result<usize, super::DemodError> {
        // Pre-AGC raw-sample tap (FT8/JS8/FT4 decode): the untouched decimated
        // `f32` stream, in arrival order.
        if let Some(tap) = self.tap.as_ref() {
            tap.append(slice);
        }
        // S-meter: advance the one-pole smoothed level from the pre-AGC
        // decimated stream (before the AGC rescales it).
        if let Some(meter) = self.meter.as_ref() {
            meter_tick(meter, &mut self.meter_smooth, slice);
        }
        let mut out = vec![0i16; slice.len()];
        let written =
            normalize_to_i16_with_agc(slice, &mut out, self.audio_cfg.gain_db, &mut self.agc_gain);
        if std::env::var("HL2_DEBUG").is_ok() {
            let c = EMIT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if c % 50 == 1 {
                let in_max = slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let in_rms =
                    (slice.iter().map(|v| v * v).sum::<f32>() / slice.len().max(1) as f32).sqrt();
                let o_max = out[..written].iter().map(|v| v.abs()).max().unwrap_or(0);
                eprintln!(
                    "[aud] {mode_label} in_rms={in_rms:.6e} in_max={in_max:.6e} agc={agc:.3e} out_i16_max={o_max} (n={written})",
                    mode_label = self.mode_label,
                    agc = self.agc_gain,
                );
            }
        }
        sink.write(&out[..written])
    }
}

/// Advance the one-pole smoothed S-meter: compute the in-band RMS of the
/// pre-AGC decimated `f32` stream, express it in dB FS (0 dBFS = full-scale
/// i16 — the natural "1.0" amplitude maps to −∞ dB, the AGC target
/// of 0.1 × 32767 maps to ≈ −20 dBFS), apply fast-attack / slow-release
/// one-pole smoothing, and store the result into the meter atomic.
///
/// `now_db` is floored at −140 dBFS: below that the receiver is "silent"
/// (only the ADI chain's inherent quantisation noise) and it's kinder to
/// the gauge to pin the needle at its bottom instead of dropping into the
/// f64 noise floor.
fn meter_tick(meter: &MeterHandle, smooth: &mut f64, slice: &[f32]) {
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
