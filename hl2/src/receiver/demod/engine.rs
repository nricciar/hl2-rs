//! The shared **audio tail**: everything a demodulated sample runs through
//! after the mode-specific DSP — the pre-AGC [`RawSampleTap`] seam, the
//! DC-block + RMS-targeted AGC normalisation, and the `i16` sink write.
//!
//! A [`DemodCore`](super::core::DemodCore) decides *what* the decimated
//! sample is (USB arm / LSB Hilbert / envelope / phase rate / …); this
//! [`AudioEngine`] decides how it is normalised and delivered.
//! [`StandardDemod`](super::core::StandardDemod) composes the two into a
//! full [`Demodulator`](super::Demodulator).
//!
//! The normalisation itself — DC block, slow RMS-targeted AGC, gain — is in
//! [`normalize_to_i16_with_agc`].

use alloc::sync::Arc;
use alloc::vec::Vec;
#[cfg(feature = "std")]
use core::sync::atomic::AtomicUsize;
#[cfg(feature = "std")]
use core::sync::atomic::Ordering;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

use super::RawSampleTap;
use crate::receiver::AudioConfig;
use crate::receiver::sink::AudioSink;

/// Minimum decimated samples to accumulate before a block is normalised and
/// emitted. A single wire chunk (63 complex samples at 192 kHz → 4.8 kHz
/// decimation) yields only ≈ 2 samples, far too few for a meaningful
/// DC-block / AGC target (a 1-sample mean cancels itself). ~50 ms of 4.8 kHz
/// audio (240 samples) gives stable statistics without noticeable latency.
const AUDIO_EMIN: usize = 240;

/// Gated (HL2_DEBUG) instrument counter so we don't flood on every emit.
#[cfg(feature = "std")]
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
    /// Persistent `i16` output scratch (see [`emit_block`](Self::emit_block)).
    /// Sized once in [`new`](Self::new) and reused on every emit, so steady
    /// state never hits the heap — safe under a `no_std` bump allocator with no
    /// free. `1024` is the max `sink.write` block, the cap used by both `push`
    /// paths, so `resize` below only sets the length, it never reallocates.
    out_buf: Vec<i16>,
    /// Optional pre-AGC raw-sample tap (e.g. FT8 decode). `Arc` so the API
    /// layer keeps one handle for the demod and hands a clone to the decode
    /// task.
    tap: Option<Arc<dyn RawSampleTap>>,
    /// A short label for the HL2_DEBUG gated emit trace (`"ssb"` / `"am"` / …).
    #[cfg(feature = "std")]
    mode_label: &'static str,
}

impl AudioEngine {
    /// Build an empty tail at `rate_hz` / `gain_db`, AGC gain seeded at
    /// 1000.0 (updated on the first emit).
    pub fn new(rate_hz: u32, gain_db: f32, mode_label: &'static str) -> Self {
        #[cfg(not(feature = "std"))]
        let _ = mode_label;
        Self {
            audio_cfg: AudioConfig { rate_hz, gain_db },
            agc_gain: 1000.0,
            audio_buf: Vec::with_capacity(256),
            out_buf: Vec::with_capacity(1024),
            tap: None,
            #[cfg(feature = "std")]
            mode_label,
        }
    }

    /// Attach a pre-AGC [`RawSampleTap`] (builder; cheap — just a pointer).
    /// Pass `None` to keep no tap (the default).
    pub fn with_tap(mut self, tap: Option<Arc<dyn RawSampleTap>>) -> Self {
        if let Some(t) = tap {
            self.tap = Some(t);
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
    ///
    /// Allocation-free in steady state: each block is a *window* into
    /// [`audio_buf`](Self::audio_buf) (no copy), and the normalisation writes
    /// into the reused [`out_buf`](Self::out_buf). The block is drained with a
    /// single `drain` (one `memmove`, no new allocation) once it is emitted.
    pub fn emit_full_blocks(
        &mut self,
        sink: &mut dyn AudioSink,
    ) -> Result<usize, super::DemodError> {
        let mut frames_written = 0usize;
        while self.audio_buf.len() >= AUDIO_EMIN {
            let block_len = self.audio_buf.len().min(1024);
            // Split-borrow the disjoint fields (a view of `audio_buf`, the
            // output scratch, and the AGC state) so the window stays live
            // while `audio_buf` is read without a copy.
            let (input, out_slot, tap_ref) =
                (&self.audio_buf[..block_len], &mut self.out_buf, &self.tap);
            let audio_cfg = self.audio_cfg;
            let agc = &mut self.agc_gain;
            #[cfg(feature = "std")]
            let label = self.mode_label;
            // Pre-AGC raw-sample tap (FT8/JS8/FT4 decode): the untouched
            // decimated `f32` stream, in arrival order.
            if let Some(t) = tap_ref.as_deref() {
                t.append(input);
            }
            out_slot.resize_with(input.len(), i16::default);
            let out = &mut out_slot[..input.len()];
            let written = normalize_to_i16_with_agc(input, out, audio_cfg.gain_db, agc);
            #[cfg(feature = "std")]
            if std::env::var("HL2_DEBUG").is_ok() {
                let c = EMIT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                if c % 50 == 1 {
                    let in_max = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let in_rms =
                        (input.iter().map(|v| v * v).sum::<f32>() / input.len().max(1) as f32).sqrt();
                    let o_max = out[..written].iter().map(|v| v.abs()).max().unwrap_or(0);
                    eprintln!(
                        "[aud] {label} in_rms={in_rms:.6e} in_max={in_max:.6e} agc={agc:.3e} out_i16_max={o_max} (n={written})",
                        agc = *agc,
                    );
                }
            }
            sink.write(&out[..written])?;
            self.audio_buf.drain(..block_len);
            frames_written = frames_written.saturating_add(written);
        }
        Ok(frames_written)
    }

    /// Emit the residual partial block (sub-`AUDIO_EMIN` tail) so the final
    /// ~50 ms isn't silently dropped at shutdown. Reads the buffer as a
    /// window and clears it in place — no allocation, even on `no_std`.
    pub fn flush_residue(&mut self, sink: &mut dyn AudioSink) -> Result<usize, super::DemodError> {
        if self.audio_buf.is_empty() {
            return Ok(0);
        }
        let (input, out_slot, tap_ref) = (&self.audio_buf, &mut self.out_buf, &self.tap);
        let audio_cfg = self.audio_cfg;
        let agc = &mut self.agc_gain;
        #[cfg(feature = "std")]
        let label = self.mode_label;
        if let Some(t) = tap_ref.as_deref() {
            t.append(input);
        }
        out_slot.resize_with(input.len(), i16::default);
        let out = &mut out_slot[..input.len()];
        let written = normalize_to_i16_with_agc(input, out, audio_cfg.gain_db, agc);
        #[cfg(feature = "std")]
        if std::env::var("HL2_DEBUG").is_ok() {
            let c = EMIT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
            if c % 50 == 1 {
                let in_max = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let in_rms =
                    (input.iter().map(|v| v * v).sum::<f32>() / input.len().max(1) as f32).sqrt();
                let o_max = out[..written].iter().map(|v| v.abs()).max().unwrap_or(0);
                eprintln!(
                    "[aud] {label} in_rms={in_rms:.6e} in_max={in_max:.6e} agc={agc:.3e} out_i16_max={o_max} (n={written})",
                    agc = *agc,
                );
            }
        }
        sink.write(&out[..written])?;
        self.audio_buf.clear();
        Ok(written)
    }
}

/// Normalise an `f32` audio block to `i16`. The audio comes in "natural"
/// units (the channel-filter output — typically `[-1, +1]` for a unit
/// amplitude input). This function applies:
///
///   1. **DC blocking** — subtract the block mean so a constant I/Q offset
///      doesn't dominate the output level.
///   2. **Slow RMS-targeted AGC**. The AGC gain is a **natural→i16 scale
///      factor** mapping the block's RMS to the target
///      `TARGET_RMS_I16 = 0.1 × 32767` (≈ −20 dBFS). It is smoothed with a
///      one-pole filter — fast attack (alpha=0.4), slow release (alpha=0.08) —
///      to minimise "pumping" on voice gaps, and clamped to `[1.0, 1e7]`.
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
    // The HL2 EP6 baseband is full-Nyquist, with in-band voice RMS typically
    // ~1e-5–1e-4 (≈ −100…−80 dBFS). Reaching the −20 dBFS target needs
    // ~1e6–1e8× of gain, so the clamp ceiling (1e7 ≈ 140 dB) must cover
    // that.
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
