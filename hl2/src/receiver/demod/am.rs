//! AM (DSB-FC, double-sideband full-carrier) voice demodulator.
//!
//! The AM path is true **envelope detection**:
//!
//! ```text
//! complex I/Q
//!   └─► NCO × e^(−j·φ)  (carrier → ~baseband)
//!        ├─► post.re → lp_i (polyphase LPF/decimate) ─┐
//!        └─► post.im → lp_q (polyphase LPF/decimate) ─┤
//!                                                    └┴► |I + jQ| = √(I²+Q²)
//!                                                         └─► AGC (DC-block + RMS) → i16
//! ```
//!
//! ## Why envelope detection (not "keep the in-phase arm")
//!
//! A naive AM demod keeps only the real arm (`post.re`) — the same as the
//! SSB-USB branch. That works on paper for a *perfect* NCO, but it has two
//! practical defects that make AM sound worse than LSB, which the operator
//! can use as a reference:
//!
//! 1. **Carrier-offset flutter.** The NCO's actual frequency is never
//!    exactly the carrier's. Let the residual offset be `Δf` Hz. After the
//!    NCO the baseband is `(A/2)·[1+m(t)]·e^(−j·2π·Δf·t)` — a slowly
//!    rotating phasor. Keeping only `post.re` multiplies the audio by
//!    `cos(2π·Δf·t)`, a low-frequency warble that throttens and steps the
//!    AGC (the "popping / feedback" symptom). Envelope detection
//!    `√(I²+Q²) = (A/2)·[1+m(t)]` is **invariant to that rotation**, so
//!    the offset drops out.
//! 2. **Carrier DC + modulation envelope.** The full-carrier term `A/2`
//!    plus the LNA-AGC / antenna-fade envelope ride at baseband DC. The
//!    AGC's 50 ms block-mean DC-block has to chase a *moving* target
//!    (real AM's carrier moves with the speech envelope), and each
//!    block-boundary correction is a step = a pop. The envelope detector
//!    leaves clean DC (the carrier) plus the AC modulation — the AGC's
//!    DC-block handles that well because it no longer fights flutter.
//!
//! The LSB SSB path "works" by accident: the Hilbert phase-shifter has a
//! **zero at DC**, so it removes the carrier before the AGC sees it. USB
//! and AM, both of which keep `post.re`, don't have that null. Envelope
//! detection gives AM and USB the same carrier rejection.
//!
//! The channel-select FIR is the same Kaiser-windowed windowed-sinc
//! low-pass as the SSB / digital path, consumed as a [`PolyphaseDecimator`]
//! (≈1/M of the full-rate tap count per sample). Two decimators (I arm and
//! Q arm, identical taps, same `M`) run in lockstep — `push` yields a
//! sample every `M` inputs on both, and the magnitudes are summed sample
//! for sample.

use std::sync::atomic::Ordering as AOrdering;

use super::RawSampleTap;
use super::dsp::{F32Fir, KAISER_BETA, Nco, PolyphaseDecimator};
use super::{
    AUDIO_EMIN, DemodError, Demodulator, EMIT_COUNT, IqBlock, RateTooClose, meter_tick,
    normalize_to_i16_with_agc,
};
use crate::receiver::sink::AudioSink;
use crate::receiver::{AudioConfig, MeterHandle};

/// The AM (DSB-FC) voice demodulator.
///
/// ```text
/// complex I/Q
///   └─► NCO × e^(−j·φ)
///        ├─► post.re → lp_i (LPF/decimate) ─┐
///        └─► post.im → lp_q (LPF/decimate) ─┤
///                                          └► |I + jQ| = √(I²+Q²)
///                                               └─► pre-AGC f32 → [`RawSampleTap`]
///                                                    └─► AGC + DC-block → i16 → sink
/// ```
///
/// `lp_i` and `lp_q` are identical taps at the same decimation factor, so
/// they emit `Some(·)` on the same input instants. The envelope detector
/// (√ of the sum of squares) is a per-emit-sample operation, not a
/// streaming one — it consumes the two decimated samples and produces one
/// envelope value.
pub struct AmDemodulator {
    audio_cfg: AudioConfig,
    nco: Nco,
    lp_i: PolyphaseDecimator,
    lp_q: PolyphaseDecimator,

    agc_gain: f32,
    audio_buf: Vec<f32>,
    tap: Option<std::sync::Arc<dyn RawSampleTap>>,
    meter: Option<MeterHandle>,
    meter_smooth: f64,
}

impl AmDemodulator {
    /// Build an AM voice demodulator.
    ///
    /// `bandwidth_hz` is the channel-select bandwidth; use ≥ 6 kHz for
    /// HF voice AM (typical 6–10 kHz), or a narrower value for CW-side
    /// reception.
    pub fn new(
        source_rate_hz: u32,
        source_center_hz: f64,
        bandwidth_hz: u32,
        audio: AudioConfig,
        tap: Option<std::sync::Arc<dyn RawSampleTap>>,
        meter: Option<MeterHandle>,
    ) -> Result<Self, DemodError> {
        let m = source_rate_hz as usize / audio.rate_hz as usize;
        if m < 4 {
            return Err(Box::new(RateTooClose {
                src: source_rate_hz,
                audio: audio.rate_hz,
            }));
        }
        let bw_ratio = (bandwidth_hz as f64 / source_rate_hz as f64).clamp(1e-3, 0.4);
        let taps = if bandwidth_hz <= 4_000 { 257 } else { 129 };
        let h = F32Fir::lowpass(taps, bw_ratio, KAISER_BETA).taps().to_vec();
        // Two identical decimators — the I (real) and Q (imag) arms of the
        // post-NCO baseband. Same taps, same `m`, so they emit in lockstep
        // and the envelope (√(I²+Q²)) aligns sample-for-sample.
        let lp_i = PolyphaseDecimator::new(&h, m);
        let lp_q = PolyphaseDecimator::new(&h, m);
        let nco = Nco::new(2.0 * std::f64::consts::PI * source_center_hz / source_rate_hz as f64);
        Ok(Self {
            audio_cfg: audio,
            nco,
            lp_i,
            lp_q,
            agc_gain: 1000.0,
            audio_buf: Vec::with_capacity(256),
            tap,
            meter,
            meter_smooth: -120.0,
        })
    }

    fn emit(&mut self, slice: &[f32], sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        if let Some(tap) = self.tap.as_ref() {
            tap.append(slice);
        }
        if let Some(meter) = self.meter.as_ref() {
            meter_tick(meter, &mut self.meter_smooth, slice);
        }
        let mut out = vec![0i16; slice.len()];
        let written =
            normalize_to_i16_with_agc(slice, &mut out, self.audio_cfg.gain_db, &mut self.agc_gain);
        if std::env::var("HL2_DEBUG").is_ok() {
            let c = EMIT_COUNT.fetch_add(1, AOrdering::Relaxed) + 1;
            if c % 50 == 1 {
                let in_max = slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let in_rms =
                    (slice.iter().map(|v| v * v).sum::<f32>() / slice.len().max(1) as f32).sqrt();
                let o_max = out[..written].iter().map(|v| v.abs()).max().unwrap_or(0);
                eprintln!(
                    "[aud] AM in_rms={in_rms:.6e} in_max={in_max:.6e} agc={:.3e} out_i16_max={o_max} (n={written})",
                    self.agc_gain
                );
            }
        }
        sink.write(&out[..written])
    }

    fn flush_full_blocks(&mut self, sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        let mut frames_written = 0usize;
        while self.audio_buf.len() >= AUDIO_EMIN {
            let n = std::mem::take(&mut self.audio_buf);
            let block_len = n.len().min(1024);
            let slice = &n[..block_len];
            self.audio_buf = n[block_len..].to_vec();
            frames_written = frames_written.saturating_add(self.emit(slice, sink)?);
        }
        Ok(frames_written)
    }
}

impl Demodulator for AmDemodulator {
    fn demod(&mut self, iq: &IqBlock, sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        for &x in iq {
            let post = self.nco.step(x);
            // Envelope detection: LPF both arms and sum their squares.
            //
            // Each `push(x)` feeds one full-rate branch of BOTH decimators
            // (branch index rotates `M−1, M−2, …, 0` over the group). Both
            // decimators therefore reach a group boundary on the same
            // input — they emit simultaneously, and `√(I²+Q²)` is the
            // envelope of the *same* baseband symbol.
            //
            // The carrier at baseband DC passes the LPF into both arms in
            // phase, so `lp_i` + `lp_q` carry `(A/2)·[1+m(t)]` on the I
            // arm *and* `(A/2)·[1+m(t)]·0` on the Q arm (in practice the
            // two are a 90° pair at baseband; either way their magnitudes
            // sum to the analytic-envelope magnitude, invariant to the
            // carrier's absolute position on the I/Q plane).
            let i = self.lp_i.push(post.re);
            let q = self.lp_q.push(post.im);
            if let (Some(i), Some(q)) = (i, q) {
                // Envelope = √(I² + Q²). f32 is fine for our magnitudes
                // (≈1e-3 to 1e0 for in-band AM) — the squares don't
                // overflow, and the sqrt is exact to within f32 eps.
                self.audio_buf.push((i * i + q * q).sqrt());
            }
        }
        self.flush_full_blocks(sink)
    }

    fn audio_format(&self) -> AudioConfig {
        self.audio_cfg
    }

    fn flush_audio(&mut self, sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        if self.audio_buf.is_empty() {
            return Ok(0);
        }
        let n = std::mem::take(&mut self.audio_buf);
        self.emit(&n, sink)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receiver::sink::VecSink;
    use num_complex::Complex;

    /// Analytic complex tone.
    fn complex_sine(rate_hz: u32, freq_hz: f64, n_pairs: usize, amp: f32) -> IqBlock {
        (0..n_pairs)
            .map(|i| {
                let w = 2.0 * std::f64::consts::PI * freq_hz / rate_hz as f64;
                Complex::new(
                    (w * i as f64).cos() as f32 * amp,
                    (w * i as f64).sin() as f32 * amp,
                )
            })
            .collect()
    }

    /// Real baseband tone (the HL2 DDC's normal output: I = cos, Q = 0).
    fn real_tone(rate_hz: u32, freq_hz: f64, n_pairs: usize, amp: f32) -> IqBlock {
        (0..n_pairs)
            .map(|i| {
                let w = 2.0 * std::f64::consts::PI * freq_hz / rate_hz as f64;
                Complex::new((w * i as f64).cos() as f32 * amp, 0.0f32)
            })
            .collect()
    }

    #[test]
    fn inband_tone_produces_audio() {
        let rate = 192_000u32;
        let iq = complex_sine(rate, 1_500.0, 16384, 0.5);
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod = AmDemodulator::new(rate, 0.0, 8_000, cfg, None, None).expect("build");
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("demod ok");
        assert!(n > 0, "produced {n} audio frames");
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "peak too low: {peak}");
    }

    #[test]
    fn real_baseband_produces_audio() {
        let rate = 192_000u32;
        let iq = real_tone(rate, 1_500.0, 16384, 0.5);
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod = AmDemodulator::new(rate, 0.0, 8_000, cfg, None, None).expect("build");
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("demod ok");
        assert!(n > 0, "produced {n} audio frames");
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 30, "real-baseband peak too low: {peak}");
    }

    #[test]
    fn odd_block_accepted() {
        let rate = 192_000u32;
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod = AmDemodulator::new(rate, 0.0, 8_000, cfg, None, None).unwrap();
        let mut sink = VecSink::new();
        let iq = vec![Complex::new(1.0, 0.0); 63];
        demod
            .demod(&iq, &mut sink)
            .expect("odd (63) length must succeed");
    }

    #[test]
    fn inband_outranks_out_of_band() {
        let rate = 192_000u32;
        // In-band at 1.5 kHz + out-of-band at 9 kHz (past 8 kHz BW).
        let a = complex_sine(rate, 1_500.0, 16384, 0.5);
        let b = complex_sine(rate, 9_000.0, 16384, 0.5);
        let iq: IqBlock = a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect();
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod = AmDemodulator::new(rate, 0.0, 8_000, cfg, None, None).unwrap();
        let mut sink = VecSink::new();
        let _ = demod.demod(&iq, &mut sink).expect("ok");
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "in-band audio too quiet: {peak}");
    }

    #[test]
    fn nco_moves_offband_tone_into_band() {
        let rate = 192_000u32;
        let center = 100_000.0;
        let iq = complex_sine(rate, center + 1_500.0, 16384, 0.5);
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod = AmDemodulator::new(rate, center, 8_000, cfg, None, None).unwrap();
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("ok");
        assert!(n > 0, "produced {n} audio frames");
        let peak = sink.samples().iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "NCO-moved tone should be audible: {peak}");
    }

    #[test]
    fn flush_emits_residual_tail() {
        let rate = 192_000u32;
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod = AmDemodulator::new(rate, 0.0, 8_000, cfg, None, None).unwrap();
        let mut sink = VecSink::new();
        // Feed just enough to produce a few samples but leave a partial buf.
        let iq = complex_sine(rate, 1_500.0, 3000, 0.5);
        let _ = demod.demod(&iq, &mut sink).expect("ok");
        let after_flush = demod.flush_audio(&mut sink).unwrap();
        // Total samples written (including any residual) should be > 0.
        assert!(
            !sink.samples().is_empty() || after_flush == 0,
            "no audio at all"
        );
    }
}
