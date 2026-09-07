//! AM (DSB-FC, double-sideband full-carrier) voice demodulator — the **mode core**.
//!
//! The AM path is true **envelope detection**:
//!
//! ```text
//! complex I/Q
//!   └─► NCO × e^(−j·φ)  (carrier → ~baseband)
//!        ├─► post.re → lp_i (polyphase LPF/decimate) ─┐
//!        └─► post.im → lp_q (polyphase LPF/decimate) ─┤
//!                                                    └┴► |I + jQ| = √(I²+Q²)
//! ```
//!
//! The audio tail (AGC DC-block + RMS target, pre-AGC [`RawSampleTap`],
//! `i16` sink write) is shared with every mode via the [`AudioEngine`];
//! this module implements only the envelope-detection core.
//!
//! ## Why envelope detection (not "keep the in-phase arm")
//!
//! Keeping only the real arm (`post.re`) — as SSB-USB does — works for a
//! perfect NCO, but the residual carrier offset `Δf` rotates the baseband
//! phasor and multiplies the audio by `cos(2π·Δf·t)` — a low-frequency
//! warble that stutters the AGC ("popping"). Envelope detection
//! `√(I²+Q²) = (A/2)·[1+m(t)]` is invariant to that rotation, so the offset
//! drops out.
//!
//! The full-carrier term `A/2` plus the antenna-fade envelope ride at
//! baseband DC; the AGC's DC-block then chases a *moving* target on the
//! raw arm. The envelope detector leaves clean DC (the carrier) plus the AC
//! modulation, so the DC-block no longer fights flutter.
//!
//! `lp_i` and `lp_q` are identical taps at the same decimation factor, so
//! they emit `Some(·)` on the same input instants. The envelope
//! (√(I²+Q²)) is a per-sample operation that consumes the two decimated
//! samples and produces one envelope value.

use num_complex::Complex;

use super::core::DemodCore;
use super::dsp::{F32Fir, KAISER_BETA, Nco, PolyphaseDecimator};
use crate::receiver::AudioConfig;

/// The AM (DSB-FC) voice demodulator **core**.
///
/// ```text
/// complex I/Q
///   └─► NCO × e^(−j·φ)
///        ├─► post.re → lp_i (LPF/decimate) ─┐
///        └─► post.im → lp_q (LPF/decimate) ─┤
///                                          └► |I + jQ| = √(I²+Q²)  ──► AudioEngine
/// ```
pub struct AmCore {
    nco: Nco,
    lp_i: PolyphaseDecimator,
    lp_q: PolyphaseDecimator,
}

impl AmCore {
    /// Build an AM voice demodulator **core**.
    ///
    /// `bandwidth_hz` is the channel-select bandwidth; use ≥ 6 kHz for
    /// HF voice AM (typical 6–10 kHz), or a narrower value for CW-side
    /// reception.
    ///
    /// `audio` is used to (a) set the decimation factor
    /// (`m = source_rate_hz / audio.rate_hz`, with the ≥4 sanity bound) and
    /// (b) let [`DemodCore::demodulator`] build the audio tail at that rate.
    pub fn new(
        source_rate_hz: u32,
        source_center_hz: f64,
        bandwidth_hz: u32,
        audio: AudioConfig,
    ) -> Self {
        let m = source_rate_hz as usize / audio.rate_hz as usize;
        let bw_ratio = (bandwidth_hz as f64 / source_rate_hz as f64).clamp(1e-3, 0.4);
        let taps = if bandwidth_hz <= 4_000 { 257 } else { 129 };
        let h = F32Fir::lowpass(taps, bw_ratio, KAISER_BETA).taps().to_vec();
        // Two identical decimators — the I (real) and Q (imag) arms of the
        // post-NCO baseband. Same taps, same `m`, so they emit in lockstep
        // and the envelope (√(I²+Q²)) aligns sample-for-sample.
        let lp_i = PolyphaseDecimator::new(&h, m);
        let lp_q = PolyphaseDecimator::new(&h, m);
        let nco = Nco::new(2.0 * std::f64::consts::PI * source_center_hz / source_rate_hz as f64);
        Self { nco, lp_i, lp_q }
    }
}

impl DemodCore for AmCore {
    fn process(&mut self, x: Complex<f32>) -> Option<f32> {
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
            Some((i * i + q * q).sqrt())
        } else {
            None
        }
    }

    fn kind(&self) -> &'static str {
        "am"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receiver::Demodulator;
    use crate::receiver::sink::VecSink;

    /// Analytic complex tone.
    fn complex_sine(rate_hz: u32, freq_hz: f64, n_pairs: usize, amp: f32) -> super::super::IqBlock {
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
    fn real_tone(rate_hz: u32, freq_hz: f64, n_pairs: usize, amp: f32) -> super::super::IqBlock {
        (0..n_pairs)
            .map(|i| {
                let w = 2.0 * std::f64::consts::PI * freq_hz / rate_hz as f64;
                Complex::new((w * i as f64).cos() as f32 * amp, 0.0f32)
            })
            .collect()
    }

    fn build(rate: u32, center: f64, bw: u32) -> Box<dyn Demodulator> {
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        AmCore::new(rate, center, bw, cfg).demodulator(cfg, None)
    }

    #[test]
    fn inband_tone_produces_audio() {
        let rate = 192_000u32;
        let iq = complex_sine(rate, 1_500.0, 16384, 0.5);
        let mut demod = build(rate, 0.0, 8_000);
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
        let mut demod = build(rate, 0.0, 8_000);
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
        let mut demod = build(rate, 0.0, 8_000);
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
        let iq: super::super::IqBlock = a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect();
        let mut demod = build(rate, 0.0, 8_000);
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
        let mut demod = build(rate, center, 8_000);
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("ok");
        assert!(n > 0, "produced {n} audio frames");
        let peak = sink.samples().iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "NCO-moved tone should be audible: {peak}");
    }

    #[test]
    fn flush_emits_residual_tail() {
        let rate = 192_000u32;
        let iq = complex_sine(rate, 1_500.0, 3000, 0.5);
        let mut demod = build(rate, 0.0, 8_000);
        let mut sink = VecSink::new();
        let _ = demod.demod(&iq, &mut sink).expect("ok");
        let after_flush = demod.flush_audio(&mut sink).unwrap();
        assert!(
            !sink.samples().is_empty() || after_flush == 0,
            "no audio at all"
        );
    }
}
