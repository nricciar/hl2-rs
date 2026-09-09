//! FM (wide) / NFM (narrow) demodulator — the **mode core**.
//!
//! Both amateur-radio "FM" (standard, ≈ 15 kHz channel) and "NFM" (narrow,
//! ≈ 5 kHz channel) are the **same demod DSP**; they differ only in the
//! channel-select bandwidth the anti-alias/decimate LPF is tuned to
//! (`bandwidth_hz`). Standard FM and NFM are therefore one [`FmCore`],
//! instantiated twice (with different `bandwidth_hz`) by
//! [`super::make_demod_tap`].
//!
//! ## FM demodulation
//!
//! The HL2 EP6 wire delivers **genuine complex I/Q**, so both arms carry
//! signal energy in every configuration — a Hilbert phase-shifter (as the
//! SSB core uses for sideband selection) would be pure overhead here.
//!
//! After the NCO brings the FM carrier to baseband the sample is
//! `A·e^{j·φ(t)}`, where the *phase* `φ(t)` carries the audio (the FM
//! deviation) and the amplitude `A` is (near) constant. The audio is the
//! instantaneous frequency — the time-derivative of the phase.
//!
//! ### Pipeline
//!
//! ```text
//! complex I/Q
//!   └► NCO × e^(−j·φ)                        (carrier → baseband)
//!        ├► post.re ─► polyphase LPF/decimate  (lp_i)
//!        └► post.im ─► polyphase LPF/decimate  (lp_q)
//!        └► at each decimated (i, q) pair:
//!             phase  = atan2(q, i)                       ∈ (−π, π]
//!             emit   = wrap(phase − prev_phase)          ∈ (−π, π]  (radians)
//! ```
//!
//! The polyphase LPF (a unit-gain windowed-sinc low-pass at the channel-
//! select bandwidth, split into `M = source_rate / audio_rate` polyphase
//! branches by [`PolyphaseDecimator`]) band-limits the I and Q arms
//! **before** the phase is read. That is the real work of the demod:
//! it rejects wideband adjacent-channel and out-of-band noise so the
//! `atan2` reads a clean phase. Both arms go through identical filters at
//! the same decimation factor, so they emit in lockstep and the `(i, q)`
//! pair at each group boundary is a true complex sample.
//!
//! The **phase-derivative** read — `atan2` → difference → principal-value
//! wrap — is **naturally amplitude-invariant**: `atan2` returns a phase, not
//! a magnitude, so a weak carrier and a strong one produce the same `delta`
//! for the same deviation. No `A²` scaling enters, which matters because the
//! HL2 DDC output is weak and a cross-product demod would be too low to
//! reach the AGC target.
//!
//! The deviation `[−f_dev, +f_dev]` is symmetric about DC, so the audio is
//! zero-mean for symmetric deviation and the shared AGC's DC-block has no
//! work beyond removing any small residual offset.
//!
//! The audio tail (DC-block + RMS-target AGC + pre-AGC
//! [`RawSampleTap`](super::RawSampleTap) + `i16` sink write) is shared with
//! every mode by the [`AudioEngine`](super::AudioEngine).

use core::f32::consts as f32_consts;
use core::f64::consts;
use num_complex::Complex;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

use super::core::DemodCore;
use super::dsp::{F32Fir, KAISER_BETA, Nco, PolyphaseDecimator};
use crate::receiver::AudioConfig;

/// The FM / NFM demodulator **core**.
///
/// Standard FM (≈ 15 kHz channel) and NFM (≈ 5 kHz channel) share this core;
/// they are distinguished only by the `bandwidth_hz` passed to
/// [`FmCore::new`] (the channel-select / anti-alias LPF width). See the
/// module docs for the DSP.
pub struct FmCore {
    /// NCO (bring the FM carrier to baseband). Identity for a baseband source.
    nco: Nco,
    /// Polyphase anti-alias + decimation over the in-phase arm. Unit-gain
    /// windowed-sinc LPF at the channel-select bandwidth, split into `M`
    /// branches so ≈ 1/M of the taps run per input sample.
    lp_i: PolyphaseDecimator,
    /// Same filter/decimation over the quadrature arm. Runs in lockstep with
    /// [`Self::lp_i`] (same taps, same `M`) so the `(i, q)` pair at each
    /// group boundary is a true complex sample.
    lp_q: PolyphaseDecimator,
    /// A short identifier for this instance — `"fm"` (standard) / `"nfm"`
    /// (narrow), surfaced as `kind()`.
    label: &'static str,
    /// Previous decimated arm's phase `atan2(q, i)` (radians, principal
    /// value). Updated on every emitted sample; the phase-derivative read
    /// is `phase − prev_phase` wrapped to `(−π, π]`.
    prev_phase: f32,
}

impl FmCore {
    /// Build an FM / NFM demodulator **core**.
    ///
    /// `bandwidth_hz` is the channel-select bandwidth (the LPF cutoff on each
    /// side of baseband DC); use ≈ 15 kHz for standard FM voice and ≈ 5 kHz
    /// for narrow (NFM) voice — the two "modes" differ only here.
    ///
    /// `label` is the debug kind tag (`"fm"` / `"nfm"`).
    ///
    /// `audio` is used only to (a) set the decimation factor
    /// (`m = source_rate_hz / audio.rate_hz`, with the ≥ 4 sanity bound) and
    /// (b) let [`DemodCore::demodulator`] build the audio tail at that rate.
    pub fn new(
        source_rate_hz: u32,
        source_center_hz: f64,
        bandwidth_hz: u32,
        audio: AudioConfig,
        label: &'static str,
    ) -> Self {
        let m = source_rate_hz as usize / audio.rate_hz as usize;
        // Channel-select / anti-alias LPF: pass the requested channel width.
        // The same heuristic as SSB/AM: 257 taps for the narrow (voice) bands,
        // 129 for the wider — as a unit-gain windowed-sinc low-pass split into
        // `M` polyphase branches so ≈ 1/M of the taps run per input sample.
        let bw_ratio = (bandwidth_hz as f64 / source_rate_hz as f64).clamp(1e-3, 0.4);
        let taps = if bandwidth_hz <= 4_000 { 257 } else { 129 };
        let h = F32Fir::lowpass(taps, bw_ratio, KAISER_BETA).taps().to_vec();
        // Two independent polyphase decimators, one per complex arm. Both
        // share the same tap vector and decimation factor, so they emit in
        // lockstep and the (i, q) pair at each group boundary is a true
        // complex sample.
        let lp_i = PolyphaseDecimator::new(&h, m);
        let lp_q = PolyphaseDecimator::new(&h, m);
        let nco = Nco::new(2.0 * consts::PI * source_center_hz / source_rate_hz as f64);
        Self {
            nco,
            lp_i,
            lp_q,
            label,
            prev_phase: 0.0,
        }
    }

    /// The kind tag this instance was built with (`"fm"` / `"nfm"`).
    pub fn label(&self) -> &'static str {
        self.label
    }
}

impl DemodCore for FmCore {
    fn process(&mut self, x: Complex<f32>) -> Option<f32> {
        let post = self.nco.step(x);
        // 1. Band-limit + decimate each arm independently, with the same taps
        //    and the same decimation factor, so the decimator group boundaries
        //    are aligned between the two (every `m`-th input index).
        let i_dec = self.lp_i.push(post.re);
        let q_dec = self.lp_q.push(post.im);

        // 2. FM demod only at the decimated rate: read the phase, difference
        //    it against the previous phase, and wrap to the principal value
        //    `(−π, π]`. That wrapped difference *is* the audio (radians/sample)
        //    and is naturally amplitude-invariant (see module docs).
        if let (Some(i), Some(q)) = (i_dec, q_dec) {
            let phase = q.atan2(i);
            let mut delta = phase - self.prev_phase;
            self.prev_phase = phase;
            if delta > f32_consts::PI {
                delta -= 2.0 * f32_consts::PI;
            } else if delta < -f32_consts::PI {
                delta += 2.0 * f32_consts::PI;
            }
            Some(delta)
        } else {
            None
        }
    }

    fn kind(&self) -> &'static str {
        self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receiver::Demodulator;
    use crate::receiver::sink::VecSink;
    use alloc::{boxed::Box, vec};
    use std::eprintln;

    const PI: f64 = std::f64::consts::PI;

    /// A **complex (analytic)** FM signal: `z[n] = A·e^{j·φ[n]}` with
    /// `φ[n] = 2π·f_c·n/fs + β·sin(2π·f_mod·n/fs)`,
    /// `β = dev_hz / f_mod` (deviation index). The carrier is at `carrier_hz`;
    /// the 1/f-mod sinusoid is the "voice" deviating the carrier by `dev_hz`
    /// Hz. `amp` scales the carrier amplitude (1.0 for full-scale synthesis,
    /// `1e-3` to reproduce the HL2's weak DDC output).
    fn fm_complex(
        rate_hz: u32,
        carrier_hz: f64,
        mod_hz: f64,
        dev_hz: f64,
        n_pairs: usize,
        amp: f32,
    ) -> super::super::IqBlock {
        let beta = (dev_hz / mod_hz).max(1.0);
        (0..n_pairs)
            .map(|i| {
                let n = i as f64;
                let w_c = 2.0 * PI * carrier_hz / rate_hz as f64;
                let w_m = 2.0 * PI * mod_hz / rate_hz as f64;
                let phi = w_c * n + beta * (w_m * n).sin();
                Complex::new((phi.cos() as f32) * amp, (phi.sin() as f32) * amp)
            })
            .collect()
    }

    fn build(rate: u32, center: f64, bw: u32, label: &'static str) -> Box<dyn Demodulator> {
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        FmCore::new(rate, center, bw, cfg, label).demodulator(cfg, None)
    }

    /// Goertzel: power at a single frequency (no FFT needed). Used by the
    /// tone-purity assertion below.
    fn goertzel(sig: &[i16], rate_out_hz: u32, f_hz: f64) -> f64 {
        let w = 2.0 * PI * f_hz / rate_out_hz as f64;
        let c = 2.0 * w.cos();
        let mut s0 = 0.0f64;
        let mut s1 = 0.0f64;
        let mut s2 = 0.0f64;
        for &x in sig {
            s0 = x as f64 + c * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - c * s1 * s2).max(0.0)
    }

    #[test]
    fn inband_fm_produces_audio() {
        let rate = 192_000u32;
        // Standard FM channel (15 kHz): carrier at centre, 1.5 kHz "voice"
        // deviating the carrier by 2 kHz.
        let iq = fm_complex(rate, 0.0, 1_500.0, 2_000.0, 32_768, 1.0);
        let mut demod = build(rate, 0.0, 15_000, "fm");
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("demod ok");
        assert!(n > 0, "produced {n} audio frames");
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "peak too low: {peak}");
    }

    /// The weak-carrier guarantee for FM. The phase-derivative demod is
    /// **amplitude-invariant by construction** (`atan2` returns a phase
    /// regardless of carrier magnitude) so the same weak signal that used
    /// to be too low on a cross-product / `/|z|` version is still audible
    /// here: pre-AGC `delta` is ~ O(1) for any `A > 0`.
    #[test]
    fn weak_complex_baseband_fm_produces_audio() {
        let rate = 192_000u32;
        let iq = fm_complex(rate, 0.0, 1_500.0, 2_000.0, 32_768, 1e-3);
        let mut demod = build(rate, 0.0, 15_000, "fm");
        let mut sink = VecSink::new();
        let _ = demod.demod(&iq, &mut sink).expect("demod ok");
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(
            peak > 200,
            "weak complex-baseband FM must still reach the AGC target: {peak}"
        );
    }

    /// Same weak-carrier guarantee for NFM (the mode the user reported as
    /// silent). `A = 1e-3`, the narrow 5 kHz channel, 1.5 kHz voice. The
    /// deviation matches the FM weak test (2 kHz → β ≈ 1.33) so the only
    /// difference between the two is the channel-select width.
    #[test]
    fn weak_complex_baseband_nfm_produces_audio() {
        let rate = 192_000u32;
        let iq = fm_complex(rate, 0.0, 1_500.0, 2_000.0, 32_768, 1e-3);
        let mut demod = build(rate, 0.0, 5_000, "nfm");
        let mut sink = VecSink::new();
        let _ = demod.demod(&iq, &mut sink).expect("demod ok");
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(
            peak > 50,
            "weak complex-baseband NFM must still reach the AGC target: {peak}"
        );
    }

    /// The decisive test: the demodulated audio must actually *contain the
    /// voice tone at the correct frequency*, not just a peak at any frequency.
    /// Goertzel the output at the 1.5 kHz voice frequency and at 3.0 kHz (2f);
    /// the voice bin must clearly dominate.
    #[test]
    fn weak_complex_fm_is_clean_tone() {
        let rate = 192_000u32;
        let mod_hz = 1_500.0; // the "voice"
        // Weak complex baseband — the actual HL2 condition.
        let iq = fm_complex(rate, 0.0, mod_hz, 1_500.0, 262_144, 1e-3);
        let mut demod = build(rate, 0.0, 15_000, "fm");
        let mut sink = VecSink::new();
        let _ = demod.demod(&iq, &mut sink).expect("demod ok");
        let samples = sink.samples().to_vec();
        assert!(
            samples.len() > 2_000,
            "need ≥2 kHz of audio to do a DFT, got {}",
            samples.len()
        );

        let audio_rate = 4_800u32; // matches AudioConfig in build()
        let p_voice = goertzel(&samples, audio_rate, mod_hz);
        let p_2f = goertzel(&samples, audio_rate, 2.0 * mod_hz);
        let p_static = goertzel(&samples, audio_rate, 5_000.0);
        eprintln!(
            "CLEAN-TONE DIAG  voice@1.5k={p_voice:.3e}  2f@3.0k={p_2f:.3e}  static@5k={p_static:.3e}  voice/2f={:.2}",
            p_voice / p_2f.max(1e-18)
        );
        assert!(
            p_voice > p_2f * 1.5,
            "voice tone (1.5 kHz) must clearly dominate its 2× harmonic (3.0 kHz); \
             if it doesn't, the demod is leaking image (2f ripple — the 'static' symptom). \
             voice={p_voice:.3e} 2f={p_2f:.3e}"
        );
        assert!(
            p_voice > p_static.max(1e-18) * 0.25,
            "voice tone must not be drowned by broadband (5 kHz) static; \
             voice={p_voice:.3e} 5k={p_static:.3e}"
        );
    }

    #[test]
    fn nfm_narrow_channel_passes_in_band_deviation() {
        let rate = 192_000u32;
        // NFM (5 kHz): same 1.5 kHz voice, moderate deviation.
        let iq = fm_complex(rate, 0.0, 1_500.0, 1_500.0, 32_768, 1.0);
        let mut demod = build(rate, 0.0, 5_000, "nfm");
        let mut sink = VecSink::new();
        let _ = demod.demod(&iq, &mut sink).expect("demod ok");
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 40, "NFM in-band peak too low: {peak}");
    }

    #[test]
    fn nco_moves_offband_fm_into_band() {
        let rate = 192_000u32;
        let center = 50_000.0;
        // Carrier sits at NCO + 2 kHz (in the 15 kHz standard-FM channel).
        let iq = fm_complex(rate, center + 2_000.0, 1_500.0, 2_000.0, 32_768, 1.0);
        let mut demod = build(rate, center, 15_000, "fm");
        let mut sink = VecSink::new();
        let _ = demod.demod(&iq, &mut sink).expect("ok");
        let peak = sink.samples().iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "NCO-moved FM tone should be audible: {peak}");
    }

    #[test]
    fn odd_block_accepted() {
        let rate = 192_000u32;
        let mut demod = build(rate, 0.0, 15_000, "fm");
        let mut sink = VecSink::new();
        let iq = vec![Complex::new(1.0, 0.0); 63];
        demod
            .demod(&iq, &mut sink)
            .expect("odd (63) length must succeed");
    }
}
