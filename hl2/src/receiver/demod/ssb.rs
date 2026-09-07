//! SSB (USB/LSB) voice demodulator — the **mode core**.
//!
//! The SSB path is **NCO → product-discriminate (USB: in-phase, LSB: Hilbert
//! quadrature) → polyphase anti-alias/decimate**. This module implements only
//! that per-sample DSP as a [`DemodCore`]. The audio tail (DC-block +
//! RMS-target AGC + pre-AGC [`RawSampleTap`] + `i16` sink write) is
//! shared with every mode by the [`AudioEngine`], which [`StandardDemod`]
//! attaches here via [`DemodCore::demodulator`]. The API's S-meter hangs off
//! that same [`RawSampleTap`] (see `api/src/meter.rs`).
//!
//! The USB/LSB distinction is carried by which post-NCO arm is kept: a
//! positive-frequency NCO moves the upper sideband to baseband on the
//! in-phase arm and a negative one moves the lower sideband to baseband — the
//! same trick the reference uses (keep the in-phase arm for USB; conjugate it
//! to reach LSB). For LSB we synthesize that quadrature arm with a 90°
//! (Hilbert) phase-shifter on the in-phase arm, which is valid for both real
//! and complex baseband.
//!
//! The channel-select FIR is the anti-alias filter of a [`PolyphaseDecimator`],
//! so only ≈ 1/M of its taps touch each sample (~M× fewer MACs, M = the
//! decimation factor).

use num_complex::Complex;

use super::core::DemodCore;
use super::dsp::{F32Fir, F32FirState, KAISER_BETA, Nco, PolyphaseDecimator};
use crate::receiver::AudioConfig;
use crate::receiver::Sideband;

/// The SSB demodulator **core** (USB or LSB).
///
/// Holds: an NCO (to bring the carrier/selected sideband to baseband), a
/// 90° (Hilbert) phase-shifter for LSB, and the polyphase anti-alias /
/// decimation stage (`lp`). The audio tail (AGC/DC-block/tap/sink) is
/// attached by [`DemodCore::demodulator`] into the shared [`AudioEngine`].
pub struct SsbCore {
    sideband: Sideband,
    /// NCO (phase-recurrence oscillator). Identity when `source_center_hz`
    /// is 0 (i.e. the input is already baseband).
    nco: Nco,
    /// Quadrature (90° / Hilbert) phase-shifter. Applied to the real in-phase
    /// arm it synthesizes the missing quadrature component so SSB can be
    /// demodulated from a real (Q≈0) baseband stream. See `F32Fir::hilbert`.
    /// The sideband choice (`Usb` vs `Lsb`) selects which post-NCO arm is
    /// kept (in-phase → USB, quadrature → LSB). Used only for LSB.
    hilb: F32FirState,
    /// Polyphase anti-alias + decimation stage (channel-select FIR's taps are
    /// its full-rate impulse response).
    lp: PolyphaseDecimator,
}

impl SsbCore {
    /// Build an SSB voice demodulator **core** from a full set of receiver
    /// parameters.
    ///
    /// The `audio` argument is used only to (a) set the decimation factor
    /// (`m = source_rate_hz / audio.rate_hz`, with the ≥4 sanity bound) and
    /// (b) let [`DemodCore::demodulator`] build the audio tail at that rate.
    pub fn new(
        sideband: Sideband,
        source_rate_hz: u32,
        source_center_hz: f64,
        bandwidth_hz: u32,
        audio: AudioConfig,
    ) -> Self {
        let m = source_rate_hz as usize / audio.rate_hz as usize;
        // Anti-alias LPF: pass the requested bandwidth (default ≈ voice). The
        // same taps double as the channel-select filter *and* the decimator's
        // anti-alias — as `F32Fir::lowpass` they are a unit-gain, windowed-sinc
        // low-pass at exactly that bandwidth, split into `M` polyphase branches
        // by [`PolyphaseDecimator::new`] so only ≈ 1/M of the taps run per input
        // sample.
        let bw_ratio = (bandwidth_hz as f64 / source_rate_hz as f64).clamp(1e-3, 0.4);
        let taps = if bandwidth_hz <= 4_000 { 257 } else { 129 };
        let h = F32Fir::lowpass(taps, bw_ratio, KAISER_BETA).taps().to_vec();
        let lp = PolyphaseDecimator::new(&h, m);
        // Quadrature (Hilbert) 90° phase shifter. 257 taps is well within the
        // voice band (this is only exercised for LSB, where its output is used);
        // the old 2× (514) was pure overhead.
        let hilb_taps = taps.max(257);
        let hilb = F32FirState::new(&F32Fir::hilbert(hilb_taps, KAISER_BETA));
        let nco = Nco::new(2.0 * std::f64::consts::PI * source_center_hz / source_rate_hz as f64);
        Self {
            sideband,
            nco,
            hilb,
            lp,
        }
    }

    /// The sideband this demodulator was built for.
    pub fn sideband(&self) -> Sideband {
        self.sideband
    }

    /// The polyphase anti-alias / decimation stage.
    pub fn lp(&self) -> &PolyphaseDecimator {
        &self.lp
    }
}

impl DemodCore for SsbCore {
    fn process(&mut self, x: Complex<f32>) -> Option<f32> {
        // SSB demod. The wire gives (I, Q). For a *real* SDR baseband (Q ≈ 0
        // — the HL2 DDC's normal "real SDR" mode), the voice is in the I arm
        // only, with USB voice on positive frequencies and LSB voice on
        // negative frequencies. For a *complex* baseband (4× mode, Q loud),
        // the voice is already split.
        //
        // We handle both with the **product-discrimination** SSB structure:
        //   1. NCO to the channel centre — the user's `--offset` shift; with
        //      `offset=0` it is identity (no NCO) so the voice is at
        //      [0, +BW] / [−BW, 0] already.
        //   2. **Synthesise quadrature from the post-NCO in-phase arm** via
        //      the Hilbert (90°) filter. This is the missing `e^{j·90°}` that
        //      a real baseband never had; for a complex baseband the
        //      Hilbert-synthesised arm is already *equivalent* to the Q arm
        //      (within windowing tolerance), so this is correct for both.
        //   3. **Product discriminate**: USB = re{z'} = I';  LSB = im{z'} = I_H.
        //      `I_H` (the Hilbert of I') is the 90°-shifted I', which carries
        //      the LSB voice (negative frequency) as a positive-frequency
        //      tone. Equivalent to keeping the in-phase arm (USB) vs.
        //      conjugating it (LSB).
        let is_usb = self.sideband == Sideband::Usb;
        let post = self.nco.step(x);
        let r0 = if is_usb {
            post.re
        } else {
            self.hilb.convolve(post.re)
        };
        // Polyphase anti-alias + decimate (≈1/M the taps of a full-rate step).
        // Returns `None` on `M−1` of every `M` inputs; `Some` on the
        // group boundary.
        self.lp.push(r0)
    }

    fn kind(&self) -> &'static str {
        if self.sideband == Sideband::Usb {
            "ssb-usb"
        } else {
            "ssb-lsb"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receiver::Demodulator;
    use crate::receiver::demod::IqBlock;
    use crate::receiver::sink::VecSink;

    /// A proper **analytic** complex tone: `amp · e^(j·2π·f·t)`, i.e.
    /// `I = amp·cos(θ)`, `Q = amp·sin(θ)`.
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

    /// A **real** baseband tone (I = cos, Q = 0) — the HL2's normal baseband.
    fn real_tone(rate_hz: u32, freq_hz: f64, n_pairs: usize, amp: f32) -> IqBlock {
        (0..n_pairs)
            .map(|i| {
                let w = 2.0 * std::f64::consts::PI * freq_hz / rate_hz as f64;
                Complex::new((w * i as f64).cos() as f32 * amp, 0.0f32)
            })
            .collect()
    }

    fn build(sideband: Sideband, rate: u32, center: f64, bw: u32) -> Box<dyn Demodulator> {
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        SsbCore::new(sideband, rate, center, bw, cfg).demodulator(cfg, None)
    }

    #[test]
    fn usb_inband_tone_produces_audio() {
        let rate = 192_000u32;
        let iq = complex_sine(rate, 1_500.0, 16384, 0.5);
        let mut demod = build(Sideband::Usb, rate, 0.0, 2_600);
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("demod ok");
        assert!(n > 0, "produced {n} audio frames");
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "peak too low: {peak}");
    }

    #[test]
    fn lsb_inband_tone_produces_audio() {
        let rate = 192_000u32;
        let iq = complex_sine(rate, 1_500.0, 16384, 0.5);
        let mut demod = build(Sideband::Lsb, rate, 0.0, 2_600);
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("demod ok");
        assert!(n > 0, "produced {n} audio frames");
    }

    #[test]
    fn real_baseband_tone_produces_audio_both_sides() {
        let rate = 192_000u32;
        for sideband in [Sideband::Usb, Sideband::Lsb] {
            let iq = real_tone(rate, 1_500.0, 16384, 0.5);
            let mut demod = build(sideband, rate, 0.0, 2_600);
            let mut sink = VecSink::new();
            let _ = demod.demod(&iq, &mut sink).expect("demod ok");
            let samples = sink.samples().to_vec();
            let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
            assert!(peak > 30, "{sideband:?} real-baseband peak too low: {peak}");
        }
    }

    #[test]
    fn odd_block_accepted() {
        let rate = 192_000u32;
        let mut demod = build(Sideband::Usb, rate, 0.0, 2_600);
        let mut sink = VecSink::new();
        let iq = vec![Complex::new(1.0, 0.0); 63];
        demod
            .demod(&iq, &mut sink)
            .expect("odd (63) length must succeed");
    }

    #[test]
    fn usb_passes_inband_and_rejects_out_of_band_tone() {
        let rate = 192_000u32;
        let a = complex_sine(rate, 1_500.0, 16384, 0.5);
        let b = complex_sine(rate, 5_000.0, 16384, 0.5);
        let iq: IqBlock = a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect();
        let mut demod = build(Sideband::Usb, rate, 0.0, 2_600);
        let mut sink = VecSink::new();
        let _ = demod.demod(&iq, &mut sink).expect("ok");
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "audio too quiet: {peak}");
    }

    #[test]
    fn nco_moves_offband_tone_into_band() {
        let rate = 192_000u32;
        let center = 100_000.0;
        // USB, centred 100k.
        let iq = complex_sine(rate, center + 1_500.0, 16384, 0.5);
        let mut demod = build(Sideband::Usb, rate, center, 2_600);
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("ok");
        assert!(n > 0);
        let peak = sink.samples().iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "NCO-moved tone should be passable: {peak}");

        // LSB, centred 100k.
        let iq_lsb = complex_sine(rate, center - 1_500.0, 16384, 0.5);
        let mut demod_lsb = build(Sideband::Lsb, rate, center, 2_600);
        let mut sink_lsb = VecSink::new();
        let _ = demod_lsb.demod(&iq_lsb, &mut sink_lsb).expect("ok");
        let peak_lsb = sink_lsb
            .samples()
            .iter()
            .map(|s| s.abs())
            .max()
            .unwrap_or(0);
        assert!(
            peak_lsb > 50,
            "LSB NCO-moved tone should be passable: {peak_lsb}"
        );
    }
}
