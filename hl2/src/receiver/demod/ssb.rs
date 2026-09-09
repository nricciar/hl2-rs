//! SSB (USB/LSB) voice demodulator — the **mode core**.
//!
//! The SSB path is **NCO → phasing discriminate → polyphase
//! anti-alias/decimate**. This module implements only that per-sample DSP as
//! a [`DemodCore`]. The audio tail (DC-block + RMS-target AGC + pre-AGC
//! [`RawSampleTap`] + `i16` sink write) is shared with every mode by the
//! [`AudioEngine`], which [`StandardDemod`] attaches here via
//! [`DemodCore::demodulator`].
//!
//! ## Phasing discriminate (single-sideband selection)
//!
//! The HL2 EP6 baseband is **genuine complex I/Q** (both arms carry energy —
//! see PROTOCOL.md §16.1/§16.3d), arriving as a *two-sided* band around the
//! NCO. To isolate one sideband we use the standard **phasing method** (a.k.a.
//! Hartley / product-discrimination) on that complex baseband. With `H{·}` the
//! `F32Fir::hilbert` transform (`H{cos ωt} = sin ωt`, `H{sin ωt} = −cos ωt`,
//! i.e. `H ≡ −j` on positive-frequency tones), the one-sided selections are:
//!
//! ```text
//!   pass +f (upper,  real "USB-of-baseband"):  I − H{Q}
//!   pass −f (lower,  real "LSB-of-baseband"):  I + H{Q}
//! ```
//!
//! This HL2 DDC is **frequency-inverted**: a signal *above* the NCO (real USB)
//! appears at a *negative* complex baseband frequency, and one *below* the NCO
//! (real LSB) at a positive one (measured live, `hl2 ft8 --probe`; hub.rs).
//! So the mapping used by `process` is:
//!
//! ```text
//!   real USB (above NCO = complex −f) → I + H{Q}   (passes −f)
//!   real LSB (below NCO = complex +f) → I − H{Q}   (passes +f)
//! ```
//!
//! which is the inverse of the textbook sign convention — the two are swapped
//! only because of the DDC's inversion. This reuses **both** arms, so it is
//! correct whether the input is the HL2's complex baseband or a legacy real
//! (Q≈0) stream (there `H{Q}=0` and both branches reduce to `I`, the honest
//! result — a real stream genuinely carries both sidebands).
//!
//! Because `F32Fir::hilbert` has group delay `(n−1)/2` samples, the I arm is
//! delayed by exactly that much before the combine so the two terms stay in
//! phase across the voice band. Without the delay the cancellation is only
//! partial and image rejection degrades with frequency.
//!
//! The channel-select FIR is the anti-alias filter of a
//! [`PolyphaseDecimator`], so only ≈ 1/M of its taps touch each sample (~M×
//! fewer MACs, M = the decimation factor).

use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts;
use num_complex::Complex;

use super::core::DemodCore;
use super::dsp::{F32Fir, F32FirState, KAISER_BETA, Nco, PolyphaseDecimator};
use crate::receiver::AudioConfig;
use crate::receiver::Sideband;

/// The SSB demodulator **core** (USB or LSB).
///
/// Implements the **phasing method**: NCO → Hilbert(Q) + I-delay combine
/// (USB: `I + H{Q}`, LSB: `I − H{Q}` — the DDC's frequency inversion swaps
/// the textbook signs) → polyphase anti-alias/decimate. See the module
/// doc for the full derivation.
///
/// The audio tail (AGC/DC-block/tap/sink) is attached by
/// [`DemodCore::demodulator`] into the shared [`AudioEngine`].
pub struct SsbCore {
    sideband: Sideband,
    /// NCO (phase-recurrence oscillator). Identity when `source_center_hz`
    /// is 0 (i.e. the input is already baseband).
    nco: Nco,
    /// Quadrature (90° / Hilbert) phase-shifter. Applied to the post-NCO Q arm
    /// it produces `H{Q}` (90°-shifted Q) for the phasing combine. See
    /// `F32Fir::hilbert`. Used for **both** `Usb` and `Lsb` (the sideband is
    /// chosen by the add/subtract below, not by which arm we keep).
    hilb: F32FirState,
    /// Polyphase anti-alias + decimation stage (channel-select FIR's taps are
    /// its full-rate impulse response).
    lp: PolyphaseDecimator,
    /// Delay line matching the Hilbert FIR's group delay (its symmetric
    /// impulse is centred at `(len−1)/2`, so delay = `(len−1)/2` samples).
    /// The I arm is read out delayed by this much so `I[n−D]` aligns in phase
    /// with `H{Q}[n]` — required for coherent sideband cancellation.
    i_delay: Vec<f32>,
    /// Write position in [`Self::i_delay`] (holds the oldest sample to be read
    /// as the delayed I output, then overwritten).
    i_idx: usize,
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
        // Quadrature (Hilbert) 90° phase shifter on the Q arm. 257 taps is
        // well within the voice band — a wider window is pure overhead.
        let hilb_fir = F32Fir::hilbert(taps.max(257), KAISER_BETA);
        let hilb = F32FirState::new(&hilb_fir);
        // The I arm must be delayed by the Hilbert FIR's group delay so its
        // samples stay in phase with `H{Q}[n]` across the voice band. `hilb_fir`
        // has a symmetric impulse centred on sample `(len−1)/2`, so the group
        // delay is exactly `(len−1)/2` full-rate samples.
        let i_delay_samples = (hilb_fir.len() - 1) / 2;
        let nco = Nco::new(2.0 * consts::PI * source_center_hz / source_rate_hz as f64);
        Self {
            sideband,
            nco,
            hilb,
            lp,
            i_delay: vec![0.0f32; i_delay_samples.max(1)],
            i_idx: 0,
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
        // 1. NCO to the channel centre — the user's `--offset` shift; with
        //    `offset=0` it is identity (no NCO) so the voice is already at
        //    [0, +BW] (USB) / [−BW, 0] (LSB).
        let post = self.nco.step(x);

        // 2. Phase-shift the **Q arm** by 90° (Hilbert) for both sidebands, so
        //    below we combine it with the delayed I arm to pick a sideband.
        //    (The old code collapsed to `post.re`, discarding Q entirely,
        //    which is why image rejection was 0 dB: `Re{e^{±jωt}}` are
        //    identical.)
        let q_h = self.hilb.convolve(post.im);

        // 3. Delay the **I arm** by the Hilbert FIR's group delay (`D`
        //    samples) so the two terms stay in phase across the voice band.
        let i_delayed = self.i_delay[self.i_idx];
        self.i_delay[self.i_idx] = post.re;
        self.i_idx += 1;
        if self.i_idx == self.i_delay.len() {
            self.i_idx = 0;
        }

        // 4. **Phasing combine** (single-sideband selection).
        //
        // Phasing-method identities (with `H{cos}=sin`, `H{sin}=−cos`
        // — `F32Fir::hilbert`):
        //    I − H{Q} → passes +f, cancels −f
        //    I + H{Q} → passes −f, cancels +f
        //
        // This DDC is **frequency-inverted** (an above-NCO signal lands at a
        // *negative* complex frequency — measured live: signal at +14 kHz
        // above NCO sits at −14 kHz in baseband, 9 dB above the +14 kHz
        // image; see `hl2 ft8 --probe`, hub.rs:1631-1633). So:
        //    real USB (above NCO) = complex −f  → pass −f → I + H{Q}
        //    real LSB (below NCO) = complex +f  → pass +f → I − H{Q}
        let r0 = if self.sideband == Sideband::Usb {
            i_delayed + q_h
        } else {
            i_delayed - q_h
        };

        // 5. Polyphase anti-alias + decimate (≈1/M the taps of a full-rate
        //    step). Returns `None` on `M−1` of every `M` inputs; `Some` on the
        //    group boundary.
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
    use crate::receiver::demod::{IqBlock, RawSampleTap};
    use crate::receiver::sink::VecSink;
    use alloc::boxed::Box;
    use std::eprintln;

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

    /// A tap that forwards every pre-AGC f32 slice to a shared `Vec`, so the
    /// tests can measure the *signal spectrum* without the RMS-AGC masking it.
    #[derive(Debug)]
    struct VecF32Tap {
        sink: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    }
    impl RawSampleTap for VecF32Tap {
        fn append(&self, samples: &[f32]) {
            self.sink
                .lock()
                .expect("tap not poisoned")
                .extend_from_slice(samples);
        }
    }

    /// Build a plain SSB voice demod that forwards pre-AGC f32 audio to `sink`.
    fn build_tapped(
        sb: Sideband,
        rate: u32,
    ) -> (
        Box<dyn Demodulator>,
        std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    ) {
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let sink: std::sync::Arc<std::sync::Mutex<Vec<f32>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let demod = SsbCore::new(sb, rate, 0.0, 2_600, cfg).demodulator(
            cfg,
            Some(std::sync::Arc::new(VecF32Tap { sink: sink.clone() })),
        );
        (demod, sink)
    }

    /// The peak magnitude of the DFT of `x` around the bin nearest `freq_hz`
    /// (audio-rate axis). A clean in-band voice tone rides there; a strong
    /// mirror on the *other* sideband lands at the *same* bin, so this is the
    /// honest measure of image rejection (the RMS-AGC would hide it).
    fn level_at(x: &[f64], freq_hz: f64) -> f64 {
        let n = x.len();
        let bin = (freq_hz / 4_800.0 * n as f64).round() as usize;
        let lo = bin.saturating_sub(2);
        let hi = (bin + 2).min(n - 1);
        x[lo..=hi].iter().cloned().fold(0.0f64, f64::max)
    }

    /// Feed `sideband` a complex tone at **`freq`** (Hz, signed, in the
    /// DDC's baseband — where *negative* is real-USB / above-NCO) and return
    /// the pre-AGC audio's in-band level at `|freq|`.
    fn inband_level(sb: Sideband, rate: u32, freq: f64) -> f64 {
        let (mut demod, sink) = build_tapped(sb, rate);
        let iq = complex_sine(rate, freq, 16_384, 0.5);
        let mut vec_sink = VecSink::new();
        demod.demod(&iq, &mut vec_sink).expect("demod");
        demod.flush_audio(&mut vec_sink).ok();
        let audio: Vec<f64> = sink
            .lock()
            .expect("tap")
            .iter()
            .map(|v| *v as f64)
            .collect();
        level_at(&audio, freq.abs())
    }

    /// **Image-rejection regression.** A correct SSB demod must pass a tone on
    /// its own sideband and reject a tone on the *other* sideband by a
    /// meaningful margin.
    ///
    /// This DDC is frequency-inverted (an above-NCO real-USB signal lands at a
    /// *negative* complex frequency; below-NCO real-LSB at **+f** — see
    /// `hl2 ft8 --probe`, hub.rs:1631-1633). So:
    ///   * **complex −f** = **real USB** → USB demod passes it, LSB rejects it.
    ///   * **complex +f** = **real LSB** → LSB demod passes it, USB rejects it.
    ///
    /// Before the phasing fix `SsbCore` collapsed to `post.re`, so *both*
    /// sidebands passed at nearly full level (≈ 0 dB image rejection). The
    /// pre-fix measurement was:
    ///   +f tone (real-LSB): USB demod 2.275e-1, LSB demod 3.954e-1  (≈ 0 dB)
    /// After the fix (at the same tone):
    ///   −f tone (real-USB): USB demod 6.469e-1, LSB demod 1.424e-1  (≈ 13 dB)
    /// The 13 dB floor is finite because: at 192 kHz full rate, 1.5 kHz is
    /// only 1.5% of Nyquist — right inside a 257-tap Hilbert's transition
    /// region (≈5 kHz wide at Kaiser β=12), where magnitude and phase are
    /// least accurate. This test uses **> 3× (= 9.5 dB)**, which is well
    /// above the pre-fix ≈ 0 dB floor and well below the fixed ≈ 13 dB
    /// ceiling, so it cleanly discriminates fixed from broken.
    #[test]
    fn image_rejection_by_sideband() {
        let rate = 192_000u32;
        // Real-USB tone (complex −f), and real-LSB tone (complex +f).
        let usb_pass = inband_level(Sideband::Usb, rate, -1_500.0);
        let usb_reject = inband_level(Sideband::Usb, rate, 1_500.0);
        let lsb_pass = inband_level(Sideband::Lsb, rate, 1_500.0);
        let lsb_reject = inband_level(Sideband::Lsb, rate, -1_500.0);
        eprintln!(
            "[ssb] image-rejection test:\
             \n  real-USB (complex −f): USB-demod PASS  {usb_pass:.3e}   LSB-demod REJECT {lsb_reject:.3e}\
             \n  real-LSB (complex +f): LSB-demod PASS  {lsb_pass:.3e}   USB-demod REJECT {usb_reject:.3e}\
             \n  USB: {usb_pass:.3e}/{usb_reject:.3e} = {:.1} dB   LSB: {lsb_pass:.3e}/{lsb_reject:.3e} = {:.1} dB",
            20.0 * (usb_pass / usb_reject.max(1e-12)).log10(),
            20.0 * (lsb_pass / lsb_reject.max(1e-12)).log10(),
        );
        assert!(
            usb_pass > 1e-4,
            "real-USB tone must be passable by the USB demod: {usb_pass:.2e}"
        );
        assert!(
            lsb_pass > 1e-4,
            "real-LSB tone must be passable by the LSB demod: {lsb_pass:.2e}"
        );
        assert!(
            usb_pass / usb_reject.max(1e-12) > 3.0,
            "USB demod leaks real-LSB image: pass {usb_pass:.2e} vs reject {usb_reject:.2e} \
             (rejection must be > 3× = 9.5 dB; pre-fix was ≈ 1× = 0 dB)"
        );
        assert!(
            lsb_pass / lsb_reject.max(1e-12) > 3.0,
            "LSB demod leaks real-USB image: pass {lsb_pass:.2e} vs reject {lsb_reject:.2e} \
             (rejection must be > 3× = 9.5 dB; pre-fix was ≈ 1× = 0 dB)"
        );
    }
}
