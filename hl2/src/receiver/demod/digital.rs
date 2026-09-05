//! FT8 / JS8 / FT4 12 kHz digital-mode demodulator — the **mode core**.
//!
//! The FT8/JS8/FT4 path is **NCO → in-phase (USB) arm → polyphase
//! anti-alias/decimate**. The audio tail (AGC + DC-block, pre-AGC
//! [`RawSampleTap`] — which `Ft8Tap` / `Js8Tap` / `Ft4Tap` attach to for the
//! slot decoders, S-meter, `i16` sink write) is shared with every mode via
//! the [`AudioEngine`]; this module implements only the USB in-phase core.
//!
//! The FT8 passband (~1.4–2.9 kHz) sits in the *in-phase* arm after the USB
//! NCO, so the quadrature synthesis an LSB SSB demod needs is pure overhead
//! here and is dropped. The anti-alias / channel filter is the same
//! Kaiser-windowed windowed-sinc design as the SSB path, consumed as a
//! [`PolyphaseDecimator`] — ~M× fewer MACs per sample than a full-rate
//! 257-tap convolution.
//!
//! For `Mode::Ft8` / `Mode::Js8` / `Mode::Ft4` the pipeline skips the
//! quadrature synthesis entirely (USB) and decimates directly to the
//! 12 kHz digital-mode window rate — see [`super::ssb`] for the voice-rate
//! variant.
use num_complex::Complex;

use super::core::DemodCore;
use super::dsp::{F32Fir, KAISER_BETA, Nco, PolyphaseDecimator};
use crate::receiver::AudioConfig;

/// The FT8/JS8/FT4 demodulator **core**: USB SSB mix at a 12 kHz output
/// rate (the `mfsk-core` / JS8 decoder's fixed-window rate), decimated by
/// the polyphase anti-alias stage. The audio tail (AGC, pre-AGC tap, S-meter,
/// sink write) is attached by [`DemodCore::demodulator`] into the shared
/// [`AudioEngine`].
///
/// ```text
/// complex I/Q
///   └─► NCO × e^(−j·φ)
///        └─► in-phase (USB) arm directly — no Hilbert, no LSB path
///             └─► PolyphaseDecimator (LPF BW from `bandwidth`, M = src/12k)
///                  └─► [AudioEngine: pre-AGC tap → AGC + DC-block → i16 → sink]
/// ```
pub struct DigitalCore {
    nco: Nco,
    lp: PolyphaseDecimator,
    /// The mode this core was built for (`ft8` / `js8` / `ft4`), for debug
    /// logs (`DigitalCore::kind` returns this).
    mode_label: &'static str,
}

impl DigitalCore {
    /// Build a digital-mode **core** from `source_rate_hz` Hz complex I/Q
    /// down to the 12 kHz output. `source_rate_hz` must be an integer
    /// multiple of 12 kHz of at least 4×.
    ///
    /// `audio` carries the output rate (e.g. `12_000` for FT8/JS8/FT4) and
    /// gain; used to (a) set the decimation factor and (b) let
    /// [`DemodCore::demodulator`] build the audio tail. `mode_label` is a
    /// short identifier for debug logs (e.g. `"ft8"`) — surfaced via
    /// [`DemodCore::kind`].
    pub fn new(
        source_rate_hz: u32,
        source_center_hz: f64,
        audio: AudioConfig,
        mode_label: &'static str,
    ) -> Self {
        let m = source_rate_hz as usize / audio.rate_hz as usize;
        let bw_ratio = (2_600.0f64 / source_rate_hz as f64).clamp(1e-3, 0.4);
        // 511 taps (→ 513 odd) on the 12 kHz digital path: with Kaiser β = 12
        // this puts the out-of-band rejection floor at ~107 dB across the
        // 3.8 – 6.0 kHz band — well past the 80 dB floor the
        // `ft8_js8_path_rejects_out_of_passband_signal_by_at_least_80_db`
        // regression guards, and gives operator-visible headroom for a
        // stronger adjacent-channel signal. 257 taps was borderline (worst
        // case ~82 dB) because a 4 kHz offset lands right in the *transition
        // band* of a 257-tap / 2.6 kHz / 192 kHz filter, where sidelobe peak
        // height is β- and tap-count-sensitive. Cost is modest: 513 taps ÷ 16
        // polyphase ≈ 32 MACs per input sample.
        let h = F32Fir::lowpass(511, bw_ratio, KAISER_BETA).taps().to_vec();
        let lp = PolyphaseDecimator::new(&h, m);
        let nco = Nco::new(2.0 * std::f64::consts::PI * source_center_hz / source_rate_hz as f64);
        Self {
            nco,
            lp,
            mode_label,
        }
    }
}

impl DemodCore for DigitalCore {
    fn process(&mut self, x: Complex<f32>) -> Option<f32> {
        let post = self.nco.step(x);
        // USB: the in-phase arm directly — no Hilbert, no LSB path.
        self.lp.push(post.re)
    }

    fn kind(&self) -> &'static str {
        self.mode_label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receiver::Demodulator;
    use crate::receiver::demod::{IqBlock, RawSampleTap};
    use crate::receiver::sink::VecSink;

    /// A proper **analytic** complex tone.
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

    /// A real FFT of `x` via DFT (test-sized inputs) → magnitude spectrum.
    fn spec_mag(x: &[f32]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for (t, v) in x.iter().enumerate() {
                    let w = 2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
                    re += *v as f64 * w.cos();
                    im -= *v as f64 * w.sin();
                }
                (re * re + im * im).sqrt()
            })
            .collect()
    }

    /// A [`RawSampleTap`] that appends every `f32` slice to an
    /// `Arc<Mutex<Vec<f32>>>`.
    #[derive(Debug)]
    struct VecF32Tap {
        sink: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    }

    impl VecF32Tap {
        fn new(sink: std::sync::Arc<std::sync::Mutex<Vec<f32>>>) -> Self {
            Self { sink }
        }
    }

    impl RawSampleTap for VecF32Tap {
        fn append(&self, samples: &[f32]) {
            self.sink
                .lock()
                .expect("tap not poisoned")
                .extend_from_slice(samples);
        }
    }

    /// Build a `DigitalCore` with the given source rate + NCO + label.
    /// `tap` is the optional pre-AGC raw-sample tap. Returns a
    /// `Box<dyn Demodulator>` via `DemodCore::demodulator`.
    fn build(
        rate: u32,
        nco_hz: f64,
        label: &'static str,
        tap: Option<std::sync::Arc<dyn RawSampleTap>>,
    ) -> Box<dyn Demodulator> {
        let cfg = AudioConfig {
            rate_hz: 12_000,
            gain_db: 0.0,
        };
        DigitalCore::new(rate, nco_hz, cfg, label).demodulator(cfg, tap, None)
    }

    /// The DigitalCore must produce 12 kHz monitor audio (AGC'd), and
    /// must dispatch for `Mode::Ft8`.
    #[test]
    fn ft8_demod_tone_produces_12k_audio() {
        let rate = 192_000u32;
        let iq = complex_sine(rate, 1_500.0, 16384, 0.5);
        let mut demod = build(rate, 0.0, "ft8", None);
        assert_eq!(demod.audio_format().rate_hz, 12_000);
        let mut sink = VecSink::new();
        let _ = demod.demod(&iq, &mut sink).expect("demod ok");
        // 16384 inputs at M = 192k/12k = 16 → exactly 1024 decimated samples,
        // one whole emission (1024 ≥ AUDIO_EMIN = 240).
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 30, "FT8 monitor audio too quiet: {peak}");
        assert_eq!(samples.len(), 1024, "got {} samples", samples.len());
    }

    /// Regression: the digital path rejects a signal just outside the
    /// passband by **≥ 80 dB** relative to an in-band reference, in the
    /// 12 kHz FT8/JS8 path.
    #[test]
    fn ft8_js8_path_rejects_out_of_passband_signal_by_at_least_80_db() {
        let rate = 192_000u32;
        let n = 32_768;
        let audio_rate = 12_000.0;

        let amplitude_at = |f_hz: f64| -> f64 {
            let tap_sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
            let tap = std::sync::Arc::new(VecF32Tap::new(tap_sink.clone()));
            let mut demod = build(rate, 0.0, "ft8", Some(tap));
            let iq = complex_sine(rate, f_hz, n, 0.5);
            let mut vec_sink = VecSink::new();
            let _ = demod.demod(&iq, &mut vec_sink).expect("demod");
            demod.flush_audio(&mut vec_sink).ok();
            let audio: Vec<f32> = tap_sink.lock().expect("tap not poisoned").clone();
            let spec = spec_mag(&audio);
            let bin = (f_hz / audio_rate * spec.len() as f64).round() as usize;
            let lo = bin.saturating_sub(2);
            let hi = (bin + 2).min(spec.len().saturating_sub(1));
            spec[lo..=hi].iter().cloned().fold(0.0f64, f64::max)
        };

        let in_amp = amplitude_at(1_300.0);
        assert!(in_amp > 1e-6, "in-band tone not audible: {in_amp:.3e}");
        for offset in [4_000.0, 5_000.0, 8_000.0] {
            let out_amp = amplitude_at(offset);
            let rej_db = 20.0 * f64::log10(in_amp / out_amp.max(1e-12));
            assert!(
                rej_db > 80.0,
                "offset {offset} Hz: in-band(1.3 kHz) {in_amp:.2e} vs \
                 out-of-band {out_amp:.2e} → rejection only {rej_db:.1} dB \
                 (want > 80 dB)",
            );
        }
    }

    /// Regression: the digital path must pass a signal placed at a non-zero
    /// **NCO offset** into the 12 kHz passband at usable gain, and reject an
    /// out-of-band offset.
    #[test]
    fn ft8_path_passes_signal_at_nonzero_nco_offset() {
        let rate = 192_000u32;
        let n = 65_536;
        let nco_hz = 24_000.0;

        let inband_bin = |f_in: f64, expect_hz: f64| -> f64 {
            let tap_sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
            let tap = std::sync::Arc::new(VecF32Tap::new(tap_sink.clone()));
            let mut demod = build(rate, nco_hz, "ft8", Some(tap));
            let iq = complex_sine(rate, f_in, n, 0.5);
            let mut vec_sink = VecSink::new();
            let _ = demod.demod(&iq, &mut vec_sink).expect("demod");
            demod.flush_audio(&mut vec_sink).ok();
            let audio: Vec<f32> = tap_sink.lock().expect("poisoned").clone();
            let spec = spec_mag(&audio);
            let bin = (expect_hz / 12_000.0 * spec.len() as f64).round() as usize;
            let lo = bin.saturating_sub(2);
            let hi = (bin + 2).min(spec.len().saturating_sub(1));
            spec[lo..=hi].iter().cloned().fold(0.0f64, f64::max)
        };

        let in_amp = inband_bin(nco_hz + 1_500.0, 1_500.0);
        eprintln!("[test] in-band(NCO+1.5k) amplitude @1.5kHz = {in_amp:.4e}");
        let out_amp = inband_bin(nco_hz + 5_000.0, 5_000.0);
        eprintln!("[test] out-of-band(NCO+5k) amplitude @5kHz = {out_amp:.4e}");
        let dc_sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
        let dc_tap = std::sync::Arc::new(VecF32Tap::new(dc_sink.clone()));
        let mut demod_dc = build(rate, 0.0, "ft8", Some(dc_tap));
        let iq_dc = complex_sine(rate, 1_500.0, n, 0.5);
        let mut ws = VecSink::new();
        let _ = demod_dc.demod(&iq_dc, &mut ws).expect("demod");
        demod_dc.flush_audio(&mut ws).ok();
        let audio_dc: Vec<f32> = dc_sink.lock().expect("poisoned").clone();
        let spec_dc = spec_mag(&audio_dc);
        let b = (1_500.0 / 12_000.0 * spec_dc.len() as f64).round() as usize;
        let dc_amp = spec_dc[(b - 2).max(0)..=b + 2]
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);
        eprintln!("[test] reference(DC, offset-0) amplitude @1.5kHz = {dc_amp:.4e}");

        assert!(
            in_amp > 1e-4,
            "in-band NCO-offset tone too quiet: {in_amp:.3e} (reference {dc_amp:.3e})",
        );
        assert!(
            dc_amp / in_amp < 15.0,
            "NCO offset lost the signal: in-band {in_amp:.3e} vs reference {dc_amp:.3e} = {:.1} dB",
            20.0 * f64::log10(dc_amp / in_amp.max(1e-12)),
        );
        assert!(
            dc_amp / out_amp > 100.0,
            "out-of-band insufficient: in {in_amp:.3e} vs out {out_amp:.3e} (want ≥ 40 dB)",
        );
    }
}
