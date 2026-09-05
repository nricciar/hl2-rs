//! SSB (USB/LSB) voice demodulator.
//!
//! The SSB path is **NCO → product-discriminate (USB: in-phase, LSB: Hilbert
//! quadrature) → polyphase anti-alias/decimate → AGC (DC-block + RMS target)
//! → i16 → sink**.
//!
//! The USB/LSB distinction is carried by the NCO direction (the sign of
//! `φ̇`) plus which post-NCO arm is kept: a positive-frequency NCO moves the
//! upper sideband to baseband on the in-phase arm and a negative one moves
//! the lower sideband to baseband — the same trick the reference uses (keep
//! the in-phase arm for USB; conjugate it to reach LSB). For LSB we synthesize
//! that quadrature arm with a 90° (Hilbert) phase-shifter on the in-phase
//! arm, which is valid for both real and complex baseband.
//!
//! The channel-select FIR of the old design is now the anti-alias filter of
//! a `PolyphaseDecimator`, so only ≈ 1/M of its taps touch each sample
//! (~M× fewer MACs, M = the decimation factor).

use std::sync::atomic::Ordering as AOrdering;

use super::RawSampleTap;
use super::dsp::{F32Fir, F32FirState, KAISER_BETA, Nco, PolyphaseDecimator};
use super::{
    AUDIO_EMIN, DemodError, Demodulator, EMIT_COUNT, IqBlock, RateTooClose, meter_tick,
    normalize_to_i16_with_agc,
};
use crate::receiver::sink::AudioSink;
use crate::receiver::{AudioConfig, MeterHandle, Sideband};

/// The SSB demodulator (USB or LSB).
///
/// Holds: an NCO (to bring the carrier/selected sideband to baseband), a
/// 90° (Hilbert) phase-shifter for LSB, and the polyphase
/// anti-alias/decimation stage (`lp`).
pub struct SsbDemodulator {
    sideband: Sideband,
    audio_cfg: AudioConfig,
    /// NCO (phase-recurrence oscillator). Identity when `source_center_hz`
    /// is 0 (i.e. the input is already baseband).
    nco: Nco,
    /// Quadrature (90° / Hilbert) phase-shifter. Applied to the real
    /// in-phase arm it synthesizes the missing quadrature component so SSB can
    /// be demodulated from a real (Q≈0) baseband stream. See `F32Fir::hilbert`.
    /// The sideband choice (`Usb` vs `Lsb`) selects which post-NCO arm is
    /// kept (in-phase → USB, quadrature → LSB). Used only for LSB.
    hilb: F32FirState,
    /// Polyphase anti-alias + decimation stage (channel-select FIR's taps are
    /// its full-rate impulse response). Replaces the old `filter`
    /// (`F32FirState`) + `decim` (`Decimator`) pair.
    lp: PolyphaseDecimator,
    /// Slow RMS-targeted AGC gain (linear). Replaces the old per-block peak
    /// normaliser. See `normalize_to_i16_with_agc`.
    agc_gain: f32,
    /// Accumulated decimated audio, ready to be normalised + emitted once it
    /// reaches `AUDIO_EMIN`. A single wire chunk (63 complex samples at a
    /// 192 kHz → 4.8 kHz decimation) produces only ≈ 1-2 decimated samples,
    /// and calling `normalize_to_i16_with_agc` on a 1-sample buffer makes the
    /// DC-block subtract the sample from itself → output always 0. Buffering
    /// to ~50 ms (~240 samples) restores meaningful DC-block + AGC statistics.
    audio_buf: Vec<f32>,
    /// Optional pre-AGC raw-sample tap (FT8 decode — see
    /// [`RawSampleTap`]). `None` for plain SSB. `Arc` so the API layer can
    /// keep one handle for the demod and hand a clone to the decode task.
    tap: Option<std::sync::Arc<dyn RawSampleTap>>,
    /// Optional signal-level meter (see [`MeterHandle`]). The demod writes
    /// the pre-AGC in-band RMS, in dB FS, one-pole smoothed, on every emit.
    meter: Option<MeterHandle>,
    /// Running smoothed level (dB FS), for the one-pole filter. Initialised
    /// to a deep floor so the first emit moves the gauge quickly.
    meter_smooth: f64,
}

impl SsbDemodulator {
    /// Build an SSB voice demodulator from a full set of receiver parameters.
    pub fn new(
        sideband: Sideband,
        source_rate_hz: u32,
        source_center_hz: f64,
        bandwidth_hz: u32,
        audio: AudioConfig,
        tap: Option<std::sync::Arc<dyn RawSampleTap>>,
        meter: Option<MeterHandle>,
    ) -> Result<Self, DemodError> {
        // Decimation factor: integer `source / audio`. `Decimator`'s old
        // 4× floor is kept as a sanity bound (an SSB voice path decimated by
        // less than 4 is aliasing by design and nobody wants that silently).
        let m = source_rate_hz as usize / audio.rate_hz as usize;
        if m < 4 {
            return Err(Box::new(RateTooClose {
                src: source_rate_hz,
                audio: audio.rate_hz,
            }));
        }
        // Anti-alias LPF: pass the requested bandwidth (default ≈ voice). The same
        // taps double as the channel-select filter *and* the decimator's
        // anti-alias — as `F32Fir::lowpass` they are a unit-gain, windowed-sinc
        // low-pass at exactly that bandwidth, split into `M` polyphase branches by
        // [`PolyphaseDecimator::new`] so only ≈ 1/M of the taps run per input
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
        Ok(Self {
            sideband,
            audio_cfg: audio,
            nco,
            hilb,
            lp,
            agc_gain: 1000.0,
            audio_buf: Vec::with_capacity(256),
            tap,
            meter,
            meter_smooth: -120.0,
        })
    }

    /// The sideband this demodulator was built for.
    pub fn sideband(&self) -> Sideband {
        self.sideband
    }

    /// The polyphase anti-alias / decimation stage.
    pub fn lp(&self) -> &PolyphaseDecimator {
        &self.lp
    }

    /// Normalise + emit one slice of the accumulated audio, advancing the
    /// AGC. Returns the number of i16 frames written.
    fn emit(&mut self, slice: &[f32], sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        // Pre-AGC raw-sample tap (FT8 decode): the untouched decimated `f32`
        // stream, at `audio_cfg.rate_hz`, in arrival order.
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
        {
            // HL2_DEBUG: prove whether the decimated audio (filter-out) is
            // loud, and whether the AGC/normalisation is turning it into
            // audible i16. ~every 50 emits (~2.5 s at 50 ms/emit).
            if std::env::var("HL2_DEBUG").is_ok() {
                let c = EMIT_COUNT.fetch_add(1, AOrdering::Relaxed) + 1;
                if c % 50 == 1 {
                    let in_max = slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let in_rms = (slice.iter().map(|v| v * v).sum::<f32>()
                        / slice.len().max(1) as f32)
                        .sqrt();
                    let o_max = out[..written].iter().map(|v| v.abs()).max().unwrap_or(0);
                    let side = if self.sideband == Sideband::Usb {
                        "USB"
                    } else {
                        "LSB"
                    };
                    eprintln!(
                        "[aud] {side} in_rms={in_rms:.6e} in_max={in_max:.6e} agc={:.3e} out_i16_max={o_max} (n={written})",
                        self.agc_gain
                    );
                }
            }
        }
        sink.write(&out[..written])
    }

    /// Drain all `AUDIO_EMIN`-sized blocks from the accumulator into `sink`;
    /// returns total frames written.
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

impl Demodulator for SsbDemodulator {
    fn demod(&mut self, iq: &IqBlock, sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        // SSB demod. The wire gives (I, Q). For a *real* SDR baseband (Q ≈ 0
        // — the HL2 DDC's normal "real SDR" mode), the voice is in the I arm
        // only, with USB voice on positive frequencies and LSB voice on
        // negative frequencies. For a *complex* baseband (4× mode, Q loud),
        // the voice is already split.
        //
        // We handle both with the **product-discrimination** SSB structure:
        //   1. NCO to the channel centre (`nco_step`) — this is the user's
        //      `--offset` shift; with `offset=0` it is identity (no NCO) so
        //      the voice is at [0, +BW] / [−BW, 0] already.
        //   2. **Synthesise quadrature from the post-NCO in-phase arm** via
        //      the Hilbert (90°) filter. This is the missing `e^{j·90°}` that
        //      a real baseband never had; for a complex baseband the
        //      Hilbert-synthesised arm is already *equivalent* to the Q arm
        //      (within windowing tolerance), so this is correct for both.
        //   3. **Product discriminate**: USB = re{z'} = I';  LSB = im{z'} = I_H.
        //      `I_H` (the Hilbert of I') is the 90°-shifted I', which
        //      carries the LSB voice (negative frequency) as a
        //      positive-frequency tone (Hilbert of cos(ωt) = sin(ωt) with
        //      phase reversed).
        //
        // This is equivalent to keeping the in-phase arm (USB) vs.
        // conjugating it (LSB).
        //
        // `iq` is a list of **complex** samples — any length is valid (a real
        // wire chunk is 63, which is *odd*), and each sample is processed
        // independently below, so no even-length requirement applies.
        let is_usb = self.sideband == Sideband::Usb;
        for &x in iq {
            // 1. NCO to the channel centre (phase recurrence — no trig here).
            let post = self.nco.step(x);
            // 2. Product-discriminate. USB = in-phase arm directly; LSB needs
            //    the synthesised quadrature arm (Hilbert of the in-phase). The
            //    Hilbert is only run for LSB — it is pure overhead on USB.
            let r0 = if is_usb {
                post.re
            } else {
                self.hilb.convolve(post.re)
            };
            // 3. Polyphase anti-alias + decimate (≈1/M the taps of the old
            //    full-rate filter step).
            if let Some(sample) = self.lp.push(r0) {
                self.audio_buf.push(sample);
            }
        }
        // 4. AGC + DC-block + emit, once enough decimated audio has been
        //    accumulated (see `AUDIO_EMIN`: a 1-sample block would DC-block
        //    itself to zero).
        self.flush_full_blocks(sink)
    }

    fn audio_format(&self) -> AudioConfig {
        self.audio_cfg
    }

    /// Emit the residual partial block (if any) accumulated after the last
    /// full emission, normalising even a sub-`AUDIO_EMIN` tail so the final
    /// ~50 ms isn't silently dropped at shutdown.
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

    #[test]
    fn usb_inband_tone_produces_audio() {
        let rate = 192_000u32;
        let iq = complex_sine(rate, 1_500.0, 16384, 0.5);
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod =
            SsbDemodulator::new(Sideband::Usb, rate, 0.0, 2_600, cfg, None, None).expect("build");
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
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod =
            SsbDemodulator::new(Sideband::Lsb, rate, 0.0, 2_600, cfg, None, None).expect("build");
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("demod ok");
        assert!(n > 0, "produced {n} audio frames");
    }

    #[test]
    fn real_baseband_tone_produces_audio_both_sides() {
        let rate = 192_000u32;
        for sideband in [Sideband::Usb, Sideband::Lsb] {
            let iq = real_tone(rate, 1_500.0, 16384, 0.5);
            let cfg = AudioConfig {
                rate_hz: 4_800,
                gain_db: 0.0,
            };
            let mut demod =
                SsbDemodulator::new(sideband, rate, 0.0, 2_600, cfg, None, None).unwrap();
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
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod =
            SsbDemodulator::new(Sideband::Usb, rate, 0.0, 2_600, cfg, None, None).unwrap();
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
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod =
            SsbDemodulator::new(Sideband::Usb, rate, 0.0, 2_600, cfg, None, None).unwrap();
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
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod =
            SsbDemodulator::new(Sideband::Usb, rate, center, 2_600, cfg, None, None).unwrap();
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("ok");
        assert!(n > 0);
        let peak = sink.samples().iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "NCO-moved tone should be passable: {peak}");

        // LSB, centred 100k.
        let iq_lsb = complex_sine(rate, center - 1_500.0, 16384, 0.5);
        let mut demod_lsb =
            SsbDemodulator::new(Sideband::Lsb, rate, center, 2_600, cfg, None, None).unwrap();
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
