//! Per-decode-window peak normalisation for the digital-mode decoders
//! (FT8, JS8).
//!
//! WSJT-X and js8call were tuned against "green" audio levels — the
//! operator guidance is "keep the signal in the green, just under the
//! red mark" (i.e. a peak in the top of the safe band, roughly
//! −1…−6 dBFS). Most of the decoders' ratio-based machinery
//! (percentile-normalised sync scores, relative baseline fits, LLR
//! mean/std) is invariant to a uniform scale, but a number of
//! **absolute** gates were calibrated for that input level: JS8's
//! `sync < 2.0` hard gate ([`super::js8::decode`]) and mfsk-core's
//! fixed-point auto-shift headroom cap are two that bite at very low
//! input level.
//!
//! The HL2 EP6 baseband produces a very quiet natural-units stream
//! (in-band voice/RMS is typically ~1e-5, ≈ −100 dBFS relative to
//! full-scale), far below the reference decoders' assumed operating
//! point. Without normalisation the decoders see signals near the
//! floor of their assumptions: absolute gates that were "just under
//! threshold on a quiet signal" are now "deeply under threshold", and
//! the marginal-SNR regime the reference was tuned for shifts down.
//!
//! This module rescales **every decode window** to [`PEAK_TARGET`]
//! (the top of the green band) in a single pass, off the demod hot
//! loop, before the decoders consume it. It is *not* an AGC — there is
//! no running state, no attack/release coupling to the monitor-audio
//! path, and no feedback from block to block. One `f32` scan + one
//! multiply pass is the whole price; that is a tiny fraction of any
//! decode it precedes (FFT + sync + FEC).
//!
//! The SSB voice path's own AGC (`demod::normalize_to_i16_with_agc`)
//! is a separate concern for human listening (RMS-targeted, with
//! asymmetric attack/release to minimise pumping on voice gaps) and is
//! untouched by this module.

/// The "green, just under red" level every reference decoder is tuned
/// against, as a fraction of full-scale (1.0 = 0 dBFS). 0.85 ≈
/// −1.4 dBFS, i.e. the upper boundary of the green zone in WSJT-X's
/// level meter and the "just under red" mark in the operator guidance.
///
/// Any value in [0.5, 0.95] is defensible; 0.85 maximises signal level
/// (hence SNR in the decoders) while leaving visible headroom in the
/// meter the operator is told to watch.
pub const PEAK_TARGET: f32 = 0.85;

/// A practical silence floor. Windows whose absolute peak is below
/// this are treated as silence and left untouched: the decoders see a
/// numerically-zero window either way, and amplifying a residue-only
/// window (e.g. `peak = 1e-30`) by an enormous factor produces
/// sub-normals in the fixed-point back-ends the decoders exercise.
const PEAK_FLOOR: f32 = 1e-9;

/// Scan `window` for its absolute peak and, if the peak is at or above
/// [`PEAK_FLOOR`], rescale the whole window to [`PEAK_TARGET`] in
/// place.
///
/// Returns the linear gain that was applied:
///
/// * `0.0` — the window was silent (peak < `PEAKFLOOR`) and was left
///   untouched; treat as "no signal, no adjustment".
/// * `PEAK_TARGET / peak` — otherwise. This was multiplied uniformly
///   into every sample, so the resulting window is exactly
///   [`PEAK_TARGET`] at its peak and unchanged in shape / relative
///   amplitudes / noise floor (up to the reference decoders' ratios).
///
/// The operation is **idempotent only up to the floor**: calling it
/// again on an already normalised window is a no-op (peak =
/// `PEAK_TARGET`, `g = 1`). Calling it on a *louder-than-target* input
/// (peak > `PEAK_TARGET`) brings it down to target — the reference
/// decoders do not like over-full input either (mfsk-core's
/// fixed-point auto-shift caps at +8 bits).
pub fn peak_normalize(window: &mut [f32]) -> f32 {
    if window.is_empty() {
        return 0.0;
    }
    let mut peak: f32 = 0.0;
    for v in window.iter() {
        let a = v.abs();
        if a > peak {
            peak = a;
        }
    }
    if peak < PEAK_FLOOR {
        return 0.0;
    }
    let g = PEAK_TARGET / peak;
    for v in window.iter_mut() {
        *v *= g;
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_window_is_silence() {
        let mut w: Vec<f32> = Vec::new();
        let g = peak_normalize(&mut w);
        assert_eq!(g, 0.0);
    }

    #[test]
    fn silence_is_left_alone() {
        let mut w = vec![0.0f32; 1024];
        let g = peak_normalize(&mut w);
        assert_eq!(g, 0.0);
        assert!(w.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn sub_floor_residue_is_left_alone() {
        let mut w = vec![1e-12f32; 1024];
        let g = peak_normalize(&mut w);
        assert_eq!(g, 0.0);
        assert!(w.iter().all(|&v| v == 1e-12f32));
    }

    #[test]
    fn quiet_signal_is_brought_to_target() {
        let mut w = vec![1e-5f32, -1e-5, 5e-6, -5e-6];
        let g = peak_normalize(&mut w);
        assert!((g - PEAK_TARGET / 1e-5).abs() < 1e-7, "g={g}");
        let mut peak: f32 = 0.0;
        for v in w.iter() {
            let a = v.abs();
            if a > peak {
                peak = a;
            }
        }
        assert!((peak - PEAK_TARGET).abs() < 1e-6, "peak={peak}");
        // Shape preserved: the ratio of max to a smaller sample is the
        // same as the input ratio (1e-5 : 5e-6 = 2:1 → 0.85 : 0.425).
        assert!((w[0].abs() / w[2].abs() - 2.0).abs() < 1e-3);
    }

    #[test]
    fn loud_signal_is_brought_down_to_target() {
        let mut w = vec![2.0f32, -2.0, 1.5, -1.5];
        let g = peak_normalize(&mut w);
        assert!((g - PEAK_TARGET / 2.0).abs() < 1e-6, "g={g}");
        assert!((w[0].abs() - PEAK_TARGET).abs() < 1e-6);
    }

    #[test]
    fn already_at_target_is_a_noop() {
        let mut w = vec![0.85f32, -0.85, 0.4, -0.4];
        let g = peak_normalize(&mut w);
        assert!((g - 1.0).abs() < 1e-6, "g={g}");
        assert!((w[0].abs() - PEAK_TARGET).abs() < 1e-6);
    }

    /// Normalising a known sine at amplitude `A` produces a window at
    /// peak `PEAK_TARGET` regardless of `A` (as long as `A` clears the
    /// silence floor). This is the property the decoders depend on:
    /// absolute gating thresholds see the same input level every time.
    #[test]
    fn sine_at_any_amplitude_landson_target() {
        for a in [1e-4f32, 1e-3, 1e-2, 0.1, 0.5, 1.0, 3.0] {
            let n = 12_000; // 1 s at 12 kHz
            let mut w = (0..n)
                .map(|i| a * (2.0 * std::f32::consts::PI * 1_500.0 * i as f32 / n as f32).sin())
                .collect::<Vec<_>>();
            let _ = peak_normalize(&mut w);
            let mut peak: f32 = 0.0;
            for v in w.iter() {
                let x = v.abs();
                if x > peak {
                    peak = x;
                }
            }
            assert!((peak - PEAK_TARGET).abs() < 1e-3, "a={a} peak={peak}");
        }
    }
}
