//! "FFT output → display magnitudes" step for the panadapter / waterfall
//!
//! The heavy lifting (accumulating I/Q, Hann-windowing, DC-blocking and the
//! FFT itself) lives in `hub.rs::run_spectral`; this module is only the layout
//! of an already-computed forward FFT into the display bins.

use num_complex::Complex;

/// Reorder and max-pool a raw forward-FFT output for a centered panadapter.
pub fn display_mags_into(x: &[Complex<f32>], bins: usize, out: &mut Vec<u16>) {
    let n = x.len();
    let target = bins.min(n);
    out.clear();
    if n == 0 || target == 0 {
        return;
    }
    let nyq = n / 2;
    let stride = (n / target).max(1);
    out.reserve(n.min(target));
    for d in 0..target {
        // Display slice k ∈ [lo, hi) maps — under `bin = (nyq − k) mod n` — to
        // the cyclic bin interval {A, A−1, …, A−(h−l)} (h−l bins, going
        // downwards from A), i.e. either one contiguous range or two ranges
        // wrapping past 0. Max-pool it on squared magnitudes (exact for
        // argmax), then take the square root once.
        let lo = d * stride;
        let hi = ((d + 1) * stride).min(n);
        let span = hi - lo; // ≥ 1
        // The span display indices k ∈ [lo, hi) map, under
        // `bin = (nyq − k) mod n`, to the span distinct bins
        // `{(nyq−lo) mod n, (nyq−lo−1) mod n, …, (nyq−lo−span+1) mod n}` —
        // span consecutive raw bins counting downwards. Max-pool them on
        // squared magnitudes (exact for argmax), then take the square root.
        let start = (nyq as i64 - lo as i64).rem_euclid(n as i64) as usize;
        let mut peak = 0.0f32;
        for j in 0..span {
            let k = (start.wrapping_sub(j)).rem_euclid(n);
            let v = x[k];
            let s = v.re * v.re + v.im * v.im;
            if s > peak {
                peak = s;
            }
        }
        out.push((peak.sqrt() * 65535.0_f32).min(65535.0_f32) as u16);
    }
}

/// Reorder and max-pool a raw forward-FFT output for a centered panadapter.
#[cfg(test)]
pub fn display_mags(x: &[Complex<f32>], bins: usize) -> Vec<u16> {
    let mut out = Vec::new();
    display_mags_into(x, bins, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peak_index(mags: &[u16]) -> usize {
        (0..mags.len()).max_by_key(|&i| mags[i]).unwrap()
    }

    fn single_bin(n: usize, bin: usize) -> Vec<Complex<f32>> {
        let mut x = vec![Complex::new(0.0, 0.0); n];
        x[bin] = Complex::new(1.0, 0.0);
        x
    }

    /// DC (bin 0) must land on the centre display bin
    #[test]
    fn dc_sits_at_centre() {
        let n = 1024usize;
        let mags = display_mags(&single_bin(n, 0), 128);
        assert_eq!(mags.len(), 128);
        assert_eq!(peak_index(&mags), mags.len() / 2, "DC must be centred");
    }

    #[test]
    fn below_ncs_is_on_left() {
        let n = 1024usize;
        let x = single_bin(n, n / 8); // +1/8 of the Nyquist, positive
        let mags = display_mags(&x, 128);
        let c = peak_index(&mags);
        assert!(
            c < mags.len() / 2,
            "below-NCO baseband must be left of centre; got {c} of {}",
            mags.len()
        );
    }

    #[test]
    fn above_ncs_is_on_right() {
        let n = 1024usize;
        let x = single_bin(n, n - n / 8); // −1/8 (bin index n − n/8)
        let mags = display_mags(&x, 128);
        let c = peak_index(&mags);
        assert!(
            c > mags.len() / 2,
            "above-NCO baseband must be right of centre; got {c} of {}",
            mags.len()
        );
    }

    #[test]
    fn far_below_is_far_left() {
        let n = 1024usize;
        let x = single_bin(n, n / 2 - 1);
        let mags = display_mags(&x, 128);
        let c = peak_index(&mags);
        assert!(
            c < mags.len() / 4,
            "most-positive baseband must be far left; got {c} of {}",
            mags.len()
        );
    }

    #[test]
    fn both_halves_visible_symmetric() {
        let n = 1024usize;
        let mut x = vec![Complex::new(0.0, 0.0); n];
        x[n / 8] = Complex::new(1.0, 0.0); // below, left
        x[n - n / 8] = Complex::new(1.0, 0.0); // above, right
        let mags = display_mags(&x, 128);
        let c = mags.len() / 2;
        let left = peak_index(&mags[..c]);
        let right = peak_index(&mags[c..]) + c;
        assert!(left < c && right > c, "peaks must straddle the centre");
        // Symmetric about the centre (both tones at ±n/8).
        assert_eq!(left as i64, 2 * c as i64 - right as i64, "peaks symmetric");
    }

    #[test]
    fn into_matches_owned_and_reuses() {
        let n = 1024usize;
        let x = single_bin(n, n / 8);
        let mut out = vec![0u16; 4096];
        display_mags_into(&x, 128, &mut out);
        assert_eq!(out, display_mags(&x, 128));
        let y = single_bin(n, 3);
        display_mags_into(&y, 64, &mut out);
        assert_eq!(out.len(), 64);
        assert_eq!(out, display_mags(&y, 64));
    }

    #[test]
    fn bins_gt_n_clamps() {
        let n = 64usize;
        let x = single_bin(n, n / 4);
        let mags = display_mags(&x, 256);
        assert_eq!(mags.len(), n);
        assert_eq!(peak_index(&mags), n / 4);
    }

    #[test]
    fn pooled_matches_naive_reference() {
        for &(n, bins) in &[
            (64usize, 8),
            (64, 16),
            (512, 1024),
            (512, 256),
            (512, 64),
            (100, 7),
        ] {
            // Deterministic "pseudorandom" magnitudes in both halves.
            let mut x: Vec<Complex<f32>> = (0..n)
                .map(|i| {
                    let a = (i as u32).wrapping_mul(2654435761);
                    let r = (a >> 8) as f32 / 255.0;
                    let s = ((a >> 16) ^ (a >> 4)) as f32 / 4095.0;
                    let m = r * s;
                    let a2 = a.wrapping_mul(40503);
                    let ph = ((a2 >> 10) as u32 % 62832) as f32 / 10000.0;
                    let m = m.max(0.25);
                    Complex::new(m * ph.cos(), m * ph.sin())
                })
                .collect();
            let mags = display_mags(&x, bins);
            let target = bins.min(n);
            assert_eq!(mags.len(), target);

            // Reference per the public docs: bin `d` = max of |x[k]| over the
            // stride-wide block centred on raw bin (nyq − d) mod n.
            let stride = (n / target).max(1);
            let want = (0..target).map(|d| {
                let lo = d * stride;
                let hi = ((d + 1) * stride).min(n);
                let mut peak = 0.0f64;
                for dd in lo..hi {
                    let idx = ((n / 2) as i64 - dd as i64).rem_euclid(n as i64) as usize;
                    let v = x[idx];
                    let m = ((v.re * v.re + v.im * v.im) as f64).sqrt();
                    if m > peak {
                        peak = m;
                    }
                }
                (peak * 65535.0).min(65535.0) as u16
            });
            let want: Vec<u16> = want.collect();
            assert_eq!(mags, want, "n={n} bins={bins}");
        }
    }
}
