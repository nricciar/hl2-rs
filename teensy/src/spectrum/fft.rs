//! Radix-2 in-place complex FFT for power-of-two sizes.
//!
//! `no_std` with an allocated twiddle table and libm via `num-traits`.
//! Pre-computes the twiddle-factor table once, so steady-state frames run
//! without any trig calls.

extern crate alloc;

use alloc::vec::Vec;
use num_traits::float::Float;

pub struct Fft {
    n: usize,
    tw: Vec<(f32, f32)>,
}

impl Fft {
    /// Build an FFT for a power-of-two size of at least four. Every stage
    /// uses twiddle indices below n/2; the final stage reaches n/2 - 1.
    pub fn new(n: usize) -> Result<Self, &'static str> {
        if n < 4 || n & (n - 1) != 0 {
            return Err("n must be a power of two ≥ 4");
        }
        let tw: Vec<(f32, f32)> = (0..n / 2)
            .map(|k| {
                let ang = -2.0_f32 * core::f32::consts::PI * (k as f32 / n as f32);
                (Float::cos(ang), Float::sin(ang))
            })
            .collect();
        Ok(Self { n, tw })
    }

    /// In-place, unnormalized forward DFT, returned in natural bin order.
    /// Panics unless both input slices have exactly the configured length.
    pub fn process(&self, re: &mut [f32], im: &mut [f32]) {
        let n = self.n;
        assert_eq!(re.len(), n, "re.len() must equal fft.n");
        assert_eq!(im.len(), n, "im.len() must equal fft.n");
        let tw = &self.tw;

        // Bit-reversal permutation.
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j &= !bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }

        // Butterfly stages, smallest to largest.
        let mut len = 2usize;
        while len <= n {
            let half = len / 2;
            let w_step = n / len;
            for start in (0..n).step_by(len) {
                let mut k = 0usize;
                let mut m = start;
                while m < start + half {
                    let (wr, wi) = tw[k];
                    let a_re = re[m + half];
                    let a_im = im[m + half];
                    let b_re = a_re * wr - a_im * wi;
                    let b_im = a_re * wi + a_im * wr;
                    re[m + half] = re[m] - b_re;
                    im[m + half] = im[m] - b_im;
                    re[m] += b_re;
                    im[m] += b_im;
                    m += 1;
                    k += w_step;
                }
            }
            len = len.saturating_mul(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::f32::consts;

    #[test]
    fn new_rejects_bad_n() {
        assert!(Fft::new(1).is_err());
        assert!(Fft::new(2).is_err());
        assert!(Fft::new(3).is_err());
        assert!(Fft::new(6).is_err());
        assert!(Fft::new(4).is_ok());
        assert!(Fft::new(2_048).is_ok());
    }

    /// A unit impulse in the *time* domain has a flat, unit-magnitude
    /// spectrum (its DFT is 1 at every bin). A clean correctness property
    /// that pairs with the complex-exponential (frequency-delta) test.
    #[test]
    fn time_impulse_has_flat_unit_spectrum() {
        let n = 128;
        let f = Fft::new(n).unwrap();
        for p in [0usize, 1, 7, 63, 100] {
            let mut re = vec![0f32; n];
            let mut im = vec![0f32; n];
            re[p] = 1.0;
            f.process(&mut re, &mut im);
            for k in 0..n {
                let mag = (re[k] * re[k] + im[k] * im[k]).sqrt();
                assert!(
                    (mag - 1.0).abs() < 1e-3,
                    "impulse p={p} bin {k}: mag={mag} (want 1)"
                );
            }
        }
    }

    /// A complex exponential at frequency bin `k` must be a delta in that bin.
    #[test]
    fn complex_exponential_maps_to_its_bin() {
        let n = 256;
        let f = Fft::new(n).unwrap();
        for tone in [1usize, 32, 127] {
            let mut re = vec![0f32; n];
            let mut im = vec![0f32; n];
            for i in 0..n {
                let ang = 2.0 * consts::PI * (i as f32 * tone as f32 / n as f32);
                re[i] = Float::cos(ang);
                im[i] = Float::sin(ang);
            }
            f.process(&mut re, &mut im);
            let mag_at = |k: usize| -> f32 { (re[k] * re[k] + im[k] * im[k]).sqrt() };
            // bin `tone` is n; bin `n - tone` is its conjugate for a real tone,
            // 0 here since we used a complex (one-sided) exponential.
            assert!(
                (mag_at(tone) - n as f32).abs() < 1e-2,
                "peak at {tone}: got {} want {}",
                mag_at(tone),
                n
            );
            assert!(mag_at(0) < 1e-2, "DC leaked: {}", mag_at(0));
        }
    }

    /// Parseval's theorem: ∑|x|² = (1/N) ∑|X|².
    #[test]
    fn parseval_holds() {
        let n = 256;
        let f = Fft::new(n).unwrap();
        let mut re = vec![0f32; n];
        let mut im = vec![0f32; n];
        for (i, v) in re.iter_mut().enumerate() {
            *v = (i as f32 * 0.37).sin();
            im[i] = (i as f32 * 0.11).cos();
        }
        let before: f32 = re.iter().zip(im.iter()).map(|(a, b)| a * a + b * b).sum();
        f.process(&mut re, &mut im);
        let after: f32 = re
            .iter()
            .zip(im.iter())
            .map(|(a, b)| (a * a + b * b) / n as f32)
            .sum();
        assert!(
            (before - after).abs() < 1e-2 * before.max(1.0),
            "parseval violated: before={before} after={after}"
        );
    }

    #[test]
    fn new_fft_matches_dft_for_small_n() {
        let n = 16;
        let f = Fft::new(n).unwrap();
        // Deterministic "random-ish" sequence, but simple to reproduce.
        let mut x_re = vec![0f32; n];
        let mut x_im = vec![0f32; n];
        for (i, v) in x_re.iter_mut().enumerate() {
            *v = ((i * 7) % 13) as f32 - 6.0;
            x_im[i] = ((i * 3) % 11) as f32 - 5.0;
        }
        let mut y_re = x_re.clone();
        let mut y_im = x_im.clone();
        f.process(&mut y_re, &mut y_im);

        // Naive DFT (O(n²)).
        for k in 0..n {
            let mut sum_re = 0f32;
            let mut sum_im = 0f32;
            for j in 0..n {
                let ang = -2.0 * consts::PI * (j as f32 * k as f32 / n as f32);
                sum_re += x_re[j] * Float::cos(ang) - x_im[j] * Float::sin(ang);
                sum_im += x_re[j] * Float::sin(ang) + x_im[j] * Float::cos(ang);
            }
            let d_re = (y_re[k] - sum_re).abs();
            let d_im = (y_im[k] - sum_im).abs();
            assert!(d_re < 1e-2 && d_im < 1e-2, "bin {k}: diff ({d_re}, {d_im})");
        }
    }

    #[test]
    #[should_panic(expected = "re.len() must equal fft.n")]
    fn rejects_short_real_input() {
        Fft::new(4).unwrap().process(&mut [0.0; 3], &mut [0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "im.len() must equal fft.n")]
    fn rejects_long_imaginary_input() {
        Fft::new(4).unwrap().process(&mut [0.0; 4], &mut [0.0; 5]);
    }
}
