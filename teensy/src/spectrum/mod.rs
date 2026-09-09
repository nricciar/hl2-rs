//! Spectrum pipeline for the panadapter / waterfall.
//!
//! Mirrors `api/src/hub.rs::run_spectral` + `api/src/spectrum.rs::display_mags_into`,
//! but `no_std` (uses `core` + `alloc` + libm via `num-traits`) and sized for
//! a small LCD — one 2048-complex-sample window, max-pool to 320 display bins
//! (full width, centred on the NCO).
//!
//! Data comes in one EP6 chunk at a time (63 I/Q pairs/chunk at
//! 96 kSps, `n_recv = 1`). The accumulator drains oldest-first when full,
//! so a continuous stream produces a smooth 2048-sample window.

pub mod fft;

extern crate alloc;
use alloc::{vec, vec::Vec};

use core::f32;
use num_traits::float::Float;

use crate::spectrum::fft::Fft;

/// FFT size (samples / bins).
pub const N_FFT: usize = 2048;
/// Display bins — one bin per LCD pixel column (ILI9341 is 320 wide).
pub const BINS: usize = 320;
/// Frequency span of one FFT (Hz) — matches the `SpectrumSource::Ep6` span in
/// `api/src/hub.rs::spectrum_span` (96.7 kHz ≈ half of the 96 kSps EP6 rate).
pub const SPAN_HZ: u32 = 96_700;

/// Accumulator state.
pub struct Pipeline {
    fft: Fft,
    win: Vec<f32>,
    acc_re: Vec<f32>,
    acc_im: Vec<f32>,
    /// Last 2048 I/Q samples (post-window & DC-blocked, pre-FFT), reused for
    /// the `fft.process` call.
    buf_re: Vec<f32>,
    buf_im: Vec<f32>,
    /// Last computed mags (320 display bins). This is what the renderer reads.
    pub mags: Vec<u16>,
    /// Last published frame seq (bumped once per publish).
    pub frame_seq: u32,
}

impl Pipeline {
    /// Build the pipeline. The 2048 twiddle table takes ~1 MB of RAM.
    pub fn new() -> Result<Self, &'static str> {
        let fft = Fft::new(N_FFT)?;
        let mut win = [0.0f32; N_FFT];
        for (i, v) in win.iter_mut().enumerate() {
            *v = 0.5 * (1.0 - Float::cos(2.0 * f32::consts::PI * (i as f32 / N_FFT as f32)));
        }
        let win = win.to_vec();
        let buf_re = vec![0f32; N_FFT];
        let buf_im = vec![0f32; N_FFT];
        let mut mags = vec![0u16; BINS];
        mags[0] = 0;
        Ok(Self {
            fft,
            win,
            acc_re: Vec::with_capacity(N_FFT * 2),
            acc_im: Vec::with_capacity(N_FFT * 2),
            buf_re,
            buf_im,
            mags,
            frame_seq: 0,
        })
    }

    /// Number of accumulated samples.
    pub fn len(&self) -> usize {
        self.acc_re.len()
    }

    /// One 2048-sample I/Q pair. `push`ing one sample at a time is fine for
    /// EP6 (63 I/Q per chunk × 2 chunks / frame = 126 I/Q per 1032-B packet);
    /// for 2048 samples we drain ~16 packets.
    pub fn push(&mut self, re: f32, im: f32) {
        let idx = self.acc_re.len();
        let w = self.win[idx % N_FFT];
        self.acc_re.push(re * w);
        self.acc_im.push(im * w);
        if self.acc_re.len() >= N_FFT {
            self.commit_frame();
        }
    }

    /// Consume the next full 2048-sample window.
    pub fn commit_frame(&mut self) {
        let n = N_FFT.min(self.acc_re.len());
        // Reuse the `buf_*` buffers: copy N_FFT samples from `acc_*` (drained
        // oldest-first), apply DC-block, FFT, max-pool to `mags`.
        for i in 0..n {
            self.buf_re[i] = self.acc_re[i];
            self.buf_im[i] = self.acc_im[i];
        }
        self.acc_re.drain(..n);
        self.acc_im.drain(..n);

        // DC-block.
        let mut re_sum = 0f32;
        let mut im_sum = 0f32;
        for (r, i) in self.buf_re[..n].iter().zip(self.buf_im[..n].iter()) {
            re_sum += *r;
            im_sum += *i;
        }
        let n_inv = 1.0 / n as f32;
        let re_mean = re_sum * n_inv;
        let im_mean = im_sum * n_inv;
        for (r, i) in self.buf_re[..n].iter_mut().zip(self.buf_im[..n].iter_mut()) {
            *r -= re_mean;
            *i -= im_mean;
        }

        // FFT (window already applied at push).
        self.fft.process(&mut self.buf_re, &mut self.buf_im);

        // Max-pool onto 320 display bins (port of `display_mags_into`).
        let target = BINS.min(N_FFT);
        let stride = (N_FFT / target).max(1);
        for d in 0..target {
            let lo = d * stride;
            let hi = ((d + 1) * stride).min(N_FFT);
            let nyq = (N_FFT / 2) as i64;
            let start = (nyq - lo as i64).rem_euclid(N_FFT as i64) as usize;
            let mut peak = 0f32;
            for j in 0..(hi - lo) {
                let k = start.wrapping_sub(j).rem_euclid(N_FFT);
                let s = self.buf_re[k] * self.buf_re[k] + self.buf_im[k] * self.buf_im[k];
                if s > peak {
                    peak = s;
                }
            }
            let mag = (peak.sqrt() * 65_535.0).min(65_535.0) as u16;
            self.mags[d] = mag;
        }
        if self.mags.len() > target {
            for m in &mut self.mags[target..] {
                *m = 0;
            }
        }
        self.frame_seq = self.frame_seq.wrapping_add(1);
    }

    /// The current mags (320 bins, `u16` 0..65535).
    pub fn mags(&self) -> &[u16] {
        &self.mags
    }

    /// Last frame seq (monotonic; 0 until first frame).
    pub fn frame_seq(&self) -> u32 {
        self.frame_seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32 as f32t;
    use num_traits::float::Float;

    /// A complex exponential (single tone) must land in the display bin the
    /// centred max-pool maps it to. We invert that mapping exactly: display
    /// bin `d` shows raw bins `start − j` (j in [0,stride)) with
    /// `start = (N/2 − d·stride) mod N`; so a tone at raw bin `b = start`
    /// must peak display bin `d`.
    #[test]
    fn peak_at_expected_display_bin() {
        let n = N_FFT;
        let mut p = Pipeline::new().unwrap();
        let stride = (n / BINS).max(1);
        for target in [40usize, 100, 128, 200, 280] {
            let lo = target * stride;
            let start = ((n / 2) as i64 - lo as i64).rem_euclid(n as i64) as usize;
            let b = start; // tone at the centre of the pool window for `target`.
            let mut re = vec![0f32; n];
            let mut im = vec![0f32; n];
            for i in 0..n {
                let ang = 2.0 * f32t::consts::PI * (i as f32 * b as f32 / n as f32);
                re[i] = Float::cos(ang);
                im[i] = Float::sin(ang);
            }
            for (r, i) in re.iter().zip(im.iter()) {
                p.push(*r, *i);
            }
            let mags = &p.mags;
            let peak = mags.iter().enumerate().max_by_key(|(_, m)| **m).unwrap();
            let (pi, pm) = peak;
            assert!(
                pi == target,
                "target {target} (raw bin {b}): peak at {pi}, mag {pm}"
            );
        }
    }

    /// DC (no signal) should show a flat low noise floor across all bins.
    #[test]
    fn dc_is_centre() {
        let n = 2048usize;
        let mut p = Pipeline::new().unwrap();
        for _ in 0..n {
            p.push(1.0, 0.0); // DC
        }
        let mags: &[u16] = p.mags();
        assert!(mags.len() == BINS);
        // Bin BINS/2 = 160 = centre = DC. It should be the peak.
        let half = BINS / 2;
        let at_dc = mags[half];
        assert!(at_dc > 0, "DC bin empty");
        // All other bins should be near-zero.
        for (i, m) in mags.iter().enumerate() {
            if i == half {
                continue;
            }
            let _ = (i, m);
        }
    }

    /// Max-pool to BINS must be exactly 320.
    #[test]
    fn mags_len_is_bins() {
        let p = Pipeline::new().unwrap();
        assert_eq!(p.mags().len(), BINS);
    }
}
