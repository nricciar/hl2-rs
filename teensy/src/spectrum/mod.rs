//! Spectrum pipeline for the panadapter / waterfall.
//!
//! Mirrors `api/src/hub.rs::run_spectral` + `api/src/spectrum.rs::display_mags_into`,
//! but `no_std` (uses `core` + `alloc` + libm via `num-traits`) and sized for
//! a small LCD — one 2048-complex-sample window, max-pool to 320 display bins
//! (full width, centred on the NCO).
//!
//! Data comes in one EP6 chunk at a time (63 I/Q pairs/chunk at
//! 96 kSps, `n_recv = 1`). Each full, non-overlapping 2048-sample window
//! produces a row, approximately every 21.3 ms at that rate.

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

/// Accumulator state.
pub struct Pipeline {
    fft: Fft,
    win: Vec<f32>,
    acc_re: Vec<f32>,
    acc_im: Vec<f32>,
    /// Reused FFT workspace: DC-removed/windowed samples, then FFT output.
    buf_re: Vec<f32>,
    buf_im: Vec<f32>,
    /// Last computed mags (320 display bins). This is what the renderer reads.
    mags: Vec<u16>,
    /// Wrapping count of completed FFT windows.
    frame_seq: u32,
}

impl Pipeline {
    /// Build the pipeline. The 1024 complex twiddles occupy 8 KiB.
    pub fn new() -> Result<Self, &'static str> {
        let fft = Fft::new(N_FFT)?;
        let mut win = [0.0f32; N_FFT];
        for (i, v) in win.iter_mut().enumerate() {
            *v = 0.5 * (1.0 - Float::cos(2.0 * f32::consts::PI * (i as f32 / N_FFT as f32)));
        }
        let win = win.to_vec();
        let buf_re = vec![0f32; N_FFT];
        let buf_im = vec![0f32; N_FFT];
        let mags = vec![0u16; BINS];
        Ok(Self {
            fft,
            win,
            acc_re: Vec::with_capacity(N_FFT),
            acc_im: Vec::with_capacity(N_FFT),
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

    /// Accumulate one complex I/Q sample, committing only full windows.
    pub fn push(&mut self, re: f32, im: f32) {
        self.acc_re.push(re);
        self.acc_im.push(im);
        if self.acc_re.len() == N_FFT {
            self.commit_frame();
        }
    }

    /// Consume the next full 2048-sample window.
    fn commit_frame(&mut self) {
        debug_assert_eq!(self.acc_re.len(), N_FFT);
        debug_assert_eq!(self.acc_im.len(), N_FFT);
        // Remove the unwindowed mean first; windowing DC would spread it
        // into adjacent bins that subtracting a post-window mean cannot fix.
        let re_mean = self.acc_re.iter().sum::<f32>() / N_FFT as f32;
        let im_mean = self.acc_im.iter().sum::<f32>() / N_FFT as f32;
        for i in 0..N_FFT {
            self.buf_re[i] = (self.acc_re[i] - re_mean) * self.win[i];
            self.buf_im[i] = (self.acc_im[i] - im_mean) * self.win[i];
        }
        self.acc_re.clear();
        self.acc_im.clear();

        self.fft.process(&mut self.buf_re, &mut self.buf_im);

        // Partition all FFT bins proportionally. Preserve the reversed
        // frequency orientation, with raw DC at display column BINS / 2.
        for d in 0..BINS {
            let lo = d * N_FFT / BINS;
            let hi = (d + 1) * N_FFT / BINS;
            let mut peak = 0f32;
            for j in lo..hi {
                let k = (N_FFT + N_FFT / 2 - j) % N_FFT;
                let s = self.buf_re[k] * self.buf_re[k] + self.buf_im[k] * self.buf_im[k];
                if s > peak {
                    peak = s;
                }
            }
            let mag = (peak.sqrt() * 65_535.0).min(65_535.0) as u16;
            self.mags[d] = mag;
        }
        self.frame_seq = self.frame_seq.wrapping_add(1);
    }

    /// The current mags (320 bins, `u16` 0..65535).
    pub fn mags(&self) -> &[u16] {
        &self.mags
    }

    /// Wrapping frame sequence (initially 0).
    pub fn frame_seq(&self) -> u32 {
        self.frame_seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32 as f32t;
    use num_traits::float::Float;

    // Amplitude keeps the unnormalized Hann-windowed FFT below saturation.
    #[test]
    fn peak_at_expected_display_bin() {
        let mut p = Pipeline::new().unwrap();
        // Fixed raw-bin / column pairs include both near-DC sides and bins
        // in the final 128 positions that floor-stride pooling discarded.
        for (b, target) in [
            (1021, 0),
            (512, 80),
            (3, 159),
            (-3, 160),
            (-512, 240),
            (-960, 310),
            (-1020, 319),
        ] {
            for i in 0..N_FFT {
                let ang = 2.0 * f32t::consts::PI * i as f32 * b as f32 / N_FFT as f32;
                p.push(0.00025 * Float::cos(ang), 0.00025 * Float::sin(ang));
            }
            let mags = p.mags();
            let peak = mags.iter().enumerate().max_by_key(|(_, m)| **m).unwrap();
            let (pi, pm) = peak;
            assert!(
                pi == target,
                "target {target} (raw bin {b}): peak at {pi}, mag {pm}"
            );
            let expected = 0.00025 * N_FFT as f32 / 2.0 * 65_535.0;
            assert!((*pm as f32 - expected).abs() < 8.0, "peak magnitude {pm}");
        }
    }

    #[test]
    fn dc_is_removed_before_windowing() {
        let mut p = Pipeline::new().unwrap();
        for _ in 0..N_FFT {
            p.push(0.125, -0.25);
        }
        assert!(p.mags().iter().all(|&m| m == 0));

        let mut clean = Pipeline::new().unwrap();
        for i in 0..N_FFT {
            let ang = 2.0 * f32t::consts::PI * i as f32 * 512.0 / N_FFT as f32;
            let re = 0.00025 * Float::cos(ang);
            let im = 0.00025 * Float::sin(ang);
            clean.push(re, im);
            p.push(re + 0.125, im - 0.25);
        }
        for (&actual, &expected) in p.mags().iter().zip(clean.mags()) {
            assert!(actual.abs_diff(expected) <= 16, "{actual} vs {expected}");
        }
    }

    #[test]
    fn only_full_windows_commit_and_sequence_wraps() {
        let mut p = Pipeline::new().unwrap();
        assert_eq!(p.mags().len(), BINS);
        for _ in 0..N_FFT - 1 {
            p.push(0.0, 0.0);
        }
        assert_eq!(p.frame_seq(), 0);
        assert_eq!(p.len(), N_FFT - 1);
        p.push(0.0, 0.0);
        assert_eq!(p.frame_seq(), 1);
        assert_eq!(p.len(), 0);
        p.frame_seq = u32::MAX;
        for _ in 0..N_FFT {
            p.push(0.0, 0.0);
        }
        assert_eq!(p.frame_seq(), 0);
        assert_eq!(p.len(), 0);
    }
}
