//! Spectrum pipeline for the panadapter / waterfall.
//!
//! Each 2048-sample complex I/Q window produces a 320-column row every
//! 21.3 ms at 96 kSps. The view spans 24 kHz about the NCO: 75 Hz/column,
//! with DC at column 160 and USB (complex-negative frequencies) to the right.
//!
//! The mean is removed before Hann windowing. Zero-padding to 4096 samples
//! halves FFT-bin spacing to 23.4375 Hz but does not improve the resolution
//! of the 2048-sample window (46.875 Hz bin spacing before padding).
//!
//! Columns max-pool FFT magnitudes, falling back to the nearest bin for
//! narrow zooms. Output is unnormalized and clipped to u16 display scale;
//! it is not a calibrated power spectrum, and the DC carrier is suppressed.

pub mod fft;

extern crate alloc;
use alloc::{vec, vec::Vec};

use core::f32;
use core::sync::atomic::{AtomicUsize, Ordering};
use num_traits::float::Float;

use crate::spectrum::fft::Fft;

/// Complex samples in the Hann analysis window.
pub const WIN_LEN: usize = 2048;
/// Zero-padded FFT length; padding interpolates without adding resolution.
pub const N_FFT: usize = 4096;
/// One display bin per LCD column.
pub const BINS: usize = 320;
/// Complex I/Q sample rate (Hz).
pub const SAMPLE_RATE_HZ: usize = 96_000;
/// Validated display span (Hz), shared by pooling, meter and passband overlay.
/// The default is 24 kHz (nominally +/-12 kHz); zero or spans above Fs fail
/// at compile time rather than silently changing one consumer's geometry.
pub const WF_BAND_HZ: usize = {
    let span = 24_000;
    assert!(span > 0 && span <= SAMPLE_RATE_HZ);
    span
};

/// Constructor progress for render-task hang diagnostics: 0 entered,
/// 1 FFT ready, 2 Hann ready, 3 FFT buffers ready, 4 magnitudes ready,
/// 5 accumulators ready, 6 done (also the initial idle value).
pub static PIPELINE_NEW_STEP: AtomicUsize = AtomicUsize::new(6);

/// Render task / panic handler sample this to find the stuck step.
pub fn pipeline_new_step() -> usize {
    PIPELINE_NEW_STEP.load(Ordering::Acquire)
}

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
    /// Allocate tables and buffers directly on the heap; RTIC tasks share
    /// a 16 KiB stack. The 2048 complex twiddles alone occupy 16 KiB.
    pub fn new() -> Result<Self, &'static str> {
        PIPELINE_NEW_STEP.store(0, Ordering::Relaxed);
        let fft = Fft::new(N_FFT)?;
        PIPELINE_NEW_STEP.store(1, Ordering::Relaxed);
        let mut win = vec![0f32; WIN_LEN];
        for (i, v) in win.iter_mut().enumerate() {
            *v = 0.5 * (1.0 - Float::cos(2.0 * f32::consts::PI * (i as f32 / WIN_LEN as f32)));
        }
        PIPELINE_NEW_STEP.store(2, Ordering::Relaxed);
        let buf_re = vec![0f32; N_FFT];
        let buf_im = vec![0f32; N_FFT];
        PIPELINE_NEW_STEP.store(3, Ordering::Relaxed);
        let mags = vec![0u16; BINS];
        PIPELINE_NEW_STEP.store(4, Ordering::Relaxed);
        let acc_re = Vec::with_capacity(WIN_LEN);
        let acc_im = Vec::with_capacity(WIN_LEN);
        PIPELINE_NEW_STEP.store(5, Ordering::Relaxed);
        PIPELINE_NEW_STEP.store(6, Ordering::Relaxed);
        Ok(Self {
            fft,
            win,
            acc_re,
            acc_im,
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
        if self.acc_re.len() == WIN_LEN {
            self.commit_frame();
        }
    }

    /// Consume the next full 2048-sample window.
    fn commit_frame(&mut self) {
        debug_assert_eq!(self.acc_re.len(), WIN_LEN);
        debug_assert_eq!(self.acc_im.len(), WIN_LEN);
        // Remove the unwindowed mean first; windowing DC would spread it
        // into adjacent bins that subtracting a post-window mean cannot fix.
        let re_mean = self.acc_re.iter().sum::<f32>() / WIN_LEN as f32;
        let im_mean = self.acc_im.iter().sum::<f32>() / WIN_LEN as f32;
        for i in 0..WIN_LEN {
            self.buf_re[i] = (self.acc_re[i] - re_mean) * self.win[i];
            self.buf_im[i] = (self.acc_im[i] - im_mean) * self.win[i];
        }
        self.acc_re.clear();
        self.acc_im.clear();
        // Clear the previous FFT output from the zero-padding region.
        for i in WIN_LEN..N_FFT {
            self.buf_re[i] = 0.0;
            self.buf_im[i] = 0.0;
        }

        self.fft.process(&mut self.buf_re, &mut self.buf_im);

        self.pool_display(WF_BAND_HZ);
        self.frame_seq = self.frame_seq.wrapping_add(1);
    }

    fn pool_display(&mut self, band: usize) {
        debug_assert!(band > 0 && band <= SAMPLE_RATE_HZ);
        // Column centres are (BINS/2 - d) * band/BINS Hz. Include raw-bin
        // centres within half a column on either side (3-4 bins at 24 kHz).
        let col_bins_f = band as f32 * N_FFT as f32 / (SAMPLE_RATE_HZ as f32 * BINS as f32);
        let half_b = BINS / 2;
        for d in 0..BINS {
            let k_c = (half_b as f32 - d as f32) * col_bins_f;
            let mut lo = Float::ceil(k_c - col_bins_f * 0.5) as i32;
            let mut hi = Float::floor(k_c + col_bins_f * 0.5) as i32;
            // Below 7.5 kHz columns can contain no FFT-bin centre.
            if lo > hi {
                lo = Float::round(k_c) as i32;
                hi = lo;
            }
            let mut best = 0f32;
            for k in lo..=hi {
                let idx = k.rem_euclid(N_FFT as i32) as usize;
                let re = self.buf_re[idx];
                let im = self.buf_im[idx];
                let m = re * re + im * im;
                if m > best {
                    best = m;
                }
            }
            let mag = Float::sqrt(best);
            self.mags[d] = (mag * 65_535.0).min(65_535.0) as u16;
        }
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
    fn peak_at_expected_display_column() {
        let mut p = Pipeline::new().unwrap();
        // Coherent WIN_LEN-bin tones map to even padded FFT bins. Include
        // both near-DC sides and the first/last columns of the 24 kHz view.
        for (b, target) in [
            (256, 0),
            (254, 1),
            (40, 135),
            (3, 158),
            (-3, 162),
            (-40, 185),
            (-254, 319),
        ] {
            for i in 0..WIN_LEN {
                let ang = 2.0 * f32t::consts::PI * i as f32 * b as f32 / WIN_LEN as f32;
                p.push(0.00025 * Float::cos(ang), 0.00025 * Float::sin(ang));
            }
            let mags = p.mags();
            let peak = mags.iter().enumerate().max_by_key(|(_, m)| **m).unwrap();
            let (pi, pm) = peak;
            assert_eq!(pi, target, "signed bin {b}: magnitude {pm}");
            let expected = 0.00025 * WIN_LEN as f32 / 2.0 * 65_535.0;
            assert!(
                (*pm as f32 - expected).abs() < 8.0,
                "signed bin {b}: peak {pm}, expected {expected}"
            );
        }
    }

    #[test]
    fn dc_is_removed_before_windowing() {
        let mut p = Pipeline::new().unwrap();
        for _ in 0..WIN_LEN {
            p.push(0.125, -0.25);
        }
        assert!(p.mags().iter().all(|&m| m == 0));

        let mut clean = Pipeline::new().unwrap();
        for i in 0..WIN_LEN {
            let ang = 2.0 * f32t::consts::PI * i as f32 * 40.0 / WIN_LEN as f32;
            let re = 0.00025 * Float::cos(ang);
            let im = 0.00025 * Float::sin(ang);
            clean.push(re, im);
            p.push(re + 0.125, im - 0.25);
        }
        let expected_peak = 0.00025 * WIN_LEN as f32 / 2.0 * 65_535.0;
        assert!((clean.mags()[135] as f32 - expected_peak).abs() < 8.0);
        for (&actual, &expected) in p.mags().iter().zip(clean.mags()) {
            assert!(actual.abs_diff(expected) <= 16, "{actual} vs {expected}");
        }
    }

    #[test]
    fn narrow_zoom_uses_nearest_bin_for_empty_columns() {
        let mut p = Pipeline::new().unwrap();
        // A varying, nonzero spectrum makes both holes and wrong-bin
        // fallbacks observable, including wrapped negative frequencies.
        for (i, re) in p.buf_re.iter_mut().enumerate() {
            *re = (i + 1) as f32 / 8192.0;
        }
        for band in [1, 100, 3_000, 6_001, 7_100] {
            p.pool_display(band);
            let col_bins = band as f64 * N_FFT as f64 / (SAMPLE_RATE_HZ as f64 * BINS as f64);
            let mut empty_columns = 0;
            for (d, &mag) in p.mags().iter().enumerate() {
                assert!(mag > 0, "band {band}, column {d}");
                let centre = (BINS as f64 / 2.0 - d as f64) * col_bins;
                if (centre - col_bins / 2.0).ceil() > (centre + col_bins / 2.0).floor() {
                    empty_columns += 1;
                    let nearest = (centre.round() as i32).rem_euclid(N_FFT as i32) as usize;
                    let expected = (p.buf_re[nearest] * 65_535.0) as u16;
                    assert_eq!(mag, expected, "band {band}, column {d}");
                }
            }
            assert!(empty_columns > 0, "band {band} must exercise fallback");
        }
    }

    #[test]
    fn only_full_windows_commit_and_sequence_wraps() {
        let mut p = Pipeline::new().unwrap();
        assert_eq!(p.mags().len(), BINS);
        for _ in 0..WIN_LEN - 1 {
            p.push(0.0, 0.0);
        }
        assert_eq!(p.frame_seq(), 0);
        assert_eq!(p.len(), WIN_LEN - 1);
        p.push(0.0, 0.0);
        assert_eq!(p.frame_seq(), 1);
        assert_eq!(p.len(), 0);
        p.frame_seq = u32::MAX;
        for _ in 0..WIN_LEN {
            p.push(0.0, 0.0);
        }
        assert_eq!(p.frame_seq(), 0);
        assert_eq!(p.len(), 0);
    }
}
