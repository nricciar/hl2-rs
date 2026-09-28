//! Spectrum pipeline for the panadapter / waterfall.
//!
//! Mirrors `api/src/hub.rs::run_spectral` + `api/src/spectrum.rs::display_mags_into`,
//! but `no_std` (uses `core` + `alloc` + libm via `num-traits`) and sized for
//! a small LCD — one 2048-complex-sample window (zero-padded to 4096 before
//! the FFT) → 320-column display bins (a centred slice of the band, centred
//! on the NCO).
//!
//! The display shows a configurable centred slice of the 96 kHz EP6 band —
//! the default is the middle 10 kHz ([`WF_BAND_HZ`]): 5 kHz either side of
//! the NCO. The NCO stays at display column `BINS / 2`; one display column
//! spans `WF_BAND_HZ / BINS` Hz (31.25 Hz at the default). Widen the view by
//! raising [`WF_BAND_HZ`] (capped at the sample rate, where the old full-band
//! behaviour resumes).
//!
//! Zero-padding the windowed 2048-sample signal to 4096 before the FFT makes
//! the DFT a 2×-densely-sampled (sinc-interpolated) version of the same
//! frequency range *without* changing the analysis window — so each
//! display column samples a raw, analytically correct bin instead of
//! linearly interpolating between sparser ones. The true frequency
//! resolution is still set by the 2048-sample Hann window (96 kHz / 2048
//! ≈ 47 Hz); the padding just makes the waterfall shape between carriers
//! smooth.
//!
//! Data comes in one EP6 chunk at a time (63 I/Q pairs/chunk at
//! 96 kSps, `n_recv = 1`). Each full, non-overlapping 2048-sample window
//! produces a row, approximately every 21.3 ms at that rate.

pub mod fft;

extern crate alloc;
use alloc::{vec, vec::Vec};

use core::f32;
use core::sync::atomic::{AtomicUsize, Ordering};
use num_traits::float::Float;

use crate::spectrum::fft::Fft;

/// Length of the *analysis window* (real I/Q samples, Hann-windowed) that
/// is zero-padded to [`N_FFT`] before the FFT. This is what sets the
/// measureable frequency resolution (~47 Hz at 96 kHz / 2048).
pub const WIN_LEN: usize = 2048;
/// Padded FFT length. The windowed `WIN_LEN`-sample signal is
/// zero-padded to this length and FFT'ed, so the display can read
/// raw bins that are a `N_FFT / WIN_LEN`×-densely-sampled (sinc-interpolated)
/// version of the true spectrum *without* changing the resolution.
pub const N_FFT: usize = 4096;
/// Display bins — one bin per LCD pixel column (ILI9341 is 320 wide).
pub const BINS: usize = 320;
/// Complex I/Q sample rate feeding the FFT (Hz) — one EP6 sample per tick at
/// the DDC's 96 kSps setting.
pub const SAMPLE_RATE_HZ: usize = 96_000;
/// Width of the centred spectrum slice the waterfall displays (Hz),
/// symmetric about the NCO (5 kHz either side at the default). Clamp this to
/// [`SAMPLE_RATE_HZ`]; the display always spans exactly `BINS` columns.
pub const WF_BAND_HZ: usize = 24_000;

/// Diagnostic: current step inside `Pipeline::new()`. Read by the render
/// task (which keeps running as long as the RTIC scheduler preempts the
/// blocked *radio* task) so we can find out *which* step of `new` is
/// hung. Steps, in order:
///
///   0 — entered, before `Fft::new` (2048 cos/sin — software
///       `num-traits/libm` trig).
///   1 — `Fft::new` returned.
///   2 — Hann window computed (2048 `Float::cos`, then `to_vec`).
///   3 — heap `buf_re` / `buf_im` (32 KB total) allocated + zeroed.
///   4 — heap `mags` allocated.
///   5 — `acc_re` / `acc_im` created (`Vec::with_capacity(WIN_LEN)`).
///   6 — `Ok(Self { .. })` returned.
///
/// Any step that stays constant across render-heartbeat polls is the
/// step the hang is stuck on.
/// Initial value is 6 (= "not currently building / already done"), so the
/// render-task heartbeat stays quiet at boot.
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
    /// Build the pipeline. The 1024 complex twiddles occupy 8 KiB.
    ///
    /// The Hann-window table is allocated on the heap directly (as
    /// a [`Vec<f32>`]) so the caller's stack never has to hold a
    /// full 8 KB local — under RTIC 2.x all tasks share the 16 KB
    /// main stack, and a synchronous function that peaks at ~8 KB
    /// stack *on top of* three pre-allocated task-future frames is
    /// already close to the stack limit. A 1-2 KB overrun here is
    /// silent (no guard pages on `thumbv7em-none`) and would
    /// corrupt a neighbouring frame and hang the core with no
    /// panic blink.
    pub fn new() -> Result<Self, &'static str> {
        PIPELINE_NEW_STEP.store(0, Ordering::Relaxed); // entered, before Fft::new
        let fft = Fft::new(N_FFT)?;
        PIPELINE_NEW_STEP.store(1, Ordering::Relaxed); // Fft::new done
        // Build the Hann window into a pre-sized heap buffer — zero
        // stack beyond this single `Vec` (24 bytes).
        let mut win = vec![0f32; WIN_LEN];
        for (i, v) in win.iter_mut().enumerate() {
            *v = 0.5 * (1.0 - Float::cos(2.0 * f32::consts::PI * (i as f32 / WIN_LEN as f32)));
        }
        PIPELINE_NEW_STEP.store(2, Ordering::Relaxed); // window done
        let buf_re = vec![0f32; N_FFT];
        let buf_im = vec![0f32; N_FFT];
        PIPELINE_NEW_STEP.store(3, Ordering::Relaxed); // buf_re/im done
        let mags = vec![0u16; BINS];
        PIPELINE_NEW_STEP.store(4, Ordering::Relaxed); // mags done
        let acc_re = Vec::with_capacity(WIN_LEN);
        let acc_im = Vec::with_capacity(WIN_LEN);
        PIPELINE_NEW_STEP.store(5, Ordering::Relaxed); // acc done; about to return
        PIPELINE_NEW_STEP.store(6, Ordering::Relaxed); // returned
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
        // Zero-paste: the rest of the FFT input must be 0+0j.
        for i in WIN_LEN..N_FFT {
            self.buf_re[i] = 0.0;
            self.buf_im[i] = 0.0;
        }

        self.fft.process(&mut self.buf_re, &mut self.buf_im);

        // Display the centred `WF_BAND_HZ` slice (5 kHz either side of the
        // NCO by default), keeping DC at column BINS/2 and the band's
        // frequency orientation (USB / complex-negative on the right).
        //
        // Column `d` spans the signed frequencies `[(BINS/2 - d - 0.5)
        // · band/BINS, (BINS/2 - d + 0.5) · band/BINS]` Hz, i.e.
        // `band · N_FFT / (Fs · BINS) ≈ 1.33` raw 4096-DFT bins wide. Each
        // column max-pools every raw bin whose *centre* falls inside that
        // range — with 2× zero-padding that is now always 1–2 bins, so the
        // waterfall samples raw, sinc-interpolated bins with no
        // interpolation artefacts between them.
        let band = WF_BAND_HZ.min(SAMPLE_RATE_HZ);
        let col_bins_f = band as f32 * N_FFT as f32 / (SAMPLE_RATE_HZ as f32 * BINS as f32);
        let half_b = BINS / 2;
        for d in 0..BINS {
            // Raw-bin centre at this column (signed bins; negative = USB).
            let k_c = (half_b as f32 - d as f32) * col_bins_f;
            let lo = (k_c - col_bins_f * 0.5).ceil() as i64;
            let hi = (k_c + col_bins_f * 0.5).floor() as i64;
            let mut best = 0f32;
            for k in lo..=hi {
                let idx = (k.rem_euclid(N_FFT as i64)) as usize;
                let re = self.buf_re[idx];
                let im = self.buf_im[idx];
                let m = re * re + im * im;
                if m > best {
                    best = m;
                }
            }
            let mag = best.sqrt();
            self.mags[d] = (mag * 65_535.0).min(65_535.0) as u16;
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
    fn peak_at_expected_display_column() {
        let mut p = Pipeline::new().unwrap();
        // Fixed raw-bin / expected-column pairs. The display spans the
        // central `WF_BAND_HZ` slice (±5 kHz by default) about the NCO, so
        // signed carrier bin `b` lands at column `d = BINS/2 - b·BINS /
        // (N_FFT·band / sample-rate)`. Carriers are generated at integer
        // 2048-DFT frequencies (i.e. exact multiples of the *true* 47 Hz
        // resolution); zero-padding to 4096 maps them to exact half-integer
        // 4096-bin positions, and each column max-pools the 1–2 raw bins
        // that cover its frequency window.
        let band = WF_BAND_HZ.min(SAMPLE_RATE_HZ) as f32;
        for b in [40i64, -40, 50, -50, 98, -98, 106, -106] {
            for i in 0..WIN_LEN {
                let ang = 2.0 * f32t::consts::PI * i as f32 * b as f32 / WIN_LEN as f32;
                p.push(0.00025 * Float::cos(ang), 0.00025 * Float::sin(ang));
            }
            let mags = p.mags();
            let peak = mags.iter().enumerate().max_by_key(|(_, m)| **m).unwrap();
            let (pi, pm) = peak;
            // The carrier sits at signed `b` bins of the 2048-sample window,
            // i.e. `SAMPLE_RATE_HZ · b / WIN_LEN` Hz; the display maps
            // that to column `BINS/2 − f · BINS / band`.
            let f_hz = SAMPLE_RATE_HZ as f32 * b as f32 / WIN_LEN as f32;
            let expected_col = (BINS / 2) as i32 - (f_hz * (BINS as f32) / band) as i32;
            assert!(
                (pi as i32 - expected_col).abs() <= 1,
                "signed bin {b} (expect col {}): peak at {pi}, mag {pm}",
                expected_col
            );
            assert!(
                (*pm as f32) > (0.00025 * N_FFT as f32 / 2.0 * 65_535.0) * 0.2,
                "peak magnitude {pm} too low for signed bin {b}"
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
            let ang = 2.0 * f32t::consts::PI * i as f32 * 512.0 / WIN_LEN as f32;
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
