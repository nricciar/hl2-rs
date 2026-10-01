//! Display-derived S-meter: passband RMS relative to a display-row percentile.
//!
//! This is a clipped estimate, not absolute RF strength or full-band power.
//! The waterfall supplies max-pooled, u16-saturated magnitudes from the
//! centred 24 kHz view (75 Hz/column). DC is suppressed before the FFT, so
//! a carrier at the NCO is removed. Wide passbands are clipped to the view;
//! zoom, pooling and saturation can all change the reading.
//!
//! Level and floor use display full scale (65535) as their dB reference.
//! Their nonnegative difference maps to S0..S9 at 6 dB per step. The exact
//! percentile uses one stack histogram, with no heap allocation or sorting.

use core::f32;
use num_traits::float::Float;

use crate::spectrum::{BINS, WF_BAND_HZ};

/// USB SSB default channel-select bandwidth (Hz).
pub const USB_PASSBAND_HZ: u32 = 2_600;
/// Hz per display column, without integer-Hz truncation.
pub const DISPLAY_BIN_HZ: f32 = WF_BAND_HZ as f32 / BINS as f32;
/// USB passband extent from the NCO, rounded up to whole columns.
pub const USB_PASSBAND_BINS: usize = passband_bins(USB_PASSBAND_HZ, BINS);
/// Display noise-floor percentile; selects zero-based sorted rank `n * 25 / 100`.
pub const FLOOR_PERCENTILE: usize = 25;

/// The S margin (dB over the band floor) at which the meter first reads S1.
pub const S1_DB_OVER_FLOOR: f32 = 6.0;
/// dB per S-unit step (S1..S9, the classic 6-dB S-unit spacing).
pub const DB_PER_UNIT: f32 = 6.0;
/// S9 — the top of the scale.
pub const S9: u8 = 9;

/// One S-meter reading for a spectrum frame.
pub struct Smeter {
    /// Visible passband RMS, dB relative to display full scale (65535).
    pub level_db: f32,
    /// Display-row percentile, dB relative to display full scale.
    pub floor_db: f32,
    /// Nonnegative `level_db - floor_db`.
    pub margin_db: f32,
    /// The S0..S9 index derived from `margin_db` (`S9` = [`S9`], 0 = noise).
    pub sunits: u8,
}

impl Smeter {
    /// The S0..S9 index (0 = below S1 / noise floor, [`S9`] = pegged full).
    pub fn sunits(&self) -> u8 {
        self.sunits
    }

    /// The raw dB-over-floor margin (what the render shows as "+N dB").
    pub fn margin_db(&self) -> f32 {
        self.margin_db
    }
}

/// Convert a display magnitude to dB; values <= 1 map to -120 dB.
fn mag_to_db(m: u16) -> f32 {
    let v = f32::from(m);
    if v <= 1.0 {
        return -120.0;
    }
    (20.0 * Float::log10(v / 65_535.0)).clamp(-120.0, 0.0)
}

/// Map a signal-over-noise margin (dB over the band floor) to an S0..S9 index.
///
/// `S1` at [`S1_DB_OVER_FLOOR`] over the floor, [`DB_PER_UNIT`] per step, so
/// S9 lands at 54 dB over the floor. Below S1 or non-finite reads zero.
pub fn margin_to_sunits(margin_db: f32) -> u8 {
    if margin_db < S1_DB_OVER_FLOOR || !margin_db.is_finite() {
        return 0;
    }
    let steps = Float::floor((margin_db - S1_DB_OVER_FLOOR) / DB_PER_UNIT);
    (1u32 + steps as u32).clamp(1, S9 as u32) as u8
}

/// Compute the USB (mode 0) reading from a centred display row.
pub fn compute(mags: &[u16]) -> Smeter {
    compute_for_mode(mags, 0)
}

/// Compute a display-derived reading for a valid `crate::mode::MODES` index.
/// The row spans [`WF_BAND_HZ`], with the NCO at `len / 2`. SSB uses one
/// side; AM/FM/NFM use the default bandwidth on each side, clipped to the
/// display. The floor uses the entire row regardless of mode.
pub fn compute_for_mode(mags: &[u16], mode_index: usize) -> Smeter {
    let n = mags.len();

    if n == 0 {
        return Smeter {
            level_db: -120.0,
            floor_db: -120.0,
            margin_db: 0.0,
            sunits: 0,
        };
    }
    let (lo, hi) = passband_window(n, mode_index);
    let mut sum_sq = 0.0f32;
    for &m in &mags[lo..=hi] {
        let v = f32::from(m);
        sum_sq += v * v;
    }
    let rms = Float::sqrt(sum_sq / (hi - lo + 1) as f32);
    let level_db = if rms <= 1.0 {
        -120.0
    } else {
        (20.0 * Float::log10(rms / 65_535.0)).clamp(-120.0, 0.0)
    };

    // Radix selection avoids sort code in ITCM. Reuse the histogram for
    // the low byte of the selected high-byte bucket to retain weak signals.
    let mut hist = [0usize; 256];
    for &m in mags {
        hist[m as usize >> 8] += 1;
    }
    let mut rank = n * FLOOR_PERCENTILE / 100;
    let mut high = 0u16;
    for (byte, &count) in hist.iter().enumerate() {
        if rank < count {
            high = byte as u16;
            break;
        }
        rank -= count;
    }
    hist.fill(0);
    for &m in mags {
        if m >> 8 == high {
            hist[(m & 0xff) as usize] += 1;
        }
    }
    let mut floor = high << 8;
    for (byte, &count) in hist.iter().enumerate() {
        if rank < count {
            floor |= byte as u16;
            break;
        }
        rank -= count;
    }
    let floor_db = mag_to_db(floor);

    let margin_db = (level_db - floor_db).max(0.0);
    let sunits = margin_to_sunits(margin_db);
    Smeter {
        level_db,
        floor_db,
        margin_db,
        sunits,
    }
}

/// Round the bandwidth/visible-span ratio once, at column precision.
#[inline]
const fn passband_bins(bw_hz: u32, columns: usize) -> usize {
    (bw_hz as usize * columns).div_ceil(WF_BAND_HZ)
}

/// Inclusive column bounds; complex-negative frequencies (USB) are right.
fn passband_window(n: usize, mode_index: usize) -> (usize, usize) {
    let c = n / 2;
    let entry = &crate::mode::MODES[mode_index];
    let bins = passband_bins(entry.mode.default_bandwidth_hz(), n);
    if matches!(entry.mode, hl2::receiver::Mode::Ssb(_)) {
        match entry
            .mode
            .sideband()
            .unwrap_or(hl2::receiver::Sideband::Usb)
        {
            hl2::receiver::Sideband::Usb => (c, (c + bins).min(n.saturating_sub(1))),
            hl2::receiver::Sideband::Lsb => (c.saturating_sub(bins), c),
        }
    } else {
        (c.saturating_sub(bins), (c + bins).min(n.saturating_sub(1)))
    }
}

/// Inclusive passband bounds shared by the meter and waterfall overlay.
pub fn passband_columns(mode_index: usize) -> (usize, usize) {
    passband_window(BINS, mode_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_band_level_matches_floor() {
        for mag in [0, 1, 2, 10, 100, 200, 255, 256, 257, 65_535] {
            let s = compute(&[mag; BINS]);
            assert_eq!(s.floor_db, mag_to_db(mag), "magnitude {mag}");
            assert!((s.level_db - s.floor_db).abs() < 1e-4);
            assert!(s.margin_db < 1e-4);
            assert_eq!(s.sunits, 0);
        }
    }

    #[test]
    fn floor_matches_sorted_rank() {
        assert_eq!(FLOOR_PERCENTILE, 25);
        let mut mags = [0u16; BINS + 17];
        for pattern in 0..4 {
            for (i, m) in mags.iter_mut().enumerate() {
                *m = match pattern {
                    0 => (i * 197 + 17) as u16,
                    1 => (i % 256) as u16,
                    2 => [0, 1, 2, 255, 256, 257, 511, 512, 65_535][i % 9],
                    _ => (65_535 - i) as u16,
                };
            }
            for n in 1..=mags.len() {
                let mut sorted = mags;
                sorted[..n].sort_unstable();
                let expected = mag_to_db(sorted[n * FLOOR_PERCENTILE / 100]);
                assert_eq!(
                    compute(&mags[..n]).floor_db,
                    expected,
                    "pattern {pattern}, n={n}"
                );
            }
        }
        // A bucket boundary exactly at the requested rank must select the
        // next bucket, not the last value of the preceding one.
        mags.fill(256);
        mags[..BINS * 20 / 100].fill(255);
        assert_eq!(compute(&mags[..BINS]).floor_db, mag_to_db(256));
    }

    #[test]
    fn empty_row_is_silent() {
        let s = compute(&[]);
        assert_eq!(s.level_db, -120.0);
        assert_eq!(s.floor_db, -120.0);
        assert_eq!(s.margin_db, 0.0);
        assert_eq!(s.sunits, 0);
    }

    #[test]
    fn weak_inband_signal_retains_margin() {
        let mut mags = [2; BINS];
        let (lo, hi) = passband_columns(0);
        mags[lo..=hi].fill(20);
        let s = compute(&mags);
        assert_eq!(s.floor_db, mag_to_db(2));
        assert!((s.margin_db - 20.0).abs() < 1e-4);
        assert_eq!(s.sunits, 3);
    }

    /// Raise a run of columns above uniform noise.
    fn with_signal(m: &mut [u16], start: usize, width: usize, noise: u16, sig: u16) {
        for i in 0..m.len() {
            m[i] = noise;
        }
        for i in start..start + width {
            if let Some(x) = m.get_mut(i) {
                *x = sig;
            }
        }
    }

    #[test]
    fn inband_usb_signal_reads_over_floor() {
        // USB is right of the NCO (complex-negative frequencies).
        let mut mags = [0u16; BINS];
        let c = BINS / 2;
        with_signal(&mut mags, c + 4, 6, 100, 60_000);
        let s = compute(&mags);
        assert!(
            s.margin_db > 20.0,
            "in-band USB: margin {} dB must be well over the floor",
            s.margin_db
        );
        assert!(
            s.sunits >= 1,
            "in-band USB should read ≥ S1 (got S{})",
            s.sunits
        );
    }

    #[test]
    fn outofband_lsb_signal_reads_at_floor() {
        // The USB window excludes the line left of the NCO.
        let mut mags = [0u16; BINS];
        let c = BINS / 2;
        with_signal(&mut mags, c - 12, 6, 100, 60_000);
        let s = compute(&mags);
        assert!(
            s.margin_db < 6.0,
            "out-of-band (lower-side) line should read ≈ floor (margin {} dB)",
            s.margin_db
        );
        assert_eq!(s.sunits, 0);
    }

    #[test]
    fn floor_is_not_pulled_up_by_out_of_passband_signal() {
        // A few occupied columns do not shift the display percentile.
        let mut m = [0u16; BINS];
        with_signal(&mut m, 80, 4, 100, 60_000); // well left of centre
        let s = compute(&m);
        assert_eq!(s.floor_db, mag_to_db(100));
        assert!(s.margin_db < 1e-4);
    }

    #[test]
    fn sunits_map_is_monotonic_and_clamped() {
        assert_eq!(margin_to_sunits(0.0), 0);
        assert_eq!(margin_to_sunits(3.0), 0, "below S1 threshold → 0");
        assert_eq!(margin_to_sunits(S1_DB_OVER_FLOOR), 1, "S1 at the threshold");
        assert_eq!(margin_to_sunits(S1_DB_OVER_FLOOR + DB_PER_UNIT), 2);
        let s9 = S1_DB_OVER_FLOOR + 8.0 * DB_PER_UNIT;
        assert_eq!(margin_to_sunits(s9), 9, "S9 at {s9} dB over floor");
        assert_eq!(margin_to_sunits(s9 + 100.0), 9, "clamp at S9");
        assert_eq!(margin_to_sunits(f32::NAN), 0);
        assert_eq!(margin_to_sunits(-5.0), 0);
    }

    #[test]
    fn passband_width_rounds_only_at_column_precision() {
        assert_eq!(WF_BAND_HZ, 24_000);
        assert_eq!(DISPLAY_BIN_HZ, 75.0);
        assert_eq!(USB_PASSBAND_BINS, 35);
        for n in [1, 2, 159, 319, BINS, 321] {
            let width = passband_bins(USB_PASSBAND_HZ, n);
            let hz_columns = USB_PASSBAND_HZ as usize * n;
            assert!(width * WF_BAND_HZ >= hz_columns);
            assert!((width - 1) * WF_BAND_HZ < hz_columns);
            assert_eq!(passband_window(n, 0), (n / 2, (n / 2 + width).min(n - 1)));
        }
    }

    #[test]
    fn passband_columns_geometry() {
        let c = BINS / 2;
        // mode.rs order: [0] USB, [1] LSB, [2] AM, [3] FM, [4] NFM.
        let (u_lo, u_hi) = passband_columns(0);
        assert_eq!((u_lo, u_hi), (160, 195));
        assert_eq!(u_lo, c, "USB lower edge is the NCO column");
        assert!(u_hi > c && u_hi < BINS, "USB extends to the upper side");

        let (l_lo, l_hi) = passband_columns(1);
        assert_eq!(l_hi, c, "LSB upper edge is the NCO column");
        assert!(l_lo < c, "LSB extends to the lower side");
        assert_eq!(c - l_lo, u_hi - c, "LSB/USB bands are mirror-symmetric");

        for (idx, expected) in [(2, (53, 267)), (3, (0, 319)), (4, (93, 227))] {
            let (lo, hi) = passband_columns(idx);
            assert_eq!((lo, hi), expected);
            assert!(lo < c && hi > c, "[{idx}] double-sided straddles the NCO");
            // Wide FM is clipped to the display edges.
            let left_span = c - lo;
            let right_span = hi - c;
            assert!(
                left_span == right_span || lo == 0 || hi == BINS - 1,
                "[{idx}] double-sided should be symmetric (got {left_span} vs {right_span})\
                 or clamped to a window edge (lo={lo}, hi={hi}, BINS={BINS})"
            );
        }
    }
}
