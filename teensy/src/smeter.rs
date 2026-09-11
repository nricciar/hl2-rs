//! S-meter for the virtual USB receiver — a pure function of the spectrum
//! row, mirroring `api/src/meter.rs::compute_s_meter` (PROTOCOL.md §16.3e).
//!
//! The UI's S-meter is *not* computed off the demod output: it is
//! signal-over-noise read from the *spectrum* (`mags`) — the same 320-bin
//! row the waterfall already displays. That module (`api/meter.rs`) even says
//! "the S-meter is a consumer of the spectrum stream, which is the single
//! correct source of the band floor." So this is a faithful `no_std` port of
//! that method, and the virtual USB-SSB [`VirtualReceiver`] (see
//! `radio::rx`) exists to prove the demod runs at CPU speed, not to drive the
//! meter.
//!
//! The reading is **signal over noise** in the running USB receiver's
//! passband:
//!
//!   * `level` = RMS of the passband bins (the receiver's own channel-select
//!     window — for USB, the upper side of the NCO), in dB relative to full
//!     scale (`65535`). Measuring energy, not a single peak bin, means an AM
//!     carrier's DC line blends with its modulated sidebands so the reading
//!     follows the modulation rather than latching onto the carrier.
//!   * `floor` = the [`FLOOR_PERCENTILE`]th magnitude of the WHOLE display —
//!     an unbiased band-noise estimate (a single carrier is a few bins out of
//!     320, so the percentile lands on the noise regardless of the signal).
//!
//!   `margin_db = level_db - floor_db` is the scale-independent S margin. The
//!   render maps it to an S0..S9 index via [`margin_to_sunits`] — `S1` at
//!   [`S1_DB_OVER_FLOOR`] over the floor, [`DB_PER_UNIT`] per step (S9 ≈ 54
//!   dB over floor) — and shows the raw dB alongside.
//!
//! Pure + allocation-light: [`compute`] copies the magnitudes into a stack
//! buffer for the percentile selection (no heap); the caller (the radio task)
//! owns the buffer. `no_std` (uses `core` + `num_traits::Float` for the
//! `log10`/`sqrt`, consistent with `crate::spectrum`).

use core::f32;
use num_traits::float::Float;

use crate::spectrum::BINS;

/// Complex I/Q sample rate feeding the waterfall (Hz). One `mags` display bin
/// spans this divided by the bin count (96 kSps / 320 = 300 Hz).
pub const SAMPLE_RATE_HZ: u32 = 96_000;
/// The USB receiver's channel-select bandwidth (Hz) — `Mode::Ssb`'s default
/// passband (`hl2::receiver::Mode::default_bandwidth_hz` = 2600). The meter's
/// passband window spans this many Hz above the NCO.
pub const USB_PASSBAND_HZ: u32 = 2_600;
/// Hz per display bin = sample rate / display bins (96 000 / 320 = 300).
pub const DISPLAY_BIN_HZ: usize = SAMPLE_RATE_HZ as usize / BINS;
/// Passband window width in display bins (`USB_PASSBAND_HZ`, rounded up).
pub const USB_PASSBAND_BINS: usize =
    (USB_PASSBAND_HZ as usize + DISPLAY_BIN_HZ - 1) / DISPLAY_BIN_HZ;
/// Band-noise percentile (percent); `25` matches the UI (`api/meter.rs`).
pub const FLOOR_PERCENTILE: usize = 25;

/// The S margin (dB over the band floor) at which the meter first reads S1.
pub const S1_DB_OVER_FLOOR: f32 = 6.0;
/// dB per S-unit step (S1..S9, the classic 6-dB S-unit spacing).
pub const DB_PER_UNIT: f32 = 6.0;
/// S9 — the top of the scale.
pub const S9: u8 = 9;

/// One S-meter reading for a spectrum frame.
pub struct Smeter {
    /// Passband level, dB relative to full scale (65535).
    pub level_db: f32,
    /// Band noise floor (percentile), dB relative to full scale.
    pub floor_db: f32,
    /// `level_db - floor_db` — the scale-independent S margin (≥ 0).
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

/// Convert a `u16` magnitude to dB relative to full scale (`65535`). A zero or
/// sub-1 magnitude maps to `-120 dB` (a "silent bin").
fn mag_to_db(m: u16) -> f32 {
    let v = f32::from(m);
    if v <= 1.0 {
        return -120.0;
    }
    (20.0 * (v / 65_535.0).log10()).clamp(-120.0, 0.0)
}

/// Map a signal-over-noise margin (dB over the band floor) to an S0..S9 index.
///
/// `S1` at [`S1_DB_OVER_FLOOR`] over the floor, [`DB_PER_UNIT`] per step, so
/// S9 lands at `S1_DB_OVER_FLOOR + 8·DB_PER_UNIT` (≈ 54 dB over floor). Below
/// S1 (or non-finite) reads 0 (= noise floor). See the module doc for the
/// calibration rationale; the two named constants are the recalibration knobs
/// once the LNA / system sensitivity is nailed.
pub fn margin_to_sunits(margin_db: f32) -> u8 {
    if margin_db < S1_DB_OVER_FLOOR || !margin_db.is_finite() {
        return 0;
    }
    let steps = ((margin_db - S1_DB_OVER_FLOOR) / DB_PER_UNIT).floor();
    (1u32 + steps as u32).clamp(1, S9 as u32) as u8
}

/// Compute the USB S-meter reading from one `mags` row.
///
/// `mags` is the *display* band magnitudes (centered: the NCO/DC at
/// `len/2`, scaled `0..=65535`), i.e. the same 320-bin row the waterfall
/// shows. The passband window is the *upper* side of the display centre
/// (`[c, c + USB_PASSBAND_BINS]`) — the USB sideband for this DDC
/// (complex negative-frequency = real upper sideband, see
/// `hl2::receiver::demod::ssb`). The floor is the [`FLOOR_PERCENTILE`]th
/// magnitude of the whole display.
pub fn compute(mags: &[u16]) -> Smeter {
    let n = mags.len();
    let c = n / 2;
    let w = USB_PASSBAND_BINS.min(n.max(1));
    let lo = c;
    let hi = (c + w).min(n.saturating_sub(1));
    if n == 0 {
        return Smeter {
            level_db: -120.0,
            floor_db: -120.0,
            margin_db: 0.0,
            sunits: 0,
        };
    }
    // Level: RMS magnitude over the upper passband window (its average power).
    // Sum the *squared* magnitudes then take the root, so the dB conversion is
    // `20·log10(rms / full-scale)` — this is what lets an AM carrier blend with
    // its audio-modulated sidebands so the reading tracks the modulation.
    let mut sum_sq = 0.0f32;
    for &m in &mags[lo..=hi] {
        let v = f32::from(m);
        sum_sq += v * v;
    }
    let rms = (sum_sq / (hi - lo + 1) as f32).sqrt();
    let level_db = if rms <= 1.0 {
        -120.0
    } else {
        (20.0 * (rms / 65_535.0).log10()).clamp(-120.0, 0.0)
    };

    // Floor: the `FLOOR_PERCENTILE`-th magnitude of the *whole* display. A
    // single passband peak sits in a handfull of the `n` bins, so the
    // percentile lands on the noise regardless of where the signal is — the
    // clean, mode-agnostic "band noise floor". A small stack buffer + a
    // partial selection (no heap, no full sort).
    let k = n.min(BINS);
    let mut scratch = [0u16; BINS];
    scratch[..k].copy_from_slice(&mags[..k]);
    let target = ((k as u32 * FLOOR_PERCENTILE as u32 / 100) as usize).min(k - 1);
    let (_, nth, _) = scratch[..k].select_nth_unstable_by(target, |a, b| a.cmp(b));
    let floor_db = mag_to_db(*nth);

    let margin_db = (level_db - floor_db).max(0.0);
    let sunits = margin_to_sunits(margin_db);
    Smeter {
        level_db,
        floor_db,
        margin_db,
        sunits,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A uniform `mags` row (whole display = a single weak magnitude).
    fn uniform(n: usize, mag: u16) -> [u16; BINS] {
        let mut v = [0u16; BINS];
        for x in v[..n.min(BINS)].iter_mut() {
            *x = mag;
        }
        v
    }

    #[test]
    fn quiet_band_reads_at_floor() {
        let mags = uniform(BINS, 200);
        let s = compute(&mags);
        // A uniform display: passband RMS ≈ the single magnitude, the
        // percentile (floor) is the same — margin ≈ 0, no S-unit.
        assert!(
            (s.level_db - s.floor_db).abs() < 1e-3,
            "quiet band: level {} ≈ floor {}",
            s.level_db,
            s.floor_db
        );
        assert_eq!(s.sunits, 0, "quiet band should not read an S-unit");
    }

    #[test]
    fn all_full_scale_reads_full() {
        let mags = uniform(BINS, 65_535);
        let s = compute(&mags);
        // Full-scale: level_db == 0 (the ceiling), floor == 0 (uniform),
        // margin == 0 — a uniform full-scale display is "all passband, no
        // contrast". The meter itself can't see signal-over-noise in a flat
        // all-max row.
        assert!(
            (s.level_db - 0.0).abs() < 1e-6,
            "full-scale level: {} dB",
            s.level_db
        );
        assert_eq!(s.margin_db, 0.0, "full scale: level == floor (margin 0)");
    }

    /// Raise a contiguous run of bins (a "signal") to `sig`, keep the rest at
    /// `noise`. Mirrors `api/meter.rs::carrier_at_dc`.
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
        // A USB line to the RIGHT of the display centre (the complex
        // negative-frequency / real-upper side) at +100 dB over the noise.
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
        // A line on the LOWER (left) side of the centre is outside the USB
        // passband window; the upper-passband RMS level does not see it, so
        // the margin stays ≈ 0.
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
        // A strong carrier well away from the passband (adjacent station) sits
        // in a handful of the 320 bins; the 25th percentile still lands on the
        // noise, and the passband RMS does not see it.
        let mut m = [0u16; BINS];
        with_signal(&mut m, 80, 4, 100, 60_000); // well left of centre
        let s = compute(&m);
        assert!(
            s.floor_db < 0.0,
            "floor must be near the noise: {} dB",
            s.floor_db
        );
        assert!(
            s.margin_db < 6.0,
            "out-of-band adjacent station: margin {} dB",
            s.margin_db
        );
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
    fn passband_width_is_sane() {
        let hz_per_bin = DISPLAY_BIN_HZ;
        let wbins = USB_PASSBAND_BINS;
        // The passband must be a small handful of the 320 bins, not the whole
        // display, and span roughly the voice bandwidth.
        assert!(wbins >= 1 && wbins < BINS / 20, "passband {wbins} bins");
        assert!(
            (USB_PASSBAND_HZ as usize / hz_per_bin) <= wbins
                && wbins <= USB_PASSBAND_HZ as usize / hz_per_bin + 1,
            "passband {wbins} bins ≈ {USB_PASSBAND_HZ} Hz at {hz_per_bin} Hz/bin"
        );
    }
}
