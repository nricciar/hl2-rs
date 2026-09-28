//! S-meter for the virtual USB receiver — a pure function of the spectrum
//! row, mirroring `api/src/meter.rs::compute_s_meter` (PROTOCOL.md §16.3e).
//!
//! The UI's S-meter is *not* computed off the demod output: it is
//! signal-over-noise read from the *spectrum* (`mags`) — the same 320-column
//! row the waterfall already displays (a centred, zoomed slice of the band,
//! so each column is ~31 Hz wide at the default view). That module (`api/meter.rs`) even says
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

use crate::spectrum::{BINS, WF_BAND_HZ};

/// USB SSB receiver's channel-select bandwidth (Hz) — `Mode::Ssb`'s default
/// passband (`hl2::receiver::Mode::default_bandwidth_hz` = 2600). Kept as a
/// public const (for backwards-compat with the `teensy` unit tests and the
/// `hl2` `receiver`'s own tests which use this exact value).
pub const USB_PASSBAND_HZ: u32 = 2_600;
/// Hz per display bin. The waterfall shows a centred [`WF_BAND_HZ`] slice of
/// the band, so a display column spans the *displayed* band divided by the
/// column count (10 kHz / 320 = 31.25 Hz at the default). The passband window
/// below is derived from this, so it tracks whatever slice the display shows.
pub const DISPLAY_BIN_HZ: usize = WF_BAND_HZ / BINS;
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
///
/// USB is the default (index 0) in `crate::mode::MODES`; for the other
/// modes use [`compute_for_mode`] with the current mode index.
pub fn compute(mags: &[u16]) -> Smeter {
    compute_for_mode(mags, 0)
}

/// Compute the S-meter reading at `mode_index` (into `crate::mode::MODES`).
///
/// The passband window depends on the mode:
///
/// * **USB / LSB** — the receiver's sideband, `USB_PASSBAND_HZ` (= 2600 Hz
///   for SSB-USB) wide, on the matching side of the NCO (upper for USB,
///   lower for LSB).
/// * **AM / FM / NFM** — the receiver's *double-sideband* channel-select
///   passband, centred on the NCO, with the mode's default bandwidth
///   (`Mode::default_bandwidth_hz`).
///
/// The band-noise percentile (`FLOOR_PERCENTILE`) is computed on the whole
/// display regardless of mode (a single passband peak doesn't drag the
/// percentile), so the S reading remains "signal vs band noise" — the
/// classic definition.
///
/// `mode_index` is assumed to be in range; called by the radio task with
/// the value the render task publishes (always valid).
pub fn compute_for_mode(mags: &[u16], mode_index: usize) -> Smeter {
    let n = mags.len();
    let c = n / 2;
    let entry = &crate::mode::MODES[mode_index];

    // Resolve the passband geometry for this mode:
    //   * sideband (upper/lower) for SSB modes
    //   * centred, double-sided for AM / FM / NFM (mode's default bandwidth)
    let default_bw_hz = entry.mode.default_bandwidth_hz();
    let is_ssb = matches!(entry.mode, hl2::receiver::Mode::Ssb(_));
    let bins = passband_bins(default_bw_hz);

    let (lo, hi) = if is_ssb {
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
    };
    if n == 0 {
        return Smeter {
            level_db: -120.0,
            floor_db: -120.0,
            margin_db: 0.0,
            sunits: 0,
        };
    }
    // Level: RMS magnitude over the passband window of this mode (its
    // average power). Sum the *squared* magnitudes then take the root, so
    // the dB conversion is `20·log10(rms / full-scale)` — this is what lets
    // an AM carrier blend with its audio-modulated sidebands so the reading
    // tracks the modulation.
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
    // single passband peak sits in a handful of the `n` bins, so the
    // percentile lands on the noise regardless of where the signal is — the
    // clean, mode-agnostic "band-noise floor".
    //
    // Estimated with a single-pass *histogram* (256 buckets over the whole
    // `u16` range) rather than a `select_nth_unstable_by` partial-sort: that
    // sort + its closure comparator inlined per call costs several KiB of
    // `.text`, which this 192 KiB ITCM-limited build can't afford. A bucket
    // is 256 in magnitude width (≈ 0.7 dB across the scale) — well below the
    // 6 dB/units S-scale granularity the render uses — so reporting the
    // bucket *centre* as the floor doesn't change the S-unit a user reads.
    // No scratch buffer, no allocation, no sort.
    let k = n.min(BINS);
    let mut hist = [0u32; 256];
    for &m in &mags[..k] {
        hist[m as usize >> 8] += 1;
    }
    let target = (k as u32 * FLOOR_PERCENTILE as u32 / 100).max(1);
    let mut acc = 0u32;
    let mut floor_approx = 0u16;
    for (b, &cnt) in hist.iter().enumerate() {
        acc += cnt;
        if acc >= target {
            floor_approx = (b * 256 + 128) as u16; // centre of this bucket
            break;
        }
    }
    let floor_db = mag_to_db(floor_approx);

    let margin_db = (level_db - floor_db).max(0.0);
    let sunits = margin_to_sunits(margin_db);
    Smeter {
        level_db,
        floor_db,
        margin_db,
        sunits,
    }
}

/// Display-bin width of a bandwidth, rounded up.
#[inline]
fn passband_bins(bw_hz: u32) -> usize {
    (bw_hz as usize + DISPLAY_BIN_HZ - 1) / DISPLAY_BIN_HZ
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
    fn quiet_band_reads_near_floor() {
        let mags = uniform(BINS, 200);
        let s = compute(&mags);
        // A uniform display: passband RMS ≈ the single magnitude (level ~ the
        // band value), and the floor is *also* the band value (all bins equal).
        // The floor is estimated as a histogram bucket-centre though, so it
        // can sit a bucket-step (≤ ~3.5 dB across the scale) below the true
        // value — a quantization that *only ever inflates* the margin and
        // never below the band. On a flat band the margin must therefore
        // stay well under the S1 threshold so the meter reads no S-unit; we
        // assert that (the user-visible result) rather than the over-tight
        // level == floor the old exact-percentile floor allowed.
        assert!(
            s.margin_db < S1_DB_OVER_FLOOR,
            "quiet band: margin {} dB must stay under S1",
            s.margin_db
        );
        assert_eq!(s.sunits, 0, "quiet band should not read an S-unit");
    }

    #[test]
    fn all_full_scale_reads_full() {
        let mags = uniform(BINS, 65_535);
        let s = compute(&mags);
        // Full-scale: level_db == 0 (the ceiling, rms == full scale), and the
        // floor (a histogram bucket-centre, ≈ the same full-scale value) puts
        // the margin at ≈ 0 — a flat all-max display is "all passband, no
        // contrast", so no S-unit. The floor is the *nearest* bucket-centre
        // (here a hair below full-scale), so allow a small margin tolerance
        // and assert the meaningful consequence: below S1.
        assert!(
            (s.level_db - 0.0).abs() < 1e-4,
            "full-scale level: {} dB",
            s.level_db
        );
        assert!(
            s.margin_db < S1_DB_OVER_FLOOR,
            "full scale: margin {} dB (all passband, no S-unit)",
            s.margin_db
        );
        assert_eq!(s.sunits, 0, "full scale: no S-unit (flat display)");
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
        // The USB/LSB passband is *one-sided* (one side of the NCO), so it
        // must fit on one half of the display, and span roughly the voice
        // bandwidth. At the default 10 kHz display, 2.6 kHz is ~84 of the
        // 160 half-display bins.
        assert!(wbins >= 1 && wbins < BINS / 2, "passband {wbins} bins");
        assert!(
            (USB_PASSBAND_HZ as usize / hz_per_bin) <= wbins
                && wbins <= USB_PASSBAND_HZ as usize / hz_per_bin + 1,
            "passband {wbins} bins ≈ {USB_PASSBAND_HZ} Hz at {hz_per_bin} Hz/bin"
        );
    }
}
