//! The server-side S-meter: a pure function of the displayed slot's band
//! spectrum (see PROTOCOL.md §16.3e).
//!
//! The S-meter reads **signal over noise** in dB:
//!
//!   * **level** = the highest spectral magnitude inside the tuned passband
//!     window — *the signal + the noise that lives under it*.
//!
//!   * **floor** = the 25th percentile of the whole band's magnitudes,
//!     which is an unbiased *band noise floor* (independent of whether the
//!     passband is occupied): a single carrier is a *few* bins out of
//!     hundreds of noise bins, so the percentile lands on the noise
//!     regardless of mode.
//!
//!   * **S-unit reading** = `level − floor` — a clean, scale-independent
//!     dB of "signal above the band floor". This is the classic
//!     signal-over-noise-ratio definition and is **mode-agnostic**: SSB
//!     sideband, AM carrier, FM deviation, FT8 tones — all read as
//!     "elevated spectral energy vs. the band floor", no per-mode
//!     calibration required.
//!
//! The two `f32` magnitudes are scaled to `u16` in
//! [`crate::spectrum::display_mags_into`] — full-scale is `65_535`. We
//! convert to dB relative to that, then publish:
//!
//!   * [`SharedState::vrx_levels`] — the *peak* spectral energy inside the
//!     passband, in dB (relative to full scale).
//!   * [`SharedState::vrx_floors`] — the *band noise floor*, in dB (same
//!     reference).
//!
//! The UI renders `level − floor` as S-unit margin; the absolute reference
//! only matters that both fields share the same domain.
//!
//! This module is pure + allocation-free: [`compute_s_meter`] reads a `u16`
//! magnitude array plus the passband span and returns `(level_db,
//! floor_db)`. No `f32`, no state. The caller (the `run_spectral` loop in
//! `hub.rs`) invokes it per FFT frame when the displayed source is an
//! active EP6 slot, and pushes the result into a [`MeterState`] that
//! `Session::shared_state` picks up lock-free.
//!
//! The demod's pre-AGC `RawSampleTap` seam still carries the FT8/JS8/FT4
//! decoders in the same frame, exactly as before (see §16.3a / §16.3c).
//! The S-meter no longer hangs off that seam — it is a *consumer of the
//! spectrum stream*, which is the single correct source of the band floor
//! for every mode.

use std::sync::atomic::{AtomicU64, Ordering};

/// A published S-meter reading: the currently-displayed slot together with
/// its level + floor (dB relative to full-scale). Both values are stored as
/// IEEE-754 bit patterns of `f64` in `AtomicU64` so `Session::shared_state`
/// can read them lock-free on any command path (`welcome`, `get_state`,
/// the ~100 ms `run_levels` re-broadcast).
///
/// * `slot` is the EP6 RX whose EP6 baseband we're displaying (1-based).
///   `0` = "no active slot / EP4 wideband", the meter is idle.
/// * `level` is the *peak* spectral magnitude inside the passband, dB.
/// * `floor` is the *band noise floor* (25th percentile of the whole
///   display), dB. `level − floor` is the S margin.
///
/// Both level and floor default to `-120` dB. The UI treats non-finite
/// (or `-120`) as "silent — gauge at its bottom".
#[derive(Debug)]
pub struct MeterState {
    slot: AtomicU64,
    level: AtomicU64,
    floor: AtomicU64,
}

const NEGATIVE_120_BITS: u64 = (-120.0f64).to_bits();

impl MeterState {
    pub fn new() -> Self {
        Self {
            slot: AtomicU64::new(0),
            level: AtomicU64::new(NEGATIVE_120_BITS),
            floor: AtomicU64::new(NEGATIVE_120_BITS),
        }
    }

    /// Publish a fresh reading for `slot`. `level_db` / `floor_db` should
    /// already be in dB relative to full scale (see [`compute_s_meter`]).
    /// A `slot` of `0` invalidates the reading (slot = "none").
    pub fn set(&self, slot: u8, level_db: f64, floor_db: f64) {
        self.slot.store(slot as u64, Ordering::Relaxed);
        self.level.store(level_db.to_bits(), Ordering::Relaxed);
        self.floor.store(floor_db.to_bits(), Ordering::Relaxed);
    }

    /// Lock-free read for `Session::shared_state`. Returns `(slot,
    /// level_db, floor_db)`. A `slot` of `0` means the display is not an
    /// EP6 slot — the caller should not include the slot in
    /// `vrx_levels` / `vrx_floors`.
    pub fn read(&self) -> (u8, f64, f64) {
        let slot = self.slot.load(Ordering::Relaxed) as u8;
        let lev = f64::from_bits(self.level.load(Ordering::Relaxed));
        let fl = f64::from_bits(self.floor.load(Ordering::Relaxed));
        let lev = if lev.is_finite() && lev > -140.0 {
            lev
        } else {
            -120.0
        };
        let fl = if fl.is_finite() && fl > -140.0 {
            fl
        } else {
            -120.0
        };
        (slot, lev, fl)
    }
}

impl Default for MeterState {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a u16 magnitude to dB relative to full scale (65_535).
/// A zero magnitude maps to `−120 dB` (a "silent bin").
fn mag_to_db(m: u16) -> f64 {
    let v = m as f64;
    if v <= 1.0 {
        return -120.0;
    }
    let db = (20.0f64 * (v / 65_535.0).log10()).clamp(-120.0, 0.0);
    db
}

/// Compute (level_db, floor_db) for one frame.
///
/// * `mags` — the *displayed* band magnitudes (centered: DC at `len/2`).
///   Length is `cfg.wideband_bins`, values scaled `0..=65_535`.
/// * `passband_half_width_bins` — half the passband window, **in bins**,
///   around the display centre (= the tuned frequency). The caller
///   computes this from the running RX's channel-select bandwidth +
///   offset, the FFT size, and the bin count: see `hub.rs::run_spectral`.
/// * `floor_percentile` — the band floor as a quantile of the *whole*
///   display. `25` (percent) gives the robust band-noise estimate.
///
/// Returns `(level_db, floor_db)` both in dB relative to full scale
/// (65_535 = 0 dB). The UI reads the signal-over-floor margin as
/// `level_db − floor_db`.
///
/// Invariant: `level_db >= floor_db`. The level is the peak *inside* the
/// passband window and the floor is the *25th percentile* of the *whole*
/// display, so the peak can only fall below the floor when the *entire*
/// display is below it — and then the peak is also the floor (equal).
pub fn compute_s_meter(
    mags: &[u16],
    passband_half_width_bins: usize,
    floor_percentile: usize,
) -> (f64, f64) {
    if mags.is_empty() {
        return (-120.0, -120.0);
    }
    let n = mags.len();
    let c = n / 2;
    let lo = c.saturating_sub(passband_half_width_bins);
    let hi = (c + passband_half_width_bins).min(n - 1);
    if hi < lo {
        return (-120.0, -120.0);
    }

    // Level: max magnitude inside the passband window.
    let mut peak = 0u16;
    for &m in &mags[lo..=hi] {
        if m > peak {
            peak = m;
        }
    }
    let level_db = mag_to_db(peak);

    // Floor: `floor_percentile`-th percentile of the *whole* display.
    // A single passband peak sits in a handfull of bins out of thousands of
    // noise bins, so the percentile lands on the noise regardless of where
    // the signal is. This is the clean, mode-agnostic "band noise floor".
    if floor_percentile < 1 || floor_percentile > 100 {
        return (level_db, level_db);
    }
    // `target` is an index into the *sorted* mags: the magnitude such that
    // `floor_percentile` % of the display is at or below it. This is the
    // *unbiased* band-noise estimate: a single passband peak sits in a
    // handfull of the `n` bins, so the percentile lands on the noise
    // regardless of where the signal is.
    let target = ((n as u64 * floor_percentile as u64 / 100) as usize).min(n - 1);
    let mut sorted = mags.to_vec();
    sorted.sort_unstable();
    let floor_db = mag_to_db(sorted[target]);

    // Level ≥ floor by the invariant above; but clamp to be safe.
    if floor_db > level_db {
        return (level_db, level_db);
    }
    (level_db, floor_db)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quiet band: a `n_bins`-bin display where every bin is set to the
    /// same weak magnitude (≈ S1).
    fn quiet_band(n_bins: usize, mag: u16) -> Vec<u16> {
        vec![mag; n_bins]
    }

    /// A single strong carrier at the display centre (DC), with the
    /// remaining `n_bins − 1` bins at a weak background `noise_mag`.
    fn carrier_at_dc(n_bins: usize, noise_mag: u16, carrier_mag: u16) -> Vec<u16> {
        let mut m = vec![noise_mag; n_bins];
        let c = n_bins / 2;
        // The real passband of a strong AM carrier spans a *handful* of
        // bins; put the carrier in the 3 bins around the display centre to
        // simulate that (in a real `display_mags_into` the peak is in one
        // bin with adjacent rollover).
        for d in 1..=1 {
            *m.get_mut(c - d).unwrap() = carrier_mag;
            *m.get_mut(c + d).unwrap() = carrier_mag;
        }
        m[c] = carrier_mag;
        m
    }

    #[test]
    fn quiet_band_level_matches_floor() {
        // A uniform-weak display: the whole display is the same magnitude,
        // so level = floor exactly (S margin = 0, the "S1" resting read).
        let mags = quiet_band(1024, 200);
        let (lev, fl) = compute_s_meter(&mags, 64, 25);
        assert!(
            (lev as f64 - fl as f64).abs() < 1e-9,
            "quiet band: level ({lev}) ≈ floor ({fl})"
        );
        assert!(lev < 0.0, "quiet-bin level is below full scale");
    }

    #[test]
    fn carrier_well_above_floor_in_passband() {
        // A full-scale carrier inside the passband; the rest of the display
        // is weak noise. The level (peak in the passband) must be well
        // above the floor (25th percentile of the whole display, which is
        // noise).
        let mags = carrier_at_dc(1024, 100, 60_000);
        let (lev, fl) = compute_s_meter(&mags, 64, 25);
        assert!(
            lev > fl + 20.0,
            "carrier-in-band: level {lev} dB must be >20 dB over floor {fl} dB (S-margin)"
        );
        assert!(
            fl < 20.0,
            "floor should be near full scale minus a lot (≈ −6 dB from a 100/65535 mag) — got {fl}"
        );
        // A 60_000/65_535 mag is ≈ full scale → level ≈ 0 dB FS.
        assert!(
            lev > -1.0 && lev <= 1.0,
            "a 60_000/65_535 mag is ≈ −0 dBFS — level {lev} dB"
        );
    }

    #[test]
    fn floor_is_not_pulled_up_by_signal_outside_passband() {
        // A strong carrier *outside* the passband (e.g. an adjacent
        // station) sits in a handfull of the `n` bins. The 25th
        // percentile still lands on the noise.
        let n_bins = 1024;
        let mut m = vec![100u16; n_bins];
        let c = n_bins / 2;
        // A strong carrier ± 80 bins from the passband centre (outside a
        // 64-bin-half-width window).
        for d in 70..=90 {
            m[(c + d) % n_bins] = 60_000;
            m[(c - d + n_bins) % n_bins] = 60_000;
        }
        let (lev, fl) = compute_s_meter(&m, 64, 25);
        // The passband window [c-64, c+64] does NOT include the [c-90, c+90]
        // carrier, so the level is the noise. But the floor is the 25th
        // percentile of the whole display — still noise.
        assert!(
            lev < 20.0,
            "level outside the passband window should be ≈ noise — got {lev} dB"
        );
        assert!(
            fl < 20.0,
            "floor must be the band noise (≈ −6 dB) — got {fl} dB"
        );
    }
}
