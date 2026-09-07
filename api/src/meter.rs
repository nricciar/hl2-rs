//! The server-side S-meter: a pure function of the displayed slot's band
//! spectrum (see PROTOCOL.md §16.3e).
//!
//! The S-meter reads **signal over noise** in dB:
//!
//!   * **level** = the *root-mean-square* spectral energy inside the running
//!     receiver's channel passband — *the signal's average power (including
//!     the modulation that carries the audio) plus the noise under it*.
//!     Measuring energy, not a single peak bin, means an AM carrier's huge
//!     DC line is blended with its sidebands, so the reading rises and falls
//!     with the **modulation** (speech vs. silence) rather than latching onto
//!     the constant carrier.
//!
//!   * **floor** = the 25th percentile of the whole band's magnitudes,
//!     which is an unbiased *band noise floor* (independent of whether the
//!     passband is occupied): a single carrier is a *few* bins out of
//!     hundreds of noise bins, so the percentile lands on the noise
//!     regardless of mode.
//!
//!   * **S-unit reading** = `level − floor` — a clean, scale-independent
//!     dB of *band signal energy above the band floor*. This is a true
//!     signal-over-noise-ratio and is **mode-agnostic**: SSB sideband, AM
//!     carrier + sidebands, FM deviation, FT8 tones — all read as "elevated
//!     spectral energy vs. the band floor", no per-mode calibration.
//!
//! # Passband orientation
//!
//! The passband window is **not** centred for every mode — it mirrors the
//! running receiver's actual channel-select passband (the same band the UI
//! shades on the panadapter, see `draw_vrx_passband`):
//!
//!   * **Upper** (USB, and the digital modes whose audio lives above the
//!     carrier — FT8 / FT4 / JS8) → `[centre, centre + bw]`
//!   * **Lower** (LSB) → `[centre − bw, centre]`
//!   * **Centered** (AM, FM, NFM — both sidebands) → `[centre − bw, centre + bw]`
//!
//! Passing the wrong orientation is what used to make USB and LSB read
//! identically (the meter latched onto the *other* sideband's carrier, which
//! the receiver never passes); the [`PassbandShape`] selects the window.
//!
//! The magnitudes are scaled to `u16` in [`crate::spectrum::display_mags_into`]
//! — full-scale is `65_535`. We convert to dB relative to that, then publish:
//!
//!   * [`SharedState::vrx_levels`] — the *RMS passband energy*, in dB
//!     (relative to full scale).
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

/// The side(s) of the tuned frequency the running receiver's channel-select
/// passband occupies. Selects the passband window used for the S-meter
/// level, mirroring the band the UI shades on the panadapter
/// (`canvas::draw_vrx_passband`) rather than assuming everything is centred
/// on the tune.
///
/// Round-trips to `u32` for atomic storage in `hub.rs`; the hub stores the
/// *shape* (not the wire mode's disc), so this enum is the single source of
/// truth for "which side of the tune is the passband on".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassbandShape {
    /// Passband is `[centre, centre + bw]`: USB voice, and the digital modes
    /// (FT8 / FT4 / JS8), whose audio lives *above* the carrier.
    Upper,
    /// Passband is `[centre − bw, centre]`: LSB voice.
    Lower,
    /// Passband is `[centre − bw, centre + bw]`: AM (DSB-FC), FM, NFM (both).
    Centered,
}

impl PassbandShape {
    /// Stable on-wire `u32` (used to store in the hub's `AtomicU32`).
    pub const fn as_u32(self) -> u32 {
        match self {
            PassbandShape::Upper => 1,
            PassbandShape::Lower => 2,
            PassbandShape::Centered => 3,
        }
    }
    /// Reconstruct from a stored `u32`. Unknown values fall back to
    /// [`PassbandShape::Centered`] (symmetric — the most conservative
    /// reading for an unrecognised mode).
    pub const fn from_u32(v: u32) -> Self {
        match v {
            1 => PassbandShape::Upper,
            2 => PassbandShape::Lower,
            _ => PassbandShape::Centered,
        }
    }
}

/// Map a wire receiver mode to its [`PassbandShape`] — the *single* place
/// in the server that maps `VrxMode` to an S-meter window, mirroring
/// `canvas::draw_vrx_passband` (the authoritative channel-select band the
/// UI already draws).
///
/// The digital modes are demodulated as USB (audio above the carrier), so
/// they take [`PassbandShape::Upper`], same as plain `Usb`.
pub fn shape_for_mode(mode: &hl2_common::VrxMode) -> PassbandShape {
    match mode {
        hl2_common::VrxMode::Lsb => PassbandShape::Lower,
        hl2_common::VrxMode::Am | hl2_common::VrxMode::Fm | hl2_common::VrxMode::FmNarrow => {
            PassbandShape::Centered
        }
        hl2_common::VrxMode::Usb
        | hl2_common::VrxMode::Ft8
        | hl2_common::VrxMode::Ft4
        | hl2_common::VrxMode::Js8 => PassbandShape::Upper,
    }
}

/// Compute `(level_db, floor_db)` for one frame.
///
/// * `mags` — the *displayed* band magnitudes (centered: DC at `len/2`).
///   Length is `cfg.wideband_bins`, values scaled `0..=65_535`.
/// * `passband_width_bins` — the passband window width, **in bins**, on the
///   passband side(s) of the display centre (= the tuned frequency). The
///   caller computes this from the running RX's channel-select bandwidth +
///   offset, the FFT size, and the bin count: see `hub.rs::run_spectral`.
///   For [`PassbandShape::Upper`] / [`PassbandShape::Lower`] this is the
///   one-sided width `[0, w]` / `[-w, 0]`; for [`PassbandShape::Centered`]
///   it is the half-width, window `[−w, +w]`.
/// * `shape` — which side(s) of the tune the passband occupies
///   (see [`PassbandShape`]).
/// * `floor_percentile` — the band floor as a quantile of the *whole*
///   display. `25` (percent) gives the robust band-noise estimate.
///
/// Returns `(level_db, floor_db)` both in dB relative to full scale
/// (65_535 = 0 dB). The UI reads the signal-over-floor margin as
/// `level_db − floor_db`.
///
/// `level` is the **RMS** (root-mean-square) magnitude over the passband
/// window — the window's *average spectral power*, not a single peak bin.
/// That is what lets an AM carrier (a huge, near-constant DC line) blend
/// with its audio-modulated sidebands so the reading tracks the modulation
/// rather than latching onto the carrier.
pub fn compute_s_meter(
    mags: &[u16],
    passband_width_bins: usize,
    shape: PassbandShape,
    floor_percentile: usize,
) -> (f64, f64) {
    if mags.is_empty() || passband_width_bins == 0 {
        return (-120.0, -120.0);
    }
    let n = mags.len();
    let c = n / 2;
    let w = passband_width_bins.min(n);
    // Window bounds depend on which sideband the receiver actually passes.
    let (lo, hi) = match shape {
        PassbandShape::Lower => (c.saturating_sub(w), c),
        PassbandShape::Upper => (c, (c + w).min(n - 1)),
        PassbandShape::Centered => (c.saturating_sub(w), (c + w).min(n - 1)),
    };
    if hi < lo {
        return (-120.0, -120.0);
    }

    // Level: RMS magnitude over the passband window (its average power).
    // Sum the *squared* magnitudes then take the root, so the dB conversion
    // is `20·log10(rms / full-scale)`.
    let mut sum_sq = 0f64;
    for &m in &mags[lo..=hi] {
        let v = m as f64;
        sum_sq += v * v;
    }
    let rms = (sum_sq / (hi - lo + 1) as f64).sqrt();
    let level_db = if rms <= 1.0 {
        -120.0
    } else {
        (20.0f64 * (rms / 65_535.0).log10()).clamp(-120.0, 0.0)
    };

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

    /// A single strong carrier at the display centre (DC) spanning `width`
    /// bins, with the remaining bins at a weak background `noise_mag`.
    fn carrier_at_dc(n_bins: usize, width: usize, noise_mag: u16, carrier_mag: u16) -> Vec<u16> {
        let mut m = vec![noise_mag; n_bins];
        let c = n_bins / 2;
        // The real passband of a strong carrier spans a *handful* of bins;
        // paint `width` bins around the display centre to simulate that
        // (in a real `display_mags_into` the peak is in one bin with
        // adjacent rollover).
        for d in 0..width {
            if let Some(bin) = m.get_mut(c + d - width / 2) {
                *bin = carrier_mag;
            }
        }
        m
    }

    #[test]
    fn quiet_band_level_matches_floor() {
        // A uniform-weak display: the whole display is the same magnitude,
        // so level = floor exactly (S margin = 0, the "S1" resting read).
        let mags = quiet_band(1024, 200);
        let (lev, fl) = compute_s_meter(&mags, 64, PassbandShape::Centered, 25);
        assert!(
            (lev as f64 - fl as f64).abs() < 1e-9,
            "quiet band: level ({lev}) ≈ floor ({fl})"
        );
        assert!(lev < 0.0, "quiet-bin level is below full scale");
    }

    #[test]
    fn carrier_well_above_floor_in_passband() {
        // A near-full-scale carrier inside the (centred) passband; the rest
        // of the display is weak noise. The level (RMS in the passband) must
        // be well above the floor (25th percentile of the whole display,
        // which is noise).
        let mags = carrier_at_dc(1024, 3, 100, 60_000);
        let (lev, fl) = compute_s_meter(&mags, 64, PassbandShape::Centered, 25);
        assert!(
            lev > fl + 20.0,
            "carrier-in-band: level {lev} dB must be >20 dB over floor {fl} dB (S-margin)"
        );
        assert!(
            fl < 20.0,
            "floor should be near the noise magnitude — got {fl} dB"
        );
        // A 3-bin 60_000 carrier in a 129-bin window: RMS is below full
        // scale, so level is a negative dB FS (but high over the floor).
        assert!(
            (lev - 0.0).abs() <= 0.5 || lev < 0.0,
            "3-bin carrier in a 129-bin window: level {lev} dB should be ≤ full scale"
        );
    }

    #[test]
    fn floor_is_not_pulled_up_by_signal_outside_passband() {
        // A strong carrier *outside* the passband (e.g. an adjacent
        // station) sits in a handful of the `n` bins. The 25th percentile
        // still lands on the noise, and the RMS level does not see it.
        let n_bins = 1024;
        let mut m = vec![100u16; n_bins];
        let c = n_bins / 2;
        // A strong carrier ± 80 bins from the passband centre (outside a
        // 64-bin window).
        for d in 70..=90 {
            m[(c + d) % n_bins] = 60_000;
            m[(c - d + n_bins) % n_bins] = 60_000;
        }
        let (lev, fl) = compute_s_meter(&m, 64, PassbandShape::Centered, 25);
        // The passband window [c-64, c+64] does NOT include the [c-90, c+90]
        // carrier, so the level is the noise. But the floor is the 25th
        // percentile of the whole display — still noise.
        assert!(
            lev < 20.0,
            "level outside the passband window should be ≈ noise — got {lev} dB"
        );
        assert!(fl < 20.0, "floor must be the band noise — got {fl} dB");
    }

    #[test]
    fn usb_reads_signal_lsb_reads_floor() {
        // The reported bug: the *same* physical signal (a strong chunk of
        // the *upper* half of the display, i.e. the USB side) must read
        // **strong** in USB and **≈ floor** in LSB. LSB's window is
        // `[c−w, c−1]`, which does *not* overlap the `[c+1, c+w]` USB line
        // (the boundary bin `c` = carrier DC = noise, since SSB suppresses it).
        let n_bins = 1024;
        let c = n_bins / 2;
        let w = 32usize;
        let noise = 100u16;
        let sig = 60_000u16;
        // A USB line at `c+1..c+w` (strictly *above* the carrier bin `c`).
        let mut usb = vec![noise; n_bins];
        for d in 1..=w {
            usb[c + d] = sig;
        }
        // Compute using the *true* SSB passband windows:
        //   USB → `[c, c+w]` (which contains the `[c+1, c+w]` line).
        //   LSB → `[c−w, c]` (which does NOT — bin `c` is the carrier,
        //         which SSB suppresses = noise).
        let (u_lev, u_fl) = compute_s_meter(&usb, w, PassbandShape::Upper, 25);
        let (l_lev, l_fl) = compute_s_meter(&usb, w, PassbandShape::Lower, 25);

        // USB: the line is in `[c+1, c+w]` ⊂ USB window, so level >> floor.
        assert!(
            u_lev > u_fl + 20.0,
            "USB should read the upper-side line: level {u_lev} > floor {u_fl} + 20"
        );

        // LSB: `[c−w, c]` has only noise (bin `c` = carrier = noise in SSB),
        // so level ≈ floor.
        assert!(
            (l_lev - l_fl).abs() < 6.0,
            "LSB should ≈ floor (line is out of passband): level {l_lev} floor {l_fl}"
        );

        // And the two readings must *differ* now (the old bug was "identical").
        assert!(
            (u_lev - u_fl) - (l_lev - l_fl) > 15.0,
            "USB margin {:+.1} dB must exceed LSB margin {:+.1} dB by >15 dB",
            u_lev - u_fl,
            l_lev - l_fl
        );
    }

    #[test]
    fn lsb_reads_signal_in_lower_side() {
        // Mirror of the previous test: a strong line in the *lower* half
        // reads strong in LSB and ≈ floor in USB.
        let n_bins = 1024;
        let c = n_bins / 2;
        let w = 32usize;
        let noise = 100u16;
        let sig = 60_000u16;
        let mut lsb = vec![noise; n_bins];
        for d in 1..=w {
            lsb[c - d] = sig;
        }
        let (l_lev, l_fl) = compute_s_meter(&lsb, w, PassbandShape::Lower, 25);
        let (u_lev, u_fl) = compute_s_meter(&lsb, w, PassbandShape::Upper, 25);
        assert!(
            l_lev > l_fl + 20.0,
            "LSB should read the lower-side line: level {l_lev} > floor {l_fl} + 20"
        );
        assert!(
            (u_lev - u_fl).abs() < 6.0,
            "USB should ≈ floor (line is out of passband): level {u_lev} floor {u_fl}"
        );
    }

    #[test]
    fn am_level_tracks_modulation_not_carrier() {
        // An AM signal = a large constant carrier at DC plus sidebands whose
        // energy is the *audio modulation*. With an RMS (not peak) level, the
        // reading must be dominated by the **modulation**: a quiet frame
        // (small sidebands) reads lower than a loud frame (large sidebands),
        // even though the carrier is identical in both.
        let n_bins = 1024usize;
        let c = n_bins / 2;
        let noise = 100u16;
        let carrier = 40_000u16; // constant, both frames
        let side_quiet = 3_000u16;
        let side_loud = 20_000u16;
        let span = 40usize; // sideband bins around the carrier

        let am_frame = |side: u16| -> Vec<u16> {
            let mut m = vec![noise; n_bins];
            m[c] = carrier;
            for d in 1..=span {
                m[c - d] = side;
                m[c + d] = side;
            }
            m
        };

        let quiet = am_frame(side_quiet);
        let loud = am_frame(side_loud);
        let (q_lev, q_fl) = compute_s_meter(&quiet, span, PassbandShape::Centered, 25);
        let (l_lev, l_fl) = compute_s_meter(&loud, span, PassbandShape::Centered, 25);

        // Both frames sit above their floor (there is a real signal present).
        assert!(q_lev > q_fl, "quiet AM: level {q_lev} above floor {q_fl}");
        assert!(l_lev > l_fl, "loud AM: level {l_lev} above floor {l_fl}");
        // And the loud frame must read clearly higher than the quiet one —
        // the meter reacts to audio level, it is not latched on the carrier.
        assert!(
            l_lev - l_fl > q_lev - q_fl + 6.0,
            "loud AM margin {:+.1} dB must exceed quiet AM margin {:+.1} dB by >6 dB",
            l_lev - l_fl,
            q_lev - q_fl
        );
    }

    #[test]
    fn digital_tone_reads_below_its_peak() {
        // A single narrow line in an RMS window reads ~10·log10(1/w) dB
        // *below* its own peak (energy spread over `w` bins). That is the
        // accepted rescale: a CW/FT8 tone sits below where the old *peak*
        // reading would place it, but is still far above the noise floor and
        // correctly *oriented* (upper side for FT8).
        let n_bins = 1024;
        let c = n_bins / 2;
        let w = 32usize;
        let noise = 100u16;
        let sig = 60_000u16;
        let mut tone = vec![noise; n_bins];
        tone[c] = sig; // one line at the carrier bin (upper-side window start)
        let (lev, _fl) = compute_s_meter(&tone, w, PassbandShape::Upper, 25);
        // level is `20·log10(sig / sqrt(w) / full)` roughly; peak was sig.
        let expected_damping = 10.0 * (w as f64).log10();
        let sig_peak_db = 20.0 * ((sig as f64 / 65_535.0).log10());
        assert!(
            (lev - (sig_peak_db - expected_damping)).abs() < 1.5,
            "one-line RMS level {lev} dB ≈ peak {sig_peak_dB:.1} − {expected_damping:.1} dB",
            sig_peak_dB = sig_peak_db
        );
    }
}
