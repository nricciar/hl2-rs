//! Auto floor/ceil for the waterfall colour ramp.
//!
//! A `no_std` lift-and-shift of the UI's auto-scale
//! (`ui/src/app.rs::Shared::update_auto_scale`). The UI tracks the
//! panadapter/waterfall dB window automatically from the live spectrum: a
//! **robust noise-floor estimate** (the *median* of the display band, so a few
//! strong carriers don't drag it up) sets the floor, and the **peak** sets the
//! ceiling. Both are smoothed with an asymmetric attack/release follower and a
//! peak-hold so a keyed CW / bursty digital signal doesn't strobe the ramp.
//!
//! The Teensy has one fixed spectrum source (RX1 @ the NCO), so this is
//! always on: the [`AutoScale`] state is owned by the `radio` task, advanced
//! once per spectrum frame, and the resulting `(floor, ceil)` window is
//! published for the `render` task to feed `display::palette::bin_color`.
//!
//! `f32` throughout (the display palette is `f32`); magnitudes are the same
//! `u16 0..=65535` linear scale everywhere (`65535` = `0 dBFS` full scale).

use num_traits::float::Float;

use crate::spectrum::BINS;

/// dB clamps — the same full-scale reference the palette renders with.
const LOG_FLOOR: f32 = -100.0;
const LOG_CEIL: f32 = 10.0;

/// Smoothing constants — mirrors `ui/src/app.rs::Shared::update_auto_scale`.
const HEADROOM_DB: f32 = 5.0;
const MIN_SPAN_DB: f32 = 25.0;
const FLOOR_ATTACK_ALPHA: f32 = 0.05;
const FLOOR_RELEASE_ALPHA: f32 = 0.01;
const DEADBAND_DB: f32 = 1.0;
const CEIL_HOLD_MARGIN_DB: f32 = 4.0;
const CEIL_HOLD_FRAMES: u32 = 12;
const CEIL_RELEASE_DB_PER_FRAME: f32 = 0.1;

/// Seed values — the UI's initial auto floor/ceil (`ui/src/app.rs:289-290`).
const SEED_FLOOR: f32 = -85.0;
const SEED_CEIL: f32 = -15.0;

/// A `u16` magnitude → dB, in the same units the palette draws with
/// (`65535` = `0 dBFS`). Mirrors `ui/src/app.rs::mag_to_db` (and the palette's
/// `lin_to_db`).
#[inline]
fn mag_to_db(mag: u16) -> f32 {
    let v = f32::from(mag) / 65_535.0;
    if v <= 0.0 {
        LOG_FLOOR
    } else {
        (20.0 * Float::log10(v)).clamp(LOG_FLOOR, LOG_CEIL)
    }
}

/// Robust estimate of the `(floor, ceil)` targets from one frame: the
/// **median** (noise floor, insensitive to a few strong carriers) and the
/// **max** (signal peak). Mirrors `ui/src/app.rs::auto_estimate`.
///
/// `lin_to_db` is monotone in the magnitude, so the median of the dB values is
/// the dB of the median magnitude — we select the median *magnitude* (a small
/// stack copy + `select_nth_unstable_by`, no heap — the same trick as
/// `crate::smeter::compute`) and convert, so no per-bin dB array is needed.
fn estimate(mags: &[u16]) -> Option<(f32, f32)> {
    let n = mags.len();
    if n == 0 {
        return None;
    }
    let max = mags.iter().copied().max()?;

    let k = n.min(BINS);
    let mut scratch = [0u16; BINS];
    scratch[..k].copy_from_slice(&mags[..k]);
    let mid = k / 2;
    let (_, median, _) = scratch[..k].select_nth_unstable_by(mid, |a, b| a.cmp(b));

    Some((mag_to_db(*median), mag_to_db(max)))
}

/// One step of the asymmetric exponential follower (fast attack, slow release)
/// with a deadband. Mirrors `ui/src/app.rs::auto_step_follower`.
fn step_follower(
    val: &mut f32,
    target: f32,
    alpha_attack: f32,
    alpha_release: f32,
    deadband: f32,
) -> bool {
    let diff = target - *val;
    if diff.abs() <= deadband {
        return false;
    }
    let alpha = if diff > 0.0 {
        alpha_attack
    } else {
        alpha_release
    };
    *val += diff * alpha;
    true
}

/// Keep the window at least `min_span` wide by expanding symmetrically about
/// the midpoint. Mirrors `ui/src/app.rs::enforce_min_span`.
fn enforce_min_span(floor: &mut f32, ceil: &mut f32, min_span: f32) {
    if *ceil - *floor >= min_span {
        return;
    }
    let mid = (*floor + *ceil) / 2.0;
    *floor = mid - min_span / 2.0;
    *ceil = mid + min_span / 2.0;
}

/// The auto-scale state machine: a smoothed `(floor, ceil)` dB window for the
/// waterfall, advanced once per spectrum frame. Owned by the `radio` task
/// (single-threaded, so no lock is needed).
pub struct AutoScale {
    floor: f32,
    ceil: f32,
    /// Peak-hold frame counter for the ceiling release window.
    ceil_age: u32,
    /// One-shot: snap both edges to the next frame's estimate instead of
    /// blending (the UI sets this on a source change / re-enable).
    recenter: bool,
}

impl AutoScale {
    /// Seed with the UI's initial auto values (floor −85, ceil −15), with no
    /// pending re-center — the first frame blends up from the seed exactly as
    /// the UI does on start.
    pub fn new() -> Self {
        Self {
            floor: SEED_FLOOR,
            ceil: SEED_CEIL,
            ceil_age: 0,
            recenter: false,
        }
    }

    /// Snap both edges to the next frame's estimate instead of blending
    /// (after a source / regime change). The UI sets this on a spectrum-source
    /// change and on `floor_auto` re-enable.
    pub fn recenter(&mut self) {
        self.recenter = true;
    }

    /// The current computed `(floor, ceil)`.
    pub fn scale(&self) -> (f32, f32) {
        (self.floor, self.ceil)
    }

    /// Advance the window from one spectrum frame; returns `true` if either
    /// edge moved. Mirrors `ui/src/app.rs::Shared::update_auto_scale`.
    pub fn step(&mut self, mags: &[u16]) -> bool {
        let Some((median, max)) = estimate(mags) else {
            return false;
        };

        let noise_min = (median - HEADROOM_DB).max(-100.0);
        let ceil_target = max.max(noise_min) - HEADROOM_DB;

        let mut floor = self.floor;
        let mut ceil = self.ceil;
        let mut age = self.ceil_age;

        if self.recenter {
            // A different magnitude regime: snap both edges instead of
            // blending from the old regime.
            floor = noise_min;
            ceil = ceil_target.min(10.0);
            age = 0;
            self.recenter = false;
        } else {
            step_follower(
                &mut floor,
                noise_min,
                FLOOR_ATTACK_ALPHA,
                FLOOR_RELEASE_ALPHA,
                DEADBAND_DB,
            );

            if ceil_target >= ceil - CEIL_HOLD_MARGIN_DB {
                // The signal that set the ceiling (or a stronger one) is still
                // present: rise only, freeze otherwise — keying dips within the
                // margin never move it.
                ceil = ceil.max(ceil_target);
                age = 0;
            } else if age < CEIL_HOLD_FRAMES {
                // Max left the margin but the grace window is still open (a
                // brief keying gap / fading): hold.
                age += 1;
            } else {
                // Signal genuinely gone: slow release toward the current target.
                ceil -= (ceil - ceil_target).min(CEIL_RELEASE_DB_PER_FRAME);
            }
        }

        enforce_min_span(&mut floor, &mut ceil, MIN_SPAN_DB);

        let moved = floor != self.floor || ceil != self.ceil;
        self.floor = floor;
        self.ceil = ceil;
        self.ceil_age = age;
        moved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inverse of `20*log10(mag/65535)`: `mag/65535 = 10^(db/20)`.
    fn db_to_mag(db: f32) -> u16 {
        let v = 10.0_f32.powf(db / 20.0_f32).clamp(0.0, 1.0);
        (v * 65_535.0) as u16
    }

    /// A full `BINS` row: mostly `noise_db`, with (optionally) two `sig_db` spikes.
    fn row(noise_db: f32, sig_db: Option<f32>) -> [u16; BINS] {
        let mut v = [db_to_mag(noise_db); BINS];
        if let Some(s) = sig_db {
            v[40] = db_to_mag(s);
            v[220] = db_to_mag(s);
        }
        v
    }

    /// The display's `lin_to_db` is inverse to `db_to_mag` within u16 quantization.
    #[test]
    fn mag_to_db_is_inverse_of_db_to_mag() {
        for &db in &[-70.0, -50.0, -20.0, -6.0, 0.0] {
            let back = mag_to_db(db_to_mag(db));
            assert!((back - db).abs() < 0.5, "db {db} → {back}");
        }
        assert!((mag_to_db(0).abs() - 100.0).abs() < 1e-6);
        assert!(mag_to_db(65535).abs() < 1e-3);
    }

    /// The estimate is the (median, peak) — the median is the *noise* level
    /// regardless of a few strong carriers; the peak is the signal.
    #[test]
    fn estimate_is_median_and_peak() {
        let (med, max) = estimate(&row(-50.0, Some(0.0))).unwrap();
        assert!((med - (-49.7)).abs() < 0.5, "median was {med}");
        assert!(max.abs() < 0.5, "max was {max}");
    }

    /// Adding more/strange spikes doesn't move the median (the signal count is
    /// a handful of the 320 bins) — only the peak changes.
    #[test]
    fn estimate_median_unaffected_by_spike_count() {
        let (a, _) = estimate(&row(-50.0, Some(0.0))).unwrap();
        let (b, _) = estimate(&row(-50.0, Some(-10.0))).unwrap();
        assert!((a - b).abs() < 1e-3, "median {a} vs {b} must not move");
    }

    #[test]
    fn follower_deadband_holds() {
        let mut val = -55.0;
        assert!(!step_follower(&mut val, -54.0, 0.3, 0.05, 1.0));
        assert_eq!(val, -55.0);
        assert!(!step_follower(&mut val, -55.9, 0.3, 0.05, 1.0));
        assert_eq!(val, -55.0);
    }

    #[test]
    fn follower_attack_is_faster_than_release() {
        let mut a = -55.0;
        let mut r = -55.0;
        step_follower(&mut a, -40.0, 0.30, 0.05, 1.0); // rise
        step_follower(&mut r, -70.0, 0.30, 0.05, 1.0); // fall
        let a_move = (a - -55.0).abs();
        let r_move = (-55.0 - r).abs();
        assert!(
            a_move > r_move,
            "attack {a_move} should exceed release {r_move}"
        );
        assert!(a < -40.0, "approach the target from below, never overshoot");
        assert!(r > -70.0, "approach the target from above, never overshoot");
    }

    #[test]
    fn enforce_min_span_expands_symmetrically() {
        let mut f = -40.0;
        let mut c = -30.0;
        enforce_min_span(&mut f, &mut c, 25.0);
        assert!((c - f - 25.0).abs() < 1e-6);
        assert!((f - (-47.5)).abs() < 1e-6, "floor was {f}");
        assert!((c - (-22.5)).abs() < 1e-6, "ceil was {c}");

        // A healthy span is untouched.
        let mut f2 = -85.0;
        let mut c2 = -15.0;
        enforce_min_span(&mut f2, &mut c2, 25.0);
        assert_eq!((f2, c2), (-85.0, -15.0));
    }

    /// From a far seed window, the first frame begins closing the gap to the
    /// estimate — both edges move toward it.
    #[test]
    fn scale_moves_toward_a_new_signal() {
        let mut s = AutoScale::new();
        s.step(&row(-50.0, Some(0.0)));
        let (f, c) = s.scale();
        assert!(f > -85.0 + 0.5, "floor should rise from the seed, got {f}");
        assert!(c != -15.0, "ceil should move from the seed, got {c}");
    }

    /// A CW-like strobe (keying within the 4 dB hold margin) must NOT make the
    /// ceiling bounce up and down: the ceiling holds at the peak.
    #[test]
    fn strobing_does_not_oscillate_the_ceiling() {
        let mut s = AutoScale::new();
        // Settle with a held −25 dB signal (a couple hundred frames).
        for _ in 0..400 {
            s.step(&row(-50.0, Some(-25.0)));
        }
        let settled = s.scale().1;

        // Key between −25 dB (on) and −28 dB (off), 10 frames each — both within
        // the 4 dB hold margin, so the ceiling must stay frozen.
        let (mut hi, mut lo) = (f32::MIN, f32::MAX);
        for _ in 0..10 {
            for _ in 0..10 {
                s.step(&row(-50.0, Some(-25.0)));
                hi = hi.max(s.scale().1);
                lo = lo.min(s.scale().1);
            }
            for _ in 0..10 {
                s.step(&row(-50.0, Some(-28.0)));
                hi = hi.max(s.scale().1);
                lo = lo.min(s.scale().1);
            }
        }
        let span = (hi - lo).abs();
        assert!(
            span < 1.0,
            "ceil strobed {lo}…{hi} (span {span:.2} dB; settled {settled})"
        );
    }

    /// A genuinely gone signal (falls well below the hold margin) must still
    /// release the ceiling — the ceiling may never be truly stuck above a
    /// signal that has disappeared.
    #[test]
    fn gone_signal_releases_the_ceiling() {
        let mut s = AutoScale::new();
        for _ in 0..400 {
            s.step(&row(-50.0, Some(-20.0)));
        }
        let with = s.scale().1;
        // The signal is now just the −50 dB noise floor (well below the hold
        // margin) → the ceiling releases after the ~12-frame hold window.
        for _ in 0..600 {
            s.step(&row(-50.0, None));
        }
        let after = s.scale().1;
        assert!(
            (with - after).abs() > 5.0,
            "ceil did not release after the signal went away: {with} → {after}"
        );
    }
}
