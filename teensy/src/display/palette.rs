//! Magnitude/dB to RGB565 waterfall palette.
//!
//! A four-segment black/blue/green/yellow/red ramp using the UI's dB scale.

use num_traits::float::Float;

/// Floor / ceiling of the dB scale for the display (same as the UI).
pub const LOG_FLOOR: f32 = -100.0;
pub const LOG_CEIL: f32 = 10.0;

/// Linear magnitude value (0..65535) → dB.
#[inline]
pub fn lin_to_db(x: u16) -> f32 {
    let v = x as f32 / 65_535.0_f32;
    if v <= 0.0 {
        LOG_FLOOR
    } else {
        (20.0 * Float::log10(v)).clamp(LOG_FLOOR, LOG_CEIL)
    }
}

#[inline]
fn db_to_frac(db: f32, floor: f32, ceil: f32) -> f32 {
    if db <= floor {
        0.0
    } else if db >= ceil {
        1.0
    } else {
        (db - floor) / (ceil - floor)
    }
}

fn pack565(r: u16, g: u16, b: u16) -> u16 {
    (r << 11) | (g << 5) | b
}

/// Palette fraction (0..1) → RGB565:
/// black → blue → green → yellow → red (the "waterfall" ramp).
pub fn frac_to_rgb565(frac: f32) -> u16 {
    let f = frac.clamp(0.0, 1.0) * 4.0;
    let t = (f as usize).min(3);
    let g = f - t as f32;
    let l = |a: u16, b: u16| -> u16 {
        let v = (a as f32) + (b as f32 - a as f32) * g;
        v.round() as u16
    };
    match t {
        0 => pack565(0, 0, l(0, 31)),
        1 => pack565(0, l(0, 63), l(31, 0)),
        2 => pack565(l(0, 31), 63, 0),
        _ => pack565(31, l(63, 0), 0),
    }
}

/// A u16 magnitude → RGB565 for an *explicit* dB window `[floor, ceil]`.
/// Everything below `floor` is black, everything at/above `ceil` is red, and
/// the band in between spreads across the black/blue/green/yellow/red ramp.
///
/// The window is normally the auto-scale's current `(floor, ceil)` (see
/// `crate::autoscale`), seeded at (−85, −15) before the first frame lands.
/// [`LOG_FLOOR`]/[`LOG_CEIL`] are the clamps `lin_to_db` uses for the
/// magnitude→dB mapping — *not* the render window.
/// `db_to_frac` (and `frac_to_rgb565`) clamp to 0..1, so a degenerate window
/// (`ceil ≤ floor`) just yields black/red — never a panic or NaN.
#[inline]
pub fn bin_color(mag: u16, floor: f32, ceil: f32) -> u16 {
    frac_to_rgb565(db_to_frac(lin_to_db(mag), floor, ceil))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A strong carrier maps to a non-black colour; the floor is black.
    #[test]
    fn peak_is_visible_and_floor_is_black() {
        let (floor, ceil) = (LOG_FLOOR, LOG_CEIL);
        assert_ne!(
            bin_color(60000, floor, ceil),
            0,
            "strong signal should not be black"
        );
        assert_eq!(bin_color(0, floor, ceil), 0, "floor should be black");
    }

    /// Centering the window on a carrier's own level lifts it off the floor:
    /// the same magnitude is black in a window whose floor sits on it, but
    /// mid-ramp in a window centred on it — the whole point of auto floor/ceil
    /// is to spread the *live* band across the ramp.
    #[test]
    fn window_centering_lifts_a_carrier() {
        let mag = 60_000u16;
        let db = lin_to_db(mag); // the carrier's own level
        // A wide window whose floor sits on the carrier → it is black.
        let black = bin_color(mag, db, db + 80.0);
        // A window centred on the carrier's level → it is mid-ramp.
        let lifted = bin_color(mag, db - 5.0, db + 5.0);
        assert_eq!(black, 0, "a carrier on the window floor should be black");
        assert!(
            u32::from(lifted) > u32::from(black),
            "centering should lift the carrier: {black:#06x} → {lifted:#06x}"
        );
    }

    #[test]
    fn palette_endpoints_and_segment_boundaries() {
        for (frac, color) in [
            (0.0, 0x0000),
            (0.25, 0x001F),
            (0.5, 0x07E0),
            (0.75, 0xFFE0),
            (1.0, 0xF800),
        ] {
            assert_eq!(frac_to_rgb565(frac), color);
            assert_eq!(frac_to_rgb565(frac - 0.00001), color);
            assert_eq!(frac_to_rgb565(frac + 0.00001), color);
        }
        assert_eq!(frac_to_rgb565(-1.0), 0x0000);
        assert_eq!(frac_to_rgb565(2.0), 0xF800);
    }

    /// `lin_to_db(0)` is the floor; full scale (`65535`) is 0 dB (the
    /// natural reference of `20*log10(x/65535)`, where 1.0 == 0 dB).
    #[test]
    fn lin_to_db_endpoints() {
        assert!(lin_to_db(0) <= LOG_FLOOR);
        assert!((lin_to_db(65535) - 0.0_f32).abs() < 1e-3);
    }
}
