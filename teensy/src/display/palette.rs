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

/// A u16 magnitude → RGB565 (using the floor/ceil above).
#[inline]
pub fn bin_color(mag: u16) -> u16 {
    frac_to_rgb565(db_to_frac(lin_to_db(mag), LOG_FLOOR, LOG_CEIL))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A strong carrier maps to a non-black colour; the floor is black.
    #[test]
    fn peak_is_visible_and_floor_is_black() {
        assert_ne!(bin_color(60000), 0, "strong signal should not be black");
        assert_eq!(bin_color(0), 0, "floor should be black");
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
