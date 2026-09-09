//! dB → RGB565 palette and the per-frame mags → framebuffer layout.
//!
//! Port of `ui/src/canvas.rs::bin_color` / `color_for_frac` (a 4-segment
//! black→blue→green→yellow→red waterfall palette) to `no_std` + RGB565.

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

/// Palette fraction (0..1) → RGB565. The 4-segment scheme from the UI:
/// black → blue → green → yellow → red (the "waterfall" ramp).
pub fn frac_to_rgb565(frac: f32) -> u16 {
    // The UI uses 4 equal-length segments with 1-bit interpolation. We
    // mirror the endpoints exactly (each pair is a colour on a 5/6-bit
    // grid); the midpoints are just the average of the endpoints (which
    // is exactly `a + (b − a) * 0.5` when `frac * 4` falls mid-segment).
    let f = (frac.clamp(0.0, 1.0) * 4.0).min(3.999_);
    let t = f as i32;
    let g = f - t as f32;
    let l = |a: u16, b: u16| -> u16 {
        let v = (a as f32) + (b as f32 - a as f32) * g;
        (v.round() as u16).min(31)
    };
    match t {
        0 => pack565(l(0, 0), l(0, 0), l(0, 20)),
        1 => pack565(l(0, 0), l(0, 21), l(25, 1)),
        2 => pack565(l(28, 31), l(29, 31), l(13, 0)),
        _ => {
            // 3: yellow → red
            pack565(l(31, 31), l(29, 29), l(13, 1))
        }
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

    /// Monotone brightness across the ramp, and each colour is distinct.
    #[test]
    fn palette_is_monotone_and_distinct() {
        let mut vals = [0u16; 4];
        for (i, v) in vals.iter_mut().enumerate() {
            *v = frac_to_rgb565((i as f32 + 0.5) / 4.0);
        }
        // Each segment's midpoint is a distinct colour.
        assert_eq!(vals[0] != vals[1] && vals[1] != vals[2] && vals[2] != vals[3], true);
        // Roughly monotone (each is darker than the next): use luma.
        let luma = |c: u16| -> u32 {
            let r = ((c >> 11) & 0x1F) as u32;
            let g = ((c >> 5) & 0x3F) as u32;
            let b = (c & 0x1F) as u32;
            r * 9 + g * 9 + b * 9
        };
        for i in 0..3 {
            assert!(
                luma(vals[i + 1]) >= luma(vals[i]),
                "luma should increase: {} vs {}",
                luma(vals[i]),
                luma(vals[i + 1])
            );
        }
    }

    /// `lin_to_db(0)` is the floor; full scale (`65535`) is 0 dB (the
    /// natural reference of `20*log10(x/65535)`, where 1.0 == 0 dB).
    #[test]
    fn lin_to_db_endpoints() {
        assert!(lin_to_db(0) <= LOG_FLOOR);
        assert!((lin_to_db(65535) - 0.0_f32).abs() < 1e-3);
    }
}
