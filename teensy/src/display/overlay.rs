//! NCO cursor and passband tint, composited without changing waterfall history.

/// Solid amber cursor, RGB565.
pub const NCO_LINE: u16 = 0xFE8A;
/// Light-grey passband tint, RGB565.
pub const PASSBAND_GREY: u16 = 0xDEFB;
/// Passband opacity on a 0..=255 scale.
pub const BAND_ALPHA: u8 = 46;

/// Rounded per-channel RGB565 blend with exact alpha endpoints.
#[inline]
pub fn blend(orig: u16, over: u16, a: u8) -> u16 {
    let a = a as u32;
    // Map alpha to a 0..=32768 fixed-point weight, avoiding division by 255.
    let weight = (a * 257 + (a >> 7)) >> 1;
    let o = orig as u32;
    let n = over as u32;
    let mix = |oc: u32, nc: u32| (nc * weight + oc * (32768 - weight) + 16384) >> 15;
    let r = mix((o >> 11) & 0x1F, (n >> 11) & 0x1F);
    let g = mix((o >> 5) & 0x3F, (n >> 5) & 0x3F);
    let b = mix(o & 0x1F, n & 0x1F);
    ((r << 11) | (g << 5) | b) as u16
}

/// Composite one pixel using ordered, inclusive passband columns.
#[inline]
pub fn pixel(color: u16, column: usize, pb_lo: usize, pb_hi: usize, nco: usize) -> u16 {
    if column == nco {
        NCO_LINE
    } else if (pb_lo..=pb_hi).contains(&column) {
        blend(color, PASSBAND_GREY, BAND_ALPHA)
    } else {
        color
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_endpoints_and_identity() {
        for base in [0, 0x4020, 0xF800, 0x07E0, 0x001F, 0xFFFF] {
            for over in [0, NCO_LINE, PASSBAND_GREY, 0xFFFF] {
                assert_eq!(blend(base, over, 0), base);
                assert_eq!(blend(base, over, 255), over);
            }
            for alpha in 0..=255 {
                assert_eq!(blend(base, base, alpha), base);
            }
        }
    }

    #[test]
    fn blend_monotonic_per_channel() {
        for (base, over) in [(0, NCO_LINE), (0xFFFF, PASSBAND_GREY)] {
            let mut previous = base;
            for alpha in 0..=255 {
                let current = blend(base, over, alpha);
                for (shift, mask) in [(11, 31), (5, 63), (0, 31)] {
                    let old = (previous >> shift) & mask;
                    let new = (current >> shift) & mask;
                    if base == 0 {
                        assert!(new >= old);
                    } else {
                        assert!(new <= old);
                    }
                }
                previous = current;
            }
        }
    }

    #[test]
    fn passband_endpoints_and_cursor_precedence() {
        let base = 0x4020;
        let tinted = blend(base, PASSBAND_GREY, BAND_ALPHA);
        assert_ne!(tinted, base);
        assert_ne!(tinted, PASSBAND_GREY);
        assert_eq!(pixel(base, 2, 3, 7, 5), base);
        assert_eq!(pixel(base, 3, 3, 7, 5), tinted);
        assert_eq!(pixel(base, 7, 3, 7, 5), tinted);
        assert_eq!(pixel(base, 8, 3, 7, 5), base);
        for nco in [0, 3, 5, 7, 9] {
            assert_eq!(pixel(base, nco, 3, 7, nco), NCO_LINE);
        }
        assert_eq!(pixel(base, 0, 0, 0, 5), tinted);
        assert_eq!(pixel(base, 1, 0, 0, 5), base);
        assert_eq!(pixel(base, 9, 0, 9, 5), tinted);
    }

    #[test]
    fn mode_switch_recomposites_unchanged_history() {
        const COLS: usize = 320;
        let history: [u16; COLS * 3] = core::array::from_fn(|i| [0x4020, 0x001F, 0xFFFF][i / COLS]);
        let original = history;
        let nco = COLS / 2;
        // USB, LSB, AM, full-width FM, then USB again.
        for (lo, hi) in [
            (nco, nco + 35),
            (nco - 35, nco),
            (80, 240),
            (0, 319),
            (nco, nco + 35),
        ] {
            for (i, &color) in history.iter().enumerate() {
                let column = i % COLS;
                let expected = if column == nco {
                    NCO_LINE
                } else if (lo..=hi).contains(&column) {
                    blend(original[i], PASSBAND_GREY, BAND_ALPHA)
                } else {
                    original[i]
                };
                assert_eq!(pixel(color, column, lo, hi, nco), expected);
            }
        }
        assert_eq!(history, original);
    }
}
