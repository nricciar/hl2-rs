//! Waterfall band — pure function, no ILI9341 dep.
//!
//! The renderer (in the `render` task) owns the ILI9341 and calls
//! [`Waterfall::advance_row`] each spectrum publish, then SPIs the whole
//! framebuffer. See `ui/src/canvas.rs::paint_waterfall` for the intended
//! row ordering (newest row at the *top*, oldest at the bottom).

/// A scrolling waterfall band: width `W`, height `H` per row, `ROWS` rows.
///
/// The band is `W * H * ROWS` u16 pixels (RGB565). Row index 0 is the
/// newest (top) and `ROWS - 1` is the oldest (bottom).
pub struct Waterfall<'a, const W: usize, const H: usize, const ROWS: usize> {
    fb: &'a mut [u16],
}

impl<'a, const W: usize, const H: usize, const ROWS: usize> Waterfall<'a, W, H, ROWS> {
    /// The framebuffer must be at least `W * H * ROWS` u16 long.
    pub fn new(fb: &'a mut [u16]) -> Self {
        assert!(fb.len() >= W * H * ROWS, "Waterfall needs fb ≥ {len}", len = W * H * ROWS);
        Self { fb }
    }

    const fn row_len() -> usize {
        W * H
    }

    /// Advance the scrolling band by one row, writing the new top row from
    /// `mags` (320 bins → 320 columns). Older rows are all shifted down one
    /// row; the bottom row is overwritten.
    ///
    /// `mags.len()` must equal `W` (320). If it's shorter, the remaining
    /// columns are painted to `bin_color(0)` (black).
    pub fn advance_row(&mut self, mags: &[u16]) {
        let stride = Self::row_len();
        let total = W * ROWS;
        if ROWS > 1 {
            self.fb.copy_within(..total - stride, stride);
        }
        // Write the new top row. Column `c` (0..W) is one bin; it spans `H`
        // pixel-rows. We store the *raw* magnitude in every cell so the
        // renderer can re-palette at paint time (keeps the band a pure
        // data buffer).
        for c in 0..W {
            let mag = mags.get(c).copied().unwrap_or(0u16);
            for r in 0..H {
                self.fb[c * H + r] = mag;
            }
        }
    }

    /// A mutable view of the full framebuffer (for the render task to SPI out
    /// via `draw_raw_slice`).
    pub fn buf(&mut self) -> &mut [u16] {
        self.fb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shift preserves the "older rows" at the bottom, and the new row lands
    /// at the top. Use W=2, H=1, ROWS=3 (tiny; still exercises the shift).
    #[test]
    fn shift_preserves_order() {
        let mut fb = [0u16; 2 * 1 * 3];
        // Initial: row0=[a0, a1], row1=[b0, b1], row2=[c0, c1].
        fb[0] = 0; // row0 col0
        fb[1] = 0; // row0 col1
        fb[2] = 1; // row1 col0
        fb[3] = 1; // row1 col1
        fb[4] = 2; // row2 col0
        fb[5] = 2; // row2 col1
        let mut w = Waterfall::<2, 1, 3>::new(&mut fb);
        w.advance_row(&[10u16, 20u16]);
        // Expected:
        //   row0 = [10, 20]  (new)
        //   row1 = [0, 0]    (was row0)
        //   row2 = [1, 1]    (was row1)
        // (the old row2 [2,2] fell off the bottom)
        assert_eq!(fb, [10, 20, 0, 0, 1, 1]);
    }

    /// With H = 2 the top row takes two pixels per column.
    #[test]
    fn h2_rows_work() {
        let mut fb = [0u16; 2 * 2 * 3];
        let mut w = Waterfall::<2, 2, 3>::new(&mut fb);
        w.advance_row(&[10u16, 20u16]);
        // Row 0, col0 = [10,10], col1 = [20,20]. Everything else stays 0.
        assert_eq!(fb, [10, 10, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    /// A short mags slice doesn't panic; missing columns are painted 0.
    #[test]
    fn short_mags_does_not_panic() {
        let mut fb = [0u16; 4 * 1 * 2];
        let mut w = Waterfall::<4, 1, 2>::new(&mut fb);
        w.advance_row(&[5u16]);
        // Row0: col0=5, cols 1..4 = 0. Row1: old row0, still all 0.
        assert_eq!(fb, [5, 0, 0, 0, 0, 0, 0, 0]);
    }
}
