//! NCO line + passband overlay, composited into the RGB565 waterfall frame.
//!
//! The panel is RGB565 (no alpha channel), so "transparent" is achieved by a
//! per-channel CPU blend of a constant colour over the already-colourised
//! pixels. [`apply`] runs on the render task's offscreen `WF` buffer *after*
//! the new row is colourised and *before* the single eDMA blit, so the
//! overlay rides the existing blit — no extra SPI traffic, no tearing.
//!
//! Geometry note: the waterfall is NCO-centred (`spectrum` keeps the NCO at
//! display column `BINS / 2`), so the NCO line and the mode's passband are a
//! *fixed number of columns* (USB/LSB one-sided, AM/FM double-sided, see
//! `crate::smeter::passband_columns`). They therefore never distort as the
//! user tunes — only the data scrolling under them shifts, which is exactly
//! the "waterfall moves left/right with tuning" behaviour the UI exhibits.

use super::WF_COLS;

/// UI centre-cursor `rgba(255, 210, 80)` → RGB565 `0xFE8A` (amber). Solid
/// (full-alphas) so it reads crisply on top of the passband band.
pub const NCO_LINE: u16 = 0xFE8A;

/// Neutral light-grey (`≈ RGB(222,222,222)`), for the passband band. A
/// light, neutral (R≈G≈B) grey so it tints most frames *toward light*
/// — the visual cue that's what a UI alpha-tint looks like. A dark or
/// chroma tint (e.g. a `0x4xxx` value) would wash out *bright* signal
/// peaks the same way a black overlay does.
///
/// The 5/6/5 bit widths mean some "even" L values don't encode to a
/// perfectly neutral grey (the 6-bit plane has only 0/6/18/32… 4/28-
/// steps). `0xDEFB` is the closest L220 neutral (222, 222, 222 in 8-bit).
pub const PASSBAND_GREY: u16 = 0xDEFB;

/// Passband opacity, 0..255 (≈ 0.18–0.25 alpha) — matches the UI's
/// `rgba(·,·,·,0.18)` overlay on the waterfall. See [`blend`] for the
/// mixing: the underlying water-fall pixel is preserved, the grey is
/// composited lightly on top.
pub const BAND_ALPHA: u8 = 46;

/// Per-channel lerp of `over` onto `orig` by `a` (0..255), honouring the
/// 5/6/5 RGB565 planes. `a=0` → `orig`; `a=255` → `over`.
///
/// Implemented as a fixed-point lerp in the full 16-bit integer domain —
/// one mul/acc per plane, no 255-division (the division is the code-size
/// killer at the ITCM cap; a 255-division per pixel × 3 planes × 120×44
/// px/frame would overflow). The `a*257 + (a>>7)` expansion is exact for
/// `a=0` and `a=255`, and within one ULP for everything in between — far
/// below the 5/6/5 plane's 60-31-60 dynamic range, so nothing visible shifts.
#[inline]
pub fn blend(orig: u16, over: u16, a: u8) -> u16 {
    // `a` is 0..=255 (8-bit). Promote to a 15-bit lerp weight `t` such that
    // `t / 32767 ≈ a / 255`. We use the equivalent fixed point:
    //   t = a * 2^9 + a / 8   (i.e. a * 257 + (a >> 7))
    // Because 256 / 2^16 = 1 / 256 ≈ 1 / 255, so `t * nc + (2^16 - t) * oc
    //   / 2^16` is an 8-bit lerp with at most 1 ULP error.
    let a = a as u32;
    let t = a * 257u32 + (a >> 7); // in 0..=65791 ≈ 0..=16*4096
    let t16 = (t >> 1) as u32; // scale into 0..=32767 (15-bit)
    let m = 32768u32; // 2^15
    let o = orig as u32;
    let n = over as u32;
    // Lerp per plane: (n * t16 + o * (m - t16) + m/2) / m  → round-half-up;
    // m/2 = 16384; m = 32768 is a power of two so / is a shift.
    let mix = |oc: u32, nc: u32| ((nc * t16) + (oc * (m - t16)) + (m >> 1)) >> 15;
    let r = mix((o >> 11) & 0x1F, (n >> 11) & 0x1F); // R plane, 5-bit
    let g = mix((o >> 5) & 0x3F, (n >> 5) & 0x3F); // G plane, 6-bit
    let b = mix(o & 0x1F, n & 0x1F); // B plane, 5-bit
    // Each `mix` output is on 0..=plane_max (since inputs are), so we can
    // pack directly; clamp to the plane width just in case of rounding.
    ((r.min(31) << 11) | (g.min(63) << 5) | b.min(31)) as u16
}

/// Paint the NCO line + passband band into the **newest row only** of `fb`
/// (row 0), in place. The render task shifts the *already-decorated* waterfall
/// down by one row before colourizing a new frame and *after* calling this;
/// the shift carries each row's one-fold blend downward with it, so a given
/// pixel is never blended more than once during its lifetime. Blending every
/// row every frame would compound the alpha (`blend^N → over` for N rows) and
/// wash the band region out to the overlay's own colour after a few dozen
/// frames.
///
/// `fb` is the flattened `[WF_ROWS * WF_COLS]` waterfall (newest row = index 0).
/// `pb_lo`/`pb_hi` are the passband's inclusive column range (from
/// `crate::smeter::passband_columns`, in `BINS` == `WF_COLS` space); `nco` is
/// the NCO column (always `BINS / 2`). The band blends [`PASSBAND_GREY`] over
/// the underlying waterfall pixels at [`BAND_ALPHA`]; the NCO column is set
/// solid so it stays crisp where it meets the band.
pub fn apply(fb: &mut [u16], pb_lo: usize, pb_hi: usize, nco: usize) {
    if fb.is_empty() || WF_COLS == 0 {
        return;
    }
    let lo = pb_lo.min(pb_hi).min(WF_COLS - 1);
    let hi = pb_hi.max(pb_lo).min(WF_COLS - 1);
    let nco_col = nco.min(WF_COLS - 1);
    // Row 0 only. Subsequent rows already carry the prior frame's one-fold
    // blend — the shift-down in the caller carries it into row 1, etc.
    for x in lo..=hi {
        fb[x] = blend(fb[x], PASSBAND_GREY, BAND_ALPHA);
    }
    fb[nco_col] = NCO_LINE;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_endpoints() {
        let base: u16 = 0x4020;
        let over: u16 = 0xF800;
        assert_eq!(blend(base, over, 0), base, "a=0 keeps the original");
        assert_eq!(blend(base, over, 255), over, "a=255 fully replaces");
    }

    #[test]
    fn blend_monotonic_in_alpha() {
        let base: u16 = 0x0000;
        let over: u16 = NCO_LINE;
        let lo = blend(base, over, 32);
        let hi = blend(base, over, 192);
        assert!(
            u32::from(lo) < u32::from(hi),
            "more alpha → closer to the overlay colour ({lo:#06x} < {hi:#06x})"
        );
    }

    #[test]
    fn apply_marks_band_and_line_only_row0() {
        // Start with a fully uniform non-zero base so the "out-of-band
        // unchanged" assertions are meaningful.
        let base: u16 = 0x4020;
        let mut fb = [base; WF_COLS * 3];
        let c = WF_COLS / 2;
        let pb_lo = c + 4; // a one-sided (USB-like) band, right of centre
        let pb_hi = c + 40;
        apply(&mut fb, pb_lo, pb_hi, c);

        // Only row 0 (the fresh frame) is blended. Each in-band column on
        // row 0 must move; the other rows must still equal the base.
        for x in pb_lo..=pb_hi {
            assert_ne!(
                fb[x], base,
                "row-0 band pixel {x} must change from the base"
            );
            for r in 1..3 {
                assert_eq!(
                    fb[r * WF_COLS + x],
                    base,
                    "row-{} band pixel {x} must stay as drawn (shift-down will move it)",
                    r,
                );
            }
        }
        // A column outside the band and other than the NCO column is
        // untouched on row 0 as well.
        let outside = 3usize;
        assert_eq!(
            fb[outside], base,
            "out-of-band col {outside} row 0 must stay base"
        );
        // The NCO line is solid on row 0. (For older rows the caller's
        // shift-down carries the previous frame's NCO line downward, so we
        // only assert row 0 here.)
        assert_eq!(fb[c], NCO_LINE, "NCO line must be stamped on row 0");
        // A column inside the band (c+8 ∈ [c+4, c+40]) is a partial blend of
        // the base and the grey — not either extreme — proving the blend was
        // applied (not a full replace) and did not bleed past the band.
        let blended = fb[c + 8];
        assert!(
            blended != base && blended != PASSBAND_GREY,
            "in-band pixel should be a partial blend, got {blended:#06x}"
        );
    }

    /// The key invariant: apply is idempotent per-row, so after each frame's
    /// `fb.copy_within(.., cols)` shift + fresh `apply(row 0)`, any given
    /// pixel gets at most ONE `blend` over its lifetime — the exact alpha
    /// the operator configured, not `blend^N`.
    #[test]
    fn apply_does_not_compound_across_frames() {
        let base: u16 = 0x4020;
        let c = WF_COLS / 2;
        let pb_lo = c + 4;
        let pb_hi = c + 40;

        // Frame 1: paint row 0, then shift down (simulating what the caller
        // does before the next frame's colorise).
        let mut fb = [base; WF_COLS * 3];
        apply(&mut fb, pb_lo, pb_hi, c);
        let expected_row1 = fb[c + 8]; // the one-fold blend we just made
        fb.copy_within(..WF_COLS * 3 - WF_COLS, WF_COLS);

        // Frame 2: paint row 0 (which is `base`-ish after the shift) and
        // shift again.
        for i in 0..WF_COLS {
            fb[i] = base; // fresh colorized row
        }
        apply(&mut fb, pb_lo, pb_hi, c);
        fb.copy_within(..WF_COLS * 3 - WF_COLS, WF_COLS);

        // Frame 3: same.
        for i in 0..WF_COLS {
            fb[i] = base;
        }
        apply(&mut fb, pb_lo, pb_hi, c);
        fb.copy_within(..WF_COLS * 3 - WF_COLS, WF_COLS);

        // After three frames, the pixel originally at row 0 should be at
        // the bottom (row 2) and still be exactly one-fold blend, NOT
        // `base → blend → blend^2 → blend^3`.
        let aged = fb[2 * WF_COLS + (c + 8)];
        assert_eq!(
            aged, expected_row1,
            "the aged pixel must equal ONE blend of base, not a compounding blend;\
             got {aged:#06x} expected {expected_row1:#06x}"
        );
        // And it must differ from the "compounded" value (blend of blend).
        let compounded = blend(expected_row1, PASSBAND_GREY, BAND_ALPHA);
        assert_ne!(
            aged, compounded,
            "aged pixel must be single-blend, not compounded"
        );
    }

    #[test]
    fn apply_handles_degenerate_band() {
        // A zero-width band (lo == hi) must not panic; the NCO line must
        // still show on its column.
        let mut fb = [0x4020u16; WF_COLS];
        let c = WF_COLS / 2;
        apply(&mut fb, 0, 0, c);
        assert_eq!(fb[c], NCO_LINE);
        assert_ne!(fb[c + 1], NCO_LINE);
    }
}
