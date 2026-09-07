//! Canvas rendering for the panadapter (live spectrum) and waterfall
//!
//! Both panels draw a frequency ruler so the user can read what the
//! panadapter is actually displaying. An **EP6** (complex, per-slot) source
//! is centred on its NCO: the ruler runs `centre − span/2 … centre + span/2`
//! with a centre cursor at the tune. An **EP4** (real / I-only) wideband
//! source is conjugate-symmetric about DC (0 MHz), so it is drawn *folded*:
//! once, left-to-right, with the left edge at 0 MHz and the right edge at the
//! full span (no mirror, no centre NCO cursor — there is no NCO on EP4).

use web_sys::{CanvasRenderingContext2d, ImageData};

const LOG_FLOOR: f64 = -100.0;
const LOG_CEIL: f64 = 10.0;

fn lin_to_db(x: u16) -> f64 {
    let v = x as f64 / 65535.0;
    if v <= 0.0 {
        LOG_FLOOR
    } else {
        (20.0 * v.log10()).clamp(LOG_FLOOR, LOG_CEIL)
    }
}

fn db_to_frac(db: f64, floor: f64, ceil: f64) -> f64 {
    if db <= floor {
        0.0
    } else if db >= ceil {
        1.0
    } else {
        (db - floor) / (ceil - floor)
    }
}

fn color_for_frac(frac: f64) -> (u8, u8, u8) {
    // black -> blue -> green -> yellow -> red (waterfall palette).
    let t = (frac.clamp(0.0, 1.0) * 4.0).min(3.999) as i32;
    let f = (frac.clamp(0.0, 1.0) * 4.0).min(3.999) - t as f64;
    let l = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * f) as u8;
    match t {
        0 => (l(0, 0), l(0, 0), l(64, 255)),
        1 => (l(0, 0), l(0, 180), l(255, 40)),
        2 => (l(0, 180), l(180, 230), l(40, 0)),
        3 => (l(180, 255), l(230, 40), l(0, 0)),
        _ => unreachable!(),
    }
}

fn bin_color(mag: u16, floor: f64, ceil: f64) -> (u8, u8, u8) {
    color_for_frac(db_to_frac(lin_to_db(mag), floor, ceil))
}

/// Human-readable frequency label for a ruler. kHz for <1 MHz, else MHz, both
/// to one decimal so the axis stays compact. `0` is printed as "0 MHz" so a
/// DC label doesn't carry a meaningless trailing ".0".
fn fmt_hz(hz: f64) -> String {
    if hz.abs() < 0.5 {
        String::from("0 MHz")
    } else if hz.abs() < 1.0e6 {
        format!("{:.1} kHz", hz / 1.0e3)
    } else {
        format!("{:.2} MHz", hz / 1.0e6)
    }
}

/// Draw the frequency ruler for the currently-displayed source.
///
/// * **EP6** (`folded == false`): the source is *centred* on its NCO — so a
///   vertical **centre cursor** (the tune, at `x = w/2`) is drawn, and the
///   ruler carries three labels: the band edges (`centre ± span/2`) on either
///   side and the tune at the centre.
/// * **EP4** (`folded == true`): the source is a *real* (I-only) stream,
///   conjugate-symmetric about DC, so the spectrum is shown **once** (no
///   mirror): 0 MHz on the far left rising to Nyquist (`span / 2`) on the far
///   right. There is no NCO / centre cursor here (the axis is baseband, not
///   centred on a tune), so the ruler shows only the two band edges: `0 MHz`
///   on the left and `span / 2` on the right.
///
/// `center_hz` is the left-edge reference for the centred EP6 axis (a tune
/// offset); it is unused for EP4 (whose left edge is always 0 MHz).
/// `span_hz` is the *full* displayed bandwidth for the source (EP6: both
/// sidebands about the NCO; EP4: the real baseband width, so one visible
/// side is `span / 2`). When `span_hz` is `0` the ruler is skipped.
pub fn draw_center_cursor_and_ruler(
    ctx: &CanvasRenderingContext2d,
    center_hz: Option<u32>,
    span_hz: u32,
    w: usize,
    h: usize,
    top: bool,
    folded: bool,
) {
    if w == 0 || h == 0 {
        return;
    }
    let cx = (w as f64) / 2.0;

    // 1. Centre cursor — the tune / NCO, at the horizontal centre of the band.
    //    EP4 has no NCO, so the cursor only applies to the centred (EP6) view.
    if !folded {
        ctx.set_stroke_style_str("rgba(255, 210, 80, 0.85)");
        ctx.set_line_width(1.0);
        ctx.begin_path();
        ctx.move_to(cx, 0.0);
        ctx.line_to(cx, h as f64);
        let _ = ctx.stroke();
    }

    // 2. Frequency ruler. The labels depend on the axis layout (the band
    //    edges for EP4, plus the centre tune for EP6), but both need the
    //    displayed `span`; skip it when the server hasn't reported one yet.
    let span = span_hz as f64;
    if span <= 0.0 {
        return;
    }
    // Label sits just inside the canvas: near the top for the panadapter, just
    // above the bottom for the waterfall (which scrolls down, so the bottom
    // holds the oldest rows — the label covers those rather than the newest).
    let text_y = if top { 11.0 } else { h as f64 - 2.0 };

    ctx.set_fill_style_str("rgba(240, 245, 255, 0.85)");
    ctx.set_font("11px monospace");

    if folded {
        // EP4 (real / I-only) wideband: drawn once (no mirror). The left edge
        // is DC (0 MHz) and the right edge is Nyquist — half the full real
        // band (`span / 2`) — since only the positive-frequency side is shown.
        // There is no NCO / centre to label, so just the two band edges.
        let left = 0.0;
        let right = span / 2.0;
        ctx.set_text_align("left");
        let _ = ctx.fill_text(&fmt_hz(left), 2.0, text_y);
        ctx.set_text_align("right");
        let _ = ctx.fill_text(&fmt_hz(right), (w as f64) - 2.0, text_y);
        return;
    }

    // EP6 (complex): centred on the NCO. The tune must be known to carry the
    // centre label; without it the two band-edge labels would have no basis.
    let Some(center) = center_hz else { return };
    let left = center as f64 - span / 2.0;
    let right = center as f64 + span / 2.0;
    // Left edge — left-aligned at x = 2.
    ctx.set_text_align("left");
    let _ = ctx.fill_text(&fmt_hz(left), 2.0, text_y);
    // Centre — centred on the cursor.
    ctx.set_text_align("center");
    let _ = ctx.fill_text(&fmt_hz(center as f64), cx, text_y);
    // Right edge — right-aligned at x = w-2.
    ctx.set_text_align("right");
    let _ = ctx.fill_text(&fmt_hz(right), (w as f64) - 2.0, text_y);
}

fn paint(ctx: &CanvasRenderingContext2d, data: &[u8], w: usize, h: usize) {
    if w == 0 || h == 0 || data.is_empty() {
        return;
    }
    let imgd = match ImageData::new_with_u8_clamped_array_and_sh(
        wasm_bindgen::Clamped(data),
        w as u32,
        h as u32,
    ) {
        Ok(v) => v,
        Err(_) => return,
    };
    let _ = ctx.put_image_data(&imgd, 0.0, 0.0);
}

/// Draw the latest frame as a scope-style spectrum: a filled trace under a
/// dB grid. One bin per canvas column; y is mapped linearly between `floor`
/// (bottom) and `ceil` (top).
///
/// `folded` selects the axis layout: `false` (EP6, complex) maps the full
/// `mags` array across the width (centred on the NCO); `true` (EP4, real /
/// I-only) maps only the *positive-frequency half* of the conjugate-symmetric
/// spectrum across the width, left edge = 0 MHz, right edge = Nyquist
/// (`span / 2`) — i.e. the spectrum is drawn once, no mirror.
pub fn draw_panadapter(
    ctx: &CanvasRenderingContext2d,
    mags: &[u16],
    floor: f64,
    ceil: f64,
    w: usize,
    h: usize,
    center_hz: Option<u32>,
    span_hz: u32,
    folded: bool,
) {
    if mags.is_empty() || w == 0 || h == 0 || (ceil - floor) <= 0.0 {
        return;
    }

    // 1. Background fill.
    ctx.set_fill_style_str("black");
    ctx.fill_rect(0.0, 0.0, w as f64, h as f64);

    // 2. Grid: horizontal lines every 10 dB inside [floor, ceil], vertical
    //    lines every one-eighth of the width.
    ctx.set_stroke_style_str("rgba(80, 90, 110, 0.35)");
    ctx.set_line_width(1.0);

    let span = ceil - floor;
    let mut db = (floor / 10.0).floor() * 10.0;
    while db <= ceil {
        if db >= floor {
            let y = h as f64 * (1.0 - (db - floor) / span);
            ctx.begin_path();
            ctx.move_to(0.0, y);
            ctx.line_to(w as f64, y);
            let _ = ctx.stroke();
        }
        db += 10.0;
    }
    for i in 1..8 {
        let x = w as f64 * (i as f64 / 8.0);
        ctx.begin_path();
        ctx.move_to(x, 0.0);
        ctx.line_to(x, h as f64);
        let _ = ctx.stroke();
    }

    // 3. Trace: one sample per canvas column. Map each mag bin to a (x, y)
    //    point using the same dB scale as the grid so the trace lines up.
    //
    // The `bin` index depends on the axis layout. The display `mags` array
    // (from `spectrum::display_mags_into`) has +Nyquist at index 0, DC
    // (0 MHz) at the centre (`len/2`), and −Nyquist at `len−1`.
    //
    // * EP6 (complex, centred on the NCO): the full array runs left→right
    //   across the width, so `x` sweeps `0 .. len`.
    // * EP4 (real / I-only): the spectrum is conjugate-symmetric about DC, so
    //   its two halves are identical (the *mirror* the user sees). Show it
    //   once — left edge = DC (0 MHz), rising to Nyquist (span/2 = fs/2) on
    //   the right — by sweeping only one half: the centre index (`half`) out
    //   to `len−1`.
    let n = w;
    let half = (mags.len() / 2).max(1);
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(n);
    for x in 0..n {
        let frac = x as f64 / n as f64;
        let bin = if folded {
            (half + (frac * half as f64) as usize).min(mags.len() - 1)
        } else {
            ((frac * mags.len() as f64) as usize).min(mags.len() - 1)
        };
        let db = lin_to_db(mags[bin]);
        let y_frac = 1.0 - ((db - floor) / span).clamp(0.0, 1.0);
        pts.push((x as f64, y_frac * h as f64));
    }

    // 4. Semi-transparent fill under the trace (gives a "scope" glow look).
    ctx.set_fill_style_str("rgba(76, 175, 125, 0.18)");
    ctx.begin_path();
    ctx.move_to(0.0, h as f64);
    for &(x, y) in &pts {
        ctx.line_to(x, y);
    }
    ctx.line_to((n - 1) as f64, h as f64);
    ctx.close_path();
    let _ = ctx.fill();

    // 5. Stroke the trace.
    ctx.set_stroke_style_str("rgba(76, 175, 125, 0.95)");
    ctx.set_line_width(1.5);
    ctx.begin_path();
    for (i, &(x, y)) in pts.iter().enumerate() {
        if i == 0 {
            ctx.move_to(x, y);
        } else {
            ctx.line_to(x, y);
        }
    }
    let _ = ctx.stroke();

    // 6. Centre cursor (the tune) + frequency ruler, on top of the trace.
    draw_center_cursor_and_ruler(ctx, center_hz, span_hz, w, h, true, folded);
}

/// Paint the full waterfall from a rolling history of spectrum frames.
///
/// `folded` selects the same axis layout as [`draw_panadapter`]: `false` (EP6,
/// complex) maps the full frame across the width (centred on the NCO); `true`
/// (EP4, real / I-only) maps only the conjugate-symmetric spectrum's
/// positive-frequency half, left edge = 0 MHz, no mirror.
pub fn paint_waterfall(
    ctx: &CanvasRenderingContext2d,
    history: &[Vec<u16>],
    floor: f64,
    ceil: f64,
    w: usize,
    h: usize,
    center_hz: Option<u32>,
    span_hz: u32,
    folded: bool,
) {
    if history.is_empty() || w == 0 || h == 0 {
        return;
    }

    let hist_len = history.len();
    // Canvas row 0 (top) = newest, row h-1 (bottom) = oldest — each new frame
    // draws over the top and pushes older frames downward. If there are
    // fewer history frames than rows, the bottom rows are blank.
    let newest = hist_len.saturating_sub(1);
    let mut data = vec![0u8; w * h * 4];
    for y in 0..h {
        if y > newest {
            continue;
        }
        let row = &history[newest - y];
        if row.is_empty() {
            continue;
        }
        let half = (row.len() / 2).max(1);
        for x in 0..w {
            let frac = x as f64 / w as f64;
            let bin = if folded {
                (half + (frac * half as f64) as usize).min(row.len() - 1)
            } else {
                ((frac * row.len() as f64) as usize).min(row.len() - 1)
            };
            let (r, g, b) = bin_color(row[bin], floor, ceil);
            let idx = (y * w + x) * 4;
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = 255;
        }
    }
    paint(ctx, &data, w, h);
    // Centre cursor + frequency ruler, drawn after the image so they aren't
    // overwritten by `put_image_data`.
    draw_center_cursor_and_ruler(ctx, center_hz, span_hz, w, h, false, folded);
}

/// Shade the virtual-receiver channel-select passband on the spectrum.
pub fn draw_vrx_passband(
    ctx: &CanvasRenderingContext2d,
    sideband: &str,
    bw_hz: u32,
    w: usize,
    h: usize,
    center_hz: Option<u32>,
    span_hz: u32,
    offset_hz: i32,
) {
    let Some(center) = center_hz else { return };
    if bw_hz == 0 || w == 0 || h == 0 {
        return;
    }
    let span = span_hz as f64;
    if span <= 0.0 {
        return;
    }
    let left = center as f64 - span / 2.0;
    let width = span; // full displayed span
    // Pixel column for a frequency, clamped to [0, w−1].
    let xf = |hz: f64| {
        let t = (hz - left) / width;
        (t.clamp(0.0, 1.0) * w as f64)
            .floor()
            .clamp(0.0, (w as f64) - 1.0)
    };
    // The receiver's band is centred relative to `tune + offset`, so shift
    // the displayed centre by that offset before placing the passband.
    let centre = center as f64 + offset_hz as f64;
    let (lo, hi) = if sideband == "lsb" {
        (centre - bw_hz as f64, centre)
    } else if sideband == "am" || sideband == "fm" {
        // AM (DSB-FC) and FM (including NFM / narrow FM): both sidebands
        // pass — a symmetric band about the carrier,
        // `[centre−bw, centre+bw]`.
        (centre - bw_hz as f64, centre + bw_hz as f64)
    } else {
        (centre, centre + bw_hz as f64)
    };
    // Skip passbands entirely off the displayed span (a neighbour slot's
    // receiver parked past the edge) so we don't paint a sliver that
    // overlaps the edge cursor.
    if hi < left || lo > left + span {
        return;
    }
    let x0 = xf(lo);
    let x1 = xf(hi);
    if x1 <= x0 {
        return;
    }
    let color = if sideband == "lsb" {
        "rgba(96, 160, 255, 0.18)" // blue — LSB below the tune
    } else if sideband == "am" {
        "rgba(245, 187, 80, 0.18)" // amber — AM, symmetric about the tune
    } else if sideband == "fm" {
        "rgba(255, 140, 100, 0.18)" // coral — FM/NFM, symmetric about the tune
    } else {
        "rgba(76, 215, 125, 0.18)" // green — USB above the tune
    };
    ctx.set_fill_style_str(color);
    let bw = (x1 - x0 + 1.0).max(1.0);
    ctx.fill_rect(x0 - 0.5, 0.0, bw, h as f64);
    // Hairline at each band edge so a very-narrow passband is still visible.
    ctx.set_stroke_style_str("rgba(255, 255, 255, 0.25)");
    ctx.set_line_width(1.0);
    for &x in &[x0 - 0.5, x1 + 0.5] {
        ctx.begin_path();
        ctx.move_to(x, 0.0);
        ctx.line_to(x, h as f64);
        let _ = ctx.stroke();
    }
}

/// Shade the auto-decoder passbands on the spectrum: one grey, mostly
/// transparent strip per monitored frequency, covering the *upper* sideband
/// (all digital modes — FT8, JS8, FT4 — are USB-only, so the band runs from
/// the carrier up to `carrier + bw`). `bands` is a list of
/// `(freq_hz, bw_hz)` — the caller supplies the mode-appropriate width.
/// A faint hairline marks the carrier itself so the exact tuned frequency
/// stays readable inside the strip.
pub fn draw_auto_passbands(
    ctx: &CanvasRenderingContext2d,
    bands: &[(u32, u32)],
    w: usize,
    h: usize,
    center_hz: Option<u32>,
    span_hz: u32,
) {
    let Some(center) = center_hz else { return };
    if w == 0 || h == 0 {
        return;
    }
    let span = span_hz as f64;
    if span <= 0.0 {
        return;
    }
    let left = center as f64 - span / 2.0;
    let width = span; // full displayed span
    // Pixel column for a frequency, clamped to [0, w−1].
    let xf = |hz: f64| {
        let t = (hz - left) / width;
        (t.clamp(0.0, 1.0) * w as f64)
            .floor()
            .clamp(0.0, (w as f64) - 1.0)
    };
    for &(f, bw) in bands {
        let lo = f as f64;
        let hi = (f + bw) as f64;
        if hi < left || lo > left + span {
            continue; // entirely off-screen
        }
        let x0 = xf(lo);
        let x1 = xf(hi);
        let bw_px = (x1 - x0 + 1.0).max(2.0);
        ctx.set_fill_style_str("rgba(210, 218, 230, 0.16)");
        ctx.fill_rect(x0, 0.0, bw_px, h as f64);
        // Hairline at the carrier (lower edge, the USB reference) so the
        // exact tuned frequency is readable inside the strip.
        let xc = xf(lo);
        ctx.set_stroke_style_str("rgba(210, 218, 230, 0.45)");
        ctx.set_line_width(1.0);
        ctx.begin_path();
        ctx.move_to(xc + 0.5, 0.0);
        ctx.line_to(xc + 0.5, h as f64);
        let _ = ctx.stroke();
    }
}

/// Clear a canvas region to black.
#[allow(dead_code)]
#[inline]
pub fn clear(ctx: &CanvasRenderingContext2d, w: usize, h: usize) {
    ctx.set_fill_style_str("black");
    ctx.fill_rect(0.0, 0.0, w as f64, h as f64);
}
