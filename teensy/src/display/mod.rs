//! ILI9341 panadapter / waterfall. `no_std`.
//!
//! Modules:
//!   * `palette` — dB → RGB565 (waterfall ramp).
//!   * `overlay` — NCO line + passband band composited into the frame.
//!   * `glyphs`  — 5×7 font.
//!   * `driver`  — ILI9341 + LPSPI4 + CS + text helpers (task-local).

pub mod driver;
pub mod glyphs;
pub mod overlay;
pub mod palette;

/// Waterfall band: 320 cols × 120 rows (u16 each).
pub const WF_COLS: usize = 320;
pub const WF_ROWS: usize = 120;

/// Top edge (panel y) of the waterfall band: the *bottom* half of the 240-row
/// panel. The top half (status header + large frequency + S-meter) sits above
/// it (see `main::app` layout constants `HEADER_Y` / `FREQ_Y` / `SMETER_Y`).
pub const WF_TOP: u16 = (240 - WF_ROWS) as u16;
