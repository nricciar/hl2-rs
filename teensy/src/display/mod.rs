//! ILI9341 panadapter / waterfall. `no_std`.
//!
//! Modules:
//!   * `palette` — dB → RGB565 (waterfall ramp).
//!   * `glyphs`  — 5×7 font.
//!   * `driver`  — ILI9341 + LPSPI4 + CS + text helpers (task-local).

pub mod driver;
pub mod glyphs;
pub mod palette;

/// Waterfall band: 320 cols × 120 rows (u16 each).
pub const WF_COLS: usize = 320;
pub const WF_ROWS: usize = 120;
