//! ILI9341 panadapter / waterfall. `no_std`.
//!
//! Modules:
//!   * `palette` — dB → RGB565 (waterfall ramp).
//!   * `frame`   — scrolling waterfall band (pure, no hardware).
//!   * `glyphs`  — 5×7 font.
//!   * `driver`  — ILI9341 + LPSPI4 + CS + text helpers (task-local).

pub mod palette;
pub mod frame;
pub mod glyphs;
pub mod driver;

/// LCD is 320×240 landscape. The waterfall occupies the top half;
/// the bottom half is status text.
pub const WIDTH: u16 = 320;
pub const HEIGHT: u16 = 240;
/// Waterfall band: 320 cols × 120 rows (u16 each).
pub const WF_COLS: usize = 320;
pub const WF_ROWS: usize = 120;
/// One row of waterfall: `WF_COLS` u16.
pub const WF_ROW_LEN: usize = WF_COLS;
