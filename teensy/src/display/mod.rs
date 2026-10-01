//! ILI9341 panadapter / waterfall. `no_std`.
//!
//! Modules:
//!   * `palette` — dB → RGB565 (waterfall ramp).
//!   * `overlay` — NCO cursor + passband tint composited in DMA staging.
//!   * `glyphs`  — 5×7 font.
//!   * `driver`  — ILI9341 + LPSPI4 + CS + text helpers (task-local).

pub mod driver;
pub mod glyphs;
pub mod overlay;
pub mod palette;

/// Undecorated waterfall history: 320 cols × 120 rows (RGB565).
pub const WF_COLS: usize = 320;
pub const WF_ROWS: usize = 120;

/// Top edge of the waterfall in the bottom half of the 240-row panel.
pub const WF_TOP: u16 = (240 - WF_ROWS) as u16;
