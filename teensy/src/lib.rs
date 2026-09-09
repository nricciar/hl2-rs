//! `hl2-teensy` — `no_std` spectrum / display / radio stack for the Teensy 4.1.
//!
//! Thin leaf over the workspace's `hl2` protocol crate:
//!   * `spectrum` — FFT + Hann window + max-pool to 320 display bins.
//!   * `display`  — ILI9341 palette, scrolling waterfall band, 5×7 font,
//!                  SPI text helpers (task-local on the `render` task).
//!   * `radio`    — DHCP / discovery / tune / keep-alive + EP6 receive,
//!                  driven through smoltcp + the `hl2::protocol` builders.
//!   * `ethernet` — DP83825 (ENET1) bring-up and the smoltcp device.
//!   * `shared`   — radio ⇄ render watermark row + status line.
//!
//! The bin (`src/main.rs`) is a thin RTIC app: `radio` task pumps the
//! radio and publishes frames; `render` task blits frames to the ILI9341
//! and redraws status. The two tasks are single-core, no locks.

#![no_std]

extern crate alloc;

pub mod spectrum;
pub mod display;
pub mod radio;
pub mod ethernet;
pub mod shared;
