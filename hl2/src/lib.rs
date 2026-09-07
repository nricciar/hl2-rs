//! Hermes-Lite 2 SDR client library.
//!
//! Provides discovery, start/stop, per-receiver tuning, and a streaming
//! receive pump for the Hermes-Lite 2 software-defined radio over Ethernet.
//!
//! ## Features
//!
//! * `std` (default) — the full library. Off = the lean `no_std` protocol
//!   core: [`protocol`] plus the std-free slice of [`receiver`].
//! * `client` (default) — the `tokio` control client ([`hl2`]: `Hl2`,
//!   `discover`, `start`/`stop`/`tune`, the receive pump).
//! * `ft8` / `ft4` / `js8` (default, via `digital`) — the WSJT / JS8 slot
//!   decoders. Drop any to slim a build (e.g. `default-features = false,
//!   features = ["std", "client", "ft8", "ft4"]` drops JS8).
//! * `alsa` — ALSA playback sink via `cpal`.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

/// The `tokio`-based control client ([`hl2`]): one owned, clonable handle to a
/// radio plus a spawned receive pump. Requires `client`.
#[cfg(feature = "client")]
pub mod hl2;
/// Metis / protocol-1 wire codec: packet classes, endpoints, C0–C4 framing,
/// discovery, and the wideband / baseband payload parse + assemble. This is
/// the `no_std` core of the crate (only `num-complex` + `alloc`).
pub mod protocol;
/// Software audio receiver: baseband demod (SSB/AM/FM + digital 12 kHz) and
/// the FT8/FT4/JS8 slot decoders. `std` (allocation + sync).
#[cfg(feature = "std")]
pub mod receiver;

#[cfg(feature = "client")]
pub use hl2::{
    Hl2, Hl2Event, RX1_ADDR, RX2_ADDR, RX3_ADDR, RX4_ADDR, RX5_ADDR, RX6_ADDR, RX7_ADDR, StartInfo,
    TX1_ADDR, discover, discover_single,
};
pub use protocol::data::{BasebandChunk, IQBlock};
pub use protocol::discovery::DiscoveryInfo;
/// Per-slot EP6 baseband fan-out (one [`BasebandRing`] per active slot).
#[cfg(feature = "std")]
pub use receiver::fanout::BasebandFanout;

/// RX LNA gain register + default (dB). See PROTOCOL.md §11.3.
pub use protocol::{DEFAULT_LNA_GAIN_DB, LNA_ADDR};
