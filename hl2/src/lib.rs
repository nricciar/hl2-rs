//! Hermes-Lite 2 SDR client library.
//!
//! Provides discovery, start/stop, per-receiver tuning, and a streaming
//! receive pump for the Hermes-Lite 2 software-defined radio over Ethernet.
//!
//! ## Features
//!
//! * `dsp` (default) — the DSP core: SSB/AM/FM mode cores, `Nco`, `F32Fir`,
//!   `PolyphaseDecimator`, `AudioEngine`, `VecSink`/`DropSink`,
//!   `VirtualReceiver`, `IqBlock`, `make_demod`. Requires only `alloc` +
//!   `num-complex`. This is the **`no_std`-usable** demod path — a
//!   single-threaded consumer can drive [`receiver::VirtualReceiver::process`]
//!   over whatever I/Q source is wired (socket, file, a SPSC ring, or a live
//!   EP6 stream).
//! * `std` (default) — heap + sync + trait-object `std::error::Error`.
//!   Required for the multi-threaded `client` (the `Arc<Mutex<...>>` shared
//!   rings / decoders). Off = pure `no_std`.
//! * `client` (default) — the `tokio` control client ([`hl2`]: `Hl2`,
//!   `discover`, `start`/`stop`/`tune`, the receive pump + per-slot
//!   `BasebandFanout`).
//! * `ft8` / `ft4` / `js8` (default, via `digital`) — the WSJT / JS8 slot
//!   decoders. Drop any to slim a build.
//! * `alsa` — ALSA playback sink via `cpal`.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
#[cfg(all(test, not(feature = "std")))]
extern crate std;

/// The `tokio`-based control client ([`hl2`]): one owned, clonable handle to a
/// radio plus a spawned receive pump. Requires `client`.
#[cfg(feature = "client")]
pub mod hl2;
/// Metis / protocol-1 wire codec: packet classes, endpoints, C0–C4 framing,
/// discovery, and the wideband / baseband payload parse + assemble. This is
/// the `no_std` core of the crate (only `num-complex` + `alloc`).
pub mod protocol;
/// Software audio receiver: baseband demod (SSB/AM/FM + digital 12 kHz) and
/// the FT8/FT4/JS8 slot decoders. `dsp` (allocation only) for the demod core;
/// `std` adds the thread-safe `BufSink` / `BasebandFanout` / digital decoders.
#[cfg(feature = "dsp")]
pub mod receiver;

#[cfg(feature = "client")]
pub use hl2::{
    Hl2, Hl2Event, RX1_ADDR, RX2_ADDR, RX3_ADDR, RX4_ADDR, RX5_ADDR, RX6_ADDR, RX7_ADDR, StartInfo,
    TX1_ADDR, discover, discover_single,
};
pub use protocol::data::{BasebandChunk, IQBlock};
pub use protocol::discovery::DiscoveryInfo;
pub use protocol::session::{DatagramKind, Session, SessionConfig};
/// Per-slot EP6 baseband fan-out (one [`BasebandRing`] per active slot).
/// `std`-flavoured multi-reader `Arc` API or a `no_std` single-owner `&mut` API,
/// depending on whether `std` is on; either way it's `dsp`-available.
#[cfg(feature = "dsp")]
pub use receiver::fanout::BasebandFanout;

/// RX LNA gain register + default (dB). See PROTOCOL.md §11.3.
pub use protocol::{DEFAULT_LNA_GAIN_DB, LNA_ADDR};
