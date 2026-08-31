//! Hermes-Lite 2 SDR client library.
//!
//! Provides discovery, start/stop, per-receiver tuning, and a streaming
//! receive pump for the Hermes-Lite 2 software-defined radio over Ethernet.

pub mod hl2;
pub mod protocol;
pub mod receiver;

pub use hl2::{
    Hl2, Hl2Event, RX1_ADDR, RX2_ADDR, RX3_ADDR, RX4_ADDR, RX5_ADDR, RX6_ADDR, RX7_ADDR, StartInfo,
    TX1_ADDR, discover, discover_single,
};
pub use protocol::data::{BasebandChunk, IQBlock};
pub use protocol::discovery::DiscoveryInfo;
/// Per-slot EP6 baseband fan-out (one [`BasebandRing`] per active slot).
pub use receiver::fanout::BasebandFanout;

/// RX LNA gain register + default (dB). See PROTOCOL.md §11.3.
pub use protocol::{DEFAULT_LNA_GAIN_DB, LNA_ADDR};
