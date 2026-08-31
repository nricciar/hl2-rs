//! JS8Call (all speeds) decode.
//!
//! A self-contained port of the `js8call` reference decoder for the four
//! JS8 speeds (A/B/C/E — 15 s / 10 s / 6 s / 30 s cycles, 8-FSK,
//! LDPC(174,87), 12-char varicode payload). Each speed has its own
//! cycle length and per-speed timing/sync constants (`params.rs`); all
//! share one 12 kHz baseband path and one rolling buffer. Decode is
//! readiness-driven (`js8_step`), re-arming each speed independently.
//!
//! See PROTOCOL.md (JS8Call section) for the wire-level description.

pub mod decode;
pub mod decoder;
pub mod frame;
pub mod jsc_map;
pub mod ldpc;
pub mod ldpc_tables;
pub mod msg;
pub mod params;
pub mod sync;

pub use frame::ALPHABET;
pub use ldpc::bpdecode174;
