//! HL2 radio stack: UDP control (discovery/start/tune/keepalive) and the
//! EP6 baseband receive path that feeds the spectrum pipeline.
//!
//! Wire bytes live in `hl2::protocol` (AGENTS.md layering rule); this
//! module only sequences them over smoltcp and drives the DSP.

pub mod control;
pub mod rx;

pub use control::Radio;
pub use rx::Rx;
