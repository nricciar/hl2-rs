//! Auto-decode: a data registry of "digital modes that know which RF
//! frequencies to listen for", so the API layer can spawn a headless
//! decoder per (slot, in-window frequency) for each registered mode
//! without the mode list living in the server.
//!
//! The registry lives in the `hl2` crate (not `hl2-common`) because the
//! frequency tables live next to the decoders they describe
//! ([`super::ft8::KNOWN_FREQS`] / [`super::js8::decoder::KNOWN_FREQS`]).
//! The API (`hl2-api`) maps [`AutoMode`] to `hl2_common::VrxMode` and to
//! a concrete shared decoder + tap + decode-spawner in `spawn_auto` — the
//! same single `match` site `spawn_vrx` already has for
//! `hl2_common::VrxMode`.
//!
//! Adding a new digital mode (FT4, Q65, CONTEST-C…) is:
//!
//! 1. add a variant of [`AutoMode`] + a `known_freqs` entry pointing at
//!    the new decoder module's table,
//! 2. one more arm in the API's `spawn_auto` `match`,
//! 3. (if not already a variant) a new `hl2_common::VrxMode` variant so
//!    the `auto_monitors` wire entry can name it.

/// A digital mode that participates in auto-decode.
///
/// One entry here corresponds to one `hl2_common::VrxMode` variant on the
/// wire (lowercase in `wire()`), and to one
/// decoder/tap/spawner triple in the API layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoMode {
    /// FT8, the 15-second-slot WSJT-family mode
    /// ([`super::Ft8Message`] / [`super::Ft8Decoder`]).
    #[cfg(feature = "ft8")]
    Ft8,
    /// JS8Call, the multi-speed continuous mode
    /// ([`super::Js8Message`] / [`super::js8::Js8Decoder`]).
    #[cfg(feature = "js8")]
    Js8,
    /// FT4, the 7.5-second-slot WSJT-family mode
    /// ([`super::Ft4Message`] / [`super::ft4::Ft4Decoder`]).
    #[cfg(feature = "ft4")]
    Ft4,
}

impl AutoMode {
    /// The wire name, matching `hl2_common::VrxMode`'s lowercase serde
    /// rename — the UI renders it in the "Auto Decode" readout.
    pub const fn wire(&self) -> &'static str {
        match self {
            #[cfg(feature = "ft8")]
            Self::Ft8 => "ft8",
            #[cfg(feature = "js8")]
            Self::Js8 => "js8",
            #[cfg(feature = "ft4")]
            Self::Ft4 => "ft4",
        }
    }

    /// The operator-known frequencies (Hz) this mode listens on. The API
    /// intersects this with the slot's NCO (± EP6 half-span) to decide
    /// which (slot, freq) pairs a decodable target is reachable from.
    /// Adding a band is a one-line change in the decoder module's table.
    pub fn known_freqs(&self) -> &'static [u32] {
        match self {
            #[cfg(feature = "ft8")]
            Self::Ft8 => super::ft8::KNOWN_FREQS,
            #[cfg(feature = "js8")]
            Self::Js8 => super::js8::decoder::KNOWN_FREQS,
            #[cfg(feature = "ft4")]
            Self::Ft4 => super::ft4::KNOWN_FREQS,
        }
    }
}

/// The full registry. Order is the UI render order and the order the API
/// iterates when building a slot's auto-decoder set. Each entry is present
/// only for an enabled digital mode.
pub const AUTO_MODES: &[AutoMode] = &[
    #[cfg(feature = "ft8")]
    AutoMode::Ft8,
    #[cfg(feature = "js8")]
    AutoMode::Js8,
    #[cfg(feature = "ft4")]
    AutoMode::Ft4,
];
