//! Shared wire types and constants for the HL2 service + UI.
//!
//! Two channels are defined on the WebSocket:
//!
//! * `CH_WIDEBAND` — binary, one frame per wideband spectrum acquisition.
//!   Payload: `[u16 bin_count][16-bit little-endian bin mags ...]`.
//! * `CH_AUDIO` — post-demod mono `i16` audio from one of the virtual
//!   receivers. Payload: `[u16 LE slot][u32 LE seq][u16 LE rate_hz][i16 LE
//!   samples ...]`; the leading slot byte lets several receivers' audio share
//!   the channel and be routed per slot by the client.
//! * `CH_RX_IQ` — reserved for per-receiver DDC IQ (wire slot held; not yet
//!   decoded by the MVP).
//!
//! Control / status traffic uses JSON.

#![allow(dead_code)] // MVP: not every type is used by both sides yet
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::collections::BTreeMap;
use core::fmt;

use serde::{Deserialize, Serialize};

/// WebSocket channel IDs. The first two bytes of a binary frame are this ID
/// (little-endian `u16`).
pub const CH_WIDEBAND: u16 = 0x01;
/// Reserved: per-receiver DDC IQ (24-bit I/Q interleaved, one receiver per
/// sub-frame). Wire slot held even though the MVP does not yet decode it.
pub const CH_RX_IQ: u16 = 0x02;
/// Reserved: post-demod audio (16-bit stereo interleaved). Wire slot held.
pub const CH_AUDIO: u16 = 0x03;

/// Sample bit depth as reported by the hardware via discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SampleFormat {
    Sample12,
    Sample16,
}

/// A single discovered HL2 device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveryInfo {
    pub mac: [u8; 6],
    pub gateware_major: u8,
    pub gateware_minor: u8,
    pub board_id: u8,
    pub rx_count: u8,
    pub is_sending: bool,
    pub sample_16bit: bool,
    pub ip: [u8; 4],
    /// Human-readable source address (e.g. "169.254.19.221:1024").
    pub addr: String,
    #[serde(default)]
    pub in_service: bool,
}

/// Per-receiver NCO state. `slot` is 1-based (RX1 = 1, RX2 = 2, ...).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SlotTuning {
    pub slot: u8,
    pub freq_hz: u32,
}

/// Shared state of the HL2 as seen by all connected WS clients.
///
/// This is what every WS client "owns" in the sense that a tune from one
/// client tunes the hardware for everyone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedState {
    pub started: bool,
    pub rx_count: u8,
    /// Currently set NCO frequency per slot. Missing slot = not set.
    pub tuning: BTreeMap<u8, u32>,
    pub sample_format: SampleFormat,
    pub adc_sample_rate_hz: u32,
    /// RX low-noise-amplifier gain, in dB (register range −12…+48).
    /// Reported back to clients so the UI can mirror the current setting.
    pub lna_gain_db: i8,
    /// RX open-collector filter-bank relay mask, LSB-first: bit 0 = relay 1
    /// … bit 6 = relay 7 (the 7 selectable relays on the companion filter
    /// board, e.g. N2ADR / MRF101).
    #[serde(default)]
    pub oc_bits: u8,
    /// Which receive stream feeds the spectrum (panadapter/waterfall)
    #[serde(default)]
    pub spectrum_source: SpectrumSource,
    /// Monotonic server revision of this snapshot (increments per command).
    #[serde(default)]
    pub state_at: u64,
    /// The NCO frequency (Hz) that the *currently selected* spectrum source is
    /// centred on
    #[serde(default)]
    pub spectrum_center_hz: Option<u32>,
    /// Total displayed baseband bandwidth of the current spectrum source
    #[serde(default)]
    pub spectrum_span_hz: u32,
    /// The active virtual receivers
    #[serde(default)]
    pub vrx: BTreeMap<u8, VrxState>,
    /// Auto-decode monitors (headless, per-slot). Each entry describes one
    /// (slot, target-frequency) that the server is currently running a
    /// digital-mode decoder on. The UI uses this for the readout + the
    /// auto-enabled state of a slot's "Auto Decode" checkbox.
    ///
    /// Keyed by RX slot so the UI can look up "is auto on for this tab?"
    /// with a simple `contains_key`.
    #[serde(default)]
    pub auto_monitors: BTreeMap<u8, Vec<AutoMonitor>>,
    /// Per-slot signal level, in **dB relative to the band spectrum's full
    /// scale** (the same reference as `vrx_floors`). Computed server-side
    /// from the *displayed slot's* band spectrum (the FFT we already run for
    /// the panadapter): the highest magnitude inside the running receiver's
    /// channel passband window around the tuned frequency. This is
    /// *signal + noise* in dB. The S-unit map (S1 = at the floor, +6 dB per
    /// S unit, red past S9 / +20 dB) is applied by the client from
    /// `vrx_levels[slot] − vrx_floors[slot]`. A missing slot = the display
    /// source is not an EP6 per-slot stream (EP4 wideband, or that slot is
    /// not currently being displayed). The value is republished by the
    /// server's periodic state heartbeat (~100 ms), so the UI's S-meter
    /// needle tracks the live reading without any command traffic.
    #[serde(default)]
    pub vrx_levels: BTreeMap<u8, f64>,
    /// Per-slot **band noise floor**, in the same dB reference as
    /// `vrx_levels`. Computed as the 25th percentile of the *whole
    /// displayed band's* magnitudes — a strong signal occupies only a few
    /// of the hundreds of noise bins, so the percentile lands on the noise
    /// regardless of where (or whether) the signal sits, and regardless of
    /// the mode (SSB sideband, AM carrier, FM deviation, FT8 tones — all
    /// read as "elevated spectral energy vs. the noise floor"). The UI uses
    /// `vrx_levels[slot] − vrx_floors[slot]` (dB of signal over the band
    /// floor) as the S-meter gauge's input. Before this field existed the
    /// UI self-calibrated the floor from its own level samples
    /// (`track_floors` in ui/src/app.rs), which broke for carriers
    /// (continuous signals never dip to noise, so the floor crept up to the
    /// carrier and `level − floor → 0`).
    #[serde(default)]
    pub vrx_floors: BTreeMap<u8, f64>,
}

impl SharedState {
    pub fn new(rx_count: u8) -> Self {
        Self {
            started: false,
            rx_count,
            tuning: BTreeMap::new(),
            sample_format: SampleFormat::Sample16,
            adc_sample_rate_hz: 76_800_000,
            lna_gain_db: 6,
            oc_bits: 0,
            spectrum_source: SpectrumSource::default(),
            state_at: 0,
            spectrum_center_hz: None,
            spectrum_span_hz: 0,
            vrx: BTreeMap::new(),
            auto_monitors: BTreeMap::new(),
            vrx_levels: BTreeMap::new(),
            vrx_floors: BTreeMap::new(),
        }
    }
}

/// One auto-decode monitor: a headless digital-mode decoder the server is
/// running against one target frequency within one slot's EP6 window.
///
/// Unlike a virtual receiver (`VrxState`) these produce no audio, are not
/// user-addressable, and can co-exist with a slot's live VRX. Their decoded
/// rows are pushed on the same `ft8log` / `js8log` event envelopes as live
/// vrx decodes; the UI distinguishes them by the embedded
/// [`VrxState::muted`] flag (always `true` for auto-decode rows) or by
/// presence in [`SharedState::auto_monitors`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AutoMonitor {
    /// The RX slot this decoder is attached to (1-based).
    pub slot: u8,
    /// The digital mode (Ft8 / Js8 / Ft4 today).
    pub mode: VrxMode,
    /// The target RF frequency (Hz) being decoded — the operator-known
    /// frequency for this mode + band (e.g. 7_074_000 for FT8 40 m).
    /// The decode's audio-passband centre is this frequency, so a spot's
    /// `freq_hz = freq_hz + m.freq_hz` (the decoder's tone-0 offset).
    pub freq_hz: u32,
}

/// Virtual-receiver mode, for the virtual audio channel.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VrxMode {
    Usb,
    Lsb,
    /// AM (DSB-FC, full-carrier) voice demod.
    Am,
    /// FM (standard, ≈ 15 kHz channel) voice demod (phase-derivative read
    /// of a polyphase-LPF'd complex baseband).
    Fm,
    /// NFM (narrow, ≈ 5 kHz channel) voice demod — same DSP as [`VrxMode::Fm`],
    /// narrower channel-select bandwidth. Amateur-radio "NFM". The on-wire
    /// tag is `"fm_narrow"` (the `lowercase` rename); `"nfm"` is accepted as
    /// an alias so the UI may use either spelling.
    #[serde(alias = "nfm")]
    FmNarrow,
    Ft8,
    Js8,
    Ft4,
}

/// A client's request to create (or update) the single virtual receiver
/// for one RX slot.
///
/// `slot` is the receiver to demodulate (RX1 = 1, RX2 = 2, …). `offset_hz`
/// is the NCO offset applied to that slot's baseband before demodulation —
/// `0` for a receiver centred on the tune (the default); a non-zero value
/// lets one slot host more than one virtual receiver at a later offset.
/// `bw_hz` is the channel-select bandwidth; `gain_db` is applied on top of
/// the receiver's internal AGC.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VrxCfg {
    pub slot: u8,
    #[serde(default)]
    pub offset_hz: i32,
    pub mode: VrxMode,
    pub bw_hz: u32,
    pub gain_db: f32,
}

/// A snapshot of the active virtual receiver, echoed in
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VrxState {
    /// The RX slot this receiver is demodulating (1-based).
    pub slot: u8,
    #[serde(default)]
    pub offset_hz: i32,
    pub mode: VrxMode,
    pub bw_hz: u32,
    pub gain_db: f32,
    /// The audio sample rate (Hz) the server is emitting `CH_AUDIO` at for
    /// this slot.
    pub rate_hz: u32,
    /// Whether this receiver is muted: `true` means the demod + (in FT8 mode)
    /// the decoder keep running but the server skips broadcasting the slot's
    /// `CH_AUDIO` frames.
    #[serde(default)]
    pub muted: bool,
}

/// One decoded digital-mode message (a **row** of the decode log), as pushed
/// by the server to every connected client.
///
/// This is the *mode-agnostic* wire shape shared by every digital decoder
/// (FT8 / FT4 / JS8 today). `text` is the human-readable row: FT8/FT4 carry
/// the resolved 77-bit WSJT payload, JS8 carries the decoder's display
/// string. `vrx` attributes the row to the exact receiver (slot, NCO
/// offset, mode) that demodulated and decoded it — it is the only place the
/// mode of the producing receiver is visible on the wire; the UI reads it
/// from `vrx.mode`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeRow {
    /// The decoded message text / display string (mode-dependent).
    pub text: String,
    /// Decoded tone-0 offset (Hz) in the audio passband — the column
    /// wsjtx/js8call show as the signal's offset, not absolute RF.
    pub freq_hz: f32,
    /// Signal start offset relative to the slot anchor, signed seconds
    /// (`+` = late).
    pub dt_sec: f32,
    /// Estimated SNR (dB, wsjtx reference-bandwidth convention).
    pub snr_db: f32,
    /// The wall-clock slot anchor this decode belongs to (ms since Unix
    /// epoch). FT8 is a multiple of 15 000, FT4 of 7 500, JS8 is the cycle
    /// window start.
    pub slot_ms: u64,
    /// The virtual receiver that produced this decode (slot / mode / offset).
    pub vrx: VrxState,
}

/// A batch of decoded messages from one slot / cycle window, pushed over the
/// WebSocket as a JSON text frame with envelope
/// `{"cmd":"log","data":[{"text":…, …}, …]}`.
///
/// There is exactly one envelope for all digital modes — the server maps
/// every mode's decoded rows to [`DecodeRow`] before broadcasting, so the
/// client never branches on mode to receive a log row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeLog {
    pub decodes: Vec<DecodeRow>,
}

// ────────────────────────────────────────────────────────────────────────────
// Decoded-message product + spot sink (the mode-agnostic junction shared by
// hl2 (impls), hl2-api (the sink) and — indirectly — the hl2-ui row shape.
// The `hl2` crate implements [`DecodedMessage`] per mode; `hl2-api`
// implements [`SpotSink`] for its PSK Reporter sender. See PROTOCOL.md /
// AGENTS.md "Refactor: Spotting".

/// Our own spot station ("us"): the callsign + grid of the local rig,
/// supplied to [`DecodedMessage::spot_fields`] so it can decide whether a
/// decode is a self-spot and suppress it. Mode-agnostic: the caller is
/// whatever the message says it is, the locator is whatever the message
/// says it is; the station is only used to *reject* self-spots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpotStation {
    /// Our callsign, e.g. `"K9ABC"`. May be empty (no station configured) in
    /// which case self-spot suppression is a no-op.
    pub callsign: String,
    /// Our grid (Maidenhead), e.g. `"FN31pq"`. May be empty.
    pub grid: String,
}

impl SpotStation {
    pub fn new(callsign: impl Into<String>, grid: impl Into<String>) -> Self {
        Self {
            callsign: callsign.into(),
            grid: grid.into(),
        }
    }
}

/// The extracted identifying "who / where" of a decoded message that is
/// worth spotting. `caller` is the callsign, `locator` the grid square (or
/// empty when the message carried none).
///
/// Everything else in a wire spot (absolute frequency, SNR, epoch, mode)
/// is derived from the [`DecodedMessage`] envelope + the local tune by the
/// sink, which is mode-agnostic — so it lives here, not in each impl.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpotFields {
    pub caller: String,
    pub locator: String,
}

/// The mode-agnostic decoded-message product: the shared interface over
/// every digital decoder (FT8 / FT4 / JS8 / …).
///
/// Implemented by each mode's concrete message type in the `hl2` crate
/// (`Ft8Message`, `Ft4Message`, `Js8Message`). A consumer (e.g. `hl2-api`)
/// only ever sees this trait — never the concrete types — and thus never
/// branches on mode to log or spot a decode. The *only* mode-specific
/// surface is [`spot_fields`]: each impl knows how to pull a
/// callsign + locator out of its own message shape (FT8/FT4 token-parsing,
/// JS8 structured fields), and how to suppress self-spots against the
/// supplied station.
///
/// Object-safe: `dyn DecodedMessage` is the unit of the decode pipeline.
pub trait DecodedMessage {
    /// Decoded tone-0 offset (Hz) in the audio passband.
    fn freq_hz(&self) -> f32;
    /// Signal start offset relative to the slot anchor (signed seconds).
    fn dt_sec(&self) -> f32;
    /// Estimated SNR (dB, wsjtx reference-bandwidth convention).
    fn snr_db(&self) -> f32;
    /// The wall-clock slot anchor this decode belongs to (ms since Unix).
    fn slot_ms(&self) -> u64;
    /// The ADIF mode tag for this message's mode, e.g. `"FT8"` / `"FT4"` /
    /// `"JS8"`. The sink uses this as the wire mode field.
    fn mode(&self) -> &'static str;
    /// The human-readable text of this message: FT8/FT4 → the 77-bit
    /// resolved payload (`"CQ DE W1AW"`); JS8 → the decoder's display
    /// string. This becomes the [`DecodeRow::text`] shown in the UI log.
    fn display(&self) -> &str;
    /// Extract the spot fields (who / where) from this message, or `None`
    /// if it is not worth spotting (no sender, or a self-spot against
    /// `st`). Each impl bakes in its mode's own message grammar *and* its
    /// self-spot rule, so callers are never `if ft8 … if js8`.
    fn spot_fields(&self, st: &SpotStation) -> Option<SpotFields>;
}

/// A place to send a decoded message that qualifies as a spot. The sink is
/// mode-agnostic: given any [`DecodedMessage`] + the local RF tune
/// (`rf_hz`, the passband centre it was decoded against — `rf_hz == 0` means
/// "not tuned, do not spot") it derives a full wire spot (caller, locator,
/// absolute frequency, SNR, epoch, mode) and accepts or rejects it
/// (e.g. per `(caller, band)` dedup / queue cap).
///
/// The only sink today is the PSK Reporter sender in `hl2-api`; more may be
/// added (e.g. a log file, an internal database) without changing the
/// pipeline.
pub trait SpotSink {
    /// Report one decoded message as a spot, if worth posting.
    /// `rf_hz` is the passband centre the message was decoded against
    /// (absolute-RF reference); `rf_hz == 0` means "tune unknown, skip".
    /// Returns `true` if the spot is now pending in the sink.
    fn spot(&mut self, m: &dyn DecodedMessage, rf_hz: u32) -> bool;
}

/// Shared helpers for the spot-selection grammar (Maidenhead validation,
/// mobile-portable base-callsign stripping) — used by the `DecodedMessage`
/// impls in `hl2` and tested in `hl2` / `hl2-api`.
pub mod spot {
    /// Strip a mobile / portable prefix from a callsign: the base call to
    /// the right of the last `/` (e.g. `"W1AW/K9ABC"` → `"K9ABC"`,
    /// `"K9ABC"` → `"K9ABC"`). Used for self-spot comparison so a message
    /// from a base call with a POTA suffix still suppresses.
    pub fn base_callsign(call: &str) -> &str {
        match call.rfind('/') {
            Some(j) => &call[j + 1..],
            None => call,
        }
    }

    /// True if `g` is a valid Maidenhead **grid square** (4- or 6-char):
    /// `[A-R]{2}[0-9]{2}` with optional `[A-X]{2}` suffix, case-insensitive,
    /// excluding the well-known `"RR73"` (a 73 / "best regards" greeting,
    /// not a locator). Mirrors wsjtx's `grid_regexp`
    /// (widgets/mainwindow.cpp:308) — this is the locator gate for the
    /// WSJT-family (`"CQ R7IW LN35"` style) messages and the *usable*
    /// locator gate for JS8 frames (a non-square grid becomes an empty
    /// locator on the wire spot, but the callsign still posts).
    pub fn grid_is_square(g: &str) -> bool {
        let b = g.as_bytes();
        if b.len() != 4 && b.len() != 6 {
            return false;
        }
        let sq =
            |c: u8| c.to_ascii_uppercase().is_ascii_uppercase() && c.to_ascii_uppercase() <= b'R';
        if !sq(b[0]) || !sq(b[1]) || !b[2].is_ascii_digit() || !b[3].is_ascii_digit() {
            return false;
        }
        if b.len() == 6 {
            let sub = |c: u8| {
                c.to_ascii_uppercase().is_ascii_uppercase() && c.to_ascii_uppercase() <= b'X'
            };
            if !sub(b[4]) || !sub(b[5]) {
                return false;
            }
        }
        !(b[0].to_ascii_uppercase() == b'R'
            && b[1].to_ascii_uppercase() == b'R'
            && b[2].to_ascii_uppercase() == b'7'
            && b[3].to_ascii_uppercase() == b'3')
    }
}

/// Choose which per-adapter stream the server's spectrum pipeline (panadapter
/// + waterfall) consumes.
///
/// * `Ep4`   — the wideband real sample stream (122.88 MSps, full bandwidth).
/// * `Ep6(slot)` — the DDC'd complex baseband of one receiver slot, at the
///   per-receiver DDC rate programmed at Start. `slot` is 1-based (RX1…RX7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpectrumSource {
    /// Raw wideband EP4 stream.
    Ep4,
    /// DDC'd complex baseband of the given slot (EP6).
    Ep6 { slot: u8 },
}

impl Default for SpectrumSource {
    fn default() -> Self {
        Self::Ep4
    }
}

/// Commands a client may send. `id` is an opaque 64-bit token used to match
/// responses back to the original request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "cmd", content = "data", rename_all = "lowercase")]
pub enum ClientCmd {
    /// Run a discovery sweep and reply with the found devices.
    Discover,
    /// Start the HL2
    Start {
        initial_tuning: Option<BTreeMap<u8, u32>>,
    },
    /// Stop the HL2 (all clients see `started=false`).
    Stop,
    /// Write the NCO for a single receiver slot.
    Tune { slot: u8, freq_hz: u32 },
    /// Set the RX low-noise-amplifier gain (dB). Range −12…+48
    SetLnaGain { gain_db: i8 },
    /// Set the RX open-collector filter-bank relay mask (LSB-first: bit 0 =
    /// relay 1 … bit 6 = relay 7; take effect on the next keep-alive, ≤40 ms).
    SetOcBits { oc_bits: u8 },
    /// Switch which stream feeds the panadapter/waterfall spectrum pipeline:
    /// `Ep4` (raw wideband) or `Ep6 { slot }` (per-slot baseband).
    SetSpectrumSource { source: SpectrumSource },
    /// Create / update / destroy the virtual receiver for one RX slot
    /// (audio channel).
    SetVrx { cfg: Option<VrxCfg> },
    /// Mute / unmute the virtual receiver on one RX slot without rebuilding
    SetVrxMute { slot: u8, muted: bool },
    /// Tear down the virtual receiver
    SetVrxOff { slot: u8 },
    /// Fetch the current shared state.
    State,
    /// Enable / disable the auto-decoder on one RX slot. When enabled the
    /// server spawns a headless decoder per known digital-mode
    /// frequency in the slot's EP6 window (e.g. FT8 @ 7.074 and JS8 @ 7.078
    /// when the NCO is near 7 MHz); when disabled it tears them down.
    ///
    /// Enabling also locks the slot's NCO in the UI (the stepper is
    /// disabled while auto is on) so the decoders don't have to be
    /// rebuilt on every 10 Hz nudge. A `Tune` command on a slot that has
    /// auto-decode on (from another client, or after a UI race) tears the
    /// slot's auto-decoders down (the NCO and the built decoders would
    /// disagree) — the checkbox mirrors `auto_monitors`, so it flips to
    /// off in the UI; the operator re-enables to rebuild them.
    AutoDecode { slot: u8, enabled: bool },
}

/// Response envelope: every command gets exactly one JSON response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerResponse {
    /// Echo of the `id` from the request, so the UI can correlate.
    pub id: u64,
    pub ack: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub state: SharedState,
    /// Populated on `Discover` reply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub devices: Option<Vec<DiscoveryInfo>>,
}

impl ServerResponse {
    pub fn ok(id: u64, state: &SharedState) -> Self {
        Self {
            id,
            ack: true,
            error: None,
            state: state.clone(),
            devices: None,
        }
    }

    pub fn err(id: u64, state: &SharedState, msg: impl Into<String>) -> Self {
        Self {
            id,
            ack: false,
            error: Some(msg.into()),
            state: state.clone(),
            devices: None,
        }
    }

    pub fn with_devices(mut self, devices: Vec<DiscoveryInfo>) -> Self {
        self.devices = Some(devices);
        self
    }

    /// Not a reply to any specific client request. The UI does not match this
    /// against a pending `cmd()` id — it just applies `state`. Used for the
    /// on-connect snapshot and periodic heartbeats.
    pub const WELCOME_ID: u64 = u64::MAX;

    pub fn welcome(state: &SharedState) -> Self {
        Self {
            id: Self::WELCOME_ID,
            ack: true,
            error: None,
            state: state.clone(),
            devices: None,
        }
    }
}

/// Binary spectrum frame, as sent on `CH_WIDEBAND`.
///
/// Layout:
/// ```text
///   [0..2)   : u16 LE bin_count (=1024 for the MVP)
///   [2..2+2*bin_count): u16 LE magnitude per bin
/// ```
///
/// Values are raw `|X[k]|` from the real-stream FFT (no dB, no floor); the UI
/// is expected to apply its own log / dB scaling and colormap.
#[derive(Debug, Clone)]
pub struct SpectrumFrame {
    /// Block sequence start (radio-provided; useful to the UI for correlation
    /// with a future audio frame).
    pub seq_start: u32,
    /// Number of bins (1024 in the MVP).
    pub bin_count: u16,
    /// Per-bin magnitudes, in `bin_count` bins.
    pub mags: Vec<u16>,
}

impl SpectrumFrame {
    /// Encode to the wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + 2 * self.bin_count as usize);
        out.extend_from_slice(&self.bin_count.to_le_bytes());
        for b in &self.mags {
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }
}

/// Binary audio frame, sent on [`CH_AUDIO`].
///
/// Layout:
/// ```text
///   [0..2)  : u16 LE RX slot (1-based; which virtual receiver this is)
///   [2..6)  : u32 LE frame seq (monotonic per-slot server counter)
///   [6..8)  : u16 LE sample rate (Hz)
///   [8..)   : i16 LE mono samples (samples.len() × 2)
/// ```
#[derive(Debug, Clone)]
pub struct AudioFrame {
    /// The RX slot (1-based) this audio block is demodulated from.
    pub slot: u16,
    /// Monotonic server frame counter (correlates with a virtual-receiver
    /// session; the UI discards frames whose rate doesn't match the last
    /// configured one).
    pub seq: u32,
    /// Sample rate of `samples` in Hz (typically 4 800).
    pub rate_hz: u16,
    /// Mono `i16` audio samples at `rate_hz`.
    pub samples: Vec<i16>,
}

impl AudioFrame {
    /// Encode to the wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + 2 * self.samples.len());
        out.extend_from_slice(&self.slot.to_le_bytes());
        out.extend_from_slice(&self.seq.to_le_bytes());
        out.extend_from_slice(&self.rate_hz.to_le_bytes());
        for s in &self.samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }
}

/// Encode a server→client binary frame (channel ID + payload bytes).
pub fn encode_ws_binary(channel: u16, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + payload.len());
    out.extend_from_slice(&channel.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Decode a client→server binary frame. Returns `(channel, payload)`.
pub fn decode_ws_binary(bytes: &[u8]) -> Option<(u16, &[u8])> {
    if bytes.len() < 2 {
        return None;
    }
    let channel = u16::from_le_bytes([bytes[0], bytes[1]]);
    Some((channel, &bytes[2..]))
}

pub struct Hl2Error;

impl fmt::Display for Hl2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hl2Error")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_server_response() {
        let s = SharedState::new(4);
        let r = ServerResponse::ok(42, &s);
        let j = serde_json::to_string(&r).unwrap();
        let p: ServerResponse = serde_json::from_str(&j).unwrap();
        assert_eq!(p.id, 42);
        assert!(p.ack);
    }

    #[test]
    fn setocbits_cmd_shape() {
        // The UI emits `{"id":N,"cmd":"setocbits","data":{"oc_bits":B}}` —
        // the inner-tag + content `ClientCmd::SetOcBits` must decode exactly
        // that (wire name `setocbits`, `oc_bits` in the `data` object).
        let cmd = r#"{"cmd":"setocbits","data":{"oc_bits":8}}"#;
        let c: ClientCmd = serde_json::from_str(cmd).unwrap();
        assert_eq!(c, ClientCmd::SetOcBits { oc_bits: 8 });

        // And the shared-state field round-trips (UI mirrors this on connect).
        let mut s = SharedState::new(4);
        s.oc_bits = 0b0101100; // relays 1, 2, 4, 6 (== 44)
        let j = serde_json::to_string(&s).unwrap();
        // `oc_bits:44` appears literally.
        assert!(j.contains(r#""oc_bits":44"#), "got: {j}");
        let p: SharedState = serde_json::from_str(&j).unwrap();
        assert_eq!(p.oc_bits, 0b0101100);
    }

    #[test]
    fn spectrum_source_serde_shape() {
        // The UI builds and consumes these JSON fragments by hand, so the shape
        // must be stable and match serde's externally-tagged encoding for the
        // unit/struct variants.
        let e4 = serde_json::to_string(&SpectrumSource::Ep4).unwrap();
        let e6 = serde_json::to_string(&SpectrumSource::Ep6 { slot: 2 }).unwrap();
        assert_eq!(e4, "\"ep4\"");
        assert_eq!(e6, "{\"ep6\":{\"slot\":2}}");
        // And the deserializer accepts exactly that shape — i.e. the UI's
        // `setSpectrumSource` wire format is valid on the server.
        assert_eq!(
            serde_json::from_str::<SpectrumSource>("\"ep4\"").unwrap(),
            SpectrumSource::Ep4
        );
        assert_eq!(
            serde_json::from_str::<SpectrumSource>("{\"ep6\":{\"slot\":1}}").unwrap(),
            SpectrumSource::Ep6 { slot: 1 }
        );
        // Full command envelope, as the UI emits it.
        let cmd = r#"{"cmd":"setspectrumsource","data":{"source":{"ep6":{"slot":1}}}}"#;
        let c: ClientCmd = serde_json::from_str(cmd).unwrap();
        assert!(matches!(
            c,
            ClientCmd::SetSpectrumSource {
                source: SpectrumSource::Ep6 { slot: 1 }
            }
        ));
    }

    #[test]
    fn start_cmd_initial_tuning_shape() {
        // The UI emits `{"cmd":"start","data":{"initial_tuning":{"1":14074000}}}`
        // (a BTreeMap<u8,u32> keyed by the RX slot number — a JSON *string* key,
        // since object keys are always strings). With the inner-tag + content
        // encoding on `ClientCmd` it must decode to slot 1 → 14.074 MHz, which
        // is the default start frequency the UI now sends.
        let cmd = r#"{"cmd":"start","data":{"initial_tuning":{"1":14074000}}}"#;
        let c: ClientCmd = serde_json::from_str(cmd).unwrap();
        let ClientCmd::Start { initial_tuning } = c else {
            panic!("expected Start, got {c:?}");
        };
        let mut expected = std::collections::BTreeMap::new();
        expected.insert(1u8, 14_074_000u32);
        assert_eq!(initial_tuning.as_ref(), Some(&expected));

        // And `null` (the legacy no-initial-tune form) still decodes.
        let cmd = r#"{"cmd":"start","data":{"initial_tuning":null}}"#;
        let c: ClientCmd = serde_json::from_str(cmd).unwrap();
        let ClientCmd::Start { initial_tuning } = c else {
            panic!("expected Start, got {c:?}");
        };
        assert_eq!(initial_tuning, None);
    }

    #[test]
    fn binary_decode_roundtrip() {
        let frame = SpectrumFrame {
            seq_start: 7,
            bin_count: 1024,
            mags: (0..1024).map(|i| (i as u16) % 1000).collect(),
        };
        let bytes = encode_ws_binary(CH_WIDEBAND, &frame.to_bytes());
        let (ch, payload) = decode_ws_binary(&bytes).unwrap();
        assert_eq!(ch, CH_WIDEBAND);
        assert_eq!(payload.len(), 2 + 2 * 1024);
    }

    #[test]
    fn setvrx_cmd_shape() {
        // `{"cmd":"setvrx","data":{"cfg":{"slot":1,"offset_hz":0,"mode":"usb","bw_hz":2600,"gain_db":-3.0}}}`
        // must decode to `ClientCmd::SetVrx { cfg: Some(…) }`.
        let c: ClientCmd = serde_json::from_str(
            r#"{"cmd":"setvrx","data":{"cfg":{"slot":1,"mode":"usb","bw_hz":2600,"gain_db":-3.0}}}"#,
        )
        .unwrap();
        assert_eq!(
            c,
            ClientCmd::SetVrx {
                cfg: Some(VrxCfg {
                    slot: 1,
                    offset_hz: 0,
                    mode: VrxMode::Usb,
                    bw_hz: 2_600,
                    gain_db: -3.0,
                })
            }
        );
        // A non-zero NCO offset decodes too (a receiver not at the slot tune).
        let c: ClientCmd = serde_json::from_str(
            r#"{"cmd":"setvrx","data":{"cfg":{"slot":2,"offset_hz":1500,"mode":"ft8","bw_hz":2600,"gain_db":0.0}}}"#,
        )
        .unwrap();
        assert_eq!(
            c,
            ClientCmd::SetVrx {
                cfg: Some(VrxCfg {
                    slot: 2,
                    offset_hz: 1_500,
                    mode: VrxMode::Ft8,
                    bw_hz: 2_600,
                    gain_db: 0.0,
                })
            }
        );
        // `cfg: null` is the "turn off" form.
        let c: ClientCmd = serde_json::from_str(r#"{"cmd":"setvrx","data":{"cfg":null}}"#).unwrap();
        assert_eq!(c, ClientCmd::SetVrx { cfg: None });
    }

    #[test]
    fn setvrx_mute_off_cmd_shape() {
        // `setvrxmute` — mute/unmute a slot's receiver without a rebuild.
        let c: ClientCmd =
            serde_json::from_str(r#"{"cmd":"setvrxmute","data":{"slot":2,"muted":true}}"#).unwrap();
        assert_eq!(
            c,
            ClientCmd::SetVrxMute {
                slot: 2,
                muted: true
            }
        );
        // `setvrxoff` — the explicit, slot-addressed teardown.
        let c: ClientCmd =
            serde_json::from_str(r#"{"cmd":"setvrxoff","data":{"slot":3}}"#).unwrap();
        assert_eq!(c, ClientCmd::SetVrxOff { slot: 3 });
    }

    #[test]
    fn vrx_state_roundtrip_and_default() {
        let s = VrxState {
            slot: 1,
            offset_hz: 0,
            mode: VrxMode::Lsb,
            bw_hz: 2_600,
            gain_db: 0.0,
            rate_hz: 4_800,
            muted: false,
        };
        let j = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<VrxState>(&j).unwrap(), s);
        let ft8 = VrxState {
            slot: 2,
            offset_hz: 1_500,
            mode: VrxMode::Ft8,
            bw_hz: 2_600,
            gain_db: 0.0,
            rate_hz: 12_000,
            muted: true,
        };
        assert_eq!(
            serde_json::from_str::<VrxState>(&serde_json::to_string(&ft8).unwrap()).unwrap(),
            ft8
        );
        // An older payload without `offset_hz` / `muted` still decodes
        // (both default) — backward compatibility.
        let legacy = r#"{"slot":1,"mode":"usb","bw_hz":2600,"gain_db":0.0,"rate_hz":4800}"#;
        let p: VrxState = serde_json::from_str(legacy).unwrap();
        assert_eq!(p.offset_hz, 0);
        assert!(!p.muted);
        // `SharedState.vrx` is now a per-slot map: it must tolerate a
        // *missing* `vrx` (old server) by defaulting to empty, and carry a
        // keyed entry per active slot when present.
        let mut st = SharedState::new(4);
        let mut vrx = std::collections::BTreeMap::new();
        vrx.insert(1u8, s);
        vrx.insert(2u8, ft8);
        st.vrx = vrx.clone();
        let j = serde_json::to_string(&st).unwrap();
        let p = serde_json::from_str::<SharedState>(&j).unwrap();
        assert_eq!(p.vrx, vrx);
        let mut obj: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&j).unwrap();
        obj.remove("vrx");
        let p = serde_json::from_value::<SharedState>(serde_json::Value::Object(obj)).unwrap();
        assert!(p.vrx.is_empty());
    }

    #[test]
    fn vrx_levels_roundtrip_and_default() {
        // `vrx_levels` is a per-slot map of signal levels (dB FS) the UI
        // renders as the S-meter. It must round-trip and default to empty
        // when absent (older servers) so the UI gracefully shows no meter.
        let mut st = SharedState::new(4);
        let mut lv = std::collections::BTreeMap::new();
        lv.insert(1u8, -42.5_f64);
        lv.insert(2u8, -30.0_f64);
        st.vrx_levels = lv.clone();
        let j = serde_json::to_string(&st).unwrap();
        let p = serde_json::from_str::<SharedState>(&j).unwrap();
        assert_eq!(p.vrx_levels, lv);
        // A value appears literally in the serialized form (keys are strings,
        // values are JSON numbers).
        assert!(j.contains(r#""1":-42.5"#), "got: {j}");
        // An older payload without `vrx_levels` still decodes (defaults empty),
        // and the empty map round-trips.
        let legacy = r#"{"started":true,"rx_count":4,"tuning":{},"sample_format":"sample16","adc_sample_rate_hz":76800000,"lna_gain_db":6,"oc_bits":0,"spectrum_source":"ep4","state_at":0}"#;
        let p = serde_json::from_str::<SharedState>(legacy).unwrap();
        assert!(p.vrx_levels.is_empty());
        let e = serde_json::to_string(&SharedState::new(4)).unwrap();
        assert!(e.contains(r#""vrx_levels":{}"#), "got: {e}");
    }

    #[test]
    fn vrx_floors_roundtrip_and_default() {
        // `vrx_floors` mirrors `vrx_levels` (per-slot, dB FS) — the server's
        // noise-floor estimate the UI pairs with the raw level to render the
        // S-meter. It must round-trip and default to empty when absent
        // (older servers).
        let mut st = SharedState::new(4);
        let mut fl = std::collections::BTreeMap::new();
        fl.insert(1u8, -62.3_f64);
        fl.insert(2u8, -71.0_f64);
        st.vrx_floors = fl.clone();
        let j = serde_json::to_string(&st).unwrap();
        let p = serde_json::from_str::<SharedState>(&j).unwrap();
        assert_eq!(p.vrx_floors, fl);
        // Backward-compat: a legacy payload without `vrx_floors` still decodes.
        let legacy = r#"{"started":true,"rx_count":4,"tuning":{},"sample_format":"sample16","adc_sample_rate_hz":76800000,"lna_gain_db":6,"oc_bits":0,"spectrum_source":"ep4","state_at":0,"vrx_levels":{"1":-40.5}}"#;
        let p = serde_json::from_str::<SharedState>(legacy).unwrap();
        assert!(p.vrx_floors.is_empty());
        assert!(!p.vrx_levels.is_empty());
        // And the empty map round-trips in the serialised form.
        let e = serde_json::to_string(&SharedState::new(4)).unwrap();
        assert!(e.contains(r#""vrx_floors":{}"#), "got: {e}");
    }

    #[test]
    fn decode_row_roundtrip() {
        // One unified row: FT8-shaped (mode visible only via `vrx.mode`) and
        // a JS8-shaped row in the same batch — proving the wire carries both
        // under one type.
        let vrx_ft8 = VrxState {
            slot: 1,
            offset_hz: 0,
            mode: VrxMode::Ft8,
            bw_hz: 2_600,
            gain_db: 0.0,
            rate_hz: 12_000,
            muted: false,
        };
        let vrx_js8 = VrxState {
            slot: 2,
            offset_hz: 1_500,
            mode: VrxMode::Js8,
            bw_hz: 2_600,
            gain_db: 0.0,
            rate_hz: 12_000,
            muted: true,
        };
        let ft8 = DecodeRow {
            text: "CQ DE W1AW".into(),
            freq_hz: 1512.0,
            dt_sec: 0.48,
            snr_db: -3.0,
            slot_ms: 85_000_000_000,
            vrx: vrx_ft8,
        };
        let js8 = DecodeRow {
            text: "N1MM: SNR +20".into(),
            freq_hz: 1500.0,
            dt_sec: 0.5,
            snr_db: 9.8,
            slot_ms: 85_000_001_000,
            vrx: vrx_js8,
        };
        assert_eq!(
            serde_json::from_str::<DecodeRow>(&serde_json::to_string(&ft8).unwrap()).unwrap(),
            ft8.clone()
        );
        let l = DecodeLog {
            decodes: vec![ft8, js8],
        };
        let l2: DecodeLog = serde_json::from_str(&serde_json::to_string(&l).unwrap()).unwrap();
        assert_eq!(l2, l);
    }

    #[test]
    fn decoded_message_trait_is_object_safe() {
        // The pipeline (`hl2-api` / `hl2`) handles decodes as `dyn
        // DecodedMessage` — this proves the trait object compiles and that
        // every method routes through the vtable, not a concrete type.
        struct W1Aw {
            caller: String,
            grid: String,
        }
        impl DecodedMessage for W1Aw {
            fn freq_hz(&self) -> f32 {
                1_500.0
            }
            fn dt_sec(&self) -> f32 {
                0.48
            }
            fn snr_db(&self) -> f32 {
                6.0
            }
            fn slot_ms(&self) -> u64 {
                85_000_000_000
            }
            fn mode(&self) -> &'static str {
                "FT8"
            }
            fn display(&self) -> &str {
                "CQ DE W1AW"
            }
            fn spot_fields(&self, st: &SpotStation) -> Option<SpotFields> {
                if !st.callsign.is_empty() && st.callsign == self.caller {
                    return None;
                }
                Some(SpotFields {
                    caller: self.caller.clone(),
                    locator: self.grid.clone(),
                })
            }
        }
        let m: &dyn DecodedMessage = &W1Aw {
            caller: "W1AW".into(),
            grid: "FN42".into(),
        };
        assert_eq!(m.mode(), "FT8");
        assert_eq!(m.display(), "CQ DE W1AW");
        let st = SpotStation::new("K9ABC", "FN31");
        assert_eq!(
            m.spot_fields(&st),
            Some(SpotFields {
                caller: "W1AW".into(),
                locator: "FN42".into()
            }),
            "a foreign call spots"
        );
        let self_m: &dyn DecodedMessage = &W1Aw {
            caller: "K9ABC".into(),
            grid: "FN31".into(),
        };
        assert_eq!(self_m.spot_fields(&st), None, "a self-spot is suppressed");

        // And a sink sees only the trait — never the concrete type.
        struct S {
            accepted: std::cell::Cell<usize>,
            station: SpotStation,
        }
        impl SpotSink for S {
            fn spot(&mut self, m: &dyn DecodedMessage, rf_hz: u32) -> bool {
                if rf_hz != 0 && m.spot_fields(&self.station).is_some() {
                    self.accepted.set(self.accepted.get() + 1);
                    true
                } else {
                    false
                }
            }
        }
        let mut sink = S {
            accepted: std::cell::Cell::new(0usize),
            station: st.clone(),
        };
        {
            let s: &mut dyn SpotSink = &mut sink;
            assert!(s.spot(m, 7_074_000));
        }
        assert_eq!(sink.accepted.get(), 1);
        {
            let s: &mut dyn SpotSink = &mut sink;
            assert!(!s.spot(self_m, 7_074_000), "self-spot");
        }
        assert_eq!(sink.accepted.get(), 1);
        {
            let s: &mut dyn SpotSink = &mut sink;
            assert!(!s.spot(m, 0), "no tune");
        }

        const _: fn() = || {
            let _ = spot::grid_is_square("LN35PQ");
            let _ = spot::base_callsign("W1AW/K9ABC");
        };
    }

    #[test]
    fn spot_helpers() {
        // grid square gates
        use crate::spot::grid_is_square;
        assert!(grid_is_square("LN35"));
        assert!(grid_is_square("ln35"));
        assert!(grid_is_square("FN42UR"));
        assert!(!grid_is_square("RR73"));
        assert!(!grid_is_square("12"));
        assert!(!grid_is_square("LN3"));
        assert!(!grid_is_square("S1"));
        assert!(!grid_is_square("LN35Y"));
        assert!(!grid_is_square("+10"));
        assert!(!grid_is_square("RR73AB"));
        assert!(grid_is_square("RR74")); // starts RR but is a square
        // base-call stripping
        use crate::spot::base_callsign;
        assert_eq!(base_callsign("K9ABC"), "K9ABC");
        assert_eq!(base_callsign("W1AW/K9ABC"), "K9ABC");
        assert_eq!(base_callsign("A1/AB2"), "AB2");
        assert_eq!(base_callsign(""), "");
        assert_eq!(base_callsign("/"), "");
        // SpotStation
        let s = SpotStation::new("K9ABC", "FN31pq");
        assert_eq!(s.callsign, "K9ABC");
        assert!(s.grid.starts_with("FN31"));
    }

    #[test]
    fn vrx_mode_js8_roundtrip() {
        let v = VrxState {
            slot: 1,
            offset_hz: 0,
            mode: VrxMode::Js8,
            bw_hz: 2_600,
            gain_db: 0.0,
            rate_hz: 12_000,
            muted: false,
        };
        assert_eq!(
            serde_json::from_str::<VrxState>(&serde_json::to_string(&v).unwrap()).unwrap(),
            v
        );
    }

    #[test]
    fn vrx_mode_fm_narrow_roundtrip() {
        // Both FM and NFM (narrow) are voice modes: 4.8 kHz audio, ≈ 15 kHz /
        // ≈ 5 kHz channel width respectively. The `lowercase` serde tag on
        // `VrxMode` means the on-wire name for `Fm` is `"fm"` and for
        // `FmNarrow` is `"fmnarrow"` (the lowercase of the Rust variant, no
        // underscore). The UI may additionally use `"nfm"` (the operator-
        // facing spelling); both decode to `VrxMode::FmNarrow`.
        let std_fm = VrxState {
            slot: 1,
            offset_hz: 0,
            mode: VrxMode::Fm,
            bw_hz: 15_000,
            gain_db: 0.0,
            rate_hz: 4_800,
            muted: false,
        };
        let narrow = VrxState {
            mode: VrxMode::FmNarrow,
            bw_hz: 5_000,
            ..std_fm
        };
        for v in [std_fm, narrow] {
            assert_eq!(
                serde_json::from_str::<VrxState>(&serde_json::to_string(&v).unwrap()).unwrap(),
                v
            );
        }
        // Enum-level round-trip through the `lowercase` wire names.
        assert_eq!(
            serde_json::from_str::<VrxMode>("\"fm\"").unwrap(),
            VrxMode::Fm
        );
        assert_eq!(
            serde_json::from_str::<VrxMode>("\"fmnarrow\"").unwrap(),
            VrxMode::FmNarrow
        );
        // And the operator-facing alias `"nfm"` must also decode to the same
        // variant — so the UI can use either spelling without a server change.
        assert_eq!(
            serde_json::from_str::<VrxMode>("\"nfm\"").unwrap(),
            VrxMode::FmNarrow
        );
        assert_eq!(serde_json::to_string(&VrxMode::Fm).unwrap(), "\"fm\"");
        assert_eq!(
            serde_json::to_string(&VrxMode::FmNarrow).unwrap(),
            "\"fmnarrow\""
        );
    }

    #[test]
    fn audio_frame_wire_layout() {
        let f = AudioFrame {
            slot: 1,
            seq: 0x0123_4567,
            rate_hz: 4_800,
            samples: vec![100, -200, 300],
        };
        let b = f.to_bytes();
        // 2 (slot) + 4 (seq) + 2 (rate) + 2 × 3 (samples) + nothing else.
        assert_eq!(b.len(), 2 + 4 + 2 + 6);
        assert_eq!(&b[0..2], &1u16.to_le_bytes());
        assert_eq!(&b[2..6], &0x0123_4567u32.to_le_bytes());
        assert_eq!(&b[6..8], &4_800u16.to_le_bytes());
        assert_eq!(&b[8..10], &100i16.to_le_bytes());
        assert_eq!(&b[10..12], &(-200i16).to_le_bytes());
        assert_eq!(&b[12..14], &300i16.to_le_bytes());
    }
}
