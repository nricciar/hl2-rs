//! Protocol constants and packet formats for the Hermes-Lite 2 SDR (Metis / protocol1).

pub mod command;
pub mod data;
pub mod discovery;
pub mod session;

/// Metis protocol marker bytes (frame type byte == 0x01).
pub const METIS_MARKER: [u8; 2] = [0xEF, 0xFE];

/// UDP port used by the HL2 for all communication.
pub const HL2_PORT: u16 = 1024;

/// Board ID for Hermes-Lite 2.
pub const BOARD_ID_HL2: u8 = 0x06;

/// UDP packet sizes.
pub const DISCOVERY_REQUEST_SIZE: usize = 63;
pub const DISCOVERY_RESPONSE_SIZE: usize = 60;
pub const START_REQUEST_SIZE: usize = 64;
pub const DATA_PACKET_SIZE: usize = 1032;
pub const CHUNK_SIZE: usize = 512;
pub const DATA_PAYLOAD_SIZE: usize = 1024;

/// Endpoint values.
///
/// * `ENDPOINT_CONTROL` = 0x02 — host → radio C&C / keep-alive frame (feeds
///   the watchdog). Every keep-alive must be sent on this endpoint.
/// * `ENDPOINT_WIDEBAND` = 0x04 — radio → host wideband IQ data stream.
/// * `ENDPOINT_DATA_TX` = 0x06 — radio → host command/ACK data stream.
pub const ENDPOINT_CONTROL: u8 = 0x02;
pub const ENDPOINT_WIDEBAND: u8 = 0x04;
pub const ENDPOINT_DATA_TX: u8 = 0x06;

/// Each 512-byte chunk holds 256 16-bit samples (2 bytes each); a 1032-byte EP4
/// frame carries 2 chunks = 512 samples.
pub const SAMPLES_PER_CHUNK: usize = 256;
pub const IQ_PAIRS_PER_PACKET: usize = SAMPLES_PER_CHUNK * 2;
/// A "block" is one full wideband spectrum acquisition: 2048 samples
/// (4 EP4 frames), per PROTOCOL.md "Wideband data" — samples are continuous
/// within a block and discontinuous across blocks. The API may use a larger
/// FFT window by accumulating multiple blocks.
pub const IQ_PAIRS_PER_BLOCK: usize = 2048;

/// ADC clock.
pub const ADC_CLOCK_HZ: u32 = 76_800_000;

/// Watchdog: the HL2 resets to "waiting" if no control frame arrives within
/// ~168 ms. We send keep-alives every 40 ms for margin.
pub const KEEPALIVE_INTERVAL_MS: u64 = 40;

/// C1 configuration bits in a host-side control frame.
///
/// * `CONFIG_BOTH (0x60)` — both TX and RX engines enabled; the reference
///   implementation calls this "critical to getting the board to respond".
/// * `SPEED_*` — per-receiver DDC output rate in kSps. Reference: the C1
///   byte's lowest two bits are the speed (`00`=48k, `01`=96k, `10`=192k,
///   `11`=384k) — the reference drives the board the same way; see PROTOCOL.md §7.
pub const C1_CONFIG_BOTH: u8 = 0x60;
/// C1 speed bits, raw value (do not use directly; use one of the
/// `C1_SPEED_*` constants below).
#[allow(dead_code)]
pub const C1_SPEED_MASK: u8 = 0x03;
pub const C1_SPEED_48K: u8 = 0x00;
pub const C1_SPEED_96K: u8 = 0x01;
/// 192 kSps per-receiver DDC output rate.
pub const C1_SPEED_192K: u8 = 0x02;
pub const C1_SPEED_384K: u8 = 0x03;

/// The RX open-collector filter selection is carried in **C2** of the keep-alive
/// (and any host→radio C&C) frame, bits `[7:1] = OC1..OC7`, bit `[0]` reserved.
/// This is the mechanism that drives the MRF101-companion filter board's relay
/// switching (via the board's own I2C-0x20 MCP23008 — see PROTOCOL.md §11.4)
/// and matches the reference (`output_buffer[C2] |= band->OCrx << 1`).
///
/// Convention: the user-facing `oc_bits` mask uses bit 0 = relay/checkbox 1
/// (LSB-first) and bit 6 = relay/checkbox 7. The wire puts them in C2[7:1].
/// `0x00` = all relays off (board default, e.g. 160 m receive).
/// `0x44` = relays 3 and 7 (the 7.074 MHz / 40 m band case).
///
/// Only the lower 7 bits are meaningful; any higher bits are ignored by
/// `build_keepalive_packet`.
pub const OC_MASK_RX: u8 = 0x7F;

/// Convert a baseband rate (kHz) to C1 speed bits, choosing the nearest legal
/// option (`48`, `96`, `192`, `384`).
///
/// * 48  → `C1_SPEED_48K`
/// * 96  → `C1_SPEED_96K`
/// * 192 → `C1_SPEED_192K`
/// * 384 → `C1_SPEED_384K`
pub fn speed_bits_for_khz(khz: u32) -> u8 {
    match khz {
        48 => C1_SPEED_48K,
        96 => C1_SPEED_96K,
        192 => C1_SPEED_192K,
        384 => C1_SPEED_384K,
        // Round other values to the nearest legal option.
        v if v < 72 => C1_SPEED_48K,
        v if v < 144 => C1_SPEED_96K,
        v if v < 288 => C1_SPEED_192K,
        _ => C1_SPEED_384K,
    }
}

/// 3-byte sync prefix in every host-to-radio / radio-to-host C&C frame.
pub const C_SYNC: u8 = 0x7F;

/// LNA gain register address (PROTOCOL.md §7 / §11.3).
///
/// A write is a C&C register write with `C0 = LNA_ADDR << 1` (MOX=0) = `0x14`,
/// which matches the reference case 4 (`C0=0x14`). Only
/// C4 carries the value: bit[6] selects "Set" mode (LNA sent straight to the
/// AD9866, full −12…+48 dB) and bits[5:0] hold the gain. See `build_lna_gain_frame`.
pub const LNA_ADDR: u8 = 0x0A;
/// LNA "Set" (straight-to-AD9866) mode bit in the gain byte (C4 bit[6]).
/// The reference sets `output_buffer[C4] = 0x40`.
pub const LNA_MODE_SET: u8 = 0x40;

/// Default LNA gain, in dB. The register range is −12…+48 dB; this is a
/// moderate default that suits a compromised-antenna setup (see §11.3). The
/// reference defaults to 0 dB; we start a touch higher.
pub const DEFAULT_LNA_GAIN_DB: i8 = 6;

/// Discovery status byte values.
pub const STATUS_NOT_SENDING: u8 = 0x02;
pub const STATUS_SENDING: u8 = 0x03;
