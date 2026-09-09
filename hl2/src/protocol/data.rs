//! Parsing of 1032-byte HL2 data packets, IQ extraction, and block assembly.
//!
//! `no_std`-eligible: allocation comes from `alloc` (see the top-level
//! `crate::extern crate alloc`), so the heap types here are `alloc::vec::Vec`
//! (identical to `std`'s `Vec` when `std` is enabled).

use alloc::vec::Vec;

use num_complex::Complex;

use crate::protocol::command::CommandData;
use crate::protocol::{
    C_SYNC, C1_CONFIG_BOTH, C1_SPEED_MASK, CHUNK_SIZE, DATA_PACKET_SIZE, ENDPOINT_CONTROL,
    ENDPOINT_DATA_TX, ENDPOINT_WIDEBAND, IQ_PAIRS_PER_BLOCK, METIS_MARKER, SAMPLES_PER_CHUNK,
    START_REQUEST_SIZE,
};

/// Size of the 8-byte data packet header.
pub const HEADER_SIZE: usize = 8;

/// EP4 wideband frames carry raw IQ samples starting immediately after the
/// 8-byte header (no C0-C4 or eaddr prefix, no SYNC).
pub const IQ_OFFSET: usize = 0;

/// Each EP4 sample is 2 bytes (16-bit little-endian single real value).
/// HL2 wideband is a single real-valued stream sampled at 76.8 MSps ADC
/// (`ADC_CLOCK_HZ`, PROTOCOL.md §10) — a 2048-sample block spans 0–38.4 MHz.
pub const BYTES_PER_SAMPLE: usize = 2;

/// 3×0x7F sync prefix that every EP6 (control/ACK) frame starts with.
/// C0–C4 follow immediately after, at absolute offset 11 from the frame start.
pub const EP6_SYNC_LEN: usize = 3;

/// The sample format negotiated with the HL2 (informational; the wire format
/// is always 16-bit little-endian in EP4 frames).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleFormat {
    /// 12-bit signed samples, sign-extended to 16 bits on the wire.
    Sample12,
    /// Full 16-bit signed samples.
    Sample16,
}

/// Parsed 8-byte header of a 1032-byte data packet.
///
/// Layout: `[0xEF 0xFE][frame_type(1B)][endpoint(1B)][seq(4B BE)]`
///
/// `frame_type` is always 0x01 for data frames.  `endpoint` is:
///   0x02 = host→radio C&C,  0x04 = radio→host wideband IQ,  0x06 = radio→host C&C/ACK
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DataHeader {
    pub endpoint: u8,
    pub seq: u32,
}

pub fn parse_data_header(bytes: &[u8; HEADER_SIZE]) -> Option<DataHeader> {
    if bytes[0] != 0xEF || bytes[1] != 0xFE {
        return None;
    }
    Some(DataHeader {
        endpoint: bytes[3],
        seq: u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    })
}

/// Extract 256 × 16-bit little-endian samples from a 512-byte EP4 chunk.
pub fn extract_iq_from_chunk(chunk: &[u8; CHUNK_SIZE]) -> Vec<i16> {
    (0..SAMPLES_PER_CHUNK)
        .map(|i| i16::from_le_bytes([chunk[i * 2], chunk[i * 2 + 1]]))
        .collect()
}

/// Extract 256 × 16-bit little-endian samples from a 512-byte EP4 chunk into
/// a caller-provided slice (capacity ≥ `SAMPLES_PER_CHUNK`). Reuses the
/// `Vec`'s allocation so the receiver loop can keep one buffer alive across
/// packets.
#[inline]
pub fn extract_iq_into_chunk(chunk: &[u8; CHUNK_SIZE], out: &mut [i16]) {
    for (i, o) in (0..SAMPLES_PER_CHUNK).zip(out.iter_mut()) {
        *o = i16::from_le_bytes([chunk[i * 2], chunk[i * 2 + 1]]);
    }
}

// ── EP6 baseband I/Q (24-bit, multi-receiver interleave) ──────────────────

/// Header: `3×SYNC (0x7F) + C0..C4` — 8 bytes at the start of every 512-byte
/// EP6 baseband chunk. C0-C4 are control/status bytes (PTT, ADC overload, IO,
/// AIN), NOT a slot selector.
pub const BASEBAND_HEADER_OFFSET: usize = 8;

/// Bytes per 24-bit I/Q sample: 24-bit I (3 B, big-endian) + 24-bit Q (3 B).
pub const BASEBAND_BYTES_PER_IQ: usize = 6;
/// Bytes per mic/audio sample (16-bit, appended after each receiver group).
pub const BASEBAND_BYTES_PER_MIC: usize = 2;
/// A 24-bit signed sample, sign-extended from 3 big-endian bytes.
#[inline]
pub fn sample_24be(b: &[u8]) -> f32 {
    let mut v: i32 =
        ((b[0] as i32) & 0xFF) << 16 | ((b[1] as i32) & 0xFF) << 8 | (b[2] as i32) & 0xFF;
    if v & 0x80_0000 != 0 {
        v |= 0xFF00_0000u32 as i32; // sign-extend
    }
    v as f32 / 8_388_607.0
}

/// A chunk of complex I/Q baseband for one (or more) receiver slot(s).
#[derive(Debug, Clone, PartialEq)]
pub struct BasebandChunk {
    /// Complex I/Q samples per receiver (normalised to `f32` in [-1, +1]).
    /// `per_rx[0]` == RX1, `per_rx[1]` == RX2, …
    pub per_rx: Vec<Vec<Complex<f32>>>,
}

impl BasebandChunk {
    /// Number of receiver groups present in this chunk.
    pub fn rx_count(&self) -> usize {
        self.per_rx.len()
    }
    /// The receiver-0 (RX1) stream — the common single-receiver case.
    pub fn iq(&self) -> &[Complex<f32>] {
        &self.per_rx[0]
    }
    /// Reuse the `per_rx` inner buffers for exactly `n` receiver streams,
    /// clearing their contents. `per_rx` is grown/shrunk if needed; existing
    /// inner `Vec`s retain their capacity, so a subsequent
    /// `parse_baseband_chunk_into` fills without reallocating in the common
    /// (same-rate, same-N) case.
    pub fn resize_rx(&mut self, n: usize) {
        while self.per_rx.len() < n {
            self.per_rx.push(Vec::new());
        }
        if self.per_rx.len() > n {
            self.per_rx.truncate(n);
        }
        for v in self.per_rx.iter_mut() {
            v.clear();
        }
    }
}

/// Bytes per record for `n_recv` receivers (N×(I+Q) + 1 mic group):
/// `n_recv * 6 + 2`. Only receiver counts where 504 is divisible by this
/// are physically meaningful: 1 (8 B), 2 (14 B), 5 (32 B).
pub fn baseband_record_len(n_recv: u8) -> usize {
    n_recv as usize * BASEBAND_BYTES_PER_IQ + BASEBAND_BYTES_PER_MIC
}

/// True if `n_recv` receivers produce a record length that evenly divides the
/// 504-byte payload of a chunk.
pub fn baseband_valid_n(n_recv: u8) -> bool {
    n_recv >= 1 && (CHUNK_SIZE - BASEBAND_HEADER_OFFSET) % baseband_record_len(n_recv) == 0
}

/// Parse one 512-byte EP6 baseband chunk given `n_recv` active receivers.
///
/// Layout (reference EP6 receive path):
///   offset 0: 0x7F 0x7F 0x7F  (3×SYNC)
///   offset 3: C0-C4 (control/status: PTT, ADC overload, IO, AIN)
///   offset 8: `count` ×  [ I 3B BE | Q 3B BE | ...N×... | mic 2B ]
///
/// The payload holds `count = (504) / (n_recv*6 + 2)` whole records (plain
/// integer division, same as the reference). Only values
/// of N whose record size 6·N+2 is a divisor of 504 (N = 1, 2, 7, 16, 42,
/// 251) yield no partial trailing bytes; other N — like N = 3, 4, 5, 6 —
/// decode `504 / (6N+2)` whole records and truncate the remainder (e.g.
/// N=4 → 19 whole records, 10 trailing bytes dropped).
///
/// Returns None if the 3×SYNC prefix is missing or `n_recv` is 0, or if
/// `n_recv` is so large that even one record doesn't fit (i.e. 6·N+2 > 504,
/// N > 83).
pub fn parse_baseband_chunk(chunk: &[u8; CHUNK_SIZE], n_recv: u8) -> Option<BasebandChunk> {
    if chunk[0] != C_SYNC || chunk[1] != C_SYNC || chunk[2] != C_SYNC {
        return None;
    }
    let record = baseband_record_len(n_recv);
    let payload_len = CHUNK_SIZE - BASEBAND_HEADER_OFFSET;
    if record > payload_len {
        return None;
    }
    let count = payload_len / record;
    let mut per_rx: Vec<Vec<Complex<f32>>> = Vec::with_capacity(n_recv as usize);
    for _ in 0..n_recv {
        per_rx.push(Vec::with_capacity(count));
    }
    let base = BASEBAND_HEADER_OFFSET;
    for r in 0..count {
        let base_r = base + r * record;
        for rx in 0..n_recv as usize {
            let o = base_r + rx * BASEBAND_BYTES_PER_IQ;
            let i_re = sample_24be(&chunk[o..o + 3]);
            let q_im = sample_24be(&chunk[o + 3..o + 6]);
            per_rx[rx].push(Complex::new(i_re, q_im));
        }
    }
    Some(BasebandChunk { per_rx })
}

/// Parse one 512-byte EP6 baseband chunk into a caller-provided
/// [`BasebandChunk`], reusing its `per_rx` inner buffers (see
/// [`BasebandChunk::resize_rx`]) so a steady-state receive loop at constant
/// rate + `n_recv` does not reallocate. Returns `true` if the chunk had a
/// valid 3×SYNC prefix and filled `bc`; `false` if `bc` should be skipped
/// (no sync, or `n_recv` too large for one record).
///
/// See [`parse_baseband_chunk`] for the exact field layout and the
/// truncation semantics.
pub fn parse_baseband_chunk_into(
    chunk: &[u8; CHUNK_SIZE],
    n_recv: u8,
    bc: &mut BasebandChunk,
) -> bool {
    if chunk[0] != C_SYNC || chunk[1] != C_SYNC || chunk[2] != C_SYNC {
        return false;
    }
    let record = baseband_record_len(n_recv);
    let payload_len = CHUNK_SIZE - BASEBAND_HEADER_OFFSET;
    if record > payload_len {
        return false;
    }
    let count = payload_len / record;
    bc.resize_rx(n_recv as usize);
    for v in bc.per_rx.iter_mut() {
        v.reserve_exact(count);
    }
    let base = BASEBAND_HEADER_OFFSET;
    for r in 0..count {
        let base_r = base + r * record;
        for rx in 0..n_recv as usize {
            let o = base_r + rx * BASEBAND_BYTES_PER_IQ;
            let i_re = sample_24be(&chunk[o..o + 3]);
            let q_im = sample_24be(&chunk[o + 3..o + 6]);
            bc.per_rx[rx].push(Complex::new(i_re, q_im));
        }
    }
    true
}

/// Parse a 1032-byte EP6 frame (2 × 512-byte baseband chunks) into a list of
/// [`BasebandChunk`]s (one per 512-byte block present). Each chunk's `per_rx`
/// is sized to `n_recv`.
pub fn parse_baseband_frame(buf: &[u8; DATA_PACKET_SIZE], n_recv: u8) -> Vec<BasebandChunk> {
    let mut out = Vec::with_capacity(2);
    for off in [HEADER_SIZE, HEADER_SIZE + CHUNK_SIZE] {
        let slice = &buf[off..off + CHUNK_SIZE];
        if let Ok(chunk) = <[u8; CHUNK_SIZE]>::try_from(slice) {
            if let Some(bc) = parse_baseband_chunk(&chunk, n_recv) {
                out.push(bc);
            }
        }
    }
    out
}

/// Parse a 1032-byte EP6 frame into a caller-provided vector of
/// [`BasebandChunk`]s, **reusing** any chunks (and their `per_rx` inner
/// buffers) already present in `out`. In steady state (constant rate +
/// `n_recv`) this means `out` keeps the same `Vec`s and capacities and no
/// allocation happens per frame — the chunks are simply refilled. `out` ends
/// with exactly as many valid chunks as were found (0, 1 or 2), in original
/// order.
pub fn parse_baseband_frame_into(
    buf: &[u8; DATA_PACKET_SIZE],
    n_recv: u8,
    out: &mut Vec<BasebandChunk>,
) {
    // Reserve up to 2 reusable chunk slots (keeping any pre-existing ones).
    while out.len() < 2 {
        out.push(BasebandChunk { per_rx: Vec::new() });
    }
    let mut kept = 0;
    for off in [HEADER_SIZE, HEADER_SIZE + CHUNK_SIZE] {
        let slice = &buf[off..off + CHUNK_SIZE];
        let Some(chunk) = <[u8; CHUNK_SIZE]>::try_from(slice).ok() else {
            continue;
        };
        if parse_baseband_chunk_into(&chunk, n_recv, &mut out[kept]) {
            kept += 1;
        }
    }
    out.truncate(kept);
}

/// Parse a 5-byte C0-C4 payload found in an EP6 control/ACK frame.
///
/// The caller is responsible for passing `chunk[3..8]` (the bytes right after
/// the 3×0x7F sync prefix).
pub fn parse_chunk_command(chunk: &[u8; CHUNK_SIZE]) -> Option<CommandData> {
    if chunk[0] == C_SYNC && chunk[1] == C_SYNC && chunk[2] == C_SYNC {
        let five: [u8; 5] = [chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]];
        CommandData::parse(&five)
    } else {
        None
    }
}

/// Build the "baseline" 512-byte chunk that *every* host→radio C&C frame uses
/// for its **first** chunk. This is the control/status baseline the reference
/// emits for the first half of a frame:
///   chunk 0: 0x7F 0x7F 0x7F  (3×SYNC)
///   C0 = 0x00
///   C1 = CONFIG_BOTH(0x60) | SPEED bits
///   C2 = (oc_bits << 1) & 0xFF — RX open-collector filter bits
///        (C2[7:1]=OC1..OC7, bit 0 reserved; LSB-first `oc_bits` bit 0 = relay 1)
///   C3 = 0x00
///   C4 = 0x04 (duplex) | ((n_recv-1) << 3) — `output_buffer[C4]=0x04 | nreceivers<<3`
///
/// The reference comment — *"CONFIG_BOTH seems to be critical to getting
/// ozy to respond"* — is why the baseline must ride in the
/// **same 1032-byte frame** as any register write: the gateware commits the
/// write only when the frame it arrives in is a well-formed baseline+command
/// pair. See [`build_nco_packet`] / [`build_lna_gain_frame`].
pub fn build_baseline_chunk(c1_speed_bits: u8, oc_bits: u8, n_recv: u8) -> [u8; CHUNK_SIZE] {
    let mut chunk = [0u8; CHUNK_SIZE];
    chunk[0] = C_SYNC;
    chunk[1] = C_SYNC;
    chunk[2] = C_SYNC;
    chunk[3] = 0x00; // C0
    chunk[4] = C1_CONFIG_BOTH | (c1_speed_bits & C1_SPEED_MASK);
    chunk[5] = (oc_bits & 0x7F) << 1; // C2: [7:1]=OC1..OC7, [0]=reserved
    chunk[6] = 0x00; // C3
    chunk[7] = 0x04 | (((n_recv.saturating_sub(1)) & 0x07) << 3); // C4: duplex + receivers
    chunk
}

/// Build the 1032-byte host→radio keepalive (C&C) frame.
///
/// Endpoint must be 0x02 (EP2) — sending on EP3 starves the HL2 watchdog and
/// triggers a reset to "waiting" after ~4 s.
///
/// Both 512-byte chunks carry the reference baseline chunk (see
/// [`build_baseline_chunk`]); the baseline alone (C0=0x00) is what tells the
/// gateware the device is being actively driven and re-asserts the per-receiver
/// DDC rate and open-collector relay holding between register writes.
pub fn build_keepalive_packet(
    seq: u32,
    c1_speed_bits: u8,
    oc_bits: u8,
    n_recv: u8,
) -> [u8; DATA_PACKET_SIZE] {
    let chunk = build_baseline_chunk(c1_speed_bits, oc_bits, n_recv);

    let mut buf = [0u8; DATA_PACKET_SIZE];
    buf[0] = 0xEF;
    buf[1] = 0xFE;
    buf[2] = 0x01; // frame type = data
    buf[3] = ENDPOINT_CONTROL;
    buf[4..8].copy_from_slice(&seq.to_be_bytes());
    buf[8..8 + CHUNK_SIZE].copy_from_slice(&chunk);
    buf[8 + CHUNK_SIZE..8 + CHUNK_SIZE * 2].copy_from_slice(&chunk);
    buf
}

/// Build the 64-byte host→radio start/stop command frame.
///
/// The wire layout is `[0xEFFE][0x04][CommandByte][60 bytes zero]`; the
/// sequence number lives in the *same* field position as the data-frame
/// header when the packet is sent over UDP (HL2 ignores it but we keep it so
/// `build_nco_packet` / `build_keepalive_packet` can stay consistent with a
/// single 1032-byte layout for C&C frames).
pub fn build_start_stop_frame(seq: u32, start: bool) -> [u8; START_REQUEST_SIZE] {
    use crate::protocol::command::StartCommand;
    let mut out = [0u8; START_REQUEST_SIZE];
    out[0] = METIS_MARKER[0];
    out[1] = METIS_MARKER[1];
    out[2] = 0x04;
    out[3] = StartCommand::new(start, start, false).into_byte();
    out[4..8].copy_from_slice(&seq.to_be_bytes());
    out
}

/// Build a 1032-byte command frame to write a 32-bit NCO frequency to a
/// receiver slot.
///
/// `slot` is 1-based (RX1 = 1) and maps to the memory-map NCO register
/// (PROTOCOL.md §7: `0x02`=RX1, `0x03`=RX2, …). Since C0 encodes the address
/// left-shifted (`C0 = ADDR << 1`), the byte is:
///
///   `C0 = (0x01 + slot) << 1`
///
///   RX1 (slot=1) → 0x04, RX2 (slot=2) → 0x06, RX3 (slot=3) → 0x08, …
///
/// This matches the reference (`output_buffer[C0]=0x04+(current_rx*2)` with a
/// 0-based `current_rx`, so RX1 → 0x04).
///
/// **Two-chunk frame layout (critical to the commit handshake):** the
/// reference emits every C&C register write as *two 512-byte chunks in one
/// 1032-byte frame* — the first chunk is the baseline (see
/// [`build_baseline_chunk`]) and the second carries the NCO register write.
/// The reference fills chunk 1, then chunk 2, then sends. Its comment —
/// *"CONFIG_BOTH seems to be critical to getting ozy to respond"* — makes
/// the pairing
/// non-optional: an NCO frame that arrives without a baseline in the same
/// frame is received but not committed.
///
/// Our historical bug: this function put the NCO write in **both** chunks,
/// so a tune-only frame carried no baseline and the DDC latched at the last
/// committed frequency (the last one any client had written). Fixed by using
/// the reference two-chunk split.
///
/// `c1_speed_bits` / `oc_bits` / `n_recv` configure the baseline chunk so the
/// frame is well-formed and the baseline is re-asserted even when the caller
/// does not send a separate keep-alive for the same tune (the reference
/// relies on this: a tune-only frame *is* its own keep-alive for baseline
/// purposes).
///
/// A 1032-byte EP2 C&C frame (both 512-byte chunks in one send).
pub fn build_nco_packet(
    seq: u32,
    slot: u8,
    freq_hz: u32,
    c1_speed_bits: u8,
    oc_bits: u8,
    n_recv: u8,
) -> [u8; DATA_PACKET_SIZE] {
    let c0 = (0x01 + slot) << 1;
    let freq_bytes = freq_hz.to_be_bytes();

    let mut chunk = [0u8; CHUNK_SIZE];
    chunk[0] = C_SYNC;
    chunk[1] = C_SYNC;
    chunk[2] = C_SYNC;
    chunk[3] = c0;
    chunk[4] = freq_bytes[0];
    chunk[5] = freq_bytes[1];
    chunk[6] = freq_bytes[2];
    chunk[7] = freq_bytes[3];

    let baseline = build_baseline_chunk(c1_speed_bits, oc_bits, n_recv);

    let mut buf = [0u8; DATA_PACKET_SIZE];
    buf[0] = 0xEF;
    buf[1] = 0xFE;
    buf[2] = 0x01;
    buf[3] = ENDPOINT_CONTROL;
    buf[4..8].copy_from_slice(&seq.to_be_bytes());
    // First chunk: baseline (C0=0x00, CONFIG_BOTH, …) — the commit handshake.
    buf[8..8 + CHUNK_SIZE].copy_from_slice(&baseline);
    // Second chunk: the NCO register write.
    buf[8 + CHUNK_SIZE..8 + CHUNK_SIZE * 2].copy_from_slice(&chunk);
    buf
}

/// Build a 1032-byte C&C frame that sets the RX low-noise-amplifier gain.
///
/// Mirrors [`build_nco_packet`]: same EP2 control endpoint, 3×SYNC prefix in
/// both 512-byte chunks, big-endian sequence in the header. The register
/// write itself:
///   * C0 = `LNA_ADDR << 1` (`0x0A << 1 = 0x14`, MOX=0). This is the address
///     `0x0a` from PROTOCOL.md §7 (the same "Set" case the reference uses).
///   * C1 = C2 = C3 = 0.
///   * C4 = `LNA_MODE_SET | ((gain_db + 12) & 0x3F)` — bit[6] "Set" mode plus
///     the 6-bit gain. PROTOCOL.md §11.3 "Set": LNA[5:0] is passed straight to
///     the AD9866 (full −12…+48 dB). `gain_db` is expected in that range; out
///     of range it wraps on the low bits (−13 → 0x41, 49 → 0x7D). The
///     reference sets `output_buffer[C4] = 0x40 | ((attenuation + 12) & 0x3F)`.
pub fn build_lna_gain_frame(
    seq: u32,
    gain_db: i8,
    c1_speed_bits: u8,
    oc_bits: u8,
    n_recv: u8,
) -> [u8; DATA_PACKET_SIZE] {
    use crate::protocol::{LNA_ADDR, LNA_MODE_SET};
    let c0 = LNA_ADDR << 1; // 0x14, MOX=0
    let c4 = LNA_MODE_SET | (((gain_db as u8).wrapping_add(12)) & 0x3F);

    let mut chunk = [0u8; CHUNK_SIZE];
    chunk[0] = C_SYNC;
    chunk[1] = C_SYNC;
    chunk[2] = C_SYNC;
    chunk[3] = c0;
    chunk[4] = 0x00; // C1
    chunk[5] = 0x00; // C2
    chunk[6] = 0x00; // C3
    chunk[7] = c4;

    let baseline = build_baseline_chunk(c1_speed_bits, oc_bits, n_recv);

    let mut buf = [0u8; DATA_PACKET_SIZE];
    buf[0] = 0xEF;
    buf[1] = 0xFE;
    buf[2] = 0x01;
    buf[3] = ENDPOINT_CONTROL;
    buf[4..8].copy_from_slice(&seq.to_be_bytes());
    // First chunk: baseline (commit handshake). Second: the LNA register write.
    buf[8..8 + CHUNK_SIZE].copy_from_slice(&baseline);
    buf[8 + CHUNK_SIZE..8 + CHUNK_SIZE * 2].copy_from_slice(&chunk);
    buf
}

/// A block of `IQ_PAIRS_PER_BLOCK` samples from the EP4 wideband stream.
#[derive(Debug, Clone)]
pub struct IQBlock {
    pub seq_start: u32,
    pub samples: Vec<i16>,
    pub sample_rate_hz: u32,
}

/// Items emitted from the block assembler.
pub enum Item {
    Block(IQBlock),
}

/// Parse a received 1032-byte data packet into its components.
///
/// *   EP4 frames: `iq_samples` is populated.
/// *   EP6 frames: either `chunk1_command` or `chunk2_command` may hold a
///     `CommandData` (PTT / state / ACK).
#[derive(Debug)]
pub struct ParsedPacket {
    pub header: DataHeader,
    pub chunk1_command: Option<CommandData>,
    pub chunk2_command: Option<CommandData>,
    pub iq_samples: Option<Vec<i16>>,
    /// EP6 complex baseband I/Q (one chunk per slot present in the frame).
    pub baseband: Vec<BasebandChunk>,
}

pub fn parse_receive_packet(buf: &[u8; DATA_PACKET_SIZE], n_recv: u8) -> Option<ParsedPacket> {
    let header = parse_data_header(&buf[..HEADER_SIZE].try_into().unwrap())?;

    let chunk1: [u8; CHUNK_SIZE] = buf[HEADER_SIZE..HEADER_SIZE + CHUNK_SIZE].try_into().ok()?;
    let chunk2: [u8; CHUNK_SIZE] = buf[HEADER_SIZE + CHUNK_SIZE..HEADER_SIZE + CHUNK_SIZE * 2]
        .try_into()
        .ok()?;

    // EP4 (wideband): both 512-byte chunks carry 16-bit little-endian samples.
    // EP6 (control): C0-C4 follow a 3×0x7F sync prefix in the first chunk.
    let iq_samples = if header.endpoint == ENDPOINT_WIDEBAND {
        let mut samples = Vec::with_capacity(SAMPLES_PER_CHUNK * 2);
        samples.extend(extract_iq_from_chunk(&chunk1));
        samples.extend(extract_iq_from_chunk(&chunk2));
        Some(samples)
    } else {
        None
    };

    // Command/ACK chunks only appear on EP2 (control). EP6 is the baseband
    // data stream; parsing its C0s as commands would falsely trigger CmdAck.
    let is_cc = header.endpoint == ENDPOINT_CONTROL;
    let chunk1_command = if is_cc {
        parse_chunk_command(&chunk1)
    } else {
        None
    };
    let chunk2_command = if is_cc {
        parse_chunk_command(&chunk2)
    } else {
        None
    };

    let baseband = if header.endpoint == ENDPOINT_DATA_TX {
        parse_baseband_frame(buf, n_recv)
    } else {
        Vec::new()
    };

    Some(ParsedPacket {
        header,
        chunk1_command,
        chunk2_command,
        iq_samples,
        baseband,
    })
}

/// Parse a received 1032-byte data packet into caller-provided buffers,
/// reusing them across calls so the steady-state receive loop does not
/// reallocate.
///
/// `iq` is a `Vec<i16>` reused across calls: it is filled (len set to
/// `SAMPLES_PER_CHUNK * 2`) iff the frame is an EP4 (`ENDPOINT_WIDEBAND`)
/// data frame, and cleared (len 0) otherwise. `baseband` is similarly
/// refilled (0–2 chunks) iff the frame is an EP6 (`ENDPOINT_DATA_TX`) frame,
/// cleared otherwise; the two are independent and can be reused alternately.
#[derive(Debug)]
pub struct ParsedPacketInto<'a> {
    pub header: DataHeader,
    pub chunk1_command: Option<CommandData>,
    pub chunk2_command: Option<CommandData>,
    /// EP4 samples (len 0 when the frame is not EP4 — this buffer IS
    /// cleared on non-EP4 frames, so `is_empty()` is the correct probe).
    pub iq: &'a mut Vec<i16>,
    /// EP6 baseband chunks. **Left alone on non-EP6 frames** — the caller
    /// must check `header.endpoint == ENDPOINT_DATA_TX` before reading
    /// `baseband`. This preserves the inner `per_rx` `Vec<Complex<f32>>`
    /// capacity across frames even when the caller interleaves EP6 data
    /// with EP2 control (ack, keep-alive) traffic.
    pub baseband: &'a mut Vec<BasebandChunk>,
}

pub fn parse_receive_packet_into<'a>(
    buf: &[u8; DATA_PACKET_SIZE],
    n_recv: u8,
    iq: &'a mut Vec<i16>,
    baseband: &'a mut Vec<BasebandChunk>,
) -> Option<ParsedPacketInto<'a>> {
    let header = parse_data_header(&buf[..HEADER_SIZE].try_into().unwrap())?;

    let chunk1: [u8; CHUNK_SIZE] = buf[HEADER_SIZE..HEADER_SIZE + CHUNK_SIZE].try_into().ok()?;
    let chunk2: [u8; CHUNK_SIZE] = buf[HEADER_SIZE + CHUNK_SIZE..HEADER_SIZE + CHUNK_SIZE * 2]
        .try_into()
        .ok()?;

    // EP4 (wideband) carries 16-bit LE samples in both chunks; EP6 baseband
    // and EP2 control do not.
    iq.clear();
    if header.endpoint == ENDPOINT_WIDEBAND {
        iq.resize(SAMPLES_PER_CHUNK * 2, 0);
        extract_iq_into_chunk(&chunk1, &mut iq[..SAMPLES_PER_CHUNK]);
        extract_iq_into_chunk(&chunk2, &mut iq[SAMPLES_PER_CHUNK..SAMPLES_PER_CHUNK * 2]);
    }

    // Command/ACK chunks only appear on EP2 (control).
    let is_cc = header.endpoint == ENDPOINT_CONTROL;
    let chunk1_command = if is_cc {
        parse_chunk_command(&chunk1)
    } else {
        None
    };
    let chunk2_command = if is_cc {
        parse_chunk_command(&chunk2)
    } else {
        None
    };

    if header.endpoint == ENDPOINT_DATA_TX {
        parse_baseband_frame_into(buf, n_recv, baseband);
    }
    // Non-EP6 frames leave `baseband` alone so the inner `per_rx` buffers
    // keep their capacity — critical on a no-op-dealloc allocator where a
    // `clear()` + next EP6 `push(Vec::new())` is a permanent leak.

    Some(ParsedPacketInto {
        header,
        chunk1_command,
        chunk2_command,
        iq,
        baseband,
    })
}

/// Block assembler: accumulates EP4 packets and yields a complete IQ block
/// (`IQ_PAIRS_PER_BLOCK` samples) when enough have been collected.
#[derive(Debug)]
pub struct BlockAssembler {
    buffer: Vec<i16>,
    seq_start: u32,
}

impl BlockAssembler {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(IQ_PAIRS_PER_BLOCK),
            seq_start: 0,
        }
    }

    pub fn feed(&mut self, header: &DataHeader, samples: &[i16]) -> Option<Item> {
        if header.endpoint != ENDPOINT_WIDEBAND {
            return None;
        }

        if self.buffer.is_empty() {
            self.seq_start = header.seq;
        }
        self.buffer.extend_from_slice(samples);

        if self.buffer.len() >= IQ_PAIRS_PER_BLOCK {
            let samples: Vec<i16> = self.buffer.drain(..IQ_PAIRS_PER_BLOCK).collect();
            Some(Item::Block(IQBlock {
                seq_start: self.seq_start,
                samples,
                sample_rate_hz: crate::protocol::ADC_CLOCK_HZ,
            }))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{C1_SPEED_192K, ENDPOINT_CONTROL, ENDPOINT_WIDEBAND, LNA_MODE_SET};

    #[test]
    fn parse_valid_header() {
        let header = [0xEF, 0xFE, 0x01, 0x04, 0x00, 0x00, 0x00, 0x42];
        let h = parse_data_header(&header).unwrap();
        assert_eq!(h.endpoint, 0x04);
        assert_eq!(h.seq, 0x42);
    }

    #[test]
    fn parse_invalid_header_marker() {
        let header = [0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x00, 0x42];
        assert!(parse_data_header(&header).is_none());
    }

    #[test]
    fn keepalive_uses_ep2_and_c1_config() {
        let pkt = build_keepalive_packet(1, C1_SPEED_192K, 0x00, 1);
        assert_eq!(pkt[3], ENDPOINT_CONTROL);
        // chunk0 C1 at absolute offset 12
        assert_eq!(pkt[12], C1_CONFIG_BOTH | C1_SPEED_192K);
        // C2 (offset 13) is 0 when no OC bits are set (all relays off)
        assert_eq!(pkt[13], 0x00);
        // C4 (offset 15) carries duplex (0x04) + receiver-count bits (0 for n=1)
        assert_eq!(pkt[15], 0x04);
        // sync prefix
        assert_eq!(pkt[8], C_SYNC);
        assert_eq!(pkt[9], C_SYNC);
        assert_eq!(pkt[10], C_SYNC);
    }

    #[test]
    fn keepalive_honors_requested_speed_bit() {
        use crate::protocol::{C1_SPEED_48K, C1_SPEED_96K, C1_SPEED_384K};
        for speed in [C1_SPEED_48K, C1_SPEED_96K, C1_SPEED_192K, C1_SPEED_384K] {
            let pkt = build_keepalive_packet(0, speed, 0x00, 1);
            assert_eq!(pkt[12] & C1_SPEED_MASK, speed, "speed {speed:#04x}");
            assert_eq!(pkt[12] & C1_CONFIG_BOTH, C1_CONFIG_BOTH);
        }
    }

    #[test]
    fn keepalive_chunk_replicated_in_both_chunks() {
        let pkt = build_keepalive_packet(0, C1_SPEED_192K, 0x44, 1);
        let c0 = &pkt[8..8 + CHUNK_SIZE];
        let c1 = &pkt[8 + CHUNK_SIZE..8 + CHUNK_SIZE * 2];
        assert_eq!(c0, c1);
    }

    #[test]
    fn keepalive_encodes_oc_rx_bits_in_c2() {
        // Wire convention: C2[7:1] = OC1..OC7, C2[0] = reserved. So the
        // user-facing LSB-first 7-bit mask must be left-shifted by one to land
        // in the wire position (the reference shifts the relay bits the same way).
        // 0x44 (relays 3 + 7) → C2 = 0x88
        let pkt = build_keepalive_packet(0, C1_SPEED_192K, 0x44, 1);
        assert_eq!(pkt[13], 0x44 << 1);
        // all seven relays on (0x7F) → C2 = 0xFE (bit 0 stays 0)
        let pkt = build_keepalive_packet(0, C1_SPEED_192K, 0x7F, 1);
        assert_eq!(pkt[13], 0x7F << 1);
        // bit 7 is reserved → dropped from C2 (C2 only has 7 usable bits)
        let pkt = build_keepalive_packet(0, C1_SPEED_192K, 0x80, 1);
        assert_eq!(pkt[13] & 0xFE, 0x00);
        // bit 0 must not set C2[0] for any mask
        for mask in [0u8, 0x01, 0x04, 0x44, 0x7F, 0xFF] {
            let pkt = build_keepalive_packet(0, C1_SPEED_192K, mask, 1);
            assert_eq!(pkt[13] & 0x01, 0x00);
        }
    }

    #[test]
    fn keepalive_c4_sets_duplex_and_receiver_count() {
        // reference: C4 = 0x04 | ((n_recv-1) << 3)
        //   n_recv=1 → 0x04; n_recv=2 → 0x0C; n_recv=5 → 0x24.
        let pkt = build_keepalive_packet(0, C1_SPEED_192K, 0x00, 1);
        assert_eq!(pkt[15], 0x04);
        let pkt = build_keepalive_packet(0, C1_SPEED_192K, 0x00, 2);
        assert_eq!(pkt[15], 0x04 | (1 << 3)); // 0x0C
        let pkt = build_keepalive_packet(0, C1_SPEED_192K, 0x00, 5);
        assert_eq!(pkt[15], 0x04 | (4 << 3)); // 0x24
    }

    #[test]
    fn build_lna_frame_layout() {
        let pkt = build_lna_gain_frame(0, 0, C1_SPEED_192K, 0x00, 1);
        assert_eq!(pkt[3], ENDPOINT_CONTROL);
        // chunk 1: baseline — sync prefix + C0=0x00 at offset 11
        assert_eq!(pkt[8], C_SYNC);
        assert_eq!(pkt[9], C_SYNC);
        assert_eq!(pkt[10], C_SYNC);
        assert_eq!(pkt[11], 0x00);
        // chunk 2: LNA register write — C0 at 8+512+3, C4 at 8+512+7.
        // 0 dB → 0x40 | (0+12)&0x3F = 0x4C.
        assert_eq!(pkt[8 + CHUNK_SIZE + 3], 0x14);
        assert_eq!(pkt[8 + CHUNK_SIZE + 7], LNA_MODE_SET | 0x0C);
        // chunk 1 and chunk 2 must be *different* (baseline vs. write)
        let c0 = &pkt[8..8 + CHUNK_SIZE];
        let c1 = &pkt[8 + CHUNK_SIZE..8 + CHUNK_SIZE * 2];
        assert_ne!(c0, c1);
    }

    #[test]
    fn build_lna_frame_gain_range() {
        // LNA write is now in chunk 2 (offset 8+CHUNK_SIZE+7).
        let off = 8 + CHUNK_SIZE + 7;
        // −12 dB → gain bits 0x00 → C4 = 0x40
        assert_eq!(
            build_lna_gain_frame(0, -12, C1_SPEED_192K, 0x00, 1)[off],
            LNA_MODE_SET | 0x00
        );
        // 0 dB → +12 → 0x0C → 0x4C
        assert_eq!(
            build_lna_gain_frame(0, 0, C1_SPEED_192K, 0x00, 1)[off],
            LNA_MODE_SET | 0x0C
        );
        // +48 dB → +60 → 0x3C → 0x7C
        assert_eq!(
            build_lna_gain_frame(0, 48, C1_SPEED_192K, 0x00, 1)[off],
            LNA_MODE_SET | 0x3C
        );
        // +6 dB (our default) → +18 → 0x12 → 0x52
        assert_eq!(
            build_lna_gain_frame(0, 6, C1_SPEED_192K, 0x00, 1)[off],
            LNA_MODE_SET | 0x12
        );
    }

    #[test]
    fn build_nco_packet_baseline_plus_write_two_chunks() {
        // Layout: baseline in chunk 1, then the NCO write in
        // the *second* 512-byte chunk:
        //   chunk 1: 0x7F 0x7F 0x7F  0x00  (C1=CONFIG_BOTH|SPEED)  C2  0x00  C4(duplex|rcv)
        //   chunk 2: 0x7F 0x7F 0x7F  C0    freq[3]  freq[2]  freq[1]  freq[0]
        let pkt = build_nco_packet(0, 1, 14_200_000, C1_SPEED_192K, 0x00, 1);
        assert_eq!(pkt[3], ENDPOINT_CONTROL);

        // chunk 1 = baseline
        assert_eq!(pkt[8], C_SYNC);
        assert_eq!(pkt[9], C_SYNC);
        assert_eq!(pkt[10], C_SYNC);
        assert_eq!(pkt[11], 0x00); // C0
        assert_eq!(pkt[12], C1_CONFIG_BOTH | C1_SPEED_192K); // C1
        assert_eq!(pkt[15], 0x04); // C4: duplex, n_recv=1

        // chunk 2 = NCO write (offset 8 + CHUNK_SIZE + …)
        let c2 = 8 + CHUNK_SIZE;
        assert_eq!(pkt[c2], C_SYNC);
        assert_eq!(pkt[c2 + 1], C_SYNC);
        assert_eq!(pkt[c2 + 2], C_SYNC);
        assert_eq!(pkt[c2 + 3], 0x04); // slot=1 (RX1) → C0 = 0x04
        // frequency BE: 14200000 = 0x00D8_7700 → C1=00 C2=D8 C3=77 C4=00
        let f = u32::from_be_bytes([pkt[c2 + 4], pkt[c2 + 5], pkt[c2 + 6], pkt[c2 + 7]]);
        assert_eq!(f, 14_200_000);

        // chunk 1 and chunk 2 must differ (this is the whole point of the fix)
        let baseline = &pkt[8..8 + CHUNK_SIZE];
        let write = &pkt[c2..c2 + CHUNK_SIZE];
        assert_ne!(baseline, write);
    }

    #[test]
    fn build_nco_packet_slot_to_c0_table() {
        // Lock the full RX1..RX7 → C0 table against PROTOCOL.md §7
        // (0x02..0x08) → (0x04,0x06,0x08,…).
        for (slot, expected_c0) in [
            (1u8, 0x04u8), // RX1
            (2, 0x06),     // RX2
            (3, 0x08),     // RX3
            (4, 0x0A),     // RX4
            (5, 0x0C),     // RX5
            (6, 0x0E),     // RX6
            (7, 0x10),     // RX7
        ] {
            let pkt = build_nco_packet(0, slot, 7_074_000, C1_SPEED_192K, 0x00, 1);
            let c2 = 8 + CHUNK_SIZE;
            assert_eq!(pkt[3], ENDPOINT_CONTROL, "slot {slot}");
            assert_eq!(pkt[c2 + 3], expected_c0, "slot {slot} → C0");
        }
    }

    #[test]
    fn build_nco_packet_baseline_carries_receiver_count() {
        // n_recv=5 → C4 = 0x04 | (4 << 3) = 0x24, in the *baseline* chunk.
        let pkt = build_nco_packet(0, 1, 10_000_000, C1_SPEED_192K, 0x00, 5);
        assert_eq!(pkt[15], 0x04 | (4 << 3));
        // NCO still in chunk 2 with C0 = 0x04 (RX1) and the 10 MHz frequency:
        // 10,000,000 = 0x0098_9680 → C1=00 C2=98 C3=96 C4=80
        let c2 = 8 + CHUNK_SIZE;
        assert_eq!(pkt[c2 + 3], 0x04);
        let f = u32::from_be_bytes([pkt[c2 + 4], pkt[c2 + 5], pkt[c2 + 6], pkt[c2 + 7]]);
        assert_eq!(f, 10_000_000);
    }

    #[test]
    fn extract_iq_little_endian() {
        let mut chunk = [0u8; CHUNK_SIZE];
        // sample[0] = 32767 (0x7FFF LE), sample[1] = -1 (0xFFFF LE)
        chunk[0] = 0xFF;
        chunk[1] = 0x7F;
        chunk[2] = 0xFF;
        chunk[3] = 0xFF;
        let s = extract_iq_from_chunk(&chunk);
        assert_eq!(s[0], 32767);
        assert_eq!(s[1], -1);
    }

    #[test]
    fn extract_iq_count() {
        let chunk = [0u8; CHUNK_SIZE];
        let s = extract_iq_from_chunk(&chunk);
        assert_eq!(s.len(), SAMPLES_PER_CHUNK);
    }

    #[test]
    fn ep6_sync_detection() {
        let mut chunk = [0u8; CHUNK_SIZE];
        // no sync prefix → None
        assert!(parse_chunk_command(&chunk).is_none());
        // with sync → Some, but C0=0
        chunk[0] = C_SYNC;
        chunk[1] = C_SYNC;
        chunk[2] = C_SYNC;
        let cmd = parse_chunk_command(&chunk).unwrap();
        assert_eq!(cmd.data, 0);
    }

    /// Helper: write a 24-bit signed value as 3 big-endian bytes at `off`.
    fn put_24be(buf: &mut [u8], off: usize, v: i32) {
        buf[off] = ((v >> 16) & 0xFF) as u8;
        buf[off + 1] = ((v >> 8) & 0xFF) as u8;
        buf[off + 2] = (v & 0xFF) as u8;
    }

    #[test]
    fn baseband_parse_24be_roundtrip() {
        // Positive
        {
            let b = [0x00, 0x01, 0x00u8]; // 256
            assert!((sample_24be(&b) - 256.0 / 8_388_607.0).abs() < 1e-9);
        }
        // Negative (sign-extension)
        {
            let b = [0xFF, 0xFE, 0x00u8]; // -512
            assert!((sample_24be(&b) + 512.0 / 8_388_607.0).abs() < 1e-9);
        }
        // Most-negative 24-bit (0x80_0000 = -2^23) → close to -1.0
        {
            let b = [0x80, 0x00, 0x00u8];
            assert!((sample_24be(&b) + 1.0).abs() < 1e-6);
        }
        // -1 (all ones in 24-bit two's complement)
        {
            let b = [0xFF, 0xFF, 0xFFu8];
            assert!((sample_24be(&b) + 1.0 / 8_388_607.0).abs() < 1e-9);
        }
    }

    #[test]
    fn baseband_valid_n_table() {
        // Valid iff 504 % (6n+2) == 0: n=1 → 8 divides 504 (63×);
        // n=2 → 14 divides 504 (36×). 3→20, 4→26, … are not divisors.
        assert!(baseband_valid_n(1));
        assert!(baseband_valid_n(2));
        assert!(!baseband_valid_n(3));
        assert!(!baseband_valid_n(4));
        // Recompute: valid iff 504 % (6n+2)==0
        for n in 1..=8u8 {
            let expected = 504 % (6usize * n as usize + 2) == 0;
            assert_eq!(baseband_valid_n(n), expected, "n={n}");
        }
    }

    #[test]
    fn baseband_chunk_valid_n1() {
        let mut chunk = [0u8; CHUNK_SIZE];
        chunk[0] = C_SYNC;
        chunk[1] = C_SYNC;
        chunk[2] = C_SYNC;
        chunk[3] = 0x01; // C0 (PTT bit set — doesn't matter anymore)
        // Record 0: I=+1000, Q=-2000, mic=12345
        put_24be(&mut chunk, 8, 1000);
        put_24be(&mut chunk, 11, -2000);
        chunk[14] = 0x30;
        chunk[15] = 0x39; // 12345
        let bc = parse_baseband_chunk(&chunk, 1).unwrap();
        assert_eq!(bc.rx_count(), 1);
        let count = (CHUNK_SIZE - BASEBAND_HEADER_OFFSET) / baseband_record_len(1);
        assert_eq!(bc.iq().len(), count);
        assert_eq!(count, 63); // 504 / 8
        let c = bc.iq()[0];
        assert!((c.re - 1000.0 / 8_388_607.0).abs() < 1e-9);
        assert!((c.im + 2000.0 / 8_388_607.0).abs() < 1e-9);
    }

    #[test]
    fn baseband_chunk_valid_n2() {
        let mut chunk = [0u8; CHUNK_SIZE];
        chunk[0] = C_SYNC;
        chunk[1] = C_SYNC;
        chunk[2] = C_SYNC;
        // Record 0: RX1 I=+1, RX2 I=+2, mic=0
        put_24be(&mut chunk, 8, 1);
        put_24be(&mut chunk, 14, 2);
        let bc = parse_baseband_chunk(&chunk, 2).unwrap();
        assert_eq!(bc.rx_count(), 2);
        let count = (CHUNK_SIZE - BASEBAND_HEADER_OFFSET) / baseband_record_len(2);
        assert_eq!(count, 36); // 504 / 14
        assert_eq!(bc.per_rx[0][0].re, 1.0 / 8_388_607.0);
        assert_eq!(bc.per_rx[1][0].re, 2.0 / 8_388_607.0);
    }

    #[test]
    fn baseband_chunk_valid_n3_truncates() {
        // N=3 → record = 20 B → 504 / 20 = 25 whole records + 4 trailing bytes.
        // We take the plain integer division (same as the reference), so
        // exactly 25 records are decoded
        // and the 4 trailing bytes of the 504-byte payload are ignored.
        //
        // Synthetic: fill all 25 records with distinct I values per receiver.
        let mut chunk = [0u8; CHUNK_SIZE];
        chunk[0] = C_SYNC;
        chunk[1] = C_SYNC;
        chunk[2] = C_SYNC;
        let rec = baseband_record_len(3);
        let count = (CHUNK_SIZE - BASEBAND_HEADER_OFFSET) / rec;
        assert_eq!(count, 25, "N=3 → 25 whole records per chunk");
        for r in 0..count {
            for rx in 0..3 {
                // Encode a unique value per (record, rx) so we can verify
                // placement.
                let v = (r * 3 + rx) as i32 + 1;
                put_24be(&mut chunk, BASEBAND_HEADER_OFFSET + r * rec + rx * 6, v);
            }
        }
        let bc = parse_baseband_chunk(&chunk, 3).unwrap();
        assert_eq!(bc.rx_count(), 3);
        for rx in 0..3 {
            assert_eq!(
                bc.per_rx[rx].len(),
                count,
                "receiver {rx} gets {count} samples"
            );
        }
        // Spot check: receiver 2 (index 2), record 0 → I should be 3.
        assert_eq!(bc.per_rx[2][0].re, 3.0 / 8_388_607.0);
        // Spot check: receiver 0 (index 0), record 10 → I should be 10*3+1 = 31.
        assert_eq!(bc.per_rx[0][10].re, 31.0 / 8_388_607.0);
    }

    #[test]
    fn baseband_chunk_valid_n4_truncates() {
        // N=4 → record = 26 B → 504 / 26 = 19 whole records + 10 trailing bytes.
        // Same plain-integer-division rule as the reference; decode 19 records.
        let mut chunk = [0u8; CHUNK_SIZE];
        chunk[0] = C_SYNC;
        chunk[1] = C_SYNC;
        chunk[2] = C_SYNC;
        let rec = baseband_record_len(4);
        let count = (CHUNK_SIZE - BASEBAND_HEADER_OFFSET) / rec;
        assert_eq!(
            count, 19,
            "N=4 → 19 whole records per chunk (504/26 truncated)"
        );
        for r in 0..count {
            for rx in 0..4 {
                let v = (r * 4 + rx) as i32 + 1;
                put_24be(&mut chunk, BASEBAND_HEADER_OFFSET + r * rec + rx * 6, v);
            }
        }
        let bc = parse_baseband_chunk(&chunk, 4).unwrap();
        assert_eq!(bc.rx_count(), 4);
        for rx in 0..4 {
            assert_eq!(
                bc.per_rx[rx].len(),
                count,
                "receiver {rx} gets {count} samples"
            );
        }
        // Receiver 3 (index 3), record 0 → I = 4.
        assert_eq!(bc.per_rx[3][0].re, 4.0 / 8_388_607.0);
        // Receiver 0 (index 0), record 5 → I = 5*4 + 1 = 21.
        assert_eq!(bc.per_rx[0][5].re, 21.0 / 8_388_607.0);
    }

    #[test]
    fn baseband_chunk_requires_sync() {
        let chunk = [0u8; CHUNK_SIZE];
        assert!(parse_baseband_chunk(&chunk, 1).is_none());
    }

    #[test]
    fn baseband_frame_yields_two_chunks() {
        let mut pkt = [0u8; DATA_PACKET_SIZE];
        pkt[0] = 0xEF;
        pkt[1] = 0xFE;
        pkt[2] = 0x01;
        pkt[3] = ENDPOINT_DATA_TX;
        let c0_off = HEADER_SIZE;
        pkt[c0_off] = C_SYNC;
        pkt[c0_off + 1] = C_SYNC;
        pkt[c0_off + 2] = C_SYNC;
        let c1_off = HEADER_SIZE + CHUNK_SIZE;
        pkt[c1_off] = C_SYNC;
        pkt[c1_off + 1] = C_SYNC;
        pkt[c1_off + 2] = C_SYNC;
        let v = parse_baseband_frame(&pkt, 1);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].rx_count(), 1);
        assert_eq!(v[1].rx_count(), 1);
    }

    #[test]
    fn assembler_yields_after_enough_frames() {
        use crate::protocol::DATA_PACKET_SIZE;
        let mut pkt = [0u8; DATA_PACKET_SIZE];
        pkt[0] = 0xEF;
        pkt[1] = 0xFE;
        pkt[2] = 0x01;
        pkt[3] = ENDPOINT_WIDEBAND;

        let mut assembler = BlockAssembler::new();
        // one real EP4 frame yields 2 chunks * 256 = 512 samples
        let frame: Vec<i16> = vec![1i16; SAMPLES_PER_CHUNK * 2];
        let mut got = false;
        for i in 0..10 {
            if let Some(Item::Block(b)) = assembler.feed(
                &DataHeader {
                    endpoint: ENDPOINT_WIDEBAND,
                    seq: i as u32,
                },
                &frame,
            ) {
                assert_eq!(b.samples.len(), IQ_PAIRS_PER_BLOCK);
                got = true;
                break;
            }
        }
        assert!(got, "assembler should have emitted a block");
    }

    /// `extract_iq_into_chunk` must produce the same values (and length) as
    /// `extract_iq_from_chunk` for arbitrary chunk bytes.
    #[test]
    fn ep4_into_matches_owned() {
        use crate::protocol::ENDPOINT_DATA_TX as _;
        let mut chunk = [0u8; CHUNK_SIZE];
        // Fill with a deterministic pseudo pattern.
        for (i, b) in chunk.iter_mut().enumerate() {
            *b = (i * 31 + i / 2) as u8;
        }
        let owned = extract_iq_from_chunk(&chunk);
        let mut into: Vec<i16> = Vec::with_capacity(SAMPLES_PER_CHUNK);
        into.resize(SAMPLES_PER_CHUNK, 0);
        extract_iq_into_chunk(&chunk, &mut into[..]);
        assert_eq!(into, owned);
    }

    /// `parse_baseband_frame_into` must yield the same chunks as
    /// `parse_baseband_frame` for valid (n_recv=1, 2) and the two-chunk frame.
    #[test]
    fn frame_into_matches_owned() {
        for n in [1u8, 2, 3] {
            let mut pkt = [0u8; DATA_PACKET_SIZE];
            pkt[0] = 0xEF;
            pkt[1] = 0xFE;
            pkt[2] = 0x01;
            pkt[3] = ENDPOINT_DATA_TX;
            // Deterministic payload with valid sync in both chunks and
            // interleaved I/Q bytes.
            for (i, b) in pkt.iter_mut().enumerate() {
                *b = (i * 17 + 3) as u8;
            }
            for off in [HEADER_SIZE, HEADER_SIZE + CHUNK_SIZE] {
                pkt[off] = C_SYNC;
                pkt[off + 1] = C_SYNC;
                pkt[off + 2] = C_SYNC;
            }

            let owned = parse_baseband_frame(&pkt, n);

            let mut reused: Vec<BasebandChunk> = Vec::new();
            // Call twice with alternating sizes to catch stale-state bugs.
            parse_baseband_frame_into(&pkt, 1, &mut reused);
            parse_baseband_frame_into(&pkt, n, &mut reused);

            assert_eq!(reused.len(), owned.len(), "n={n}");
            for (a, b) in reused.iter().zip(owned.iter()) {
                assert_eq!(a.rx_count(), b.rx_count());
                for (sa, sb) in a.per_rx.iter().zip(b.per_rx.iter()) {
                    assert_eq!(sa, sb, "n={n}");
                }
            }
        }
    }

    /// `parse_baseband_frame_into` must clear on a non-matching frame
    /// (so an EP4 frame after an EP6 frame leaves 0 chunks).
    #[test]
    fn frame_into_clears_on_bad_frame() {
        use crate::protocol::ENDPOINT_DATA_TX;
        let mut good = [0u8; DATA_PACKET_SIZE];
        good[3] = ENDPOINT_DATA_TX;
        for off in [HEADER_SIZE, HEADER_SIZE + CHUNK_SIZE] {
            good[off] = C_SYNC;
            good[off + 1] = C_SYNC;
            good[off + 2] = C_SYNC;
        }
        good[0] = 0xEF;
        good[1] = 0xFE;
        good[2] = 0x01;

        let mut out: Vec<BasebandChunk> = Vec::new();
        parse_baseband_frame_into(&good, 1, &mut out);
        assert_eq!(out.len(), 2);

        // Now feed a "bad" frame (missing sync in either chunk).
        let mut bad = good;
        bad[HEADER_SIZE] = 0x00; // destroy first chunk sync
        parse_baseband_frame_into(&bad, 1, &mut out);
        // Only the second chunk remains.
        assert_eq!(out.len(), 1);
        assert!(!out[0].per_rx[0].is_empty());
    }

    /// `parse_receive_packet_into` must agree with `parse_receive_packet`
    /// on both the EP4 (`iq_samples`) and EP6 (`baseband`) paths, and the
    /// non-EP6 branches leave the caller-provided baseband buffer alone
    /// (so a no-op-dealloc bump-allocator doesn't leak on interleaved
    /// ACK / data traffic).
    #[test]
    fn receive_packet_into_matches_owned() {
        use crate::protocol::ENDPOINT_DATA_TX;
        // EP4 (wideband).
        let mut ep4 = [0u8; DATA_PACKET_SIZE];
        ep4[0] = 0xEF;
        ep4[1] = 0xFE;
        ep4[2] = 0x01;
        ep4[3] = ENDPOINT_WIDEBAND;
        for (i, b) in ep4.iter_mut().enumerate() {
            *b = (i ^ 0x55) as u8;
        }
        ep4[0] = 0xEF;
        ep4[1] = 0xFE;
        ep4[2] = 0x01;
        ep4[3] = ENDPOINT_WIDEBAND;

        let a = parse_receive_packet(&ep4, 1).unwrap();
        let a_iq = a.iq_samples.unwrap();
        let mut iq: Vec<i16> = Vec::new();
        let mut bb: Vec<BasebandChunk> = Vec::new();
        {
            let _b = parse_receive_packet_into(&ep4, 1, &mut iq, &mut bb).unwrap();
            // `header` is `Copy` — capture the endpoint so we can verify the
            // non-EP6 gate even after `_b` (and its `&mut` borrows on
            // `iq`/`bb`) is dropped.
            assert!(_b.header.endpoint != ENDPOINT_DATA_TX);
        } // `_b` dropped here; `iq`/`bb` borrow released.
        assert_eq!(&*a_iq, &*iq);
        // EP4: the `Into` variant leaves the caller-provided `baseband`
        // buffer alone (freshly-empty here, so it stays len 0). The owned
        /// variant returns an empty `Vec` for `baseband` on EP4 frames.
        assert!(a.baseband.is_empty());
        assert!(bb.is_empty());
        assert_eq!(a.baseband, bb);

        // EP6 (baseband).
        let mut ep6 = [0u8; DATA_PACKET_SIZE];
        ep6[0] = 0xEF;
        ep6[1] = 0xFE;
        ep6[2] = 0x01;
        ep6[3] = ENDPOINT_DATA_TX;
        for (i, byte) in ep6.iter_mut().enumerate() {
            *byte = (i.wrapping_mul(7) + 1) as u8;
        }
        ep6[0] = 0xEF;
        ep6[1] = 0xFE;
        ep6[2] = 0x01;
        ep6[3] = ENDPOINT_DATA_TX;
        for off in [HEADER_SIZE, HEADER_SIZE + CHUNK_SIZE] {
            ep6[off] = C_SYNC;
            ep6[off + 1] = C_SYNC;
            ep6[off + 2] = C_SYNC;
        }
        let a6 = parse_receive_packet(&ep6, 2).unwrap();
        let mut iq2: Vec<i16> = Vec::new();
        let mut bb2: Vec<BasebandChunk> = Vec::new();
        {
            let _b6 = parse_receive_packet_into(&ep6, 2, &mut iq2, &mut bb2).unwrap();
            assert!(_b6.header.endpoint == ENDPOINT_DATA_TX);
        } // `_b6` dropped; `iq2`/`bb2` borrow released.
        assert!(a6.iq_samples.is_none());
        assert!(iq2.is_empty());
        assert_eq!(a6.baseband.len(), bb2.len());
        assert_eq!(a6.baseband, bb2);
    }

    /// The `Into` variant must *preserve* the caller's baseband chunk (and
    /// its inner `per_rx` capacity) across a non-EP6 frame. This is the
    /// invariant the no-std bump-allocator path relies on: on Teensy,
    /// `dealloc` is a no-op, so a `clear()` followed by the next EP6
    /// `push(Vec::new())` is a permanent leak of the inner `Vec<Complex>`.
    /// The owned variant still returns a fresh (empty) `Vec` on non-EP6
    /// frames — only the `Into` variant's reuse contract differs.
    #[test]
    fn receive_packet_into_preserves_baseband_on_non_ep6() {
        use crate::protocol::ENDPOINT_DATA_TX;
        // A valid EP6 frame (populates baseband).
        let mut ep6 = [0u8; DATA_PACKET_SIZE];
        ep6[0] = 0xEF;
        ep6[1] = 0xFE;
        ep6[2] = 0x01;
        ep6[3] = ENDPOINT_DATA_TX;
        for (i, byte) in ep6.iter_mut().enumerate() {
            *byte = (i.wrapping_mul(7) + 1) as u8;
        }
        ep6[0] = 0xEF;
        ep6[1] = 0xFE;
        ep6[2] = 0x01;
        ep6[3] = ENDPOINT_DATA_TX;
        for off in [HEADER_SIZE, HEADER_SIZE + CHUNK_SIZE] {
            ep6[off] = C_SYNC;
            ep6[off + 1] = C_SYNC;
            ep6[off + 2] = C_SYNC;
        }
        // A non-EP6 frame (EP2 control) with a well-formed header.
        let mut cc = [0u8; DATA_PACKET_SIZE];
        cc[0] = 0xEF;
        cc[1] = 0xFE;
        cc[2] = 0x01;
        cc[3] = ENDPOINT_CONTROL;
        cc[7] = 1;

        let mut bb: Vec<BasebandChunk> = Vec::new();
        // 1. Feed the EP6 frame → populates two chunks with non-empty `per_rx`.
        {
            let _p = parse_receive_packet_into(&ep6, 1, &mut Vec::new(), &mut bb).unwrap();
            assert_eq!(bb.len(), 2, "two chunks from the EP6 frame");
            assert!(!bb[0].per_rx[0].is_empty(), "chunk 0 populated");
        }
        let cap_before = bb[0].per_rx[0].capacity();
        assert!(cap_before > 0, "inner buffer allocated at least once");

        // 2. Feed a non-EP6 frame into the *same* buffer.
        {
            let _p = parse_receive_packet_into(&cc, 1, &mut Vec::new(), &mut bb).unwrap();
        }
        // The new contract: the chunks and their inner capacity are
        // preserved (not cleared), so a subsequent EP6 frame refills
        // without reallocating.
        assert_eq!(
            bb.len(),
            2,
            "non-EP6 frame must not clear the caller's baseband"
        );
        assert_eq!(
            bb[0].per_rx[0].capacity(),
            cap_before,
            "inner capacity must survive a non-EP6 frame"
        );
    }
}
