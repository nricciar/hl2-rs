//! Shared HL2 session state + frame drivers.
//!
//! The wire *bytes* live in [`super::data`] / [`super::discovery`]; this module
//! is the thin "how a client drives the radio" layer on top: it owns the
//! **send-sequence counter** and the **persistent session config**
//! (C1 SPEED bits, RX open-collector relay mask, active receiver count) and
//! hands back ready-to-send frames, each stamped with the next sequence
//! number.
//!
//! It exists so the two transports — the tokio `hl2::hl2` client and the
//! `no_std` `teensy` smoltcp radio — stop duplicating:
//!
//! * the take-the-sequence-and-increment step (was
//!   `Shared::next_send_seq` in `hl2::hl2` and `Radio::next_seq` in
//!   `teensy::radio::control`, byte-for-byte identical),
//! * passing the same `(c1_speed, oc_bits, n_recv)` triple into every
//!   `build_*` call on every client,
//! * re-deriving "is this datagram a discovery reply / an EP4 wideband frame /
//!   an EP6 baseband frame / an EP2 C&C frame" inline in each receive loop.
//!
//! ## What it is *not*
//!
//! * The async / polling **driver** — the tokio client drives a `select!`
//!   loop, the smoltcp client a hand-rolled poll loop, each with its own
//!   socket + timers. The shared part here is the *bytes + ordering*, not the
//!   IO plumbing.
//! * The START-ack phase machine, the per-slot `BasebandRing` / demod, or the
//!   block assembler — those stay in `crate::hl2` / `crate::receiver`
//!   (`dsp`). `Session` is deliberately small and heap-free: `Copy`, no
//!   interior mutability, no allocation. Wrap it in your own
//!   `Arc<Mutex<…>>` if you need to share it across threads (the tokio client
//!   does exactly that).
//!
//! `no_std`-clean: depends only on the wire builders in [`super::data`],
//! `super::discovery` and the constants in [`super`]. No `alloc`, no `std`.

use super::data::{
    build_keepalive_packet, build_lna_gain_frame, build_nco_packet, build_start_stop_frame,
};
use super::discovery::{DiscoveryInfo, parse_discovery_response};
use super::{
    C1_SPEED_96K, DATA_PACKET_SIZE, DISCOVERY_RESPONSE_SIZE, ENDPOINT_CONTROL, ENDPOINT_DATA_TX,
    ENDPOINT_WIDEBAND, METIS_MARKER, START_REQUEST_SIZE, STATUS_NOT_SENDING, STATUS_SENDING,
};

/// Persistent session configuration for one HL2.
///
/// These three values are re-asserted in *every* host→radio C&C frame (the
/// keep-alive baseline and the first chunk of every NCO/LNA write) so the
/// gateware holds the per-receiver DDC rate and the open-collector filter
/// relay between register writes. They are stable for the life of a session;
/// LNA gain and the NCO frequency are *commands* ([`Session::lna_frame`],
/// [`Session::tune_frame`]), not config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionConfig {
    /// C1 SPEED bits, raw (one of `C1_SPEED_48K / _96K / _192K / _384K`).
    pub c1_speed: u8,
    /// RX open-collector filter relay mask, LSB-first (bit 0 = relay 1 … bit 6 = relay 7).
    /// `0x00` = all relays off (board default); the wire puts it in C2[7:1].
    pub oc_bits: u8,
    /// Number of active receiver slots (`1` is the common single-RX1 case).
    pub n_recv: u8,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            c1_speed: C1_SPEED_96K,
            oc_bits: 0,
            n_recv: 1,
        }
    }
}

/// The kind of a received UDP datagram, as the receive side needs to classify
/// it before dispatch. Both the tokio pump and the smoltcp radio task used to
/// re-derive this inline; [`Session::classify`] is the single place now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatagramKind {
    /// A 60-byte discovery **response** (marker + status `0x02`/`0x03`).
    Discovery,
    /// A 1032-byte **EP4 wideband** frame (endpoint `0x04`).
    Ep4Wideband,
    /// A 1032-byte **EP6 baseband** frame (endpoint `0x06`).
    Ep6Baseband,
    /// A 1032-byte **EP2 C&C** frame (endpoint `0x02`) — keep-alive / ACKs.
    Ep2Control,
    /// Anything else (wrong size, wrong marker, unknown endpoint).
    Other,
}

/// One driven HL2 session: the send-sequence counter plus the persistent
/// [`SessionConfig`]. `Copy` by design — it is value state the caller owns and
/// mutates in its own loop; no interior mutability, so no lock is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Session {
    config: SessionConfig,
    /// Monotonic send sequence. The HL2 does not strictly enforce monotonicity;
    /// wrap-around is allowed. Each [`Session::stop_frame`] / [`start_frame`]
    /// / [`lna_frame`] / [`tune_frame`] / [`keepalive_frame`] stamps it and
    /// increments it, exactly as the two transports did independently.
    seq: u32,
}

impl Session {
    /// Begin a session with `config` and sequence reset to `0`.
    pub fn new(config: SessionConfig) -> Self {
        Self { config, seq: 0 }
    }

    /// The persistent session config this session drives frames with.
    #[inline]
    pub fn config(&self) -> SessionConfig {
        self.config
    }

    /// Return the current send sequence and advance it (wrapping at 2³²).
    ///
    /// This is the one-line `cur; self = cur.wrapping_add(1); cur` that used to
    /// be copy-pasted into both transports. Kept public so a client that builds
    /// its own frames (rare) can stay consistent; the frame helpers below use
    /// it internally.
    #[inline]
    pub fn next_seq(&mut self) -> u32 {
        let cur = self.seq;
        self.seq = cur.wrapping_add(1);
        cur
    }

    /// Build the 64-byte **STOP** frame (C&C). Stamps the next sequence.
    pub fn stop_frame(&mut self) -> [u8; START_REQUEST_SIZE] {
        build_start_stop_frame(self.next_seq(), false)
    }

    /// Build the 64-byte **START** frame (C&C). Stamps the next sequence.
    pub fn start_frame(&mut self) -> [u8; START_REQUEST_SIZE] {
        build_start_stop_frame(self.next_seq(), true)
    }

    /// Build the 1032-byte **LNA-gain** C&C frame for `gain_db`
    /// (−12…+48). Stamps the next sequence.
    pub fn lna_frame(&mut self, gain_db: i8) -> [u8; DATA_PACKET_SIZE] {
        let c = self.config;
        build_lna_gain_frame(self.next_seq(), gain_db, c.c1_speed, c.oc_bits, c.n_recv)
    }

    /// Build the 1032-byte **NCO** C&C frame tuning `slot` (1-based) to `hz`.
    /// Stamps the next sequence.
    pub fn tune_frame(&mut self, slot: u8, hz: u32) -> [u8; DATA_PACKET_SIZE] {
        let c = self.config;
        build_nco_packet(self.next_seq(), slot, hz, c.c1_speed, c.oc_bits, c.n_recv)
    }

    /// Build the 1032-byte **keep-alive** C&C frame. Stamps the next sequence.
    pub fn keepalive_frame(&mut self) -> [u8; DATA_PACKET_SIZE] {
        let c = self.config;
        build_keepalive_packet(self.next_seq(), c.c1_speed, c.oc_bits, c.n_recv)
    }

    /// Classify a received datagram into a [`DatagramKind`].
    ///
    /// Distincts discovery (60 B) from data frames (1032 B) by length + marker,
    /// then the four data-frame endpoints by the header endpoint byte. Returns
    /// [`DatagramKind::Other`] when it matches none — the caller drops or logs
    /// it.
    #[inline]
    pub fn classify(dgram: &[u8]) -> DatagramKind {
        // Discovery response: `[EFEF][status 0x02|0x03][…]`, 60 bytes.
        if dgram.len() == DISCOVERY_RESPONSE_SIZE
            && dgram[0] == METIS_MARKER[0]
            && dgram[1] == METIS_MARKER[1]
            && (dgram[2] == STATUS_NOT_SENDING || dgram[2] == STATUS_SENDING)
        {
            return DatagramKind::Discovery;
        }
        // Data frame: `[EFEF][0x01][endpoint][seq 4B]`, 1032 bytes.
        if dgram.len() == DATA_PACKET_SIZE
            && dgram[0] == METIS_MARKER[0]
            && dgram[1] == METIS_MARKER[1]
            && dgram[2] == 0x01
        {
            return match dgram[3] {
                ENDPOINT_WIDEBAND => DatagramKind::Ep4Wideband,
                ENDPOINT_DATA_TX => DatagramKind::Ep6Baseband,
                ENDPOINT_CONTROL => DatagramKind::Ep2Control,
                _ => DatagramKind::Other,
            };
        }
        DatagramKind::Other
    }

    /// Attempt to parse `dgram` as a discovery reply. `Some` only if it is a
    /// 60-byte, marker-valid HL2 response — the caller then sets its peer from
    /// `DiscoveryInfo`.
    #[inline]
    pub fn try_parse_discovery(dgram: &[u8]) -> Option<DiscoveryInfo> {
        let slice = dgram.get(..DISCOVERY_RESPONSE_SIZE)?.try_into().ok()?;
        parse_discovery_response(&slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    /// Fixtures: the "real" build_* calls for a known config, so the Session
    /// wrappers can be pinned to the wire codec they delegate to.
    const CFG: SessionConfig = SessionConfig {
        c1_speed: C1_SPEED_96K,
        oc_bits: 0x40,
        n_recv: 1,
    };

    #[test]
    fn default_config_is_96k_single_rx_relay_off() {
        let c = SessionConfig::default();
        assert_eq!(c.c1_speed, C1_SPEED_96K);
        assert_eq!(c.oc_bits, 0);
        assert_eq!(c.n_recv, 1);
    }

    #[test]
    fn next_seq_increments_and_wraps() {
        let mut s = Session::new(CFG);
        assert_eq!(s.next_seq(), 0);
        assert_eq!(s.next_seq(), 1);
        assert_eq!(s.next_seq(), 2);
        // force wrap: consume u32::MAX, counter rolls to 0
        s.seq = u32::MAX;
        assert_eq!(s.next_seq(), u32::MAX);
        assert_eq!(s.next_seq(), 0);
    }

    #[test]
    fn start_stop_frames_match_codec_and_burn_seq() {
        let mut s = Session::new(CFG);
        let stop = s.stop_frame();
        assert_eq!(stop, build_start_stop_frame(0, false));
        let start = s.start_frame();
        assert_eq!(start, build_start_stop_frame(1, true));
    }

    #[test]
    fn lna_frame_matches_codec_and_burns_seq() {
        let mut s = Session::new(CFG);
        // gain +6 dB (DEFAULT_LNA_GAIN_DB)
        assert_eq!(
            s.lna_frame(6),
            build_lna_gain_frame(0, 6, C1_SPEED_96K, 0x40, 1)
        );
    }

    #[test]
    fn tune_frame_matches_codec_and_burns_seq() {
        let mut s = Session::new(CFG);
        // RX1 @ 7.074 MHz
        let frame = s.tune_frame(1, 7_074_000);
        assert_eq!(
            frame,
            build_nco_packet(0, 1, 7_074_000, C1_SPEED_96K, 0x40, 1)
        );
        // chunk-2 C0 for RX1 is 0x04
        assert_eq!(frame[8 + 512 + 3], 0x04);
    }

    #[test]
    fn keepalive_frame_matches_codec_and_burns_seq() {
        let mut s = Session::new(CFG);
        assert_eq!(
            s.keepalive_frame(),
            build_keepalive_packet(0, C1_SPEED_96K, 0x40, 1)
        );
    }

    fn dgram(len_marker_status_endpoint: (usize, u8, u8, u8)) -> Vec<u8> {
        let (len, m0, status, endpoint) = len_marker_status_endpoint;
        let mut v = vec![0u8; len];
        if !v.is_empty() {
            v[0] = 0xEF;
        }
        if v.len() > 1 {
            v[1] = 0xFE;
        }
        if v.len() > 2 {
            v[2] = status;
        }
        if v.len() > 3 {
            v[3] = endpoint;
        }
        // m0 unused (always 0xEF); keep the destructure tidy.
        let _ = m0;
        v
    }

    #[test]
    fn classify_discovery() {
        let d = dgram((DISCOVERY_RESPONSE_SIZE, 0xEF, STATUS_NOT_SENDING, 0));
        assert_eq!(Session::classify(&d), DatagramKind::Discovery);
    }

    #[test]
    fn classify_data_frames_by_endpoint() {
        for (ep, kind) in [
            (ENDPOINT_WIDEBAND, DatagramKind::Ep4Wideband),
            (ENDPOINT_DATA_TX, DatagramKind::Ep6Baseband),
            (ENDPOINT_CONTROL, DatagramKind::Ep2Control),
        ] {
            let d = dgram((DATA_PACKET_SIZE, 0xEF, 0x01, ep));
            assert_eq!(Session::classify(&d), kind, "endpoint {ep:#04x}");
        }
    }

    #[test]
    fn classify_rejects_wrong_size_and_marker() {
        // right length, wrong marker (status byte where marker should be)
        let mut d = vec![0u8; DATA_PACKET_SIZE];
        d[0] = 0xEF;
        d[2] = 0x01;
        d[3] = ENDPOINT_WIDEBAND;
        assert_eq!(Session::classify(&d), DatagramKind::Other, "bad marker");

        // data length but discovery status → not discovery (needs 60 B) and
        // not a data frame (status 0x02 != 0x01)
        let d = dgram((DATA_PACKET_SIZE, 0xEF, STATUS_SENDING, ENDPOINT_WIDEBAND));
        assert_eq!(Session::classify(&d), DatagramKind::Other);

        // discovery status but data length → Other (discovery is 60 B only)
        let d = dgram((DATA_PACKET_SIZE, 0xEF, STATUS_NOT_SENDING, 0));
        assert_eq!(Session::classify(&d), DatagramKind::Other);

        // empty
        let d: Vec<u8> = Vec::new();
        assert_eq!(Session::classify(&d), DatagramKind::Other);
    }

    #[test]
    fn try_parse_discovery_delegates_to_codec() {
        let mut buf = [0u8; DISCOVERY_RESPONSE_SIZE];
        buf[0] = 0xEF;
        buf[1] = 0xFE;
        buf[2] = STATUS_NOT_SENDING;
        buf[3..9].copy_from_slice(&[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC]);
        buf[0x13] = 4;

        // exact-size slice parses
        assert_eq!(
            Session::try_parse_discovery(&buf[..]).map(|i| i.rx_count),
            Some(4)
        );
        // longer slice (padding after the 60 B) still parses the head
        let mut padded = [0u8; DISCOVERY_RESPONSE_SIZE + 16];
        padded[..60].copy_from_slice(&buf);
        assert_eq!(
            Session::try_parse_discovery(&padded[..]).map(|i| i.mac),
            Some([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC])
        );
        // too short → None
        assert!(Session::try_parse_discovery(&buf[..10]).is_none());
        // wrong marker → None
        let mut bad = buf;
        bad[0] = 0x00;
        assert!(Session::try_parse_discovery(&bad[..]).is_none());
    }
}
