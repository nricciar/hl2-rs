//! HL2 radio control: DHCP, discovery, START/STOP, LNA, NCO tune,
//! keep-alive, EP6 datagram RX.
//!
//! Wire bytes come from `hl2::protocol` (AGENTS.md layering rule); this
//! module only sequences them through smoltcp + tracks the send sequence.
//!
//! Fixed session parameters:
//!
//!   RX1          7.074 MHz   (OC filter bank relay: `OC_RELAY`)
//!   DDC rate     96 kSps     (C1 SPEED = `C1_SPEED_96K`)
//!   n_recv       1
//!   LNA          +6 dB       (`DEFAULT_LNA_GAIN_DB`)
//!   keepalive    25 ms target (watchdog limit approximately 168 ms)
//!
//! Socket handles are captured from `SocketSet::add` during setup, not
//! constructed from assumed slot indices.
//!
//! Port model: `LOCAL_PORT` is the local UDP source port the Teensy binds
//! AND uses when sending C&C frames to the HL2. The HL2 echoes this
//! source port back as the destination of its EP6 data stream (the same
//! mechanism the `hl2` crate uses with an ephemeral source port).
//! smoltcp `udp::Socket::bind` requires a non-zero
//! port; once bound, all packets destined to `LOCAL_PORT` land on this
//! socket regardless of source (UDP is connectionless on this socket).

use core::net::Ipv4Addr;

use smoltcp::iface::{Config, Interface, SocketHandle, SocketSet};
use smoltcp::socket::{dhcpv4, udp};
use smoltcp::time::Instant;
use smoltcp::wire::{EthernetAddress, IpAddress, IpEndpoint, IpListenEndpoint};
use static_cell::ConstStaticCell;

use hl2::protocol::discovery::{DiscoveryInfo, discovery_request};
use hl2::protocol::session::Session;
use hl2::protocol::{
    C1_SPEED_96K, DATA_PACKET_SIZE, DEFAULT_LNA_GAIN_DB, HL2_PORT, OC_MASK_RX, START_REQUEST_SIZE,
};

/// OC filter bank relay mask.
/// LSB-first: bit 0 = relay 1 … bit 6 = relay 7 (see `hl2::protocol::OC_MASK_RX`).
/// Current value selects relay 7 (`0b100_0000` = `0x40`).
pub const OC_RELAY: u8 = 0x40 & OC_MASK_RX;

/// RX1 NCO target (Hz) — 7.074 MHz.
pub const TUNE_HZ: u32 = 7_074_000;

/// RX1 slot (1-based; `build_nco_packet` maps slot 1 to C0 = 0x04).
const RX1_SLOT: u8 = 1;

/// Active receiver count (1 = single-slot RX1).
const N_RECV: u8 = 1;

/// DDC rate (kHz) for the status line.
pub const SAMPLE_RATE_KHZ: u32 = 96;

/// Default LNA gain (dB), re-exported for the status text.
pub const LNA_GAIN_DB: i8 = 30; //DEFAULT_LNA_GAIN_DB;

/// Target keep-alive cadence in ms. Blocking work can delay actual sends;
/// the caller must keep gaps below the approximately 168 ms watchdog limit.
pub const KEEPALIVE_INTERVAL_MS_CONST: u64 = 25;

/// Local (Teensy) MAC. Change for each board on the same LAN.
pub const MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];

/// Local UDP source port. Must be non-zero (smoltcp `bind` rejects 0).
/// Any ephemeral value works — the HL2 echoes it on the EP6 stream.
pub const LOCAL_PORT: u16 = 40000;

/// UDP datagram slot count per direction (rx and tx are separate arrays).
/// At 96 kSps, 126 pairs per EP6 packet means about 1.31 ms per packet.
/// Eight slots buffer about 10.5 ms; the ENET ring buffers additional traffic.
const HL2_SLOTS: usize = 8;
/// Payload capacity budget per slot. UDP metadata is stored separately;
/// link, IP and UDP headers are not part of this payload buffer.
const HL2_SLOT: usize = 1280;

static SOCKET_STORAGE: ConstStaticCell<[smoltcp::iface::SocketStorage<'static>; 2]> =
    ConstStaticCell::new([smoltcp::iface::SocketStorage::EMPTY; 2]);
static UDP_RX_META: ConstStaticCell<[udp::PacketMetadata; HL2_SLOTS]> =
    ConstStaticCell::new([udp::PacketMetadata::EMPTY; HL2_SLOTS]);
static UDP_TX_META: ConstStaticCell<[udp::PacketMetadata; HL2_SLOTS]> =
    ConstStaticCell::new([udp::PacketMetadata::EMPTY; HL2_SLOTS]);
static UDP_RX: ConstStaticCell<[u8; HL2_SLOTS * HL2_SLOT]> =
    ConstStaticCell::new([0; HL2_SLOTS * HL2_SLOT]);
static UDP_TX: ConstStaticCell<[u8; HL2_SLOTS * HL2_SLOT]> =
    ConstStaticCell::new([0; HL2_SLOTS * HL2_SLOT]);

/// Build the HL2 UDP socket, taking its separate RX/TX storage once.
/// Panics if called a second time.
pub fn make_udp_socket() -> udp::Socket<'static> {
    let rx = udp::PacketBuffer::new(&mut UDP_RX_META.take()[..], &mut UDP_RX.take()[..]);
    let tx = udp::PacketBuffer::new(&mut UDP_TX_META.take()[..], &mut UDP_TX.take()[..]);
    let mut s = udp::Socket::new(rx, tx);
    s.bind(IpListenEndpoint::from(LOCAL_PORT))
        .expect("nonzero local port on a new socket");
    s
}

/// The radio-side state the `radio` task owns.
///
/// The send-sequence counter + the persistent C&C config (SPEED / OC relay /
/// receiver count) live in [`Session`] (`hl2::protocol::session`), so this
/// module only sequences frames through smoltcp and does no wire-encoding of
/// its own (the layering rule: protocol bytes stay in `hl2`).
pub struct Radio {
    peer: Ipv4Addr,
    session: Session,
}

impl Radio {
    pub fn new() -> Self {
        Self {
            peer: Ipv4Addr::UNSPECIFIED,
            session: Session::new(C1_SPEED_96K, OC_RELAY),
        }
    }

    /// Build a stop frame (64 B).
    pub fn build_stop(&mut self) -> [u8; START_REQUEST_SIZE] {
        self.session.stop_frame()
    }

    /// Build a start frame (64 B).
    pub fn build_start(&mut self) -> [u8; START_REQUEST_SIZE] {
        self.session.start_frame()
    }

    /// Build an LNA-gain frame (1032 B).
    pub fn build_lna(&mut self, gain_db: i8) -> [u8; DATA_PACKET_SIZE] {
        self.session.lna_frame(gain_db, N_RECV)
    }

    /// Build an RX1 NCO frame (1032 B) at `hz`.
    pub fn build_tune(&mut self, hz: u32) -> [u8; DATA_PACKET_SIZE] {
        self.session.tune_frame(RX1_SLOT, hz, N_RECV)
    }

    /// Build a keep-alive frame (1032 B).
    pub fn build_keepalive(&mut self) -> [u8; DATA_PACKET_SIZE] {
        self.session.keepalive_frame(N_RECV)
    }

    /// Attempt to parse `dgram` as an HL2 discovery reply.
    pub fn try_discovery(dgram: &[u8]) -> Option<DiscoveryInfo> {
        Session::try_parse_discovery(dgram)
    }
}

/// Socket handles returned by `SocketSet::add` at setup time.
#[derive(Debug)]
pub struct SocketHandles {
    pub dhcp: SocketHandle,
    pub hl2: SocketHandle,
}

/// Build the smoltcp `Interface` + a two-socket `SocketSet` (DHCP + HL2 UDP).
///
/// Takes static socket storage once; panics if called a second time.
pub fn build_iface_and_sockets(
    dev: &mut (impl smoltcp::phy::Device + ?Sized),
    mac: [u8; 6],
    now: Instant,
) -> (Interface, SocketSet<'static>, SocketHandles) {
    let iface = Interface::new(Config::new(EthernetAddress(mac).into()), dev, now);
    let mut set = SocketSet::new(&mut SOCKET_STORAGE.take()[..]);
    let dhcp = set.add(dhcpv4::Socket::new());
    let hl2 = set.add(make_udp_socket());
    (iface, set, SocketHandles { dhcp, hl2 })
}

/// `Radio` + the smoltcp surfaces the `radio` task needs to drive
/// `Interface` / `SocketSet` / `Device` per poll call. The `radio` task
/// owns all three for the life of the task, passing them in to these
/// helpers (smoltcp 0.13's passing-device pattern).
///
/// The caller (the radio task) keeps `dev`, `iface`, `sockets` as
/// `&mut`-references and threads them through `radio_handle.<op>(…)`.
pub struct RadioHandle<'a, D: smoltcp::phy::Device + ?Sized> {
    pub radio: Radio,
    pub dev: &'a mut D,
    pub iface: &'a mut Interface,
    pub sockets: &'a mut SocketSet<'static>,
    pub handles: &'a SocketHandles,
    now: Instant,
}

impl<'a, D: smoltcp::phy::Device + ?Sized> RadioHandle<'a, D> {
    pub fn new(
        radio: Radio,
        dev: &'a mut D,
        iface: &'a mut Interface,
        sockets: &'a mut SocketSet<'static>,
        handles: &'a SocketHandles,
    ) -> Self {
        Self {
            radio,
            dev,
            iface,
            sockets,
            handles,
            now: Instant::from_millis(0),
        }
    }

    /// Update the clock used by the next poll.
    pub fn set_now(&mut self, now: Instant) {
        self.now = now;
    }

    /// One poll iteration: ingress (bounded) + egress (drained).
    pub fn pump(&mut self) {
        self.iface
            .poll_ingress_single(self.now, self.dev, self.sockets);
        self.iface.poll_egress(self.now, self.dev, self.sockets);
    }

    /// Poll the socket set for DHCP state transitions and return the
    /// current `is_configured` state.
    pub fn poll_dhcp(&mut self) -> bool {
        self.iface.poll(self.now, self.dev, self.sockets);
        let s = self.sockets.get_mut::<dhcpv4::Socket>(self.handles.dhcp);
        match s.poll() {
            Some(dhcpv4::Event::Configured(cfg)) => {
                let ip = cfg.address.address();
                let prefix = cfg.address.prefix_len();
                log::info!("DHCP: {ip}/{prefix}");
                self.iface.update_ip_addrs(|addrs| {
                    addrs.clear();
                    let _ = addrs
                        .push(smoltcp::wire::IpCidr::new(IpAddress::Ipv4(ip), prefix))
                        .map_err(|e| log::error!("iface.update_ip_addrs: push {e}"));
                });
                let _ = self.iface.routes_mut().remove_default_ipv4_route();
                if let Some(rt) = cfg.router {
                    let _ = self.iface.routes_mut().add_default_ipv4_route(rt);
                }
            }
            Some(dhcpv4::Event::Deconfigured) => {
                log::info!("DHCP: waiting for lease");
                self.iface.update_ip_addrs(|addrs| addrs.clear());
                let _ = self.iface.routes_mut().remove_default_ipv4_route();
            }
            None => {}
        }
        self.is_configured()
    }

    /// True if the interface has an IPv4 address.
    pub fn is_configured(&self) -> bool {
        matches!(
            self.iface.ip_addrs().iter().next().map(|c| c.address()),
            Some(IpAddress::Ipv4(_))
        )
    }

    /// The interface's IPv4, if any (first configured address, if it is
    /// IPv4; IPv6 is reported as `None`).
    pub fn our_ip(&self) -> Option<Ipv4Addr> {
        match self.iface.ip_addrs().iter().next()?.address() {
            IpAddress::Ipv4(ip) => Some(ip),
        }
    }

    /// Destination endpoint for C&C frames: the peer HL2's `HL2_PORT`
    /// (1024).
    fn cc_dst(&self) -> IpEndpoint {
        IpEndpoint {
            addr: IpAddress::Ipv4(self.radio.peer),
            port: HL2_PORT,
        }
    }

    /// Send a UDP datagram over the HL2 socket to the current peer at
    /// `HL2_PORT`.
    pub fn send_cc(&mut self, buf: &[u8]) {
        let dst = self.cc_dst();
        let s = self.sockets.get_mut::<udp::Socket>(self.handles.hl2);
        if let Err(e) = s.send_slice(buf, dst) {
            log::warn!("HL2 send_cc: {e}");
        }
        self.pump();
    }

    /// Broadcast a discovery request (63 B datagram) to `HL2_PORT`.
    pub fn send_discovery(&mut self) {
        let s = self.sockets.get_mut::<udp::Socket>(self.handles.hl2);
        let dst = IpEndpoint {
            addr: IpAddress::Ipv4(Ipv4Addr::BROADCAST),
            port: HL2_PORT,
        };
        let req = discovery_request();
        if let Err(e) = s.send_slice(&req, dst) {
            log::warn!("HL2 discovery send: {e}");
        }
        self.pump();
    }

    /// Receive the next datagram into `buf` (non-blocking). Returns
    /// `(len, src_ip)` or `None` if empty or the datagram exceeds `buf`.
    pub fn recv(&mut self, buf: &mut [u8]) -> Option<(usize, Ipv4Addr)> {
        let s = self.sockets.get_mut::<udp::Socket>(self.handles.hl2);
        if !s.can_recv() {
            return None;
        }
        match s.recv_slice(buf) {
            Ok((n, meta)) => match meta.endpoint.addr {
                IpAddress::Ipv4(ip) => Some((n, ip)),
            },
            Err(_) => None,
        }
    }

    /// Drain the HL2 socket until empty.
    pub fn drain(&mut self) {
        let mut buf = [0u8; DATA_PACKET_SIZE + 64];
        while let Some((n, src)) = self.recv(&mut buf) {
            log::trace!("HL2 drain: {n} B from {src}");
        }
    }

    // ── C&C frames (built by `Radio::build_*`, sent over `send_cc`) ──
    pub fn send_stop(&mut self) {
        let buf = self.radio.build_stop();
        self.send_cc(&buf);
    }

    pub fn send_start(&mut self) {
        let buf = self.radio.build_start();
        self.send_cc(&buf);
    }

    pub fn send_lna(&mut self, gain_db: i8) {
        let buf = self.radio.build_lna(gain_db);
        self.send_cc(&buf);
    }

    pub fn send_tune(&mut self, hz: u32) {
        let buf = self.radio.build_tune(hz);
        self.send_cc(&buf);
    }

    /// Send one keep-alive through the normal C&C path.
    pub fn send_keepalive(&mut self) {
        let buf = self.radio.build_keepalive();
        self.send_cc(&buf);
    }

    /// Parse a datagram as a discovery reply.
    pub fn try_discovery(&self, dgram: &[u8]) -> Option<DiscoveryInfo> {
        Radio::try_discovery(dgram)
    }

    /// Update the peer IP.
    pub fn set_peer(&mut self, ip: Ipv4Addr) {
        self.radio.peer = ip;
    }

    pub fn peer(&self) -> Ipv4Addr {
        self.radio.peer
    }
}

/// A bounded `core::fmt::Write` target borrowing caller-owned storage.
pub struct WriteBuf<'a> {
    pub target: &'a mut [u8],
    pub pos: usize,
}

impl core::fmt::Write for WriteBuf<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let end = self.pos.checked_add(s.len()).ok_or(core::fmt::Error)?;
        let dst = self.target.get_mut(self.pos..end).ok_or(core::fmt::Error)?;
        dst.copy_from_slice(s.as_bytes());
        self.pos = end;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_buf_borrows_local_storage_and_rejects_overflow() {
        use core::fmt::Write;
        let mut buf = [0; 8];
        let mut w = WriteBuf {
            target: &mut buf,
            pos: 0,
        };
        write!(w, "F {}", 123).unwrap();
        assert_eq!(&w.target[..w.pos], b"F 123");
        assert!(w.write_str("4567").is_err());
        assert_eq!(w.pos, 5);
        w.write_str("456").unwrap();
        assert_eq!(&w.target[..w.pos], b"F 123456");
        w.write_str("").unwrap();
        assert!(w.write_str("7").is_err());
        w.pos = usize::MAX;
        assert!(w.write_str("7").is_err());
    }

    #[test]
    fn udp_storage_is_separate_and_taken_once() {
        let mut socket = make_udp_socket();
        assert_eq!(socket.endpoint().port, LOCAL_PORT);
        socket
            .send_slice(b"control", (Ipv4Addr::LOCALHOST, HL2_PORT))
            .unwrap();
        assert!(!socket.can_recv());
        assert!(UDP_RX_META.try_take().is_none());
        assert!(UDP_TX_META.try_take().is_none());
        assert!(UDP_RX.try_take().is_none());
        assert!(UDP_TX.try_take().is_none());
    }

    /// `OC_RELAY` (currently relay 7 = 0x40) encodes to the C2 wire byte
    /// expected by `build_keepalive_packet`: bits 7:1 ← `oc_bits & 0x7F`,
    /// so 0x40 → (0x40 & 0x7F) << 1 = 0x80.
    #[test]
    fn oc_relay_wire_value() {
        let mut radio = Radio::new();
        let pkt = radio.build_keepalive();
        // C2 is at chunk[5] = buf[8 + 5] = buf[13].
        assert_eq!(pkt[13], (OC_RELAY & 0x7F) << 1);
    }

    /// `build_lna_gain_frame(+6 dB)` encodes to C4 = 0x40 | (18 & 0x3F).
    #[test]
    fn lna_frame_c4_value() {
        let mut radio = Radio::new();
        let pkt = radio.build_lna(LNA_GAIN_DB);
        let c4_off = 8 + 512 + 7;
        assert_eq!(
            pkt[c4_off],
            0x40 | ((LNA_GAIN_DB as u8).wrapping_add(12) & 0x3F)
        );
    }

    /// NCO frame for RX1 @ 7.074 MHz has C0 = 0x04.
    #[test]
    fn nco_frame_rx1() {
        let mut radio = Radio::new();
        let pkt = radio.build_tune(TUNE_HZ);
        assert_eq!(pkt[8 + 512 + 3], 0x04);
        let f = u32::from_be_bytes([
            pkt[8 + 512 + 4],
            pkt[8 + 512 + 5],
            pkt[8 + 512 + 6],
            pkt[8 + 512 + 7],
        ]);
        assert_eq!(f, TUNE_HZ);
    }
}
