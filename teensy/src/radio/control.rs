//! HL2 radio control: DHCP, discovery, START/STOP, LNA, NCO tune,
//! keep-alive, EP6 datagram RX.
//!
//! Wire bytes come from `hl2::protocol` (AGENTS.md layering rule); this
//! module only sequences them through smoltcp + tracks the send sequence.
//!
//! Session parameters (hardwired, matches the old monolith):
//!
//!   RX1          7.074 MHz   (OC filter bank relay: `OC_RELAY`)
//!   DDC rate     96 kSps     (C1 SPEED = `C1_SPEED_96K`)
//!   n_recv       1
//!   LNA          +6 dB       (`DEFAULT_LNA_GAIN_DB`)
//!   keepalive    40 ms       (watchdog limit ≈ 168 ms)
//!
//! Socket handles: `add()` returns a `SocketHandle` equal to the insertion
//! index, so the DHCP socket added first is `#0` and the HL2 socket added
//! second is `#1`. `make_radio` captures both return values into a
//! `SocketHandles` struct; pass `&handles` to each `RadioHandle` call.
//!
//! Port model: `LOCAL_PORT` is the local UDP source port the Teensy binds
//! AND uses when sending C&C frames to the HL2. The HL2 echoes this
//! source port back as the destination of its EP6 data stream (the same
//! mechanism the `hl2` crate uses with an ephemeral port + `SO_REUSEADDR`
//! on a real socket). smoltcp `udp::Socket::bind` requires a non-zero
//! port; once bound, all packets destined to `LOCAL_PORT` land on this
//! socket regardless of source (UDP is connectionless on this socket).

use core::net::Ipv4Addr;

use smoltcp::iface::{Config, Interface, SocketHandle, SocketSet};
use smoltcp::socket::{dhcpv4, udp};
use smoltcp::time::Instant;
use smoltcp::wire::{EthernetAddress, IpEndpoint, IpListenEndpoint, IpAddress};

use hl2::protocol::discovery::{discovery_request, parse_discovery_response, DiscoveryInfo};
use hl2::protocol::{
    C1_SPEED_96K, DATA_PACKET_SIZE, DEFAULT_LNA_GAIN_DB, DISCOVERY_RESPONSE_SIZE, HL2_PORT,
    OC_MASK_RX, START_REQUEST_SIZE,
};
use hl2::protocol::data::{
    build_keepalive_packet, build_lna_gain_frame, build_nco_packet, build_start_stop_frame,
};

/// OC filter bank relay mask.
/// LSB-first: bit 0 = relay 1 … bit 6 = relay 7 (see `hl2::protocol::OC_MASK_RX`).
/// Current value selects relay 7 (`0b010_0000` = `0x40`).
pub const OC_RELAY: u8 = 0x40 & OC_MASK_RX;

/// RX1 NCO target (Hz) — 7.074 MHz.
pub const TUNE_HZ: u32 = 7_074_000;

/// RX1 slot (1-based; `build_nco_packet` maps slot → C0 = slot << 1).
const RX1_SLOT: u8 = 1;

/// Active receiver count (1 = single-slot RX1).
const N_RECV: u8 = 1;

/// DDC rate (kHz) for the status line.
pub const SAMPLE_RATE_KHZ: u32 = 96;

/// Default LNA gain (dB), re-exported for the status text.
pub const LNA_GAIN_DB: i8 = DEFAULT_LNA_GAIN_DB;

/// Keep-alive cadence in ms (HL2 resets after ≈ 168 ms of silence).
///
/// The `hl2` crate ships 40 ms as the host default; this board runs a
/// single-core cooperative RTIC app where the render task can hold the
/// core for tens of milliseconds per waterfull row, so we tighten the
/// cadence to 25 ms. Even if the render task eats one tick, the worst
/// gap between two actual sends is ~50 ms — well inside the 168 ms
/// watchdog, instead of ~80 ms at the 40 ms cadence.
pub const KEEPALIVE_INTERVAL_MS_CONST: u64 = 25;

/// Local (Teensy) MAC. Change for each board on the same LAN.
pub const MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];

/// Local UDP source port. Must be non-zero (smoltcp `bind` rejects 0).
/// Any ephemeral value works — the HL2 echoes it on the EP6 stream.
pub const LOCAL_PORT: u16 = 40000;

/// UDP datagram slot count per direction (rx and tx are separate arrays).
/// Each slot holds one in-flight datagram; 8 is enough for a 1032 B EP6
/// burst with ~6 ms inter-frame spacing (≈ 3 in flight at peak).
const HL2_SLOTS: usize = 8;
/// Payload length per slot. Must be ≥ `DATA_PACKET_SIZE` (1032) + 8 B
/// for the smoltcp UDP metadata envelope + L2/L3/IP/UDP overhead.
const HL2_SLOT: usize = 1280;

// Per-socket storage (`SocketStorage<'static>`). Held in an
// `UnsafeCell` because we need a `&'static mut [..]` slice to pass to
// `SocketSet::new` while also returning a `SocketSet<'static>` owned by
// the caller. Single-core system; `get()` is called exactly once from
// `build_iface_and_sockets`, which is called exactly once at startup.
// Safety invariant: after the first `get()` call, no second call happens.
struct SocketStorageUnsafe {
    inner: core::cell::UnsafeCell<[smoltcp::iface::SocketStorage<'static>; 2]>,
}
unsafe impl Sync for SocketStorageUnsafe {}
static SOCKET_STORAGE_UNSAFE: SocketStorageUnsafe = SocketStorageUnsafe {
    inner: core::cell::UnsafeCell::new([smoltcp::iface::SocketStorage::EMPTY; 2]),
};

/// `udp::PacketMetadata` ring + payload rings. One per slot. Consumed
/// exactly once by `make_udp_socket` via raw pointer reassembly.
static UDP_META: [udp::PacketMetadata; HL2_SLOTS] = [udp::PacketMetadata::EMPTY; HL2_SLOTS];
static UDP_RX: [u8; HL2_SLOTS * HL2_SLOT] = [0u8; HL2_SLOTS * HL2_SLOT];
static UDP_TX: [u8; HL2_SLOTS * HL2_SLOT] = [0u8; HL2_SLOTS * HL2_SLOT];

/// Build the HL2 `udp::Socket`. `UDP_META` / `UDP_RX` / `UDP_TX` are
/// declared once per process and consumed exactly once here.
pub fn make_udp_socket() -> udp::Socket<'static> {
    // Safety: `UDP_META`/`UDP_RX`/`UDP_TX` are declared once per process,
    // passed in here exactly once, and are `static` (no aliasing). smoltcp
    // does not copy the metadata; it owns them from this point.
    unsafe {
        let rx_meta = core::slice::from_raw_parts_mut(
            UDP_META.as_ptr() as *mut udp::PacketMetadata,
            UDP_META.len(),
        );
        let rx_payload = core::slice::from_raw_parts_mut(
            UDP_RX.as_ptr() as *mut u8,
            UDP_RX.len(),
        );
        let tx_meta = core::slice::from_raw_parts_mut(
            UDP_META.as_ptr() as *mut udp::PacketMetadata,
            UDP_META.len(),
        );
        let tx_payload = core::slice::from_raw_parts_mut(
            UDP_TX.as_ptr() as *mut u8,
            UDP_TX.len(),
        );
        let rx = udp::PacketBuffer::new(rx_meta, rx_payload);
        let tx = udp::PacketBuffer::new(tx_meta, tx_payload);
        let mut s = udp::Socket::new(rx, tx);
        // Bind to `LOCAL_PORT` — this becomes both the local source port
        // for C&C sends (below: `IpEndpoint { addr: peer, port: HL2_PORT }`)
        // and the destination we receive EP6 data frames on (the HL2
        // echoes our source port as the data-stream destination).
        if let Err(e) = s.bind(IpListenEndpoint::from(LOCAL_PORT)) {
            log::error!("HL2 socket bind: {e}");
        }
        s
    }
}

/// The radio-side state the `radio` task owns.
pub struct Radio {
    pub peer: Ipv4Addr,
    /// Monotonically-increasing send sequence. The HL2 does not
    /// strictly enforce monotonicity; wrap-around is allowed.
    seq: u32,
}

impl Radio {
    pub fn new() -> Self {
        Self {
            peer: Ipv4Addr::UNSPECIFIED,
            seq: 0,
        }
    }

    pub fn set_peer(&mut self, ip: Ipv4Addr) {
        self.peer = ip;
    }

    pub fn peer(&self) -> Ipv4Addr {
        self.peer
    }

    fn next_seq(&mut self) -> u32 {
        let s = self.seq;
        self.seq = s.wrapping_add(1);
        s
    }

    /// Build a stop frame (64 B).
    pub fn build_stop(&mut self) -> [u8; START_REQUEST_SIZE] {
        build_start_stop_frame(self.next_seq(), false)
    }

    /// Build a start frame (64 B).
    pub fn build_start(&mut self) -> [u8; START_REQUEST_SIZE] {
        build_start_stop_frame(self.next_seq(), true)
    }

    /// Build an LNA-gain frame (1032 B).
    pub fn build_lna(&mut self, gain_db: i8) -> [u8; DATA_PACKET_SIZE] {
        build_lna_gain_frame(self.next_seq(), gain_db, C1_SPEED_96K, OC_RELAY, N_RECV)
    }

    /// Build an RX1 NCO frame (1032 B) at `hz`.
    pub fn build_tune(&mut self, hz: u32) -> [u8; DATA_PACKET_SIZE] {
        build_nco_packet(self.next_seq(), RX1_SLOT, hz, C1_SPEED_96K, OC_RELAY, N_RECV)
    }

    /// Build a keep-alive frame (1032 B).
    pub fn build_keepalive(&mut self) -> [u8; DATA_PACKET_SIZE] {
        build_keepalive_packet(self.next_seq(), C1_SPEED_96K, OC_RELAY, N_RECV)
    }

    /// Attempt to parse `dgram` as an HL2 discovery reply.
    pub fn try_discovery(dgram: &[u8]) -> Option<DiscoveryInfo> {
        let slice = dgram.get(..DISCOVERY_RESPONSE_SIZE)?.try_into().ok()?;
        parse_discovery_response(&slice)
    }
}

/// Socket handles captured at setup time. `add()` returns handles matching
/// the insertion index, so a fixed `SocketHandles { dhcp: 0, hl2: 1 }`
/// works for a `SocketSet` whose first two slots are DHCP + HL2.
#[derive(Debug)]
pub struct SocketHandles {
    pub dhcp: SocketHandle,
    pub hl2: SocketHandle,
}

/// Build the smoltcp `Interface` + a two-socket `SocketSet` (DHCP + HL2 UDP).
///
/// The `socket_storage` array must outlive the returned `SocketSet` — it
/// holds per-socket metadata. It is declared `static` in this module
/// (`SOCKET_STORAGE`), not a local, so the returned `SocketSet<'static>`
/// can borrow it for the life of the process.
pub fn build_iface_and_sockets(
    dev: &mut (impl smoltcp::phy::Device + ?Sized),
    mac: [u8; 6],
    now: Instant,
) -> (Interface, SocketSet<'static>, SocketHandles) {
    let iface = Interface::new(Config::new(EthernetAddress(mac).into()), dev, now);
    // SAFETY: `SOCKET_STORAGE_UNSAFE.inner` is `static`, written to exactly
    // once (this `get()` call) and read thereafter only by the `SocketSet`
    // that borrows from it. Single-core, cooperative scheduling.
    let storage: &mut [smoltcp::iface::SocketStorage<'static>] =
        unsafe { &mut *SOCKET_STORAGE_UNSAFE.inner.get() };
    let mut set = SocketSet::new(storage);
    let dhcp = set.add(dhcpv4::Socket::new());
    let hl2 = set.add(make_udp_socket());
    (
        iface,
        set,
        SocketHandles {
            dhcp,
            hl2,
        },
    )
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

    /// Broadcast a discovery request (60 B datagram) to `HL2_PORT`.
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
    /// `(len, src_ip)` or `None` if the socket is empty.
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

    /// Send one keep-alive. Logs the result so a USB capture shows the
    /// keep-alive cadence (HL2 watchdog is ≈168 ms; our cadence is 40 ms).
    pub fn send_keepalive(&mut self) {
        let buf = self.radio.build_keepalive();
        let dst = self.cc_dst();
        let s = self.sockets.get_mut::<udp::Socket>(self.handles.hl2);
        match s.send_slice(&buf, dst) {
            Ok(()) => {
                log::info!("keepalive → {dst}");
                self.pump();
            }
            Err(e) => {
                log::warn!("keepalive → {dst} FAILED: {e:?}");
            }
        }
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

    /// Keep-alive cadence (ms).
    pub const fn keepalive_interval_ms() -> u64 {
        KEEPALIVE_INTERVAL_MS_CONST
    }
}

/// A minimal `core::fmt::Write` target over a byte slice, used by
/// `main.rs::fmt_ip` to format the peer IP without `alloc`.
pub struct WriteBuf {
    pub target: &'static mut [u8],
    pub pos: usize,
}

impl core::fmt::Write for WriteBuf {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let end = (self.pos + s.len()).min(self.target.len());
        self.target[self.pos..end].copy_from_slice(&s.as_bytes()[..end - self.pos]);
        self.pos = end;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// NCO frame for RX1 @ 7.074 MHz has C0 = 0x04 (slot 1 → << 1).
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
