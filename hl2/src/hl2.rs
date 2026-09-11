//! High-level Hermes-Lite 2 client.
//!
//! Two roles are split across two types:
//!
//! * **`Hl2`** — an *owned, clonable* control handle for one device. It wraps
//!   the single shared UDP socket via `Arc`, so many websocket clients can
//!   share one physical radio; control writes are serialized by an internal
//!   mutex.
//! * **`Hl2Pump`** — spawns the receive/keep-alive task that owns the socket
//!   read side and emits completed `IQBlock`s + C&C ACKs over an `mpsc`.
//!
//! One socket, bound once at start time, is shared by both.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::net::UdpSocket;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::time;

use crate::protocol::data::{
    BasebandChunk, BlockAssembler, IQBlock, Item, SampleFormat, parse_receive_packet_into,
};
use crate::protocol::discovery::{DiscoveryInfo, discovery_request, parse_discovery_response};
use crate::protocol::session::Session;
use crate::protocol::{
    BOARD_ID_HL2, DATA_PACKET_SIZE, DEFAULT_LNA_GAIN_DB, DISCOVERY_RESPONSE_SIZE, HL2_PORT,
    KEEPALIVE_INTERVAL_MS, speed_bits_for_khz,
};
use crate::receiver::fanout::BasebandFanout;

/// NCo slot address constants for `tune`. Slot N → `RXn_ADDR`. Slot values
/// are 1-based (RX1 = 1, RX2 = 2, …, RX7 = 7). The HL2 hardware has at most
/// 7 receiver DDCs, but the gateware interleaves EP6 baseband per *active*
/// slot (those that have been NCO-tuned), so the client tracks the active
/// set explicitly via [`Hl2::tune`] and the [`BasebandFanout`].
pub const TX1_ADDR: u8 = 0x00;
pub const RX1_ADDR: u8 = 0x01;
pub const RX2_ADDR: u8 = 0x02;
pub const RX3_ADDR: u8 = 0x03;
pub const RX4_ADDR: u8 = 0x04;
pub const RX5_ADDR: u8 = 0x05;
pub const RX6_ADDR: u8 = 0x06;
pub const RX7_ADDR: u8 = 0x07;

/// Information returned from a successful start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StartInfo {
    /// Sample bit depth detected from the HL2.
    pub sample_format: SampleFormat,
    /// Number of hardware receivers.
    pub rx_count: usize,
    /// The local UDP port the HL2 streams wideband data to.
    pub local_port: u16,
}

/// An event the pump emits for every interesting thing on the receive stream.
#[derive(Debug, Clone)]
pub enum Hl2Event {
    /// A complete wideband block from the EP4 stream (`IQ_PAIRS_PER_BLOCK`).
    Block(IQBlock),
    /// A C&C (EP6) response from the hardware.
    CmdAck {
        ack: bool,
        raddr: u8,
        ptt: bool,
        data: u32,
    },
    /// A complex baseband I/Q frame from the EP6 stream, de-interleaved into
    /// one stream **per active receiver**: `per_rx[p]` is the p-th active slot
    /// in ascending order (e.g. slots {1,3} active → `per_rx[0]`=RX1,
    /// `per_rx[1]`=RX3). 24-bit I/Q, normalised to `f32` in [-1, +1].
    /// Each stream's rate = C1 SPEED / 2 complex Sps, independent of the
    /// receiver count. For the single-receiver case this is one element.
    ///
    /// Per-receiver fan-out to the shared rings is done by the pump — see
    /// [`Hl2::baseband_ring`].
    Baseband(BasebandChunk),
}

/// Shared handle state for one physical HL2 device.
#[derive(Debug)]
struct Shared {
    socket: Arc<UdpSocket>,
    /// The peer (device) address we send control frames to.
    hl2_peer: IpAddr,
    local_port: u16,
    started: Mutex<bool>,
    rx_count: Mutex<u8>,
    /// Serializes control writes (start/stop/tune) across clients.
    write_lock: Mutex<()>,
    /// The shared HL2 [`Session`]: the monotonic send-sequence counter **plus**
    /// the persistent C&C config — the C1 SPEED bits (per-receiver DDC rate
    /// 48/96/192/384 kHz, re-asserted by every keep-alive so the board holds
    /// that rate between tune writes) and the RX open-collector filter relay
    /// mask (LSB-first, bit 0 = relay 1 … bit 6 = relay 7; re-asserted in
    /// every keep-alive's C2 so the MRF101 companion filter board holds its
    /// band between NCO tunes — see PROTOCOL.md §11.4 and
    /// `build_keepalive_packet`).
    ///
    /// Held behind a [`tokio::sync::Mutex`] because it is written by any
    /// control client and read/built by the pump's keep-alive tick; the
    /// critical section is a few-byte copy + a frame build, never an `.await`
    /// point inside it. `n_recv` is deliberately **not** stored here — it is
    /// derived live from the [`BasebandFanout`] per frame (it changes as slots
    /// are tuned on/off) and passed to the frame builders as a parameter.
    session: Mutex<Session>,
    /// The RX LNA gain currently programmed on the board (dB), so a `State`
    /// snapshot can report it without a register read.
    lna_gain_db: Mutex<i8>,
    /// The per-slot EP6 baseband fan-out: one [`BasebandRing`] per active
    /// receiver, tracked from the host's own tune history. This replaces the
    /// old single `ring` field (RX1-only). The pump is the sole writer; any
    /// number of `VirtualReceiver`s can read the ring of the slot they were
    /// configured for. Decouples the socket RX loop from the (potentially
    /// slow) demod so the pump never blocks on DSP.
    ///
    /// `BasebandFanout` wraps a `std::sync::Mutex` so it is `Arc`-shareable
    /// across the pump and every demod reader; the critical section is
    /// memcpy/insert sized and never awaits. See
    /// [`crate::receiver::fanout::BasebandFanout`] for the position↔slot
    /// mapping and PROTOCOL.md §16.
    fanout: Arc<BasebandFanout>,
    /// Cumulative complex samples pushed by the pump (all slots, per chunk),
    /// for the `[pump] pair_rate` diagnostic. Reset by the log line.
    delivered: AtomicUsize,
}

impl Shared {
    async fn send_packet(&self, bytes: &[u8]) -> std::io::Result<usize> {
        let dst = SocketAddr::new(self.hl2_peer, HL2_PORT);
        self.socket.send_to(bytes, dst).await
    }
}

/// A clonable control handle for one HL2 device.
#[derive(Debug, Clone)]
pub struct Hl2 {
    inner: Arc<Shared>,
}

impl Hl2 {
    /// Create a new controller on an *already-bound* shared socket and send
    /// the Metis start+wideband command. Returns controller, event receiver,
    /// and start info. The receive task is spawned and runs to completion.
    pub async fn start(
        hl2_addr: IpAddr,
    ) -> Result<
        (Self, mpsc::UnboundedReceiver<Hl2Event>, StartInfo),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        // Historical default: 192 kSps per receiver.
        Self::start_with_speed(hl2_addr, 192_000).await
    }

    /// Create a controller for `hl2_addr` and run a receive pump that emits
    /// [`Hl2Event`]s over an mpsc. `c1_speed_hz` is the *per-receiver* DDC
    /// rate (kHz) requested from the hardware — one of 48/96/192/384 (the
    /// standard openHPSDR DDC options). Use this to pair the C1 bit with the
    /// matching `source_rate_hz` you pass to a downstream
    /// [`crate::receiver::VirtualReceiver`]; the radio delivers the option
    /// rate per receiver, so the downstream decimator must see it too.
    pub async fn start_with_speed(
        hl2_addr: IpAddr,
        c1_speed_hz: u32,
    ) -> Result<
        (Self, mpsc::UnboundedReceiver<Hl2Event>, StartInfo),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let c1_speed = speed_bits_for_khz(c1_speed_hz / 1000);
        let socket = UdpSocket::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0))
            .await
            .map_err(|e| e.to_string())?;
        let local_port = socket.local_addr().map_err(|e| e.to_string())?.port();
        let socket = Arc::new(socket);

        // Fetch the device's real receiver count via a directed discovery
        // before we start streaming. While in "waiting" the HL2 answers this
        // with its capability (rx_count in `DiscoveryInfo`). If the radio
        // doesn't respond (e.g. we were passed a stale IP and nothing is
        // listening, or it's already locked into sending for a different
        // host) we fall back to the hardware default of 4 — the HL2 always
        // has 4 NCO slots available for use, so the tune API keeps working,
        // and the fan-out map grows as we tune.
        let discovered = discover_single(hl2_addr).await;
        let rx_count: u8 = discovered
            .as_ref()
            .map(|(_, info)| info.rx_count)
            .unwrap_or(4);
        if std::env::var("HL2_DEBUG").is_ok() {
            eprintln!(
                "[DBG] rx_count={rx_count} (from discovery: {})",
                discovered.is_some()
            );
        }

        let session = Session::new(c1_speed, 0u8);
        let shared = Arc::new(Shared {
            socket: socket.clone(),
            hl2_peer: hl2_addr,
            local_port,
            started: Mutex::new(false),
            rx_count: Mutex::new(0),
            write_lock: Mutex::new(()),
            session: Mutex::new(session),
            lna_gain_db: Mutex::new(DEFAULT_LNA_GAIN_DB),
            fanout: Arc::new(BasebandFanout::new()),
            delivered: AtomicUsize::new(0),
        });

        let self_ = Self {
            inner: shared.clone(),
        };

        // Phase 1 — pre-START: force the device into a clean "waiting" state
        // and drain any in-flight wideband/C&C frames still being sent from a
        // previous session. Without this, the first frames we see after START
        // may be old-session leftovers (typically all-zero wideband buffers)
        // and the handshake falsely fails.
        if std::env::var("HL2_DEBUG").is_ok() {
            eprintln!("[DBG] sending STOP + drain to reset device state");
        }
        let stop = {
            let mut s = self_.inner.session.lock().await;
            s.stop_frame()
        };
        let _ = self_.inner.send_packet(&stop).await;
        let drain_deadline = time::Instant::now() + time::Duration::from_secs(3);
        let mut last_frame_at = time::Instant::now() - time::Duration::from_millis(300);
        let mut drained = 0u32;
        while time::Instant::now() < drain_deadline {
            let mut buf = [0u8; 2048];
            match time::timeout(time::Duration::from_millis(300), {
                let s = self_.inner.socket.clone();
                async move { s.recv_from(&mut buf).await }
            })
            .await
            {
                Ok(Ok((len, _src))) => {
                    drained += 1;
                    if std::env::var("HL2_DEBUG").is_ok() {
                        eprintln!(
                            "[DBG] drain[{}] len={} zeros={} first8={:02x?}",
                            drained,
                            len,
                            buf[..len].iter().all(|b| *b == 0),
                            &buf[..8.min(len)],
                        );
                    }
                    last_frame_at = time::Instant::now();
                }
                Ok(Err(e)) => return Err(e.to_string().into()),
                Err(_) => {
                    // 300 ms of silence → device is idle. Stop draining.
                    if last_frame_at.elapsed() >= time::Duration::from_millis(300) {
                        break;
                    }
                }
            }
        }
        if std::env::var("HL2_DEBUG").is_ok() {
            eprintln!("[DBG] drain complete, {} frames discarded", drained);
        }

        // Phase 2 — send the START command. The HL2 answers with 1032-byte
        // frames immediately. Depending on device state (fresh boot vs.
        // recently-reset via watchdog), the first frame(s) may be all-zero
        // wideband buffers or valid C&C status frames. In every case a
        // 1032-byte datagram from `hl2_peer:1024` is the ACK — the device is
        // streaming. We accept it after seeing at least one non-zero frame or
        // after collecting 5+ frames (device can take a few frames to produce
        // non-zero output after a watchdog reset). The pump task will handle
        // the rest of the stream.
        let pkt = {
            let mut s = self_.inner.session.lock().await;
            s.start_frame()
        };
        if std::env::var("HL2_DEBUG").is_ok() {
            let dst = SocketAddr::new(self_.inner.hl2_peer, HL2_PORT);
            eprintln!(
                "[DBG] sending START frame len={} to={} pkt={:02x?}",
                pkt.len(),
                dst,
                &pkt[..8],
            );
        }
        self_
            .inner
            .send_packet(&pkt)
            .await
            .map_err(|e| e.to_string())?;

        // Await the first acknowledgement datagram from `hl2_peer`. After
        // START the device emits either a C&C status frame (non-zero `EF FE`
        // header) or wideband frames (1032 B, possibly all-zero while the ADC
        // is settling). Either one is the ACK that the device is in "sending"
        // mode.
        let deadline = time::Instant::now() + time::Duration::from_secs(3);
        let mut got_ack = false;
        let mut frame_count: u32 = 0;
        while !got_ack && time::Instant::now() < deadline {
            let remain = deadline.saturating_duration_since(time::Instant::now());
            let mut buf = [0u8; DATA_PACKET_SIZE + 64];
            let (len, _src) = match time::timeout(remain, {
                let s = self_.inner.socket.clone();
                async move { s.recv_from(&mut buf).await }
            })
            .await
            {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => return Err(e.to_string().into()),
                Err(_) => break,
            };
            frame_count += 1;
            let all_zero = len > 0 && buf[..len].iter().all(|b| *b == 0);
            if std::env::var("HL2_DEBUG").is_ok() {
                eprintln!(
                    "[DBG] post-start frame[{}] len={} zeros={} first8={:02x?}",
                    frame_count,
                    len,
                    all_zero,
                    &buf[..8.min(len)],
                );
            }
            // Any datagram from the device = it responded to START.
            got_ack = true;
            // Drain a few more frames so the socket buffer isn't full by the
            // time the pump task starts (avoids a burst of stale data).
            let drain_n = 4u32.min(frame_count / 2 + 1);
            for _ in 0..drain_n {
                let mut b2 = [0u8; 2048];
                let _ = time::timeout(time::Duration::from_millis(50), {
                    let s = self_.inner.socket.clone();
                    async move { s.recv_from(&mut b2).await }
                })
                .await;
            }
        }
        if !got_ack {
            return Err("timeout: no reply after START".into());
        }

        // rx_count / sample_16bit aren't recoverable from the C&C status burst
        // in this path; use the HL2 defaults (4 RX slots, 16-bit). Refine
        // later via a read of the sample-depth register if needed.
        // Fold what we learned from discovery into the stub `DiscoveryInfo`
        // (the fields the C&C burst does not let us recover). When we did not
        // get a discovery reply, keep the HL2 defaults (4 RX slots, 16-bit).
        let (info_rx_count, info_sample_16bit) = match discovered.as_ref() {
            Some((_, info)) => (info.rx_count, info.sample_16bit),
            None => (4, true),
        };
        let info = DiscoveryInfo {
            mac: [0u8; 6],
            gateware_major: 0,
            gateware_minor: 0,
            board_id: BOARD_ID_HL2,
            rx_count: info_rx_count,
            is_sending: true,
            sample_16bit: info_sample_16bit,
            ip: [0, 0, 0, 0],
        };

        *self_.inner.started.lock().await = true;
        *self_.inner.rx_count.lock().await = info.rx_count;

        // Program the RX LNA so the board amplifies weak signals from the
        // moment we start listening (e.g. the `ssb` ALSA path). The default is
        // `DEFAULT_LNA_GAIN_DB`; clients can change it later via `set_lna_gain`.
        let lna = {
            let g = self_.inner.lna_gain_db.lock().await;
            *g
        };
        let lna_pkt = {
            let mut s = self_.inner.session.lock().await;
            s.lna_frame(lna, 1)
        };
        self_
            .inner
            .send_packet(&lna_pkt)
            .await
            .map_err(|e| e.to_string())?;
        if std::env::var("HL2_DEBUG").is_ok() {
            eprintln!("[DBG] LNA gain set to {lna:+} dB at start");
        }

        let start_info = StartInfo {
            sample_format: if info.sample_16bit {
                SampleFormat::Sample16
            } else {
                SampleFormat::Sample12
            },
            rx_count: info.rx_count as usize,
            local_port,
        };

        // Spawn the receive + keep-alive loop.
        let (tx, rx) = mpsc::unbounded_channel::<Hl2Event>();
        let pump = self_.clone();
        tokio::spawn(async move { run_loop(pump, tx).await });

        Ok((self_, rx, start_info))
    }

    /// Send the Metis stop command.
    pub async fn stop(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let pkt = {
            let mut s = self.inner.session.lock().await;
            s.stop_frame()
        };
        self.inner
            .send_packet(&pkt)
            .await
            .map_err(|e| e.to_string())?;
        *self.inner.started.lock().await = false;
        Ok(())
    }

    /// Tune the NCO for a receiver slot.
    ///
    /// * **A new slot** is registered active: the fan-out gains a ring for it
    ///   and the next EP6 record starts carrying its I/Q pair. This is the
    ///   "add receiver" step (exactly what the reference tracks as the
    ///   active-receiver list).
    /// * **A repeated tune of an already-active slot** re-tunes the NCO in
    ///   place; the interleaved count is unchanged and no new ring is created.
    /// * **Passing `freq_hz == 0`** unregisters the slot (idempotent). This is
    ///   the "remove receiver" step and shrinks `N` so the gateware stops
    ///   interleaving that position.
    ///
    /// The active-slot set is the source of truth for the de-interleaver,
    /// the C4[6:3] receiver-count field, and which rings exist. The tune frame
    /// already embeds the prospective count in C4 for this commit; the
    /// keep-alive keeps reasserting it on every tick (so the count and the
    /// rate stay stable between tunes).
    pub async fn tune(
        &self,
        slot: u8,
        freq_hz: u32,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _guard = self.inner.write_lock.lock().await;
        let closing = freq_hz == 0;
        if closing {
            self.inner.fanout.unregister_slot(slot);
            // Keep the "at least one active receiver" invariant: the gateware
            // and the C4 receiver-count field (N-1) assume N >= 1, and the
            // slot-1 NCO is the system's anchor receiver. If closing leaves
            // the set empty, re-anchor on slot 1 so N stays >= 1.
            if self.inner.fanout.rx_count() == 0 {
                self.inner.fanout.register_slot(RX1_ADDR);
            }
        } else {
            self.inner.fanout.register_slot(slot);
        }
        if std::env::var("HL2_DEBUG").is_ok() && closing {
            eprintln!(
                "[DBG] closing RX{slot}: active={:?} (note: gateware NCO for this slot is not reset — \
                 re-tune to 0 or restart to fully retire it on the radio side)",
                self.inner.fanout.slots()
            );
        }
        // Compute the *prospective* receiver set (after this tune commits) so
        // the baseline chunk's C4 receiver-count field matches what the
        // gateware will see once this tune lands. Mirrors the reference
        // (`nreceivers = radio->receivers`, which already includes the
        // receiver being tuned).
        let n_recv = self.inner.fanout.rx_count() as u8;
        let pkt = {
            let mut s = self.inner.session.lock().await;
            s.tune_frame(slot, freq_hz, n_recv)
        };
        if std::env::var("HL2_DEBUG").is_ok() {
            // C0 now lives in the *second* chunk (the register-write chunk).
            let c0 = pkt[8 + crate::protocol::CHUNK_SIZE + 3];
            eprintln!(
                "[DBG] tune RX{slot} (close={closing}): freq={freq_hz} C0=0x{c0:02x} n_recv={n_recv} active={:?} (CONFIG_BOTH baseline)",
                self.inner.fanout.slots()
            );
        }
        self.inner
            .send_packet(&pkt)
            .await
            .map_err(|e| e.to_string())?;
        // Give the hardware a few ms to emit its ACK before another client's
        // write could starve the classic EP6 response.
        time::sleep(time::Duration::from_millis(5)).await;
        Ok(())
    }

    /// The currently active receiver slots, in ascending slot order. This is
    /// the set that will be interleaved in the EP6 baseband stream (one I/Q
    /// pair per slot, in this order, in every record). Tuning a slot adds it;
    /// `tune(slot, 0)` removes it. The pump de-interleaves against this exact
    /// set per frame.
    pub fn active_slots(&self) -> Vec<u8> {
        self.inner.fanout.slots()
    }

    /// Whether `slot` is currently *active* — i.e. registered in the fan-out
    /// and owning a dedicated [`BasebandRing`]. Slot 1 is seeded active at
    /// construction; a later `tune(slot, f≠0)` activates other slots;
    /// `tune(slot, 0)` deactivates.
    ///
    /// Callers that consume a slot's baseband stream (`VirtualReceiver`
    /// demods, spectrum taps, mode decoders) typically bind to the slot's
    /// ring at spawn time; `baseband_ring(slot)` resolves to that dedicated
    /// ring **only if the slot is active** (otherwise it falls back to the
    /// position-0 ring, which is still the RX1 anchor). Comparing ring
    /// handles to detect this is fragile — apps should ask the library
    /// directly:
    ///
    /// ```text
    /// let ring = hl2.baseband_ring(slot);
    /// let active = hl2.is_slot_active(slot);
    /// // ... if a running consumer must be re-bound when `active`
    /// // transitions from `false` to `true`, the app knows to rebuild it.
    /// ```
    pub fn is_slot_active(&self, slot: u8) -> bool {
        self.inner.fanout.ring_for_slot(slot).is_some()
    }

    /// The number of active receiver slots (the N that goes in C4[6:3] of the
    /// baseline chunk and that the de-interleaver partitions the EP6 payload
    /// into). Always ≥ 1 because slot 1 is seeded at construction and a
    /// `tune(slot, 0)` that empties the set re-seeds slot 1 on the next tune.
    pub fn rx_count_active(&self) -> usize {
        self.inner.fanout.rx_count()
    }

    /// Set the RX low-noise-amplifier gain.
    ///
    /// `gain_db` is the AD9866 LNA gain in dB (register range −12…+48 dB; the
    /// 6-bit "Set" field passed straight to the part). Clamped to that range.
    /// This is what the `hl2 ssb` ALSA path relies on to amplify a weak
    /// signal; the board otherwise sits at its power-on default.
    pub async fn set_lna_gain(
        &self,
        gain_db: i8,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let clamped = gain_db.clamp(-12, 48);
        let _guard = self.inner.write_lock.lock().await;
        let n_recv = self.inner.fanout.rx_count() as u8;
        let pkt = {
            let mut s = self.inner.session.lock().await;
            s.lna_frame(clamped, n_recv)
        };
        self.inner
            .send_packet(&pkt)
            .await
            .map_err(|e| e.to_string())?;
        *self.inner.lna_gain_db.lock().await = clamped;
        // Give the hardware a few ms to emit its ACK before another client's
        // write could starve the classic EP6 response.
        time::sleep(time::Duration::from_millis(5)).await;
        Ok(())
    }

    /// The RX LNA gain (dB) currently programmed on the board.
    pub async fn lna_gain(&self) -> i8 {
        *self.inner.lna_gain_db.lock().await
    }

    /// Set the RX open-collector filter relay mask.
    ///
    /// `oc_bits` is an LSB-first 7-bit mask: bit **0** = relay/checkbox **1**,
    /// bit **6** = relay/checkbox **7**. This is re-asserted in the C2 byte of
    /// *every* subsequent keep-alive (wire position C2[7:1]) so the HL2-driven
    /// companion filter board (e.g. MRF101) holds the selected band. See
    /// PROTOCOL.md §11.4 (the reference re-asserts the mask the same way).
    ///
    /// `0x00` = all relays off (board factory default). Bit 7 and above are
    /// ignored. This takes effect on the *next* keep-alive tick (≤ 40 ms), so
    /// there is no blocking round-trip and no register ACK to await.
    pub async fn set_oc_bits(&self, oc_bits: u8) {
        let mut s = self.inner.session.lock().await;
        s.set_oc_bits(oc_bits & crate::protocol::OC_MASK_RX);
    }

    /// The RX open-collector filter relay mask currently held, LSB-first
    /// (bit 0 = relay/checkbox 1 … bit 6 = relay/checkbox 7).
    pub async fn oc_bits(&self) -> u8 {
        self.inner.session.lock().await.oc_bits() & crate::protocol::OC_MASK_RX
    }

    /// The local port the hardware streams wideband data to.
    pub fn local_port(&self) -> u16 {
        self.inner.local_port
    }

    /// The target HL2 address.
    pub fn hl2_addr(&self) -> IpAddr {
        self.inner.hl2_peer
    }

    pub async fn is_started(&self) -> bool {
        *self.inner.started.lock().await
    }

    pub async fn rx_count(&self) -> u8 {
        *self.inner.rx_count.lock().await
    }

    /// A clone of the [`Arc<Mutex<BasebandRing>>`] for the shared EP6
    /// baseband fan-out ring for **one specific receiver slot**.
    ///
    /// The pump is the sole writer for every slot's ring; any number of
    /// `VirtualReceiver`s (one per audio sink) can clone the handle for the
    /// slot they demodulate and peek it from their own thread/task. Hold the
    /// lock only long enough to `push`/`peek` (a memcpy); never `await` under
    /// it.
    ///
    /// Resolution: if `slot` is active (tuned) we return its dedicated ring.
    /// Otherwise we fall back to the position-0 (slot 1) ring so the call
    /// never returns a `None` and the legacy RX1-only call sites keep working
    /// unchanged (`baseband_ring()` == `baseband_ring(1)`).
    ///
    /// For the SSB CLI / `hl2-api` audio sinks:
    /// `let ring = hl2.baseband_ring(2);` for RX2.
    pub fn baseband_ring(
        &self,
        slot: u8,
    ) -> Arc<std::sync::Mutex<crate::receiver::baseband_ring::BasebandRing>> {
        self.inner
            .fanout
            .ring_for_slot(slot)
            .or_else(|| self.inner.fanout.rings_in_order().into_iter().next())
            .expect("fan-out always has ≥1 ring (slot 1 is seeded at start)")
    }

    /// Backward-compatible no-arg form of [`Self::baseband_ring`]: the RX1
    /// slot's ring (position 0). Existing call sites that were written for the
    /// single-receiver design keep compiling unchanged.
    pub fn baseband_ring_slot1(
        &self,
    ) -> Arc<std::sync::Mutex<crate::receiver::baseband_ring::BasebandRing>> {
        self.baseband_ring(RX1_ADDR)
    }

    /// Atomic handle to the pump's cumulative RX1 delivered counter (for
    /// external `[pump] pair_rate`-style diagnostics). The pump resets it to
    /// `0` each second when `HL2_DEBUG` is set; when not, it monotonically
    /// increases since `start_with_speed`.
    pub fn baseband_delivered(&self) -> &AtomicUsize {
        &self.inner.delivered
    }
}

/// Discover HL2 devices on the local network.
pub async fn discover() -> Vec<(SocketAddr, DiscoveryInfo)> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").unwrap();
    socket.set_broadcast(true).unwrap();

    let request = discovery_request();
    let _ = socket.send_to(
        &request,
        SocketAddr::new(IpAddr::V4(Ipv4Addr::BROADCAST), HL2_PORT),
    );

    for addr in ["169.254.19.221", "192.168.1.67"] {
        if let Ok(ip) = addr.parse::<IpAddr>() {
            let _ = socket.send_to(&request, SocketAddr::new(ip, HL2_PORT));
        }
    }

    let local_addr = socket.local_addr().unwrap();
    drop(socket);

    let receive_socket = UdpSocket::bind(local_addr).await.unwrap();
    let mut results = Vec::new();
    let mut buf = [0u8; DISCOVERY_RESPONSE_SIZE];

    time::timeout(time::Duration::from_secs(3), async {
        loop {
            let (len, src) = match time::timeout(time::Duration::from_millis(500), {
                async { receive_socket.recv_from(&mut buf).await }
            })
            .await
            {
                Ok(Ok((len, src))) => (len, src),
                Ok(Err(_)) => continue,
                Err(_) => break,
            };
            if len >= DISCOVERY_RESPONSE_SIZE {
                if let Some(info) = parse_discovery_response(&buf[..].try_into().unwrap()) {
                    if !results.iter().any(|(s, _)| *s == src) {
                        results.push((src, info));
                    }
                }
            }
        }
    })
    .await
    .ok();

    results
}

/// Discover a single HL2 by directed packet.
pub async fn discover_single(hl2_addr: IpAddr) -> Option<(SocketAddr, DiscoveryInfo)> {
    let socket = UdpSocket::bind("0.0.0.0:0").await.unwrap();
    let request = discovery_request();
    let _ = socket
        .send_to(&request, SocketAddr::new(hl2_addr, HL2_PORT))
        .await
        .ok();

    let mut buf = [0u8; DISCOVERY_RESPONSE_SIZE];
    let (len, src) = time::timeout(time::Duration::from_secs(3), {
        async { socket.recv_from(&mut buf).await.unwrap() }
    })
    .await
    .ok()?;

    if len >= DISCOVERY_RESPONSE_SIZE {
        if let Some(info) = parse_discovery_response(&buf[..].try_into().unwrap()) {
            return Some((src, info));
        }
    }
    None
}

/// The receive + keep-alive loop. Runs until the event channel closes (the
/// consumer dropped the receiver, e.g. `hl2-api` aborted the spectrum task on
/// Stop), or the socket errors.
///
/// For every decoded EP6 baseband chunk the pump de-interleaves the payload
/// against the **current active-slot set** (from the [`BasebandFanout`]) and
/// pushes each receiver's stream into that receiver's dedicated
/// [`crate::receiver::baseband_ring::BasebandRing`]. The cumulative
/// `delivered` counter tracks all slots combined. A 1 Hz diagnostic
/// (`HL2_DEBUG`) prints the total pair rate.
async fn run_loop(hl2: Hl2, tx: mpsc::UnboundedSender<Hl2Event>) {
    let socket = hl2.inner.socket.clone();
    let mut assembler = BlockAssembler::new();
    let mut keepalive = time::interval(time::Duration::from_millis(KEEPALIVE_INTERVAL_MS));
    keepalive.set_missed_tick_behavior(time::MissedTickBehavior::Delay);

    let mut buf = [0u8; DATA_PACKET_SIZE];
    // Reused across frames so the steady-state parse loop never reallocates.
    let mut iq: Vec<i16> = Vec::new();
    let mut baseband: Vec<BasebandChunk> = Vec::new();
    let dbg = std::env::var("HL2_DEBUG").is_ok();
    let mut rate_start = time::Instant::now();
    let delivered = &hl2.inner.delivered;
    let fanout = &hl2.inner.fanout;

    loop {
        tokio::select! {
            biased;

            // Exit as soon as the event receiver is dropped — the `tx.send(..)`
            // `is_err()` checks below cover the *next* branch that fires, but a
            // `select!` without this arm never woken by a closed channel keeps
            // the loop (and its keep-alive ticks) alive for the life of the
            // runtime. Without this, the pump outlives the spectrum pipeline
            // that Stop/Start aborts, and a stale keep-alive with the old
            // session's `oc_bits` races the new pump's keep-alives (filter
            // relays chattering).
            _ = tx.closed() => break,

            _ = keepalive.tick() => {
                let n_recv = fanout.rx_count() as u8;
                let pkt = {
                    let mut s = hl2.inner.session.lock().await;
                    s.keepalive_frame(n_recv)
                };
                let _ = socket.send_to(&pkt, SocketAddr::new(hl2.inner.hl2_peer, HL2_PORT)).await;

                // 1 Hz delivery-rate diagnostic.
                if dbg {
                    let now = time::Instant::now();
                    let dt = now.duration_since(rate_start).as_secs_f64();
                    if dt >= 1.0 {
                        let n = delivered.swap(0, Ordering::Relaxed);
                        eprintln!(
                            "[pump] pair_rate={:.0}/s (all slots, {} active, {:.1}s)",
                            n as f64 / dt,
                            fanout.rx_count(),
                            dt,
                        );
                        rate_start = now;
                    }
                }
            }

            recv_res = socket.recv_from(&mut buf) => match recv_res {
                Ok((len, _)) => {
                    if len != DATA_PACKET_SIZE {
                        continue;
                    }
                    // Bind the de-interleave N and the destination rings to
                    // one consistent snapshot so a tune() arriving mid-frame
                    // cannot change the set between parse and push.
                    let (n_recv, rings) = fanout.snapshot();
                    let n = n_recv as u8;
                    // Parse into reused buffers (no per-frame allocation in
                    // steady state). The buffers are shared across events, so
                    // the baseband chunk is `clone`d for the by-value `Hl2Event`;
                    // the ring push is by reference and needs no clone.
                    let Some(parsed) = parse_receive_packet_into(&buf, n, &mut iq, &mut baseband)
                    else {
                        continue;
                    };

                    if let Some(cmd) = parsed.chunk1_command.or(parsed.chunk2_command) {
                        if tx.send(Hl2Event::CmdAck {
                            ack: cmd.header.ack,
                            raddr: cmd.header.raddr,
                            ptt: cmd.header.ptt,
                            data: cmd.data,
                        }).is_err() {
                            break;
                        }
                    }

                    if !parsed.iq.is_empty() {
                        if let Some(Item::Block(block)) = assembler.feed(&parsed.header, parsed.iq) {
                            if tx.send(Hl2Event::Block(block)).is_err() {
                                break;
                            }
                        }
                    }

                    for chunk in parsed.baseband.iter() {
                        // Fan every per_rx stream into its slot's ring.
                        // `rings` is in position order (= ascending slot
                        // order), which is the same order the de-interleaver
                        // produced `per_rx`. `per_rx.len()` is `n_recv` (=
                        // `rings.len()`), so index one-to-one.
                        for (p, samples) in chunk.per_rx.iter().enumerate() {
                            if p < rings.len() {
                                // Keep reception running if a reader poisoned the lock.
                                rings[p]
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                                    .push(samples);
                                delivered.fetch_add(samples.len(), Ordering::Relaxed);
                            }
                        }
                        if tx.send(Hl2Event::Baseband(chunk.clone())).is_err() {
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }
    }
}
