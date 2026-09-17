//! The `RadioHub` is Rocket's shared `State` — the single owner of the HL2
//! session. `Hl2` is `Arc`-based inside, so the hub is what lets many
//! websocket clients drive one physical radio.
//!
//! # Roles
//! * **Lazily start** the radio (ensure-running) on first `Start`.
//! * **Control path**: apply `ClientCmd`s and reply per-client via `mpsc`.
//! * **Fan-out**: consume the `Hl2` pump events in a background task, FFT
//!   the wideband blocks into 1024 linear bins, and broadcast to every
//!   connected websocket via `tokio::sync::broadcast`.
//! * **Virtual receiver** (audio): a `ClientCmd::SetVrx` request creates /
//!   tears down a [`hl2::receiver::VirtualReceiver`] that demods the radio's
//!   RX1 baseband stream (from the shared [`hl2::receiver::BasebandRing`])
//!   to `i16` mono at 4 800 Hz. A std demod thread feeds a shared
//!   [`hl2::receiver::BufSink`]; a coalescing tokio task drains it every
//!   ~24 ms and broadcasts `CH_AUDIO` frames to every connected websocket.

use std::collections::BTreeMap;
use std::net::IpAddr;

use rustfft::FftPlanner;
use tokio::sync::Mutex;
use tokio::sync::mpsc;

use crate::spectrum::display_mags_into;

use hl2::protocol::{ADC_CLOCK_HZ, IQ_PAIRS_PER_BLOCK};
use hl2::receiver::{
    AUTO_MODES, AudioConfig, AutoMode, BufSink, BufSinkHandle, DropSink, Ft4Tap, Ft8Tap,
    Js8SharedDecoder, Js8Tap, Mode, ReceiverConfig, Sideband, VirtualReceiver, closed_slot_for,
    decode_closed_slot, ft4_closed_slot_for, ft4_decode_closed_slot, ft4_shared, js8_shared,
    js8_step, shared,
};
use hl2::{DEFAULT_LNA_GAIN_DB, Hl2, Hl2Event, discover};
use hl2_common::{
    AutoMonitor, ClientCmd, DecodeLog, DiscoveryInfo, SampleFormat, ServerResponse, SharedState,
    SpectrumSource, SpotSink, VrxCfg, VrxMode, VrxState,
};

/// Map `hl2::receiver::AutoMode` (the auto-decode registry in `hl2`) to
/// the wire `hl2_common::VrxMode`. Kept inline in `shared_state` instead
/// of as a `From` impl because both `AutoMode` and `VrxMode` come from
/// other crates (orphan-trait rule); the two enums are kept in lockstep
/// manually — both lowercase — so this is the single mapping site.
fn auto_to_vrx_mode(m: AutoMode) -> VrxMode {
    match m {
        AutoMode::Ft8 => VrxMode::Ft8,
        AutoMode::Js8 => VrxMode::Js8,
        AutoMode::Ft4 => VrxMode::Ft4,
    }
}

/// One fan-out event as delivered to a single websocket subscriber.
#[derive(Debug, Clone)]
pub enum WsEvent {
    /// A JSON response / status snapshot.
    Json(ServerResponse),
    /// A wideband spectrum frame (server → client binary channel).
    Wideband { seq: u32, mags: Vec<u16> },
    /// A post-demod audio frame from one virtual receiver (server → client
    /// binary channel on `CH_AUDIO`). `samples` are `i16` mono at `rate_hz`,
    /// demodulated from the RX slot `slot`. Several receivers' audio share
    /// the channel; the client routes each block by `slot`.
    Audio {
        slot: u8,
        seq: u32,
        rate_hz: u16,
        samples: Vec<i16>,
    },
    /// A batch of decoded digital-mode messages (server → client JSON text
    /// frame `{"cmd":"log","data":[…]}`), for **any** digital mode (FT8 /
    /// FT4 / JS8 / …) — the mode is carried per-row in `vrx.mode`, so the
    /// client never branches on mode to receive a log batch. Sent at most
    /// once per wall-clock slot (FT8 15 s / FT4 7.5 s) or JS8 readiness
    /// tick, and only when the decoder produced ≥ 1 CRC-passing row.
    Log(DecodeLog),
}

/// Tunable knobs for the spectrum pipeline.
#[derive(Debug, Clone, Copy)]
pub struct HubConfig {
    /// Display bin count (1024 in the MVP).
    pub wideband_bins: usize,
    /// Number of 2048-sample blocks to accumulate per FFT. Must be power of 2.
    pub accumulate_blocks: usize,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            wideband_bins: 1024,
            accumulate_blocks: 8,
        }
    }
}

#[derive(Debug)]
struct Session {
    pub ctrl: Hl2,
    pub started: bool,
    pub rx_count: u8,
    pub sample_format: SampleFormat,
    pub tuning: BTreeMap<u8, u32>,
    pub lna_gain_db: i8,
    pub oc_bits: u8,
    pub device: DiscoveryInfo,
    vrx: std::collections::BTreeMap<u8, VrxTask>,
    /// Per-slot auto-decode pipelines: `(slot) -> [ AutoTask ]`. One
    /// [`AutoTask`] per (mode, in-window frequency) — see
    /// [`RadioHub::auto_decode_cmd`]. Populated by `AutoDecode { enabled:
    /// true }` and torn down by `AutoDecode { enabled: false }`, `Tune`
    /// (the UI locks the NCO while auto is on, but a stray Tune still
    /// tears it down — the UI re-enables), or `Stop`.
    auto: std::collections::BTreeMap<u8, Vec<AutoTask>>,
    /// The S-meter reading (level + floor, dB, and which slot they are for).
    /// Written by `run_spectral` (which already runs the band FFT) and read
    /// by `shared_state` into `vrx_levels` / `vrx_floors`. See
    /// `api/src/meter.rs` and PROTOCOL.md §16.3e.
    meter: std::sync::Arc<crate::meter::MeterState>,
}

/// A running virtual-receiver pipeline, demodulating one RX slot (optionally
/// at a non-zero NCO offset).
struct VrxTask {
    /// The currently active receiver configuration, echoed in
    pub state: VrxState,
    pub buf: BufSinkHandle,
    /// Stop flag shared with the demod std thread. Set on stop/torn-down.
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Mute flag shared with the audio fan-out task. Set via
    /// `ClientCmd::SetVrxMute`; while true the fan-out drains + discards
    muted: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The demod std thread — owns the `VirtualReceiver` and reads the
    /// shared `BasebandRing` it was spawned against. `JoinHandle` lets
    /// `stop` flush + join. The hub re-spawns this whole task whenever the
    /// slot's activation changes (see `rebind_vrx_on_activate`) — we do
    /// NOT need to track the captured ring here; the library's own
    /// `Hl2::is_slot_active` tells the hub whether the slot is currently
    /// activated, and that is the only activation state the idempotency
    /// checks need.
    demod: Option<std::thread::JoinHandle<()>>,
    /// The coalescing tokio fan-out task that drains `buf` and broadcasts
    /// `WsEvent::Audio` frames. Aborted on stop.
    fanout: Option<tokio::task::JoinHandle<()>>,
    // TODO: these really belong somewhere else
    ft8: Option<DecodeTask>,
    js8: Option<DecodeTask>,
    ft4: Option<DecodeTask>,
}

/// Stack size for a mode-decode thread (FT8/JS8): deep DSP frame chains
/// (FFT plans + multi-MB complex temporaries) have overflowed a Rocket
/// worker's default 8 MiB stack once, so the decode threads get their own
/// generous stack.
const MODE_DECODE_THREAD_STACK: usize = 32 * 1024 * 1024; // 32 MiB

/// A dedicated std-thread mode decoder (FT8/JS8 slot decode).
///
/// A mode decode burst runs the full window through many large FFTs + the
/// LDPC/BP stack (tens of MB of temporaries, several seconds of CPU on a
/// busy slot); running it inline on a Rocket worker thread starves the
/// fan-out / WS delivery paths that share that worker (spectrum + audio
/// lag in the UI), and large DSP frame chains have overflowed a worker's
/// stack before. The decode therefore lives on its own std thread with a
/// generous stack — exactly like the demod thread — and the thread is
/// parked (not joined) while it sleeps between 1 s ticks, so `stop` is a
/// flag + bounded `thread::join`.
struct DecodeTask {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// The shared JS8 decoder handle type (re-exported here to keep `spawn_auto`'s
/// locals typed identically whether we're in the FT8 or JS8 arm).
/// A headless auto-decode pipeline for one (slot, mode, target-frequency)
/// triple — i.e. the auto-decoder's per-frequency instance.
///
/// Same shape as [`VrxTask`] but with no audio in the loop: the demod
/// thread reads the slot's [`hl2::receiver::BasebandRing`], runs
/// [`hl2::receiver::VirtualReceiver`] at 12 kHz with the mode's
/// decoder-attached tap, and hands `i16` audio to a
/// [`hl2::receiver::DropSink`] (a no-op sink). The mode's decoder
/// (FT8's 15-second slot / JS8's continuous) accumulates the pre-AGC
/// `f32` the tap sampled inside the demod — exactly the way a
/// [`VrxTask`] does in FT8/JS8 mode — so both pipelines are byte-for-byte
/// the same DSP, just with the audio sink + fan-out task stripped out.
///
/// Decoded rows go to the shared `WsEvent::Log` envelope
/// broadcast the live vrx uses (the UI distinguishes auto rows by the
/// embedded `VrxState.muted == true`). PSK Reporter spot-ability is
/// driven by `PSK_CALL` as usual; no separate auto-spot path.
///
/// A `VrxState` is still constructed for the decode row so the UI's
/// existing table (which keys on `m.vrx.slot` / `m.vrx.offset_hz`)
/// attributes the row correctly: the slot is the real slot, the offset
/// is the (f − NCO) delta to the target frequency, the mode is the
/// registered mode, and `muted = true` flags "auto" to the UI.
struct AutoTask {
    /// The per-frequency identity this task is decoding — the UI's
    /// `auto_monitors[slot]` carries the (mode, freq) pair so the user
    /// can see exactly what is running.
    pub mode: AutoMode,
    pub freq_hz: u32,
    /// Stop flag shared with the demod std thread. Set on stop/torn-down.
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The demod std thread — owns the `VirtualReceiver` and reads the
    /// shared `BasebandRing`. `Option` so `stop` can `take()` + `join`
    /// without needing `std::mem::ManuallyDrop` tricks.
    demod: Option<std::thread::JoinHandle<()>>,
    /// The decode task (FT8's 15-second-slot thread or JS8's 500 ms-step
    /// thread). `stop` joins before the demod thread is joined so the
    /// decoder isn't mid-window when the input stops arriving.
    decode: DecodeTask,
}

impl std::fmt::Debug for AutoTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoTask")
            .field("mode", &self.mode)
            .field("freq_hz", &self.freq_hz)
            .finish_non_exhaustive()
    }
}

impl AutoTask {
    /// Stop the pipeline: signal the shared stop flag, then join the decode
    /// thread (long-lived, ~1-3 s at worst if mid-burst) and the demod
    /// thread (short — bounded by one `rx.process` + one 500 µs sleep
    /// iteration, plus the final `rx.flush`).
    ///
    /// **Callers must not be holding the `tokio::sync::Mutex` session lock
    /// while this joins.** The decode-thread join can block for up to one
    /// burst (~1-3 s) plus the next tick cycle. The two-phase pattern in
    /// each `*_cmd` (extract-under-lock → stop-outside-lock) exists to keep
    /// this off the hot path. `Drop` is a backstop for the unexpected
    /// (e.g. `Session` being dropped without an explicit stop) and accepts
    /// the risk of joining there.
    fn stop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.decode.stop();
        if let Some(d) = self.demod.take() {
            let _ = d.join();
        }
    }
}

impl std::ops::Drop for AutoTask {
    fn drop(&mut self) {
        self.stop();
    }
}

impl DecodeTask {
    fn stop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for DecodeTask {
    fn drop(&mut self) {
        self.stop();
    }
}

impl std::fmt::Debug for VrxTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VrxTask")
            .field("state", &self.state)
            .field("buf.len", &self.buf.len())
            .finish_non_exhaustive()
    }
}

impl VrxTask {
    /// Stop the pipeline: signal the shared stop flag, then stop the
    /// mode-decoder threads (long-lived — up to ~1-3 s at worst if one of
    /// them is mid-burst at the moment of teardown), then abort the audio
    /// fan-out tokio task, then finally join the demod std thread (short —
    /// bounded by one `rx.process` + one 500 µs sleep iteration, plus the
    /// final `rx.flush`).
    ///
    /// The ordering is deliberate: stop the decoders **before** the demod so
    /// the decoder isn't mid-window when the input stops arriving (matches
    /// [`AutoTask::stop`]), and stop the demod **last** so that if the decode
    /// thread is sleeping between ticks it still exits promptly instead of
    /// outliving the demod by up to a tick.
    ///
    /// **Callers must not be holding the `tokio::sync::Mutex` session lock
    /// while this runs.** The decode-thread join can block for up to one
    /// burst (~1-3 s) at a time per mode. The two-phase pattern in each
    /// `*_cmd` (extract-under-lock → stop-outside-lock) keeps this off the
    /// hot path. `Drop` is a backstop for the unexpected.
    fn stop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(mut t) = self.ft8.take() {
            t.stop();
        }
        if let Some(mut t) = self.js8.take() {
            t.stop();
        }
        if let Some(mut t) = self.ft4.take() {
            t.stop();
        }
        if let Some(t) = self.fanout.take() {
            t.abort();
        }
        if let Some(t) = self.demod.take() {
            let _ = t.join();
        }
    }
}

impl std::ops::Drop for VrxTask {
    fn drop(&mut self) {
        self.stop();
    }
}

impl Session {
    fn shared_state(&self, state_at: u64, src: &SpectrumSource) -> SharedState {
        // The S-meter (signal level + noise floor, both in dB relative to
        // full-scale) for the currently-displayed EP6 slot, written by the
        // `run_spectral` band-FFT loop and read here into the shared
        // contract's `vrx_levels` / `vrx_floors`. A slot of `0` (EP4 wideband
        // or no vrx running) means "no reading" — the UI parks the gauge at
        // its bottom.
        let (slot, lev, fl) = self.meter.read();
        let mut vrx_levels = std::collections::BTreeMap::new();
        let mut vrx_floors = std::collections::BTreeMap::new();
        if slot != 0 {
            vrx_levels.insert(slot, lev);
            vrx_floors.insert(slot, fl);
        }
        SharedState {
            started: self.started,
            rx_count: self.rx_count,
            tuning: self.tuning.clone(),
            sample_format: self.sample_format,
            adc_sample_rate_hz: 76_800_000,
            lna_gain_db: self.lna_gain_db,
            oc_bits: self.oc_bits,
            spectrum_source: src.clone(),
            state_at,
            spectrum_center_hz: spectrum_center(&self.tuning, src),
            spectrum_span_hz: spectrum_span(src),
            vrx: self.vrx.values().map(|t| (t.state.slot, t.state)).collect(),
            vrx_levels,
            vrx_floors,
            auto_monitors: self
                .auto
                .iter()
                .map(|(&slot, v)| {
                    (
                        slot,
                        v.iter()
                            .map(|t| AutoMonitor {
                                slot,
                                mode: auto_to_vrx_mode(t.mode),
                                freq_hz: t.freq_hz,
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        }
    }
}

/// The NCO the currently-selected spectrum source is centred on, in Hz.
fn spectrum_center(tuning: &BTreeMap<u8, u32>, src: &SpectrumSource) -> Option<u32> {
    match src {
        SpectrumSource::Ep6 { slot } => tuning.get(slot).copied(),
        SpectrumSource::Ep4 => tuning.values().next().copied(),
    }
}

/// Total displayed baseband bandwidth (Hz) for the current source.
fn spectrum_span(src: &SpectrumSource) -> u32 {
    match src {
        SpectrumSource::Ep6 { .. } => VRX_SOURCE_RATE_HZ,
        SpectrumSource::Ep4 => ADC_CLOCK_HZ,
    }
}

#[derive(Clone)]
pub struct RadioHub {
    cfg: HubConfig,
    hint: Option<IpAddr>,
    session: std::sync::Arc<Mutex<Option<Session>>>,
    /// Last discovery result; always present, independent of whether a
    /// session is live.
    last_devices: std::sync::Arc<Mutex<Vec<DiscoveryInfo>>>,
    fanout: std::sync::Arc<tokio::sync::broadcast::Sender<WsEvent>>,
    task: std::sync::Arc<std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Monotonic revision, bumped once per applied command. Stamped into
    /// `SharedState::state_at` so UI clients can discard out-of-order
    /// snapshots (last-writer-wins without a lock).
    state_rev: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// The periodic "S-meter" re-broadcast task — started alongside
    /// `run_spectral` on `Start`, aborted on `Stop`. Sends a
    /// `ServerResponse::welcome` (which carries `vrx_levels`) on a ~100 ms
    /// cadence so the UI's per-slot gauges don't wait for the next command
    /// to see a new meter value. Idle when no vrx is running (we check the
    /// map's length and skip the broadcast; the demod's own emission is
    /// the write-side, so the read-side is free to tick faster than that
    /// — the atomic's value just doesn't change).
    levels_task: std::sync::Arc<std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Which stream feeds the spectrum pipeline.
    spectrum_source: std::sync::Arc<std::sync::Mutex<SpectrumSource>>,
    /// Bumped every time `spectrum_source` is replaced so the accumulator in
    /// `run_spectral` knows to discard samples from the previous source.
    spectrum_rev: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// The S-meter reading for the *currently displayed* slot, written by
    /// the `run_spectral` loop (which already runs the band FFT) and read by
    /// `Session::shared_state` into `vrx_levels` / `vrx_floors`. See
    /// `api/src/meter.rs` and PROTOCOL.md §16.3e. Live for the whole process
    /// so a fresh client's `welcome` still carries the last reading.
    meter: std::sync::Arc<crate::meter::MeterState>,
    /// The channel-select bandwidth (Hz) of the running virtual receiver on
    /// the displayed slot, written by `set_vrx` / `ensure_vrx` and read by
    /// `run_spectral` to size the S-meter's passband window. `0` = no vrx.
    passband_bw: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Which side(s) of the tune the running receiver's passband occupies,
    /// as a [`VrxMode`] discriminant (`0` = Usb, `1` = Lsb, …). Written by
    /// `set_vrx` and read by `run_spectral` to select the S-meter's
    /// [`crate::meter::PassbandShape`].
    passband_mode: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// PSK Reporter sink (spot queue + station identity + send bookkeeping),
    /// shared with the `pskrep_hook` UDP send task. The hub appends
    /// spot-able FT8/JS8 decodes into it (see `spawn_ft8_decode` /
    /// `spawn_js8_decode`) and the
    /// send task drains + transmits; both live for the whole process.
    pskrep: crate::pskrep_hook::SharedPsk,
}

impl RadioHub {
    pub fn new(
        cfg: HubConfig,
        hint: Option<IpAddr>,
        pskrep: crate::pskrep_hook::SharedPsk,
    ) -> Self {
        let (fanout, _) = tokio::sync::broadcast::channel(256);
        Self {
            cfg,
            hint,
            session: std::sync::Arc::new(Mutex::new(None)),
            last_devices: std::sync::Arc::new(Mutex::new(Vec::new())),
            fanout: std::sync::Arc::new(fanout),
            task: std::sync::Arc::new(std::sync::Mutex::new(None)),
            state_rev: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            levels_task: std::sync::Arc::new(std::sync::Mutex::new(None)),
            spectrum_source: std::sync::Arc::new(std::sync::Mutex::new(SpectrumSource::default())),
            spectrum_rev: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            meter: std::sync::Arc::new(crate::meter::MeterState::new()),
            passband_bw: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            passband_mode: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(
                crate::meter::PassbandShape::Centered.as_u32(),
            )),
            pskrep,
        }
    }

    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<WsEvent> {
        self.fanout.subscribe()
    }

    /// Snapshot the current shared state + last-discovered devices, and
    /// broadcast a `ServerResponse::welcome` to every connected client.
    ///
    /// Called once per WebSocket connection so a freshly-opened tab immediately
    /// sees the authoritative radio state (started?, tuning, LNA) instead of
    /// relying on the spectrum stream implying "started". Also usable on
    /// reconnect after a network blip.
    pub async fn push_welcome(&self) {
        let (st, _) = self.snapshot().await;
        let mut devices = self.last_devices.lock().await.clone();
        // Guarantee the in-service radio is listed even if the last
        // Discover run *excluded* it (the hardware is streaming and does not
        // respond to a broadcast).
        if let Some(svc) = current_in_service(self).await {
            devices = merge_devices(devices, Some(svc));
        }
        let resp = ServerResponse::welcome(&st).with_devices(devices);
        if std::env::var("HL2_DEBUG").is_ok() {
            eprintln!(
                "[HUB] welcome: started={} lna={} state_at={}",
                st.started, st.lna_gain_db, st.state_at
            );
        }
        let _ = self.fanout.send(WsEvent::Json(resp));
    }

    /// Apply a command and broadcast the JSON response to all subscribers.
    ///
    /// The response is the same for every client (shared state + status), so
    /// fanning out through the broadcast channel is correct and simpler than a
    /// per-client reply channel.
    pub async fn handle_cmd(&self, id: u64, cmd: &ClientCmd) {
        if std::env::var("HL2_DEBUG").is_ok() {
            eprintln!("[HUB] cmd id={id} : {cmd:?}");
        }
        let _prev_rev = self
            .state_rev
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let resp = match cmd {
            ClientCmd::Discover => self.discover_cmd(id).await,
            ClientCmd::Start { initial_tuning } => self.start_cmd(id, initial_tuning).await,
            ClientCmd::Stop => self.stop_cmd(id).await,
            ClientCmd::Tune { slot, freq_hz } => self.tune_cmd(id, *slot, *freq_hz).await,
            ClientCmd::SetLnaGain { gain_db } => self.set_lna_gain_cmd(id, *gain_db).await,
            ClientCmd::SetOcBits { oc_bits } => self.set_oc_bits_cmd(id, *oc_bits).await,
            ClientCmd::SetSpectrumSource { source } => self.set_spectrum_source(id, source).await,
            ClientCmd::SetVrx { cfg } => self.set_vrx_cmd(id, cfg).await,
            ClientCmd::SetVrxMute { slot, muted } => self.set_vrx_mute_cmd(id, *slot, *muted).await,
            ClientCmd::SetVrxOff { slot } => self.set_vrx_off_cmd(id, *slot).await,
            ClientCmd::State => self.state_cmd(id).await,
            ClientCmd::AutoDecode { slot, enabled } => {
                self.auto_decode_cmd(id, *slot, *enabled).await
            }
        };

        let _ = self.fanout.send(WsEvent::Json(resp));
    }

    /// Periodically re-broadcast the current `SharedState` (wrapping in
    /// `ServerResponse::welcome`) so the UI's per-slot S-meter / gauge is
    /// kept alive between commands.
    ///
    /// The demod thread is the *writer* of `vrx_levels` (one atomic per
    /// vrx, updated every audio emit); nothing else changes the value, so
    /// the only thing the UI ever sees moving between commands is this
    /// meter. Commands (Start / Tune / SetVrx / …) each broadcast a fresh
    /// snapshot, but the meter itself only changes as audio streams — hence
    /// this ~100 ms tick, which is fast enough for a needle to feel alive
    /// but slow enough to be negligible CPU (a single snapshot + one
    /// `BroadcastChannel::send` per tick).
    ///
    /// We skip the broadcast when there is no vrx (nothing to show) and when
    /// the level map hasn't changed since the last tick (no point re-sending
    /// an identical snapshot — the UI would re-apply the same numbers). The
    /// comparison is against the *previous* tick's map, not a global "ever
    /// seen" cache, so a single vrx whose level is stable still produces
    /// periodic traffic (a needle that should be frozen IS frozen, which is
    /// the correct visual).
    ///
    /// Runs for the life of a session (started in `Start`, aborted in
    /// `Stop`) — see `levels_task`.
    async fn run_levels(&self) {
        let mut tick = tokio::time::interval(tokio::time::Duration::from_millis(100));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut last: BTreeMap<u8, f64> = BTreeMap::new();
        loop {
            let _ = tick.tick().await;
            let (state, _) = self.snapshot().await;
            if state.vrx.is_empty() {
                last.clear();
                continue;
            }
            if state.vrx_levels == last {
                continue;
            }
            last = state.vrx_levels.clone();
            let resp = ServerResponse::welcome(&state);
            let _ = self.fanout.send(WsEvent::Json(resp));
        }
    }

    /// Snapshot the current shared state (release the session lock first).
    ///
    /// The snapshot is stamped with the current `state_rev`, so a command that
    /// bumped the revision before its snapshot produces a strictly-newer state
    /// than any earlier command's response.
    async fn snapshot(&self) -> (SharedState, Option<Hl2>) {
        let rev = self.state_rev.load(std::sync::atomic::Ordering::Relaxed);
        let src = self.spectrum_source.lock().unwrap().clone();
        let span_hz = spectrum_span(&src);
        let guard = self.session.lock().await;
        match guard.as_ref() {
            Some(s) => (s.shared_state(rev, &src), Some(s.ctrl.clone())),
            None => (
                SharedState {
                    started: false,
                    rx_count: 0,
                    tuning: BTreeMap::new(),
                    sample_format: SampleFormat::Sample16,
                    adc_sample_rate_hz: 76_800_000,
                    lna_gain_db: DEFAULT_LNA_GAIN_DB,
                    oc_bits: 0,
                    spectrum_source: src,
                    state_at: rev,
                    spectrum_center_hz: None,
                    spectrum_span_hz: span_hz,
                    vrx: std::collections::BTreeMap::new(),
                    auto_monitors: std::collections::BTreeMap::new(),
                    vrx_levels: std::collections::BTreeMap::new(),
                    vrx_floors: std::collections::BTreeMap::new(),
                },
                None,
            ),
        }
    }

    /// Switch which stream (EP4 wideband vs. a specific slot's EP6 baseband)
    /// feeds the spectrum pipeline.
    pub async fn set_spectrum_source(&self, id: u64, source: &SpectrumSource) -> ServerResponse {
        if std::env::var("HL2_DEBUG").is_ok() {
            eprintln!("[HUB] set spectrum source → {source:?}");
        }
        *self.spectrum_source.lock().unwrap() = source.clone();
        self.spectrum_rev
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (state, _) = self.snapshot().await;
        ServerResponse::ok(id, &state)
    }

    async fn discover_cmd(&self, id: u64) -> ServerResponse {
        let found = discover().await;
        let devices = convert_devices(found);
        // Fold the radio we're currently serving back in if it is missing
        // (the hardware will not answer a discovery broadcast while in use).
        let in_service = current_in_service(self).await;
        let devices = merge_devices(devices, in_service);
        *self.last_devices.lock().await = devices.clone();
        let (state, _) = self.snapshot().await;
        ServerResponse::ok(id, &state).with_devices(devices)
    }

    async fn start_cmd(
        &self,
        id: u64,
        initial_tuning: &Option<BTreeMap<u8, u32>>,
    ) -> ServerResponse {
        {
            let guard = self.session.lock().await;
            if let Some(s) = guard.as_ref() {
                if s.started {
                    let rev = self.state_rev.load(std::sync::atomic::Ordering::Relaxed);
                    let src = self.spectrum_source.lock().unwrap().clone();
                    return ServerResponse::ok(id, &s.shared_state(rev, &src));
                }
            }
        }

        let (state0, _) = self.snapshot().await;

        // Resolve the target address: hint, else first discovered.
        let mut addr = self.hint;
        if addr.is_none() {
            let found = discover().await;
            let devices = convert_devices(found.clone());
            *self.last_devices.lock().await = devices;
            addr = found.into_iter().next().map(|(a, _)| a.ip());
        }
        let addr = match addr {
            Some(a) => a,
            None => {
                let devices = self.last_devices.lock().await.clone();
                if let Some(ip) = devices.first().and_then(|d| {
                    d.addr
                        .split(':')
                        .next()
                        .and_then(|ip| ip.parse::<std::net::IpAddr>().ok())
                }) {
                    ip
                } else {
                    return ServerResponse::err(
                        id,
                        &state0,
                        "no HL2 discovered and no IP hint configured",
                    );
                }
            }
        };

        match Hl2::start(addr).await {
            Ok((ctrl, ev_rx, info)) => {
                let sample_format = match info.sample_format {
                    hl2::protocol::data::SampleFormat::Sample12 => SampleFormat::Sample12,
                    hl2::protocol::data::SampleFormat::Sample16 => SampleFormat::Sample16,
                };
                // Capture the in-service radio's identity now: it will NOT
                // respond to further discovery broadcasts while streaming
                let last_dev = self.last_devices.lock().await.clone();
                let device = session_device(
                    addr,
                    &last_dev,
                    info.rx_count,
                    sample_format == SampleFormat::Sample16,
                );
                {
                    let mut lg = self.last_devices.lock().await;
                    let ip_bytes = device.ip;
                    if let Some(ex) = lg.iter_mut().find(|d| d.ip == ip_bytes) {
                        *ex = device.clone();
                    } else {
                        lg.push(device.clone());
                    }
                }
                let mut session = Session {
                    ctrl: ctrl.clone(),
                    started: true,
                    rx_count: info.rx_count as u8,
                    sample_format,
                    tuning: BTreeMap::new(),
                    lna_gain_db: ctrl.lna_gain().await,
                    oc_bits: ctrl.oc_bits().await,
                    device,
                    vrx: std::collections::BTreeMap::new(),
                    auto: std::collections::BTreeMap::new(),
                    meter: self.meter.clone(),
                };

                if let Some(t) = initial_tuning {
                    for (&slot, &freq) in t {
                        if let Err(e) = session.ctrl.tune(slot, freq).await {
                            eprintln!("tune slot={slot} failed: {e}");
                        } else {
                            session.tuning.insert(slot, freq);
                        }
                    }
                }

                *self.session.lock().await = Some(session);

                // (Re)spawn the spectrum pipeline task, aborting any prior one.
                if let Some(old) = self.task.lock().unwrap().take() {
                    old.abort();
                }
                let cfg = self.cfg;
                let fanout = self.fanout.clone();
                let src = self.spectrum_source.clone();
                let src_rev = self.spectrum_rev.clone();
                let meter = self.meter.clone();
                let passband_bw = self.passband_bw.clone();
                let passband_mode = self.passband_mode.clone();
                let handle = tokio::spawn(async move {
                    run_spectral(
                        cfg,
                        ev_rx,
                        fanout,
                        src,
                        src_rev,
                        meter,
                        passband_bw,
                        passband_mode,
                    )
                    .await
                });
                *self.task.lock().unwrap() = Some(handle);

                // Start the periodic S-meter re-broadcast task.
                if let Some(old) = self.levels_task.lock().unwrap().take() {
                    old.abort();
                }
                let levels_handle = tokio::spawn({
                    let self2 = self.clone();
                    async move { self2.run_levels().await }
                });
                *self.levels_task.lock().unwrap() = Some(levels_handle);

                let (state, _) = self.snapshot().await;
                ServerResponse::ok(id, &state)
            }
            Err(e) => ServerResponse::err(id, &state0, e.to_string()),
        }
    }

    async fn stop_cmd(&self, id: u64) -> ServerResponse {
        // Copy out what we need then run.
        let ctrl = {
            let guard = self.session.lock().await;
            guard.as_ref().and_then(|s| {
                if s.started {
                    Some(s.ctrl.clone())
                } else {
                    None
                }
            })
        };

        if let Some(ctrl) = ctrl {
            if let Err(e) = ctrl.stop().await {
                eprintln!("stop failed: {e}");
            }
            // Phase 1 — under the session lock, mark stopped + extract the
            // vrx/auto pipelines out of the maps. Holding the lock while we
            // do this is the same as before, but we do NOT call `.stop()`
            // (which joins std threads — a decode thread mid-burst can be
            // 1-3 s) on this phase.
            //
            // Phase 2 — outside the lock, call `.stop()` on each extracted
            // pipeline. Between phases the session lock is released, so
            // other commands (tune, mute, start, another vrx stop) can make
            // progress while we join threads. `snapshot()` re-acquires it
            // once we're done.
            //
            // The extracted pipelines are dropped at the end of the
            // `if let Some(ctrl)` scope; the `Drop` impls for VrxTask /
            // AutoTask are a backstop (idempotent — they `take()` the
            // handles so a double-stop is harmless).
            let mut vrx_drained: Vec<VrxTask> = Vec::new();
            let mut auto_drained: Vec<Vec<AutoTask>> = Vec::new();
            {
                let mut guard = self.session.lock().await;
                if let Some(s) = guard.as_mut() {
                    s.started = false;
                    vrx_drained = std::mem::take(&mut s.vrx).into_values().collect();
                    auto_drained = std::mem::take(&mut s.auto).into_values().collect();
                }
            }
            let vrx_torn = vrx_drained.len();
            let auto_torn = auto_drained.iter().map(|v| v.len()).sum::<usize>();

            // Abort the spectrum task BEFORE phase 2 — it's the pump's
            // consumer; once the pump notices the channel is closed it
            // exits (see `hl2::Hl2::run_loop`), which stops the write-side
            // of every slot's `BasebandRing`. Demod threads see that as a
            // silent ring (0 peeks) and exit on the stop flag's next 500 µs
            // check. (Ordering doesn't matter here — the stop flags below
            // are independent of the pump — but aborting first lets the
            // ring's write side close out before we join the demods.)
            if let Some(t) = self.task.lock().unwrap().take() {
                t.abort();
            }
            if let Some(t) = self.levels_task.lock().unwrap().take() {
                t.abort();
            }

            // Phase 2 — outside the session lock. Stop each vrx pipeline
            // (each stops its decode threads + demod thread) and then each
            // auto pipeline. Order is vrx-first then auto — purely cosmetic,
            // each pipeline is independent.
            for v in vrx_drained.iter_mut() {
                v.stop();
            }
            for vec in auto_drained.iter_mut() {
                for t in vec.iter_mut() {
                    t.stop();
                }
            }

            if vrx_torn > 0 && std::env::var("HL2_DEBUG").is_ok() {
                eprintln!("[HUB] {vrx_torn} vrx pipeline(s) torn down (Stop)");
            }
            if auto_torn > 0 && std::env::var("HL2_DEBUG").is_ok() {
                eprintln!("[HUB] {auto_torn} auto pipeline(s) torn down (Stop)");
            }
        }
        let (state, _) = self.snapshot().await;
        ServerResponse::ok(id, &state)
    }

    async fn tune_cmd(&self, id: u64, slot: u8, freq_hz: u32) -> ServerResponse {
        let ctrl = {
            let guard = self.session.lock().await;
            guard.as_ref().and_then(|s| {
                if s.started {
                    Some(s.ctrl.clone())
                } else {
                    None
                }
            })
        };
        let (state0, _) = self.snapshot().await;

        // Does this tune *activate* the slot (i.e. it wasn't active before)?
        // We capture the pre-tune state now so the predicate is stable by
        // the time `h.tune` has run and registered the slot in the
        // `BasebandFanout` (side effect of `Hl2::tune`; see the
        // `Hl2::is_slot_active` doc). If the slot was already active and
        // this is a plain retune, the fan-out retains the same ring `Arc`
        // across `register_slot` calls, so any vrx running on this slot
        // still points at the right ring and needs no re-spawn.
        let activates = match &ctrl {
            Some(c) => !c.is_slot_active(slot),
            None => false,
        };

        match ctrl {
            Some(c) => match c.tune(slot, freq_hz).await {
                Ok(()) => {
                    // Phase 1 — under the session lock, update the tuning map
                    // and (if auto-decode is on for this slot) extract the
                    // auto pipelines out of the map. Holding the lock here is
                    // fine — we don't do any thread joins inside.
                    //
                    // Phase 2 — outside the lock, stop (join threads) the
                    // extracted pipelines. Without this split a Tune on a
                    // slot with auto-decode on would hold the session lock
                    // for the duration of up to 3 * ~1-3 s decode-thread
                    // joins (one per mode, one per in-window frequency),
                    // blocking every other command.
                    let mut auto_drained: Vec<AutoTask> = Vec::new();
                    {
                        let mut guard = self.session.lock().await;
                        if let Some(s) = guard.as_mut() {
                            s.tuning.insert(slot, freq_hz);
                            // Safety net: a Tune on a slot that has auto-decode
                            // on (stray client / UI race) invalidates the
                            // (NCO, target) pairs the decoders were built
                            // against. Tear the slot's auto pipelines down;
                            // the UI re-enables (its checkbox mirrors
                            // `auto_monitors`, which now lacks the slot).
                            if let Some(v) = s.auto.remove(&slot) {
                                auto_drained = v;
                            }
                        }
                    }
                    let torn = auto_drained.len();
                    for t in auto_drained.iter_mut() {
                        t.stop();
                    }
                    if torn > 0 && std::env::var("HL2_DEBUG").is_ok() {
                        eprintln!("[HUB] {torn} auto pipeline(s) torn down (Tune slot={slot})");
                    }
                    if activates {
                        self.rebind_vrx_on_activate(slot).await;
                    }
                    let (state, _) = self.snapshot().await;
                    ServerResponse::ok(id, &state)
                }
                Err(e) => ServerResponse::err(id, &state0, e.to_string()),
            },
            None => ServerResponse::err(id, &state0, "HL2 not started"),
        }
    }

    async fn set_lna_gain_cmd(&self, id: u64, gain_db: i8) -> ServerResponse {
        let ctrl = {
            let guard = self.session.lock().await;
            guard.as_ref().and_then(|s| {
                if s.started {
                    Some(s.ctrl.clone())
                } else {
                    None
                }
            })
        };
        let (state0, _) = self.snapshot().await;

        match ctrl {
            Some(c) => match c.set_lna_gain(gain_db).await {
                Ok(()) => {
                    let active = c.lna_gain().await;
                    {
                        let mut guard = self.session.lock().await;
                        if let Some(s) = guard.as_mut() {
                            s.lna_gain_db = active;
                        }
                    }
                    let (state, _) = self.snapshot().await;
                    ServerResponse::ok(id, &state)
                }
                Err(e) => ServerResponse::err(id, &state0, e.to_string()),
            },
            None => ServerResponse::err(id, &state0, "HL2 not started"),
        }
    }

    async fn state_cmd(&self, id: u64) -> ServerResponse {
        let (st, _) = self.snapshot().await;
        ServerResponse::ok(id, &st)
    }

    async fn set_oc_bits_cmd(&self, id: u64, oc_bits: u8) -> ServerResponse {
        if std::env::var("HL2_DEBUG").is_ok() {
            eprintln!("[HUB] set oc bits → 0x{oc_bits:02x}");
        }
        let ctrl = {
            let guard = self.session.lock().await;
            guard.as_ref().and_then(|s| {
                if s.started {
                    Some(s.ctrl.clone())
                } else {
                    None
                }
            })
        };
        let (state0, _) = self.snapshot().await;

        match ctrl {
            Some(c) => {
                c.set_oc_bits(oc_bits).await;
                let active = c.oc_bits().await;
                {
                    let mut guard = self.session.lock().await;
                    if let Some(s) = guard.as_mut() {
                        s.oc_bits = active;
                    }
                }
                let (state, _) = self.snapshot().await;
                ServerResponse::ok(id, &state)
            }
            None => ServerResponse::err(id, &state0, "HL2 not started"),
        }
    }

    /// Create / update / destroy the virtual receiver for one RX slot.
    async fn set_vrx_cmd(&self, id: u64, cfg: &Option<VrxCfg>) -> ServerResponse {
        if std::env::var("HL2_DEBUG").is_ok() {
            eprintln!("[HUB] set vrx → {cfg:?}");
        }
        let (state0, _) = self.snapshot().await;
        let started = state0.started;

        match cfg.as_ref() {
            // Global off: tear down every slot's pipeline.
            None => {
                // Phase 1 — extract all vrx pipelines under the lock. Phase 2
                // — stop (join threads) them outside the lock, so other
                // commands aren't blocked while we join decode/demod threads.
                let mut vrx_drained: Vec<VrxTask> = Vec::new();
                {
                    let mut guard = self.session.lock().await;
                    if let Some(s) = guard.as_mut() {
                        vrx_drained = std::mem::take(&mut s.vrx).into_values().collect();
                    }
                }
                let torn = vrx_drained.len();
                for v in vrx_drained.iter_mut() {
                    v.stop();
                }
                if std::env::var("HL2_DEBUG").is_ok() {
                    eprintln!("[HUB] vrx off — {torn} slot(s) torn down");
                }
                let (state, _) = self.snapshot().await;
                ServerResponse::ok(id, &state)
            }
            Some(c) => {
                if !started {
                    return ServerResponse::err(id, &state0, "HL2 not started");
                }
                // Idempotency: if this slot already has a vrx whose config
                // matches the request *and* the slot is currently activated,
                // return the current snapshot without tearing down and
                // re-spawning the demod thread. UI reconciliation can (and
                // does) re-send `setvrx` for the active tab on every stale
                // echo it cannot match against a fresh response — a
                // rebuild-per-command would churn the demod, reset AGC
                // state, and spam `vrx started` logs endlessly. Equality is
                // on (slot, mode, bw, gain, offset); `muted` is deliberately
                // NOT part of it (mute is a separate command, and a stale-mute
                // `setvrx` must not force a rebuild just because the mute
                // intent changed).
                //
                // Activation matters too: `Hl2::baseband_ring(slot)` resolves
                // to the slot's *dedicated* ring only once `tune(slot, ..)`
                // has registered it in the fan-out (the first successful
                // tune does this), and falls back to the position-0 (RX1)
                // ring until then. A receiver spawned *before* its slot was
                // tuned is therefore demodulating the wrong ring; we must
                // re-spawn it against the now-dedicated ring. The library
                // exposes `Hl2::is_slot_active(slot)` for exactly this
                // predicate — the hub asks the library instead of re-deriving
                // "is this slot alive" from its own bookkeeping. `tune_cmd`
                // re-binds any affected vrx as the slot comes online (see
                // `rebind_vrx_on_activate`), so the two paths agree without
                // comparing ring-handle identity.
                let up_to_date = {
                    let guard = self.session.lock().await;
                    guard.as_ref().is_some_and(|s| {
                        s.ctrl.is_slot_active(c.slot)
                            && s.vrx.get(&c.slot).is_some_and(|existing| {
                                existing.state.mode == c.mode
                                    && existing.state.bw_hz == c.bw_hz
                                    && (existing.state.gain_db - c.gain_db).abs() < 1e-6
                                    && existing.state.offset_hz == c.offset_hz
                            })
                    })
                };
                if up_to_date {
                    let (state, _) = self.snapshot().await;
                    return ServerResponse::ok(id, &state);
                }
                // Rebuild only *this* slot's pipeline (other slots keep
                // streaming). Extract-under-lock, stop-outside-lock (see
                // `VrxTask::stop` doc): the decode-thread join can block for
                // up to one burst, so keep the session lock free of it.
                let mut old_vrx: Option<VrxTask> = None;
                {
                    let mut guard = self.session.lock().await;
                    if let Some(s) = guard.as_mut() {
                        old_vrx = s.vrx.remove(&c.slot);
                    }
                }
                if let Some(mut v) = old_vrx {
                    v.stop();
                }
                // Fetch the per-slot ring for the receiver we are about to
                // demodulate.
                let ring = {
                    let guard = self.session.lock().await;
                    guard.as_ref().map(|s| s.ctrl.baseband_ring(c.slot))
                };
                match ring {
                    Some(ring) => match spawn_vrx(
                        ring,
                        self.fanout.clone(),
                        c,
                        self.session.clone(),
                        self.pskrep.clone(),
                    )
                    .await
                    {
                        Ok(vrx) => {
                            {
                                let mut guard = self.session.lock().await;
                                if let Some(s) = guard.as_mut() {
                                    s.vrx.insert(c.slot, vrx);
                                }
                            }
                            if std::env::var("HL2_DEBUG").is_ok() {
                                eprintln!(
                                    "[HUB] vrx started (slot={} offset={})",
                                    c.slot, c.offset_hz
                                );
                            }
                            self.passband_bw
                                .store(c.bw_hz as u64, std::sync::atomic::Ordering::Relaxed);
                            let shape = crate::meter::shape_for_mode(&c.mode);
                            self.passband_mode
                                .store(shape.as_u32(), std::sync::atomic::Ordering::Relaxed);
                            let (state, _) = self.snapshot().await;
                            ServerResponse::ok(id, &state)
                        }
                        Err(e) => {
                            eprintln!("[HUB] vrx spawn failed: {e}");
                            ServerResponse::err(id, &state0, e)
                        }
                    },
                    None => ServerResponse::err(id, &state0, "HL2 not started"),
                }
            }
        }
    }

    /// Mute / unmute the virtual receiver on one RX slot without rebuilding
    async fn set_vrx_mute_cmd(&self, id: u64, slot: u8, muted: bool) -> ServerResponse {
        if std::env::var("HL2_DEBUG").is_ok() {
            eprintln!("[HUB] set vrx mute → slot={slot} muted={muted}");
        }
        {
            let mut guard = self.session.lock().await;
            if let Some(s) = guard.as_mut() {
                if let Some(vrx) = s.vrx.get_mut(&slot) {
                    vrx.muted.store(muted, std::sync::atomic::Ordering::Relaxed);
                    vrx.state.muted = muted;
                }
            }
        }
        let (state, _) = self.snapshot().await;
        ServerResponse::ok(id, &state)
    }

    /// Tear down the virtual receiver on one RX slot
    async fn set_vrx_off_cmd(&self, id: u64, slot: u8) -> ServerResponse {
        if std::env::var("HL2_DEBUG").is_ok() {
            eprintln!("[HUB] set vrx off → slot={slot}");
        }
        let mut old_vrx: Option<VrxTask> = None;
        {
            let mut guard = self.session.lock().await;
            if let Some(s) = guard.as_mut() {
                old_vrx = s.vrx.remove(&slot);
            }
        }
        if let Some(mut v) = old_vrx {
            v.stop();
            if std::env::var("HL2_DEBUG").is_ok() {
                eprintln!("[HUB] vrx {slot} torn down (off)");
            }
        }
        let (state, _) = self.snapshot().await;
        ServerResponse::ok(id, &state)
    }

    /// Re-bind the virtual receiver on `slot` to the slot's *current*
    /// baseband ring, preserving the operator's existing config (mode / bw /
    /// gain / offset).
    ///
    /// Why: a `VrxTask`'s demod thread binds to a specific `BasebandRing`
    /// `Arc` at spawn time (see `spawn_vrx`). `Hl2::baseband_ring(slot)`
    /// resolves to the slot's *dedicated* ring only once the slot has been
    /// registered in the `BasebandFanout` — which happens as a side effect
    /// of the first successful `tune(slot, f≠0)`. Before that, it falls
    /// back to the position-0 (RX1) ring. A vrx therefore spawned *before*
    /// its slot was tuned is demodulating the wrong ring: the UI's audio
    /// (and any decoders on it) track RX1, not this slot.
    ///
    /// The hub is the one that sends `tune`, so *it* can keep the vrx in
    /// sync: when a tune *activates* a slot (it wasn't active before), any
    /// vrx currently running on that slot is still bound to the fallback
    /// (RX1) ring. Re-resolve the ring and re-spawn the vrx against the
    /// newly-registered dedicated one. The "was it active?" predicate is
    /// the library's own — `Hl2::is_slot_active(slot)` — so the hub's
    /// decision is a contract with the library, not a re-derivation from
    /// hub-internal bookkeeping. `set_vrx_cmd` uses the same predicate (via
    /// `s.ctrl.is_slot_active`) in its idempotency gate, so the two paths
    /// agree.
    ///
    /// Only reachable when there is actually a vrx running on the slot and
    /// the tune just activated it. A plain *retune* of an already-active
    /// slot is not routed here: the fan-out retains the same `Arc` for the
    /// slot across `register_slot` calls (see
    /// `BasebandFanout::register_slot`), so the running demod's ring handle
    /// is still valid for the new NCO.
    ///
    /// Mirrors the extract-under-lock → stop-outside-lock → spawn → insert
    /// sequence in `set_vrx_cmd`: there is never a window with two vrx tasks
    /// on the same slot, and the session `tokio::sync::Mutex` is held only
    /// long enough to do the map operations (`VrxTask::stop` can block for
    /// up to one decoder burst, ~1-3 s, on the decode-thread join).
    async fn rebind_vrx_on_activate(&self, slot: u8) {
        // Extract the slot's existing vrx under the lock, reading its config
        // and mute intent out (we're about to drop the `VrxTask`, so we
        // must carry both forward for the replacement). Mirrors the
        // extract-under-lock start of `set_vrx_cmd`'s rebuild path — except
        // that `set_vrx_cmd` rebuilds from an explicit operator request, so
        // the mute intent re-asserts via a follow-up `setvrxmute` the UI
        // already sends; here we must preserve the prior state directly so
        // the new vrx comes up in the operator's mute mode.
        let mut old_vrx: Option<VrxTask> = None;
        let mut cfg: Option<hl2_common::VrxCfg> = None;
        let mut kept_muted: bool = false;
        {
            let mut guard = self.session.lock().await;
            if let Some(s) = guard.as_mut() {
                if let Some(v) = s.vrx.remove(&slot) {
                    kept_muted = v.muted.load(std::sync::atomic::Ordering::Relaxed);
                    cfg = Some(hl2_common::VrxCfg {
                        slot: v.state.slot,
                        offset_hz: v.state.offset_hz,
                        mode: v.state.mode,
                        bw_hz: v.state.bw_hz,
                        gain_db: v.state.gain_db,
                    });
                    old_vrx = Some(v);
                }
            }
        }
        // Nothing to re-bind — no vrx was running on this slot. `set_vrx`
        // will spawn one (now that the slot is activated, it picks up the
        // correct dedicated ring).
        let Some(cfg) = cfg else {
            return;
        };
        // Stop the old pipeline off the lock. `muted` has already been
        // captured; the new vrx will be set back to it after `spawn_vrx`.
        if let Some(mut v) = old_vrx {
            v.stop();
        }
        // Resolve the slot's *current* ring. Now that the tune has run (and
        // this is only called on activation transitions), this is the
        // slot's dedicated ring, not the fallback.
        let ring = {
            let guard = self.session.lock().await;
            guard.as_ref().map(|s| s.ctrl.baseband_ring(slot))
        };
        let Some(ring) = ring else {
            return;
        };
        match spawn_vrx(
            ring,
            self.fanout.clone(),
            &cfg,
            self.session.clone(),
            self.pskrep.clone(),
        )
        .await
        {
            Ok(mut new_vrx) => {
                // Restore the operator's mute intent onto the fresh task
                // before the map swap (see the capture above).
                new_vrx
                    .muted
                    .store(kept_muted, std::sync::atomic::Ordering::Relaxed);
                new_vrx.state.muted = kept_muted;
                {
                    let mut guard = self.session.lock().await;
                    if let Some(s) = guard.as_mut() {
                        s.vrx.insert(slot, new_vrx);
                    }
                }
                if std::env::var("HL2_DEBUG").is_ok() {
                    eprintln!("[HUB] vrx re-bound for slot={slot} after tune (dedicated ring)");
                }
            }
            Err(e) => {
                eprintln!("[HUB] vrx rebind after tune slot={slot} failed: {e}");
            }
        }
    }

    /// Enable / disable auto-decode on one RX slot.
    ///
    /// `enabled=true` computes the set of (mode, target-frequency) pairs
    /// registered in [`hl2::receiver::AutoMode::known_freqs`] that fall
    /// within the slot's EP6 baseband (± half-span) against the slot's
    /// current tune, tears down any existing slot auto, and spawns a
    /// headless `VirtualReceiver` + the mode's decoder pipeline for each
    /// in-window entry. `enabled=false` tears down the slot's pipeline.
    ///
    /// Decodes are pushed on the shared `WsEvent::Log` envelope
    /// the live vrx uses; the UI distinguishes auto rows by
    /// `Ft8Decode.vrx.muted == true` (or JS8's `Js8Decode.vrx.muted`),
    /// which `spawn_auto` sets via the synthetic `VrxState`.
    ///
    /// The UI is expected to lock the slot's NCO while auto is on (the
    /// stepper becomes read-only) so the (NCO, target) pair the decoders
    /// were built against doesn't silently drift. A `Tune` command on a
    /// slot that has auto on tears the slot's auto down (the UI re-enables
    /// after showing "Auto Decode" as off — this matches the "lock"
    /// mental model; we don't try to be clever about mid-tune rebuilds).
    async fn auto_decode_cmd(&self, id: u64, slot: u8, enabled: bool) -> ServerResponse {
        if std::env::var("HL2_DEBUG").is_ok() {
            eprintln!("[HUB] auto decode → slot={slot} enabled={enabled}");
        }

        if !enabled {
            // Phase 1 — extract under the lock; phase 2 — stop (join threads)
            // outside it. See `VrxTask::stop` / `AutoTask::stop` for why the
            // join must not run under the session lock.
            let mut auto_drained: Vec<AutoTask> = Vec::new();
            {
                let mut guard = self.session.lock().await;
                if let Some(s) = guard.as_mut() {
                    auto_drained = s.auto.remove(&slot).unwrap_or_default();
                }
            }
            let torn = auto_drained.len();
            for t in auto_drained.iter_mut() {
                t.stop();
            }
            if std::env::var("HL2_DEBUG").is_ok() {
                eprintln!("[HUB] auto slot={slot}: {torn} pipeline(s) torn down");
            }
            let (state, _) = self.snapshot().await;
            return ServerResponse::ok(id, &state);
        }

        let (state0, _) = self.snapshot().await;
        let Some(nco_hz) = state0.tuning.get(&slot).copied() else {
            return ServerResponse::err(id, &state0, &format!("slot {slot} is not tuned"));
        };
        if !state0.started {
            return ServerResponse::err(id, &state0, "HL2 not started");
        }

        // The slot's EP6 half-span (Hz): in-window targets are within
        // `half_span` of the NCO. `spectrum_span` is the full window.
        let half_span = spectrum_span(&self.spectrum_source.lock().unwrap()) / 2;

        // Tear down the slot's existing auto pipelines (if any), so the
        // rebuilt set is exactly the in-window set. Extract-under-lock,
        // stop-outside-lock (same two-phase pattern as elsewhere): a torn
        // auto set can be up to 3 * (in-window freqs) pipelines, each with
        // a decode-thread join of up to one burst, so don't hold the session
        // lock through it.
        let mut auto_drained: Vec<AutoTask> = Vec::new();
        {
            let mut guard = self.session.lock().await;
            if let Some(s) = guard.as_mut() {
                auto_drained = s.auto.remove(&slot).unwrap_or_default();
            }
        }
        for t in auto_drained.iter_mut() {
            t.stop();
        }

        let ring = {
            let guard = self.session.lock().await;
            guard.as_ref().map(|s| s.ctrl.baseband_ring(slot))
        };
        let Some(ring) = ring else {
            return ServerResponse::err(id, &state0, "HL2 not started");
        };

        let mut spawned: Vec<AutoTask> = Vec::new();
        for &m in AUTO_MODES {
            for &freq in m.known_freqs() {
                if (freq as i64 - nco_hz as i64).unsigned_abs() <= half_span as u64 {
                    match spawn_auto(
                        ring.clone(),
                        self.fanout.clone(),
                        self.session.clone(),
                        slot,
                        nco_hz,
                        freq,
                        m,
                        self.pskrep.clone(),
                    )
                    .await
                    {
                        Ok(task) => spawned.push(task),
                        Err(e) => {
                            eprintln!(
                                "[HUB] auto slot={slot} {} freq={freq}: spawn failed: {e}",
                                m.wire()
                            );
                        }
                    }
                }
            }
        }

        let summary: Vec<(AutoMode, u32)> = spawned.iter().map(|t| (t.mode, t.freq_hz)).collect();
        if std::env::var("HL2_DEBUG").is_ok() {
            for (m, f) in &summary {
                eprintln!(
                    "[HUB] auto slot={slot} {} @ freq={} (nco={}, offset={:>8} Hz)",
                    m.wire(),
                    f,
                    nco_hz,
                    *f as i64 - nco_hz as i64
                );
            }
        }

        {
            let mut guard = self.session.lock().await;
            if let Some(s) = guard.as_mut() {
                s.auto.insert(slot, spawned);
            }
        }
        let (state, _) = self.snapshot().await;
        ServerResponse::ok(id, &state)
    }
}

impl std::fmt::Debug for RadioHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RadioHub")
            .field("cfg", &self.cfg)
            .field("hint", &self.hint)
            .finish_non_exhaustive()
    }
}

fn convert_devices(found: Vec<(std::net::SocketAddr, hl2::DiscoveryInfo)>) -> Vec<DiscoveryInfo> {
    found
        .into_iter()
        .map(|(addr, info)| DiscoveryInfo {
            mac: info.mac,
            gateware_major: info.gateware_major,
            gateware_minor: info.gateware_minor,
            board_id: info.board_id,
            rx_count: info.rx_count,
            is_sending: info.is_sending,
            sample_16bit: info.sample_16bit,
            ip: info.ip,
            addr: addr.to_string(),
            in_service: false,
        })
        .collect()
}

/// Build the `in_service: true` `DiscoveryInfo` the hub reports for the
/// currently-started radio.
fn session_device(
    addr: std::net::IpAddr,
    last: &[DiscoveryInfo],
    rx_count: usize,
    sample_16bit: bool,
) -> DiscoveryInfo {
    // Try to find the matching entry in the last discover result by IP.
    let ip_bytes = if let std::net::IpAddr::V4(v4) = addr {
        v4.octets()
    } else {
        [0u8; 4]
    };
    let addr_str = format!("{}:{}", addr, hl2::protocol::HL2_PORT);
    if let Some(existing) = last.iter().find(|d| d.ip == ip_bytes.as_slice()) {
        return DiscoveryInfo {
            mac: existing.mac,
            gateware_major: existing.gateware_major,
            gateware_minor: existing.gateware_minor,
            board_id: existing.board_id,
            rx_count: rx_count as u8,
            is_sending: true,
            sample_16bit,
            ip: ip_bytes,
            addr: addr_str,
            in_service: true,
        };
    }
    DiscoveryInfo {
        mac: [0u8; 6],
        gateware_major: 0,
        gateware_minor: 0,
        board_id: hl2::protocol::BOARD_ID_HL2,
        rx_count: rx_count as u8,
        is_sending: true,
        sample_16bit,
        ip: ip_bytes,
        addr: addr_str,
        in_service: true,
    }
}

/// Union of `discovered` and the in-service devices
fn merge_devices(
    discovered: Vec<DiscoveryInfo>,
    in_service: Option<DiscoveryInfo>,
) -> Vec<DiscoveryInfo> {
    let mut out = discovered;
    if let Some(svc) = in_service {
        let ip = svc.ip;
        let exists = out.iter().any(|d| d.ip == ip);
        if !exists {
            out.push(svc);
        } else if let Some(ex) = out.iter_mut().find(|d| d.ip == ip) {
            ex.in_service = true;
            ex.is_sending = true;
        }
    }
    out
}

/// The in-service device snapshot from the hub (or `None` if no session is
/// started). The hub holds the `Session` in a tokio `Mutex<Option<Session>>`;
/// we grab a momentary read, clone the `DiscoveryInfo` out, and release.
async fn current_in_service(hub: &RadioHub) -> Option<DiscoveryInfo> {
    let guard = hub.session.lock().await;
    guard
        .as_ref()
        .and_then(|s| s.started.then(|| s.device.clone()))
}

/// Per-receiver DDC baseband rate the server programs at Start (Hz).
const VRX_SOURCE_RATE_HZ: u32 = 192_000;
//const VRX_SOURCE_RATE_HZ: u32 = 96_000;

/// Audio sample rate (Hz) the virtual receiver outputs for SSB voice.
const VRX_AUDIO_RATE_HZ: u32 = 4_800;

/// `hl2::receiver`'s fixed FT8 input rate (12 kHz). Aliased here so
/// `spawn_vrx` can set `audio.rate_hz` *and* `VrxState.rate_hz` from the
/// same source.
const VRX_FT8_RATE_HZ: u32 = hl2::receiver::FT8_SAMPLE_RATE_HZ;

/// Build a [`VrxTask`] for the given [`VrxCfg`] by wiring a
/// [`hl2::receiver::VirtualReceiver`] (reading the shared [`hl2::receiver::BasebandRing`])
/// onto a fresh [`BufSink`]
async fn spawn_vrx(
    ring: std::sync::Arc<std::sync::Mutex<hl2::receiver::baseband_ring::BasebandRing>>,
    fanout: std::sync::Arc<tokio::sync::broadcast::Sender<WsEvent>>,
    cfg: &VrxCfg,
    session: std::sync::Arc<Mutex<Option<Session>>>,
    pskrep: crate::pskrep_hook::SharedPsk,
) -> Result<VrxTask, String> {
    let (mode, rate_hz, ft8_dec, js8_dec, ft4_dec) = match cfg.mode {
        VrxMode::Usb => (
            Mode::Ssb(Sideband::Usb),
            VRX_AUDIO_RATE_HZ,
            None,
            None,
            None,
        ),
        VrxMode::Lsb => (
            Mode::Ssb(Sideband::Lsb),
            VRX_AUDIO_RATE_HZ,
            None,
            None,
            None,
        ),
        VrxMode::Am => (Mode::Am, VRX_AUDIO_RATE_HZ, None, None, None),
        // FM (standard) and NFM (narrow) are the same demod core,
        // distinguished only by the channel-select bandwidth that
        // `ReceiverConfig::bandwidth()` resolves (the UI's `bw_hz` wins —
        // typically 15 kHz for FM voice, 5 kHz for NFM voice).
        VrxMode::Fm => (Mode::Fm, VRX_AUDIO_RATE_HZ, None, None, None),
        VrxMode::FmNarrow => (Mode::FmNarrow, VRX_AUDIO_RATE_HZ, None, None, None),
        VrxMode::Ft8 => (Mode::Ft8, VRX_FT8_RATE_HZ, Some(shared()), None, None),
        VrxMode::Js8 => (Mode::Js8, VRX_FT8_RATE_HZ, None, Some(js8_shared()), None),
        VrxMode::Ft4 => (Mode::Ft4, VRX_FT8_RATE_HZ, None, None, Some(ft4_shared())),
    };
    let state = VrxState {
        slot: cfg.slot,
        offset_hz: cfg.offset_hz,
        mode: cfg.mode,
        bw_hz: cfg.bw_hz,
        gain_db: cfg.gain_db,
        rate_hz,
        muted: false,
    };
    let ft8_task = ft8_dec.as_ref().map(|dec| {
        spawn_ft8_decode(
            dec.clone(),
            fanout.clone(),
            session.clone(),
            cfg.slot,
            state,
            pskrep.clone(),
        )
    });
    let js8_task = js8_dec.as_ref().map(|dec| {
        spawn_js8_decode(
            dec.clone(),
            fanout.clone(),
            session.clone(),
            cfg.slot,
            state,
            pskrep.clone(),
        )
    });
    let ft4_task = ft4_dec.as_ref().map(|dec| {
        spawn_ft4_decode(
            dec.clone(),
            fanout.clone(),
            session,
            cfg.slot,
            state,
            pskrep,
        )
    });
    // The demod's pre-AGC `RawSampleTap` seam carries only the digital-mode
    // decoders (FT8/JS8/FT4). Voice modes (SSB/AM/FM) have no decoder → no
    // tap. The S-meter no longer hangs off this seam — it is a *consumer of
    // the band spectrum* (see `api/src/meter.rs` and `run_spectral`), which
    // is the one source that has a correct band-noise floor for every mode.
    let tap: Option<std::sync::Arc<dyn hl2::receiver::RawSampleTap>> =
        if let Some(dec) = ft8_dec.as_ref() {
            Some(std::sync::Arc::new(Ft8Tap::from_shared(dec.clone()))
                as std::sync::Arc<dyn hl2::receiver::RawSampleTap>)
        } else if let Some(dec) = js8_dec.as_ref() {
            Some(std::sync::Arc::new(Js8Tap::from_shared(dec.clone()))
                as std::sync::Arc<dyn hl2::receiver::RawSampleTap>)
        } else if let Some(dec) = ft4_dec.as_ref() {
            Some(std::sync::Arc::new(Ft4Tap::from_shared(dec.clone()))
                as std::sync::Arc<dyn hl2::receiver::RawSampleTap>)
        } else {
            None
        };
    let rx_cfg = ReceiverConfig {
        mode,
        source_rate_hz: VRX_SOURCE_RATE_HZ,
        // `cfg.offset_hz` is "target − NCO" (Hz), positive for an FT8 at 7.074
        // with NCO at 7.050. Pass the ***negated*** value as `source_center_hz`:
        // the demod's own unit test (`ft8_path_passes_signal_at_nonzero_nco_offset`)
        // fixes the DSP convention as "source_center_hz = the carrier's baseband
        // location", and the firmware's DDC puts an **above-NCO** target at a
        // *negative* baseband frequency (measured live via the `hl2 ft8 --probe`
        // diagnostic: NCO 7.060 / signal 7.074 → strongest complex-DFT bin at
        // −14 kHz, 9 dB above the +14 kHz bin; below-NCO is the mirror). So the
        // carrier's baseband location is `−(target − nco)`. This is the sign fix
        // for the non-zero-NCO FT8/JS8 bug: the old code passed `+offset_hz`.
        source_center_hz: -(cfg.offset_hz as f64),
        bandwidth_hz: Some(cfg.bw_hz.max(1)),
        audio: AudioConfig {
            rate_hz,
            gain_db: cfg.gain_db,
        },
        tap,
    };
    let (sink, buf) = BufSink::pair(hl2::receiver::BUF_SINK_DEFAULT_CAP);
    let rx = VirtualReceiver::new(rx_cfg, Box::new(sink))
        .map_err(|e| format!("vrx build failed: {e}"))?;

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop2 = stop.clone();
    let ring2 = ring.clone();
    let demod = std::thread::Builder::new()
        .name("vrx-demod".into())
        .spawn(move || {
            let mut rx = rx;
            let mut iq_buf: Vec<num_complex::Complex<f32>> = Vec::with_capacity(1024);
            let mut cursor: u64 = 0; // (first peek sees base_seq=0 → reads from head)
            loop {
                if stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = rx.flush();
                    return;
                }
                // Peek non-destructively from our own reader cursor; the
                // buffer is shared with every other receiver on this slot
                // (auto-decode decoders etc.), so reading does not remove
                // samples from their view. On 0 we either caught up (idle)
                // or fell behind (resync: jump to the head and re-derive
                // the AGC/DC-block over one frame).
                let got = {
                    let mut g = ring2.lock().unwrap();
                    let base = g.base_seq();
                    if cursor < base {
                        cursor = base;
                        0
                    } else {
                        g.peek(cursor, &mut iq_buf, 1024)
                    }
                };
                if got == 0 {
                    std::thread::sleep(std::time::Duration::from_micros(500));
                    continue;
                }
                cursor += got as u64;
                if let Err(e) = rx.process(&iq_buf) {
                    eprintln!("[HUB] vrx demod err: {e}");
                }
            }
        })
        .map_err(|e| format!("spawn vrx demod thread: {e}"))?;

    let fanout2 = fanout.clone();
    let buf2 = buf.clone();
    let muted = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let muted2 = muted.clone();
    let slot = cfg.slot;
    let rate_u16 = state.rate_hz as u16;
    let fanout_task = tokio::spawn(async move {
        let mut tick = tokio::time::interval(tokio::time::Duration::from_millis(24));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut seq: u32 = 0;
        loop {
            let _ = tick.tick().await;
            let n = buf2.len();
            if n == 0 {
                continue;
            }
            if muted2.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = buf2.drain(n);
                continue;
            }
            let max = n.min(480);
            let samples = buf2.drain(max);
            if samples.is_empty() {
                continue;
            }
            let frame = WsEvent::Audio {
                slot,
                seq,
                rate_hz: rate_u16,
                samples,
            };
            let _ = fanout2.send(frame);
            seq = seq.wrapping_add(1);
        }
    });

    Ok(VrxTask {
        state,
        buf,
        stop,
        muted,
        demod: Some(demod),
        fanout: Some(fanout_task),
        ft8: ft8_task,
        js8: js8_task,
        ft4: ft4_task,
    })
}

/// Build an [`AutoTask`] for the given (slot, auto_mode, target_freq) by
/// wiring a [`hl2::receiver::VirtualReceiver`] (reading the slot's
/// [`hl2::receiver::BasebandRing`]) to a [`hl2::receiver::DropSink`], with
/// the mode's decoder (FT8's wall-clock slot decoder / JS8's continuous
/// decoder) attached as a [`hl2::receiver::RawSampleTap`], and spawning
/// the matching decode thread from `spawn_ft8_decode` /
/// `spawn_js8_decode` — reusing the existing decoder + broadcast + PSK
/// paths exactly the way a [`VrxTask`] does, just without the `BufSink`
/// + WebSocket fan-out.
///
/// The synthetic `VrxState` that is stamped onto decoded rows is
/// *auto* — `muted: true` (so the UI can distinguish auto rows from
/// live vrx rows by `vrx.muted`), `offset_hz = freq − nco_hz` (the
/// NCO offset that produced the target), `mode = auto_mode`, `rate_hz =
/// 12_000` (fixed by the FT8/JS8 decoders), and a bandwidth matching
/// `hl2::receiver::Mode::default_bandwidth_hz()` for the auto mode.
/// `gain_db` is `0.0` — the AGC + peak-normalisation in the decoders
/// is what actually shapes the decode; the `gain_db` in `ReceiverConfig`
/// only affects the audio the demod emits, which is dropped here, so
/// keeping it at 0 is honest.
async fn spawn_auto(
    ring: std::sync::Arc<std::sync::Mutex<hl2::receiver::baseband_ring::BasebandRing>>,
    fanout: std::sync::Arc<tokio::sync::broadcast::Sender<WsEvent>>,
    session: std::sync::Arc<Mutex<Option<Session>>>,
    slot: u8,
    nco_hz: u32,
    freq_hz: u32,
    auto_mode: AutoMode,
    pskrep: crate::pskrep_hook::SharedPsk,
) -> Result<AutoTask, String> {
    // The concrete decoder + tap + spawner, keyed on the auto mode. Both
    // arms end up at 12 kHz output via `Mode::{Ft8,Js8}`.
    let (mode, tap, decode_task) = match auto_mode {
        AutoMode::Ft8 => {
            let dec = hl2::receiver::shared();
            let tap: std::sync::Arc<dyn hl2::receiver::RawSampleTap> =
                std::sync::Arc::new(Ft8Tap::from_shared(dec.clone()));
            (
                Mode::Ft8,
                tap,
                DecodeTaskOrNone::Ft8(spawn_ft8_decode(
                    dec,
                    fanout.clone(),
                    session.clone(),
                    slot,
                    synthetic_vrx_state(slot, nco_hz, freq_hz, auto_mode),
                    pskrep.clone(),
                )),
            )
        }
        AutoMode::Js8 => {
            let dec = hl2::receiver::js8_shared();
            let tap: std::sync::Arc<dyn hl2::receiver::RawSampleTap> =
                std::sync::Arc::new(Js8Tap::from_shared(dec.clone()));
            (
                Mode::Js8,
                tap,
                DecodeTaskOrNone::Js8(spawn_js8_decode(
                    dec,
                    fanout.clone(),
                    session.clone(),
                    slot,
                    synthetic_vrx_state(slot, nco_hz, freq_hz, auto_mode),
                    pskrep.clone(),
                )),
            )
        }
        AutoMode::Ft4 => {
            let dec = hl2::receiver::ft4_shared();
            let tap: std::sync::Arc<dyn hl2::receiver::RawSampleTap> =
                std::sync::Arc::new(Ft4Tap::from_shared(dec.clone()));
            (
                Mode::Ft4,
                tap,
                DecodeTaskOrNone::Ft4(spawn_ft4_decode(
                    dec,
                    fanout.clone(),
                    session.clone(),
                    slot,
                    synthetic_vrx_state(slot, nco_hz, freq_hz, auto_mode),
                    pskrep,
                )),
            )
        }
    };

    let rx_cfg = ReceiverConfig {
        mode,
        source_rate_hz: VRX_SOURCE_RATE_HZ,
        // Same sign fix as `spawn_vrx`: "above-NCO" targets sit at
        // **negative** baseband frequency per the firmware DDC, so
        // `source_center_hz = −(target − nco)`. The old `freq_hz − nco_hz`
        // (positive for above-NCO) was the non-zero-NCO FT8/JS8 bug.
        source_center_hz: (nco_hz as i64 - freq_hz as i64) as f64,
        bandwidth_hz: Some(mode.default_bandwidth_hz()),
        audio: hl2::receiver::AudioConfig {
            rate_hz: VRX_FT8_RATE_HZ,
            gain_db: 0.0,
        },
        tap: Some(tap),
    };
    let rx = hl2::receiver::VirtualReceiver::new(rx_cfg, Box::new(DropSink::new()))
        .map_err(|e| format!("auto vrx build failed: {e}"))?;

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop2 = stop.clone();
    let ring2 = ring.clone();
    let demod = std::thread::Builder::new()
        .name(format!(
            "auto-demod-rx{slot}-{}-{freq_hz}",
            auto_mode.wire()
        ))
        .spawn(move || {
            let mut rx = rx;
            let mut iq_buf: Vec<num_complex::Complex<f32>> = Vec::with_capacity(1024);
            let mut cursor: u64 = 0;
            loop {
                if stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    let _ = rx.flush();
                    return;
                }
                let got = {
                    let mut g = ring2.lock().unwrap();
                    let base = g.base_seq();
                    if cursor < base {
                        cursor = base;
                        0
                    } else {
                        g.peek(cursor, &mut iq_buf, 1024)
                    }
                };
                if got == 0 {
                    std::thread::sleep(std::time::Duration::from_micros(500));
                    continue;
                }
                cursor += got as u64;
                if let Err(e) = rx.process(&iq_buf) {
                    eprintln!("[HUB] auto demod err: {e}");
                }
            }
        })
        .map_err(|e| format!("spawn auto demod thread: {e}"))?;

    let decode = match decode_task {
        DecodeTaskOrNone::Ft8(t) => t,
        DecodeTaskOrNone::Js8(t) => t,
        DecodeTaskOrNone::Ft4(t) => t,
    };

    Ok(AutoTask {
        mode: auto_mode,
        freq_hz,
        stop,
        demod: Some(demod),
        decode,
    })
}

/// Helper enum to let `spawn_auto`'s single match arm bind to the right
/// `spawn_*_decode` function without leaking a `Box<dyn>` through two
/// branches (the concrete functions have different return types —
/// both `DecodeTask`, but the *shared-decoder* arguments are
/// `Ft8SharedDecoder` vs `Js8SharedDecoder`, so the arm body has to be
/// typed inside the branch).
enum DecodeTaskOrNone {
    Ft8(DecodeTask),
    Js8(DecodeTask),
    Ft4(DecodeTask),
}

/// The synthetic [`VrxState`] stamped onto auto-decode rows (see
/// `spawn_auto` doc for why). `muted: true` is the UI's "this was an
/// auto row" flag; everything else mirrors `auto_to_vrx_mode`'s output.
fn synthetic_vrx_state(slot: u8, nco_hz: u32, freq_hz: u32, auto_mode: AutoMode) -> VrxState {
    VrxState {
        slot,
        offset_hz: (freq_hz as i64 - nco_hz as i64) as i32,
        mode: auto_to_vrx_mode(auto_mode),
        bw_hz: 2_600,
        gain_db: 0.0,
        rate_hz: VRX_FT8_RATE_HZ,
        muted: true,
    }
}

/// Spawn the wall-clock-aligned FT8 slot-decode **thread**: every ≤ 1 s it
/// computes the most-recently-closed 15 s slot (`closed_slot_for(now_ms)`)
/// and, if it has not already been decoded, runs the `mfsk-core` batch
/// decode (the demod thread's critical section is bounded by
/// `Ft8Tap::append` and never includes a decode). On success (≥ 1 row) it
/// broadcasts a `WsEvent::Log`.
///
/// Same dedicated-standard-thread shape as the JS8 decode thread
/// ([`spawn_js8_decode`]): all mode decodes live on their own thread with a
/// generous stack, so none of them can starve (or overflow) a Rocket worker
/// that is also delivering the WS fan-out. `AtomicBool` stop flag checked
/// between 1 s ticks, `thread::join` on `VrxTask::stop`. `broadcast::Sender
/// ::send` and the mutexes touched here are all std (non-blocking /
/// bounded), so no runtime is needed on this thread.
fn spawn_ft8_decode(
    shared: hl2::receiver::SharedDecoder,
    fanout: std::sync::Arc<tokio::sync::broadcast::Sender<WsEvent>>,
    session: std::sync::Arc<Mutex<Option<Session>>>,
    vrx_slot: u8,
    vrx: VrxState,
    pskrep: crate::pskrep_hook::SharedPsk,
) -> DecodeTask {
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop2 = stop.clone();
    let slot_name = vrx_slot;
    let th = std::thread::Builder::new()
        .name(format!("vrx-ft8-decode-rx{slot_name}"))
        .stack_size(MODE_DECODE_THREAD_STACK)
        .spawn(move || {
            let mut last_decoded_slot = 0u64;
            let mut last_tick = std::time::Instant::now() - std::time::Duration::from_secs(1);
            loop {
                if stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                // Park ~1 s between wall-clock checks (250 ms resolution so
                // `stop` joins quickly — the worst-case join while parked is
                // one 250 ms sleep + the top-of-loop check).
                let now = std::time::Instant::now();
                if now - last_tick < std::time::Duration::from_secs(1) {
                    std::thread::sleep(std::time::Duration::from_millis(250));
                    continue;
                }
                last_tick = now;

                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                let slot = closed_slot_for(now_ms);
                if slot == last_decoded_slot {
                    continue;
                }
                // Defensive: if stop was set between the park and here (a
                // teardown racing the 1 s tick), skip the burst. The burst
                // (mfsk-core's `req.decode()`) is 1-3 s of CPU and cannot be
                // interrupted cheaply, so catching stop *before* we enter it
                // is the only cheap way to keep the join bounded.
                if stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                let res = decode_closed_slot(&shared, slot);
                if res.is_ok() {
                    last_decoded_slot = slot;
                }
                match res {
                    Ok(msgs) if !msgs.is_empty() => {
                        let n = msgs.len();
                        if std::env::var("HL2_DEBUG").is_ok() {
                            eprintln!("[HUB] ft8 slot {slot} decoded {n} row(s)");
                        }
                        let rf_hz = rf_for_vrx(&session, vrx_slot, vrx.offset_hz);
                        let (log, spotted) = on_decodes(&msgs, vrx, rf_hz, &pskrep);
                        let _ = fanout.send(WsEvent::Log(log));
                        if spotted > 0 && std::env::var("HL2_DEBUG").is_ok() {
                            eprintln!("[HUB] ft8 slot {slot}: {spotted} spot(s) queued");
                        }
                    }
                    Ok(_) => {} // silence / no CRC-pass hit — no broadcast.
                    Err(e) => {
                        if std::env::var("HL2_DEBUG").is_ok() {
                            eprintln!("[HUB] ft8 decode err: {e}");
                        }
                    }
                }
            }
        })
        .expect("spawn ft8 decode thread");
    DecodeTask {
        stop,
        thread: Some(th),
    }
}

/// The shared decode tail (all digital modes): build the mode-agnostic
/// [`hl2_common::DecodeLog`] by reading each message through the
/// [`hl2_common::DecodedMessage`] trait, spot each qualifying message into
/// the PSK Reporter sink, and return the log (to broadcast) + the number of
/// spots queued.
///
/// `rf_hz` is the passband centre the rows were decoded against
/// (`0` = untun­ed — no spots are posted). This is the *single* place the
/// API branches on "was this spot-able" — everything mode-specific (the
/// call/locator grammar, self-spot suppression) already lives inside each
/// mode's `DecodedMessage::spot_fields` impl.
fn on_decodes<T: hl2_common::DecodedMessage>(
    msgs: &[T],
    vrx: VrxState,
    rf_hz: u32,
    pskrep: &crate::pskrep_hook::SharedPsk,
) -> (hl2_common::DecodeLog, usize) {
    let log = hl2_common::DecodeLog {
        decodes: msgs.iter().map(|m| decode_row(m, vrx)).collect(),
    };
    let mut spotted = 0usize;
    if rf_hz != 0 {
        let mut g = pskrep.lock().unwrap();
        for m in msgs {
            if g.spot(m, rf_hz) {
                spotted += 1;
            }
        }
    }
    (log, spotted)
}

/// The passband centre an vrx's rows were decoded against: the slot's NCO
/// plus the receiver's own offset. `0` when the slot is untuned (the sink
/// then posts no spots).
fn rf_for_vrx(session: &std::sync::Arc<Mutex<Option<Session>>>, slot: u8, offset_hz: i32) -> u32 {
    let tune_hz = match session.try_lock() {
        Ok(g) => g
            .as_ref()
            .and_then(|s| s.tuning.get(&slot).copied())
            .unwrap_or(0),
        Err(_) => 0,
    };
    if tune_hz == 0 {
        0
    } else {
        ((tune_hz as i64) + offset_hz as i64).max(0) as u32
    }
}

/// Map one decoded message (any digital mode) to its wire
/// [`hl2_common::DecodeRow`], stamping the producing receiver's
/// [`VrxState`] (slot, NCO offset, mode, bandwidth, gain) onto the row so
/// the UI can attribute the decode to the exact receiver — not just the
/// slot. Reads everything through the [`hl2_common::DecodedMessage`] trait,
/// so a new mode needs no API-layer change here.
fn decode_row<T: hl2_common::DecodedMessage>(m: &T, vrx: VrxState) -> hl2_common::DecodeRow {
    hl2_common::DecodeRow {
        text: m.display().to_string(),
        freq_hz: m.freq_hz(),
        dt_sec: m.dt_sec(),
        snr_db: m.snr_db(),
        slot_ms: m.slot_ms(),
        vrx,
    }
}

/// Spawn the wall-clock-aligned FT4 slot-decode **thread**: every ≤ 1 s it
/// computes the most-recently-closed 7.5 s slot (`closed_slot_for(now_ms)`)
/// and, if it has not already been decoded, runs the `mfsk-core` batch
/// decode (the demod thread's critical section is bounded by
/// `Ft4Tap::append` and never includes a decode). On success (≥ 1 row) it
/// broadcasts a `WsEvent::Ft4Log`.
///
/// Same dedicated-standard-thread shape as the FT8/JS8 decode threads: all
/// mode decodes live on their own thread with a generous stack, so none of
/// them can starve (or overflow) a Rocket worker that is also delivering
/// the WS fan-out. `AtomicBool` stop flag checked between 1 s ticks,
/// `thread::join` on `VrxTask::stop`.
fn spawn_ft4_decode(
    shared: hl2::receiver::Ft4SharedDecoder,
    fanout: std::sync::Arc<tokio::sync::broadcast::Sender<WsEvent>>,
    session: std::sync::Arc<Mutex<Option<Session>>>,
    vrx_slot: u8,
    vrx: VrxState,
    pskrep: crate::pskrep_hook::SharedPsk,
) -> DecodeTask {
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop2 = stop.clone();
    let slot_name = vrx_slot;
    let th = std::thread::Builder::new()
        .name(format!("vrx-ft4-decode-rx{slot_name}"))
        .stack_size(MODE_DECODE_THREAD_STACK)
        .spawn(move || {
            let mut last_decoded_slot = 0u64;
            let mut last_tick = std::time::Instant::now() - std::time::Duration::from_secs(1);
            loop {
                if stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                // Park ~1 s between wall-clock checks (250 ms resolution so
                // `stop` joins quickly).
                let now = std::time::Instant::now();
                if now - last_tick < std::time::Duration::from_secs(1) {
                    std::thread::sleep(std::time::Duration::from_millis(250));
                    continue;
                }
                last_tick = now;

                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                let slot = ft4_closed_slot_for(now_ms);
                if slot == last_decoded_slot {
                    continue;
                }
                // Defensive: skip the burst if stop was set since the park
                // (see the FT8 loop for the full rationale) so the join stays
                // bounded to one park slice, not one burst.
                if stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                let res = ft4_decode_closed_slot(&shared, slot);
                if res.is_ok() {
                    last_decoded_slot = slot;
                }
                match res {
                    Ok(msgs) if !msgs.is_empty() => {
                        let n = msgs.len();
                        if std::env::var("HL2_DEBUG").is_ok() {
                            eprintln!("[HUB] ft4 slot {slot} decoded {n} row(s)");
                        }
                        let rf_hz = rf_for_vrx(&session, vrx_slot, vrx.offset_hz);
                        let (log, spotted) = on_decodes(&msgs, vrx, rf_hz, &pskrep);
                        let _ = fanout.send(WsEvent::Log(log));
                        if spotted > 0 && std::env::var("HL2_DEBUG").is_ok() {
                            eprintln!("[HUB] ft4 slot {slot}: {spotted} spot(s) queued");
                        }
                    }
                    Ok(_) => {} // silence / no CRC-pass hit — no broadcast.
                    Err(e) => {
                        if std::env::var("HL2_DEBUG").is_ok() {
                            eprintln!("[HUB] ft4 decode err: {e}");
                        }
                    }
                }
            }
        })
        .expect("spawn ft4 decode thread");
    DecodeTask {
        stop,
        thread: Some(th),
    }
}

/// Spawn the continuous JS8 decode **thread**: every 500 ms it calls
/// [`js8_step`], which — per the js8call `decodeEnqueueReadyExperiment`
/// rules — snapshots and decodes any of the four JS8 speeds (A/B/C/E)
/// whose cycle data is (now or already) fully in the rolling buffer since
/// the last re-arm (~1.5 s). Duplicate frames (same call + grid at the
/// same frequency, within 45 s) are suppressed by the decoder before we
/// see them, so a station that keeps transmitting is not re-logged every
/// 1.5 s: the UI sees one row per distinct signal, the way the reference
/// merges `decodes` into a single entry.
///
/// A dedicated std thread, not a tokio task: the decode burst is heavy
/// (FFT temporaries + multi-pass decode across up to four speeds) and
/// must not share a Rocket worker with the WS fan-out path — that is what
/// stalled the UI's spectrum/audio on a hot decode and overflowed a
/// worker's stack once. Same shape as the demod thread: `AtomicBool` stop
/// flag checked between ticks, `thread::join` on `VrxTask::stop`.
/// `broadcast::Sender::send` and the mutexes touched here are all std
/// (non-blocking / bounded), so no runtime is needed.
fn spawn_js8_decode(
    shared: Js8SharedDecoder,
    fanout: std::sync::Arc<tokio::sync::broadcast::Sender<WsEvent>>,
    session: std::sync::Arc<Mutex<Option<Session>>>,
    vrx_slot: u8,
    vrx: VrxState,
    pskrep: crate::pskrep_hook::SharedPsk,
) -> DecodeTask {
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop2 = stop.clone();
    let slot_name = vrx_slot;
    let th = std::thread::Builder::new()
        .name(format!("vrx-js8-decode-rx{slot_name}"))
        .stack_size(MODE_DECODE_THREAD_STACK)
        .spawn(move || {
            loop {
                if stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                // Tick every 500 ms: the decoder re-arms each speed
                // independently every ~1.5 s, so sub-1 s ticks only add
                // CPU without adding decodes; 500 ms still lets `stop`
                // join quickly (we sleep in 100 ms slices).
                for _ in 0..5 {
                    if stop2.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }

                // Defensive: skip the burst if stop was set since the last
                // slice check — see the FT8/FT4 loops for the full
                // rationale. `js8_step` is lighter than the FT8/FT4 bursts
                // but can still take a few hundred ms for four speeds.
                if stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                match js8_step(&shared, now_ms) {
                    Ok(msgs) if !msgs.is_empty() => {
                        let n = msgs.len();
                        if std::env::var("HL2_DEBUG").is_ok() {
                            eprintln!("[HUB] js8 step decoded {n} frame(s)");
                        }
                        let rf_hz = rf_for_vrx(&session, vrx_slot, vrx.offset_hz);
                        let (log, spotted) = on_decodes(&msgs, vrx, rf_hz, &pskrep);
                        if spotted > 0 && std::env::var("HL2_DEBUG").is_ok() {
                            eprintln!("[HUB] js8: {spotted} spot(s) queued");
                        }
                        // Fire-and-forget broadcast: if all websockets are
                        // gone (or lagged) the send returns `Err`, and
                        // that's fine — dedup / re-arm is in the decoder,
                        // not in us, so no retry.
                        let _ = fanout.send(WsEvent::Log(log));
                    }
                    Ok(_) => {}
                    Err(e) => {
                        if std::env::var("HL2_DEBUG").is_ok() {
                            eprintln!("[HUB] js8 decode err: {e}");
                        }
                    }
                }
            }
        })
        .expect("spawn js8 decode thread");
    DecodeTask {
        stop,
        thread: Some(th),
    }
}

/// The spectrum pipeline. Reads `Hl2Event`s from the Hl2 pump; accumulates
/// raw samples into one FFT window, computes the magnitude spectrum,
/// max-pools into `cfg.wideband_bins` linear bins. The most-recently-computed
/// frame is held in `latest` and re-broadcast to every subscriber at most
/// once every `BROADCAST_INTERVAL_MS`
///
/// Exits when the pump's channel closes.
const BROADCAST_INTERVAL_MS: u64 = 33; // ~30 fps
async fn run_spectral(
    cfg: HubConfig,
    mut ev_rx: mpsc::UnboundedReceiver<Hl2Event>,
    fanout: std::sync::Arc<tokio::sync::broadcast::Sender<WsEvent>>,
    spectrum_source: std::sync::Arc<std::sync::Mutex<SpectrumSource>>,
    spectrum_rev: std::sync::Arc<std::sync::atomic::AtomicU64>,
    meter: std::sync::Arc<crate::meter::MeterState>,
    passband_bw: std::sync::Arc<std::sync::atomic::AtomicU64>,
    passband_mode: std::sync::Arc<std::sync::atomic::AtomicU32>,
) {
    let n_fft = cfg.accumulate_blocks * IQ_PAIRS_PER_BLOCK;
    use num_complex::Complex;

    // Hann window over one FFT window (identical for both sources since the
    // window length is fixed at `n_fft`).
    let mut window = vec![0.0f32; n_fft];
    for (i, v) in window.iter_mut().enumerate() {
        *v = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos());
    }

    // Accumulator of complex I/Q samples (re,im pairs) ready for one FFT.
    let mut acc: Vec<Complex<f32>> = Vec::with_capacity(n_fft);
    let mut buf: Vec<Complex<f32>> = Vec::with_capacity(n_fft);
    let mut mags: Vec<u16> = Vec::with_capacity(cfg.wideband_bins);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    // The `spectrum_rev` value we last saw; if it has a different value when
    // we're about to accept a frame, the accumulator may contain samples from
    // the previous (now-mismatched) source and must be discarded.
    let mut seen_src_rev = spectrum_rev.load(std::sync::atomic::Ordering::Relaxed);
    // Monotonic counter of completed FFT frames, used as the frame sequence
    // tag (the EP6 baseband pump exposes no per-chunk seq, so we synthesize
    // one).
    let mut frame_seq: u32 = 0;

    // Latest-computed frame, awaited by the broadcast tick below.
    let mut latest: Option<(u32, Vec<u16>)> = None;

    let mut tick = tokio::time::interval(tokio::time::Duration::from_millis(BROADCAST_INTERVAL_MS));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    // Push one accumulated value (either the real value for EP4 or the complex
    // I/Q pair for EP6) into the accumulator, applying the Hann taper.
    let _ = &window;
    macro_rules! push_iq {
        ($re:expr, $im:expr) => {{
            // We flush the moment `acc.len() == n_fft`, so no sample is ever
            // seen past position `n_fft-1` while `acc.len() < n_fft`.
            let idx = acc.len() % n_fft;
            acc.push(Complex::new($re * window[idx], $im * window[idx]));
        }};
    }

    macro_rules! finish_frame {
        () => {{
            let n = n_fft.min(acc.len());
            buf.clear();
            buf.extend_from_slice(&acc[..n]);
            acc.drain(..n);

            if n > 0 {
                let mut re_sum = 0.0f32;
                let mut im_sum = 0.0f32;
                for x in buf.iter() {
                    re_sum += x.re;
                    im_sum += x.im;
                }
                let re_mean = re_sum / n as f32;
                let im_mean = im_sum / n as f32;
                for x in buf.iter_mut() {
                    x.re -= re_mean;
                    x.im -= im_mean;
                }
            }

            fft.process(buf.as_mut_slice());

            display_mags_into(&buf, cfg.wideband_bins, &mut mags);

            let seq = frame_seq;
            frame_seq = frame_seq.wrapping_add(1);
            latest = Some((seq, mags.clone()));
        }};
    }

    'outer: loop {
        tokio::select! {
            biased; // Keep up with the pump first; broadcast is best-effort.

            res = ev_rx.recv() => {
                let Some(ev) = res else { break 'outer };

                let cur_src = spectrum_source.lock().unwrap().clone();
                let cur_rev = spectrum_rev.load(std::sync::atomic::Ordering::Relaxed);
                if cur_rev != seen_src_rev {
                    acc.clear();
                    seen_src_rev = cur_rev;
                }

                match ev {
                    Hl2Event::Block(b) if matches!(cur_src, SpectrumSource::Ep4) => {
                        for &s in &b.samples {
                            let v = s as f32 / 32768.0;
                            push_iq!(v, 0.0);
                        }
                        if acc.len() >= n_fft {
                            finish_frame!();
                        }
                    }
                    Hl2Event::Baseband(bc) => {
                        if let SpectrumSource::Ep6 { slot } = &cur_src {
                            if let Some(iq) = bc.per_rx.get(slot.wrapping_sub(1) as usize) {
                                for &c in iq {
                                    push_iq!(c.re, c.im);
                                }
                                if acc.len() >= n_fft {
                                    finish_frame!();
                                }
                            }
                        }
                    }
                    Hl2Event::CmdAck { ack, raddr, data, .. } => {
                        eprintln!("EP6 ack={} raddr=0x{:02x} data=0x{:08x}", ack, raddr, data);
                    }
                    _ => {}
                }
            }

            _ = tick.tick() => {
                if let Some((seq, mags)) = latest.take() {
                    // S-meter: for an EP6 (per-slot) display, the band
                    // spectrum's *quietest bins* are the noise floor and the
                    // *passband peak* is the signal — both mode-agnostic
                    // (see `crate::meter::compute_s_meter`). The passband
                    // window is the running receiver's channel bandwidth
                    // mapped to display bins.
                    {
                        let cur_src = spectrum_source.lock().unwrap().clone();
                        if let SpectrumSource::Ep6 { slot } = cur_src {
                            let bw_hz = passband_bw.load(std::sync::atomic::Ordering::Relaxed);
                            if bw_hz > 0 {
                                // Each display bin spans `(span / bins)` Hz
                                // (full-bandwidth baseband, `span` ≈ source
                                // rate). The passband in display bins =
                                // bw / bin_span.
                                let span_hz = spectrum_span(&SpectrumSource::Ep6 { slot }) as f64;
                                let bin_span = span_hz / mags.len() as f64;
                                let width = if bin_span > 0.0 {
                                    ((bw_hz as f64 / bin_span) as usize).max(1)
                                } else {
                                    1
                                };
                                let shape = crate::meter::PassbandShape::from_u32(
                                    passband_mode.load(std::sync::atomic::Ordering::Relaxed),
                                );
                                let (lev, fl) =
                                    crate::meter::compute_s_meter(&mags, width, shape, 25);
                                meter.set(slot, lev, fl);
                            } else {
                                meter.set(0, -120.0, -120.0);
                            }
                        } else {
                            meter.set(0, -120.0, -120.0);
                        }
                    }
                    let _ = fanout.send(WsEvent::Wideband { seq, mags });
                }
            }
        }
    }
}
