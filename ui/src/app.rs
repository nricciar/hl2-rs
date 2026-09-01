use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use web_sys::{HtmlInputElement, HtmlSelectElement};
use yew::prelude::*;

use crate::audio::Audio;
use crate::canvas;
use crate::client::{WsClient, parse_audio, parse_binary};
use hl2_common::{CH_AUDIO, CH_WIDEBAND};

/// Maximum rows kept in the FT8 decode log (newest-first, oldest dropped).
const FT8_LOG_CAP: usize = 200;

fn ws_url() -> String {
    let w = web_sys::window().unwrap();
    let l = w.location();
    let proto = if l
        .protocol()
        .as_deref()
        .unwrap_or("http://")
        .starts_with("https")
    {
        "wss"
    } else {
        "ws"
    };
    let host = l.host().unwrap_or_else(|_| "localhost:8080".into());
    format!("{proto}://{host}/api/ws")
}

struct Shared {
    client: RefCell<Option<WsClient>>,
    status: RefCell<String>,
    started: RefCell<bool>,
    latest: RefCell<Vec<u16>>,
    history: RefCell<Vec<Vec<u16>>>,
    /// Currently set NCO frequency (Hz) per RX slot, mirrored from
    /// `SharedState.tuning`. A missing slot = not yet tuned.
    tuning: RefCell<std::collections::BTreeMap<u8, u32>>,
    lna: RefCell<i8>,
    /// RX open-collector filter-bank relay mask (LSB-first: bit 0 = relay/F1 …
    /// bit 6 = relay/F7).
    oc_bits: RefCell<u8>,
    floor: RefCell<f64>,
    ceil: RefCell<f64>,
    /// `true` = the panadapter/waterfall dB scale is tracked automatically
    /// from the live spectrum (see [`Shared::update_auto_scale`]); the
    /// Floor/Ceil inputs are then read-only and show the current computed
    /// range.
    floor_auto: RefCell<bool>,
    auto_floor: RefCell<f64>,
    auto_ceil: RefCell<f64>,
    ceil_age: RefCell<u32>,
    auto_recenter: RefCell<bool>,
    force: RefCell<Option<UseForceUpdateHandle>>,
    timers: RefCell<Vec<Closure<dyn FnMut()>>>,
    rearming: RefCell<bool>,
    devices: RefCell<Vec<hl2_common::DiscoveryInfo>>,
    active_device: RefCell<Option<String>>,
    last_state_at: RefCell<u64>,
    /// User's spectrum source. `None` means "EP4 — wideband". `Some(slot)`
    /// means "EP6 slot N".
    spectrum_source: RefCell<Option<u8>>,
    /// NCO the current spectrum source is centred on (Hz)
    spectrum_center_hz: RefCell<Option<u32>>,
    /// Displayed bandwidth (Hz) of the current spectrum source
    spectrum_span_hz: RefCell<u32>,
    /// The Web Audio playback engine for the virtual receiver (`CH_AUDIO`).
    /// The client enforces one-audible-at-a-time by muting the other slots
    /// on the server, so at most one slot's audio streams in at a time.
    audio: Audio,
    /// The RX slot (1-based) the virtual-receiver panel is currently targeting
    /// Defaults to RX1.
    vrx_slot: RefCell<u8>,
    /// Active virtual receivers
    vrx: RefCell<std::collections::BTreeMap<u8, hl2_common::VrxState>>,
    /// Auto-decode monitors (headless), keyed by RX slot, mirrored from
    /// `SharedState.auto_monitors`. Presence of a slot = auto decode is on
    /// for that slot; the `Vec` carries the (mode, target-frequency) readout.
    auto_monitors: RefCell<std::collections::BTreeMap<u8, Vec<hl2_common::AutoMonitor>>>,
    vrx_sideband: RefCell<String>,
    vrx_bw_hz: RefCell<u32>,
    vrx_gain_db: RefCell<f32>,
    /// Decoded digital-mode rows (FT8 / FT4 / JS8), newest-first, capped
    /// (see [`FT8_LOG_CAP`]). All modes share one buffer + one wire
    /// envelope; the mode of each row is read from its `vrx` snapshot.
    decode_log: RefCell<Vec<hl2_common::DecodeRow>>,
    vrx_play_gain: RefCell<f32>,
    started_prev: RefCell<bool>,
    started_local: RefCell<bool>,
}

impl Shared {
    fn new() -> Rc<Self> {
        Rc::new(Self {
            client: RefCell::new(None),
            status: RefCell::new("connecting…".into()),
            started: RefCell::new(false),
            started_prev: RefCell::new(false),
            started_local: RefCell::new(false),
            latest: RefCell::new(Vec::new()),
            history: RefCell::new(Vec::new()),
            tuning: RefCell::new(std::collections::BTreeMap::new()),
            lna: RefCell::new(6),
            oc_bits: RefCell::new(0),
            floor: RefCell::new(-85.0),
            ceil: RefCell::new(-15.0),
            floor_auto: RefCell::new(true),
            auto_floor: RefCell::new(-85.0),
            auto_ceil: RefCell::new(-15.0),
            ceil_age: RefCell::new(0),
            auto_recenter: RefCell::new(false),
            force: RefCell::new(None),
            timers: RefCell::new(Vec::new()),
            rearming: RefCell::new(false),
            devices: RefCell::new(Vec::new()),
            active_device: RefCell::new(None),
            last_state_at: RefCell::new(0),
            spectrum_source: RefCell::new(Some(1)),
            spectrum_center_hz: RefCell::new(None),
            spectrum_span_hz: RefCell::new(0),
            audio: Audio::new(),
            vrx_slot: RefCell::new(1),
            vrx: RefCell::new(std::collections::BTreeMap::new()),
            auto_monitors: RefCell::new(std::collections::BTreeMap::new()),
            vrx_sideband: RefCell::new("usb".into()),
            vrx_bw_hz: RefCell::new(2_600),
            vrx_gain_db: RefCell::new(0.0),
            decode_log: RefCell::new(Vec::new()),
            vrx_play_gain: RefCell::new(0.8),
        })
    }

    fn notify(&self) {
        if let Some(f) = self.force.borrow().as_ref() {
            f.force_update();
        }
    }

    fn on_text(sh: &Rc<Self>, json: &str) {
        // One shared decode-log envelope for every digital mode (FT8 / FT4 /
        // JS8): `{"cmd":"log","data":[{…}, …]}`. The mode of each row is
        // carried in its `vrx` snapshot, so the client never branches on
        // mode to receive a decode.
        if json.trim_start().contains("\"cmd\":\"log\"")
            || json.trim_start().contains("\"cmd\": \"log\"")
        {
            #[derive(serde::Deserialize)]
            struct Envelope {
                data: Vec<hl2_common::DecodeRow>,
            }
            if let Ok(env) = serde_json::from_str::<Envelope>(json) {
                let mut log = sh.decode_log.borrow_mut();
                log.extend(env.data);
                if log.len() > FT8_LOG_CAP {
                    let excess = log.len() - FT8_LOG_CAP;
                    log.drain(..excess);
                }
            }
            sh.notify();
            return;
        }

        let resp: hl2_common::ServerResponse = match serde_json::from_str(json) {
            Ok(r) => r,
            Err(_) => return,
        };

        // Apply the embedded state only if it isn't stale.
        let fresh = resp.state.state_at >= *sh.last_state_at.borrow();
        if fresh {
            *sh.last_state_at.borrow_mut() = resp.state.state_at;

            // Apply the default RX1 tune *only once* per user-initiated start:
            //
            // 1. The local user just toggled the On switch (so `started_local`
            //    is true — set in the checkbox handler). This excludes a
            //    passive connect where the radio is already running from
            //    another tab and its welcome would otherwise clobber the
            //    user's existing RX1 frequency.
            // 2. The `started` field just transitioned false→true (an edge,
            //    not a level). This excludes repeat acks and is only safe to
            //    `tune` in because the server has already installed the
            //    session and completed the START handshake. A mid-run
            //    reconnect also has `started_prev==true`, so it's a no-op —
            //    exactly the "if already in use we don't change it" rule.
            // 3. The server's authoritative tuning map does not yet have an
            //    RX1 entry. If the user had already picked an RX1 frequency
            //    before the last stop/restart cycle, we preserve it rather
            //    than stomping on it with the default.
            //
            // Once fired, `started_local` is cleared so subsequent fresh
            // starts of the same session (welcome, other commands) don't
            // re-issue it.
            let local_start = *sh.started_local.borrow();
            let prev_started = *sh.started_prev.borrow();
            let now_started = resp.state.started;
            let edge = !prev_started && now_started && local_start;
            *sh.started_prev.borrow_mut() = now_started;
            *sh.started.borrow_mut() = now_started;
            if edge {
                *sh.started_local.borrow_mut() = false;
                let already_set = resp.state.tuning.get(&1u8).copied().unwrap_or(0) != 0;
                if !already_set {
                    // Default start frequency: RX1 at 14.074 MHz. Server-side
                    // this registers the RX1 slot in the fanout
                    sh.tuning.borrow_mut().insert(1, 14_074_000);
                    Shared::send_tune(sh, 1, 14_074_000);
                }
            }

            *sh.lna.borrow_mut() = resp.state.lna_gain_db;
            *sh.oc_bits.borrow_mut() = resp.state.oc_bits;
            // Mirror the per-slot NCO map so the frequency heading can show
            // the currently selected source's tune (see `freq_for`).
            *sh.tuning.borrow_mut() = resp.state.tuning.clone();
            // `spectrum_source: Ep4` → None; `Ep6 { slot }` → Some(slot).
            // A source change is a different magnitude regime (EP4 raw wideband
            // vs a DDC'd EP6 slot, or another slot) — recompute the auto scale
            // from scratch
            let prev_source = *sh.spectrum_source.borrow();
            let next_source: Option<u8> = match &resp.state.spectrum_source {
                hl2_common::SpectrumSource::Ep4 => {
                    *sh.spectrum_source.borrow_mut() = None;
                    None
                }
                hl2_common::SpectrumSource::Ep6 { slot } => {
                    *sh.spectrum_source.borrow_mut() = Some(*slot);
                    Some(*slot)
                }
            };
            if *sh.floor_auto.borrow() && next_source != prev_source {
                *sh.auto_recenter.borrow_mut() = true;
            }
            *sh.spectrum_center_hz.borrow_mut() = resp.state.spectrum_center_hz;
            *sh.spectrum_span_hz.borrow_mut() = resp.state.spectrum_span_hz;
            // Mirror the per-slot virtual receivers (audio) so the control
            // panel + spectrum passband overlays track every client. Each
            // entry is a full `VrxState` (slot / offset / mode / bw / gain /
            // rate / muted), keyed by RX slot.
            *sh.vrx.borrow_mut() = resp.state.vrx.clone();
            // Mirror the per-slot auto-decode monitors (headless digital-mode
            // decoders the server is running against known band frequencies).
            *sh.auto_monitors.borrow_mut() = resp.state.auto_monitors.clone();
            // Re-anchor the panel editor to the targeted slot: if it has an
            // active receiver pick up its settings, otherwise fall back to
            // the band defaults (so the panel always shows a usable state).
            Shared::refocus_vrx_panel(sh);
        }
        if let Some(devs) = &resp.devices {
            let ips: Vec<String> = devs
                .iter()
                .map(|d| {
                    d.ip.iter()
                        .map(|o| o.to_string())
                        .collect::<Vec<_>>()
                        .join(".")
                })
                .collect();
            // Keep the current selection if it is still present; otherwise
            // default to the first discovered radio.
            if !ips.contains(&sh.active_device.borrow().clone().unwrap_or_default()) {
                *sh.active_device.borrow_mut() = ips.into_iter().next();
            }
            *sh.devices.borrow_mut() = devs.clone();
        }

        if let Some(err) = &resp.error {
            let new = format!("error: {}", err);
            let changed = *sh.status.borrow() != new;
            *sh.status.borrow_mut() = new;
            if changed {
                sh.notify();
            }
        } else if resp.ack {
            let new = "connected".to_string();
            let changed = *sh.status.borrow() != new;
            *sh.status.borrow_mut() = new;
            if changed {
                sh.notify();
            }
        }
        sh.notify();
    }

    fn on_binary(sh: &Rc<Self>, bytes: &[u8]) {
        if bytes.len() < 2 {
            return;
        }
        let ch = u16::from_le_bytes([bytes[0], bytes[1]]);
        match ch {
            CH_WIDEBAND => {
                let (ch2, mags) = match parse_binary(bytes) {
                    Ok(v) => v,
                    Err(_) => return,
                };
                if ch2 != CH_WIDEBAND {
                    return;
                }
                *sh.latest.borrow_mut() = mags.clone();
                // Track the display scale from this frame (auto mode only).
                // Runs before `mags` is moved into the history below.
                // Re-render only if the scale actually moved, so the floor/ceil
                // readout stays live without churn while the scale is stable.
                let scale_moved = Shared::update_auto_scale(sh, &mags);
                {
                    let mut hist = sh.history.borrow_mut();
                    hist.push(mags);
                    if hist.len() > 256 {
                        let drop = hist.len() - 256;
                        hist.drain(..drop);
                    }
                }
                if scale_moved {
                    sh.notify();
                }
                draw_all(sh);
            }
            CH_AUDIO => {
                let (slot, _seq, rate_hz, samples) = match parse_audio(bytes) {
                    Some(v) => v,
                    None => return,
                };
                if samples.is_empty() {
                    return;
                }
                // Only feed audio the operator is currently *listening* on:
                // the targeted slot's receiver — and only while that receiver
                // is active and unmuted on the server (the server stops
                // streaming a muted slot, but a stale frame from before the
                // mute is discarded here too). A single shared `Audio` engine
                // plays the routed block; the master gain honours the
                // user's slider and the mute state.
                let target = *sh.vrx_slot.borrow();
                let is_listening = {
                    let vrx = sh.vrx.borrow();
                    match vrx.get(&target) {
                        Some(v) => v.slot as u16 == slot && !v.muted,
                        None => false,
                    }
                };
                if !is_listening {
                    return;
                }
                // Play the block. The `Audio` engine lazily creates the
                // `AudioContext` on first use and chains source nodes so the
                // stream is continuous across 25 ms server ticks. The master
                // gain is set from the user's slider (see `vrx_play_gain`) —
                // applied on connect and on every slider move.
                sh.audio.set_gain(*sh.vrx_play_gain.borrow());
                sh.audio.push_samples(&samples, rate_hz);
            }
            _ => {}
        }
    }

    fn on_open(sh: &Rc<Self>) {
        // A fresh connection means a fresh snapshot sequence: on reconnect the
        // server may have restarted (its `state_at` would have reset to 0), so
        // clear our high-water mark to avoid discarding the welcome state.
        *sh.last_state_at.borrow_mut() = 0;
        let was_reconnecting = {
            let s = sh.status.borrow();
            s.contains("reconnecting") || s.contains("reconnected")
        };
        let new = if was_reconnecting {
            "reconnected".to_string()
        } else {
            "connected".to_string()
        };
        let changed = *sh.status.borrow() != new;
        *sh.status.borrow_mut() = new;
        if changed {
            sh.notify();
        }
        Shared::send_discover(sh);
    }

    fn send_discover(sh: &Rc<Self>) {
        if let Some(c) = sh.client.borrow().as_ref() {
            let _ = c.send_text(r#"{"id":1,"cmd":"discover"}"#);
        }
    }

    fn on_error(sh: &Rc<Self>) {
        let new = "connect error".to_string();
        let changed = *sh.status.borrow() != new;
        *sh.status.borrow_mut() = new;
        if changed {
            sh.notify();
        }
    }

    /// Send a `setSpectrumSrc` command. `src = None` → EP4; `Some(slot)` →
    /// EP6 slot. Takes effect on the server's next computed FFT frame
    fn send_spectrum_source(sh: &Rc<Self>, src: Option<u8>) {
        let inner = match src {
            None => r#""ep4""#.to_string(),
            Some(slot) => format!(r#"{{"ep6":{{"slot":{slot}}}}}"#),
        };
        // `{"id":N,"cmd":"setspectrumsource","data":{"source":<SpectrumSource>}}`
        // where `<SpectrumSource>` in serde's external-tag form is either
        // `"ep4"` or `{"ep6":{"slot":N}}`.
        let msg = format!(r#"{{"id":1234,"cmd":"setspectrumsource","data":{{"source":{inner}}}}}"#);
        if let Some(c) = sh.client.borrow().as_ref() {
            let _ = c.send_text(&msg);
        }
    }

    fn send_oc_bits(sh: &Rc<Self>, oc_bits: u8) {
        let msg = format!(r#"{{"id":6,"cmd":"setocbits","data":{{"oc_bits":{oc_bits}}}}}"#);
        if let Some(c) = sh.client.borrow().as_ref() {
            let _ = c.send_text(&msg);
        }
    }

    fn send_lna(sh: &Rc<Self>, gain_db: i8) {
        let msg = format!(r#"{{"id":5,"cmd":"setlnagain","data":{{"gain_db":{gain_db}}}}}"#);
        if let Some(c) = sh.client.borrow().as_ref() {
            let _ = c.send_text(&msg);
        }
    }

    /// Send a `setvrx` command **for the currently-targeted slot**
    /// ([`Shared::vrx_slot`]), enabling / rebuilding that slot's virtual
    /// receiver with the panel's current editor values (`vrx_sideband` /
    /// `vrx_bw_hz` / `vrx_gain_db`). Offset is fixed at 0 (one receiver per
    /// slot, centred on the tune). NCO offset is part of the wire `VrxCfg`
    /// but not exposed in the panel yet.
    fn send_vrx(sh: &Rc<Self>, mode: &str, bw_hz: u32, gain_db: f32) {
        let slot = *sh.vrx_slot.borrow();
        let Some(msg) = vrx_cmd_msg(slot, mode, bw_hz, gain_db) else {
            return;
        };
        if let Some(c) = sh.client.borrow().as_ref() {
            let _ = c.send_text(&msg);
        }
    }

    /// Tear down the targeted slot's virtual receiver (`setvrxoff { slot }`).
    fn send_vrx_off(sh: &Rc<Self>) {
        let slot = *sh.vrx_slot.borrow();
        let Some(msg) = vrx_cmd_msg_off(slot) else {
            return;
        };
        if let Some(c) = sh.client.borrow().as_ref() {
            let _ = c.send_text(&msg);
        }
    }

    /// Mute / unmute the receiver on `slot` **without rebuilding it**: while
    /// muted the demod (and the FT8 decoder, in FT8 mode) keep running on the
    /// server, but it stops broadcasting that slot's `CH_AUDIO`
    fn send_vrx_mute(sh: &Rc<Self>, slot: u8, muted: bool) {
        let msg =
            format!(r#"{{"id":9,"cmd":"setvrxmute","data":{{"slot":{slot},"muted":{muted}}}}}"#);
        if let Some(c) = sh.client.borrow().as_ref() {
            let _ = c.send_text(&msg);
        }
    }

    /// Enable / disable the auto-decoder on one RX slot (`autodecode { slot,
    /// enabled }`). While enabled, the server runs a headless FT8/JS8 decoder
    /// for every known band frequency inside the slot's EP6 window and the
    /// slot's NCO stepper is locked in the UI (the decoders are built against
    /// that NCO; a stray `Tune` tears them down on the server).
    fn send_auto_decode(sh: &Rc<Self>, slot: u8, enabled: bool) {
        let msg = format!(
            r#"{{"id":11,"cmd":"autodecode","data":{{"slot":{slot},"enabled":{enabled}}}}}"#
        );
        if let Some(c) = sh.client.borrow().as_ref() {
            let _ = c.send_text(&msg);
        }
    }

    #[allow(dead_code)]
    fn send_vrx_unmute_all_except(sh: &Rc<Self>, except: u8) {
        let slots: Vec<u8> = sh
            .vrx
            .borrow()
            .iter()
            .filter(|(s, v)| **s != except && v.muted)
            .map(|(s, _)| *s)
            .collect();
        for s in slots {
            Self::send_vrx_mute(sh, s, false);
        }
    }

    /// String form of a `VrxMode` for the panel `<select>` value.
    fn vrx_mode_str(m: &hl2_common::VrxMode) -> &'static str {
        match m {
            hl2_common::VrxMode::Usb => "usb",
            hl2_common::VrxMode::Lsb => "lsb",
            hl2_common::VrxMode::Ft8 => "ft8",
            hl2_common::VrxMode::Js8 => "js8",
            hl2_common::VrxMode::Ft4 => "ft4",
        }
    }

    fn default_vrx_bw_hz() -> u32 {
        2_600
    }

    /// Re-anchor the virtual-receiver panel editor to what is actually
    /// configured on the server for the *targeted* slot ([`Shared::vrx_slot`]):
    ///
    /// * If the slot has an active [`hl2_common::VrxState`], mirror its mode /
    ///   channel bandwidth / gain into the editor (`vrx_sideband` /
    ///   `vrx_bw_hz` / `vrx_gain_db`) so the panel shows the live values.
    /// * Otherwise seed the editor with the slot's **band defaults** (see
    ///   [`Shared::panel_defaults`]) so a fresh tab still has a usable
    ///   sideband/bandwidth/gain to press **On** with.
    ///
    /// Called on every fresh [`SharedState`] and immediately when the targeted
    /// slot changes, so multiple tabs always converge on the same config.
    fn refocus_vrx_panel(sh: &Rc<Self>) {
        let slot = *sh.vrx_slot.borrow();
        let (sb, bw, g) = {
            let vrx = sh.vrx.borrow();
            match vrx.get(&slot) {
                Some(v) => (Self::vrx_mode_str(&v.mode).to_string(), v.bw_hz, v.gain_db),
                None => {
                    let def_mode = sh.panel_defaults();
                    (def_mode, Self::default_vrx_bw_hz(), 0.0f32)
                }
            }
        };
        *sh.vrx_sideband.borrow_mut() = sb;
        *sh.vrx_bw_hz.borrow_mut() = bw;
        *sh.vrx_gain_db.borrow_mut() = g;
    }

    fn panel_defaults(&self) -> String {
        let slot = *self.vrx_slot.borrow();
        let hz = self.tuning.borrow().get(&slot).copied().unwrap_or(0);
        if hz > 0 && hz < 10_000_000 {
            "lsb".to_string()
        } else {
            "usb".to_string()
        }
    }

    /// The NCO (Hz) the frequency heading shows:
    ///
    /// * **EP6 slot**: the slot's NCO from the tuning map, or `0` when it
    ///   hasn't been tuned yet. A `0` readout is meaningful and *active* —
    ///   the stepper is greyed off (`.off`) only for EP4 or when the radio
    ///   is stopped; for an untuned EP6 slot it renders `0.000.000` and
    ///   the first digit-click sends `tune(slot, new_hz)` to activate it.
    /// * **EP4** (raw wideband): no NCO exists, so `0` — the stepper
    ///   additionally gets the `ep4` flag from the caller, forcing the `.off`
    ///   class and a descriptive tooltip.
    fn freq_for(&self) -> Option<u32> {
        let tuning = self.tuning.borrow();
        match *self.spectrum_source.borrow() {
            Some(slot) => Some(tuning.get(&slot).copied().unwrap_or(0)),
            None => Some(0),
        }
    }

    fn send_tune(sh: &Rc<Self>, slot: u8, freq_hz: u32) {
        let msg =
            format!(r#"{{"id":4,"cmd":"tune","data":{{"slot":{slot},"freq_hz":{freq_hz}}}}}"#);
        if let Some(c) = sh.client.borrow().as_ref() {
            let _ = c.send_text(&msg);
        }
    }

    fn on_close(sh: &Rc<Self>) {
        let new = "disconnected (reconnecting in 3s…)".to_string();
        let changed = *sh.status.borrow() != new;
        *sh.status.borrow_mut() = new;
        if changed {
            sh.notify();
        }
        if *sh.rearming.borrow() {
            return;
        }
        *sh.rearming.borrow_mut() = true;

        let sh_clone = sh.clone();
        let timer = Closure::<dyn FnMut()>::new(move || {
            sh_clone.rearming.replace(false);
            match Shared::connect(&sh_clone) {
                Ok(client) => {
                    *sh_clone.client.borrow_mut() = Some(client);
                }
                Err(_) => {
                    let st = "ws connect failed — retrying…".to_string();
                    let changed = *sh_clone.status.borrow() != st;
                    *sh_clone.status.borrow_mut() = st;
                    if changed {
                        sh_clone.notify();
                    }
                }
            }
        });
        if let Some(w) = web_sys::window() {
            let fn_ref = timer.as_ref().unchecked_ref::<js_sys::Function>();
            let _ = w.set_timeout_with_callback_and_timeout_and_arguments_0(fn_ref, 3000);
        }
        sh.timers.borrow_mut().push(timer);
    }

    fn connect(sh: &Rc<Self>) -> Result<WsClient, String> {
        if let Some(old) = sh.client.borrow_mut().take() {
            old.close();
        }
        let url = ws_url();
        let id_slot = Rc::new(RefCell::new(0u64));

        let s0 = sh.clone();
        let s1 = sh.clone();
        let s2 = sh.clone();
        let s3 = sh.clone();
        let s4 = sh.clone();
        let sl0 = id_slot.clone();
        let sl1 = id_slot.clone();
        let sl2 = id_slot.clone();
        let sl3 = id_slot.clone();
        let sl4 = id_slot.clone();

        let client = WsClient::connect(
            &url,
            move || {
                if is_current(&s0, *sl0.borrow()) {
                    Shared::on_open(&s0);
                }
            },
            move |text| {
                if is_current(&s1, *sl1.borrow()) {
                    Shared::on_text(&s1, text);
                }
            },
            move |bytes| {
                if is_current(&s2, *sl2.borrow()) {
                    Shared::on_binary(&s2, bytes);
                }
            },
            move || {
                if is_current(&s3, *sl3.borrow()) {
                    Shared::on_close(&s3);
                }
            },
            move || {
                if is_current(&s4, *sl4.borrow()) {
                    Shared::on_error(&s4);
                }
            },
        )
        .map_err(|e| format!("{:?}", e))?;

        *id_slot.borrow_mut() = client.id();
        Ok(client)
    }
}

fn vrx_cmd_msg(slot: u8, mode: &str, bw_hz: u32, gain_db: f32) -> Option<String> {
    let mode = match mode {
        "lsb" => hl2_common::VrxMode::Lsb,
        "ft8" => hl2_common::VrxMode::Ft8,
        "js8" => hl2_common::VrxMode::Js8,
        "ft4" => hl2_common::VrxMode::Ft4,
        _ => hl2_common::VrxMode::Usb,
    };
    let cfg = hl2_common::VrxCfg {
        slot,
        offset_hz: 0, // centred on the tune (one receiver per slot for now)
        mode,
        bw_hz,
        gain_db,
    };
    Some(serde_json::json!({ "id": 7u64, "cmd": "setvrx", "data": { "cfg": cfg } }).to_string())
}

fn vrx_cmd_msg_off(slot: u8) -> Option<String> {
    Some(
        serde_json::json!({
            "id": 8u64,
            "cmd": "setvrxoff",
            "data": { "slot": slot }
        })
        .to_string(),
    )
}

fn is_current(sh: &Shared, my_id: u64) -> bool {
    sh.client.borrow().as_ref().map(|c| c.id()) == Some(my_id)
}

/// The display string for the frequency heading: fixed 8-digit decimal Hz,
/// zero-padded (`14_074_000` → `"14074000"`). The digit stepper works on this
/// string, one position per digit. Positions run MSB→LSB, so position `p`
/// (0-indexed) is the `10^(7-p)` decimal place.
fn fmt_freq(hz: u32) -> String {
    format!("{:08}", hz)
}

/// Step the digit at position `pos` (0 = most significant) up by 1 (`delta > 0`)
/// or down by 1 (`delta < 0`). "Stepping a digit by one" is exactly adding or
/// subtracting that digit's decimal place value — `10^(7-pos)` — so the normal
/// carry / borrow (a `9` wrapping to `0` and carrying into the digit to its
/// left, or a `0` wrapping to `9` and borrowing from it) falls out of the
/// integer arithmetic with no special-casing. Only positions 0..8 are honoured;
/// `hz` is clamped to `u32` so it can never underflow or overflow.
fn step_freq(hz: u32, pos: usize, delta: i8) -> u32 {
    if pos >= 8 || delta == 0 {
        return hz;
    }
    let place: u32 = 10u32.pow(7 - pos as u32);
    let step = place * delta.abs() as u32;
    if delta > 0 {
        hz.saturating_add(step)
    } else {
        hz.saturating_sub(step)
    }
}

/// The dotted-quad IP of a discovered radio (its identity for selection).
fn dev_ip(d: &hl2_common::DiscoveryInfo) -> String {
    d.ip.iter()
        .map(|o| o.to_string())
        .collect::<Vec<_>>()
        .join(".")
}

/// The frequency readout, rendered as an 8-digit stepper in three-digit
/// groups (`14.074.000`). Each digit can be stepped independently: the
/// **top half** increments (with normal decimal carry — a `9` wraps to `0`
/// and carries into the digit to its left), the **bottom half** decrements
/// (a `0` borrows from the digit to its left). Hovering a half reveals its
/// indicator bar (`.fstep-in` above for +, `.fstep-out` below for −).
fn freq_stepper(sh: Rc<Shared>, freq: Option<u32>, started: bool, ep4: bool) -> Html {
    // EP4 (raw wideband) has no NCO: render all-zeros, and disable it
    // (greyed out) whether or not the radio is started. An EP6 slot shows its
    // tune, or the em-dash placeholder if it hasn't been tuned yet.
    let hz = if ep4 { Some(0u32) } else { freq };
    let Some(hz) = hz else {
        return html! { <div class="fstep off"><span class="fstep-dot">{"— . — — — . — — —"}</span></div> };
    };
    let str_freq = fmt_freq(hz);
    // Lock the stepper while the *displayed* slot has auto-decode running:
    // the decoders are built against this NCO, and a `Tune` here would tear
    // them down on the server.
    let src = *sh.spectrum_source.borrow();
    let auto_locked = src.is_some_and(|s| {
        let am = sh.auto_monitors.borrow();
        am.get(&s).is_some_and(|v| !v.is_empty())
    });
    let enabled = started && !ep4 && !auto_locked;
    let cls = if enabled { "fstep" } else { "fstep off" };
    let mut parts: Vec<Html> = Vec::new();
    for (pos, ch) in str_freq.chars().enumerate() {
        let shc_up = sh.clone();
        let shc_dn = sh.clone();
        let pos_s = pos.to_string();
        parts.push(html! {
            <span class="fstep-digit" title={if enabled { "hover top half for +1 (carries left) · bottom half for −1 (borrows left); click to step" } else if auto_locked { "auto decode is on — NCO locked (toggle it off to tune)" } else if ep4 { "EP4 (raw wideband) has no NCO — pick an RX slot to tune" } else { "start the radio (On) to tune" }}>
                <span class="fstep-half fstep-half-top" role="button"
                      aria-label={format!("increment digit at position {pos_s} by one")}
                      onclick={Callback::from(move |_| step_and_send(&shc_up, move |prev| step_freq(prev, pos, 1)))}></span>
                <span class="fstep-half fstep-half-bottom" role="button"
                      aria-label={format!("decrement digit at position {pos_s} by one")}
                      onclick={Callback::from(move |_| step_and_send(&shc_dn, move |prev| step_freq(prev, pos, -1)))}></span>
                <span class="fstep-num" aria-hidden="true">{ch}</span>
                <span class="fstep-in" aria-hidden="true"></span>
                <span class="fstep-out" aria-hidden="true"></span>
            </span>
        });
        // `14.074.000` grouping: separator after positions 1 and 4.
        if pos == 1 || pos == 4 {
            parts.push(html! { <span class="fstep-dot" aria-hidden="true">{"."}</span> });
        }
    }
    html! {
        <div class={cls}>
            {parts}
        </div>
    }
}

fn step_and_send(sh: &Rc<Shared>, f: impl Fn(u32) -> u32) {
    let cur = sh.freq_for().unwrap_or(14_000_000);
    let next = f(cur);
    let slot = match *sh.spectrum_source.borrow() {
        Some(s) => s,
        None => *sh.tuning.borrow().keys().next().unwrap_or(&1),
    };
    sh.tuning.borrow_mut().insert(slot, next);
    if *sh.started.borrow() {
        Shared::send_tune(sh, slot, next);
    }
    sh.notify();
}

fn oc_box_row(current: u8, sh: Rc<Shared>) -> Html {
    html! {
        <div class="oc-toggles">
            {(0..7u8)
                .map(|i| oc_box(i, current, sh.clone()))
                .collect::<Html>()}
        </div>
    }
}

/// One RX filter-bank relay (F1…F7) as a compact toggle. Relay N (1-based)
/// drives bit (N−1) of the open-collector mask. Toggling it independently
/// sets/clears that bit and re-sends the whole mask; the server re-asserts
/// the mask in every keep-alive and echoes it back so all tabs converge.
fn oc_box(bit: u8, current: u8, sh: Rc<Shared>) -> Html {
    let mask_bit = 1u8 << bit;
    let checked = current & mask_bit != 0;
    let relay = (bit + 1).to_string();
    let sh_clone = sh.clone();
    html! {
        <label class={if checked { "ochip on" } else { "ochip" }} title={format!("filter bank relay F{relay}")}>
            <input
                type="checkbox"
                checked={checked}
                onchange={Callback::from(move |e: web_sys::Event| {
                    if let Some(dom) = e.target() {
                        if let Ok(cx) = dom.dyn_into::<HtmlInputElement>() {
                            let new_mask = if cx.checked() {
                                *sh_clone.oc_bits.borrow() | mask_bit
                            } else {
                                *sh_clone.oc_bits.borrow() & !mask_bit
                            };
                            *sh_clone.oc_bits.borrow_mut() = new_mask;
                            Shared::send_oc_bits(&sh_clone, new_mask);
                            sh_clone.notify();
                        }
                    }
                })}
            />
            {"F"}{relay}
        </label>
    }
}

/// `mag` → dB, mirroring `canvas::lin_to_db` (full-scale = 65535 = 0 dBFS,
/// clamped to [-100, +10] dB) so the auto scale is in the same units the
/// renderers draw with.
fn mag_to_db(mag: u16) -> f64 {
    const LOG_FLOOR: f64 = -100.0;
    const LOG_CEIL: f64 = 10.0;
    let v = mag as f64 / 65535.0;
    if v <= 0.0 {
        LOG_FLOOR
    } else {
        (20.0 * v.log10()).clamp(LOG_FLOOR, LOG_CEIL)
    }
}

fn auto_estimate(mags: &[u16]) -> Option<(f64, f64)> {
    if mags.is_empty() {
        return None;
    }
    let mut dbs: Vec<f64> = mags.iter().map(|&m| mag_to_db(m)).collect();
    dbs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = dbs[dbs.len() / 2];
    let max = *dbs.last().unwrap();
    Some((median, max))
}

fn auto_step_follower(
    val: &mut f64,
    target: f64,
    alpha_attack: f64,
    alpha_release: f64,
    deadband: f64,
) -> bool {
    let diff = target - *val;
    if diff.abs() <= deadband {
        return false;
    }
    let alpha = if diff > 0.0 {
        alpha_attack
    } else {
        alpha_release
    };
    *val += diff * alpha;
    true
}

impl Shared {
    pub fn update_auto_scale(sh: &Self, mags: &[u16]) -> bool {
        if !*sh.floor_auto.borrow() || mags.is_empty() {
            return false;
        }
        let Some((median, max)) = auto_estimate(mags) else {
            return false;
        };

        const HEADROOM_DB: f64 = 5.0;
        const MIN_SPAN_DB: f64 = 25.0;

        const FLOOR_ATTACK_ALPHA: f64 = 0.05;
        const FLOOR_RELEASE_ALPHA: f64 = 0.01;
        const DEADBAND_DB: f64 = 1.0;

        const CEIL_HOLD_MARGIN_DB: f64 = 4.0;
        const CEIL_HOLD_FRAMES: u32 = 12;
        const CEIL_RELEASE_DB_PER_FRAME: f64 = 0.1;

        let noise_min = (median - HEADROOM_DB).max(-100.0);
        let ceil_target = max.max(noise_min) - HEADROOM_DB;

        let mut floor = *sh.auto_floor.borrow();
        let mut ceil = *sh.auto_ceil.borrow();
        let mut age = *sh.ceil_age.borrow();

        if *sh.auto_recenter.borrow() {
            // A different magnitude regime (source change / re-enable): snap
            // both edges instead of blending from the old regime.
            floor = noise_min;
            ceil = ceil_target.min(10.0);
            age = 0;
            *sh.auto_recenter.borrow_mut() = false;
        } else {
            auto_step_follower(
                &mut floor,
                noise_min,
                FLOOR_ATTACK_ALPHA,
                FLOOR_RELEASE_ALPHA,
                DEADBAND_DB,
            );

            if ceil_target >= ceil - CEIL_HOLD_MARGIN_DB {
                // The signal that established the ceiling (or a stronger
                // one) is still present: instant attack upward, freeze
                // otherwise — keying dips within the margin never move it.
                ceil = ceil.max(ceil_target);
                age = 0;
            } else if age < CEIL_HOLD_FRAMES {
                // Max left the margin, but the grace window is still open
                // (brief keying gap / fading): hold.
                age += 1;
            } else {
                // Signal genuinely gone: slow release toward the current
                // minimum (never below the noise-floor-based target).
                ceil -= (ceil - ceil_target).min(CEIL_RELEASE_DB_PER_FRAME);
            }
        }

        enforce_min_span(&mut floor, &mut ceil, MIN_SPAN_DB);

        let moved = (*sh.auto_floor.borrow() - floor).abs() > 1e-9
            || (*sh.auto_ceil.borrow() - ceil).abs() > 1e-9;
        *sh.auto_floor.borrow_mut() = floor;
        *sh.auto_ceil.borrow_mut() = ceil;
        *sh.ceil_age.borrow_mut() = age;
        moved
    }

    /// Test accessor for the current auto scale (floor, ceil).
    #[cfg(test)]
    pub fn auto_scale(&self) -> (f64, f64) {
        (*self.auto_floor.borrow(), *self.auto_ceil.borrow())
    }
}

fn enforce_min_span(floor: &mut f64, ceil: &mut f64, min_span: f64) {
    let span = *ceil - *floor;
    if span >= min_span {
        return;
    }
    let mid = (*floor + *ceil) / 2.0;
    *floor = mid - min_span / 2.0;
    *ceil = mid + min_span / 2.0;
}

fn canvas_2d(id: &str, h: u32) -> Option<(web_sys::CanvasRenderingContext2d, u32)> {
    let el: web_sys::HtmlCanvasElement = web_sys::window()?
        .document()?
        .get_element_by_id(id)?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .ok()?;
    let w = el.client_width().max(1) as u32;
    if el.width() != w {
        el.set_width(w);
    }
    if el.height() != h {
        el.set_height(h);
    }
    let ctx = el
        .get_context("2d")
        .ok()??
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .ok()?;
    Some((ctx, w))
}

/// Draw the panadapter (latest frame) and waterfall (history), then overlay
/// each active virtual-receiver passband (one per slot) on top of both.
fn draw_all(sh: &Shared) {
    // Auto mode: the live follower's scale. Manual: the user's inputs.
    let (floor, ceil) = if *sh.floor_auto.borrow() {
        (*sh.auto_floor.borrow(), *sh.auto_ceil.borrow())
    } else {
        (*sh.floor.borrow(), *sh.ceil.borrow())
    };
    let center_hz = *sh.spectrum_center_hz.borrow();
    let span_hz = *sh.spectrum_span_hz.borrow();
    // Per-slot virtual receivers currently active (offset, mode, bw) —
    // each shades its own passband on the currently-displayed source.
    // (Mode → tint key via `vrx_mode_str_sideband`: LSB tints blue, USB/FT8
    // green.)
    let vrx_bands: Vec<(i32, String, u32)> = sh
        .vrx
        .borrow()
        .iter()
        .map(|(_, v)| (v.offset_hz, vrx_mode_str_sideband(v), v.bw_hz))
        .collect();

    // Auto-decoder passbands: only for the *displayed* slot, and only when it
    // has auto-decode enabled (a non-empty monitor list). Each entry shades a
    // one-sided USB strip (carrier → carrier + bw) covering the decoder's
    // channel-select passband, so the user sees what the headless decoders are
    // listening to on the spectrum / waterfall. The digital modes all share the
    // 2 600 Hz channel-select bandwidth, so `(freq, 2600)` is a faithful band.
    const AUTO_BW_HZ: u32 = 2_600;
    let auto_bands: Vec<(u32, u32)> = {
        let src = *sh.spectrum_source.borrow();
        src.and_then(|slot| {
            sh.auto_monitors.borrow().get(&slot).map(|v| {
                v.iter()
                    .map(|m| (m.freq_hz, AUTO_BW_HZ))
                    .collect::<Vec<_>>()
            })
        })
        .unwrap_or_default()
    };

    if let Some((pan, w)) = canvas_2d("panadapter", 250) {
        let mags = sh.latest.borrow().clone();
        canvas::draw_panadapter(
            &pan, &mags, floor, ceil, w as usize, 250, center_hz, span_hz,
        );
        for (offset, sb, bw) in &vrx_bands {
            canvas::draw_vrx_passband(&pan, sb, *bw, w as usize, 250, center_hz, span_hz, *offset);
        }
        if !auto_bands.is_empty() {
            canvas::draw_auto_passbands(&pan, &auto_bands, w as usize, 250, center_hz, span_hz);
        }
    }

    if let Some((wf, w)) = canvas_2d("waterfall", 256) {
        let history = sh.history.borrow().clone();
        canvas::paint_waterfall(
            &wf, &history, floor, ceil, w as usize, 256, center_hz, span_hz,
        );
        for (offset, sb, bw) in &vrx_bands {
            canvas::draw_vrx_passband(&wf, sb, *bw, w as usize, 256, center_hz, span_hz, *offset);
        }
        if !auto_bands.is_empty() {
            canvas::draw_auto_passbands(&wf, &auto_bands, w as usize, 256, center_hz, span_hz);
        }
    }
}

fn vrx_mode_str_sideband(v: &hl2_common::VrxState) -> String {
    match v.mode {
        hl2_common::VrxMode::Lsb => "lsb".to_string(),
        _ => "usb".to_string(),
    }
}

/// Uppercase label for a receiver's mode, for the decode-log `Mode` column.
fn vrx_mode_label(m: &hl2_common::VrxMode) -> String {
    match m {
        hl2_common::VrxMode::Ft8 => "FT8".to_string(),
        hl2_common::VrxMode::Js8 => "JS8".to_string(),
        hl2_common::VrxMode::Ft4 => "FT4".to_string(),
        hl2_common::VrxMode::Usb => "USB".to_string(),
        hl2_common::VrxMode::Lsb => "LSB".to_string(),
    }
}

#[function_component(App)]
pub fn app() -> Html {
    let sh = use_state(|| Shared::new());
    let sh = (*sh).clone();
    let force = use_force_update();
    *sh.force.borrow_mut() = Some(force.clone());

    {
        let sh = sh.clone();
        use_effect_with((), move |_| {
            *sh.client.borrow_mut() = match Shared::connect(&sh) {
                Ok(client) => Some(client),
                Err(_) => None,
            }
        });
    }

    let status = sh.status.borrow().clone();
    let started = *sh.started.borrow();
    let freq = sh.freq_for();
    let devices = sh.devices.borrow().clone();
    let active_device = sh.active_device.borrow().clone();
    let online = status.contains("connected") || status.contains("reconnected");
    let status_class = "status".to_string()
        + if online {
            " ok"
        } else if status.to_lowercase().contains("failed") || status.contains("error") {
            " err"
        } else {
            " warn"
        };
    let offline_class = if online { "" } else { " offline" };
    let sh2 = sh.clone();
    let sh7 = sh.clone();
    let sh8 = sh.clone();
    let sh10 = sh.clone();
    let sh12 = sh.clone();
    let sh13 = sh.clone();
    let sh14 = sh.clone();
    let sh15 = sh.clone();
    let sh16 = sh.clone();
    let sh17 = sh.clone();
    let sh18 = sh.clone();
    let sh19 = sh.clone();
    let sh21 = sh.clone();
    let sh23 = sh.clone();
    let sh25 = sh.clone();

    let floor_auto = *sh.floor_auto.borrow();
    let scale_floor_display = if floor_auto {
        *sh.auto_floor.borrow()
    } else {
        *sh.floor.borrow()
    };
    let scale_ceil_display = if floor_auto {
        *sh.auto_ceil.borrow()
    } else {
        *sh.ceil.borrow()
    };

    // Auto decode (headless FT8/JS8 decoders on the *displayed* slot):
    // on/off state + the (mode, freq) readout of the active decoders.
    let auto_slot: Option<u8> = *sh.spectrum_source.borrow();
    let auto_on: bool = auto_slot.is_some_and(|s| {
        sh.auto_monitors
            .borrow()
            .get(&s)
            .is_some_and(|v| !v.is_empty())
    });
    let empty = yew::virtual_dom::VNode::VList(std::rc::Rc::new(yew::virtual_dom::VList::new()));
    let auto_row: Html = if let Some(slot) = auto_slot {
        let items: String = sh
            .auto_monitors
            .borrow()
            .get(&slot)
            .map(|v| {
                v.iter()
                    .map(|m| {
                        let name = match m.mode {
                            hl2_common::VrxMode::Ft8 => "FT8",
                            hl2_common::VrxMode::Js8 => "JS8",
                            hl2_common::VrxMode::Ft4 => "FT4",
                            _ => "MODE",
                        };
                        format!("{name}({})", fmt_freq(m.freq_hz))
                    })
                    .collect::<Vec<_>>()
                    .join(" · ")
            })
            .unwrap_or_default();
        let sh_ad = sh.clone();
        html! {
            <label class="autobox">
                <input
                    type="checkbox"
                    checked={auto_on}
                    title={"Run headless FT8/JS8/FT4 decoders on every known band frequency in this slot's window; locks the NCO while on. Grey strips on the spectrum/waterfall mark the frequencies being decoded."}
                    onchange={Callback::from(move |e: web_sys::Event| {
                        if let Some(dom) = e.target() {
                            if let Ok(cx) = dom.dyn_into::<HtmlInputElement>() {
                                let on = cx.checked();
                                if *sh_ad.started.borrow() {
                                    Shared::send_auto_decode(&sh_ad, slot, on);
                                }
                            }
                        }
                        sh_ad.notify();
                    })}
                />
                {"Auto decode"}
                {if auto_on {
                    format!("({items})")
                } else {
                    String::new()
                }}
            </label>
        }
    } else {
        empty
    };

    let vrx_slot: u8 = *sh13.vrx_slot.borrow();
    let vrx_on: bool = sh13.vrx.borrow().contains_key(&vrx_slot);
    let vrx_muted_cur: bool = sh13
        .vrx
        .borrow()
        .get(&vrx_slot)
        .map(|v| v.muted)
        .unwrap_or(false);
    // The decode tables show while *any* slot demodulates the mode —
    // including headless auto-decode monitors (which carry no `VrxState`
    // of their own and don't appear in `vrx`).
    let auto_modes: Vec<hl2_common::VrxMode> = sh13
        .auto_monitors
        .borrow()
        .values()
        .flatten()
        .map(|m| m.mode)
        .collect();
    // The single decode-log table shows while *any* slot is demodulating a
    // digital mode (FT8 / FT4 / JS8) — including headless auto-decode
    // monitors. Which mode produced a given row is read from that row's
    // `vrx` snapshot, so no per-mode gate is needed here.
    let digital_on: bool = sh13.vrx.borrow().values().any(|v| {
        matches!(
            v.mode,
            hl2_common::VrxMode::Ft8 | hl2_common::VrxMode::Js8 | hl2_common::VrxMode::Ft4
        )
    }) || auto_modes.contains(&hl2_common::VrxMode::Ft8)
        || auto_modes.contains(&hl2_common::VrxMode::Js8)
        || auto_modes.contains(&hl2_common::VrxMode::Ft4);
    let vrx_sideband_cur: String = sh14.vrx_sideband.borrow().clone();
    let vrx_bw_cur: u32 = *sh15.vrx_bw_hz.borrow();
    let vrx_gain_cur: f32 = *sh16.vrx_gain_db.borrow();
    let vrx_play_gain_cur: f32 = *sh17.vrx_play_gain.borrow();
    let decode_log: Vec<hl2_common::DecodeRow> = sh17.decode_log.borrow().clone();
    let vrx_desc: Html = if vrx_sideband_cur == "ft8" {
        html! {
            <p class="muted">
                {format!("Demodulates RX{vrx_slot} baseband as FT8 (12 kHz sample rate). Decoded messages appear in the log below — one decode attempt per completed 15 s slot.")}
            </p>
        }
    } else if vrx_sideband_cur == "js8" {
        html! {
            <p class="muted">
                {format!("Demodulates RX{vrx_slot} baseband as JS8 (all speeds: A/B/C/E, 12 kHz sample rate). Each speed is decoded continuously — a cycle (15/10/6/30 s) is decoded as soon as its data is ready, not on a fixed slot boundary. Decoded frames appear in the log below, tagged with their speed.")}
            </p>
        }
    } else if vrx_sideband_cur == "ft4" {
        html! {
            <p class="muted">
                {format!("Demodulates RX{vrx_slot} baseband as FT4 (12 kHz sample rate). FT4 is a 7.5 s-slot WSJT-family mode — decoded messages appear in the log below, one decode attempt per completed 7.5 s slot.")}
            </p>
        }
    } else {
        html! {
            <p class="muted">
                {format!("Demodulates RX{vrx_slot} baseband as SSB — ")}{if vrx_sideband_cur == "usb" {"USB" } else { "LSB" }}{". Shaded band on the spectrum = channel-select passband."}
            </p>
        }
    };

    // The decode-log table; rendered while *any* slot demodulates a digital
    // mode (FT8 / FT4 / JS8) — the newest-first list of decoded messages
    // pushed by the server across all modes. Each row is stamped with the RX
    // slot (from the decode's `vrx` snapshot) that demodulated it and the
    // mode of that receiver, so the single table mixes FT8 / FT4 / JS8 rows.
    let decode_log_html: Html = if digital_on {
        let rows: Html = decode_log
            .iter()
            .map(|m| {
                html! {
                    <tr>
                        <td class="ft8-rx">{format!("RX{}", m.vrx.slot)}</td>
                        <td>{vrx_mode_label(&m.vrx.mode)}</td>
                        <td>{format!("{:.1}", m.freq_hz)}</td>
                        <td>{format!("{:+.2}", m.dt_sec)}</td>
                        <td>{format!("{:.1}", m.snr_db)}</td>
                        <td>{m.slot_ms.to_string()}</td>
                        <td class="ft8-text">{m.text.as_str()}</td>
                    </tr>
                }
            })
            .collect();
        html! {
            <div class="ft8-log">
                <h4>{"decode log"}</h4>
                {if decode_log.is_empty() {
                    html! {
                        <p class="muted">{"Waiting for a decoded message…"}</p>
                    }
                } else {
                    html! {
                        <table class="ft8-table">
                            <thead>
                                <tr>
                                    <th>{"Rx"}</th>
                                    <th>{"Mode"}</th>
                                    <th>{"Freq (Hz)"}</th>
                                    <th>{"Δt (s)"}</th>
                                    <th>{"SNR dB"}</th>
                                    <th>{"Slot (ms)"}</th>
                                    <th>{"Text"}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rows}
                            </tbody>
                        </table>
                    }
                }}
            </div>
        }
    } else {
        Html::default()
    };

    // Mute/Unmute label for the targeted slot (drives the toggle button text).
    let mute_label: String = if vrx_muted_cur {
        "Mute".to_string()
    } else {
        "Unmute".to_string()
    };

    // The RX filter-bank relay checkboxes. Relay N (1-based) drives bit
    // (N-1) of the open-collector mask. Each box independently toggles its own
    // bit and sends a `setocbits` command; the server re-asserts the resulting
    // mask in every subsequent keep-alive (≤40 ms) and echoes it back in the
    // shared state
    let oc_cur = *sh12.oc_bits.borrow();
    let oc_row: Html = oc_box_row(oc_cur, sh12.clone());
    // Current LNA gain (dB), mirrored from the server — value for the slider
    // and the dB readout.
    let lna_cur: i8 = *sh10.lna.borrow();
    // LNA slider fill position (−12…48 → 0…100 %) for the gradient track.
    let lna_pct = ((lna_cur as f32) + 12.0) / 60.0 * 100.0;

    // The EP4 (raw wideband) source tab — added after the last RX tab.
    let ep4_active = sh.spectrum_source.borrow().is_none();
    let sh_ep4 = sh.clone();
    let ep4_tab: Html = {
        let active = ep4_active;
        let shc = sh_ep4;
        html! {
            <li class="nav-item">
                <a class={if active { "nav-link active" } else { "nav-link" }}
                   aria-current={if active { Some("page") } else { None }}
                   href="#"
                    title="EP4 — raw wideband (122.88 MSps); no NCO, so the frequency readout is disabled"
                    onclick={Callback::from(move |_| {
                        *shc.spectrum_source.borrow_mut() = None;
                        *shc.vrx_slot.borrow_mut() = 1;
                        Shared::refocus_vrx_panel(&shc);
                        Shared::send_spectrum_source(&shc, None);
                        shc.notify();
                    })}>
                    {"EP4"}
                </a>
            </li>
        }
    };

    html! {
            <div class={format!("app{offline_class}")}>
                <h1>{"Hermes Lite 2"}</h1>

                <div class="devrow">
                    {if devices.is_empty() {
                        html! { <span class="dev muted-dev">{"no radios discovered"}</span> }
                    } else {
                        devices.iter().map(|d| {
                            let ip = dev_ip(d);
                            let display = ip.clone();
                            let in_service = d.in_service;
                            html! {
                                <button
                                    class={if active_device.as_deref() == Some(&ip) { "dev active" } else { "dev" }}
                                    disabled={!online}
                                    title={if in_service { "In use by this server" } else { "Idle (discovered)" }}
                                    onclick={Callback::from({
                                        let shc = sh.clone();
                                        let ip = ip.clone();
                                        move |_| {
                                            *shc.active_device.borrow_mut() = Some(ip.clone());
                                            shc.notify();
                                        }
                                    })}
                                >
                                    <span class="dev-ip">{display}</span>
                                    <span class="dev-meta">
                                        {format!("{} RX · {}", d.rx_count, if d.sample_16bit {"16-bit"} else {"12-bit"})}
                                        {if in_service { html! { <span class="dev-inuse" title="In use by this server">{"in use"}</span> } } else { html! { <span></span> } }}
                                    </span>
                                </button>
                            }
                        }).collect::<Html>()
                    }}
                    <span class={status_class}>
                        {format!("{} {}", if online { "●" } else { "○" }, status)}
                    </span>
                </div>

                <ul class="nav nav-tabs">
                    {(1..5u8).map(|n| {
                        let active = *sh21.spectrum_source.borrow() == Some(n);
                        let shc = sh.clone();
                        html! {
                            <li class="nav-item">
                                <a class={if active { "nav-link active" } else { "nav-link" }}
                                   aria-current={if active { Some("page") } else { None }}
                                href="#"
                                onclick={Callback::from(move |_| {
                                    let src = Some(n);
                                    *shc.spectrum_source.borrow_mut() = src;
                                    *shc.vrx_slot.borrow_mut() = n;
                                    Shared::refocus_vrx_panel(&shc);
                                    Shared::send_spectrum_source(&shc, src);
                                    shc.notify();
                                })}>
                                    {format!("RX{n}")}
                                </a>
                            </li>
                        }
                    }).collect::<Html>()}
                    {ep4_tab.clone()}
                    <li class="ms-auto">
                    <div class="form-check form-switch">
                    <div class="d-flex">
        { "off" }
        <div class="form-switch ms-2">
            <input type="checkbox" class="form-check-input" id="site_state"
                checked={started}
                onchange={Callback::from(move |e: web_sys::Event| {
                    if let Some(dom) = e.target() {
                        if let Ok(cx) = dom.dyn_into::<HtmlInputElement>() {
                             let on = cx.checked();
                             if let Some(c) = sh2.client.borrow().as_ref() {
                                 let msg = if on {
                                     r#"{"id":2,"cmd":"start","data":{"initial_tuning":null}}"#.to_string()
                                 } else {
                                     r#"{"id":3,"cmd":"stop"}"#.to_string()
                                 };
                                 let _ = c.send_text(&msg);
                             }
                              *sh2.started.borrow_mut() = on;
                              *sh2.started_local.borrow_mut() = on;
                              sh2.notify();
                         }
                     }
                 })} />
        </div>
        <label for="site_state" class="form-check-label">{ "on" }</label>
    </div>
                    </div>

                    </li>
                </ul>

                <div class="tune-row">
                    {freq_stepper(sh.clone(), freq, started, sh.spectrum_source.borrow().is_none())}

                    <div class={if online && started { "fpanel" } else { "fpanel idle" }}>
                        <div class="fpanel-block">
                            <span class="fpanel-label">{"RX filter bank"}</span>
                            {oc_row}
                        </div>
                        <div class="fpanel-sep"></div>
                        <div class="fpanel-block">
                            <span class="fpanel-label">{"LNA"}</span>
                            <input type="range" min="-12" max="48" step="1"
                                value={lna_cur.to_string()}
                                style={format!("--fill: {}%", lna_pct)}
                                class="lna-slider"
                                aria-label="LNA gain (dB)"
                                oninput={Callback::from(move |e: web_sys::InputEvent| {
                                    if let Some(inp) = e.target_dyn_into::<HtmlInputElement>() {
                                        let v: i8 = inp.value().parse::<i8>().unwrap_or(6).clamp(-12, 48);
                                        *sh10.lna.borrow_mut() = v;
                                        if *sh10.started.borrow() {
                                            Shared::send_lna(&sh10, v);
                                        }
                                    }
                                    sh10.notify();
                                })}/>
                            <span class="lna-val">{format!("{:+} dB", lna_cur)}</span>
                        </div>
                    </div>
                </div>

                     <canvas id="panadapter" width="800" height="250"></canvas>
                     <canvas id="waterfall" width="800" height="256" class="waterfall"></canvas>
                       // Auto decode on its own row so a long monitor list
                       // (3 modes x multiple bands) can't wrap the Floor/Ceil
                       // inputs onto a different line from the Auto scale box.
                        <div class="row autodecode-row">{auto_row.clone()}</div>
                        <div class="row scale-row">
                          <label class="autobox">
                            <input
                                type="checkbox"
                                checked={floor_auto}
                                onchange={Callback::from(move |e: web_sys::Event| {
                                    if let Some(dom) = e.target() {
                                        if let Ok(cx) = dom.dyn_into::<HtmlInputElement>() {
                                            let on = cx.checked();
                                            if on {
                                                *sh18.auto_recenter.borrow_mut() = true;
                                            } else {
                                                *sh18.floor.borrow_mut() = *sh18.auto_floor.borrow();
                                                *sh18.ceil.borrow_mut() = *sh18.auto_ceil.borrow();
                                            }
                                            *sh18.floor_auto.borrow_mut() = on;
                                        }
                                    }
                                    sh18.notify();
                                })}
                            />
                            {"Auto scale"}
                        </label>
                        <label>{"Floor dB "}
                            <input type="number" min="-120" max="0" disabled={floor_auto}
                                value={scale_floor_display.to_string()} oninput={Callback::from(move |e: web_sys::InputEvent| {
                                if let Some(inp) = e.target_dyn_into::<HtmlInputElement>() {
                                    let v: f64 = inp.value().parse::<f64>().unwrap_or(*sh7.floor.borrow());
                                    *sh7.floor.borrow_mut() = v;
                                    *sh7.floor_auto.borrow_mut() = false;
                                }
                                sh7.notify();
                            })} />
                        </label>
                        <label>{"Ceil dB "}
                            <input type="number" min="-100" max="30" disabled={floor_auto}
                                value={scale_ceil_display.to_string()} oninput={Callback::from(move |e: web_sys::InputEvent| {
                                if let Some(inp) = e.target_dyn_into::<HtmlInputElement>() {
                                    let v: f64 = inp.value().parse::<f64>().unwrap_or(*sh8.ceil.borrow());
                                    *sh8.ceil.borrow_mut() = v;
                                    *sh8.floor_auto.borrow_mut() = false;
                                }
                                sh8.notify();
                            })} />
                        </label>
                    </div>

                <div class="panel vrx-panel">
                    <h3>{"Virtual receiver"}</h3>
                    {vrx_desc}
                    <div class="row">
                        <span class={format!("vrx-badge {}", if vrx_on {"on"} else {"off"})}>
                            {format!("● RX{vrx_slot} {}", if vrx_on { if vrx_muted_cur { "ON (muted)" } else { "ON" } } else { "OFF" })}
                        </span>
                         <label>{"Mode "}
                             <select onchange={Callback::from(move |e: web_sys::Event| {
                                 if let Some(dom) = e.target() {
                                     if let Ok(sel) = dom.dyn_into::<HtmlSelectElement>() {
                                         let v = sel.value().to_lowercase();
                                          if v == "usb" || v == "lsb" || v == "ft8" || v == "js8" || v == "ft4" {
                                             *sh14.vrx_sideband.borrow_mut() = v.clone();
                                         }
                                         if sh14.vrx.borrow().contains_key(&*sh14.vrx_slot.borrow()) {
                                             let bw = *sh14.vrx_bw_hz.borrow();
                                             let g  = *sh14.vrx_gain_db.borrow();
                                             sh14.audio.reset();
                                             Shared::send_vrx(&sh14, &v, bw, g);
                                         }
                                         sh14.notify();
                                     }
                                 }
                             })}>
                                 <option value="usb" selected={vrx_sideband_cur == "usb"}>{"USB"}</option>
                                 <option value="lsb" selected={vrx_sideband_cur == "lsb"}>{"LSB"}</option>
                                  <option value="ft8" selected={vrx_sideband_cur == "ft8"}>{"FT8"}</option>
                                  <option value="js8" selected={vrx_sideband_cur == "js8"}>{"JS8"}</option>
                                  <option value="ft4" selected={vrx_sideband_cur == "ft4"}>{"FT4"}</option>
                             </select>
                         </label>
                        <label>{"BW Hz "}
                            <input type="number" min="300" max="12000" step="100"
                                   value={vrx_bw_cur.to_string()} oninput={Callback::from(move |e: web_sys::InputEvent| {
                                if let Some(inp) = e.target_dyn_into::<HtmlInputElement>() {
                                    let v: u32 = inp.value().parse::<u32>().unwrap_or(2600).max(300).min(12000);
                                    *sh15.vrx_bw_hz.borrow_mut() = v;
                                    if sh15.vrx.borrow().contains_key(&*sh15.vrx_slot.borrow()) {
                                        let sb = sh15.vrx_sideband.borrow().clone();
                                        let g  = *sh15.vrx_gain_db.borrow();
                                        sh15.audio.reset();
                                        Shared::send_vrx(&sh15, &sb, v, g);
                                    }
                                }
                                sh15.notify();
                            })} />
                        </label>
                        <label>{"Gain dB "}
                            <input type="number" min="-40" max="40" step="1"
                                   value={format!("{:.1}", vrx_gain_cur)} oninput={Callback::from(move |e: web_sys::InputEvent| {
                                if let Some(inp) = e.target_dyn_into::<HtmlInputElement>() {
                                    let v: f32 = inp.value().parse::<f32>().unwrap_or(0.0).clamp(-40.0, 40.0);
                                    *sh16.vrx_gain_db.borrow_mut() = v;
                                    if sh16.vrx.borrow().contains_key(&*sh16.vrx_slot.borrow()) {
                                        let sb = sh16.vrx_sideband.borrow().clone();
                                        let bw = *sh16.vrx_bw_hz.borrow();
                                        sh16.audio.reset();
                                        Shared::send_vrx(&sh16, &sb, bw, v);
                                    }
                                }
                                sh16.notify();
                            })} />
                        </label>
                        <label class="vol"><span>{"Vol"}</span>
                            <input type="range" min="0" max="100" step="1"
                                   value={((vrx_play_gain_cur * 100.0).round() as i32).to_string()} oninput={Callback::from(move |e: web_sys::InputEvent| {
                                if let Some(inp) = e.target_dyn_into::<HtmlInputElement>() {
                                    let v: i32 = inp.value().parse::<i32>().unwrap_or(80).max(0).min(100);
                                    let lin = (v as f32) / 100.0;
                                    *sh19.vrx_play_gain.borrow_mut() = lin;
                                    sh19.audio.set_gain(lin);
                                }
                                sh19.notify();
                            })} />
                            <span class="vol-val">{((vrx_play_gain_cur * 100.0).round() as i32).to_string()}{"%"}</span>
                        </label>
                        <button disabled={!vrx_on} onclick={Callback::from(move |_| {
                            let slot = *sh21.vrx_slot.borrow();
                            let muted_now = sh21.vrx.borrow().get(&slot).map(|v| v.muted).unwrap_or(false);
                            let new_muted = !muted_now;
                            Shared::send_vrx_mute(&sh21, slot, new_muted);
                            if new_muted {
                                sh21.audio.mute();
                            } else {
                                sh21.audio.unmute();
                            }
                             let mut vrx = sh21.vrx.borrow_mut();
                             if let Some(v) = vrx.get_mut(&slot) {
                                 v.muted = new_muted;
                             }
                             sh21.notify();
                        })}>
                            {mute_label.clone()}
                        </button>
                        <button disabled={vrx_on} onclick={Callback::from(move |_| {
                            let slot = *sh23.vrx_slot.borrow();
                            let sb = sh23.vrx_sideband.borrow().clone();
                            let bw = *sh23.vrx_bw_hz.borrow();
                            let g  = *sh23.vrx_gain_db.borrow();
                            let others: Vec<u8> = sh23.vrx.borrow().iter()
                                .filter(|(s, _)| **s != slot)
                                .map(|(s, _)| *s)
                                .collect();
                             for os in others {
                                 Shared::send_vrx_mute(&sh23, os, true);
                                 let mut vrx = sh23.vrx.borrow_mut();
                                 if let Some(v) = vrx.get_mut(&os) { v.muted = true; }
                             }
                            Shared::send_vrx_mute(&sh23, slot, false);
                            sh23.audio.unmute();
                            Shared::send_vrx(&sh23, &sb, bw, g);
                            sh23.notify();
                        })}>
                            {"On"}
                        </button>
                        <button disabled={!vrx_on} onclick={Callback::from(move |_| {
                            let slot = *sh25.vrx_slot.borrow();
                            sh25.audio.mute();
                            Shared::send_vrx_off(&sh25);
                            sh25.vrx.borrow_mut().remove(&slot);
                            sh25.notify();
                        })}>
                            {"Off"}
                        </button>
                    </div>

                    {decode_log_html}
                </div>

                <footer class="hint">
                    <p>{"Binary frames: "}<code>{"0x01 0x00"}</code>{" (CH_WIDEBAND) → magnitude spectrum. Control via JSON."}</p>
                </footer>
            </div>
        }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_freq_zero_pads_to_eight_digits() {
        assert_eq!(fmt_freq(14_074_000), "14074000");
        assert_eq!(fmt_freq(7), "00000007");
        assert_eq!(fmt_freq(10_000_000), "10000000");
        assert_eq!(fmt_freq(0), "00000000");
    }

    #[test]
    fn step_increments_single_digit() {
        // 14 074 000 → "14074000".
        // pos 3 = ten-thousands place (the `7`): 7 → 8.
        assert_eq!(step_freq(14_074_000, 3, 1), 14_084_000);
        // pos 4 = thousands place (the `4`): 4 → 5.
        assert_eq!(step_freq(14_074_000, 4, 1), 14_075_000);
        // pos 7 = ones (always 0 at the start): 0 → 1.
        assert_eq!(step_freq(14_074_000, 7, 1), 14_074_001);
    }

    #[test]
    fn step_carry_on_increment_wraps_nine() {
        // 14 074 900 → "14074900"; pos 5 (hundreds) is `9`.
        // 9 wraps to 0 and carries into pos 4: 14074900 + 100 → 14075000.
        assert_eq!(step_freq(14_074_900, 5, 1), 14_075_000);
        // 99 → "00000099"; pos 6 (tens) is `9`.
        // 9 → 0, carry into pos 5 (hundreds 0 → 1): 99 + 10 → 109.
        assert_eq!(step_freq(99, 6, 1), 109);
    }

    #[test]
    fn step_decrements_single_digit() {
        // pos 4 (the `5` in 14 075 000): 5 → 4.
        assert_eq!(step_freq(14_075_000, 4, -1), 14_074_000);
        // pos 7 (ones `1` in 14 074 001): 1 → 0.
        assert_eq!(step_freq(14_074_001, 7, -1), 14_074_000);
    }

    #[test]
    fn step_borrow_on_decrement_wraps_zero() {
        // 14 075 000 → "14075000"; pos 5 (hundreds) is `0`.
        // 0 → 9, borrow from pos 4 (thousands 5 → 4): 14075000 − 100 → 14074900.
        assert_eq!(step_freq(14_075_000, 5, -1), 14_074_900);
        // 1000 → "00001000"; pos 5 (hundreds) is `0`.
        // 0 → 9, borrow from pos 4 (thousands 1 → 0): 1000 − 100 → 900.
        assert_eq!(step_freq(1_000, 5, -1), 900);
    }

    #[test]
    fn step_out_of_range_positions_are_ignored() {
        assert_eq!(step_freq(14_074_000, 8, 1), 14_074_000);
        assert_eq!(step_freq(14_074_000, 99, 1), 14_074_000);
        assert_eq!(step_freq(14_074_000, 8, -1), 14_074_000);
    }

    #[test]
    fn step_saturates_at_extremes() {
        // Max: 99.999.999 → incrementing the MSB wraps to 0.900.000.0 (i.e.
        // adding 10^7 to 99 999 999 → 109 999 999) — fits in u32.
        assert_eq!(step_freq(99_999_999, 0, 1), 109_999_999);
        // 0 − 1 (ones) saturates at 0 (can't go negative).
        assert_eq!(step_freq(0, 7, -1), 0);
        // 0 − 10 (tens, pos 6) also saturates at 0.
        assert_eq!(step_freq(0, 6, -1), 0);
    }

    #[test]
    fn step_is_inverse_of_itself_mid_digit() {
        let base = 14_074_321u32;
        for pos in 0..8usize {
            let up = step_freq(base, pos, 1);
            assert_eq!(step_freq(up, pos, -1), base, "pos {pos} not inverse");
        }
    }

    fn db_to_mag(db: f64) -> u16 {
        // Inverse of `20*log10(mag/65535)`: `mag/65535 = 10^(db/20)`.
        let v = 10.0f64.powf(db / 20.0).clamp(0.0, 1.0);
        (v * 65535.0) as u16
    }

    /// 1000 bins mostly at -50 dB, with two spikes at 0 dB.
    fn signal_with_noise() -> Vec<u16> {
        let mut v = vec![db_to_mag(-50.0); 1000];
        v[100] = db_to_mag(0.0);
        v[500] = db_to_mag(0.0);
        v
    }

    #[test]
    fn mag_to_db_is_inverse_of_db_to_mag() {
        // u16 magnitude quantization: near 0 dB LSB ≈ 0.12 dB, and truncation
        // at low mags is coarser (≈0.3 dB around −70 dB) → small tolerance.
        for &db in &[-70.0, -50.0, -20.0, -6.0, 0.0] {
            let m = db_to_mag(db);
            let back = mag_to_db(m);
            assert!((back - db).abs() < 0.5, "db {db} → mag {m} → {back}");
        }
        // Bounds match canvas.rs:lin_to_db: 0 mag → −100 dB floor; full scale
        // is 0 dBFS (the +10 clamp only guards mags past full scale).
        assert!((mag_to_db(0) + 100.0).abs() < 1e-9);
        assert!(mag_to_db(65535).abs() < 1e-9);
    }

    #[test]
    fn auto_estimate_returns_median_and_peak() {
        let (f, c) = auto_estimate(&signal_with_noise()).unwrap();
        // 2 of 1000 bins are spikes, so the median is the noise level (−50,
        // ±u16 quantization); the max is the spike level (0). Headroom and
        // the peak-hold are applied downstream in `update_auto_scale`.
        assert!((f + 50.0).abs() < 2e-2, "median was {f}");
        assert!(c.abs() < 2e-2, "max was {c}");
    }

    #[test]
    fn estimate_is_unaffected_by_spike_count() {
        let mut v = signal_with_noise();
        v[250] = db_to_mag(0.0); // one more spike
        v[750] = db_to_mag(-10.0); // a weaker one too
        let (f, c) = auto_estimate(&v).unwrap();
        assert!((f + 50.0).abs() < 2e-2, "median was {f}");
        assert!(c.abs() < 2e-2, "max was {c}");
    }

    #[test]
    fn follower_deadband_holds() {
        let mut val = -55.0;
        // Target within the deadband → no move.
        assert!(!auto_step_follower(&mut val, -54.0, 0.3, 0.05, 1.0));
        assert_eq!(val, -55.0);
        assert!(!auto_step_follower(&mut val, -55.9, 0.3, 0.05, 1.0));
        assert_eq!(val, -55.0);
    }

    #[test]
    fn follower_attack_is_faster_than_release() {
        let mut a = -55.0;
        let mut r = -55.0;
        auto_step_follower(&mut a, -40.0, 0.30, 0.05, 1.0); // rise
        auto_step_follower(&mut r, -70.0, 0.30, 0.05, 1.0); // fall
        let a_move = (a - -55.0).abs();
        let r_move = (-55.0 - r).abs();
        assert!(
            a_move > r_move,
            "attack {a_move} should exceed release {r_move}"
        );
        // Both move monotonically toward the target (never overshoot).
        assert!(a < -40.0);
        assert!(r > -70.0);
    }

    #[test]
    fn follower_tracks_over_time_without_overshoot() {
        let mut val = -85.0;
        for _ in 0..10_000 {
            auto_step_follower(&mut val, -55.0, 0.05, 0.01, 1.0);
        }
        // Converged to within the deadband of the target, from below, never
        // overshooting (exponential approach is strictly monotone).
        assert!((val + 55.0).abs() <= 1.0 + 1e-9, "val {val}");
        assert!(
            val <= -55.0 + 1e-9,
            "should approach from below, not overshoot"
        );
    }

    #[test]
    fn enforce_min_span_expands_symmetrically() {
        let mut f = -40.0;
        let mut c = -30.0;
        enforce_min_span(&mut f, &mut c, 25.0);
        assert!((c - f - 25.0).abs() < 1e-9);
        assert!((f + 47.5).abs() < 1e-9, "floor was {f}");
        assert!((c - -22.5).abs() < 1e-9, "ceil was {c}");
        // A healthy span is untouched.
        let mut f2 = -85.0;
        let mut c2 = -15.0;
        enforce_min_span(&mut f2, &mut c2, 25.0);
        assert_eq!((f2, c2), (-85.0, -15.0));
    }

    #[test]
    fn auto_scale_moves_upward_toward_a_new_signal() {
        let sh = Shared::new();
        // A frame well outside the seed range (−85/−15): ceiling target is
        // peak(0) − headroom(5) = −5, floor (median −50) − 5 = −55, so both
        // edges must start closing the gap upward immediately.
        assert!(Shared::update_auto_scale(&sh, &signal_with_noise()));
        let (f, c) = sh.auto_scale();
        assert!(
            f > -85.0 + 0.5,
            "floor should rise from seed toward estimate, got {f}"
        );
        assert!(
            c > -15.0 + 0.5,
            "ceil should rise from seed toward estimate, got {c}"
        );
    }

    /// A signal "strobe" — strong, then weak, then strong again (a keyed CW
    /// tone / bursty digital signal, or anything flapping a few dB) — must
    /// NOT make the computed ceiling bounce up and down: the ceiling follower's
    /// slow release (α=0.02/frame) moves it only a fraction of a dB per keying
    /// burst, while the fast attack re-covers the small drop instantly.
    fn frame_at(peak_db: f64) -> Vec<u16> {
        let mut v = vec![db_to_mag(-50.0); 1000];
        if peak_db != f64::MIN {
            v[42] = db_to_mag(peak_db);
        }
        v
    }

    /// A CW-like strobe: the signal keys between −25 dB (on) and −28 dB (off)
    /// around a peak of −25 — i.e. within the 4 dB hold margin, which is the
    /// real "aggressive Morse / bursty signal" scenario. The ceiling must NOT
    /// track the −25 ↔ −28 keying; it must hold at the peak.
    #[test]
    fn strobing_signal_does_not_oscillate_the_ceiling() {
        let sh = Shared::new();

        // Establish the ceiling with a held −25 dB signal. The ceiling starts
        // above the target (seed −15 vs target −30) and closes it on the slow
        // release tail (0.1 dB/frame), so this needs a couple hundred frames.
        for _ in 0..250 {
            Shared::update_auto_scale(&sh, &frame_at(-25.0));
        }
        let c_settled = sh.auto_scale().1;
        // Steady state: floor has converged to the median-based target
        // (−50 − 5 headroom ≈ −55), so the 25 dB min-span expansion pins the
        // ceiling just above the headroom target (−25 − 5 = −30) — it lands at
        // −26.1 with the headroom's 5 dB preserved minus the span expansion.
        assert!(
            (c_settled + 26.1).abs() <= 1.0,
            "warmup not settled: {c_settled}"
        );

        // Now key it between −25 dB (on) and −28 dB (off), 10 frames each.
        // Both phases stay within the 4 dB hold margin of the −25 peak, so
        // the ceiling must stay frozen — the 3 dB keying must not strobe it.
        let mut hi = f64::MIN;
        let mut lo = f64::MAX;
        for _ in 0..10 {
            for _ in 0..10 {
                Shared::update_auto_scale(&sh, &frame_at(-25.0)); // on
                hi = hi.max(sh.auto_scale().1);
                lo = lo.min(sh.auto_scale().1);
            }
            for _ in 0..10 {
                Shared::update_auto_scale(&sh, &frame_at(-28.0)); // off
                hi = hi.max(sh.auto_scale().1);
                lo = lo.min(sh.auto_scale().1);
            }
        }
        let span = hi - lo;
        assert!(
            span < 1.0,
            "ceil strobed {lo}…{hi} (span {span:.2} dB; settled {c_settled})"
        );
    }

    /// A genuinely gone signal (falls well below the hold margin) must still
    /// release the ceiling — the ceiling may never be truly stuck above a
    /// signal that has disappeared.
    #[test]
    fn gone_signal_still_releases_the_ceiling() {
        let sh = Shared::new();
        // Hold the ceiling up with a −20 dB signal.
        for _ in 0..50 {
            Shared::update_auto_scale(&sh, &frame_at(-20.0));
        }
        let c_with = sh.auto_scale().1;
        // Now the signal is gone (median floor −50, max −50 is well below the
        // 4 dB hold margin of the peak, so it starts releasing after the
        // ~12-frame hold window).
        for _ in 0..500 {
            Shared::update_auto_scale(&sh, &frame_at(-50.0));
        }
        let c_after = sh.auto_scale().1;
        assert!(
            c_with - c_after > 5.0,
            "ceil did not release after signal gone: {c_with} → {c_after}"
        );
    }
}
