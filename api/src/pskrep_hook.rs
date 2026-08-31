//! PSK Reporter spot posting for decoded FT8 rows.
//!
//! Two halves:
//!
//! * [`ft8_spot`] — the pure selector: given one decoded
//!   [`hl2::receiver::Ft8Message`] plus the RF tune it was decoded at, decide
//!   whether it is spot-able and, if so, produce a [`pskrep::Spot`. This
//!   ports wsjtx's `MainWindow::pskPost` (widgets/mainwindow.cpp:7565) +
//!   `DecodedText::deCallAndGrid` (Decoder/decodedtext.cpp:212) + the
//!   `grid_regexp` (widgets/mainwindow.cpp:308):
//!
//!   * word1 of the 77-bit text is `CQ` / `QRZ` (optionally with a
//!     direction / `DX` / 3-digit suffix) or a bare callsign;
//!   * **caller** = the next word (wsjtx "word2"), **locator** = the one
//!     after it ("word3"; "R" → "word4");
//!   * post only when the locator is a valid grid square
//!     (`[A-R]{2}[0-9]{2}([A-X]{2})?`, not RR73) **or** the text contains
//!     `" CQ "`;
//!   * suppress self-spots: text containing both our base callsign and our
//!     4-char grid (wsjtx mainwindow.cpp:7568).
//!
//! * [`run_sender`] — the UDP send loop (60 s cadence): drains
//!   [`PskReporter::drain`], encodes with [`pskrep::wire::build_packets`],
//!   sends to the collector. Sequence / observation-domain id / the
//!   "send descriptors on the next 3 reports" counter (wsjtx
//!   `PSKReporter.cpp` — `send_descriptors_ = 3` at startup and on every
//!   reconnect) live in [`Pskrep::state`].
//!
//! Transport and collector address follow wsjtx:
//! `report.pskreporter.info:4739` (PSKReporter.cpp:35-37).

use std::sync::{Arc, Mutex};

use pskrep::wire::{MAX_UDP_IPFIX_PAYLOAD_BYTES, build_packets};
use pskrep::{Clock, PskCfg, PskReporter, Spot, Station};

pub type SharedPsk = Arc<Mutex<Pskrep>>;

/// Collector host
pub const DEFAULT_COLLECTOR: &str = "report.pskreporter.info:4739";

#[derive(Debug)]
pub struct Pskrep {
    pub station: Station,
    pub reporter: PskReporter,
    /// `PSK_CALL` was given — spot posting is active.
    pub enabled: bool,
    /// Descriptors still to embed on the next reports (wsjtx
    /// `send_descriptors_`: 3 at startup and after every reconnect).
    pub desc_left: u32,
    /// Running IPFIX sequence counter (wsjtx `sequence_number_`).
    pub seq: u32,
    /// Observation-domain id (wsjtx `observation_id_` — random at startup).
    pub obs: u32,
}

impl Pskrep {
    /// Build from the `PSK_*` environment (see AGENTS.md):
    /// `PSK_CALL`, `PSK_GRID`, `PSK_ANTENNA`, `PSK_RIG`.
    /// Empty `PSK_CALL` → `enabled == false` (decode rows are still
    /// broadcast, just not posted. the send loop never puts datagrams on
    /// the wire).
    pub fn from_env() -> Self {
        let v = |k: &str| std::env::var(k).unwrap_or_default();
        let station = Station {
            callsign: v("PSK_CALL"),
            grid: v("PSK_GRID"),
            program_info: format!("hl2-api/{}", env!("CARGO_PKG_VERSION")),
            antenna: v("PSK_ANTENNA"),
            rig_info: v("PSK_RIG"),
        };
        let enabled = !station.callsign.is_empty();
        let obs = {
            let n = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u32)
                .unwrap_or(0);
            n.wrapping_mul(0x9E37_79B9).wrapping_add(0x85_EB_CA_6B)
        };
        Self {
            station,
            reporter: PskReporter::new(PskCfg::default()),
            enabled,
            desc_left: 3,
            seq: 0,
            obs,
        }
    }

    /// Spot-selection + queue-append for one decoded FT8 row at RF tune
    /// `rf_hz` (the NCO / passband centre the row was decoded against).
    /// Returns `true` if the spot is now pending.
    pub fn add_ft8(&mut self, m: &hl2::receiver::Ft8Message, rf_hz: u32) -> bool {
        if !self.enabled || rf_hz == 0 {
            if self.enabled {
                dbg_log(&format!("ft8 spot dropped: no rf tune (rf_hz={rf_hz})"));
            }
            return false;
        }
        match ft8_spot(m, rf_hz, &self.station) {
            Some(s) => {
                let ok = self.reporter.add(s);
                if !ok {
                    dbg_log("ft8 spot dropped by reporter dedup/cap");
                }
                ok
            }
            None => {
                dbg_log(&format!("ft8 spot not spot-able: text={:?}", m.text));
                false
            }
        }
    }

    /// Spot-selection + queue-append for one decoded FT4 row at RF tune
    /// `rf_hz`. FT4 shares FT8's free-text message format (callsign +
    /// grid / CQ), so this mirrors `add_ft8`. Returns `true` if the spot is
    /// now pending.
    pub fn add_ft4(&mut self, m: &hl2::receiver::Ft4Message, rf_hz: u32) -> bool {
        if !self.enabled || rf_hz == 0 {
            if self.enabled {
                dbg_log(&format!("ft4 spot dropped: no rf tune (rf_hz={rf_hz})"));
            }
            return false;
        }
        match ft4_spot(m, rf_hz, &self.station) {
            Some(s) => {
                let ok = self.reporter.add(s);
                if !ok {
                    dbg_log("ft4 spot dropped by reporter dedup/cap");
                }
                ok
            }
            None => {
                dbg_log(&format!("ft4 spot not spot-able: text={:?}", m.text));
                false
            }
        }
    }

    /// Spot-selection + queue-append for one decoded JS8 frame at RF tune
    /// `rf_hz`. Mirrors js8call's spotting (mainwindow.cpp `logCallActivity`
    /// / `processSpots`): a spot is any frame with a **sender callsign**, the
    /// locator is optional. That covers heartbeats + compounds (call + grid),
    /// directed frames (the `from` callsign, no grid), and data frames that
    /// lead their text with `CALL:`. Returns `true` if the spot is now
    /// pending.
    pub fn add_js8(&mut self, m: &hl2::receiver::Js8Message, rf_hz: u32) -> bool {
        if !self.enabled || rf_hz == 0 {
            if self.enabled {
                let f = &m.frame;
                dbg_log(&format!(
                    "js8 spot dropped: no rf tune (rf_hz={rf_hz}) call={:?} grid={:?}",
                    f.callsign, f.grid
                ));
            }
            return false;
        }
        match js8_spot(m, rf_hz, &self.station) {
            Some(s) => {
                let caller = s.caller.clone();
                let ok = self.reporter.add(s);
                if !ok {
                    dbg_log(&format!(
                        "js8 spot dropped by reporter dedup/cap (caller={caller})"
                    ));
                }
                ok
            }
            None => {
                let f = &m.frame;
                dbg_log(&format!(
                    "js8 spot not spot-able: kind={} call={:?} grid={:?} self_call={:?} self_grid={:?}",
                    f.kind, f.callsign, f.grid, self.station.callsign, self.station.grid
                ));
                false
            }
        }
    }
}

fn is_cq_tag(w: &str) -> bool {
    let b = w.as_bytes();
    (1..=4).contains(&b.len()) && b.iter().all(|c| c.is_ascii_uppercase())
        || b.len() == 3 && b.iter().all(|c| c.is_ascii_digit())
}

/// Spot-selection per wsjtx `MainWindow::pskPost` + `deCallAndGrid`.
///
/// `rf_hz` is the passband centre (the vrx slot's tune); the row's
/// `freq_hz` is the tone-0 offset in the audio band, so the spot frequency
/// is `rf_hz + freq_hz` (wsjtx `m_freqNominalPeriod + audioFrequency`).
pub fn ft8_spot(m: &hl2::receiver::Ft8Message, rf_hz: u32, st: &Station) -> Option<Spot> {
    wsjt_text_spot(
        m.text.as_str(),
        m.freq_hz,
        m.dt_sec,
        m.snr_db,
        m.slot_ms,
        rf_hz,
        st,
        "FT8",
    )
}

/// Spot-selection for one decoded FT4 row, mirroring `ft8_spot`.
///
/// FT4 uses the same WSJT-77 free-text message format as FT8 (callsign +
/// grid / CQ), so the identical token parsing applies — only the mode tag
/// on the spot differs.
pub fn ft4_spot(m: &hl2::receiver::Ft4Message, rf_hz: u32, st: &Station) -> Option<Spot> {
    wsjt_text_spot(
        m.text.as_str(),
        m.freq_hz,
        m.dt_sec,
        m.snr_db,
        m.slot_ms,
        rf_hz,
        st,
        "FT4",
    )
}

/// The shared free-text WSJT spot selector (FT8 / FT4).
///
/// `rf_hz` is the passband centre (the vrx slot's tune); the row's
/// `freq_hz` is the tone-0 offset in the audio band, so the spot frequency
/// is `rf_hz + freq_hz` (wsjtx `m_freqNominalPeriod + audioFrequency`).
fn wsjt_text_spot(
    text: &str,
    freq_hz: f32,
    dt_sec: f32,
    snr_db: f32,
    slot_ms: u64,
    rf_hz: u32,
    st: &Station,
    mode: &'static str,
) -> Option<Spot> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 2 {
        dbg_log(&format!("{mode} spot reject: too few words in {text:?}"));
        return None;
    }
    let mut i = 1;
    if matches!(words[0], "CQ" | "QRZ") && i + 1 < words.len() && is_cq_tag(words[i]) {
        i += 1;
    }
    let call = words.get(i).copied()?;
    let mut grid = words.get(i + 1).copied().unwrap_or_default();
    if grid == "R" {
        grid = words.get(i + 2).copied().unwrap_or_default();
    }
    // Self-spotting prevention
    let base = base_callsign(&st.callsign);
    let g4: String = st.grid.chars().take(4).collect();
    if !base.is_empty() && !g4.is_empty() && text.contains(base) && text.contains(g4.as_str()) {
        dbg_log(&format!("{mode} spot reject: self-spot in {text:?}"));
        return None;
    }
    // Spot only if the locator is a grid square or the message is a CQ
    if !(grid_is_square(grid) || text.contains(" CQ ") || text == "CQ" || text.starts_with("CQ ")) {
        dbg_log(&format!(
            "{mode} spot reject: no grid square (call={call:?} grid={grid:?} text={text:?})"
        ));
        return None;
    }
    Some(Spot {
        caller: call.to_ascii_uppercase(),
        locator: grid.to_ascii_uppercase(),
        freq_hz: ((rf_hz as i64) + (freq_hz).round() as i64).clamp(1, u32::MAX as i64) as u32,
        snr: (snr_db).clamp(i8::MIN as f32, i8::MAX as f32) as i8,
        time_epoch: ((slot_ms as i64 / 1000) + (dt_sec).round() as i64).clamp(0, u32::MAX as i64)
            as u32,
        mode: mode.into(),
    })
}

/// Spot-selection for one decoded JS8 frame, mirroring `ft8_spot` but driven
/// by the structured fields (JS8 is a protocol, not free text).
///
/// js8call spots **any** received frame that carries a sender — the locator
/// is only added when a grid square is present (mainwindow.cpp
/// `logCallActivity` + `processSpots`: directed frames and `CALL:`-prefixed
/// data frames are queued without a grid). So a spot requires a non-empty
/// **caller**, with `locator = grid` when that is a valid square.
///
/// The caller is taken from:
///   * the structured `callsign` field — heartbeats / compounds /
///     compound-directed (`callsign` + optional `grid`) and directed frames
///     (`callsign` = the `from`) which carry no grid;
///   * else the first word of the free-text data payload only when it leads
///     the message as `CALL:` (js8call mainwindow.cpp:8292-8306), the other
///     common case; e.g. `K1ABC: ...`.
///
/// Self-spotting: suppress when the caller matches our base callsign **and**
/// either (a) the locator matches our 4-char grid, or (b) the frame carries
/// no locator at all — this is how our *own* heartbeats / directed / data
/// messages would look. A different grid with the same base call still spots
/// (mobile on our base).
pub fn js8_spot(m: &hl2::receiver::Js8Message, rf_hz: u32, st: &Station) -> Option<Spot> {
    use hl2::receiver::js8::msg::{
        FRAME_COMPOUND, FRAME_COMPOUND_DIRECTED, FRAME_DIRECTED, FRAME_HEARTBEAT,
    };

    let f = &m.frame;
    let kind = f.kind;

    // Choose the sender: structured fields first, then a leading `CALL:`
    // word on data frames.
    let structured_call = f.callsign.trim();
    let call = if matches!(
        kind,
        FRAME_HEARTBEAT | FRAME_COMPOUND | FRAME_COMPOUND_DIRECTED | FRAME_DIRECTED
    ) && !structured_call.is_empty()
    {
        Some(structured_call)
    } else {
        data_frame_caller(f)
    };
    let Some(call) = call else {
        dbg_log(&format!(
            "js8_spot reject: no sender (kind={kind} call={:?} text={:?})",
            f.callsign, f.text
        ));
        return None;
    };

    // Locator: only a valid square, never a command name; otherwise empty.
    let grid = f.grid.as_deref().unwrap_or("").trim();
    if !grid.is_empty() && !grid_is_square(grid) {
        dbg_log(&format!(
            "js8_spot: non-square grid {grid:?} → empty locator"
        ));
    }
    let grid: Option<&str> = grid_is_square(grid).then_some(grid);

    // Self-spot suppression: caller matches our base call **and** the grid
    // matches ours too — or the frame carries no grid at all (directed /
    // data-frame `CALL:`), in which case the matching callsign is enough
    // (that is how *our* directed messages would look). A different grid
    // with the same base call (a mobile on our base) still spots.
    let base_call = base_callsign(&st.callsign);
    if !base_call.is_empty() {
        let g4: String = st.grid.chars().take(4).collect();
        let caller_matches = call.eq_ignore_ascii_case(base_call);
        let grid_matches = grid.is_some_and(|g| g.eq_ignore_ascii_case(g4.as_str()));
        if caller_matches && (grid.is_none() || grid_matches) {
            dbg_log(&format!(
                "js8_spot reject: self-spot (call={call:?} ~ {base_call:?}, grid={grid:?} ~ {g4:?})"
            ));
            return None;
        }
    }

    Some(Spot {
        caller: call.to_ascii_uppercase(),
        locator: grid.map(str::to_ascii_uppercase).unwrap_or_default(),
        freq_hz: ((rf_hz as i64) + (m.freq_hz).round() as i64).clamp(1, u32::MAX as i64) as u32,
        snr: (m.snr_db).clamp(i8::MIN as f32, i8::MAX as f32) as i8,
        time_epoch: ((m.slot_ms as i64 / 1000) + (m.dt_sec).round() as i64)
            .clamp(0, u32::MAX as i64) as u32,
        // Plain "JS8" — js8call posts a single mode name (mainwindow.cpp:9611)
        // regardless of speed, so the collector's mode filter matches. We do
        // not post speed-specific JS8A/B/C/E (those are not a filter option).
        mode: "JS8".into(),
    })
}

/// For a JS8 free-text (data) frame, the leading sender when the payload
/// starts a directed message as `CALL:` — js8call's data-frame spotting rule
/// (mainwindow.cpp:8292-8306: first word callsign immediately followed by
/// `:`). Returns `None` when the text does not follow that shape.
fn data_frame_caller(f: &hl2::receiver::js8::msg::DecodedFrame) -> Option<&str> {
    let t = f.text.trim();
    let pos = t.find(':')?;
    let head = t[..pos].trim();
    let w: Vec<&str> = head.split_whitespace().collect();
    if w.len() != 1 {
        return None;
    }
    let call = w[0];
    let b = call.as_bytes();
    let ok = !call.is_empty()
        && b.iter()
            .all(|c| c.is_ascii_alphanumeric() || *c == b'/' || *c == b'.' || *c == b'@');
    if ok { Some(call) } else { None }
}

/// Mobile / portable prefix stripping
fn base_callsign(call: &str) -> &str {
    match call.rfind('/') {
        Some(j) => &call[j + 1..],
        None => call,
    }
}

fn grid_is_square(g: &str) -> bool {
    let b = g.as_bytes();
    if b.len() != 4 && b.len() != 6 {
        return false;
    }
    let sq = |c: u8| c.to_ascii_uppercase().is_ascii_uppercase() && c.to_ascii_uppercase() <= b'R';
    if !sq(b[0]) || !sq(b[1]) || !b[2].is_ascii_digit() || !b[3].is_ascii_digit() {
        return false;
    }
    if b.len() == 6 {
        let sub =
            |c: u8| c.to_ascii_uppercase().is_ascii_uppercase() && c.to_ascii_uppercase() <= b'X';
        if !sub(b[4]) || !sub(b[5]) {
            return false;
        }
    }
    !(b[0].to_ascii_uppercase() == b'R'
        && b[1].to_ascii_uppercase() == b'R'
        && b[2].to_ascii_uppercase() == b'7'
        && b[3].to_ascii_uppercase() == b'3')
}

pub async fn run_sender(state: SharedPsk) {
    use std::time::Duration;
    use tokio::net::{UdpSocket, lookup_host};

    let addr_str = std::env::var("PSKREP_ADDR").unwrap_or_else(|_| DEFAULT_COLLECTOR.to_string());

    loop {
        let addr = match lookup_host(&addr_str).await {
            Ok(mut it) => match it.next() {
                Some(a) => a,
                None => {
                    dbg_log(&format!("pskrep: {addr_str} resolved to no addresses"));
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    continue;
                }
            },
            Err(e) => {
                dbg_log(&format!("pskrep: lookup {addr_str}: {e}"));
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };
        let socket = match UdpSocket::bind("0.0.0.0:0").await {
            Ok(s) => s,
            Err(e) => {
                dbg_log(&format!("pskrep: bind: {e}"));
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };
        let _ = socket.connect(addr).await;
        {
            let mut g = state.lock().unwrap();
            g.desc_left = 3; // reconnect → wsjtx PSKReporter.cpp:152
        }
        dbg_log(&format!("pskrep: connected to {addr}"));

        let mut tick = tokio::time::interval(Duration::from_secs(60));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let _ = tick.tick().await;
            let (station, spots, include_descriptors, seq, obs) = {
                let mut g = state.lock().unwrap();
                if g.reporter.is_empty() {
                    continue;
                }
                let inc = g.desc_left > 0;
                if inc {
                    g.desc_left -= 1;
                }
                (g.station.clone(), g.reporter.drain(), inc, g.seq, g.obs)
            };
            let packets = build_packets(
                &station,
                &spots,
                include_descriptors,
                seq,
                obs,
                Clock::Real.now(),
                MAX_UDP_IPFIX_PAYLOAD_BYTES,
            );
            let mut seq_adv = 0u32;
            let mut all_sent = true;
            for p in &packets {
                match socket.send(&p.payload).await {
                    Ok(_) => seq_adv += p.spot_count as u32,
                    Err(e) => {
                        dbg_log(&format!("pskrep: send: {e}"));
                        all_sent = false;
                        break;
                    }
                }
            }
            if all_sent {
                dbg_log(&format!(
                    "pskrep: sent {} spot(s) in {} packet(s)",
                    spots.len(),
                    packets.len()
                ));
                let mut g = state.lock().unwrap();
                g.seq = seq.checked_add(seq_adv).unwrap_or(seq_adv);
            } else {
                // Connected-UDP send failures are rare (ECONNREFUSED); give
                // the collector a moment, then the outer loop reconnects.
                tokio::time::sleep(Duration::from_secs(1)).await;
                break;
            }
        }
    }
}

fn dbg_log(s: &str) {
    if std::env::var("HL2_DEBUG").is_ok() {
        eprintln!("[PSKREP] {s}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hl2::receiver::Ft8Message;

    fn st() -> Station {
        Station {
            callsign: "K9ABC".into(),
            grid: "FN31".into(),
            program_info: "hl2-api/test".into(),
            antenna: String::new(),
            rig_info: String::new(),
        }
    }

    fn msg(text: &str) -> Ft8Message {
        Ft8Message {
            text: text.into(),
            freq_hz: 1_500.0,
            dt_sec: -0.5,
            snr_db: 12.0,
            slot_ms: 1_700_000_015_000,
        }
    }

    #[test]
    fn cq_message_spots_call_and_grid() {
        let s = ft8_spot(&msg("CQ R7IW LN35"), 14_075_000, &st()).expect("CQ + grid spots");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "LN35");
        assert_eq!(s.freq_hz, 14_076_500);
        assert_eq!(s.mode, "FT8");
        assert_eq!(s.time_epoch, 1_700_000_015 - 1); // slot start + dt (−0.5 s, rounded)
    }

    fn f4msg(text: &str) -> hl2::receiver::Ft4Message {
        hl2::receiver::Ft4Message {
            text: text.into(),
            freq_hz: 1_500.0,
            dt_sec: -0.5,
            snr_db: 12.0,
            slot_ms: 1_700_000_015_000,
        }
    }

    #[test]
    fn ft4_cq_message_spots_call_and_grid() {
        let s = ft4_spot(&f4msg("CQ R7IW LN35"), 14_075_000, &st()).expect("CQ + grid spots");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "LN35");
        assert_eq!(s.freq_hz, 14_076_500);
        assert_eq!(s.mode, "FT4");
    }

    #[test]
    fn ft4_self_spot_is_suppressed() {
        assert!(ft4_spot(&f4msg("CQ K9ABC FN31"), 14_075_000, &st()).is_none());
        assert!(
            ft4_spot(
                &f4msg("CQ K9ABC FN31"),
                14_075_000,
                &Station {
                    grid: "PM95".into(),
                    ..st()
                },
            )
            .is_some()
        );
    }

    #[test]
    fn cq_dx_tag_is_eaten_by_word1() {
        let s = ft8_spot(&msg("CQ DX R6WA LN32"), 14_075_000, &st()).expect("CQ DX spots");
        assert_eq!(s.caller, "R6WA");
        assert_eq!(s.locator, "LN32");
    }

    #[test]
    fn cq_number_tag_is_eaten_by_word1() {
        let s = ft8_spot(&msg("CQ 001 3Y0Z JD34"), 14_075_000, &st());
        // "001" is a 3-digit CQ tag → word1 = "CQ 001", caller 3Y0Z.
        assert_eq!(s.as_ref().map(|s| s.caller.clone()), Some("3Y0Z".into()));
        assert_eq!(s.as_ref().map(|s| s.locator.clone()), Some("JD34".into()));
    }

    #[test]
    fn bare_call_grid_spots() {
        // None of station's base call (K9ABC) or 4-char grid (FN31) appears
        // in the text, so the self-spot guard does not fire.
        let s = ft8_spot(&msg("K1ABC K9ABD PN32"), 7_074_000, &st()).expect("spots");
        assert_eq!(s.caller, "K9ABD");
        assert_eq!(s.locator, "PN32");
        assert_eq!(s.freq_hz, 7_075_500); // tune + 1500 Hz offset
    }

    #[test]
    fn reply_without_grid_does_not_spot() {
        // "JA1ABC 3Y0Z -12": caller 3Y0Z, grid "-12" — not a square, no CQ.
        assert!(ft8_spot(&msg("JA1ABC 3Y0Z -12"), 14_075_000, &st()).is_none());
    }

    #[test]
    fn self_spot_is_suppressed() {
        // text contains our base call K9ABC *and* our grid FN31.
        assert!(ft8_spot(&msg("CQ K9ABC FN31"), 14_075_000, &st()).is_none());
        // but the same message from a different grid (POTA suffix style) spots.
        assert!(
            ft8_spot(
                &msg("CQ K9ABC FN31"),
                14_075_000,
                &Station {
                    grid: "PM95".into(),
                    ..st()
                }
            )
            .is_some()
        );
        // and the mobile-prefix base call (W1AW/K9ABC) still suppresses K9ABC.
        assert!(
            ft8_spot(
                &msg("CQ K9ABC FN31"),
                14_075_000,
                &Station {
                    callsign: "W1AW/K9ABC".into(),
                    ..st()
                }
            )
            .is_none()
        );
    }

    #[test]
    fn qrz_message_spots() {
        let s = ft8_spot(&msg("QRZ K1ABC PM95"), 14_075_000, &st()).expect("QRZ spots");
        assert_eq!(s.caller, "K1ABC");
        assert_eq!(s.locator, "PM95");
    }

    #[test]
    fn grid_square_rules() {
        assert!(grid_is_square("LN35"));
        assert!(grid_is_square("ln35"));
        assert!(grid_is_square("FN42ur".to_uppercase().as_str()));
        assert!(!grid_is_square("RR73"));
        assert!(!grid_is_square("12"));
        assert!(!grid_is_square("LN3"));
        assert!(!grid_is_square("S1")); // S is outside A-R
        assert!(!grid_is_square("LN35Y")); // 5 chars
        assert!(!grid_is_square("+10"));
        assert!(!grid_is_square("RR73AB")); // starts with RR73 → excluded by the lookahead
    }

    #[test]
    fn free_text_with_cq_spots() {
        // "HELLO FT8 WLD" style free text: caller = word-after-CQ… free text
        // has no structured words — wsjtx still posts when " CQ " appears
        // (word2 = the word after word1).
        let s = ft8_spot(&msg("CQ 73 W1AW FN42"), 14_075_000, &st());
        // "73" is not a CQ tag (2 chars) → word2 = "73" (caller), word3 =
        // "W1AW" (locator), exactly wsjtx's tokens_re outcome. The *spot*
        // condition is still met via " CQ " (wsjtx mainwindow.cpp:7591).
        let s = s.expect("CQ-phrase spots");
        assert_eq!(s.caller, "73");
        assert_eq!(s.locator, "W1AW");
    }

    use hl2::receiver::js8::msg::{DecodedFrame, FRAME_COMPOUND, FRAME_DIRECTED, FRAME_HEARTBEAT};

    fn js8_base() -> hl2::receiver::Js8Message {
        hl2::receiver::Js8Message {
            frame: DecodedFrame {
                kind: FRAME_HEARTBEAT,
                callsign: String::new(),
                to: None,
                grid: None,
                cmd: None,
                num: None,
                bits3: 0,
                is_alt: false,
                text: String::new(),
                message: String::new(),
            },
            submode: "JS8A",
            mode_id: 0,
            freq_hz: 1_500.0,
            dt_sec: 0.48,
            snr_db: 11.0,
            slot_ms: 1_700_000_015_000,
        }
    }

    fn js8msg(callsign: &str, grid: Option<&str>) -> hl2::receiver::Js8Message {
        let mut m = js8_base();
        m.frame.kind = FRAME_HEARTBEAT;
        m.frame.callsign = callsign.into();
        m.frame.grid = grid.map(str::to_string);
        m
    }

    fn js8_dir(from: &str, to: &str, cmd: &str) -> hl2::receiver::Js8Message {
        let mut m = js8_base();
        m.frame.kind = FRAME_DIRECTED;
        m.frame.callsign = from.into();
        m.frame.to = Some(to.into());
        m.frame.cmd = Some(cmd.into());
        m.frame.grid = None;
        m
    }

    fn js8_compound(callsign: &str, grid: &str) -> hl2::receiver::Js8Message {
        let mut m = js8_base();
        m.frame.kind = FRAME_COMPOUND;
        m.frame.callsign = callsign.into();
        m.frame.grid = Some(grid.into());
        m
    }

    fn js8_data(text: &str) -> hl2::receiver::Js8Message {
        let mut m = js8_base();
        m.frame.kind = DecodedFrame::KIND_DATA_JSC;
        m.frame.callsign = String::new();
        m.frame.grid = None;
        m.frame.text = text.into();
        m.frame.message = text.into();
        m
    }

    #[test]
    fn js8_heartbeat_spots_call_and_grid() {
        let s = js8_spot(&js8msg("R7IW", Some("LN35")), 14_075_000, &st()).expect("spots");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "LN35");
        assert_eq!(s.freq_hz, 14_076_500);
        assert_eq!(s.mode, "JS8");
    }

    #[test]
    fn js8_heartbeat_without_grid_still_spots() {
        // js8call spots any frame that carries a sender, grid optional.
        let s = js8_spot(&js8msg("R7IW", None), 14_075_000, &st()).expect("spots");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "");
    }

    #[test]
    fn js8_directed_frame_spots_from_call_without_grid() {
        let s = js8_spot(&js8_dir("R7IW", "K9ABC", "SNR"), 14_075_000, &st()).expect("directed");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "");
        assert_eq!(s.mode, "JS8");
    }

    #[test]
    fn js8_compound_spots_call_and_grid() {
        let s = js8_spot(&js8_compound("R7IW", "LN35"), 14_075_000, &st()).expect("compound");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "LN35");
    }

    #[test]
    fn js8_data_frame_leading_call_spots() {
        let s = js8_spot(&js8_data("K1ABC: Hello there"), 14_075_000, &st())
            .expect("data CALL: lead spots");
        assert_eq!(s.caller, "K1ABC");
        assert_eq!(s.locator, "");
    }

    #[test]
    fn js8_data_frame_without_call_does_not_spot() {
        // No leading `CALL:` — e.g. a free-text line not addressed to anyone.
        assert!(js8_spot(&js8_data("HELLO WORLD"), 14_075_000, &st()).is_none());
    }

    #[test]
    fn js8_bad_grid_becomes_empty_locator() {
        // A non-square grid is not a usable locator; the spot still goes
        // through on the callsign.
        let s = js8_spot(&js8msg("K1ABC", Some("FN4")), 14_075_000, &st()).expect("spots");
        assert_eq!(s.caller, "K1ABC");
        assert_eq!(s.locator, "");
    }

    #[test]
    fn js8_no_call_does_not_spot() {
        let mut m = js8_base();
        m.frame.kind = FRAME_HEARTBEAT;
        m.frame.callsign = String::new();
        assert!(js8_spot(&m, 14_075_000, &st()).is_none());
    }

    #[test]
    fn js8_self_spot_is_suppressed() {
        // callsign + grid both match ours → suppress.
        assert!(js8_spot(&js8msg("K9ABC", Some("FN31")), 14_075_000, &st()).is_none());
        // different grid → spots.
        assert!(js8_spot(&js8msg("K9ABC", Some("PM95")), 14_075_000, &st()).is_some());
    }

    #[test]
    fn js8_self_directed_is_suppressed() {
        // Our own base call, no grid (directed) → self-spot, suppress.
        assert!(js8_spot(&js8_dir("K9ABC", "R7IW", "SNR"), 14_075_000, &st()).is_none());
    }
}
