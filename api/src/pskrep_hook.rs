//! PSK Reporter spot posting for decoded digital-mode rows.
//!
//! Two halves:
//!
//! * [`Pskrep::spot`] (an [`hl2_common::SpotSink`] impl) — the pure
//!   selector + queue-append: given one decoded [`hl2_common::DecodedMessage`]
//!   (any mode: FT8 / FT4 / JS8 / …) plus the RF tune it was decoded at,
//!   derive the wire spot (caller, locator, absolute frequency, SNR, epoch,
//!   mode) and append it to the per-reporter queue. Frequency is
//!   `rf_hz + m.freq_hz` (wsjtx `m_freqNominalPeriod + audioFrequency`);
//!   epoch is `(slot_ms / 1000) + round(dt_sec)` (wsjtx `time_stamp`);
//!   mode is the message's own ADIF mode tag (`m.mode()`).
//!
//!   The per-mode **call/locator extraction** and its **self-spot
//!   suppression** is the `DecodedMessage::spot_fields` impl — so this sink
//!   is completely mode-agnostic (no `if FT8 … if JS8 … if FT4` branches).
//!   See `hl2/src/receiver/spot.rs` (WSJT) and
//!   `hl2/src/receiver/js8/decoder.rs` (JS8) for the per-mode selectors.
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

use hl2_common::{DecodedMessage, SpotSink};
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

    /// Our own station as a [`hl2_common::SpotStation`]: the per-mode
    /// `spot_fields` impls use this for self-spot suppression.
    pub fn spot_station(&self) -> hl2_common::SpotStation {
        hl2_common::SpotStation::new(self.station.callsign.clone(), self.station.grid.clone())
    }

    /// The mode-agnostic sink tail: given one decoded message (already in
    /// the form the per-mode selector wants to spot) and the passband
    /// centre it was decoded against, derive a wire spot and queue it.
    fn build_spot(&self, m: &dyn DecodedMessage, rf_hz: u32) -> Option<Spot> {
        if rf_hz == 0 {
            return None;
        }
        let f = m.spot_fields(&self.spot_station())?;
        Some(Spot {
            caller: f.caller,
            locator: f.locator,
            freq_hz: ((rf_hz as i64) + (m.freq_hz()).round() as i64).clamp(1, u32::MAX as i64)
                as u32,
            snr: (m.snr_db()).clamp(i8::MIN as f32, i8::MAX as f32) as i8,
            time_epoch: ((m.slot_ms() as i64 / 1000) + (m.dt_sec()).round() as i64)
                .clamp(0, u32::MAX as i64) as u32,
            mode: m.mode().to_string(),
        })
    }
}

impl SpotSink for Pskrep {
    fn spot(&mut self, m: &dyn DecodedMessage, rf_hz: u32) -> bool {
        if !self.enabled {
            return false;
        }
        match self.build_spot(m, rf_hz) {
            Some(s) => {
                let ok = self.reporter.add(s);
                if !ok {
                    dbg_log(&format!(
                        "spot dropped by reporter dedup/cap (caller={:?})",
                        m.spot_fields(&self.spot_station()).map(|f| f.caller)
                    ));
                }
                ok
            }
            None => {
                dbg_log(&format!(
                    "spot not spot-able: mode={} display={:?}",
                    m.mode(),
                    m.display()
                ));
                false
            }
        }
    }
}

fn dbg_log(s: &str) {
    if std::env::var("HL2_DEBUG").is_ok() {
        eprintln!("[PSKREP] {s}");
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use hl2::receiver::js8::msg::{DecodedFrame, FRAME_DIRECTED, FRAME_HEARTBEAT};
    use hl2::receiver::{Ft4Message, Ft8Message, Js8Message};
    use hl2_common::SpotSink;

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

    fn f4msg(text: &str) -> Ft4Message {
        Ft4Message {
            text: text.into(),
            freq_hz: 1_500.0,
            dt_sec: -0.5,
            snr_db: 12.0,
            slot_ms: 1_700_000_015_000,
        }
    }

    fn js8base() -> Js8Message {
        Js8Message {
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

    fn js8msg(callsign: &str, grid: Option<&str>) -> Js8Message {
        let mut m = js8base();
        m.frame.kind = FRAME_HEARTBEAT;
        m.frame.callsign = callsign.into();
        m.frame.grid = grid.map(str::to_string);
        m
    }

    fn js8dir(from: &str, to: &str, cmd: &str) -> Js8Message {
        let mut m = js8base();
        m.frame.kind = FRAME_DIRECTED;
        m.frame.callsign = from.into();
        m.frame.to = Some(to.into());
        m.frame.cmd = Some(cmd.into());
        m.frame.grid = None;
        m
    }

    fn js8data(text: &str) -> Js8Message {
        let mut m = js8base();
        m.frame.kind = DecodedFrame::KIND_DATA_JSC;
        m.frame.callsign = String::new();
        m.frame.grid = None;
        m.frame.text = text.into();
        m.frame.message = text.into();
        m
    }

    /// Drive one decoded message through the sink. Proves the sink only
    /// sees `&dyn DecodedMessage` + `rf_hz` (no per-mode branch) and that
    /// the golden wsjtx / js8call spot expectations still hold end-to-end.
    fn spot(m: &dyn DecodedMessage, rf_hz: u32) -> Option<Spot> {
        let mut p = Pskrep {
            station: st(),
            reporter: PskReporter::new(PskCfg::default()),
            enabled: true,
            desc_left: 3,
            seq: 0,
            obs: 0,
        };
        let ok = p.spot(m, rf_hz);
        assert_eq!(ok, !p.reporter.is_empty());
        assert!(p.reporter.len() <= 1);
        p.reporter.drain().into_iter().next()
    }

    #[test]
    fn ft8_cq_message_spots_call_and_grid() {
        let s = spot(&msg("CQ R7IW LN35"), 14_075_000).expect("CQ + grid");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "LN35");
        assert_eq!(s.freq_hz, 14_076_500); // tune + 1500 Hz offset
        assert_eq!(s.mode, "FT8");
        assert_eq!(s.time_epoch, 1_700_000_015 - 1); // slot start + dt (−0.5 s, rounded)
    }

    #[test]
    fn ft4_cq_message_spots_call_and_grid() {
        let s = spot(&f4msg("CQ R7IW LN35"), 14_075_000).expect("CQ + grid");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "LN35");
        assert_eq!(s.freq_hz, 14_076_500);
        assert_eq!(s.mode, "FT4");
    }

    #[test]
    fn ft4_self_spot_is_suppressed() {
        assert!(spot(&f4msg("CQ K9ABC FN31"), 14_075_000).is_none());
    }

    #[test]
    fn self_spot_is_suppressed() {
        assert!(spot(&msg("CQ K9ABC FN31"), 14_075_000).is_none());
    }

    #[test]
    fn bare_call_grid_spots() {
        let s = spot(&msg("K1ABC K9ABD PN32"), 7_074_000).expect("spots");
        assert_eq!(s.caller, "K9ABD");
        assert_eq!(s.locator, "PN32");
        assert_eq!(s.freq_hz, 7_075_500); // tune + 1500 Hz offset
    }

    #[test]
    fn no_tune_is_no_spot() {
        assert!(spot(&msg("CQ R7IW LN35"), 0).is_none());
        assert!(spot(&js8msg("R7IW", Some("LN35")), 0).is_none());
    }

    #[test]
    fn js8_heartbeat_spots_call_and_grid() {
        let s = spot(&js8msg("R7IW", Some("LN35")), 14_075_000).expect("spots");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "LN35");
        assert_eq!(s.freq_hz, 14_076_500);
        assert_eq!(s.mode, "JS8");
    }

    #[test]
    fn js8_heartbeat_without_grid_still_spots() {
        let s = spot(&js8msg("R7IW", None), 14_075_000).expect("spots");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "");
    }

    #[test]
    fn js8_directed_frame_spots_from_call_without_grid() {
        let s = spot(&js8dir("R7IW", "K9ABC", "SNR"), 14_075_000).expect("directed");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "");
        assert_eq!(s.mode, "JS8");
    }

    #[test]
    fn js8_data_frame_leading_call_spots() {
        let s = spot(&js8data("K1ABC: Hello there"), 14_075_000).expect("data CALL: lead");
        assert_eq!(s.caller, "K1ABC");
        assert_eq!(s.locator, "");
    }

    #[test]
    fn js8_data_frame_without_call_does_not_spot() {
        assert!(spot(&js8data("HELLO WORLD"), 14_075_000).is_none());
    }
}
