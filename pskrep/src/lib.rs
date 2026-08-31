//! PSK Reporter client: the wire encoder ([`wire`]) plus the spot queue /
//! dedup ([`PskReporter`]).
//!
//! Mode-agnostic: a spot is just `(caller, locator, freq, snr, time, mode)`
//! — any digital mode that can produce a decoded row can feed one in. The
//! wire format is PSK Reporter's IPFIX dialect (see [`wire`] and wsjtx
//! `Network/PSKReporterIPFIX.cpp` for the byte reference). Transport (UDP
//! to `report.pskreporter.info:4739`) is the caller's responsibility.

pub mod wire;

pub use wire::{MAX_TCP_IPFIX_PAYLOAD_BYTES, MAX_UDP_IPFIX_PAYLOAD_BYTES, Packet};

use std::collections::HashMap;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

pub const CALLER_LIMIT: usize = 32;
pub const LOCATOR_LIMIT: usize = 16;
pub const MODE_LIMIT: usize = 16;
pub const PROGRAM_INFO_LIMIT: usize = 80;
pub const ANTENNA_LIMIT: usize = 128;
pub const RIG_INFO_LIMIT: usize = 128;

/// A decoded spot: one decoded row from any digital mode.
#[derive(Debug, Clone, PartialEq)]
pub struct Spot {
    /// The decoded (spotting) callsign.
    pub caller: String,
    /// The decoded grid square, or empty when the message carried none.
    pub locator: String,
    /// Absolute RF frequency, Hz.
    pub freq_hz: u32,
    /// SNR as the decoder reported it, dB (the wire field is `i8`).
    pub snr: i8,
    /// UTC epoch seconds of the signal start (slot start + dt).
    pub time_epoch: u32,
    /// ADIF mode string.
    pub mode: String,
}

/// Station identity for the receiver-information record.
#[derive(Debug, Clone, PartialEq)]
pub struct Station {
    pub callsign: String,
    pub grid: String,
    pub program_info: String,
    pub antenna: String,
    pub rig_info: String,
}

/// Clock abstraction so tests can drive the wall clock.
#[derive(Debug, Clone, Copy)]
pub enum Clock {
    /// `SystemTime::now()`.
    Real,
    /// Fixed epoch-seconds value (tests).
    Fixed(u32),
}

impl Clock {
    pub fn now(&self) -> u32 {
        match self {
            Self::Real => SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as u32)
                .unwrap_or(0),
            Self::Fixed(t) => *t,
        }
    }
}

/// The amateur band a frequency falls in, as a small index. Used to build
/// the per-(callsign, band) dedup key so that the same DX on *different*
/// bands (e.g. 20 m vs 40 m — normal for a multi-vrx station) is posted on
/// each, while the same DX on the *same* band is not double-posted.
///
/// Only HF distinctions matter for dedup (anything ≥ 49 MHz is exempt from
/// dedup entirely, wsjtx `PSKReporter.cpp:382`), but the buckets cover the
/// whole range so the key is well-defined everywhere.
fn band_of(freq_hz: u32) -> u32 {
    match freq_hz {
        f if f < 3_500_000 => 0,    // 160 m
        f if f < 5_000_000 => 1,    // 80 m
        f if f < 7_000_000 => 2,    // 60 m
        f if f < 10_000_000 => 3,   // 40 m
        f if f < 14_000_000 => 4,   // 30 m
        f if f < 18_000_000 => 5,   // 20 m
        f if f < 21_000_000 => 6,   // 17 m
        f if f < 24_000_000 => 7,   // 15 m
        f if f < 28_000_000 => 8,   // 12 m
        f if f < 29_700_000 => 9,   // 10 m
        f if f < 54_000_000 => 10,  // 6 m / below
        f if f < 90_000_000 => 11,  // 4 m
        f if f < 144_000_000 => 12, // 2 m / below
        f if f < 174_000_000 => 13, // 2 m
        _ => 14,                    // 70 cm and up
    }
}

/// Queue / dedup policy for one reporter instance.
#[derive(Debug, Clone)]
pub struct PskCfg {
    /// Re-spot suppression window per (callsign, band), in seconds.
    /// (wsjtx `PSKReporter.cpp:42` — `CACHE_TIMEOUT`.)
    pub dedup_ttl_secs: u32,
    /// Cache entries older than this (seconds) are pruned on every add.
    /// (wsjtx `PSKReporter.cpp:418` — literal `600`.)
    pub cache_prune_secs: u32,
    /// Maximum pending spots before the head is dropped.
    /// (wsjtx `PSKReporter.cpp:41` — `MAX_PENDING_SPOTS`.)
    pub max_pending: usize,
    pub clock: Clock,
}

impl Default for PskCfg {
    fn default() -> Self {
        Self {
            dedup_ttl_secs: 300,
            cache_prune_secs: 600,
            max_pending: 2048,
            clock: Clock::Real,
        }
    }
}

/// Queue + dedup of pending spots, in arrival order.
///
/// Dedup is per **(callsign, band)** — a callsign has its own re-spot window
/// on each band independently (so a multi-receiver station running the same
/// band on several VRXes collapses to one spot, but the same DX on a
/// different band is posted too). The key is case-insensitive on the call.
/// TTL-bounded, and disabled at / above 49 MHz (wsjtx
/// `PSKReporter.cpp:382` — VHF/UHF spots always go through). The queue is
/// capped at `cfg.max_pending`; overflow drops oldest first and
/// [`PskReporter::add`] reports `false`.
#[derive(Debug)]
pub struct PskReporter {
    cfg: PskCfg,
    cache: HashMap<String, u32>,
    queue: VecDeque<Spot>,
}

impl PskReporter {
    pub fn new(cfg: PskCfg) -> Self {
        Self {
            cfg,
            cache: HashMap::new(),
            queue: VecDeque::with_capacity(64),
        }
    }

    /// Enqueue `spot`, applying the dedup / cap rules of [`PskCfg`].
    ///
    /// Returns `true` when the spot is now in the queue, `false` when it
    /// was dropped (duplicate callsign+band within the TTL, or capped off).
    pub fn add(&mut self, spot: Spot) -> bool {
        let now = self.cfg.clock.now();
        let key = format!("{}|{}", spot.caller.to_uppercase(), band_of(spot.freq_hz));
        let vhf_up = spot.freq_hz >= 49_000_000;
        if !vhf_up {
            if let Some(&last) = self.cache.get(&key) {
                if now.saturating_sub(last) < self.cfg.dedup_ttl_secs {
                    return false;
                }
            }
        }
        self.cache.insert(key, now);
        self.queue.push_back(spot);
        let ok = self.queue.len() <= self.cfg.max_pending;
        while self.queue.len() > self.cfg.max_pending {
            self.queue.pop_front();
        }
        // wsjtx `PSKReporter.cpp:414-419`.
        self.cache
            .retain(|_, ts| now.saturating_sub(*ts) <= self.cfg.cache_prune_secs);
        ok
    }

    /// Pending spots in arrival order (oldest first). Borrowed view.
    pub fn pending(&self) -> &VecDeque<Spot> {
        &self.queue
    }

    /// Take all pending spots, clearing the queue. Cache survives — a
    /// drained-and-sent spot stays suppressed until the TTL expires.
    pub fn drain(&mut self) -> Vec<Spot> {
        self.queue.drain(..).collect()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Swap the clock (so tests can advance it without rebuilding the
    /// reporter).
    pub fn set_clock(&mut self, clock: Clock) {
        self.cfg.clock = clock;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(ttl: u32, max_pending: usize) -> PskCfg {
        PskCfg {
            dedup_ttl_secs: ttl,
            cache_prune_secs: 600,
            max_pending,
            clock: Clock::Real,
        }
    }

    fn spot(call: &str, freq: u32) -> Spot {
        Spot {
            caller: call.to_string(),
            locator: String::new(),
            freq_hz: freq,
            snr: -10,
            time_epoch: 1_700_000_000,
            mode: "FT8".into(),
        }
    }

    #[test]
    fn dedup_suppresses_within_ttl() {
        let mut r = PskReporter::new(cfg(300, 16));
        r.set_clock(Clock::Fixed(1_700_000_000));
        assert!(r.add(spot("W1AW", 14_075_000)));
        r.set_clock(Clock::Fixed(1_700_000_000 + 299));
        assert!(!r.add(spot("W1AW", 14_075_000)));
        r.set_clock(Clock::Fixed(1_700_000_000 + 301));
        assert!(r.add(spot("W1AW", 14_075_000)));
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn vhf_up_is_exempt_from_dedup() {
        let mut r = PskReporter::new(cfg(300, 16));
        r.set_clock(Clock::Fixed(1_700_000_000));
        assert!(r.add(spot("W1AW", 145_500_000)));
        assert!(r.add(spot("W1AW", 145_500_000)));
        assert_eq!(r.len(), 2);
        assert!(r.add(spot("K9ABC", 14_075_000)));
        assert!(!r.add(spot("K9ABC", 14_075_000)));
    }

    #[test]
    fn caller_is_case_insensitive() {
        let mut r = PskReporter::new(cfg(300, 16));
        r.set_clock(Clock::Fixed(1_700_000_000));
        assert!(r.add(spot("w1aW", 14_075_000)));
        assert!(!r.add(spot("W1AW", 14_075_000)));
    }

    #[test]
    fn same_call_different_band_is_not_suppressed() {
        // 14 MHz is 20 m, 7 MHz is 40 m — same callsign on a different band
        // (both HF, so dedup-eligible) must still post.
        let mut r = PskReporter::new(cfg(300, 16));
        r.set_clock(Clock::Fixed(1_700_000_000));
        assert!(r.add(spot("W1AW", 14_075_000)));
        assert!(r.add(spot("W1AW", 7_074_000)));
        assert_eq!(r.len(), 2);
        // Same call, same band → suppressed.
        assert!(!r.add(spot("W1AW", 14_076_000)));
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn max_pending_caps_and_drops_oldest() {
        let mut r = PskReporter::new(cfg(3600, 3));
        r.set_clock(Clock::Fixed(1_700_000_000));
        assert!(r.add(spot("A1", 14_075_000)));
        assert!(r.add(spot("B2", 14_075_000)));
        assert!(r.add(spot("C3", 14_075_000)));
        // At capacity — the next one pushes `len` to 4 > 3, so the head
        // drops and `add` reports `false`.
        assert!(!r.add(spot("D4", 14_075_000)));
        let got: Vec<String> = r.pending().iter().map(|s| s.caller.clone()).collect();
        let want: Vec<String> = vec!["B2".into(), "C3".into(), "D4".into()];
        assert_eq!(got, want);
    }

    #[test]
    fn drain_clears_and_cache_survives() {
        let mut r = PskReporter::new(cfg(300, 8));
        r.set_clock(Clock::Fixed(1_700_000_000));
        assert!(r.add(spot("W1AW", 14_075_000)));
        r.set_clock(Clock::Fixed(1_700_000_001));
        let _ = r.drain();
        assert!(r.is_empty());
        assert!(!r.add(spot("W1AW", 14_075_000)));
    }
}
