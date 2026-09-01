//! Multi-speed JS8Call decode (A / B / C / E).
//!
//! Mirror of the js8call `decodeEnqueueReadyExperiment` + decode loop:
//! the demod thread appends raw pre-AGC `f32` audio via [`Js8Tap`] (a short
//! critical section), and the API's decode thread polls the rolling buffer
//! on a fixed tick and runs [`js8_step`]. Each JS8 speed has its own cycle
//! length (A = 15 s, B = 10 s, C = 6 s, E = 30 s of 12 kHz audio); a mode
//! is decoded *as soon as its cycle data is ready* (not on a closed-slot
//! boundary), re-arming roughly every 1.5 s afterwards — the same
//! continuous-decode cadence the reference uses.
//!
//! The buffer is shared across all modes: it is 12 kHz audio in arrival
//! order, and each mode slices its own `mode.nmax`-sample window
//! (its full cycle). All modes consume the same samples with different
//! symbol rates / sync waveforms, so one rolling buffer serves all four
//! decodes (the reference keeps one `d2[]` ring and runs each submode's
//! decoder over it too).
//!
//! The decode itself is the same three-pass loop the previous single-mode
//! port used ([`super::decode::decode_candidate`] over each candidate from
//! [`super::sync::sync_pass`], passes 1–2 subtracting decoded signals to
//! reveal weaker interferers).

use std::sync::{Arc, Mutex};

use crate::receiver::demod::RawSampleTap;
use crate::receiver::js8::decode::{Ctx, decode_candidate, subtract};
use crate::receiver::js8::msg::DecodedFrame;
use crate::receiver::js8::params::MODES;
use crate::receiver::js8::sync::{Spectra, sync_pass};

use hl2_common::{DecodedMessage, SpotFields, SpotStation};

// ────────────────────────────────────────────────────────────────────────────
// Constants

/// 12 kHz — the JS8 decoder's fixed input rate (all speeds).
pub const JS8_SAMPLE_RATE_HZ: u32 = 12_000;

/// Samples per second at the fixed rate (re-exported for cycle math).
pub const JS8_SAMPLES_PER_SEC: u64 = JS8_SAMPLE_RATE_HZ as u64;

/// Re-arm interval (samples, 12 kHz): a mode whose cycle is ready is
/// decoded, then re-arms about 1.5 s later — the reference's "within
/// every 3/2 seconds" cadence from `decodeEnqueueReadyExperiment`.
const REARM_SAMPLES: u64 = 12_000 * 15 / 10; // = 18_000

/// Rolling buffer cap (samples): 60 s at 12 kHz — the reference's
/// `JS8_RX_SAMPLE_SIZE` (`JS8_NTMAX=60` seconds), so even the slowest
/// speed (Mode E, 30 s cycle) always has a full cycle of history plus
/// 30 s of additional margin for its "last 1.5 s of cycle" partial.
pub const JS8_BUFFER_CAP: usize = 60 * 12_000;

/// Frequency search range passed to the sync pass (Hz), matching the
/// reference's `nfa`/`nfb` for the audio passband.
const NFA_HZ: f32 = 100.0;
const NFB_HZ: f32 = 4_910.0;

/// Operator-known JS8Call frequencies (Hz) across the amateur bands, used
/// by the auto-decoder the same way [`super::ft8::KNOWN_FREQS`] is for
/// FT8. The convention is typically 4–10 kHz above the band's FT8
/// frequency; the table is the static, user-facing list of "where should
/// I be listening for JS8".
pub const KNOWN_FREQS: &[u32] = &[
    1_810_000,   // 160 m
    3_577_000,   // 80 m
    7_078_000,   // 40 m
    10_130_000,  // 30 m
    14_078_000,  // 20 m
    18_100_000,  // 17 m
    21_078_000,  // 15 m
    24_920_000,  // 12 m
    28_078_000,  // 10 m
    50_313_000,  // 6 m
    145_500_000, // 2 m
    432_075_000, // 70 cm
];

/// Duplicate-emission suppression window (ms): the continuous decoder
/// re-arms every ~1.5 s, so a clean signal keeps decoding — without
/// suppression the UI log + PSK Reporter would see the same frame many
/// times per minute. A repeat is re-emitted only after this window or
/// a ≥ half-tone frequency shift.
const EMIT_DEDUP_MS: u64 = 45_000;

// ────────────────────────────────────────────────────────────────────────────
// Decoded message

/// One decoded JS8 frame, as produced by a decode step.
#[derive(Debug, Clone, PartialEq)]
pub struct Js8Message {
    /// The unpacked frame (kind, callsign, grid, text…).
    pub frame: DecodedFrame,
    /// The JS8 speed this frame was decoded as ("JS8A"/"JS8B"/"JS8C"/"JS8E").
    pub submode: &'static str,
    /// The submode id (`js8call` varicode: A=0, B=2? … — see
    /// [`crate::receiver::js8::params::Mode::id`]) for the UI.
    pub mode_id: u8,
    /// Decoded frequency (Hz) in the audio passband.
    pub freq_hz: f32,
    /// Signal start offset into the cycle window, seconds
    /// (the reference's `xdt`, measured from the window start).
    pub dt_sec: f32,
    /// Estimated SNR (dB, WSJT-family reference bandwidth).
    pub snr_db: f32,
    /// The wall-clock start of the cycle window this decode came from
    /// (ms since Unix epoch; `decode_step` stamps it from the demod's
    /// sample count, which is monotonic and matches the buffer content).
    pub slot_ms: u64,
}

impl DecodedMessage for Js8Message {
    fn freq_hz(&self) -> f32 {
        self.freq_hz
    }
    fn dt_sec(&self) -> f32 {
        self.dt_sec
    }
    fn snr_db(&self) -> f32 {
        self.snr_db
    }
    fn slot_ms(&self) -> u64 {
        self.slot_ms
    }
    fn mode(&self) -> &'static str {
        // Plain "JS8" — js8call posts a single mode name (mainwindow.cpp:9611)
        // regardless of speed, so the collector's mode filter matches. We do
        // not post speed-specific JS8A/B/C/E (those are not a filter option).
        "JS8"
    }
    fn display(&self) -> &str {
        &self.frame.message
    }
    fn spot_fields(&self, st: &SpotStation) -> Option<SpotFields> {
        js8_spot_fields(&self.frame, st)
    }
}

/// Spot-selection for one decoded JS8 frame, mirroring the WSJT selector but
/// driven by the structured fields (JS8 is a protocol, not free text).
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
/// no locator at all. A different grid with the same base call still spots
/// (mobile on our base).
fn js8_spot_fields(
    f: &crate::receiver::js8::msg::DecodedFrame,
    st: &SpotStation,
) -> Option<SpotFields> {
    use crate::receiver::js8::msg::{
        FRAME_COMPOUND, FRAME_COMPOUND_DIRECTED, FRAME_DIRECTED, FRAME_HEARTBEAT,
    };
    use hl2_common::spot::{base_callsign, grid_is_square};

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
        return None;
    };

    // Locator: only a valid square, never a command name; otherwise empty.
    let grid_raw = f.grid.as_deref().unwrap_or("").trim();
    let grid: Option<&str> = grid_is_square(grid_raw).then_some(grid_raw);

    // Self-spot suppression: caller matches our base call **and** the grid
    // matches ours too — or the frame carries no grid at all (directed /
    // data-frame `CALL:`), in which case the matching callsign is enough
    // (that is how *our* directed messages would look). A different grid with
    // the same base call (a mobile on our base) still spots.
    let base_call = base_callsign(&st.callsign);
    if !base_call.is_empty() {
        let g4: String = st.grid.chars().take(4).collect();
        let caller_matches = call.eq_ignore_ascii_case(base_call);
        let grid_matches = grid.is_some_and(|g| g.eq_ignore_ascii_case(g4.as_str()));
        if caller_matches && (grid.is_none() || grid_matches) {
            return None;
        }
    }

    Some(SpotFields {
        caller: call.to_ascii_uppercase(),
        locator: grid.map(str::to_ascii_uppercase).unwrap_or_default(),
    })
}

/// For a JS8 free-text (data) frame, the leading sender when the payload
/// starts a directed message as `CALL:` — js8call's data-frame spotting rule
/// (mainwindow.cpp:8292-8306: first word callsign immediately followed by
/// `:`). Returns `None` when the text does not follow that shape.
fn data_frame_caller(f: &crate::receiver::js8::msg::DecodedFrame) -> Option<&str> {
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

// ────────────────────────────────────────────────────────────────────────────
// Decoder (accumulator)

/// The JS8 multi-speed decoder.
///
/// Accumulates raw decimated `f32` audio at 12 kHz in a bounded rolling
/// window (arrival order, oldest first). The decode thread snapshots
/// per-mode windows (under the lock) and decodes them off-lock. Per-mode
/// readiness state (`last_decode_total`) and duplicate-emission
/// suppression state live here too, so the whole API layer is stateless.
///
/// Not itself thread-safe — share it behind a [`Mutex`]. The demod's
/// critical section is a bounded `Vec` copy + cap trim, never a decode.
#[derive(Debug)]
pub struct Js8Decoder {
    buf: Vec<f32>,
    /// Total samples ever appended (monotonic) — used for per-mode
    /// cycle math, independent of the buffer cap trimming.
    total: u64,
    /// Per-mode (indexed by `MODES` position) total-sample count at the
    /// last successful re-arm. 0 = "never re-armed".
    last_decode_total: [u64; 4],
    /// Duplicate-emission suppression: the last time (ms) we emitted each
    /// (mode, callsign, grid) combination. Key is owned so the map
    /// survives message drops.
    last_emit: std::collections::HashMap<(u8, String, Option<String>), (u64, f32)>,
}

impl Js8Decoder {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(JS8_BUFFER_CAP),
            total: 0,
            last_decode_total: [0; 4],
            last_emit: std::collections::HashMap::new(),
        }
    }

    pub fn buffered_samples(&self) -> usize {
        self.buf.len()
    }

    /// Total samples ever appended (monotonic, survives buffer trims).
    pub fn total_samples(&self) -> u64 {
        self.total
    }

    /// Append a block of raw decimated `f32` audio (the demod's pre-AGC
    /// output), in arrival order. A `Vec` copy + cap trim; never blocks on a
    /// decode.
    pub fn add_samples(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        self.buf.extend_from_slice(samples);
        self.total = self.total.saturating_add(samples.len() as u64);
        if self.buf.len() > JS8_BUFFER_CAP {
            let drop = self.buf.len() - JS8_BUFFER_CAP;
            self.buf.drain(..drop);
        }
    }

    /// Snapshot the per-mode decode windows that are ready, per the
    /// reference's readiness rules, and mark each as re-armed. Returns
    /// `(mode_index, window)` for each ready mode (in `MODES` order).
    ///
    /// `now_ms` is the wall-clock (ms) for the readiness tick; used only
    /// to stamp `Js8Message::slot_ms` on the caller side (the windows
    /// themselves are pure buffer slices).
    fn snapshot_ready(&mut self) -> Vec<(usize, Vec<f32>)> {
        let mut out = Vec::new();
        if self.total == 0 {
            return out;
        }
        // The reference's `framesNeeded`: the signal's full symbol span
        // PLUS a minimum 0.5 s decode margin AND the `astart` delay —
        // `floor(framesForSymbols + (0.5 + astart) * JS8_RX_SAMPLE_RATE)`.
        // Without the `astart` component the test fires at t = `NN*nsps`
        // samples (= 12.64 s for Mode A), but the signal ends at
        // `astart + NN*nsps` seconds (= 13.14 s), so the last ~0.5 s of
        // data symbols is cut from the window.
        let frames_needed = |m: &crate::receiver::js8::params::Mode| {
            crate::receiver::js8::params::NN as u64 * m.nsps as u64
                + ((0.5 + m.astart as f64) * crate::receiver::js8::params::SAMPLE_RATE as f64)
                    as u64
        };
        for (mi, mode) in MODES.iter().enumerate() {
            // Cycle length in *samples* (nmax == cycle_ms * 12 at 12 kHz).
            let cycle = mode.nmax as u64;
            let nmax = mode.nmax as u64;
            // The cycle the most recent sample belongs to (sample `total-1`
            // is in cycle `(total-1)/cycle`), and how much of it we hold.
            let cycle_start = ((self.total - 1) / cycle) * cycle;
            let ready = (self.total - cycle_start).min(nmax);

            let since_last = if self.last_decode_total[mi] == 0 {
                u64::MAX
            } else {
                self.total.saturating_sub(self.last_decode_total[mi])
            };
            let needed = frames_needed(mode);
            // The reference's `decodeEnqueueReadyExperiment` fires when
            // (≥ REARM since the last decode):
            //   * the cycle is < 1.5 s old — a signal may straddle the
            //     cycle boundary (`ready < 1.5 s`), or
            //   * the cycle holds the signal's span minus the last 1.5 s
            //     (`ready >= needed - 1.5 s`); that tail covers both
            //     "within the last 3/2 seconds of a new cycle" (partial)
            //     and the fully ready case.
            let fired = since_last >= REARM_SAMPLES
                && (ready < REARM_SAMPLES || ready >= needed.saturating_sub(REARM_SAMPLES));
            if !fired {
                continue;
            }

            // Snapshot the trailing `ready` samples (zero-padded forward
            // if the buffer holds less than `ready` — e.g. right after
            // start-up or a receiver restart).
            let want = ready.min(self.buf.len() as u64) as usize;
            let mut window = vec![0.0f32; want];
            let src_start = self.buf.len() - want;
            window.copy_from_slice(&self.buf[src_start..]);

            // Re-arm (the reference's `m_lastDecodeStartMap[submode] = k`).
            self.last_decode_total[mi] = self.total;
            out.push((mi, window));
        }
        out
    }

    /// Suppression check for an outgoing decode. Returns `true` if the
    /// decode should be emitted now and records it; `false` if an
    /// identical (mode, call, grid) frame was emitted within
    /// [`EMIT_DEDUP_MS`] *and* the frequency has not moved by at least
    /// half a tone (the reference's "repeats merge into the existing
    /// `decodes` entry" behaviour, which keeps one row per sender).
    fn should_emit(
        &mut self,
        mode_id: u8,
        mode_name: &'static str,
        msg: &Js8Message,
        now_ms: u64,
    ) -> bool {
        let key = (mode_id, msg.frame.callsign.clone(), msg.frame.grid.clone());
        let tone = JS8_SAMPLE_RATE_HZ as f32
            / MODES
                .iter()
                .find(|m| m.id == mode_id)
                .map(|m| m.nsps)
                .unwrap_or(1920) as f32;
        let half_tone = tone * 0.5;
        let prev = self.last_emit.get(&key);
        if let Some(&(ts, f)) = prev {
            let dt = now_ms.saturating_sub(ts);
            if dt < EMIT_DEDUP_MS && (f - msg.freq_hz).abs() < half_tone {
                if std::env::var_os("JS8_TRACE").is_some() {
                    eprintln!(
                        "JS8_TRACE   dedup suppress {}: dt={dt}ms dF={:.1}Hz (mode {mode_name})",
                        msg.frame.callsign,
                        (f - msg.freq_hz)
                    );
                }
                return false;
            }
        }
        self.last_emit.insert(key, (now_ms, msg.freq_hz));
        true
    }
}

impl Default for Js8Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared handle to a [`Js8Decoder`] behind a [`Mutex`] — the type both the
/// demod tap and the tokio decode task hold.
pub type SharedDecoder = Arc<Mutex<Js8Decoder>>;

/// Build the shared [`Arc<Mutex<Js8Decoder>>`].
pub fn shared() -> SharedDecoder {
    Arc::new(Mutex::new(Js8Decoder::new()))
}

// ────────────────────────────────────────────────────────────────────────────
// Demod-thread tap

/// The demod-thread half of the JS8 pipeline: adapts a [`SharedDecoder`] to
/// the [`RawSampleTap`] trait the demodulator calls while demodulating.
#[derive(Debug, Clone)]
pub struct Js8Tap {
    inner: SharedDecoder,
}

impl Js8Tap {
    pub fn from_shared(inner: SharedDecoder) -> Self {
        Self { inner }
    }

    pub fn shared(&self) -> SharedDecoder {
        self.inner.clone()
    }
}

impl Default for Js8Tap {
    fn default() -> Self {
        Self { inner: shared() }
    }
}

impl RawSampleTap for Js8Tap {
    fn append(&self, samples: &[f32]) {
        let mut g = self.inner.lock().expect("Js8Decoder poisoned");
        g.add_samples(samples);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Decode step

struct Hit {
    frame: DecodedFrame,
    freq: f32,
    dt: f32,
    snr: f32,
}

/// Run the reference's three-pass decode loop on one mode window
/// (0-padded to `mode.nmax`): sync → decode → subtract, stopping early
/// if a pass improves nothing. Returns the unique frames (a repeat frame
/// keeps the best SNR, the way the reference's `decodes` map merges
/// duplicates).
fn decode_window(window: &[f32], mode: &crate::receiver::js8::params::Mode) -> Vec<Hit> {
    let nmax = mode.nmax;
    let mut dd: Vec<f32> = window.to_vec();
    if dd.len() < nmax {
        dd.resize(nmax, 0.0);
    } else {
        dd.truncate(nmax);
    }
    // Rescale this window's absolute peak to [`super::super::audio_scale::PEAK_TARGET`]
    // (≈ −1.4 dBFS — "green, just under red", the level js8call's
    // `JS8.cpp` reference was tuned against). The EP6 baseband is
    // typically 10×-100× quieter than that, and a number of absolute
    // gates in the decoder (the `sync < 2.0` pass gate, the `s2[]`
    // amplitude / 1000.0 scale in `decode_candidate`) are calibrated
    // for inputs near it. One pass over the window, a tiny fraction of
    // the FFT + sync + BP-FEC cost that follows; see
    // [`super::super::audio_scale`] for the rationale. The one
    // normalisation is shared across passes 1-3 of the subtraction
    // loop: once `dd` is at the reference amplitude, the `subtract`
    // step's scale estimate and the `sync` power gate both see the
    // level the C++ reference expects.
    let gain = crate::receiver::audio_scale::peak_normalize(&mut dd);
    if std::env::var_os("HL2_DEBUG").is_some() {
        let db = if gain > 0.0 && gain.is_finite() {
            20.0 * gain.log10()
        } else {
            0.0
        };
        eprintln!(
            "[aud] {} peaknorm={db:+.1} dB (peak→{:.3}, pre-AGC into decoder)",
            mode.name,
            crate::receiver::audio_scale::PEAK_TARGET
        );
    }
    let csyncs = Ctx::csyncs(mode);
    let mut hits: Vec<Hit> = Vec::new();

    for ipass in 1..=3 {
        // (Re)compute the per-step spectrum from the (possibly subtracted)
        // signal and run the sync pass. `savg` is mutated by the baseline
        // fit, so it is re-derived on every pass — as the reference does
        // (`syncjs8` accumulates `savg` fresh each call).
        let mut sp = Spectra::new(mode, &dd);
        let cands = sync_pass(mode, &mut sp, NFA_HZ, NFB_HZ);
        if std::env::var_os("JS8_TRACE").is_some() {
            let top: String = cands
                .iter()
                .take(5)
                .map(|c| format!("f={:.1}/dt={:.2}/s={:.1} ", c.freq, c.dt, c.sync))
                .collect();
            eprintln!(
                "JS8_TRACE decode_window mode={} ipass={ipass} cands={} top5=[{top}]",
                mode.name,
                cands.len()
            );
        }
        if cands.is_empty() {
            break;
        }

        let do_subtract = ipass < 3;
        let mut improved = false;

        for cand in cands {
            let ctx = Ctx {
                mode,
                dd: &dd,
                sp: &sp,
                csyncs: csyncs.clone(),
                cd0: Vec::new(),
            };

            let Some((frame, tones, xdt, snr)) = decode_candidate(ctx, &cand, &sp) else {
                continue;
            };

            // Reference convention: `xdt` is the absolute offset into the
            // window (js8dec refines it at the fine-sync step). We keep
            // it as-is: the operator-visible "signal start relative to
            // cycle anchor" is `xdt - astart`, but the window *is* the
            // cycle (it starts at the cycle boundary), so `xdt` is the
            // cycle-anchored offset.
            let dt = xdt;

            // Subtract on passes 1–2 so the next pass can find weaker
            // signals underneath. Use the refined `xdt` and the corrected
            // frequency.
            if do_subtract {
                subtract(mode, &mut dd, &tones, cand.freq, xdt);
            }

            // Merge: a repeat frame keeps the best SNR; a new frame is added.
            if let Some(existing) = hits.iter_mut().find(|h| h.frame == frame) {
                if snr > existing.snr {
                    existing.snr = snr;
                    existing.freq = cand.freq;
                    existing.dt = dt;
                }
            } else {
                hits.push(Hit {
                    frame,
                    freq: cand.freq,
                    dt,
                    snr,
                });
            }
            improved = true;
        }

        if !improved {
            break;
        }
    }

    hits
}

/// One decode tick. Called by the API's decode thread on a fixed cadence
/// (recommended ~500 ms).
///
/// For each JS8 speed in [`MODES`] (A / B / C / E), if the rolling
/// buffer holds enough of that mode's cycle *and* ≥ 1.5 s have passed
/// since this mode last re-armed, snapshot the trailing window (under
/// the lock) and run the three-pass sync + decode + subtract loop on it
/// (off the lock). Duplicates (same mode + callsign + grid within
/// [`EMIT_DEDUP_MS`], frequency within half a tone) are suppressed to
/// keep the UI log / PSK Reporter from flooding the same frame every
/// 1.5 s — the reference merges repeated decodes into one row instead.
///
/// Returns the unique decoded frames across all modes (empty if nothing
/// decoded or everything was suppressed).
pub fn js8_step(
    decoder: &SharedDecoder,
    now_ms: u64,
) -> Result<Vec<Js8Message>, Box<dyn std::error::Error + Send + Sync>> {
    // Snapshot ready windows under the lock; decode off-lock so the
    // demod tap's critical section stays bounded to the (small) buffer
    // copy.
    let pending: Vec<(usize, Vec<f32>)> = {
        let mut g = decoder.lock().expect("Js8Decoder poisoned");
        g.snapshot_ready()
    };
    if pending.is_empty() {
        return Ok(Vec::new());
    }

    let mut all: Vec<Js8Message> = Vec::new();
    for (mi, window) in pending {
        let mode = &MODES[mi];
        let hits = decode_window(&window, mode);
        if hits.is_empty() {
            continue;
        }
        // Approximate the wall-clock start of the window: the decode
        // thread calls `js8_step` on a wall-clock tick and the buffer is
        // arrival-ordered, so the oldest sample in the window is
        // `total - window.len()` samples ago ≈ `now_ms - window.len()/12`
        // seconds ago. Good enough for UI / PSK Reporter timestamps.
        let start_ms = now_ms.saturating_sub(window.len() as u64 / JS8_SAMPLES_PER_SEC * 1000);
        for h in hits {
            let msg = Js8Message {
                frame: h.frame,
                submode: mode.name,
                mode_id: mode.id,
                freq_hz: h.freq,
                dt_sec: h.dt,
                snr_db: h.snr,
                slot_ms: start_ms,
            };
            let emit = {
                let mut g = decoder.lock().expect("Js8Decoder poisoned");
                g.should_emit(mode.id, mode.name, &msg, now_ms)
            };
            if emit {
                all.push(msg);
            }
        }
    }
    Ok(all)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receiver::js8::params::SAMPLE_RATE;

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    /// Feed a clean Mode `mode` signal through the decoder in 500 ms
    /// chunks (the way the demod's [`Js8Tap`] delivers it in production),
    /// calling [`js8_step`] after each chunk with the running wall-clock.
    /// Chunk (not all-at-once) feeding models the continuous re-arm loop:
    /// the reference decodes as soon as a signal's full span is in the
    /// window and re-arms every ~1.5 s. Returns the first decoded frame
    /// whose callsign is `want`, or `None` if the signal never decoded.
    fn feed_until_decoded(
        mode: &crate::receiver::js8::params::Mode,
        sig: &[f32],
        want: &str,
    ) -> Option<crate::receiver::Js8Message> {
        let d = shared();
        const CHUNK: usize = 6000; // 500 ms @ 12 kHz
        let total = sig.len();
        let mut off = 0usize;
        let mut now_ms: u64 = 0;
        while off < total {
            let n = CHUNK.min(total - off);
            d.lock().unwrap().add_samples(&sig[off..off + n]);
            now_ms = (now_ms as usize + n) as u64 / 12;
            let msgs = js8_step(&d, now_ms).unwrap();
            if let Some(m) = msgs
                .iter()
                .find(|m| m.frame.callsign == want && m.submode == mode.name)
            {
                return Some(m.clone());
            }
            off += n;
        }
        None
    }

    #[test]
    fn window_caps_at_60s() {
        let mut d = Js8Decoder::new();
        d.add_samples(&silence(70 * 12_000)); // 70 s — over the 60 s cap
        assert_eq!(d.buf.len(), 60 * 12_000);
        assert_eq!(d.total_samples(), 70 * 12_000);
    }

    #[test]
    fn append_is_the_tap() {
        let d = shared();
        let tap = Js8Tap::from_shared(d.clone());
        tap.append(&[0.1, -0.2, 0.3]);
        let got = d.lock().unwrap().buf.clone();
        assert_eq!(got, vec![0.1, -0.2, 0.3]);
    }

    #[test]
    fn silence_step_yields_nothing() {
        let d = shared();
        // 40 s of silence — enough for every mode to have a full cycle in
        // the buffer, but silence decodes nothing.
        d.lock().unwrap().add_samples(&silence(40 * 12_000));
        let msgs = js8_step(&d, 40 * 1000).expect("step");
        assert!(msgs.is_empty(), "silence should yield no rows: {msgs:?}");
    }

    /// Full end-to-end: build a real JS8 signal in each of the four modes
    /// (encoded heartbeat frame → 79 tones → 12 kHz cosine at 1.5 kHz,
    /// `nsps` samples per symbol, mode-appropriate `astart` offset) and
    /// run it through the continuous decoder. The pipeline must recover
    /// the frame per mode (callsign + grid, per `unpack_frame` for the
    /// heartbeat family), tagged with the right submode.
    #[test]
    fn self_synthesised_js8_all_modes_decode() {
        use crate::receiver::js8::frame::encode_tones;
        use crate::receiver::js8::msg::{pack_compound_frame, pack_grid};
        use crate::receiver::js8::params::MODES;

        let grid = pack_grid("FN42");
        let words =
            pack_compound_frame("N1MM", crate::receiver::js8::msg::FRAME_HEARTBEAT, grid, 0)
                .expect("pack_compound_frame");

        for mode in MODES {
            let tones_vec = encode_tones(words, 0, &mode);
            assert_eq!(tones_vec.len(), 79, "mode {} tone count", mode.name);

            // Build `ntxdur` seconds of 12 kHz passband audio; the signal
            // (79 tones, `nsps` samples each) occupies
            // [astart, astart + NN*nsps/12000) inside the cycle.
            let nmax = mode.nmax;
            let f0 = 1_500.0f32;
            let mut sig = vec![0.0f32; nmax];
            let off = (mode.astart * SAMPLE_RATE).round() as usize;
            let mut phi = 0.0f32;
            let bfpi = std::f32::consts::TAU * f0 / SAMPLE_RATE;
            for (i, t) in tones_vec.iter().enumerate() {
                let dphi = bfpi + std::f32::consts::TAU * (*t as f32) / mode.nsps as f32;
                for s in 0..mode.nsps {
                    let idx = off + i * mode.nsps + s;
                    if idx < nmax {
                        sig[idx] = 0.5 * phi.cos();
                    }
                    phi = (phi + dphi).rem_euclid(std::f32::consts::TAU);
                }
            }

            let m = feed_until_decoded(&mode, &sig, "N1MM")
                .unwrap_or_else(|| panic!("mode {}: no decode at {f0} Hz", mode.name));
            assert_eq!(m.mode_id, mode.id);
            assert!(
                (m.dt_sec - mode.astart).abs() < 0.4,
                "mode {}: dt {} should be ~ astart ({})",
                mode.name,
                m.dt_sec,
                mode.astart
            );
            assert!(
                m.snr_db > 0.0,
                "mode {}: SNR {} positive",
                mode.name,
                m.snr_db
            );
            assert!(
                m.snr_db > -59.0,
                "mode {}: SNR {} at the -60 dB floor (scale regression)",
                mode.name,
                m.snr_db
            );
            assert_eq!(m.frame.callsign, "N1MM");
            assert_eq!(m.frame.grid.as_deref(), Some("FN42"));
        }
    }

    /// The decoder should not re-emit the same (mode, callsign, grid,
    /// same-frequency) frame again within [`EMIT_DEDUP_MS`] — the
    /// continuous re-arm cadence would otherwise flood the log and PSK
    /// Reporter with the same station every ~1.5 s.
    #[test]
    fn repeat_suppressed_within_window() {
        use crate::receiver::js8::frame::encode_tones;
        use crate::receiver::js8::msg::{pack_compound_frame, pack_grid};
        use crate::receiver::js8::params::MODE_A;

        let grid = pack_grid("FN42");
        let words =
            pack_compound_frame("N1MM", crate::receiver::js8::msg::FRAME_HEARTBEAT, grid, 0)
                .expect("pack");
        let tones_vec = encode_tones(words, 0, &MODE_A);
        let nmax = MODE_A.nmax;
        let f0 = 1_500.0f32;
        let mut sig = vec![0.0f32; nmax];
        let off = (MODE_A.astart * SAMPLE_RATE).round() as usize;
        let mut phi = 0.0f32;
        let bfpi = std::f32::consts::TAU * f0 / SAMPLE_RATE;
        for (i, t) in tones_vec.iter().enumerate() {
            let dphi = bfpi + std::f32::consts::TAU * (*t as f32) / MODE_A.nsps as f32;
            for s in 0..MODE_A.nsps {
                sig[off + i * MODE_A.nsps + s] = 0.5 * phi.cos();
                phi = (phi + dphi).rem_euclid(std::f32::consts::TAU);
            }
        }

        let d = shared();
        d.lock().unwrap().add_samples(&sig);
        {
            let msgs = js8_step(&d, 15_000).unwrap();
            assert!(
                !msgs.is_empty(),
                "first step should decode the clean signal"
            );
        }
        // Append another full cycle of the same signal with the decoder's
        // re-arm elapsed; the same (call, grid) at the same freq should be
        // suppressed.
        d.lock().unwrap().add_samples(&sig);
        // Re-arm has elapsed (1.5 s); the same (call, grid) at the same
        // freq should be suppressed now.
        let msgs2 = js8_step(&d, 16_500).unwrap();
        let same: Vec<_> = msgs2
            .iter()
            .filter(|m| m.submode == "JS8A" && m.frame.callsign == "N1MM")
            .collect();
        assert!(
            same.is_empty(),
            "repeat within {EMIT_DEDUP_MS} ms at the same freq should be suppressed, got: {same:?}"
        );
    }

    /// The reported JS8 SNR must track the *true* injected SNR
    /// (monotonic, correct order, sane magnitude) — pinning the
    /// `xsig`(cd0/s2) versus `xbase`(fit_baseline) relative scale.
    #[test]
    fn reported_snr_tracks_injected_snr() {
        use crate::receiver::js8::frame::encode_tones;
        use crate::receiver::js8::msg::{pack_compound_frame, pack_grid};
        use crate::receiver::js8::params::MODE_A;

        let grid = pack_grid("FN42");
        let words =
            pack_compound_frame("N1MM", crate::receiver::js8::msg::FRAME_HEARTBEAT, grid, 0)
                .expect("pack");
        let tones_vec = encode_tones(words, 0, &MODE_A);
        let nmax = MODE_A.nmax;
        let f0 = 1_500.0f32;
        let mut sig = vec![0.0f32; nmax];
        let off = (MODE_A.astart * SAMPLE_RATE).round() as usize;
        let mut phi = 0.0f32;
        let bfpi = std::f32::consts::TAU * f0 / SAMPLE_RATE;
        for (i, t) in tones_vec.iter().enumerate() {
            let dphi = bfpi + std::f32::consts::TAU * (*t as f32) / MODE_A.nsps as f32;
            for s in 0..MODE_A.nsps {
                sig[off + i * MODE_A.nsps + s] = 0.5 * phi.cos();
                phi = (phi + dphi).rem_euclid(std::f32::consts::TAU);
            }
        }
        let sig_power: f32 = sig.iter().map(|v| v * v).sum::<f32>() / sig.len() as f32;

        let mut reported: Vec<(f32, f32)> = Vec::new();
        for snr_in_db in [20.0f32, 10.0, 5.0, 0.0] {
            let mut seed: u64 = 0x9e37_79b9u64;
            let amp = (sig_power / (10f32.powf(snr_in_db / 10.0))).sqrt();
            let mut mixed = vec![0.0f32; nmax];
            for v in mixed.iter_mut() {
                seed = seed.wrapping_mul(63_641).wrapping_add(1_357_911);
                let u = (seed >> 40) as f32 / 8_388_608.0 - 1.0;
                *v = amp * u;
            }
            let mean: f32 = mixed.iter().sum::<f32>() / mixed.len() as f32;
            let (s, n) = mixed.split_at_mut(sig.len());
            for (v, sv) in s.iter_mut().zip(sig.iter()) {
                *v = *v - mean + sv;
            }
            for v in n.iter_mut() {
                *v -= mean;
            }

            let d = shared();
            d.lock().unwrap().add_samples(&mixed);
            let msgs = js8_step(&d, (nmax / 12_000) as u64 * 1000).expect("step");
            let m = msgs
                .iter()
                .find(|m| m.freq_hz == f0 && m.submode == "JS8A")
                .expect("decode");
            reported.push((snr_in_db, m.snr_db));
        }

        let (b, br) = reported[0]; // 20 dB
        let (a, ar) = reported[1]; // 10 dB
        assert!(
            br >= ar - 3.0,
            "20 dB ({b}) reported {br} but 10 dB ({a}) reported {ar} — not monotonic"
        );
        let (c, cr) = reported[3]; // 0 dB
        assert!(
            ar >= cr - 3.0,
            "10 dB ({a}) reported {ar} but 0 dB ({c}) reported {cr} — not monotonic"
        );
        assert!(ar > -10.0, "10 dB true SNR reported {ar} dB (floor?)");
    }

    use crate::receiver::js8::msg::{
        DecodedFrame, FRAME_COMPOUND, FRAME_DIRECTED, FRAME_HEARTBEAT,
    };
    use hl2_common::SpotStation;

    fn js8_base() -> Js8Message {
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
        let mut m = js8_base();
        m.frame.kind = FRAME_HEARTBEAT;
        m.frame.callsign = callsign.into();
        m.frame.grid = grid.map(str::to_string);
        m
    }

    fn js8_dir(from: &str, to: &str, cmd: &str) -> Js8Message {
        let mut m = js8_base();
        m.frame.kind = FRAME_DIRECTED;
        m.frame.callsign = from.into();
        m.frame.to = Some(to.into());
        m.frame.cmd = Some(cmd.into());
        m.frame.grid = None;
        m
    }

    fn js8_compound(callsign: &str, grid: &str) -> Js8Message {
        let mut m = js8_base();
        m.frame.kind = FRAME_COMPOUND;
        m.frame.callsign = callsign.into();
        m.frame.grid = Some(grid.into());
        m
    }

    fn js8_data(text: &str) -> Js8Message {
        let mut m = js8_base();
        m.frame.kind = DecodedFrame::KIND_DATA_JSC;
        m.frame.callsign = String::new();
        m.frame.grid = None;
        m.frame.text = text.into();
        m.frame.message = text.into();
        m
    }

    fn st() -> SpotStation {
        SpotStation::new("K9ABC", "FN31")
    }

    #[test]
    fn js8_heartbeat_spots_call_and_grid() {
        let s = js8msg("R7IW", Some("LN35"))
            .spot_fields(&st())
            .expect("spots");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "LN35");
    }

    #[test]
    fn js8_heartbeat_without_grid_still_spots() {
        let s = js8msg("R7IW", None).spot_fields(&st()).expect("spots");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "");
    }

    #[test]
    fn js8_directed_frame_spots_from_call_without_grid() {
        let s = js8_dir("R7IW", "K9ABC", "SNR")
            .spot_fields(&st())
            .expect("directed");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "");
    }

    #[test]
    fn js8_compound_spots_call_and_grid() {
        let s = js8_compound("R7IW", "LN35")
            .spot_fields(&st())
            .expect("compound");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "LN35");
    }

    #[test]
    fn js8_data_frame_leading_call_spots() {
        let s = js8_data("K1ABC: Hello there")
            .spot_fields(&st())
            .expect("data CALL: lead");
        assert_eq!(s.caller, "K1ABC");
        assert_eq!(s.locator, "");
    }

    #[test]
    fn js8_data_frame_without_call_does_not_spot() {
        assert!(js8_data("HELLO WORLD").spot_fields(&st()).is_none());
    }

    #[test]
    fn js8_bad_grid_becomes_empty_locator() {
        let s = js8msg("K1ABC", Some("FN4"))
            .spot_fields(&st())
            .expect("spots");
        assert_eq!(s.caller, "K1ABC");
        assert_eq!(s.locator, "");
    }

    #[test]
    fn js8_no_call_does_not_spot() {
        let mut m = js8_base();
        m.frame.kind = FRAME_HEARTBEAT;
        m.frame.callsign = String::new();
        assert!(m.spot_fields(&st()).is_none());
    }

    #[test]
    fn js8_self_spot_is_suppressed() {
        assert!(js8msg("K9ABC", Some("FN31")).spot_fields(&st()).is_none());
        assert!(js8msg("K9ABC", Some("PM95")).spot_fields(&st()).is_some());
    }

    #[test]
    fn js8_self_directed_is_suppressed() {
        assert!(js8_dir("K9ABC", "R7IW", "SNR").spot_fields(&st()).is_none());
    }
}
