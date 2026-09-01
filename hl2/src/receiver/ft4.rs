//! Wall-clock-aligned FT4 decode.
//!
//! FT4 is a 7.5-second-slot mode: a WSJT-family 4-GFSK symbol stream, 103
//! symbols per slot (~48 ms each), occupying ~210-250 Hz of the USB SSB
//! audio passband. `mfsk-core` 0.9.1 does a **batch** decode of a fixed
//! 90 000-sample (7.5 s at 12 kHz) `i16` window — there is no streaming
//! `add_sample` API.
//!
//! This module bridges that batch model to the live HL2 receive loop, in
//! exactly the shape of [`super::ft8`]:
//!
//! 1. The virtual-receiver demod (in `hl2-api`, in FT4 mode) is a USB
//!    [`super::demod::DigitalDemodulator`] at 12 kHz output, with an
//!    [`Ft4Tap`] attached. The demod feeds the tap the raw pre-AGC
//!    decimated `f32` audio inline ([`Ft4Tap`'s `RawSampleTap::append`]);
//!    independently it emits the AGC'd `i16` to the `CH_AUDIO` sink — the
//!    operator hears the same audio that is being decoded.
//! 2. A std-thread decode task in `hl2-api` polls the wall-clock on a 1 s
//!    tick. On each tick it computes the most-recently-closed 7.5 s slot
//!    boundary (`now_ms / 7500 × 7500`) and, if that slot has not already
//!    been decoded, calls [`decode_closed_slot`]: lock the shared buffer,
//!    clone out the trailing 90 000-sample window (7.5 s at 12 kHz),
//!    release the lock, then run the `mfsk-core` batch decode off-lock —
//!    so the demod hot loop only ever pays a short critical section to
//!    append samples.
//! 3. If the decode produces >= 1 CRC-passing row, the API task broadcasts
//!    an `hl2_common::Ft4Log` to every connected client.
//!
//! Concurrency: one writer thread (the demod)'s [`Ft4Tap`] and one reader
//! (the decode task) share a single `Arc<Mutex<Ft4Decoder>>`. The writer's
//! critical section is a bounded `Vec` copy + cap trim (never a decode);
//! the reader snapshots the trailing window under the same lock and
//! decodes off-lock. A slot is decodable only *after* its wall-clock
//! boundary, so a decode that finishes late always lands in the *current*
//! 7.5 s slot — no cross-slot starvation.

use std::sync::{Arc, Mutex};

use mfsk_core::engine::protocol::ProtocolId;
use mfsk_core::ft4::Ft4;
use mfsk_core::msg::decode_request::DecodeRequest;

use hl2_common::{DecodedMessage, SpotFields, SpotStation};

use super::demod::RawSampleTap;

// ────────────────────────────────────────────────────────────────────────────
// Constants

/// Fixed FT4 slot duration (ms): the WSJT family's 7.5-second transmit
/// window. A signal runs from ~0.5 s into a slot to ~7.0 s into it; the
/// slot boundary (a clean multiple of 7500) is the canonical decode anchor
/// (and the `slot_ms` [`Ft4Message`] is stamped with).
pub const FT4_SLOT_MS: u64 = 7_500;

/// Number of 12 kHz samples in one full 7.5 s FT4 slot (90 000). This is
/// the fixed input window `mfsk-core`'s FT4 decoder expects.
pub const FT4_SLOT_WINDOW_SAMPLES: usize = 90_000;

/// 12 kHz — the `mfsk-core` FT4 decoder's fixed input rate (`NSPS = 576`
/// samples per 48 ms symbol). The HL2's demod must therefore emit audio at
/// 12 kHz when the virtual receiver is in FT4 mode. Kept here so the API
/// layer pulls the same constant.
pub const FT4_SAMPLE_RATE_HZ: u32 = 12_000;

/// Rolling-buffer cap (samples): 10 s at 12 kHz — one full 7.5 s slot of
/// decode margin plus 2.5 s of headroom before the slot boundary crosses.
/// `add_samples` trims the front past this cap (drop-oldest semantics —
/// matching `BasebandRing` / `BufSink` in this codebase).
const FT4_BUFFER_CAP: usize = 10 * 12_000;

/// Operator-known FT4 frequencies (Hz) across the amateur bands, used by
/// the auto-decoder to pick which in-window target frequencies to attach a
/// FT4 pipeline to. FT4 runs in the same band plans as FT8 (the standard
/// WSJT digital frequencies). The table is static and band-agnostic — the
/// API layer intersects it with the slot's current NCO (± EP6 half-span)
/// to figure out which are reachable from that NCO. Adding a new known
/// frequency is a one-line change here.
pub const KNOWN_FREQS: &[u32] = &[
    3_575_000,   // 80 m
    7_047_500,   // 40 m
    10_140_000,  // 30 m
    14_080_000,  // 20 m
    18_104_000,  // 17 m
    21_140_000,  // 15 m
    24_919_000,  // 12 m
    28_180_000,  // 10 m
    50_318_000,  // 6 m
    144_170_000, // 2 m
];

// ────────────────────────────────────────────────────────────────────────────
// Decoded message

/// One decoded FT4 message, as produced by the slot decode.
///
/// `text` is the resolved 77-bit WSJT payload (e.g. `"CQ DE W1AW"`).
/// `freq_hz` is the decoded tone-0 offset in the audio passband — the same
/// number WSJT-X shows in its "freq" column (Hz). `dt_sec` is the
/// signal-start offset relative to the slot anchor, in signed seconds
/// (`+` = late). `snr_db` is the estimated SNR (WSJT-X 2.5 kHz reference
/// bandwidth convention). `slot_ms` is the 7.5-second wall-clock bucket
/// (a multiple of 7500) this decode belongs to — the anchor the UI log
/// groups by.
#[derive(Debug, Clone, PartialEq)]
pub struct Ft4Message {
    pub text: String,
    pub freq_hz: f32,
    pub dt_sec: f32,
    pub snr_db: f32,
    pub slot_ms: u64,
}

impl DecodedMessage for Ft4Message {
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
        "FT4"
    }
    fn display(&self) -> &str {
        &self.text
    }
    fn spot_fields(&self, st: &SpotStation) -> Option<SpotFields> {
        super::spot::wsjt_spot_fields(&self.text, st)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Decoder (accumulator + decode)

/// Tunable search-range / detector parameters for one decode window.
#[derive(Debug, Clone, Copy)]
struct DecodeParams {
    /// Lower bound of the tone-0 search range (Hz), in the audio passband.
    freq_min: f32,
    /// Upper bound of the tone-0 search range (Hz).
    freq_max: f32,
    /// Minimum normalised sync score to accept a candidate.
    sync_min: f32,
    /// Maximum sync candidates to attempt full FEC decode on.
    ///
    /// A valid FT4 signal typically produces 1–3 strong sync candidates
    /// (the true tone-0 and any images). 20 is generous: it catches a
    /// signal with up to 5× the expected candidates while keeping the
    /// worst-case decode latency ≈ 20 full FEC attempts.
    max_cand: usize,
}

impl Default for DecodeParams {
    fn default() -> Self {
        Self {
            freq_min: 100.0,
            freq_max: 3_000.0,
            sync_min: 1.0,
            max_cand: 20,
        }
    }
}

/// The FT4 slot decoder.
///
/// Accumulates raw decimated `f32` audio at 12 kHz in a bounded rolling
/// window (arrival order, oldest first, capped at 10 s) and, on demand,
/// slices the trailing 7.5 s and decodes it with `mfsk-core`.
///
/// Not itself thread-safe — share it behind a [`Mutex`] (see
/// [`Ft4Tap`] and [`shared`]). The critical sections are short (a copy into
/// / out of a bounded `Vec`) so the demod hot path never stalls on a decode.
#[derive(Debug)]
pub struct Ft4Decoder {
    buf: Vec<f32>,
    /// The most-recent 7.5 s slot anchor (a multiple of 7500) already
    /// decoded. 0 means "never decoded". Prevents double-decoding the same
    /// slot (a duplicate boundary tick, a retry).
    last_decoded_slot: u64,
    params: DecodeParams,
}

impl Ft4Decoder {
    /// Build an empty decoder over 12 kHz raw `f32` audio (the demod's
    /// pre-AGC output). The slot window is [`FT4_SLOT_WINDOW_SAMPLES`]
    /// samples (7.5 s), decodable once a full slot has been buffered.
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(FT4_BUFFER_CAP),
            last_decoded_slot: 0,
            params: DecodeParams::default(),
        }
    }

    /// Number of samples currently buffered. For diagnostics / tests.
    pub fn buffered_samples(&self) -> usize {
        self.buf.len()
    }

    /// The trailing [`FT4_SLOT_WINDOW_SAMPLES`] samples (a copy), or `None`
    /// if the buffer is short. Does **not** mark any slot decoded — a
    /// diagnostic view over the same window `snapshot_for_slot` would slice.
    pub fn buf_trailing(&self, n: usize) -> Vec<f32> {
        let start = self.buf.len().saturating_sub(n);
        self.buf[start..].to_vec()
    }

    /// Append a block of raw decimated `f32` audio (the demod's pre-AGC
    /// output), in arrival order. A `Vec` copy + cap trim. Never blocks on
    /// a decode.
    pub fn add_samples(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        self.buf.extend_from_slice(samples);
        if self.buf.len() > FT4_BUFFER_CAP {
            let drop = self.buf.len() - FT4_BUFFER_CAP;
            self.buf.drain(..drop);
        }
    }

    /// Snapshot the trailing [`FT4_SLOT_WINDOW_SAMPLES`] samples (a copy —
    /// the demod thread may be appending while the caller holds the lock)
    /// and the decode parameters, *marking* `slot_ms` as decoded. Returns
    /// `None` if the slot was already decoded or the buffer is still short
    /// of a full 7.5 s window.
    ///
    /// Returns the owned window (ready for [`decode_window`]) plus the
    /// parameters to decode with. Call while holding the mutex, then decode
    /// off-lock.
    fn snapshot_for_slot(&mut self, slot_ms: u64) -> Option<(Vec<f32>, DecodeParams)> {
        if self.last_decoded_slot == slot_ms {
            return None;
        }
        if self.buf.len() < FT4_SLOT_WINDOW_SAMPLES {
            return None;
        }
        let start = self.buf.len() - FT4_SLOT_WINDOW_SAMPLES;
        let window = self.buf[start..].to_vec();
        self.last_decoded_slot = slot_ms;
        Some((window, self.params))
    }
}

impl Default for Ft4Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared handle to a [`Ft4Decoder`] behind a [`Mutex`] — the type both the
/// demod tap and the decode task hold.
pub type SharedDecoder = Arc<Mutex<Ft4Decoder>>;

/// Build the shared [`Arc<Mutex<Ft4Decoder>>`] both halves of the pipeline
/// hold: the demod thread (via [`Ft4Tap`]) appends, the decode task decodes
/// via [`decode_closed_slot`].
pub fn shared() -> SharedDecoder {
    Arc::new(Mutex::new(Ft4Decoder::new()))
}

// ────────────────────────────────────────────────────────────────────────────
// Demod-thread tap

/// The demod-thread half of the FT4 pipeline: adapts a
/// [`SharedDecoder`] to the [`RawSampleTap`] trait the
/// [`super::demod::DigitalDemodulator`] calls while demodulating. `append`
/// is a short critical section (a bounded `Vec` copy + cap trim) and never
/// runs a decode.
///
/// `Send + Sync` (it only holds an `Arc<Mutex<_>>`), so it boxes cleanly
/// into the demod's `Option<Box<dyn RawSampleTap>>`.
#[derive(Debug, Clone)]
pub struct Ft4Tap {
    inner: SharedDecoder,
}

impl Ft4Tap {
    /// Wrap an existing [`SharedDecoder`] so the demod and a decode task
    /// share the same buffer.
    pub fn from_shared(inner: SharedDecoder) -> Self {
        Self { inner }
    }

    /// A clone of the shared decoder, for the decode task to hand to
    /// [`decode_closed_slot`].
    pub fn shared(&self) -> SharedDecoder {
        self.inner.clone()
    }
}

impl Default for Ft4Tap {
    fn default() -> Self {
        Self { inner: shared() }
    }
}

impl RawSampleTap for Ft4Tap {
    fn append(&self, samples: &[f32]) {
        let mut g = self.inner.lock().expect("Ft4Decoder poisoned");
        g.add_samples(samples);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Decode

/// Compute the most-recently-closed FT4 slot boundary (a multiple of
/// [`FT4_SLOT_MS`]) for a wall-clock instant in ms — the slot whose window
/// is now fully in the past.
pub fn closed_slot_for(now_ms: u64) -> u64 {
    (now_ms / FT4_SLOT_MS) * FT4_SLOT_MS
}

/// Decode the trailing 7.5 s window of `decoder` for `slot_ms`, returning
/// the CRC-passing rows (empty if the slot was already decoded, the buffer
/// is short, or no message decoded). A `slot_ms` not a multiple of
/// [`FT4_SLOT_MS`] is a caller bug -> `Err`.
///
/// The `mfsk-core` decode runs *off* the buffer lock: the trailing window
/// is cloned out under the lock, the lock is released, and the decode
/// proceeds on the caller's thread (the decode task — never the demod hot
/// loop).
pub fn decode_closed_slot(
    decoder: &SharedDecoder,
    slot_ms: u64,
) -> Result<Vec<Ft4Message>, Box<dyn std::error::Error + Send + Sync>> {
    if slot_ms % FT4_SLOT_MS != 0 {
        return Err(format!("slot_ms={slot_ms} is not a multiple of {FT4_SLOT_MS}").into());
    }
    // Benign cases (the slot was already decoded, or the buffer is still
    // short of a full 7.5 s window) yield empty rows, not an error — the
    // decode task just skips the broadcast.
    let (window, params) = {
        let mut g = decoder.lock().expect("Ft4Decoder poisoned");
        match g.snapshot_for_slot(slot_ms) {
            Some(wp) => wp,
            None => return Ok(Vec::new()),
        }
    };
    decode_window(&window, slot_ms, &params)
}

/// Run the `mfsk-core` FT4 batch decode on one trailing 7.5 s window and
/// return the decoded rows (empty if the window held silence / no
/// CRC-passing hit). Free function so the decode can run off the decoder
/// lock — see [`decode_closed_slot`].
fn decode_window(
    window: &[f32],
    slot_ms: u64,
    params: &DecodeParams,
) -> Result<Vec<Ft4Message>, Box<dyn std::error::Error + Send + Sync>> {
    // `f32` (natural [-1, +1] units, the demod's pre-AGC output) -> `i16`
    // full-scale. The decoder estimates SNR from the *relative* amplitudes
    // of the 4-GFSK tones; a uniform full-scale clamp is exactly the
    // normalisation it expects, and the AGC's per-block RMS retargeting (a
    // cosmetic normalisation for human listening) would only perturb the
    // SNR a digital decoder should see.
    //
    // Before the `f32` -> `i16` conversion, rescale the window's absolute
    // peak to [`super::audio_scale::PEAK_TARGET`] (≈ −1.4 dBFS — "green,
    // just under red", the level the reference decoders were tuned
    // against). The EP6 baseband is typically ~10×-100× quieter than that
    // level, and mfsk-core's absolute-threshold gates and headroom
    // assumptions are calibrated for inputs near it. See
    // [`super::audio_scale`] for the rationale; the one pass over the
    // window is a tiny fraction of the decode cost that follows.
    let mut window = window.to_vec();
    let gain = crate::receiver::audio_scale::peak_normalize(&mut window);
    if std::env::var_os("HL2_DEBUG").is_some() {
        let db = if gain > 0.0 && gain.is_finite() {
            20.0 * gain.log10()
        } else {
            0.0
        };
        eprintln!(
            "[aud] ft4 slot={slot_ms} peaknorm={db:+.1} dB (peak→{:.3}, pre-AGC into decoder)",
            crate::receiver::audio_scale::PEAK_TARGET
        );
    }
    let samples: Vec<i16> = window
        .iter()
        .map(|v| (v.clamp(-1.0, 1.0) * 32_767.0) as i16)
        .collect();

    let req = DecodeRequest::<Ft4>::new(
        &samples,
        params.freq_min,
        params.freq_max,
        params.sync_min,
        params.max_cand,
    );
    let out = req.decode();
    let mut msgs = Vec::with_capacity(out.results.len());
    for r in &out.results {
        if let Some(d) = r.to_decoded(ProtocolId::Ft4, None) {
            msgs.push(Ft4Message {
                text: d.text,
                freq_hz: d.freq_hz,
                dt_sec: d.dt_sec,
                snr_db: d.snr_db,
                slot_ms,
            });
        }
    }
    Ok(msgs)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    #[test]
    fn window_caps_at_10s() {
        let mut d = Ft4Decoder::new();
        d.add_samples(&silence(12 * 12_000)); // 12 s — over the 10 s cap
        assert_eq!(d.buf.len(), 10 * 12_000);
    }

    #[test]
    fn append_is_the_tap() {
        let d = shared();
        let tap = Ft4Tap::from_shared(d.clone());
        tap.append(&[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]);
        tap.append(&[1.0]);
        let got = d.lock().unwrap().buf.clone();
        assert_eq!(got, vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 1.0]);
    }

    /// A silence slot at 12 kHz should decode cleanly (0 rows) and exercise
    /// the `mfsk-core` path without panicking.
    #[test]
    fn silence_slot_roundtrip() {
        let d = shared();
        d.lock()
            .unwrap()
            .add_samples(&silence(FT4_SLOT_WINDOW_SAMPLES));
        let msgs = decode_closed_slot(&d, 15_000).unwrap();
        assert!(msgs.is_empty(), "silence should yield no rows");
    }

    #[test]
    fn non_multiple_of_slot_errors() {
        let d = shared();
        d.lock()
            .unwrap()
            .add_samples(&silence(FT4_SLOT_WINDOW_SAMPLES));
        assert!(decode_closed_slot(&d, 17_500).is_err());
    }

    /// Double-decoding the same slot is benign (returns empty, not an
    /// error) — a duplicate boundary tick must not surface as a failure.
    #[test]
    fn decode_same_slot_twice_is_benign() {
        let d = shared();
        d.lock()
            .unwrap()
            .add_samples(&silence(FT4_SLOT_WINDOW_SAMPLES));
        assert!(decode_closed_slot(&d, 15_000).unwrap().is_empty());
        assert!(decode_closed_slot(&d, 15_000).unwrap().is_empty());
    }

    /// A synthetic FT4-like burst: a full-slot 1.5 kHz sine. Exercises the
    /// `mfsk-core` decode path end-to-end (a bare sine will not decode to a
    /// real message — that is the live-radio test's job).
    #[test]
    fn tone_burst_roundtrip() {
        let d = shared();
        let n = FT4_SLOT_WINDOW_SAMPLES;
        let mut sig = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / 12_000.0;
            sig.push((2.0 * std::f32::consts::PI * 1_500.0 * t).sin() * 0.3);
        }
        d.lock().unwrap().add_samples(&sig);
        let _ = decode_closed_slot(&d, 15_000).unwrap();
    }

    /// A real self-synthesised FT4 signal (mfsk-core's own TX chain —
    /// `pack77` -> `message_to_tones` -> `tones_to_f32`) placed at the 0.5 s
    /// TX offset inside a 7.5 s slot must decode back to the original 77-bit
    /// message through our `decode_window` path.
    #[test]
    fn self_synthesised_ft4_decodes() {
        use mfsk_core::ft4::encode::{message_to_tones, tones_to_f32};
        use mfsk_core::msg::wsjt77::pack77;

        let m77 = pack77("CQ", "K1ABC", "FN42").expect("pack77");
        let tone = message_to_tones(&m77);
        let pcm = tones_to_f32(&tone, 1_500.0, 0.5); // f32, peak 0.5

        let mut window = vec![0f32; FT4_SLOT_WINDOW_SAMPLES];
        // 0.5 s TX offset => 6 000 samples at 12 kHz. Same convention
        // mfsk-core's own decode tests use.
        let off = 6_000usize;
        let len = pcm.len().min(window.len() - off);
        window[off..off + len].copy_from_slice(&pcm[..len]);

        let msgs = decode_window(&window, 22_500, &DecodeParams::default()).expect("decode");
        assert!(
            !msgs.is_empty(),
            "expected >= 1 CRC-passing row from a clean self-synthesized FT4 signal"
        );
        let first = &msgs[0];
        assert_eq!(first.slot_ms, 22_500);
        assert!(
            first.text.contains("K1ABC"),
            "decoded text should contain the callsign, got {:?}",
            first.text
        );
        // The tone-0 search should land within a bin or two of 1.5 kHz.
        assert!(
            (first.freq_hz - 1_500.0).abs() < 4.0,
            "freq_hz {} should be ~1500 Hz",
            first.freq_hz
        );
    }
}
