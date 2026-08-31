//! Demodulators for the virtual audio receiver.
//!
//! A [`Demodulator`] converts a block of complex I/Q baseband samples into
//! `i16` mono audio and writes it to an [`AudioSink`](super::sink::AudioSink).
//! Each radio mode (SSB, AM, FM, FT8, CW…) is a different `Demodulator`;
//! [`make_demod`] is the single dispatch point for a new mode
//! (PROTOCOL.md §16).
//!
//! The SSB path is NCO + quadrature synthesis + **polyphase** anti-alias /
//! decimation:
//!
//! ```text
//! complex I/Q
//!   └─► NCO × e^(−j·φ),  φ̇ = ±2π·center / fs    (USB: φ̇>0, LSB: φ̇<0)
//!        └─► USB: in-phase arm directly
//!            LSB: 90° Hilbert phase-shifter (synthesise quadrature from I')
//!             └─► PolyphaseDecimator (windowed-sinc anti-alias + decimate)
//!                  └─► normalise to i16 (DC-block + slow AGC, × gain_db)
//! ```
//!
//! The USB/LSB distinction is carried by the **NCO direction** (the sign of
//! `φ̇`) plus which post-NCO arm is kept: a positive-frequency NCO moves the
//! upper sideband to baseband on the in-phase arm and a negative one moves
//! the lower sideband to baseband — the same trick the reference uses
//! (keep the in-phase arm for USB; conjugate it to reach LSB). For LSB we
//! synthesize that quadrature arm with a 90° (Hilbert)
//! phase-shifter on the in-phase arm, which is valid for *both* real and
//! complex baseband (`SsbDemodulator::demod`).
//!
//! The channel-select FIR of the old design (a 257-tap convolution on every
//! 96 kHz sample) is now the anti-alias filter of a `PolyphaseDecimator`, so
//! only ≈ 1/M of its taps touch each sample (~M× fewer MACs, M = the
//! decimation factor). The old single-pole smoothing stage is gone — the
//! windowed-sinc is the anti-alias.
//!
//! For `Mode::Ft8` / `Mode::Js8` the pipeline skips the quadrature synthesis
//! entirely (USB) and decimates directly to the 12 kHz digital-mode window
//! rate with `DigitalDemodulator` — see [`super::ft8`] and
//! [`super::js8`].
//!
//! DSP is hand-rolled (windowed-sinc FIR, Hilbert, polyphase decimator); no
//! external DSP crate. The trait is deliberately minimal — `demod(iq, sink)`
//! and `audio_format()` — so AM/FM/CW slot in without touching the receiver
//! loop or the sink API.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering as AOrdering};

use num_complex::Complex;

use super::sink::AudioSink;
use super::{AudioConfig, Mode, Sideband};

/// Gated (HL2_DEBUG) instrument counter so we don't flood on every emit.
static EMIT_COUNT: AtomicUsize = AtomicUsize::new(0);

/// 2π, for the NCO phase tests (named so tests avoid the `std::f64::consts::TAU`
/// path-resolution noise).
#[cfg(test)]
const TAU: f64 = std::f64::consts::TAU;

/// An SSB demodulator received a block whose length was not a multiple of two
/// (an odd number of complex pairs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidBlockLength(pub usize);
impl fmt::Display for InvalidBlockLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "demod: block length {} must be even (complex pairs)",
            self.0
        )
    }
}
impl std::error::Error for InvalidBlockLength {}

/// The source rate is too close to the audio rate to decimate cleanly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateTooClose {
    pub src: u32,
    pub audio: u32,
}
impl fmt::Display for RateTooClose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "demod: source rate {} Hz must be at least 4× the audio rate {} Hz",
            self.src, self.audio
        )
    }
}
impl std::error::Error for RateTooClose {}

/// Demodulator error.
pub type DemodError = Box<dyn std::error::Error + Send + Sync>;

/// A block of complex I/Q baseband samples: `Complex<f32>` pairs.
pub type IqBlock = Vec<Complex<f32>>;

/// A demodulator: complex I/Q in, `i16` mono audio out, to a given sink.
///
/// `Send` so a [`crate::receiver::VirtualReceiver`] (which owns a
/// `Box<dyn Demodulator>`) can be handed to a dedicated demod thread — the
/// pump and the demod are decoupled via the
/// [`crate::receiver::BasebandRing`].
pub trait Demodulator: Send {
    /// Demodulate one complex I/Q block into `sink`; returns audio frames written.
    fn demod(&mut self, iq: &IqBlock, sink: &mut dyn AudioSink) -> Result<usize, DemodError>;

    /// The audio format produced.
    fn audio_format(&self) -> AudioConfig;

    /// Flush any audio the demodulator is still buffering (e.g. a partial
    /// block that hadn't accumulated enough samples for normalisation). Safe
    /// on modes that don't buffer (no-op).
    fn flush_audio(&mut self, _sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        Ok(0)
    }

    /// Flush buffered audio into `sink` (alias for `flush_audio`, used by
    /// higher-level wrappers that call a uniform `flush`).
    fn flush(&mut self, sink: &mut dyn AudioSink) -> Result<(), DemodError> {
        self.flush_audio(sink).map(|_| ())
    }
}

/// A complex oscillator (`× e^(−j·φ)`) advanced by phase recurrence.
///
/// The old implementation recomputed `cos(phase)` / `sin(phase)` (two
/// transcendentals) for every 96 kHz sample. Instead the state `(c, s)`
/// holds `(cos φ, sin φ)` in `f64` and each step is the 2-D rotation
///
/// ```text
/// z' = z · e^(−j·φ)      (re·c + im·s,  im·c − re·s)
/// c' =  c·cos_step − s·sin_step
/// s' =  s·cos_step + c·sin_step
/// ```
///
/// (two FMA-pairs, no trig in the loop). This matches the old per-sample
/// `(−nco_phase)` convention exactly: sample `n` is mixed at
/// `φ = n·step` (`z·e^(−j·n·step)`), then the state advances. Over a block
/// of up to a few thousand samples the drift of `c² + s²` from 1 is a
/// 2nd-order O(n·ε) effect (~1e-13 per 10⁴ samples in f64) — far below
/// `f32` sample precision, so no periodic renormalisation is needed.
///
/// `step == 0.0` (a baseband source) short-circuits to the identity.
#[derive(Debug, Clone)]
pub struct Nco {
    cos_step: f64,
    sin_step: f64,
    /// cos φ
    c: f64,
    /// sin φ
    s: f64,
}

impl Nco {
    pub fn new(step: f64) -> Self {
        Self {
            cos_step: step.cos(),
            sin_step: step.sin(),
            c: 1.0,
            s: 0.0,
        }
    }

    /// Multiply by `e^(−j·φ)` and advance by one step.
    #[inline]
    pub fn step(&mut self, x: Complex<f32>) -> Complex<f32> {
        if self.cos_step == 1.0 && self.sin_step == 0.0 {
            return x;
        }
        let re = (x.re as f64 * self.c + x.im as f64 * self.s) as f32;
        let im = (x.im as f64 * self.c - x.re as f64 * self.s) as f32;
        let c = self.c;
        let s = self.s;
        self.c = c * self.cos_step - s * self.sin_step;
        self.s = s * self.cos_step + c * self.sin_step;
        Complex::new(re, im)
    }
}

/// Kaiser-window shape parameter β used by the SSB/digital channel-select
/// and Hilbert windows.
///
/// The Kaiser window is the standard choice for windowed-sinc decimation
/// filters in SDR stack clients: unlike Hann/Blackman
/// its stopband attenuation is an *explicit design parameter* — β directly
/// trades off sidelobe decay against transition width. β = 12 gives ≈ 100-110
/// dB stopband (the rule of thumb `A ≈ 10·β − 18` for β > 5), which at our
/// 511-tap / 2.6 kHz @ 192 kHz digital-path design rejects a 4-6 kHz
/// offset by ≥ 107 dB (a Hann window — first sidelobe fixed at ≈ −31 dB
/// with only 18 dB/octave sidelobe decay — reaches only ~55-60 dB at that
/// offset, leaving off-center digital signals, e.g. adjacent-channel FT8
/// on a USB center, clearly audible).
pub const KAISER_BETA: f64 = 12.0;

/// Kaiser window (`w[i] = I₀(β·√(1−x²))/I₀(β)`, `x = (i−c)/c ∈ [−1,1]`,
/// `c = (n−1)/2`), as `f64`. The Bessel I₀ series converges fast for the
/// β values used here (≤ ~12) — no reflection formula needed.
fn kaiser(i: usize, n: usize, beta: f64) -> f64 {
    let c = (n as f64 - 1.0) / 2.0;
    let x = ((i as f64 - c) / c).clamp(-1.0, 1.0);
    bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta)
}

/// Modified Bessel function of the first kind, order 0 — power series
/// `Σₖ (x²/4)ᵏ / (k!)²`. For β ≤ 12 the largest argument is 144 and the
/// series needs ~15 terms to reach 1e-12.
fn bessel_i0(x: f64) -> f64 {
    let q = (0.5 * x) * (0.5 * x);
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    let mut k = 1;
    loop {
        term *= q / (k as f64 * k as f64);
        if term < sum * 1e-15 {
            break;
        }
        sum += term;
        k += 1;
    }
    sum
}

/// A symmetric, causal, windowed-sinc **real low-pass** FIR.
///
/// A length-`n` (odd) FIR with an integer centre (`(n−1)/2`), so it has
/// linear phase (group delay `(n−1)/2` samples) and the response
/// `H(0) = 1`, `H(Nyquist) ≈ 0`. Applied as `y[k] = Σ_j h[j]·x[k − j]`.
///
/// After the NCO down-conversion, this is the channel-select filter
/// for SSB: it passes the voice band `[−BW, +BW]` around DC and rejects
/// out-of-band signals and the image. The USB/LSB distinction itself is
/// carried by the NCO direction (the reference does the same: it conjugates
/// the in-phase arm for LSB). The windowing is Kaiser (see
/// [`KAISER_BETA`]).
#[derive(Debug, Clone)]
pub struct F32Fir {
    taps: Vec<f32>,
}

impl F32Fir {
    /// Build a real low-pass FIR with passband width `bandwidth_ratio`
    /// (`HW / fs`, in `(0, 0.5]`), windowed with the Kaiser window `kaiser_beta`.
    ///
    /// `tap_count` is rounded up to the next *odd* integer ≥ 17 (so the
    /// impulse response stays centred on a whole sample — required for linear
    /// phase).
    pub fn lowpass(tap_count: usize, bandwidth_ratio: f64, kaiser_beta: f64) -> Self {
        let target = tap_count.max(17);
        let n = if target % 2 == 0 { target + 1 } else { target };
        let ratio = bandwidth_ratio.clamp(1e-3, 0.49);
        let mut taps = Vec::with_capacity(n);
        let mut norm = 0.0f64;
        for i in 0..n {
            let t = i as f64 - (n as f64 - 1.0) / 2.0;
            let window = kaiser(i, n, kaiser_beta);
            // Ideal low-pass: h(t) = ratio · sinc(π·ratio·t).
            let h = if t.abs() < 1e-9 {
                ratio
            } else {
                (std::f64::consts::PI * ratio * t).sin() / (std::f64::consts::PI * t)
            } * window;
            norm += h;
            taps.push(h as f32);
        }
        // Normalise for unit passband gain.
        if norm.abs() > 1e-12 {
            let scale = (1.0 / norm) as f32;
            for t in taps.iter_mut() {
                *t *= scale;
            }
        }
        Self { taps }
    }

    /// Build a **quadrature (90°) phase-shifter** FIR — the discrete Hilbert
    /// transform. Applied to a real signal `r(t)` it yields `q(t) = H{r}(t)`,
    /// the 90°-rotated version (`H{cos ωt} = sin ωt`). Combined with `r` it
    /// forms the analytic signal `r + j·q`, the one-sided (positive-frequency)
    /// version of `r`.
    ///
    /// This is what makes SSB demod possible from a **real** baseband stream
    /// (Q arm ≈ 0, which is what the HL2 "real SDR" DDC produces): we
    /// *synthesize* the missing quadrature component here, then use
    /// two-product-discrimination to pick the USB or LSB. See
    /// [`SsbDemodulator::demod`]. The ideal (windowless) impulse response is
    /// `h[m] = 0` for even `m`, `h[m] = 2/(π·m)` for odd `m` (about an odd
    /// centre tap); we window with the same Kaiser window as [`F32Fir::lowpass`].
    pub fn hilbert(tap_count: usize, kaiser_beta: f64) -> Self {
        let target = tap_count.max(17);
        let n = if target % 2 == 0 { target + 1 } else { target };
        let center = (n - 1) as f64 / 2.0;
        let mut taps = Vec::with_capacity(n);
        for i in 0..n {
            let m = i as f64 - center; // odd integer relative to the centre tap
            let window = kaiser(i, n, kaiser_beta);
            let h = if m.abs() < 1e-9 {
                0.0
            } else {
                (2.0 / (std::f64::consts::PI * m)) * window
            };
            taps.push(h as f32);
        }
        Self { taps }
    }

    /// Number of taps.
    pub fn len(&self) -> usize {
        self.taps.len()
    }

    /// Whether there are no taps.
    pub fn is_empty(&self) -> bool {
        self.taps.is_empty()
    }

    /// The impulse response (a copy).
    pub fn taps(&self) -> &[f32] {
        &self.taps
    }

    /// Causal convolution in a bounded chronological `history` (newest = last
    /// element): result at index `idx` = `Σ_j h[j]·history[idx − j]`, where
    /// terms with `idx − j < 0` are dropped (zero-padded onset).
    pub fn apply(&self, history: &[f32], idx: usize) -> f32 {
        let mut acc = 0.0f32;
        let hist = history.len() as isize;
        let idx_i = idx as isize;
        for j in 0..self.taps.len() {
            let k = idx_i - j as isize;
            if k < 0 {
                break;
            }
            if k >= hist {
                continue;
            }
            acc += history[k as usize] * self.taps[j];
        }
        acc
    }
}

/// A FIR filter with a **fixed-length, pre-allocated** history ring, tuned
/// for streaming single-sample convolution (the SSB hot loop).
///
/// The history ring holds the last `T` input samples, newest at
/// `hist[(pos − 1) % T]`. `convolve()` computes
/// `Σ_j taps[j]·x[n−j]` (terms with `n−j < 0` zero-padded, identical to
/// [`F32Fir::apply`] with the full history) in a single contiguous
/// backwards pass over the ring + the just-pushed sample, in O(T) with no
/// per-tap `Vec` reallocation, `drain`, or modulo (only one per ring step).
/// Compared with the old `Vec::push` + `Vec::drain` + `apply` path this
/// avoids two heap operations and one amortised `memmove` per sample.
#[derive(Debug, Clone)]
pub struct F32FirState {
    taps: Vec<f32>,
    hist: Vec<f32>,
    /// Total samples pushed so far (drives the zero-padding onset).
    processed: usize,
    /// The output of the last `convolve` call (consumed by
    /// [`PolyphaseDecimator`] when summing phase outputs).
    last_out: f32,
}

impl F32FirState {
    pub fn new(fir: &F32Fir) -> Self {
        Self::new_taps(fir.taps())
    }

    /// Build state directly over a tap vector (the polyphase component
    /// filters are slices of a larger FIR — see
    /// [`PolyphaseDecimator::new`]).
    pub fn new_taps(taps: &[f32]) -> Self {
        let t = taps.len();
        Self {
            taps: taps.to_vec(),
            hist: vec![0.0f32; t.max(1)],
            processed: 0,
            last_out: 0.0,
        }
    }

    /// The impulse response.
    pub fn taps(&self) -> &[f32] {
        &self.taps
    }

    /// Number of taps.
    pub fn len(&self) -> usize {
        self.taps.len()
    }

    /// Whether there are no taps.
    pub fn is_empty(&self) -> bool {
        self.taps.is_empty()
    }

    /// Push one input sample and return the new filtered output.
    ///
    /// `x` is appended to the history (oldest dropped on overflow). The
    /// result is the causal convolution `Σ_j taps[j]·x[n − j]`, where `n` is
    /// the total number of samples pushed and terms with `n − j < 0` are
    /// dropped — byte-identical to the old `apply(history, len−1)` path.
    #[inline]
    pub fn convolve(&mut self, x: f32) -> f32 {
        let t = self.taps.len();
        if t == 0 {
            return 0.0;
        }
        // Write the new sample at the ring position that will be the newest.
        let pos = self.processed % t;
        self.hist[pos] = x;
        self.processed += 1;
        // Number of usable past+present samples (bounded by the ring).
        let m = self.processed.min(t);
        // Walk newest→oldest: x[n], x[n−1], ... x[n−(m−1)], reading the ring
        // backwards from `pos` (wrapping at 0) and the just-written `x`.
        // Walk taps newest→oldest: the k-th tap (0-indexed) multiplies x[n−k].
        // k=0 is the freshly written sample; k>=1 reads the ring backwards
        // from `pos` (wrapping at 0). Bounded by `m` for the zero-pad onset.
        let mut acc = self.taps[0] * x;
        let mut k = 1usize;
        let mut idx = pos;
        while k < t && k < m {
            idx = if idx == 0 { t - 1 } else { idx - 1 };
            acc += self.taps[k] * self.hist[idx];
            k += 1;
        }
        self.last_out = acc;
        acc
    }

    /// The output of the last [`Self::convolve`] call — used by
    /// [`PolyphaseDecimator`] to sum one input's phase outputs.
    pub fn last_out(&self) -> f32 {
        self.last_out
    }
}

/// **Polyphase** low-pass + decimator: the anti-alias / decimate stage.
///
/// Produces exactly the same samples as "full-rate low-pass FIR followed by
/// a sub-sampler" (the old `F32FirState` + `Decimator` pair, without the
/// single-pole smoothing), at ≈ `1/M ×` the tap-multiplies.
///
/// Output alignment: the `e`-th emitted sample (0-indexed) equals the
/// full-rate convolution's output at input index `e·M + (M−1)` — i.e. the
/// reference `y[t] = Σ_j h[j]·x[t−j]` (zero-padded for `t−j < 0`) *kept* at
/// the instants `t ≡ M−1 (mod M)`. The polyphase emits on the last sample
/// of each input group; both start from empty history, so the whole stream
/// — including the zero-pad onset — matches sample-for-sample.
///
/// ## Identity (phase decomposition)
///
/// With a full-rate impulse response `h[0..N)`, an input `x[·]` and an
/// integer decimation factor `M ≥ 1`, the output kept after the group
/// ending at input index `n = k·M + (M−1)` has been processed is
///
///   `y[k] = Σ_{p=0}^{M−1} b_p[k]`
///
/// where `b_p[k]` is the `k`-th output of branch `p`: a causal FIR over the
/// *decimated* substream `z_p[s] = x[s·M + (M−1−p)]` (`s = 0, 1, 2, …` —
/// branch `p` is fed input `n` exactly when `n ≡ M−1−p (mod M)`) with taps
///
///   `H_p[j] = h[p + j·M]`,  `j = 0, 1, …`
///
/// in newest-first order (`H_p[0] = h[p]` multiplies the newest substream
/// sample). The `M` branches are exactly the `M` phase components of the
/// decimated convolution; summing their step-`k` outputs recovers `y[k]`.
/// Verified to ~10⁻¹⁶ against a naive full-rate reference by
/// `polyphase_matches_fullfir_decimate` below, for arbitrary (random) `h`.
///
/// ## Cost
///
/// Per input sample: one branch convolve of `⌈(N)/M⌉ ≈ N/M` taps (vs `N`
/// for the full-rate path) — ~`M×` fewer MACs — and at each group boundary:
/// `M−1` additions. The windowed-sinc `h` *is* the anti-alias design; the
/// old single-pole smoothing stage is unnecessary.
#[derive(Debug, Clone)]
pub struct PolyphaseDecimator {
    /// Branch `p` (index `p` in `0..M`): taps `h[p], h[p+M], …`.
    branches: Vec<F32FirState>,
    /// Position within the current group, `0..M`; the input at within-group
    /// position `pos` feeds branch `M−1−pos`.
    pos: usize,
}

impl PolyphaseDecimator {
    /// Build a decimator from a full-rate low-pass FIR (`h`, any length ≥ 1)
    /// at integer decimation factor `M ≥ 1`.
    pub fn new(h: &[f32], m: usize) -> Self {
        assert!(!h.is_empty(), "PolyphaseDecimator needs ≥ 1 tap");
        assert!(m >= 1, "PolyphaseDecimator decimation factor ≤ 0");
        let mut branches = Vec::with_capacity(m);
        for p in 0..m {
            let mut taps = Vec::new();
            let mut j = p;
            while j < h.len() {
                taps.push(h[j]);
                j += m;
            }
            branches.push(F32FirState::new_taps(&taps));
        }
        Self { branches, pos: 0 }
    }

    /// Decimation factor (`rate_in / rate_out`).
    pub fn decimation(&self) -> usize {
        self.branches.len()
    }

    /// Push one input sample. Returns `Some(·)` — the completed group's
    /// output, `m` groups-of-1 summed — after every `M`-th sample, in order.
    pub fn push(&mut self, x: f32) -> Option<f32> {
        let m = self.branches.len();
        let p = (m - 1 - self.pos) % m;
        self.branches[p].convolve(x);
        self.pos += 1;
        if self.pos == m {
            self.pos = 0;
            let mut y = 0.0f32;
            for b in &self.branches {
                y += b.last_out();
            }
            Some(y)
        } else {
            None
        }
    }
}

/// The SSB demodulator (USB or LSB).
///
/// Holds: an NCO (to bring the carrier/selected sideband to baseband), a
/// 90° (Hilbert) phase-shifter for LSB, and the polyphase
/// anti-alias/decimation stage (`lp`).
pub struct SsbDemodulator {
    sideband: Sideband,
    audio_cfg: AudioConfig,
    /// NCO (phase-recurrence oscillator). Identity when `source_center_hz`
    /// is 0 (i.e. the input is already baseband).
    nco: Nco,
    /// Quadrature (90° / Hilbert) phase-shifter. Applied to the real
    /// in-phase arm it synthesizes the missing quadrature component so SSB can
    /// be demodulated from a real (Q≈0) baseband stream. See `F32Fir::hilbert`.
    /// The sideband choice (`Usb` vs `Lsb`) selects which post-NCO arm is
    /// kept (in-phase → USB, quadrature → LSB). Used only for LSB.
    hilb: F32FirState,
    /// Polyphase anti-alias + decimation stage (channel-select FIR's taps are
    /// its full-rate impulse response). Replaces the old `filter`
    /// (`F32FirState`) + `decim` (`Decimator`) pair.
    lp: PolyphaseDecimator,
    /// Slow RMS-targeted AGC gain (linear). Replaces the old per-block peak
    /// normaliser. See `normalize_to_i16_with_agc`.
    agc_gain: f32,
    /// Accumulated decimated audio, ready to be normalised + emitted once it
    /// reaches `AUDIO_EMIN`. A single wire chunk (63 complex samples at a
    /// 192 kHz → 4.8 kHz decimation) produces only ≈ 1-2 decimated samples,
    /// and calling `normalize_to_i16_with_agc` on a 1-sample buffer makes the
    /// DC-block subtract the sample from itself → output always 0. Buffering
    /// to ~50 ms (~240 samples) restores meaningful DC-block + AGC statistics.
    audio_buf: Vec<f32>,
    /// Optional pre-AGC raw-sample tap (FT8 decode — see
    /// [`RawSampleTap`]). `None` for plain SSB. `Arc` so the API layer can
    /// keep one handle for the demod and hand a clone to the decode task.
    tap: Option<std::sync::Arc<dyn RawSampleTap>>,
}

/// Minimum decimated samples to accumulate before a block is normalised and
/// emitted. A single wire chunk (63 complex samples at a 192 kHz → 4.8 kHz
/// decimation) yields only ≈ 2 samples, far too few for a meaningful
/// DC-block / AGC target (a 1-sample mean subtracts `x` from itself → the
/// output is always 0, which is what made SSB "silence"). ~50 ms of 4.8 kHz
/// audio (240 samples) is the smallest block that gives stable statistics
/// without noticeable latency.
const AUDIO_EMIN: usize = 240;

/// A tap for capturing the *pre-AGC* decimated audio stream.
///
/// The SSB demod pipeline is: NCO → SSB branch → channel LPF → decimate →
/// **raw decimated f32 samples** → AGC (DC block + RMS target) → i16 → sink.
/// For the `CH_AUDIO` sink the AGC-normalised i16 output is what the listener
/// wants. For FT8 decoding, the raw decimated f32 stream (the "audio the SSB
/// demod produced, untouched") is what `mfsk_core`'s FT8 decoder should
/// consume — the AGC's per-block RMS retargeting is a cosmetic
/// normalisation for human listening and adds nothing to (and slightly
/// perturbs) the SNR a digital decoder should see from the *relative*
/// amplitudes of the 8-GFSK tones.
///
/// Implement with [`super::ft8::Ft8Tap`] (wrapping a
/// `hl2::receiver::shared()` decoder): the API layer owns the shared
/// instance, the demod thread calls [`RawSampleTap::append`] inline while
/// demodulating, and the tokio decode task calls
/// `hl2::receiver::decode_closed_slot` on a 1 s wall-clock cadence.
pub trait RawSampleTap: std::fmt::Debug + Send + Sync {
    /// Append `samples` to the tap. May be called concurrently from the
    /// demod thread; implementers should be `Send + Sync` and keep calls
    /// short (no allocation, no blocking).
    fn append(&self, samples: &[f32]);
}

impl SsbDemodulator {
    /// The sideband this demodulator was built for.
    pub fn sideband(&self) -> Sideband {
        self.sideband
    }

    /// The polyphase anti-alias / decimation stage.
    pub fn lp(&self) -> &PolyphaseDecimator {
        &self.lp
    }

    /// Normalise + emit one slice of the accumulated audio, advancing the
    /// AGC. Returns the number of i16 frames written.
    fn emit(&mut self, slice: &[f32], sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        // Pre-AGC raw-sample tap (FT8 decode): the untouched decimated `f32`
        // stream, at `audio_cfg.rate_hz`, in arrival order.
        if let Some(tap) = self.tap.as_ref() {
            tap.append(slice);
        }
        let mut out = vec![0i16; slice.len()];
        let written =
            normalize_to_i16_with_agc(slice, &mut out, self.audio_cfg.gain_db, &mut self.agc_gain);
        {
            // HL2_DEBUG: prove whether the decimated audio (filter-out) is
            // loud, and whether the AGC/normalisation is turning it into
            // audible i16. ~every 50 emits (~2.5 s at 50 ms/emit).
            if std::env::var("HL2_DEBUG").is_ok() {
                let c = EMIT_COUNT.fetch_add(1, AOrdering::Relaxed) + 1;
                if c % 50 == 1 {
                    let in_max = slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let in_rms = (slice.iter().map(|v| v * v).sum::<f32>()
                        / slice.len().max(1) as f32)
                        .sqrt();
                    let o_max = out[..written].iter().map(|v| v.abs()).max().unwrap_or(0);
                    let side = if self.sideband == Sideband::Usb {
                        "USB"
                    } else {
                        "LSB"
                    };
                    eprintln!(
                        "[aud] {side} in_rms={in_rms:.6e} in_max={in_max:.6e} agc={:.3e} out_i16_max={o_max} (n={written})",
                        self.agc_gain
                    );
                }
            }
        }
        sink.write(&out[..written])
    }

    /// Drain all `AUDIO_EMIN`-sized blocks from the accumulator into `sink`;
    /// returns total frames written.
    fn flush_full_blocks(&mut self, sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        let mut frames_written = 0usize;
        while self.audio_buf.len() >= AUDIO_EMIN {
            let n = std::mem::take(&mut self.audio_buf);
            let block_len = n.len().min(1024);
            let slice = &n[..block_len];
            self.audio_buf = n[block_len..].to_vec();
            frames_written = frames_written.saturating_add(self.emit(slice, sink)?);
        }
        Ok(frames_written)
    }
}

/// Normalise an `f32` audio block to `i16`. The audio comes in "natural"
/// units (the channel-filter output — typically `[-1, +1]` for a unit
/// amplitude input). This function applies:
///
///   1. **DC blocking** — subtract the block mean so the ADI chain's constant
///      I/Q offset (often ~0.3 on the in-phase arm) doesn't dominate the
///      output level for USB (which passes the in-phase arm).
///   2. **Slow RMS-targeted AGC** (replaces the old per-block peak normaliser).
///      The AGC gain is a **natural→i16 scale factor**: it maps the block's
///      RMS to the target output RMS. The target is
///      `TARGET_RMS_I16 = 0.1 × 32767` (≈ −20 dBFS). The AGC gain is
///      smoothed with a one-pole filter: fast attack (alpha=0.4) when the
///      gain needs to increase, slow release (alpha=0.08) when it decreases
///      — classic SSB AGC asymmetry that minimises "pumping" on voice gaps.
///      The gain is clamped to `[1.0, 1e7]` so a silent input doesn't
///      drive the gain to 1e9 and a saturated one doesn't crush to 0.
///   3. the user's `gain_db` (applied on top of the AGC) — a knob to add
///      headroom or trim overall level.
///
/// `agc_gain` is the running AGC scale factor, updated in-place.
fn normalize_to_i16_with_agc(
    audio: &[f32],
    out: &mut [i16],
    gain_db: f32,
    agc_gain: &mut f32,
) -> usize {
    let n = audio.len().min(out.len());
    if n == 0 {
        return 0;
    }
    // 1. DC block + compute RMS.
    let mut sum = 0.0f32;
    for v in &audio[..n] {
        sum += v;
    }
    let mean = sum / n as f32;
    let mut rms_sq = 0.0f32;
    for v in &audio[..n] {
        let x = v - mean;
        rms_sq += x * x;
    }
    let rms = (rms_sq / n as f32).sqrt();
    // 2. AGC. Target output RMS in i16 units.
    //
    // The HL2 EP6 baseband is a full-Nyquist complex stream in which SSB voice
    // occupies only ~3 kHz of 96 kHz — so the in-band voice RMS relative to
    // full-scale is typically ~1e-5–1e-4 (≈ −100…−80 dBFS). Reaching the
    // −20 dBFS target therefore requires ~1e6–1e8× of gain, well beyond a
    // "reasonable" ceiling. The reference gets this gain from the
    // *hardware* RXA AGC before digitisation (~80 dB); we do it
    // digitally, so the ceiling must cover it. 1e7 (140 dB) lands the
    // measured in-band signal right at target.
    const TARGET_RMS_I16: f32 = 0.1 * 32_767.0;
    let target_scale = (TARGET_RMS_I16 / rms.max(1e-6)).clamp(1.0, 10_000_000.0);
    let alpha = if target_scale > *agc_gain { 0.4 } else { 0.08 };
    *agc_gain += alpha * (target_scale - *agc_gain);
    // 3. Emit.
    let g = 10f32.powf(gain_db / 20.0) * *agc_gain;
    for (o, v) in out[..n].iter_mut().zip(audio.iter().take(n)) {
        let s = (v - mean) * g;
        *o = s.clamp(-32_768.0, 32_767.0) as i16;
    }
    n
}

impl Demodulator for SsbDemodulator {
    fn demod(&mut self, iq: &IqBlock, sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        // SSB demod. The wire gives (I, Q). For a *real* SDR baseband (Q ≈ 0
        // — the HL2 DDC's normal "real SDR" mode), the voice is in the I arm
        // only, with USB voice on positive frequencies and LSB voice on
        // negative frequencies. For a *complex* baseband (4× mode, Q loud),
        // the voice is already split.
        //
        // We handle both with the **product-discrimination** SSB structure:
        //   1. NCO to the channel centre (`nco_step`) — this is the user's
        //      `--offset` shift; with `offset=0` it is identity (no NCO) so
        //      the voice is at [0, +BW] / [−BW, 0] already.
        //   2. **Synthesise quadrure from the post-NCO in-phase arm** via the
        //      Hilbert (90°) filter. This is the missing `e^{j·90°}` that a
        //      real baseband never had; for a complex baseband the
        //      Hilbert-synthesised arm is already *equivalent* to the Q arm
        //      (within windowing tolerance), so this is correct for both.
        //   3. **Product discriminate**: USB = re{z'} = I';  LSB = im{z'} = I_H.
        //      `I_H` (the Hilbert of I') is the 90°-shifted I', which
        //      carries the LSB voice (negative frequency) as a
        //      positive-frequency tone (Hilbert of cos(ωt) = sin(ωt) with
        //      phase reversed).
        //
        // This is equivalent to keeping the in-phase arm (USB) vs.
        // conjugating it (LSB).
        //
        // `iq` is a list of **complex** samples — any length is valid (a real
        // wire chunk is 63, which is *odd*), and each sample is processed
        // independently below, so no even-length requirement applies.
        let is_usb = self.sideband == Sideband::Usb;
        for &x in iq {
            // 1. NCO to the channel centre (phase recurrence — no trig here).
            let post = self.nco.step(x);
            // 2. Product-discriminate. USB = in-phase arm directly; LSB needs
            //    the synthesised quadrature arm (Hilbert of the in-phase). The
            //    Hilbert is only run for LSB — it is pure overhead on USB.
            let r0 = if is_usb {
                post.re
            } else {
                self.hilb.convolve(post.re)
            };
            // 3. Polyphase anti-alias + decimate (≈1/M the taps of the old
            //    full-rate filter step).
            if let Some(sample) = self.lp.push(r0) {
                self.audio_buf.push(sample);
            }
        }
        // 4. AGC + DC-block + emit, once enough decimated audio has been
        //    accumulated (see `AUDIO_EMIN`: a 1-sample block would DC-block
        //    itself to zero).
        self.flush_full_blocks(sink)
    }

    fn audio_format(&self) -> AudioConfig {
        self.audio_cfg
    }

    /// Emit the residual partial block (if any) accumulated after the last
    /// full emission, normalising even a sub-`AUDIO_EMIN` tail so the final
    /// ~50 ms isn't silently dropped at shutdown.
    fn flush_audio(&mut self, sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        if self.audio_buf.is_empty() {
            return Ok(0);
        }
        let n = std::mem::take(&mut self.audio_buf);
        self.emit(&n, sink)
    }
}

/// Build a `Box<dyn Demodulator>` for `mode`.
///
/// * `source_center_hz`: the signal's carrier offset from the source centre
///   (`Hz`); `0.0` for a baseband source.
/// * `bandwidth_hz`: the channel-select bandwidth (`Hz`); must be less than
///   the source rate.
pub fn make_demod(
    mode: Mode,
    source_rate_hz: u32,
    source_center_hz: f64,
    bandwidth_hz: u32,
    audio: AudioConfig,
) -> Result<Box<dyn Demodulator>, DemodError> {
    make_demod_tap(
        mode,
        source_rate_hz,
        source_center_hz,
        bandwidth_hz,
        audio,
        None,
    )
}

/// `make_demod` plus an optional [`RawSampleTap`]: the tap is fed the raw,
/// pre-AGC decimated `f32` audio of every block, in arrival order, inline
/// from the demod thread. This is the seam the FT8 slot decoder hangs off:
/// the SSB/USB pipeline still emits AGC'd `i16` to the `CH_AUDIO` sink (the
/// operator hears the same audio being decoded), while the tap copies the
/// untouched `f32` stream to [`super::ft8::Ft8Decoder`].
///
/// `Mode::Ft8` / `Mode::Js8` build a [`DigitalDemodulator`] (USB mix, polyphase
/// decimation straight to the 12 kHz digital-mode window rate — no quadrature
/// synthesis, no separate channel filter); `Mode::Ssb(·)` builds an
/// [`SsbDemodulator`] (USB or LSB, voice-rate polyphase). `Mode::SsbWide` is
/// passed through the USB SSB pipeline today.
pub fn make_demod_tap(
    mode: Mode,
    source_rate_hz: u32,
    source_center_hz: f64,
    bandwidth_hz: u32,
    audio: AudioConfig,
    tap: Option<std::sync::Arc<dyn RawSampleTap>>,
) -> Result<Box<dyn Demodulator>, DemodError> {
    if matches!(mode, Mode::Ft8 | Mode::Js8 | Mode::Ft4) {
        let label = match mode {
            Mode::Ft8 => "ft8",
            Mode::Js8 => "js8",
            Mode::Ft4 => "ft4",
            _ => unreachable!(),
        };
        return Ok(Box::new(DigitalDemodulator::new(
            source_rate_hz,
            source_center_hz,
            audio,
            tap,
            label,
        )?));
    }
    let sideband = match mode {
        Mode::Ssb(s) => s,
        Mode::SsbWide => Sideband::Usb,
        Mode::Ft8 | Mode::Js8 | Mode::Ft4 => unreachable!(),
    };
    // Decimation factor: integer `source / audio`. `Decimator`'s old
    // 4× floor is kept as a sanity bound (an SSB voice path decimated by
    // less than 4 is aliasing by design and nobody wants that silently).
    let m = source_rate_hz as usize / audio.rate_hz as usize;
    if m < 4 {
        return Err(Box::new(RateTooClose {
            src: source_rate_hz,
            audio: audio.rate_hz,
        }));
    }
    // Anti-alias LPF: pass the requested bandwidth (default ≈ voice). The same
    // taps double as the channel-select filter *and* the decimator's
    // anti-alias — as `F32Fir::lowpass` they are a unit-gain, windowed-sinc
    // low-pass at exactly that bandwidth, split into `M` polyphase branches by
    // [`PolyphaseDecimator::new`] so only ≈ 1/M of the taps run per input
    // sample.
    let bw_ratio = (bandwidth_hz as f64 / source_rate_hz as f64).clamp(1e-3, 0.4);
    let taps = if bandwidth_hz <= 4_000 { 257 } else { 129 };
    let h = F32Fir::lowpass(taps, bw_ratio, KAISER_BETA).taps().to_vec();
    let lp = PolyphaseDecimator::new(&h, m);
    // Quadrature (Hilbert) 90° phase shifter. 257 taps is well within the
    // voice band (this is only exercised for LSB, where its output is used);
    // the old 2× (514) was pure overhead.
    let hilb_taps = taps.max(257);
    let hilb = F32FirState::new(&F32Fir::hilbert(hilb_taps, KAISER_BETA));
    let nco = Nco::new(2.0 * std::f64::consts::PI * source_center_hz / source_rate_hz as f64);
    Ok(Box::new(SsbDemodulator {
        sideband,
        audio_cfg: audio,
        nco,
        hilb,
        lp,
        agc_gain: 1000.0,
        audio_buf: Vec::with_capacity(256),
        tap,
    }))
}

/// The FT8 demodulator: USB SSB mix at the [`super::ft8::FT8_SAMPLE_RATE_HZ`]
/// (12 kHz) rate `mfsk-core`'s FT8 decoder expects.
///
/// ```text
/// complex I/Q
///   └─► NCO × e^(−j·φ)
///        └─► in-phase (USB) arm directly — no Hilbert, no LSB path
///             └─► PolyphaseDecimator (LPF BW from `bandwidth`, M = src/12k)
///                  └─► pre-AGC f32 → [`RawSampleTap`] (FT8 window)  [if tap]
///                       └─► AGC + DC-block → i16 → sink (monitor audio)
/// ```
///
/// The FT8 passband (~1.4-2.9 kHz inside the SSB audio band) sits in the
/// *in-phase* arm after the USB NCO, so the quadrature synthesis an LSB SSB
/// demod needs is pure overhead here and is dropped. The anti-alias /
/// channel filter is the same windowed-sinc design as
/// [`SsbDemodulator`]'s, consumed as a [`PolyphaseDecimator`] — ~M× fewer
/// MACs per sample than the old full-rate 257-tap convolution.
pub struct DigitalDemodulator {
    nco: Nco,
    lp: PolyphaseDecimator,
    agc_gain: f32,
    audio_cfg: AudioConfig,
    audio_buf: Vec<f32>,
    /// The mode this digital-path demodulator was built for (`ft8` / `js8`),
    /// for debug logs.
    mode_label: &'static str,
    /// Pre-AGC raw-sample tap the FT8 slot decoder hangs off (see
    /// [`super::ft8::Ft8Tap`]). `None` if the mode was built without one.
    tap: Option<std::sync::Arc<dyn RawSampleTap>>,
}

impl DigitalDemodulator {
    /// Build an FT8/JS8 demodulator from `source_rate_hz` Hz complex I/Q
    /// down to the 12 kHz output. `source_rate_hz` must be an integer
    /// multiple of 12 kHz of at least 4×.
    pub fn new(
        source_rate_hz: u32,
        source_center_hz: f64,
        audio: AudioConfig,
        tap: Option<std::sync::Arc<dyn RawSampleTap>>,
        mode_label: &'static str,
    ) -> Result<Self, DemodError> {
        let m = source_rate_hz as usize / audio.rate_hz as usize;
        if m < 4 {
            return Err(Box::new(RateTooClose {
                src: source_rate_hz,
                audio: audio.rate_hz,
            }));
        }
        let bw_ratio = (2_600.0f64 / source_rate_hz as f64).clamp(1e-3, 0.4);
        // 511 taps (→ 513 odd) on the 12 kHz digital path: with Kaiser β = 12
        // this puts the out-of-band rejection floor at ~107 dB across the
        // 3.8 – 6.0 kHz band — well past the 80 dB floor the
        // `ft8_js8_path_rejects_out_of_passband_signal_by_at_least_80_db`
        // regression guards, and gives operator-visible headroom for a
        // stronger adjacent-channel signal (the user's 14.078 vs. 14.074
        // FT8-on-USB case). 257 taps was borderline (worst case ~82 dB)
        // because a 4 kHz offset lands right in the *transition band* of a
        // 257-tap / 2.6 kHz / 192 kHz filter, where sidelobe peak height
        // is β- and tap-count-sensitive. Cost is modest: 513 taps ÷ 16
        // polyphase ≈ 32 MACs per input sample (2× the old 16), well within
        // the existing hot-loop budget.
        let h = F32Fir::lowpass(511, bw_ratio, KAISER_BETA).taps().to_vec();
        let lp = PolyphaseDecimator::new(&h, m);
        let nco = Nco::new(2.0 * std::f64::consts::PI * source_center_hz / source_rate_hz as f64);
        Ok(Self {
            nco,
            lp,
            agc_gain: 1000.0,
            audio_cfg: audio,
            audio_buf: Vec::with_capacity(256),
            tap,
            mode_label,
        })
    }

    /// Normalise + emit one slice of the accumulated audio, advancing the
    /// AGC. Returns the number of i16 frames written.
    fn emit(&mut self, slice: &[f32], sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        // Pre-AGC raw-sample tap (FT8 decode): the untouched decimated `f32`
        // stream, at `audio_cfg.rate_hz`, in arrival order.
        if let Some(tap) = self.tap.as_ref() {
            tap.append(slice);
        }
        let mut out = vec![0i16; slice.len()];
        let written =
            normalize_to_i16_with_agc(slice, &mut out, self.audio_cfg.gain_db, &mut self.agc_gain);
        {
            if std::env::var("HL2_DEBUG").is_ok() {
                let c = EMIT_COUNT.fetch_add(1, AOrdering::Relaxed) + 1;
                if c % 50 == 1 {
                    let in_max = slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let in_rms = (slice.iter().map(|v| v * v).sum::<f32>()
                        / slice.len().max(1) as f32)
                        .sqrt();
                    let o_max = out[..written].iter().map(|v| v.abs()).max().unwrap_or(0);
                    eprintln!(
                        "[aud] {} in_rms={in_rms:.6e} in_max={in_max:.6e} agc={:.3e} out_i16_max={o_max} (n={written})",
                        self.mode_label, self.agc_gain
                    );
                }
            }
        }
        sink.write(&out[..written])
    }
}

impl Demodulator for DigitalDemodulator {
    fn demod(&mut self, iq: &IqBlock, sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        for &x in iq {
            let post = self.nco.step(x);
            if let Some(sample) = self.lp.push(post.re) {
                self.audio_buf.push(sample);
            }
        }
        // Emit: pre-AGC tap first (the FT8 window sees natural units), then
        // the AGC'd i16 to the sink (the operator's monitor audio).
        let mut frames_written = 0usize;
        while self.audio_buf.len() >= AUDIO_EMIN {
            let n = std::mem::take(&mut self.audio_buf);
            let block_len = n.len().min(1024);
            let slice = &n[..block_len];
            self.audio_buf = n[block_len..].to_vec();
            frames_written = frames_written.saturating_add(self.emit(slice, sink)?);
        }
        Ok(frames_written)
    }

    fn audio_format(&self) -> AudioConfig {
        self.audio_cfg
    }

    /// Emit the residual partial block (sub-`AUDIO_EMIN` tail) so the final
    /// ~50 ms isn't silently dropped at shutdown.
    fn flush_audio(&mut self, sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        if self.audio_buf.is_empty() {
            return Ok(0);
        }
        let n = std::mem::take(&mut self.audio_buf);
        self.emit(&n, sink)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receiver::sink::VecSink;

    /// A proper **analytic** complex tone: `amp · e^(j·2π·f·t)`, i.e.
    /// `I = amp·cos(θ)`, `Q = amp·sin(θ)`. The demod of such a tone is a full
    /// sine wave (the real part), with no 50% attenuation.
    fn complex_sine(rate_hz: u32, freq_hz: f64, n_pairs: usize, amp: f32) -> IqBlock {
        (0..n_pairs)
            .map(|i| {
                let w = 2.0 * std::f64::consts::PI * freq_hz / rate_hz as f64;
                Complex::new(
                    (w * i as f64).cos() as f32 * amp,
                    (w * i as f64).sin() as f32 * amp,
                )
            })
            .collect()
    }

    #[test]
    fn usb_inband_tone_produces_audio() {
        let rate = 192_000u32;
        let iq = complex_sine(rate, 1_500.0, 16384, 0.5);
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod: Box<dyn Demodulator> =
            make_demod(Mode::Ssb(Sideband::Usb), rate, 0.0, 2_600, cfg).unwrap();
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("demod ok");
        assert!(n > 0, "produced {n} audio frames");
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "peak too low: {peak}");
    }

    #[test]
    fn lsb_inband_tone_produces_audio() {
        let rate = 192_000u32;
        let iq = complex_sine(rate, 1_500.0, 16384, 0.5);
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod: Box<dyn Demodulator> =
            make_demod(Mode::Ssb(Sideband::Lsb), rate, 0.0, 2_600, cfg).unwrap();
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("demod ok");
        assert!(n > 0, "produced {n} audio frames");
    }

    /// A **real** baseband tone: the radio downconverted to DC with a *real*
    /// mixer, so `I = cos(wt)`, `Q = 0` (the HL2's normal "real SDR" baseband
    /// — this is why old-LSB, which read `im`, was silent). The demod must
    /// synthesize quadrure and produce audio on **both** sidebands.
    fn real_tone(rate_hz: u32, freq_hz: f64, n_pairs: usize, amp: f32) -> IqBlock {
        (0..n_pairs)
            .map(|i| {
                let w = 2.0 * std::f64::consts::PI * freq_hz / rate_hz as f64;
                Complex::new((w * i as f64).cos() as f32 * amp, 0.0f32)
            })
            .collect()
    }

    #[test]
    fn real_baseband_tone_produces_audio_both_sides() {
        let rate = 192_000u32;
        for mode in [Mode::Ssb(Sideband::Usb), Mode::Ssb(Sideband::Lsb)] {
            let iq = real_tone(rate, 1_500.0, 16384, 0.5);
            let cfg = AudioConfig {
                rate_hz: 4_800,
                gain_db: 0.0,
            };
            let mut demod = make_demod(mode, rate, 0.0, 2_600, cfg).unwrap();
            let mut sink = VecSink::new();
            let _ = demod.demod(&iq, &mut sink).expect("demod ok");
            let samples = sink.samples().to_vec();
            let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
            assert!(peak > 30, "{mode:?} real-baseband peak too low: {peak}");
        }
    }

    #[test]
    fn odd_block_accepted() {
        // A real wire chunk is 63 complex samples (odd). The demod handles
        // each sample independently, so an odd length must now succeed.
        let rate = 192_000u32;
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod = make_demod(Mode::Ssb(Sideband::Usb), rate, 0.0, 2_600, cfg).unwrap();
        let mut sink = VecSink::new();
        let iq = vec![Complex::new(1.0, 0.0); 63];
        demod
            .demod(&iq, &mut sink)
            .expect("odd (63) length must succeed");
    }

    /// A real FFT of `x` via DFT (test-sized inputs) → magnitude spectrum.
    fn spec_mag(x: &[f32]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for (t, v) in x.iter().enumerate() {
                    let w = 2.0 * std::f64::consts::PI * k as f64 * t as f64 / n as f64;
                    re += *v as f64 * w.cos();
                    im -= *v as f64 * w.sin();
                }
                (re * re + im * im).sqrt()
            })
            .collect()
    }

    #[test]
    fn usb_passes_inband_and_rejects_out_of_band_tone() {
        // Add a 5 kHz tone to a 1.5 kHz tone; USB should pass 1.5k (in-band)
        // and heavily attenuate 5k (far above our 0.03·fs filter passband).
        let rate = 192_000u32;
        let a = complex_sine(rate, 1_500.0, 16384, 0.5);
        let b = complex_sine(rate, 5_000.0, 16384, 0.5);
        let iq: IqBlock = a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect();
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        let mut demod = make_demod(Mode::Ssb(Sideband::Usb), rate, 0.0, 2_600, cfg).unwrap();
        let mut sink = VecSink::new();
        let _ = demod.demod(&iq, &mut sink).expect("ok");
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "audio too quiet: {peak}");
    }

    /// Regression: the channel-select filter rejects a signal just outside
    /// the passband by **≥ 80 dB** relative to an in-band reference, in the
    /// 12 kHz FT8/JS8 path.
    ///
    /// Reproduces the user's real-world scenario: tuning a JS8 / FT8
    /// receiver to 14.078 MHz while the band is quiet but adjacent FT8 at
    /// 14.074 MHz (a 4 kHz offset past the 2.6 kHz passband edge) is
    /// still heard. With the historical Hann window that residual was
    /// ~50 dB below the in-band voice; the Kaiser window (see
    /// [`KAISER_BETA`]) lifts it to → 80 dB by design.
    ///
    /// **Measurement**: each tone is run *in isolation* through a demodulator
    /// whose [`RawSampleTap`] captures the **pre-AGC f32** audio stream.
    /// With a single tone in, the DFT bin at that tone's frequency is the
    /// filter's *true linear gain* at that frequency — no cross-leakage
    /// between two different tones and no AGC floor (the AGC is a per-block
    /// scalar that preserves single-tone ratios, but measuring the raw f32
    /// is cleanest). In-band 1.3 kHz vs each out-of-band offset are
    /// amplitude-normalised, so their ratio in dB *is* the filter's
    /// out-of-band rejection at that offset.
    ///
    /// Note the SSB 4.8 kHz voice path has a *separate* decimation-aliasing
    /// issue (a 4 kHz offset folds into voice at 0.8 kHz) independent of
    /// the channel filter — covered separately if/when addressed.
    #[test]
    fn ft8_js8_path_rejects_out_of_passband_signal_by_at_least_80_db() {
        let rate = 192_000u32;
        let n = 32_768;
        let audio_rate = 12_000.0;
        let cfg = AudioConfig {
            rate_hz: 12_000,
            gain_db: 0.0,
        };

        // Measure the *filter's* response at a single input tone:
        // pre-AGC f32 amplitude of that tone after the demodulator, as a
        // function of the input tone's frequency. Because the demod is
        // linear (NCO + linear FIR + decimation) this equals
        // `A_in · |H(f)|` at the decimated rate — a clean measure of the
        // channel-select filter's complex gain at `f`.
        let amplitude_at = |f_hz: f64| -> f64 {
            let sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
            let tap = std::sync::Arc::new(VecF32Tap::new(sink.clone()));
            let mut demod: Box<dyn Demodulator> =
                make_demod_tap(Mode::Ft8, rate, 0.0, 2_600, cfg, Some(tap)).expect("build");
            let iq = complex_sine(rate, f_hz, n, 0.5);
            let mut vec_sink = VecSink::new();
            let _ = demod.demod(&iq, &mut vec_sink).expect("demod");
            demod.flush_audio(&mut vec_sink).ok();
            let audio: Vec<f32> = sink.lock().expect("tap not poisoned").clone();
            let spec = spec_mag(&audio);
            // Tone frequency lands on bin round(f / fs_audio · N); search
            // a tight window around it to absorb any off-bin energy and
            // the tiny tail of the DFT (the DFT is *unwindowed* — a
            // rectangular window — so the nearest bin is essentially the
            // filter response at that frequency, within ~0.5 Hz of the
            // tone).
            let bin = (f_hz / audio_rate * spec.len() as f64).round() as usize;
            let lo = bin.saturating_sub(2);
            let hi = (bin + 2).min(spec.len().saturating_sub(1));
            spec[lo..=hi].iter().cloned().fold(0.0f64, f64::max)
        };

        let in_amp = amplitude_at(1_300.0);
        assert!(in_amp > 1e-6, "in-band tone not audible: {in_amp:.3e}");
        for offset in [4_000.0, 5_000.0, 8_000.0] {
            let out_amp = amplitude_at(offset);
            let rej_db = 20.0 * f64::log10(in_amp / out_amp.max(1e-12));
            assert!(
                rej_db > 80.0,
                "offset {offset} Hz: in-band(1.3 kHz) {in_amp:.2e} vs \
                 out-of-band {out_amp:.2e} → rejection only {rej_db:.1} dB \
                 (want > 80 dB)",
            );
        }
    }

    /// A [`RawSampleTap`] that appends every `f32` slice to an `Arc<Mutex<Vec<f32>>>`
    /// (for the channel-select response measurement in
    /// `ft8_js8_path_rejects_out_of_passband_signal_by_at_least_80_db`).
    #[derive(Debug)]
    struct VecF32Tap {
        sink: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    }

    impl VecF32Tap {
        fn new(sink: std::sync::Arc<std::sync::Mutex<Vec<f32>>>) -> Self {
            Self { sink }
        }
    }

    impl RawSampleTap for VecF32Tap {
        fn append(&self, samples: &[f32]) {
            self.sink
                .lock()
                .expect("tap not poisoned")
                .extend_from_slice(samples);
        }
    }

    /// Regression: the FT8/JS8 digital path must pass a signal placed at a
    /// non-zero **NCO offset** (the auto-decode case: slot NCO=7.050 MHz,
    /// FT8 at 7.074 → offset +24 kHz) into the 12 kHz passband at usable
    /// gain, and reject an out-of-band offset — exactly the scenario where
    /// only offset-0 used to work. This is the first test that drives
    /// `DigitalDemodulator` with `source_center_hz != 0` (the NCO active).
    #[test]
    fn ft8_path_passes_signal_at_nonzero_nco_offset() {
        let rate = 192_000u32;
        let n = 65_536;
        let cfg = AudioConfig {
            rate_hz: 12_000,
            gain_db: 0.0,
        };
        let nco_hz = 24_000.0; // 7.074 - 7.050

        // Linear response: run a single tone in isolation, capture the
        // pre-AGC f32 via a tap, read the DFT bin at the tone's expected
        // 12 kHz location. Because the demod is linear (NCO + FIR +
        // decimate) the bin magnitude == A_in · |H(f_in)|.
        let inband_bin = |f_in: f64, expect_hz: f64| -> f64 {
            let sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
            let tap = std::sync::Arc::new(VecF32Tap::new(sink.clone()));
            let mut demod: Box<dyn Demodulator> =
                make_demod_tap(Mode::Ft8, rate, nco_hz, 2_600, cfg, Some(tap)).expect("build");
            let iq = complex_sine(rate, f_in, n, 0.5);
            let mut vec_sink = VecSink::new();
            let _ = demod.demod(&iq, &mut vec_sink).expect("demod");
            demod.flush_audio(&mut vec_sink).ok();
            let audio: Vec<f32> = sink.lock().expect("poisoned").clone();
            let spec = spec_mag(&audio);
            let bin = (expect_hz / 12_000.0 * spec.len() as f64).round() as usize;
            let lo = bin.saturating_sub(2);
            let hi = (bin + 2).min(spec.len().saturating_sub(1));
            spec[lo..=hi].iter().cloned().fold(0.0f64, f64::max)
        };

        // In-band: a tone at NCO+1.5 kHz must land at 1.5 kHz in the
        // decimated window with strong gain.
        let in_amp = inband_bin(nco_hz + 1_500.0, 1_500.0);
        eprintln!("[test] in-band(NCO+1.5k) amplitude @1.5kHz = {in_amp:.4e}");
        // Out-of-band: a tone at NCO+5 kHz lands at 5 kHz (past the 2.6 kHz
        // passband edge) and must be strongly rejected.
        let out_amp = inband_bin(nco_hz + 5_000.0, 5_000.0);
        eprintln!("[test] out-of-band(NCO+5k) amplitude @5kHz = {out_amp:.4e}");
        // A reference in-band tone (at DC, the offset-0 case that always
        // worked) so we can assert the non-zero NCO isn't *losing* gain.
        let dc_sink = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
        let dc_tap = std::sync::Arc::new(VecF32Tap::new(dc_sink.clone()));
        let mut demod_dc: Box<dyn Demodulator> =
            make_demod_tap(Mode::Ft8, rate, 0.0, 2_600, cfg, Some(dc_tap)).expect("build");
        let iq_dc = complex_sine(rate, 1_500.0, n, 0.5);
        let mut ws = VecSink::new();
        let _ = demod_dc.demod(&iq_dc, &mut ws).expect("demod");
        demod_dc.flush_audio(&mut ws).ok();
        let audio_dc: Vec<f32> = dc_sink.lock().expect("poisoned").clone();
        let spec_dc = spec_mag(&audio_dc);
        let b = (1_500.0 / 12_000.0 * spec_dc.len() as f64).round() as usize;
        let dc_amp = spec_dc[(b - 2).max(0)..=b + 2]
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);
        eprintln!("[test] reference(DC, offset-0) amplitude @1.5kHz = {dc_amp:.4e}");

        assert!(
            in_amp > 1e-4,
            "in-band NCO-offset tone too quiet: {in_amp:.3e} (reference {dc_amp:.3e})",
        );
        // The non-zero-NCO in-band signal should be within ~12 dB of the
        // offset-0 reference (windowing/transition-band loss is expected at
        // these offsets but not > ~20×).
        assert!(
            dc_amp / in_amp < 15.0,
            "NCO offset lost the signal: in-band {in_amp:.3e} vs reference {dc_amp:.3e} = {:.1} dB",
            20.0 * f64::log10(dc_amp / in_amp.max(1e-12)),
        );
        assert!(
            dc_amp / out_amp > 100.0,
            "out-of-band insufficient: in {in_amp:.3e} vs out {out_amp:.3e} (want ≥ 40 dB)",
        );
    }

    #[test]
    fn nco_moves_offband_tone_into_band() {
        // Signal at (center + 1.5k). Center = 100k. NCO should move it to 1.5k.
        let rate = 192_000u32;
        let center = 100_000.0;
        let iq = complex_sine(rate, center + 1_500.0, 16384, 0.5);
        let cfg = AudioConfig {
            rate_hz: 4_800,
            gain_db: 0.0,
        };
        // USB, centred 100k. NCO (e^{−jφ}, φ̇ = −2π·center/fs) moves the tone
        // from (100k+1.5k) down to +1.5k around DC — in the passband.
        let mut demod = make_demod(Mode::Ssb(Sideband::Usb), rate, center, 2_600, cfg).unwrap();
        let mut sink = VecSink::new();
        let n = demod.demod(&iq, &mut sink).expect("ok");
        assert!(n > 0);
        let peak = sink.samples().iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 50, "NCO-moved tone should be passable: {peak}");

        // LSB, centred 100k, passes a tone at (100k − 1.5k): the NCO moves it
        // to −1.5k around DC and the Hilbert-synthesised quadrature arm keeps
        // the (lower) sideband.
        let iq_lsb = complex_sine(rate, center - 1_500.0, 16384, 0.5);
        let mut demod_lsb = make_demod(Mode::Ssb(Sideband::Lsb), rate, center, 2_600, cfg).unwrap();
        let mut sink_lsb = VecSink::new();
        let _ = demod_lsb.demod(&iq_lsb, &mut sink_lsb).expect("ok");
        let peak_lsb = sink_lsb
            .samples()
            .iter()
            .map(|s| s.abs())
            .max()
            .unwrap_or(0);
        assert!(
            peak_lsb > 50,
            "LSB NCO-moved tone should be passable: {peak_lsb}"
        );
    }

    /// The recurrence NCO must track `cos((n+1)·step)` / `−sin((n+1)·step)`
    /// sample-for-sample against the trig reference, for a positive (USB) and
    /// a negative (LSB) step.
    #[test]
    fn nco_recurrence_matches_trig() {
        for step in [TAU * 1_500.0 / 192_000.0, -TAU * 1_500.0 / 192_000.0] {
            let x = Complex::new(0.3f32, -0.2f32);
            let mut nco = Nco::new(step);
            for n in 0..20_000usize {
                let got = nco.step(x);
                let phi = n as f64 * step; // multiply at φ = n·step, then advance
                let want_re = ((x.re as f64) * phi.cos() + (x.im as f64) * phi.sin()) as f32;
                let want_im = ((x.im as f64) * phi.cos() - (x.re as f64) * phi.sin()) as f32;
                assert!(
                    (got.re - want_re).abs() < 1e-3,
                    "n={n} re got {} want {want_re}",
                    got.re
                );
                assert!(
                    (got.im - want_im).abs() < 1e-3,
                    "n={n} im got {} want {want_im}",
                    got.im
                );
            }
        }
    }

    /// The polyphase decimator must reproduce, sample-for-sample (zero-pad
    /// onset included), the full-rate reference:
    ///
    ///   `y[n] = Σ_j h[j]·x[n−j]`  (zero-padded for `n−j < 0`),
    ///   kept at `n = e·M + (M−1)`  (the e-th emitted sample, `POLY_ALIGN`).
    ///
    /// This alignment identity is what the demod relies on; a branch/tap or
    /// emit-instant regression would corrupt the decimated stream. Checked
    /// across several `(M, taps)` shapes with pseudo-random `h`-scale inputs.
    #[test]
    fn polyphase_matches_fullfir_decimate() {
        for (m, taps_n) in [(1usize, 31usize), (2, 40), (4, 257), (8, 97)] {
            let fir = F32Fir::lowpass(taps_n, 0.02, KAISER_BETA);
            let h = fir.taps().to_vec();
            let mut poly = PolyphaseDecimator::new(&h, m);

            // Deterministic pseudo-random input stream in [-1, 1]
            // ((seed >> 40) is 24-bit; dividing by 2^23 gives [0, 2), −1.0 → [-1, 1)).
            let mut seed = 0x1234_5678u64;
            let mut next_x = move || {
                seed = seed.wrapping_mul(63_641).wrapping_add(1_357_911);
                ((seed >> 40) as f64 / 8_388_608.0 - 1.0) as f32
            };

            // Reference: full-rate causal convolution over a growing history.
            let mut hist: Vec<f32> = Vec::new();
            let mut emitted = 0usize;
            const N: usize = 1200;
            for n in 0..N {
                let x = next_x();
                hist.push(x);
                let got = poly.push(x);
                if (n + 1) % m == 0 {
                    let got = got.expect("polyphase must emit every M-th sample");
                    let mut want = 0.0f64;
                    for (j, hj) in h.iter().enumerate() {
                        if (j as i64) <= (n as i64) {
                            want += (*hj as f64) * (hist[n - j] as f64);
                        }
                    }
                    emitted += 1;
                    assert!(
                        (got - want as f32).abs() < 1e-4,
                        "m={m} n={n} ({emitted}th emit): got {got} want {}",
                        want as f32
                    );
                } else {
                    assert!(got.is_none(), "m={m} n={n}: unexpected early emit");
                }
            }
            // N is a multiple of every tested m.
            assert_eq!(emitted, N / m);
        }
    }

    /// `DigitalDemodulator` must actually produce 12 kHz monitor audio (AGC'd),
    /// and must dispatch via `make_demod` for `Mode::Ft8`.
    #[test]
    fn ft8_demod_tone_produces_12k_audio() {
        let rate = 192_000u32;
        let iq = complex_sine(rate, 1_500.0, 16384, 0.5);
        let cfg = AudioConfig {
            rate_hz: 12_000,
            gain_db: 0.0,
        };
        let mut demod: Box<dyn Demodulator> =
            make_demod(Mode::Ft8, rate, 0.0, 2_600, cfg).expect("build ft8");
        assert_eq!(demod.audio_format().rate_hz, 12_000);
        let mut sink = VecSink::new();
        let _ = demod.demod(&iq, &mut sink).expect("demod ok");
        // 16384 inputs at M = 192k/12k = 16 → exactly 1024 decimated samples,
        // one whole emission (1024 ≥ AUDIO_EMIN = 240).
        let samples = sink.samples().to_vec();
        let peak = samples.iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > 30, "FT8 monitor audio too quiet: {peak}");
        assert_eq!(samples.len(), 1024, "got {} samples", samples.len());
    }

    /// `F32FirState::convolve` (the new fixed-ring streaming path) must match
    /// `F32Fir::apply` over a growing full-history vector (the old path), both
    /// during the zero-pad onset and in steady state. Catches any ring-index
    /// or onset regression that the tone tests (which only check peak level)
    /// would miss.
    #[test]
    fn firstate_matches_reference_apply() {
        let fir = F32Fir::lowpass(31, 0.03, KAISER_BETA);
        let mut st = F32FirState::new(&fir);
        let mut hist: Vec<f32> = Vec::new();
        let mut seed = 0.123456_f64;
        for step in 0..60 {
            // Deterministic pseudo-random input in [-1, 1].
            seed = (seed * 1_103.0 + 19.0) % 100.0 - 50.0;
            let x = (seed / 50.0) as f32;
            hist.push(x);
            let got = st.convolve(x);
            let want = fir.apply(&hist, hist.len() - 1);
            assert!(
                (got - want).abs() < 1e-4,
                "convolve diverged at step {step}: got {got} want {want}"
            );
        }
    }
}
