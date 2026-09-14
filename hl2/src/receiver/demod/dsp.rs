//! Hand-rolled DSP primitives shared by every virtual-receiver mode: a
//! complex phase-recurrence [`Nco`], a Kaiser-windowed windowed-sinc
//! low-pass / Hilbert FIR ([`F32Fir`] + its streaming [`F32FirState`]), and
//! the [`PolyphaseDecimator`] that turns the full-rate anti-alias FIR into
//! an ≈`1/M`-taps-per-sample decimator. All three are consumed by the SSB
//! voice path ([`super::ssb`]) and the 12 kHz digital path
//! ([`super::digital`]) — no external DSP crate.

use alloc::vec;
use alloc::vec::Vec;
use num_complex::Complex;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

use core::f64::consts;

/// Kaiser-window shape parameter β for the channel-select and Hilbert
/// windows.
///
/// Unlike Hann/Blackman, the Kaiser window's stopband attenuation is an
/// explicit design parameter — β trades sidelobe decay against transition
/// width. β = 12 gives ≈ 100–110 dB stopband (rule of thumb `A ≈ 10·β − 18`
/// for β > 5), which at our 511-tap / 2.6 kHz @ 192 kHz digital design
/// rejects a 4–6 kHz offset by ≥ 107 dB.
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

/// Choose a Kaiser-windowed-sinc tap count so the **transition width** is
/// ≈ `transition_hz` at sample rate `rate_hz`, for stopband attenuation
/// `A ≈ 10·beta − 18` dB (the standard Kaiser rule). Returns an odd length
/// ≥ 17 (so the impulse response stays centred on a whole sample — linear
/// phase).
///
/// This is the design companion to [`F32Fir`]: give it the anti-alias /
/// quadrature headroom you actually have (the transition you can afford) and
/// it returns the tap count that achieves it at `10·beta − 18` dB. Used by
/// [`super::ssb::SsbCore`] to size the pre-decimation anti-alias, the
/// intermediate-rate Hilbert and the final channel-select with *different*
/// transitions (each serves a different anti-alias / quadrature goal).
pub fn fir_taps(rate_hz: u32, transition_hz: u32, beta: f64) -> usize {
    let a = (10.0 * beta - 18.0).max(20.0);
    let dw = if rate_hz > 0 && transition_hz > 0 {
        2.0 * consts::PI * (transition_hz as f64) / (rate_hz as f64)
    } else {
        f64::INFINITY
    };
    let n = if dw.is_finite() && dw > 0.0 {
        (((a - 7.95) / (14.36 * dw)).ceil() as usize).max(17)
    } else {
        63
    };
    if n % 2 == 0 { n + 1 } else { n }
}

/// A complex oscillator (`× e^(−j·φ)`) advanced by phase recurrence — no
/// `cos`/`sin` in the loop. The state `(c, s)` holds `(cos φ, sin φ)` in
/// `f64` and each step is the 2-D rotation
///
/// ```text
/// z' = z · e^(−j·φ)      (re·c + im·s,  im·c − re·s)
/// c' =  c·cos_step − s·sin_step
/// s' =  s·cos_step + c·sin_step
/// ```
///
/// Sample `n` is mixed at `φ = n·step`, then the state advances. Over the
/// few-thousand samples of a block the drift of `c² + s²` from 1 is ~1e-13
/// in f64 — far below `f32` sample precision, so no renormalisation is
/// needed.
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

/// A symmetric, causal, windowed-sinc **real low-pass** FIR.
///
/// A length-`n` (odd) FIR with an integer centre (`(n−1)/2`), so it has
/// linear phase (group delay `(n−1)/2` samples) and the response
/// `H(0) = 1`, `H(Nyquist) ≈ 0`. Applied as `y[k] = Σ_j h[j]·x[k − j]`.
///
/// After the NCO down-conversion this is the SSB channel-select filter:
/// it passes the voice band `[−BW, +BW]` around DC and rejects out-of-band
/// signals and the image. Windowed with the Kaiser window (see
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
                (consts::PI * ratio * t).sin() / (consts::PI * t)
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
    /// the 90°-rotated version (`H{cos ωt} = sin ωt`, `H{sin ωt} = −cos ωt`),
    /// i.e. `H ≡ −j` on positive-frequency tones.
    ///
    /// In the SSB demod the HL2 EP6 wire already delivers **genuine complex
    /// I/Q** (both arms carry energy, no "real SDR" / Q≈0 special case). The
    /// Hilbert transform is applied to that *complex* Q arm so the two arms
    /// can be combined (`I ± H{Q}`) to pass one sideband and reject the other
    /// (the **phasing method**) — see [`super::ssb`](super::ssb). It is *not*
    /// synthesising a missing quadrature arm; it is the 90° phase element of
    /// the one-sided / two-sided selection. The ideal (windowless) impulse
    /// response is `h[m] = 0` for even `m`, `h[m] = 2/(π·m)` for odd `m`
    /// (about an odd centre tap); we window with the same Kaiser window as
    /// [`F32Fir::lowpass`].
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
                (2.0 / (consts::PI * m)) * window
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
/// The history ring holds the last `T` input samples. `convolve()` computes
/// `Σ_j taps[j]·x[n−j]` (terms with `n−j < 0` zero-padded, identical to
/// [`F32Fir::apply`] with the full history) in a single contiguous
/// backwards pass over the ring + the just-pushed sample — O(T), no
/// per-tap reallocation or `vec` drain.
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
    /// dropped — byte-identical to the `apply(history, len−1)` path.
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
        // `taps[0]` multiplies the just-written `x`; `taps[1..m]` walk the
        // ring backwards from `pos` (wrapping to `t−1`). Split into at most
        // two *straight* loops over contiguous slices — no per-element branch
        // in the ring walk — so on `-C eabihf` the compiler emits contiguous
        // loads and folds each term into an `fma`. This is the hot path the
        // SSB Hilbert (`super::ssb`) and the polyphase channel-select both
        // hit per full-rate sample.
        let need = (m - 1).min(t - 1);
        let taps = &self.taps[1..];
        let h = self.hist.as_ptr();
        let mut acc = self.taps[0] * x;
        // Segment 1 (no wrap): i in 0..seg1, hist index = pos−1−i.
        let seg1 = need.min(pos);
        for i in 0..seg1 {
            acc += taps[i] * unsafe { *h.add(pos - 1 - i) };
        }
        // Segment 2 (wrapped): hist index = base2−i, continuing the ring walk.
        let n2 = need - seg1;
        if n2 > 0 {
            let base2 = t - 1 - seg1 + pos;
            for i in 0..n2 {
                acc += taps[seg1 + i] * unsafe { *h.add(base2 - i) };
            }
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
/// a sub-sampler", at ≈ `1/M ×` the tap-multiplies.
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
/// sample). The `M` branches are the `M` phase components of the decimated
/// convolution; summing their step-`k` outputs recovers `y[k]`.
/// Checked against a naive full-rate reference by
/// `polyphase_matches_fullfir_decimate` below, for arbitrary (random) `h`.
///
/// ## Cost
///
/// Per input sample: one branch convolve of `⌈N/M⌉` taps (vs `N` for the
/// full-rate path) — ~`M×` fewer MACs — and at each group boundary `M−1`
/// additions. The windowed-sinc `h` *is* the anti-alias filter.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 2π, for the NCO phase tests (named so tests avoid the `std::f64::consts::TAU`
    /// path-resolution noise).
    const TAU: f64 = std::f64::consts::TAU;

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

    /// `F32FirState::convolve` must match `F32Fir::apply` over a full
    /// history vector, both during the zero-pad onset and in steady state.
    /// Catches any ring-index or onset regression that the tone tests (which
    /// only check peak level) would miss.
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
