//! JS8Call (Mode A) synchronization: noise baseline + Costas sync search.
//!
//! Faithful port of `baselinejs8` and `syncjs8` from `js8call/JS8.cpp`. The
//! sync pass computes a per-symbol-step power spectrum (`s`), accumulates
//! the average spectrum, replaces the average between the search edges with
//! a fitted noise baseline (dB scale), and then scores the three Costas
//! sync blocks across frequency (`i`) × symbol offset (`j`). The
//! 40th-percentile-normalised scores yield the decode candidates.
//!
//! Numerics are `f32` throughout, matching the reference.

use num_complex::Complex;
use rustfft::FftPlanner;

use crate::receiver::js8::params::{
    ASYNCMIN, BASELINE_MAX, BASELINE_MIN, BASELINE_NODES_R, BASELINE_SAMPLE, Mode, NFOS, NMAXCAND,
};

/// A sync candidate: (freq offset in Hz, start offset in seconds,
/// normalised power).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sync {
    pub freq: f32,
    pub dt: f32,
    pub sync: f32,
}

/// One 15 s sync/decode working set: the per-step power spectrum plus the
/// average spectrum. `s[freq * NHSYM + step]`; `freq` is a bin index in
/// 0..NSPS, `step` a quarter-symbol step in 0..NHSYM.
///
/// `savg` is in **power** units before [`Spectra::fit_baseline`] (which
/// overwrites the fitted band with dB-scaled baseline values) and on the
/// dB scale afterwards — the decode pass (`js8dec`) consumes it in its
/// fitted form (`0.1 * (savg[i] - BASESUB)` then `10^x`).
pub struct Spectra {
    /// Power spectrum, index `freq * nhsym + step` (`nhsym` mode-dependent).
    s: Vec<f32>,
    /// Average spectrum (dB scale over the fitted band after
    /// [`fit_baseline`], else power).
    pub savg: Vec<f32>,
    /// Symbol-spectra count for the mode this `Spectra` was built for
    /// (replaces the old global `NHSYM`).
    pub nhsym: usize,
}

impl Spectra {
    /// Build the spectra for one cycle (`mode.nmax`) of `f32` (12 kHz
    /// passband audio), port of the first loop in `syncjs8`. `window` must
    /// be `mode.nmax` samples; any excess is ignored.
    pub fn new(mode: &Mode, window: &[f32]) -> Self {
        let n = window.len().min(mode.nmax);
        let nsps = mode.nsps;
        let nfft1 = mode.nfft1;
        let nhsym = mode.nhsym;
        let nstep = mode.nstep;
        let mut s = vec![0.0f32; nsps * nhsym];
        let mut savg = vec![0.0f32; nsps];
        let w = nuttal4(nfft1);

        for j in 0..nhsym {
            let ia = j * nstep;
            let ib = ia + nfft1;
            if ib > n {
                break;
            }
            let mut win = vec![0.0f32; nfft1];
            for i in 0..nfft1 {
                win[i] = window[ia + i] * w[i];
            }
            let c = fft(&win);
            for i in 0..nsps {
                let power = c[i].norm_sqr();
                s[i * nhsym + j] = power;
                savg[i] += power;
            }
        }

        Self { s, savg, nhsym }
    }

    /// `s` at (freq bin `i`, step `j`).
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f32 {
        self.s[i * self.nhsym + j]
    }

    /// The noise baseline fit: converts `savg` (over the closed bin range
    /// [BASELINE_MIN, BASELINE_MAX] at `DF` resolution) from power to dB,
    /// fits the degree-`BASELINE_DEGREE` polynomial through the
    /// lower-envelope points at the reference's **Chebyshev node**
    /// positions (Vandermonde normal equations in `f64`, the
    /// reference's `real*8 polyfit`, abscissa centred on the band
    /// midpoint), then overwrites `savg` with the evaluated baseline
    /// + 0.65 dB over `[ia, ib]`. Port of `baselinejs8`.
    ///
    /// The node positions matter, not just their count: the reference's
    /// `BASELINE_NODES` are Chebyshev points at relative positions
    /// `0.5·(1 − cos(π(2i+1)/(2·n)))`, i.e. {0.017, 0.146, 0.371, 0.629,
    /// 0.854, 0.983} for 6 points — clustered at the band edges, so NO
    /// envelope window is centred inside the 1.2-1.8 kHz region where
    /// JS8 signals live. Evenly-spaced segments (our earlier port) put a
    /// window at 40% of the band, directly under a strong in-band signal:
    /// the 10th-percentile there samples the signal's own spectral
    /// skirt, inflating the fitted baseline (`xbase`) exactly at the
    /// decoded frequency and collapsing the reported SNR toward the
    /// −60 dB floor on real (off-bin, real-noise) audio.
    ///
    /// `ia`/`ib` are the caller's search edge bin indices (the domain over
    /// which `savg` will be replaced). `bmin/bmax` use the mode's `df`
    /// bin resolution.
    pub fn fit_baseline(&mut self, df: f32, ia: usize, ib: usize) {
        let bmin = (BASELINE_MIN / df).round() as usize;
        let bmax = (BASELINE_MAX / df).round() as usize;
        let size = bmax - bmin + 1;

        const N_NODES: usize = BASELINE_NODES_R.len();
        let arm = size / (2 * N_NODES);
        let mid = size as f64 / 2.0;
        let mut x = [0.0f64; N_NODES];
        let mut y = [0.0f64; N_NODES];
        for i in 0..N_NODES {
            // `baselinejs8`: `node = size * BASELINE_NODES[i]`, rounded;
            // window `±arm` around it (clamped to the data range).
            let node = (size as f64 * BASELINE_NODES_R[i]).round() as usize;
            let lo = node.saturating_sub(arm);
            let hi = (node + arm).min(size - 1);
            // Convert to dB the reference way (`baselinejs8`):
            // `10·log10(power)`.
            let mut bucket: Vec<f64> = self.savg[bmin + lo..bmin + hi + 1]
                .iter()
                .map(|v| 10.0 * (v.max(f32::EPSILON) as f64).log10())
                .collect();
            bucket.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let rank = (bucket.len() as usize * BASELINE_SAMPLE / 100).min(bucket.len() - 1);
            y[i] = bucket[rank];
            x[i] = node as f64 - mid;
        }

        let coeffs = polyfit_f64(&x, &y);

        // Evaluate at centred abscissas over [ia, ib]; overwrite savg with
        // the baseline (+ 0.65, reference convention).
        let span = (ib.saturating_sub(ia)).max(1) as f64;
        let last = (size - 1) as f64;
        for i in ia..=ib {
            let t = ((i as f64 - ia as f64) * last / span) - mid;
            let v = poly_eval_f64(&coeffs, t);
            self.savg[i] = (v as f32) + 0.65;
        }
    }
}

/// Fit a degree-`c.len()-1` polynomial (ascending powers) through the
/// points `(x[k], y[k])` using the normal equations + Gauss elimination
/// with partial pivoting, in `f64` (the reference's `real*8 polyfit`).
fn polyfit_f64(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    // A = X^T X, b = X^T y, V[k][r] = x[k]^r.
    let mut a = vec![vec![0.0f64; n]; n];
    let mut b = vec![0.0f64; n];
    for r in 0..n {
        for cc in r..n {
            let mut s = 0.0f64;
            for k in 0..n {
                s += x[k].powi(r as i32) * x[k].powi(cc as i32);
            }
            a[r][cc] = s;
            a[cc][r] = s;
        }
        let mut s = 0.0f64;
        for k in 0..n {
            s += x[k].powi(r as i32) * y[k];
        }
        b[r] = s;
    }
    gauss_solve_f64(&mut a, &mut b)
}

/// Evaluate a polynomial (ascending powers) at `t`, in `f64`.
fn poly_eval_f64(c: &[f64], t: f64) -> f64 {
    c.iter()
        .enumerate()
        .fold(0.0f64, |acc, (i, &ci)| acc + ci * t.powi(i as i32))
}

/// Gauss elimination with partial pivoting; `a` is mutated in place.
fn gauss_solve_f64(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for c in 0..n {
        let mut piv = c;
        for r in (c + 1)..n {
            if a[r][c].abs() > a[piv][c].abs() {
                piv = r;
            }
        }
        if piv != c {
            a.swap(c, piv);
            b.swap(c, piv);
        }
        let ap = a[c][c];
        for r in (c + 1)..n {
            let f = a[r][c] / ap;
            if f.abs() > 1e-30 {
                for cc in c..n {
                    a[r][cc] -= f * a[c][cc];
                }
                b[r] -= f * b[c];
            }
        }
    }
    let mut xs = vec![0.0f64; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for cc in (r + 1)..n {
            s -= a[r][cc] * xs[cc];
        }
        xs[r] = s / a[r][r];
    }
    xs
}

/// Nuttall-4 window of length `n`, normalised to `n / 300`. The reference
/// normalises to that constant (it is an *approximate* unit-window — not
/// exactly 1.0; the downstream FFT compensates in the power computation
/// and the constant only matters for the absolute scale of `s`).
fn nuttal4(n: usize) -> Vec<f32> {
    const A: [(f32, f32); 4] = [
        (0.3635819, 0.0),
        (-0.4891775, 2.0),
        (0.1365995, 4.0),
        (-0.0106411, 6.0),
    ];
    let pi = 4.0f32.atan();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = A[0].0;
        for k in 1..4 {
            sum += A[k].0 * ((A[k].1 * pi * i as f32 / n as f32).cos());
        }
        out[i] = sum;
    }
    let norm: f32 = out.iter().sum::<f32>().max(1e-9);
    for v in out.iter_mut() {
        *v = *v / norm * n as f32 / 300.0;
    }
    out
}

/// Complex FFT of length `n` (power-of-two), returning the first `n/2+1` bins.
fn fft(x: &[f32]) -> Vec<Complex<f32>> {
    let mut planner = FftPlanner::new();
    let f = planner.plan_fft_forward(x.len());
    let mut c: Vec<Complex<f32>> = x.iter().map(|v| Complex::new(*v, 0.0)).collect();
    f.process(&mut c);
    c.truncate(x.len() / 2 + 1);
    c
}

/// Score one (freq bin `i`, symbol offset `j`, in ±`mode.jz`) over the
/// three Costas blocks, returning `max(compute(0,2), compute(0,1),
/// compute(1,2))`. `sp` is the power spectrum (see [`Spectra`]).
pub fn score(mode: &Mode, i: usize, j: isize, sp: &Spectra) -> f32 {
    let mut t0 = [0.0f32; 3];
    let mut t1 = [0.0f32; 3];
    for p in 0..3 {
        for n in 0..7 {
            let offset = j as isize + mode.jstrt as isize + n as isize * 4 + p as isize * 36 * 4;
            if offset >= 0 && (offset as usize) < sp.nhsym {
                let off = offset as usize;
                t0[p] += sp.at(i + NFOS * mode.costas[p][n], off);
                for freq in 0..7 {
                    t1[p] += sp.at(i + NFOS * freq, off);
                }
            }
        }
    }
    let compute = |lo: usize, hi: usize| {
        let mut tx = 0.0f32;
        let mut t1v = 0.0f32;
        for k in lo..=hi {
            tx += t0[k];
            t1v += t1[k];
        }
        tx / ((t1v - tx) / 6.0)
    };
    compute(0, 2).max(compute(0, 1)).max(compute(1, 2))
}

/// The full sync pass over `sp`: fit the baseline over `[nfa, nfb]`,
/// score all (freq, offset) pairs, normalise to the 40th percentile,
/// prune near-duplicates by frequency, and return the strongest
/// `NMAXCAND` candidates (sync ≥ [`ASYNCMIN`]), ordered for the caller
/// (strongest first — the caller re-sorts toward `nfqso` in hl2-api).
pub fn sync_pass(mode: &Mode, sp: &mut Spectra, nfa: f32, nfb: f32) -> Vec<Sync> {
    let df = mode.df;
    let mut nfa = nfa.max(100.0);
    let mut nfb = nfb.min(4910.0);
    let nwin = (nfb - nfa).abs();
    if nfb < 100.0 {
        nfb = nfa;
    }
    if nfa < 100.0 {
        nfa = 100.0;
        if nwin < 100.0 {
            nfb = nfa + nwin;
        }
    }
    if nfb > 4910.0 {
        nfb = 4910.0;
        if nwin < 100.0 {
            nfa = nfb - nwin;
        }
    }
    let ia = (nfa / df).round().max(0.0) as usize;
    let ib = (nfb / df).round() as usize;
    sp.fit_baseline(df, ia, ib);

    #[derive(Clone, Copy)]
    struct Raw {
        i: usize,
        j: isize,
        v: f32,
    }
    let mut raw: Vec<Raw> = (ia..=ib)
        .map(|i| {
            let mut best = f32::NEG_INFINITY;
            let mut bj = -1;
            for j in -(mode.jz as isize)..=mode.jz as isize {
                let v = score(mode, i, j, sp);
                if v > best {
                    best = v;
                    bj = j;
                }
            }
            Raw { i, j: bj, v: best }
        })
        .collect();

    if raw.is_empty() {
        return Vec::new();
    }

    // 40th percentile normalisation (rank `size * 4 / 10`, 0-based).
    raw.sort_unstable_by(|a, b| a.v.total_cmp(&b.v));
    let rank = (raw.len() * 4 / 10).max(1).min(raw.len() - 1);
    let norm = raw[rank].v;

    if std::env::var_os("JS8_TRACE").is_some() {
        let top = raw.last().map(|r| r.v);
        eprintln!(
            "JS8_TRACE sync_pass mode={} raw size {} p40 {norm} top {top:?}",
            mode.name,
            raw.len()
        );
    }

    for e in raw.iter_mut() {
        e.v /= norm;
    }
    raw.sort_unstable_by(|a, b| b.v.total_cmp(&a.v));

    let mut out: Vec<Sync> = Vec::new();
    let mut used: Vec<bool> = vec![false; raw.len()];
    for k in 0..raw.len() {
        if used[k] {
            continue;
        }
        if raw[k].v < ASYNCMIN || raw[k].v.is_nan() {
            break;
        }
        used[k] = true;
        out.push(Sync {
            freq: df * raw[k].i as f32,
            dt: mode.tstep * (raw[k].j as f32 + 0.5),
            sync: raw[k].v,
        });
        for (l, other) in raw.iter().enumerate() {
            if l == k {
                continue;
            }
            let f = df * other.i as f32;
            if (f - df * raw[k].i as f32).abs() <= mode.az {
                used[l] = true;
            }
        }
        if out.len() >= NMAXCAND {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nuttal_is_positive_and_normalised() {
        let w = nuttal4(32);
        assert!(w.iter().all(|&v| v > 0.0));
        let sum: f32 = w.iter().sum();
        assert!((sum - 32.0 / 300.0).abs() < 1e-3, "sum {sum}");
    }

    #[test]
    fn polyfit_recovers_square() {
        let xs: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x * x).collect();
        let c = polyfit_f64(&xs, &ys);
        assert!((c[0] - 0.0).abs() < 1e-2);
        assert!((c[1] - 0.0).abs() < 1e-2);
        assert!((c[2] - 1.0).abs() < 1e-3);
    }
}
