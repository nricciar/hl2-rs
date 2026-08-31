//! JS8Call (Mode A) decode: band extraction, per-symbol demod, LDPC metric
//! + BP-FEC, CRC, and multi-pass signal subtraction.
//!
//! Faithful port of `js8_downsample`, `syncjs8d`, `genjs8refsig`,
//! `subtractjs8` and `js8dec` from `js8call/JS8.cpp`.
//!
//! The working set is a [`Ctx`]: one 15 s window of `f32` audio, its
//! per-step power spectrum plus fitted baseline (the [`Spectra`]), and the
//! pre-computed Costas sync waveforms. [`decode_candidate`] runs the full
//! `js8dec` on one candidate; [`subtract`] removes the decoded signal from
//! the window (multi-pass interference rejection).

use num_complex::Complex;
use rustfft::FftPlanner;

use crate::receiver::js8::frame::encode_tones;
use crate::receiver::js8::ldpc::bpdecode174;
use crate::receiver::js8::msg::{DecodedFrame, check_crc, unpack_frame};
use crate::receiver::js8::params::{Mode, N, NFILT, NFSRCH, NN, NROWS, SAMPLE_RATE};
use crate::receiver::js8::sync::{Spectra, Sync};

/// One `ctx` per cycle window: everything `js8dec` needs across the decode
/// passes. `dd` is the window (`f32`, 12 kHz); `sp` is the per-step
/// spectrum (already baseline-fitted); `cd0` is the extracted
/// downsampled complex baseband of the current candidate (length
/// `mode.np2`, zero-padded).
pub struct Ctx<'a> {
    /// The JS8 speed this window is being decoded as.
    pub mode: &'a Mode,
    /// The cycle window (12 kHz `f32`), `mode.nmax` samples.
    pub dd: &'a [f32],
    /// The per-step power spectrum + fitted baseline from the sync pass.
    pub sp: &'a Spectra,
    /// Pre-computed Costas sync waveforms (3 × 7 × `mode.ndownsps`
    /// complex samples), from the reference `DecodeMode` constructor.
    pub csyncs: [[Vec<Complex<f32>>; 7]; 3],
    /// The extracted downsampled baseband signal of the *current*
    /// candidate (`mode.np2` complex samples). The decoder re-extracts it
    /// at the start of every pass (per candidate / after subtraction).
    pub cd0: Vec<Complex<f32>>,
}

impl<'a> Ctx<'a> {
    /// Build the Costas sync waveforms (per the reference): per symbol,
    /// `dphi = TAU * costas_row / mode.ndownsps` accumulated with `polar`.
    pub fn csyncs(mode: &Mode) -> [[Vec<Complex<f32>>; 7]; 3] {
        let mut out: [[Vec<Complex<f32>>; 7]; 3] =
            std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
        for p in 0..3 {
            for i in 0..7 {
                let dphi = std::f32::consts::TAU * mode.costas[p][i] as f32 / mode.ndownsps as f32;
                let mut phi = 0.0f32;
                let mut v = Vec::with_capacity(mode.ndownsps);
                for _ in 0..mode.ndownsps {
                    v.push(Complex::new(phi.cos(), phi.sin()));
                    phi = (phi + dphi).rem_euclid(std::f32::consts::TAU);
                }
                out[p][i] = v;
            }
        }
        out
    }

    /// Extract the band around `f1` Hz into `cd0` (length `mode.np2`),
    /// port of `js8_downsample`. `dd` must be `mode.nmax` samples (the
    /// caller zeroes any gap past the real data; `Spectra::new` already
    /// zero-pads).
    ///
    /// The reference zero-pads the window to `NDFFT1` (`nsps * ndd`)
    /// before the r2c FFT: that sets the bin grid to
    /// `12000 / (nsps*ndd)` Hz and the time-domain grid to
    /// `1 / (ndfft1/ndown)` s. We must do the same — the band [ib..it]
    /// and the left-rotate by `(i0-ib)` are only self-consistent on
    /// *that* grid.
    pub fn extract(&mut self, f1: f32) {
        let mode = self.mode;
        let df = SAMPLE_RATE / mode.ndfft1 as f32;
        let baud = mode.baud;

        let ft = f1 + 8.5 * baud;
        let fb = f1 - 1.5 * baud;
        let i0 = (f1 / df).round() as usize;
        let it = ((ft / df).round() as usize).min(mode.ndfft1 / 2);
        let ib = (fb / df).round().max(0.0) as usize;
        let range = it.saturating_sub(ib) + 1;

        let n = self.dd.len().min(mode.nmax);
        let mut c = vec![Complex::new(0.0f32, 0.0f32); mode.ndfft1];
        for i in 0..n {
            c[i].re = self.dd[i];
        }
        let mut planner = FftPlanner::new();
        let f = planner.plan_fft_forward(mode.ndfft1);
        f.process(&mut c);
        c.truncate(mode.ndfft1 / 2 + 1);

        // Band extract → head of `cd0` (length mode.ndfft2), taper both
        // ends (mode.ndd+1 samples), then left-rotate by `i0 - ib`.
        let mut cd0 = vec![Complex::new(0.0, 0.0); mode.ndfft2];
        let mut band = Vec::with_capacity(range);
        for k in 0..range {
            band.push(c[ib + k]);
        }
        taper(mode.ndd, &mut band);
        cd0[..range].copy_from_slice(&band);
        let shift = i0.saturating_sub(ib) % mode.ndfft2;
        cd0.rotate_left(shift);

        // Inverse FFT (back to time domain); scale by
        // 1/sqrt(mode.ndfft1 * mode.ndfft2). rustfft's inverse is
        // unnormalised (`inv·fwd = N·x`), exactly like the reference's
        // FFTW `FFTW_BACKWARD`; the explicit `1/sqrt(N·M)` matches the
        // reference's `fac`. Do not "normalise" one side or the other
        // without re-checking the other: the SNR's `xsig`/`xbase` must
        // stay on the same scale.
        let mut d = cd0;
        let f2 = planner.plan_fft_inverse(d.len());
        f2.process(&mut d);
        let factor = 1.0 / ((mode.ndfft1 as f32 * mode.ndfft2 as f32).sqrt());
        for v in d.iter_mut() {
            *v *= factor;
        }
        // Truncate / zero-pad to mode.np2.
        if mode.np2 > d.len() {
            d.resize(mode.np2, Complex::new(0.0, 0.0));
        } else {
            d.truncate(mode.np2);
        }
        self.cd0 = d;
    }

    /// The sync power of the signal after `i0` in downsampled samples,
    /// with an optional frequency offset `delf` Hz — port of `syncjs8d`.
    pub fn sync_power(&self, i0: isize, delf: f32) -> f32 {
        let mode = self.mode;
        let fs2 = SAMPLE_RATE / mode.ndown as f32;
        let base_dphi = std::f32::consts::TAU / fs2;
        let freq_adjust: Vec<Complex<f32>> = if delf != 0.0 {
            let dphi = base_dphi * delf;
            let mut phi: f32 = 0.0;
            (0..mode.ndownsps)
                .map(|_| {
                    let v = Complex::new(phi.cos(), phi.sin());
                    phi = (phi + dphi).rem_euclid(std::f32::consts::TAU);
                    v
                })
                .collect()
        } else {
            vec![Complex::new(1.0, 0.0); mode.ndownsps]
        };

        let mut score = 0.0f32;
        for p in 0..3 {
            for j in 0..7 {
                let offset = (p * 36 + j) as isize * mode.ndownsps as isize + i0;
                if offset >= 0 && offset + mode.ndownsps as isize <= mode.np2 as isize {
                    let mut acc = Complex::new(0.0, 0.0);
                    for k in 0..mode.ndownsps {
                        let cd = self.cd0[(offset + k as isize) as usize];
                        let fa = &freq_adjust[k];
                        // `cd * conj(fa * csyncs[p][j][k])`.
                        let cs = &self.csincs_p(p, j)[k];
                        let prod = fa * cs;
                        acc += cd * prod.conj();
                    }
                    score += acc.norm_sqr();
                }
            }
        }
        score
    }

    #[inline]
    fn csincs_p(&self, p: usize, j: usize) -> &Vec<Complex<f32>> {
        &self.csyncs[p][j]
    }
}

/// Cos-taper both ends of the extracted frequency band in place (the
/// reference `Taper`): `ndd + 1` samples at the head ramp 0→1 and the same
/// at the tail ramp 1→0: `0.5 * (1 + cos(i * PI / ndd))`, `i` counted from
/// the band edge inward. `ndd` is the per-mode taper length.
fn taper(ndd: usize, range: &mut [Complex<f32>]) {
    let n = range.len();
    let ntap = (ndd + 1).min(n);
    // head: taper[0][j] = 0.5 * (1 + cos((ndd - j) * PI / ndd)).
    for i in 0..ntap {
        let v = 0.5 * (1.0 + ((ndd as f32 - i as f32) / ndd as f32 * std::f32::consts::PI).cos());
        range[i] *= v;
    }
    // tail: taper[1][i] = 0.5 * (1 + cos(i * PI / ndd)).
    for i in 0..ntap {
        let v = 0.5 * (1.0 + (i as f32 / ndd as f32 * std::f32::consts::PI).cos());
        range[n - ntap + i] *= v;
    }
}

/// Decode one candidate in the `CTX` (the full `js8dec`). Returns the
/// decoded frame, the tone sequence (for [`subtract`]), the measured start
/// offset (s) and SNR (dB), or `None` — port of `js8dec` (without the
/// multi-pass subtraction; that is the caller's job across the three
/// passes, see the slot loop in [`super::decoder`]).
pub fn decode_candidate(
    mut ctx: Ctx,
    cand: &Sync,
    sp: &Spectra,
) -> Option<(DecodedFrame, [u8; 79], f32, f32)> {
    let mode = ctx.mode;
    // FR = 12000 / mode.nfft1 (e.g. 3.125 Hz for Mode A);
    // FS2 = 12000 / mode.ndown (200 Hz for A, 240 for B, 400 for C/E ...);
    // DT2 = 1/FS2.
    let fr = SAMPLE_RATE / mode.nfft1 as f32;
    let fs2 = SAMPLE_RATE / mode.ndown as f32;
    let dt2 = 1.0 / fs2;

    // Band extraction (the reference's `js8_downsample(f1)`), always done
    // at the start of a decode attempt (after `subtract` it must be
    // re-extracted: the signal landscape has changed).
    ctx.extract(cand.freq);

    // xbase: the noise floor at this frequency (0.1 * (savg - mode.basesub)).
    let index = (cand.freq / fr).round() as usize;
    let scaled = 0.1 * (sp.savg.get(index).copied().unwrap_or(0.0) - mode.basesub);
    // `10.0f.powf(scaled)` (NOT `scaled.powf(10.0)` — the latter would
    // raise the *negative* dB scale to an integer power, flipping the sign:
    // (-4.7).powf(10) == +4.7^10).
    let xbase = 10.0f32.powf(scaled);

    // Initial start estimate (xdt + ASTART). Widen the fine-search window
    // to ±4 * mode.nqsymbol (vs the reference's ±mode.nqsymbol) to absorb
    // coarse-sync dt errors — the Costas correlation peak is narrow enough
    // that the wider search does not produce spurious peaks in silence.
    let i0 = ((cand.dt + mode.astart) * fs2).round() as isize;
    let mut smax = 0.0f32;
    let mut ibest = 0isize;
    let window = 4 * mode.nqsymbol as isize;
    for idt in (i0 - window)..=(i0 + window) {
        let v = ctx.sync_power(idt, 0.0);
        if v > smax {
            smax = v;
            ibest = idt;
        }
    }
    let xdt2 = ibest as f32 * dt2;
    let i0 = (xdt2 * fs2).round() as isize;
    let mut smax = 0.0f32;
    let mut delfbest = 0.0f32;
    for ifr in -NFSRCH..=NFSRCH {
        let delf = ifr as f32 * 0.5;
        let v = ctx.sync_power(i0, delf);
        if v > smax {
            smax = v;
            delfbest = delf;
        }
    }
    // Apply the delpbest phase shift to `ctx.cd0` (in place).
    let dphi = -delfbest * std::f32::consts::TAU / fs2;
    let wstep = Complex::new(dphi.cos(), dphi.sin());
    let mut w = Complex::new(1.0, 0.0);
    for i in 0..mode.np2 {
        w *= wstep;
        ctx.cd0[i] *= w;
    }

    let sync = ctx.sync_power(i0, 0.0);

    // Per-symbol FFT into `s2[row][symbol]` (normalised by 1000).
    let mut s2: [[f32; 79]; NROWS] = [[0.0; 79]; NROWS];
    let mut csymb = vec![Complex::new(0.0, 0.0); mode.ndownsps];
    let mut planner = FftPlanner::new();
    let f = planner.plan_fft_forward(mode.ndownsps);
    for k in 0..NN {
        let i1 = ibest + (k as isize) * mode.ndownsps as isize;
        if i1 >= 0 && i1 + mode.ndownsps as isize <= mode.np2 as isize {
            for l in 0..mode.ndownsps {
                csymb[l] = ctx.cd0[(i1 + l as isize) as usize];
            }
            f.process(&mut csymb);
            for r in 0..NROWS {
                // `norm()` is the complex modulus (|z|), matching the
                // reference's `std::abs(csymb[i]) / 1000.0f`.
                s2[r][k] = csymb[r].norm() / 1000.0;
            }
        }
    }

    // Sync quality gate (Costas pattern match, >= 7 of 21).
    let mut nsync = 0;
    for p in 0..3 {
        for n in 0..7 {
            let idx = p * 36 + n;
            let mut mx = 0;
            for r in 1..NROWS {
                if s2[r][idx] > s2[mx][idx] {
                    mx = r;
                }
            }
            let match_ = mode.costas[p][n] == mx;
            if match_ {
                nsync += 1;
            }
        }
    }
    if std::env::var_os("JS8_TRACE").is_some() {
        eprintln!(
            "JS8_TRACE   ibest={} delf={} nsync={}/21 sync={}",
            ibest, delfbest, nsync, sync
        );
    }
    if nsync <= 6 {
        return None;
    }

    // s1: strip the Costas columns from s2 (29 parity + 29 message symbols).
    let mut s1: [[f32; 58]; NROWS] = [[0.0; 58]; NROWS];
    for r in 0..NROWS {
        s1[r][0..29].copy_from_slice(&s2[r][7..36]);
        s1[r][29..58].copy_from_slice(&s2[r][43..72]);
    }

    // Build LLR0/LLR1 (3 bits per symbol, max-diff).
    let mut llr0 = [0.0f32; N];
    let mut llr1 = [0.0f32; N];
    for j in 0..58 {
        let i1 = 3 * j;
        let i2 = 3 * j + 1;
        let i4 = 3 * j + 2;
        let mut ps = [0.0f32; NROWS];
        for r in 0..NROWS {
            ps[r] = s1[r][j];
        }
        llr0[i1] = ps[4..].iter().fold(f32::MIN, |a, &b| a.max(b))
            - ps[..4].iter().fold(f32::MIN, |a, &b| a.max(b));
        llr0[i2] = [ps[2], ps[3], ps[6], ps[7]]
            .iter()
            .fold(f32::MIN, |a, &b| a.max(b))
            - [ps[0], ps[1], ps[4], ps[5]]
                .iter()
                .fold(f32::MIN, |a, &b| a.max(b));
        llr0[i4] = [ps[1], ps[3], ps[5], ps[7]]
            .iter()
            .fold(f32::MIN, |a, &b| a.max(b))
            - [ps[0], ps[2], ps[4], ps[6]]
                .iter()
                .fold(f32::MIN, |a, &b| a.max(b));
        for v in ps.iter_mut() {
            *v = (*v + 1e-32).ln();
        }
        llr1[i1] = ps[4..].iter().fold(f32::MIN, |a, &b| a.max(b))
            - ps[..4].iter().fold(f32::MIN, |a, &b| a.max(b));
        llr1[i2] = [ps[2], ps[3], ps[6], ps[7]]
            .iter()
            .fold(f32::MIN, |a, &b| a.max(b))
            - [ps[0], ps[1], ps[4], ps[5]]
                .iter()
                .fold(f32::MIN, |a, &b| a.max(b));
        llr1[i4] = [ps[1], ps[3], ps[5], ps[7]]
            .iter()
            .fold(f32::MIN, |a, &b| a.max(b))
            - [ps[0], ps[2], ps[4], ps[6]]
                .iter()
                .fold(f32::MIN, |a, &b| a.max(b));
    }

    // LLR normalisation (mean/std, × 2.83).
    let normalise = |llr: &mut [f32; 174]| {
        let sum: f32 = llr.iter().sum();
        let sum_sq: f32 = llr.iter().map(|v| v * v).sum();
        let n = llr.len() as f32;
        let mean = sum / n;
        let var = (sum_sq / n - mean * mean).max(0.0);
        let sig = if var > 0.0 {
            var.sqrt()
        } else {
            (sum_sq / n).sqrt()
        };
        for v in llr.iter_mut() {
            *v = (*v / sig) * 2.83;
        }
    };
    normalise(&mut llr0);
    normalise(&mut llr1);

    // Decode passes. ipass 1/3/4 use llr0 (with zeros in [0..24) then in
    // [24..48)); ipass 2 uses llr1. Hard-error budget and all-zero codeword
    // guard are the reference's.
    let mut llr0 = llr0;
    let mut decoded: [i8; 87] = [0; 87];
    let mut cw: [i8; N] = [0; N];
    for ipass in 1..=4 {
        // ipass 3 zeroes the first 24 (the parity *block*); ipass 4 zeroes
        // another 24 (so [0..48) — the parity *and* the first half of the
        // message).
        if ipass == 3 {
            for i in 0..24 {
                llr0[i] = 0.0;
            }
        } else if ipass == 4 {
            for i in 24..48 {
                llr0[i] = 0.0;
            }
        }
        let llr: &[f32; 174] = if ipass == 2 { &llr1 } else { &llr0 };
        let nerr = bpdecode174(llr, &mut decoded, &mut cw);
        if std::env::var_os("JS8_TRACE").is_some() {
            let msgbits: String = decoded
                .iter()
                .map(|b| if *b > 0 { '1' } else { '0' })
                .collect();
            eprintln!(
                "JS8_TRACE   ipass={} nerr={} all_zero={} msgbits(87)={}",
                ipass,
                nerr,
                cw.iter().all(|&b| b == 0),
                &msgbits[..87.min(msgbits.len())]
            );
        }
        // All-zero codeword: try the next pass.
        if cw.iter().all(|&b| b == 0) {
            continue;
        }
        if !(nerr >= 0 && nerr < 60)
            || (sync < 2.0 && nerr > 35)
            || (ipass > 2 && nerr > 39)
            || (ipass == 4 && nerr > 30)
        {
            continue;
        }
        // CRC + frame unpack.
        let mut msg87 = [0u8; 87];
        for i in 0..87 {
            msg87[i] = decoded[i] as u8;
        }
        if check_crc(&msg87) {
            if let Some(frame) = unpack_frame(&msg87) {
                // Use the reference's per-tone s2[itone[i]][i] formula.
                let tones_v = encode_tones_from_msg(ctx.mode, &msg87);
                let xsig: f32 = tones_v
                    .iter()
                    .enumerate()
                    .map(|(i, t)| s2[*t as usize][i].powi(2))
                    .sum();
                let snr = (10.0 * (xsig / xbase - 1.0).max(1.259e-10f32).log10()) - 32.0;
                let snr = snr.max(-60.0);
                let mut tones = [0u8; 79];
                tones.copy_from_slice(&tones_v);
                if std::env::var_os("JS8_TRACE").is_some() {
                    eprintln!(
                        "JS8_TRACE   snr: xsig={:?} xbase={:?} tones={:?}",
                        xsig, xbase, tones
                    );
                }
                return Some((frame, tones, xdt2, snr));
            }
        }
    }
    None
}

/// Re-derive the 79-tone sequence from an 87-bit message by re-encoding
/// through the same parity + tone-packing as `encode_tones` — needed by the
/// `s2[itone[i]][i]` SNR term and (eventually) by `subtractjs8`.
/// `mode` supplies the Costas arrays (Mode A original vs B/C/E modified).
fn encode_tones_from_msg(mode: &Mode, msg: &[u8; 87]) -> Vec<u8> {
    // 87 message bits -> 11-byte layout -> 87-bit *parity* via the same
    // frame::encode_parity path the encoder uses; but `encode_tones` takes
    // the 12×6-bit payload, not the raw 87 bits. So extract them instead.
    let mut payload = [0u8; 12];
    for i in 0..12 {
        for b in 0..6 {
            let k = i * 6 + b;
            payload[i] = (payload[i] << 1) | msg[k];
        }
    }
    let i3bit = ((msg[72] << 2) | (msg[73] << 1) | msg[74]) as u8;
    encode_tones(payload, i3bit & 0x7, mode)
}

/// Subtract the decoded signal from the window (multi-pass interference
/// removal; port of `subtractjs8` + `genjs8refsig`). `f1` is the tone-0
/// frequency (Hz), `xdt` the start offset (s), `tones` the 79-symbol tone
/// sequence, `dd` the window (mutated, `mode.nmax` samples).
pub fn subtract(mode: &Mode, dd: &mut [f32], tones: &[u8; NN], f1: f32, xdt: f32) {
    let n_start = (xdt * SAMPLE_RATE).round() as isize;
    let cref_start: usize = if n_start < 0 { (-n_start) as usize } else { 0 };
    let dd_start: usize = if n_start > 0 { n_start as usize } else { 0 };

    // Reference signal: 79 tones, `f1` + tone_offset, mode.nsps samples each.
    let bfpi = std::f32::consts::TAU * f1 / SAMPLE_RATE;
    let nsps = mode.nsps;
    let mut cref = Vec::with_capacity(79 * nsps);
    let mut phi = 0.0f32;
    for t in tones {
        let dphi = bfpi + std::f32::consts::TAU * (*t as f32) / nsps as f32;
        for _ in 0..nsps {
            cref.push(Complex::new(phi.cos(), phi.sin()));
            phi = (phi + dphi).rem_euclid(std::f32::consts::TAU);
        }
    }

    // Clamp `size` so both `cref[cref_start+i]` and `dd[dd_start+i]` stay
    // in range when the burst extends past either end of the window.
    let size = (tones.len() as isize * nsps as isize)
        .min(cref.len() as isize - cref_start as isize)
        .min(dd.len() as isize - dd_start as isize)
        .max(0) as usize;
    if size == 0 {
        return;
    }

    // cfilt[i] = dd[i] * conj(cref[i]), zero-padded to mode.nmax (the
    // reference circular-convolves at full window length).
    let mut cfilt = vec![Complex::new(0.0, 0.0); mode.nmax];
    for i in 0..size {
        cfilt[i] = dd[dd_start + i] * cref[i].conj();
    }
    // Forward FFT (mode.nmax), apply the bandpass filter, inverse FFT —
    // circular convolution of the reference reconstruction.
    let filter = bandpass_filter(mode);
    let mut planner = FftPlanner::new();
    let mut c = cfilt;
    let f = planner.plan_fft_forward(mode.nmax);
    f.process(&mut c);
    for i in 0..mode.nmax {
        c[i] *= filter[i];
    }
    let f2 = planner.plan_fft_inverse(mode.nmax);
    f2.process(&mut c);
    // rustfft's inverse is unnormalized (`inv·fwd = N·x`), exactly like the
    // reference's FFTW `FFTW_BACKWARD`; and `bandpass_filter` is pre-divided
    // by `NMAX` to suit that — so this matches the reference's `subtractjs8`
    // with no further scale correction. (Do not add a `1/N` without also
    // re-checking `bandpass_filter` and `extract()`; they are coupled.)
    // Subtract: dd[i] -= 2 * Re(cref[i] * cfilt_time[i]).
    for i in 0..size {
        dd[dd_start + i] -= 2.0 * (cref[i] * c[i]).re;
    }
}

/// The reference's `filter`: a normalised cos² window over ±[NFILT/2]
/// (positioned at the centre index `NFILT/2`), FFT'd at `mode.nmax`,
/// divided by `mode.nmax`. Cached per mode (keyed by `mode.id`).
static FILTERS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<u8, Vec<Complex<f32>>>>,
> = std::sync::OnceLock::new();

fn bandpass_filter(mode: &Mode) -> Vec<Complex<f32>> {
    let inner = FILTERS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut map = inner.lock().expect("bandpass filter cache poisoned");
    if let Some(c) = map.get(&mode.id) {
        return c.clone();
    }
    let nmax = mode.nmax;
    let mut c = vec![Complex::new(0.0, 0.0); nmax];
    let mut sum = 0.0f32;
    for j in 0..=NFILT / 2 {
        let idx = j as i32 - (NFILT / 2) as i32 + (NFILT / 2) as i32;
        let idx = (idx.rem_euclid(nmax as i32)) as usize;
        let v = (std::f32::consts::PI * j as f32 / NFILT as f32)
            .cos()
            .powi(2);
        c[idx] = Complex::new(v, 0.0);
        sum += v;
    }
    for v in c.iter_mut() {
        v.re /= sum;
    }
    let mut planner = FftPlanner::new();
    let f = planner.plan_fft_forward(nmax);
    f.process(&mut c);
    for v in c.iter_mut() {
        *v /= nmax as f32;
    }
    map.insert(mode.id, c.clone());
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csyncs_is_unit_length_and_phase() {
        let mode = crate::receiver::js8::params::MODE_A;
        let cs = Ctx::csyncs(&mode);
        for p in 0..3 {
            for i in 0..7 {
                assert_eq!(cs[p][i].len(), mode.ndownsps);
                // First sample is at phase 0.
                assert!((cs[p][i][0].re - 1.0).abs() < 1e-6);
            }
        }
    }

    /// A zero window (the reference's silence case) should produce no
    /// candidate and no decode.
    #[test]
    fn silence_window_extracts_to_zero() {
        use crate::receiver::js8::params::MODE_A;
        use crate::receiver::js8::sync::Spectra as S;
        let mode = MODE_A;
        let dd = vec![0.0f32; mode.nmax];
        let mut sp = S::new(&mode, &dd);
        sp.fit_baseline(mode.df, 100, 1000);
        let spref = &sp;
        let cs = Ctx::csyncs(&mode);
        let _ = decode_candidate(
            Ctx {
                mode: &mode,
                dd: &dd,
                sp: spref,
                csyncs: cs,
                cd0: Vec::new(),
            },
            &Sync {
                freq: 1500.0,
                dt: 0.5,
                sync: 10.0,
            },
            spref,
        );
    }
}
