//! Demodulator micro-benchmark.
//!
//! Measures the hot-path building blocks this optimization work introduced, on
//! synthetic data (one 2-second 1.5 kHz tone at 192 kHz, plus per-build-unit
//! micro-benches):
//!
//!   * `Nco::step` — the phase-recurrence mixer, vs. the per-sample
//!     `cos`/`sin` it replaced (reported as a ratio).
//!   * `PolyphaseDecimator` — one output tap, vs. the old full-FIR decimate
//!     (≈ M taps of work, reported as the equivalent cost).
//!   * Full `SsbDemodulator` (4.8 kHz) and `DigitalDemodulator` (12 kHz) end-to-end
//!     sample throughput.
//!
//! Timing uses `Instant` over N=5 repetitions and reports the best (min) run,
//! which is the most stable under the scheduler.
//!
//! This is a rough guide, not a calibrated benchmark: `cargo run --release` it
//! on the target machine. The absolute numbers will vary (CPU, cache state,
//! turbo) but the *shape* — recurrence vs. trig, polyphase vs. full-FIR —
//! is the point.

use std::time::Instant;

use num_complex::Complex;

use hl2::receiver::demod::{F32Fir, Nco, PolyphaseDecimator};
use hl2::receiver::{AudioConfig, Demodulator, IqBlock, Mode, Sideband, VecSink, make_demod};

/// Repetitions for the "best-of" timing (the scheduler is noisier for the
/// short micro-benches, so more reps = more stable result).
const REPS: usize = 5;
/// One full 1 500 Hz tone, 2 s long at `rate` Hz — the "realistic" block for
/// the end-to-end demod benches.
const RATE: u32 = 192_000;
const TONE_N: usize = 2 * RATE as usize;

fn tone(freq_hz: f64, rate_hz: u32, n: usize, amp: f32) -> IqBlock {
    let w = 2.0 * std::f64::consts::PI * freq_hz / rate_hz as f64;
    (0..n)
        .map(|i| {
            let a = (w * i as f64).cos();
            let b = (w * i as f64).sin();
            Complex::new(a as f32 * amp, b as f32 * amp)
        })
        .collect()
}

/// Best (minimum) wall time over `REPS` runs of `f`.
fn bench<F, R>(f: &mut F) -> std::time::Duration
where
    F: FnMut() -> R,
{
    let mut best = None;
    for _ in 0..REPS {
        let t = Instant::now();
        f();
        let d = t.elapsed();
        if best.is_none_or(|b| d < b) {
            best = Some(d);
        }
    }
    best.unwrap()
}

fn main() {
    println!("hl2 bench — {} reps, best-of", REPS);
    println!("tone: {:.1} kHz, {} samples @{} Hz", 1.5, TONE_N, RATE);
    println!();

    // ── Nco micro-bench ──────────────────────────────────────────────────
    // The new oscillator: phase recurrence (a few FMA, no transcendentals).
    // The old one: `cos(φ)` + `sin(φ)` per sample (libm, ~tens of ns).
    {
        let step = 2.0 * std::f64::consts::PI * 1_500.0 / RATE as f64;
        // New.
        let d_new = bench(&mut || {
            let mut nco = Nco::new(step);
            let mut acc = 0.0f64;
            for _ in 0..RATE as usize {
                let z = nco.step(Complex::new(0.7f32, 0.3f32));
                acc += z.re as f64;
            }
            std::hint::black_box(acc)
        });
        // Old (reference).
        let d_old = bench(&mut || {
            let mut acc = 0.0f64;
            let mut phi = 0.0f64;
            for _ in 0..RATE as usize {
                let c = phi.cos();
                let _s = phi.sin();
                acc += c;
                phi += step;
            }
            std::hint::black_box(acc)
        });
        let ns_new = d_new.as_nanos() as f64 / RATE as f64;
        let ns_old = d_old.as_nanos() as f64 / RATE as f64;
        println!(
            "Nco 1.5 kHz (per sample, {} Hz): new={:.2} ns  old={:.2} ns  x{:.1} faster",
            RATE,
            ns_new,
            ns_old,
            ns_old / ns_new.max(1e-9)
        );
    }

    // ── Polyphase micro-bench ────────────────────────────────────────────
    // One output tap of the decimator. The polyphase form runs ≈1/M of the
    // old full-FIR's taps per input sample, at identical output rate.
    {
        let m = 40usize; // 192k / 4.8k
        let taps = F32Fir::lowpass(257, 2_600.0 / RATE as f64, 8.6)
            .taps()
            .to_vec();
        let pp = PolyphaseDecimator::new(&taps, m);
        let d_new = bench(&mut || {
            let mut dec = pp.clone();
            let mut acc = 0.0f32;
            // Push m samples → m completed output taps (one per input, steady
            // state after the warm-up that the bench loop naturally hides).
            for _ in 0..m {
                if let Some(y) = dec.push(0.7f32) {
                    acc += y;
                }
            }
            std::hint::black_box(acc)
        });
        // Old (reference): a full 257-tap FIR, taking every M-th output —
        // ≈ m × the work of the polyphase form for the same 1 output sample.
        let d_old = bench(&mut || {
            let h = taps.clone();
            let mut acc = 0.0f32;
            // One "full-FIR decimate" step ≈ M full convolutions to produce
            // 1 output in the old scheme.
            for _ in 0..m {
                let mut sum = 0.0f32;
                for t in &h {
                    sum += t * 0.7f32;
                }
                acc += sum;
            }
            std::hint::black_box(acc)
        });
        let ns_new = d_new.as_nanos() as f64 / m as f64;
        let ns_old = d_old.as_nanos() as f64 / m as f64;
        println!(
            "Polyphase (per output tap, M={m}, {} taps): new={:.2} ns  old={:.2} ns  x{:.1} faster",
            taps.len(),
            ns_new,
            ns_old,
            ns_old / ns_new.max(1e-9)
        );
    }

    // ── End-to-end demod throughput ──────────────────────────────────────
    for (label, mode, cfg) in [
        (
            "Ssb/USB 4.8k",
            Mode::Ssb(Sideband::Usb),
            AudioConfig {
                rate_hz: 4_800,
                gain_db: 0.0,
            },
        ),
        (
            "Ft8/USB  12k",
            Mode::Ft8,
            AudioConfig {
                rate_hz: 12_000,
                gain_db: 0.0,
            },
        ),
    ] {
        let iq = tone(1_500.0, RATE, TONE_N, 0.5);
        let d = bench(&mut || {
            let mut demod: Box<dyn Demodulator> = match make_demod(mode, RATE, 0.0, 2_600, cfg) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("  {label}: make_demod failed: {e:?}");
                    std::process::exit(1);
                }
            };
            let mut sink = VecSink::new();
            let _ = demod.demod(&iq, &mut sink);
            std::hint::black_box(sink.samples().len())
        });
        let ms = d.as_secs_f64() * 1e3;
        let sps = TONE_N as f64 / (d.as_secs_f64().max(1e-9));
        let real_time_factor = d.as_secs_f64() / (TONE_N as f64 / RATE as f64);
        println!(
            "{label:14} {} samples: {:.1} ms  →  {:.0} samples/s  ({:.1}x faster than real-time)",
            TONE_N,
            ms,
            sps,
            (1.0 / real_time_factor.max(1e-9))
        );
    }
}
