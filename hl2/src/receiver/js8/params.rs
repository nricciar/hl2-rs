//! JS8Call protocol and DSP constants + per-speed `Mode` table.
//!
//! JS8Call has four user-selectable speeds ("submodes"), all sharing the
//! same 79-symbol (21 Costas sync + 58 LDPC data) frame structure, the
//! same LDPC(174,87) + CRC12 + 12-char varicode payload — and differing
//! only in samples-per-symbol, transmit duration, and a small set of tuned
//! sync/codec constants (Costas array, sync-search range, baseline offset,
//! downsample factor, taper length, etc.).
//!
//! This module holds both:
//! * the truly shared constants (frame shape, LDPC, baseline, Costas arrays);
//! * a `Mode` struct that encapsulates every mode-specific constant, with
//!   `MODE_A` / `MODE_B` / `MODE_C` / `MODE_E` covering the four speeds the
//!   reference implements (js8call `ModeA`–`ModeE`; `ModeI`/Ultra is
//!   hidden/disabled there and is not ported).
//!
//! Source of truth (mode constants): `js8call/JS8.cpp`
//! (`ModeA`/`ModeB`/`ModeC`/`ModeE` struct bodies around `JS8.cpp:211-380`)
//! and `js8call/JS8Submode.cpp` (`Data` constructor, `JS8Submode.cpp:55-77`).

/// Total channel symbols per transmit: 21 sync (3 Costas x 7) + 58 data.
pub const NN: usize = 79;

/// Sync (Costas) symbols: 3 blocks of 7.
pub const NS: usize = 21;

/// Data symbols: 58 (3 bits each = 174 codeword bits).
pub const ND: usize = 58;

/// LDPC codeword length.
pub const N: usize = 174;

/// LDPC information (message) bits: 72 payload + 3 frame-type + 12 CRC.
pub const K: usize = 87;

/// LDPC parity bits (N - K).
pub const M: usize = N - K;

/// Rows per symbol FFT (8-FSK).
pub const NROWS: usize = 8;

/// Symbol spectrum over-sampling factor.
pub const NFOS: usize = 2;

/// Symbol FFT decimation (quarter-symbol steps).
pub const NSSY: usize = 4;

/// 12 kHz — the JS8 decoder's fixed input rate (all speeds).
pub const SAMPLE_RATE: f32 = 12_000.0;

/// Normalised minimum sync score a candidate must reach.
pub const ASYNCMIN: f32 = 1.5;

/// Frequency search half-range in Hz (±2.5 Hz, fine sync).
pub const NFSRCH: i32 = 5;

/// Max candidate signals to attempt decoding on.
pub const NMAXCAND: usize = 300;

/// Impulse-response length for the bandpass filter in `subtract`.
pub const NFILT: usize = 1_400;

/// BP decoder limits (`JS8.cpp`).
pub const BP_MAX_ROWS: usize = 7;
pub const BP_MAX_CHECKS: usize = 3;
pub const BP_MAX_ITERATIONS: usize = 30;

/// Baseline polynomial degree (must be odd for the Estrin evaluation).
pub const BASELINE_DEGREE: usize = 5;

/// The reference's Chebyshev node positions (`JS8.cpp` `BASELINE_NODES`,
/// `0.5·(1 − cos(π(2i+1)/(2·n)))` with `n = BASELINE_DEGREE + 1 = 6`).
/// Length is `BASELINE_DEGREE + 1`.
pub const BASELINE_NODES_R: [f64; BASELINE_DEGREE + 1] = [
    0.01700404329549476,
    0.14589803375031546,
    0.36602540378443865,
    0.6339745962155614,
    0.8541019662496846,
    0.9829959567045052,
];

/// Baseline lower-envelope percentile (0-100).
pub const BASELINE_SAMPLE: usize = 10;

/// Closed Hz band considered for the baseline fit.
pub const BASELINE_MIN: f32 = 500.0;
pub const BASELINE_MAX: f32 = 2_500.0;

/// The 7x7 Costas arrays. Mode A uses the **original** (FT8) set;
/// B/C/E use the **modified** set. Each entry is a 7-symbol row (0..7 =
/// FSK row within the symbol).
pub const COSTAS_ORIGINAL: [[usize; 7]; 3] = [
    [4, 2, 5, 6, 1, 3, 0],
    [4, 2, 5, 6, 1, 3, 0],
    [4, 2, 5, 6, 1, 3, 0],
];

pub const COSTAS_MODIFIED: [[usize; 7]; 3] = [
    [0, 6, 2, 3, 5, 4, 1],
    [1, 5, 0, 2, 3, 6, 4],
    [2, 5, 0, 6, 4, 1, 3],
];

/// One JS8Call speed (submode): a full DSP/timing parameter set.
///
/// Derived (`#[derive(Copy)]`, `const`) fields are computed from the raw
/// inputs and mirror the reference's per-mode struct bodies
/// (`JS8.cpp:211-380`) one-for-one:
///
/// | field      | formula                                        |
/// |---|---|
/// | `nmax`     | `ntxdur * 12_000`                              |
/// | `nfft1`    | `nsps * NFOS`                                   |
/// | `nstep`    | `nsps / NSSY`                                   |
/// | `nhsym`    | `nmax / nstep - 3`                              |
/// | `ndown`    | `nsps / ndownsps`                               |
/// | `nqsymbol` | `ndownsps / 4`                                  |
/// | `ndfft1`   | `nsps * ndd`                                    |
/// | `ndfft2`   | `ndfft1 / ndown`                                |
/// | `np2`      | `NN * ndownsps`                                 |
/// | `tstep`    | `nstep` as f32 / 12000                          |
/// | `jstrt`    | (astart / tstep) as usize                       |
/// | `df`       | 12000 / nfft1 as f32                            |
/// | `baud`     | 12000 / nsps as f32                             |
/// | `cycle_ms` | `ntxdur as u64 * 1_000`                         |
pub const MODE_A: Mode = mode_const(
    0,
    "JS8A",
    1920,
    15,
    32,
    100,
    62,
    0.5,
    40.0,
    4.0,
    COSTAS_ORIGINAL,
);
/// Fast (Mode B): 10 s cycle, 10 baud.
pub const MODE_B: Mode = mode_const(
    1,
    "JS8B",
    1200,
    10,
    20,
    100,
    144,
    0.2,
    39.0,
    8.0,
    COSTAS_MODIFIED,
);
/// Turbo (Mode C): 6 s cycle, 20 baud.
pub const MODE_C: Mode = mode_const(
    2,
    "JS8C",
    600,
    6,
    12,
    120,
    172,
    0.1,
    38.0,
    12.0,
    COSTAS_MODIFIED,
);
/// Slow (Mode E): 30 s cycle, 3.125 baud.
pub const MODE_E: Mode = mode_const(
    4,
    "JS8E",
    3840,
    30,
    32,
    94,
    32,
    0.5,
    42.0,
    2.0,
    COSTAS_MODIFIED,
);

/// All user-selectable speeds (A, B, C, E) — multi-decode iterates these.
pub const MODES: [Mode; 4] = [MODE_A, MODE_B, MODE_C, MODE_E];

#[inline]
const fn mode_const(
    id: u8,
    name: &'static str,
    nsps: usize,
    ntxdur: usize,
    ndownsps: usize,
    ndd: usize,
    jz: usize,
    astart: f32,
    basesub: f32,
    az: f32,
    costas: [[usize; 7]; 3],
) -> Mode {
    let nmax = ntxdur * (SAMPLE_RATE as usize);
    let nfft1 = nsps * NFOS;
    let nstep = nsps / NSSY;
    let nhsym = nmax / nstep - 3;
    let ndown = nsps / ndownsps;
    let nqsymbol = ndownsps / 4;
    let ndfft1 = nsps * ndd;
    let ndfft2 = ndfft1 / ndown;
    let np2 = NN * ndownsps;
    let tstep = nstep as f32 / SAMPLE_RATE;
    let jstrt = (astart / tstep) as usize;
    let df = SAMPLE_RATE / nfft1 as f32;
    let baud = SAMPLE_RATE / nsps as f32;
    Mode {
        id,
        name,
        nsps,
        ntxdur,
        ndownsps,
        ndd,
        jz,
        astart,
        basesub,
        az,
        costas,
        nmax,
        nfft1,
        nstep,
        nhsym,
        ndown,
        nqsymbol,
        ndfft1,
        ndfft2,
        np2,
        tstep,
        jstrt,
        df,
        baud,
        cycle_ms: (ntxdur as u64) * 1_000,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mode {
    /// Submode id per `js8call/varicode.h` (A=0, C=1, B=2, E=4).
    pub id: u8,
    /// Short tag for logs / PSK Reporter (e.g. "JS8A", "JS8C", "JS8B", "JS8E").
    pub name: &'static str,
    /// 12 kHz samples per symbol (NSPS).
    pub nsps: usize,
    /// Transmit duration, seconds (NTXDUR).
    pub ntxdur: usize,
    /// Samples per symbol after downsampling (NDOWNSPS).
    pub ndownsps: usize,
    /// Taper length (NDD).
    pub ndd: usize,
    /// Coarse time-sync search half-range in quarter-symbol steps (JZ).
    pub jz: usize,
    /// Transmit start delay within the cycle, seconds (ASTART).
    pub astart: f32,
    /// Baseline-subtraction offset in the SNR term (BASESUB).
    pub basesub: f32,
    /// Near-duplicate candidate pruning window in Hz (AZ).
    pub az: f32,
    /// The Costas arrays this mode uses (ORIGINAL for A, MODIFIED for B/C/E).
    pub costas: [[usize; 7]; 3],

    // ────────────────────────── derived ──────────────────────────
    /// Cycle window length in 12 kHz samples (NTXDUR * 12 000).
    pub nmax: usize,
    /// Per-symbol spectrum FFT size (NSPS * NFOS).
    pub nfft1: usize,
    /// Quarter-symbol step in 12 kHz samples (NSPS / NSSY).
    pub nstep: usize,
    /// Symbol-spectra count for the cycle (NMAX / NSTEP - 3).
    pub nhsym: usize,
    /// Decimation factor (NSPS / NDOWNSPS).
    pub ndown: usize,
    /// Quarter-symbol index (NDOWNSPS / 4).
    pub nqsymbol: usize,
    /// Baseband FFT-1 size (NSPS * NDD).
    pub ndfft1: usize,
    /// Baseband FFT-2 size (NDFFT1 / NDOWN).
    pub ndfft2: usize,
    /// Downsampled decode window (NN * NDOWNSPS).
    pub np2: usize,
    /// Quarter-symbol time step in seconds (NSTEP / 12 000).
    pub tstep: f32,
    /// Start search offset in quarter-symbol steps (ASTART / TSTEP).
    pub jstrt: usize,
    /// Per-symbol-spectrum frequency resolution (12 000 / NFFT1).
    pub df: f32,
    /// Symbol rate (12 000 / NSPS) in baud.
    pub baud: f32,
    /// Cycle duration in wall-clock milliseconds (NTXDUR * 1000).
    pub cycle_ms: u64,
}

impl Mode {
    /// Look up a mode by its `id` (A=0, C=1, B=2, E=4), if any.
    pub fn by_id(id: u8) -> Option<&'static Mode> {
        MODES.iter().find(|m| m.id == id)
    }
}
