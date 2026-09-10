//! # Virtual receiver (audio channel demod)
//!
//! The software audio receiver for the HL2: takes the radio's per-slot
//! **complex I/Q baseband** and demodulates it to **audio** (`i16` mono).
//!
//! ## Pipeline
//!
//! ```text
//! BasebandSource            Demodulator                     AudioSink
//! (complex I/Q f32)  ──►   (bandpass + decimate)   ──►   (play / capture)
//!   e.g. VecSource            SsbDemodulator             VecSink | AlsaSink
//! ```
//!
//! The three ends are separate traits so each can be swapped independently:
//!
//! * `BasebandSource` — where the I/Q comes from (socket, file, synth).
//! * `Demodulator` — the mode DSP. New modes implement [`DemodCore`] and are
//!   added by extending [`Mode`] / [`make_demod`].
//! * `AudioSink` — where the audio goes. `VecSink` (capture) and
//!   `AlsaSink` (playback, `alsa` feature) are built-in.
//!
//! [`VirtualReceiver`] wires these together and runs the loop.
//!
//! ## Layering note
//!
//! This is DSP, not wire protocol: it may be used from `hl2-api` / `hl2-ui`
//! without touching the byte layout. The only seam that meets the wire is
//! the abstract (not socket-specific) `BasebandSource`.

/// Peak normalisation for the digital-mode decode windows (FT8/FT4/JS8).
#[cfg(any(feature = "ft8", feature = "ft4", feature = "js8"))]
pub mod audio_scale;
/// Auto-decode registry: which digital modes are present + their known freqs.
#[cfg(any(feature = "ft8", feature = "ft4", feature = "js8"))]
pub mod auto;
/// Bounded, drop-oldest complex I/Q ring the EP6 pump writes into (pump
/// plumbing). `no_std`-capable (only `alloc` + `num-complex`); the
/// multi-reader `Arc<Mutex<..>>` view is an additive `std`-only layer the
/// fan-out hands out.
#[cfg(feature = "dsp")]
pub mod baseband_ring;
pub mod demod;
/// Per-slot EP6 baseband fan-out. `no_std`-capable (owned core); the
/// `Arc<Mutex<..>>` multi-reader view is an additive `std`-only layer.
#[cfg(feature = "dsp")]
pub mod fanout;
/// FT4 slot decoder (`mfsk-core`). `ft4` feature.
#[cfg(feature = "ft4")]
pub mod ft4;
/// FT8 slot decoder (`mfsk-core`). `ft8` feature.
#[cfg(feature = "ft8")]
pub mod ft8;
/// JS8Call decoder (`rustfft`). `js8` feature.
#[cfg(feature = "js8")]
pub mod js8;
pub mod sink;
pub mod source;
/// WSJT-family spot (PSK Reporter) extraction — the shared FT8/FT4 free-text
/// grammar. (JS8 has its own varicode selector in `js8::decoder`.) Present
/// iff a WSJT mode (FT8 or FT4) is enabled, since those need `hl2-common`.
#[cfg(any(feature = "ft8", feature = "ft4"))]
pub mod spot;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Auto-decode registry + per-mode known-frequency tables (whichever modes are
/// enabled; the API iterates `AUTO_MODES` to build slot decoders).
#[cfg(any(feature = "ft8", feature = "ft4", feature = "js8"))]
pub use auto::{AUTO_MODES, AutoMode};
#[cfg(feature = "dsp")]
pub use baseband_ring::{BASEBAND_RING_CAP, BasebandRing};
pub use demod::AudioEngine;
pub use demod::{
    AmCore, AmDemodulator, DemodCore, Demodulator, F32Fir, F32FirState, IqBlock, Nco,
    PolyphaseDecimator, RawSampleTap, SsbCore, SsbDemodulator, StandardDemod, make_demod,
    make_demod_tap,
};
/// Digital-mode core (FT8/FT4/JS8 share one USB/12 kHz core), present only when
/// at least one digital mode is enabled.
#[cfg(any(feature = "ft8", feature = "ft4", feature = "js8"))]
pub use demod::{DigitalCore, DigitalDemodulator};
#[cfg(feature = "dsp")]
pub use fanout::BasebandFanout;
#[cfg(feature = "ft4")]
pub use ft4::{
    FT4_SAMPLE_RATE_HZ, FT4_SLOT_MS, FT4_SLOT_WINDOW_SAMPLES, Ft4Decoder, Ft4Message, Ft4Tap,
    closed_slot_for as ft4_closed_slot_for, decode_closed_slot as ft4_decode_closed_slot,
    shared as ft4_shared,
};
#[cfg(feature = "ft8")]
pub use ft8::{
    FT8_SAMPLE_RATE_HZ, FT8_SLOT_MS, FT8_SLOT_WINDOW_SAMPLES, Ft8Decoder, Ft8Message, Ft8Tap,
    SharedDecoder, closed_slot_for, decode_closed_slot, shared,
};
#[cfg(feature = "js8")]
pub use js8::decoder::{
    JS8_BUFFER_CAP, JS8_SAMPLE_RATE_HZ, JS8_SAMPLES_PER_SEC, Js8Decoder, Js8Message, Js8Tap,
    js8_step, shared as js8_shared,
};
/// Shared FT4 decoder handle ([`Arc<Mutex<Ft4Decoder>>`]), held by both the
/// demod tap and the API's decode task (cf. [`SharedDecoder`] for FT8).
#[cfg(feature = "ft4")]
pub type Ft4SharedDecoder = ft4::SharedDecoder;
/// Shared JS8 decoder handle ([`Arc<Mutex<Js8Decoder>>`]), held by both the
/// demod tap and the API's decode task (cf. [`SharedDecoder`] for FT8).
#[cfg(feature = "js8")]
pub type Js8SharedDecoder = js8::decoder::SharedDecoder;
#[cfg(feature = "alsa")]
pub use sink::AlsaSink;
pub use sink::{AudioSink, DropSink, SinkError, VecSink};
/// Thread-safe `BufSink` / `BufSinkHandle` (the shared buffer the pump +
/// demod + API share) live in the `std` world.
#[cfg(feature = "std")]
pub use sink::{BUF_SINK_DEFAULT_CAP, BufSink, BufSinkHandle};
pub use source::{BasebandSource, VecSource};

/// A receiver-mode-specific error.
pub type ReceiverError = Box<dyn core::error::Error + Send + Sync>;

/// SSB sideband.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sideband {
    /// Upper sideband — passband above the carrier.
    Usb,
    /// Lower sideband — passband below the carrier.
    Lsb,
}

/// The modulation / decode mode of a virtual receiver.
///
/// Adding a mode: extend this enum, add a [`Demodulator`] impl, and a branch
/// in [`make_demod`]. `Ssb`'s passband is set by its channel-select
/// `bandwidth_hz` override, so a wide SSB passband needs no dedicated mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// AM (DSB-FC, full-carrier) voice. Both sidebands pass; the carrier DC
    /// is removed by the AGC DC-block step. See [`am::AmDemodulator`].
    Am,
    /// FM (standard, ≈ 15 kHz channel) voice demod. Audio is recovered as
    /// the phase derivative of the (LPF'd) complex baseband — see
    /// [`FmCore`](crate::receiver::demod::FmCore). Distinct from
    /// [`Mode::FmNarrow`] only by the default channel-select bandwidth.
    Fm,
    /// NFM (narrow, ≈ 5 kHz channel) voice demod — the same phase-derivative
    /// DSP as [`Mode::Fm`] with a narrower channel-select bandwidth.
    /// Amateur-radio "NFM".
    FmNarrow,
    /// Single-sideband voice, with the selected sideband. The passband is
    /// the mode's `bandwidth_hz` (a wide passband is just a large override —
    /// no dedicated wide mode needed).
    Ssb(Sideband),
    /// FT8: USB SSB audio at 12 kHz, decoded externally in `hl2-api`
    /// ([`hl2::receiver::Ft8Decoder`]) on a wall-clock-aligned 15 s slot.
    /// Mode is SSB/USB at a 12 kHz output rate; the flag tells the API layer
    /// what rate to drive and when to decode. Slot details in PROTOCOL.md.
    /// Requires the `ft8` feature.
    #[cfg(feature = "ft8")]
    Ft8,
    /// JS8Call (Mode A): same USB/12 kHz pipeline as [`Mode::Ft8`], decoded
    /// in `hl2-api` ([`Js8Decoder`]) on a 15 s slot. Requires the `js8` feature.
    #[cfg(feature = "js8")]
    Js8,
    /// FT4: same USB/12 kHz pipeline as [`Mode::Ft8`], decoded in `hl2-api`
    /// ([`Ft4Decoder`]) on a 7.5 s slot. Requires the `ft4` feature.
    #[cfg(feature = "ft4")]
    Ft4,
}

impl Mode {
    /// The default channel-select bandwidth for this mode (`Hz`).
    ///
    /// SSB / voice / digital modes use typical voice bands (≈ 2.6–15 kHz); a
    /// wide passband is set by a `bandwidth_hz` override on
    /// [`ReceiverConfig`].
    pub fn default_bandwidth_hz(&self) -> u32 {
        match self {
            Mode::Am => 8_000,
            Mode::Fm => 15_000,
            Mode::FmNarrow => 5_000,
            Mode::Ssb(_) => 2_600,
            #[cfg(feature = "ft8")]
            Mode::Ft8 => 2_600,
            #[cfg(feature = "js8")]
            Mode::Js8 => 2_600,
            #[cfg(feature = "ft4")]
            Mode::Ft4 => 2_600,
        }
    }

    /// The sideband, if the mode is SSB. `Ft8` uses USB (its digital
    /// passband is in the upper sideband), so returns `Some(Usb)` for
    /// `Ft8`.
    pub fn sideband(&self) -> Option<Sideband> {
        match self {
            Mode::Am => None,
            Mode::Fm => None,
            Mode::FmNarrow => None,
            Mode::Ssb(s) => Some(*s),
            #[cfg(feature = "ft8")]
            Mode::Ft8 => Some(Sideband::Usb),
            #[cfg(feature = "js8")]
            Mode::Js8 => Some(Sideband::Usb),
            #[cfg(feature = "ft4")]
            Mode::Ft4 => Some(Sideband::Usb),
        }
    }
}

/// Audio output parameters (mono).
#[derive(Debug, Clone, Copy)]
pub struct AudioConfig {
    /// Decimated audio sample rate (Hz). Typically 4 800.
    pub rate_hz: u32,
    /// Playback gain, in dB (applied before normalisation).
    pub gain_db: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            rate_hz: 4_800,
            gain_db: 0.0,
        }
    }
}

/// Configuration for a [`VirtualReceiver`].
#[derive(Debug, Clone)]
pub struct ReceiverConfig {
    /// The demod mode.
    pub mode: Mode,
    /// The complex baseband sample rate of the source (`Hz`). Must be at
    /// least 4× the audio rate (polyphase antialiasing sanity bound).
    pub source_rate_hz: u32,
    /// Offset of the signal carrier from the source centre, in `Hz`, before
    /// the channel filter is applied. `0.0` for a baseband source.
    pub source_center_hz: f64,
    /// Override for the channel-select bandwidth. `None` uses
    /// [`Mode::default_bandwidth_hz`].
    pub bandwidth_hz: Option<u32>,
    /// Audio output config.
    pub audio: AudioConfig,
    /// Optional pre-AGC raw-sample tap for FT8/JS8/FT4 decode (see
    /// [`RawSampleTap`]). `None` for plain SSB / AM / FM.
    pub tap: Option<Arc<dyn RawSampleTap>>,
}

impl Default for ReceiverConfig {
    fn default() -> Self {
        Self {
            mode: Mode::Ssb(Sideband::Usb),
            // Default complex I/Q wire rate: 96 kSps — the `SPEED_192K` option's
            // Nyquist rate (each complex pair carries two real samples). See
            // PROTOCOL.md §16.1: the DSP's source_rate_hz = C1 option / 2.
            source_rate_hz: 96_000,
            source_center_hz: 0.0,
            bandwidth_hz: None,
            audio: AudioConfig::default(),
            tap: None,
        }
    }
}

impl ReceiverConfig {
    /// The receiver's effective channel bandwidth (`Hz`).
    ///
    /// `bandwidth_hz` wins if set, otherwise [`Mode::default_bandwidth_hz`].
    pub fn bandwidth(&self) -> u32 {
        self.bandwidth_hz
            .unwrap_or_else(|| self.mode.default_bandwidth_hz())
    }
}

/// A virtual audio receiver: a configured demodulator + a sink.
///
/// Construct with a sink, then feed it I/Q either block-by-block
/// ([`process`]) or by streaming an entire [`BasebandSource`] (`run`).
pub struct VirtualReceiver {
    cfg: ReceiverConfig,
    demod: Box<dyn Demodulator>,
    sink: Box<dyn AudioSink>,
    frames: AtomicUsize,
}

impl VirtualReceiver {
    /// Create a receiver with the given config and sink.
    pub fn new(cfg: ReceiverConfig, sink: Box<dyn AudioSink>) -> Result<Self, ReceiverError> {
        let mode = cfg.mode;
        let tap = cfg.tap.clone();
        let demod = make_demod_tap(
            mode,
            cfg.source_rate_hz,
            cfg.source_center_hz,
            cfg.bandwidth(),
            cfg.audio,
            tap,
        )?;
        Ok(Self {
            cfg,
            demod,
            sink,
            frames: AtomicUsize::new(0),
        })
    }

    /// The receiver's configuration.
    pub fn config(&self) -> &ReceiverConfig {
        &self.cfg
    }

    /// The audio format the receiver produces.
    pub fn audio_format(&self) -> AudioConfig {
        self.demod.audio_format()
    }

    /// Demodulate one complex I/Q block into the receiver's sink.
    ///
    /// Returns the number of audio frames written.
    pub fn process(&mut self, iq: &IqBlock) -> Result<usize, ReceiverError> {
        let n = self.demod.demod(iq, &mut *self.sink)?;
        self.frames.fetch_add(n, Ordering::Relaxed);
        Ok(n)
    }

    /// Flush any buffered audio: the demodulator's residual partial block
    /// (its AGC/DC-block accumulator) and then the sink itself.
    pub fn flush(&mut self) -> Result<(), ReceiverError> {
        self.demod.flush(&mut *self.sink)?;
        self.sink.flush()
    }

    /// Stream an entire [`BasebandSource`] through the demodulator to the
    /// sink until the source is exhausted, returning total audio frames.
    ///
    /// The source's `rate_hz` must match `cfg.source_rate_hz` for the
    /// demodulation to be correct (it is not re-read here).
    pub fn run(&mut self, source: &mut dyn BasebandSource) -> Result<usize, ReceiverError> {
        // Allocate a buffer sized for a full decimation period + margin so a
        // source that yields one big block at a time (e.g. `VecSource` in
        // tests) is not truncated before the demod can emit a full
        // `AUDIO_EMIN` block. 16384 samples is ~85 ms at 192 kSps.
        let mut buf: IqBlock = vec![num_complex::Complex::new(0.0, 0.0); 16384];
        let mut total = 0usize;
        loop {
            let n = source.next_block(&mut buf);
            if n == 0 {
                break;
            }
            let iq = buf[..n].to_vec();
            total += self.process(&iq)?;
        }
        Ok(total)
    }

    /// Total audio frames produced by this receiver so far.
    pub fn frames_processed(&self) -> usize {
        self.frames.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn tone(rate_hz: u32, freq_hz: f64, n_pairs: usize, amp: f32) -> IqBlock {
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
    fn virtual_receiver_usb_produces_audio_from_source() {
        let rate = 192_000u32;
        let mut source = VecSource::new(rate);
        // 1.5 kHz USB tone, a few ms worth.
        let iq = tone(rate, 1_500.0, 12000, 0.5);
        source.push(iq);

        let mut rx = VirtualReceiver::new(
            ReceiverConfig {
                source_rate_hz: rate,
                ..Default::default()
            },
            Box::new(VecSink::new()),
        )
        .expect("receiver build");

        let frames = rx.run(&mut source).expect("run");
        assert!(frames > 0, "expected audio frames, got {frames}");
        assert_eq!(rx.frames_processed(), frames);
        assert_eq!(source.queued_blocks(), 0, "source should be drained");
    }

    #[test]
    fn virtual_receiver_sidebands_and_modes_build() {
        let cfg = |m: Mode| ReceiverConfig {
            mode: m,
            source_rate_hz: 192_000,
            ..Default::default()
        };
        for m in [
            Mode::Ssb(Sideband::Usb),
            Mode::Ssb(Sideband::Lsb),
            Mode::Fm,
            Mode::FmNarrow,
        ] {
            VirtualReceiver::new(cfg(m), Box::new(VecSink::new())).expect("receiver build");
        }
    }

    #[test]
    fn virtual_receiver_preserves_sink_errors() {
        struct FailingSink;
        impl AudioSink for FailingSink {
            fn write(&mut self, _: &[i16]) -> Result<usize, SinkError> {
                Err(std::io::Error::other("write failed").into())
            }

            fn flush(&mut self) -> Result<(), SinkError> {
                Err(std::io::Error::other("flush failed").into())
            }
        }

        let mut rx = VirtualReceiver::new(ReceiverConfig::default(), Box::new(FailingSink))
            .expect("receiver build");
        let err = rx.flush().unwrap_err();
        assert!(err.downcast_ref::<std::io::Error>().is_some());
        assert_eq!(alloc::format!("{err}"), "flush failed");

        let iq = tone(rx.config().source_rate_hz, 1_500.0, 12000, 0.5);
        let err = rx.process(&iq).unwrap_err();
        assert!(err.downcast_ref::<std::io::Error>().is_some());
        assert_eq!(alloc::format!("{err}"), "write failed");
    }
}
