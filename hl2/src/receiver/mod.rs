//! # Virtual receiver (audio channel demod)
//!
//! This module is the software audio receiver for the HL2: it takes the
//! radio's per-slot **complex I/Q baseband** and demodulates it to **audio**
//! (`i16` mono), for SSB (USB/LSB) today and (AM, FM, FT8, CW…) later.
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
//! * `BasebandSource` — where the I/Q comes from (socket, file, synth). The
//!   production socket-backed source is a follow-up (PROTOCOL.md §16).
//! * `Demodulator` — the mode DSP. `SsbDemodulator` is built-in; new modes
//!   are added by implementing this trait and extending [`Mode`] /
//!   [`make_demod`].
//! * `AudioSink` — where the audio goes. `VecSink` (capture) and
//!   `AlsaSink` (playback, `alsa` feature) are built-in.
//!
//! [`VirtualReceiver`] wires these together and runs the loop.
//!
//! ## Layering note
//!
//! This is DSP, not wire protocol, so unlike the `protocol` module it may be
//! used from `hl2-api` / `hl2-ui` without touching the byte layout. The
//! layering rule still applies: nothing here writes or reads protocol bytes
//! — the incoming `BasebandSource` is the only seam that meets the wire
//! (and it is abstract, not socket-specific).

pub mod audio_scale;
pub mod auto;
pub mod baseband_ring;
pub mod demod;
pub mod fanout;
pub mod ft4;
pub mod ft8;
pub mod js8;
pub mod sink;
pub mod source;
pub mod spot;

use std::sync::atomic::{AtomicUsize, Ordering};

pub use auto::{AUTO_MODES, AutoMode};
pub use baseband_ring::{BASEBAND_RING_CAP, BasebandRing};
pub use demod::AudioEngine;
pub use demod::{
    AmCore, AmDemodulator, DemodCore, Demodulator, DigitalCore, DigitalDemodulator, F32Fir,
    F32FirState, IqBlock, Nco, PolyphaseDecimator, RawSampleTap, SsbCore, SsbDemodulator,
    StandardDemod, make_demod, make_demod_tap,
};
pub use fanout::BasebandFanout;
pub use ft4::{
    FT4_SAMPLE_RATE_HZ, FT4_SLOT_MS, FT4_SLOT_WINDOW_SAMPLES, Ft4Decoder, Ft4Message, Ft4Tap,
    closed_slot_for as ft4_closed_slot_for, decode_closed_slot as ft4_decode_closed_slot,
    shared as ft4_shared,
};
pub use ft8::{
    FT8_SAMPLE_RATE_HZ, FT8_SLOT_MS, FT8_SLOT_WINDOW_SAMPLES, Ft8Decoder, Ft8Message, Ft8Tap,
    SharedDecoder, closed_slot_for, decode_closed_slot, shared,
};
pub use js8::decoder::{
    JS8_BUFFER_CAP, JS8_SAMPLE_RATE_HZ, JS8_SAMPLES_PER_SEC, Js8Decoder, Js8Message, Js8Tap,
    js8_step, shared as js8_shared,
};
/// Shared JS8 decoder handle ([`Arc<Mutex<Js8Decoder>>`]), the type both the
/// demod tap and the API's decode task hold (cf. [`SharedDecoder`] for FT8).
pub type Js8SharedDecoder = js8::decoder::SharedDecoder;
/// Shared FT4 decoder handle ([`Arc<Mutex<Ft4Decoder>>`]), the type both the
/// demod tap and the API's decode task hold (cf. [`SharedDecoder`] for FT8).
pub type Ft4SharedDecoder = ft4::SharedDecoder;
#[cfg(feature = "alsa")]
pub use sink::AlsaSink;
pub use sink::{
    AudioSink, BUF_SINK_DEFAULT_CAP, BufSink, BufSinkHandle, DropSink, SinkError, VecSink,
};
pub use source::{BasebandSource, VecSource};

/// A receiver-mode-specific error.
pub type ReceiverError = Box<dyn std::error::Error + Send + Sync>;

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
/// Adding a mode is: extend this enum, add a `Demodulator`, and a branch in
/// [`make_demod`]. `SsbWide` is a placeholder for "pass the wideband complex
/// through unfiltered" (FT8/other digital work sits here — see the
/// `BasebandTap` idea in PROTOCOL.md §16.3).
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
    /// Single-sideband voice, with the selected sideband.
    Ssb(Sideband),
    /// Wideband complex pass-through (digital-mode placeholder).
    SsbWide,
    /// FT8: USB SSB audio at 12 kHz, plus an external FT8 decode pass (in
    /// `hl2-api`, via [`hl2::receiver::Ft8Decoder`]). The demodulator itself
    /// is the SSB/USB pipeline at a 12 kHz output rate; `Ft8` is a *mode
    /// flag* that tells the API layer (a) to drive the demod at 12 kHz, and
    /// (b) to spin up a wall-clock-aligned 15-second slot decode task.
    ///
    /// See the FT8 section in PROTOCOL.md for the slot / wall-clock
    /// alignment details, and [`hl2::receiver::Ft8Decoder`] for the
    /// decoder itself.
    Ft8,
    /// JS8Call (Mode A): USB SSB audio at 12 kHz, plus an external
    /// JS8 decode pass (in `hl2-api`, via
    /// [`Js8Decoder`]). The demodulator is the same USB/12 kHz pipeline as
    /// [`Mode::Ft8`]; `Js8` is a *mode flag* that tells the API layer to
    /// spawn a wall-clock-aligned 15-second slot decode task. See the
    /// JS8Call section in PROTOCOL.md and [`Js8Decoder`] for details.
    Js8,
    /// FT4: USB SSB audio at 12 kHz, plus an external FT4 decode pass (in
    /// `hl2-api`, via [`Ft4Decoder`]). The demodulator is the same USB/12 kHz
    /// pipeline as [`Mode::Ft8`]; `Ft4` is a *mode flag* that tells the API
    /// layer (a) to drive the demod at 12 kHz, and (b) to spin up a
    /// wall-clock-aligned 7.5-second slot decode task.
    ///
    /// See the FT4 section in PROTOCOL.md for the slot / wall-clock
    /// alignment details, and [`Ft4Decoder`] for the decoder itself.
    Ft4,
}

impl Mode {
    /// The default channel-select bandwidth for this mode (`Hz`).
    ///
    /// SSB uses a typical voice band (≈ 2.6 kHz); wide pass-through uses half
    /// the source rate (Nyquist). A [`ReceiverConfig`] may override this with
    /// `bandwidth_hz`.
    pub fn default_bandwidth_hz(&self) -> u32 {
        match self {
            Mode::Am => 8_000,
            Mode::Fm => 15_000,
            Mode::FmNarrow => 5_000,
            Mode::Ssb(_) => 2_600,
            Mode::SsbWide => 2_400_000,
            Mode::Ft8 => 2_600,
            Mode::Js8 => 2_600,
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
            Mode::SsbWide => None,
            Mode::Ft8 => Some(Sideband::Usb),
            Mode::Js8 => Some(Sideband::Usb),
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
    /// Optional pre-AGC raw-sample tap (FT8/JS8/FT4 decode — see
    /// [`RawSampleTap`]). `None` for plain SSB / AM / FM. The S-meter is
    /// *not* part of this seam — it is computed in the API layer from the
    /// displayed slot's band spectrum (see `api/src/meter.rs` and
    /// PROTOCOL.md §16.3e), so this crate carries no level bookkeeping of
    /// its own.
    pub tap: Option<std::sync::Arc<dyn RawSampleTap>>,
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
    fn virtual_receiver_lsb_and_wide_build() {
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
            Mode::SsbWide,
        ] {
            VirtualReceiver::new(cfg(m), Box::new(VecSink::new())).expect("receiver build");
        }
    }
}
