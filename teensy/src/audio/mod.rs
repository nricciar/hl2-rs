//! I2S audio output: demod audio (4.8 kHz `i16` mono) → WM8731 over SAI1.
//!
//! Pieces:
//!
//!   * [`upsample`] — the 10× 4.8 kHz → 48 kHz resampler.
//!   * [`ring`]     — the cross-task FIFO the radio *producer* and audio
//!     *consumer* tasks share.
//!   * [`sai1`]     — SAI1 slave-TX bring-up (I2S 16-bit, L = R mono fold).
//!   * [`wm8731`]   — WM8731 I2C register init (the codec is I2S *master*).
//!   * [`sink`]     — the [`hl2::receiver::sink::AudioSink`] impl + the
//!     eDMA-to-TDR tick the audio task runs.
//!
//! The radio task owns a [`sink::Sink`] and hands it to its
//! `VirtualReceiver`; the audio task owns the eDMA `Channel` + SAI `Tx` and
//! calls [`sink::process_chunk`] every millisecond. Everything between the
//! two rides on the cross-task [`ring::RING`].

pub mod ring;
pub mod sai1;
pub mod sink;
pub mod upsample;
pub mod wm8731;
