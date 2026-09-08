//! The [`DemodCore`] trait: the **one contract** a mode's DSP must satisfy
//! before it can be wrapped into a full [`Demodulator`].
//!
//! A core owns only the mode-specific per-sample DSP — NCO multiplication,
//! sideband/envelope/phase discrimination, the polyphase anti-alias/decimate
//! stage. It yields zero, one, or many decimated `f32` samples per complex
//! input and must remember at most one pending sample between calls (drained
//! by [`flush`](Self::flush)). It owns **none** of the audio tail: the
//! DC-block + RMS-targeted AGC + pre-AGC [`RawSampleTap`] + `i16` sink write
//! — that lives in [`AudioEngine`](super::engine::AudioEngine) and is shared
//! by [`StandardDemod`] (this module).
//!
//! ```text
//! complex I/Q ──► DemodCore::process ──► 0..=N f32s per block ──► AudioEngine ──► i16 → sink
//!                     (mode DSP)                                    (AGC, tap)
//! ```
//!
//! To add a mode: implement `DemodCore` (≤ 30 lines — reuse
//! [`Nco`](super::dsp::Nco) + [`PolyphaseDecimator`](super::dsp::PolyphaseDecimator)
//! + a mode-specific discriminator), add the `Mode` enum variant and a
//! dispatch arm in `demod/mod.rs`, and that's the whole change.

use alloc::boxed::Box;
use alloc::sync::Arc;
use num_complex::Complex;

use super::engine::AudioEngine;
use super::{DemodError, Demodulator, IqBlock, RawSampleTap};
use crate::receiver::AudioConfig;
use crate::receiver::sink::AudioSink;

/// A mode's per-sample DSP (see module docs).
///
/// `Send` (required so a [`StandardDemod<C>`] can live on the demod thread
/// alongside a [`crate::receiver::VirtualReceiver`]).
pub trait DemodCore: Send + 'static {
    /// Consume one complex input sample; produce one decimated `f32` output
    /// if a full decimator group is complete, else `None`. For the SSB
    /// voice/digital path this is the polyphase branch boundary (one in every
    /// `M = rate_in / rate_out` inputs). For the AM envelope discriminator,
    /// this is the I/Q magnitude at the boundary. Cores that need to emit a
    /// *pending* sample on a later input call [`flush`](Self::flush).
    fn process(&mut self, x: Complex<f32>) -> Option<f32>;

    /// Drain a *pending* sample accumulated by a previous `process` call (no-op
    /// by default; the AM envelope discriminator overrides this). Called once
    /// per input block in [`StandardDemod`]'s `demod`, and once more at
    /// shutdown.
    fn flush(&mut self) -> Option<f32> {
        None
    }

    /// A short, human-readable identifier for this mode's DSP — used in the
    /// HL2_DEBUG emit trace (`[aud] {kind} in_rms=…`). Defaults to the
    /// type name (e.g. `SsbCore` → `"SsbCore"`); override per-mode when you
    /// want `"ssb-usb"` etc.
    fn kind(&self) -> &'static str {
        // `core::any::type_name` is not (yet) stabilised for `no_std`; the
        // `std`-gated fallback below keeps the nice type-name in `std` builds
        // while `no_std` consumers get a stable `"DemodCore"` label (per-mode
        // cores override it with a fixed string anyway).
        #[cfg(feature = "std")]
        {
            std::any::type_name::<Self>()
                .rsplit("::")
                .next()
                .unwrap_or("DemodCore")
        }
        #[cfg(not(feature = "std"))]
        {
            "DemodCore"
        }
    }

    /// Compose this core with the shared audio tail into a full [`Demodulator`].
    ///
    /// `audio` carries the decimated output rate + playback gain; `tap` is
    /// the optional pre-AGC raw-sample seam (the FT8/JS8/FT4 decoders hang
    /// off it in the same frame). This is the **one call site** that turns
    /// a mode's core into a ready-to-use demodulator; new modes do not need a
    /// separate `Demodulator` impl.
    fn demodulator(
        self,
        audio: AudioConfig,
        tap: Option<Arc<dyn RawSampleTap>>,
    ) -> Box<dyn Demodulator>
    where
        Self: Sized,
    {
        let kind = self.kind();
        Box::new(StandardDemod::<Self> {
            core: self,
            engine: AudioEngine::new(audio.rate_hz, audio.gain_db, kind).with_tap(tap),
        })
    }
}

/// The "core + shared audio tail → full [`Demodulator`]" wrapper. This is
/// the only `Demodulator` impl in the crate — new modes add a `DemodCore`
/// impl and a single dispatch arm, not a new `Demodulator` impl.
pub struct StandardDemod<C: DemodCore> {
    /// The mode-specific DSP.
    core: C,
    /// The shared audio tail (AGC, DC-block, pre-AGC tap, sink
    /// write). See [`AudioEngine`].
    engine: AudioEngine,
}

impl<C: DemodCore> Demodulator for StandardDemod<C> {
    fn demod(&mut self, iq: &IqBlock, sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        for &x in iq {
            if let Some(out) = self.core.process(x) {
                self.engine.push(out);
            } else if let Some(out) = self.core.flush() {
                self.engine.push(out);
            }
        }
        self.engine.emit_full_blocks(sink)
    }

    fn audio_format(&self) -> AudioConfig {
        self.engine.audio_format()
    }

    fn flush_audio(&mut self, sink: &mut dyn AudioSink) -> Result<usize, DemodError> {
        // Two steps: drain any pending sample the core still has (the AM
        // envelope path may have one), then drain the audio tail's residual
        // partial block. Order matters: the core's pending sample belongs to
        // the current block, so it is normalised in the same `emit` window
        // as the tail's residue.
        let mut written = 0usize;
        if let Some(out) = self.core.flush() {
            self.engine.push(out);
        }
        written += self.engine.flush_residue(sink)?;
        Ok(written)
    }
}
