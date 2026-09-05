//! The [`DemodCore`] trait: the **one contract** a mode's DSP must satisfy
//! before it can be wrapped into a full [`Demodulator`].
//!
//! A core owns only the mode-specific per-sample DSP — NCO multiplication,
//! sideband/envelope/phase discrimination, the polyphase anti-alias/decimate
//! stage. It yields **zero, one, or many** decimated `f32` samples per
//! complex input and must remember at most one pending sample between calls
//! (drained by [`flush`](Self::flush)). It owns **none** of the audio tail:
//! the DC-block + RMS-targeted AGC + pre-AGC [`RawSampleTap`] + S-meter +
//! `i16` sink write — that lives in [`AudioEngine`](super::engine::AudioEngine)
//! and is shared with every mode by [`StandardDemod`] (this module).
//!
//! ```text
//! complex I/Q ──► DemodCore::process ──► 0..=N f32s per block ──► AudioEngine ──► i16 → sink
//!                     (mode DSP)                                    (AGC, tap, meter)
//! ```
//!
//! This is the extension point. To add FM/NFM/CW: implement `DemodCore`
//! (≤ 30 lines — reuse [`Nco`](super::dsp::Nco) +
//! [`PolyphaseDecimator`](super::dsp::PolyphaseDecimator) + a mode-specific
//! discriminator), then add the `Mode` enum variant + a registry row
//! (`demod/mod.rs`). Nothing else changes.
//!
//! The default [`demodulator`](Self::demodulator) builds the audio tail (with
//! the engine's default AGC seed of 1000.0, matching the old per-mode initial
//! gains) and returns a `Box<dyn Demodulator>` — so `VirtualReceiver`,
//! `hl2-api`, the CLI and the existing unit tests continue to consume the
//! modes through the stable `Box<dyn Demodulator>` API.

use num_complex::Complex;

use super::engine::AudioEngine;
use super::{DemodError, Demodulator, IqBlock, RawSampleTap};
use crate::receiver::sink::AudioSink;
use crate::receiver::{AudioConfig, MeterHandle};

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
        std::any::type_name::<Self>()
            .rsplit("::")
            .next()
            .unwrap_or("DemodCore")
    }

    /// Compose this core with the shared audio tail into a full [`Demodulator`].
    ///
    /// `audio` carries the decimated output rate + playback gain; `tap` and
    /// `meter` are the optional pre-AGC raw-sample and signal-level seams.
    /// This is the **one call site** that turns a mode's core into a ready-to-
    /// use demodulator; new modes do not need a separate `Demodulator` impl.
    fn demodulator(
        self,
        audio: AudioConfig,
        tap: Option<std::sync::Arc<dyn RawSampleTap>>,
        meter: Option<MeterHandle>,
    ) -> Box<dyn Demodulator>
    where
        Self: Sized,
    {
        let kind = self.kind();
        Box::new(StandardDemod::<Self> {
            core: self,
            engine: AudioEngine::new(audio.rate_hz, audio.gain_db, kind)
                .with_tap(tap)
                .with_meter(meter),
        })
    }
}

/// The generic "core + engine → full [`Demodulator`]" wrapper. This is the
/// *only* `Demodulator` impl in the crate (the three concrete per-mode impls
/// — in `ssb.rs` / `am.rs` / `digital.rs` — are gone after the refactor).
/// New modes do not need a new `Demodulator` impl — they need a new
/// `DemodCore` impl and a single line in the dispatch.
pub struct StandardDemod<C: DemodCore> {
    /// The mode-specific DSP.
    core: C,
    /// The shared audio tail (AGC, DC-block, pre-AGC tap, S-meter, sink
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
        // partial block. Order matters: `flush()`'s pending sample belongs to
        // the current block, so it is normalised in the same `emit` window as
        // the tail's residue (matching the old `flush_residue` semantics).
        let mut written = 0usize;
        if let Some(out) = self.core.flush() {
            self.engine.push(out);
        }
        written += self.engine.flush_residue(sink)?;
        Ok(written)
    }
}
