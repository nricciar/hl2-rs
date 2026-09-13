//! The audio sink: demod → upsample → ring → SAI1-TDR eDMA.
//!
//! Wiring:
//!
//! ```text
//!   demod (hl2)  ──sink.write(samples: &[i16])──►  [`Sink::write`]
//!      4.8 kHz i16 mono                                    │
//!                                                           ▼
//!                                                     (Upsampler) 10× linear
//!                                                           │
//!                                                           ▼
//!                                                     [`crate::audio::ring::push`]
//!                                                     cross-task FIFO
//!                                                           │
//!                                                           ▼
//!   audio_task  ◄────  DMA-completion paced  ◄──────────┘
//!      pop up to 960 samples (20 ms of audio at 48 kHz)
//!        → repack into 1920 SAI TDR `u32` words (L = R, interleaved)
//!        → eDMA ch1 → SAI1.TDR[0]   (await completion, ~20 ms)
//!        → SAI slave clocks them out to the WM8731 on the I2S bus
//!
//! The [`Sink`] is task-local to the *radio* task (the `AudioSink` impl
//! the `VirtualReceiver` drives). The *audio* task owns the eDMA `Channel`
//! + SAI `Tx` and runs [`process_chunk`] as each DMA transfer completes.
//! The two tasks coordinate through a critical-section-protected ring.
//!
//! SAI + eDMA bring-up (the pin muxing, `init_tx`, and the `Channel
//! reset`) happens once in `main.rs` at init; the audio task just hands
//! its two handles to [`process_chunk`] each tick.

use alloc::boxed::Box;
use teensy4_bsp::hal::dma::channel::Channel;
use teensy4_bsp::hal::dma::peripheral;

use crate::audio::ring as r;
use crate::audio::sai1::TDR_WORDS_PER_SAMPLE;
use crate::audio::upsample::{RATE_RATIO, Upsampler};
use hl2::receiver::sink::{AudioSink, SinkError};

/// Pack one mono `i16` sample into the SAI TDR `u32` required by the
/// 32-bit slot / MSB-first / `Packing::None` config. The codec consumes
/// the upper 16 bits; process_chunk writes this word twice for L = R.
#[inline]
fn pack(sample: i16) -> u32 {
    let w = sample as u16;
    (w as u32) << 16
}

/// Upsample local capacity. The demod caps `sink.write` blocks at
/// 1024 samples (see `hl2::receiver::demod::AudioEngine`'s `out_buf`),
/// so `1024 * RATE_RATIO` upsampled `i16`s covers the maximum block the
/// virtual receiver can hand us.
const UP_OUT_CAP: usize = 1024 * RATE_RATIO;

/// The `AudioSink` impl the radio task hands to its `VirtualReceiver`.
///
/// Owns the stateful 10× upsample step and a pre-sized upsample buffer (so
/// every `write` runs allocation-free). The cross-task ring, eDMA `Channel`,
/// and the SAI `Tx` live in the audio task — the two halves talk over
/// [`ring::RING`](crate::audio::ring::RING).
pub struct Sink {
    up: Upsampler,
    up_out: Box<[i16]>,
}

impl Sink {
    pub fn new() -> Self {
        Self {
            up: Upsampler::new(),
            up_out: alloc::vec![0i16; UP_OUT_CAP].into_boxed_slice(),
        }
    }
}

impl core::fmt::Debug for Sink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Sink")
            .field("up", &self.up)
            .finish_non_exhaustive()
    }
}

impl AudioSink for Sink {
    /// Upsample `samples` (≤ 1024) into `up_out` and push the result into
    /// the cross-task ring. The ring drops its oldest samples on overflow,
    /// so this is a bounded, non-blocking call.
    fn write(&mut self, samples: &[i16]) -> Result<usize, SinkError> {
        let n = self.up.upsample(samples, &mut self.up_out);
        r::push(&self.up_out[..n]);

        // Diagnostic: the demod's first ~30 writes tell us whether the
        // virtual receiver is actually producing audio. A peak of 0 or a
        // single 0/1/2 value on every write means the AGC hasn't locked or
        // the demod output is ~zero. A healthy USB receiver sees `max`
        // climbing to thousands within a few hundred blocks as AGC engages.
        // Count every write; retain a sparse heartbeat beyond startup.
        crate::shared::audio_writes_bump();
        let writes = crate::shared::audio_writes();
        if writes <= 30 || writes % 200 == 0 {
            let peak = samples
                .iter()
                .map(|s| *s as i32)
                .fold(0i32, |a, b| a.max(b.abs()));
            let non_zero = samples.iter().filter(|s| **s != 0).count();
            log::info!(
                "audio sink #{}: in_{} peak={} non_zero={}/{}",
                crate::shared::audio_writes(),
                samples.len(),
                peak,
                non_zero,
                samples.len()
            );
        }
        Ok(samples.len())
    }

    fn flush(&mut self) -> Result<(), SinkError> {
        Ok(())
    }
}

// `Sink` holds one `i16` (the upsample `prev`) + a `Box<[i16]>` — no `Cell`
// / `RefCell` / `Rc` / pointers. Rust auto-implements `Send` when all fields
// are `Send`, so the `AudioSink: Send` bound is satisfied; no `unsafe impl`.

/// How many upsampled `i16` samples one audio tick drains. 960 = 20 ms of
/// 48 kHz audio (× 2 TDR words/sample = 1920 u32 words per chunk).
pub const STAGE_MAX_SAMPLES: usize = 960;

/// The `'static` eDMA source buffer: `STAGE_MAX_SAMPLES` ×
/// `TDR_WORDS_PER_SAMPLE` `u32`s, so the source address is stable for the
/// channel's lifetime (the eDMA reads it while the audio task awaits).
///
/// `ConstStaticCell::take()` is a one-shot, so the *audio task* calls it
/// exactly once at startup (see `main.rs::audio_task`) and threads the
/// resulting `&'static mut [u32]` into [`process_chunk`] on every tick —
/// the same pattern the display `DmaDisplay` uses for its pixel stage.
pub static STAGE: static_cell::ConstStaticCell<[u32; STAGE_MAX_SAMPLES * TDR_WORDS_PER_SAMPLE]> =
    static_cell::ConstStaticCell::new([0u32; STAGE_MAX_SAMPLES * TDR_WORDS_PER_SAMPLE]);

/// One audio-task tick: pop up to [`STAGE_MAX_SAMPLES`] samples from the
/// cross-task ring, repack them into the `'static` SAI buffer, and drive the
/// eDMA as one linear transfer to SAI1.TDR[0].
///
/// On underrun the ring zero-fills the tail (silence); the eDMA always
/// emits `STAGE_MAX_SAMPLES * TDR_WORDS_PER_SAMPLE` words. The codec's
/// sample clock is independent of DMA; late rearming can still underrun
/// the SAI FIFO, and transitions to silence can click.
///
/// # Errors
///
/// DMA errors are returned; the caller bounds the wait with a timeout.
/// Diagnostic counters for the audio path. `CHUNKS_ENTERED` increments
/// once per `process_chunk` call (before the DMA). `CHUNKS_COMPLETED`
/// increments once per successful completion. The difference (entered -
/// completed) includes failed/cancelled attempts as well as the current
/// transfer. It is not an in-flight count after a failure.
///
/// `AtomicUsize` (not `AtomicU32`) so the values fit naturally on 32-bit
/// `thumbv7`; used from the audio task (the writer) and read from
/// `radio_task` and the panic handler (no lock needed).
///
/// These exist so a single log line can answer, in real time, *which
/// state the audio task is in* from the outside — the audio task
/// itself is either blocked inside `peripheral::write(...).await` or
/// it is on the next `loop` iteration about to call `process_chunk`
/// again. The counters disambiguate that without needing to race on
/// the log.
pub static CHUNKS_ENTERED: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

pub static CHUNKS_COMPLETED: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

pub fn chunks_entered() -> usize {
    CHUNKS_ENTERED.load(core::sync::atomic::Ordering::Acquire)
}

pub fn chunks_completed() -> usize {
    CHUNKS_COMPLETED.load(core::sync::atomic::Ordering::Acquire)
}

// Keep the scratch buffer out of the async future. RTIC stores futures on
// the main stack, and timeout wrappers can multiply their storage overhead.
#[inline(never)]
fn fill_stage(stage: &mut [u32]) {
    let mut local = [0i16; STAGE_MAX_SAMPLES];
    r::pop_into(&mut local);
    for (i, sample) in local.iter().enumerate() {
        let packed = pack(*sample);
        stage[2 * i] = packed;
        stage[2 * i + 1] = packed;
    }
}

pub async fn process_chunk(
    chan: &mut Channel,
    tx: &mut crate::audio::sai1::Tx,
    stage: &mut [u32],
) -> Result<(), teensy4_bsp::hal::dma::Error> {
    CHUNKS_ENTERED.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

    fill_stage(stage);

    // 3. One linear eDMA transfer of `N * TDR_WORDS_PER_SAMPLE` × 32-bit
    //    words from `stage` to the SAI TDR[0]. `peripheral::write` programs
    //    the TCD (incl. the DMAMUX slot + linear-buffer source) and enables
    //    the SAI's DMA request (FRDE); the future's `.await` parks the
    //    audio task on the DMA waker — the `audio_dma_irq` ISR in `main` fires
    //    `DMA.on_interrupt(1)` on completion, which wakes us. This is the
    //    same interrupt-driven pattern the display `DmaDisplay` uses; the
    //    core goes on running `radio_task` + `render` while the eDMA
    //    moves the 1920 words, instead of holding the CPU in a spin loop.
    //    `Drop` on the future clears FRDE on completion or cancellation.
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    let res = peripheral::write(chan, stage, tx).await;
    let now_c = cortex_m::peripheral::DWT::cycle_count();
    let ms = u64::from(now_c.wrapping_sub(t0)) * 1_000 / (teensy4_bsp::board::ARM_FREQUENCY as u64);
    if res.is_ok() {
        CHUNKS_COMPLETED.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        // Raw TCSR: request/warning/error/sync/word-start flags are bits
        // 16..20; TE is bit 31. FIFO positions wrap, so compare over time.
        let tcsr = tx.reg_dump()[5];
        let (wfp, rfp) = tx.fifo_position();
        crate::shared::set_sai_status(tcsr, (wfp as u32) << 16 | (rfp as u32), ms as u32);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dma_future_does_not_hold_sample_scratch_buffer() {
        // Measure the returned type without constructing hardware handles.
        fn future_size<'a, F: core::future::Future>(
            _: impl FnOnce(&'a mut Channel, &'a mut crate::audio::sai1::Tx, &'a mut [u32]) -> F,
        ) -> usize {
            core::mem::size_of::<F>()
        }
        let size = future_size(process_chunk);
        assert!(size < 256, "DMA future unexpectedly holds {size} bytes");
    }

    #[test]
    fn pack_left_aligns_sample_in_32_bit_slot() {
        assert_eq!(pack(1i16), 0x0001_0000);
        assert_eq!(pack(-1i16), 0xFFFF_0000);
        assert_eq!(pack(i16::MIN), 0x8000_0000);
        assert_eq!(pack(i16::MAX), 0x7FFF_0000);
    }

    #[test]
    fn sink_writable_and_send_bound() {
        fn takes_send<T: AudioSink + Send>(_: &mut T) {}
        let mut s = Sink::new();
        takes_send(&mut s);
        s.flush().unwrap();
    }

    #[test]
    fn sink_upsampler_starts_from_silence() {
        let mut s = Sink::new();
        assert_eq!(s.up.upsample(&[7; 240], &mut s.up_out), 2400);
        assert_eq!(&s.up_out[..RATE_RATIO], &[0; RATE_RATIO]);
        assert!(s.up_out[RATE_RATIO..2400].iter().all(|&x| x == 7));
    }
}
