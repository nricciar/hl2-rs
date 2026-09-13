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
//!   audio_task  ◄────  periodic 1 ms Systick delay  ◄─────┘
//!      pop up to 960 samples (20 ms of audio at 48 kHz)
//!        → repack into 1920 SAI TDR `u32` words (L = R, interleaved)
//!        → eDMA ch1 → SAI1.TDR[0]   (spin-on-completion, ~200 µs)
//!        → SAI slave clocks them out to the WM8731 on the I2S bus
//!
//! The [`Sink`] is task-local to the *radio* task (the `AudioSink` impl
//! the `VirtualReceiver` drives). The *audio* task owns the eDMA `Channel`
//! + SAI `Tx` and runs [`process_chunk`] on a 1 ms tick. The two tasks
//! coordinate only through the atomics on [`crate::audio::ring::RING`).
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
/// 16-bit / 2-words-per-frame / MSB-first / `Packing::None` config.
///
/// The WM8731 is stereo (no mono mode), so L = R = the sample: one `i16`
/// becomes two 16-bit words `[s, s]`; each half lands in both words (upper
/// 16 = L, lower 16 = R).
#[inline]
fn pack(sample: i16) -> u32 {
    let w = sample as u16;
    (w as u32) << 16 | (w as u32)
}

/// Upsample local capacity. The demod caps `sink.write` blocks at
/// 1024 samples (see `hl2::receiver::demod::AudioEngine`'s `out_buf`),
/// so `1024 * RATE_RATIO` upsampled `i16`s covers the maximum block the
/// virtual receiver can hand us.
const UP_OUT_CAP: usize = 1024 * RATE_RATIO; // 10 240 samples = 20 ms at 48 kHz.

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
        // Gated to the first 30 writes — the ring has no lock, so a log
        // call from here cannot deadlock, but we keep the log cadence low.
        if crate::shared::audio_writes() < 30 {
            crate::shared::audio_writes_bump();
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
/// channel's lifetime (the eDMA reads it while the audio task spins).
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
/// emits a full `STAGE_MAX_SAMPLES` words, so the WM8731 sample clock is
/// constant even when the radio task briefly stalls (a normal demod gap
/// must not cause clicks).
///
/// # Errors
///
/// Returns the imxrt-dma `Error` on source/destination address or
/// master/slave error; the caller logs it and keeps ticking.
/// Diagnostic counters for the audio path. `CHUNKS_ENTERED` increments
/// once per `process_chunk` call (before the DMA). `CHUNKS_COMPLETED`
/// increments once per successful completion. The difference (entered -
/// completed) is the number of transfers that were started but never
/// completed — that is, either currently in flight (normal, ≤ 1 at a
/// time in single-channel DMA) or permanently parked (fault, or the
/// SAI FIFO never drains so eDMA stalls mid-transfer).
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

pub async fn process_chunk(
    chan: &mut Channel,
    tx: &mut teensy4_bsp::hal::sai::Tx,
    stage: &mut [u32],
) -> Result<(), teensy4_bsp::hal::dma::Error> {
    const N: usize = STAGE_MAX_SAMPLES;
    CHUNKS_ENTERED.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

    // 1. Pull up to N upsampled samples (FIFO order); shortfalls zero-filled.
    let mut local = [0i16; N];
    r::pop_into(&mut local);

    // 2. Fold each mono sample into its `[L, R]` TDR pair (L = R).
    for (i, s) in local.iter().enumerate() {
        let packed = pack(*s);
        stage[2 * i] = packed;
        stage[2 * i + 1] = packed;
    }

    // 3. One linear eDMA transfer of `N * TDR_WORDS_PER_SAMPLE` × 32-bit
    //    words from `stage` to the SAI TDR[0]. `peripheral::write` programs
    //    the TCD (incl. the DMAMUX slot + linear-buffer source) and enables
    //    the SAI's DMA request (FWDE); the future's `.await` parks the
    //    audio task on the DMA waker — the `dma_irq` ISR in `main` fires
    //    `DMA.on_interrupt(1)` on completion, which wakes us. This is the
    //    same interrupt-driven pattern the display `DmaDisplay` uses; the
    //    core goes on running `radio_task` + `render` while the eDMA
    //    moves the 1920 words, instead of holding the CPU in a spin loop.
    //    `Drop` on the future clears FWDE when done.
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    let res = peripheral::write(chan, stage, tx).await;
    let now_c = cortex_m::peripheral::DWT::cycle_count();
    let ms = (now_c as u64 - t0 as u64) * 1_000 / (teensy4_bsp::board::ARM_FREQUENCY as u64);
    if res.is_ok() {
        CHUNKS_COMPLETED.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        // Diagnostic publish: read the SAI TCSR + TFR after the DMA
        // completes. TCSR bits we care about
        //   * WORD_START (bit 12)     — last word started (should be set
        //                               in steady state)
        //   * SYNC_ERROR (bit 11)     — externally-generated FSYNC mismatch
        //   * FIFO_ERROR (bit 10)     — TX FIFO underrun
        //   * FIFO_WARNING (bit 9)    — TX FIFO at its watermark
        //   * FIFO_REQUEST (bit 8)    — DMA request currently active
        // and TFR FIFO positions (WFP = write pos, RFP = read pos).
        // A SAI that is shifting will have WFP ≈ RFP in flight (draining).
        // A SAI that is *not* clocking will have WFP = 32 (full), RFP = 0.
        let _st = tx.status();
        let tcsr = _st.bits();
        // `fifo_position` returns `(WFP, RFP)`.
        let (wfp, rfp) = tx.fifo_position(0);
        crate::shared::set_sai_status(tcsr, (wfp as u32) << 16 | (rfp as u32), ms as u32);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_lr_is_l_and_r_identical() {
        assert_eq!(pack(1i16), 0x0001_0001);
        assert_eq!(pack(-1i16), 0xFFFF_FFFF);
        assert_eq!(pack(i16::MIN), 0x8000_8000);
        assert_eq!(pack(i16::MAX), 0x7FFF_7FFF);
    }

    #[test]
    fn sink_writable_and_send_bound() {
        fn takes_send<T: AudioSink + Send>(s: &mut T) -> usize {
            s.write(&[1, 2, 3]).unwrap()
        }
        let mut s = Sink::new();
        assert_eq!(takes_send(&mut s), 3);
        s.flush().unwrap();
    }

    #[test]
    fn sink_write_feeds_ring() {
        let mut s = Sink::new();
        s.write(&[7; 240]).unwrap(); // one demod block
        let mut out = [0i16; 20];
        r::pop_into(&mut out);
        assert!(out.iter().all(|&x| x == 7));
    }
}
