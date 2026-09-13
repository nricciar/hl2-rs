//! Cross-task FIFO of `i16` mono samples.
//!
//! The producer is the radio task (it `push`es upsampled audio from
//! `AudioSink::write`); the consumer is the audio task (it `pop`s the next
//! DMA chunk in `AudioEngine::process_chunk`). The single-writer /
//! single-reader discipline is enforced by the two tasks (the ring has no
//! locks); the *indices* are `AtomicUsize` so each end sees a consistent
//! snapshot. The sample buffer sits behind an `UnsafeCell` so both ends can
//! write through a shared `&'static` reference even though they are distinct
//! RTIC tasks.
//!
//! Oldest samples are dropped on overflow and a shortfall on pop is filled
//! with zeros (underrun = silence), so neither side can wedge the pipeline.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Ring capacity in `i16` samples. 2^13 ≈ 170 ms of 48 kHz audio.
pub const CAP: usize = 1 << 13;

/// The cross-task sample FIFO. `UnsafeCell` wraps the buffer so it is `Sync`
/// (the atomics + single-writer/single-reader discipline guarantee coherence;
/// `UnsafeCell` is the standard `Sync`-by-manual-discipline marker).
pub struct RingState {
    buf: UnsafeCell<[i16; CAP]>,
    head: AtomicUsize,
    len: AtomicUsize,
}

impl RingState {
    const fn new() -> Self {
        RingState {
            buf: UnsafeCell::new([0i16; CAP]),
            head: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
        }
    }
}

// `UnsafeCell<T>` is `Sync` when `T` is `Send`; `AtomicUsize` is `Send + Sync`.
// SAFETY: single producer + single reader by task design; the atomics
// (head/len) are the synchronization for the shared indices, and the buffer
// slots a producer and consumer can touch simultaneously are distinct.
unsafe impl Sync for RingState {}

/// The one-and-only ring for the audio path, zero-initialized at boot
/// (silence).
static RING: RingState = RingState::new();

/// Push `src` into the ring, dropping the oldest on overflow.
///
/// Single-writer in practice (one `Sink` pushes from `AudioSink::write`),
/// but the atomic update keeps the index consistent for the reader.
/// Push `src` into the ring, dropping the oldest on overflow.
///
/// Single-writer (one `Sink` in one task) — the local `len` tracks the
/// producer's view and is published exactly once with `Release` at the end,
/// so the reader only ever sees a coherent snapshot.
pub fn push(src: &[i16]) {
    let buf = RING.buf.get();
    let mut head = RING.head.load(Ordering::Relaxed);
    let mut len = RING.len.load(Ordering::Relaxed);
    for s in src {
        if len == CAP {
            // Full: overwrite the oldest, advance the read head.
            // SAFETY: single writer (the Sink in the radio task) is the only
            // one writing to `buf[head]` right now; the reader has not yet
            // published the matching `head` advance.
            unsafe { (*buf)[head & (CAP - 1)] = *s };
            head = (head + 1) & (CAP - 1);
        } else {
            // SAFETY: producer-only slot (see the full case above).
            unsafe { (*buf)[(head + len) & (CAP - 1)] = *s };
            len += 1;
        }
    }
    // Publish: the ring's contents are fully visible before the len /
    // head values, and the reader's `Acquire` on `len` pairs with this.
    RING.head.store(head, Ordering::SeqCst);
    RING.len.store(len, Ordering::SeqCst);
}

/// Fill `out` from the ring in FIFO order, zero-filling any shortfall
/// (underrun = silence). Advances the read head by `out.len()` regardless
/// of how much was real — this keeps the consumer's pacing constant even
/// when the producer is briefly starved.
///
/// Single-reader (the audio task), so the `fetch_sub` is race-free; the
/// pair of atomic reads (`len` then `head`) are consistent because the
/// producer publishes them in order and the reader only ever reads them.
pub fn pop_into(out: &mut [i16]) {
    let buf = RING.buf.get();
    let len = RING.len.load(Ordering::Acquire);
    let head = RING.head.load(Ordering::Relaxed);
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = if i < len {
            // SAFETY: single reader (the audio task) reads `buf[head ..
            // head + len]`. The producer only writes to slots *at or above
            // head+len* (or at head when it overflows), so the reader's
            // read range and the producer's write range are disjoint.
            unsafe { (*buf)[(head + i) & (CAP - 1)] }
        } else {
            0
        };
    }
    let adv = core::cmp::min(out.len(), len);
    RING.head.store((head + adv) & (CAP - 1), Ordering::SeqCst);
    // `fetch_sub` on an underflowing atomic still saturates to 0 in
    // wrapping mode but we want 0, so clamp to len-adv.
    RING.len.fetch_sub(adv, Ordering::SeqCst);
}

/// Samples currently buffered.
pub fn len() -> usize {
    RING.len.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_order_preserved() {
        push(&[1, 2, 3, 4]);
        let mut out = [0i16; 3];
        pop_into(&mut out);
        assert_eq!(out, [1, 2, 3]);
        let mut out2 = [0i16; 2];
        pop_into(&mut out2);
        assert_eq!(out2, [4, 0]); // 4 then silence
    }

    #[test]
    fn overflow_drops_oldest() {
        for i in 0..=(CAP as i16) {
            push(&[i]);
        }
        assert_eq!(len(), CAP);
        let mut out = [0i16; 2];
        pop_into(&mut out);
        // Pushed 0..=CAP (CAP+1 samples, one dropped): head at index 1 now.
        assert_eq!(out, [1, 2]);
    }

    #[test]
    fn underrun_zero_fills_and_advances() {
        push(&[9]);
        let mut out = [0i16; 4];
        pop_into(&mut out);
        assert_eq!(out, [9, 0, 0, 0]);
        assert_eq!(len(), 0);
    }
}
