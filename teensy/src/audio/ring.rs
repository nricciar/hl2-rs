//! Cross-task FIFO of `i16` mono samples.
//!
//! The producer is the radio task (it `push`es upsampled audio from
//! `AudioSink::write`); the consumer is the audio task (it `pop`s the next
//! DMA chunk in `AudioEngine::process_chunk`). Interrupt critical sections
//! protect both the samples and indices from task preemption on the
//! single-core Cortex-M, including when overflow moves the read head.
//!
//! Oldest samples are dropped on overflow and a shortfall on pop is filled
//! with zeros (underrun = silence), so neither side can wedge the pipeline.

use core::cell::RefCell;
use cortex_m::interrupt::{self, Mutex};

/// Ring capacity in `i16` samples. 2^13 ≈ 170 ms of 48 kHz audio.
pub const CAP: usize = 1 << 13;

/// Plain FIFO data; shared access is protected by `RING`'s mutex.
pub struct RingState {
    buf: [i16; CAP],
    head: usize,
    len: usize,
}

impl RingState {
    const fn new() -> Self {
        RingState {
            buf: [0i16; CAP],
            head: 0,
            len: 0,
        }
    }

    fn push(&mut self, src: &[i16]) {
        // Earlier input cannot survive overflow; bound critical-section work.
        let src = &src[src.len().saturating_sub(CAP)..];
        for &sample in src {
            if self.len == CAP {
                self.buf[self.head] = sample;
                self.head = (self.head + 1) & (CAP - 1);
            } else {
                self.buf[(self.head + self.len) & (CAP - 1)] = sample;
                self.len += 1;
            }
        }
    }

    fn pop_into(&mut self, out: &mut [i16]) {
        let count = out.len().min(self.len);
        for (i, slot) in out[..count].iter_mut().enumerate() {
            *slot = self.buf[(self.head + i) & (CAP - 1)];
        }
        out[count..].fill(0);
        self.head = (self.head + count) & (CAP - 1);
        self.len -= count;
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// The one-and-only ring for the audio path, initially empty.
static RING: Mutex<RefCell<RingState>> = Mutex::new(RefCell::new(RingState::new()));

/// Push `src` into the ring, dropping the oldest samples on overflow.
///
/// Retains the newest `CAP` samples of the buffered audio followed by `src`.
/// Copies at most `CAP` samples with interrupts masked, even for larger input.
pub fn push(src: &[i16]) {
    interrupt::free(|cs| RING.borrow(cs).borrow_mut().push(src));
}

/// Fill `out` in FIFO order, zero-filling any shortfall (underrun = silence).
/// Only available samples are consumed; silence does not advance the head.
/// Work with interrupts masked is bounded by `out.len()`.
pub fn pop_into(out: &mut [i16]) {
    interrupt::free(|cs| RING.borrow(cs).borrow_mut().pop_into(out));
}

/// Samples currently buffered.
pub fn len() -> usize {
    interrupt::free(|cs| RING.borrow(cs).borrow().len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_order_preserved() {
        let mut ring = RingState::new();
        ring.push(&[1, 2, 3, 4]);
        let mut out = [0i16; 3];
        ring.pop_into(&mut out);
        assert_eq!(out, [1, 2, 3]);
        assert_eq!(ring.len(), 1);
        let mut out2 = [0i16; 2];
        ring.pop_into(&mut out2);
        assert_eq!(out2, [4, 0]); // 4 then silence
        assert_eq!(ring.len(), 0);
    }

    #[test]
    fn overflow_drops_oldest() {
        let mut ring = RingState::new();
        for i in 0..=(CAP as i16) {
            ring.push(&[i]);
        }
        assert_eq!(ring.len(), CAP);
        let mut out = [0i16; CAP];
        ring.pop_into(&mut out);
        // Pushed 0..=CAP (CAP+1 samples, one dropped): head at index 1 now.
        for (i, sample) in out.iter().enumerate() {
            assert_eq!(*sample, (i + 1) as i16);
        }
        assert_eq!(ring.len(), 0);
    }

    #[test]
    fn underrun_consumes_only_available_samples() {
        let mut ring = RingState::new();
        ring.push(&[9]);
        let mut out = [-1i16; 4];
        ring.pop_into(&mut out);
        assert_eq!(out, [9, 0, 0, 0]);
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.head, 1);
        ring.pop_into(&mut out);
        assert_eq!(out, [0; 4]);
        assert_eq!(ring.head, 1);
        ring.push(&[10, 11]);
        ring.pop_into(&mut out);
        assert_eq!(out, [10, 11, 0, 0]);
    }

    #[test]
    fn oversized_push_retains_newest_capacity() {
        let mut ring = RingState::new();
        ring.push(&[-1, -2, -3]);
        let src: [i16; CAP * 2 + 3] = core::array::from_fn(|i| i as i16);
        ring.push(&src);
        assert_eq!(ring.len(), CAP);
        let mut out = [-1i16; CAP + 2];
        ring.pop_into(&mut out);
        assert_eq!(&out[..CAP], &src[src.len() - CAP..]);
        assert_eq!(&out[CAP..], &[0, 0]);
        assert_eq!(ring.len(), 0);
    }

    #[test]
    fn push_and_pop_wrap_around() {
        let mut ring = RingState::new();
        let src: [i16; CAP] = core::array::from_fn(|i| i as i16);
        ring.push(&src);
        let mut prefix = [0i16; CAP - 2];
        ring.pop_into(&mut prefix);
        assert_eq!(&prefix, &src[..CAP - 2]);
        ring.push(&[100, 101, 102]);
        assert_eq!(ring.len(), 5);
        let mut out = [-1i16; 7];
        ring.pop_into(&mut out);
        assert_eq!(
            out,
            [(CAP - 2) as i16, (CAP - 1) as i16, 100, 101, 102, 0, 0]
        );
        assert_eq!(ring.len(), 0);
    }

    #[test]
    fn empty_operations_leave_samples_unchanged() {
        let mut ring = RingState::new();
        ring.push(&[1, 2]);
        ring.push(&[]);
        ring.pop_into(&mut []);
        assert_eq!(ring.len(), 2);
        let mut out = [0i16; 2];
        ring.pop_into(&mut out);
        assert_eq!(out, [1, 2]);
    }
}
