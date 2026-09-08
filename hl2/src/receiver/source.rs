use alloc::vec::Vec;

use num_complex::Complex;

/// A continuous stream of complex I/Q baseband samples, per radio slot.
///
/// This is the audio-receiver input contract. Implementations feed the
/// demodulator one interleaved `[I, Q, I, Q, …]` block of `f32` at a time —
/// the canonical complex baseband representation (`Complex<f32>`), the same
/// layout `SoapySDR`'s `CF32` stream uses.
///
/// The demodulator does not care where the samples come from: in production
/// the radio's EP6 per-slot complex I/Q (`Hl2Event::Baseband`) is fed to a
/// [`crate::receiver::VirtualReceiver`] block by block, while tests and
/// offline analysis use [`VecSource`]. Keeping the source trait separate means
/// the demodulator, sinks and audio pipeline are all transport-agnostic
/// (PROTOCOL.md §16).
///
/// A source is consumed once on one thread.
pub trait BasebandSource {
    /// The source sample rate of this stream (Hz).
    ///
    /// For the HL2 this is the slot's **complex** baseband rate — the C1 SPEED
    /// option at `Hl2::start_with_speed` divided by two (PROTOCOL.md §16.1) —
    /// and it is the value to pass as `source_rate_hz` to the demodulator.
    fn rate_hz(&self) -> u32;

    /// Fill `out` with the next interleaved complex I/Q samples and return the
    /// count actually written (≤ `out.len()`). Returns `0` when there is no
    /// data to feed yet — the demodulator should then skip this iteration.
    ///
    /// Successive calls yield successive blocks in order.
    fn next_block(&mut self, out: &mut [Complex<f32>]) -> usize;
}

/// A `BasebandSource` backed by an in-memory queue of complex blocks, for
/// tests and offline audio analysis.
///
/// Each [`push`](VecSource::push) enqueue block must have an even number of
/// samples (an integer number of complex pairs). `next_block` returns blocks
/// in FIFO order; once the queue is empty it returns `0`.
#[derive(Debug, Default)]
pub struct VecSource {
    rate: u32,
    queue: Vec<Vec<Complex<f32>>>,
}

impl VecSource {
    pub fn new(rate_hz: u32) -> Self {
        Self {
            rate: rate_hz,
            queue: Vec::new(),
        }
    }

    /// Enqueue one complex I/Q block (even length) and return its length.
    pub fn push(&mut self, samples: Vec<Complex<f32>>) -> usize {
        let n = samples.len();
        self.queue.push(samples);
        n
    }

    /// Number of blocks still queued.
    pub fn queued_blocks(&self) -> usize {
        self.queue.len()
    }
}

impl BasebandSource for VecSource {
    fn rate_hz(&self) -> u32 {
        self.rate
    }

    fn next_block(&mut self, out: &mut [Complex<f32>]) -> usize {
        match self.queue.first() {
            Some(blk) => {
                let n = blk.len().min(out.len());
                out[..n].copy_from_slice(&blk[..n]);
                self.queue.remove(0);
                n
            }
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f32, im: f32) -> Complex<f32> {
        Complex::new(re, im)
    }

    #[test]
    fn vecsource_drains_blocks_in_order() {
        let mut src = VecSource::new(48_000);
        assert_eq!(src.queued_blocks(), 0);
        let a: Vec<Complex<f32>> = vec![c(1.0, 0.0), c(0.0, 1.0)];
        let b: Vec<Complex<f32>> = vec![c(-1.0, 0.0), c(0.0, -1.0), c(0.5, 0.5), c(2.0, 1.0)];
        src.push(a.clone());
        src.push(b.clone());
        assert_eq!(src.queued_blocks(), 2);

        let mut buf = vec![c(0.0, 0.0); 8];
        assert_eq!(src.next_block(&mut buf), 2);
        assert_eq!(&buf[..2], a.as_slice());

        assert_eq!(src.next_block(&mut buf), 4);
        assert_eq!(&buf[..4], b.as_slice());

        assert_eq!(src.next_block(&mut buf), 0);
        assert_eq!(src.queued_blocks(), 0);
    }

    #[test]
    fn vecsource_truncates_to_buffer_capacity() {
        let mut src = VecSource::new(48_000);
        let blk: Vec<Complex<f32>> = vec![c(1.0, 0.0), c(0.0, 1.0), c(2.0, 0.0), c(0.0, 3.0)];
        assert_eq!(blk.len(), 4);
        src.push(blk);
        let mut buf = vec![c(0.0, 0.0); 2];
        assert_eq!(src.next_block(&mut buf), 2);
        assert_eq!(buf[0], c(1.0, 0.0));
        assert_eq!(buf[1], c(0.0, 1.0));
    }
}
