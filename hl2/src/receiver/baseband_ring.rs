//! A bounded, drop-oldest complex I/Q ring shared between the EP6 RX pump
//! (sole writer) and **any number** of downstream `VirtualReceiver`
//! consumers (readers — demodulators, spectrum taps, analysis).
//!
//! ## Design: peek, not pop
//!
//! The pump writes each slot's baseband stream into the slot's ring
//! (one ring per EP6 slot — see [`super::fanout::BasebandFanout`]).
//! Several readers may follow the *same* ring concurrently — the live vrx
//! and, under auto-decode, every headless FT8/JS8 decoder attached to that
//! slot. Each reader keeps its own **cursor**: the sequence number of the
//! next sample it wants to read from. `Peek` copies samples from the
//! buffer (it does not remove them), so a slow reader can never starve a
//! fast one and the pump's push does not have to care how many readers are
//! downstream — it is still the sole writer.
//!
//! The buffer is bounded; on overflow the oldest samples are dropped.
//! When a reader's cursor falls behind the oldest buffered sample
//! (because the pump outpaced it by more than `cap` samples) the caller
//! **resyncs** the cursor to `base_seq()` (the oldest *still buffered*
//! sample) and keeps reading from there — `peek` signals the case by
//! returning 0 — the demod's AGC/DC-block re-accumulate one frame before
//! it can emit again, which is exactly the behaviour we want from a
//! "kept-up-with-the-radio" contract (agreed in PROTOCOL.md §16.9).
//!
//! ## Threading
//!
//! Backed by a `std::vec::Vec` of fixed capacity (reserved once at
//! `new`) guarded by the outer `Arc<Mutex<BasebandRing>>` the fan-out
//! hands out. No per-reader lock: the mutex serialises the writer's
//! push with the readers' peeks, which is a memcpy-sized critical
//! section on either side. The mutex is the same one the pump already
//! holds while writing, so the reader's peek never contends against any
//! other writer (there is only one).
//!
//! ## Sequence numbers
//!
//! The writer maintains `base_seq` = the sequence number of the oldest
//! buffered sample, and increments it each time it drops an old sample
//! to make room. `count` is the number of buffered samples (valid range
//! `[base_seq, base_seq + count)`). A reader's cursor is a sequence
//! number in the same space; `peek` translates it to a buffer offset
//! (`cursor - base_seq`), clamps by `count`, and copies. Because both
//! sides live under the same mutex, the cursor is consistent with the
//! buffer contents the moment peek returns.
//!
//! ## Note on the `peek` API
//!
//! `peek` writes into a caller-provided `Vec`, reusing its allocation so
//! a demod loop can keep one buffer alive across calls. The caller is
//! expected to **advance its cursor by the returned count**; the buffer
//! does not track per-reader cursors itself (the reader's cursor is a
//! piece of thread-local state that belongs with the demod, not the
//! ring).

use num_complex::Complex;

/// Default ring capacity in complex samples. Kept **small on purpose**
/// (~68 ms of 96 kHz complex pairs) so the virtual receiver's
/// receive→audio-out latency stays well under one FT8 symbol window and
/// demod threads stay locked to "current" radio state (fast TX↔RX
/// switchover, low end-to-end latency). A demod thread drains in 1024-
/// sample chunks and only sleeps ~1 ms when empty, so a small ring
/// doesn't starve it; when the pump does outpace a reader by more than
/// `cap`, the reader resyncs at the head (see module docs) rather than
/// reading increasingly stale samples. Overridable by re-creating a
/// [`BasebandRing`] with a different `cap`.
pub const BASEBAND_RING_CAP: usize = 8_192;

/// A fixed-capacity ring of complex I/Q samples with **drop-oldest**
/// semantics on overflow and **non-destructive** peek reads (multiple
/// concurrent readers each follow the stream from their own cursor).
///
/// `push` appends, dropping the *oldest* `n + count − cap` samples if
/// the ring would overflow (and advancing `base_seq` by the same amount).
/// `peek` copies up to `max` samples from the reader's cursor into a
/// caller-supplied `Vec` without removing them; it returns 0 if the
/// cursor has caught up to the head or fallen behind (resync).
#[derive(Debug)]
pub struct BasebandRing {
    /// Storage ring. `buf[i % cap]` holds the sample at logical position
    /// `(i + cap) % cap`; valid samples live at logical positions
    /// `[base_seq, base_seq + count)`. A `None` slot means "empty" (only
    /// observable before the first `push` — after, every slot is a valid
    /// sample because `count ≤ cap`).
    buf: Vec<Complex<f32>>,
    /// Logical sequence number of the oldest buffered sample.
    base_seq: u64,
    /// Number of samples buffered (0 ≤ count ≤ cap).
    count: usize,
    /// Capacity. Kept separately because `buf` may be grown on reallocation
    /// (it isn't on the push/peek path — we pre-reserve at `new`).
    cap: usize,
}

impl BasebandRing {
    /// Allocate a ring with `cap` slots. `cap` must be > 0.
    ///
    /// The buffer is pre-sized at `cap` so the normal `n < cap` push
    /// path never allocates.
    pub fn new(cap: usize) -> Self {
        debug_assert!(cap > 0, "BasebandRing::new(cap=0)");
        let buf = vec![Complex::new(0.0, 0.0); cap];
        Self {
            buf,
            base_seq: 0,
            count: 0,
            cap,
        }
    }

    /// Ring size in complex samples (max `count` before drop-oldest kicks
    /// in).
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Number of samples currently queued.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn is_full(&self) -> bool {
        self.count == self.cap
    }

    /// Logical sequence number of the **oldest** buffered sample. A reader
    /// whose cursor equals this is at the base of the window and sees every
    /// buffered sample; if the cursor falls below it the sample has been
    /// dropped and the reader must resync the cursor here.
    pub fn base_seq(&self) -> u64 {
        self.base_seq
    }

    /// Logical sequence number one past the newest buffered sample.
    pub fn head_seq(&self) -> u64 {
        self.base_seq + self.count as u64
    }

    /// Append `samples`, dropping the *oldest* `n + count − cap` samples if
    /// the ring would overflow (advancing `base_seq` correspondingly).
    /// Never blocks, never allocates (the ring's capacity is reserved once
    /// at `new`). Returns the number of samples queued after the call
    /// (≤ `cap`).
    ///
    /// Sole-writer API: the pump is the only caller. Readers (`peek`)
    /// never mutate and do not need to coordinate with `push` beyond the
    /// shared mutex.
    pub fn push(&mut self, samples: &[Complex<f32>]) -> usize {
        let n = samples.len();
        if n == 0 {
            return self.len();
        }
        // Drop-oldest: the ring ends up holding the `new_count` newest
        // samples of (old buffered ++ incoming). `dropped` covers both
        // old samples pushed out by overflow *and* incoming samples too
        // old to survive (when `n` alone exceeds `cap`), so
        // `head_seq() == total samples ever pushed` always holds.
        let new_count = (self.count + n).min(self.cap);
        let dropped = (self.count + n).saturating_sub(new_count);
        self.base_seq += dropped as u64;
        // How much of the *new* data survives (its tail), and how much of
        // the *old* buffered data survives (its tail).
        let keep_new = n.min(new_count);
        let keep_old = new_count - keep_new;
        self.count = keep_old;
        // Write the trailing `keep_new` incoming samples at logical
        // positions `[base_seq + keep_old, base_seq + keep_old + keep_new)`,
        // i.e. starting at physical index `(base_seq + keep_old) % cap`,
        // wrapping around `buf`.
        let mut idx = ((self.base_seq + keep_old as u64) % self.cap as u64) as usize;
        for s in &samples[n - keep_new..] {
            self.buf[idx] = *s;
            idx = if idx + 1 == self.cap { 0 } else { idx + 1 };
        }
        self.count = new_count;
        self.len()
    }

    /// `peek` from `cursor` (the next sequence the reader wants) into `out`,
    /// copying up to `max` samples without removing them. Returns the count
    /// actually copied. `out`'s allocation is reused — the caller is
    /// expected to keep `out` alive across calls and **advance its cursor
    /// by the returned count**.
    ///
    /// - If `cursor < base_seq` (fell behind, its samples were dropped):
    ///   returns `0`; the caller should resync the cursor to
    ///   `self.base_seq()` (the ring does not do this on its behalf).
    /// - If `cursor >= self.head_seq()` (caught up or ahead): returns `0`
    ///   and the reader is idle.
    /// - Otherwise: copies `[offset, offset + min(max, head_seq - cursor))`
    ///   into `out`, in sample order (wrapping the physical ring as needed).
    ///
    /// A 0 return is ambiguous ("fell behind" vs. "caught up") — the caller
    /// distinguishes by comparing its cursor to `base_seq()` / `head_seq()`
    /// in the same locked critical section. Only when `cursor < base_seq()`
    /// should the resync happen; when merely caught up the cursor is left
    /// alone (resyncing it to `base_seq()` there would re-read samples the
    /// caller already consumed).
    pub fn peek(&mut self, cursor: u64, out: &mut Vec<Complex<f32>>, max: usize) -> usize {
        if self.count == 0 {
            out.clear();
            return 0;
        }
        if cursor < self.base_seq {
            // Fell behind — resync. Caller will set cursor = self.base_seq()
            // on seeing 0 from a stale cursor (checked via base_seq() before
            // the peek; if they only check "peek returned 0" they should
            // resync unconditionally — same outcome at the head).
            out.clear();
            return 0;
        }
        let off = cursor as usize - self.base_seq as usize;
        if off >= self.count {
            out.clear();
            return 0;
        }
        let n = (self.count - off).min(max);
        // The requested window `[off, off + n)` (logical offset from
        // base_seq, `off + n <= cap`) may wrap the physical ring —
        // physical index is `(base_seq + logical) % cap`. Copy in at
        // most two runs.
        let phys = ((self.base_seq + off as u64) % self.cap as u64) as usize;
        // How many samples fit between the window start and the end of
        // the physical buffer (i.e. before it wraps back to buf[0]).
        let first = (self.cap - phys).min(n);
        out.clear();
        out.extend_from_slice(&self.buf[phys as usize..phys as usize + first]);
        if n > first {
            out.extend_from_slice(&self.buf[0..n - first]);
        }
        n
    }
}

impl Default for BasebandRing {
    fn default() -> Self {
        Self::with_capacity(BASEBAND_RING_CAP)
    }
}

impl BasebandRing {
    /// Convenience: build with the crate default [`BASEBAND_RING_CAP`].
    pub fn with_capacity(cap: usize) -> Self {
        Self::new(cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f32, im: f32) -> Complex<f32> {
        Complex::new(re, im)
    }

    fn seq(from: u64, n: usize) -> Vec<Complex<f32>> {
        (from..from + n as u64).map(|i| c(i as f32, 0.0)).collect()
    }

    #[test]
    fn peek_simple_fifo() {
        let mut r = BasebandRing::new(8);
        let mut out = Vec::new();
        // Ring sequence starts at 0 regardless of the values pushed:
        // `vals(10,4)` = [10,11,12,13] lands at logical seq 0..=3.
        let d = seq(10, 4);
        r.push(&d);
        assert_eq!(r.len(), 4);
        assert_eq!(r.base_seq(), 0);
        // Single reader at the head: reads all 4 in order.
        assert_eq!(r.peek(0, &mut out, 10), 4);
        assert_eq!(out, d);
        // Caught up (cursor == head): no samples.
        assert_eq!(r.peek(4, &mut out, 10), 0);
        // Re-read from a mid-stream cursor — non-destructive, still there.
        assert_eq!(r.peek(1, &mut out, 10), 3);
        assert_eq!(out, &d[1..4]);
    }

    #[test]
    fn push_larger_than_cap_keeps_trailing() {
        let mut r = BasebandRing::new(4);
        let data = seq(100, 10); // 10 samples
        r.push(&data);
        assert!(r.is_full());
        // The 4 newest survive; logical base_seq advanced to 10 - 4 = 6.
        assert_eq!(r.base_seq(), 6);
        let mut out = Vec::new();
        assert_eq!(r.peek(6, &mut out, 100), 4);
        assert_eq!(out, &data[10 - 4..]);
    }

    #[test]
    fn peek_beyond_head_returns_zero() {
        let mut r = BasebandRing::new(16);
        r.push(&seq(0, 5));
        let mut out = Vec::new();
        // `5` is one past the head → zero.
        assert_eq!(r.peek(5, &mut out, 100), 0);
        assert!(out.is_empty());
        // `10` is ahead → zero (catch-up case).
        assert_eq!(r.peek(10, &mut out, 100), 0);
    }

    #[test]
    fn peek_multiple_readers_independent_cursors() {
        // A: slow, behind the head. B: fast, caught up. C: mid-stream.
        // All read the *same* ring, each from its own cursor, and the
        // buffer is not consumed by peek (all three see the same data
        // when they ask).
        let mut r = BasebandRing::new(16);
        r.push(&seq(0, 8));

        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();

        // A reads 2 samples from cursor 0, then "advances" to 2.
        assert_eq!(r.peek(0, &mut a, 2), 2);
        assert_eq!(a, seq(0, 2));
        // C reads 2 samples from cursor 4 (independent of A's cursor).
        assert_eq!(r.peek(4, &mut c, 2), 2);
        assert_eq!(c, seq(4, 2));
        // B reads from cursor 2 (A's new cursor) — sees the 2 samples A
        // already got, because peek does not consume.
        assert_eq!(r.peek(2, &mut b, 2), 2);
        assert_eq!(b, seq(2, 2));
        // All three re-read their windows: still there.
        assert_eq!(r.peek(0, &mut a, 2), 2);
        assert_eq!(a, seq(0, 2));
        assert_eq!(r.peek(4, &mut c, 2), 2);
        assert_eq!(c, seq(4, 2));
        assert_eq!(r.peek(2, &mut b, 2), 2);
        assert_eq!(b, seq(2, 2));
    }

    #[test]
    fn push_more_than_cap_advances_base_seq() {
        // cap 4. Push 4 (seq 0..=3, full). Push 4 more: all 4 oldest are
        // dropped, base_seq = 4, count back to 4.
        let mut r = BasebandRing::new(4);
        let first = seq(0, 4);
        let second = seq(4, 4);
        r.push(&first);
        r.push(&second);
        assert!(r.is_full());
        assert_eq!(r.base_seq(), 4);
        assert_eq!(r.len(), 4);
        let mut out = Vec::new();
        // A reader still at seq 0 is behind the head → 0 (resync case).
        assert_eq!(r.peek(0, &mut out, 10), 0);
        assert!(out.is_empty());
        // The 4 newest are fully available from the new base_seq.
        assert_eq!(r.peek(4, &mut out, 10), 4);
        assert_eq!(out, second);
    }

    #[test]
    fn push_then_peek_preserves_order() {
        let mut r = BasebandRing::new(6);
        let first = seq(0, 4);
        r.push(&first);
        let second = seq(10, 3); // 4 + 3 = 7 > 6: drop the 1 oldest
        r.push(&second);
        assert!(r.is_full());
        let mut out = Vec::new();
        // The 6 newest samples are first[1..] ++ second.
        let mut expected = Vec::new();
        expected.extend_from_slice(&first[1..]);
        expected.extend_from_slice(&second);
        assert_eq!(r.peek(r.base_seq(), &mut out, 100), 6);
        assert_eq!(out, expected);
    }

    #[test]
    fn peek_window_wraps_physical_ring() {
        // Force a non-zero base_seq whose physical start is mid-buffer so a
        // peek window wraps past buf[cap-1] back to buf[0].
        let mut r = BasebandRing::new(6);
        r.push(&seq(0, 6)); // full, base_seq 0
        r.push(&seq(6, 4)); // drop 4 oldest: logical window seq 4..=9, values
        // [4,5,6,7,8,9], physical layout buf[4..6]=[4,5], buf[0..4]=[6..9].
        assert_eq!(r.base_seq(), 4);
        let mut out = Vec::new();
        // A window starting at physical index 4 (seq 6) must wrap past
        // buf[5] back around buf[0..] to stay in order.
        assert_eq!(r.peek(6, &mut out, 10), 4);
        assert_eq!(out, [c(6.0, 0.0), c(7.0, 0.0), c(8.0, 0.0), c(9.0, 0.0)]);
    }

    #[test]
    fn peek_reuses_out_allocation() {
        let mut r = BasebandRing::new(32);
        let d = seq(0, 8);
        r.push(&d);
        let mut out: Vec<Complex<f32>> = Vec::with_capacity(64);
        let n = r.peek(0, &mut out, 100);
        assert_eq!(n, 8);
        assert_eq!(out.len(), 8);
        assert!(out.capacity() >= 8);
    }

    #[test]
    fn fuzz_push_peek_stays_in_buffer() {
        // Random push/peek sequences against a reference deque: same
        // drop-oldest semantics, any cursor, any window — catches
        // off-by-ones in the physical-offset arithmetic without needing a
        // specific wrap position.
        let mut r = BasebandRing::new(13);
        let mut refq: Vec<Complex<f32>> = Vec::new(); // oldest..newest tail
        let mut next_seq = 0u64;
        let mut out = Vec::new();
        // Deterministic LCG so a failure is reproducible.
        let mut seed: u64 = 0x5DEECE66D;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        for iter in 0..20_000 {
            // Push a random 0..=30 samples.
            let n = rnd() % 31;
            if n > 0 {
                let s = seq(next_seq, n);
                r.push(&s);
                next_seq += n as u64;
                // Reference model: the ring ends holding the `cap` NEWEST
                // of (old ++ incoming).
                let combined = refq.len() + n;
                if combined <= r.capacity() {
                    refq.extend(s);
                } else if n >= r.capacity() {
                    // The push alone overflows: only its tail survives.
                    refq = s[n - r.capacity()..].to_vec();
                } else {
                    // Push fits but combined overflows: drop the oldest of
                    // what was buffered.
                    refq.drain(..combined - r.capacity());
                    refq.extend(s);
                }
                assert!(r.len() <= r.capacity());
                assert_eq!(r.len(), refq.len());
            }
            // Peek from a random cursor spanning [base-seq - 2, head + 4]
            // and a random max.
            let base = r.base_seq();
            let head = base + refq.len() as u64;
            let cursor = base.saturating_sub(2) + rnd() as u64 % (refq.len() as u64 + 8);
            let max = 1 + rnd() % 64;
            let got = r.peek(cursor, &mut out, max);
            if cursor < base || cursor >= head {
                assert_eq!(got, 0, "cursor {cursor}, base {base}, head {head}");
            } else {
                let off = (cursor - base) as usize;
                let want = ((refq.len() - off).min(max)) as usize;
                assert_eq!(got, want);
                assert_eq!(out, &refq[off..off + got]);
            }
        }
    }

    #[test]
    fn default_capacity_is_const() {
        let r = BasebandRing::default();
        assert_eq!(r.capacity(), BASEBAND_RING_CAP);
    }
}
