//! EP6 baseband receive path: 1032-byte frame → 2 chunks × 63 I/Q pairs
//! (n_recv = 1) → pushed one sample at a time into [`crate::spectrum::Pipeline`].
//!
//! We reuse the `hl2::protocol::data` parse functions (single source of
//! truth for the wire layout, per AGENTS.md layering rule) and feed the
//! pipeline directly. Parser buffers allocate on the first valid chunk
//! and are retained even across malformed EP6 packets.

use alloc::boxed::Box;
use alloc::vec::Vec;
use num_complex::Complex;

use cortex_m::peripheral::DWT;

use hl2::protocol::data::{
    BasebandChunk, HEADER_SIZE, parse_baseband_chunk_into, parse_data_header,
};
use hl2::protocol::{CHUNK_SIZE, DATA_PACKET_SIZE, ENDPOINT_DATA_TX};
use hl2::receiver::{Demodulator, Mode, VirtualReceiver, make_demod};

use crate::mode::len as mode_len;
use crate::spectrum::Pipeline;

/// The receive-side state the radio task owns (task-local; no sync needed).
pub struct Rx {
    /// Fixed slots prevent malformed frames from dropping bump-allocated buffers.
    baseband: [BasebandChunk; 2],
    /// The virtual-RX1 receiver (offset 0) — the *same* demod pipeline the
    /// UI runs. Built on [`hl2::ReceiverConfig::default()`] (USB, 2.6 kHz)
    /// so it mirrors the UI receiver byte-for-byte; its audio is routed
    /// through the `crate::audio::sink::Sink` → SAI1 → WM8731 path. The
    /// running demodulator is always `active`'s entry — the other
    /// `crate::mode::len() - 1` live in [`pool`](Self::pool).
    vrx: VirtualReceiver,
    /// The index into `crate::mode::MODES` the *running* demodulator is
    /// (0 = USB at boot — must agree with `ReceiverConfig::default()`).
    active: usize,
    /// The *other* demodulators: index `i` holds `MODES[i]`'s demod (for
    /// `i != self.active`), and `pool[self.active]` is `None` (that
    /// demodulator is the one the `vrx` is running). Entries start `None`
    /// and are built **once**, on first switch to them ([`set_mode`]), then
    /// cached forever — so the *live* demod count is always ≤
    /// `crate::mode::len()` (each slot only ever goes `None → Some` in the
    /// one direction, back to `Some` holding its *own previous occupant*,
    /// never rebuilt on top of a live one). That cap is what keeps the
    /// no-free bump heap alive across a long button-tournament: a per-press
    /// *rebuild* would leak a whole demod's DSP state every toggle and the
    /// 160 KiB arena would run out. Once built, a switch is a pure
    /// `mem::replace` pointer swap — zero heap traffic.
    pool: [Option<Box<dyn Demodulator>>; mode_len()],
    /// Reused I/Q block handed to the virtual receiver each frame. Held as a
    /// field (not a per-call local) because the heap is a no_std bump arena
    /// that never frees — `clear()`, don't reallocate.
    iq_acc: Vec<Complex<f32>>,
    /// Cumulative DWT cycles spent parsing the EP6 baseband + accumulating
    /// the passband samples into the spectrum pipeline. The *demod* stage of
    /// the CPU readout (see `crate::shared::set_cpu_demod_pct`).
    demod_cycles: u64,
    /// Cumulative DWT cycles spent committing the FFT windows (the
    /// `mags`/S-meter compute inside `Pipeline::push` once a full
    /// `WIN_LEN`-sample window arrives, zero-padded to `N_FFT`). The
    /// *FFT* stage of the CPU readout.
    fft_cycles: u64,
}

impl Rx {
    pub fn new() -> Self {
        // USB-SSB, offset 0, 96 kSps, 2.6 kHz, 4.8 kHz audio — the `hl2`
        // crate's default receiver config, so the Teensy's virtual receiver is
        // byte-for-byte the same DSP the UI drives. Audio is routed through
        // the I2S path (`audio::sink::Sink`) → WM8731; the S-meter is a
        // spectrum consumer (see `crate::smeter`).
        let sink = Box::new(crate::audio::sink::Sink::new());
        let vrx = VirtualReceiver::new(Default::default(), sink)
            .expect("virtual USB receiver at offset 0");
        // The running demod is MODES[0]' (USB — `ReceiverConfig::default()`);
        // the other entries start unbuilt ([`pool`](Self::pool) = all-`None`)
        // and are **built once, on first switch, then cached for the life of
        // the machine** (see `activate` below). Building is deferred out of
        // `new` so the boot path stays light, and because each entry's `Some(..)`
        // is written exactly once (the slot is never rebuilt back from `Some`
        // — `activate` only ever writes `pool[old]=Some(..)` and
        // `pool[old]` was `None` at that moment) the *live* demod count is
        // always ≤ `crate::mode::len()`. That bound is what keeps the no-free
        // bump heap alive across a long press-tournament: a per-press rebuild
        // would allocate-and-leak a whole demod's DSP state every toggle.
        Self {
            baseband: core::array::from_fn(|_| BasebandChunk {
                per_rx: alloc::vec::Vec::new(),
            }),
            vrx,
            active: 0,
            pool: core::array::from_fn(|_| None),
            iq_acc: Vec::new(),
            demod_cycles: 0,
            fft_cycles: 0,
        }
    }

    /// Cumulative DWT cycles spent in the *demod* stage (EP6 baseband parse +
    /// per-sample passband accumulate + the virtual-receiver demod). Monotonic
    /// (u64, wraps at 2²⁶⁴); the radio task publishes the per-second delta as
    /// a percent of the wall window (see `crate::shared::set_cpu_demod_pct`).
    pub fn demod_cycles(&self) -> u64 {
        self.demod_cycles
    }

    /// Cumulative DWT cycles spent committing the spectrum FFT windows
    /// (`Pipeline::push` once a full `WIN_LEN`-sample window arrives — the
    /// `mags`/S-meter compute over the `[WIN_LEN, N_FFT)`-zero-padded FFT
    /// output). Monotonic; the radio task publishes the
    /// per-second delta (see `crate::shared::set_cpu_fft_pct`).
    pub fn fft_cycles(&self) -> u64 {
        self.fft_cycles
    }

    /// Switch the running demodulator to `index` (into
    /// `crate::mode::MODES`). On success the receiver runs `MODES[index]`;
    /// on error it is left exactly as it was (callers can walk the shared
    /// index back).
    ///
    /// The [`pool`](Self::pool) entry for `index` is either already built
    /// (a pure pointer swap — zero heap traffic) or `None` on the *very
    /// first* switch to that mode, in which case it is built now (a one-time
    /// allocation; the pool's `None → Some`-only invariant means a mode is
    /// never built twice, so a long button-tournament cannot exhaust the
    /// no-free bump heap). The I2S sink is reused (the 10× upsample → SAI1 →
    /// WM8731 path is unchanged), so no audio re-wiring — only the new demod's
    /// fresh DSP state (AGC / DC-blocker / channel-select) is a one-block
    /// transition on its first `process`.
    pub fn set_mode(&mut self, index: usize) -> Result<(), hl2::receiver::ReceiverError> {
        if index == self.active {
            return Ok(());
        }
        let mode = crate::mode::mode_at(index);
        let bw = mode.default_bandwidth_hz();
        // Lazily build this mode's demod once (first switch to it); `None` is
        // only ever reached for a slot that was never built, so `make_demod`
        // runs at most once per entry — repeated toggles cost no heap.
        let cfg = self.vrx.config();
        let incoming = match self.pool[index].take() {
            Some(d) => d,
            None => make_demod(
                mode,
                cfg.source_rate_hz,
                cfg.source_center_hz,
                bw,
                cfg.audio,
            )?,
        };
        let retired = self.vrx.swap_demod(incoming, mode, bw);
        self.pool[self.active] = Some(retired);
        self.active = index;
        Ok(())
    }

    /// The index the *running* demodulator is (into `crate::mode::MODES`).
    pub fn mode_index(&self) -> usize {
        self.active
    }

    /// The virtual receiver's current demod mode (the one at `mode_index()`
    /// — `MODES[0]` = USB at boot, the most recent [`set_mode`] after that).
    pub fn mode(&self) -> Mode {
        crate::mode::mode_at(self.active)
    }

    /// Consume one 1032-byte datagram. Returns the number of complex I/Q
    /// pairs pushed into the pipeline (0 for a non-EP6 frame or a short /
    /// unparseable one).
    pub fn feed(&mut self, datagram: &[u8], pipeline: &mut Pipeline) -> usize {
        if datagram.len() != DATA_PACKET_SIZE {
            return 0;
        }
        let Some(header) =
            parse_data_header(datagram[..HEADER_SIZE].try_into().expect("checked len"))
        else {
            return 0;
        };
        if header.endpoint != ENDPOINT_DATA_TX {
            return 0;
        }
        // One I/Q pair per record, per chunk; 63 per chunk at n_recv = 1,
        // 2 chunks per frame → 126 pair/frame in steady state. Each `push`
        // either just accumulates a sample into the `WIN_LEN`-window (cost:
        // one vector append — the *demod* stage) or, once every `WIN_LEN`
        // samples, commits the window (cost: mean/window/pad/FFT/max-pool —
        // the *FFT* stage over the `[WIN_LEN, N_FFT)`-zero-padded input). The DWT taps below attribute each call to the one
        // that actually dominated (a new `frame_seq` appeared → FFT bucket,
        // otherwise demod).
        let mut pushed = 0usize;
        for (bytes, chunk) in datagram[HEADER_SIZE..]
            .chunks_exact(CHUNK_SIZE)
            .zip(self.baseband.iter_mut())
        {
            if !parse_baseband_chunk_into(bytes.try_into().expect("chunk size"), 1, chunk) {
                continue;
            }
            self.iq_acc.clear();
            for rx in chunk.per_rx.iter().take(1) {
                for c in rx.iter() {
                    let seq_before = pipeline.frame_seq();
                    let t0 = DWT::cycle_count();
                    // Waterfall spectrum (unchanged path).
                    pipeline.push(c.re, c.im);
                    let t1 = DWT::cycle_count();
                    let delta = u64::from(t1.wrapping_sub(t0));
                    if pipeline.frame_seq() != seq_before {
                        self.fft_cycles += delta;
                    } else {
                        self.demod_cycles += delta;
                    }
                    pushed += 1;
                    // …and the virtual USB-SSB receiver (offset 0), accumulated
                    // per chunk and fed below.
                    self.iq_acc.push(*c);
                }
            }
            // Drive the virtual receiver with this chunk's I/Q (its audio is
            // pushed to the I2S sink; the S-meter itself reads the spectrum,
            // not this demod output).
            let vr0 = DWT::cycle_count();
            let _ = self.vrx.process(&self.iq_acc);
            self.demod_cycles += u64::from(DWT::cycle_count().wrapping_sub(vr0));
        }
        pushed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hl2::protocol::data::EP6_SYNC_LEN;
    use hl2::protocol::{C_SYNC, ENDPOINT_CONTROL, ENDPOINT_WIDEBAND};

    fn put_24be(buf: &mut [u8], off: usize, v: i32) {
        buf[off] = ((v >> 16) & 0xFF) as u8;
        buf[off + 1] = ((v >> 8) & 0xFF) as u8;
        buf[off + 2] = (v & 0xFF) as u8;
    }

    fn make_frame(endpoint: u8) -> [u8; DATA_PACKET_SIZE] {
        let mut buf = [0u8; DATA_PACKET_SIZE];
        buf[0] = 0xEF;
        buf[1] = 0xFE;
        buf[2] = 0x01;
        buf[3] = endpoint;
        for off in [HEADER_SIZE, HEADER_SIZE + CHUNK_SIZE] {
            buf[off] = C_SYNC;
            buf[off + 1] = C_SYNC;
            buf[off + 2] = C_SYNC;
            // Fill the records: I = r, Q = -r; 63 records at n_recv = 1.
            for r in 0..63 {
                let base = off + EP6_SYNC_LEN + 5 + r * 8;
                put_24be(&mut buf, base, r as i32);
                put_24be(&mut buf, base + 3, -(r as i32));
            }
        }
        buf
    }

    #[test]
    fn ep6_yields_126_pairs() {
        let buf = make_frame(ENDPOINT_DATA_TX);
        let mut rx = Rx::new();
        let mut pipeline = Pipeline::new().unwrap();
        assert_eq!(rx.feed(&buf, &mut pipeline), 126);
        assert_eq!(pipeline.len(), 126);
        assert_eq!(pipeline.frame_seq(), 0);
    }

    #[test]
    fn packets_retain_buffers_and_only_consume_valid_chunks() {
        let mut rx = Rx::new();
        let mut pipeline = Pipeline::new().unwrap();
        let valid = make_frame(ENDPOINT_DATA_TX);
        assert_eq!(rx.feed(&valid, &mut pipeline), 126);
        let buffers = rx.baseband.each_ref().map(|chunk| {
            (
                chunk.per_rx.as_ptr(),
                chunk.per_rx.capacity(),
                chunk.per_rx[0].as_ptr(),
                chunk.per_rx[0].capacity(),
            )
        });
        for endpoint in [ENDPOINT_CONTROL, ENDPOINT_WIDEBAND, 0xFF] {
            assert_eq!(rx.feed(&make_frame(endpoint), &mut pipeline), 0);
        }
        assert_eq!(rx.feed(&[], &mut pipeline), 0);
        assert_eq!(rx.feed(&valid[..DATA_PACKET_SIZE - 1], &mut pipeline), 0);
        assert_eq!(rx.feed(&[0; DATA_PACKET_SIZE + 1], &mut pipeline), 0);
        for offset in [0, 1] {
            let mut invalid = valid;
            invalid[offset] = 0;
            assert_eq!(rx.feed(&invalid, &mut pipeline), 0);
        }
        assert_eq!(pipeline.len(), 126);
        assert_eq!(pipeline.frame_seq(), 0);
        let mut expected_len = 126;
        for (bad_first, bad_second, pairs) in
            [(true, false, 63), (false, true, 63), (true, true, 0)]
        {
            let mut invalid = valid;
            if bad_first {
                invalid[HEADER_SIZE] = 0;
            }
            if bad_second {
                invalid[HEADER_SIZE + CHUNK_SIZE] = 0;
            }
            for (frame, pairs) in [(&invalid, pairs), (&valid, 126)] {
                assert_eq!(rx.feed(frame, &mut pipeline), pairs);
                expected_len += pairs;
                assert_eq!(pipeline.len(), expected_len);
                assert_eq!(pipeline.frame_seq(), 0);
                assert_eq!(
                    rx.baseband.each_ref().map(|chunk| {
                        (
                            chunk.per_rx.as_ptr(),
                            chunk.per_rx.capacity(),
                            chunk.per_rx[0].as_ptr(),
                            chunk.per_rx[0].capacity(),
                        )
                    }),
                    buffers,
                );
            }
        }
    }

    #[test]
    fn feed_matches_direct_pipeline_across_windows() {
        let mut rx = Rx::new();
        let mut received = Pipeline::new().unwrap();
        let mut direct = Pipeline::new().unwrap();
        let mut sample = 0;
        // A low-amplitude, exactly representable quarter-rate complex tone.
        // Comparing rows catches scaling, I/Q order, and dropped chunks.
        for _ in 0..34 {
            let mut buf = make_frame(ENDPOINT_DATA_TX);
            for off in [HEADER_SIZE, HEADER_SIZE + CHUNK_SIZE] {
                for r in 0..63 {
                    let (re, im) = [(1024, 0), (0, 1024), (-1024, 0), (0, -1024)][sample % 4];
                    sample += 1;
                    let base = off + EP6_SYNC_LEN + 5 + r * 8;
                    put_24be(&mut buf, base, re);
                    put_24be(&mut buf, base + 3, im);
                    direct.push(re as f32 / 8_388_607.0, im as f32 / 8_388_607.0);
                }
            }
            assert_eq!(rx.feed(&buf, &mut received), 126);
            assert_eq!(received.len(), direct.len());
            assert_eq!(received.frame_seq(), direct.frame_seq());
            assert_eq!(received.mags(), direct.mags());
        }
        assert_eq!(received.frame_seq(), 2);
        assert_eq!(received.len(), sample % crate::spectrum::WIN_LEN);
        assert!(received.mags()[80] > 8000);
    }
}
