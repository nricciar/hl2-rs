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

use hl2::protocol::data::{
    BasebandChunk, HEADER_SIZE, parse_baseband_chunk_into, parse_data_header,
};
use hl2::protocol::{CHUNK_SIZE, DATA_PACKET_SIZE, ENDPOINT_DATA_TX};
use hl2::receiver::{Demodulator, Mode, VirtualReceiver, make_demod};

use crate::mode::len as mode_len;
use crate::spectrum::Pipeline;

#[inline]
fn cycle_count() -> u32 {
    #[cfg(target_arch = "arm")]
    {
        cortex_m::peripheral::DWT::cycle_count()
    }
    #[cfg(not(target_arch = "arm"))]
    {
        0
    }
}

/// The receive-side state the radio task owns (task-local; no sync needed).
pub struct Rx {
    /// Fixed slots prevent malformed frames from dropping bump-allocated buffers.
    baseband: [BasebandChunk; 2],
    /// RX1 at offset 0, feeding the I2S audio sink.
    vrx: VirtualReceiver,
    /// Running mode index; starts at 0 to match `ReceiverConfig::default()`.
    active: usize,
    /// Inactive demodulators, built once and retained for the no-free bump heap.
    /// The active slot and modes not yet visited are `None`.
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
        let sink = Box::new(crate::audio::sink::Sink::new());
        let vrx =
            VirtualReceiver::new(Default::default(), sink).expect("virtual receiver at offset 0");
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

    /// Select a mode index, building it on first use and reusing its DSP state later.
    /// The sink is unchanged; build errors leave the current mode running.
    pub fn set_mode(&mut self, index: usize) -> Result<(), hl2::receiver::ReceiverError> {
        if index == self.active {
            return Ok(());
        }
        let mode = crate::mode::mode_at(index);
        let bw = mode.default_bandwidth_hz();
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

    /// Running index into `crate::mode::MODES`.
    pub fn mode_index(&self) -> usize {
        self.active
    }

    /// The virtual receiver's current demod mode.
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
                    let t0 = cycle_count();
                    // Waterfall spectrum (unchanged path).
                    pipeline.push(c.re, c.im);
                    let t1 = cycle_count();
                    let delta = u64::from(t1.wrapping_sub(t0));
                    if pipeline.frame_seq() != seq_before {
                        self.fft_cycles += delta;
                    } else {
                        self.demod_cycles += delta;
                    }
                    pushed += 1;
                    // Accumulate RX1 samples for demodulation below.
                    self.iq_acc.push(*c);
                }
            }
            // Drive the virtual receiver with this chunk's I/Q (its audio is
            // pushed to the I2S sink; the S-meter itself reads the spectrum,
            // not this demod output).
            let vr0 = cycle_count();
            let _ = self.vrx.process(&self.iq_acc);
            self.demod_cycles += u64::from(cycle_count().wrapping_sub(vr0));
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
    fn mode_pool_reuses_demodulators_across_full_cycles() {
        let mut rx = Rx::new();
        let mut pointers = [None; mode_len()];
        assert!(rx.pool.iter().all(Option::is_none));
        for _ in 0..3 {
            for step in 1..=mode_len() {
                let previous = rx.mode_index();
                let index = step % mode_len();
                rx.set_mode(index).unwrap();
                rx.set_mode(index).unwrap();
                let retired =
                    rx.pool[previous].as_deref().unwrap() as *const dyn Demodulator as *const ();
                if let Some(expected) = pointers[previous] {
                    assert_eq!(retired, expected);
                } else {
                    pointers[previous] = Some(retired);
                }
                assert!(rx.pool[index].is_none());
                assert_eq!(rx.mode_index(), index);
                let mode = crate::mode::mode_at(index);
                assert_eq!(rx.mode(), mode);
                let cfg = rx.vrx.config();
                assert_eq!(cfg.mode, mode);
                assert_eq!(cfg.bandwidth(), mode.default_bandwidth_hz());
                assert_eq!(cfg.source_rate_hz, 96_000);
                assert_eq!(cfg.source_center_hz, 0.0);
                assert_eq!(cfg.audio.rate_hz, 4_800);
                assert_eq!(cfg.audio.gain_db, 0.0);
                assert!(cfg.tap.is_none());
            }
            assert_eq!(rx.mode_index(), 0);
            assert_eq!(
                rx.pool.iter().filter(|slot| slot.is_some()).count(),
                mode_len() - 1
            );
        }
    }

    #[test]
    fn feed_matches_direct_pipeline_across_windows() {
        let mut rx = Rx::new();
        let mut received = Pipeline::new().unwrap();
        let mut direct = Pipeline::new().unwrap();
        let mut sample = 0;
        // Integer I/Q for a coherent 6 kHz tone inside the 24 kHz view.
        // Comparing rows catches scaling, I/Q order, and dropped chunks.
        let sine = [
            0, 392, 724, 946, 1024, 946, 724, 392, 0, -392, -724, -946, -1024, -946, -724, -392,
        ];
        for _ in 0..34 {
            let mut buf = make_frame(ENDPOINT_DATA_TX);
            for off in [HEADER_SIZE, HEADER_SIZE + CHUNK_SIZE] {
                for r in 0..63 {
                    let (re, im) = (sine[(sample + 4) % 16], sine[sample % 16]);
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
