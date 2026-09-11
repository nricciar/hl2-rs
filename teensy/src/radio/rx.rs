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
use hl2::receiver::{DropSink, VirtualReceiver};

use crate::spectrum::Pipeline;

/// The receive-side state the radio task owns (task-local; no sync needed).
pub struct Rx {
    /// Fixed slots prevent malformed frames from dropping bump-allocated buffers.
    baseband: [BasebandChunk; 2],
    /// The virtual USB-SSB receiver (offset 0) — the *same* demod pipeline as
    /// the UI (`Mode::Ssb(Usb)`, 96 kSps → 2.6 kHz channel select → 4.8 kHz
    /// audio). Built on [`hl2::ReceiverConfig::default()`] so it mirrors the
    /// UI receiver exactly. It runs so the demod is exercised at CPU speed and
    /// its cost is surfaced (see the radio task's CPU% readout); its audio is
    /// **muted** (a [`DropSink`]) — the I2S out for real audio is a later step.
    vrx: VirtualReceiver,
    /// Reused I/Q block handed to the virtual receiver each frame. Held as a
    /// field (not a per-call local) because the heap is a no_std bump arena
    /// that never frees — `clear()`, don't reallocate.
    iq_acc: Vec<Complex<f32>>,
}

impl Rx {
    pub fn new() -> Self {
        // USB-SSB, offset 0, 96 kSps, 2.6 kHz, 4.8 kHz audio — the `hl2`
        // crate's default receiver config, so the Teensy's virtual receiver is
        // byte-for-byte the same DSP the UI drives. Audio goes to a DropSink
        // (muted); the S-meter is a spectrum consumer (see `crate::smeter`).
        let sink = Box::new(DropSink);
        let vrx = VirtualReceiver::new(Default::default(), sink)
            .expect("virtual USB receiver at offset 0");
        Self {
            baseband: core::array::from_fn(|_| BasebandChunk {
                per_rx: alloc::vec::Vec::new(),
            }),
            vrx,
            iq_acc: Vec::new(),
        }
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
        // 2 chunks per frame → 126 pair/frame in steady state.
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
                    // Waterfall spectrum (unchanged path).
                    pipeline.push(c.re, c.im);
                    pushed += 1;
                    // …and the virtual USB-SSB receiver (offset 0), accumulated
                    // per chunk and fed below.
                    self.iq_acc.push(*c);
                }
            }
            // Drive the virtual receiver with this chunk's I/Q (audio is muted
            // by its DropSink); the S-meter itself reads the spectrum, not this.
            let _ = self.vrx.process(&self.iq_acc);
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
        assert_eq!(received.len(), sample % crate::spectrum::N_FFT);
        assert!(received.mags()[80] > 8000);
    }
}
