//! EP6 baseband receive path: 1032-byte frame → 2 chunks × 63 I/Q pairs
//! (n_recv = 1) → pushed one sample at a time into [`crate::spectrum::Pipeline`].
//!
//! We reuse the `hl2::protocol::data` parse functions (single source of
//! truth for the wire layout, per AGENTS.md layering rule) and feed the
//! pipeline directly. The pipeline's `Vec`s are heap; on Teensy (thumbv7em,
//! a bump allocator) the one-time allocation in `new()` is all we need.

use hl2::protocol::data::{BasebandChunk, parse_receive_packet_into};
use hl2::protocol::{DATA_PACKET_SIZE, ENDPOINT_DATA_TX};

use crate::spectrum::Pipeline;

/// The receive-side state the radio task owns (task-local; no sync needed).
pub struct Rx {
    /// Reused across frames so the steady-state parse never reallocates.
    pub iq: alloc::vec::Vec<i16>,
    pub baseband: alloc::vec::Vec<BasebandChunk>,
    /// How many 1032-byte EP6 frames we have consumed (for status lines).
    pub frames: u32,
    /// How many complex pairs we have pushed into the pipeline (for
    /// `S 126 Hz`-style rate lines).
    pub pairs: u32,
}

impl Rx {
    pub fn new() -> Self {
        Self {
            iq: alloc::vec::Vec::new(),
            baseband: alloc::vec::Vec::new(),
            frames: 0,
            pairs: 0,
        }
    }

    /// Consume one 1032-byte datagram. Returns the number of complex I/Q
    /// pairs pushed into the pipeline (0 for a non-EP6 frame or a short /
    /// unparseable one).
    pub fn feed(&mut self, datagram: &[u8], pipeline: &mut Pipeline) -> usize {
        if datagram.len() != DATA_PACKET_SIZE {
            return 0;
        }
        let buf: &[u8; DATA_PACKET_SIZE] =
            datagram.try_into().expect("checked len");
        let Some(parsed) = parse_receive_packet_into(buf, 1, &mut self.iq, &mut self.baseband)
        else {
            return 0;
        };
        if parsed.header.endpoint != ENDPOINT_DATA_TX {
            return 0;
        }
        // One I/Q pair per record, per chunk; 63 per chunk at n_recv = 1,
        // 2 chunks per frame → 126 pair/frame in steady state.
        let mut pushed = 0usize;
        for chunk in parsed.baseband.iter() {
            for rx in chunk.per_rx.iter().take(1) {
                for c in rx.iter() {
                    pipeline.push(c.re, c.im);
                    pushed += 1;
                }
            }
        }
        self.frames = self.frames.wrapping_add(1);
        self.pairs = self.pairs.wrapping_add(pushed as u32);
        pushed
    }
}

/// A tiny, host-testable helper that mirrors [`Rx::feed`] over the
/// `parse_receive_packet_into` output — no `Rx` state needed. Extracted
/// so the unit tests below can drive it with synthetic 1032-byte frames
/// without touching smoltcp or the pipeline.
#[cfg(test)]
fn feed_pairs(
    buf: &[u8; DATA_PACKET_SIZE],
    iq: &mut alloc::vec::Vec<i16>,
    baseband: &mut alloc::vec::Vec<BasebandChunk>,
) -> (hl2::protocol::data::DataHeader, usize) {
    let parsed = parse_receive_packet_into(buf, 1, iq, baseband).expect("valid frame");
    let pairs = parsed
        .baseband
        .iter()
        .map(|c| c.per_rx.iter().map(|s| s.len()).sum::<usize>())
        .sum::<usize>();
    (parsed.header, pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hl2::protocol::data::EP6_SYNC_LEN;
    use hl2::protocol::{C_SYNC, CHUNK_SIZE, ENDPOINT_DATA_TX};

    fn put_24be(buf: &mut [u8], off: usize, v: i32) {
        buf[off] = ((v >> 16) & 0xFF) as u8;
        buf[off + 1] = ((v >> 8) & 0xFF) as u8;
        buf[off + 2] = (v & 0xFF) as u8;
    }

    fn make_frame(endpoint: u8, n_records: usize) -> [u8; DATA_PACKET_SIZE] {
        let mut buf = [0u8; DATA_PACKET_SIZE];
        buf[0] = 0xEF;
        buf[1] = 0xFE;
        buf[2] = 0x01;
        buf[3] = endpoint;
        for off in [8, 8 + CHUNK_SIZE] {
            buf[off] = C_SYNC;
            buf[off + 1] = C_SYNC;
            buf[off + 2] = C_SYNC;
            // Fill the records: I = r, Q = -r. n_records = 63 at n_recv = 1.
            for r in 0..n_records {
                let base = off + EP6_SYNC_LEN + 5 + r * 8;
                put_24be(&mut buf, base, r as i32);
                put_24be(&mut buf, base + 3, -(r as i32));
            }
        }
        buf
    }

    #[test]
    fn ep6_yields_126_pairs() {
        let buf = make_frame(ENDPOINT_DATA_TX, 63);
        let mut iq = alloc::vec::Vec::new();
        let mut baseband = alloc::vec::Vec::new();
        let (h, pairs) = feed_pairs(&buf, &mut iq, &mut baseband);
        assert_eq!(h.endpoint, ENDPOINT_DATA_TX);
        assert_eq!(pairs, 126, "63 records × 2 chunks = 126 pairs");
        assert!(iq.is_empty(), "EP6 does not populate EP4 iq");
    }

    #[test]
    fn non_ep6_yields_0_pairs() {
        let buf = make_frame(0x04, 63); // EP4 wideband
        let mut iq = alloc::vec::Vec::new();
        let mut baseband = alloc::vec::Vec::new();
        let (_h, pairs) = feed_pairs(&buf, &mut iq, &mut baseband);
        assert_eq!(pairs, 0, "EP4 does not populate EP6 baseband");
    }
}
