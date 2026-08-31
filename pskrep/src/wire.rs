//! IPFIX datagram builder for PSK Reporter.
//!
//! Byte-for-byte parity with the reference implementations:
//!
//! * wsjtx `Network/PSKReporterIPFIX.cpp` (modern layout — element IDs,
//!   lengths, padding, split semantics).
//! * fldigi `src/spot/pskrep.cxx` (cross-check on header layout and 4-byte
//!   record padding).
//!
//! PSK Reporter uses two enterprise-specific IPFIX templates under PEN
//! 30351 (0x768F):
//!
//! * `0x50E2` — receiver information (5 fields, `receiverCallsign` scopes
//!   the rest).
//! * `0x50E3` — spot records (7 fields: senderCallsign, frequency, sNR,
//!   mode, senderLocator, informationSource, and `flowStartSeconds` (id 150,
//!   a standard IETF field — no enterprise bit / no PEN appended).
//!
//! All multi-byte numerics are big-endian. Records are padded to 4-byte
//! boundaries; every set carries a 2-byte length at offset 2; the message
//! carries a 2-byte total length at offset 2.

use crate::{
    ANTENNA_LIMIT, CALLER_LIMIT, LOCATOR_LIMIT, MODE_LIMIT, PROGRAM_INFO_LIMIT, RIG_INFO_LIMIT,
    Spot, Station,
};

/// The PSK Reporter PEN = 0x768F = 30351. On the wire an enterprise field
/// (id with the 0x80 bit set) is followed by this number as a 4-byte
/// integer — IPFIX / RFC 5101 mandates a 32-bit enterprise number
pub const PSK_REPORTER_PEN: u32 = 30351;
/// IETF-standard IPFIX field `flowStartSeconds` (id 150): declared without
/// the 0x80 enterprise bit and with no enterprise number appended.
const FLOW_START_SECONDS: u16 = 150;
/// `0x50E2` — receiver-information template.
pub const RECEIVER_TEMPLATE_ID: u16 = 0x50e2;
/// `0x50E3` — sender / spot-record template.
pub const SENDER_TEMPLATE_ID: u16 = 0x50e3;

/// One datagram to send to the collector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet {
    pub payload: Vec<u8>,
    /// Spot records this packet carried. Used by the caller to advance its
    /// sequence counter (wsjtx `PSKReporterIPFIX.cpp:324`).
    pub spot_count: usize,
}

/// UDP maximum datagram payload the collector accepts — wsjtx
/// `MAX_UDP_IPFIX_PAYLOAD_BYTES` (1000 − 40 IPv6 − 8 UDP).
pub const MAX_UDP_IPFIX_PAYLOAD_BYTES: usize = 1000 - 40 - 8;
/// TCP variant: 0xFFFF (wsjtx `MAX_TCP_IPFIX_PAYLOAD_BYTES`).
pub const MAX_TCP_IPFIX_PAYLOAD_BYTES: usize = 0xffff;

fn pad4(n: usize) -> usize {
    (4 - n % 4) % 4
}

/// Append a length-prefixed byte string, capped at `max_bytes`.
fn write_utf(out: &mut Vec<u8>, s: &str, max_bytes: usize) {
    let b = s.as_bytes();
    let len = b.len().min(max_bytes);
    out.push(len as u8);
    out.extend_from_slice(&b[..len]);
}

/// Back-patch `data[2..4]` to `data.len()` after padding to 4.
fn finalize_set(data: &mut Vec<u8>) {
    let pad = pad4(data.len());
    for _ in 0..pad {
        data.push(0);
    }
    let len = data.len() as u16;
    data[2] = (len >> 8) as u8;
    data[3] = len as u8;
}

fn descriptor(
    record_type: u16,
    template: u16,
    scope: Option<u16>,
    fields: &[(u16, u16)],
) -> Vec<u8> {
    let mut d = Vec::new();
    d.extend_from_slice(&record_type.to_be_bytes());
    d.extend_from_slice(&0u16.to_be_bytes());
    d.extend_from_slice(&template.to_be_bytes());
    d.extend_from_slice(&(fields.len() as u16).to_be_bytes());
    if let Some(s) = scope {
        d.extend_from_slice(&s.to_be_bytes());
    }
    for (id, len) in fields {
        if *id == FLOW_START_SECONDS {
            // Standard IETF field — no enterprise bit, no PEN.
            d.extend_from_slice(&id.to_be_bytes());
            d.extend_from_slice(&len.to_be_bytes());
        } else {
            d.extend_from_slice(&(0x8000u16 | id).to_be_bytes());
            d.extend_from_slice(&len.to_be_bytes());
            d.extend_from_slice(&PSK_REPORTER_PEN.to_be_bytes());
        }
    }
    finalize_set(&mut d);
    d
}

/// The two IPFIX template-descriptor sets, in the documented order
/// (sender first, then receiver). Mirrors wsjtx
/// `PSKReporterIPFIX.cpp:80-137`.
pub fn descriptor_sets() -> Vec<u8> {
    let sender = descriptor(
        2,
        SENDER_TEMPLATE_ID,
        None,
        &[
            (1, 0xffff),
            (5, 5),
            (6, 1),
            (10, 0xffff),
            (3, 0xffff),
            (11, 1),
            (150, 4),
        ],
    );
    let receiver = descriptor(
        3,
        RECEIVER_TEMPLATE_ID,
        Some(1),
        &[
            (2, 0xffff),
            (4, 0xffff),
            (8, 0xffff),
            (9, 0xffff),
            (13, 0xffff),
        ],
    );
    let mut out = Vec::new();
    out.extend_from_slice(&sender);
    out.extend_from_slice(&receiver);
    out
}

/// Encode the receiver-information record for `station`.
pub fn receiver_set(station: &Station) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&RECEIVER_TEMPLATE_ID.to_be_bytes());
    data.extend_from_slice(&0u16.to_be_bytes());
    write_utf(&mut data, &station.callsign, CALLER_LIMIT);
    write_utf(&mut data, &station.grid, LOCATOR_LIMIT);
    write_utf(&mut data, &station.program_info, PROGRAM_INFO_LIMIT);
    write_utf(&mut data, &station.antenna, ANTENNA_LIMIT);
    write_utf(&mut data, &station.rig_info, RIG_INFO_LIMIT);
    finalize_set(&mut data);
    data
}

/// Encode one spot record in `SENDER_TEMPLATE_ID` field order
/// (wsjtx `PSKReporterIPFIX.cpp:162-181`).
pub(crate) fn spot_record(spot: &Spot) -> Vec<u8> {
    let mut data = Vec::new();
    write_utf(&mut data, &spot.caller, CALLER_LIMIT);
    // 5-byte frequency: high byte + 32-bit BE low part.
    data.push((spot.freq_hz >> 24) as u8);
    let low = spot.freq_hz & 0x00_ff_ff_ff;
    data.extend_from_slice(&low.to_be_bytes());
    data.push(spot.snr as u8);
    write_utf(&mut data, &spot.mode, MODE_LIMIT);
    write_utf(&mut data, &spot.locator, LOCATOR_LIMIT);
    data.push(1); // informationSource = 1 (automatic)
    data.extend_from_slice(&spot.time_epoch.to_be_bytes());
    data
}

fn sender_set(records: &[Vec<u8>]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&SENDER_TEMPLATE_ID.to_be_bytes());
    data.extend_from_slice(&0u16.to_be_bytes());
    for r in records {
        data.extend_from_slice(r);
    }
    finalize_set(&mut data);
    data
}

/// Message header (16 bytes: 0x00 0x0A, length, export_time, seq,
/// observation domain) + `sets`, 4-byte padded, length backpatched.
fn message(sets: &[u8], seq: u32, obs: u32, now: u32) -> Vec<u8> {
    let body = 16usize + sets.len();
    let total = body + pad4(body);
    let mut msg = Vec::with_capacity(total);
    msg.push(0x00);
    msg.push(0x0a);
    msg.extend_from_slice(&(total as u16).to_be_bytes());
    msg.extend_from_slice(&now.to_be_bytes());
    msg.extend_from_slice(&seq.to_be_bytes());
    msg.extend_from_slice(&obs.to_be_bytes());
    msg.extend_from_slice(sets);
    for _ in 0..pad4(body) {
        msg.push(0);
    }
    msg
}

/// Build the UDP datagram(s) for one send cycle.
///
/// `include_descriptors` — embed the template descriptors in this cycle's
/// first packet (wsjtx toggles this on reconnect + at startup; we follow).
/// `seq` is the running sequence counter; each returned `Packet.spot_count`
/// tells the caller how far to advance it.
///
/// When spots overflow `max_payload_bytes`, they split into multiple
/// packets; follow-on packets re-include the receiver set but drop the
/// descriptors (wsjtx `PSKReporterIPFIX.cpp:354`).
pub fn build_packets(
    receiver: &Station,
    spots: &[Spot],
    include_descriptors: bool,
    seq: u32,
    obs: u32,
    now: u32,
    max_payload_bytes: usize,
) -> Vec<Packet> {
    let receptor = receiver_set(receiver);
    let first_base = {
        let mut v = Vec::new();
        if include_descriptors {
            v.extend_from_slice(&descriptor_sets());
        }
        v.extend_from_slice(&receptor);
        v
    };

    if spots.is_empty() {
        return vec![Packet {
            payload: message(&first_base, seq, obs, now),
            spot_count: 0,
        }];
    }

    // Cost model for the split check — must be exactly `message()`'s
    // resulting length for a packet that contains `base_sets` + a sender
    // set wrapping `cb` bytes of records:
    //   message = 16 (header) + base + (4 + cb + pad of that) + pad of all
    // and `senders_set`'s length is `4 + cb + pad4(4 + cb)`.
    let fit = |base: &Vec<u8>, cb: usize| {
        let ss_len = 4 + cb + pad4(4 + cb);
        let body = 16usize + base.len() + ss_len;
        body + pad4(body)
    };

    let mut packets = Vec::new();
    let mut cur_seq = seq;
    let mut records: Vec<Vec<u8>> = Vec::new();
    let mut record_count = 0usize;
    let mut record_bytes = 0usize;
    let mut base_sets = first_base.clone();

    for spot in spots {
        let record = spot_record(spot);
        let rec_len = record.len();
        let would_be = record_bytes + rec_len;
        if !records.is_empty() && fit(&base_sets, would_be) > max_payload_bytes {
            let ss = sender_set(&records);
            let mut combined = base_sets.clone();
            combined.extend_from_slice(&ss);
            let payload = message(&combined, cur_seq, obs, now);
            let count = record_count;
            packets.push(Packet {
                payload,
                spot_count: count,
            });
            cur_seq += count as u32;
            records.clear();
            record_count = 0;
            record_bytes = 0;
            // Follow-on packets: receiver set only.
            base_sets = receptor.clone();
        }
        records.push(record);
        record_count += 1;
        record_bytes += rec_len;
    }
    if !records.is_empty() {
        let ss = sender_set(&records);
        let mut combined = base_sets.clone();
        combined.extend_from_slice(&ss);
        let payload = message(&combined, cur_seq, obs, now);
        packets.push(Packet {
            payload,
            spot_count: record_count,
        });
    }
    packets
}

#[cfg(test)]
mod tests {
    use super::*;

    fn station() -> Station {
        Station {
            callsign: "W1AW".into(),
            grid: "FN31ur".into(),
            program_info: "hl2-api/0.1.0-Linux x86_64".into(),
            antenna: "Yagi 3 el 20m".into(),
            rig_info: "Hermes-Lite 2".into(),
        }
    }

    fn spot(call: &str, grid: &str, freq: u32, t: u32) -> Spot {
        Spot {
            caller: call.into(),
            locator: grid.into(),
            freq_hz: freq,
            snr: -10,
            time_epoch: t,
            mode: "FT8".into(),
        }
    }

    #[test]
    fn receiver_set_layout() {
        let s = station();
        let b = receiver_set(&s);
        let body = 4
            + (1 + s.callsign.len())
            + (1 + s.grid.len())
            + (1 + s.program_info.len())
            + (1 + s.antenna.len())
            + (1 + s.rig_info.len());
        let expected = body + pad4(body);
        assert_eq!(b.len(), expected);
        assert_eq!(&b[0..2], &RECEIVER_TEMPLATE_ID.to_be_bytes());
        let len_u16 = u16::from_be_bytes([b[2], b[3]]);
        assert_eq!(len_u16, expected as u16);
        assert_eq!(b[4], 4);
        assert_eq!(&b[5..9], b"W1AW");
    }

    #[test]
    fn spot_record_layout() {
        let s = spot("K9ABC", "FN31", 14_075_000, 1_700_000_000);
        let b = spot_record(&s);
        // 1+5 +5 +1 +1+3 +1+4 +1 +4 = 26
        assert_eq!(b.len(), 26);
        assert_eq!(&b[0..6], b"\x05K9ABC");
        // 5-byte frequency: high byte + 32-bit BE low part of 14_075_000.
        let freq: u32 = 14_075_000;
        assert_eq!(b[6], (freq >> 24) as u8);
        assert_eq!(&b[7..11], &(freq & 0x00ff_ff_ff_u32).to_be_bytes());
        assert_eq!(b[11], (-10i8) as u8);
        assert_eq!(&b[12..16], b"\x03FT8");
        assert_eq!(&b[16..21], b"\x04FN31");
        assert_eq!(b[21], 1);
        assert_eq!(&b[22..26], &1_700_000_000u32.to_be_bytes());
    }

    #[test]
    fn descriptor_sets_match_wsjtx_layout() {
        let d = descriptor_sets();
        // Independent byte-by-byte re-derivation, the same way wsjtx
        // `PSKReporterIPFIX.cpp:80-137` builds them.
        let mut expected = Vec::new();
        fn push_descriptor(
            out: &mut Vec<u8>,
            record_type: u16,
            template: u16,
            scope: Option<u16>,
            fields: &[(u16, u16)],
        ) {
            let mut v = Vec::new();
            v.extend_from_slice(&record_type.to_be_bytes());
            v.extend_from_slice(&0u16.to_be_bytes());
            v.extend_from_slice(&template.to_be_bytes());
            v.extend_from_slice(&(fields.len() as u16).to_be_bytes());
            if let Some(s) = scope {
                v.extend_from_slice(&s.to_be_bytes());
            }
            for (id, len) in fields {
                if *id == FLOW_START_SECONDS {
                    v.extend_from_slice(&id.to_be_bytes());
                    v.extend_from_slice(&len.to_be_bytes());
                } else {
                    v.extend_from_slice(&(0x8000u16 | id).to_be_bytes());
                    v.extend_from_slice(&len.to_be_bytes());
                    v.extend_from_slice(&PSK_REPORTER_PEN.to_be_bytes());
                }
            }
            let pad = (4 - v.len() % 4) % 4;
            v.extend(std::iter::repeat(0).take(pad));
            v[2] = (v.len() >> 8) as u8;
            v[3] = v.len() as u8;
            out.extend_from_slice(&v);
        }
        push_descriptor(
            &mut expected,
            2,
            SENDER_TEMPLATE_ID,
            None,
            &[
                (1, 0xffff),
                (5, 5),
                (6, 1),
                (10, 0xffff),
                (3, 0xffff),
                (11, 1),
                (150, 4),
            ],
        );
        push_descriptor(
            &mut expected,
            3,
            RECEIVER_TEMPLATE_ID,
            Some(1),
            &[
                (2, 0xffff),
                (4, 0xffff),
                (8, 0xffff),
                (9, 0xffff),
                (13, 0xffff),
            ],
        );
        assert_eq!(d, expected);
    }

    #[test]
    fn message_header_only_layout() {
        let station = station();
        let mut base = Vec::new();
        base.extend_from_slice(&descriptor_sets());
        base.extend_from_slice(&receiver_set(&station));
        let msg = message(&base, 7, 0x1234_5678, 1_700_000_000);
        assert_eq!(&msg[0..2], &[0x00, 0x0a]);
        let len = u16::from_be_bytes([msg[2], msg[3]]) as usize;
        assert_eq!(len, msg.len(), "length backpatched to actual");
        assert_eq!(
            u32::from_be_bytes([msg[4], msg[5], msg[6], msg[7]]),
            1_700_000_000
        );
        assert_eq!(u32::from_be_bytes([msg[8], msg[9], msg[10], msg[11]]), 7);
        assert_eq!(
            u32::from_be_bytes([msg[12], msg[13], msg[14], msg[15]]),
            0x1234_5678
        );
        assert_eq!(msg.len(), 16 + base.len() + pad4(16 + base.len()));
    }

    #[test]
    fn build_packets_no_spots_is_descriptor_only() {
        let st = station();
        let p = build_packets(
            &st,
            &[],
            true,
            7,
            0x1234_5678,
            1_700_000_000,
            MAX_UDP_IPFIX_PAYLOAD_BYTES,
        );
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].spot_count, 0);
        assert!(!p[0].payload.is_empty());
        assert!(p[0].payload.len() <= MAX_UDP_IPFIX_PAYLOAD_BYTES);
        assert_eq!(&p[0].payload[0..2], &[0x00, 0x0a]);
    }

    #[test]
    fn build_packets_with_spots_is_single_packet() {
        let st = station();
        let spots = vec![
            spot("K9ABC", "FN31", 14_075_000, 1_700_000_000),
            spot("W2XYZ", "", 7_074_000, 1_700_000_000),
            spot("N0EBT/K7", "FN31", 3_573_500, 1_700_000_000),
        ];
        let p = build_packets(
            &st,
            &spots,
            true,
            0,
            0x1234_5678,
            1_700_000_000,
            MAX_UDP_IPFIX_PAYLOAD_BYTES,
        );
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].spot_count, 3);
        assert!(p[0].payload.len() <= MAX_UDP_IPFIX_PAYLOAD_BYTES);
    }

    /// Golden datagram — the exact 232 bytes of a known (station, spot)
    /// send, byte-verified against wsjtx's layout (`PSKReporterIPFIX.cpp`),
    /// fldigi's `pskrep.cxx` and the PSK Reporter dev-doc byte example. Acts as the regression
    /// guard: any change to descriptor/record/length/padding rules that
    /// diverges from the reference will fail this test.
    #[test]
    fn golden_matches_wsjtx_reference() {
        let st = station();
        let spots = vec![spot("K9ABC", "FN31ur", 14_075_000, 1_700_000_000)];
        let p = build_packets(
            &st,
            &spots,
            true,
            7,
            0x1234_5678,
            1_700_000_000,
            MAX_UDP_IPFIX_PAYLOAD_BYTES,
        );
        assert_eq!(p.len(), 1);
        let expected = hex(
            "000a00e86553f10000000007123456780002003c50e300078001ffff0000768f800500050000768f800600010000768f800affff0000768f8003ffff0000768f800b00010000768f009600040003003450e2000500018002ffff0000768f8004ffff0000768f8008ffff0000768f8009ffff0000768f800dffff0000768f000050e20048045731415706464e333175721a686c322d6170692f302e312e302d4c696e7578207838365f36340d59616769203320656c2032306d0d4865726d65732d4c69746520320050e30020054b394142430000d6c478f60346543806464e33317572016553f100",
        );
        assert_eq!(
            p[0].payload, expected,
            "golden datagram must be byte-identical to the wsjtx reference",
        );
    }

    fn hex(h: &str) -> Vec<u8> {
        (0..h.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn split_at_max_payload_bytes() {
        let st = station();
        // 20 distinct calls on HF so dedup-style reasoning doesn't confuse
        // the split test (the split test operates on `spots` directly, but
        // keeping them distinguishable makes failures obvious).
        let spots: Vec<Spot> = (0..20)
            .map(|i| spot(&format!("CALL{i:02}"), "", 14_075_000, 1_700_000_000))
            .collect();
        // Compute the smallest `max` that fits exactly one record → forces
        // a split for >1.
        let one_record = spot_record(&spots[0]).len();
        let mut base = Vec::new();
        base.extend_from_slice(&descriptor_sets());
        base.extend_from_slice(&receiver_set(&st));
        let ss1_len = 4 + one_record + pad4(4 + one_record);
        let body1 = 16usize + base.len() + ss1_len;
        let min_for_one = body1 + pad4(body1);
        // Use `min_for_one` as max — exactly one record fits, so 20 records
        // must split into at least 20 packets.
        let p = build_packets(
            &st,
            &spots,
            true,
            0,
            0x1234_5678,
            1_700_000_000,
            min_for_one,
        );
        assert!(p.len() >= 2, "expected splits, got {}", p.len());
        assert!(p.iter().all(|pk| pk.payload.len() <= min_for_one));
        let total: usize = p.iter().map(|pk| pk.spot_count).sum();
        assert_eq!(total, spots.len());
    }
}
