//! HL2 Metis Discovery protocol implementation.

use crate::protocol::{DISCOVERY_REQUEST_SIZE, DISCOVERY_RESPONSE_SIZE, METIS_MARKER};

/// Parsed state of the discovery response.
#[derive(Debug, Clone)]
pub struct DiscoveryInfo {
    /// HL2 MAC address.
    pub mac: [u8; 6],
    /// Gateware major version.
    pub gateware_major: u8,
    /// Gateware minor version / patch.
    pub gateware_minor: u8,
    /// Board ID (0x06 = HL2).
    pub board_id: u8,
    /// Number of hardware receivers.
    pub rx_count: u8,
    /// Whether the HL2 is currently sending data.
    pub is_sending: bool,
    /// Sample format: true = 16-bit, false = 12-bit.
    pub sample_16bit: bool,
    /// HL2 IP address (from EEPROM config bytes).
    pub ip: [u8; 4],
}

/// Build the 63-byte discovery request packet.
///
/// Format: `[METIS_MARKER][0x02][60 bytes of 0x00]`
pub fn discovery_request() -> [u8; DISCOVERY_REQUEST_SIZE] {
    let mut buf = [0u8; DISCOVERY_REQUEST_SIZE];
    buf[0..2].copy_from_slice(&METIS_MARKER);
    buf[2] = 0x02;
    buf
}

/// Parse a 60-byte discovery response.
///
/// Returns `None` if the packet doesn't contain a valid HL2 response.
pub fn parse_discovery_response(buf: &[u8; DISCOVERY_RESPONSE_SIZE]) -> Option<DiscoveryInfo> {
    // Validate marker
    if buf[0] != 0xEF || buf[1] != 0xFE {
        return None;
    }

    let is_sending = buf[2] == 0x03;
    let is_16bit = (buf[0x14] >> 6) & 0x03 == 0x01;

    Some(DiscoveryInfo {
        mac: [
            buf[0x03], buf[0x04], buf[0x05], buf[0x06], buf[0x07], buf[0x08],
        ],
        gateware_major: buf[0x09],
        board_id: buf[0x0A],
        rx_count: buf[0x13],
        is_sending,
        sample_16bit: is_16bit,
        gateware_minor: buf[0x15],
        ip: buf[0x0D..=0x10].try_into().unwrap_or([0, 0, 0, 0]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_request_size() {
        let pkt = discovery_request();
        assert_eq!(pkt.len(), DISCOVERY_REQUEST_SIZE);
        assert_eq!(&pkt[0..2], &METIS_MARKER);
        assert_eq!(pkt[2], 0x02);
        assert_eq!(&pkt[3..], &[0u8; 60]);
    }

    #[test]
    fn parse_valid_12bit_response() {
        let mut buf = [0u8; DISCOVERY_RESPONSE_SIZE];
        buf[0] = 0xEF;
        buf[1] = 0xFE;
        buf[2] = 0x02; // not sending
        buf[3..9].copy_from_slice(&[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC]);
        buf[0x09] = 0x06; // gateware major
        buf[0x0A] = 0x06; // HL2 board ID
        buf[0x13] = 0x04; // 7 hardware receivers
        buf[0x14] = 0x00; // bits [7:6] = 00 → 12-bit
        buf[0x15] = 0x01; // gateware minor

        let info = parse_discovery_response(&buf).unwrap();
        assert_eq!(info.mac, [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC]);
        assert_eq!(info.gateware_major, 6);
        assert_eq!(info.gateware_minor, 1);
        assert_eq!(info.board_id, 0x06);
        assert_eq!(info.rx_count, 4);
        assert!(!info.is_sending);
        assert!(!info.sample_16bit);
    }

    #[test]
    fn parse_valid_16bit_response() {
        let mut buf = [0u8; DISCOVERY_RESPONSE_SIZE];
        buf[0] = 0xEF;
        buf[1] = 0xFE;
        buf[2] = 0x03; // sending
        buf[0x0A] = 0x06;
        buf[0x14] = 0x40; // bits [7:6] = 01 → 16-bit

        let info = parse_discovery_response(&buf).unwrap();
        assert!(info.is_sending);
        assert!(info.sample_16bit);
    }

    #[test]
    fn parse_invalid_marker_returns_none() {
        let buf = [0u8; DISCOVERY_RESPONSE_SIZE];
        assert!(parse_discovery_response(&buf).is_none());
    }
}
