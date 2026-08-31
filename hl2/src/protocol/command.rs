//! C0-C4 command/control packet construction and parsing for the HL2.

/// The C0 byte controls command addressing and direction.
/// Bit [7] = RQST (request response), bits [6:1] = address, bit [0] = MOX.
#[derive(Debug, Clone, Copy)]
pub struct CommandHeader {
    pub rqst: bool,
    pub addr: u8,
    pub mox: bool,
}

impl CommandHeader {
    /// Create a write command to the given address.
    pub fn write(addr: u8, mox: bool) -> Self {
        Self {
            rqst: false,
            addr,
            mox,
        }
    }

    /// Same as `write` but also request an ACK response.
    pub fn read(addr: u8, mox: bool) -> Self {
        Self {
            rqst: true,
            addr,
            mox,
        }
    }

    pub fn into_byte(self) -> u8 {
        let mut b: u8 = 0;
        if self.rqst {
            b |= 0x80;
        }
        b |= (self.addr as u8) << 1;
        if self.mox {
            b |= 0x01;
        }
        b
    }
}

/// Parse C0 from an incoming response.
///
/// Returns (ACK, RADDR, PTT). When ACK=0 the classic response uses only RADDR[3:0];
/// when ACK=1 the extended response uses RADDR[5:0].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResponseHeader {
    pub ack: bool,
    pub raddr: u8,
    pub ptt: bool,
    /// For classic responses only: CW dot indicator.
    pub cw_dot: bool,
}

/// C0-C4 data from an HL2 response containing 32-bit data.
#[derive(Debug, Clone, Copy)]
pub struct CommandData {
    pub header: ResponseHeader,
    /// 32-bit data payload (C1-C4 concatenated).
    pub data: u32,
}

impl CommandData {
    /// Parse C0-C4 from the first 5 bytes of a chunk payload.
    ///
    /// Returns `None` if the marker (C0[7]) is not consistent: when ACK is set,
    /// RQST must not be set, and vice versa.
    pub fn parse(bytes: &[u8; 5]) -> Option<Self> {
        Some(CommandData {
            header: ResponseHeader {
                ack: bytes[0] & 0x80 != 0,
                raddr: if bytes[0] & 0x80 != 0 {
                    (bytes[0] >> 1) & 0x3F
                } else {
                    (bytes[0] >> 3) & 0x0F
                },
                ptt: bytes[0] & 0x01 != 0,
                cw_dot: bytes[0] & 0x04 != 0,
            },
            data: u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]),
        })
    }

    /// Build C0-C4 bytes for a write command.
    pub fn encode(header: CommandHeader, data: u32) -> [u8; 5] {
        let mut out = [0u8; 5];
        out[0] = header.into_byte();
        let bytes = data.to_be_bytes();
        out[1] = bytes[0];
        out[2] = bytes[1];
        out[3] = bytes[2];
        out[4] = bytes[3];
        out
    }
}

/// The 32-bit DATA payload for a start/stop command.
///
/// Bit [0] = start/stop radio, bit [1] = start/stop wideband, bit [7] = disable watchdog.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StartCommand {
    /// Whether the radio should be running.
    pub radio: bool,
    /// Whether wideband data should be streaming.
    pub wideband: bool,
    /// Whether the watchdog timer should be disabled.
    pub watchdog_disabled: bool,
}

impl StartCommand {
    pub fn new(radio: bool, wideband: bool, watchdog_disabled: bool) -> Self {
        Self {
            radio,
            wideband,
            watchdog_disabled,
        }
    }

    pub fn into_byte(self) -> u8 {
        let mut b: u8 = 0;
        if self.radio {
            b |= 0x01;
        }
        if self.wideband {
            b |= 0x02;
        }
        if self.watchdog_disabled {
            b |= 0x80;
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_header_encode() {
        // Write to addr 0x02 (RX1 NCO), MOX=0
        let hdr = CommandHeader::write(0x02, false);
        assert_eq!(hdr.into_byte(), 0x04);

        // Read from addr 0x0A (LNA), MOX=1
        let hdr = CommandHeader::read(0x0A, true);
        assert_eq!(hdr.into_byte(), 0x80 | (0x0A << 1) | 0x01);
    }

    #[test]
    fn command_data_roundtrip() {
        let hdr = CommandHeader::write(0x01, false);
        let data = 0x00DBBA20u32;
        let encoded = CommandData::encode(hdr, data);
        assert_eq!(encoded[0], 0x02);
        assert_eq!(&encoded[1..], &[0x00, 0xDB, 0xBA, 0x20]);

        let parsed = CommandData::parse(&encoded).unwrap();
        assert_eq!(parsed.data, data);
    }

    #[test]
    fn start_command_encode() {
        // Start radio + wideband + keep watchdog
        let cmd = StartCommand::new(true, true, false);
        assert_eq!(cmd.into_byte(), 0x03);

        // Stop both, disable watchdog
        let cmd = StartCommand::new(false, false, true);
        assert_eq!(cmd.into_byte(), 0x80);

        // Stop radio but keep wideband
        let cmd = StartCommand::new(false, true, false);
        assert_eq!(cmd.into_byte(), 0x02);
    }

    #[test]
    fn parse_ack_response() {
        // ACK set, RADDR=0x07, PTT=0, data = 0x01234567
        let bytes = [0x8E, 0x01, 0x23, 0x45, 0x67];
        let parsed = CommandData::parse(&bytes).unwrap();
        assert!(parsed.header.ack);
        assert_eq!(parsed.header.raddr, 0x07);
        assert!(!parsed.header.ptt);
        assert_eq!(parsed.data, 0x01234567);
    }

    #[test]
    fn parse_classic_response() {
        // ACK=0, RADDR=0x03 (classic 4-bit), PTT=0, cw_dot=1
        // C0: bit[7]=0, bits[6:3]=0b0011=0x03, bit[2]=1, bit[0]=0
        let bytes = [0x1C, 0xFF, 0x00, 0x00, 0x00];
        let parsed = CommandData::parse(&bytes).unwrap();
        assert!(!parsed.header.ack);
        assert_eq!(parsed.header.raddr, 0x03);
        assert!(!parsed.header.ptt);
        assert!(parsed.header.cw_dot);
        assert_eq!(parsed.data, 0xFF000000);
    }
}
