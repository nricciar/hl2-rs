//! Radio status decoded from the classic (ACK=0) C0–C4 responses that ride in
//! every EP6 chunk (PROTOCOL.md §9). The HL2 cycles RADDR 0x00 → 0x01 → 0x02
//! across successive chunks; each address carries a different 32-bit word
//! (`data` = C1..C4, big-endian):
//!
//! | RADDR | Bits      | Field                         |
//! |-------|-----------|-------------------------------|
//! | 0x00  | `[24]`    | RF ADC overload               |
//! | 0x00  | `[15]`    | TX under/overflow recovery    |
//! | 0x00  | `[14:8]`  | TX IQ FIFO count MSBs         |
//! | 0x00  | `[7:0]`   | Gateware version              |
//! | 0x01  | `[31:16]` | Temperature (12-bit ADC)      |
//! | 0x01  | `[15:0]`  | Forward power (12-bit ADC)    |
//! | 0x02  | `[31:16]` | Reverse power (12-bit ADC)    |
//! | 0x02  | `[15:0]`  | PA bias current (12-bit ADC)  |

use super::command::CommandData;

/// Latest values from the classic status cycle. Raw ADC readings are kept so
/// callers can apply their own calibration; [`Self::temperature_c`] converts
/// the one with a fixed formula.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RadioStatus {
    /// Number of overload events: RADDR 0x00 responses where the overload
    /// bit went from clear to set. Monotonic (wrapping); a consumer detects
    /// "clipped since I last looked" by comparing counts.
    pub adc_overload_events: u32,
    /// Overload bit from the most recent RADDR 0x00 response.
    pub adc_overload: bool,
    pub tx_recovery: bool,
    pub tx_fifo_msbs: u8,
    pub gateware_version: u8,
    /// `None` until the first RADDR 0x01 response arrives.
    pub temperature_raw: Option<u16>,
    pub forward_power_raw: u16,
    pub reverse_power_raw: u16,
    pub bias_current_raw: u16,
}

impl RadioStatus {
    /// Fold one response into the status. ACK=1 responses (replies to
    /// RQST writes/reads) carry request data, not status, and are ignored.
    pub fn apply(&mut self, cmd: &CommandData) {
        if cmd.header.ack {
            return;
        }
        let d = cmd.data;
        match cmd.header.raddr {
            0x00 => {
                let overload = d & (1 << 24) != 0;
                if overload && !self.adc_overload {
                    self.adc_overload_events = self.adc_overload_events.wrapping_add(1);
                }
                self.adc_overload = overload;
                self.tx_recovery = d & (1 << 15) != 0;
                self.tx_fifo_msbs = ((d >> 8) & 0x7F) as u8;
                self.gateware_version = (d & 0xFF) as u8;
            }
            0x01 => {
                self.temperature_raw = Some((d >> 16) as u16);
                self.forward_power_raw = d as u16;
            }
            0x02 => {
                self.reverse_power_raw = (d >> 16) as u16;
                self.bias_current_raw = d as u16;
            }
            _ => {}
        }
    }

    /// Board temperature in °C: the sensor outputs 10 mV/°C with a 500 mV
    /// offset, read by a 12-bit ADC with a 3.26 V reference (the formula
    /// Quisk and other HL2 hosts use).
    pub fn temperature_c(&self) -> Option<f32> {
        self.temperature_raw
            .map(|raw| (3.26 * (raw as f32 / 4096.0) - 0.5) / 0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resp(c0: u8, data: u32) -> CommandData {
        let b = data.to_be_bytes();
        CommandData::parse(&[c0, b[0], b[1], b[2], b[3]]).unwrap()
    }

    // Classic response C0: [6:3] = RADDR.
    const RADDR0: u8 = 0x00 << 3;
    const RADDR1: u8 = 0x01 << 3;
    const RADDR2: u8 = 0x02 << 3;

    #[test]
    fn decodes_the_classic_cycle() {
        let mut s = RadioStatus::default();
        assert_eq!(s.temperature_c(), None);

        s.apply(&resp(RADDR0, (1 << 24) | (1 << 15) | (0x12 << 8) | 0x49));
        assert!(s.adc_overload);
        assert_eq!(s.adc_overload_events, 1);
        assert!(s.tx_recovery);
        assert_eq!(s.tx_fifo_msbs, 0x12);
        assert_eq!(s.gateware_version, 0x49);

        // raw 1005 → 3.26 * 1005/4096 = 0.7999 V → 29.99 °C
        s.apply(&resp(RADDR1, (1005 << 16) | 0x0123));
        assert!((s.temperature_c().unwrap() - 30.0).abs() < 0.1);
        assert_eq!(s.forward_power_raw, 0x0123);

        s.apply(&resp(RADDR2, (0x0456 << 16) | 0x0789));
        assert_eq!(s.reverse_power_raw, 0x0456);
        assert_eq!(s.bias_current_raw, 0x0789);

        // A sustained overload is one event; it clears on the next 0x00
        // without the bit, and the next rising edge is a new event.
        s.apply(&resp(RADDR0, (1 << 24) | 0x49));
        assert_eq!(s.adc_overload_events, 1);
        s.apply(&resp(RADDR0, 0x49));
        assert!(!s.adc_overload);
        assert_eq!(s.adc_overload_events, 1);
        s.apply(&resp(RADDR0, (1 << 24) | 0x49));
        assert_eq!(s.adc_overload_events, 2);
    }

    #[test]
    fn ignores_ack_responses() {
        let mut s = RadioStatus::default();
        // ACK=1, RADDR=0 in C0[6:1], data with the overload bit position set.
        s.apply(&resp(0x80, 1 << 24));
        assert_eq!(s, RadioStatus::default());
    }
}
