//! WM8731 (Cirrus Logic / Wolfson) I2C control plane.
//!
//! The two-byte I2C control word contains a 7-bit register address followed
//! by 9 data bits: `[(register << 1) | data[8], data[7:0]]`.
//!
//! In the example above, `AudioOutputI2Sslave` describes the Teensy.
//! `AudioControlWM8731master` configures the codec as I2S master, supplying
//! BCLK/LRCLK from its crystal. It remains a slave on the separate I2C bus.
//! At 48 kHz in normal mode, BCLK is 64fs: two 32-bit slots containing
//! 16-bit samples. See WM8731 datasheet "Master and Slave Mode Operation".

use cortex_m::prelude::*; // `embedded_hal::i2c::I2c` for the `write` below.
use teensy4_bsp::board::Lpi2c;
use wm8731::WM8731;

/// WM8731 7-bit I2C read/write address.
pub const WM8731_ADDR: u8 = 0x1A;

/// One register write as the two-byte I2C payload.
#[inline]
fn reg_bytes(addr: u8, value: u16) -> [u8; 2] {
    [(addr << 1) | ((value >> 8) as u8 & 1), value as u8]
}

/// Push one `wm8731::Register` across the bus.
fn write_reg(i2c: &mut Lpi2c, reg: wm8731::Register) -> Result<(), ()> {
    let bytes = reg_bytes(reg.address, reg.value);
    i2c.write(WM8731_ADDR, &bytes).map_err(|e| {
        log::error!("wm8731 register {:#04x} write failed: {:?}", reg.address, e);
    })
}

/// Configure I2S master / 16-bit / 48 kHz, DAC to headphone, soft-mute off.
///
/// Returns `Ok(())` once every register write has ACK'd (the codec NAKs when
/// it is not on the bus, so this is also a live bus check at 0x1A).
pub fn init(i2c: &mut Lpi2c) -> Result<(), ()> {
    use embedded_hal::delay::DelayNs;
    let mut delay = crate::display::driver::DwtDelay;
    configure(|reg| write_reg(i2c, reg), |ms| delay.delay_ms(ms))
}

fn configure(
    mut write: impl FnMut(wm8731::Register) -> Result<(), ()>,
    mut delay_ms: impl FnMut(u32),
) -> Result<(), ()> {
    write(WM8731::reset())?;

    // Keep outputs off while the analog supplies settle. POWEROFF and
    // oscillator power-down must remain CLEAR for the crystal to run.
    write(WM8731::power_down(|p| {
        p.mic().power_off();
        p.output().power_off();
    }))?;

    write(WM8731::left_line_in(|l| l.volume().nearest_dB(0)))?;
    write(WM8731::right_line_in(|l| l.volume().nearest_dB(0)))?;
    // 0 dB (0x79), simultaneous L/R update (bit 8). No zero-cross wait:
    // initial silence must not prevent the volume update from taking effect.
    write(wm8731::Register {
        address: 2,
        value: 0x0179,
    })?;
    write(WM8731::analog_audio_path(|a| {
        a.mute_mic().enable();
        a.dac_select().select();
    }))?;
    write(WM8731::digital_audio_path(|d| d.dac_mut().enable()))?;

    // 6. Digital interface format: I2S, 16-bit, **master** (the WM8731 is
    //    the I2S-bus master — it drives BCLK/FSYNC from its 12.288 MHz MCLK;
    //    our SAI1 is the slave that receives those clocks and shifts TDR out).
    //    Normal L/R phase. `master()` sets bit 6 of this register; the
    //    crate's `slave()` (clear bit 6) would put *both* the SAI and the
    //    codec in slave mode and leave the bus with no one clocking it.
    write(WM8731::digital_audio_interface_format(|f| {
        f.format().i2s();
        f.bit_length().bits_16();
        f.master_slave().master();
        f.left_right_phase().data_when_daclrc_low();
    }))?;

    // 7. Sampling: normal (non-USB) mode, 256× oversample (→ 48 kHz),
    //    normal core + clkout divisors.
    write(WM8731::sampling(|s| {
        s.usb_normal().normal();
        s.base_oversampling_rate().normal_256();
        s.sample_rate().adc_48().dac_48();
        s.core_clock_divider_select().normal();
        s.clock_out_divider_select().normal();
    }))?;

    delay_ms(100);
    write(WM8731::active().active())?;
    write(WM8731::power_down(|p| p.mic().power_off()))?;
    delay_ms(5);
    write(WM8731::digital_audio_path(|d| d.dac_mut().disable()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn control_words_pack_address_and_ninth_data_bit() {
        assert_eq!(reg_bytes(15, 0), [0x1e, 0]);
        assert_eq!(reg_bytes(7, 0x42), [0x0e, 0x42]);
        assert_eq!(reg_bytes(2, 0x179), [0x05, 0x79]);
    }

    #[test]
    fn init_powers_and_routes_dac_with_master_clocks() {
        let mut writes = Vec::new();
        let mut delays = Vec::new();
        configure(
            |r| {
                writes.push((r.address, r.value));
                Ok(())
            },
            |ms| delays.push(ms),
        )
        .unwrap();
        assert_eq!(
            writes,
            [
                (15, 0),
                (6, 0x12),
                (0, 0x17),
                (1, 0x17),
                (2, 0x179),
                (4, 0x12),
                (5, 8),
                (7, 0x42),
                (8, 0),
                (9, 1),
                (6, 2),
                (5, 0),
            ]
        );
        assert_eq!(delays, [100, 5]);
    }

    #[test]
    fn init_stops_at_each_failed_write() {
        for fail_at in 0..12 {
            let mut calls = 0;
            assert!(
                configure(
                    |_| {
                        calls += 1;
                        if calls == fail_at + 1 {
                            Err(())
                        } else {
                            Ok(())
                        }
                    },
                    |_| {}
                )
                .is_err()
            );
            assert_eq!(calls, fail_at + 1);
        }
    }
}
