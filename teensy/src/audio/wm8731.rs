//https://github.com/PaulStoffregen/Audio/blob/master/examples/HardwareTesting/WM8731MikroSine/WM8731MikroSine.ino
// When using AudioInputI2Sslave & AudioOutputI2Sslave with MikroE-506,
// the sample rate will be the crystal frequency divided by 256.  The
// MikroE-506 comes with a 12.288 MHz crystal, for 48 kHz sample rate.
// To get 44.1 kHz (as expected by the Teensy Audio Library) the crystal
// should be replaced with 11.2896 MHz.
//
// Recommended connections:
//    MikroE    Teensy 4
//    ------    --------
//     SCK         21
//     MISO         8
//     MOSI         7
//     ADCL        20   
//     DACL        20
//     SDA         18
//     SCL         19
//     3.3V       +3.3V
//     GND         GND
//! WM8731 (Cirrus Logic / Wolfson) I2C control plane.
//!
//! The codec speaks a 16-bit register file over I2C, addressed as
//! `[addr, value >> 8, value & 0xFF]` (three data bytes per register). The
//! `wm8731` crate builds those `{ address, value }` pairs; here we flatten
//! them to bytes and push them over the existing `Lpi2c` (embedded-hal 0.2
//! `I2c` — `write(addr, &[u8])`, the same call `crate::i2c` uses for the
//! bus probe).
//!
//! The WM8731 is **master** on the I2S bus and **slave** on the I2C bus; its
//! 16-bit register file is only reachable over I2C, so this is the only knob
//! that tells it when to start clocking.
//!
//! Register order follows the datasheet init sequence:
//! `reset → power-down → line-in → analog path → digital path →
//! digital-interface → sampling → active`. `active` is written **last**:
//! until that bit is set the WM8731 holds BCLK / FSYNC Low (slave idle) and
//! the SAI1 slave sees no clock, so it emits nothing — a hardware "mute".

use cortex_m::prelude::*; // `embedded_hal::i2c::I2c` for the `write` below.
use teensy4_bsp::board::Lpi2c;
use wm8731::WM8731;

/// WM8731 7-bit I2C read/write address.
pub const WM8731_ADDR: u8 = 0x1A;

/// One register write as the 3-byte I2C payload.
#[inline]
fn reg_bytes(addr: u8, value: u16) -> [u8; 3] {
    [addr, (value >> 8) as u8, value as u8]
}

/// Push one `wm8731::Register` across the bus.
fn write_reg(i2c: &mut Lpi2c, reg: wm8731::Register) -> Result<(), ()> {
    let bytes = reg_bytes(reg.address, reg.value);
    let _ = i2c.write(WM8731_ADDR, &bytes).map_err(|_| ());
    Ok(())
}

/// Configure the WM8731 for I2S / 16-bit / 48 kHz / **slave**, L = R mono,
/// DAC → headphone, soft-mute **off**.
///
/// Returns `Ok(())` once every register write has ACK'd (the codec NAKs when
/// it is not on the bus, so this is also a live bus check at 0x1A).
pub fn init(i2c: &mut Lpi2c) -> Result<(), ()> {
    // 0. Reset to the chip's power-on state so earlier junk does not
    //    survive between config writes.
    write_reg(i2c, WM8731::reset())?;

    // 1. Power down. A bit **set** = that block is in power-down. We power
    //    down only the mic and the master `POWEROFF` bit; every other block
    //    (line-in, ADC, DAC, output, oscillator, clkout) is left powered on.
    write_reg(
        i2c,
        WM8731::power_down(|p| {
            p.mic().power_off();
            p.power_off().power_off();
        }),
    )?;

    // 2. Line input: 0 dB (nearest 1.5-dB step = -1.5 dB ≈ unity), unmuted.
    //    Left (reg 0) carries the value; right (reg 1) is linked to it
    //    (bit 8 set) so the one left config covers both channels. Capture
    //    `li.value` before `write_reg` moves `li`, so we can reuse it.
    let li = WM8731::left_line_in(|l| l.volume().nearest_dB(0));
    let li_value = li.value;
    write_reg(i2c, li)?;
    write_reg(
        i2c,
        wm8731::Register {
            address: 1,
            value: li_value | 0x0100,
        },
    )?;

    // 3. Headphone: max volume (0x01FF ≈ +6 dB) unmuted, left = right linked.
    write_reg(
        i2c,
        wm8731::Register {
            address: 2,
            value: 0x01FF,
        },
    )?;
    write_reg(
        i2c,
        wm8731::Register {
            address: 3,
            value: 0x03FF,
        },
    )?;

    // 4. Analog path: line input (not mic) routed to the ADC.
    write_reg(
        i2c,
        WM8731::analog_audio_path(|a| a.input_select().line_input()),
    )?;

    // 5. Digital path: DAC **unmuted**, deemphasis off, HPF off, L/R linked.
    let dap = WM8731::digital_audio_path(|d| {
        d.dac_mut().disable(); // bit 3 clear = unmuted
        d.deemphasis().disable();
        d.adc_hpf().disable();
    });
    write_reg(
        i2c,
        wm8731::Register {
            address: dap.address,
            value: dap.value | 0x0001,
        },
    )?;

    // 6. Digital interface format: I2S, 16-bit, **master** (the WM8731 is
    //    the I2S-bus master — it drives BCLK/FSYNC from its 12.288 MHz MCLK;
    //    our SAI1 is the slave that receives those clocks and shifts TDR out).
    //    Normal L/R phase. `master()` sets bit 6 of this register; the
    //    crate's `slave()` (clear bit 6) would put *both* the SAI and the
    //    codec in slave mode and leave the bus with no one clocking it.
    write_reg(
        i2c,
        WM8731::digital_audio_interface_format(|f| {
            f.format().i2s();
            f.bit_length().bits_16();
            f.master_slave().master();
            f.left_right_phase().data_when_daclrc_low();
        }),
    )?;

    // 7. Sampling: normal (non-USB) mode, 256× oversample (→ 48 kHz),
    //    normal core + clkout divisors.
    write_reg(
        i2c,
        WM8731::sampling(|s| {
            s.usb_normal().normal();
            s.base_oversampling_rate().normal_256();
            s.sample_rate().adc_48().dac_48();
            s.core_clock_divider_select().normal();
            s.clock_out_divider_select().normal();
        }),
    )?;

    // 8. **Activate** the interface so the WM8731 starts driving BCLK /
    //    FSYNC. Until this, the slave SAI1 sees no clock at all.
    write_reg(i2c, WM8731::active().active())?;

    Ok(())
}
