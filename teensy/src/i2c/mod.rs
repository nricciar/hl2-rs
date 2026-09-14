//! LPI2C1 bring-up + startup bus probe. `no_std`.
//!
//! Pins: SDA = p18, SCL = p19 (LPI2C1 on the RT1060, alt 3).

// The `Lpi2c` driver implements the embedded-hal 0.2 blocking traits; they
// are reachable through the cortex-m prelude (the same eh0 crate the HAL
// uses), which avoids pulling a second embedded-hal 0.2 path into our deps.
use cortex_m::prelude::*;
use teensy4_bsp::board::{self, Lpi2c, Lpi2cClockSpeed};
use teensy4_bsp::pins::t41::{P18, P19};

pub const TEMP_ADDR: u8 = 0x40;

/// Internal pull-up strength applied to SDA / SCL (the LPI2C `prepare()`
/// only sets open-drain, so the bus needs the pull explicitly).
///
/// 47 kΩ is the canonical I2C value for standard/fast-mode buses and is the
/// weakest available RT1060 pull-up that still guarantees timing margins at
/// 400 kHz over a few-centimetre cable.
//const PULL_CFG: imxrt_iomuxc::Config = imxrt_iomuxc::Config::modify()
//    .set_pull_keeper(Some(imxrt_iomuxc::PullKeeper::Pullup22k));
const PULL_CFG: imxrt_iomuxc::Config = imxrt_iomuxc::Config::modify()
    .set_open_drain(imxrt_iomuxc::OpenDrain::Enabled) // <-- Add this line
    .set_pull_keeper(Some(imxrt_iomuxc::PullKeeper::Pullup47k));

/// Initialize LPI2C1 on pins 18 (SDA) / 19 (SCL), 400 kHz.
pub fn init(lpi2c1: teensy4_bsp::ral::lpi2c::Instance<1>, mut scl: P19, mut sda: P18) -> Lpi2c {
    imxrt_iomuxc::configure(&mut scl, PULL_CFG);
    imxrt_iomuxc::configure(&mut sda, PULL_CFG);
    board::lpi2c(lpi2c1, scl, sda, Lpi2cClockSpeed::KHz400)
}

/// Address probe: a zero-length write = START + ADDR + STOP.
fn probes_ok(i2c: &mut Lpi2c, addr: u8) -> bool {
    i2c.write(addr, &[0]).is_ok()
}

/// Scan 0x03..=0x77 and log anything that ACKs.
pub fn scan(i2c: &mut Lpi2c) {
    log::info!("i2c: begin scan");
    let mut found = 0usize;
    for addr in 0x03..=0x77 {
        if probes_ok(i2c, addr) {
            match addr {
                TEMP_ADDR => log::info!("i2c: temp sensor at 0x{addr:02X}"),
                0x1A => log::info!("i2c: wm8731 (audio) at 0x{addr:02X}"),
                _ => log::info!("i2c: device at 0x{addr:02X}"),
            }
            found += 1;
        }
    }
    if found == 0 {
        log::warn!("i2c: no devices responding on the bus");
    } else {
        log::info!("i2c: {found} device(s) found");
    }
}
