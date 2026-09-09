//! ILI9341 + LPSPI4 + text helpers. `no_std`.
//!
//! Extracted from the old `main.rs`: `DisplaySpi` (software-CS), `NopPin`
//! (placeholder reset), `new_display`, `fill_rect`, `draw_text`. The display
//! stays task-local (owned by the `render` task) so no cross-task locking is
//! required.

use core::convert::Infallible;
use core::result::Result;

use embedded_hal::delay::DelayNs;
use embedded_hal::digital::{ErrorType as DigitalErrorType, OutputPin, StatefulOutputPin};
use embedded_hal::spi::{ErrorType, Operation, SpiBus, SpiDevice};
use teensy4_bsp::{board, hal};

use crate::display::glyphs::glyph;

/// LPSPI driver's error type (kept for the `ErrorType` impls below).
pub type LpspiError = hal::lpspi::LpspiError;

/// A no-op `OutputPin` standing in for the display's RESET line, which is
/// not wired to Teensy and is held high by the module's external pull-up.
/// `Ili9341::new` requires an `OutputPin` to "reset" the panel; the real
/// reset happens via the software RESET command it also sends.
pub struct NopPin;

impl DigitalErrorType for NopPin {
    type Error = Infallible;
}
impl OutputPin for NopPin {
    fn set_high(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn set_low(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}
impl StatefulOutputPin for NopPin {
    fn is_set_high(&mut self) -> Result<bool, Self::Error> {
        Ok(true)
    }
    fn is_set_low(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

/// `SpiBus<u8>` → `SpiDevice<u8>` adapter with a software chip-select.
pub struct DisplaySpi {
    pub spi: board::Lpspi,
    pub cs: hal::gpio::Output,
}

impl ErrorType for DisplaySpi {
    type Error = LpspiError;
}
impl SpiBus<u8> for DisplaySpi {
    fn read(&mut self, words: &mut [u8]) -> Result<(), Self::Error> {
        SpiBus::<u8>::read(&mut self.spi, words)
    }
    fn write(&mut self, words: &[u8]) -> Result<(), Self::Error> {
        SpiBus::<u8>::write(&mut self.spi, words)
    }
    fn transfer(&mut self, read: &mut [u8], write: &[u8]) -> Result<(), Self::Error> {
        SpiBus::<u8>::transfer(&mut self.spi, read, write)
    }
    fn transfer_in_place(&mut self, words: &mut [u8]) -> Result<(), Self::Error> {
        SpiBus::<u8>::transfer_in_place(&mut self.spi, words)
    }
    fn flush(&mut self) -> Result<(), Self::Error> {
        SpiBus::<u8>::flush(&mut self.spi)
    }
}
impl SpiDevice<u8> for DisplaySpi {
    fn transaction(&mut self, operations: &mut [Operation<'_, u8>]) -> Result<(), Self::Error> {
        self.cs.set_low();
        let mut result: Result<(), LpspiError> = Ok(());
        for op in operations.iter_mut() {
            result = match op {
                Operation::Read(buf) => SpiBus::<u8>::read(&mut self.spi, buf).map(|_| ()),
                Operation::Write(buf) => SpiBus::<u8>::write(&mut self.spi, buf),
                Operation::Transfer(read, write) => {
                    SpiBus::<u8>::transfer(&mut self.spi, read, write)
                }
                Operation::TransferInPlace(buf) => {
                    SpiBus::<u8>::transfer_in_place(&mut self.spi, buf)
                }
                Operation::DelayNs(ns) => {
                    let mut d = DwtDelay;
                    d.delay_ns(*ns);
                    Ok(())
                }
            };
            if result.is_err() {
                break;
            }
        }
        let flush = SpiBus::<u8>::flush(&mut self.spi);
        self.cs.set_high();
        result.and(flush)
    }
}

/// ILI9341 + `SPIInterface` + `NopPin`. Owned by the `render` task.
pub type Display =
    ili9341::Ili9341<display_interface_spi::SPIInterface<DisplaySpi, hal::gpio::Output>, NopPin>;

/// A trivial busy-wait `DelayNs` on the DWT cycle counter. The ILI9341 init
/// path calls `delay_ms(_)` after "reset"; a real delay is the right answer
/// (the panel needs the settle time even without a hard reset).
pub struct DwtDelay;
impl DelayNs for DwtDelay {
    fn delay_ns(&mut self, ns: u32) {
        let ticks =
            (ns as u64 * board::ARM_FREQUENCY as u64).div_ceil(1_000_000_000) as u32;
        let start = cortex_m::peripheral::DWT::cycle_count();
        while cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) < ticks {
            core::hint::spin_loop();
        }
    }
}

/// Construct + initialize the display (landscape + orientation +
/// invert-off + full brightness).
///
/// Caller must have already configured GPIO for CS/DC and the LPSPI4 pins,
/// and enabled DEMCR.TRCENA + DWT.CYCCNT (done in `main::init`).
pub fn new_display(
    spi: board::Lpspi,
    cs: hal::gpio::Output,
    dc: hal::gpio::Output,
) -> Result<Display, ili9341::DisplayError> {
    let iface = display_interface_spi::SPIInterface::<DisplaySpi, hal::gpio::Output>::new(
        DisplaySpi { spi, cs },
        dc,
    );
    let mut delay = DwtDelay;
    let mut display = ili9341::Ili9341::new(
        iface,
        NopPin,
        &mut delay,
        ili9341::Orientation::Landscape,
        ili9341::DisplaySize240x320,
    )?;
    let _ = display.invert_mode(ili9341::ModeState::Off);
    let _ = display.brightness(255);
    Ok(display)
}

/// Fill a rectangle with a solid colour (one SPI transaction, `w*h` u16).
pub fn fill_rect(display: &mut Display, x: u16, y: u16, w: u16, h: u16, color: u16) {
    let count = (w as usize) * (h as usize);
    display
        .draw_raw_iter(
            x,
            y,
            x.saturating_add(w).saturating_sub(1),
            y.saturating_add(h).saturating_sub(1),
            core::iter::repeat_n(color, count),
        )
        .expect("fill_rect draw");
}

/// Draw `text` at (x0, y0) with each source pixel scaled to a `scale` ×
/// `scale` block. Uses the 5×7 `glyphs::glyph`. Each character uses 6
/// columns (5 glyph + 1 gap); space is skipped as blank.
pub fn draw_text(
    display: &mut Display,
    x0: u16,
    y0: u16,
    scale: u16,
    text: &str,
    fg: u16,
    bg: u16,
) {
    let mut x = x0;
    for ch in text.chars() {
        if ch != ' ' {
            let g = glyph(ch);
            for (r, row) in g.iter().enumerate() {
                for c in 0..5u16 {
                    let bit = (row >> (5 - c as u32 - 1)) & 1 != 0;
                    fill_rect(
                        display,
                        x + c * scale,
                        y0 + r as u16 * scale,
                        scale,
                        scale,
                        if bit { fg } else { bg },
                    );
                }
            }
        }
        x += 6 * scale;
    }
}
