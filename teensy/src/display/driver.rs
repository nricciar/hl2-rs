//! ILI9341 + LPSPI4 + text helpers. `no_std`.
//!
//! The display stays task-local (owned by the `render` task), so no
//! cross-task locking is required.

use core::convert::Infallible;
use core::result::Result;

use embedded_hal::delay::DelayNs;
use embedded_hal::digital::{ErrorType as DigitalErrorType, OutputPin};
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

/// `SpiBus<u8>` → `SpiDevice<u8>` adapter with a software chip-select.
pub struct DisplaySpi {
    spi: board::Lpspi,
    cs: hal::gpio::Output,
}

impl ErrorType for DisplaySpi {
    type Error = LpspiError;
}
impl SpiDevice<u8> for DisplaySpi {
    fn transaction(&mut self, operations: &mut [Operation<'_, u8>]) -> Result<(), Self::Error> {
        self.cs.set_low();
        let mut result: Result<(), LpspiError> = Ok(());
        for op in operations.iter_mut() {
            result = match op {
                Operation::Read(buf) => SpiBus::<u8>::read(&mut self.spi, buf),
                Operation::Write(buf) => SpiBus::<u8>::write(&mut self.spi, buf),
                Operation::Transfer(read, write) => {
                    SpiBus::<u8>::transfer(&mut self.spi, read, write)
                }
                Operation::TransferInPlace(buf) => {
                    SpiBus::<u8>::transfer_in_place(&mut self.spi, buf)
                }
                Operation::DelayNs(ns) => SpiBus::<u8>::flush(&mut self.spi).map(|()| {
                    let mut d = DwtDelay;
                    d.delay_ns(*ns);
                }),
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
        let ticks = (ns as u64 * board::ARM_FREQUENCY as u64).div_ceil(1_000_000_000) as u32;
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
    display.invert_mode(ili9341::ModeState::Off)?;
    display.brightness(255)?;
    Ok(display)
}

/// Fill a rectangle with a solid colour (`w*h` u16 pixels).
pub fn fill_rect(display: &mut Display, x: u16, y: u16, w: u16, h: u16, color: u16) {
    if w == 0 || h == 0 {
        return;
    }
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
/// columns (5 glyph + 1 gap), including an opaque background.
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
        draw_char(display, x, y0, scale, ch, fg, bg);
        x += 6 * scale;
    }
}

fn draw_char(display: &mut Display, x: u16, y: u16, scale: u16, ch: char, fg: u16, bg: u16) {
    if scale == 0 {
        return;
    }
    let g = glyph(ch);
    let width = 6 * scale;
    let height = 7 * scale;
    let pixels = (0..height).flat_map(|r| {
        (0..width).map(move |c| {
            let col = c / scale;
            if col < 5 && g[(r / scale) as usize] & (1 << (4 - col)) != 0 {
                fg
            } else {
                bg
            }
        })
    });
    // Replace the cell directly, without a separate blanking pass.
    display
        .draw_raw_iter(x, y, x + width - 1, y + height - 1, pixels)
        .expect("text draw");
}

/// Cached ASCII text on a cleared background. Position and colors must stay
/// fixed, and no other drawing may overwrite its cells between updates.
pub struct TextLine<const N: usize> {
    text: [u8; N],
}

impl<const N: usize> TextLine<N> {
    pub const fn new() -> Self {
        Self { text: [b' '; N] }
    }

    pub fn update(&mut self, display: &mut Display, x: u16, y: u16, text: &str, fg: u16, bg: u16) {
        self.update_cells(text, |col, ch| {
            draw_char(display, x + col as u16 * 6, y, 1, ch as char, fg, bg);
        });
    }

    fn update_cells(&mut self, text: &str, mut draw: impl FnMut(usize, u8)) {
        assert!(text.is_ascii() && text.len() <= N);
        for (col, old) in self.text.iter_mut().enumerate() {
            let new = text.as_bytes().get(col).copied().unwrap_or(b' ');
            if *old != new {
                draw(col, new);
                *old = new;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    #[test]
    fn unchanged_text_does_not_draw() {
        let mut line = TextLine::<9>::new();
        let mut cells = Vec::new();
        line.update_cells("STREAMING", |col, ch| cells.push((col, ch)));
        assert_eq!(cells.len(), 9);
        line.update_cells("STREAMING", |_, _| panic!("unchanged cell redrawn"));
    }

    #[test]
    fn shorter_text_erases_old_suffix_and_spaces() {
        let mut line = TextLine::<9>::new();
        line.update_cells("STREAMING", |_, _| {});
        let mut cells = Vec::new();
        line.update_cells("WAIT IP", |col, ch| cells.push((col, ch)));
        assert!(cells.contains(&(4, b' ')));
        assert!(cells.contains(&(7, b' ')));
        assert!(cells.contains(&(8, b' ')));
        assert_eq!(&line.text, b"WAIT IP  ");
    }

    #[test]
    fn counter_updates_leave_address_untouched() {
        let mut line = TextLine::<32>::new();
        line.update_cells("192.168.1.5 F 99", |_, _| {});
        let mut cells = Vec::new();
        line.update_cells("192.168.1.5 F 100", |col, ch| cells.push((col, ch)));
        assert_eq!(cells, vec![(14, b'1'), (15, b'0'), (16, b'0')]);
    }
}
