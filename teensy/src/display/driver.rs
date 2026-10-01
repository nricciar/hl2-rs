//! ILI9341 + LPSPI4 + text helpers. `no_std`.
//!
//! The display stays task-local (owned by the `render` task), so no
//! cross-task locking is required.

use core::convert::Infallible;
use core::result::Result;

use embedded_hal::delay::DelayNs;
use embedded_hal::digital::{ErrorType as DigitalErrorType, OutputPin};
use embedded_hal::spi::{ErrorType, Operation, SpiBus, SpiDevice};
use static_cell::ConstStaticCell;
use teensy4_bsp::{board, hal};

use super::overlay;
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
    display.brightness(99)?;
    Ok(display)
}

/// ILI9341 memory-write window + pixel header commands (single-byte).
const MEM_COLUMN: u8 = 0x2A;
const MEM_ROW: u8 = 0x2B;
const MEM_WRITE: u8 = 0x2C;

/// LPSPI's 4096-bit frame limit: 128 words, each holding two RGB565 pixels.
const MAX_TX_WORDS: usize = 128;

/// Owned by one `DmaDisplay`; reused only after each DMA write completes.
static DMA_STAGE: ConstStaticCell<[u32; MAX_TX_WORDS]> = ConstStaticCell::new([0u32; MAX_TX_WORDS]);

/// DMA blitter sharing SPI and GPIO hardware with the blocking display.
/// Both handles must stay in one task and must not be used concurrently.
pub struct DmaDisplay {
    spi: board::Lpspi,
    cs: hal::gpio::Output,
    dc: hal::gpio::Output,
    chan: hal::dma::channel::Channel,
    stage: &'static mut [u32; MAX_TX_WORDS],
}

/// Send command/address bytes and drain SPI before the next DC change.
fn lpspi_write_u8(spi: &mut board::Lpspi, bytes: &[u8]) {
    SpiBus::<u8>::write(spi, bytes).expect("lpspi cmd write");
    SpiBus::<u8>::flush(spi).expect("lpspi flush");
}

impl DmaDisplay {
    /// Copy the blocking display's hardware handles and own the DMA channel.
    /// SPI configuration must remain consistent between the two handles.
    pub fn new(
        spi: &board::Lpspi,
        cs: &hal::gpio::Output,
        dc: &hal::gpio::Output,
        chan: hal::dma::channel::Channel,
    ) -> Self {
        Self {
            // These HAL handles have no Drop; access is serialized by the render task.
            spi: unsafe { core::ptr::read(spi) },
            cs: unsafe { core::ptr::read(cs) },
            dc: unsafe { core::ptr::read(dc) },
            chan,
            stage: DMA_STAGE.take(),
        }
    }

    /// Select an inclusive pixel window and begin a memory write.
    fn send_window(&mut self, x0: u16, y0: u16, x1: u16, y1: u16) {
        // Keep CS asserted through the header and all pixel chunks.
        self.cs.set_low();
        self.dc.set_low();
        lpspi_write_u8(&mut self.spi, &[MEM_COLUMN]);
        self.dc.set_high();
        lpspi_write_u8(
            &mut self.spi,
            &[(x0 >> 8) as u8, x0 as u8, (x1 >> 8) as u8, x1 as u8],
        );
        self.dc.set_low();
        lpspi_write_u8(&mut self.spi, &[MEM_ROW]);
        self.dc.set_high();
        lpspi_write_u8(
            &mut self.spi,
            &[(y0 >> 8) as u8, y0 as u8, (y1 >> 8) as u8, y1 as u8],
        );
        self.dc.set_low();
        lpspi_write_u8(&mut self.spi, &[MEM_WRITE]);
        self.dc.set_high();
    }

    /// Blit undecorated RGB565 history, compositing overlays in DMA staging.
    /// `passband` is ordered and inclusive, relative to the window's columns;
    /// the NCO cursor is at its centre. Pixels must exactly fill the inclusive
    /// window and have even length (two pixels per DMA word).
    pub async fn draw_pixels(
        &mut self,
        x0: u16,
        y0: u16,
        x1: u16,
        y1: u16,
        pixels: &[u16],
        passband: (usize, usize),
    ) {
        assert!(x0 <= x1 && y0 <= y1, "invalid pixel window");
        let cols = usize::from(x1) - usize::from(x0) + 1;
        let rows = usize::from(y1) - usize::from(y0) + 1;
        assert_eq!(pixels.len() % 2, 0, "DMA requires paired pixels");
        assert_eq!(
            Some(pixels.len()),
            cols.checked_mul(rows),
            "pixel window size mismatch"
        );
        let (pb_lo, pb_hi) = passband;
        self.send_window(x0, y0, x1, y1);
        // The DMA IRQ wakes the render task between chunks.
        self.chan.set_interrupt_on_completion(true);
        let mut k = 0usize;
        let n = pixels.len();
        while k < n {
            let chunk_end = (k + MAX_TX_WORDS * 2).min(n);
            let pairs = (chunk_end - k) / 2;
            for i in 0..pairs {
                let index = k + 2 * i;
                let first = overlay::pixel(pixels[index], index % cols, pb_lo, pb_hi, cols / 2);
                let second = overlay::pixel(
                    pixels[index + 1],
                    (index + 1) % cols,
                    pb_lo,
                    pb_hi,
                    cols / 2,
                );
                // First pixel in the upper half for MSB-first SPI.
                self.stage[i] = (u32::from(first) << 16) | u32::from(second);
            }
            let result = self
                .spi
                .dma_write(&mut self.chan, &self.stage[..pairs])
                .expect("dma_write setup")
                .await;
            if let Err(e) = result {
                self.cs.set_high();
                panic!("dma transfer error: {e:?}");
            }
            k = chunk_end;
        }
        // DMA completion need not mean SPI is idle; drain before releasing CS.
        SpiBus::<u8>::flush(&mut self.spi).expect("lpspi blit flush");
        self.cs.set_high();
        if self.chan.is_error() {
            log::error!(
                "dma channel 0 error end-of-blit: {:?}",
                self.chan.error_status()
            );
        }
        if self.chan.is_complete() {
            log::warn!("dma channel 0 still COMPLETE at end-of-blit");
            self.chan.clear_complete();
        }
    }
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

/// Cached ASCII text. Colors must stay fixed; `update` also requires fixed
/// placement. No other drawing may overwrite the cached cells.
pub struct TextLine<const N: usize> {
    text: [u8; N],
    /// Last scaled text length and placement (-1 = never painted).
    len: usize,
    lx: i32,
    ly: i32,
    ls: i32,
}

impl<const N: usize> TextLine<N> {
    pub const fn new() -> Self {
        Self {
            text: [b' '; N],
            len: 0,
            lx: -1,
            ly: -1,
            ls: -1,
        }
    }

    pub fn update(&mut self, display: &mut Display, x: u16, y: u16, text: &str, fg: u16, bg: u16) {
        self.update_cells(text, |col, ch| {
            draw_char(display, x + col as u16 * 6, y, 1, ch as char, fg, bg);
        });
    }

    /// On content or placement changes, erase the old rectangle and paint
    /// the new opaque glyph cells.
    pub fn update_scaled(
        &mut self,
        display: &mut Display,
        x: u16,
        y: u16,
        scale: u16,
        text: &str,
        fg: u16,
        bg: u16,
    ) {
        assert!(text.is_ascii() && text.len() <= N);
        if self.text[..text.len()] == *text.as_bytes()
            && self.lx == x as i32
            && self.ly == y as i32
            && self.ls == scale as i32
            && self.len == text.len()
        {
            return;
        }
        if self.ls >= 0 {
            let old_x = (self.lx as u16).min(320);
            let old_y = (self.ly as u16).min(240);
            let old_w = (self.len as u32 * 6 * self.ls as u32).min(u32::from(320 - old_x));
            let old_h = (7 * self.ls as u32).min(u32::from(240 - old_y));
            fill_rect(display, old_x, old_y, old_w as u16, old_h as u16, bg);
        }
        for (i, b) in text.as_bytes().iter().enumerate() {
            draw_char(
                display,
                x + i as u16 * 6 * scale,
                y,
                scale,
                *b as char,
                fg,
                bg,
            );
        }
        for (i, slot) in self.text.iter_mut().enumerate() {
            *slot = text.as_bytes().get(i).copied().unwrap_or(b' ');
        }
        self.len = text.len();
        self.lx = x as i32;
        self.ly = y as i32;
        self.ls = scale as i32;
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
