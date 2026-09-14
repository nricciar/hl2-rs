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

/// ILI9341 memory-write window + pixel header commands (single-byte).
const MEM_COLUMN: u8 = 0x2A;
const MEM_ROW: u8 = 0x2B;
const MEM_WRITE: u8 = 0x2C;

/// Largest LPSPI transaction: the HAL caps `Transaction::new_words` at 128
/// `u32` (4096-bit max frame size, see `lpspi::Transaction`), so one DMA
/// transfer through this peripheral is bounded to 128 words (256
/// RGB565 pixels — each `u32` holds two packed RGB565 halves, MSB-first).
/// Chunk the pixel stream accordingly.
const MAX_TX_WORDS: usize = 128;

/// Staging buffer for a 128-`u32` DMA transfer. Held in `.uninit` (OC-RAM)
/// like the framebuffer. One `DmaDisplay` owns the `&'static mut` for its
/// lifetime; the DMA engine reads it while the LPSPI streams the chunk to
/// the panel, so it must not be mutated until the transfer's `DONE` bit
/// clears. A `DmaDisplay` is single-task-owned, so the reference is
/// uniquely held across the blit's lifetime.
static DMA_STAGE: ConstStaticCell<[u32; MAX_TX_WORDS]> = ConstStaticCell::new([0u32; MAX_TX_WORDS]);

/// DMA-accelerated ILI9341 pixel blit.
///
/// Holds an LPSPI4 instance bitwise-copied from the blocking `DisplaySpi`
/// (same register file, same clock settings — the LPSPI is not reconfigured
/// after `Ili9341::new`, so a second handle on the same `Lpspi` state sees
/// the same `ccr_cache` / `bit_order` / `mode` / `pcs` the DMA path uses),
/// the display's CS and DC GPIO pins (bitwise-copied the same way), and a
/// DMA `Channel` allocated from `resources.dma`.
pub struct DmaDisplay {
    spi: board::Lpspi,
    cs: hal::gpio::Output,
    dc: hal::gpio::Output,
    chan: hal::dma::channel::Channel,
    stage: &'static mut [u32; MAX_TX_WORDS],
}

/// Blocking u8 SPI write through `board::Lpspi` (via the embedded-hal
/// `SpiBus` impl the blocking `DisplaySpi` already depends on). Used for the
/// ILI9341 header bytes — a small number of command/address tokens that the
/// DMA engine has no job here (they're interleaved with DC pin toggles).
fn lpspi_write_u8(spi: &mut board::Lpspi, bytes: &[u8]) {
    SpiBus::<u8>::write(spi, bytes).expect("lpspi cmd write");
    SpiBus::<u8>::flush(spi).expect("lpspi flush");
}

impl DmaDisplay {
    /// Build a `DmaDisplay` by bitwise-copying the LPSPI + CS pin + DC pin
    /// from their `&` references, and taking ownership of the eDMA channel.
    ///
    /// The caller retains ownership of the originals — those get moved into
    /// the blocking `Display` (via `new_display`) later in the same
    /// `init()` body. Both handles share the same LPSPI register file and
    /// the same GPIO bits at the hardware level (the type system just
    /// doesn't know about the second handle).
    ///
    /// # Safety
    ///
    /// `board::Lpspi` (`hal::lpspi::Lpspi`) has all-pod fields —
    /// `NonZeroCell`, `ccr_cache`, `bit_order`, `mode`, `pcs` — none with
    /// `Drop`. `hal::gpio::Output` is `{gpio: AnyInstance, offset: u32}` —
    /// same story. `core::ptr::read`-ing each through a shared reference is
    /// therefore a valid bitwise copy.
    pub fn new(
        spi: &board::Lpspi,
        cs: &hal::gpio::Output,
        dc: &hal::gpio::Output,
        chan: hal::dma::channel::Channel,
    ) -> Self {
        Self {
            spi: unsafe { core::ptr::read(spi) },
            cs: unsafe { core::ptr::read(cs) },
            dc: unsafe { core::ptr::read(dc) },
            chan,
            stage: DMA_STAGE.take(),
        }
    }

    /// Write the ILI9341 memory-window + memory-write header (column, row,
    /// then the "stream pixels next" command). DC is toggled per segment in
    /// exactly the byte order the blocking ILI9341 init path used.
    fn send_window(&mut self, x0: u16, y0: u16, x1: u16, y1: u16) {
        // Assert CS for the whole pixel transfer — the ILI9341 holds its
        // memory-window latch across CS boundaries only when the memory
        // write command has been issued, and we don't want CS to deassert
        // between header bytes because that would split the command stream
        // the panel is expecting.
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

    /// DMA-blit `pixels` (RGB565, in the row-major order the ILI9341
    /// expects for the current memory window) into the window
    /// `(x0..=x1, y0..=y1)`. This drives the header (memory window + write
    /// command) synchronously, then hands the pixel stream to the eDMA
    /// engine in `MAX_TX_WORDS`-sized chunks.
    ///
    /// Per chunk:
    ///   1. Repack 256 u16 pixels → 128 u32s (upper half = left half of
    ///      the u32 word, matching the LPSPI's MSB-first shift).
    ///   2. `Lpspi::dma_write` programs the LPSPI TCR for the chunk and
    ///      enables the LPSPI's DMA-tx bit.
    ///   3. The returned `Write` future resolves when the eDMA signals the
    ///      channel's `DONE` bit. The channel is armed with
    ///      `set_interrupt_on_completion(true)`, so completion fires the
    ///      `DMA0_DMA16` IRQ; the `dma_irq` handler in `main` calls
    ///      `hal::dma::DMA.on_interrupt(0)`, which clears the flag and
    ///      wakes the registered waker. The `.await` below stores that
    ///      waker, so `render` yields its timeslice back to the RTIC loop
    ///      and `radio_task` (network demod + spectrum) runs while the
    ///      eDMA moves bytes — the exact win the DMA path was added for.
    ///   4. `SpiBus::flush` waits until the panel has actually clocked
    ///      out the last frame (the LPSPI's `BUSY` flag may still be set
    ///      when the DMA's `DONE` bit clears) — necessary before the next
    ///      chunk's TCR is enqueued.
    pub async fn draw_pixels(&mut self, x0: u16, y0: u16, x1: u16, y1: u16, pixels: &[u16]) {
        self.send_window(x0, y0, x1, y1);
        // Arm interrupt-on-completion once. `prepare_write` (called inside
        // `dma_write`) does not touch INTMAJOR, so this stays set across all
        // chunks in this blit, and also stays set for subsequent blits — the
        // channel is ours exclusively, so there's no contention to worry
        // about. The `DMA0_DMA16` IRQ (channel 0) wakes whatever task is
        // currently `.await`-ing on `self.chan`.
        self.chan.set_interrupt_on_completion(true);
        let mut k = 0usize;
        let mut cyc_setup = 0u64;
        let n = pixels.len();
        let t0 = cortex_m::peripheral::DWT::cycle_count();
        // Diagnostic: per-chunk wall time for the first 6 chunks + min/max
        // across the rest, so we can tell "each of 150 chunks takes 15 ms"
        // (all samples ~15000) from "one chunk stalls, rest fine" (one
        // sample in the millions, rest ~124).
        const SAMPLE_N: usize = 6;
        let mut sample: [u32; SAMPLE_N] = [0; SAMPLE_N];
        let mut sample_i: usize = 0;
        let mut max_wait_us: u64 = 0;
        let mut min_wait_us: u64 = u64::MAX;
        // IRQ count across the blit — how many times the ISR actually fired.
        let irq_start = crate::shared::irq_fires();
        while k < n {
            let chunk_end = (k + MAX_TX_WORDS * 2).min(n);
            let pairs = (chunk_end - k) / 2;
            for i in 0..pairs {
                // Upper half first so the LPSPI MSB-first shift emits
                // `pixel[0].hi, pixel[0].lo, pixel[1].hi, pixel[1].lo` —
                // the exact byte order the ILI9341 samples.
                self.stage[i] = ((pixels[k + 2 * i] as u32) << 16) | (pixels[k + 2 * i + 1] as u32);
            }
            let a = cortex_m::peripheral::DWT::cycle_count();
            let w = self
                .spi
                .dma_write(&mut self.chan, &self.stage[..pairs])
                .expect("dma_write setup");
            let setup = cortex_m::peripheral::DWT::cycle_count().wrapping_sub(a);
            let result = w.await;
            let b = cortex_m::peripheral::DWT::cycle_count();
            let wait_us = ((b.wrapping_sub(a)) as u64 * 1_000_000) / (board::ARM_FREQUENCY as u64);
            if wait_us < min_wait_us {
                min_wait_us = wait_us;
            }
            if wait_us > max_wait_us {
                max_wait_us = wait_us;
            }
            if sample_i < SAMPLE_N {
                sample[sample_i] = wait_us as u32;
                sample_i += 1;
            }
            cyc_setup += setup as u64;
            if let Err(e) = result {
                self.cs.set_high();
                panic!("dma transfer error: {e:?}");
            }
            k = chunk_end;
        }
        let irq_fires = crate::shared::irq_fires().wrapping_sub(irq_start);
        // One final drain: every chunk's `Write::drop` already waited for
        // `TDDE` to deassert (LPSPI idle), but belt-and-suspenders before
        // we release CS so the last frame is on the wire before the panel
        // latches.
        SpiBus::<u8>::flush(&mut self.spi).expect("lpspi blit flush");
        self.cs.set_high();
        let cycles = cortex_m::peripheral::DWT::cycle_count().wrapping_sub(t0);
        let ms = cycles as f64 / (board::ARM_FREQUENCY as f64) * 1e3;
        let fsr = self.spi.fifo_status();
        let sr = self.spi.status().bits();
        let cc = self.spi.clock_configs();
        let sck_mhz = 132_000_000u64 as f64 / ((cc.sckdiv as u64 + 2) as f64) / 1e6;
        // `info!` (not `debug!`) so the diagnostics survive release builds.
        let eff_mbit_s = (n as f64 * 32.0) / (ms * 1e6);
        let setup_us = cyc_setup as f64 / (board::ARM_FREQUENCY as f64) * 1e6;
        let chunks = n / (MAX_TX_WORDS * 2);
        /*log::info!(
            "draw_pixels {n} px: total {ms:.3} ms (setup {setup_us:.1}us); wait_us min={min_wait_us} max={max_wait_us} first=[{s0},{s1},{s2},{s3},{s4},{s5}]; irq_fires={irq_fires} (expect {chunks}); sckdiv={sd} => sck={sck_mhz:.2} MHz; eff={eff_mbit_s:.2} Mbit/s; sr=0x{sr_val:08x} tx={tx}/{tmc} rx={rx}/{rmc}",
            s0 = sample[0],
            s1 = sample[1],
            s2 = sample[2],
            s3 = sample[3],
            s4 = sample[4],
            s5 = sample[5],
            sd = cc.sckdiv,
            sr_val = sr,
            tx = fsr.txcount,
            tmc = fsr.txcap,
            rx = fsr.rxcount,
            rmc = fsr.rxcap,
        );*/
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
