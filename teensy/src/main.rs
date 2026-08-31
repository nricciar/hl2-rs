//! Proof of concept: bring up an ILI9341 LCD over LPSPI4 and paint a
//! "HELLO WORLD" banner with color bands at boot.
//!
//! Pin map (all present on the LPSPI4 + GPIO2 pads):
//!   MOSI  = 11 (LPSPI4_SDO)   CS = 10
//!   MISO  = 12 (LPSPI4_SDI)   DC = 9
//!   SCK   = 13 (LPSPI4_SCK)
//!   RESET = not wired (pulled high on the display module)
//!
//! Note: pin 13 is also the onboard LED, but is shared as SCK here.
//!
//! Target a Teensy 4.1 (board::t41). Uses RTIC v2.

#![no_std]
#![no_main]

use teensy4_panic as _;

#[rtic::app(device = teensy4_bsp, peripherals = true, dispatchers = [KPP])]
mod app {
    use bsp::board;
    use bsp::hal;
    use teensy4_bsp as bsp;

    use core::convert::Infallible;

    use embedded_hal::delay::DelayNs;
    use embedded_hal::digital::{ErrorType as DigitalErrorType, OutputPin};
    use embedded_hal::spi::{ErrorType, Operation, SpiBus, SpiDevice};
    use imxrt_log as logging;

    use rtic_monotonics::systick::{Systick, *};
    use rtic_monotonics::Monotonic;

    const WIDTH: u16 = 320;
    const HEIGHT: u16 = 240;

    type Display = ili9341::Ili9341<
        display_interface_spi::SPIInterface<DisplaySpi, hal::gpio::Output>,
        NopPin,
    >;

    type LpspiError = hal::lpspi::LpspiError;

    fn dwt_cycles_now() -> u32 {
        cortex_m::peripheral::DWT::cycle_count()
    }

    /// Rough microsecond counter from the DWT cycle counter, for timing logs.
    fn us_now() -> u64 {
        dwt_cycles_now() as u64 / (board::ARM_FREQUENCY as u64 / 1_000_000)
    }

    /// Busy-wait delay built on the DWT cycle counter (core at `ARM_FREQUENCY`).
    struct DwtDelay {}

    impl DelayNs for DwtDelay {
        fn delay_ns(&mut self, ns: u32) {
            let hz = board::ARM_FREQUENCY as u128;
            // Cycles = ns * hz / 1e9. Use u128 so ns*hz can't overflow u64.
            let ticks = (ns as u128) * hz / 1_000_000_000u128;
            let start = dwt_cycles_now() as u64;
            while (dwt_cycles_now() as u64 - start) < ticks as u64 {
                core::hint::spin_loop();
            }
        }
    }

    /// Wrap the LPSPI driver (which only implements `SpiBus<u8>`) into a
    /// `SpiDevice`, toggling the software chip-select pin around each
    /// transaction. `display_interface_spi::SPIInterface` needs a `SpiDevice`.
    struct DisplaySpi {
        spi: board::Lpspi,
        cs: hal::gpio::Output,
    }

    impl ErrorType for DisplaySpi {
        type Error = LpspiError;
    }

    /// A no-op `OutputPin` standing in for the display's RESET line, which is
    /// not wired to Teensy and is held high by the module's external pull-up.
    /// `Ili9341::new` requires an `OutputPin` to "reset" the panel; the real
    /// reset happens via the software RESET command it also sends.
    struct NopPin;

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

    impl embedded_hal::digital::StatefulOutputPin for NopPin {
        fn is_set_high(&mut self) -> Result<bool, Self::Error> {
            Ok(true)
        }
        fn is_set_low(&mut self) -> Result<bool, Self::Error> {
            Ok(false)
        }
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
        fn transaction(
            &mut self,
            operations: &mut [Operation<'_, u8>],
        ) -> Result<(), Self::Error> {
            self.cs.set_low();
            let mut result: Result<(), LpspiError> = Ok(());
            for op in operations.iter_mut() {
                result = match op {
                    Operation::Read(buf) => SpiBus::<u8>::read(&mut self.spi, buf).map(|_| ()),
                    Operation::Write(buf) => SpiBus::<u8>::write(&mut self.spi, buf),
                    Operation::Transfer(read, write) => SpiBus::<u8>::transfer(&mut self.spi, read, write),
                    Operation::TransferInPlace(buf) => SpiBus::<u8>::transfer_in_place(&mut self.spi, buf),
                    Operation::DelayNs(ns) => {
                        let mut d = DwtDelay {};
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

    /// There are no resources shared across tasks.
    #[shared]
    struct Shared {}

    /// Resources local to individual tasks.
    #[local]
    struct Local {
        poller: logging::Poller,
    }

    #[init]
    fn init(cx: init::Context) -> (Shared, Local) {
        let board::Resources {
            mut gpio2,
            pins,
            usb,
            lpspi4,
            ..
        } = board::t41(cx.device);

        let poller = logging::log::usbd(usb, logging::Interrupts::Enabled).unwrap();

        // FIRST things we do in init (before anything else that uses DwtDelay):
        // enable the DWT cycle counter. On Cortex-M7 DWT.CYCCNT is gated by
        // DCB.C_DEBUGEN (enable_trace) AND DWT.CTRL.CYCCNTENA. Without both,
        // the counter never ticks and our DelayNs implementation hangs forever.
        let mut dcb = cx.core.DCB;
        dcb.enable_trace();
        let mut dwt = cx.core.DWT;
        dwt.enable_cycle_counter();

        Systick::start(
            cx.core.SYST,
            board::ARM_FREQUENCY,
            rtic_monotonics::create_systick_token!(),
        );

        log::info!("DWT enabled: {} Hz, systick running", board::ARM_FREQUENCY);

        // Control lines (all on GPIO2): CS=10, DC=9. RESET is not wired.
        let cs = gpio2.output(pins.p10).expect("p10 is GPIO2");
        let dc = gpio2.output(pins.p9).expect("p9 is GPIO2");

        // SPI on LPSPI4: SDO/MOSI=11, SDI/MISO=12, SCK=13. SPI MODE 0 (default).
        let spi: board::Lpspi = board::lpspi(
            lpspi4,
            board::LpspiPins {
                sdo: pins.p11,
                sdi: pins.p12,
                sck: pins.p13,
            },
            8_000_000,
        );

        let iface = display_interface_spi::SPIInterface::<DisplaySpi, hal::gpio::Output>::new(
            DisplaySpi { spi, cs },
            dc,
        );

        let mut delay = DwtDelay {};

        // --- Sanity check: is DWT.CYCCNT actually ticking at ARM_FREQUENCY? ---
        // We ask DwtDelay to wait 50 ms, then measure with Systick (which the
        // BSP starts from ARM_FREQUENCY, so it's an independent clock source).
        // If the two disagree by a lot, DWT is running slower than we assume
        // and every DelayNs call is being stretched.
        {
            // Cross-check: DwtDelay asks for 50 ms; measure the real time with
            // Systick (a *different* clock source). Systick runs at 600 MHz
            // (core clock), so 50 ms = 30_000_000 ticks. We want to see a value
            // close to 50 ms — if DWT were broken (running slower than that,
            // or not running at all) the measured time would be much larger.
            let systick_start = Systick::now();
            delay.delay_ms(50);
            let systick_elapsed = Systick::now() - systick_start;
            let elapsed_ticks = systick_elapsed.ticks();
            let elapsed_ms = elapsed_ticks / 600_000;
            log::info!(
                "Delay cross-check: DwtDelay(50 ms) took {elapsed_ms} ms on systick clock"
            );
        }

        log::info!("Ili9341::new starting (t0 = {} µs)", us_now());
        let mut display = ili9341::Ili9341::new(
            iface,
            NopPin,
            &mut delay,
            ili9341::Orientation::Landscape,
            ili9341::DisplaySize240x320,
        )
        .expect("ILI9341 init");
        log::info!("Ili9341::new done (t = {} µs)", us_now());

        // Some ILI9341 panels power up with inverted colors (INVO on) or a low
        // brightness — set both explicitly so a successful fill is guaranteed
        // to be visible.
        let _ = display.invert_mode(ili9341::ModeState::Off);
        let _ = display.brightness(255);
        log::info!("Ili9341 settings applied (t = {} µs)", us_now());

        {
            let t0 = us_now();
            let total = (WIDTH as usize) * (HEIGHT as usize);
            display
                .draw_raw_iter(
                    0,
                    0,
                    WIDTH - 1,
                    HEIGHT - 1,
                    core::iter::repeat(0xF800).take(total),
                )
                .expect("red screen draw");
            let dt = us_now() - t0;
            log::info!(
                "Painted full-screen red in {dt} µs ({} px; ~{} bytes/s effective)",
                total,
                (total * 2) as u64 * 1_000_000 / dt.max(1)
            );
        }

        {
            let t0 = us_now();
            paint_banner(&mut display);
            let dt = us_now() - t0;
            log::info!("Painted banner in {dt} µs");
        }
        log::info!("Banner painted");

        hello_world::spawn().unwrap();
        (Shared {}, Local { poller })
    }

    /// A 5x7 bitmap for one glyph (7 rows, 5 cols, MSB = leftmost bit).
    fn glyph(ch: char) -> [u8; 7] {
        match ch {
            'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
            'E' => [0b11111, 0b10000, 0b10000, 0b11111, 0b10000, 0b10000, 0b11111],
            'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
            'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
            'W' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011],
            'R' => [0b11100, 0b10010, 0b10010, 0b11100, 0b10100, 0b10010, 0b10001],
            'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
            _ => [0u8; 7],
        }
    }

    fn fill_rect(display: &mut Display, x: u16, y: u16, w: u16, h: u16, color: u16) {
        let count = (w as usize) * (h as usize);
        display
            .draw_raw_iter(
                x,
                y,
                x.saturating_add(w) - 1,
                y.saturating_add(h) - 1,
                core::iter::repeat(color).take(count),
            )
            .expect("fill_rect draw");
    }

    /// Draw `text` starting at (x0, y0) with each source pixel scaled to a
    /// `scale` x `scale` block.
    fn draw_text(
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
            x += 6 * scale; // 5 glyph columns + 1 gap
        }
    }

    /// Paint the welcome banner: a dark background, colored side bands, and the
    /// "HELLO WORLD" text.
    fn paint_banner(display: &mut Display) {
        // Background.
        fill_rect(display, 0, 0, WIDTH, HEIGHT, 0x001E);

        // Colored vertical bands down the left and right edges.
        let bands: [u16; 6] = [0xF800, 0x07E0, 0x001F, 0xFFE0, 0xF81F, 0x1F00];
        let band_w = 12u16;
        let seg = HEIGHT / 6;
        for (i, color) in bands.iter().enumerate() {
            let yy = (i as u16) * seg;
            fill_rect(display, 0, yy, band_w, seg, *color);
            fill_rect(display, WIDTH - band_w, yy, band_w, seg, *color);
        }

        // "HELLO" / "WORLD" centered.
        let scale = 5u16;
        let line_w = (5 * 6 - 1) * scale;
        let x0 = (WIDTH / 2) - line_w / 2;
        draw_text(display, x0, 70, scale, "HELLO", 0xFFFF, 0x001E);
        draw_text(display, x0, 140, scale, "WORLD", 0xFFFF, 0x001E);
    }

    /// Periodically log over USB so we can see the radio's "companion" device
    /// is alive. This is the actual "hello world" heartbeat.
    #[task]
    async fn hello_world(_cx: hello_world::Context) {
        let mut n = 0u32;
        loop {
            log::info!("Hello from your Teensy 4.1 — the ILI9341 is up and running! ({n})");
            n = n.wrapping_add(1);
            Systick::delay(1_000.millis()).await;
        }
    }

    #[task(binds = USB_OTG1, local = [poller])]
    fn log_over_usb(cx: log_over_usb::Context) {
        cx.local.poller.poll();
    }
}
