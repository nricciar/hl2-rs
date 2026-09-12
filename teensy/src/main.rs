//! Teensy 4.1 → Hermes Lite 2 SDR panadapter.
//!
//! RTIC v2, single-core, two tasks:
//!
//!   * `radio_task` — DHCP, HL2 discovery, START/TUNE/LNA, EP6 receive,
//!     publish 320-bin spectrum per frame to `hl2_teensy::shared`.
//!   * `render`     — blits the latest row to the ILI9341 waterfall
//!     (shift-and-blit) and redraws state / peer-IP / freq status text.
//!
//! Both tasks run at the same priority and yield cooperatively. Shared
//! spectrum rows are copied in a bounded critical section.
//!
//! DWT measures short intervals; SysTick provides the network clock.
//! SysTick (via `rtic_monotonics::systick`) is used to yield back to the
//! RTIC scheduler with `Systick::delay(...).await`.

#![no_std]
#![no_main]

use rtic_monotonics::Monotonic;
use rtic_monotonics::systick::Systick;

extern crate alloc;
use alloc::alloc::{GlobalAlloc, Layout};
use core::cell::RefCell;
use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};
use cortex_m::interrupt::Mutex;

/// Startup-only heap. Deallocation is a no-op; streaming must reuse buffers.
const HEAP_SIZE: usize = 160 * 1024;
// CPU-only allocations can use OCRAM; reserve DTCM for DMA buffers and stacks.
#[unsafe(link_section = ".uninit.heap")]
static mut BUMP_BUF: core::mem::MaybeUninit<[u8; HEAP_SIZE]> = core::mem::MaybeUninit::uninit();
static BUMP_OFF: AtomicUsize = AtomicUsize::new(0);

static POLLER: Mutex<RefCell<Option<imxrt_log::Poller>>> = Mutex::new(RefCell::new(None));

/// Call `f` with the log `Poller` (no-op before `init` publishes it).
fn with_poller<F: FnOnce(&mut imxrt_log::Poller)>(f: F) {
    // Move ownership out while polling so interrupts remain enabled and a
    // reentrant panic cannot access the same poller.
    let poller = cortex_m::interrupt::free(|cs| POLLER.borrow(cs).borrow_mut().take());
    if let Some(mut poller) = poller {
        f(&mut poller);
        cortex_m::interrupt::free(|cs| POLLER.borrow(cs).replace(Some(poller)));
    }
}

/// Drive USB logging, if initialization has completed.
fn poll_log() {
    with_poller(|p| p.poll());
}

/// Busy-wait `us` microseconds using the DWT cycle counter. Safe from the
/// panic handler (no `Systick::delay` there — it's `.await`-based).
fn busy_wait_us(us: u32) {
    let start = cortex_m::peripheral::DWT::cycle_count();
    let need = (us as u64) * (teensy4_bsp::board::ARM_FREQUENCY as u64 / 1_000_000);
    while (cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) as u64) < need {
        core::hint::spin_loop();
    }
}

/// Attempt to flush the panic log before blinking SOS.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    let bump_off = BUMP_OFF.load(Ordering::Relaxed);
    log::error!("[PANIC] heap={bump_off}/{HEAP_SIZE} | {info}");
    for _ in 0..200 {
        poll_log();
        busy_wait_us(20);
    }
    teensy4_panic::sos()
}

struct BumpAllocator;

// SAFETY: the atomic cursor reserves disjoint ranges of the mutable arena.
// No references to the arena are created, and allocations are never freed.
unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let align = layout.align();
        let base = ptr::addr_of_mut!(BUMP_BUF).cast::<u8>();
        let mut off = BUMP_OFF.load(Ordering::Acquire);
        loop {
            // Align the address, not just the offset: the arena is byte-aligned.
            let Some(address) = (base as usize)
                .checked_add(off)
                .and_then(|address| address.checked_add(align - 1))
            else {
                return ptr::null_mut();
            };
            let aligned_off = (address & !(align - 1)) - base as usize;
            let Some(next_off) = aligned_off.checked_add(layout.size()) else {
                return ptr::null_mut();
            };
            if next_off > HEAP_SIZE {
                return ptr::null_mut();
            }
            match BUMP_OFF.compare_exchange_weak(off, next_off, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return unsafe { base.add(aligned_off) },
                Err(actual) => off = actual,
            }
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { self.alloc(layout) };
        if !p.is_null() {
            unsafe { ptr::write_bytes(p, 0, layout.size()) };
        }
        p
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static GLOBAL: BumpAllocator = BumpAllocator;

/// DWT cycle-counter helper.
fn cycles_now() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}

/// Elapsed milliseconds for intervals shorter than one DWT wrap (~7 seconds).
fn ms_since(start: u32, now: u32) -> u64 {
    let diff = now.wrapping_sub(start);
    (diff as u64 * 1_000) / (teensy4_bsp::board::ARM_FREQUENCY as u64)
}

/// Current time in milliseconds, from the SysTick monotonic.
fn now_millis() -> i64 {
    Systick::now().ticks() as i64
}

#[rtic::app(device = teensy4_bsp, peripherals = true, dispatchers = [KPP])]
mod app {
    use bsp::board;
    use teensy4_bsp as bsp;

    use super::{POLLER, cycles_now, ms_since, now_millis, poll_log};
    use cortex_m::peripheral::DWT;
    use hl2_teensy::{display, radio, shared, spectrum};
    use imxrt_log as logging;
    use rtic_monotonics::systick::ExtU64;
    use rtic_monotonics::systick::Systick;

    /// DMA0-15 completion IRQ. The `render` task `.await`s the LPSPI
    /// `dma_write` future on channel 0, which stores its waker on the
    /// `imxrt-dma` channel before enabling it. When the eDMA sets the
    /// channel's `DONE` bit the `INTMAJOR` request fires this vector; the
    /// handler routes it into the driver's `on_interrupt`, which clears the
    /// flag (`DONE` + `INT`) and wakes the parked `render` task. The RTIC
    /// executor then re-polls `render`, which observes `is_complete()` and
    /// resumes — freeing the CPU to `radio_task` meanwhile.
    /// Diagnostic counter incremented in the `dma_irq` ISR. Lets us count
    /// how many DMA completion interrupts actually fired during a blit —
    /// if we only see ~15 IRQs instead of 150, most chunks never completed
    /// (or completed so fast the IRQ coalesced).
    #[task(binds = DMA0_DMA16, priority = 3)]
    fn dma_irq(_cx: dma_irq::Context) {
        shared::irq_fires_inc();
        // Safety: channel 0 is exclusively owned by the `render` task's
        // `DmaDisplay` (`init` hands it `dma[0]`), so no concurrent
        // `on_interrupt` can race this one. Only channels 0..=15 share the
        // `DMA0_DMA16` vector; we only ever arm channel 0 here.
        unsafe {
            bsp::hal::dma::DMA.on_interrupt(0);
        }
    }

    /// Task-local type alias for the ILI9341 panel.
    type Panel = display::driver::Display;

    /// RGB565 waterfall, newest row at the top. Stored outside the task stack.
    static WF: static_cell::ConstStaticCell<[u16; display::WF_COLS * display::WF_ROWS]> =
        static_cell::ConstStaticCell::new([0u16; display::WF_COLS * display::WF_ROWS]);

    #[shared]
    struct Shared {}

    #[local]
    struct Local {}

    #[init]
    fn init(cx: init::Context) -> (Shared, Local) {
        let board::Resources {
            mut gpio2,
            pins,
            usb,
            lpspi4,
            ccm,
            ccm_analog,
            iomuxc_gpr,
            mut dma,
            ..
        } = board::t41(cx.device);

        // eDMA channel for the ILI9341 pixel blit. USB (imxrt-usbd) on this
        // chip is CPU-polled by the logger, so all 32 channels are free;
        // channel 0 is the smallest index the RT1060's eDMA can address on
        // any bus domain that can reach both OCRAM (source) and LPSPI4 TDR
        // (destination).
        let display_dma = dma[0].take().expect("dma channel 0 available");

        let poller = logging::log::usbd(usb, logging::Interrupts::Disabled).unwrap();
        cortex_m::interrupt::free(|cs| POLLER.borrow(cs).replace(Some(poller)));

        // DWT cycle counter (DwtDelay + MDIO timeouts + ms_since helper).
        let mut dcb = cx.core.DCB;
        dcb.enable_trace();
        let mut dwt = cx.core.DWT;
        dwt.enable_cycle_counter();

        Systick::start(
            cx.core.SYST,
            board::ARM_FREQUENCY,
            rtic_monotonics::create_systick_token!(),
        );

        // Pins: CS=p10, DC=p9 (GPIO2); LPSPI4 SDO=p11, SDI=p12, SCK=p13.
        //
        // Keep SPI fast enough that full-band blits fit within the watchdog.
        let cs = gpio2.output(pins.p10).expect("p10 is GPIO2");
        let dc = gpio2.output(pins.p9).expect("p9 is GPIO2");
        let mut spi: board::Lpspi = board::lpspi(
            lpspi4,
            board::LpspiPins {
                sdo: pins.p11,
                sdi: pins.p12,
                sck: pins.p13,
            },
            33_000_000,
        );
        // Override: the HAL clamps sckdiv to >=4 (22 MHz). Push to 33 MHz
        // (sckdiv=2: 132/(2+2)). ILI9341 max SPI is 50 MHz so this is safe.
        spi.disabled(|d| {
            d.set_clock_configs(bsp::hal::lpspi::ClockConfigs {
                sckdiv: 2,
                dbt: 0,
                pcssck: 0,
                sckpcs: 0,
            })
        });

        // Construct the DMA-accelerated panel *before* the ILI9341 driver
        // moves `spi` into itself, so `DmaDisplay::new` can bitwise-copy
        // the `board::Lpspi` + CS pin out of a temporary `DisplaySpi`
        // (which then keeps its own copy for the boot splash + status text).
        // Build the DMA handle first: it bitwise-copies the LPSPI + CS + DC
        // pins from their `&` references. The originals are then moved into
        // the blocking `Display` (which owns them for the boot splash +
        // status text).
        let dma = display::driver::DmaDisplay::new(&spi, &cs, &dc, display_dma);
        let mut panel = display::driver::new_display(spi, cs, dc).expect("ILI9341 init");
        display::driver::fill_rect(&mut panel, 0, 0, 320, 240, 0xF800);
        display::driver::draw_text(&mut panel, 10, 10, 2, "HL2 TEENSY 4.1", 0x0000, 0xF800);

        let _ = render::spawn(panel, dma);
        let _ = radio_task::spawn(ccm, ccm_analog, iomuxc_gpr);

        (Shared {}, Local {})
    }

    #[task]
    async fn radio_task(
        _cx: radio_task::Context,
        mut ccm: bsp::ral::ccm::CCM,
        mut ccm_analog: bsp::ral::ccm_analog::CCM_ANALOG,
        mut iomuxc_gpr: bsp::ral::iomuxc_gpr::IOMUXC_GPR,
    ) {
        // Let USB-serial enumerate before MDIO.
        Systick::delay(2_000.millis()).await;
        poll_log();

        shared::set_state(shared::STATE_WAITING_IP);

        // 1. Bring up the DP83825 + ENET MAC.
        let mut delay = display::driver::DwtDelay;
        let mut device = match hl2_teensy::ethernet::Ethernet::new(
            &mut ccm,
            &mut ccm_analog,
            &mut iomuxc_gpr,
            &mut delay,
            &radio::control::MAC,
        ) {
            Ok(d) => d,
            Err(e) => {
                log::error!("Ethernet init failed: {e}");
                shared::set_state(shared::STATE_ERROR);
                return;
            }
        };

        // 2. Build smoltcp Interface + SocketSet + socket handles.
        let (mut iface, mut sockets, handles) = radio::control::build_iface_and_sockets(
            &mut device,
            radio::control::MAC,
            smoltcp::time::Instant::from_millis(now_millis()),
        );

        // 3. Pipeline.
        let mut pipeline = match spectrum::Pipeline::new() {
            Ok(p) => p,
            Err(e) => {
                log::error!("pipeline init: {e}");
                shared::set_state(shared::STATE_ERROR);
                return;
            }
        };
        let mut rx = radio::Rx::new();

        // 4. Wait for IP. Poll every ~500 ms; also link-check.
        let mut link_up = false;
        let mut last_link = cycles_now();
        let mut last_dhcp = cycles_now();

        // Poller needs a mutable reference to the socket set + iface for
        // each call; we build the handle at the start.
        let mut handle = radio::control::RadioHandle::new(
            radio::control::Radio::new(),
            &mut device,
            &mut iface,
            &mut sockets,
            &handles,
        );

        loop {
            let now_c = cycles_now();
            if !link_up && ms_since(last_link, now_c) >= 500 {
                last_link = now_c;
                match handle.dev.link_up() {
                    Ok(up) => {
                        if up != link_up {
                            link_up = up;
                            log::info!("link: {}", if up { "up" } else { "down" });
                        }
                    }
                    Err(e) => log::error!("link check: {e}"),
                }
            }
            if link_up && ms_since(last_dhcp, now_c) >= 500 {
                last_dhcp = now_c;
                handle.set_now(smoltcp::time::Instant::from_millis(now_millis()));
                let _configured = handle.poll_dhcp();
                poll_log();
            }
            if handle.is_configured() {
                if let Some(ip) = handle.our_ip() {
                    log::info!("IP acquired: {ip}");
                    break;
                }
            }
            Systick::delay(10.millis()).await;
            poll_log();
        }

        // 5. Discovery (500 ms cadence until a valid reply).
        shared::set_state(shared::STATE_DISCOVERING);
        let mut last_disc = cycles_now();
        'discovery: loop {
            handle.set_now(smoltcp::time::Instant::from_millis(now_millis()));
            handle.pump();
            let mut dgram = [0u8; hl2::protocol::DISCOVERY_RESPONSE_SIZE + 16];
            // Drain until empty; a valid reply ends the loop.
            while let Some((n, src)) = handle.recv(&mut dgram) {
                if n >= hl2::protocol::DISCOVERY_RESPONSE_SIZE {
                    if let Some(info) = handle.try_discovery(&dgram[..n]) {
                        // Discovery's stored IP can differ from its current DHCP lease.
                        let ip = src;
                        log::info!(
                            "HL2 FOUND ip={} mac={:02x?} rx={} 16bit={} sending={}",
                            ip,
                            info.mac,
                            info.rx_count,
                            info.sample_16bit,
                            info.is_sending,
                        );
                        handle.set_peer(ip);
                        shared::set_peer(ip);
                        break 'discovery;
                    }
                }
                log::info!("discovery: {n} B from {src}; not HL2");
            }
            let now_c = cycles_now();
            if ms_since(last_disc, now_c) >= 500 {
                last_disc = now_c;
                handle.send_discovery();
            }
            Systick::delay(5.millis()).await;
            poll_log();
        }

        // 6. START.
        shared::set_state(shared::STATE_STARTING);
        handle.send_stop();
        Systick::delay(150.millis()).await;
        poll_log();
        handle.set_now(smoltcp::time::Instant::from_millis(now_millis()));
        handle.drain();
        handle.send_start();

        // 7. LNA + NCO.
        shared::set_state(shared::STATE_TUNING);
        handle.send_lna(radio::control::LNA_GAIN_DB);
        handle.send_tune(radio::control::TUNE_HZ);

        // 8. Streaming.
        shared::set_state(shared::STATE_STREAMING);
        log::info!("EP6 streaming at 96 kSps, NCO 7.074 MHz (RX1)");
        poll_log();

        let mut last_keepalive = cycles_now();
        // S-meter + per-stage CPU% instrumentation.
        //   * `last_row_pub` — the frame_seq at which we last republished the
        //     row. The S-level is computed from the *same* `mags()` snapshot we
        //     hand to the render task, so the meter and the waterfall agree.
        //   * `cpu_*` — three workloads reported against the *same* 1-second
        //     shared denominator (the one rolling wall window, in cycles):
        //       * `demod` — EP6 baseband parse + the virtual-USB demod DSP,
        //                   attributed by `Rx`'s DWT taps.
        //       * `fft`   — the 2048-commit spectrum / S-meter window, also
        //                   attributed by `Rx`'s DWT taps.
        //       * `lcd`   — the render task's waterfall shift + colourise,
        //                   accumulated on `shared::LCD_CYCLES` (the render
        //                   task appends; we read + zero it here on the same
        //                   second boundary).
        //     Because a single core can't run two things at once,
        //     demod + fft + lcd cannot sum past 100 — the whole point is
        //     seeing *which* slice ate the CPU, not three independent 100s.
        let mut last_row_pub: u32 = pipeline.frame_seq();
        let mut cpu_start: u32 = cycles_now();
        let mut window_start: u64 = rx.demod_cycles();
        let mut fft_start: u64 = rx.fft_cycles();

        loop {
            handle.set_now(smoltcp::time::Instant::from_millis(now_millis()));

            // Drain a bounded batch from the MAC as well as the UDP socket.
            // One ingress poll per millisecond cannot keep up at 96 kSps.
            let mut dgram = [0u8; hl2::protocol::DATA_PACKET_SIZE + 64];
            for _ in 0..64 {
                handle.pump();
                while let Some((n, src)) = handle.recv(&mut dgram) {
                    if src == handle.peer() && rx.feed(&dgram[..n], &mut pipeline) > 0 {
                        shared::bump_frames();
                    }
                }
            }

            // Publish only completed FFT frames, coalescing to the latest row.
            let cur_seq = pipeline.frame_seq();
            if cur_seq != last_row_pub {
                last_row_pub = cur_seq;
                shared::publish(pipeline.mags());
                // The S-meter is a spectrum consumer (PROTOCOL.md §16.3e) — the
                // same 320-bin row the render task blits. Compute it here, off
                // the render path, so the render task just paints.
                let m = hl2_teensy::smeter::compute(pipeline.mags());
                shared::set_slevel(m.sunits, m.margin_db);
            }

            // Roll the CPU window forward every ~1 s and publish all three
            // shares against the *same* shared wall (the one rolling window
            // in cycles). The demod / fft deltas come from `Rx`'s DWT taps;
            // the lcd delta is accumulated cross-task on `shared::LCD_CYCLES`.
            // On one core they cannot overlap, so each is ≤ 100 and the sum
            // is the honest total CPU busy fraction (≤ 100).
            let now_c = cycles_now();
            if ms_since(cpu_start, now_c) >= 1_000 {
                // Shared wall = the rolling window's total elapsed cycles
                // (the 1-second denominator both tasks' work is measured
                // against — *not* each task's own busy wall, which is how
                // the three 100s happened).
                let wall = now_c as u64 - cpu_start as u64;
                shared::publish_cpu_stages(
                    rx.demod_cycles() - window_start,
                    rx.fft_cycles() - fft_start,
                    shared::take_lcd_cycles(),
                    wall,
                );
                window_start = rx.demod_cycles();
                fft_start = rx.fft_cycles();
                cpu_start = now_c;
            }

            if ms_since(last_keepalive, now_c) >= radio::control::KEEPALIVE_INTERVAL_MS_CONST {
                last_keepalive = now_c;
                handle.send_keepalive();
            }

            Systick::delay(1.millis()).await;
            poll_log();
        }
    }

    /// DMA-accelerated waterfall blit, task-local (owned by the `render`
    /// task). See `display::driver::DmaDisplay` for the bitwise-copy
    /// safety argument behind sharing the LPSPI + GPIO bits with the
    /// blocking `Display` (which owns the original handles).
    type Dma = display::driver::DmaDisplay;

    /// Priority 1 (> the default 0 used by `radio_task`), so that when a
    /// DMA chunk's completion IRQ wakes a suspended `render` continuation
    /// the RTIC scheduler is allowed to **preempt** `radio_task` — letting
    /// the ~150 × 125 µs blit chunks run back-to-back without queueing
    /// behind `radio_task`'s ~200 ms blocking FFT/demod/AGC slices.
    /// Without this inversion fix, every render `.await` that landed
    /// inside one of those blocking slices stalled ~200 ms.
    #[task(priority = 1)]
    async fn render(_cx: render::Context, mut panel: Panel, mut dma: Dma) {
        let fb = WF.take();

        // Boot: fill black (covers the red "HL2 TEENSY" splash), draw the
        // status bar once.
        display::driver::fill_rect(&mut panel, 0, 0, 320, 240, 0x0000);
        display::driver::draw_text(
            &mut panel,
            6,
            display::WF_ROWS as u16 + 76,
            1,
            "7.074MHz RX1 96k",
            0xF800,
            0x0000,
        );
        let mut status_label = display::driver::TextLine::<9>::new();
        let mut status_detail = display::driver::TextLine::<32>::new();
        let mut status_meter = display::driver::TextLine::<16>::new();
        let mut status_cpu = display::driver::TextLine::<32>::new();

        let mut painted_seq: u32 = 0;
        let mut last_state: u32 = u32::MAX;
        let mut last_peer: u32 = u32::MAX;
        let mut last_status_ms: i64 = now_millis();

        loop {
            poll_log();

            let (seq, row) = shared::latest();
            let st = shared::state();
            let peer_u32 = shared::peer()
                .map(|ip| {
                    let o = ip.octets();
                    ((o[0] as u32) << 24)
                        | ((o[1] as u32) << 16)
                        | ((o[2] as u32) << 8)
                        | (o[3] as u32)
                })
                .unwrap_or(u32::MAX);

            // Shift in RAM, then repaint the band because every row has moved.
            if seq != painted_seq {
                let cols = display::WF_COLS;
                let rows = display::WF_ROWS;
                let total = cols * rows;

                // 1. shift down in RAM (newest → top row). The shift + the
                //    colourise below *is* the *lcd* CPU stage for the status
                //    readout — the DWT delta is appended to the shared
                //    accumulator the radio task drains on its 1-second rollover
                //    (same shared wall as demod / fft). The eDMA blit below is
                //    offloaded to the engine and is *not* counted as CPU.
                let c0 = DWT::cycle_count();
                fb.copy_within(..total - cols, cols);
                // 2. new row at the top.
                for (i, px) in fb[..cols].iter_mut().enumerate() {
                    *px = display::palette::bin_color(row.get(i).copied().unwrap_or(0u16));
                }
                shared::add_lcd_cycles(DWT::cycle_count() as u64 - c0 as u64);
                // 3. Blit the band via eDMA. This hands the pixel work over
                //    to the eDMA engine (which drives the LPSPI TDR while
                //    the panel clocks in the bytes), so the CPU is free to
                //    poll the radio's MAC ring between chunks — the exact
                //    "we're not stalling the network" win the DMA path was
                //    meant to buy.
                dma.draw_pixels(0, 0, (cols as u16) - 1, (rows as u16) - 1, &fb[..total])
                    .await;
                painted_seq = seq;
                // Let the radio drain its MAC ring before further SPI work.
                Systick::delay(1.millis()).await;
            }

            // Refresh status on changes and keep the RX counter live at 2 Hz.
            let now_ms = now_millis();
            let st_changed = st != last_state || peer_u32 != last_peer;
            let tick = (now_ms - last_status_ms) >= 500;
            if st_changed || tick {
                last_state = st;
                last_peer = peer_u32;
                last_status_ms = now_ms;
                status_redraw(
                    &mut panel,
                    &mut status_label,
                    &mut status_detail,
                    &mut status_meter,
                    &mut status_cpu,
                );
            }

            Systick::delay(2.millis()).await;
        }
    }

    /// Update only changed status characters, including the live RX counter
    /// and the per-stage CPU readout.
    fn status_redraw(
        panel: &mut Panel,
        status_label: &mut display::driver::TextLine<9>,
        status_detail: &mut display::driver::TextLine<32>,
        status_meter: &mut display::driver::TextLine<16>,
        status_cpu: &mut display::driver::TextLine<32>,
    ) {
        let label = match shared::state() {
            shared::STATE_WAITING_IP => "WAIT IP",
            shared::STATE_LINK => "LINK",
            shared::STATE_DISCOVERING => "DISCOVERY",
            shared::STATE_STARTING => "STARTING",
            shared::STATE_TUNING => "TUNING",
            shared::STATE_STREAMING => "STREAMING",
            shared::STATE_ERROR => "ERROR",
            _ => "?",
        };
        let wf_y = display::WF_ROWS as u16;
        status_label.update(panel, 6, wf_y + 4, label, 0xF800, 0x0000);
        use core::fmt::Write;
        let mut line = [0u8; 32];
        let mut w = radio::control::WriteBuf {
            target: &mut line,
            pos: 0,
        };
        if let Some(ip) = shared::peer() {
            write!(w, "{ip}").unwrap();
        } else {
            write!(w, "-").unwrap();
        }
        write!(w, " F {}", shared::frames()).unwrap();
        let line_s = core::str::from_utf8(&w.target[..w.pos]).unwrap();
        status_detail.update(panel, 6, wf_y + 30, line_s, 0xF800, 0x0000);

        // S-meter readout. The bar is `S1..S9` (one segment per unit), the
        // raw dB-over-floor margin to its right. The bar sits below the
        // state label / IP/F counter row; the CPU split is on its own row
        // further down (see `status_cpu` at the end of this fn).
        //
        // `fill_rect(display, x, y, w, h, color)` — the last 4 args are
        // width/height, NOT x2/y2.
        let s = shared::slevel();
        let margin = shared::smargin_db();
        let bar_y = wf_y + 14u16;
        let seg_w = 8u16;
        let seg_h = 8u16;
        let x0 = 6u16;
        for i in 0..9u16 {
            let x = x0 + (i as u32 * ((seg_w + 1) as u32)) as u16;
            let on = (i as u8 + 1) <= s;
            let fg: u16 = if on { 0xFFFF } else { 0x4020 };
            display::driver::fill_rect(panel, x, bar_y, seg_w, seg_h, fg);
        }

        // Right of the bar: "S{n} +{N} dB" (≤ 12 cells), same row as the bar.
        // The `TextLine<16>` is 96 px wide — plenty for "S9 +54 dB".
        let mut meter = [0u8; 16];
        let mut w2 = radio::control::WriteBuf {
            target: &mut meter,
            pos: 0,
        };
        if s == 0 {
            let _ = write!(w2, "S0 +{:.0} dB", margin);
        } else {
            let _ = write!(w2, "S{s} +{:.0} dB", margin);
        }
        let meter_s = core::str::from_utf8(&w2.target[..w2.pos]).unwrap();
        let meter_x = x0 + (9u16 * (seg_w + 1) as u16) + 2u16;
        status_meter.update(panel, meter_x, bar_y, meter_s, 0xF800, 0x0000);

        // CPU split, on its own row **below the frequency line** (the "7.074 MHz…"
        // line drawn at boot at `WF_ROWS + 76`). The three stages are the
        // pipeline's three workloads, each reported as a percent (≤ 100) on its
        // *own* task's rolling 1-second window:
        //   * D — demod: EP6 baseband parse + virtual-USB demod  (radio task)
        //   * F — fft:   the 2048-commit spectrum / S-meter window (radio task)
        //   * L — lcd:   the waterfall shift + colourise           (render task)
        // They run on one shared core, so the three can each read ~100 and
        // overlap — that's the honest "what's eating the CPU" number.
        let mut cpu = [0u8; 32];
        let mut w3 = radio::control::WriteBuf {
            target: &mut cpu,
            pos: 0,
        };
        let _ = write!(
            w3,
            "CPU D:{:03} F:{:03} L:{:03} %",
            shared::cpu_demod_pct(),
            shared::cpu_fft_pct(),
            shared::cpu_lcd_pct(),
        );
        let cpu_s = core::str::from_utf8(&w3.target[..w3.pos]).unwrap();
        status_cpu.update(
            panel,
            6,
            display::WF_ROWS as u16 + 85,
            cpu_s,
            0xF800,
            0x0000,
        );
    }
}
