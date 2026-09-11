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
    use hl2_teensy::{display, radio, shared, spectrum};
    use imxrt_log as logging;
    use rtic_monotonics::systick::ExtU64;
    use rtic_monotonics::systick::Systick;

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
            ..
        } = board::t41(cx.device);

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
        let spi: board::Lpspi = board::lpspi(
            lpspi4,
            board::LpspiPins {
                sdo: pins.p11,
                sdi: pins.p12,
                sck: pins.p13,
            },
            33_000_000,
        );

        // ILI9341 + red background so the operator knows the display is
        // alive before the radio comes up.
        let mut panel = display::driver::new_display(spi, cs, dc).expect("ILI9341 init");
        display::driver::fill_rect(&mut panel, 0, 0, 320, 240, 0xF800);
        display::driver::draw_text(&mut panel, 10, 10, 2, "HL2 TEENSY 4.1", 0x0000, 0xF800);

        let _ = render::spawn(panel);
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
        // S-meter + CPU% instrumentation.
        //   * `last_row_pub` — the frame_seq at which we last republished the
        //     row. The S-level is computed from the *same* `mags()` snapshot we
        //     hand to the render task, so the meter and the waterfall agree.
        //   * `cpu_*` — a rolling 1-second DWT window. `busy_cycles` is the
        //     time inside the pump + feed loop (the real per-DP work: MAC
        //     poll + EP6 parse + virtual-receiver demod + FFT commit); the
        //     `Systick::delay(1)` between iterations is the scheduler's free
        //     time and is *not* counted. `busy / wall × 100` is the radio
        //     task's CPU utilisation — the honest "we are running at CPU
        //     speed" number. Because a no-std bump heap never frees, the
        //     demod's per-emit buffers are amortised against persistent `Vec`s
        //     (see `hl2::receiver::AudioEngine`); if DSP grows, this climbs.
        let mut last_row_pub: u32 = pipeline.frame_seq();
        let mut cpu_start: u32 = cycles_now();
        let mut busy_cycles: u64 = 0;
        let mut wall_cycles: u64 = 0;

        loop {
            let iter_t0 = cycles_now();

            handle.set_now(smoltcp::time::Instant::from_millis(now_millis()));

            // Drain a bounded batch from the MAC as well as the UDP socket.
            // One ingress poll per millisecond cannot keep up at 96 kSps.
            let mut dgram = [0u8; hl2::protocol::DATA_PACKET_SIZE + 64];
            let busy_t0 = cycles_now();
            for _ in 0..64 {
                handle.pump();
                while let Some((n, src)) = handle.recv(&mut dgram) {
                    if src == handle.peer() && rx.feed(&dgram[..n], &mut pipeline) > 0 {
                        shared::bump_frames();
                    }
                }
            }
            busy_cycles += cycles_now() as u64 - busy_t0 as u64;

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

            // Roll the CPU% window forward every ~1 s and publish.
            let now_c = cycles_now();
            wall_cycles += now_c as u64 - iter_t0 as u64;
            if ms_since(cpu_start, now_c) >= 1_000 {
                let wall = wall_cycles;
                // `busy <= wall` always (the busy window is a sub-interval), so
                // the clamped ratio is naturally ≤ 100 and never wraps.
                let pct = if wall == 0 {
                    0
                } else {
                    (100u64.saturating_mul(busy_cycles)).min(wall) as u32
                };
                shared::set_cpu_pct(pct);
                busy_cycles = 0;
                wall_cycles = 0;
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

    #[task]
    async fn render(_cx: render::Context, mut panel: Panel) {
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

                // 1. shift down in RAM (newest → top row).
                fb.copy_within(..total - cols, cols);
                // 2. new row at the top.
                for (i, px) in fb[..cols].iter_mut().enumerate() {
                    *px = display::palette::bin_color(row.get(i).copied().unwrap_or(0u16));
                }
                panel
                    .draw_raw_slice(0, 0, (cols as u16) - 1, (rows as u16) - 1, &fb[..total])
                    .expect("waterfall draw");
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
                );
            }

            Systick::delay(2.millis()).await;
        }
    }

    /// Update only changed status characters, including the live RX counter.
    fn status_redraw(
        panel: &mut Panel,
        status_label: &mut display::driver::TextLine<9>,
        status_detail: &mut display::driver::TextLine<32>,
        status_meter: &mut display::driver::TextLine<16>,
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

        // S-meter + CPU% readout. The bar is `S1..S9` (one segment per unit);
        // below it, the raw dB margin + the DSP/network CPU% — the "we are
        // running at CPU speed" proof. The bar sits below the state label
        // (y = `wf_y + 14`) and the meter text one row further down
        // (y = `wf_y + 24`), so the left-side status label at `y = wf_y + 4`
        // and the IP / F counter at `y = wf_y + 30` stay clear.
        let s = shared::slevel();
        let margin = shared::smargin_db();
        let pct = shared::cpu_pct();

        // `fill_rect(display, x, y, w, h, color)` — the last 4 args are
        // width/height, NOT x2/y2. This was the source of the "big red box":
        // segment i=8 (x=78) had `w = x+seg_w-1 = 85` and `h = bar_y+seg_h-1 = 141`,
        // covering the IP text at (6, 150) and the state label at (6, 124).
        let bar_y = wf_y + 14u16;
        let seg_w = 8u16;
        let seg_h = 8u16;
        // Nine 8×8 segments spaced 1 px apart — 80 px wide, x = [6..85].
        let x0 = 6u16;
        for i in 0..9u16 {
            let x = x0 + (i as u32 * ((seg_w + 1) as u32)) as u16;
            let on = (i as u8 + 1) <= s;
            let fg: u16 = if on { 0xFFFF } else { 0x4020 };
            display::driver::fill_rect(panel, x, bar_y, seg_w, seg_h, fg);
        }

        // Right of the bar: "S{n} +{N} dB {pct} C" (≤ 14 cells), same row as the
        // bar. `TextLine<16>` is 96 px wide — more than enough for "S9 +54 dB 100 C".
        let mut meter = [0u8; 16];
        let mut w2 = radio::control::WriteBuf {
            target: &mut meter,
            pos: 0,
        };
        if s == 0 {
            let _ = write!(w2, "S0 +{:.0} dB {pct} C", margin);
        } else {
            let _ = write!(w2, "S{s} +{:.0} dB {pct} C", margin);
        }
        let meter_s = core::str::from_utf8(&w2.target[..w2.pos]).unwrap();
        let meter_x = x0 + (9u16 * (seg_w + 1) as u16) + 2u16;
        status_meter.update(panel, meter_x, bar_y, meter_s, 0xF800, 0x0000);
    }
}
