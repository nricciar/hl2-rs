//! Teensy 4.1 → Hermes Lite 2 SDR panadapter.
//!
//! RTIC v2, single-core, two tasks:
//!
//!   * `radio_task` — DHCP, HL2 discovery, START/TUNE/LNA, EP6 receive,
//!     publish 320-bin spectrum per frame to `hl2_teensy::shared`.
//!   * `render`     — blits the latest row to the ILI9341 waterfall
//!     (shift-and-blit) and redraws state / peer-IP / freq status text.
//!
//! Single-core cooperative scheduling: `hl2_teensy::shared` is a pair of
//! `static` byte arrays guarded by atomic sequence numbers (no locks).
//!
//! DWT cycle counter is used for all internal timing — 1 cycle = 1/600 MHz
//! seconds, so "every 40 ms" is a delta of `40 * ARM_FREQ / 1000`.
//! SysTick (via `rtic_monotonics::systick`) is used to yield back to the
//! RTIC scheduler with `Systick::delay(...).await`.

#![no_std]
#![no_main]

use rtic_monotonics::Monotonic;
use rtic_monotonics::systick::Systick;

extern crate alloc;
use alloc::alloc::{GlobalAlloc, Layout};
use core::ptr;
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// Heap arena. 160 KiB covers `pipeline` (~96 KiB for the twiddle table +
/// window + accumulators), the `Rx` `Vec`s, and smoltcp scratch. The
/// waterfall framebuffer is a `static`.
static BUMP_BUF: [u8; 160 * 1024] = [0u8; 160 * 1024];
static BUMP_OFF: AtomicUsize = AtomicUsize::new(0);

/// Global view of the log poller, reachable from `#[panic_handler]`.
/// `init` stores the RTIC-owned `Poller` here once after construction; from
/// then on every `poll_log()` in the tasks and the flush in the panic
/// handler drives the *same* object. `Poller` is `!Send`/`!Sync` but is
/// only ever touched from one thread (cooperative scheduling) or from the
/// panic handler after both tasks have stopped being scheduled.
static POLLER_CELL: static_cell::StaticCell<imxrt_log::Poller> = static_cell::StaticCell::new();
// `AtomicPtr` (`Copy` + `Sync`) holding the address of the `Poller` that
// `POLLER_CELL` owns — so the cooperative tasks and the panic handler can
// each get a `&mut` to it without moving anything. `Poller` is `!Sync`,
// but single-core cooperative scheduling (one task at a time, the panic
// handler only after the tasks stop) keeps the aliasing sound.
static POLLER_PTR: AtomicPtr<imxrt_log::Poller> =
    AtomicPtr::new(core::ptr::null_mut());

/// Call `f` with the log `Poller` (no-op before `init` publishes it).
fn with_poller<F: FnOnce(&mut imxrt_log::Poller)>(f: F) {
    let raw = POLLER_PTR.load(Ordering::Acquire);
    if !raw.is_null() {
        f(unsafe { &mut *raw });
    }
}

/// Drive the log poller once (no-op before `init` publishes the reference).
/// Used by the cooperative tasks in place of the old `cx.shared.poller`.
fn poll_log() {
    with_poller(|p| p.poll());
}

/// Busy-wait `us` microseconds using the DWT cycle counter. Safe from the
/// panic handler (no `Systick::delay` there — it's `.await`-based).
fn busy_wait_us(us: u32) {
    let start = cortex_m::peripheral::DWT::cycle_count();
    let need = (us as u64) * (teensy4_bsp::board::ARM_FREQUENCY as u64 / 1_000_000);
    let now = || cortex_m::peripheral::DWT::cycle_count() as u64;
    while now().wrapping_sub(start as u64) < need {
        core::hint::spin_loop();
    }
}

/// `#[panic_handler]`: log the panic message, flush the USB log ring in a
/// bounded loop so the message actually reaches the host, then hand off to
/// `teensy4_panic::sos()` for the LED S.O.S. blink. `default-features =
/// false` on `teensy4-panic` disables its built-in handler — that one
/// `sos()`ed *immediately* after `log::error!`, so the message sat in the
/// 8 KB `bbqueue` ring and was never clocked out to the CDC-ACM endpoint.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    use core::fmt::Write;

    // Format the `PanicInfo` into a local stack buffer (a panic handler must
    // never allocate; 256 B on the stack is fine — the handler runs on the
    // faulting task's original stack, which is still valid). 256 B is large
    // enough for `{:?}` of a `PanicInfo` to carry the `location: {file, line,
    // col}` through, which is what we need to name the failing alloc site.
    let mut buf = [0u8; 256];
    struct W<'a> { buf: &'a mut [u8], pos: usize }
    impl<'a> core::fmt::Write for W<'a> {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let b = s.as_bytes();
            let room = self.buf.len().saturating_sub(self.pos);
            let n = b.len().min(room);
            let dst = &mut self.buf[self.pos..self.pos + n];
            dst.copy_from_slice(&b[..n]);
            self.pos += n;
            Ok(())
        }
    }
    let mut w = W { buf: &mut buf, pos: 0 };
    let _ = write!(w, "{:?}", info);
    let pos = w.pos;
    let msg = core::str::from_utf8(&buf[..pos]).unwrap_or("<non-utf8 panic>");
    let bump_off = BUMP_OFF.load(Ordering::Relaxed);

    // `log::error!` never panics on a full ring (the frontend's `fmt::Write`
    // impl returns `Err` — no `expect!`), so this is safe in a panic
    // handler.
    log::error!("[PANIC] bump_used=0x{:05x}/0x{:05x} | {}", bump_off, BUMP_BUF.len(), msg);

    // Bounded flush. Each `poll()` moves at most one 512-byte grant; 8 KB
    // ring = 16 grants max, but the CDC-ACM driver needs a GPT tick (~4 ms)
    // to see a grant release before it will accept the next write. So we
    // interleave `poll()` with a short busy-wait. 200 iterations × 20 µs is
    // ~4 ms of real time, more than enough for the host to drain the ring.
    // If we haven't emptied it by then, `sos()` spins on the LED anyway.
    with_poller(|p| {
        for _ in 0..200u32 {
            p.poll();
            busy_wait_us(20);
        }
    });
    teensy4_panic::sos()
}

struct BumpAllocator;

/// SAFETY: single-core; only one allocation at a time; allocations are
/// never freed.
unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let align = layout.align();
        let mut off = BUMP_OFF.load(Ordering::Acquire);
        loop {
            // Honor layout.align() — Cortex-M7 faults on misaligned
            // f32/usize/Vec metadata otherwise. `align` is a power of two
            // (Layout invariant), so the mask form is valid.
            let aligned_off = (off + align - 1) & !(align - 1);
            let next_off = aligned_off + layout.size();
            if next_off > BUMP_BUF.len() {
                return ptr::null_mut();
            }
            match BUMP_OFF.compare_exchange_weak(
                off,
                next_off,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return unsafe { BUMP_BUF.as_ptr().add(aligned_off) as *mut u8 },
                Err(actual) => off = actual,
            }
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { self.alloc(layout) };
        if !p.is_null() {
            unsafe { ptr::write_bytes(p, 0, layout.size().max(1)) };
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

/// Elapsed milliseconds since `start` (saturating to `u32::MAX`) using
/// DWT. Handles wrap-around correctly (u64 math on two u32 operands).
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

    use super::{cycles_now, now_millis, ms_since, poll_log, POLLER_PTR};
    use core::sync::atomic::Ordering;
    use hl2_teensy::{display, radio, shared, spectrum};
    use imxrt_log as logging;
    use rtic_monotonics::systick::Systick;
    use rtic_monotonics::systick::ExtU64;

    /// Task-local type alias for the ILI9341 panel.
    type Panel = display::driver::Display;

    /// Waterfall framebuffer: 320 × 120 u16 = 76.8 KiB. `static`.
    /// Row 0 is at the *top* (newest), row `WF_ROWS − 1` is at the
    /// bottom (oldest). We write the newest row *into the top row* after
    /// shifting the rest of the band down one row in RAM, and we SPI only
    /// that one row to the display — a 120× SPI-load reduction over the
    /// older full-screen repaint.
    static WF: static_cell::ConstStaticCell<[u16; display::WF_COLS * display::WF_ROWS]> =
        static_cell::ConstStaticCell::new([0u16; display::WF_COLS * display::WF_ROWS]);

    /// Format the RX frame counter as "F <u32>". Single-byte buffer in a
    /// static; the render task alone reads it (via `status_redraw`).
    fn fmt_frames() -> &'static str {
        use core::fmt::Write;
        static BUF: [u8; 12] = [0u8; 12];
        let mut w = radio::control::WriteBuf {
            target: unsafe { core::slice::from_raw_parts_mut(BUF.as_ptr() as *mut u8, 12) },
            pos: 0,
        };
        let _ = write!(&mut w, "F {}", shared::frames());
        core::str::from_utf8(&w.target[..w.pos]).unwrap()
    }

    fn fmt_ip() -> &'static str {
        use core::fmt::Write;
        static BUF: [u8; 16] = [0u8; 16];
        let w = radio::control::WriteBuf {
            target: unsafe { core::slice::from_raw_parts_mut(BUF.as_ptr() as *mut u8, 16) },
            pos: 0,
        };
        let mut w = w;
        let peer = match shared::peer() {
            Some(ip) => ip,
            None => {
                let _ = write!(&mut w, "-");
                return core::str::from_utf8(&w.target[..1]).unwrap();
            }
        };
        let o = peer.octets();
        let _ = write!(&mut w, "{}.{}.{}.{}", o[0], o[1], o[2], o[3]);
        core::str::from_utf8(&w.target[..w.pos]).unwrap()
    }

    /// `imxrt_log::Poller` drains buffered log messages to the USB CDC ACM
    /// The log `Poller` lives in the crate-root `super::POLLER_CELL` static
    /// (one instance per program) so `#[panic_handler]` can drive it after
    /// the tasks stop being scheduled. Tasks poll it via `super::poll_log()`.
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

        // Install the log `Poller` into the crate-root static so the
        // `#[panic_handler]` (in the crate root) can drive it after the RTIC
        // tasks stop being scheduled. `POLLER_REF` caches the `&'static mut`
        // so the handler / `poll_log()` don't need to move it around.
        use super::POLLER_CELL;
        let poller = POLLER_CELL
            .init(logging::log::usbd(usb, logging::Interrupts::Enabled).unwrap());
        POLLER_PTR.store(poller as *mut _, Ordering::Release);

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
        // LPSPI clock. The BSP derives it from a 132 MHz source with an
        // even `sckdiv` in [4, 254], so the max is 132 MHz / 4 = 33 MHz.
        // The ILI9341 datasheet allows far more (62.5 MHz in SPI mode),
        // so 33 MHz is the hardware-limited ceiling. At 8 MHz the
        // 320 × 120 full-screen blit takes ≈154 ms of uninterrupted SPI
        // — long enough that the radio task's 40 ms keepalive tick
        // can't fire in time (HL2 watchdog ≈ 168 ms) and the EP6 stream
        // drops after ~2 spectrum frames. At 33 MHz the same blit is
        // ~37 ms, comfortably inside 40 ms.
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
        let mut panel =
            display::driver::new_display(spi, cs, dc).expect("ILI9341 init");
        display::driver::fill_rect(&mut panel, 0, 0, 320, 240, 0xF800);
        display::driver::draw_text(
            &mut panel,
            10, 10, 2, "HL2 TEENSY 4.1", 0x0000, 0xF800,
        );

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
                        let ip = core::net::Ipv4Addr::new(
                            info.ip[0], info.ip[1], info.ip[2], info.ip[3],
                        );
                        log::info!(
                            "HL2 FOUND ip={} mac={:02x?} rx={} 16bit={} sending={}",
                            ip, info.mac, info.rx_count, info.sample_16bit, info.is_sending,
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
        handle.drain();
        handle.send_start();
        Systick::delay(200.millis()).await;
        poll_log();
        handle.drain();

        // 7. LNA + NCO.
        shared::set_state(shared::STATE_TUNING);
        handle.send_lna(radio::control::LNA_GAIN_DB);
        Systick::delay(50.millis()).await;
        poll_log();
        handle.send_tune(radio::control::TUNE_HZ);
        Systick::delay(200.millis()).await;
        poll_log();
        handle.drain();

        // 8. Streaming.
        shared::set_state(shared::STATE_STREAMING);
        log::info!("EP6 streaming at 96 kSps, NCO 7.074 MHz (RX1)");
        poll_log();

        let mut last_keepalive = cycles_now();
        let mut last_hb = cycles_now();
        let mut seen_frame_seq: u32 = pipeline.frame_seq();
        let mut total_pkts: u32 = 0;
        let mut non_ep6: u32 = 0;
        let mut first_dlogged = 0u32;
        let mut last_dlog = cycles_now();
        loop {
            handle.set_now(smoltcp::time::Instant::from_millis(now_millis()));
            handle.pump();

            // Drain + push into pipeline.
            let mut dgram = [0u8; hl2::protocol::DATA_PACKET_SIZE + 64];
            while let Some((n, _src)) = handle.recv(&mut dgram) {
                total_pkts = total_pkts.wrapping_add(1);
                let is_ep6 = n == hl2::protocol::DATA_PACKET_SIZE
                    && dgram.get(3).copied().unwrap_or(0) == hl2::protocol::ENDPOINT_DATA_TX;
                if !is_ep6 {
                    non_ep6 = non_ep6.wrapping_add(1);
                }
                // Log at most one dgram per ~500 ms (plus the first 5), so we
                // still confirm the endpoint/parse path without spamming the
                // 1 KB log ring and dropping the render-side output.
                let now_c = cycles_now();
                let log_this = first_dlogged < 5
                    || (is_ep6 && ms_since(last_dlog, now_c) >= 500);
                if log_this {
                    first_dlogged = first_dlogged.wrapping_add(1);
                    last_dlog = now_c;
                    let ep = dgram.get(3).copied().unwrap_or(0);
                    log::warn!("dgram #{}: n={} ep=0x{:02x}", first_dlogged, n, ep);
                }
                let pushed = rx.feed(&dgram[..n], &mut pipeline);
                if pushed > 0 {
                    shared::bump_frames();
                    if log_this {
                        log::warn!("   → pushed {} (acc={})", pushed, pipeline.len());
                    }
                }
            }

            // Publish a new spectrum row **only when the pipeline actually
            // produced a new FFT frame**. This matters because the HL2
            // stream is faster (~1.3 ms/FFT-frame) than the old 5 ms
            // publish cadence — the old code republished the *same* mags
            // multiple times between real FFT commits, so the render task
            // shifted + blitted identical content (a frozen band with a
            // scrolling seam). Publishing on frame_seq bump fixes that.
            let cur_seq = pipeline.frame_seq();
            if cur_seq != seen_frame_seq {
                seen_frame_seq = cur_seq;
                shared::publish(pipeline.mags());
            }

            let now_c = cycles_now();
            // Keep-alive: 40 ms cadence (~168 ms HL2 watchdog limit).
            if ms_since(last_keepalive, now_c) >= radio::control::KEEPALIVE_INTERVAL_MS_CONST
            {
                last_keepalive = now_c;
                handle.send_keepalive();
            }

            // Diagnostic heartbeat to USB (every ~100 ms): lets the operator
            // distinguish "radio alive but render dead" (F climbing) from
            // "HL2 stream dead" (F frozen). `rx.frames` is local; `acc` is
            // the pipeline accumulator depth.
            if ms_since(last_hb, now_c) >= 100 {
                last_hb = now_c;
                log::info!(
                    "hb F={} acc={} seq={} total={} non_ep6={}",
                    rx.frames,
                    pipeline.len(),
                    seen_frame_seq,
                    total_pkts,
                    non_ep6
                );
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
        status_redraw(&mut panel);

        let mut painted_seq: u32 = 0;
        let mut last_state: u32 = u32::MAX;
        let mut last_peer: u32 = u32::MAX;
        let mut last_status_ms: i64 = now_millis();
        let mut last_render_hb_ms: i64 = now_millis();

        loop {
            poll_log();

            // Render-side heartbeat: confirms this task is still alive even
            // if no new `seq` has been published. Compare with the
            // radio-side `hb` line: if both are streaming, the two tasks
            // are both up and the freeze is a pipeline / render-content
            // problem. If only one is streaming, we know which task died.
            let now_ms_r = now_millis();
            if now_ms_r - last_render_hb_ms >= 100 {
                last_render_hb_ms = now_ms_r;
                log::info!(
                    "rdr loop seq={} painted={}",
                    shared::latest().0,
                    painted_seq
                );
            }

            let (seq, row) = shared::latest();
            let st = shared::state();
            let peer_u32 = shared::peer().map(|ip| {
                let o = ip.octets();
                ((o[0] as u32) << 24) | ((o[1] as u32) << 16) | ((o[2] as u32) << 8) | (o[3] as u32)
            }).unwrap_or(u32::MAX);

            // ── Waterfall band ──
            // Per new published `seq`:
            //   1. shift the whole band DOWN one row in RAM (fast memcpy,
            //      no SPI);
            //   2. palette the new spectrum into the top row;
            //   3. blit the full 320 × 120 band to the panel (one
            //      uninterrupted 80 ms SPI call).
            //
            // The panel's framebuffer is independent of our RAM band —
            // after the shift, every panel row other than row 0 is stale
            // (it still shows the previous frame's contents). Only a
            // full-band blit keeps the panel in sync. One uninterrupted
            // SPI call is the fastest option the driver offers and leaves
            // the radio task with an ~80 ms window to send its 40 ms
            // keep-alive inside the ~168 ms HL2 watchdog — much more
            // generous than the 3-millisecond windows of the chunked
            // version the code used before this fix.
            if seq > painted_seq {
                let cols = display::WF_COLS as usize;
                let rows = display::WF_ROWS as usize;
                let total = cols * rows;

                // 1. shift down in RAM (newest → top row).
                fb.copy_within(..total - cols, cols);
                // 2. new row at the top.
                for (i, px) in fb[..cols].iter_mut().enumerate() {
                    *px = display::palette::bin_color(row.get(i).copied().unwrap_or(0u16));
                }
                // 3. blit the whole band (one SPI transaction).
                //    Heartbeats bracket the SPI writes so a freeze pins them:
                //      "rdr blit>  …"  with no "rdr blit ok" → stuck in
                //        draw_raw_slice (LPSPI transfer loop).
                //      "rdr blit ok" present but no further loop → stuck in
                //        the status redraw or the next iteration's poll.
                log::info!("rdr blit> seq={seq} painted={painted_seq}");
                let _ = panel.draw_raw_slice(
                    0, 0,
                    (cols as u16) - 1, (rows as u16) - 1,
                    &fb[..total],
                );
                log::info!("rdr blit ok seq={seq}");
                painted_seq = seq;
            }

            // ── Status bar ── Redraw on (a) state / peer-IP change, or
            // (b) a 500 ms tick (to keep the "F N" live — a *frozen* F
            // counter on the panel is the diagnostic for "HL2 stream died").
            // 500 ms is still ~8× less repaint pressure than the old
            // 40 ms seizure.
            let now_ms = now_millis();
            let st_changed = st != last_state || peer_u32 != last_peer;
            let tick = (now_ms - last_status_ms) >= 500;
            if st_changed || tick {
                last_state = st;
                last_peer = peer_u32;
                last_status_ms = now_ms;
                status_redraw(&mut panel);
            }

            Systick::delay(2.millis()).await;
        }
    }

    /// Redraw the bottom-half status bar (label + peer IP + frequency line).
    /// Called on first paint and on every state / peer-IP change; NOT on a
    /// timer. A state transition is a rare event (WAIT IP → DISCOVERY →
    /// STARTING → TUNING → STREAMING), so this cost is amortised.
    fn status_redraw(panel: &mut Panel) {
        let label = match shared::state() {
            shared::STATE_WAITING_IP => "WAIT IP",
            shared::STATE_LINK      => "LINK",
            shared::STATE_DISCOVERING => "DISCOVERY",
            shared::STATE_STARTING   => "STARTING",
            shared::STATE_TUNING     => "TUNING",
            shared::STATE_STREAMING  => "STREAMING",
            shared::STATE_ERROR      => "ERROR",
            _                        => "?",
        };
        let wf_y = display::WF_ROWS as u16;
        // Diagnostic brackets: status_redraw is the *only* SPI work in the
        // render body that does NOT carry its own "before"/"after" log pair,
        // and it is where render wedged (last line was "rdr blit ok" with no
        // further output). Bracket each SPI call so the next capture shows
        // whether it hangs in the fill, in a draw_text, or never returns.
        log::info!("status> fill {label}");
        // Fill the full status region (y=120..=239, 120 rows tall) black
        // before the text. Runs on a state / peer-IP change and the 100 ms
        // live-F tick.
        display::driver::fill_rect(panel, 0, wf_y, 320, 120, 0x0000);
        log::info!("status mid1 (fill ok)");
        display::driver::draw_text(panel, 6, wf_y + 4,  1, label,          0xF800, 0x0000);
        log::info!("status mid2 (label ok)");
        // IP + RX-frame counter on one line: "192.168.1.5 F 1234".
        // If the F counter stops climbing, the HL2 has dropped the stream.
        let mut line = [0u8; 32];
        let mut len = 0;
        for b in fmt_ip().as_bytes() {
            line[len] = *b; len += 1;
        }
        line[len] = b' '; len += 1;
        for b in fmt_frames().as_bytes() {
            line[len] = *b; len += 1;
        }
        let line_s = core::str::from_utf8(&line[..len]).unwrap();
        display::driver::draw_text(panel, 6, wf_y + 30, 1, line_s,        0xF800, 0x0000);
        log::info!("status mid3 (ip/F ok)");
        display::driver::draw_text(panel, 6, wf_y + 76, 1, "7.074MHz RX1 96k", 0xF800, 0x0000);
        log::info!("status< done");
    }
}
