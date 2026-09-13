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
    use hl2_teensy::{autoscale, display, i2c, radio, shared, spectrum};
    use imxrt_log as logging;
    use rtic_monotonics::systick::ExtU64;
    use rtic_monotonics::systick::Systick;

    /// DMA0-15 completion IRQ. Channels 0..=15 all share this vector on the
    /// RT1060; the `dma_irq` ISR is what each parked task was told to
    /// expect on completion.
    ///
    /// - **channel 0** — display blit. `DmaDisplay` (owned by `render`)
    ///   arms `set_interrupt_on_completion(true)` on that channel; the
    ///   handler here calls `DMA.on_interrupt(0)`, which clears DONE + INT
    ///   and wakes `render`.
    ///
    /// - **channel 1** — SAI1 audio. `audio_task` arms it once at startup;
    ///   each `process_chunk` `.await`s the eDMA-to-TDR transfer and parks
    ///   on that channel's waker. The completion sets DONE + INT, fires
    ///   this same DMA0-16 vector, and we route it through
    ///   `DMA.on_interrupt(1)` to wake the parked `audio_task`.
    ///
    /// Both are exclusive — one task owns channel 0, the other owns
    /// channel 1 — so `on_interrupt`'s waker-slot contract (`"associated
    /// DMA channel is exclusively referenced"`) is satisfied per channel.
    /// The handler just fans in one shared vector.
    ///
    /// The `shared::irq_fires_inc()` counter is incremented once per IRQ
    /// (used by `DmaDisplay.draw_pixels` to report "how many IRQs did we
    /// actually get?" across a 150-chunk blit; the audio path doesn't
    /// check this counter, so it's a rough total that over-counts if both
    /// channels complete at similar times).
    #[task(binds = DMA0_DMA16, priority = 3)]
    fn dma_irq(_cx: dma_irq::Context) {
        shared::irq_fires_inc();
        // Safety: `channel(0)` (display) and `channel(1)` (audio) are each
        // exclusively owned by a single task — `render` and `audio_task`
        // respectively. `on_interrupt` calls the *channel's* stored waker
        // if that channel has INT + DONE set, and is a no-op otherwise
        // (see `imxrt-dma` `Dma::on_interrupt`). Two calls into the same
        // static `Dma<32>` is safe because they touch disjoint channels
        // and their waker slots (`SharedWaker` is a `Mutex<RefCell<..>>`).
        unsafe {
            bsp::hal::dma::DMA.on_interrupt(0);
            bsp::hal::dma::DMA.on_interrupt(1);
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
            lpi2c1,
            ccm,
            ccm_analog,
            iomuxc_gpr,
            mut dma,
            sai1,
            ..
        } = board::t41(cx.device);

        // eDMA channel for the ILI9341 pixel blit. USB (imxrt-usbd) on this
        // chip is CPU-polled by the logger, so all 32 channels are free;
        // channel 0 is the smallest index the RT1060's eDMA can address on
        // any bus domain that can reach both OCRAM (source) and LPSPI4 TDR
        // (destination).
        let display_dma = dma[0].take().expect("dma channel 0 available");

        // eDMA channel 1: the SAI1 → WM8731 I2S audio path. Same bus-domain
        // argument as channel 0 (OCRAM source → SAI1 TDR[0] destination).
        let audio_dma = dma[1].take().expect("dma channel 1 available");

        // SAI1 pin mux (RT1060, Alt3):  P7 = TX_DATA00 (the one wire the
        // SAI needs to shift our L = R mono out to the WM8731),  P20 =
        // RX_SYNC (FSYNC in, driven by the WM8731 since it's the I2S
        // master),  P21 = RX_BCLK (the WM8731's bit clock in).  All three
        // are plain alt-3 multiplex (the RxSync / RxBclk "daisies" in the
        // `imxrt-iomuxc` map exist but we are the *only* SAI1 user, so we
        // don't need to set the daisy bits explicitly — `prepare()` just
        // writes the `MUX` field in `SW_MUXR`).
        let mut p7 = pins.p7;
        let mut p20 = pins.p20;
        let mut p21 = pins.p21;
        imxrt_iomuxc::sai::prepare(&mut p7);
        imxrt_iomuxc::sai::prepare(&mut p20);
        imxrt_iomuxc::sai::prepare(&mut p21);

        // Bring up SAI1 as **slave-TX** (I2S 16-bit, frame_size = 2 with
        // `Packing::None` so each 16-bit half lands in its own 32-bit TDR
        // entry — the exact "2 TDR words per mono sample" the `Sink`
        // repacks). The SAI's own clock plumbing (PLL4 → SAI1) and its
        // gate are already enabled in `teensy4-bsp::clock_power::setup_sai1_clk`
        // (see the BSP `prepare_clocks_and_power`); in slave mode we clock
        // off the WM8731's BCLK + FSYNC, so the SAI1's own BCLK divider is
        // irrelevant.
        //
        // The `i2c_bus` lives in the radio task (which owns LPI2C1), so
        // the WM8731 itself is configured there at startup before the first
        // audio sample is pushed. Until then the SAI has no clock, no
        // BCLK-driven eDMA requests, and it sits idle in the TX FIFO.
        let sai_tx = hl2_teensy::audio::sai1::init_tx(sai1).expect("sai1 tx");

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
        let _ = radio_task::spawn(ccm, ccm_analog, iomuxc_gpr, lpi2c1, pins.p19, pins.p18);
        let _ = audio_task::spawn(audio_dma, sai_tx);

        (Shared {}, Local {})
    }

    /// Drive the SAI1 → WM8731 audio path on a fixed 1 ms tick.
    ///
    /// [`hl2_teensy::audio::sink::process_chunk`] pulls up to 960 upsampled
    /// samples (20 ms of 48 kHz audio) from the cross-task ring, folds each
    /// into its `[L = R]` TDR pair, and drives one eDMA transfer from the
    /// `'static` `STAGE` buffer to SAI1.TDR[0]. The WM8731 (I2S master,
    /// driven by its own 12.288 MHz / 256 sample clock) pulls those words
    /// out of the SAI's TX FIFO on the wire.
    ///
    /// The tick deliberately runs at 1 ms (not at the radio's 1 ms poll):
    /// with 960 samples per tick the audio task only needs to touch the
    /// DMA channel ~ 50 times per 100 ms — a negligible fraction of the
    /// single-core budget, and *enough* to keep the SAI's TX FIFO above
    /// its low-watermark (8 words) without ever needing to spin for more
    /// than ~ 200 µs per transfer.
    /// Priority 1: above `radio_task`'s default 0, so the completion IRQ
    /// (routed to a priority-3 ISR that calls `DMA.on_interrupt(1)`) wins
    /// the preemption race as soon as the eDMA sets DONE.
    ///
    /// The audio task itself **yields** while the DMA runs (it `await`s
    /// the `peripheral::write` future, which parks on the channel waker).
    /// Only the ~few µs of `pop_into` + `pack` + TCD programming holds the
    /// CPU per 20 ms of 48 kHz audio (960 samples / 48 kHz). `radio_task`
    /// (priority 0) still gets its long network / I2C / DHCP slices
    /// freely. `render` (priority 1) can also preempt back in here to
    /// paint, because we're parked on a waker — the RTIC executor will
    /// resume the *highest-priority* task with a woken continuation
    /// whenever it gets to poll again.
    #[task(priority = 1)]
    async fn audio_task(
        _cx: audio_task::Context,
        mut chan: bsp::hal::dma::channel::Channel,
        mut tx: bsp::hal::sai::Tx,
    ) {
        // Yield until `radio_task` finishes configuring the WM8731 and
        // publishes `shared::AUDIO_READY = true`. Until that flag flips,
        // the SAI (slave) has no BCLK/FSYNC to shift against and the
        // eDMA-to-TDR transfer cannot complete — so the first chunk would
        // `.await` indefinitely. Yielding here (not spinning at this
        // task's priority, which would starve `radio_task`) lets the
        // radio task run, bring up the codec, and only then flip the flag.
        while !shared::audio_ready() {
            Systick::delay(5.millis()).await;
        }
        log::info!("audio task: codec ready, first chunk starting");
        poll_log();
        // `ConstStaticCell::take()` is a one-shot, so acquire the stable
        // SAI source buffer *here, once*; the eDMA's TCD references this
        // address for the channel's lifetime. Same pattern as the display.
        let stage = hl2_teensy::audio::sink::STAGE.take();
        // Arm interrupt-on-completion once. The `dma_irq` ISR fires
        // `DMA.on_interrupt(1)` when the channel's DONE + INT bits are
        // set, which wakes our registered waker (imxrt-dma stores it in
        // `Channel::waker` and calls `waker.wake()` from `on_interrupt`).
        chan.set_interrupt_on_completion(true);
        let mut chunk_idx: u32 = 0;
        loop {
            let t0 = cycles_now();
            // On any DMA fault `process_chunk` returns `Ready(Err)` *immediately*
            // (imxrt-dma's `Transfer::poll` returns `Ready` as soon as `is_error`
            // is set — it does not park). Without yielding here, `audio_task`
            // (priority 1) just re-enters `process_chunk` in a tight loop while
            // its `await` returns `Ready` again and again; RTIC's run wrapper
            // then `pend(KPP)`s after every slice, re-firing the KPP interrupt
            // that preempts the *priority-0* `radio_task` dispatcher's endless
            // loop (see `target/rtic-expansion.rs` — `radio_task` runs in
            // `__rtic_internal_async_0_prio_dispatcher`, the main-thread `loop`,
            // while `audio_task` + `render` run inside the KPP handler). The
            // priority-1 task therefore starves priority-0 forever and the
            // `Pipeline::new` line the radio task prints *after* its
            // `build_iface_and_sockets` slice never reaches USB.
            //
            // Yielding one SysTick tick on every fault guarantees the
            // priority-0 dispatcher a slice. The fault itself is the real
            // cause (logged as `info!` below — `debug!` is compiled out in
            // release), so the next log line tells us *what* the DMA is
            // complaining about.
            match hl2_teensy::audio::sink::process_chunk(&mut chan, &mut tx, stage).await {
                Ok(()) => {}
                Err(e) => {
                    log::info!("audio dma fault {e} (chunk #{chunk_idx}); yielding to dispatcher");
                    poll_log();
                    Systick::delay(2.millis()).await;
                }
            }
            chunk_idx += 1;
            // Diagnostic: per-chunk wall time + how many total DMA IRQs
            // have fired by now. If each chunk takes ~20 ms (960 samples
            // @ 48 kHz) and `irq_fires` is incrementing, the SAI + eDMA
            // are clocking and draining. If chunks take much longer than
            // 20 ms, the SAI FIFO is backing up (likely the codec is
            // *not* actually driving BCLK/FSYNC, or a config mismatch
            // between the SAI frame and the codec's expected format).
            if chunk_idx <= 3 || (chunk_idx % 50) == 0 {
                let ms = ms_since(t0, cycles_now());
                log::info!(
                    "audio chunk #{} done in {} ms; total dma_irqs={}",
                    chunk_idx,
                    ms,
                    shared::irq_fires(),
                );
            }
        }
    }

    #[task]
    async fn radio_task(
        _cx: radio_task::Context,
        mut ccm: bsp::ral::ccm::CCM,
        mut ccm_analog: bsp::ral::ccm_analog::CCM_ANALOG,
        mut iomuxc_gpr: bsp::ral::iomuxc_gpr::IOMUXC_GPR,
        lpi2c1: bsp::ral::lpi2c::Instance<1>,
        scl_pin: teensy4_bsp::pins::t41::P19,
        sda_pin: teensy4_bsp::pins::t41::P18,
    ) {
        // Let USB-serial enumerate before MDIO.
        Systick::delay(2_000.millis()).await;
        poll_log();

        // I2C bus probe (LPI2C1, SDA=p18 / SCL=p19). The bus is held for
        // the whole task; the temp sensor / audio codec will be polled
        // here later.
        let mut i2c_bus = i2c::init(lpi2c1, scl_pin, sda_pin);
        i2c::scan(&mut i2c_bus);
        poll_log();

        // Configure the WM8731 *before* the Sink starts pushing samples so
        // that by the time the first `audio_task` tick runs, the codec is
        // awake, driving BCLK + FSYNC, and our SAI1 slave TX has a clock to
        // latch data against. Without this call the SAI1's FIFO drains by
        // its low-watermark-driven eDMA request (FWDE) but the wire is
        // silent because the WM8731 isn't clocking.
        if let Err(()) = hl2_teensy::audio::wm8731::init(&mut i2c_bus) {
            log::error!(
                "wm8731 init failed on 0x{:02X}",
                hl2_teensy::audio::wm8731::WM8731_ADDR
            );
        } else {
            log::info!("WM8731 audio codec configured (I2S master, 48 kHz, 16-bit)");
            // Now (and only now) is the codec driving BCLK/FSYNC, so the SAI
            // slave-TX has a clock and the eDMA to the SAI can actually
            // complete. Release the audio task from its pre-activation yield
            // gate *after* publishing the log line so the first chunk is
            // guaranteed to come after this `poll_log` has flushed.
            shared::set_audio_ready(true);
        }
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
        log::info!("radio: Ethernet OK, now building smoltcp iface");
        poll_log();
        let (mut iface, mut sockets, handles) = radio::control::build_iface_and_sockets(
            &mut device,
            radio::control::MAC,
            smoltcp::time::Instant::from_millis(now_millis()),
        );
        log::info!("radio: iface + sockets OK");
        poll_log();

        // 3. Pipeline.
        log::info!("radio: now building spectrum pipeline");
        poll_log();
        // Diagnostic: bracket `Pipeline::new` with a wall-time + heap + audio
        // counter snapshot. `Pipeline::new` is synchronous (no `.await`), so
        // if it ever fails to return the *immediate* next log line
        // ("now building Rx") never appears — the symptom we are chasing.
        // The numbers below let us separate three possible causes:
        //   * "took 40 ms"          → a trig/allocation cost (fine, not stuck)
        //   * "heap grew 48 KB"     → normal (twiddles + win + buf_re/im + acc)
        //   * "audio in=1 done=0"   → audio task parked on its 1st DMA (normal
        //                            before BCLK) — *not* a busy loop
        //   * "audio in=42 done=42" → audio is churning; if radio still stalls,
        //                            it's a priority/preemption race, not a
        //                            stack/allocation issue
        let t_n0 = cycles_now();
        let heap_n0 = crate::BUMP_OFF.load(core::sync::atomic::Ordering::Acquire);
        let a_in_n0 = hl2_teensy::audio::sink::chunks_entered();
        let a_done_n0 = hl2_teensy::audio::sink::chunks_completed();
        let mut pipeline = match spectrum::Pipeline::new() {
            Ok(p) => p,
            Err(e) => {
                let ms = ms_since(t_n0, cycles_now());
                log::error!("pipeline init: {e} after {ms} ms");
                shared::set_state(shared::STATE_ERROR);
                return;
            }
        };
        {
            // Keep the log line short (the imxrt-log ring buffer is 1024 B and
            // silently drops writes that don't fit); use pure ASCII.
            let ms = ms_since(t_n0, cycles_now());
            let heap_kb =
                (crate::BUMP_OFF.load(core::sync::atomic::Ordering::Acquire) - heap_n0) / 1024;
            log::info!(
                "Pipeline::new done in {}ms, heap +{}KB, in {}->{} done {}->{}",
                ms,
                heap_kb,
                a_in_n0,
                hl2_teensy::audio::sink::chunks_entered(),
                a_done_n0,
                hl2_teensy::audio::sink::chunks_completed(),
            );
        }
        poll_log();
        log::info!("radio: now building Rx (VirtualReceiver + Sink)");
        poll_log();
        let mut rx = radio::Rx::new();
        log::info!("radio: Rx + Sink built; entering wait-for-IP loop");
        poll_log();

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

        let mut wait_iter: u32 = 0;
        loop {
            wait_iter += 1;
            // Diagnostic: count the loop iterations so we can tell "loop
            // isn't running" from "loop is running but link_up() keeps
            // returning false". If wait_iter increments forever, the
            // radio task has a slice every ~10 ms and the PHY is just
            // never reporting link.
            if wait_iter <= 3 || wait_iter % 100 == 0 {
                log::info!("wait-ip iter #{}", wait_iter);
            }
            let now_c = cycles_now();
            if !link_up && ms_since(last_link, now_c) >= 500 {
                last_link = now_c;
                match handle.dev.link_up() {
                    Ok(up) => {
                        log::info!("link_up() = {}", if up { "up" } else { "down" });
                        if up != link_up {
                            link_up = up;
                            if up {
                                log::info!("link is up, proceeding to DHCP...");
                            }
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
        // Liveness diagnostics: the discovery loop is silent by design (it
        // only logs when it *receives* a reply), so "broadcasting forever"
        // looks identical to "hung in a sync call" from the USB log alone.
        // These counters + the render-task tick echo a heartbeat at ~2 Hz
        // that tells us which:
        //   * iter grows, sent grows, rc flat, rtk grows  → alive, waiting
        //                                                       for HL2 reply
        //   * iter grows, sent flat                        → 500 ms gate not
        //                                                     firing (clock?)
        //   * iter flat / rtk flat                         → scheduler or
        //                                                     render starved
        let mut disc_iter: u32 = 0;
        let mut disc_sent: u32 = 0;
        let mut disc_recv: u32 = 0;
        let mut disc_last_hb = cycles_now();
        'discovery: loop {
            disc_iter += 1;
            handle.set_now(smoltcp::time::Instant::from_millis(now_millis()));
            handle.pump();
            let mut dgram = [0u8; hl2::protocol::DISCOVERY_RESPONSE_SIZE + 16];
            // Drain until empty; a valid reply ends the loop.
            while let Some((n, src)) = handle.recv(&mut dgram) {
                disc_recv += 1;
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
                disc_sent += 1;
                handle.send_discovery();
            }
            // ~2 Hz liveness heartbeat (see the counters declared above).
            // Also carries the audio path state:
            //   aud_wr = demod `Sink::write` calls so far (0 → no demod
            //            output; > 0 → virtual receiver is producing)
            //   in     = `process_chunk` calls = eDMA transfers attempted
            //   done   = `process_chunk` calls that *completed* (the eDMA
            //            IRQ fired and imxrt-dma reported success)
            //   in-flight = in - done (should be ≤ 1: one single in-flight
            //             in the single-channel DMA; > 1 = eDMA hung)
            //   tcsr   = SAI TCSR status after the last DMA (bits: 0x800
            //            = FIFO_REQUEST, 0x1000 = FIFO_WARNING, 0x2000
            //            = FIFO_ERROR,   0x4000 = SYNC_ERROR,  0x8000 =
            //            WORD_START). A slave SAI that is clocking should
            //            show FIFO_REQUEST or FIFO_WARNING. A slave SAI
            //            that is not clocking can show all-clear (FIFO
            //            never filled enough to warn) or FIFO_ERROR
            //            (underrun — the SAI tried to shift with an empty
            //            FIFO).
            //   wfp/rfp = SAI TX FIFO write / read positions at the instant
            //            of the last DMA completion (32=full). wfp=32 rfp=0
            //            = data is *in* the FIFO but not shifting; wfp≈rfp
            //            = steady drain (chain is healthy end-to-end).
            //   chunk_ms = wall milliseconds for the last completed eDMA
            //            transfer (~20 ms = 1920 TDR words ÷ 48 kHz wire
            //            rate; > 20 ms = SAI isn't shifting fast enough,
            //            < 20 ms = it IS clocking and the eDMA just runs
            //            at source rate).
            if ms_since(disc_last_hb, cycles_now()) >= 500 {
                disc_last_hb = cycles_now();
                let in_c = hl2_teensy::audio::sink::chunks_entered() as u32;
                let done_c = hl2_teensy::audio::sink::chunks_completed() as u32;
                log::info!(
                    "disc iter={} sent={} rec={} rtk={} | aud={} in={} done={} inflight={} | sai_tcsr={:x} wfp={} rfp={} ms={}",
                    disc_iter,
                    disc_sent,
                    disc_recv,
                    shared::render_ticks(),
                    shared::audio_writes(),
                    in_c,
                    done_c,
                    in_c.saturating_sub(done_c),
                    shared::sai_tcsr(),
                    (shared::sai_tfr() >> 16) as u32,
                    (shared::sai_tfr() & 0xFFFF) as u32,
                    shared::sai_chunk_ms(),
                );
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
        // Auto floor/ceil for the waterfall (UI `Shared::update_auto_scale`).
        // Seeded to the UI's initial auto window; `recenter` snaps it to the
        // first real frame's band instead of blending up from the seed.
        let mut scale = autoscale::AutoScale::new();
        scale.recenter();
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
                // Auto floor/ceil: advance the window from the *same* row and
                // publish it so the render task's `bin_color` ramps the live
                // band (only the top row repaints; prior rows keep their ramp).
                if scale.step(pipeline.mags()) {
                    let (f, c) = scale.scale();
                    shared::set_scale(f, c);
                }
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
        let mut last_heartbeat_ms: i64 = 0;
        let mut heartbeat_count: u32 = 0;

        loop {
            poll_log();
            // Liveness: bump the cross-task render counter *first thing* each
            // iteration so the radio task's discovery heartbeat can echo it.
            // A flat value while the radio task is still logging proves the
            // render task (and possibly the scheduler) has stopped running.
            shared::render_tick();

            // Out-of-band heartbeat (100 ms cadence). This runs *while* the
            // radio task is blocked in a synchronous section (e.g. inside
            // `Pipeline::new`), so it tells us *which* step of that section
            // the main thread is currently executing. Also samples the
            // audio counters, so we learn — from a different task's vantage
            // point — whether the audio task is churning (enter == done
            // and both growing) or parked (enter == 1, done == 0).
            // Only log while the radio task is mid-build (step 0..=5) or during
            // the first few beats; once `Pipeline::new` has returned (step == 6)
            // and the first beats have elapsed, the heartbeat goes quiet so it
            // doesn't flood the log in steady state. Its job is to reveal, from
            // a *running* priority-1 task, exactly which step the priority-0
            // `radio_task` is stuck on in `Pipeline::new` — a frozen
            // `pipeline_new_step` value is the smoking gun.
            let hb_now = now_millis();
            let pstep = hl2_teensy::spectrum::pipeline_new_step();
            if (hb_now - last_heartbeat_ms) >= 100 {
                last_heartbeat_ms = hb_now;
                heartbeat_count += 1;
                if heartbeat_count <= 3 || pstep != 6 {
                    log::info!(
                        "render hb #{}: pipeline_new_step={} audio enter={} done={} dma_irqs={}",
                        heartbeat_count,
                        pstep,
                        hl2_teensy::audio::sink::chunks_entered(),
                        hl2_teensy::audio::sink::chunks_completed(),
                        shared::irq_fires(),
                    );
                }
            }

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

                // The auto floor/ceil window for the ramp (the radio task
                // publishes it each frame; the seed before the first frame).
                let (floor, ceil) = shared::scale();

                // 1. shift down in RAM (newest → top row). The shift + the
                //    colourise below *is* the *lcd* CPU stage for the status
                //    readout — the DWT delta is appended to the shared
                //    accumulator the radio task drains on its 1-second rollover
                //    (same shared wall as demod / fft). The eDMA blit below is
                //    offloaded to the engine and is *not* counted as CPU.
                let c0 = DWT::cycle_count();
                fb.copy_within(..total - cols, cols);
                // 2. new row at the top, ramped by the auto floor/ceil window.
                for (i, px) in fb[..cols].iter_mut().enumerate() {
                    *px = display::palette::bin_color(
                        row.get(i).copied().unwrap_or(0u16),
                        floor,
                        ceil,
                    );
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
