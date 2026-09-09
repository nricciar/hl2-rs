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

extern crate alloc;
use alloc::alloc::{GlobalAlloc, Layout};
use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};

static BUMP_BUF: [u8; 16 * 1024] = [0u8; 16 * 1024];
static BUMP_OFF: AtomicUsize = AtomicUsize::new(0);

struct BumpAllocator;

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size().min(BUMP_BUF.len());
        let off = BUMP_OFF.fetch_add(size, Ordering::AcqRel);
        if off >= BUMP_BUF.len() {
            ptr::null_mut()
        } else {
            unsafe { BUMP_BUF.as_ptr().add(off) as *mut u8 }
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

mod ethernet;

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

    use rtic_monotonics::Monotonic;
    use rtic_monotonics::systick::{Systick, *};

    const WIDTH: u16 = 320;
    const HEIGHT: u16 = 240;

    use smoltcp::{
        iface::{Config, Interface, SocketSet, SocketStorage},
        socket::{dhcpv4, udp},
        wire::EthernetAddress,
    };

    // Discovery lifecycle shared between the `ethernet` task (producer) and
    // the `hello_world` task (painter). `AtomicU8` is enough: one writer,
    // one reader, single-core, no ordering needed between the state byte and
    // the `HL2_FOUND` pointer (the pointer is either null or final by the
    // time the state flips to `Found`).
    const STATE_WAITING: u8 = 0;
    const STATE_SEARCHING: u8 = 1;
    const STATE_FOUND: u8 = 2;
    const STATE_NOT_FOUND: u8 = 3;
    static DISCOVERY_STATE: core::sync::atomic::AtomicU8 =
        core::sync::atomic::AtomicU8::new(STATE_WAITING);

    // UDP datagram storage for HL2 discovery. The HL2 reply is a fixed 60
    // bytes (hl2/src/protocol/discovery.rs:39); 8 slots / 512 B payload hold
    // replies with ample headroom while this socket coexists with DHCP.
    static UDP_RX_META: [udp::PacketMetadata; 8] = [udp::PacketMetadata::EMPTY; 8];
    static UDP_RX_BUF: [u8; 4096] = [0u8; 4096];
    static UDP_TX_META: [udp::PacketMetadata; 8] = [udp::PacketMetadata::EMPTY; 8];
    static UDP_TX_BUF: [u8; 512] = [0u8; 512];

    fn make_udp_socket() -> udp::Socket<'static> {
        // Safety: these statics are declared once, used exactly once, and are
        // not otherwise accessed before the socket owns them.
        unsafe {
            let rx_meta = core::slice::from_raw_parts_mut(
                UDP_RX_META.as_ptr() as *mut udp::PacketMetadata,
                UDP_RX_META.len(),
            );
            let rx_payload =
                core::slice::from_raw_parts_mut(UDP_RX_BUF.as_ptr() as *mut u8, UDP_RX_BUF.len());
            let tx_meta = core::slice::from_raw_parts_mut(
                UDP_TX_META.as_ptr() as *mut udp::PacketMetadata,
                UDP_TX_META.len(),
            );
            let tx_payload =
                core::slice::from_raw_parts_mut(UDP_TX_BUF.as_ptr() as *mut u8, UDP_TX_BUF.len());
            let rx = udp::PacketBuffer::new(&mut rx_meta[..], &mut rx_payload[..]);
            let tx = udp::PacketBuffer::new(&mut tx_meta[..], &mut tx_payload[..]);
            udp::Socket::new(rx, tx)
        }
    }

    // The first parsed HL2 discovery reply, stored once so `hello_world` can
    // re-render the display without re-parsing.
    static HL2_FOUND: static_cell::StaticCell<hl2::protocol::discovery::DiscoveryInfo> =
        static_cell::StaticCell::new();

    // `NonNull` pointer to the stored reply, so we can read it back after
    // `try_init_with` consumes the `StaticCell`'s single use.
    static FOUND_PTR: core::sync::atomic::AtomicPtr<hl2::protocol::discovery::DiscoveryInfo> =
        core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

    fn hl2_found_ref() -> Option<&'static hl2::protocol::discovery::DiscoveryInfo> {
        let p = FOUND_PTR.load(core::sync::atomic::Ordering::Acquire);
        (p != core::ptr::null_mut()).then_some(unsafe { &*p })
    }

    // Locally administered test MAC. Change this for each board on the same LAN.
    const MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];

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
            let ticks = (ns as u64 * board::ARM_FREQUENCY as u64).div_ceil(1_000_000_000);
            let start = dwt_cycles_now();
            // CYCCNT wraps every ~7.16 seconds at 600 MHz.
            while dwt_cycles_now().wrapping_sub(start) < ticks as u32 {
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
            ccm,
            ccm_analog,
            iomuxc_gpr,
            ..
        } = board::t41(cx.device);

        let poller = logging::log::usbd(usb, logging::Interrupts::Enabled).unwrap();

        // Enable DEMCR.TRCENA and DWT.CTRL.CYCCNTENA before using DwtDelay.
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
                    core::iter::repeat_n(0xF800, total),
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

        let _ = hello_world::spawn(display);
        assert!(ethernet::spawn(ccm, ccm_analog, iomuxc_gpr).is_ok());
        (Shared {}, Local { poller })
    }

    #[task]
    async fn ethernet(
        _cx: ethernet::Context,
        mut ccm: bsp::ral::ccm::CCM,
        mut analog: bsp::ral::ccm_analog::CCM_ANALOG,
        mut gpr: bsp::ral::iomuxc_gpr::IOMUXC_GPR,
    ) {
        // Let init return and USB enumerate before touching experimental hardware.
        Systick::delay(2_000.millis()).await;
        log::info!("Ethernet: configuring clocks, pins and DP83825");
        let mut device = match crate::ethernet::Ethernet::new(
            &mut ccm,
            &mut analog,
            &mut gpr,
            &mut DwtDelay {},
            &MAC,
        ) {
            Ok(device) => device,
            Err(error) => loop {
                log::error!("Ethernet init failed: {error}; LCD/USB still running");
                Systick::delay(5_000.millis()).await;
            },
        };
        let now = || smoltcp::time::Instant::from_millis(Systick::now().ticks() as i64);
        let mut iface =
            Interface::new(Config::new(EthernetAddress(MAC).into()), &mut device, now());
        let mut storage = [SocketStorage::EMPTY; 2];
        let mut sockets = SocketSet::new(&mut storage[..]);
        let dhcp = sockets.add(dhcpv4::Socket::new());
        // Discovery socket. Bind to HL2_PORT (1024) so we receive the reply.
        let mut hl2_sock = make_udp_socket();
        hl2_sock
            .bind(hl2::protocol::HL2_PORT)
            .expect("bind discovery socket to HL2_PORT");
        let hl2 = sockets.add(hl2_sock);

        let mut link_up = false;
        let mut discovered = false;
        let mut search_attempts = 0u32;
        let mut next_search = Systick::now();
        let mut next_status = Systick::now();
        let mut next_report = Systick::now();
        loop {
            if Systick::now() >= next_status {
                next_status = Systick::now() + 250.millis();
                match device.link_up() {
                    Ok(up) => {
                        if up != link_up {
                            link_up = up;
                            log::info!(
                                "Ethernet link: {}",
                                if up { "100 Mbps full duplex" } else { "down" }
                            );
                            sockets.get_mut::<dhcpv4::Socket>(dhcp).reset();
                            iface.update_ip_addrs(|addrs| addrs.clear());
                            iface.routes_mut().remove_default_ipv4_route();
                        }
                    }
                    Err(error) => {
                        log::error!("Ethernet link check failed: {error}");
                        link_up = false;
                        iface.update_ip_addrs(|addrs| addrs.clear());
                        iface.routes_mut().remove_default_ipv4_route();
                        sockets.get_mut::<dhcpv4::Socket>(dhcp).reset();
                    }
                }
            }
            if link_up {
                // Bound RX work so a busy LAN cannot starve the other tasks.
                for _ in 0..4 {
                    iface.poll_ingress_single(now(), &mut device, &mut sockets);
                }
                iface.poll_egress(now(), &mut device, &mut sockets);

                let ip_configured = !iface.ip_addrs().is_empty();

                match sockets.get_mut::<dhcpv4::Socket>(dhcp).poll() {
                    Some(dhcpv4::Event::Configured(config)) => {
                        iface.update_ip_addrs(|addrs| {
                            addrs.clear();
                            addrs.push(config.address.into()).unwrap();
                        });
                        iface.routes_mut().remove_default_ipv4_route();
                        if let Some(router) = config.router {
                            iface.routes_mut().add_default_ipv4_route(router).unwrap();
                        }
                        log::info!("DHCP address: {}; try pinging this address", config.address);
                        // We have an IP — start searching.
                        if ip_configured {
                            DISCOVERY_STATE
                                .store(STATE_SEARCHING, core::sync::atomic::Ordering::Release);
                            search_attempts = 0;
                            next_search = Systick::now(); // send immediately
                        }
                    }
                    Some(dhcpv4::Event::Deconfigured) => {
                        iface.update_ip_addrs(|addrs| addrs.clear());
                        iface.routes_mut().remove_default_ipv4_route();
                        log::info!("DHCP: waiting for a lease");
                        DISCOVERY_STATE.store(STATE_WAITING, core::sync::atomic::Ordering::Release);
                    }
                    None => {}
                }

                if !ip_configured && !discovered {
                    // No IP: nothing to broadcast.
                    DISCOVERY_STATE.store(STATE_WAITING, core::sync::atomic::Ordering::Release);
                }

                // Broadcast the 63-byte discovery request every 2 s while we
                // have an IP and have not yet found the HL2.
                if ip_configured && !discovered && Systick::now() >= next_search {
                    next_search = Systick::now() + 2_000.millis();
                    search_attempts = search_attempts.wrapping_add(1);
                    let req = hl2::protocol::discovery::discovery_request();
                    let dst = smoltcp::wire::IpEndpoint {
                        addr: smoltcp::wire::IpAddress::v4(255, 255, 255, 255),
                        port: hl2::protocol::HL2_PORT,
                    };
                    match sockets.get_mut::<udp::Socket>(hl2).send_slice(&req, dst) {
                        Ok(_) => {}
                        Err(e) => log::error!("HL2 discovery: send failed: {e}"),
                    }
                }

                // Drain the UDP socket until empty; a reply is exactly 60 bytes
                // and must parse as a valid discovery response.
                if !discovered {
                    let s = sockets.get_mut::<udp::Socket>(hl2);
                    while s.can_recv() {
                        let mut buf = [0u8; 512];
                        match s.recv_slice(&mut buf) {
                            Ok((len, from)) => {
                                if len >= hl2::protocol::DISCOVERY_RESPONSE_SIZE {
                                    match hl2::protocol::discovery::parse_discovery_response(
                                        buf.get(..hl2::protocol::DISCOVERY_RESPONSE_SIZE)
                                            .unwrap()
                                            .try_into()
                                            .unwrap(),
                                    ) {
                                        Some(info) => {
                                            let mac = info.mac;
                                            log::info!(
                                                "HL2 FOUND ip={}.{}.{}.{} mac={:02x?} gw={} rx={} {}bit sending={}",
                                                info.ip[0],
                                                info.ip[1],
                                                info.ip[2],
                                                info.ip[3],
                                                mac,
                                                info.gateware_major * 10 + info.gateware_minor,
                                                info.rx_count,
                                                if info.sample_16bit { 16 } else { 12 },
                                                info.is_sending,
                                            );
                                            if let Some(found) = HL2_FOUND.try_init_with(|| info) {
                                                FOUND_PTR.store(
                                                    found as *mut _,
                                                    core::sync::atomic::Ordering::Release,
                                                );
                                                DISCOVERY_STATE.store(
                                                    STATE_FOUND,
                                                    core::sync::atomic::Ordering::Release,
                                                );
                                                discovered = true;
                                            }
                                        }
                                        None => {
                                            log::info!(
                                                "HL2 discovery: {len} B from {from}; malformed, ignoring"
                                            );
                                        }
                                    }
                                } else {
                                    log::info!(
                                        "HL2 discovery: {len} B from {from}; short, ignoring"
                                    );
                                }
                            }
                            Err(e) => log::error!("HL2 discovery: recv failed: {e}"),
                        }
                    }
                }
            }
            if Systick::now() >= next_report {
                next_report = Systick::now() + 5_000.millis();
                log::info!(
                    "Ethernet: link={}, addresses={:?}",
                    link_up,
                    iface.ip_addrs()
                );
            }
            Systick::delay(1.millis()).await;
        }
    }

    /// A 5x7 bitmap for one glyph (7 rows, 5 cols, MSB = leftmost bit).
    /// Covers the letters, digits and punctuation used by the discovery UI.
    fn glyph(ch: char) -> [u8; 7] {
        match ch {
            'A' => [
                0b00110, 0b01010, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
            ],
            'B' => [
                0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
            ],
            'C' => [
                0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
            ],
            'D' => [
                0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
            ],
            'E' => [
                0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
            ],
            'F' => [
                0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
            ],
            'G' => [
                0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
            ],
            'H' => [
                0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
            ],
            'I' => [
                0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
            ],
            'K' => [
                0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
            ],
            'L' => [
                0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
            ],
            'M' => [
                0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
            ],
            'N' => [
                0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
            ],
            'O' => [
                0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
            ],
            'P' => [
                0b01110, 0b10001, 0b10001, 0b01110, 0b10000, 0b10000, 0b10000,
            ],
            'R' => [
                0b11100, 0b10010, 0b10010, 0b11100, 0b10100, 0b10010, 0b10001,
            ],
            'S' => [
                0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
            ],
            'T' => [
                0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
            ],
            'U' => [
                0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
            ],
            'V' => [
                0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
            ],
            'W' => [
                0b10001, 0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011,
            ],
            'X' => [
                0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
            ],
            '0' => [
                0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
            ],
            '1' => [
                0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
            ],
            '2' => [
                0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111,
            ],
            '3' => [
                0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110,
            ],
            '4' => [
                0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
            ],
            '5' => [
                0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
            ],
            '6' => [
                0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
            ],
            '7' => [
                0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
            ],
            '8' => [
                0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
            ],
            '9' => [
                0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b00110,
            ],
            '.' => [
                0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100,
            ],
            '-' => [
                0b00000, 0b00000, 0b00000, 0b01110, 0b00000, 0b00000, 0b00000,
            ],
            ':' => [
                0b00000, 0b00100, 0b00000, 0b00000, 0b00100, 0b00000, 0b00000,
            ],
            '/' => [
                0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000, 0b00000,
            ],
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
                core::iter::repeat_n(color, count),
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

    /// Paint one of the four discovery states on the LCD. Full-screen paint
    /// so the user sees progress without a USB console.
    ///
    ///   WaitingIp  -> "WAITING FOR IP" / "DHCP IN PROGRESS"
    ///   Searching  -> "SEARCHING HL2" / "BROADCASTING"
    ///   NotFound   -> "HL2 NOT FOUND" / "CHECK CABLE + LAN"
    ///   Found      -> "HL2 FOUND" / <IP> / <MAC> / "GW a.b [16|12] BIT RX n"
    fn paint_discovery(
        display: &mut Display,
        info: Option<&hl2::protocol::discovery::DiscoveryInfo>,
    ) {
        let bg: u16 = 0x001E;
        let fg: u16 = 0xFFFF;
        fill_rect(display, 0, 0, WIDTH, HEIGHT, bg);

        let state = DISCOVERY_STATE.load(core::sync::atomic::Ordering::Acquire);
        let (t1, t2): (&str, &str) = match (state, info) {
            (STATE_WAITING, _) => ("WAITING FOR IP", "DHCP IN PROGRESS"),
            (STATE_SEARCHING, _) => ("SEARCHING HL2", "BROADCASTING"),
            (STATE_NOT_FOUND, _) => ("HL2 NOT FOUND", "CHECK CABLE + LAN"),
            (STATE_FOUND, Some(_)) => ("HL2 FOUND", "RX"),
            (STATE_FOUND, None) => ("HL2 FOUND", ""),
            _ => ("?", "?"),
        };

        // Title (scale 4).
        draw_text(display, 10, 12, 4, t1, fg, bg);
        // Body (scale 3).
        if !t2.is_empty() {
            draw_text(display, 10, 72, 3, t2, fg, bg);
        }

        if let Some(i) = info {
            // IP line: "a.b.c.d"
            let mut s = [0u8; 16];
            let mut p = 0usize;
            for octet in i.ip.iter() {
                if p > 0 {
                    s[p] = b'.';
                    p += 1;
                }
                s[p] = b'0' + octet / 10;
                p += 1;
                s[p] = b'0' + octet % 10;
                p += 1;
            }
            let ip_str = core::str::from_utf8(&s[..p]).unwrap();
            draw_text(display, 10, 112, 3, ip_str, fg, bg);

            // MAC line: "aa:bb:cc:dd:ee:ff"
            let mut s = [0u8; 20];
            let mut p = 0usize;
            for (n, m) in i.mac.iter().enumerate() {
                if n > 0 {
                    s[p] = b':';
                    p += 1;
                }
                for shift in [4usize, 0usize] {
                    let nib = (*m >> shift) & 0x0F;
                    s[p] = match nib {
                        0..=9 => b'0' + nib,
                        10..=15 => b'A' + nib - 10,
                        _ => 0,
                    };
                    p += 1;
                }
            }
            let mac_str = core::str::from_utf8(&s[..p]).unwrap();
            draw_text(display, 10, 152, 3, mac_str, fg, bg);

            // Gateware: "GW a.b 1[6|2]-BIT RX n"
            let mut s = [0u8; 28];
            let mut p = 0usize;
            s[p] = b'G';
            p += 1;
            s[p] = b'W';
            p += 1;
            s[p] = b' ';
            p += 1;
            s[p] = b'0' + i.gateware_major % 10;
            p += 1;
            s[p] = b'.';
            p += 1;
            s[p] = b'0' + i.gateware_minor % 10;
            p += 1;
            s[p] = b' ';
            p += 1;
            s[p] = b'1';
            p += 1;
            s[p] = if i.sample_16bit { b'6' } else { b'2' };
            p += 1;
            s[p] = b'-';
            p += 1;
            s[p] = b'B';
            p += 1;
            s[p] = b'I';
            p += 1;
            s[p] = b'T';
            p += 1;
            s[p] = b' ';
            p += 1;
            s[p] = b'R';
            p += 1;
            s[p] = b'X';
            p += 1;
            s[p] = b' ';
            p += 1;
            s[p] = b'0' + i.rx_count / 10;
            p += 1;
            s[p] = b'0' + i.rx_count % 10;
            p += 1;
            let gw_str = core::str::from_utf8(&s[..p]).unwrap();
            draw_text(display, 10, 192, 3, gw_str, fg, bg);
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

    /// Paint the current discovery state on the display. Each pass re-renders
    /// based on `DISCOVERY_STATE` / `HL2_FOUND`. The display is passed in via
    /// `spawn(display)` as a free (task-local) resource, following the same
    /// convention as `ethernet(ccm, ccm_analog, iomuxc_gpr)` in the baseline.
    /// `hello_world` is the sole owner, so no sync is required.
    #[task]
    async fn hello_world(_cx: hello_world::Context, mut display: Display) {
        loop {
            paint_discovery(&mut display, hl2_found_ref());
            Systick::delay(500.millis()).await;
        }
    }

    #[task(binds = USB_OTG1, priority = 2, local = [poller])]
    fn log_over_usb(cx: log_over_usb::Context) {
        cx.local.poller.poll();
    }
}
