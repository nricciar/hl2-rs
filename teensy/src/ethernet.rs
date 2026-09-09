//! Teensy 4.1 on-board DP83825 / ENET1 bring-up.
//!
//! Register/pad mapping: i.MX RT1060 reference manual (CCM, IOMUXC, ENET)
//! and DP83825I datasheet (reset timing, RCSR, ANAR, PHY IDs).
//! Board reference: https://github.com/PaulStoffregen/teensy41_ethernet
//! Unlike Teensyduino, teensy4-bsp uses GPIO2, not fast GPIO7.

use embedded_hal::delay::DelayNs;
use imxrt_enet::{Duplex, Enet, ReceiveBuffers, TransmitBuffers};
use smoltcp::{phy, time::Instant};
use static_cell::ConstStaticCell;
use teensy4_bsp::{board, ral};

// Keep descriptors AND payloads in non-cacheable DTCM (.bss), which is
// accessible to ENET through the CM7 backdoor. Do not move these to cached
// OCRAM without adding cache maintenance; atomics alone are not sufficient.
// 64 RX descriptors so the MAC ring absorbs EP6 bursts while the render
// task blocks the core for the full-screen SPI blit (≈37 ms at 22 MHz SCK).
// With only 4 buffers the ring overflows after ~5 ms of starvation and ~50
// datagrams are dropped per paint, starving the pipeline.
static RX: ConstStaticCell<ReceiveBuffers<64>> = ConstStaticCell::new(ReceiveBuffers::new());
static TX: ConstStaticCell<TransmitBuffers<4>> = ConstStaticCell::new(TransmitBuffers::new());

const ENET_CLOCK_HZ: u32 = 50_000_000;
const PHY_RESET: u32 = 1 << 14; // GPIO_B0_14 / GPIO2_IO14
const PHY_POWER: u32 = 1 << 15; // GPIO_B0_15 / GPIO2_IO15

pub struct Ethernet(Enet);

impl Ethernet {
    // Called exactly once, after board::t41 and LCD initialization. The BSP
    // discards ENET1 and the internal PHY pads; no other task may use them.
    pub fn new(
        ccm: &mut ral::ccm::CCM,
        analog: &mut ral::ccm_analog::CCM_ANALOG,
        gpr: &mut ral::iomuxc_gpr::IOMUXC_GPR,
        delay: &mut impl DelayNs,
        mac: &[u8; 6],
    ) -> Result<Self, &'static str> {
        // CCGR1.CG5 gates ENET register access. Accessing ENET first can hang
        // the bus, before even a software timeout gets a chance to run.
        ral::modify_reg!(ral::ccm, ccm, CCGR1, CG5: 3);
        ral::modify_reg!(ral::ccm_analog, analog, PLL_ENET,
            BYPASS: 1, POWERDOWN: 0, ENABLE: 1, DIV_SELECT: 1);
        let start = cortex_m::peripheral::DWT::cycle_count();
        while ral::read_reg!(ral::ccm_analog, analog, PLL_ENET, LOCK == 0) {
            if cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start)
                >= board::ARM_FREQUENCY / 10
            {
                return Err("PLL6 lock timeout");
            }
        }
        ral::modify_reg!(ral::ccm_analog, analog, PLL_ENET, BYPASS: 0);
        ral::modify_reg!(ral::iomuxc_gpr, gpr, GPR1,
            ENET1_CLK_SEL: 0, ENET1_TX_CLK_DIR: 1, ENET_IPG_CLK_S_EN: 1);

        // Safety: only internal PHY pads and GPIO2 bits 14/15 are touched.
        // They are not exposed by bsp::pins::t41. GPIO2 is already clocked;
        // LCD GPIO configuration has finished, and no task changes its GDIR.
        unsafe {
            let pads = &*ral::iomuxc::IOMUXC;
            let gpio = &*ral::gpio::GPIO2;
            gpr.GPR27.write(gpr.GPR27.read() & !(PHY_RESET | PHY_POWER));
            pads.SW_PAD_CTL_PAD_GPIO_B0_14.write(0x0038);
            pads.SW_PAD_CTL_PAD_GPIO_B0_15.write(0x0038);
            pads.SW_MUX_CTL_PAD_GPIO_B0_14.write(5);
            pads.SW_MUX_CTL_PAD_GPIO_B0_15.write(5);
            gpio.DR_CLEAR.write(PHY_RESET | PHY_POWER);
            gpio.GDIR.write(gpio.GDIR.read() | PHY_RESET | PHY_POWER);

            // Strap PHY address 0, RMII slave, auto-MDIX. Hold these pulls
            // through reset; RXD1's internal pulldown may still win, so RCSR
            // explicitly selects the external 50 MHz clock below.
            pads.SW_PAD_CTL_PAD_GPIO_B1_04.write(0x3038);
            pads.SW_PAD_CTL_PAD_GPIO_B1_05.write(0xF038);
            pads.SW_PAD_CTL_PAD_GPIO_B1_06.write(0x3038);
            pads.SW_PAD_CTL_PAD_GPIO_B1_11.write(0x3038);
            pads.SW_PAD_CTL_PAD_GPIO_B1_10.write(0x0031);
            pads.SW_MUX_CTL_PAD_GPIO_B1_10.write(6 | 0x10); // REF_CLK + SION
            pads.ENET_IPG_CLK_RMII_SELECT_INPUT.write(1);

            pads.SW_PAD_CTL_PAD_GPIO_B1_15.write(0xF829); // MDIO open drain + pull-up
            pads.SW_PAD_CTL_PAD_GPIO_B1_14.write(0xB0E9); // MDC
            pads.SW_MUX_CTL_PAD_GPIO_B1_15.write(0);
            pads.SW_MUX_CTL_PAD_GPIO_B1_14.write(0);
            pads.ENET_MDIO_SELECT_INPUT.write(2);

            gpio.DR_SET.write(PHY_POWER);
            delay.delay_ms(50); // Clock and power stable while RESET_N is low.
            gpio.DR_SET.write(PHY_RESET);
            delay.delay_ms(2); // DP83825 reset-to-SMI-ready time.
        }

        log::info!("Ethernet: PLL6 locked, PHY released from reset; initializing MAC");
        // Safety: board::t41 discarded this instance, and this module is its
        // sole owner. Its clocks and reference clock have now been enabled.
        let instance = unsafe { ral::enet::ENET1::instance() };
        let mut device = Self(Enet::new(
            instance,
            TX.take().take(),
            RX.take().take(),
            ENET_CLOCK_HZ,
            mac,
        ));
        device.0.enable_rmii_mode(true);
        device.0.set_duplex(Duplex::Full);

        let id1 = device.mdio(2, None)?;
        let id2 = device.mdio(3, None)?;
        log::info!("DP83825 PHY IDs: {id1:#06x} {id2:#06x}");
        if id1 != 0x2000 || id2 & 0xFFF0 != 0xA140 {
            return Err("DP83825 not found at MDIO address 0 (check clock, reset, MDIO)");
        }
        device.mdio(0x17, Some(0x0081))?; // RCSR: 50 MHz slave, 2-bit elasticity
        device.mdio(0x18, Some(0x0280))?; // LEDCR: active-high link LED, 10 Hz
        // Bring-up deliberately advertises only 100BASE-TX full duplex, so
        // the negotiated mode must match the MAC. Do not force the peer.
        device.mdio(0x04, Some(0x0101))?; // ANAR: IEEE 802.3 + 100TX full
        device.mdio(0x00, Some(0x3300))?; // BMCR: enable/restart negotiation

        // Safety: same exclusively owned internal pads as above, now that
        // PHY strap sampling is complete. None overlap the LCD's SPI pins.
        unsafe {
            let pads = &*ral::iomuxc::IOMUXC;
            for pad in [
                &pads.SW_PAD_CTL_PAD_GPIO_B1_04,
                &pads.SW_PAD_CTL_PAD_GPIO_B1_05,
                &pads.SW_PAD_CTL_PAD_GPIO_B1_06,
                &pads.SW_PAD_CTL_PAD_GPIO_B1_07,
                &pads.SW_PAD_CTL_PAD_GPIO_B1_08,
                &pads.SW_PAD_CTL_PAD_GPIO_B1_09,
                &pads.SW_PAD_CTL_PAD_GPIO_B1_11,
            ] {
                pad.write(0xB0E9);
            }
            for mux in [
                &pads.SW_MUX_CTL_PAD_GPIO_B1_04,
                &pads.SW_MUX_CTL_PAD_GPIO_B1_05,
                &pads.SW_MUX_CTL_PAD_GPIO_B1_06,
                &pads.SW_MUX_CTL_PAD_GPIO_B1_07,
                &pads.SW_MUX_CTL_PAD_GPIO_B1_08,
                &pads.SW_MUX_CTL_PAD_GPIO_B1_09,
                &pads.SW_MUX_CTL_PAD_GPIO_B1_11,
            ] {
                mux.write(3);
            }
            pads.ENET0_RXDATA_SELECT_INPUT.write(1);
            pads.ENET1_RXDATA_SELECT_INPUT.write(1);
            pads.ENET_RXEN_SELECT_INPUT.write(1);
            pads.ENET_RXERR_SELECT_INPUT.write(1);

            // imxrt-enet 0.1.0 advertises checksum offload, but initializes
            // TX descriptor IINS/PINS to zero. Use software checksums for
            // bring-up rather than transmitting zero IPv4/ICMP checksums.
            let enet = &*ral::enet::ENET1;
            ral::modify_reg!(ral::enet, enet, TACC, IPCHK: 0, PROCHK: 0);
        }
        device.0.enable_mac(true);
        log::info!("Ethernet: RMII MAC ready; waiting for 100 Mbps full-duplex link");
        Ok(device)
    }

    // The dependency's MDIO methods spin forever. Bound every transaction;
    // holding &mut self keeps raw MDIO access exclusive with MAC access.
    fn mdio(&mut self, register: u8, write: Option<u16>) -> Result<u16, &'static str> {
        // Safety: this object owns the live ENET1 peripheral, and no IRQ or
        // other task accesses it. Only its management registers are touched.
        let enet = unsafe { &*ral::enet::ENET1 };
        ral::write_reg!(ral::enet, enet, EIR, MII: 1);
        ral::write_reg!(ral::enet, enet, MMFR,
            ST: 1, OP: if write.is_some() { 1 } else { 2 }, PA: 0,
            RA: register as u32, TA: 2, DATA: write.unwrap_or(0) as u32);
        let start = cortex_m::peripheral::DWT::cycle_count();
        while ral::read_reg!(ral::enet, enet, EIR, MII == 0) {
            if cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start)
                >= board::ARM_FREQUENCY / 1_000
            {
                return Err("MDIO completion timeout");
            }
        }
        let value = ral::read_reg!(ral::enet, enet, MMFR, DATA) as u16;
        ral::write_reg!(ral::enet, enet, EIR, MII: 1);
        Ok(value)
    }

    pub fn link_up(&mut self) -> Result<bool, &'static str> {
        self.mdio(1, None)?; // BMSR link bit is latch-low: read twice.
        let status = self.mdio(1, None)?;
        if status == 0xFFFF {
            return Err("PHY stopped responding");
        }
        // Require both link and completed autonegotiation. Parallel detection
        // of a forced-mode peer must not be mistaken for full duplex.
        let partner = self.mdio(5, None)?;
        Ok(status & 0x24 == 0x24 && partner & 0x0100 != 0)
    }
}

impl phy::Device for Ethernet {
    type RxToken<'a> = <Enet as phy::Device>::RxToken<'a>;
    type TxToken<'a> = <Enet as phy::Device>::TxToken<'a>;

    fn receive(&mut self, now: Instant) -> Option<(Self::RxToken<'_>, Self::TxToken<'_>)> {
        self.0.receive(now)
    }

    fn transmit(&mut self, now: Instant) -> Option<Self::TxToken<'_>> {
        self.0.transmit(now)
    }

    fn capabilities(&self) -> phy::DeviceCapabilities {
        let mut caps = self.0.capabilities();
        caps.checksum = phy::ChecksumCapabilities::default();
        caps
    }
}
