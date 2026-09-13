//! SAI1 slave transmitter for the crystal-clocked WM8731.
//!
//! P21/P20 are RX_BCLK/RX_SYNC, so TX must follow the enabled RX clock
//! section even though we only send data (P7 = TX_DATA00). No MCLK pin is
//! needed. WM8731 normal-mode master timing is 64fs: two 32-bit slots at
//! 48 kHz, with each 16-bit sample left-aligned in its slot.

use teensy4_bsp::hal::dma::peripheral::Destination;
use teensy4_bsp::hal::sai::{self, Mode, Packing, Sai, SaiConfig, SyncMode};
use teensy4_bsp::ral;

/// RT1060 DMAMUX source for SAI1 TX (reference manual DMA request mapping).
pub const SAI1_DMA_TX_SRC: u32 = 20;
/// One FIFO word per stereo slot, with mono duplicated into L and R.
pub const TDR_WORDS_PER_SAMPLE: usize = 2;
// TCSR status flags cleared by writing one; mask them during control updates.
const TCSR_W1C: u32 =
    ral::sai::TCSR::FEF::mask | ral::sai::TCSR::SEF::mask | ral::sai::TCSR::WSF::mask;

/// Owns SAI1 TX and its RX clock section; no other SAI1 user may reconfigure it.
pub struct Tx {
    inner: sai::Tx,
}

impl Tx {
    fn regs(&self) -> &ral::sai::RegisterBlock {
        // SAFETY: init_tx consumes SAI1. The inner HAL handle never escapes;
        // all accesses, including DMA request enable/disable, use this owner.
        unsafe { &*ral::sai::SAI1 }
    }

    /// TCR1..5, TCSR, RCR2, RCR4, RCSR, including raw control/status bits.
    pub fn reg_dump(&mut self) -> [u32; 9] {
        let tx = self.inner.reg_dump();
        let regs = self.regs();
        [
            tx[0],
            tx[1],
            tx[2],
            tx[3],
            tx[4],
            tx[5],
            ral::read_reg!(ral::sai, regs, RCR2),
            ral::read_reg!(ral::sai, regs, RCR4),
            ral::read_reg!(ral::sai, regs, RCSR),
        ]
    }

    pub fn fifo_position(&mut self) -> (u32, u32) {
        self.inner.fifo_position(0)
    }
}

// SAFETY: the fixed SAI1 TX request writes one u32 to the owned TDR[0].
// Unlike imxrt-hal 0.6's FWDE implementation, FRDE services the watermark
// rather than waiting for an empty FIFO. Cancellation must clear FRDE too.
unsafe impl Destination<u32> for Tx {
    fn destination_signal(&self) -> u32 {
        SAI1_DMA_TX_SRC
    }
    fn destination_address(&self) -> *const u32 {
        self.inner.tdr(0)
    }
    fn enable_destination(&mut self) {
        let regs = self.regs();
        ral::modify_reg!(ral::sai, regs, TCSR, |v| {
            (v & !TCSR_W1C) | ral::sai::TCSR::FRDE::mask
        });
    }
    fn disable_destination(&mut self) {
        let regs = self.regs();
        ral::modify_reg!(ral::sai, regs, TCSR, |v| {
            v & !(TCSR_W1C | ral::sai::TCSR::FRDE::mask)
        });
    }
}

/// Configure while disabled, then enable RX clock synchronization and TX.
pub fn init_tx(sai1: ral::sai::Instance<1>) -> Result<Tx, sai::InvalidPackingError> {
    // without_pins does not reset the peripheral, unlike Sai::new.
    ral::write_reg!(ral::sai, sai1, TCSR, SR: 1);
    ral::write_reg!(ral::sai, sai1, RCSR, SR: 1);
    ral::write_reg!(ral::sai, sai1, TCSR, 0);
    ral::write_reg!(ral::sai, sai1, RCSR, 0);
    ral::write_reg!(ral::sai, sai1, TCSR, FR: 1);
    ral::write_reg!(ral::sai, sai1, RCSR, FR: 1);
    ral::write_reg!(ral::sai, sai1, TMR, 0);
    ral::write_reg!(ral::sai, sai1, RMR, 0);

    let sai = Sai::without_pins(sai1, 1, 1);
    let cfg = SaiConfig {
        mode: Mode::Slave,
        sync_mode: SyncMode::TxFollowRx,
        tx_fifo_wm: 16,
        ..SaiConfig::i2s(2)
    };
    let (tx, _rx) = sai.split(32, 2, Packing::None, &cfg)?;
    let mut tx = Tx {
        inner: tx.expect("TX channel 0 enabled"),
    };
    let regs = tx.regs();

    // HAL 0.6 ignores SYNC in slave mode. Match AudioOutputI2Sslave's
    // RT1062 setup explicitly: RX receives clocks, TX follows RX. TX FSD
    // selects the internal synchronized frame sync, not the external RX pin.
    ral::write_reg!(ral::sai, regs, TCR2, SYNC: 1, BCP: 1, BCD: 0);
    ral::write_reg!(ral::sai, regs, RCR2, SYNC: 0, BCP: 1, BCD: 0);
    // Continue after FIFO starvation at startup or a delayed chunk rearm.
    ral::modify_reg!(ral::sai, regs, TCR4, FSD: 1, FCONT: 1);
    // Only the RX clock section is needed; do not capture unused P8 data.
    ral::write_reg!(ral::sai, regs, RCR3, RCE: 0);
    ral::write_reg!(ral::sai, regs, RCSR, RE: 1, BCE: 1);
    tx.inner.set_enable(true);
    Ok(tx)
}
