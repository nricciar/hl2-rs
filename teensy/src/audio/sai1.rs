//! SAI1 → WM8731 I2S transmitter (slave, 16-bit stereo, TX over eDMA).
//!
//! SAI1 is the **I2S slave**: the WM8731 is the master and drives BCLK
//! (P21) + LRCLK/FSYNC (P20) from its 12.288 MHz on-shield crystal
//! (12.288 MHz / 256 = 48 kHz sample rate, I2S 16-bit × 2 channels).
//! SAI1 only shifts *audio data* out on P7 (SAI1_TX_DATA00 = TDR[0]).
//!
//! The WM8731 has its own MCLK source (the on-shield 12.288 MHz crystal),
//! so the SAI does not need to drive MCLK. No P23, P36, or P37 mux.
//!
//! The wire is one 16-bit word per channel per frame; a frame therefore carries
//! two 16-bit words (L on TDR[0] word 0, R on word 1, MSB-first). Our demod is
//! mono, so one audio sample is folded into **both** L and R (L = R) — the
//! codec has no mono mode and its headphone outputs come from the L/R DACs.
//!
//! Transmit is DMA-driven: the SAI's `FWDE` bit (enabled below + re-asserted by
//! the `Sink` before each chunk) makes the eDMA pull a TDR `u32` out of a
//! `'static` linear buffer whenever the TX FIFO drops below its watermark. See
//! [`crate::audio::Sink`] for the chunked eDMA + spin-on-completion loop.
//!
//! The clock root + `SAI1` clock gate are already configured by `teensy4-bsp`
//! (audio-PLL derived); in slave mode the SAI's own MCLK division is irrelevant
//! (it clocks off the codec's BCLK/FSYNC), so no further clock work is needed.

use teensy4_bsp::hal::sai::{self, Mode, Packing, Sai, SaiConfig};

/// DMAMUX source signal for SAI1 DMA transmit (RT1060: SAI DMA TX mapping
/// `[20, 22, 84]`, index 0 = SAI1). Mirrors `imxrt-hal`'s
/// `SAI_DMA_TX_MAPPING`.
pub const SAI1_DMA_TX_SRC: u32 = 20;

/// Build the SAI1 TX half as an I2S slave.
///
/// # Errors
///
/// [`sai::InvalidPackingError`] if 16-bit + `Packing::None` were ever made
/// inconsistent (they aren't today, but the HAL models the case).
pub fn init_tx(
    sai1: teensy4_bsp::ral::sai::Instance<1>,
) -> Result<sai::Tx, sai::InvalidPackingError> {
    // tx channel mask = 0b01 → channel 0 (TDR[0]) = SAI1_TX_DATA00 (P7).
    let sai = Sai::without_pins(sai1, 1, 0);

    // Start from the I2S defaults (MSB, standard polarity, sync_early) and
    // override the bits that matter for a slave with no RX half.
    let cfg = SaiConfig {
        mode: Mode::Slave,
        tx_fifo_wm: 8, // FIFO low-watermark that triggers the eDMA request (FWDE).
        ..SaiConfig::i2s(2)
    };

    let (tx, _rx) = sai.split(16, 2, Packing::None, &cfg)?;
    let mut tx = tx.expect("tx chan mask was 1, so split must return a Tx half");

    // Enable the SAI's DMA-request-on-FIFO-warning (FWDE) so the eDMA
    // refills the TX FIFO. The `Sink` re-asserts `enable_dma_transmit` before
    // each chunk (it's a no-op if already set) and owns the eDMA + linear
    // buffer wiring.
    tx.enable_dma_transmit();
    tx.set_enable(true);

    Ok(tx)
}

/// TDR `u32` words the SAI consumes per **one** mono audio sample (L + R fold).
pub const TDR_WORDS_PER_SAMPLE: usize = 2;
