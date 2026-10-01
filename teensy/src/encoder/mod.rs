//! RX1 tuning on p28 (A) / p29 (B), with a mode-cycle button on p30.
//!
//! A (GPIO3_IO18) and the button (GPIO3_IO23) share GPIO3_COMBINED_16_31;
//! B (GPIO4_IO31) uses GPIO4_COMBINED_16_31. RTIC owns both NVIC vectors.
//! Each valid quadrature edge counts as one step, not one detent.

#[cfg(target_arch = "arm")]
pub use hardware::*;

/// Decode an edge and resynchronize to the sampled phase, even after a missed edge.
/// Phase is A | (B << 1). Preserve the positive, B-leading cycle 0 -> 2 -> 3 -> 1 -> 0;
/// physical rotation direction depends on the encoder wiring.
#[inline(always)]
fn advance(prev: &mut u8, cur: u8) -> i32 {
    let delta = match (*prev, cur) {
        (0, 2) | (2, 3) | (3, 1) | (1, 0) => 1,
        (0, 1) | (1, 3) | (3, 2) | (2, 0) => -1,
        // Repeated samples and invalid two-bit jumps contribute no steps.
        _ => 0,
    };
    *prev = cur;
    delta
}

#[cfg(target_arch = "arm")]
mod hardware {
    use super::advance;
    use core::cell::RefCell;
    use core::sync::atomic::{AtomicBool, AtomicI32, Ordering};
    use cortex_m::interrupt::Mutex;
    use teensy4_bsp::hal::gpio::{Port, Trigger};
    use teensy4_bsp::pins::t41::{P28, P29, P30};

    /// RX1 NCO increment per signed quadrature edge (Hz).
    pub const DEFAULT_STEP_HZ: i32 = 25;

    static ENC_STEPS: AtomicI32 = AtomicI32::new(0);
    static ENC_RETUNES: AtomicI32 = AtomicI32::new(0);
    static ENC_STEP_HZ: AtomicI32 = AtomicI32::new(DEFAULT_STEP_HZ);
    static ENC_PHASE: Mutex<RefCell<u8>> = Mutex::new(RefCell::new(0));
    static BTN_TOGGLE_REQ: AtomicBool = AtomicBool::new(false);

    /// Set the per-edge NCO increment used on the next retune.
    pub fn set_step_hz(hz: i32) {
        ENC_STEP_HZ.store(hz, Ordering::Release);
    }

    /// Current per-edge NCO increment (Hz).
    pub fn step_hz() -> i32 {
        ENC_STEP_HZ.load(Ordering::Acquire)
    }

    /// Drain signed edge counts, in units of [`step_hz`], and count nonzero retunes.
    pub fn take_steps() -> i32 {
        let n = ENC_STEPS.swap(0, Ordering::AcqRel);
        ENC_RETUNES.fetch_add(if n != 0 { 1 } else { 0 }, Ordering::Relaxed);
        n
    }

    /// Cumulative number of times the encoder has caused a retune.
    pub fn retunes() -> i32 {
        ENC_RETUNES.load(Ordering::Acquire)
    }

    /// Configure GPIO triggers and seed the decoder before RTIC enables interrupts.
    pub fn init(mut gpio3: Port, mut gpio4: Port, mut p28: P28, mut p29: P29, mut p30: P30) {
        let pad = imxrt_iomuxc::Config::modify()
            .set_pull_keeper(Some(imxrt_iomuxc::PullKeeper::Pullup22k))
            .set_hysteresis(imxrt_iomuxc::Hysteresis::Enabled);
        imxrt_iomuxc::configure(&mut p28, pad);
        imxrt_iomuxc::configure(&mut p29, pad);
        imxrt_iomuxc::configure(&mut p30, pad);

        let a = gpio3.input(p28).expect("p28 is a GPIO3 pin");
        let b = gpio4.input(p29).expect("p29 is a GPIO4 pin");
        let btn = gpio3.input(p30).expect("p30 is a GPIO3 pin");
        gpio3
            .set_interrupt(&a, Some(Trigger::EitherEdge))
            .expect("a is on GPIO3 (same port)");
        gpio4
            .set_interrupt(&b, Some(Trigger::EitherEdge))
            .expect("b is on GPIO4 (same port)");
        gpio3
            .set_interrupt(&btn, Some(Trigger::FallingEdge))
            .expect("btn is on GPIO3 (same port)");

        // Discard setup edges, then sample the initial phase while RTIC holds IRQs off.
        clear_isr_gpio3_pin18();
        clear_isr_gpio4_pin31();
        clear_isr_gpio3_pin23();
        cortex_m::interrupt::free(|cs| *ENC_PHASE.borrow(cs).borrow_mut() = read_level());
    }

    /// Read A (GPIO3 bit 18) into bit 0 and B (GPIO4 bit 31) into bit 1.
    #[inline(always)]
    fn read_level() -> u8 {
        // SAFETY: GPIO3/GPIO4 are valid MMIO blocks; PSR is read-only.
        unsafe {
            let a = (*teensy4_bsp::ral::gpio::GPIO3).PSR.read() & (1u32 << 18) != 0;
            let b = (*teensy4_bsp::ral::gpio::GPIO4).PSR.read() & (1u32 << 31) != 0;
            (a as u8) | ((b as u8) << 1)
        }
    }

    /// Sample and decode an edge. The calling ISR must then clear its pending GPIO bit.
    #[inline]
    pub fn handle_edge() {
        let delta = cortex_m::interrupt::free(|cs| {
            let mut phase = ENC_PHASE.borrow(cs).borrow_mut();
            advance(&mut phase, read_level())
        });
        if delta != 0 {
            ENC_STEPS.fetch_add(delta, Ordering::AcqRel);
        }
    }

    /// W1C-clear the A-channel pending bit after [`handle_edge`] (or during init).
    #[inline]
    pub fn clear_isr_gpio3_pin18() {
        // SAFETY: GPIO3 is a valid MMIO block; ISR is W1C, so only bit 18 is cleared.
        unsafe {
            (*teensy4_bsp::ral::gpio::GPIO3).ISR.write(1u32 << 18);
        }
    }

    /// W1C-clear the B-channel pending bit after [`handle_edge`] (or during init).
    #[inline]
    pub fn clear_isr_gpio4_pin31() {
        // SAFETY: GPIO4 is a valid MMIO block; ISR is W1C, so only bit 31 is cleared.
        unsafe {
            (*teensy4_bsp::ral::gpio::GPIO4).ISR.write(1u32 << 31);
        }
    }

    /// Latch a falling-edge button request; repeated requests coalesce until drained.
    #[inline]
    pub fn handle_button_press() {
        BTN_TOGGLE_REQ.store(true, Ordering::Release);
    }

    /// W1C-clear the button pending bit after [`handle_button_press`] (or during init).
    #[inline]
    pub fn clear_isr_gpio3_pin23() {
        // SAFETY: GPIO3 is a valid MMIO block; ISR is W1C, so only bit 23 is cleared.
        unsafe {
            (*teensy4_bsp::ral::gpio::GPIO3).ISR.write(1u32 << 23);
        }
    }

    /// Pending GPIO3 upper-half bits: A is bit 18, button is bit 23.
    #[inline]
    pub fn isr_gpio3_upper_active() -> u32 {
        // SAFETY: GPIO3 is a valid MMIO block; reading ISR does not clear pending bits.
        unsafe { (*teensy4_bsp::ral::gpio::GPIO3).ISR.read() & 0xFFFF_0000 }
    }

    /// Drain the pending mode-cycle request. This flag does not debounce the button.
    #[inline]
    pub fn take_mode_request() -> bool {
        BTN_TOGGLE_REQ.swap(false, Ordering::AcqRel)
    }
}

#[cfg(test)]
mod tests {
    use super::advance;

    #[test]
    fn all_phase_transitions_preserve_sign_and_update_phase() {
        let expected = [[0, -1, 1, 0], [1, 0, 0, -1], [-1, 0, 0, 1], [0, 1, -1, 0]];
        for (prev, row) in expected.iter().enumerate() {
            for (cur, &delta) in row.iter().enumerate() {
                let mut phase = prev as u8;
                assert_eq!(advance(&mut phase, cur as u8), delta, "{prev} -> {cur}");
                assert_eq!(phase, cur as u8);
            }
        }
    }

    #[test]
    fn cycles_count_four_edges_not_one_detent() {
        for (cycle, sign) in [([0, 2, 3, 1], 1), ([0, 1, 3, 2], -1)] {
            for start in 0..4 {
                let mut phase = cycle[start];
                let mut steps = 0;
                for edge in 1..=4 {
                    let delta = advance(&mut phase, cycle[(start + edge) % 4]);
                    assert_eq!(delta, sign);
                    steps += delta;
                }
                assert_eq!(steps, 4 * sign);
                assert_eq!(phase, cycle[start]);
            }
        }
    }

    #[test]
    fn invalid_jump_resynchronizes_before_next_edge() {
        let mut phase = 0;
        assert_eq!(advance(&mut phase, 3), 0);
        assert_eq!(advance(&mut phase, 1), 1);
        assert_eq!(advance(&mut phase, 0), 1);
    }

    #[test]
    fn repeated_samples_and_reversals_do_not_add_net_steps() {
        let mut phase = 0;
        let steps: i32 = [0, 2, 2, 0, 0]
            .into_iter()
            .map(|cur| advance(&mut phase, cur))
            .sum();
        assert_eq!(steps, 0);
        assert_eq!(phase, 0);
    }
}
