//! Rotary-encoder RX1 NCO tuning: pins 28 (A) / 29 (B), edge-triggered GPIO
//! interrupts on both quadrature channels.
//!
//! Pin/port mapping on the RT1060 (i.MX RT 1062 family):
//!
//! * p28 → `GPIO_EMC_32` pad ⇒ GPIO3_IO18  (bit 18 ≥ 0x8000 = port 1 of GPIO3)
//! * p29 → `GPIO_EMC_31` pad ⇒ GPIO4_IO31  (bit 31 ≥ 0x8000 = port 1 of GPIO4)
//!
//! Because the RT1060 routes each 16-bit half of a GPIO port through a
//! separate IRQ vector, the active vectors are the *combined* "16..31"
//! ones for both ports (pin 18 is in the upper half of GPIO3; pin 31 is in
//! the upper half of GPIO4):
//!
//! * `GPIO3_COMBINED_16_31` (irq 85) → handles the A channel (pin 28)
//! * `GPIO4_COMBINED_16_31` (irq 87) → handles the B channel (pin 29)
//!
//! Both vectors are bound in the RTIC `[task(binds = …)]` handlers in
//! `main.rs`; the ISR tasks there call [`handle_edge`] (which advances the
//! quadrature state machine and accumulates a pending step count) and then
//! [`clear_isr_gpio3_pin18`] / [`clear_isr_gpio4_pin31`] (W1C).
//!
//! The `radio_task` polls [`take_steps`] every loop iteration; for each
//! pending step it calls `handle.send_tune(NCO ± step_hz())`. Per-step is
//! user-configurable via [`set_step_hz`]; the current NCO is published via
//! [`crate::shared::set_nco_hz`] / [`crate::shared::nco_hz`] so the
//! `render` task can show it, and the cumulative retune count is available
//! via [`retunes`].

use core::cell::RefCell;
use core::sync::atomic::{AtomicI32, Ordering};
use cortex_m::interrupt::Mutex;
use teensy4_bsp::hal::gpio::{Port, Trigger};
use teensy4_bsp::pins::t41::{P28, P29};

/// Per-step RX1 NCO increment (Hz). Positive = CW = up; the state machine
/// also emits `−1` for CCW, so the total shift is `step × step_hz`.
pub const DEFAULT_STEP_HZ: i32 = 500;

static ENC_STEPS: AtomicI32 = AtomicI32::new(0);
static ENC_RETUNES: AtomicI32 = AtomicI32::new(0);
static ENC_STEP_HZ: AtomicI32 = AtomicI32::new(DEFAULT_STEP_HZ);
static ENC_PHASE: Mutex<RefCell<u8>> = Mutex::new(RefCell::new(0));

/// Set the per-step NCO increment. The radio task reads it on the next
/// retune via [`step_hz`].
pub fn set_step_hz(hz: i32) {
    ENC_STEP_HZ.store(hz, Ordering::Release);
}

/// Current per-step NCO increment (Hz).
pub fn step_hz() -> i32 {
    ENC_STEP_HZ.load(Ordering::Acquire)
}

/// Atomically add `n` signed steps to the pending counter.
fn add_steps(n: i32) {
    ENC_STEPS.fetch_add(n, Ordering::AcqRel);
}

/// Drain the pending steps and bump the cumulative retune counter.
///
/// Returns the total signed delta the radio task should apply to the NCO
/// (in units of `step_hz()`).
pub fn take_steps() -> i32 {
    let n = ENC_STEPS.swap(0, Ordering::AcqRel);
    ENC_RETUNES.fetch_add(if n != 0 { 1 } else { 0 }, Ordering::Relaxed);
    n
}

/// Cumulative number of times the encoder has caused a retune.
pub fn retunes() -> i32 {
    ENC_RETUNES.load(Ordering::Acquire)
}

/// Initialize pad pulls / hysteresis and the two GPIO inputs with both-edge
/// triggers, plus the NVIC priorities for the two combined vectors.
///
/// This is the sole caller of `set_interrupt` for these pins; once called,
/// the pins are live and the ISR may fire on any subsequent edge.
pub fn init(mut gpio3: Port, mut gpio4: Port, mut p28: P28, mut p29: P29) {
    // Pull-up + hysteresis (Schmitt-trigger) for noise-robust edges.
    let pad = imxrt_iomuxc::Config::modify()
        .set_pull_keeper(Some(imxrt_iomuxc::PullKeeper::Pullup22k))
        .set_hysteresis(imxrt_iomuxc::Hysteresis::Enabled);
    imxrt_iomuxc::configure(&mut p28, pad);
    imxrt_iomuxc::configure(&mut p29, pad);

    let a = gpio3.input(p28).expect("p28 is a GPIO3 pin");
    let b = gpio4.input(p29).expect("p29 is a GPIO4 pin");
    gpio3
        .set_interrupt(&a, Some(Trigger::EitherEdge))
        .expect("a is on GPIO3 (same port)");
    gpio4
        .set_interrupt(&b, Some(Trigger::EitherEdge))
        .expect("b is on GPIO4 (same port)");

    // NVIC: RTIC binds the `[task(binds = …)]` handlers for these vectors,
    // so the GPIO *block* (EDGE_SEL/IMR) is configured above and the vector
    // entries are installed by RTIC. The NVIC *priorities* are ours to set.
    // Cortex-M4's IPR register (0xE000_E400) is byte-sized, one slot per
    // vector, and priorities are stored at 4-bit grain so logical level 1
    // => 0x10. Level 1 puts the encoder handlers above the priority-0
    // radio/render/audio software tasks, below the priority-3 DMA IRQs —
    // fast, never preempted by the radio pump, and it cannot starve the DMA
    // channels.
    unsafe {
        cortex_m::peripheral::NVIC::unmask(teensy4_bsp::Interrupt::GPIO3_COMBINED_16_31);
        cortex_m::peripheral::NVIC::unmask(teensy4_bsp::Interrupt::GPIO4_COMBINED_16_31);
        let ipr = 0xE000_E400u32 as *mut u8;
        *ipr.add(85) = 0x10; // GPIO3_COMBINED_16_31
        *ipr.add(87) = 0x10; // GPIO4_COMBINED_16_31
    }

    // Seed the quadrature phase to the current pad state so the first
    // detent is detected relative to the actual (real, next) pair, not
    // `(0, real)`, which would register one spurious step.
    set_phase(read_level());
}

/// Read the A/B pair from each port's `PSR` (the live pad-value register).
/// A = p28 (GPIO3 bit 18) — LSB; B = p29 (GPIO4 bit 31) — bit 1.
#[inline(always)]
fn read_level() -> u8 {
    // SAFETY: GPIO3/GPIO4 are MMIO base addresses for two distinct blocks;
    // PSR is a read-only register in each.
    unsafe {
        let a = (*teensy4_bsp::ral::gpio::GPIO3).PSR.read() & (1u32 << 18) != 0;
        let b = (*teensy4_bsp::ral::gpio::GPIO4).PSR.read() & (1u32 << 31) != 0;
        (a as u8) | ((b as u8) << 1)
    }
}

/// Current stored quadrature phase (A<<1 | B), 0..=3.
fn phase() -> u8 {
    cortex_m::interrupt::free(|cs| *ENC_PHASE.borrow(cs).borrow())
}

fn set_phase(p: u8) {
    cortex_m::interrupt::free(|cs| *ENC_PHASE.borrow(cs).borrow_mut() = p);
}

/// Advance the quadrature state machine. Returns +1 for a CW step, −1 for
/// a CCW step, 0 for a bounce/no-op.
///
/// A valid detent moves `prev → cur` through one of these four-state
/// sequences:
///
///   CW  : 00 → 10 → 11 → 01 → 00 → …   (A leading, B lagging)
///   CCW : 00 → 01 → 11 → 10 → 00 → …   (B leading, A lagging)
///
/// The table below is the union of both. Any other transition (e.g. a
/// double-tap on the same channel, or a missed edge between two valid
/// positions) is treated as noise and ignored — the next valid transition
/// will still be detected.
///
/// NOTE: the sign convention is chosen so that a *typical* knob with A
/// leading (i.e. A=1 first, then B=1) is **CW → +1**. If yours feels
/// inverted, swap A/B in the pin map or invert the delta on the radio side.
#[inline(always)]
fn advance(prev: u8, cur: u8) -> i32 {
    match (prev, cur) {
        // CW (A-leading convention)
        (0, 2) => 1,
        (2, 3) => 1,
        (3, 1) => 1,
        (1, 0) => 1,
        // CCW (B-leading convention)
        (0, 1) => -1,
        (1, 3) => -1,
        (3, 2) => -1,
        (2, 0) => -1,
        // No phase change (bounce / repeat edge).
        _ => 0,
    }
}

/// Common edge body invoked by both ISR tasks. Reads the current level,
/// advances the state machine, and accumulates a pending step if the
/// transition was a valid detent.
///
/// Both A and B ISR handlers call this — the ISR that owns the pin is
/// still responsible for clearing its `ISR` bit (W1C) *after* calling
/// this; see [`clear_isr_gpio3_pin18`] / [`clear_isr_gpio4_pin31`].
#[inline]
pub fn handle_edge() {
    let prev = phase();
    let cur = read_level();
    let d = advance(prev, cur);
    if d != 0 {
        add_steps(d);
        set_phase(cur);
    }
}

/// W1C-clear the A-channel pin (GPIO3_IO18) bit in `ISR`. Must be called
/// from the A-channel ISR after [`handle_edge`] has consumed the edge; the
/// ISR bit latches until written, so a missed clear would re-fire the vector.
#[inline]
pub fn clear_isr_gpio3_pin18() {
    // SAFETY: the GPIO3 base pointer is a valid MMIO block for the lifetime
    // of the run; ISR is a W1C RWRegister.
    unsafe {
        (*teensy4_bsp::ral::gpio::GPIO3).ISR.write(1u32 << 18);
    }
}

/// W1C-clear the B-channel pin (GPIO4_IO31) bit in `ISR`. Must be called
/// from the B-channel ISR after [`handle_edge`] has consumed the edge.
#[inline]
pub fn clear_isr_gpio4_pin31() {
    // SAFETY: same argument as `clear_isr_gpio3_pin18`, for GPIO4.
    unsafe {
        (*teensy4_bsp::ral::gpio::GPIO4).ISR.write(1u32 << 31);
    }
}
