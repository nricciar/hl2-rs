//! Rotary-encoder RX1 NCO tuning: pins 28 (A) / 29 (B), edge-triggered GPIO
//! interrupts on both quadrature channels, plus the encoder's push button
//! (pin 30) — a falling-edge GPIO interrupt that requests an RX1 mode change
//! (see [`handle_button_press`] / [`take_mode_request`]).
//!
//! Pin/port mapping on the RT1060 (i.MX RT 1062 family):
//!
//! * p28 → `GPIO_EMC_32` pad ⇒ GPIO3_IO18  (bit 18, in the **upper** half of GPIO3)
//! * p29 → `GPIO_EMC_31` pad ⇒ GPIO4_IO31  (bit 31, in the **upper** half of GPIO4)
//! * p30 → `GPIO_EMC_37` pad ⇒ GPIO3_IO23  (bit 23, in the **upper** half of GPIO3)
//!
//! The RT1060 routes each 16-bit half of a GPIO port through a **separate
//! IRQ vector**: bits 0..=15 → `GPIOx_COMBINED_0_15`; bits 16..=31 →
//! `GPIOx_COMBINED_16_31`. All three of our pins are in upper halves, so
//! the A channel (bit 18) **and** the button (bit 23) land on the **same
//! vector** `GPIO3_COMBINED_16_31`. The handler for that vector must
//! therefore inspect [`isr_gpio3_upper_active`], and clear each pending bit
//! individually via the W1C-clearers below. The `0_15` vector is bound too
//! (as a fallback / semantic home) but the button never fires it in the
//! current layout.
//!
//! All three vectors are bound in the RTIC `[task(binds = …)]` handlers in
//! `main.rs`. The A/B handlers call [`handle_edge`] (advance the quadrature
//! state machine, accumulate a pending step count); the button handler
//! calls [`handle_button_press`] (latch a pending mode-toggle request).
//! Each clears its own ISR bit with the matching `clear_isr_*` W1C helper.
//!
//! The `radio_task` polls [`take_steps`] and [`take_mode_request`] every
//! loop iteration. Per-step is user-configurable via [`set_step_hz`]; the
//! current NCO is published via [`crate::shared::set_nco_hz`]; the
//! cumulative retune count is available via [`retunes`].
//!
//! The encoder's push button (p30) latches a pending mode-toggle request on
//! a falling edge; the radio task drains it with [`take_mode_request`] in
//! the same drain step as [`take_steps`] and advances the shared RX1 mode
//! index (see `crate::shared::next_rx1_mode`) before rebuilding its virtual
//! receiver (`crate::radio::rx::Rx::set_mode`). The task never samples the
//! pin — the falling-edge ISR latches `true`, a later press just re-latches
//! it, and the radio task sees one request per press.

use core::cell::RefCell;
use core::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use cortex_m::interrupt::Mutex;
use teensy4_bsp::hal::gpio::{Port, Trigger};
use teensy4_bsp::pins::t41::{P28, P29, P30};

/// Per-step RX1 NCO increment (Hz). Positive = CW = up; the state machine
/// also emits `−1` for CCW, so the total shift is `step × step_hz`.
pub const DEFAULT_STEP_HZ: i32 = 500;

static ENC_STEPS: AtomicI32 = AtomicI32::new(0);
static ENC_RETUNES: AtomicI32 = AtomicI32::new(0);
static ENC_STEP_HZ: AtomicI32 = AtomicI32::new(DEFAULT_STEP_HZ);
static ENC_PHASE: Mutex<RefCell<u8>> = Mutex::new(RefCell::new(0));

/// Push button (p30, GPIO3_IO23) — pending RX1 mode-toggle request. The
/// falling-edge ISR latches this to `true` ([`handle_button_press`]); the
/// radio task drains it with [`take_mode_request`] (an atomic flag swap,
/// the same drain pattern as [`take_steps`]) and then advances the shared
/// RX1 mode index and rebuilds its virtual receiver (see
/// `crate::shared::next_rx1_mode` / `crate::radio::rx::Rx::set_mode`).
///
/// Pull-up + hysteresis (see [`init`]) make the idle pad level HIGH, so a
/// *falling* edge is always the press. Hardware debounce is present on the
/// button, so one press = one falling edge = one latch; the radio task does
/// not re-sample the pin or time the edges.
static BTN_TOGGLE_REQ: AtomicBool = AtomicBool::new(false);

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
/// triggers, plus the push button (p30) with a falling-edge trigger, and the
/// NVIC priorities for the three combined vectors.
///
/// This is the sole caller of `set_interrupt` for these pins; once called,
/// the pins are live and the ISR may fire on any subsequent edge.
pub fn init(mut gpio3: Port, mut gpio4: Port, mut p28: P28, mut p29: P29, mut p30: P30) {
    // Pull-up + hysteresis (Schmitt-trigger) for noise-robust edges.
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
        // The push button (p30, GPIO3_IO23, lower half of GPIO3) is on the
        // *other* 16-bit-half vector for the same GPIO3 block.
        cortex_m::peripheral::NVIC::unmask(teensy4_bsp::Interrupt::GPIO3_COMBINED_16_31);
        cortex_m::peripheral::NVIC::unmask(teensy4_bsp::Interrupt::GPIO4_COMBINED_16_31);
        cortex_m::peripheral::NVIC::unmask(teensy4_bsp::Interrupt::GPIO3_COMBINED_0_15);
        let ipr = 0xE000_E400u32 as *mut u8;
        *ipr.add(84) = 0x10; // GPIO3_COMBINED_0_15 (p30 push button)
        *ipr.add(85) = 0x10; // GPIO3_COMBINED_16_31 (p28 A)
        *ipr.add(87) = 0x10; // GPIO4_COMBINED_16_31 (p29 B)
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

// ──────────────────────────────────────────────────────────────────────────
// Push button (p30) → RX1 mode toggle
// ──────────────────────────────────────────────────────────────────────────

/// p30 → `GPIO_EMC_37` pad ⇒ **GPIO3_IO23** (bit 23, lower half of GPIO3 ⇒
/// `GPIO3_COMBINED_0_15`, irq 84).
///
/// Pull-up + hysteresis: idle pad is HIGH, press drives it LOW. The ISR is
/// bound on the *falling* edge only, so one press fires the vector once.
/// [`handle_button_press`] is idempotent (it only latches the flag to
/// `true`), so a repeated falling edge — bounce, or a press held across an
/// ISR re-fire before the radio task drains it — collapses to a single
/// pending request.
#[inline]
pub fn handle_button_press() {
    BTN_TOGGLE_REQ.store(true, Ordering::Release);
}

/// W1C-clear the push-button pin (GPIO3_IO23) bit in `ISR`. Must be called
/// from the button ISR after [`handle_button_press`] has consumed the edge.
#[inline]
pub fn clear_isr_gpio3_pin23() {
    // SAFETY: same argument as `clear_isr_gpio3_pin18`, for the upper half
    // of the GPIO3 block (bit 23).
    unsafe {
        (*teensy4_bsp::ral::gpio::GPIO3).ISR.write(1u32 << 23);
    }
}

/// Read back which of the GPIO3 *upper-half* interrupts (bits 16..=31) are
/// pending: bit 18 (A channel, pin 28) and bit 23 (button, pin 30). Used by
/// the `enc_a_irq` ISR to decide whether to handle the quadrature edge, the
/// button, or both.
#[inline]
pub fn isr_gpio3_upper_active() -> u32 {
    // SAFETY: ISR is a W1C RWRegister; the read returns the live pending
    // state of any bit (1 = pending, 0 = clear).
    unsafe { (*teensy4_bsp::ral::gpio::GPIO3).ISR.read() & 0xFFFF_0000 }
}

/// Drain the pending mode-toggle request. Returns `true` exactly once per
/// press (the ISR latches it; this swaps it back to `false`). The caller
/// (radio task) then advances the shared mode index and rebuilds its
/// virtual receiver.
#[inline]
pub fn take_mode_request() -> bool {
    BTN_TOGGLE_REQ.swap(false, Ordering::AcqRel)
}
