//! Radio → render shared state.
//!
//! The `radio` task (ENET + EP6 pump + FFT) publishes one 320-bin
//! `u16` magnitude row per spectrum frame; the `render` task drains the
//! newest row onto the ILI9341 waterfall and redraws status. Row publication
//! and snapshots copy a fixed-size buffer with interrupts disabled; readers
//! own their snapshots, so later publishes cannot invalidate them.

use core::cell::RefCell;
use core::sync::atomic::{AtomicU32, Ordering};
use cortex_m::interrupt::{self, Mutex};

/// One spectrum row. Must equal `crate::spectrum::BINS` == `crate::display::WF_COLS`.
pub const BINS: usize = crate::spectrum::BINS;

/// Session states for the status line (render task reads these).
pub const STATE_WAITING_IP: u32 = 0;
pub const STATE_LINK: u32 = 1;
pub const STATE_DISCOVERING: u32 = 2;
pub const STATE_STARTING: u32 = 3;
pub const STATE_TUNING: u32 = 4;
pub const STATE_STREAMING: u32 = 5;
pub const STATE_ERROR: u32 = 6;

static SNAPSHOT: Mutex<RefCell<(u32, [u16; BINS])>> = Mutex::new(RefCell::new((0, [0; BINS])));

/// Session state (see `STATE_*`).
static STATE: AtomicU32 = AtomicU32::new(STATE_WAITING_IP);

/// Peer HL2 IP packed as a `u32` (first octet in the high byte).
static PEER: AtomicU32 = AtomicU32::new(0);

/// EP6 frames consumed by the radio task. Monotonic u32 (wraps at 4.29 B).
/// The render task reads this to show "F N" on the status line — if the
/// number is not incrementing, the HL2 has stopped sending (watchdog tripped).
static FRAMES: AtomicU32 = AtomicU32::new(0);

/// Bump the RX frame counter.
pub fn bump_frames() {
    FRAMES.fetch_add(1, Ordering::Relaxed);
}

/// Read the RX frame counter.
pub fn frames() -> u32 {
    FRAMES.load(Ordering::Relaxed)
}

/// The virtual receiver's S-meter reading, an S1..S9 index (0 = below S1).
/// Set by the radio task from the spectrum passband-over-floor margin; read
/// by the render task to paint the S1-S9 bar. See `crate::smeter`.
static SLEVEL: AtomicU32 = AtomicU32::new(0);

/// The raw signal-over-noise margin (dB) behind the S-unit reading
/// (`level - floor`, scale-independent). Stored as `f32` bits so the render
/// task can show "+N dB" next to the bar.
static SMARGIN: AtomicU32 = AtomicU32::new(0.0f32.to_bits());

/// The demod + waterfall share of the CPU, in percent (0..=100). Computed by
/// the radio task over a ~1 s window (DWT cycles spent in the feed vs. wall);
/// shown on the status line so the pipeline "runs at CPU speed" is visible.
static CPU_PCT: AtomicU32 = AtomicU32::new(0);

/// Publish the S1..S9 index + the raw dB margin the render task displays.
pub fn set_slevel(sunits: u8, margin_db: f32) {
    SLEVEL.store(u32::from(sunits), Ordering::Release);
    SMARGIN.store(margin_db.to_bits(), Ordering::Release);
}

/// The current S1..S9 index (or 0).
pub fn slevel() -> u8 {
    SLEVEL.load(Ordering::Acquire) as u8
}

/// The raw dB-over-floor margin (0.0 until the first reading).
pub fn smargin_db() -> f32 {
    f32::from_bits(SMARGIN.load(Ordering::Acquire))
}

pub fn set_cpu_pct(pct: u32) {
    CPU_PCT.store(pct.min(100), Ordering::Release);
}

/// Diagnostic: increment the counter for each DMA completion interrupt.
/// The `dma_irq` ISR in `main.rs` bumps this on every DMA0_DMA16 vector
/// entry. `draw_pixels` samples the delta across a blit to see how many
/// chunks actually completed and raised their IRQ.
///
/// Uses `AtomicUsize` (not `AtomicU32`) to survive both `#[no_std]`
/// cortex-m and 32-bit pointer arithmetic without an extra import.
static IRQ_FIRES: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

pub fn irq_fires() -> usize {
    IRQ_FIRES.load(core::sync::atomic::Ordering::Acquire)
}

pub fn irq_fires_inc() {
    IRQ_FIRES.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

pub fn cpu_pct() -> u32 {
    CPU_PCT.load(Ordering::Acquire)
}

/// Publish one complete waterfall row and advance the wrapping sequence.
pub fn publish(row: &[u16]) {
    assert_eq!(row.len(), BINS, "publish requires a complete row");
    interrupt::free(|cs| {
        let mut snapshot = SNAPSHOT.borrow(cs).borrow_mut();
        snapshot.1.copy_from_slice(row);
        snapshot.0 = snapshot.0.wrapping_add(1);
    });
}

/// Copy the last published row, initially `(0, [0; BINS])`.
/// Compare sequence numbers with `!=`, not `>`, to handle wraparound.
pub fn latest() -> (u32, [u16; BINS]) {
    interrupt::free(|cs| *SNAPSHOT.borrow(cs).borrow())
}

pub fn set_state(s: u32) {
    STATE.store(s, Ordering::Release);
}

pub fn state() -> u32 {
    STATE.load(Ordering::Acquire)
}

pub fn set_peer(ip: core::net::Ipv4Addr) {
    let o = ip.octets();
    PEER.store(
        ((o[0] as u32) << 24) | ((o[1] as u32) << 16) | ((o[2] as u32) << 8) | (o[3] as u32),
        Ordering::Release,
    );
}

/// Peer IP, if one has been learned.
pub fn peer() -> Option<core::net::Ipv4Addr> {
    let p = PEER.load(Ordering::Acquire);
    (p != 0).then_some(core::net::Ipv4Addr::new(
        (p >> 24) as u8,
        (p >> 16) as u8,
        (p >> 8) as u8,
        p as u8,
    ))
}
