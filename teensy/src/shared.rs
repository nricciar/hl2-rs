//! Radio → render shared state.
//!
//! The `radio` task (ENET + EP6 pump + FFT) publishes one 320-bin
//! `u16` magnitude row per spectrum frame; the `render` task drains the
//! newest row onto the ILI9341 waterfall and redraws status. Single-core,
//! one writer / one reader, so a pair of static bands + a monotonic
//! sequence number is sufficient — no locks, no atomics-for-mutability
//! beyond the seq.

use core::sync::atomic::{AtomicU32, Ordering};

/// One spectrum row. Must equal `crate::spectrum::BINS` == `crate::display::WF_COLS`.
pub const BINS: usize = 320;

/// Session states for the status line (render task reads these).
pub const STATE_WAITING_IP: u32 = 0;
pub const STATE_LINK: u32 = 1;
pub const STATE_DISCOVERING: u32 = 2;
pub const STATE_STARTING: u32 = 3;
pub const STATE_TUNING: u32 = 4;
pub const STATE_STREAMING: u32 = 5;
pub const STATE_ERROR: u32 = 6;

struct Band {
    inner: core::cell::UnsafeCell<[u16; BINS]>,
}

// Single-core, one writer (radio) + one reader (render).  It is sound to
// share `&Band` references across tasks: the writer holds `&mut` via a
// `&'static` pointer it obtains from the static, and the reader uses `&`.
// They never execute concurrently (RTIC cooperative scheduling), so the
// interior mutability does not expose a data race.
unsafe impl Sync for Band {}

impl Band {
    const fn new() -> Self {
        Self {
            inner: core::cell::UnsafeCell::new([0u16; BINS]),
        }
    }

    /// `&[u16; BINS]` view for the reader.
    fn as_slice(&'static self) -> &'static [u16] {
        unsafe { &*self.inner.get() }
    }

    /// `[u16; BINS]` writer view for the radio task.
    fn as_mut(&'static self) -> &'static mut [u16] {
        unsafe { &mut *self.inner.get() }
    }
}

// Double buffer: the radio writes to the band it does NOT expose; the
// render always reads the band the most-recent `SEQ` published. The two
// tasks never run concurrently, so a mid-write read cannot happen, and the
// extra band is defensive (reader keeps a valid frame if it ever yields
// inside its own copy step).
const fn band() -> Band {
    Band::new()
}
static BAND0: Band = band();
static BAND1: Band = band();

/// Monotonic publish counter. Bit 0 is the band index the *reader* is to
/// read (the band just published); higher bits are a free-running count.
static SEQ: AtomicU32 = AtomicU32::new(0);

/// Session state (see `STATE_*`).
static STATE: AtomicU32 = AtomicU32::new(STATE_WAITING_IP);

/// Peer HL2 IP packed as a `u32` (octet 3 in the high byte) for the status line.
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

/// Publish the next waterfall row. Called once per spectrum frame.
///
/// Band selection: `cur & 1` alternates BAND0/BAND1 on each call
/// (cur 0→B0, cur 1→B1, cur 2→B0, …). `latest()` reads band
/// `(seq − 1) & 1` — the band just written — so every publish is
/// immediately visible to the renderer.
pub fn publish(row: &[u16]) {
    let n = BINS.min(row.len());
    let cur = SEQ.load(Ordering::Relaxed);
    // `cur & 1` alternates band on every call (0 → B0, 1 → B1, 2 → B0, …).
    // The reader (see `latest`) reads band `(SEQ − 1) & 1`, the band just
    // written by the most-recent publish. This is correct because a
    // cooperative-scheduling render task cannot read mid-publish, and the
    // two publishes are never interleaved.
    let band = cur & 1;
    let next = if band == 0 { &BAND0 } else { &BAND1 };
    next.as_mut()[..n].copy_from_slice(&row[..n]);
    // Advance: bump the counter; the reader will now read this band.
    SEQ.store(cur + 1, Ordering::Release);
}

/// The last published row. Returns `(seq, &rows[..len])` where `seq == 0`
/// means "no frames yet" (caller should skip the blit).
pub fn latest() -> (u32, &'static [u16]) {
    let seq = SEQ.load(Ordering::Acquire);
    let band = ((seq - 1) & 1) as u32;
    let body = if band == 0 { BAND0.as_slice() } else { BAND1.as_slice() };
    (seq, body)
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
