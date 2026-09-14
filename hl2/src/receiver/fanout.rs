//! Per-slot EP6 baseband fan-out.
//!
//! The EP6 baseband stream is a *single* interleaved I/Q payload in which each
//! record carries **N** 24-bit I/Q pairs — one per active receiver, in
//! ascending slot order — followed by a mic sample. The de-interleaver (see
//! [`crate::protocol::data::parse_baseband_chunk`]) splits one frame into
//! `N` per-receiver streams (`BasebandChunk::per_rx`), where `per_rx[p]` is the
//! stream of the *p-th active slot in ascending order*.
//!
//! A single shared [`BasebandRing`] can only hold one of those streams. To
//! demodulate several receivers at once we need one ring **per active slot**,
//! keyed by the slot the user tuned to:
//!
//! ```text
//! pump (sole writer)               VirtualReceiver / demod thread (readers)
//!   │                                 ┌────► ring(slot 1)  ──► demod RX1 → audio
//! parse frame ─► per_rx[0] ──────────►┼────► ring(slot 2)  ──► demod RX2 → audio
//!               per_rx[p] ──────────►└────► ring(slot p)  ──► demod RXp → audio
//! ```
//!
//! This module is the seam between the two: it tracks the set of active slots,
//! owns one [`BasebandRing`] per slot, and exposes a **position-indexed** view
//! so the pump can push `per_rx[p]` into the ring of the p-th active slot, while
//! the demod side fetches the ring by **slot number** (what the user thinks in).
//!
//! ## Position ↔ slot
//!
//! Interleaving order is the p-th non-NULL receiver in ascending slot index
//! (same order the reference uses). We therefore map
//!
//!   * position `p`  ═  p-th smallest active slot
//!   * slot `s`      ═  its rank among the active slots (0-based) = position
//!
//! The slot→ring map is an `alloc` `BTreeMap<u8, Cell>` keyed by slot number, so
//! iteration is always in ascending slot order — the canonical position order.
//! Registering / unregistering a slot just adds or removes an entry; the data
//! already in a surviving slot's ring is preserved, and the position order is
//! implicit in key order.
//!
//! ## `std` vs `no_std`
//!
//! The per-slot cell differs by feature:
//!
//! * **`std`** — `Cell = Arc<Mutex<BasebandRing>>`. The fan-out is `Sync`, so
//!   an `Arc<BasebandFanout>` can be shared across the pump task and any number
//!   of demod threads, each cloning the `Arc` for its slot and locking only for
//!   a memcpy-sized `push`/`peek`. `register_slot` / `unregister_slot` /
//!   `rx_count` / `slots` take `&self` (interior mutability via the `Mutex`).
//! * **`no_std`** — `Cell = BasebandRing` (owned). A single-threaded consumer
//!   (e.g. the Teensy RTIC radio task) holds `&mut self` and reaches the p-th /
//!   slot's ring by reference; interior mutability is a `core::cell::RefCell` on
//!   the map so the register / count / slot methods keep their `&self` shape.
//!
//! The `std` public API (`ring_for_slot`, `rings_in_order`, `snapshot` returning
//! owned `Arc<Mutex<BasebandRing>>` handles) is byte-identical to the historical
//! one, so the `hl2::hl2` pump, `hl2-api`, and the `hl2` bin compile unchanged.
//! The `no_std` surface adds two `&mut self` operations for a single-owner
//! consumer: `push_frame` (pump side — fan a position-ordered `per_rx` into the
//! matching slot rings) and `peek_slot` (reader side — non-destructive peek one
//! active slot).

#[cfg(feature = "std")]
use std::sync::{Arc, Mutex};

use alloc::collections::btree_map::BTreeMap;
#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;

use super::baseband_ring::BasebandRing;

/// The per-slot ring storage. `Arc`-shared (multi-reader, `std`) or owned
/// (single-owner, `no_std`).
#[cfg(feature = "std")]
type Cell = Arc<Mutex<BasebandRing>>;
#[cfg(not(feature = "std"))]
type Cell = BasebandRing;

/// Interior mutability for the slot→ring map: `Mutex` under `std` (so the map
/// is `Sync` across threads) or `RefCell` under `no_std` (single owner).
#[cfg(feature = "std")]
type MapCell<T> = Mutex<T>;
#[cfg(not(feature = "std"))]
type MapCell<T> = core::cell::RefCell<T>;

/// A per-slot fan-out of EP6 baseband. One [`BasebandRing`] per active slot,
/// addressed by slot number for readers and by interleaved position for the
/// pump (sole writer).
#[derive(Debug)]
pub struct BasebandFanout {
    /// The authoritative active-slot → ring map, in ascending slot order.
    inner: MapCell<BTreeMap<u8, Cell>>,
}

impl BasebandFanout {
    /// A fresh fan-out with no active slots yet.
    ///
    /// The pump starts as soon as the HL2 is started; the first `BasebandChunk`
    /// we see is for the *initial* (RX1) tune, so we seed the fan-out with slot 1
    /// so the "no user-tuned slots yet" window is already covered and
    /// baseband_ring(1) resolves to a real ring from the very first frame. This
    /// mirrors the radio's default (RX1 is active from power-on).
    pub fn new() -> Self {
        let mut m: BTreeMap<u8, Cell> = BTreeMap::new();
        m.insert(1, Self::new_ring());
        Self {
            inner: MapCell::new(m),
        }
    }

    /// A fresh per-slot ring cell.
    fn new_ring() -> Cell {
        #[cfg(feature = "std")]
        {
            Arc::new(Mutex::new(BasebandRing::default()))
        }
        #[cfg(not(feature = "std"))]
        {
            BasebandRing::default()
        }
    }

    /// A map guard: `MutexGuard` under `std`, `RefMut` under `no_std`.
    #[cfg(feature = "std")]
    fn guard(&self) -> std::sync::MutexGuard<'_, BTreeMap<u8, Cell>> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    #[cfg(not(feature = "std"))]
    fn guard(&self) -> core::cell::RefMut<'_, BTreeMap<u8, Cell>> {
        self.inner.borrow_mut()
    }

    /// Register `slot` as active (idempotent — repeated tunes of the same slot
    /// do not add a second ring). If the slot already has a ring, the same ring
    /// is retained so any in-flight samples are not lost.
    pub fn register_slot(&self, slot: u8) {
        self.guard().entry(slot).or_insert_with(Self::new_ring);
    }

    /// Mark `slot` inactive (e.g. via `tune(slot, 0)` or a future `close_slot`).
    /// Drops the ring and any in-flight samples for that slot. Idempotent.
    pub fn unregister_slot(&self, slot: u8) {
        self.guard().remove(&slot);
    }

    /// Number of currently active slots. This is the `N` that the pump passes
    /// to the de-interleaver and that goes into the C4 receiver-count field of
    /// the baseline chunk. Always ≥ 1 (slot 1 is seeded at construction).
    pub fn rx_count(&self) -> usize {
        self.guard().len()
    }

    /// The active slots in ascending order (position order).
    pub fn slots(&self) -> Vec<u8> {
        self.guard().keys().copied().collect()
    }
}

// ── `std`: owned multi-reader `Arc<Mutex<BasebandRing>>` handles ─────────────
// Identical public API to the historical std-only fan-out, so the `hl2::hl2`
// pump, `hl2-api`, and the `hl2` bin compile unchanged.
#[cfg(feature = "std")]
impl BasebandFanout {
    /// The ring for `slot`, if active. `None` if the slot is not currently
    /// tuned (callers should either register it first or fall back to the
    /// position-0 ring).
    pub fn ring_for_slot(&self, slot: u8) -> Option<Arc<Mutex<BasebandRing>>> {
        self.guard().get(&slot).cloned()
    }

    /// All active rings in position order (ascending slot). The pump iterates
    /// this with `per_rx` (which has the same length) one-to-one.
    pub fn rings_in_order(&self) -> Vec<Arc<Mutex<BasebandRing>>> {
        self.guard().values().cloned().collect()
    }

    /// A consistent snapshot for one pump frame: (active-count, position-ordered
    /// rings). The pump uses the active-count as the de-interleave N and the rings
    /// as the `per_rx` push targets, so both are bound to the same active set and
    /// a `tune()` mid-frame cannot desynchronise them.
    ///
    /// Always returns `n >= 1` and `rings.len() == n` because slot 1 is seeded
    /// at [`Self::new`].
    pub fn snapshot(&self) -> (usize, Vec<Arc<Mutex<BasebandRing>>>) {
        let g = self.guard();
        let rings: Vec<_> = g.values().cloned().collect();
        (rings.len(), rings)
    }
}

// ── `no_std`: single-owner pump/read operations (RefCell interior mutability) ─
#[cfg(not(feature = "std"))]
impl BasebandFanout {
    /// Fan one EP6 chunk's de-interleaved `per_rx` streams (p-th active slot,
    /// ascending) into the matching slot rings, in place. Sole-writer: a
    /// single-threaded pump (e.g. the Teensy RTIC radio task) calls this once
    /// per parsed chunk.
    ///
    /// `per_rx[p]` is the p-th active slot's stream (the same order
    /// [`super::super::protocol::data::parse_baseband_chunk`] used: ascending
    /// slot). `per_rx.len()` should equal the active-slot count (`n_recv`);
    /// trailing active slots with no `per_rx` entry are left untouched and extra
    /// `per_rx` entries beyond the active set are ignored — mirroring the `std`
    /// pump's position-ordered `snapshot()` + `rings[p].push(&per_rx[p])` fan-out.
    ///
    /// Accepts `&[&[Complex<f32>>]` (the borrow of `BasebandChunk::per_rx`) so
    /// the pump needs no per-slot `Arc`/lock. Returns the number of slot rings
    /// actually filled.
    pub fn push_frame(&mut self, per_rx: &[&[num_complex::Complex<f32>]]) -> usize {
        let mut filled = 0usize;
        for (p, samples) in per_rx.iter().enumerate() {
            // Position `p` ⇔ p-th active slot in ascending order (the
            // `BTreeMap` iterates in ascending slot order).
            if let Some(ring) = self.guard().values_mut().nth(p) {
                ring.push(samples);
                filled += 1;
            }
        }
        filled
    }

    /// A non-destructive peek into the ring of `slot`. Copies up to `max`
    /// samples from `cursor` (the next sequence this reader wants) into `out`
    /// (its allocation is reused; see [`BasebandRing::peek`]) without removing
    /// them. Returns the count copied; `0` if `slot` is not active, or the
    /// cursor has caught up, or it fell behind (resync via
    /// [`Self::ring_bounds`]). The reader advances its cursor by the returned
    /// count.
    pub fn peek_slot(
        &mut self,
        slot: u8,
        cursor: u64,
        out: &mut Vec<num_complex::Complex<f32>>,
        max: usize,
    ) -> usize {
        self.guard()
            .get_mut(&slot)
            .map(|ring| ring.peek(cursor, out, max))
            .unwrap_or(0)
    }

    /// The `[oldest, newest)` sequence window currently buffered in `slot`'s
    /// ring (`base_seq`, `head_seq`). `None` if the slot is not active. A reader
    /// whose cursor falls below `base_seq` should resync to it before peeking
    /// (the ring keeps `count` buffered from `base_seq`; see
    /// [`BasebandRing::peek`]).
    pub fn ring_bounds(&mut self, slot: u8) -> Option<(u64, u64)> {
        let g = self.guard();
        let ring = g.get(&slot)?;
        Some((ring.base_seq(), ring.head_seq()))
    }
}

impl Default for BasebandFanout {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn c(re: f32, im: f32) -> Complex<f32> {
        Complex::new(re, im)
    }

    // ── Common (`register_slot` / `rx_count` / `slots` — `&self` on both) ──────
    #[test]
    fn seed_slot1_is_active() {
        let f = BasebandFanout::new();
        assert_eq!(f.rx_count(), 1);
        assert_eq!(f.slots(), vec![1u8]);
    }

    #[test]
    fn unregister_removes_entry() {
        let f = BasebandFanout::new();
        f.register_slot(2);
        f.register_slot(3);
        assert_eq!(f.slots(), vec![1u8, 2, 3]);
        f.unregister_slot(2);
        assert_eq!(f.slots(), vec![1u8, 3]);
        assert_eq!(f.rx_count(), 2);
    }

    // ── `no_std`: single-owner `push_frame` / `peek_slot` / `ring_bounds` ──────
    #[cfg(not(feature = "std"))]
    mod nostd {
        use super::*;

        fn push_one(f: &mut BasebandFanout, vals: &[Complex<f32>]) {
            let borrows: Vec<&[Complex<f32>]> = vec![vals];
            f.push_frame(&borrows);
        }

        #[test]
        fn push_frame_fans_into_position_order() {
            let mut f = BasebandFanout::new(); // seeded slot 1
            f.register_slot(3);
            f.register_slot(2);
            // Active set, ascending = [1,2,3] → per_rx[0]/[1]/[2] map to slots 1/2/3.
            let per_rx: Vec<Vec<Complex<f32>>> =
                vec![vec![c(1.0, 0.0)], vec![c(2.0, 0.0)], vec![c(3.0, 0.0)]];
            let borrows: Vec<&[Complex<f32>]> = per_rx.iter().map(Vec::as_slice).collect();
            assert_eq!(f.push_frame(&borrows), 3, "all 3 active slots filled");
            let mut out = Vec::new();
            assert_eq!(f.peek_slot(1, 0, &mut out, 10), 1);
            assert_eq!(out, vec![c(1.0, 0.0)]);
            assert_eq!(f.peek_slot(2, 0, &mut out, 10), 1);
            assert_eq!(out, vec![c(2.0, 0.0)]);
            assert_eq!(f.peek_slot(3, 0, &mut out, 10), 1);
            assert_eq!(out, vec![c(3.0, 0.0)]);
        }

        #[test]
        fn peek_slot_resync_via_ring_bounds() {
            let mut f = BasebandFanout::new();
            push_one(&mut f, &[c(0.0, 0.0); 10]);
            let (base, head) = f.ring_bounds(1).expect("slot 1 active");
            assert_eq!(base, 0);
            assert_eq!(head, 10);
            let mut out = Vec::new();
            // Cursor at head ⇒ caught up ⇒ 0.
            assert_eq!(f.peek_slot(1, head, &mut out, 100), 0);
            // Resync to base ⇒ all 10 again (non-destructive).
            assert_eq!(f.peek_slot(1, base, &mut out, 100), 10);
        }

        #[test]
        fn peek_inactive_slot_zero_and_no_bounds() {
            let mut f = BasebandFanout::new();
            let mut out = Vec::new();
            assert_eq!(f.peek_slot(9, 0, &mut out, 10), 0);
            assert!(out.is_empty());
            assert!(f.ring_bounds(9).is_none());
        }

        #[test]
        fn extra_per_rx_beyond_active_set_ignored() {
            let mut f = BasebandFanout::new(); // only slot 1
            let per_rx: Vec<Vec<Complex<f32>>> =
                vec![vec![c(1.0, 0.0)], vec![c(2.0, 0.0)], vec![c(3.0, 0.0)]];
            let borrows: Vec<&[Complex<f32>]> = per_rx.iter().map(Vec::as_slice).collect();
            // Only 1 active slot → per_rx[1..] dropped, 1 filled.
            assert_eq!(f.push_frame(&borrows), 1);
            let mut out = Vec::new();
            assert_eq!(f.peek_slot(1, 0, &mut out, 10), 1);
            assert_eq!(out, vec![c(1.0, 0.0)]);
        }
    }

    // ── `std`: multi-reader `Arc` / `ring_for_slot` / `rings_in_order` / `snapshot` ─
    #[cfg(feature = "std")]
    mod stdd {
        use super::*;

        #[test]
        fn seed_slot1_has_ring() {
            let f = BasebandFanout::new();
            assert!(f.ring_for_slot(1).is_some());
            assert!(f.ring_for_slot(2).is_none());
        }

        #[test]
        fn register_is_idempotent_and_preserves_data() {
            let f = BasebandFanout::new();
            let r1 = f.ring_for_slot(1).unwrap();
            r1.lock().unwrap().push(&[c(7.0, 0.0)]);
            f.register_slot(1);
            let r1b = f.ring_for_slot(1).unwrap();
            assert!(
                Arc::ptr_eq(&r1, &r1b),
                "re-register must keep the same ring"
            );
            let mut out = Vec::new();
            assert_eq!(r1b.lock().unwrap().peek(0, &mut out, 1), 1);
            assert_eq!(out, vec![c(7.0, 0.0)]);
        }

        #[test]
        fn register_three_slots_keeps_sorted_position_order() {
            let f = BasebandFanout::new();
            f.register_slot(3);
            f.register_slot(2);
            let r3 = f.ring_for_slot(3).unwrap();
            r3.lock().unwrap().push(&[c(3.0, 0.0)]);
            assert_eq!(f.slots(), vec![1u8, 2, 3]);
            let rings = f.rings_in_order();
            assert_eq!(rings.len(), 3);
            let mut out = Vec::new();
            assert_eq!(rings[2].lock().unwrap().peek(0, &mut out, 1), 1);
            assert_eq!(out, vec![c(3.0, 0.0)], "ring[2] must be slot 3's data");
        }

        #[test]
        fn register_distinct_slots_get_distinct_rings() {
            let f = BasebandFanout::new();
            f.register_slot(2);
            f.register_slot(3);
            let r1 = f.ring_for_slot(1).unwrap();
            let r2 = f.ring_for_slot(2).unwrap();
            let r3 = f.ring_for_slot(3).unwrap();
            assert!(!Arc::ptr_eq(&r1, &r2));
            assert!(!Arc::ptr_eq(&r2, &r3));
            assert!(!Arc::ptr_eq(&r1, &r3));
        }

        #[test]
        fn snapshot_matches_rx_count() {
            let f = BasebandFanout::new();
            f.register_slot(2);
            let (n, rings) = f.snapshot();
            assert_eq!(n, 2);
            assert_eq!(rings.len(), 2);
        }
    }
}
