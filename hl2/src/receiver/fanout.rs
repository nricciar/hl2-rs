//! Per-slot EP6 baseband fan-out.
//!
//! The EP6 baseband stream is a *single* interleaved I/Q payload in which each
//! record carries **N** 24-bit I/Q pairs — one per active receiver, in
//! ascending slot order — followed by a mic sample. The de-interleaver
//! (see [`crate::protocol::data::parse_baseband_chunk`]) splits one frame into
//! `N` per-receiver streams (`BasebandChunk::per_rx`), where `per_rx[p]` is the
//! stream of the *p-th active slot in ascending order*.
//!
//! A single shared [`BasebandRing`] can only hold
//! one of those streams. To demodulate several receivers at once we need one
//! ring **per active slot**, keyed by the slot the user tuned to:
//!
//! ```text
//! pump (sole writer)               VirtualReceiver / demod thread (readers)
//!   │                                 ┌────► ring(slot 1)  ──► demod RX1 → audio
//! parse frame ─► per_rx[0] ──────────►┼────► ring(slot 2)  ──► demod RX2 → audio
//!               per_rx[p] ──────────►└────► ring(slot p)  ──► demod RXp → audio
//! ```
//!
//! This module is the seam between the two: it tracks the set of active slots,
//! owns one [`BasebandRing`] per slot, and exposes a **position-indexed**
//! view so the pump can push `per_rx[p]` into the ring of the p-th active
//! slot, while the demod side fetches the ring by **slot number** (what the
//! user thinks in).
//!
//! ## Position ↔ slot
//!
//! Interleaving order is the p-th non-NULL receiver in ascending slot index
//! (same order the reference uses). We therefore map
//!
//!   * position `p`  ═  p-th smallest active slot
//!   * slot `s`      ═  its rank among the active slots (0-based) = position
//!
//! [`BasebandFanout`]'s internals are a `BTreeMap<u8, Arc<Mutex<BasebandRing>>>`
//! so `values()` iterates in ascending slot order — the canonical position
//! order. Registering / unregistering a slot just adds or removes an entry;
//! the data already in a surviving slot's ring is preserved (we keep the same
//! `Arc`), and the position order is rebuilt implicitly by key order.
//!
//! ## Threading
//!
//! A single `std::sync::Mutex` guards the `BTreeMap`. The fan-out methods are
//! memcpy/insert sized and never `await` under the lock. The pump takes a
//! one-shot snapshot (`snapshot`) per frame to bind "which slots are active"
//! and "their rings" to a single consistent view, so a `tune()` arriving in
//! the middle of a frame cannot change the active set mid-decode.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use super::baseband_ring::BasebandRing;

/// A per-slot fan-out of EP6 baseband. One [`BasebandRing`] per active slot,
/// addressed by slot number for readers and by interleaved position for the
/// pump (sole writer).
#[derive(Debug)]
pub struct BasebandFanout {
    /// The authoritative active-slot → ring map. `BTreeMap` iterates in
    /// ascending slot order, which is exactly the EP6 interleaved position
    /// order, so `values()` is the pump's `per_rx` push target.
    inner: Mutex<BTreeMap<u8, Arc<Mutex<BasebandRing>>>>,
}

impl BasebandFanout {
    /// A fresh fan-out with no active slots yet.
    ///
    /// The pump starts as soon as the HL2 is started; the first `BasebandChunk`
    /// we see is for the *initial* (RX1) tune, so we seed the fan-out with
    /// slot 1 so the "no user-tuned slots yet" window is already covered and
    /// `baseband_ring(1)` resolves to a real ring from the very first frame.
    /// This mirrors the radio's default (RX1 is active from power-on).
    pub fn new() -> Self {
        let mut m: BTreeMap<u8, Arc<Mutex<BasebandRing>>> = BTreeMap::new();
        m.insert(1, Arc::new(Mutex::new(BasebandRing::default())));
        Self {
            inner: Mutex::new(m),
        }
    }

    /// Register `slot` as active (idempotent — repeated tunes of the same slot
    /// do not add a second ring). If the slot already has a ring, the same
    /// `Arc` is retained so any in-flight samples are not lost.
    pub fn register_slot(&self, slot: u8) {
        let mut g = self.inner.lock().unwrap();
        g.entry(slot)
            .or_insert_with(|| Arc::new(Mutex::new(BasebandRing::default())));
    }

    /// Mark `slot` inactive (e.g. via `tune(slot, 0)` or a future `close_slot`).
    /// Drops the ring and any in-flight samples for that slot. Idempotent.
    pub fn unregister_slot(&self, slot: u8) {
        self.inner.lock().unwrap().remove(&slot);
    }

    /// Number of currently active slots. This is the `N` that the pump passes
    /// to the de-interleaver and that goes into the C4 receiver-count field of
    /// the baseline chunk. Always ≥ 1 (slot 1 is seeded at construction).
    pub fn rx_count(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// The active slots in ascending order (position order).
    pub fn slots(&self) -> Vec<u8> {
        self.inner.lock().unwrap().keys().copied().collect()
    }

    /// The ring for `slot`, if active. `None` if the slot is not currently
    /// tuned (callers should either register it first or fall back to the
    /// position-0 ring).
    pub fn ring_for_slot(&self, slot: u8) -> Option<Arc<Mutex<BasebandRing>>> {
        self.inner.lock().unwrap().get(&slot).cloned()
    }

    /// All active rings in position order (ascending slot). The pump iterates
    /// this with `per_rx` (which has the same length) one-to-one.
    pub fn rings_in_order(&self) -> Vec<Arc<Mutex<BasebandRing>>> {
        self.inner.lock().unwrap().values().cloned().collect()
    }

    /// A consistent snapshot for one pump frame: (active-count, position-ordered
    /// rings). The pump uses the active-count as the de-interleave `N` and the
    /// rings as the `per_rx` push targets, so both are bound to the same active
    /// set and a `tune()` mid-frame cannot desynchronise them.
    ///
    /// Always returns `n >= 1` and `rings.len() == n` because slot 1 is seeded
    /// at [`Self::new`]; a call to `unregister_slot` that empties the set would
    /// (and is expected to) re-register slot 1 before the next frame.
    pub fn snapshot(&self) -> (usize, Vec<Arc<Mutex<BasebandRing>>>) {
        let g = self.inner.lock().unwrap();
        let rings: Vec<_> = g.values().cloned().collect();
        (rings.len(), rings)
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

    #[test]
    fn seed_slot1_is_active() {
        let f = BasebandFanout::new();
        assert_eq!(f.rx_count(), 1);
        assert_eq!(f.slots(), vec![1u8]);
        assert!(f.ring_for_slot(1).is_some());
        assert!(f.ring_for_slot(2).is_none());
    }

    #[test]
    fn register_is_idempotent_and_preserves_data() {
        let f = BasebandFanout::new();
        let r1 = f.ring_for_slot(1).unwrap();
        // Stuff a sentinel sample into the pre-existing slot-1 ring.
        r1.lock().unwrap().push(&[c(7.0, 0.0)]);
        // Register slot 1 again — the same Arc must be retained.
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
        // Inserting a slot "in the middle" must not corrupt existing data.
        let r3 = f.ring_for_slot(3).unwrap();
        r3.lock().unwrap().push(&[c(3.0, 0.0)]);
        f.register_slot(1); // (already seeded)
        assert_eq!(f.slots(), vec![1u8, 2, 3]);
        // Position order: rings[2] must be slot 3's ring.
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
    fn unregister_removes_ring() {
        let f = BasebandFanout::new();
        f.register_slot(2);
        f.register_slot(3);
        assert_eq!(f.slots(), vec![1u8, 2, 3]);
        f.unregister_slot(2);
        assert_eq!(f.slots(), vec![1u8, 3]);
        assert!(f.ring_for_slot(2).is_none());
        // Slots 1 and 3 are unaffected.
        assert!(f.ring_for_slot(1).is_some());
        assert!(f.ring_for_slot(3).is_some());
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
