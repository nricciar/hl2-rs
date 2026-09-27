//! RX1 demod modes available on the Teensy.
//!
//! The encoder push button (p30) toggles the RX1 virtual receiver through
//! this ordered list of modes. The list is the single source of truth for
//! the *order* the user cycles through; the underlying `hl2::receiver::Mode`
//! value for each entry is used by `crate::radio::rx::Rx::set_mode` to
//! rebuild the receiver, and `label()` is used by the render task to paint
//! the current mode on the LCD status line.
//!
//! The list is deliberately short: only the modes the Teensy build supports
//! today (per `hl2::receiver::Mode` with no optional features — `dsp` is
//! enabled, `ft8`/`ft4`/`js8` are not). Adding a new mode here is enough to
//! expose it to the button; the radio task's rebuild path already handles
//! any `Mode` value the `hl2` crate provides.

use hl2::receiver::{Mode, Sideband};

/// One entry in the toggle cycle: a display label + the underlying `Mode`.
#[derive(Debug, Clone, Copy)]
pub struct ModeEntry {
    /// The `hl2::receiver::Mode` the virtual receiver should be rebuilt with.
    pub mode: Mode,
    /// Short label (≤ 4 chars) shown on the LCD status line.
    pub label: &'static str,
}

/// The ordered RX1 mode cycle the push button steps through (wrapping).
///
/// Order is user-visible: each press advances to the next entry; the list
/// wraps back to the first on a further press. The default (index 0, USB)
/// matches `ReceiverConfig::default()` so the first boot and the first
/// toggle state agree.
pub const MODES: &[ModeEntry] = &[
    ModeEntry {
        mode: Mode::Ssb(Sideband::Usb),
        label: "USB",
    },
    ModeEntry {
        mode: Mode::Ssb(Sideband::Lsb),
        label: "LSB",
    },
    ModeEntry {
        mode: Mode::Am,
        label: "AM",
    },
    ModeEntry {
        mode: Mode::Fm,
        label: "FM",
    },
    ModeEntry {
        mode: Mode::FmNarrow,
        label: "NFM",
    },
];

/// The `Mode` at `index` (panicked out-of-range — the caller owns bounds).
#[inline]
pub fn mode_at(index: usize) -> Mode {
    MODES[index].mode
}

/// The display label for the mode at `index`.
#[inline]
pub fn label_at(index: usize) -> &'static str {
    MODES[index].label
}

/// Number of entries in the cycle (for the radio task's wrap-around math).
/// `const fn` so callers can size a fixed per-mode pool
/// (`[T; crate::mode::len()]`) at compile time.
#[inline]
pub const fn len() -> usize {
    MODES.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hl2::receiver::sink::VecSink;
    use hl2::receiver::{ReceiverConfig, VirtualReceiver};

    #[test]
    fn each_mode_builds_a_virtual_receiver() {
        // Every entry in `MODES` must produce a working `VirtualReceiver`
        // with a default config (the same shape `Rx::set_mode` builds).
        for e in MODES.iter() {
            let mut cfg = ReceiverConfig::default();
            cfg.mode = e.mode;
            VirtualReceiver::new(cfg, alloc::boxed::Box::new(VecSink::new()))
                .expect("mode should build a virtual receiver");
        }
    }

    #[test]
    fn labels_are_short_enough_for_the_lcd() {
        // The status line draws each label at 6px/char (see
        // `display::driver::draw_text`); keep every label ≤ 4 chars so it
        // fits next to the NCO + mode row without colliding with the S-meter.
        for e in MODES.iter() {
            assert!(e.label.chars().count() <= 4, "{} exceeds 4 chars", e.label);
        }
    }

    #[test]
    fn usb_is_the_default_entry() {
        // `Rx::new` uses `ReceiverConfig::default()` (USB). Index 0 in the
        // cycle must agree — otherwise the first button press would toggle
        // *away* from the actual initial state, and the LCD label (seeded
        // to index 0) would lie about the first press.
        assert_eq!(MODES[0].mode, hl2::receiver::ReceiverConfig::default().mode);
    }
}
