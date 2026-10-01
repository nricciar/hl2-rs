//! Ordered RX1 mode cycle for the encoder button, with LCD labels.

use hl2::receiver::{Mode, Sideband};

/// A receiver mode and its display label.
#[derive(Debug, Clone, Copy)]
pub struct ModeEntry {
    pub mode: Mode,
    /// LCD status label, at most four characters.
    pub label: &'static str,
}

/// Button-cycle order; index 0 (USB) matches `ReceiverConfig::default()`.
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

/// Mode at `index`; panics if out of range.
#[inline]
pub fn mode_at(index: usize) -> Mode {
    MODES[index].mode
}

/// Display label at `index`; panics if out of range.
#[inline]
pub fn label_at(index: usize) -> &'static str {
    MODES[index].label
}

/// Cycle length, also used to size fixed per-mode pools.
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
        for e in MODES.iter() {
            let mut cfg = ReceiverConfig::default();
            cfg.mode = e.mode;
            VirtualReceiver::new(cfg, alloc::boxed::Box::new(VecSink::new()))
                .expect("mode should build a virtual receiver");
        }
    }

    #[test]
    fn labels_are_short_enough_for_the_lcd() {
        for e in MODES.iter() {
            assert!(e.label.chars().count() <= 4, "{} exceeds 4 chars", e.label);
        }
    }

    #[test]
    fn usb_is_the_default_entry() {
        assert_eq!(MODES[0].mode, hl2::receiver::ReceiverConfig::default().mode);
    }
}
