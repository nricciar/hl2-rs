//! Per-mode [`DecodedMessage`] `spot_fields` extraction for the WSJT-family
//! (FT8 / FT4) free-text messages.
//!
//! Both FT8 and FT4 decode into a resolved 77-bit WSJT payload with the same
//! free-text grammar, so they share a single selector: [`wsjt_spot_fields`].
//!
//! Mirrors wsjtx's `MainWindow::pskPost` (widgets/mainwindow.cpp:7565) +
//! `DecodedText::deCallAndGrid` (Decoder/decodedtext.cpp:212) + the
//! `grid_regexp` (widgets/mainwindow.cpp:308):
//!
//! * word1 of the 77-bit text is `CQ` / `QRZ` (optionally with a direction /
//!   `DX` / 3-digit suffix) or a bare callsign;
//! * **caller** = the next word (wsjtx "word2"), **locator** = the one after
//!   it ("word3"; "R" → "word4");
//! * post only when the locator is a valid grid square
//!   (`[A-R]{2}[0-9]{2}([A-X]{2})?`, not RR73) **or** the text contains
//!   `" CQ "`;
//! * suppress self-spots: text containing both our base callsign and our
//!   4-char grid (wsjtx mainwindow.cpp:7568).

use hl2_common::spot::{base_callsign, grid_is_square};
use hl2_common::{SpotFields, SpotStation};

fn is_cq_tag(w: &str) -> bool {
    let b = w.as_bytes();
    (1..=4).contains(&b.len()) && b.iter().all(|c| c.is_ascii_uppercase())
        || b.len() == 3 && b.iter().all(|c| c.is_ascii_digit())
}

/// Extract [`SpotFields`] from a WSJT-77 free-text message (`text`), given
/// our own spot station for self-spot suppression. Returns `None` when the
/// message is not worth spotting (no word1/word2, no valid locator and no
/// `"CQ"` phrase, or a self-spot).
pub fn wsjt_spot_fields(text: &str, st: &SpotStation) -> Option<SpotFields> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 2 {
        return None;
    }
    let mut i = 1;
    if matches!(words[0], "CQ" | "QRZ") && i + 1 < words.len() && is_cq_tag(words[i]) {
        i += 1;
    }
    let call = words.get(i).copied()?;
    let mut grid = words.get(i + 1).copied().unwrap_or_default();
    if grid == "R" {
        grid = words.get(i + 2).copied().unwrap_or_default();
    }
    // Self-spotting prevention
    let base = base_callsign(&st.callsign);
    let g4: String = st.grid.chars().take(4).collect();
    if !base.is_empty() && !g4.is_empty() && text.contains(base) && text.contains(g4.as_str()) {
        return None;
    }
    // Spot only if the locator is a grid square or the message is a CQ
    if !(grid_is_square(grid) || text.contains(" CQ ") || text == "CQ" || text.starts_with("CQ ")) {
        return None;
    }
    Some(SpotFields {
        caller: call.to_ascii_uppercase(),
        locator: grid.to_ascii_uppercase(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st() -> SpotStation {
        SpotStation::new("K9ABC", "FN31")
    }

    #[test]
    fn cq_message_spots_call_and_grid() {
        let s = wsjt_spot_fields("CQ R7IW LN35", &st()).expect("CQ + grid spots");
        assert_eq!(s.caller, "R7IW");
        assert_eq!(s.locator, "LN35");
    }

    #[test]
    fn cq_dx_tag_is_eaten_by_word1() {
        let s = wsjt_spot_fields("CQ DX R6WA LN32", &st()).expect("CQ DX spots");
        assert_eq!(s.caller, "R6WA");
        assert_eq!(s.locator, "LN32");
    }

    #[test]
    fn cq_number_tag_is_eaten_by_word1() {
        let s = wsjt_spot_fields("CQ 001 3Y0Z JD34", &st()).expect("spots");
        assert_eq!(s.caller, "3Y0Z");
        assert_eq!(s.locator, "JD34");
    }

    #[test]
    fn bare_call_grid_spots() {
        let s = wsjt_spot_fields("K1ABC K9ABD PN32", &st()).expect("spots");
        assert_eq!(s.caller, "K9ABD");
        assert_eq!(s.locator, "PN32");
    }

    #[test]
    fn reply_without_grid_does_not_spot() {
        assert!(wsjt_spot_fields("JA1ABC 3Y0Z -12", &st()).is_none());
    }

    #[test]
    fn self_spot_is_suppressed() {
        assert!(wsjt_spot_fields("CQ K9ABC FN31", &st()).is_none());
        let st2 = SpotStation::new("K9ABC", "PM95");
        assert!(wsjt_spot_fields("CQ K9ABC FN31", &st2).is_some());
        let st3 = SpotStation::new("W1AW/K9ABC", "FN31");
        assert!(wsjt_spot_fields("CQ K9ABC FN31", &st3).is_none());
    }

    #[test]
    fn qrz_message_spots() {
        let s = wsjt_spot_fields("QRZ K1ABC PM95", &st()).expect("QRZ spots");
        assert_eq!(s.caller, "K1ABC");
        assert_eq!(s.locator, "PM95");
    }

    #[test]
    fn free_text_with_cq_spots() {
        let s = wsjt_spot_fields("CQ 73 W1AW FN42", &st()).expect("CQ-phrase spots");
        assert_eq!(s.caller, "73");
        assert_eq!(s.locator, "W1AW");
    }

    #[test]
    fn self_spot_suppressed_with_empty_station() {
        // Empty station → no self-spot rejection.
        let st = SpotStation::new("", "");
        assert!(wsjt_spot_fields("CQ K1ABC FN42", &st).is_some());
    }
}
