//! JS8Call (Mode A) frame encode / decode primitives.
//!
//! A JS8 Mode-A transmission consists of 79 8-FSK symbols
//! (3 sync Costas arrays × 7 symbols, bracketing two 29-symbol LDPC
//! data blocks):
//!
//! ```text
//!  tones[0..7]     Costas A
//!  tones[7..36]    parity  (29 × 3 bits = 87 parity bits cw[0..86])
//!  tones[36..43]   Costas B
//!  tones[43..72]   message (29 × 3 bits = 87 message bits cw[87..173])
//!  tones[72..79]   Costas C
//! ```
//!
//! Message structure (87 bits, cw[87..174]):
//!
//! ```text
//!  +------------------------------------------+---------------+
//!  | 72 bits    payload (12 × 6-bit words)   | 3 bits  type  |
//!  +------------------------------------------+---------------+
//!  | 12 bits    CRC-12                        | 1 bit  spare  |
//!  +------------------------------------------+---------------+
//!   bits cw[87..158]                         cw[159..161]    cw[162..173]
//! ```
//!
//! The first 87 bits of the 174-bit codeword (`cw[0..87]`) are the parity
//! bits computed via an 87×87 matrix (transposed from the parity-check
//! structure in `ldpc_tables.rs`).

use crate::receiver::js8::ldpc_tables::NM;
use crate::receiver::js8::params::{COSTAS_ORIGINAL, Mode};

/// JS8 Mode A payload alphabet (64 chars): digits, then A–Z, then a–z,
/// then `- + / ? .`.
/// Index = the 6-bit code assigned to each character by `alphabetWord`.
///   `alphabetWord('0') = 0`, `alphabetWord('A') = 10`,
///   `alphabetWord('a') = 36`, `alphabetWord('-') = 62`,
///   `alphabetWord('+') = 63`
///   `alphabetWord('/') = 64`, `alphabetWord('?') = 65`,
///   `alphabetWord('.') = 66` (indices 64–66 are not in JS8.hpp::alphabet
///   which is only 64 chars; use the 64-char subset for encode).
pub const ALPHABET: &[u8; 64] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-+";

/// The 7×3 Costas arrays (original / FT8-style) used as the three 7-symbol
/// sync blocks in Mode A.
pub const COSTAS: [[usize; 7]; 3] = COSTAS_ORIGINAL;

// ────────────────────────────────────────────────────────────────────────────
// Alphabet mapping

/// Map a payload byte to its 6-bit JS8 alphabet word.
/// Returns `None` if the character is not in the 64-char alphabet.
pub fn alphabet_word(c: u8) -> Option<u8> {
    ALPHABET.iter().position(|&x| x == c).map(|i| i as u8)
}

/// Map a 6-bit word (0..63) back to its payload character.
pub fn word_alphabet(w: u8) -> u8 {
    ALPHABET[w as usize]
}

// ────────────────────────────────────────────────────────────────────────────
// CRC-12

/// Compute the JS8 CRC-12 over 11 bytes (75 data bits + 12 placeholder
/// zero bits), MSB-first bit-serial, poly 0xC06, init 0, no final XOR,
/// as in `boost::augmented_crc<12, 0xC06>`.
/// The result (12 bits) is then XORed with 42 (0x2A) before insertion
/// into the message.
///
/// `bytes` must be 11 bytes: the first 9 bytes are payload (72 bits),
/// byte 9 contains the 3 type bits in bits 7–5 and placeholder zeros in
/// bits 4–0, byte 10 is all zeros.
pub fn crc12(bytes: &[u8; 11]) -> u16 {
    let mut crc: u32 = 0;
    for &b in bytes.iter() {
        for bitpos in (0..8).rev() {
            let bit = (b >> bitpos) & 1;
            let top = (crc >> 11) & 1;
            crc = (crc << 1) & 0xFFF;
            crc ^= bit as u32;
            if top == 1 {
                crc ^= 0xC06;
            }
        }
    }
    (crc ^ 42) as u16
}

// ────────────────────────────────────────────────────────────────────────────
// Parity matrix

/// Parity computation row `row` (0..87), column bits from hex string `hex`
/// (22 hex chars = 88 bits, MSB-first; first bit of string = rightmost
/// column of the 87-bit row = `coef(row, 0)`).
///
/// `coef(row, col)` returns 1 iff message bit `col` (0..87) contributes
/// to parity bit `row` (0..87).
fn coef(hex: &str, col: usize) -> u8 {
    // Parse the full 88-bit hex value (22 hex chars), MSB-first.
    let mut v: u128 = 0;
    for ch in hex.chars() {
        let d = match ch {
            '0'..='9' => ch as u8 - b'0',
            'a'..='f' => ch as u8 - b'a' + 10,
            'A'..='F' => ch as u8 - b'A' + 10,
            _ => 0,
        } as u128;
        v = (v << 4) | d;
    }
    // v is 88 bits wide; bit `87 - col` = coefficient of column `col`.
    if col < 88 && (v >> (87 - col)) & 1 == 1 {
        1
    } else {
        0
    }
}

/// The 87 parity-matrix rows, taken from `JS8.cpp` (ModeA LDPC parity
/// matrix, `parity` lambda). Each hex string is 22 hex chars (88 bits);
/// bit 0 (leftmost nibble MSB) = coef(row, 0), rightmost = coef(row, 86).
const PARITY_ROWS: [&str; 87] = [
    "23bba830e23b6b6f50982e",
    "1f8e55da218c5df3309052",
    "ca7b3217cd92bd59a5ae20",
    "56f78313537d0f4382964e",
    "6be396b5e2e819e373340c",
    "293548a138858328af4210",
    "cb6c6afcdc28bb3f7c6e86",
    "3f2a86f5c5bd225c961150",
    "849dd2d63673481860f62c",
    "56cdaec6e7ae14b43feeee",
    "04ef5cfa3766ba778f45a4",
    "c525ae4bd4f627320a3974",
    "41fd9520b2e4abeb2f989c",
    "7fb36c24085a34d8c1dbc4",
    "40fc3e44bb7d2bb2756e44",
    "d38ab0a1d2e52a8ec3bc76",
    "3d0f929ef3949bd84d4734",
    "45d3814f504064f80549ae",
    "f14dbf263825d0bd04b05e",
    "db714f8f64e8ac7af1a76e",
    "8d0274de71e7c1a8055eb0",
    "51f81573dd4049b082de14",
    "d8f937f31822e57c562370",
    "b6537f417e61d1a7085336",
    "ecbd7c73b9cd34c3720c8a",
    "3d188ea477f6fa41317a4e",
    "1ac4672b549cd6dba79bcc",
    "a377253773ea678367c3f6",
    "0dbd816fba1543f721dc72",
    "ca4186dd44c3121565cf5c",
    "29c29dba9c545e267762fe",
    "1616d78018d0b4745ca0f2",
    "fe37802941d66dde02b99c",
    "a9fa8e50bcb032c85e3304",
    "83f640f1a48a8ebc0443ea",
    "3776af54ccfbae916afde6",
    "a8fc906976c35669e79ce0",
    "f08a91fb2e1f78290619a8",
    "cc9da55fe046d0cb3a770c",
    "d36d662a69ae24b74dcbd8",
    "40907b01280f03c0323946",
    "d037db825175d851f3af00",
    "1bf1490607c54032660ede",
    "0af7723161ec223080be86",
    "eca9afa0f6b01d92305edc",
    "7a8dec79a51e8ac5388022",
    "9059dfa2bb20ef7ef73ad4",
    "6abb212d9739dfc02580f2",
    "f6ad4824b87c80ebfce466",
    "d747bfc5fd65ef70fbd9bc",
    "612f63acc025b6ab476f7c",
    "05209a0abb530b9e7e34b0",
    "45b7ab6242b77474d9f11a",
    "6c280d2a0523d9c4bc5946",
    "f1627701a2d692fd9449e6",
    "8d9071b7e7a6a2eed6965e",
    "bf4f56e073271f6ab4bf80",
    "c0fc3ec4fb7d2bb2756644",
    "57da6d13cb96a7689b2790",
    "a9fa2eefa6f8796a355772",
    "164cc861bdd803c547f2ac",
    "cc6de59755420925f90ed2",
    "a0c0033a52ab6299802fd2",
    "b274db8abd3c6f396ea356",
    "97d4169cb33e7435718d90",
    "81cfc6f18c35b1e1f17114",
    "481a2a0df8a23583f82d6c",
    "081c29a10d468ccdbcecb6",
    "2c4142bf42b01e71076acc",
    "a6573f3dc8b16c9d19f746",
    "c87af9a5d5206abca532a8",
    "012dee2198eba82b19a1da",
    "b1ca4ea2e3d173bad4379c",
    "b33ec97be83ce413f9acc8",
    "5b0f7742bca86b8012609a",
    "37d8e0af9258b9e8c5f9b2",
    "35ad3fb0faeb5f1b0c30dc",
    "6114e08483043fd3f38a8a",
    "cd921fdf59e882683763f6",
    "95e45ecd0135aca9d6e6ae",
    "2e547dd7a05f6597aac516",
    "14cd0f642fc0c5fe3a65ca",
    "3a0a1dfd7eee29c2e827e0",
    "c8b5dffc335095dcdcaf2a",
    "3dd01a59d86310743ec752",
    "8abdb889efbe39a510a118",
    "3f231f212055371cf3e2a2",
];

/// Compute the 87 parity bits for a given 87-bit message (MSB-first:
/// bit 0 = payload bit 0).
///
/// `msg` is the 87 message bits as a Vec<u8> (0 or 1), MSB-first
/// (index 0 = first transmitted bit, i.e. payload bit 0).
pub fn encode_parity(msg: &[u8; 87]) -> Vec<u8> {
    let mut parity = vec![0u8; 87];
    for (i, hex) in PARITY_ROWS.iter().enumerate() {
        let mut p = 0u8;
        for j in 0..87usize {
            if coef(hex, j) != 0 && msg[j] != 0 {
                p ^= 1;
            }
        }
        parity[i] = p;
    }
    parity
}

/// Verify that a 174-bit codeword (parity[0..87] || message[87..174])
/// satisfies all 87 parity-check equations from `NM`.
/// `cw` index 0 = parity bit 0, index 87 = message bit 0.
#[allow(dead_code)]
pub(crate) fn check_codeword(cw: &[u8; 174]) -> bool {
    for i in 0..NM.len() {
        let c = &NM[i];
        let mut s: u32 = 0;
        for j in 0..c.valid {
            s += cw[c.bits[j] as usize] as u32;
        }
        if s % 2 != 0 {
            return false;
        }
    }
    true
}

// ────────────────────────────────────────────────────────────────────────────
// Encode: message bytes → 79-tone sequence

/// Encode 12 payload characters (as 6-bit alphabet words, MSB-first) with
/// a given 3-bit `message_type` (0=Heartbeat, 1=Compound, 2=CompoundDir,
/// 3=Directed, 4=Data, 6=DataJsc) into the 79-symbol tone sequence for
/// Mode A.
///
/// `payload_words` are the 6-bit words for the 12 payload characters
/// (index 0 = first character = most-significant payload bits).
///
/// Returns `Vec<u8>` of length 79, each value 0..7 (8-FSK row).
pub fn encode_tones(payload_words: [u8; 12], message_type: u8, mode: &Mode) -> Vec<u8> {
    // Build the 11-byte message array (9×8 + 3×3 + 12 zero bits + 1 spare).
    let mut bytes = [0u8; 11];
    for (i, word) in payload_words.iter().enumerate() {
        // 6-bit word i, MSB-first, occupies bitstream positions i*6 .. i*6+5.
        for b in (0..6usize).rev() {
            let target = i * 6 + (5 - b);
            if (word >> b) & 1 == 1 {
                bytes[target / 8] |= 1 << (7 - (target % 8));
            }
        }
    }

    // Insert 3 type bits (bits 72–74) at byte 9 bits 7–5.
    bytes[9] = (message_type & 0x7) << 5;

    // Compute CRC-12 over 11 bytes (including the type bits, before zero
    // bits are filled in; bytes[9] lower 5 bits + byte 10 are zero).
    let crc = crc12(&bytes);

    // Place CRC bits: top 5 bits in bytes[9][4..0], next 7 bits in bytes[10][6..0].
    bytes[9] |= ((crc >> 7) & 0x1F) as u8;
    bytes[10] = ((crc & 0x7F) << 1) as u8;

    // Extract the 87 message bits (MSB-first).
    let mut msg_bits = [0u8; 87];
    for i in 0..87usize {
        let byte = i / 8;
        let col = i % 8;
        msg_bits[i] = (bytes[byte] >> (7 - col)) & 1;
    }

    // Compute 87 parity bits.
    let parity = encode_parity(&msg_bits);

    // Build 174-bit codeword: parity (bit 0..86) || message (bit 87..173).
    let mut cw = [0u8; 174];
    for i in 0..87usize {
        cw[i] = parity[i];
        cw[87 + i] = msg_bits[i];
    }

    // Group 174 bits into 58 3-bit tone values (29 parity + 29 message).
    let mut tones = vec![0u8; 79];

    // Place Costas arrays (3 × 7 symbols). Mode A uses the original
    // (FT8) set; B/C/E use the modified set. `mode` supplies it.
    tones[0..7].copy_from_slice(&mode.costas[0].map(|x| x as u8));
    tones[36..43].copy_from_slice(&mode.costas[1].map(|x| x as u8));
    tones[72..79].copy_from_slice(&mode.costas[2].map(|x| x as u8));

    // Parity tones: cw[0..87] → tones[7..36] (29 tones, 3 bits each, MSB-first).
    for word in 0..29 {
        let b0 = cw[word * 3];
        let b1 = cw[word * 3 + 1];
        let b2 = cw[word * 3 + 2];
        let tone = (b0 << 2) | (b1 << 1) | b2;
        tones[7 + word] = tone;
    }

    // Message tones: cw[87..174] → tones[43..72] (29 tones, 3 bits each, MSB-first).
    for word in 0..29 {
        let b0 = cw[87 + word * 3];
        let b1 = cw[87 + word * 3 + 1];
        let b2 = cw[87 + word * 3 + 2];
        let tone = (b0 << 2) | (b1 << 1) | b2;
        tones[43 + word] = tone;
    }

    tones
}

// ────────────────────────────────────────────────────────────────────────────
// Decode: 87-bit decoded → 12-char message

/// Extract the 12-character payload from an 87-bit decoded message
/// (decoded[0..87] = message bits, MSB-first; payload in bits 0..71).
///
/// `decoded[0..87]` must be the 87 message bits as returned by
/// `bpdecode174` (bits 0 = MSB = payload char 0 bit 5, …, bit 71 =
/// payload char 11 bit 0).
///
/// Returns 12 chars (alphabet indices) or `None` if invalid.
pub fn extract_message(decoded: &[i8; 87]) -> Result<[u8; 12], ()> {
    let mut words = [0u8; 12];
    for i in 0..12usize {
        words[i] = ((decoded[i * 6 + 0] as u8) << 5)
            | ((decoded[i * 6 + 1] as u8) << 4)
            | ((decoded[i * 6 + 2] as u8) << 3)
            | ((decoded[i * 6 + 3] as u8) << 2)
            | ((decoded[i * 6 + 4] as u8) << 1)
            | (decoded[i * 6 + 5] as u8);
    }

    // Verify each word is in range (0..63).
    for w in &words {
        if *w >= 64 {
            return Err(());
        }
    }

    Ok(words)
}

/// Reconstruct the 12-character UTF-8 payload from alphabet word indices.
pub fn words_to_string(words: &[u8; 12]) -> String {
    words.iter().map(|w| word_alphabet(*w) as char).collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receiver::js8::ldpc_tables::NM;

    #[test]
    fn test_crc12() {
        // From js8-rs known-answer: "DR4CNK: KN4CRD AGN?" → bytes
        // [107,159,207,211,23,23,235,222, 0, 96, 0] → CRC12 = 918
        let bytes = [107u8, 159, 207, 211, 23, 23, 235, 222, 0, 96, 0];
        assert_eq!(crc12(&bytes), 918 ^ 42, "CRC12 mismatch");
    }

    #[test]
    fn test_encode_codeword_satisfies_nm() {
        // Encode a known message and verify the resulting codeword
        // satisfies all NM check equations.
        let payload = [0u8; 12]; // all zero
        let _tones = encode_tones(payload, 0, &crate::receiver::js8::params::MODE_A);

        // Rebuild bytes manually for all-zero payload, type 0.
        let mut bytes = [0u8; 11];
        bytes[9] = 0 << 5; // type = 0
        let crc = crc12(&bytes);
        bytes[9] |= ((crc >> 7) & 0x1F) as u8;
        bytes[10] = ((crc & 0x7F) << 1) as u8;

        let mut msg_bits = [0u8; 87];
        for i in 0..87usize {
            msg_bits[i] = (bytes[i / 8] >> (7 - i % 8)) & 1;
        }

        let parity = encode_parity(&msg_bits);

        // Build codeword.
        let mut cw = [0u8; 174];
        for i in 0..87usize {
            cw[i] = parity[i];
            cw[87 + i] = msg_bits[i];
        }

        // Recompute the parity-check syndromes directly and report any
        // failing check node.
        let mut failing = Vec::new();
        for (i, c) in NM.iter().enumerate() {
            let mut s: u32 = 0;
            for j in 0..c.valid {
                s += cw[c.bits[j] as usize] as u32;
            }
            if s % 2 != 0 {
                failing.push(i);
            }
        }
        assert!(
            failing.is_empty(),
            "codeword fails NM check at {:?}",
            failing
        );
    }

    #[test]
    fn test_encode_roundtrip_extract_message() {
        // Encode "W1AW" (4 chars) + 8 spaces → 12 alphabet words.
        let payload: Vec<u8> = "W1AW        "
            .chars()
            .map(|c| {
                // Map char to 6-bit alphabet word.
                let c_b = c as u8;
                if (b'0'..=b'9').contains(&c_b) {
                    c_b - b'0'
                } else if (b'A'..=b'Z').contains(&c_b) {
                    10 + (c_b - b'A')
                } else if (b'a'..=b'z').contains(&c_b) {
                    36 + (c_b - b'a')
                } else {
                    0 // space → use '0' as placeholder
                }
            })
            .collect();
        let payload: [u8; 12] = payload.try_into().unwrap();

        let msg_type = 1; // Compound

        // Build 87-message-bits by calling encode and recovering them.
        let mut bytes = [0u8; 11];
        for (i, w) in payload.iter().enumerate() {
            for (b, bit) in (0..6usize).rev().enumerate() {
                let target = i * 6 + (5 - b);
                let byte_idx = target / 8;
                let col = target % 8;
                if (*w >> b) & 1 == 1 {
                    bytes[byte_idx] |= 1 << (7 - col);
                }
            }
        }
        bytes[9] = (msg_type & 0x7) << 5;
        let crc = crc12(&bytes);
        bytes[9] |= ((crc >> 7) & 0x1F) as u8;
        bytes[10] = ((crc & 0x7F) << 1) as u8;

        let mut decoded: [i8; 87] = [0; 87];
        for i in 0..87usize {
            decoded[i] = ((bytes[i / 8] >> (7 - i % 8)) & 1) as i8;
        }

        // Verify extract_message recovers the original words.
        let words = extract_message(&decoded).unwrap();
        assert_eq!(words, payload, "roundtrip words mismatch");
    }

    #[test]
    fn test_extract_message_and_crc_check() {
        // Manually build a valid 87-bit message: "CQDEW1AW    " (12 chars).
        // 12 chars, all in the 64-char JS8 alphabet (no spaces).
        let chars: Vec<u8> = "CQDEW1AW0123".as_bytes().to_vec();
        let mut words = [0u8; 12];
        for (i, c) in chars.iter().enumerate() {
            let c = *c;
            words[i] = if (b'0'..=b'9').contains(&c) {
                c - b'0'
            } else if (b'A'..=b'Z').contains(&c) {
                10 + (c - b'A')
            } else {
                panic!("unexpected char: {}", c as char);
            };
        }

        let msg_type = 1;

        // Build bytes, compute CRC, build decoded[0..87].
        let mut bytes = [0u8; 11];
        for (i, w) in words.iter().enumerate() {
            for (b, bit) in (0..6usize).rev().enumerate() {
                let target = i * 6 + (5 - b);
                let byte_idx = target / 8;
                let col = target % 8;
                if (*w >> b) & 1 == 1 {
                    bytes[byte_idx] |= 1 << (7 - col);
                }
            }
        }
        bytes[9] = (msg_type & 0x7) << 5;
        let crc = crc12(&bytes);
        bytes[9] |= ((crc >> 7) & 0x1F) as u8;
        bytes[10] = ((crc & 0x7F) << 1) as u8;

        let mut decoded: [i8; 87] = [0; 87];
        for i in 0..87usize {
            decoded[i] = ((bytes[i / 8] >> (7 - i % 8)) & 1) as i8;
        }

        // Verify CRC by extracting and re-checking.
        let extracted = extracted_crc12_from_decoded(&decoded);
        assert_eq!(extracted, (crc) as u16, "CRC mismatch in decoded");

        // Verify payload recovery.
        let recovered = extract_message(&decoded).unwrap();
        assert_eq!(recovered, words, "payload words mismatch");
    }

    /// Reconstruct the 12-bit CRC value from the decoded[75..87] bits
    /// (MSB-first bits 75–86 = CRC bits 11..0).
    fn extracted_crc12_from_decoded(decoded: &[i8; 87]) -> u16 {
        let mut v: u16 = 0;
        for i in 0..12usize {
            v = (v << 1) | (decoded[75 + i] as u16 & 1);
        }
        v
    }
}
