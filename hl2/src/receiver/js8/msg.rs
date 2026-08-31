//! JS8Call (Mode A) semantic frame unpacking.
//!
//! Mirrors the `js8call` reference decoder (`varicode.cpp` +
//! `decodedtext.cpp` + `jsc.cpp`): after LDPC decode yields the 87 message
//! bits (72 payload bits + 3 i3bit type bits + 12 CRC bits), this module
//! validates the CRC and decodes the 72-bit payload into a structured
//! [`DecodedFrame`] (heartbeat, compound, compound-directed, directed,
//! legacy data, or fast/JSC data).
//!
//! i3bit values (varicode.h): 1 = First, 2 = Last, 4 = Data
//! (flagged frame, no frame-type header). If Data is not set, the first bit
//! of the payload distinguishes the families:
//!   * payload bit 0 == 1  → legacy data frame (bit 1 selects JSC vs huff);
//!   * otherwise payload bits [0..3) select the compound family:
//!     0=heartbeat, 1=compound, 2=compound-directed, 3=directed.

use crate::receiver::js8::frame::crc12;
use crate::receiver::js8::jsc_map::{
    JSC_CODE_TO_POS, JSC_MAP_NWORDS, JSC_MAP_OFFSETS, JSC_MAP_SIZES, JSC_MAP_WORDS,
};

/// i3bit frame type selectors.
pub const JS8_FIRST: u8 = 1;
pub const JS8_LAST: u8 = 2;
pub const JS8_DATA: u8 = 4;

/// payload[0..3) compound-family frame types.
pub const FRAME_HEARTBEAT: u8 = 0;
pub const FRAME_COMPOUND: u8 = 1;
pub const FRAME_COMPOUND_DIRECTED: u8 = 2;
pub const FRAME_DIRECTED: u8 = 3;

/// A decoded JS8 frame, ready for logging / display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFrame {
    /// Frame class: [`FRAME_HEARTBEAT`], [`FRAME_COMPOUND`],
    /// [`FRAME_COMPOUND_DIRECTED`], [`FRAME_DIRECTED`],
    /// [`DecodedFrame::KIND_DATA_HUFF`], [`DecodedFrame::KIND_DATA_JSC_LEGACY`],
    /// [`DecodedFrame::KIND_DATA_JSC`].
    pub kind: u8,
    /// Station (compound or directed `from`). Empty for data frames.
    pub callsign: String,
    /// Directed frames: destination callsign.
    pub to: Option<String>,
    /// Grid locator (compound/heartbeat with grid).
    pub grid: Option<String>,
    /// Command name (no leading space), if present.
    pub cmd: Option<String>,
    /// Command number / SNR value, if present.
    pub num: Option<i16>,
    /// Heartbeat CQ/HB variant selector (`bits3`).
    pub bits3: u8,
    /// Heartbeat: whether this is an alt (CQ) frame.
    pub is_alt: bool,
    /// Data frames: the decoded free text.
    pub text: String,
    /// The display string (decodedtext.cpp conventions).
    pub message: String,
}

impl DecodedFrame {
    pub const KIND_DATA_HUFF: u8 = 0x10;
    pub const KIND_DATA_JSC_LEGACY: u8 = 0x11;
    pub const KIND_DATA_JSC: u8 = 0x12;
}

const NBASECALL: u32 = 37 * 36 * 10 * 27 * 27 * 27; // 944_790
const NBASEGRID: u16 = 180 * 180; // 32_400
const NUSERGRID: u32 = NBASEGRID as u32 + 10;
const NMAXGRID: u16 = (1 << 15) - 1; // 32_767

/// Callsign/grid alphabet (39 chars; index = varicode `alphanumeric`).
pub const ALPHANUMERIC: &[u8; 39] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ /@";

/// Basecalls: special "callsigns" with fixed codes (varicode.cpp).
static BASECALLS: &[(&str, u32)] = &[
    ("<....>", 1),
    ("@ALLCALL", 2),
    ("@JS8NET", 3),
    ("@DX/NA", 4),
    ("@DX/SA", 5),
    ("@DX/EU", 6),
    ("@DX/AS", 7),
    ("@DX/AF", 8),
    ("@DX/OC", 9),
    ("@DX/AN", 10),
    ("@REGION/1", 11),
    ("@REGION/2", 12),
    ("@REGION/3", 13),
    ("@GROUP/0", 14),
    ("@GROUP/1", 15),
    ("@GROUP/2", 16),
    ("@GROUP/3", 17),
    ("@GROUP/4", 18),
    ("@GROUP/5", 19),
    ("@GROUP/6", 20),
    ("@GROUP/7", 21),
    ("@GROUP/8", 22),
    ("@GROUP/9", 23),
    ("@COMMAND", 24),
    ("@CONTROL", 25),
    ("@NET", 26),
    ("@NTS", 27),
    ("@RESERVE/0", 28),
    ("@RESERVE/1", 29),
    ("@RESERVE/2", 30),
    ("@RESERVE/3", 31),
    ("@RESERVE/4", 32),
    ("@APRSIS", 33),
    ("@RAGCHEW", 34),
    ("@JS8", 35),
    ("@EMCOMM", 36),
    ("@ARES", 37),
    ("@MARS", 38),
    ("@AMRRON", 39),
    ("@RACES", 40),
    ("@RAYNET", 41),
    ("@RADAR", 42),
    ("@SKYWARN", 43),
    ("@CQ", 44),
    ("@HB", 45),
    ("@QSO", 46),
    ("@QSOPARTY", 47),
    ("@CONTEST", 48),
    ("@FIELDDAY", 49),
    ("@SOTA", 50),
    ("@IOTA", 51),
    ("@POTA", 52),
    ("@QRP", 53),
    ("@QRO", 54),
];

/// Directed command table (code -> name), per varicode.cpp `directed_cmds`.
/// Index 0 ("?") is the legacy alias for " SNR?".
static DIRECTED_CMDS: [&str; 32] = [
    " SNR?",          // 0  (also "?")
    " DIT DIT",       // 1
    " NACK",          // 2
    " HEARING?",      // 3
    " GRID?",         // 4
    ">",              // 5  (relay)
    " STATUS?",       // 6
    " STATUS",        // 7
    " HEARING",       // 8
    " MSG",           // 9
    " MSG TO:",       // 10
    " QUERY",         // 11
    " QUERY MSGS",    // 12 (also " QUERY MSGS?")
    " QUERY CALL",    // 13
    " ACK",           // 14
    " GRID",          // 15
    " INFO?",         // 16
    " INFO",          // 17
    " FB",            // 18
    " HW CPY?",       // 19
    " SK",            // 20
    " RR",            // 21
    " QSL?",          // 22
    " QSL",           // 23
    " CMD",           // 24
    " SNR",           // 25
    " NO",            // 26
    " YES",           // 27
    " 73",            // 28
    " HEARTBEAT SNR", // 29
    " AGN?",          // 30
    " ",              // 31
];

const CQS: [&str; 8] = [
    "CQ CQ CQ",
    "CQ DX",
    "CQ QRP",
    "CQ CONTEST",
    "CQ FIELD",
    "CQ FD",
    "CQ CQ",
    "CQ",
];

/// The huffman table used for legacy (non-JSC) data frames.
static HUFF_TABLE: &[(&str, &str)] = &[
    (" ", "01"),
    ("E", "100"),
    ("T", "1101"),
    ("A", "0011"),
    ("O", "11111"),
    ("I", "11100"),
    ("N", "10111"),
    ("S", "10100"),
    ("H", "00011"),
    ("R", "00000"),
    ("D", "111011"),
    ("L", "110011"),
    ("C", "110001"),
    ("U", "101101"),
    ("M", "101011"),
    ("W", "001011"),
    ("F", "001001"),
    ("G", "000101"),
    ("Y", "000011"),
    ("P", "1111011"),
    ("B", "1111001"),
    (".", "1110100"),
    ("V", "1100101"),
    ("K", "1100100"),
    ("-", "1100001"),
    ("+", "1100000"),
    ("?", "1011001"),
    ("!", "1011000"),
    ("\"", "1010101"),
    ("X", "1010100"),
    ("0", "0010101"),
    ("J", "0010100"),
    ("1", "0010001"),
    ("Q", "0010000"),
    ("2", "0001001"),
    ("Z", "0001000"),
    ("3", "0000101"),
    ("5", "0000100"),
    ("4", "11110101"),
    ("9", "11110100"),
    ("8", "11110001"),
    ("6", "11110000"),
    ("7", "11101011"),
    ("/", "11101010"),
];

const JSC_UNMAPPED: u32 = 0xFFFF_FFFF;

fn an_index(c: u8) -> Option<u32> {
    ALPHANUMERIC.iter().position(|&x| x == c).map(|i| i as u32)
}

// ────────────────────────────────────────────────────────────────────────────
// Small helpers

/// Format an SNR integer for display (varicode.cpp `formatSNR`), e.g. "+20".
pub fn format_snr(snr: i32) -> Option<String> {
    if snr < -60 || snr > 60 {
        return None;
    }
    if snr < 0 {
        Some(format!("-{0}", -snr))
    } else {
        Some(format!("+{snr}"))
    }
}

fn is_snr(cmd_idx: usize) -> bool {
    cmd_idx == 25 || cmd_idx == 29
}

/// Pack a number into 1..=62 (varicode.cpp `packNum`): `clamp(n,-30,31) + 31`.
/// Returns `None` when the text is not an integer.
pub fn pack_num(num: &str) -> Option<u8> {
    let n: i32 = num.trim().parse().ok()?;
    Some((n.clamp(-30, 31) + 31) as u8)
}

/// Look up a command by name (trimmed), returning its code.
pub fn cmd_index(name: &str) -> Option<usize> {
    let t = name.trim();
    if t == "?" {
        return Some(0);
    }
    if t == "AGN?" {
        // legacy "AGN?" is code 30.
    }
    DIRECTED_CMDS
        .iter()
        .position(|&c| c.trim() == t)
        .or_else(|| DIRECTED_CMDS.iter().position(|&c| c == t))
}

/// Pack a command (+ optional number) into 8 bits (varicode.cpp `packCmd`).
/// SNR commands: `[1][hb][6-bit num]`. Non-SNR: low 7 bits = command index.
pub fn pack_cmd(cmd_idx: usize, num: u8) -> u8 {
    if is_snr(cmd_idx) {
        let hb = if cmd_idx == 29 { 1 } else { 0 };
        (((1u8 << 1) | hb) << 6) | (num & 0x3F)
    } else {
        cmd_idx as u8 & 0x7F
    }
}

/// Unpack 8 bits into a command index + number (varicode.cpp `unpackCmd`).
/// Returns the cmd index; `*pnum` is set to the number (0 if none).
pub fn unpack_cmd(value: u8, pnum: &mut Option<i16>) -> Option<usize> {
    if value & 0x80 != 0 {
        let n = ((value & 0x3F) as i16) - 31;
        *pnum = Some(n);
        return Some(if value & 0x40 != 0 { 29 } else { 25 });
    }
    let c = value & 0x7F;
    *pnum = Some(0);
    (c < 32).then_some(c as usize)
}

// ────────────────────────────────────────────────────────────────────────────
// Grid pack / unpack (varicode.cpp deg2grid / grid2deg / packGrid / unpackGrid)

/// Grid (4- or 6-char) to degrees, using 32-bit float arithmetic to match the
/// C++ reference (`grid2deg`, `QPair<float,float>`). Padding "mm" for <6 chars.
fn grid2deg(grid: &str) -> (f32, f32) {
    let mut g: String = grid.to_string();
    if g.chars().count() < 6 {
        let g4: String = g.chars().take(4).collect();
        g = format!("{g4}mm");
    }
    let chars: Vec<char> = g.chars().take(6).collect();
    let (a, b) = (chars[0].to_ascii_uppercase(), chars[1].to_ascii_uppercase());
    let z = chars[2].to_ascii_lowercase();
    let d = chars[3].to_ascii_lowercase();
    let (e, f) = (chars[4].to_ascii_lowercase(), chars[5].to_ascii_lowercase());

    let nlong = (180 - 20 * (a as i32 - 'A' as i32)) as f32;
    let n20d = 2 * (z as i32 - '0' as i32);
    let xminlong = 5.0_f32 * ((e as u8 - b'a' as u8) as f32 + 0.5) / 60.0;
    let dlong = nlong - (n20d as f32) - xminlong;

    let nlat = (-90 + 10 * (b as i32 - 'A' as i32) + (d as i32 - '0' as i32)) as f32;
    let xminlat = 2.5_f32 * ((f as u8 - b'a' as u8) as f32 + 0.5) / 60.0;
    let dlat = nlat + xminlat;

    (dlong, dlat)
}

fn deg2grid(dlong: f32, dlat: f32) -> String {
    let mut dlong = dlong;
    if dlong < -180.0 {
        dlong += 360.0;
    }
    if dlong > 180.0 {
        dlong -= 360.0;
    }
    let nlong = (60.0_f32 * (180.0 - dlong) / 5.0) as i32;
    let nlat = (60.0_f32 * (dlat + 90.0) / 2.5) as i32;
    let mut s = [0u8; 6];
    // longitude -> cells 0, 2
    {
        let n1 = nlong / 240;
        let n2 = (nlong - 240 * n1) / 24;
        let n3 = nlong - 240 * n1 - 24 * n2;
        s[0] = b'A' + n1 as u8;
        s[2] = b'0' + n2 as u8;
        s[4] = b'a' + n3 as u8;
    }
    // latitude -> cells 1, 3
    {
        let n1 = nlat / 240;
        let n2 = (nlat - 240 * n1) / 24;
        let n3 = nlat - 240 * n1 - 24 * n2;
        s[1] = b'A' + n1 as u8;
        s[3] = b'0' + n2 as u8;
        s[5] = b'a' + n3 as u8;
    }
    String::from_utf8(s.to_vec()).unwrap()
}

/// Pack a 4-char Maidenhead grid into a 15-bit value (varicode.cpp `packGrid`).
/// Returns [`NMAXGRID`] when the input is invalid.
///
/// C++: `int ilong = pair.first; int ilat = pair.second + 90;` where the pair
/// holds *float* degrees. So the int-truncation happens on the 32-bit value,
/// NOT (f32 + 90) cast. We replicate that exactly.
pub fn pack_grid(value: &str) -> u16 {
    let grid = value.trim();
    if grid.chars().count() < 4 {
        return NMAXGRID;
    }
    let (dlong, dlat) = grid2deg(&grid[..4]);
    let ilong = dlong as i32 + 180;
    let ilat = dlat as i32 + 90;
    ((ilong / 2) as u16) * 180 + ilat as u16
}

/// Unpack a 15-bit Maidenhead value to a grid (varicode.cpp `unpackGrid`).
/// Values above `NBASEGRID` unpack to the empty string.
pub fn unpack_grid(value: u16) -> String {
    if value > NBASEGRID {
        return String::new();
    }
    let dlat = (value % 180) as f32 - 90.0;
    let dlong = (value / 180) as f32 * 2.0 - 180.0 + 2.0;
    let full = deg2grid(dlong, dlat);
    // C++ `unpackGrid` returns `.left(4)`.
    full.chars().take(4).collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Callsign / alphanumeric50 pack / unpack

/// Pack a callsign (up to 11 chars: A-Z 0-9 space /, optional leading @)
/// into a 50-bit value (varicode.cpp `packAlphaNumeric50`).
///
/// Slot radices (MSB→LSB): `[39, 38, 38, 2, 38, 38, 38, 2, 38, 38, 38]`,
/// i.e. multipliers `38^k` with interleaved 2s as in the C++ reference.
/// Slot 0 may hold '@' (index 38); slots 3 and 7 hold a binary '/' flag;
/// all other slots hold an ALPHANUMERIC index (0..37).
pub fn pack_alphanumeric50(value: &str) -> u64 {
    let mut word: Vec<u8> = value
        .to_ascii_uppercase()
        .bytes()
        .filter(|&b| {
            b.is_ascii_whitespace()
                || (b'A'..=b'Z').contains(&b)
                || (b'0'..=b'9').contains(&b)
                || b == b'/'
                || b == b'@'
        })
        .collect();

    if word.len() > 3 && word[3] != b'/' {
        word.insert(3, b' ');
    }
    if word.len() > 7 && word[7] != b'/' {
        word.insert(7, b' ');
    }
    while word.len() < 11 {
        word.push(b' ');
    }
    let w = [
        word[0], word[1], word[2], word[3], word[4], word[5], word[6], word[7], word[8], word[9],
        word[10],
    ];

    const M0: u64 = 38 * 38 * 38 * 2 * 38 * 38 * 38 * 2 * 38 * 38;
    const M1: u64 = 38 * 38 * 38 * 2 * 38 * 38 * 38 * 2 * 38;
    const M2: u64 = 38 * 38 * 38 * 38 * 38 * 38 * 2 * 2;
    const M3: u64 = 38 * 38 * 38 * 38 * 38 * 38 * 2;
    const M4: u64 = 38 * 38 * 38 * 38 * 38 * 2;
    const M5: u64 = 38 * 38 * 38 * 38 * 2;
    const M6: u64 = 38 * 38 * 38 * 2;
    const M7: u64 = 38 * 38 * 38;
    const M8: u64 = 38 * 38;
    const M9: u64 = 38;
    const M10: u64 = 1;

    let idx = |c: u8| an_index(c).unwrap_or(0) as u64;
    idx(w[0]) * M0
        + idx(w[1]) * M1
        + idx(w[2]) * M2
        + (if w[3] == b'/' { 1 } else { 0 }) * M3
        + idx(w[4]) * M4
        + idx(w[5]) * M5
        + idx(w[6]) * M6
        + (if w[7] == b'/' { 1 } else { 0 }) * M7
        + idx(w[8]) * M8
        + idx(w[9]) * M9
        + idx(w[10]) * M10
}

/// Unpack a 50-bit value to a callsign (varicode.cpp `unpackAlphaNumeric50`).
/// The 39-char slot0 alphabet includes '@', the 39-char alphanumeric table
/// is used for every slot (slots 3/7 hold '/' (1) or ' ' (0)).
pub fn unpack_alphanumeric50(mut packed: u64) -> String {
    let mut word = [0u8; 11];
    word[10] = ALPHANUMERIC[(packed % 38) as usize];
    packed /= 38;
    word[9] = ALPHANUMERIC[(packed % 38) as usize];
    packed /= 38;
    word[8] = ALPHANUMERIC[(packed % 38) as usize];
    packed /= 38;
    word[7] = if packed % 2 != 0 { b'/' } else { b' ' };
    packed /= 2;
    word[6] = ALPHANUMERIC[(packed % 38) as usize];
    packed /= 38;
    word[5] = ALPHANUMERIC[(packed % 38) as usize];
    packed /= 38;
    word[4] = ALPHANUMERIC[(packed % 38) as usize];
    packed /= 38;
    word[3] = if packed % 2 != 0 { b'/' } else { b' ' };
    packed /= 2;
    word[2] = ALPHANUMERIC[(packed % 38) as usize];
    packed /= 38;
    word[1] = ALPHANUMERIC[(packed % 38) as usize];
    packed /= 38;
    word[0] = ALPHANUMERIC[(packed % 39) as usize];
    // Trim trailing spaces / drop all spaces (C++: value.replace(" ", "")).
    let s = String::from_utf8_lossy(&word).to_string();
    s.replace(' ', "")
}

/// Pack a callsign into a 27-bit value (varicode.cpp `packCallsign`).
/// Handles basecalls, `/P` portability and 2-6 char alphanumeric signs.
pub fn pack_callsign(value: &str) -> Option<u32> {
    let cs = value.trim().to_ascii_uppercase();
    if let Some((_, off)) = BASECALLS.iter().find(|(name, _)| *name == cs) {
        return Some(NBASECALL + off);
    }

    let mut cs: Vec<u8> = cs.bytes().filter(|&b| b != b' ').collect();
    let portable = cs.len() > 2 && &cs[cs.len() - 2..] == b"/P";
    if portable {
        cs.truncate(cs.len() - 2);
    }

    // workarounds (varicode.cpp).
    if cs.starts_with(b"3DA0") && cs.len() > 4 {
        let mut new = b"3D0".to_vec();
        new.extend_from_slice(&cs[4..]);
        cs = new;
    }
    if cs.len() >= 3 && cs[0] == b'3' && cs[1] == b'X' && (b'A'..=b'Z').contains(&cs[2]) {
        cs.insert(0, b'Q');
        cs.remove(2);
    }

    let slen = cs.len();
    if slen < 2 || slen > 6 {
        return None;
    }
    let cs = String::from_utf8(cs).unwrap();

    // Padding permutations, in C++ order (varicode.cpp packCallsign). The last
    // one satisfying the LLD pattern `([0-9A-Z ])([0-9A-Z])([0-9])([A-Z ])([A-Z ])([A-Z ])`
    // wins (C++ overwrites `matched` on each successful regex match).
    let mut perms: Vec<String> = vec![cs.clone()];
    match slen {
        2 => perms.push(format!(" {cs}   ")),
        3 => {
            perms.push(format!(" {cs}  "));
            perms.push(format!("{cs}   "));
        }
        4 => {
            perms.push(format!(" {cs} "));
            perms.push(format!("{cs}  "));
        }
        5 => {
            perms.push(format!(" {cs}"));
            perms.push(format!("{cs} "));
        }
        _ => {}
    }

    let chars_ok = [
        |c: u8| c.is_ascii_digit() || c.is_ascii_uppercase() || c == b' ',
        |c: u8| c.is_ascii_digit() || c.is_ascii_uppercase(),
        |c: u8| c.is_ascii_digit(),
        |c: u8| c.is_ascii_uppercase() || c == b' ',
        |c: u8| c.is_ascii_uppercase() || c == b' ',
        |c: u8| c.is_ascii_uppercase() || c == b' ',
    ];

    let mut matched: Option<[u8; 6]> = None;
    for p in &perms {
        let bytes = p.as_bytes();
        if bytes.len() != 6 {
            continue;
        }
        if (0..6).all(|i| chars_ok[i](bytes[i])) {
            let mut w = [0u8; 6];
            w.copy_from_slice(bytes);
            matched = Some(w);
        }
    }
    let matched = matched?;

    let (i0, i1, i2, i3, i4, i5) = (
        an_index(matched[0])?,
        an_index(matched[1])?,
        an_index(matched[2])?,
        an_index(matched[3])?,
        an_index(matched[4])?,
        an_index(matched[5])?,
    );
    let mut packed = i0;
    packed = 36 * packed + i1;
    packed = 10 * packed + i2;
    packed = 27 * packed + (i3 - 10);
    packed = 27 * packed + (i4 - 10);
    packed = 27 * packed + (i5 - 10);
    Some(packed)
}

/// Map a 27-bit basecall value to its name, if any (varicode.cpp).
pub fn basecall_name(value: u32) -> Option<&'static str> {
    if value < NBASECALL {
        return None;
    }
    BASECALLS
        .iter()
        .find(|(_, off)| NBASECALL + off == value)
        .map(|(name, _)| *name)
}

/// Unpack a 27-bit value to a callsign (varicode.cpp `unpackCallsign`).
pub fn unpack_callsign(value: u32, portable: bool) -> String {
    if let Some(name) = basecall_name(value) {
        return name.to_string();
    }
    let mut word = [0u8; 6];
    let mut v = value;
    word[5] = ALPHANUMERIC[10 + (v % 27) as usize];
    v /= 27;
    word[4] = ALPHANUMERIC[10 + (v % 27) as usize];
    v /= 27;
    word[3] = ALPHANUMERIC[10 + (v % 27) as usize];
    v /= 27;
    word[2] = ALPHANUMERIC[(v % 10) as usize];
    v /= 10;
    word[1] = ALPHANUMERIC[(v % 36) as usize];
    v /= 36;
    word[0] = ALPHANUMERIC[v as usize];

    let mut cs = String::from_utf8(word.to_vec()).unwrap();
    if cs.starts_with("3D0") {
        cs = format!("3DA0{}", &cs[3..]);
    }
    if cs.starts_with('Q') && cs.chars().nth(1).is_some_and(|c| c.is_ascii_uppercase()) {
        cs = format!("3X{}", &cs[1..]);
    }
    cs = cs.trim().to_string();
    if portable {
        cs.push_str("/P");
    }
    cs
}

// ────────────────────────────────────────────────────────────────────────────
// 72-bit pack / unpack (6-bit alphabet words + 8-remainder-bits)

/// Unpack a 12-char JS8 payload (6-bit words) into the first 64 bits
/// (`value`, MSB-first) and the remaining 8 bits (`rem`).
/// Mirrors varicode.cpp `unpack72bits` exactly: `words[i]` occupies absolute
/// bit positions `58-6*i .. 63-6*i` (word0 = top 6 bits).
pub fn unpack_72bits(words: &[u8; 12]) -> (u64, u8) {
    let mut value: u64 = 0;
    for i in 0..10 {
        value |= (words[i] as u64) << (58 - 6 * i);
    }
    value |= (words[10] >> 2) as u64;
    let rem = ((words[10] & 0x03) << 6) | words[11];
    (value, rem)
}

/// Pack 64 + 8 bits into 12 6-bit JS8 alphabet words (varicode.cpp
/// `pack72bits`). Mirrors the C++ reference exactly: the low nibble of
/// `value` plus the top 2 bits of `rem` form word 10; word 11 is the low 6
/// bits of `rem`; words 0-9 carry `value >> 4` (word 0 = MSB). Always
/// succeeds for a well-formed value (it is masked to 64 bits by construction).
pub fn pack_72bits(value: u64, rem: u8) -> Option<[u8; 12]> {
    let mut out = [0u8; 12];
    out[10] = ((((value & 0x0F) as u8) << 2) | ((rem >> 6) & 0x03)) as u8;
    out[11] = (rem & 0x3F) as u8;
    let mut v = value >> 4;
    for i in 0..10 {
        out[9 - i] = (v & 0x3F) as u8;
        v >>= 6;
    }
    Some(out)
}

/// View a (u64, u8) 72-bit field as MSB-first bits.
fn bits72(value: u64, rem: u8) -> Vec<u8> {
    let mut bits = vec![0u8; 72];
    for i in 0..64 {
        bits[i] = (value >> (63 - i)) as u8 & 1;
    }
    for i in 0..8 {
        bits[64 + i] = (rem >> (7 - i)) & 1;
    }
    bits
}

fn bit_last_zero(bits: &[u8]) -> usize {
    bits.iter().rposition(|&b| b == 0).unwrap_or(0)
}

// ────────────────────────────────────────────────────────────────────────────
// Compound / heartbeat / directed framing

pub fn pack_compound_frame(
    callsign: &str,
    frame_type: u8,
    num: u16,
    bits3: u8,
) -> Option<[u8; 12]> {
    if !(FRAME_HEARTBEAT..=FRAME_DIRECTED).contains(&frame_type) {
        return None;
    }
    let packed_cs = pack_alphanumeric50(callsign);
    if packed_cs == 0 {
        return None;
    }
    let packed_11 = (num as u64) >> 5 & 0x7FF;
    let packed_8 = (((num & 0x1F) as u8) << 3) | (bits3 & 0x07);
    let value = ((frame_type as u64) << 61) | (packed_cs << 11) | packed_11;
    pack_72bits(value, packed_8)
}

/// Unpack a compound-family frame (heartbeat/compound/compound-directed).
/// Returns (type, callsign, num(16), bits3).
pub fn unpack_compound_frame(words: &[u8; 12]) -> Option<(u8, String, u16, u8)> {
    let (value, rem) = unpack_72bits(words);
    let packed_5 = rem >> 3;
    let packed_3 = rem & 0x07;
    let flag = (value >> 61) as u8;
    if flag == 4 || flag > 3 {
        return None;
    }
    let packed_cs = (value >> 11) & ((1u64 << 50) - 1);
    let packed_11 = value & 0x7FF;
    let callsign = unpack_alphanumeric50(packed_cs);
    let num = (packed_11 as u16) << 5 | packed_5 as u16;
    Some((flag, callsign, num, packed_3))
}

/// Heartbeat / CQ frame.
pub fn unpack_heartbeat(words: &[u8; 12]) -> Option<(String, Option<String>, bool, u8)> {
    let (flag, callsign, num, bits3) = unpack_compound_frame(words)?;
    if flag != FRAME_HEARTBEAT {
        return None;
    }
    let is_alt = num & (1 << 15) != 0;
    let grid = unpack_grid(num & 0x7FFF);
    Some((
        callsign,
        if grid.is_empty() { None } else { Some(grid) },
        is_alt,
        bits3,
    ))
}

/// Compound (callsign + grid | command) frame.
pub fn unpack_compound(words: &[u8; 12]) -> Option<(u8, String, Option<String>)> {
    let (flag, callsign, num, _bits3) = unpack_compound_frame(words)?;
    if flag != FRAME_COMPOUND && flag != FRAME_COMPOUND_DIRECTED {
        return None;
    }
    let mut extra = None;
    if num as u32 <= NBASEGRID as u32 {
        let g = unpack_grid(num);
        if !g.is_empty() {
            extra = Some(g);
        }
    } else if NUSERGRID <= num as u32 && (num as u32) < NMAXGRID as u32 {
        let mut n = None;
        if let Some(idx) = unpack_cmd((num as u32 - NUSERGRID) as u8, &mut n) {
            let name = DIRECTED_CMDS[idx].trim();
            if is_snr(idx) {
                if let Some(s) = n.and_then(|v| format_snr(v as i32)) {
                    extra = Some(format!("{name} {s}"));
                }
            } else {
                extra = Some(name.to_string());
            }
        }
    }
    Some((flag, callsign, extra))
}

/// Directed frame (from → to + command [+ number]).
pub fn unpack_directed(words: &[u8; 12]) -> Option<(String, String, String, Option<i16>)> {
    let (value, extra) = unpack_72bits(words);
    let flag = (value >> 61) as u8;
    if flag != FRAME_DIRECTED {
        return None;
    }
    let packed_from = (value >> 33) & ((1u64 << 28) - 1);
    let packed_to = (value >> 5) & ((1u64 << 28) - 1);
    let packed_cmd = (value & 0x1F) as u8;

    let portable_from = extra & 0x80 != 0;
    let portable_to = extra & 0x40 != 0;
    let inum = (extra & 0x3F) as u32;

    let from = unpack_callsign(packed_from as u32, portable_from);
    let to = unpack_callsign(packed_to as u32, portable_to);
    let cmd = String::from(DIRECTED_CMDS[packed_cmd as usize % 32].trim());
    let num = (inum != 0).then(|| inum as i16 - 31);
    Some((from, to, cmd, num))
}

/// Pack a directed frame (from, to, command name, number).
pub fn pack_directed(from: &str, to: &str, cmd: &str, num: Option<i16>) -> Option<[u8; 12]> {
    let cmd_idx = cmd_index(cmd)?;
    let inum = match num {
        Some(n) => (n as i32 + 31).clamp(0, 62) as u8,
        None => 0,
    };
    let raw = from.trim();
    let pf = raw.ends_with("/P") && raw.len() > 2;
    let f = if pf { &raw[..raw.len() - 2] } else { raw };
    let raw = to.trim();
    let pt = raw.ends_with("/P") && raw.len() > 2;
    let t = if pt { &raw[..raw.len() - 2] } else { raw };
    let packed_from = pack_callsign(f)?;
    let packed_to = pack_callsign(t)?;
    let value =
        (3u64 << 61) | ((packed_from as u64) << 33) | ((packed_to as u64) << 5) | cmd_idx as u64;
    let extra = ((pf as u8) << 7) | ((pt as u8) << 6) | inum;
    pack_72bits(value, extra)
}

// ────────────────────────────────────────────────────────────────────────────
// Data frames: legacy huff / legacy JSC / fast JSC

/// Huffman-decode a bit string using the JS8 table.
pub fn huff_decode(bits: &[u8]) -> String {
    let mut out = String::new();
    let mut i = 0;
    'outer: while i < bits.len() {
        for &(ch, code) in HUFF_TABLE {
            let cb: Vec<u8> = code.bytes().map(|b| (b != b'0') as u8).collect();
            if cb.len() <= bits.len() - i && bits[i..i + cb.len()] == cb[..] {
                out.push_str(ch);
                i += cb.len();
                continue 'outer;
            }
        }
        break;
    }
    out
}

/// JSC (partial-word (s,c)-dense) decompression (jsc.cpp).
///
/// `bits` is an MSB-first bit string; 4 bits = 1 byte in the dense code.
pub fn jsc_decompress(bits: &[u8]) -> Option<String> {
    const S: u32 = 7;
    const C: u32 = 9;

    let mut bytes = Vec::with_capacity(bits.len() / 4);
    let mut separators = Vec::new();
    let mut i = 0;
    while i + 4 <= bits.len() {
        let mut byte: u8 = 0;
        for k in 0..4 {
            byte = (byte << 1) | bits[i + k];
        }
        bytes.push(byte as u32);
        if u32::from(byte) < S && i + 4 < bits.len() && bits[i + 4] == 1 {
            separators.push((bytes.len() - 1) as usize);
        }
        i += 4;
    }

    // base offsets for the terminator index (jsc.cpp `base`).
    let base: Vec<u32> = {
        let mut v = vec![0u32; (bits.len() / 4 + 1).min(8)];
        if v.len() >= 2 {
            v[1] = S;
        }
        for k in 2..v.len() {
            v[k] = v[k - 1] + S * C;
        }
        v
    };

    let mut out = String::new();
    let nb = bytes.len();
    let mut start = 0usize;
    while start < nb {
        let mut k: usize = 0;
        let mut j: u32 = 0;
        while start + k < nb && bytes[start + k] >= S {
            j = j * C + (bytes[start + k] - S);
            k += 1;
        }
        if j >= JSC_MAP_SIZE_CONST {
            break;
        }
        if start + k >= nb {
            break;
        }
        let base_k = base.get(k).copied().unwrap_or(0);
        j = j * S + bytes[start + k] + base_k;
        if (j as usize) >= JSC_MAP_NWORDS {
            break;
        }
        let p = JSC_CODE_TO_POS[j as usize];
        if p == JSC_UNMAPPED {
            break;
        }
        let p = p as usize;
        let begin = JSC_MAP_OFFSETS[p] as usize;
        let len = JSC_MAP_SIZES[p] as usize;
        let word = &JSC_MAP_WORDS[begin..begin + len];
        out.push_str(&String::from_utf8_lossy(word));

        if let Some(first) = separators.first().copied() {
            if first == start + k {
                out.push(' ');
                separators.remove(0);
            }
        }

        start += k + 1;
    }
    (!out.is_empty()).then_some(out)
}

const JSC_MAP_SIZE_CONST: u32 = 262_144;

/// Legacy (payload bit 0 == 1) data message:
/// `[1][huff|jsc][payload][0][111…1]`.
pub fn unpack_legacy_data(words: &[u8; 12]) -> Option<(u8, String)> {
    let (value, rem) = unpack_72bits(words);
    let bits = bits72(value, rem);
    if bits.first() != Some(&1) {
        return None;
    }
    let compressed = bits[1] == 1;
    let last0 = bit_last_zero(&bits);
    if last0 <= 1 {
        return None;
    }
    let payload = &bits[2..last0];
    let text = if compressed {
        jsc_decompress(payload)?
    } else {
        huff_decode(payload)
    };
    let kind = if compressed {
        DecodedFrame::KIND_DATA_JSC_LEGACY
    } else {
        DecodedFrame::KIND_DATA_HUFF
    };
    Some((kind, text))
}

/// Fast data message (i3bit & Data): full 72 bits of JSC,
/// padded `[payload][0][111…1]`.
pub fn unpack_fast_data(words: &[u8; 12]) -> Option<String> {
    let (value, rem) = unpack_72bits(words);
    let bits = bits72(value, rem);
    let last0 = bit_last_zero(&bits);
    jsc_decompress(&bits[..last0])
}

// ────────────────────────────────────────────────────────────────────────────
// Top-level dispatcher + CRC validation

/// Validate the 12-bit CRC carried in message bits [75..87).
/// `msg` is the 87 message bits (MSB-first, as returned by LDPC decode).
pub fn check_crc(msg: &[u8; 87]) -> bool {
    // Rebuild the 11-byte layout the CRC was computed over: the CRC is
    // computed *before* its low 7 bits are filled into byte 10.
    let mut bytes = [0u8; 11];
    for i in 0..9 {
        for j in 0..8 {
            bytes[i] = (bytes[i] << 1) | msg[i * 8 + j];
        }
    }
    let mut b9 = 0u8;
    for j in 0..3 {
        b9 = (b9 << 1) | msg[72 + j];
    }
    bytes[9] = b9 << 5;
    bytes[10] = 0;
    let crc = crc12(&bytes) as u16;
    let mut got = 0u16;
    for i in 0..12 {
        got = (got << 1) | msg[75 + i] as u16;
    }
    got == crc
}

/// Unpack a full 87-bit JS8 message (MSB-first) into a [`DecodedFrame`].
///
/// Returns `None` if the CRC fails, the frame is unknown, or the
/// frame-family unpack fails.
pub fn unpack_frame(msg: &[u8; 87]) -> Option<DecodedFrame> {
    if !check_crc(msg) {
        return None;
    }

    let i3bit = (msg[72] << 2) | (msg[73] << 1) | msg[74];
    let mut words = [0u8; 12];
    for i in 0..12 {
        for j in 0..6 {
            words[i] = (words[i] << 1) | msg[i * 6 + j];
        }
    }
    let payload_bit0 = msg[0];

    if i3bit & JS8_DATA != 0 {
        let text = unpack_fast_data(&words)?;
        if text.is_empty() {
            return None;
        }
        let message = text.clone();
        return Some(DecodedFrame {
            kind: DecodedFrame::KIND_DATA_JSC,
            callsign: String::new(),
            to: None,
            grid: None,
            cmd: None,
            num: None,
            bits3: 0,
            is_alt: false,
            text,
            message,
        });
    }

    if payload_bit0 == 1 {
        let (kind, text) = unpack_legacy_data(&words)?;
        if text.is_empty() {
            return None;
        }
        let message = text.clone();
        return Some(DecodedFrame {
            kind,
            callsign: String::new(),
            to: None,
            grid: None,
            cmd: None,
            num: None,
            bits3: 0,
            is_alt: false,
            text,
            message,
        });
    }

    let typ = ((msg[0]) << 2) | (msg[1] << 1) | msg[2];

    match typ {
        FRAME_HEARTBEAT => {
            let (callsign, grid, is_alt, bits3) = unpack_heartbeat(&words)?;
            let mut message = format!("{callsign}: ");
            if is_alt {
                message.push_str(&CQS[(bits3 & 7) as usize]);
            } else {
                message.push_str(if bits3 == 0 { "HEARTBEAT" } else { "HB" });
            }
            if let Some(g) = &grid {
                message.push(' ');
                message.push_str(g);
            }
            Some(DecodedFrame {
                kind: FRAME_HEARTBEAT,
                callsign,
                to: None,
                grid,
                cmd: None,
                num: None,
                bits3: bits3 & 7,
                is_alt,
                text: String::new(),
                message: format!("{message} "),
            })
        }
        FRAME_COMPOUND | FRAME_COMPOUND_DIRECTED => {
            let (flag, callsign, extra) = unpack_compound(&words)?;
            let mut message = format!("{callsign}:");
            if let Some(e) = &extra {
                message.push(' ');
                message.push_str(e);
            }
            let grid = if flag == FRAME_COMPOUND {
                extra.clone()
            } else {
                None
            };
            let extra_str = extra.clone();
            let (cmd, num) = split_cmd_num(extra_str.as_deref());
            Some(DecodedFrame {
                kind: flag,
                callsign,
                to: None,
                grid,
                cmd,
                num,
                bits3: 0,
                is_alt: false,
                text: String::new(),
                message: format!("{message} "),
            })
        }
        FRAME_DIRECTED => {
            let (from, to, cmd, num) = unpack_directed(&words)?;
            let mut message = format!("{from} {to} {cmd}");
            if let Some(n) = &num {
                let disp = if cmd == "SNR" || cmd == "HEARTBEAT SNR" {
                    format_snr(*n as i32).unwrap_or_default()
                } else {
                    n.to_string()
                };
                message.push(' ');
                message.push_str(&disp);
            }
            Some(DecodedFrame {
                kind: FRAME_DIRECTED,
                callsign: from,
                to: Some(to),
                grid: None,
                cmd: Some(cmd),
                num,
                bits3: 0,
                is_alt: false,
                text: String::new(),
                message: format!("{message} "),
            })
        }
        _ => None,
    }
}

/// Split an extra string "SNR <n>" / "HEARTBEAT SNR <n>" / "GRID" etc.
/// into (cmd, num). The grid case returns `cmd = grid`, `num = None`.
fn split_cmd_num(extra: Option<&str>) -> (Option<String>, Option<i16>) {
    let Some(s) = extra else {
        return (None, None);
    };
    let t = s.trim();
    if let Some(rest) = t.strip_prefix("SNR ") {
        let n: Option<i16> = rest.parse().ok();
        return (Some("SNR".to_string()), n);
    }
    if let Some(rest) = t.strip_prefix("HEARTBEAT SNR ") {
        let n: Option<i16> = rest.parse().ok();
        return (Some("HEARTBEAT SNR".to_string()), n);
    }
    if t == "SNR" {
        return (Some("SNR".to_string()), None);
    }
    (Some(t.to_string()), None)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn msg_from_words(words: &[u8; 12], i3bit: u8) -> [u8; 87] {
        let mut msg = [0u8; 87];
        // payload: bits 0..71 (12 × 6-bit words, MSB-first)
        for b in 0..72usize {
            msg[b] = (words[b / 6] >> (5 - b % 6)) & 1;
        }
        // type/i3bit: bits 72..74
        msg[72] = (i3bit >> 2) & 1;
        msg[73] = (i3bit >> 1) & 1;
        msg[74] = i3bit & 1;
        // CRC-12: identical to encode_tones (payload in bytes[0..8], type in
        // byte9 high3, byte10 = 0), placed as 12 bits at positions 75..86.
        let mut bytes = [0u8; 11];
        for b in 0..72usize {
            if msg[b] == 1 {
                bytes[b / 8] |= 1 << (7 - b % 8);
            }
        }
        bytes[9] = i3bit << 5;
        let crc = crc12(&bytes) as u16;
        for i in 0..12 {
            msg[75 + i] = ((crc >> (11 - i)) & 1) as u8;
        }
        msg
    }

    #[test]
    fn test_alphanumeric50_roundtrip() {
        for cs in ["VE7/KN4CRD", "SP1ATOM", "N1MM", "A1"] {
            let packed = pack_alphanumeric50(cs);
            assert_eq!(unpack_alphanumeric50(packed), cs, "roundtrip {cs}");
        }
    }

    #[test]
    fn test_alphanumeric50_basecall() {
        // "@" is index 38, only valid in slot 0. unpack keeps the "@".
        let packed = pack_alphanumeric50("@ALLCALL");
        assert_ne!(packed, 0);
        assert_eq!(unpack_alphanumeric50(packed), "@ALLCALL");
    }

    #[test]
    fn test_pack_num() {
        assert_eq!(pack_num(""), None);
        assert_eq!(pack_num("0"), Some(31));
        assert_eq!(pack_num("-30"), Some(1));
        assert_eq!(pack_num("31"), Some(62));
        assert_eq!(pack_num("73"), Some(62));
        assert_eq!(pack_num("-5"), Some(26));
        assert_eq!(pack_num("+20"), Some(51));
    }

    #[test]
    fn test_cmd_index() {
        assert_eq!(cmd_index("GRID?"), Some(4));
        assert_eq!(cmd_index("SNR"), Some(25));
        assert_eq!(cmd_index("?"), Some(0));
        assert_eq!(cmd_index("SK"), Some(20));
        assert_eq!(cmd_index("UNKNOWN"), None);
    }

    #[test]
    fn test_grid_roundtrip() {
        // Only use grids whose center falls on a valid cell boundary.
        for grid in ["FN42", "EM55", "JN57"] {
            let packed = pack_grid(grid);
            assert!(packed <= NBASEGRID, "{grid} -> {packed}");
            assert_eq!(unpack_grid(packed), grid, "roundtrip {grid}");
        }
    }

    #[test]
    fn test_grid_out_of_range() {
        assert_eq!(unpack_grid(NBASEGRID + 1), "");
        assert_eq!(pack_grid("A"), NMAXGRID);
    }

    #[test]
    fn test_basecall() {
        let v = pack_callsign("@ALLCALL").unwrap();
        assert_eq!(v, NBASECALL + 2);
        assert_eq!(basecall_name(v), Some("@ALLCALL"));
        assert_eq!(unpack_callsign(v, false), "@ALLCALL");
    }

    #[test]
    fn test_callsign_roundtrip() {
        for cs in ["N1MM", "VE3NE", "DL1ABC"] {
            let packed = pack_callsign(cs);
            assert!(packed.is_some(), "pack {cs}");
            assert_eq!(
                unpack_callsign(packed.unwrap(), false),
                cs,
                "roundtrip {cs}"
            );
        }
    }

    #[test]
    fn test_portable_callsign() {
        let packed = pack_callsign("N1MM/P").unwrap();
        assert!(packed < NBASECALL);
        assert_eq!(unpack_callsign(packed, true), "N1MM/P");
    }

    #[test]
    fn test_unpack_cmd_snr() {
        let mut num = None;
        let v = pack_cmd(25, 51); // SNR +20 -> (2<<6)|51 = 179
        assert_eq!(v, (2u8 << 6) | 51);
        let idx = unpack_cmd(v, &mut num).unwrap();
        assert_eq!(idx, 25);
        assert_eq!(Some(20), num);
    }

    #[test]
    fn test_heartbeat_roundtrip() {
        let grid = pack_grid("FN42");
        // type=0, callsign "N1MM", num = grid (bit 15 clear -> HB, not CQ).
        let words = pack_compound_frame("N1MM", FRAME_HEARTBEAT, grid, 0).unwrap();
        let (cs, g, is_alt, bits3) = unpack_heartbeat(&words).unwrap();
        assert_eq!(cs, "N1MM");
        assert_eq!(g, Some("FN42".to_string()));
        assert!(!is_alt);
        assert_eq!(bits3, 0);
    }

    #[test]
    fn test_heartbeat_cq() {
        // bit 15 set -> CQ variant, bits3 = 3 -> "CQ CONTEST".
        let num = (pack_grid("EM55") as u16) | (1 << 15);
        let words = pack_compound_frame("K1ABC", FRAME_HEARTBEAT, num, 3).unwrap();
        let (cs, _g, is_alt, bits3) = unpack_heartbeat(&words).unwrap();
        assert_eq!(cs, "K1ABC");
        assert!(is_alt);
        assert_eq!(bits3, 3);
    }

    #[test]
    fn test_compound_grid_roundtrip() {
        let grid = pack_grid("JN57");
        let words = pack_compound_frame("DK7ZL", FRAME_COMPOUND, grid, 0).unwrap();
        let (flag, cs, extra) = unpack_compound(&words).unwrap();
        assert_eq!(flag, FRAME_COMPOUND);
        assert_eq!(cs, "DK7ZL");
        assert_eq!(extra, Some("JN57".to_string()));
    }

    #[test]
    fn test_compound_command() {
        let cmd = NUSERGRID + 4; // " GRID?"
        let words = pack_compound_frame("W1AW", FRAME_COMPOUND, cmd as u16, 0).unwrap();
        let (flag, cs, extra) = unpack_compound(&words).unwrap();
        assert_eq!(flag, FRAME_COMPOUND);
        assert_eq!(cs, "W1AW");
        assert_eq!(extra.as_deref(), Some("GRID?"));
    }

    #[test]
    fn test_directed_roundtrip() {
        let words = pack_directed("N1MM", "VE3NE", "SNR", Some(20)).unwrap();
        let (from, to, cmd, num) = unpack_directed(&words).unwrap();
        assert_eq!(from, "N1MM");
        assert_eq!(to, "VE3NE");
        assert_eq!(cmd, "SNR");
        assert_eq!(num, Some(20));
    }

    #[test]
    fn test_directed_no_num() {
        let words = pack_directed("N1MM", "VE3NE", "GRID?", None).unwrap();
        let (from, to, cmd, num) = unpack_directed(&words).unwrap();
        assert_eq!(from, "N1MM");
        assert_eq!(to, "VE3NE");
        assert_eq!(cmd, "GRID?");
        assert_eq!(num, None);
    }

    #[test]
    fn test_unpack_frame_heartbeat() {
        let grid = pack_grid("FN42");
        let words = pack_compound_frame("N1MM", FRAME_HEARTBEAT, grid, 0).unwrap();
        let msg = msg_from_words(&words, 0);
        let frame = unpack_frame(&msg).unwrap();
        assert_eq!(frame.kind, FRAME_HEARTBEAT);
        assert_eq!(frame.callsign, "N1MM");
        assert_eq!(frame.grid, Some("FN42".to_string()));
        assert_eq!(frame.message, "N1MM: HEARTBEAT FN42 ");
    }

    #[test]
    fn test_unpack_frame_compound_grid() {
        let grid = pack_grid("JN57");
        let words = pack_compound_frame("DK7ZL", FRAME_COMPOUND, grid, 0).unwrap();
        let msg = msg_from_words(&words, 0);
        let frame = unpack_frame(&msg).unwrap();
        assert_eq!(frame.kind, FRAME_COMPOUND);
        assert_eq!(frame.callsign, "DK7ZL");
        assert_eq!(frame.grid, Some("JN57".to_string()));
        assert_eq!(frame.message, "DK7ZL: JN57 ");
    }

    #[test]
    fn test_unpack_frame_directed() {
        let words = pack_directed("N1MM", "VE3NE", "SNR", Some(20)).unwrap();
        let msg = msg_from_words(&words, 0);
        let frame = unpack_frame(&msg).unwrap();
        assert_eq!(frame.kind, FRAME_DIRECTED);
        assert_eq!(frame.callsign, "N1MM");
        assert_eq!(frame.to, Some("VE3NE".to_string()));
        assert_eq!(frame.cmd.as_deref(), Some("SNR"));
        assert_eq!(frame.num, Some(20));
        assert_eq!(frame.message, "N1MM VE3NE SNR +20 ");
    }

    #[test]
    fn test_check_crc_bad() {
        let grid = pack_grid("FN42");
        let words = pack_compound_frame("N1MM", FRAME_HEARTBEAT, grid, 0).unwrap();
        let mut msg = msg_from_words(&words, 0);
        msg[0] ^= 1; // corrupt one payload bit
        assert!(!check_crc(&msg));
    }

    #[test]
    fn test_huff_decode_known() {
        // "HI" — H = 00011, I = 11100
        let bits: Vec<u8> = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0].to_vec();
        assert_eq!(huff_decode(&bits), "HI");
    }

    #[test]
    fn test_format_snr() {
        assert_eq!(format_snr(20), Some("+20".into()));
        assert_eq!(format_snr(-5), Some("-5".into()));
        assert_eq!(format_snr(70), None);
    }
}
