//! EP6 baseband live probe with the new 24-bit/N-receiver parser.
//!
//! Usage: ep6probe [--seconds N] [--freq HZ] [--lna DB]
//!   defaults: 6 s, 7_074_000 Hz, +30 dB LNA, filtermask 0x4.
//!
//! Captures EP6 frames, parses them as N=1 (single active receiver) and as
//! N=2 (two active receivers), and reports — for each hypothesis:
//!   * first 24 payload bytes (hex)
//!   * per-receiver I/Q DC offset and DC-removed RMS
//!   * top-5 bins of a 32768-point DFT over a rolling buffer (expect a
//!     narrow spike > 30 dB above the floor if we are pointed at a tone)
//!
//! Whichever N gives a narrow spectral spike + a sane (sub-full-scale) RMS
//! is the true active-receiver count.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use hl2::protocol::data::{BASEBAND_HEADER_OFFSET, parse_baseband_chunk, sample_24be};
use num_complex::Complex;

const P: u16 = 1024;
const CONFIG_BOTH: u8 = 0x60;
const CHUNK: usize = 512;

fn speed_bits(khz: u32) -> u8 {
    match khz {
        48 => 0x00,
        96 => 0x01,
        192 => 0x02,
        _ => 0x03,
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut seconds = 6.0f64;
    let mut freq = 7_074_000u32;
    let mut lna_db: i8 = 30;
    let mut rate_khz: u32 = 192;
    let mut c0: u8 = 0x04;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seconds" => {
                i += 1;
                seconds = args[i].parse().ok().unwrap_or(6.0);
            }
            "--freq" => {
                i += 1;
                freq = args[i].parse().ok().unwrap_or(7_074_000);
            }
            "--lna" => {
                i += 1;
                lna_db = args[i].parse().ok().unwrap_or(30);
            }
            "--rate" => {
                i += 1;
                rate_khz = args[i].parse().ok().unwrap_or(192);
            }
            "--c0" => {
                i += 1;
                c0 = u8::from_str_radix(
                    args[i].trim_start_matches("0x").trim_start_matches("0X"),
                    16,
                )
                .unwrap_or(0x04);
            }
            _ => {}
        }
        i += 1;
    }

    let addr: IpAddr = "192.168.1.68".parse().unwrap();
    let dst = SocketAddr::new(addr, P);
    let sock = tokio::net::UdpSocket::bind("0.0.0.0:0").await.unwrap();
    let sock = &sock;

    async fn ka(sock: &tokio::net::UdpSocket, dst: SocketAddr, c1: u8) {
        let mut ka = [0u8; 1032];
        ka[0] = 0xEF;
        ka[1] = 0xFE;
        ka[2] = 0x01;
        ka[3] = 0x02;
        ka[4..8].copy_from_slice(&0u32.to_be_bytes());
        for off in [8usize, 8 + 512] {
            ka[off] = 0x7F;
            ka[off + 1] = 0x7F;
            ka[off + 2] = 0x7F;
            ka[off + 3] = 0x00;
            ka[off + 4] = c1;
            ka[off + 5] = 0x04 << 1; // filtermask 0x4 → relay 3
        }
        let _ = sock.send_to(&ka, dst).await;
    }

    // STOP + drain
    let mut stop = [0u8; 1032];
    stop[0] = 0xEF;
    stop[1] = 0xFE;
    stop[2] = 0x04;
    stop[3] = 0x00;
    let _ = sock.send_to(&stop, dst).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut d = [0u8; 2048];
    while sock.try_recv_from(&mut d).is_ok() {}

    // START
    let mut start = [0u8; 1032];
    start[0] = 0xEF;
    start[1] = 0xFE;
    start[2] = 0x04;
    start[3] = 0x01;
    let _ = sock.send_to(&start, dst).await;

    let c1 = CONFIG_BOTH | speed_bits(rate_khz);
    for _ in 0..10 {
        ka(sock, dst, c1).await;
        let _ = sock.try_recv_from(&mut d);
        tokio::time::sleep(Duration::from_millis(40)).await;
    }

    // LNA gain: C0=0x14, C4 = 0x40 | ((lna+12)&0x3F)
    let lna_byte = 0x40u8 | (((lna_db as u8).wrapping_add(12)) & 0x3F);
    {
        let mut ch = [0u8; 512];
        ch[0] = 0x7F;
        ch[1] = 0x7F;
        ch[2] = 0x7F;
        ch[3] = 0x14;
        ch[7] = lna_byte;
        let mut pkt = [0u8; 1032];
        pkt[0] = 0xEF;
        pkt[1] = 0xFE;
        pkt[2] = 0x01;
        pkt[3] = 0x02;
        pkt[8..8 + 512].copy_from_slice(&ch);
        pkt[520..1032].copy_from_slice(&ch);
        let _ = sock.send_to(&pkt, dst).await;
    }

    // NCO write: C0 = (0x01 + slot) << 1 (0x04 = RX1, 0x06 = RX2, …)
    {
        let be = freq.to_be_bytes();
        let mut ch = [0u8; 512];
        ch[0] = 0x7F;
        ch[1] = 0x7F;
        ch[2] = 0x7F;
        ch[3] = c0;
        ch[4..8].copy_from_slice(&be);
        let mut pkt = [0u8; 1032];
        pkt[0] = 0xEF;
        pkt[1] = 0xFE;
        pkt[2] = 0x01;
        pkt[3] = 0x02;
        pkt[8..8 + 512].copy_from_slice(&ch);
        pkt[520..1032].copy_from_slice(&ch);
        let _ = sock.send_to(&pkt, dst).await;
    }

    for _ in 0..5 {
        ka(sock, dst, c1).await;
        tokio::time::sleep(Duration::from_millis(40)).await;
    }

    // Capture window
    let window = Duration::from_secs_f64(seconds);
    let t0 = std::time::Instant::now();
    let mut ep6_frames: u64 = 0;
    let mut valid_chunks: u64 = 0;
    let mut first_hex: Option<Vec<u8>> = None;
    let mut c0_hist: HashMap<u8, u64> = HashMap::new();
    let mut per_n_bufs: HashMap<u8, Vec<Complex<f32>>> = HashMap::new();
    let cap = (rate_khz as usize / 2)
        .saturating_mul((seconds * 1.5) as usize)
        .max(32768);
    let mut since_ka = 0u64;
    println!("freq={freq} lna={lna_db:+} rate={rate_khz}k cap={cap}");

    while t0.elapsed() < window {
        if let Ok((len, _)) = sock.try_recv_from(&mut d) {
            since_ka += 1;
            if len == 1032 && d[3] == 0x06 && d[0] == 0xEF && d[1] == 0xFE {
                ep6_frames += 1;
                for off in [8usize, 8 + 512] {
                    let chunk0 = &d[off..off + CHUNK];
                    if chunk0[0] == 0x7F && chunk0[1] == 0x7F && chunk0[2] == 0x7F {
                        valid_chunks += 1;
                        *c0_hist.entry(chunk0[3]).or_insert(0) += 1;
                        if first_hex.is_none() {
                            first_hex = Some(d[8..8 + 32].to_vec());
                        }
                        for n in [1u8, 2u8] {
                            if let Some(bc) = parse_baseband_chunk(chunk0.try_into().unwrap(), n) {
                                const CAP: usize = 32768;
                                // Track BOTH receivers' streams so we can see which DDC latched the NCO.
                                for slot in 0..bc.per_rx.len().min(2) {
                                    let entry = per_n_bufs.entry(n * 10 + slot as u8).or_default();
                                    entry.extend(bc.per_rx[slot].iter().copied());
                                    if entry.len() > CAP {
                                        let k = entry.len() - CAP;
                                        entry.drain(..k);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            tokio::time::sleep(Duration::from_millis(1)).await;
            if since_ka >= 40 {
                since_ka = 0;
                ka(sock, dst, c1).await;
            }
        }
    }

    let _ = sock.send_to(&stop, dst).await;

    let secs = seconds;
    println!(
        "\nEP6 frames during window: {ep6_frames} (approx {:.1}/s)",
        ep6_frames as f64 / secs
    );
    let mut hist: Vec<(u8, u64)> = c0_hist.iter().map(|(k, v)| (*k, *v)).collect();
    hist.sort();
    let h = hist
        .iter()
        .map(|(k, v)| format!("0x{:02x}={v}", *k))
        .collect::<Vec<_>>()
        .join(" ");
    println!("C0 histogram (must be status bytes, any value legal): {h}");
    if let Some(hx) = &first_hex {
        let hex: Vec<String> = hx.iter().map(|b| format!("{b:02x}")).collect();
        let mut groups: Vec<String> = Vec::new();
        for i in 0..hex.len() / 16 {
            groups.push(hex[i * 16..i * 16 + 16].join(" "));
        }
        println!("first 32 payload bytes:\n  {}", groups.join("\n  "));
    }

    // Per-N analysis.
    const FS_EXPECT: f64 = 96_000.0; // rate/2 complex
    for composite in [10u8, 20u8, 21u8] {
        let n = composite / 10;
        let slot = composite % 10;
        let buf = match per_n_bufs.get(&composite) {
            Some(b) if !b.is_empty() => b,
            _ => continue,
        };
        let len = buf.len();
        // DC + DC-removed RMS
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for x in buf.iter() {
            re += x.re as f64;
            im += x.im as f64;
        }
        let re_dc = re / len as f64;
        let im_dc = im / len as f64;
        let mut re2 = 0.0f64;
        let mut im2 = 0.0f64;
        for x in buf.iter() {
            let dr = x.re as f64 - re_dc;
            let di = x.im as f64 - im_dc;
            re2 += dr * dr;
            im2 += di * di;
        }
        let rms = ((re2 + im2) / 2.0 / (len as f64)).sqrt();
        println!(
            "  [N={}, slot {}] DC={re_dc:+.4e}+j{im_dc:+.4e}  (|DC|={:.3e}, {:.1} dBFS   RMS(after DC)={:.3e}, {:.1} dBFS)",
            n,
            slot,
            (re_dc * re_dc + im_dc * im_dc).sqrt(),
            20.0 * ((re_dc * re_dc + im_dc * im_dc).sqrt()).log10().max(-150.0),
            rms,
            20.0 * rms.log10().max(-150.0)
        );
        // 4096-point DFT over the tail of the buffer (fast enough, coarse
        // bins to identify spike frequency).
        let fft_n = 4096usize;
        let tail = len.saturating_sub(fft_n).max(0);
        let window_len = len - tail;
        // Goertzel-less direct DFT at log-spaced bins is heavy at 4096 x 4096;
        // instead compute 512 candidate bins (every 8th) which still has 192
        // Hz resolution at 96k — enough to separate a narrow tone.
        let mut best: (f64, i32) = (0.0, 0); // (mag, bin_index)
        let mut mags: Vec<(i32, f64)> = Vec::with_capacity(512);
        let step = (window_len as f64 / 512.0).max(1.0);
        for b in 0..512i32 {
            let w = 2.0 * std::f64::consts::PI * (b as f64 * step) / window_len as f64;
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (i, x) in buf[tail..tail + window_len].iter().enumerate() {
                let phi = w * i as f64;
                re += (x.re as f64) * phi.cos() - (x.im as f64) * phi.sin();
                im += (x.re as f64) * phi.sin() + (x.im as f64) * phi.cos();
            }
            let m = (re * re + im * im).sqrt();
            mags.push((b, m));
            if m > best.0 {
                best = (m, b);
            }
        }
        mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top5: Vec<(i32, f64)> = mags[..5].to_vec();
        let max_m = top5[0].1;
        let avg_floor = (mags.iter().map(|(_, m)| *m).sum::<f64>() / mags.len() as f64);
        let ratio_db = if avg_floor > 0.0 && max_m > 0.0 {
            (max_m / avg_floor).log10() * 20.0
        } else {
            f64::INFINITY
        };
        let measured_rate = if valid_chunks > 0 {
            // total RX1 pairs: both 512-byte chunks per frame are baseband.
            let per_chunk = if n == 1 { 63 } else { 36 };
            (valid_chunks as f64 * per_chunk as f64) / secs
        } else {
            0.0
        };
        println!("\n=== hypothesis N={n} ===");
        println!("samples captured: {len}");
        println!("measured rate: {measured_rate:.0} Sps (expect {FS_EXPECT:.0} if N is correct)");
        println!("DC offset: I = {re_dc:+.6e}  Q = {im_dc:+.6e}");
        println!(
            "RMS (DC-removed): {rms:.6}  ({:.1} dBFS)",
            20.0 * rms.log10()
        );
        let top: Vec<String> = top5
            .iter()
            .map(|(b, m)| {
                let f_khz = *b as f64 * step / window_len as f64 * (FS_EXPECT) / 4.0;
                format!(
                    "{:6.3} kHz  {:.2e} ({:+.1} dB vs mean)",
                    f_khz,
                    *m,
                    (m / avg_floor).log10() * 20.0
                )
            })
            .collect();
        println!("top-5 bins:\n  {}", top.join("\n  "));
        println!("peak:mean ratio: {ratio_db:.1} dB");
        let _ = best;
        // The 24-bit helper sanity: parse one raw chunk and print first two.
        let _ = sample_24be(&[0x7F, 0x00, 0x00]);
        let _ = BASEBAND_HEADER_OFFSET;
    }
}
