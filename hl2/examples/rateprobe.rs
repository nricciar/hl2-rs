//! Measure the true per-slot EP6 baseband I/Q rate, as a function of the
//! C1 SPEED bits (the "baseband rate" option: 48/96/192/384 kHz).
//!
//! Args (all optional):
//!   --speed <48|96|192|384>   baseband rate to request via C1 (default 192)
//!   --slot <N>                which RX slot to measure (default 1 → C0 0x08)
//!   --seconds <S>             measurement window (default 6)
//!
//! Sends keep-alives with the requested C1 = CONFIG_BOTH | SPEED throughout
//! the window (to avoid watchdog reset), then reports the per-slot complex-
//! sample rate (chunks/s × 63 complex samples/chunk, one active receiver).
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, Instant};

const P: u16 = 1024;
const CONFIG_BOTH: u8 = 0x60;

fn speed_bits(khz: u32) -> u8 {
    match khz {
        48 => 0x00,
        96 => 0x01,
        192 => 0x02,
        384 => 0x03,
        _ => 0x02,
    }
}

async fn keepalive(sock: &tokio::net::UdpSocket, dst: SocketAddr, c1: u8) {
    let mut ka = [0u8; 1032];
    ka[0] = 0xEF;
    ka[1] = 0xFE;
    ka[2] = 0x01;
    ka[3] = 0x02;
    for off in [8usize, 8 + 512] {
        ka[off] = 0x7F;
        ka[off + 1] = 0x7F;
        ka[off + 2] = 0x7F;
        ka[off + 3] = 0x00;
        ka[off + 4] = c1;
    }
    let _ = sock.send_to(&ka, dst).await;
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut speed_khz = 192u32;
    let mut slot = 1u8;
    let mut window = 6.0f64;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--speed" => {
                i += 1;
                speed_khz = args[i].parse().ok().unwrap_or(192);
            }
            "--slot" => {
                i += 1;
                slot = args[i].parse().ok().unwrap_or(1);
            }
            "--seconds" => {
                i += 1;
                window = args[i].parse().ok().unwrap_or(6.0);
            }
            _ => {}
        }
        i += 1;
    }
    let c1 = CONFIG_BOTH | speed_bits(speed_khz);

    let addr: IpAddr = "192.168.1.68".parse().unwrap();
    let sock = tokio::net::UdpSocket::bind("0.0.0.0:0").await.unwrap();
    let dst = SocketAddr::new(addr, P);

    // STOP + drain
    let _ = sock.send_to(&[0xEF, 0xFE, 0x04, 0x00], dst).await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    let mut d = [0u8; 2048];
    while sock.try_recv_from(&mut d).is_ok() {}

    // START (wideband)
    let _ = sock.send_to(&[0xEF, 0xFE, 0x04, 0x03], dst).await;
    // Keep the radio alive AND set the C1 SPEED bit we want to measure.
    for _ in 0..10 {
        keepalive(&sock, dst, c1).await;
        let _ = sock.try_recv_from(&mut d);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // NCO tune RX1
    let freq = 7_049_000u32;
    let be = freq.to_be_bytes();
    let mut ch = [0u8; 512];
    ch[0] = 0x7F;
    ch[1] = 0x7F;
    ch[2] = 0x7F;
    ch[3] = 0x06;
    ch[4..8].copy_from_slice(&be);
    let mut pkt = [0u8; 1032];
    pkt[0] = 0xEF;
    pkt[1] = 0xFE;
    pkt[2] = 0x01;
    pkt[3] = 0x02;
    pkt[8..520].copy_from_slice(&ch);
    pkt[520..1032].copy_from_slice(&ch);
    let _ = sock.send_to(&pkt, dst).await;
    for _ in 0..5 {
        keepalive(&sock, dst, c1).await;
        let _ = sock.try_recv_from(&mut d);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let duration = Duration::from_secs_f64(window);
    let t0 = Instant::now();
    let mut ep6_frames = 0u64;
    let mut slot_hist: HashMap<u8, u64> = HashMap::new();
    let mut fbuf = [0u8; 1032];
    let mut since_keepalive = 0u64;
    // Tight drain loop (no sleeps): the UDP receive buffer drops datagrams the
    // moment we stop reading fast enough, so we must read as fast as possible
    // and only send a keep-alive ~every 40 frames (well under the ~168 ms
    // watchdog) to keep the radio from resetting.
    while t0.elapsed() < duration {
        if let Ok((len, _)) = sock.try_recv_from(&mut fbuf) {
            since_keepalive += 1;
            if len == 1032 && fbuf[3] == 0x06 {
                ep6_frames += 1;
                for off in [8usize, 8 + 512] {
                    if fbuf[off] == 0x7F && fbuf[off + 1] == 0x7F && fbuf[off + 2] == 0x7F {
                        *slot_hist.entry(fbuf[off + 3]).or_insert(0) += 1;
                    }
                }
            }
        } else {
            // No datagram immediately available: brief yield.
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        if since_keepalive >= 40 {
            since_keepalive = 0;
            keepalive(&sock, dst, c1).await;
        }
    }
    let secs = window;
    let target_c0 = 0x08 + (slot.saturating_sub(1)) * 8;
    let measured = slot_hist.get(&target_c0).copied().unwrap_or(0);
    let mut v: Vec<(u8, u64)> = slot_hist.iter().map(|(k, val)| (*k, *val)).collect();
    v.sort();
    let label = v
        .iter()
        .map(|(k, val)| format!("0x{:02x}={}", *k, val))
        .collect::<Vec<_>>()
        .join("  ");
    println!("\n== C1 SPEED {} (0x{:02X}) ==", speed_khz, c1);
    println!("EP6 frames/s: {:.1}", ep6_frames as f64 / secs);
    println!("chunks/{:.0}s: {}", secs, label);
    println!(
        "RX{} (C0=0x{:02x}): {:.1} chunks/s  =>  {:.0} complex samples/s",
        slot,
        target_c0,
        measured as f64 / secs,
        (measured as f64) * 63.0 / secs
    );
    let _ = sock.send_to(&[0xEF, 0xFE, 0x04, 0x00], dst).await;
}
