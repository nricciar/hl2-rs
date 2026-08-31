//! Tuning-sweep diagnostic: for each DDC Hz, capture ~2 s of EP6 per_rx[0],
//! Hann-window, 32768-pt FFT, and report:
//!   * the *bin offset* of the strongest bin vs DC (in ±kHz; positive = above center)
//!   * total RMS after DC removal
//! Tune LO in 1 MHz steps from 3 MHz to 30 MHz. If the DDC LO is moving, the
//! strongest-bin offset should *drift* across the sweep (real RF lines shift
//! into/out of the ±48 kHz window as we retune). If the LO is stuck, the
//! same bins repeat.

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use hl2::protocol::data::parse_baseband_chunk;
use num_complex::Complex;

const P: u16 = 1024;
const CHUNK: usize = 512;

mod rfft {
    pub fn fft(x: &mut [num_complex::Complex<f64>]) {
        use num_complex::Complex;
        let n = x.len();
        assert!(n.is_power_of_two() && n > 1);
        let mut i = 1usize;
        let mut j = 0usize;
        while i < n {
            let mut m = n >> 1;
            while j & m != 0 {
                j ^= m;
                m >>= 1;
            }
            j ^= m;
            if i < j {
                x.swap(i, j);
            }
            i += 1;
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let ang = -2.0 * std::f64::consts::PI / len as f64;
            let wl = ang.cos();
            let wr = ang.sin();
            let mut i = 0usize;
            while i < n {
                let mut c = Complex::new(1.0, 0.0f64);
                let mut j = i;
                while j < i + half {
                    let t = c * x[j + half];
                    x[j + half] = x[j] - t;
                    x[j] += t;
                    c = Complex::new(c.re * wl - c.im * wr, c.re * wr + c.im * wl);
                    j += 1;
                }
                i += len;
            }
            len *= 2;
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut start_hz = 3_000_000u32;
    let mut end_hz = 30_000_000u32;
    let mut step_hz = 1_000_000u32;
    let mut lna_db = 30i8;
    let mut secs = 2.0f64;
    let mut c0: u8 = 0x04;
    let mut rate_khz = 192u32;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--start" => {
                start_hz = args[i + 1].parse().unwrap();
                i += 1;
            }
            "--end" => {
                end_hz = args[i + 1].parse().unwrap();
                i += 1;
            }
            "--step" => {
                step_hz = args[i + 1].parse().unwrap();
                i += 1;
            }
            "--lna" => {
                lna_db = args[i + 1].parse().unwrap();
                i += 1;
            }
            "--seconds" => {
                secs = args[i + 1].parse().unwrap();
                i += 1;
            }
            "--c0" => {
                c0 = u8::from_str_radix(args[i + 1].trim_start_matches("0x"), 16).unwrap();
                i += 1;
            }
            "--rate" => {
                rate_khz = args[i + 1].parse().unwrap();
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    let addr: IpAddr = "192.168.1.68".parse().unwrap();
    let dst = SocketAddr::new(addr, P);
    let sock = tokio::net::UdpSocket::bind("0.0.0.0:0").await.unwrap();
    let sock = &sock;

    fn speed_bits(khz: u32) -> u8 {
        match khz {
            48 => 0,
            96 => 1,
            192 => 2,
            _ => 3,
        }
    }

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
            ka[off + 5] = 0x04 << 1;
        }
        let _ = sock.send_to(&ka, dst).await;
    }

    async fn do_stop(sock: &tokio::net::UdpSocket, dst: SocketAddr) {
        let mut s = [0u8; 64];
        s[0] = 0xEF;
        s[1] = 0xFE;
        s[2] = 0x04;
        s[3] = 0x00;
        let _ = sock.send_to(&s, dst).await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        let mut d = [0u8; 2048];
        while sock.try_recv_from(&mut d).is_ok() {}
    }

    async fn do_start(sock: &tokio::net::UdpSocket, dst: SocketAddr) {
        let mut s = [0u8; 64];
        s[0] = 0xEF;
        s[1] = 0xFE;
        s[2] = 0x04;
        s[3] = 0x01;
        let _ = sock.send_to(&s, dst).await;
    }

    async fn lna(sock: &tokio::net::UdpSocket, dst: SocketAddr, db: i8) {
        let b = 0x40u8 | ((db as u8).wrapping_add(12) & 0x3F);
        let mut ch = [0u8; 512];
        ch[0] = 0x7F;
        ch[1] = 0x7F;
        ch[2] = 0x7F;
        ch[3] = 0x14;
        ch[7] = b;
        let mut p = [0u8; 1032];
        p[0] = 0xEF;
        p[1] = 0xFE;
        p[2] = 0x01;
        p[3] = 0x02;
        p[8..8 + 512].copy_from_slice(&ch);
        p[520..1032].copy_from_slice(&ch);
        let _ = sock.send_to(&p, dst).await;
    }

    async fn nco(sock: &tokio::net::UdpSocket, dst: SocketAddr, c0: u8, hz: u32) {
        let be = hz.to_be_bytes();
        let mut ch = [0u8; 512];
        ch[0] = 0x7F;
        ch[1] = 0x7F;
        ch[2] = 0x7F;
        ch[3] = c0;
        ch[4..8].copy_from_slice(&be);
        let mut p = [0u8; 1032];
        p[0] = 0xEF;
        p[1] = 0xFE;
        p[2] = 0x01;
        p[3] = 0x02;
        p[8..8 + 512].copy_from_slice(&ch);
        p[520..1032].copy_from_slice(&ch);
        let _ = sock.send_to(&p, dst).await;
    }

    do_stop(sock, dst).await;
    do_start(sock, dst).await;
    let c1 = 0x60u8 | speed_bits(rate_khz);
    for _ in 0..12 {
        ka(sock, dst, c1).await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        let mut d = [0u8; 2048];
        while sock.try_recv_from(&mut d).is_ok() {}
    }
    lna(sock, dst, lna_db).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let fs = rate_khz as f64;
    let N = 1usize << 15;
    let mut hann = vec![0.0f64; N];
    for (i, v) in hann.iter_mut().enumerate() {
        *v = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / N as f64).cos());
    }

    let mut d = [0u8; 2048];
    let mut hz = start_hz;
    let mut rows: Vec<(u32, f64, f64)> = Vec::new();
    while hz <= end_hz {
        nco(sock, dst, c0, hz).await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        let mut buf: Vec<Complex<f64>> = Vec::with_capacity(N);
        let mut since_ka = 0u64;
        let t0 = std::time::Instant::now();
        while t0.elapsed() < Duration::from_secs_f64(secs) {
            if let Ok((len, _)) = sock.try_recv_from(&mut d) {
                since_ka = 0;
                if len == 1032 && d[3] == 0x06 {
                    for off in [8usize, 8 + 512] {
                        let chunk = &d[off..off + CHUNK];
                        if chunk[0] == 0x7F && chunk[1] == 0x7F && chunk[2] == 0x7F {
                            if let Some(bc) = parse_baseband_chunk(chunk.try_into().unwrap(), 1) {
                                buf.extend(
                                    bc.per_rx[0]
                                        .iter()
                                        .map(|c| Complex::new(c.re as f64, c.im as f64)),
                                );
                                if buf.len() > N {
                                    let k = buf.len() - N;
                                    buf.drain(..k);
                                }
                            }
                        }
                    }
                }
            } else {
                tokio::time::sleep(Duration::from_millis(1)).await;
                since_ka += 1;
                if since_ka >= 40 {
                    since_ka = 0;
                    ka(sock, dst, c1).await;
                }
            }
        }

        let mut x: Vec<Complex<f64>> = buf
            .iter()
            .enumerate()
            .take(N)
            .map(|(i, c)| {
                let h = hann[i % N];
                Complex::new(c.re * h, c.im * h)
            })
            .collect();
        rfft::fft(&mut x);
        let mut pk: Vec<(usize, f64)> = (0..N)
            .map(|k| (k, (x[k].re * x[k].re + x[k].im * x[k].im).sqrt()))
            .collect();
        pk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (top_k, top_m) = pk[0];
        let dc = pk
            .iter()
            .find(|(k, _)| *k < 3)
            .map(|(_, m)| *m)
            .unwrap_or(0.0);
        // Strongest non-DC.
        let (top_nd_k, top_nd_m) = pk
            .iter()
            .find(|(k, _)| *k >= 4)
            .cloned()
            .unwrap_or((0, 0.0));
        let off_hz = if top_nd_k <= N / 2 {
            top_nd_k as f64 * fs / N as f64
        } else {
            (top_nd_k as f64 - N as f64) * fs / N as f64
        };
        let rms_dbfs = 20.0 * ((top_nd_m / N as f64).max(1e-15)).log10();
        let top_vs_dc_db = 20.0 * ((top_nd_m / dc.max(1e-30)).max(1e-15)).log10();
        println!(
            "LO={hz:>9} Hz   top-nonDC={off_hz:+.3} kHz   mag={top_nd_m:.3e}  ({top_vs_dc_db:+.1} dB vs DC,  {rms_dbfs:+.1} dBFS/norm)"
        );
        rows.push((hz, off_hz, top_nd_m));
        hz += step_hz;
    }
    do_stop(sock, dst).await;
    let _ = rows;
    let _ = N;
}
