//! Tuning validation: for each given DDC Hz, capture ~1s of EP6 per_rx[0],
//! Hann-window, 32768-pt FFT, and report the strongest bins in kHz.
//! Compare the *relative* bin frequencies across the runs: if the LO is
//! actually moving, a real RF line should appear at a *different* offset
//! from DC in each run. If the LO is stuck, the same bin repeats.

use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use hl2::protocol::data::{BASEBAND_HEADER_OFFSET, parse_baseband_chunk};
use num_complex::Complex;

const P: u16 = 1024;
const CHUNK: usize = 512;

mod rfft {
    pub fn fft(x: &mut [num_complex::Complex<f64>]) {
        use num_complex::Complex;
        // Radix-2 Cooley-Tukey.
        let n = x.len();
        assert!(n.is_power_of_two() && n > 1);
        // bit-reversal
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
            let wl = (ang).cos();
            let wr = (ang).sin();
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
    let mut freqs: Vec<u32> = vec![];
    let mut lna_db = 30i8;
    let mut rate_khz = 192u32;
    let mut secs = 1.5f64;
    let mut c0: u8 = 0x04;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--freqs" => {
                for tok in args[i + 1].split(',') {
                    let s = tok.trim().replace('_', "");
                    freqs.push(s.parse().unwrap());
                }
                i += 1;
            }
            "--lna" => {
                lna_db = args[i + 1].parse().ok().unwrap();
                i += 1;
            }
            "--rate" => {
                rate_khz = args[i + 1].parse().ok().unwrap();
                i += 1;
            }
            "--seconds" => {
                secs = args[i + 1].parse().ok().unwrap();
                i += 1;
            }
            "--c0" => {
                c0 = u8::from_str_radix(args[i + 1].trim_start_matches("0x"), 16).unwrap();
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
            48 => 0x00,
            96 => 0x01,
            192 => 0x02,
            _ => 0x03,
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

    // Reset and start.
    do_stop(sock, dst).await;
    do_start(sock, dst).await;
    let c1 = 0x60u8 | speed_bits(rate_khz);
    for _ in 0..10 {
        ka(sock, dst, c1).await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        let mut d = [0u8; 2048];
        while sock.try_recv_from(&mut d).is_ok() {}
    }
    lna(sock, dst, lna_db).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let fs = (rate_khz as f64);
    let N = 1 << 15; // 32768 samples.
    let mut hann = vec![0.0f64; N];
    for (i, v) in hann.iter_mut().enumerate() {
        *v = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * (i as f64) / (N as f64)).cos());
    }

    let mut d = [0u8; 2048];
    for (i, &hz) in freqs.iter().enumerate() {
        eprintln!("[{}/{}] c0=0x{c0:02x} nco={} Hz", i + 1, freqs.len(), hz);
        nco(sock, dst, c0, hz).await;
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Warm up: drain + keepalive, then capture `secs` worth of per_rx[0].
        let mut buf: Vec<Complex<f64>> = Vec::with_capacity(N);
        let t0 = std::time::Instant::now();
        let mut since_ka = 0u64;
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
                if since_ka >= 30 {
                    since_ka = 0;
                    ka(sock, dst, c1).await;
                }
            }
        }

        // FFT the last N.
        let mut x = Vec::with_capacity(N);
        for (k, c) in buf.iter().enumerate().take(N) {
            x.push(Complex::new(c.re as f64, c.im as f64) * hann[k % N]);
        }
        rfft::fft(&mut x);

        // Top-6 non-DC bins.
        let mut pk: Vec<(usize, f64)> = (0..N)
            .map(|k| (k, (x[k].re * x[k].re + x[k].im * x[k].im).sqrt()))
            .collect();
        pk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let dc = pk
            .iter()
            .find(|(k, _)| *k < 3)
            .map(|(_, m)| *m)
            .unwrap_or(0.0);
        // Report top non-DC and the DC separately.
        let top: Vec<String> = pk
            .iter()
            .filter(|(k, _)| *k >= 4)
            .take(6)
            .map(|(k, m)| {
                // bin k corresponds to k*fs/N Hz on [0, fs/N).
                // Mirror k > N/2 to negative side.
                let off_hz = if *k <= N / 2 {
                    *k as f64 * fs / N as f64
                } else {
                    (*k as f64 - N as f64) * fs / N as f64
                };
                let ratio_db = 10.0 * (m / dc.max(1e-30)).log10();
                format!("{:+9.3} kHz  ({ratio_db:.1} dB vs DC)", off_hz / 1000.0)
            })
            .collect();
        println!(
            "\n--- DDC @ {} Hz  (fs={} kHz, sample rate complex = {}) ---",
            hz,
            (fs / 2.0).round() as u32,
            (fs / 2.0).round() as u32
        );
        println!("  DC: {:.3e}  ({:.1} dBFS)", dc, 20.0 * dc.log10());
        for t in top {
            println!("  {t}");
        }
    }

    do_stop(sock, dst).await;
}
