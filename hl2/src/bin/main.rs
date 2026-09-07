use tokio::signal::unix::{SignalKind, signal};

use std::sync::atomic::Ordering;

#[cfg(feature = "alsa")]
use hl2::receiver::AlsaSink;
#[cfg(not(feature = "alsa"))]
use hl2::receiver::VecSink;
use hl2::receiver::{
    AudioConfig, DropSink, Ft8Tap, Mode, RawSampleTap, ReceiverConfig, Sideband, VirtualReceiver,
    decode_closed_slot, shared,
};
use hl2::{Hl2, Hl2Event, RX1_ADDR, discover, discover_single};
use std::time::{Duration, Instant};

const HL2_ADDR: &str = "169.254.19.221";
const DEFAULT_FREQ: u64 = 14_200_000;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");

    match cmd {
        "discover" => {
            eprintln!("Discovering HL2 devices...");
            let devices = discover().await;

            if devices.is_empty() {
                eprintln!("No HL2 devices found.");
            } else {
                eprintln!("Found {} device(s):", devices.len());
                for (i, (addr, info)) in devices.iter().enumerate() {
                    eprintln!(
                        "  [{}] {} - Board ID: 0x{:02x}, Gateware: {}.{}",
                        i, addr, info.board_id, info.gateware_major, info.gateware_minor,
                    );
                    eprintln!(
                        "      MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                        info.mac[0],
                        info.mac[1],
                        info.mac[2],
                        info.mac[3],
                        info.mac[4],
                        info.mac[5],
                    );
                    eprintln!(
                        "      RX count: {}, sample format: {}",
                        info.rx_count,
                        if info.sample_16bit {
                            "16-bit"
                        } else {
                            "12-bit"
                        }
                    );
                    eprintln!(
                        "      IP: {}.{}.{}.{}",
                        info.ip[0], info.ip[1], info.ip[2], info.ip[3]
                    );
                }
            }
        }

        "probe" => {
            let hl2_addr: std::net::IpAddr = args
                .get(1)
                .and_then(|s| s.parse().ok())
                .expect("Usage: hl2 probe <IP>");

            eprintln!("Probing HL2 at {}...", hl2_addr);
            let result = discover_single(hl2_addr).await;

            if let Some((addr, info)) = result {
                eprintln!("Found HL2 at {}:", addr);
                eprintln!(
                    "  Board ID: 0x{:02x}, Gateware: {}.{}",
                    info.board_id, info.gateware_major, info.gateware_minor
                );
                eprintln!(
                    "  RX count: {}, sample format: {}",
                    info.rx_count,
                    if info.sample_16bit {
                        "16-bit"
                    } else {
                        "12-bit"
                    }
                );
                eprintln!(
                    "  IP: {}.{}.{}.{}",
                    info.ip[0], info.ip[1], info.ip[2], info.ip[3]
                );
            } else {
                eprintln!("No HL2 found at {}.", hl2_addr);
            }
        }

        "start" => {
            let hl2_addr: std::net::IpAddr = args
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| HL2_ADDR.parse().unwrap());

            let freq = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_FREQ);

            eprintln!("Connecting to HL2 at {}", hl2_addr);

            let (hl2, rx_ev, info) = match Hl2::start(hl2_addr).await {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Failed to start HL2: {}", e);
                    return;
                }
            };

            eprintln!(
                "HL2 started (sample_format={:?}, local_port={}, rx_count={})",
                info.sample_format, info.local_port, info.rx_count
            );

            if freq > 0 {
                eprintln!("Setting RX1 frequency to {} Hz", freq);
                if let Err(e) = hl2.tune(RX1_ADDR, freq as u32).await {
                    eprintln!("Tune failed: {}", e);
                }
            }

            // Ensure the RX LNA is programmed (START already applied the +6 dB
            // default; this is a no-op unless a different value was requested).
            let lna_active = hl2.lna_gain().await;
            eprintln!("LNA gain active at {lna_active:+} dB (use DEFAULT_LNA_GAIN_DB to change)");

            println!("Receiving IQ data (Ctrl-C to stop)...");

            // Pump task is running; read events until we get Ctrl-C or the
            // receiver channel closes.
            let mut block_count: u64 = 0;
            let mut total_samples: u64 = 0;
            let mut keep_alive = rx_ev;

            let mut got_ctrl = false;
            // One persistent SIGINT listener for the whole loop. (Re-creating
            // `tokio::signal::ctrl_c()` inside the hot select! loop unsubscribed
            // it between iterations, so Ctrl-C arriving during the baseband
            // branch was silently dropped by tokio's one-shot future.)
            let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
            loop {
                tokio::select! {
                    _ = sigint.recv() => {
                        if got_ctrl {
                            break;
                        }
                        got_ctrl = true;
                        eprintln!("Ctrl-C received; stopping...");
                    }
                    ev = keep_alive.recv() => match ev {
                        Some(Hl2Event::Block(b)) => {
                            block_count += 1;
                            total_samples += b.samples.len() as u64;
                            if block_count % 10 == 0 {
                                eprintln!("Block #{} received ({} total samples)", block_count, total_samples);
                            }
                        }
                        Some(Hl2Event::CmdAck { ack, raddr, ptt, data }) => {
                            eprintln!("ACK={} addr=0x{:02x} ptt={} data=0x{:08x}", ack, raddr, ptt, data);
                        }
                        Some(Hl2Event::Baseband(_)) => {}
                        None => {
                            eprintln!("Pump channel closed");
                            break;
                        }
                    }
                }
            }

            eprintln!("Stopping HL2...");
            let _ = hl2.stop().await;
        }

        "ssb" => {
            let hl2_addr: std::net::IpAddr = args
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or("192.168.1.68".parse().unwrap());
            let freq: u32 = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(14_200_000);
            let side = match args.get(3).map(|s| s.as_str()).unwrap_or("") {
                "usb" | "USB" => Sideband::Usb,
                "lsb" | "LSB" => Sideband::Lsb,
                _ => {
                    eprintln!(
                        "usage: ssb <IP> <FREQ> <usb|lsb> [--slot <N>] [--offset <Hz>] [--rate <Hz>] [--band <Hz>] [--lna <dB>] [--filtermask <hex>]"
                    );
                    return;
                }
            };

            // Optional trailing flags (after the positional args).
            //
            // `--rate <Hz>`  Per-receiver **baseband** sample rate, Hz
            //                (48/96/192/384 kHz — the openHPSDR DDC option, and
            //                what we send the radio via the C1 SPEED bit).
            //                The wire carries *complex* I/Q that runs at **half**
            //                this (Nyquist: a complex pair packs two real samples,
            //                so R real samples/s → R/2 complex pairs/s). That R/2
            //                is what we use as the demodulator's
            //                `source_rate_hz` (measured on the live box exactly).
            //                Default 192 kHz (→ 96 kHz complex, 48 kHz bandwidth).
            // `--offset <Hz>` Carrier offset from baseband centre, Hz (NCO).
            //                Default 0 Hz (the DDC hands us baseband-centred).
            // `--band <Hz>`   Channel-select bandwidth, Hz. Default 2.6 kHz.
            let mut rate_hz = 192_000u32;
            let mut offset_hz = 0.0f64;
            let mut band_hz = 2_600u32;
            // `--slot <N>`    Receiver slot to demodulate (RX1..RX4). Default 1.
            //                The HL2 interleaves one 24-bit I/Q pair per active
            //                slot in every EP6 record; the pump fans each slot's
            //                stream into its own ring, and this picks which
            //                ring the demod consumes. (Tune the radio to the
            //                same slot first — `ssb --slot 2` demods slot 2's
            //                NCO-tuned stream.)
            let mut slot: u8 = 1;
            // `--filtermask <hex>`  RX filter-board relay mask. The 7 open-
            //                      collector relays on the HL2's companion
            //                      filter board (e.g. MRF101) are driven by the
            //                      keep-alive C2 byte. LSB-first: bit 0 =
            //                      relay/checkbox 1, bit 6 = relay/checkbox 7.
            //                      `0x00` = all relays off (board default).
            //                      `0x44` = relays 3 + 7. Default `0x00`.
            let mut filtermask: u8 = 0x00;
            // `--lna <dB>`  RX low-noise-amplifier gain, dB (−12…+48).
            //                Default `DEFAULT_LNA_GAIN_DB` (+6): a touch of
            //                headroom for weak signals without saturating strong
            //                ones. The board's LNA is set once here (and again at
            //                START) — there is no RX AGC; see PROTOCOL.md §11.3.
            let mut lna_db = hl2::DEFAULT_LNA_GAIN_DB;
            let tail: Vec<String> = args.iter().skip(4).cloned().collect();
            let mut i = 0usize;
            while i < tail.len() {
                let flag = tail[i].as_str();
                let val: Option<String> = (i + 1 < tail.len()).then(|| tail[i + 1].clone());
                match flag {
                    "--rate" => {
                        rate_hz = val.and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                            eprintln!("--rate needs a value (Hz)");
                            std::process::exit(1);
                        })
                    }
                    "--offset" => {
                        offset_hz = val.and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                            eprintln!("--offset needs a value (Hz)");
                            std::process::exit(1);
                        })
                    }
                    "--band" => {
                        band_hz = val.and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                            eprintln!("--band needs a value (Hz)");
                            std::process::exit(1);
                        })
                    }
                    "--lna" => {
                        lna_db = val.and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                            eprintln!("--lna needs a value (dB, -12..48)");
                            std::process::exit(1);
                        })
                    }
                    "--filtermask" => {
                        filtermask = val
                            .and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok())
                            .unwrap_or_else(|| {
                                eprintln!("--filtermask needs a hex value (e.g. 0x44)");
                                std::process::exit(1);
                            }) as u8;
                    }
                    "--slot" => {
                        slot = val.and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                            eprintln!("--slot needs a value (1..4)");
                            std::process::exit(1);
                        })
                    }
                    other => {
                        eprintln!(
                            "unknown flag '{other}' (try --rate/--offset/--band/--lna/--filtermask/--slot)"
                        );
                        std::process::exit(1);
                    }
                }
                i += 2;
            }

            eprintln!(
                "SSB {} {} Hz on {} (rate={rate_hz} Hz, offset={offset_hz:+.0} Hz, band={band_hz} Hz, lna={lna_db:+} dB, filtermask=0x{filtermask:02x})",
                hl2_addr,
                freq,
                if side == Sideband::Usb { "USB" } else { "LSB" }
            );

            let (hl2, rx_ev, info) = Hl2::start_with_speed(hl2_addr, rate_hz)
                .await
                .unwrap_or_else(|e| {
                    eprintln!("Failed to start HL2: {}", e);
                    std::process::exit(1);
                });
            eprintln!("HL2 started (rx_count={})", info.rx_count);
            if let Err(e) = hl2.tune(slot, freq).await {
                eprintln!("Tune RX{slot} failed: {e}");
            }
            // Program the RX filter-board relay mask. This takes effect on the
            // next keep-alive tick (≤ 40 ms), so no round-trip to wait for.
            // `filtermask` is the user-facing LSB-first 7-bit mask (bit 0 =
            // relay/checkbox 1, bit 6 = relay/checkbox 7); the wire puts it in
            // keep-alive C2[7:1].
            if filtermask != 0x00 {
                hl2.set_oc_bits(filtermask).await;
                let relays: Vec<u32> = (0..7_u32)
                    .filter(|b| (filtermask & (1 << b)) != 0)
                    .map(|b| b + 1)
                    .collect();
                eprintln!(
                    "filtermask 0x{filtermask:02x} → C2=0x{:02x} active (relays: {})",
                    filtermask << 1,
                    relays
                        .iter()
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
            // Program the LNA so weak signals are amplified for the audio path.
            // START already applied the default; this enforces `--lna` when set.
            if let Err(e) = hl2.set_lna_gain(lna_db).await {
                eprintln!("LNA set failed: gain {lna_db:+} dB ({e})");
            }

            #[cfg(feature = "alsa")]
            let sink = Box::new(AlsaSink::build(None, 4_800).expect("alsa sink"));
            #[cfg(not(feature = "alsa"))]
            let sink = Box::new(VecSink::new());

            let iq_rate_hz = rate_hz;
            let rx = {
                let cfg = ReceiverConfig {
                    mode: Mode::Ssb(side),
                    source_rate_hz: iq_rate_hz,
                    source_center_hz: offset_hz,
                    bandwidth_hz: Some(band_hz),
                    audio: AudioConfig {
                        rate_hz: 4_800,
                        gain_db: 0.0,
                    },
                    tap: None,
                };
                VirtualReceiver::new(cfg, sink).expect("build virtual receiver")
            };

            // The pump (running inside Hl2) is the *sole writer* of the
            // shared baseband ring; `rx` is one of its readers. We hand `rx`
            // + an `Arc<Mutex<…>>` clone to a dedicated **std** thread so the
            // demod's per-sample DSP (NCO + ~771-tap FIR pair + decimator)
            // never blocks the EP6 RX inner loop — that was the CPU-100% /
            // choppy-audio / wrong-pair_rate failure mode.
            //
            // See `PROTOCOL.md` §16 and `hl2/src/receiver/baseband_ring.rs`
            // for the ring design (drop-oldest when full, so the pump is
            // never backpressured by a slow demod).
            //
            // `ring` is passed as `Arc<Mutex<…>>` so the thread and the main
            // task can both read it; the main thread only calls `.len()`
            // inside the periodic gauge, which is a short critical section.
            let ring = hl2.baseband_ring(slot);
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let stop_tx = stop.clone();
            let demod_thread = std::thread::Builder::new()
                .name("ssb-demod".into())
                .spawn(move || run_demod_thread(stop_tx, ring, rx, iq_rate_hz))
                .expect("spawn demod thread");

            let dbg = std::env::var("HL2_DEBUG").is_ok();

            // The demod (DSP + ALSA) lives on its own std thread reading from
            // the shared baseband ring — see `run_demod_thread` below. The
            // pump (inside `Hl2`) is the sole writer of that ring and never
            // blocks on the demod, so the EP6 RX inner loop stays fast and
            // the pump's own `[pump] pair_rate` log reflects the radio's
            // *delivered* rate regardless of demod speed.
            //
            // This task only:
            //   * waits on SIGINT,
            //   * drains the event mpsc (which the pump still fills for Block
            //     / CmdAck / Baseband — we don't consume it in the `ssb` path,
            //     but the pump would block on a full channel; unbounded,
            //     though, so we just drop events to bound memory),
            //   * prints a periodic `ring_len=` gauge (every 2 s) so the
            //     operator can see how far the demod is behind.
            let mut got_ctrl = false;
            let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
            let mut keep_alive = rx_ev;
            let mut drain_start = std::time::Instant::now();
            loop {
                tokio::select! {
                    _ = sigint.recv() => {
                        if got_ctrl {
                            break;
                        }
                        got_ctrl = true;
                        eprintln!("Ctrl-C; stopping…");
                    }
                    // Drain the event mpsc (we don't need it in the `ssb`
                    // path — the ring is the demod's source of truth — but
                    // the pump keeps filling it, and an unbounded mpsc
                    // that nobody reads will eventually OOM under heavy
                    // wideband traffic, so we just drop each event).
                    ev = keep_alive.recv() => {
                        match ev {
                            Some(hl2::Hl2Event::Block(_))
                            | Some(hl2::Hl2Event::CmdAck { .. })
                            | Some(hl2::Hl2Event::Baseband(_)) => { /* drop */ }
                            None => break,
                        }
                    }
                    // 2 s cadence.
                    () = tokio::time::sleep(tokio::time::Duration::from_secs(2)) => {
                        let ring_len = hl2.baseband_ring(slot).lock().unwrap().len();
                        let delivered = hl2.baseband_delivered().load(Ordering::Relaxed);
                        let dt = drain_start.elapsed().as_secs_f64();
                        drain_start = std::time::Instant::now();
                        // Reset the pump's `delivered` counter so the next
                        // 2 s window is a clean "delivered in the last 2 s"
                        // gauge (the pump's own `[pump] pair_rate` resets its
                        // own window too — see `run_loop`).
                        hl2.baseband_delivered().store(0, Ordering::Relaxed);
                        if dbg {
                            eprintln!(
                                "[main] ring_len={ring_len} delivered={delivered} (≈{:.0}/s, 2s window)",
                                delivered as f64 / dt.max(1e-6)
                            );
                        }
                    }
                }
            }
            // Stop the demod: signal it (its `run_demod_thread` exits at the
            // next 1 ms slice), then join so the sink/audio thread has a
            // clean chance to finish emitting. `rx` was moved into the demod
            // thread, so the flush happens there (inside `run_demod_thread`).
            stop.store(true, Ordering::Relaxed);
            match demod_thread.join() {
                Ok(()) => {}
                Err(_) => {
                    // `thread::JoinHandle::join` returns `Box<dyn Any + Send>`
                    // on panic; we can't print it with Display, but we can
                    // report that the demod thread panicked.
                    eprintln!("demod thread panicked during join");
                }
            }
            let _ = hl2.stop().await;
            eprintln!("done");
        }

        "ft8" => {
            // FT8 decode diagnostic (the non-zero-NCO-offset bug). The radio
            // is tuned to `FREQ`; when `--offset` is non-zero the virtual
            // receiver's NCO shifts the channel that amount before the
            // 12 kHz digital demod — the hub's `spawn_vrx` FT8 path exactly,
            // minus the WS fan-out:
            //
            //   NCO-tuned baseband ─► NCO ─► LPF ─► 12 kHz f32
            //   ─► Ft8Tap ─► mfsk-core 15 s slot decode (wall-clock aligned)
            //
            // Decoded rows print on stdout; diagnostics on stderr. Exit 0 iff
            // ≥1 message decoded — scriptable:
            //
            //   hl2 ft8 192.168.1.67 7074000 --seconds 48              # offset 0, known-good
            //   hl2 ft8 192.168.1.67 7050000 --offset 24000 --seconds 48  # offset +24k
            //
            let hl2_addr: std::net::IpAddr = args
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or("192.168.1.67".parse().unwrap());
            let freq: u32 = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(7_074_000);
            let mut rate_hz = 192_000u32;
            let mut offset_hz = 0.0f64;
            let mut slot: u8 = 1;
            let mut seconds: u64 = 60;
            let mut lna_db = 30i8;
            // F4 + F7 (LSB-first: bit 3 = F4, bit 6 = F7), as set from the UI.
            let mut filtermask: u8 = 0x14;
            let probe = if args.iter().any(|a| a == "--probe") {
                8192usize
            } else {
                0
            };
            let tail: Vec<String> = args.iter().skip(3).cloned().collect();
            let mut i = 0usize;
            while i < tail.len() {
                let flag = tail[i].as_str();
                if flag == "--probe" {
                    i += 1;
                    continue;
                }
                let val: Option<String> = (i + 1 < tail.len()).then(|| tail[i + 1].clone());
                let v = || val.as_deref().expect("--flag needs a value");
                match flag {
                    "--rate" => rate_hz = v().parse().unwrap(),
                    "--offset" => offset_hz = v().parse().unwrap(),
                    "--slot" => slot = v().parse().unwrap(),
                    "--seconds" => seconds = v().parse().unwrap(),
                    "--lna" => lna_db = v().parse().unwrap(),
                    "--filtermask" => {
                        filtermask =
                            u32::from_str_radix(v().trim_start_matches("0x"), 16).unwrap() as u8
                    }
                    other => {
                        eprintln!("unknown flag '{other}' (try --offset/--rate/--slot/--seconds)");
                        std::process::exit(1);
                    }
                }
                i += 2;
            }

            let ft8_target = freq as i64 + offset_hz as i64;
            eprintln!(
                "ft8 diag | RX{slot}={freq} Hz | vrx offset {offset_hz:+.0} Hz (target RF {ft8_target} Hz) | rate={rate_hz} Hz | lna={lna_db:+} dB | {seconds} s",
            );

            let (hl2, rx_ev, _info) = Hl2::start_with_speed(hl2_addr, rate_hz)
                .await
                .unwrap_or_else(|e| {
                    eprintln!("Failed to start HL2: {e}");
                    std::process::exit(1);
                });
            if let Err(e) = hl2.tune(slot, freq).await {
                eprintln!("Tune RX{slot} -> {freq} Hz failed: {e}");
            }
            if filtermask != 0x00 {
                hl2.set_oc_bits(filtermask).await;
            }
            if let Err(e) = hl2.set_lna_gain(lna_db).await {
                eprintln!("LNA {lna_db:+} dB failed: {e}");
            }

            // FT8 mode: 12 kHz digital output (same as `VRX_FT8_RATE_HZ` in the
            // API hub). `DropSink` for monitor audio — irrelevant to decode.
            let audio_rate = 12_000u32;
            let dec = shared();
            let tap: Option<std::sync::Arc<dyn RawSampleTap>> =
                Some(std::sync::Arc::new(Ft8Tap::from_shared(dec.clone()))
                    as std::sync::Arc<dyn RawSampleTap>);
            let decoded_any = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            // (See the `source_center_hz` comment below for the sign rule.)
            let cfg = ReceiverConfig {
                mode: Mode::Ft8,
                source_rate_hz: rate_hz,
                // `--offset` = "target − NCO" (Hz), positive for a target
                // *above* the NCO. The demod's `source_center_hz` is "the
                // carrier's baseband location", and the firmware places an
                // above-NCO target at a **negative** baseband frequency, so
                // negate. (This is the same sign fix applied in
                // `api/src/hub.rs::spawn_vrx` / `spawn_auto`.)
                source_center_hz: -offset_hz,
                bandwidth_hz: Some(2600u32),
                audio: AudioConfig {
                    rate_hz: audio_rate,
                    gain_db: 0.0,
                },
                tap,
            };
            let rx = VirtualReceiver::new(cfg, Box::new(DropSink::new()))
                .expect("build ft8 virtual receiver");

            let ring = hl2.baseband_ring(slot);
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let stop_tx = stop.clone();
            let demod = std::thread::Builder::new()
                .name("ft8-diag-demod".into())
                .spawn(move || run_diagnostic_demod(stop_tx, ring, rx, rate_hz, probe))
                .expect("spawn demod thread");

            // Decode thread: 250 ms tick; once a wall-clock slot has fully
            // closed AND the buffer holds ≥ its full contents' worth of
            // samples (so the trailing 15 s we slice is exactly that slot's
            // window), run the mfsk-core decode off-lock.
            let stop_dx = stop.clone();
            let dec_d = dec.clone();
            let ft8_target_dx = ft8_target;
            let deadline_d = Instant::now() + Duration::from_secs(seconds);
            let decoded_any_dx = decoded_any.clone();
            let decode_thread = std::thread::Builder::new()
                .name("ft8-diag-decode".into())
                .stack_size(8 * 1024 * 1024)
                .spawn(move || {
                    let mut last_slot = 0u64;
                    loop {
                        if stop_dx.load(Ordering::Relaxed) || Instant::now() > deadline_d {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(250));
                        let now_ms = now_ms_since_epoch();
                        let slot_ms = (now_ms / 15_000) * 15_000;
                        if slot_ms == last_slot {
                            continue;
                        }
                        // Once the buffer holds ≥ 18 s of demod output the
                        // trailing 15 s (what `decode_closed_slot` slices)
                        // is exactly the just-closed full slot: we started
                        // ≥ 17.5 s ago and the demod outputs 12 kHz in
                        // real time, so the slot's 15 s window is fully in
                        // the buffer. (The slot-end wall-clock offset
                        // within the epoch doesn't matter — we compare
                        // *our* buffer to *our* wall-clock, both running
                        // in real time.)
                        let blen = dec_d.lock().unwrap().buffered_samples();
                        if blen < 18 * 12_000 {
                            continue;
                        }
                        last_slot = slot_ms;
                        match decode_closed_slot(&dec_d, slot_ms) {
                            Ok(msgs) => {
                                if msgs.is_empty() {
                                    eprintln!(
                                        "  slot {slot_ms}: no decode (buffer={blen} samples ≈{:.1} s)",
                                        blen as f64 / 12_000.0
                                    );
                                    let win = {
                                        let g = dec_d.lock().unwrap();
                                        let n = g.buf_trailing(180_000);
                                        n
                                    };
                                    if let Some(fp) = audio_band_fingerprint(&win, 12_000) {
                                        eprintln!("  fp {fp}");
                                    }
                                }
                                decoded_any_dx.store(true, std::sync::atomic::Ordering::Relaxed);
                                for m in &msgs {
                                    println!(
                                        "DECODED  text={:?}  offset={:+6.1} Hz  snr={:+5.1} dB  slot={}",
                                        m.text,
                                        m.freq_hz,
                                        m.snr_db,
                                        m.slot_ms
                                    );
                                    let rf = ft8_target_dx + m.freq_hz as i64;
                                    println!("         rf={rf} Hz");
                                }
                            }
                            Err(e) => eprintln!("  slot {slot_ms}: decode err: {e} (buffer={blen})"),
                        }
                    }
                })
                .expect("spawn decode thread");

            // Main: drain events, print per-second gauge, stop on SIGINT (×2)
            // or the `seconds` deadline.
            let mut got_ctrl = false;
            let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
            let mut keep_alive = rx_ev;
            let deadline = Instant::now() + Duration::from_secs(seconds);
            loop {
                tokio::select! {
                    _ = sigint.recv() => {
                        if got_ctrl {
                            break;
                        }
                        got_ctrl = true;
                        eprintln!("Ctrl-C; stopping…");
                    }
                    ev = keep_alive.recv() => {
                        match ev {
                            Some(hl2::Hl2Event::Block(_))
                            | Some(hl2::Hl2Event::CmdAck { .. })
                            | Some(hl2::Hl2Event::Baseband(_)) => {}
                            None => break,
                        }
                    }
                    () = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {
                        let ring_len = hl2.baseband_ring(slot).lock().unwrap().len();
                        let delivered = hl2.baseband_delivered().load(Ordering::Relaxed);
                        hl2.baseband_delivered().store(0, Ordering::Relaxed);
                        let now_ms = now_ms_since_epoch();
                        let blen = dec.lock().unwrap().buffered_samples();
                        eprintln!(
                            "[t] ring={ring_len} delivered={delivered} in 1 s  buffer={blen} (~{:.1} s)  next slot in {} ms",
                            blen as f64 / 12_000.0,
                            15_000 - now_ms % 15_000
                        );
                        if Instant::now() >= deadline {
                            break;
                        }
                    }
                }
            }
            stop.store(true, Ordering::Relaxed);
            demod.join().ok();
            decode_thread.join().ok();
            let _ = hl2.stop().await;
            let any = decoded_any.load(Ordering::Relaxed);
            let blen = dec.lock().unwrap().buffered_samples();
            eprintln!(
                "{} (buffered {blen} samples)",
                if any { "DECODE OK" } else { "no decode" }
            );
            std::process::exit(if any { 0 } else { 1 });
        }

        _ => {
            eprintln!("Usage:");
            eprintln!("  hl2 discover                     - Discover HL2 devices");
            eprintln!("  hl2 probe <IP>                   - Probe specific device");
            eprintln!("  hl2 start [IP] [FREQ]            - Start receiving IQ data");
            eprintln!("  hl2 ft8 <IP> <FREQ> [--offset <Hz>]");
            eprintln!("                                   [--rate <Hz>] [--slot <N>]");
            eprintln!("                                   [--seconds <N>] [--lna <dB>]");
            eprintln!(
                "                                   - FT8 decode diagnostic (exit 0 iff decoded)"
            );
            eprintln!(
                "                                   Default IP: {}",
                HL2_ADDR
            );
            eprintln!(
                "                                   Default FREQ: {} Hz",
                DEFAULT_FREQ
            );
        }
    }
}

/// The FT8-diagnostic demod loop (same shape as
/// [`run_demod_thread`] but no ALSA / no per-second spectral dump — only
/// the tap / decode path is exercised). Reads baseband from the shared
/// ring and feeds the [`VirtualReceiver`] (built in `Mode::Ft8` so the
/// `Ft8Tap` is attached and the 12 kHz decimated f32 stream lands in
/// the shared FT8 decoder).
fn run_diagnostic_demod(
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ring: std::sync::Arc<std::sync::Mutex<hl2::receiver::baseband_ring::BasebandRing>>,
    rx: hl2::receiver::VirtualReceiver,
    rate_hz: u32,
    probe: usize,
) {
    let mut rx = rx;
    let mut iq_buf: Vec<num_complex::Complex<f32>> = Vec::with_capacity(1024);
    let mut cursor: u64 = 0;
    let mut probed = probe == 0;
    let mut probe_buf: Vec<num_complex::Complex<f32>> = Vec::new();
    let probe_cap = std::cmp::max(probe, 8192);
    loop {
        if stop.load(Ordering::Relaxed) {
            let _ = rx.flush();
            return;
        }
        let got = {
            let mut g = ring.lock().unwrap();
            let base = g.base_seq();
            if cursor < base {
                cursor = base;
                0
            } else {
                g.peek(cursor, &mut iq_buf, probe_cap)
            }
        };
        if got == 0 {
            std::thread::sleep(std::time::Duration::from_micros(500));
            continue;
        }
        cursor += got as u64;
        if !probed {
            probe_buf.extend_from_slice(&iq_buf[..got]);
            if probe_buf.len() >= probe_cap {
                probed = true;
                if let Some(fp) = raw_iq_fingerprint(&probe_buf[..probe_cap], rate_hz) {
                    eprintln!("[iq] raw baseband: {fp}");
                }
            }
        }
        let chunk: Vec<num_complex::Complex<f32>> = iq_buf[..got].to_vec();
        if let Err(e) = rx.process(&chunk) {
            eprintln!("ft8 diag demod err: {e}");
        }
    }
}

/// Raw baseband arm + spectrum measurement. The *decisive* diagnostic for
/// the non-zero-NCO bug:
///
/// * `i_rms` / `q_rms` — if `q_rms` is a small fraction of `i_rms` the DDC
///   is handing us a **real/DSB** baseband (Q≈0) and the demod's
///   `post.re`-only path halves the wanted term and admits the mirror.
/// * `top5` — the spectral peaks in the **raw** (pre-NCO) baseband, as
///   signed Hz relative to baseband centre. A true FT8 target at
///   `RF target − NCO` lands there; its DSB mirror, at the negative of
///   that frequency.
fn raw_iq_fingerprint(iq: &[num_complex::Complex<f32>], rate_hz: u32) -> Option<String> {
    let n = iq.len();
    if n < 1_024 || rate_hz == 0 {
        return None;
    }
    let mut i_sq = 0.0f64;
    let mut q_sq = 0.0f64;
    let mut x_re = vec![0f32; n];
    for (m, c) in iq.iter().enumerate() {
        i_sq += (c.re as f64) * (c.re as f64);
        q_sq += (c.im as f64) * (c.im as f64);
        x_re[m] = c.re;
    }
    let i_rms = (i_sq / n as f64).sqrt();
    let q_rms = (q_sq / n as f64).sqrt();
    let mut x_im = vec![0f32; n];
    fft_inplace(&mut x_re, &mut x_im);
    let df = rate_hz as f64 / n as f64;
    let mut bins: Vec<(i32, f64)> = (0..n)
        .map(|k| {
            let f = if (k as i64) <= (n / 2) as i64 {
                k as i32
            } else {
                k as i32 - n as i32
            };
            let m =
                ((x_re[k] as f64) * (x_re[k] as f64) + (x_im[k] as f64) * (x_im[k] as f64)).sqrt();
            (f, m)
        })
        .collect();
    let mean: f64 = bins.iter().map(|(_, m)| *m).sum::<f64>() / n as f64;
    bins.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5: Vec<String> = bins
        .iter()
        .take(5)
        .map(|(k, m)| format!("{:+7.1} Hz ×{:.1}", (*k as f64) * df, *m / mean.max(1e-12)))
        .collect();
    // Signed-frequency magnitude via a **complex** DFT evaluated (directly) at
    // a small set of candidate frequencies, in ± symmetric pairs. For an
    // analytic signal exactly ONE of +f / −f dominates — that sign is the
    // decisive one for this bug (does "above NCO" map to +f or −f?).
    let candidates_hz = [6_000i64, 12_000, 14_000, 16_000, 24_000, 30_000, 36_000];
    let mut sig = Vec::new();
    for f0 in candidates_hz {
        for sgn in [1i64, -1] {
            let fk = f0 * sgn;
            let k_exact = fk as f64 * (n as f64) / (rate_hz as f64) as f64;
            // Direct complex DFT at bin k (O(n), fine for ~14 candidates).
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for m in 0..n {
                let ang = 2.0 * std::f64::consts::PI * (k_exact * (m as f64)) / (n as f64);
                re += (iq[m].re as f64) * ang.cos() - (iq[m].im as f64) * ang.sin();
                im += (iq[m].re as f64) * ang.sin() + (iq[m].im as f64) * ang.cos();
            }
            let mag = (re * re + im * im).sqrt();
            let db = if i_rms > 1e-18 {
                20.0 * (mag / (i_rms * n as f64)).log10()
            } else {
                f64::MIN
            };
            sig.push(format!("{:+5}k {:+6.0} dB", fk / 1000, db));
        }
    }
    Some(format!(
        "i_rms={i_rms:.3e} q_rms={q_rms:.3e} (q/i={:.3})  top5: {}  signed: {}",
        q_rms / i_rms.max(1e-18),
        top5.join(", "),
        sig.join(" ")
    ))
}

/// Wall-clock milliseconds since the Unix epoch; 0 on pre-epoch clocks.
fn now_ms_since_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Spectral fingerprint of a 12 kHz audio-band window: top-5 peaks (Hz, ×
/// mean) and the rms level. This is the diagnostic that says *where* the
/// energy actually is when a slot fails to decode — a true FT8 tone
/// cluster at 100–3000 Hz vs. broadband noise or energy at the wrong
/// frequency (e.g. the image, or the DSB mirror if the baseband is real).
fn audio_band_fingerprint(win: &[f32], rate_hz: u32) -> Option<String> {
    if win.len() < 2_048 || rate_hz == 0 {
        return None;
    }
    let fft_len = win.len().next_power_of_two();
    let mut x_re = vec![0f32; fft_len];
    for (m, &v) in win.iter().enumerate() {
        x_re[m] = v;
    }
    let mut x_im = vec![0f32; fft_len];
    fft_inplace(&mut x_re, &mut x_im);
    let half = fft_len / 2;
    let df = rate_hz as f64 / fft_len as f64;
    let mut mags = Vec::with_capacity(half);
    for k in 0..half {
        let m = ((x_re[k] as f64) * (x_re[k] as f64) + (x_im[k] as f64) * (x_im[k] as f64)).sqrt();
        mags.push(m);
    }
    let mean = mags.iter().sum::<f64>() / half as f64;
    let mut peaks: Vec<(usize, f64)> = (0..half).map(|k| (k, mags[k])).collect();
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5: Vec<String> = peaks
        .into_iter()
        .take(5)
        .map(|(k, m)| format!("{:+6.1} Hz ×{:.1}", (k as f64) * df, m / mean.max(1e-12)))
        .collect();
    let mut sq = 0.0f64;
    for &v in win.iter() {
        let d = v as f64;
        sq += d * d;
    }
    let rms = (sq / win.len() as f64).sqrt();
    Some(format!("rms={rms:.3e}  peaks: {}", top5.join(", ")))
}

/// The SSB demod loop, running on a dedicated **std** thread.
///
/// Owns the `VirtualReceiver` (moved in). Reads the shared baseband ring, demodulates
/// each chunk into `AlsaSink`/`VecSink`, and sleeps ~1 ms between iterations
/// so the thread does not busy-spin when the ring is empty.
///
/// When `HL2_DEBUG` is set, also prints a rolling `[dlg] pair_rate` and a
/// per-second spectral report via `dbg_spectral`.
///
/// Exits when `stop` is set to `true` (checked at the top of each loop
/// iteration). On the way out, flushes `rx` so any partial audio block is
/// pushed to the sink.
#[allow(clippy::too_many_arguments)]
fn run_demod_thread(
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ring: std::sync::Arc<std::sync::Mutex<hl2::receiver::baseband_ring::BasebandRing>>,
    rx: hl2::receiver::VirtualReceiver,
    iq_rate_hz: u32,
) {
    let mut rx = rx;
    let dbg = std::env::var("HL2_DEBUG").is_ok();
    let mut iq_buf: Vec<num_complex::Complex<f32>> = Vec::with_capacity(1024);
    let mut cursor: u64 = 0;
    let mut dbg_buf: Vec<num_complex::Complex<f32>> = Vec::with_capacity(4096);
    const DBG_BUF_CAP: usize = 4096;
    let mut dbg_pairs: usize = 0;
    let mut dbg_start = std::time::Instant::now();
    let mut idle_spins: u32 = 0;

    loop {
        if stop.load(Ordering::Relaxed) {
            if let Err(e) = rx.flush() {
                eprintln!("demod thread: rx.flush err: {e}");
            }
            if dbg && dbg_pairs > 0 && dbg_start.elapsed().as_secs() > 0 {
                let dt = dbg_start.elapsed().as_secs_f64();
                let rate = dbg_pairs as f64 / dt;
                if let Some(report) = dbg_spectral(&dbg_buf, iq_rate_hz) {
                    eprintln!("[dlg] final pair_rate={rate:.0}/s (start) {report}");
                }
            }
            return;
        }

        let got = {
            let mut g = ring.lock().unwrap();
            let base = g.base_seq();
            if cursor < base {
                cursor = base;
                0
            } else {
                g.peek(cursor, &mut iq_buf, 1024)
            }
        };
        if got == 0 {
            idle_spins = idle_spins.saturating_add(1);
            if idle_spins % 200 == 1 {
                idle_spins = 0;
            }
            std::thread::sleep(std::time::Duration::from_micros(1000));
            continue;
        }
        cursor += got as u64;
        idle_spins = 0;
        if let Err(e) = rx.process(&iq_buf) {
            eprintln!("demod err: {e}");
        }
        if dbg {
            dbg_pairs += got;
            dbg_buf.extend_from_slice(&iq_buf);
            dbg_buf.drain(..dbg_buf.len().saturating_sub(DBG_BUF_CAP));
            let now = std::time::Instant::now();
            if now.duration_since(dbg_start) >= std::time::Duration::from_secs(1) {
                let rate = dbg_pairs as f64 / dbg_start.elapsed().as_secs_f64();
                dbg_pairs = 0;
                dbg_start = now;
                if let Some(report) = dbg_spectral(&dbg_buf, iq_rate_hz) {
                    eprintln!("[dlg] pair_rate={rate:.0}/s (expect {iq_rate_hz} Hz)  {report}");
                }
            }
        }
    }
}

/// Full FFT-based spectral diagnostic for a rolling I/Q buffer. Returns
/// a formatted report line.
///
/// Reports (for the most recent 4096 samples):
///   * **rms** — per-sample √(I²+Q²). Absolute I/Q level.
///   * **mean |X|** — mean of the magnitude spectrum (noise floor).
///   * **top peak** — frequency (kHz, signed) and magnitude of the single
///     biggest |X| bin in the full FFT. A single sharp peak with
///     `mag / mean ≥ 10` is a tone; a "top peak" of only ~2× the mean
///     (with neighbours of similar magnitude) is a random noise maximum.
///   * **top-5 peaks** (kHz, mag/mean) — a single 8.7 kHz FT8-like tone
///     will dominate; 5 peaks of similar magnitude means broadband noise.
///
/// This is the definitive way to answer "is the FT8 in our baseband at all,
/// and at what offset from baseband centre." If the top-5 look like 5
/// random noise peaks (each ~1.2–1.8× the mean), the signal is *not* in the
/// channel on this slot — either the NCO is offset, the DDC is handing us
/// the wrong slot's I/Q, or the filter board is passing the wrong band.
fn dbg_spectral(iq: &[num_complex::Complex<f32>], rate_hz: u32) -> Option<String> {
    let n = iq.len();
    if n < 64 || rate_hz == 0 {
        return None;
    }
    // Pad to next power of 2 and apply a Hann window (mitigates spectral
    // leakage so we can distinguish a narrow tone from broad-band noise).
    let fft_len = n.max(256).next_power_of_two();
    let mut x_re = vec![0f32; fft_len];
    let mut x_im = vec![0f32; fft_len];
    for (m, c) in iq.iter().enumerate() {
        let w = (0.5_f64
            - 0.5_f64 * (2.0 * std::f64::consts::PI * (m as f64 / fft_len as f64)).cos())
            as f32;
        x_re[m] = c.re * w;
        x_im[m] = c.im * w;
    }
    // In-place iterative radix-2 FFT (Cooley–Tukey, bit-reversal permut).
    // ~N log N complex MACs; N = 8192 (padded up from 4096) takes ~1 ms.
    fft_inplace(&mut x_re, &mut x_im);
    // Parse the first half (positive frequencies): bin 0 = DC, bin fft_len/2 = Nyquist.
    let half = fft_len / 2;
    let fs = rate_hz as f64;
    let df = fs / fft_len as f64;
    // RMS of the input stream.
    let mut sq = 0.0f64;
    for c in iq {
        sq += (c.re as f64) * (c.re as f64) + (c.im as f64) * (c.im as f64);
    }
    let rms = (sq / n as f64).sqrt();
    // Magnitude spectrum, first half.
    let mut mags = Vec::with_capacity(half);
    for k in 0..half {
        let m = ((x_re[k] as f64) * (x_re[k] as f64) + (x_im[k] as f64) * (x_im[k] as f64)).sqrt();
        mags.push(m);
    }
    // Mean magnitude (noise floor) and top-5 peaks (sorted desc).
    let mut mags_sorted = mags.clone();
    mags_sorted.sort_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .reverse()
    });
    let mean = mags.iter().sum::<f64>() / half as f64;
    // Find the top-5 bin *indices* (to report their signed frequency).
    // We can't just use `mags_sorted` because it loses index info — find top
    // 5 by value with the bin index, ignoring ties.
    let mut peaks: Vec<(usize, f64)> = (0..half).map(|k| (k, mags[k])).collect();
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5: Vec<(f64, f64)> = peaks
        .into_iter()
        .take(5)
        .map(|(k, m)| {
            let f_hz = (k as f64) * df;
            // Positive-half indices [1, N/2] → positive freq. Bin 0 is DC.
            let f_khz = f_hz / 1000.0;
            (f_khz, m / mean.max(1e-12))
        })
        .collect();
    let peak_khz = top5.first().map(|(f, _)| *f).unwrap_or(0.0);
    let peak_ratio = top5.first().map(|(_, r)| *r).unwrap_or(0.0);
    let top5_str = top5
        .iter()
        .map(|(f, r)| format!("{:+.2}kHz×{r:.1}", f))
        .collect::<Vec<_>>()
        .join(",");
    // Per-arm energy: proves whether the baseband is analytic (I and Q both
    // loud) or real/DSB (Q ≈ 0 → the sideband selection must synthesize Q).
    let mut i_sq = 0.0f64;
    let mut q_sq = 0.0f64;
    for c in iq {
        i_sq += (c.re as f64) * (c.re as f64);
        q_sq += (c.im as f64) * (c.im as f64);
    }
    let i_rms = (i_sq / n as f64).sqrt();
    let q_rms = (q_sq / n as f64).sqrt();
    Some(format!(
        "i_rms={i_rms:.3} q_rms={q_rms:.3}  rms={rms:.3}  strongest={peak_khz:+.2} kHz (mag {peak_ratio:.1}× mean)  top5: {top5_str}"
    ))
}

/// Iterative in-place radix-2 Cooley–Tukey FFT. Length must be a power of
/// two. `x_re` / `x_im` are the real/imaginary parts of the DFT input; on
/// return they hold the DFT output (bit-reversed order is handled here).
fn fft_inplace(x_re: &mut [f32], x_im: &mut [f32]) {
    let n = x_re.len();
    if n <= 1 {
        return;
    }
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let bit = n >> 1;
        while j >= bit {
            j -= bit;
        }
        j += bit;
        if i < j {
            x_re.swap(i, j);
            x_im.swap(i, j);
        }
    }
    // Butterfly stages.
    let mut len = 2usize;
    while len <= n {
        let ang = 2.0 * std::f64::consts::PI / (len as f64);
        let w_re = ang.cos() as f32;
        let w_im = -ang.sin() as f32;
        let half = len >> 1;
        let mut i = 0usize;
        while i < n {
            let mut cur_re = 1.0f32;
            let mut cur_im = 0.0f32;
            for k in 0..half {
                let a = i + k;
                let b = i + k + half;
                let t_re = cur_re * x_re[b] - cur_im * x_im[b];
                let t_im = cur_re * x_im[b] + cur_im * x_re[b];
                x_re[b] = x_re[a] - t_re;
                x_im[b] = x_im[a] - t_im;
                x_re[a] += t_re;
                x_im[a] += t_im;
                // cur *= W
                let nre = cur_re * w_re - cur_im * w_im;
                let nim = cur_re * w_im + cur_im * w_re;
                cur_re = nre;
                cur_im = nim;
            }
            i += len;
        }
        len <<= 1;
    }
}
