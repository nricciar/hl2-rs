//! End-to-end integration tests against a real HL2 device.
//!
//! These tests require an actual HL2 on the network and are gated behind
//! the `integration-tests` feature flag:
//!
//!     cargo test --features integration-tests

#[cfg(feature = "integration-tests")]
mod integration {
    use std::net::Ipv4Addr;
    use std::time::Duration;

    use hl2::protocol::discovery::DiscoveryInfo;
    use hl2::{Hl2, Hl2Event, RX1_ADDR, discover, discover_single};

    fn ip() -> Ipv4Addr {
        std::env::var("HL2_IP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(Ipv4Addr::new(169, 254, 19, 221))
    }
    const HL2_IP: Ipv4Addr = Ipv4Addr::new(169, 254, 19, 221);

    fn assert_hl2(info: &DiscoveryInfo) {
        assert_eq!(info.board_id, 0x06, "Expected HL2 board ID");
    }

    #[tokio::test]
    async fn test_discover() {
        let devices = discover().await;
        assert!(
            !devices.is_empty(),
            "No HL2 devices discovered. Ensure an HL2 is connected."
        );
        for (_, info) in &devices {
            eprintln!(
                "Discovered HL2: Board ID=0x{:02x}, GW={}.{}",
                info.board_id, info.gateware_major, info.gateware_minor
            );
            assert_hl2(info);
        }
    }

    #[tokio::test]
    async fn test_discover_single() {
        let result = discover_single(HL2_IP).await;
        assert!(result.is_some(), "No HL2 found at {}", HL2_IP);
        let (_, info) = result.unwrap();
        eprintln!("Found HL2: Board ID=0x{:02x}", info.board_id);
        assert_hl2(&info);
    }

    #[tokio::test]
    async fn test_start_stop() {
        let (hl2, _rx, start_info) = Hl2::start(HL2_IP).await.expect("Failed to start HL2");
        eprintln!(
            "HL2 started, sample format: {:?}, local_port: {}",
            start_info.sample_format, start_info.local_port
        );
        assert!(hl2.is_started().await);

        hl2.stop().await.expect("Failed to stop HL2");
        assert!(!hl2.is_started().await);
    }

    #[tokio::test]
    async fn test_tune() {
        let (hl2, _rx, _info) = Hl2::start(HL2_IP).await.expect("Failed to start HL2");

        let freq = 14_200_000u32;
        hl2.tune(RX1_ADDR, freq).await.expect("Failed to tune RX1");
        eprintln!("Set RX1 frequency to {} Hz", freq);

        hl2.stop().await.expect("Failed to stop HL2");
    }

    #[tokio::test]
    async fn test_receive_blocks() {
        let (hl2, mut rx, _info) = Hl2::start(HL2_IP).await.expect("Failed to start HL2");

        let freq = 7_000_000u32;
        hl2.tune(RX1_ADDR, freq).await.expect("Failed to tune RX1");

        let stop_after = 5u32;
        let mut got = 0u32;
        println!("Receiving {} blocks...", stop_after);

        while got < stop_after {
            let Some(ev) = tokio::time::timeout(Duration::from_secs(10), rx.recv())
                .await
                .expect("timed out waiting for a block")
            else {
                panic!("pump channel closed early");
            };

            if let Hl2Event::Block(block) = ev {
                got += 1;
                assert_eq!(
                    block.samples.len(),
                    2048,
                    "Block should contain 2048 samples, got {}",
                    block.samples.len()
                );
                assert_eq!(
                    block.sample_rate_hz, 76_800_000,
                    "ADC clock should be 76.8 MHz"
                );
                eprintln!(
                    "Block #{} received, seq={}, samples={}",
                    got,
                    block.seq_start,
                    block.samples.len()
                );
            }
        }

        hl2.stop().await.expect("Failed to stop HL2");
    }
}
