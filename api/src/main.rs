//! HL2 API — Rocket + WebSocket server.
//!
//! Boot the Rocket server, manage the `RadioHub` as shared state, and mount
//! the `/api/ws` WebSocket route.

mod hub;
mod pskrep_hook;
mod spectrum;
mod ws;

#[cfg(feature = "embed-ui")]
mod web;

#[macro_use]
extern crate rocket;

#[launch]
fn rocket() -> _ {
    let hint: Option<std::net::IpAddr> =
        std::env::var("HL2_ADDR").ok().and_then(|s| s.parse().ok());

    let pskrep: pskrep_hook::SharedPsk =
        std::sync::Arc::new(std::sync::Mutex::new(pskrep_hook::Pskrep::from_env()));
    if std::env::var("HL2_DEBUG").is_ok() {
        let g = pskrep.lock().unwrap();
        eprintln!(
            "[PSKREP] spot posting {} (PSK_CALL={:?})",
            if g.enabled { "enabled" } else { "disabled" },
            g.station.callsign
        );
    }
    {
        let pskrep = pskrep.clone();
        std::thread::Builder::new()
            .name("pskrep-sender".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .enable_io()
                    .build()
                    .unwrap();
                rt.block_on(pskrep_hook::run_sender(pskrep));
            })
            .expect("spawn pskrep sender thread");
    }

    let hub = hub::RadioHub::new(hub::HubConfig::default(), hint, pskrep);

    let rocket = rocket::build().manage(std::sync::Arc::new(hub));

    #[cfg(feature = "embed-ui")]
    let rocket = rocket.mount("/", routes![web::index_page, web::asset]);

    rocket.mount("/", routes![ws::ws_route])
}
