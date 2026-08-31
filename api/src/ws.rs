//! The WebSocket endpoint. Each connection is handled by an async task that:
//!
//! 1. Subscribes to the hub's fan-out broadcast.
//! 2. Reads JSON `ClientCmd` envelopes from the client.
//! 3. Forwards each command to the hub. The hub replies by broadcasting a
//!    JSON `ServerResponse` and, on every subsequent wideband frame, a binary
//!    `CH_WIDEBAND` payload to all connected clients.

use rocket::State;
use rocket::futures::{SinkExt, StreamExt};
use rocket_ws::{Channel, Message, WebSocket};
use serde::Deserialize;
use std::sync::Arc;

use crate::hub::{RadioHub, WsEvent};
use hl2_common::{AudioFrame, CH_AUDIO, CH_WIDEBAND, ClientCmd, encode_ws_binary};

/// Control envelope: `{"id": N, ...ClientCmd fields}`.
#[derive(Debug, Deserialize)]
pub struct Envelope {
    #[serde(default)]
    pub id: u64,
    #[serde(flatten)]
    pub cmd: ClientCmd,
}

fn parse_envelope(json: &str) -> Option<Envelope> {
    serde_json::from_str(json).ok()
}

fn to_message(ev: &WsEvent) -> Message {
    match ev {
        WsEvent::Json(resp) => {
            Message::Text(serde_json::to_string(resp).unwrap_or_else(|_| "{}".into()))
        }
        WsEvent::Wideband { mags, .. } => {
            let frame = hl2_common::SpectrumFrame {
                seq_start: 0,
                bin_count: mags.len() as u16,
                mags: mags.clone(),
            };
            Message::Binary(encode_ws_binary(CH_WIDEBAND, &frame.to_bytes()))
        }
        WsEvent::Audio {
            slot,
            seq,
            rate_hz,
            samples,
        } => {
            let frame = AudioFrame {
                slot: *slot as u16,
                seq: *seq,
                rate_hz: *rate_hz,
                samples: samples.clone(),
            };
            Message::Binary(encode_ws_binary(CH_AUDIO, &frame.to_bytes()))
        }
        WsEvent::Ft8Log(log) => {
            // `{"cmd":"ft8log","data":[{…}, …]}` — at most one per 15 s
            // FT8 slot, `data` = every CRC-passing decode from that slot.
            // JSON text frame (not a binary channel — rows are sparse and
            // human-readable, so text keeps the log visible in any WS
            // client / browser devtools).
            #[derive(serde::Serialize)]
            struct Ft8LogEnvelope<'a> {
                cmd: &'static str,
                data: &'a [hl2_common::Ft8Decode],
            }
            let env = Ft8LogEnvelope {
                cmd: "ft8log",
                data: &log.decodes,
            };
            Message::Text(
                serde_json::to_string(&env)
                    .unwrap_or_else(|_| r#"{"cmd":"ft8log","data":[]}"#.into()),
            )
        }
        WsEvent::Js8Log(log) => {
            // `{"cmd":"js8log","data":[{…}, …]}` — at most one per 15 s
            // JS8 slot, `data` = every CRC-passing frame from that slot.
            #[derive(serde::Serialize)]
            struct Js8LogEnvelope<'a> {
                cmd: &'static str,
                data: &'a [hl2_common::Js8Decode],
            }
            let env = Js8LogEnvelope {
                cmd: "js8log",
                data: &log.decodes,
            };
            Message::Text(
                serde_json::to_string(&env)
                    .unwrap_or_else(|_| r#"{"cmd":"js8log","data":[]}"#.into()),
            )
        }
        WsEvent::Ft4Log(log) => {
            // `{"cmd":"ft4log","data":[{…}, …]}` — at most one per 7.5 s
            // FT4 slot, `data` = every CRC-passing decode from that slot.
            // JSON text frame (not a binary channel — rows are sparse and
            // human-readable, so text keeps the log visible in any WS
            // client / browser devtools).
            #[derive(serde::Serialize)]
            struct Ft4LogEnvelope<'a> {
                cmd: &'static str,
                data: &'a [hl2_common::Ft4Decode],
            }
            let env = Ft4LogEnvelope {
                cmd: "ft4log",
                data: &log.decodes,
            };
            Message::Text(
                serde_json::to_string(&env)
                    .unwrap_or_else(|_| r#"{"cmd":"ft4log","data":[]}"#.into()),
            )
        }
    }
}

#[rocket::get("/api/ws")]
pub fn ws_route(ws: WebSocket, hub: &State<Arc<RadioHub>>) -> Channel<'static> {
    let hub = hub.inner().clone();

    ws.channel(move |mut stream| {
        Box::pin(async move {
            let mut fanout_rx = hub.subscribe();

            // Immediately tell this (and everyone already connected) what the
            // radio's actual state is: started?, tuning, LNA, plus the last
            // discovered devices. Without this, a tab that opens while the radio
            // is already running sees the spectrum stream but has no
            // authoritative `started` / `tuning` to enable Stop/Tune/SetLNA or to
            // mirror the frequency field.
            hub.push_welcome().await;

            loop {
                tokio::select! {
                    biased;

                    res = stream.next() => {
                        if let Some(Err(e)) = res.as_ref() {
                            eprintln!("[WS] stream err: {e}");
                        }
                        match res {
                            Some(Ok(Message::Text(json))) => {
                                if let Some(env) = parse_envelope(&json) {
                                    let hub2 = hub.clone();
                                    let cmd = env.cmd.clone();
                                    tokio::spawn(async move {
                                        hub2.handle_cmd(env.id, &cmd).await;
                                    });
                                } else if std::env::var("HL2_DEBUG").is_ok() {
                                    eprintln!("[WS] unparseable text: {json}");
                                }
                            }
                            Some(Ok(_)) => {}
                            _ => break,
                        }
                    }

                    ev = fanout_rx.recv() => {
                        match ev {
                            Ok(ev) => {
                                if stream.send(to_message(&ev)).await.is_err() {
                                    break;
                                }
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                                if std::env::var("HL2_DEBUG").is_ok() {
                                    eprintln!("[WS] fanout lagged by {n}, skipping");
                                }
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                        }
                    }
                }
            }
            Ok(())
        })
    })
}
