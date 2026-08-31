# hl2-rs

![Screenshot of hl2-api decoding 40m FT8, JS8 and FT4 signals](https://i.imgur.com/VwVrjqi.png)

Hermes Lite 2 SDR client implemented in Rust. Project is a work in progress with known bugs and limitations [see issues].

## Workspace

| Crate    | Path    | Role                                                                    |
|----------|---------|-------------------------------------------------------------------------|
| `hl2`    | `hl2/`  | high-level client (discovery, start/stop, tune, receive pump). |
| `hl2-common` | `common/` | Serde wire types + WS contract shared by server and browser.           |
| `hl2-api`| `api/`  | Rocket + WebSocket server; Provides API access to hl2 radio |
| `hl2-ui` | `ui/`   | Yew/WASM UI.            |
| `pskrep` | `pskrep/` | PSK Reporter client + spot queue/dedup.  |
| `teensy` | `teensy/` | [TODO] Teensy hl2 client/ui |

## Supported Modes

* SSB (USB/LSB)
* FT8 - mfsk-core provided ft8 decoder
* FT4 - mfsk-core provided ft4 decoder
* JS8Call - `hl2/src/receiver/js8` decoder based on js8call

## Usage

```
PSK_CALL="MYCALL" PSK_GRID="XXYYzz" PSK_ANTENNA="dipole" PSK_RIG="Hermes Lite 2"  ./hl2-api
```

* PSKReporter spotting will be enabled if `PSK_CALL` is set
* Additional debugging can be enabled by providing `HL2_DEBUG=1` during startup

## AI SLOP WARNING

Most of the code in this repo was auto generated with Qwen 3.8 27b and OpenCode. Fair warning.

## TODO

* Modes [CW, FM, AM, NFM, and more]
* TX
* Teensy support
