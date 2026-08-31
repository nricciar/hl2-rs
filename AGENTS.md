# AGENTS.md

Guidance for agents (and humans) working in this repo.

## What this is

A software-defined radio (SDR) stack for the **Hermes-Lite 2** (HL2)
software-defined radio. It speaks the Metis / openHPSDR "protocol 1" over
plain UDP and exposes the radio over a WebSocket.

## Workspace layout

Cargo workspace (root `Cargo.toml`), five crates:

| Crate    | Path    | Role                                                                    |
|----------|---------|-------------------------------------------------------------------------|
| `hl2`    | `hl2/`  | Protocol codec + high-level client (discovery, start/stop, tune, receive pump). |
| `hl2-common` | `common/` | Serde wire types + WS contract shared by server and browser.           |
| `hl2-api`| `api/`  | Rocket + WebSocket server; owns one radio; fans out a spectrum stream; PSK Reporter sender. |
| `hl2-ui` | `ui/`   | Yew/WASM UI (panadapter + waterfall). Built with **Trunk**.            |
| `pskrep` | `pskrep/` | PSK Reporter client: IPFIX wire encoder + spot queue/dedup. No deps.  |

## Read PROTOCOL.md

`PROTOCOL.md` is the single source of truth for:

1. The HL2 wire protocol (packet classes, endpoints, C0–C4, memory map,
   discovery, wideband data, I2C/EEPROM/bias).
2. A **protocol → code** map. Nearly every protocol fact is followed by one or
   more `→ file:line` / `→ symbol` pointers, e.g. `hl2/src/protocol/data.rs:126`.
   Start there when you need to find *where* a given byte or command is built
   or parsed.

When you change protocol behavior, **update PROTOCOL.md** to keep the map
accurate. When in doubt, read the `→` pointers beside the relevant protocol
concept to see exactly which code implements it before touching that code.

## Layering rule

The protocol bytes live **only** in `hl2/`. `hl2-common/` is the contract
between `hl2-api/` and `hl2-ui/` and never links `hl2/` (the browser build
only wants serde wire types). `hl2-api/` and `hl2-ui/` speak that contract.

## Build / test

```sh
# build the server + protocol + common (skips the WASM ui)
cargo build -p hl2 -p hl2-common -p hl2-api

# run the protocol + common unit tests
cargo test -p hl2 -p hl2-common

# run the integration tests against a real HL2 (requires HL2_IP / HL2_ADDR env)
cargo test -p hl2 --features integration-tests

# build + serve the WASM UI
cd ui && trunk build --release    # (or: cargo build for a quick check)

# build a SELF-CONTAINED hl2-api binary that serves the UI from the same port.
# Trunk must build ui/dist FIRST (rust-embed inlines it at compile time);
# the `embed-ui` feature adds the `GET /` + `GET /<path>` routes in
# api/src/web.rs. Default builds are unchanged (WS-only).
cd ui && trunk build --release
cargo build --release -p hl2-api --features embed-ui
# then: ./target/release/hl2-api  →  open http://localhost:8000/
```

## Releases

- A `v*` tag (e.g. `v0.1.0`) triggers `.github/workflows/release.yml`,
  which builds the UI with Trunk, compiles `hl2-api` with `--features
  embed-ui`, and publishes a
  `hl2-api-<ver>-x86_64-unknown-linux-gnu.tar.gz` to the GitHub Release.
- `embed-ui` is **opt-in**; the ordinary `hl2-api` build stays WS-only so
  dev builds never require Trunk or a pre-existing `ui/dist`.

## Key environment variables

| Var           | Used by       | Meaning                                  |
|---------------|---------------|------------------------------------------|
| `HL2_ADDR`    | `hl2-api`      | IP of the HL2 to start (default auto-discover). |
| `HL2_IP`      | `hl2` (tests)  | IP for integration tests.                |
| `HL2_DEBUG`   | all           | Set to any value to enable debug `eprintln!` traces. |
| `PSK_CALL`    | `hl2-api`     | Our callsign; **sets this (non-empty) enables** PSK Reporter spot posting. `PSK_GRID`, `PSK_ANTENNA`, `PSK_RIG` fill the receiver-info record. |
| `PSKREP_ADDR` | `hl2-api`     | Collector endpoint override (default `report.pskreporter.info:4739`). |

## Conventions

- Protocol constants are `pub const`s in `hl2/src/protocol/mod.rs` (or the
  relevant sub-module). Add a new wire constant there, with a comment citing
  the PROTOCOL.md section or openHPSDR register it maps to.
- All binary fields are little-endian on the WebSocket contract; big-endian
  in the HL2 wire framing (see PROTOCOL.md §3).
- `hl2-common` is `no_std`-eligible (feature-gated `std`); avoid adding
  `std`-only deps there.
- Yew 0.23, CSR-only. UI state lives in a `Rc<Shared>` (see `ui/src/app.rs`);
  keep it single-threaded (`Rc`/`RefCell`).
- The UI is Trunk-built; `ui/index.html` is the entry, `ui/src/main.rs` the
  browser `main()`, `ui/dist/` is build output.

## Do not

- Do not add protocol logic to `hl2-api/` or `hl2-ui/` — keep it in `hl2/`.
- Do not add `std`-only types to `hl2-common/`.
- Do not change wire byte layouts without updating PROTOCOL.md and the tests.
- `.gitignore` covers `/target` and `/ui/dist` (build artifacts); keep them
  out of commits. `Cargo.lock` **is** committed — keep it committed.
