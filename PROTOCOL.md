# Hermes-Lite 2 — Protocol & Implementation Reference

> **Two audiences, one document.**
>
> 1. **The protocol** — what the Hermes-Lite 2 speaks over the wire, in
>    openHPSDR / Metis ("protocol 1") terms.
> 2. **The implementation** — where each byte of that protocol is built,
>    parsed, and driven in this repo, with `file:symbol` pointers you can
>    jump straight to.
>
> If a section shows `→` after a sentence, that line is the code that makes the
> sentence true. Everything in code paths is Rust (`hl2`, `hl2-common`,
> `hl2-api`) or Yew/WASM (`hl2-ui`).

This implementation targets a **core subset** of the openHPSDR protocol so that
a Hermes-Lite 2 (Board_ID `0x06`) can be driven by standard openHPSDR software,
and so this repo can talk to it over plain UDP. The sections marked
**[full spec]** describe behavior the hardware defines but this code does not
yet exercise — they are kept verbatim from the openHPSDR reference so the
document stays a complete, self-contained reference.

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [Implementation map](#2-implementation-map)
3. [Transport & wire framing](#3-transport--wire-framing)
4. [Discovery](#4-discovery)
5. [Start / Stop](#5-start--stop)
6. [Command & control (C0–C4)](#6-command--control-c0c4)
7. [Base memory map (host → radio)](#7-base-memory-map-host--radio)
8. [Watchdog & keep-alive](#8-watchdog--keep-alive)
9. [Responses (radio → host)](#9-responses-radio--host)
10. [Wideband data (EP4)](#10-wideband-data-ep4)
11. [Extended & advanced registers](#11-extended--advanced-registers) *(full spec)*
12. [Discovery reply packet (full layout)](#12-discovery-reply-packet-full-layout)
13. [Higher layers: API server & UI](#13-higher-layers-api-server--ui)
14. [Constants reference](#14-constants-reference)
15. [Open questions & TODOs](#15-open-questions--todos)
16. [Virtual receiver (audio channel demod)](#16-virtual-receiver-audio-channel-demod)

---

## 1. Repository layout

This is a Cargo workspace (see root `Cargo.toml`):

| Crate        | Name in `Cargo.toml` | What it is                                                    |
|--------------|----------------------|---------------------------------------------------------------|
| `hl2/`       | `hl2`                | The protocol codec **and** the high-level client (discovery / start / stop / tune / receive pump). |
| Common types | `hl2-common`         | Serde wire types shared by the server **and** the browser, plus the WebSocket channel/binary-frame contract. |
| `api/`       | `hl2-api`            | Rocket + WebSocket server that owns one radio and fans out a spectrum stream. |
| `ui/`        | `hl2-ui`             | Yew/WASM proof-of-concept: panadapter + waterfall. Built by **Trunk**. |

Only `hl2` knows the radio bytes. `hl2-common` is the contract between server
and client so the browser never links the protocol crate. `hl2-api` and
`hl2-ui` both speak that contract.

---

## 2. Implementation map

The fastest way into the code, by protocol concept:

| Protocol concept                         | Where it lives                                                        |
|------------------------------------------|-----------------------------------------------------------------------|
| Marker, port, board ID, all constants     | `hl2/src/protocol/mod.rs`                                            |
| Metis discovery request / reply           | `hl2/src/protocol/discovery.rs` → `discovery_request`, `parse_discovery_response` |
| Start / Stop command byte                 | `hl2/src/protocol/command.rs` → `StartCommand`; `hl2/src/protocol/data.rs` → `build_start_stop_frame` |
| C0–C4 build / parse (RQST, MOX, ACK, PTT)| `hl2/src/protocol/command.rs` → `CommandHeader`, `CommandData`, `ResponseHeader` |
| 1032-byte frame build/parse, IQ, assembler| `hl2/src/protocol/data.rs` → `build_keepalive_packet`, `build_nco_packet`, `parse_receive_packet`, `BlockAssembler` |
| High-level `Hl2` handle + receive pump    | `hl2/src/hl2.rs` → `Hl2::start/stop/tune`, `run_loop`, `discover`    |
| WebSocket command/status contract         | `common/src/lib.rs` → `ClientCmd`, `ServerResponse`, `SharedState`, `SpectrumFrame` |
| Server orchestration + FFT fan-out        | `api/src/hub.rs` → `RadioHub`, `run_spectral`                        |
| WebSocket route                           | `api/src/ws.rs` → `ws_route`                                          |
| Browser client + rendering                | `ui/src/app.rs`, `ui/src/client.rs`, `ui/src/canvas.rs`               |

---

## 3. Transport & wire framing

All HL2 traffic is **UDP on port 1024** (→ `HL2_PORT`, `hl2/src/protocol/mod.rs:11`).
Every datagram begins with the two-byte Metis marker `0xEF 0xFE`
(→ `METIS_MARKER`, `mod.rs:8`). Everything after the marker is described by the
**byte[2] discriminator** and, for data frames, the **endpoint** byte.

This is the framing this repo actually implements. The rest of this document
describes the *payload semantics* inside each frame.

### 3.1 Packet classes (byte[2])

| byte[2] | Class                     | Size   | Shape                                                                                     |
|:-------:|---------------------------|--------|-------------------------------------------------------------------------------------------|
| `0x01`  | **Data frame**            | 1032 B | marker, `0x01`, `endpoint`, `seq[4] BE`, then `1024 B` of payload = **2 × 512-byte chunks** |
| `0x02`  | **Discovery** request     | 63 B   | marker, `0x02`, then 60 × `0x00`                                                          |
| `0x04`  | **Start / Stop** command  | 64 B   | marker, `0x04`, `Command`, `seq[4]`, then zeros                                           |

> Note the byte[2] value plays three different roles: `0x01`/`0x04` mark a
> packet *class*, while a discovery **reply** reuses byte[2] as the *status*
> byte (`0x02` idle / `0x03` streaming — see §12).
>
> Sizes match the constants `DATA_PACKET_SIZE=1032`,
> `DISCOVERY_REQUEST_SIZE=63`, `START_REQUEST_SIZE=64`
> (→ `hl2/src/protocol/mod.rs:17-20`).

### 3.2 Data-frame header (8 bytes)

```
offset  value            meaning
0,1    EF FE            Metis marker
2      0x01             "data frame" class
3      endpoint         0x02 C&C, 0x04 wideband IQ, 0x06 C&C response/ACK
4..8   seq (u32 BE)     host↔radio sequence counter
8..    payload          2 × 512-byte chunks
```

→ header shape: `DataHeader` + `parse_data_header`,
`hl2/src/protocol/data.rs:41-54`
→ endpoint constants: `ENDPOINT_CONTROL/WIDEBAND/DATA_TX`, `mod.rs:30-32`.

### 3.3 Chunks and the C&C sync prefix

A 1032-byte data frame splits into two **512-byte chunks**
(→ `CHUNK_SIZE`, `mod.rs:21`).

* **Endpoint `0x04` (wideband):** each chunk is raw 16-bit IQ samples —
  256 per chunk, 512 per frame. (→ `SAMPLES_PER_CHUNK=256`, `mod.rs:36`;
  `extract_iq_from_chunk`, `data.rs:57`.)
* **Endpoint `0x02` / `0x06` (C&C):** the C0–C4 word does **not** start at the
  chunk head. It is preceded by a **3-byte sync `0x7F 0x7F 0x7F`**, then C0,
  C1, C2, C3, C4 — so C0 sits at chunk offset **3** (frame offset **11**).
  (→ `EP6_SYNC_LEN=3`, `data.rs:22`; `parse_chunk_command`, `data.rs:67`;
  `C_SYNC=0x7F`, `mod.rs:60`.)

```
chunk (512 B), C&C frame:
  [0x7F][0x7F][0x7F] [C0][C1][C2][C3][C4] [......]
     sync ×3          |
                      +-- C0..C4: the command/control word (see §6)
```

This three-byte `0x7F` prefix is part of the Metis-over-UDP dialect and is what
lets the parser distinguish C&C frames from IQ without trusting the endpoint
byte alone.

### 3.4 Endpoints

| Endpoint | Constant            | Direction      | Payload                                   |
|:--------:|---------------------|----------------|-------------------------------------------|
| `0x02`   | `ENDPOINT_CONTROL`  | host → radio   | C&C / keep-alive / register writes         |
| `0x04`   | `ENDPOINT_WIDEBAND` | radio → host   | 16-bit IQ wideband samples                 |
| `0x06`   | `ENDPOINT_DATA_TX`  | radio → host   | C&C responses / ACK / status               |

---

## 4. Discovery

OpenHPSDR Discovery is unchanged from the openHPSDR specification.

**Request (63 B):** `0xEF 0xFE 0x02` + 60 × `0x00`.
→ `discovery_request()`, `hl2/src/protocol/discovery.rs:29`.

**Reply (60 B):** extended Metis reply carrying MAC, gateware version, board
ID, receiver count, sample depth, fixed IP/MAC, and a live telemetry block.
The full byte layout is in [§12](#12-discovery-reply-packet-full-layout).
→ `parse_discovery_response()`, `discovery.rs:39`; result type
`DiscoveryInfo`, `discovery.rs:6` (mirrored as a serde type in
`common/src/lib.rs:49`).

**Sweeping the LAN:** the high-level `discover()` broadcasts the request and
also unicasts to a couple of known addresses, collecting replies for ~3 s
(→ `hl2/src/hl2.rs:340`); `discover_single()` targets one address
(→ `hl2.rs:390`).

> **[full spec]** Discovery reply repurposes the openHPSDR Metis reply so that
> a core subset of openHPSDR software can still find and identify the board.

---

## 5. Start / Stop

The openHPSDR/Metis Start and Stop frames carry a single **Command** byte
(→ `StartCommand`, `hl2/src/protocol/command.rs:103`; the bit layout and the
watchdog-disable bit[7]).

Command bits:

| Bit | Meaning                                              |
|:---:|------------------------------------------------------|
| [0] | 1 = radio on / running (Start), 0 = stopped          |
| [1] | 1 = wideband data streaming on, 0 = off             |
| [7] | 1 = **disable** the internal watchdog timer          |

The watchdog forces the host to keep sending control frames or the HL2 drops
back to "waiting". Disabling it is useful for receive-only programs (e.g. a CW
skimmer). This repo **keeps the watchdog enabled** (`watchdog_disabled=false`)
and instead sends periodic keep-alives — see [§8](#8-watchdog--keep-alive).

**Frame (64 B):** `0xEF 0xFE 0x04 <Command> <seq[4]> <zeros…>`.
→ `build_start_stop_frame(seq, start)`, `hl2/src/protocol/data.rs:111` —
on Start it sets bits [0]|[1]; on Stop it clears both.

The actual Start handshake — send STOP and drain stale frames, send START, wait
for the first acknowledgement datagram, drain a few more, then start the pump —
lives in `Hl2::start()`, `hl2/src/hl2.rs:106`. `Hl2::stop()` sends the Stop
frame, `hl2.rs:290`.

> **[full spec]** See openHPSDR for the original protocol 1 `Command`/`C&C`
> semantics that Start/Stop sit on top of.

---

## 6. Command & control (C0–C4)

A register write/read is a 5-byte word, **C0–C4**, placed after the `0x7F`
sync prefix inside a chunk (see [§3.3](#33-chunks-and-the-cc-sync-prefix)).

* **C0** — addressing & control: `[7]=RQST`, `[6:1]=ADDR[5:0]`, `[0]=MOX`.
* **C1–C4** — the 32-bit `DATA` word (MSB→LSB).

`RQST`: when set, the HL2 responds with an ACK'd result (see
[§9](#9-responses-radio--host)); when clear, it cycles the classic responses.
`MOX`: transmit direction (1 = active).

→ build: `CommandHeader::write/read` + `into_byte`, `command.rs:14-41`
→ encode word: `CommandData::encode`, `command.rs:87`
→ parse word back: `CommandData::parse`, `command.rs:70`

> **[full spec]** This is the openHPSDR "Command & Control" word; the HL2
> reinterprets bit `C0[7]` as the `RQST` request bit (see §9 for how requests
> and classic responses differ).

---

## 7. Base memory map (host → radio)

The HL2 keeps a 64-word memory map; the first 64 addresses overlap openHPSDR's
128-address space. Only a handful are used by this repo (notably the NCO
frequency writes, see the note at the end of this section). The
openHPSDR-defined addresses 0x00–0x11 are the
"stable" subset.

| ADDR | DATA | Description |
|-----:|------|-------------|
| `0x00` | `[25:24]` | Speed (`00`=48k, `01`=96k, `10`=192k, `11`=384k) |
| `0x00` | `[23:17]` | Open-collector outputs (filter selection) |
| `0x00` | `[13]`   | Rx antenna |
| `0x00` | `[12]`   | FPGA power-supply switching clock (0=on) |
| `0x00` | `[11]`   | Fan / band-volts PWM (0=Fan) |
| `0x00` | `[10]`   | VNA fixed RX gain (0=−6 dB, 1=+6 dB) |
| `0x00` | `[6:3]`  | Number of receivers (1–12) |
| `0x00` | `[2]`    | Duplex (0=off, 1=on) |
| `0x01` | `[31:0]` | **TX1 NCO frequency, Hz** |
| `0x02` | `[31:0]` | **RX1 NCO frequency, Hz** |
| `0x03` | `[31:0]` | RX2 NCO frequency, Hz |
| `0x04` | `[31:0]` | RX3 NCO frequency, Hz |
| `0x05` | `[31:0]` | RX4 NCO frequency, Hz |
| `0x06` | `[31:0]` | RX5 NCO frequency, Hz |
| `0x07` | `[31:0]` | RX6 NCO frequency, Hz |
| `0x08` | `[31:0]` | RX7 NCO frequency, Hz |
| `0x09` | `[31:24]` | Hermes TX drive level |
| `0x09` | `[23]`   | VNA mode |
| `0x09` | `[22]`   | Alex manual mode |
| `0x09` | `[20]`   | Tune request (ATU) |
| `0x09` | `[19]`   | Onboard PA on/off |
| `0x09` | `[18]`   | Force Rx T/R relay |
| `0x09` | `[17]`   | Tune: bypass vs tune |
| `0x09` | `[15:8]` | Alex Rx filter (or VNA count MSB) |
| `0x09` | `[7:0]`  | Alex Tx filter (or VNA count LSB) |
| `0x0a` | `[22]`   | PureSignal on/off |
| `0x0a` | `[6]`    | LNA gain mode (see §11.3) |
| `0x0a` | `[5:0]`  | LNA[5:0] gain |
| `0x0e` | `[15]`   | Enable HW-managed LNA gain for TX |
| `0x0e` | `[14]`   | see §11.3 |
| `0x0e` | `[13:8]` | LNA[5:0] gain during TX if enabled |
| `0x0f` | `[24]`   | Enable CWX keydown |
| `0x10` | `[31:16]`| CW hang time |
| `0x12`–`0x16` | `[31:0]` | RX8–RX12 NCO, Hz |
| `0x17` | `[12:8]` | PTT hang time (ms) |
| `0x17` | `[6:0]`  | TX buffer latency (ms) |
| `0x2b` | `[31:24]/[19:16]` | Predistortion index / enable |
| `0x39` | [see table] | Misc / Master / Sync / Clock generator commands |
| `0x3a` | `[0]`    | Reset HL2 on disconnect |
| `0x3b` | `[31:24]` | AD9866 SPI cookie (must be `0x06`) |
| `0x3c` | [see §11.1] | I2C1 bus (cookie, addr, control, data) |
| `0x3d` | [see §11.1] | I2C2 bus (cookie, addr, control, data) |
| `0x3f` | `[31:0]` | Error responses (RADDR `0x3F`) |

**NCO writes in this repo.** `build_nco_packet(seq, slot, freq, c1, oc, n_recv)`
 writes a 32-bit frequency to a slot (→ `hl2/src/protocol/data.rs`). `slot` is
 1-based (RX1 = 1) and maps to the memory-map NCO register above (`0x02`=RX1,
 `0x03`=RX2, …). Because C0 encodes the address left-shifted
 (`C0 = ADDR << 1`), the byte is `C0 = (0x01 + slot) << 1`, i.e.
 RX1 → `0x04`, RX2 → `0x06`, RX3 → `0x08` … RX7 → `0x10`. The big-endian
 frequency goes in C1–C4 **of the second 512-byte chunk**. Slot constants
 `TX1_ADDR…RX7_ADDR` are in `hl2/src/hl2.rs:33-40`, exposed as `RX1_ADDR` etc.
 via `hl2/src/lib.rs`.

 > ✅ **Confirmed — C0 mapping.** This was previously ambiguous (an earlier draft
 > used `0x04 + slot*2`, which put RX1 at `0x06` — RX2's register). It now
 > follows the memory map above (the reference sets
 > `output_buffer[C0]=0x04+(current_rx*2)` with a 0-based `current_rx`, so RX1
 > → `0x04`). Locked by
 > `build_nco_packet_slot_to_c0_table` in `hl2/src/protocol/data.rs`.

 > ✅ **Confirmed — two-chunk commit handshake (the "tune doesn't move" bug).**
 > The reference emits **every** C&C register write as *two 512-byte chunks in
 > one 1032-byte frame*: chunk 1 is the baseline (`C0=0x00`,
 > `C1=CONFIG_BOTH|SPEED`, `C2=OC`, `C4=0x04 dup | receivers<<3`) and chunk 2
 > carries the NCO register write. The reference fills chunk 1, then chunk 2,
 > then sends. The gateware only *commits* the write when it arrives in a frame
 > that also carries the baseline — *"CONFIG_BOTH seems to be critical to
 > getting ozy to respond."* Our previous builder put the NCO
 > write in **both** chunks (no baseline anywhere), so tunes were received but
 > not committed and the DDC stayed latched at whatever the *last* valid client
 > had written. `build_nco_packet` now emits the baseline in
 > chunk 1 and the write in chunk 2. Locked by
 > `build_nco_packet_baseline_plus_write_two_chunks`.

---

## 8. Watchdog & keep-alive

The HL2 watchdog reverts the board to "waiting" if it stops hearing control
frames (the spec cites ~168 ms; this repo uses a comfortable margin).

→ `KEEPALIVE_INTERVAL_MS = 40`, `hl2/src/protocol/mod.rs:49`
→ keep-alive frame: `build_keepalive_packet(seq, c1_speed_bits, oc_bits, n_recv)`,
  `hl2/src/protocol/data.rs` — an EP2 frame (both chunks) with the `0x7F` sync
  prefix, `C0=0x00`, `C1 = CONFIG_BOTH(0x60) | SPEED bits`,
  `C2 = (oc_bits & 0x7F) << 1`, `C4 = 0x04 (duplex) | ((n_recv-1) << 3)`.
  `CONFIG_BOTH` is what the reference calls "critical to getting the board to
  respond" (→ `C1_CONFIG_BOTH`, `mod.rs:56`); the C4 duplex + receiver-count
  bits mirror the reference (`output_buffer[C4]=0x04; ... output_buffer[C4]|=nreceivers<<3`).
  This baseline chunk is shared with `build_nco_packet` /
  `build_lna_gain_frame` (both put it in chunk 1), via
  `build_baseline_chunk` (→ `hl2/src/protocol/data.rs`).
→ the RX open-collector filter relay mask (if any) is re-asserted in every
  keep-alive's `C2[7:1]` — `Hl2::set_oc_bits()` writes an LSB-first 7-bit
  mask to `Shared.oc_bits`, the pump reads it each tick, and
  `build_keepalive_packet` encodes it into `C2` (→ `hl2.rs::Shared::oc_bits`,
  `protocol/mod.rs::OC_MASK_RX`, `data.rs::build_keepalive_packet`). This is
  the path that drives the HL2's companion filter board — see §11.4.
→ the pump emits one every tick: `run_loop`, `hl2/src/hl2.rs`.

---

## 9. Responses (radio → host)

A radio→host C&C frame (EP6) carries C0–C4 with C0[7]=`ACK`.

* **ACK=0 — classic response:** `C0[6:3]=RADDR[3:0]`, `C0[2]`=CW dot,
  `C0[0]`=PTT. The HL2 "cycles through" a small classic response memory map
  (RADDR cycles `0x00 → 0x01 → 0x02` on successive frames):
  `0x00` = Tx-inhibited / RF overload / under-overflow / TX FIFO count /
  firmware version; `0x01` = temperature + forward power;
  `0x02` = reverse power + current. See the original PROTOCOL.md
  "Base Memory Map when ACK==0" for the exact bit layout.
* **ACK=1 — request response:** `C0[6:1]=RADDR[5:0]`. For writes the gateware
  echoes `RADDR=ADDR, RDATA=DATA`; for I2C reads it returns 4 bytes of I2C
  data. `RADDR=0x3F` signals an error.

→ the whole response word (ACK, RADDR, PTT, CW dot, 32-bit data) is parsed by
  `CommandData::parse` into `ResponseHeader`, `command.rs:49-97`.
→ `run_loop` lifts these into `Hl2Event::CmdAck { ack, raddr, ptt, data }`
  (`hl2/src/hl2.rs:54-65`, emitted at `hl2.rs:442-451`).

> **[full spec]** To avoid starving the classic responses, software should send
> a `RQST`-set command only periodically (e.g. every other write) and never more
> than one outstanding request at a time (the next `RQST` must wait for the
> previous ACK).

---

## 10. Wideband data (EP4)

The ADC samples at **76.8 MHz**; a 2048-sample **block** spans the 0–38.4 MHz
HF band. Each 1032-byte EP4 frame carries 512 16-bit samples (two 256-sample
chunks) — so a **block is four consecutive EP4 frames**. Samples are continuous
within a block and discontinuous across blocks.

→ `IQ_PAIRS_PER_BLOCK = 2048`, `mod.rs:42`; `SAMPLES_PER_CHUNK = 256`,
  `mod.rs:36`; `extract_iq_from_chunk`, `data.rs:57`.
→ the `BlockAssembler` accumulates samples and emits a 2048-sample
  `IQBlock` once enough have arrived (→ `data.rs:223-257`); `IQBlock` is
  `data.rs:155`.
→ `run_loop` feeds the assembler on every EP4 frame and re-emits completed
  blocks as `Hl2Event::Block` (`hl2.rs:453-459`).

**Sample depth** is reported by discovery (0x14 bits `[7:6]`: `00`=12-bit
sign-extended, `01`=16-bit) and is surface-only in this repo — the wire is
always 16-bit little-endian. → `SampleFormat`, `data.rs:27`; detection in
`parse_discovery_response`, `discovery.rs:46`.

> ⚠ The spec says to detect a block boundary by "the two least significant
> bits of the sequence number == 0." This repo does **not** use the sequence to
> align blocks — it simply accumulates 2048 consecutive samples. Fine while
> frames arrive losslessly in order; see [§15](#15-open-questions--todos).

---

## 11. Extended & advanced registers

Everything in this section is defined by the hardware / openHPSDR reference.
Current status vs. this repo: **discovered/read where noted; otherwise
"defined, not yet exercised."**

### 11.1 I2C buses (0x3c / I2C1, 0x3d / I2C2) *(full spec)*

Two I2C buses are addressable. A **read** requires `RQST` set (`C0[7]=1`):
byte[1]=`0x07` (read), then 7-bit device address (+stop bit), then register,
then a reserved byte. The HL2 reads four bytes off the device and returns them
in C1–C4, with `ACK` set and `RADDR` matching the bus address. Writes are the
same but byte[1]=`0x06` and the final byte is the data (one-byte writes only).

→ The response side is parsed by `CommandData::parse` (§9). The *build* side of
I2C requests is currently **[defined, not yet exercised]** in code.

The HL2 also uses I2C1 (address `0x20`) to talk to a companion filter board
(see §11.4). The MCP4662 PA-bias device and the configuration EEPROM live on
I2C2 (§11.2, §11.5).

### 11.2 Bias (PA) — I2C2 / MCP4662 *(full spec)*

Bias is an 8-bit value (`0xFF` lowest → `0x00` highest, 256 linear steps).
Volatile vs. nonvolatile registers:

| Command                         | I2C2 word   |
|---------------------------------|-------------|
| Set Bias0 **volatile**           | `0x06ac00vv` |
| Set Bias0 **non-volatile**       | `0x06ac20vv` |
| Set Bias1 **volatile**           | `0x06ac10vv` |
| Set Bias1 **non-volatile**       | `0x06ac30vv` |

Nonvolatile values load at next power cycle. On beta2-era boards the middle
byte is `0xa8` instead of `0xac`.

### 11.3 LNA gain *(full spec)*

`0x0a bit[6]` selects the mode:

* **Set:** LNA[5:0] is passed straight to the AD9866 (full −12 dB … +48 dB).
* **Clear (Hermetic back-compat):** LNA[4:0] maps to 32 steps of attenuation
  matching the Hermes; bit[5] selects the step attenuator (off ⇒ +20 dB default).

**In this repo (we use "Set" mode).** The LNA is *static*: there is no RX AGC,
so we program it once (and re-assert at START) with a fixed dB value. The write
is a C&C register write on address `0x0a` — `C0 = LNA_ADDR << 1 = 0x14`
(MOX=0), matching the reference case 4 (`C0=0x14`, `C4=0x40|gain`).
Only C4 carries the value:

```
C4 = 0x40 | ((gain_db + 12) & 0x3F)   // "Set" mode bit + 6-bit gain (−12…+48 dB)
```

→ `LNA_ADDR`, `LNA_MODE_SET`, `DEFAULT_LNA_GAIN_DB` — `hl2/src/protocol/mod.rs`
→ frame builder `build_lna_gain_frame(seq, gain_db, c1, oc, n_recv)` —
  `hl2/src/protocol/data.rs`; chunk 1 carries the shared baseline (commit
  handshake, see the §7 NCO note) and chunk 2 carries `C0=0x14`, `C4=0x40|…`.
  (unit-tested: `build_lna_frame_layout`, `build_lna_frame_gain_range`)
→ client method `Hl2::set_lna_gain` / `Hl2::lna_gain` — `hl2/src/hl2.rs:379`;
  applied automatically after START (so the LNA is active for the `ssb`/ALSA
  path) via the stored value in `Shared.lna_gain_db`.
→ default `+6 dB` (`DEFAULT_LNA_GAIN_DB`); overridable per run.

There is no *automatic* / signal-level AGC in the hardware or the software DSP
(`gain_db` in [`ReceiverConfig.audio`](hl2/src/receiver/mod.rs) is a separate,
post-reception playback scaler, not the LNA). Adding an AGC loop is a
follow-up: it would periodically rewrite this same register from measured
baseband level.

**WS control.** `ClientCmd::SetLnaGain { gain_db }` (wire name `setlnagain`) →
`RadioHub::set_lna_gain_cmd`; the current value is mirrored back in
`SharedState.lna_gain_db` and the UI exposes an LNA dB input + "Set LNA" button
(`ui/src/app.rs`).

### 11.4 Filter selection *(full spec)*

The HL2 sends one byte to I2C address `0x20` to drive a companion filter board
(e.g. N2ADR / MRF101). In Alex-compatible mode (manual-mode bit = 0) it
forwards the openHPSDR open-collector bits (data bits [6:0]) and the
Rx-antenna bit (bit [7]). In "Alex manual" mode (bit = 1, not yet implemented
in gateware) it sends the 8-bit "Alex Rx filter" in Rx and "Alex Tx filter" in
Tx.

**How the OC bits reach the board.** The host writes the RX open-collector
mask into **C2** of a host→radio C&C frame — specifically the keep-alive.
`C2[7:1] = OC1..OC7`, `C2[0]` reserved. The reference implementation does the
same (`output_buffer[C2] |= band->OCrx << 1`).
Because the keep-alive is re-sent every 40 ms, re-asserting the mask there is
what holds the board on the selected band between NCO tunes.

→ `build_keepalive_packet(seq, c1_speed_bits, oc_bits, n_recv)` (via
  `build_baseline_chunk`), `hl2/src/protocol/data.rs` —
  `chunk[5] = (oc_bits & 0x7F) << 1`.
→ `Hl2::set_oc_bits(u8)` writes the LSB-first 7-bit mask to
  `Shared.oc_bits` (→ `hl2/src/hl2.rs`), read by the pump each tick.
→ `OC_MASK_RX = 0x7F` (`hl2/src/protocol/mod.rs`).
→ `hl2 ssb … --filtermask <hex>` (`hl2/src/bin/main.rs`) exposes the mask on
  the CLI. LSB-first convention: **bit 0 = relay/checkbox 1** (the first
  OC option the board exposes), **bit 6 = relay/checkbox 7**. `0x00` = all
  relays off (board default, e.g. 160 m receive on the MRF101 board);
  `0x44` = relays 3 + 7 (the 7.074 MHz / 40 m case). Bit 7 and above are
  ignored by `build_keepalive_packet`.

**Note.** The OC bits select *which relay* on the board closes, i.e. which
hardware path (band-pass, low-pass, preamp, attenuator — whatever the board is
built for). The board's relay map is board-specific. `--filtermask` is a
proof-of-concept: it does not *choose* the filter; it *drives* a specific
relay by number.

### 11.5 Configuration EEPROM (MCP4662 10 × 9-bit words) *(full spec)* 

| Addr | Bits   | Description                          |
|-----:|--------|--------------------------------------|
| `0x00` | `[7:0]` | Volatile Wiper 0                    |
| `0x01` | `[7:0]` | Volatile Wiper 1                    |
| `0x02` | `[7:0]` | Non-volatile Wiper 0 (PA Bias0)     |
| `0x03` | `[7:0]` | Non-volatile Wiper 1 (PA Bias1)     |
| `0x04` | `[8:0]` | Volatile TCON                       |
| `0x05` | `[8:0]` | Status                              |
| `0x06` | `[7:5]` | Valid IP / Valid MAC / Favor DHCP flags |
| `0x07` | `[7:0]` | Reserved                            |
| `0x08`–`0x0B` | `[7:0]` | Fixed IP `W.X.Y.Z`               |
| `0x0C`–`0x0D` | `[7:0]` | MAC `Y`, `Z`                   |
| `0x0E`–`0x0F` | `[7:0]` | Reserved                        |

**Write:** `32-bit word 0x06acA0vv` to ADDR `0x3d` (or `0x7d` to request a
response), where `A` is the EEPROM address and `vv` the byte.
**Read:** with `RQST` set, send `0x07acACXX` (`XX`=don't-care) to ADDR `0x7d`;
the ACK response's RDATA carries the 4-byte I2C read (the MCP4662 stores 9-bit
data, so `[31:24]=v[7:0]`, `[16]=v[8]`, `[15:8]=v[7:0]`, `[0]=v[8]`).

→ The **read** path (replies come back as ACK'd C0–C4, §9) is handled by
  `CommandData::parse`. Writes, and reading the EEPROM to learn fixed IP/MAC,
  are **[defined, not yet exercised]**; discovery already surfaces IP and MAC
  (→ `parse_discovery_response`, `discovery.rs:52-58`).

---

## 12. Discovery reply packet (full layout)

The 60-byte reply. (→ `parse_discovery_response`, `discovery.rs:39` — reads the
MAC, versions, board ID, rx count, sample depth, IP. `is_sending` from byte[2],
`sample_16bit` from `byte[0x14] >> 6`.)

The serde wire type in `hl2-common` (`DiscoveryInfo`) carries every field
above **plus** two server-added fields that are *not* in the 60-byte reply:
`addr` (human-readable source address, e.g. `169.254.19.221:1024`) and
`in_service: bool` (`#[serde(default)]`). `in_service` is `true` only for the
radio the hub has currently started — see §13 "In-service radio is always
listed" for why the hardware does not respond to further discovery while in
use. Both fields are server-side annotations; a raw radio reply is exactly as
laid out below.

| Addr | Bits | Description |
|-----:|------|-------------|
| `0x00` | `[7:0]` | `0xEF` |
| `0x01` | `[7:0]` | `0xFE` |
| `0x02` | `[7:0]` | Status: `0x02` idle, `0x03` streaming |
| `0x03`–`0x08` | `[7:0]` | MAC `U:V:W:X:Y:Z` |
| `0x09` | `[7:0]` | Gateware **major** version |
| `0x0A` | `[7:0]` | Board ID (`0x06` HL2, `0x01` Hermes emulation) |
| `0x0B`–`0x0D` | `[7:0]` | MCP4662 config bits (`0x06`/`0x07`) + Fixed IP `0x08` |
| `0x0E`–`0x10` | `[7:0]` | Fixed IP `0x09`, `0x0A`, `0x0B` (→ `ip[]`) |
| `0x11`–`0x12` | `[7:0]` | MAC `0x0C`, `0x0D` |
| `0x13` | `[7:0]` | Number of hardware receivers |
| `0x14` | `[7:6]` | `00`=12-bit (sign-extended), `01`=16-bit wideband |
| `0x14` | `[5:0]` | Board build (5/3/2) |
| `0x15` | `[7:0]` | Gateware **minor**/patch |
| `0x16` | `[5:0]` | Reserved |
| `0x17`–`0x1A` | `[7:0]` | Response data `[31:0]` |
| `0x1B` | `[7]` | External CW key |
| `0x1B` | `[6]` | PTT (TX on) |
| `0x1B` | `[1:0]` | ADC clip count |
| `0x1C` | `[3:0]` | Temperature MSB |
| `0x1D` | `[7:0]` | Temperature LSB |
| `0x1E` | `[3:0]` | Forward power MSB |
| `0x1F` | `[7:0]` | Forward power LSB |
| `0x20` | `[3:0]` | Reverse power MSB |
| `0x21` | `[7:0]` | Reverse power LSB |
| `0x22` | `[3:0]` | Bias current MSB |
| `0x23` | `[7:0]` | Bias current LSB |
| `0x24` | `[7]` | Under/overflow recovery flag |
| `0x24` | `[6:0]` | TX IQ FIFO count MSBs |
| `0x26`–`0x3B` | | Reserved |

---

## 13. Higher layers: API server & UI

The protocol in §3–§12 lives entirely in the `hl2` crate. Two consumers sit on
top; they speak a **WebSocket** contract defined in `hl2-common`.

**`hl2-api`** (`api/`)
* Boot: Rocket + `Arc<RadioHub>` as `State` (`api/src/main.rs:13`).
* **`RadioHub`** is the single owner of the `Hl2` session, so many WS clients
  can drive one physical radio (`api/src/hub.rs:72`).
* Command dispatch (`Discover` / `Start` / `Stop` / `Tune` / `State`) →
  `RadioHub::handle_cmd` (`hub.rs:105`).
* **Spectrum pipeline:** `run_spectral` consumes `Hl2Event::Block`, accumulates
  `accumulate_blocks × 2048` samples (default 8 → 16384), applies a Hann
  window, runs a forward FFT (rustfft), max-pools the Nyquist half into
  `wideband_bins` (default 1024) linear magnitude bins, and scales to `u16`
  (each result is held as the `latest` frame, `hub.rs:394`).
* **Wideband coalescing (≤ 30 fps):** the pump produces a frame on *every*
  FFT, but `run_spectral` does **not** forward each one. A `tokio::time`
  ticker (`BROADCAST_INTERVAL_MS = 33`, `hub.rs:400`) re-broadcasts the most
  recent frame at most once per tick via `fanout.send(WsEvent::Wideband)`
   (`hub.rs:431`). This "latest-wins" cap bounds each subscriber to ~30
   frames/s (≈ 900 KB/s at 1024 bins) regardless of the SDR's block rate or
   the number of connected tabs; intermediate frames are dropped, never queued.
* Route: `/api/ws` (`api/src/ws.rs:47`), envelope `{id, cmd}` (`ws.rs:19`).
* **On-connect state snapshot ("welcome"):** when a WebSocket opens, the hub
  synchronously pushes a `ServerResponse` with `id = u64::MAX` (`WELCOME_ID`,
  `lib.rs:167`) carrying the current `SharedState`   (started?, tuning, LNA,
  sample format) and the last discovered device list
  (`RadioHub::push_welcome`, `hub.rs:115` → `ws.rs:61`). This lets a page
  that opens after the radio was already started (by another tab, or a
  previous page load) mirror the live settings without having to click
  Start again — its Stop/Tune/SetLNA buttons enable and the frequency field
  updates immediately.
* **Cross-tab state ordering:** every command — accepted or not — bumps a
  monotonic server-side revision (`RadioHub::state_rev`, `hub.rs:87`,
  `AtomicU64`, `fetch_add` before dispatch in `handle_cmd`, `hub.rs:138`) and
  stamps it into `SharedState::state_at` (`hub.rs:159`). The UI keeps a
  high-water mark (`last_state_at`, `ui/src/app.rs`) and applies a JSON
  response's embedded state only when `state_at >= last_state_at`
  (`ui/src/app.rs` `on_text`). This makes a late in-flight response to an
  older command (e.g. a `Start` ack still in transit while another tab is
  clicking `Stop`) unable to revert the fresher state, even though
  server→WS fan-out and browser event delivery are both unordered.
* **Spectrum source selection (EP4 vs EP6):** the panadapter/waterfall
  pipeline (`run_spectral`) consumes either the **EP4 wideband** real stream
  or the **EP6 per-slot complex baseband** — not both at once. The choice is
  a piece of shared state: `SharedState::spectrum_source` (`lib.rs:92`,
  enum `SpectrumSource { Ep4 | Ep6 { slot } }`, `lib.rs:126`), owned by the
  hub as `RadioHub::spectrum_source` (`hub.rs:88`) and set by the
  `SetSpectrumSource` command (wire name `setspectrumsrc`,
  `common` `ClientCmd::SetSpectrumSource`). `run_spectral` re-reads the
  current source on **every** pump event and only accumulates samples from
  the matching stream (`Block` when `Ep4`; `Baseband` whose `slot` matches
  when `Ep6`); a change bumps `RadioHub::spectrum_rev` so the pipeline
  discards any half-accumulated samples from the other stream before the next
  FFT. The setting is persisted on the hub (survives Stop/Start and is
  echoed in the on-connect welcome + every `SharedState`), so all tabs and a
  reloaded page agree on which stream feeds the scope. Note EP6 baseband is
   the *per-receiver DDC output* at the C1 SPEED option rate (48/96/192/384
   kHz), so an EP6 frame takes longer to accumulate than an EP4 one.
* **RX filter bank (OC relays):** the 7 open-collector filter-relay bits
   (which relay on the companion filter board closes — band-pass/low-pass/
   preamp/attenuator — see §11.4) are a piece of *shared radio state*, not a
   per-client setting. `SharedState::oc_bits` (a 7-bit LSB-first mask, bit 0 =
   relay 1 … bit 6 = relay 7, `lib.rs`) is owned by the hub via the `Session`
   mirror (`hub.rs`) and is read back from the running client at Start time
   (`ctrl.oc_bits()`) so a reloaded page sees the mask the radio is currently
   driving; it is persisted in the session and echoed in the on-connect
   welcome + every `SharedState` so all tabs agree. `ClientCmd::SetOcBits {
   oc_bits: u8 }` (wire name `setocbits`) applies a new mask — `RadioHub::
   set_oc_bits_cmd` (`hub.rs`) clones the `Hl2` out of the session, calls the
   already-existing `Hl2::set_oc_bits` (`hl2/src/hl2.rs`), which stores the
   mask in `Shared.oc_bits` where the keep-alive pump re-asserts it in the
   C2 byte every 40 ms tick (so the board holds the selected relay between NCO
   tunes). The UI renders a row of 7 checkboxes (one per relay) in the main
    controls panel; each box toggles its own bit and sends `setocbits`, and the
    server's ack (which echoes the new `SharedState.oc_bits`) re-checks the
    boxes and keeps every tab in sync — including after a tab reload (the
    welcome carries the current mask).
* **In-service radio is always listed** — a started HL2 does **not** respond to
   further discovery broadcasts (`is_sending` status bit is 0x03, see §12; the
   radio treats itself as busy), so a freshly-opened tab (or a tab that
   `discover`s after someone else already `Start`ed) would otherwise see
   "no radios discovered" and show the Start button greyed out even though
   the server is streaming right now. The hub snapshots the in-service
   radio's `DiscoveryInfo` at Start time (`Session.device`,
   `api/src/hub.rs`) — preferring the last `Discover` result for the matching
   IP to preserve the real MAC / gateware / board fields, else stubbing from
   the `Hl2::start` result (`board_id = BOARD_ID_HL2`, the IP + port, the
   actual `StartInfo.rx_count` + bit depth, `is_sending = true`) — and
   `merge_devices` (in `hub.rs`) unions it back in on every subsequent
   `Discover` and on the on-connect `welcome`, deduplicated by IP. The new
   wire field is `DiscoveryInfo.in_service: bool` (see §12 / `common/src/lib.rs`);
   it is `#[serde(default)]` so older UIs that don't know the field still
   decode the payload. Selection semantics in the UI are unchanged — the chip
   is purely a server-reported capability, not a user-action state (so the
   user can switch selection freely even when one radio is "in use").

**`hl2-ui`** (`ui/`, Trunk-built Yew app)
* WS wrapper: `WsClient` + `parse_binary` (`ui/src/client.rs`).
* App state + control buttons: `ui/src/app.rs` (`App`, `Shared`).
* Rendering: `draw_panadapter` / `paint_waterfall` / colormap
  (`ui/src/canvas.rs`).
* **Radio selector (was the "connected/disconnected" status pill):** the old
  `Discover` button + "Discovered devices" panel were removed in favour of a
  compact radio-selection row that sits directly below the `<h1>`. On every
  `onopen` the UI auto-sends `{"cmd":"discover"}` (see `Shared::on_open` in
  `ui/src/app.rs`), so no manual trigger is required. When `resp.devices`
  arrives, the UI renders one `<button>` per radio and records the selected
  IP in `Shared.active_device`; the first radio is auto-highlighted if none
  was selected yet on a fresh connection. Selection is purely a UI-side
  affordance (it gates the `Start` button), so it is **not** part of the
  wire contract — `hl2-common` / `hl2-api` are unchanged.
* **Offline greying:** the status pill (still present, but now a small inline
  dot + short text at the right of the selection row) reports the WS state,
  and the root `<div class="app">` picks up an `offline` class derived from
  the status string. `.app.offline .panel` dims all panels and disables
  pointer events (`ui/app.css`), so buttons / inputs / selects visually and
  behaviourally freeze while the socket is down; the heading, selection row
  and status dot stay readable so a user can always see *why* the panels are
  grey and can switch their selection while they wait.

The two-sided wire contract (channels, `ClientCmd`, `ServerResponse`,
`SpectrumFrame`, binary encode/decode) is in `common/src/lib.rs` — see
[§14](#14-constants-reference) for the channel IDs.

---

## 14. Constants reference

Protocol constants all live in `hl2/src/protocol/mod.rs` unless noted.

| Constant | Value | Meaning | Ref |
|----------|-------|---------|-----|
| `METIS_MARKER` | `EF FE` | Frame marker | `mod.rs:8` |
| `HL2_PORT` | `1024` | UDP port | `mod.rs:11` |
| `BOARD_ID_HL2` | `0x06` | HL2 board ID | `mod.rs:14` |
| `DISCOVERY_REQUEST_SIZE` | `63` | Discovery request | `mod.rs:17` |
| `DISCOVERY_RESPONSE_SIZE` | `60` | Discovery reply | `mod.rs:18` |
| `START_REQUEST_SIZE` | `64` | Start/Stop frame | `mod.rs:19` |
| `DATA_PACKET_SIZE` | `1032` | Data frame | `mod.rs:20` |
| `CHUNK_SIZE` | `512` | Payload chunk | `mod.rs:21` |
| `ENDPOINT_CONTROL` | `0x02` | C&C (host→radio) | `mod.rs:30` |
| `ENDPOINT_WIDEBAND` | `0x04` | IQ (radio→host) | `mod.rs:31` |
| `ENDPOINT_DATA_TX` | `0x06` | C&C response (radio→host) | `mod.rs:32` |
| `SAMPLES_PER_CHUNK` | `256` | IQ per chunk | `mod.rs:36` |
| `IQ_PAIRS_PER_PACKET` | `512` | IQ per frame | `mod.rs:37` |
| `IQ_PAIRS_PER_BLOCK` | `2048` | IQ per block (4 frames) | `mod.rs:42` |
| `ADC_CLOCK_HZ` | `76 800 000` | ADC rate | `mod.rs:45` |
| `KEEPALIVE_INTERVAL_MS` | `40` | Watchdog keep-alive | `mod.rs:49` |
| `C1_CONFIG_BOTH` | `0x60` | C1 TX+RX engines | `mod.rs:56` |
| `C1_SPEED_192K` | `0x02` | C1 192 kSps | `mod.rs:57` |
| `C_SYNC` | `0x7F` | C&C sync byte | `mod.rs:60` |
| `STATUS_NOT_SENDING` / `STATUS_SENDING` | `0x02` / `0x03` | Discovery status | `mod.rs:63-64` |
| `HEADER_SIZE` | `8` | Data-frame header | `data.rs:10` |
| `EP6_SYNC_LEN` | `3` | `0x7F` prefix length | `data.rs:22` |

WebSocket (server↔client) contract, in `common/src/lib.rs`:

| Item | Value / role | Ref |
|------|--------------|-----|
 | `CH_WIDEBAND` | `0x01` | Binary spectrum channel | `lib.rs:30` |
 | `CH_RX_IQ` | `0x02` | Reserved: per-receiver DDC IQ | `lib.rs:33` |
 | `CH_AUDIO` | `0x03` | **Active:** post-demod mono audio (see §16.7) | `lib.rs:35` |
  | `ClientCmd` | `Discover`/`Start`/`Stop`/`Tune`/`SetLnaGain`/`SetOcBits`/`SetSpectrumSource`/`SetVrx`/`SetVrxMute`/`SetVrxOff`/`State` | JSON request enum | `lib.rs:220` |
| `ServerResponse` | `{id, ack, error, state, devices}` | JSON reply envelope | `lib.rs:135` |
| `ServerResponse::welcome` / `WELCOME_ID` | push snapshot, `id = u64::MAX` | On-connect state broadcast | `lib.rs:176` |
| `SharedState.state_at` | monotonic server revision stamp | Cross-tab state ordering | `lib.rs:100` |
| `SharedState.oc_bits` | 7-bit LSB-first OC relay mask (bit 0 = relay 1 … bit 6 = relay 7) | Shared state: companion filter-board relay selection | `lib.rs` (§13) |
 | `SpectrumSource` | `Ep4 \| Ep6 { slot }` | Shared state: which stream feeds the FFT | `lib.rs:203` |
 | `ClientCmd::SetOcBits` | `setocbits { oc_bits }` | Drive companion filter-relays (re-asserted every keep-alive, §11.4) | `lib.rs` (see §13) |
| `ClientCmd::SetSpectrumSource` | `setspectrumsource { source }` | Switch EP4↔EP6 (takes effect next frame) | `lib.rs` (see §13) |
 | `SpectrumFrame` | `{seq_start, bin_count, mags}` | Binary spectrum payload | `lib.rs:326` |
  | `AudioFrame` | `{slot, seq, rate_hz, samples}` | Binary **audio** payload on `CH_AUDIO`; `slot` routes the block to a virtual receiver | `lib.rs:470` |
  | `VrxCfg` | `{slot, offset_hz, mode, bw_hz, gain_db}` | Requested virtual-receiver config | `lib.rs:201` |
  | `VrxState` | `{slot, offset_hz, mode, bw_hz, gain_db, rate_hz, muted}` | Echo of the active receiver (see `SharedState.vrx`) | `lib.rs:217` |
    | `VrxMode` | `Usb \| Lsb \| Am \| Ft8 \| Js8 \| Ft4` | Demod mode (renamed `usb`/`lsb`/`am`/`ft8`/`js8`/`ft4` on the wire) | `lib.rs` |
  | `SharedState.vrx` | `BTreeMap<u8, VrxState>` | Per-slot active virtual receivers (keyed by RX slot). Mirrored into the UI panel + spectrum passband overlay. Re-echoed on every `SharedState` like `oc_bits`, so all tabs converge. | `lib.rs:146` |
  | `ClientCmd::SetVrx` | `setvrx { cfg: Option<VrxCfg> }` | Create / update the virtual receiver on `cfg.slot`. `cfg: null` tears down that slot. | `lib.rs:345` |
  | `ClientCmd::SetVrxMute` | `setvrxmute { slot, muted }` | Mute / unmute a slot's receiver without rebuilding. | `lib.rs:353` |
  | `ClientCmd::SetVrxOff` | `setvrxoff { slot }` | Tear down a slot's receiver (explicit off). | `lib.rs:359` |
 | `encode_ws_binary` / `decode_ws_binary` | `u16 LE` channel + payload | Binary envelope | `lib.rs:385` / `lib.rs:393` |

Binary WS frame layout: `[u16 LE channel][payload]`.
`CH_WIDEBAND`  — `[u16 LE bin_count][u16 LE mags × bin_count]` (
`SpectrumFrame::to_bytes`, `lib.rs:443`).
`CH_AUDIO`     — `[u16 LE slot][u32 LE seq][u16 LE rate_hz][i16 LE samples × n]`
(`AudioFrame::to_bytes`, `lib.rs:485`). The first payload byte-pair is the
1-based RX `slot` the audio is demodulated from — a single `CH_AUDIO` channel
carries several virtual receivers (one per slot) and the client routes each
block to the matching playback engine by this field. The `n` (sample count) is
derived from payload length: `n = (len − 8) / 2`. `rate_hz` is 4 800 for SSB
voice (12 000 for FT8 / JS8); `hl2-api` emits a ~25 ms chunk every 25 ms — see §16.7.

---

## 15. Open questions & TODOs

Things the review surfaced that should be settled before this is considered
reference-grade:

* **NCO slot → address mapping (resolved).**
  `build_nco_packet` sets `C0 = (0x01 + slot) << 1` **in the second 512-byte
  chunk** (chunk 1 carries the baseline, see the §7 "commit handshake" note)
  (→ `hl2/src/protocol/data.rs`), and `Hl2::tune` is called with slot values
  `RX{n}_ADDR = n` (→ `hl2/src/hl2.rs:33-40`). The memory map above labels
  `0x02=RX1, 0x03=RX2 … 0x08=RX7`; since `C0 = ADDR << 1`, RX1 → `0x04`,
  RX7 → `0x10` (the reference sets `output_buffer[C0]=0x04+(current_rx*2)`
  with a 0-based `current_rx`, so RX1 → `0x04`). Locked by the
  `build_nco_packet_slot_to_c0_table` test (→
  `hl2/src/protocol/data.rs:556`).
* **Wideband block boundary alignment.** Spec says use sequence-LSB==0 to
  mark a block start; `BlockAssembler` accumulates 2048 samples instead
  (`data.rs:236`). Add sequence-based alignment (or note why it's fine) —
  especially under packet loss.
* **I2C / EEPROM / bias** are fully specified (§11) but not yet exercised in
  code — discovery already surfaces IP/MAC, so those EEPROM reads may be
  redundant.
* **`start/stop` `watchdog_disabled`.** Always left `false`
  (`data.rs:117`); keep-alives cover it, but expose a knob for
  receive-only use.
* **`hl2-common` `Hl2Error`** is a marker type nothing implements (`lib.rs:208`)
  — either use it for typed errors or drop it.
 * **`api/src/hub.rs:29`** — `WsEvent::Wideband.seq` is currently
   dead (never read downstream); wire it into the spectrum frame or drop it.

---

## 16. Virtual receiver (audio channel demod)

A software **audio receiver** lives in the `hl2` crate (`hl2/src/receiver/`):
it takes the radio's per-slot **complex I/Q baseband** (`num_complex::Complex<f32>`
pairs at a DDC rate — tests use 192 kHz) and demodulates it to **`i16` mono
audio** for listening, SSB (USB/LSB) and AM (DSB-FC) today plus FT8 / JS8 /
FT4 digital modes; FM and CW later.

> **Status:** the DSP, the three trait seams (`BasebandSource`,
> `Demodulator`, `AudioSink`), a live EP6 `BasebandSource` and the working
> `hl2 ssb` CLI (§16.6) are implemented and unit-tested in `hl2`. In addition,
> the **full audio path is now implemented end-to-end**: `hl2-api` spawns a
> `VirtualReceiver` per `ClientCmd::SetVrx` and fans the demod audio out to
> every connected WebSocket on `CH_AUDIO` (§16.7), and the Yew UI decodes
> `CH_AUDIO` to a Web Audio playback engine, renders a control panel, and
> shades the SSB passband on the panadapter / waterfall.
>
> The MVP streams **RX1 only** (slot `1`) — creating a second receiver requires
> a follow-up (see §16.9).

### 16.1 Baseband wire format + sample-rate model (settled, measured)

#### EP6 512-byte baseband chunk layout

Each 512-byte baseband chunk of an EP6 (0x06) frame carries this layout
(reference implementation, EP6 receive path):

```text
offset  0: 0x7F 0x7F 0x7F                       3× SYNC
offset  3: C0 C1 C2 C3 C4                       control / status (PTT, ADC-
                                                overload, IO, AIN) — NOT a
                                                slot selector
offset  8: R records, each = N×(I 3B BE | Q 3B BE) | mic 2B
```

* `N` = the number of **active receivers** (slots with an NCO written). The
  gateware round-robins one 24-bit I/Q pair per active receiver, then one
  16-bit mic/audio sample, per record. `N` is *host state* — it is not
  negotiated on the wire; the client knows it because it chose which slots to
  tune (exactly as the reference tracks the per-radio active receivers).
  This crate tracks it the same
  way: [`Hl2::tune`](hl2/src/hl2.rs) registers the slot in the
  [`BasebandFanout`](hl2/src/receiver/fanout.rs) (a per-slot ring map),
  `N = fanout.rx_count()` (= `max(1, |active slots|)`), read by the pump each
  frame. A repeated tune of an already-active slot is idempotent (no new
  ring); `tune(slot, 0)` retires it.
* I and Q are **24-bit signed, big-endian**, normalised by `8388607.0`
  (`2^23 − 1`). 16-bit/LE was a mis-read.
* `count = 504 / (6N + 2)` **whole** records per chunk (plain integer division,
  same as the reference). This divides cleanly only for
  N ∈ {1, 2, 7, 16, 42, 251}; for other N the trailing partial record is
  truncated — e.g. **N=1 → 63**, **N=2 → 36**, **N=3 → 25**, **N=4 → 19**
  samples/chunk (the HL2's max-4-receiver case loses its last few bytes of
  every chunk on the 4th receiver). The parser truncates rather than rejects,
  so all 4 receivers decode.
* `C0-C4` are read separately as PTT/AIN status (the classic Metis C&C), not
  a slot ID — the old "C0 >> 3 = slot" gating dropped valid chunks and is what
  made the UI spectrum read flat-max.

→ Implemented in [`parse_baseband_chunk`](hl2/src/protocol/data.rs:128) /
  [`parse_baseband_frame`](hl2/src/protocol/data.rs:172); the resulting
  [`BasebandChunk.per_rx`](hl2/src/protocol/data.rs:92) holds
  `Vec<Vec<Complex<f32>>>` (one inner `Vec` per active receiver, in RX1, RX2, …
  order).

#### Per-receiver rate

The radio's **C1 SPEED option** (`48/96/192/384` kHz) is the **per-receiver**
baseband rate.

#### Panadapter / waterfall frequency axis (hl2-api)

The `hl2-api` spectrum pipeline (→ `api/src/hub.rs::run_spectral`) computes
the FFT of the accumulated baseband window and re-lays the output out
**centred on the NCO** (→ [`display_mags`](api/src/spectrum.rs)). This is the
same layout the reference uses for its panadapter / waterfall — where "low RF"
is on the left, the tune (NCO) is at the centre, and "high RF" is on the right:

* display bin 0 (left edge)        =  `center_hz − span/2`  — furthest below the NCO
* display bin `m/2` (centre)       =  NCO / `center_hz`      — the tune
* display bin `m-1` (right edge)   =  `center_hz + span/2`   — furthest above the NCO

with `m = 1024` display bins and `span = SharedState.spectrum_span_hz` (96.7
kHz for the EP6 source on the 192 kHz option, or 122.88 MHz for EP4). The UI
(→ `ui/src/canvas.rs::draw_center_cursor_and_ruler`) draws a vertical
cursor at the centre (the tune) and a frequency ruler (band-left / NCO /
band-right) on each panel, using exactly these values, so the user can read
which half of the display a signal on the air falls in. The previous
one-sided layout (pooling only FFT bins `0..nyquist`) parked the tune at one
edge of the display and threw the entire negative-frequency half of the band
away — which is why a station below the tune vanished when the user tuned to
the high end of a band, and vice versa.


### 16.2 Pipeline

The module is split into three independent traits so each end can be swapped
without touching the others (`hl2/src/receiver/mod.rs`):

 ```text
BasebandSource               Demodulator                      AudioSink
(complex I/Q f32 @ddc)  ──► (NCO + channel-select +  ──►  (play to ALSA |
   e.g. VecSource                audio LPF + decimate)     capture to Vec)
      source.rs:18                demod/mod.rs:172            sink.rs:30
 ```
 
  * [`BasebandSource`](hl2/src/receiver/source.rs:18) — where the I/Q comes from
    (socket, file, synth). `VecSource` (source.rs:41) is the in-memory stand-in
    used by tests; `next_block` (source.rs:72) returns `usize` pairs into a
    caller buffer. The socket source is a follow-up (see §16.5).
   * [`Demodulator`](hl2/src/receiver/demod/mod.rs:172) — the mode DSP. The
     stable public contract is the `Box<dyn Demodulator>` held by
     `VirtualReceiver`; its **only impl** is the generic
     [`StandardDemod<C: DemodCore>`](hl2/src/receiver/demod/core.rs:96). New
     modes add a `DemodCore` impl (≤ 30 lines) + a `Mode` variant + a match arm
     in [`make_demod_tap`](hl2/src/receiver/demod/mod.rs:133) — see the
     layering in the next section.
  * [`AudioSink`](hl2/src/receiver/sink.rs:30) — where the audio goes.
    `VecSink` (sink.rs:45) captures for tests; `AlsaSink` (sink.rs:135) plays
    via `cpal`, gated on the `alsa` feature.
  
  [`VirtualReceiver`](hl2/src/receiver/mod.rs:266) wires them together:
  `new(cfg, sink)` (mod.rs:275) constructs the demodulator via
  `make_demod_tap`, and either `process(iq)` (mod.rs:309) feeds one block,
  `flush()` (mod.rs:317) drains the demod/sink residual tails, or
  `run(&mut source)` (mod.rs:327) drains an entire `BasebandSource`.
  `ReceiverConfig` (mod.rs:210) carries `mode` / `source_rate_hz` /
  `source_center_hz` / `bandwidth_hz` / `audio` (`AudioConfig`, mod.rs:192 —
  `rate_hz`, `gain_db`). `Mode` (mod.rs:103) is `Am` (full-carrier DSB voice),
  `Ssb(Usb|Lsb)`, `SsbWide` (the complex pass-through placeholder for digital),
  or `Ft8` / `Js8` / `Ft4` (12 kHz USB digital + a slot decode task); the
  sideband enum is `Sideband` (mod.rs:89). `Mode::default_bandwidth_hz`
  (mod.rs:145) supplies the channel-select width when `bandwidth_hz` is
  `None`.

### 16.3 Demodulator layering (per-mode core + shared audio tail)

  The demod layer is split so **every mode reuses the same audio tail** — the
  part that is *mode-agnostic* (DC-block + RMS-targeted AGC + pre-AGC
  [`RawSampleTap`](hl2/src/receiver/demod/mod.rs:220) + S-meter + `i16` sink
  write) — and each mode contributes only its per-complex-sample DSP:

  ```text
  complex I/Q ──► DemodCore::process ──► 0..=N f32s per block ──► AudioEngine ──► i16 → sink
                     (per-mode DSP)                                  (AGC, tap, meter)
  ```

  * **`DemodCore`** ([`demod/core.rs:41`](hl2/src/receiver/demod/core.rs:41)) —
    the one method a mode implements: `process(Complex<f32>) -> Option<f32>`
    (per-sample DSP; `M−1` of every `M` inputs return `None`), an optional
    `flush() -> Option<f32>` (drains a *pending* sample, used by the AM
    envelope path), and `kind()` (a debug label). `DemodCore::demodulator`
    (core.rs:72) composes the core with the shared tail and returns the
    `Box<dyn Demodulator>` — the **single call site** a new mode needs; it does
    not need its own `Demodulator` impl.
  * **`StandardDemod<C: DemodCore>`**
    ([`demod/core.rs:96`](hl2/src/receiver/demod/core.rs:96)) — the **only**
    `Demodulator` impl in the crate. `demod` (core.rs:104) runs `core.process`
    per input sample, feeds each `Some` into the engine, then
    `engine.emit_full_blocks`; `flush_audio` (core.rs:120) drains the core's
    pending sample + the engine's residual partial block.
  * **`AudioEngine`** ([`demod/engine.rs:41`](hl2/src/receiver/demod/engine.rs:41)) —
    the shared tail. `push` (engine.rs:110) accumulates decimated `f32`s;
    `emit_full_blocks` (engine.rs:117) normalises + emits whole
    `AUDIO_EMIN`-complete 1024-sample chunks; `flush_residue` (engine.rs:129)
    drains the sub-threshold tail. The per-chunk body — `emit`
    (engine.rs:139) — is the pre-AGC [`RawSampleTap`](hl2/src/receiver/demod/mod.rs:220)
    copy + the one-pole S-meter (`meter_tick`, engine.rs:185) + the
    DC-block / RMS AGC normaliser (`normalize_to_i16_with_agc`, engine.rs:221)
    + the `i16` sink write, all unchanged from the old per-mode copies.
    `AUDIO_EMIN = 240` (engine.rs:34) and the AGC constants
    (`TARGET_RMS_I16 = 0.1×32767`, alpha `0.4`/`0.08`, clamp `[1.0, 1e7]`)
    live here.
  * **The three cores** — `SsbCore`
    ([`demod/ssb.rs:35`](hl2/src/receiver/demod/ssb.rs:35), USB/LSB
    product-discriminate), `AmCore`
    ([`demod/am.rs:65`](hl2/src/receiver/demod/am.rs:65), envelope
    detection), and `DigitalCore`
    ([`demod/digital.rs:39`](hl2/src/receiver/demod/digital.rs:39), 12 kHz
    USB) — each `impl DemodCore` holds only its NCO + channel-select
    decimator (see §16.3a / §16.3b / §16.3c for the DSP). The `pub type` aliases
    `SsbDemodulator` / `AmDemodulator` / `DigitalDemodulator`
    (demod/mod.rs:63–65) are now `StandardDemod<SsbCore>` etc., so existing
    names keep resolving to the real full-tail demod type.
  * **`make_demod` / `make_demod_tap`**
    ([`demod/mod.rs:118`](hl2/src/receiver/demod/mod.rs:118) /
    [`:133`](hl2/src/receiver/demod/mod.rs:133)) — the single per-`Mode`
    dispatch. `ensure_decimable` (demod/mod.rs:98) enforces the
    `RateTooClose` guard (`source / audio ≥ 4`, demod/mod.rs:69) that the
    per-mode constructors used to own; then the arm picks the `DemodCore`,
    builds it, and hands it to the shared tail via
    `DemodCore::demodulator`. To add FM/NFM/CW: implement `DemodCore`, add a
    `Mode` variant + a `default_bandwidth_hz` arm (receiver/mod.rs:145), and a
    match arm here — **nothing else changes**.

### 16.3a SSB demodulation (the DSP that's implemented)

 The SSB core is `SsbCore::process`
 ([`demod/ssb.rs:102`](hl2/src/receiver/demod/ssb.rs:102)): an **NCO +
 product-discriminator (Hilbert) + real channel-select LPF + polyphase
 decimator**. It is a [`DemodCore`](hl2/src/receiver/demod/core.rs:41) — the
 per-sample DSP only — and the audio tail (AGC / DC-block / tap / meter →
 `i16`) is the shared [`AudioEngine`](hl2/src/receiver/demod/engine.rs:41)
 attached by [`StandardDemod`](hl2/src/receiver/demod/core.rs:96) (see §16.3a):
 
 ```text
 each complex pair (I,Q) → SsbCore::process (per-sample):
   1  NCO multiply  post = x · e^(−j·φ)            demod/ssb.rs:124
   2  discriminate  USB: r0 = post.re
                     LSB: r0 = Hilbert(post.re)    demod/ssb.rs:125
   3  LPF+decimate  lp.push(r0) → Option<f32>      demod/ssb.rs:133
 per input block → StandardDemod::demod (core.rs:104):
   4  accumulate    engine.push(sample)            demod/core.rs:106
   5  AGC + DC      i16 = DC-block · AGC · gain    demod/engine.rs:221
 ```
 
 * **NCO** (phase-recurrence, stepped via [`Nco::step`](hl2/src/receiver/demod/dsp.rs:104)).
   `φ̇ = 2π·source_center_hz / fs` (installed in `SsbCore::new`,
   [`demod/ssb.rs:58`](hl2/src/receiver/demod/ssb.rs:58)). With a baseband source
   `source_center_hz == 0` so `nco_step == 0` and the multiply is the identity —
   the USB/LSB distinction is then purely the discriminator below. In the full
   stack (where the DDC hands down an *offset* carrier) a non-zero `nco_step`
   rotates the carrier to baseband; **which arm the discriminator keeps** is
 what separates USB from LSB (the reference keeps the in-phase arm for USB and
 conjugates it for LSB).
 * **Product discriminator** — the USB keeps the **in-phase** post-NCO arm
   (`post.re`); the LSB keeps the **quadrature** arm, synthesised as a 90°
   (Hilbert) phase-shift of `post.re` (`F32Fir::hilbert`,
 [`demod/dsp.rs:184`](hl2/src/receiver/demod/dsp.rs:184)) — the missing `e^{j·90°}` a
   real (Q≈0) baseband never had. The Hilbert is **only run for LSB** (it is
   pure overhead for USB, which discards it).
 * **Channel-select** — a symmetric, unit-gain, windowed-sinc **real** low-pass
   FIR designed by [`F32Fir::lowpass`](hl2/src/receiver/demod/dsp.rs:142). Tap count
   is forced *odd* (≥17) so the impulse response is centred on a whole sample
   (linear phase, `H(0)=1`). `SsbCore::new`
   ([`demod/ssb.rs:58`](hl2/src/receiver/demod/ssb.rs:58)) picks **257 taps** for
   the voice band and the Hilbert (`taps.max(257)`), over a `bandwidth_ratio`
   clamped to `[1e-3, 0.4]`. The FIR is split into `M` polyphase branches and
   convolved through
   [`PolyphaseDecimator::push`](hl2/src/receiver/demod/dsp.rs:409) (built on the
   fixed, pre-allocated ring of [`F32FirState::convolve`](hl2/src/receiver/demod/dsp.rs:300));
   the old `Vec::push`/`Vec::drain` rolling buffer is gone, giving
   ≈ `1/M` the tap-multiplies per sample with the same
   zero-padded onset as [`F32Fir::apply`](hl2/src/receiver/demod/dsp.rs:220), which
   remains the reference implementation (the `firstate_matches_reference_apply`
   test locks the two together).
   * **Decimate** — the polyphase anti-alias / decimate stage
     ([`PolyphaseDecimator`](hl2/src/receiver/demod/dsp.rs:375)), `rate_in ≥ 4·rate_out`
     (the `RateTooClose` guard, [`demod/mod.rs:98`](hl2/src/receiver/demod/mod.rs:98),
     struct at [`:69`](hl2/src/receiver/demod/mod.rs:69)). Keeps the
     post-decimate Nyquist well below the band edge.
   * **Normalise** — DC-block, then apply a slow **RMS-targeted AGC** (fast
     attack / slow release) to map each audio block onto target RMS `i16`, on
     top of the user's `gain_db`
     ([`normalize_to_i16_with_agc`](hl2/src/receiver/demod/engine.rs:221), the AGC
     state living in [`AudioEngine::agc_gain`](hl2/src/receiver/demod/engine.rs:44)).
     Shared with every mode (see §16.3a); replaces the original per-block peak
     normaliser, which let the noise floor ride the signal envelope — the
     "loud static" symptom. See §16.5.
 
 `IqBlock` is just `Vec<Complex<f32>>`
 ([`demod/mod.rs:91`](hl2/src/receiver/demod/mod.rs:91)). Error types are
 `RateTooClose` (demod/mod.rs:69) boxed into `DemodError` (demod/mod.rs:85).
 Tests in `demod/{ssb,dsp,digital,am}.rs` cover in-band USB **and** LSB tones
 producing audio, in-band-pass/out-of-band-reject, the NCO moving an off-band
 tone into band, the FIR ring / polyphase identities, the digital path
 out-of-band rejection, and the AM tone / real-baseband / reject / NCO-offset
 cases; `mod.rs` tests cover the full `VirtualReceiver` over a `VecSource`.

### 16.3b AM demodulation (DSB-FC)
 
 The AM core is `AmCore::process`
 ([`demod/am.rs:106`](hl2/src/receiver/demod/am.rs:106)): **envelope
 detection** — NCO → LPF/decimate *both* the I and Q arms → take their
 magnitude. It is **not** the "keep the in-phase arm" half of SSB; it
 deliberately discards the SSB/Hilbert discriminator and reads the envelope
 instead, which is invariant to a residual carrier offset (the NCO is never
 exactly on-carrier, and `post.re` alone would be multiplicatively warbled by
 `cos(2π·Δf·t)` — the "popping" symptom; the magnitude `√(I²+Q²)` is
 rotation-invariant). The audio tail is the shared [`AudioEngine`](hl2/src/receiver/demod/engine.rs:41):
 
 ```text
 each complex pair (I,Q) → AmCore::process (per-sample):
   1  NCO multiply  post = x · e^(−j·φ)          demod/am.rs:107
   2  LPF+decimate  i = lp_i.push(post.re) [polyphase] demod/am.rs:122
                     q = lp_q.push(post.im) [polyphase] demod/am.rs:123
   3  envelope      Some(√(i²+q²)) when both emit demod/am.rs:124
 per input block → StandardDemod::demod (core.rs:104):
   4  AGC + DC      i16 = DC-block · AGC · gain   demod/engine.rs:221
 ```
 
 Full-carrier AM is `A·[1+m(t)]·cos(2πf_c t)`; down-converting and
 low-passing both arms carries `A/2·[1+m(t)]` (the modulated envelope) as
 the magnitude of the baseband I/Q pair, and the AGC DC-block step removes the
 carrier DC. **No Hilbert and no sideband selection** — both sidebands pass
 the LPF symmetrically. `AmCore::new` (demod/am.rs:81) uses two identical
 `PolyphaseDecimator` instances (`lp_i` / `lp_q`, same taps + `M`) so they
 emit in lock-step and the magnitude aligns sample-for-sample. Default voice
 bandwidth 8 kHz (`Mode::Am::default_bandwidth_hz`, receiver/mod.rs:145).
 Dispatch: `Mode::Am` → `AmCore::new(…).demodulator(…)` in
 [`make_demod_tap`](hl2/src/receiver/demod/mod.rs:133).

### 16.3c Digital demodulation (FT8 / JS8 / FT4, 12 kHz USB)
 
 The digital core is `DigitalCore::process`
 ([`demod/digital.rs:87`](hl2/src/receiver/demod/digital.rs:87)): **NCO →
 in-phase (USB) arm → polyphase LPF/decimate to 12 kHz**. This is the SSB-USB
 pipeline with the Hilbert/quadrature synthesis dropped (pure overhead — the
 8-GFSK tones sit in the in-phase arm after the USB NCO) and the output rate
 fixed at the `mfsk` / JS8 decoder's 12 kHz window rate. `mode_label`
 (`"ft8"` / `"js8"` / `"ft4"`) is surfaced as `kind()`. The 511-tap
 (→ 513 odd) Kaiser β=12 filter floor is ~107 dB of out-of-band rejection,
 well past the 80 dB the
 `ft8_js8_path_rejects_out_of_passband_signal_by_at_least_80_db` regression
 guards. Default bandwidth 2.6 kHz; `source_rate_hz` must be an integer
 multiple of 12 kHz of at least 4×. The pre-AGC raw f32 stream is the
 [`RawSampleTap`](hl2/src/receiver/demod/mod.rs:220) the `Ft8Tap` / `Js8Tap`
 / `Ft4Tap` slot decoders attach to (the AGC'd `i16` is still emitted to
 `CH_AUDIO` for the operator). Dispatch: `Mode::{Ft8,Js8,Ft4}` →
 `DigitalCore::new(…, <label>).demodulator(…)` in
 [`make_demod_tap`](hl2/src/receiver/demod/mod.rs:133).

### 16.3d FM / NFM demodulation (phase derivative of a polyphase-LPF'd complex baseband)

  The FM core is `FmCore::process`
  ([`demod/fm.rs:148`](hl2/src/receiver/demod/fm.rs:148)); standard FM
  ([`Mode::Fm`](hl2/src/receiver/mod.rs:111), ≈ 15 kHz channel) and NFM
  (`FmNarrow`, ≈ 5 kHz channel) share it — they are the **same DSP**,
  distinguished only by the channel-select bandwidth `FmCore::new` tunes the
  anti-alias / decimate LPF to (`Mode::default_bandwidth_hz`,
  receiver/mod.rs:157–158). It is a
  [`DemodCore`](hl2/src/receiver/demod/core.rs:41); the audio tail is the
  shared [`AudioEngine`](hl2/src/receiver/demod/engine.rs:41):

  ```text
  each complex pair (I,Q) → FmCore::process (per-sample):
    1  NCO multiply  post = x · e^(−j·φ)                                  demod/fm.rs:149
    2  band-limit + decimate, per-arm:
         i_dec = lp_i.push(post.re)   (lp_i = PolyphaseDecimator, channel BW)
         q_dec = lp_q.push(post.im)   (lp_q = same taps, same M)            demod/fm.rs:153–154
    3  on each (i_dec, q_dec) group boundary:                             demod/fm.rs:160–172
         phase  = atan2(q, i)                     ∈ (−π, π]
         delta  = wrap(phase − prev_phase)        ∈ (−π, π]   (radians)
         emit   = delta
  per input block → StandardDemod::demod (core.rs:108):
    4  AGC + DC      i16 = DC-block · AGC · gain                          demod/engine.rs:221
  ```

  The HL2 EP6 wire always delivers **genuine complex I/Q** — the decoder
  ([`parse_baseband_chunk`](hl2/src/protocol/data.rs:164)) writes
  `Complex::new(i_re, q_im)` from two independent 24-bit words per I/Q
  record, so **both** arms carry real signal energy in every
  configuration. There is no "real (Q≈0)" mode to special-case; the SSB
  core's Hilbert 90°-phase-shifter (which synthesises the missing
  quadrature arm for legacy real-SDR input) is pure overhead here and is
  deliberately not used.

  The audio is the FM **deviation**, i.e. the instantaneous frequency —
  the time-derivative of the phase. The demod reads the phase of each
  decimated complex sample as `atan2(q, i)` and diffs it against the
  previous phase, wrapping to the principal value `(−π, π]`. That wrapped
  difference *is* the audio, in radians per decimated sample.

  Two properties make this the correct choice for the HL2:

  * **Band-limit first, then read phase.** The per-arm polyphase LPF is a
    windowed-sinc, unit-gain low-pass at the channel-select bandwidth,
    split into `M = source_rate / audio_rate` branches by
    [`PolyphaseDecimator`](hl2/src/receiver/demod/dsp.rs:375). Running it
    **before** `atan2` removes out-of-band and adjacent-channel noise so
    the phase is well-defined and smooth over the audio band; reading the
    phase on the unfiltered full-rate stream would mix wideband noise
    into every sample and the `delta` would be dominated by noise rather
    than by the deviation. The two arms share identical taps and the
    same decimation factor, so their group boundaries line up (every
    `M`-th input index) and the `(i, q)` pair at each boundary is a true
    complex sample.

  * **Phase (not magnitude) is the observable.** `atan2` returns a phase
    regardless of carrier magnitude, so the demod is **naturally
    amplitude-invariant** — a weak carrier and a strong one produce the
    same `delta` for the same deviation. This sidesteps the
    *amplitude-scaling* problem that broke the previous
    "cross-product + divide-by-magnitude" version of this core: the
    quad-mod numerator `|z|·|z′|·sin Δφ` scales as `A²`, and on the HL2's
    weak DDC output (carrier amplitude ~ 1e-3) that lands ~ 1e-6, under
    the AGC's useful range — *silent at any gain* while the S-meter showed
    a loud signal. The phase-derivative read is O(1) in `A` for any
    `A > 0`, which is exactly what the shared AGC (engine.rs:221) needs.

  The deviation `[−f_dev, +f_dev]` is symmetric about DC, so the audio is
  zero-mean for symmetric deviation and the AGC's DC-block has no work
  left to do beyond removing a small residual offset.

  Regression guards (all in the `fm.rs` `tests` module):
  `inband_fm_produces_audio` / `nfm_narrow_channel_passes_in_band_deviation`
  (peak level at a full-scale carrier), `weak_complex_baseband_fm_produces_audio`
  / `weak_complex_baseband_nfm_produces_audio` (`A = 1e-3` — the HL2's
  weak DDC output; asserts the output still reaches the AGC target), and
  `weak_complex_fm_is_clean_tone` (Goertzel at 1.5 kHz vs. 3.0 kHz and
  5.0 kHz — the voice bin must dominate its 2× harmonic *and* any
  broadband floor, so a demod that leaks image at 2f — the "static, no
  tone" symptom — fails the test).

  Dispatch: `Mode::{Fm,FmNarrow}` →
  `FmCore::new(…, "fm"|"nfm").demodulator(…)` in
  [`make_demod_tap`](hl2/src/receiver/demod/mod.rs:167).

### 16.4 Sinks

* [`AudioSink`](hl2/src/receiver/sink.rs:30) is deliberately `i16`-mono only
  (a complex/24-bit mode can add a sibling trait rather than complicating this
  one). `VecSink` (sink.rs:45) is `Default` with an optional frame cap.
* [`AlsaSink`](hl2/src/receiver/sink.rs:135) (feature `alsa`) owns a
  `cpal::Stream` + a shared `Ring` (sink.rs:93) that its audio callback drains
  (sink.rs:185, 117); `write` (sink.rs:197) just enqueues, so the demod thread
  never blocks on the sound card. `cpal::Stream` is `!Send`, so `AlsaSink`
  gets an `unsafe impl Send` (sink.rs:203, with the reasoning inline) — sound
  because the callback only touches the `Send+Sync` `Ring`. `cpal` is optional
  (`dep:cpal`, [Cargo.toml](hl2/Cargo.toml:18)); `StreamConfig` uses
  `channels: 1` (mono) and the requested audio rate.

### 16.5 Building / testing

```sh
# default build (ALSA sink off):
cargo build -p hl2 -p hl2-common -p hl2-api
cargo test  -p hl2 -p hl2-common          # 56 + 7 tests, all SSB/receiver & vrx

# with the ALSA sink (needs libasound2-dev / pkg-config `alsa`):
cargo build -p hl2 --features alsa
cargo clippy -p hl2 --features alsa
```

The `alsa` feature is **opt-in** so the default build (and the browser-facing
`hl2-common`/`hl2-api`) has no hard runtime dependency on ALSA.

### 16.6 Working SSB CLI (`hl2 ssb`)

The live end-to-end path lives in the `hl2` binary (`hl2/src/bin/main.rs`,
the `ssb` arm): `Hl2::start_with_speed(ip, rate)` opens the EP6 baseband
stream at the requested C1 SPEED (→ §16.1), `tune(slot, freq)` sets the NCO
for the chosen slot (registering that slot as active ⇒ `N = |active slots|`,
→ §16.1), and the pump de-interleaves each frame into one [`BasebandRing`]
(→ §16.9) per active slot and **fans `chunk.per_rx[p]` into the p-th slot's
ring**. A single `VirtualReceiver` (`rx.process(…)`) — configured via
`--slot` to read that slot's ring — demods it to `i16` audio written to an
[`AlsaSink`](hl2/src/receiver/sink.rs:135) (`alsa` feature) or `VecSink`
(`VecSink`).

```sh
# build + run (ALSA playback)
cargo build -p hl2 --features alsa
target/debug/main ssb 192.168.1.68 7049000 usb
target/debug/main ssb 192.168.1.68 14074000 usb --slot 2   # demod RX2

# flags
ssb <IP> <FREQ> <usb|lsb> \
  [--slot <N>] [--rate <Hz>] [--offset <Hz>] [--band <Hz>] \
  [--lna <dB>] [--filtermask <hex>]
```

* `--slot` — the receiver to demodulate (`--slot 1` = RX1, the default). The
  CLI tunes that slot to the given frequency and reads that slot's baseband
  ring. The same slot is re-used if you also `tune` other slots concurrently,
  but the demod always follows `--slot`. (Default 1, matching the previous
  RX1-only behaviour.)

* `--rate` — per-receiver baseband option, `48000/96000/192000/384000`
  (default 192 000). Sets the C1 SPEED bit and `source_rate_hz = rate/2`
  (→ §16.1).
* `--offset` — NCO offset from baseband centre, `Hz` (default 0; the DDC hands
  down baseband-centred I/Q, so SSB around DC with no NCO).
* `--band` — channel-select bandwidth, `Hz` (default 2 600).
* `--lna` — RX LNA gain in dB, `-12..+48` (default
  `DEFAULT_LNA_GAIN_DB` = +6). Register write (§11.3). The front-end has no
  AGC; the gain holds until the next `set_lna_gain`. (Note: the demod's *back-end*
  does have a slow RMS AGC — §16.3 — which maps the demod output to a stable
  i16 level independent of `--lna`; that's why loud signals don't drive the
  output to saturation as you raise `--lna`.)
* `--filtermask` — RX open-collector filter relay mask, `0x00..0x7F` (default
  `0x00`). LSB-first: bit 0 = relay/checkbox 1 … bit 6 = relay/checkbox 7.
  Re-asserted in every keep-alive's `C2[7:1]` (→ §11.4). Example: `0x41`
  = relays 1 + 7 (the 40 m / 7.074 MHz passband on the MRF101 companion
  filter board).

Verified against a live HL2: ` iq:audio ` tracks ` rate/2 : 4800 ` exactly
(96 k→/10, 192 k→/20, 384 k→/40), confirming the §16.1 complex-rate model.

 ### 16.7 Server (`hl2-api`) + UI (`hl2-ui`) implementation

 The full audio path is wired. Each layer sits on the one already-built in
 `hl2` — the DSP (§16.3a) and the `BasebandRing` hand-off — without touching
 them.

  **Wire contract** (`hl2-common`, shared by server + browser):

  | Item | Role | Ref |
  |------|------|-----|
  | `ClientCmd::SetVrx { cfg }` | Create / update the receiver on `cfg.slot`. `cfg: null` tears down that slot. | `lib.rs:247` |
  | `ClientCmd::SetVrxMute { slot, muted }` | Mute / unmute a slot's receiver without rebuilding (bandwidth saving). | `lib.rs:353` |
  | `ClientCmd::SetVrxOff { slot }` | Tear down a slot's receiver (explicit off form). | `lib.rs:359` |
   | `VrxCfg { slot, offset_hz, mode, bw_hz, gain_db }` | Requested settings (offset shifts the demod NCO off the tune; `mode` is `usb`/`lsb`/`ft8`/`js8`). | `lib.rs:201` |
  | `SharedState.vrx: BTreeMap<u8, VrxState>` | Per-slot active receivers (keyed by RX slot). Mirrored into **every** connected tab, like `oc_bits`. | `lib.rs:146` |
  | `AudioFrame { slot, seq, rate_hz, samples }` | `CH_AUDIO` payload (slot byte-pair first, then seq/rate). | `lib.rs:470` |
   | `Ft8Decode { text, freq_hz, dt_sec, snr_db, slot_ms, vrx }` | One decoded digital-mode message (`DecodeRow` since the FT8/FT4/JS8 wire collapse, §16.10/§16.11); `vrx` is the full [`VrxState`] of the receiver that demodulated it (slot, offset, **mode**, bw, gain). | `lib.rs:218` |

  **Server pipeline** (`hl2-api`, `api/src/hub.rs`):

   * `Session` owns `vrx: BTreeMap<u8, VrxTask>` (`hub.rs:99`) — **one per
     active RX slot**, not a single slot. Each `VrxTask` is fed from the
     **per-slot baseband ring** matching its `VrxCfg.slot`
     (`set_vrx_cmd` → `ctrl.baseband_ring(c.slot)`, `hub.rs`). Each active
     slot has its own [`BasebandRing`] (→ §16.8 fan-out) and the pump fans
     each into it, so `slot=2` demodulates RX2's stream, not RX1's.
     `set_vrx_cmd` removes + re-spawns only that slot's pipeline (other
     slots keep streaming); `set_vrx_off_cmd` removes just that entry;
     `stop_cmd` drains the whole map.
  * `ClientCmd::SetVrx` → `set_vrx_cmd` (`hub.rs`) →
    `spawn_vrx` (`hub.rs`):
    1. build a `ReceiverConfig` from `VrxCfg` (including
       `source_center_hz = cfg.offset_hz`, which parks the demod's NCO at
       `tune + offset` — `0` for a receiver centred on the tune) and a
       `VirtualReceiver` (the exact type the §16.6 CLI uses);
    2. spawn a **std** demod thread that reads the `BasebandRing`
       (`hl2/src/receiver/baseband_ring.rs`) and calls
       `VirtualReceiver::process`, writing each `i16` block to a
       `BufSinkHandle` (`hl2/src/receiver/sink.rs`); the
       demod thread is the single producer;
    3. spawn a **tokio** fan-out task that polls that `BufSinkHandle` every
       25 ms, pulls up to 480 samples (~100 ms of 4.8 kHz audio — four ticks'
       worth, so the UI never sees a gap even under slow frames). **While
       muted** (the `muted` `Arc<AtomicBool>` shared by the task) it just
       drains + discards those samples, keeping the audio buffer bounded
       without broadcasting — the demod and the FT8 decoder (if FT8) keep
       running. Otherwise it broadcasts a
       `WsEvent::Audio { slot, seq, rate_hz, samples }` to every subscriber.
     4. (FT8 / FT4 / JS8 mode only) build the mode's decoder, wrap it as a
        `RawSampleTap`, and spawn that mode's decode task — a ticker that
        stamps the slot's [`VrxState` snapshot`] (slot / offset / **mode** /
        bw / gain) onto each [`DecodeRow`] (via `on_decodes`, §16.10) before
        broadcasting, and appends any spot-able rows to
        the PSK Reporter queue (`SpotSink`, §16.12).
  * `ClientCmd::SetVrxMute` → `set_vrx_mute_cmd` (`hub.rs`) — flips the
     shared `muted` flag on the slot's `VrxTask` (no rebuild), echoed as
     `vrx[slot].muted` in the next `SharedState`.
  * `ClientCmd::SetVrxOff` → `set_vrx_off_cmd` (`hub.rs`) — removes the
     slot's entry from the map (other slots untouched).
  * `VrxTask::stop()` (called from `set_vrx_cmd` when the receiver is rebuilt,
    from `set_vrx_off_cmd`, or from `stop_cmd`) signals the demod thread,
    joins it, then aborts the fan-out + (FT8 / FT4 / JS8) decode task handles. A manual
    `impl Debug` exists because the atomics aren't `Debug`
    (`hub.rs`).
  * `api/src/ws.rs` converts `WsEvent::Audio { slot, … }` to
    `encode_ws_binary(CH_AUDIO, &AudioFrame { slot, … }.to_bytes())` — the
    frame layout is the one in the §13 table.

  **Browser pipeline** (`hl2-ui`, Yew 0.23, CSR-only):

  * `ui/src/client.rs` — `parse_audio` splits an incoming `CH_AUDIO` frame
    back into `(slot, seq, rate_hz, samples)` (after stripping the 2-byte
    channel id that `app.rs::on_binary` already routed on); the `slot`
    byte is used to route the block to the matching playback engine.
   * `ui/src/audio.rs` — the `Audio` engine (a **single** global
     `AudioContext` + master `GainNode`, shared across slots because the
     one-audible-at-a-time policy is enforced on the server — see the panel
     handlers below). On each `CH_AUDIO` frame, `push_samples` allocates an
     `AudioBuffer` of exactly the incoming length, writes the
     `f32`-normalised samples (`x / 32768`), chains a fresh
     `AudioBufferSourceNode` after the last one, and schedules it via
     `clamp_next_start` (audio.rs): the start time is re-anchored every frame
     to `max(next_start, ctx.current_time + LOOKAHEAD)`, so an underrun (a
     suspend/resume, or the server sideband/BW/gain-rebuild gap) becomes a
     ~50 ms silent head lead instead of a past-time dump that pops.
     - `mute()` / `unmute()` ramp the master `GainNode` (20 ms, no hard
       step): Off fades the tail to 0 so the already-scheduled buffers can't
       bleed into the next On; On re-anchors the timeline to just-ahead-of-now
       and restores the requested level. `set_gain` (master volume) and
       `reset()` (re-anchoring on reconfig/reconnect) are also ramp/no-op-safe;
       `is_muted()` exposes the current state for toggle UIs.
   * `ui/src/app.rs`:
     - `Shared` state carries `vrx_slot: RefCell<u8>` (the panel's target,
       driven by the RX nav tabs / EP4 tab), `vrx: RefCell<BTreeMap<u8,
       VrxState>>` (mirror of `SharedState.vrx`), and the panel editor fields
       (`vrx_sideband` / `vrx_bw_hz` / `vrx_gain_db`). `SharedState` mirrors
       per-slot via `Shared::refocus_vrx_panel`, which re-anchors the panel
       editor to the targeted slot's live values (or the band defaults —
       `lsb` < 10 MHz, else `usb` — when that slot has no active receiver).
     - `on_binary` (`CH_AUDIO` arm) — parses `slot` and only feeds
       `sh.audio.push_samples(…)` when the targeted slot's receiver is active
       and unmuted **on the server** (stale frames from a muted / switched slot
       are discarded locally too). Re-applies the user's master volume.
     - **Control panel** — one panel that follows the current nav tab's slot
       (RX1–4 or EP4). Mode (`usb`/`lsb`/`ft8`), bandwidth `Hz`, gain `dB`,
       master volume `0–100%`, and **Mute / Unmute** (no rebuild — just
       `setvrxmute`), **On** (enforce
       one-audible-at-a-time by muting every other active slot, then
       `setvrx`), **Off** (mutated locally via `setvrxoff`, and the local
       `vrx[slot]` entry is dropped immediately so the badge updates without
       waiting for the `on_text` mirror). Every receiver param change
       (sideband / BW / gain) calls `audio.reset()` first (re-anchor across
       the server's rebuild gap) then is pushed immediately if that slot's
       receiver is active.
       `send_vrx` / `send_vrx_off` build their JSON from `VrxCfg` / the slot
       via `serde_json::json!` (no hand-escaped braces); `send_vrx_mute`
       builds the `setvrxmute` variant.
     - `draw_all` overlays the passband of **every** active receiver (one
       band per slot, using the receiver's `offset_hz`) on **both** the
       panadapter and the waterfall (see
       `canvas::draw_vrx_passband` below).
  * `ui/src/canvas.rs` — `draw_vrx_passband` shades **each receiver's**
     channel-select passband (centred at `center + offset`) on **both** the
     panadapter and the waterfall. The demod (§16.3a) is USB =
     `[centre, centre+bw]` (above the tune + offset), LSB =
     `[centre−bw, centre]` (below) — the overlay matches that band exactly.
     Green = USB, blue = LSB. Passbands entirely off the displayed span are
     skipped (no sliver at the edge).

  **End-to-end flow** (one command → one stream):

  ```text
  [UI]  {"id":7,"cmd":"setvrx","data":{"cfg":{"slot":2,"offset_hz":0,"mode":"usb","bw_hz":2600,"gain_db":0.0}}}
    │   (ClientCmd::SetVrx, hl2-common/src/lib.rs)
    ▼
  [api]  Session::set_vrx_cmd → spawn_vrx
    │   • demod thread (std): BasebandRing(slot 2) → VirtualReceiver → BufSinkHandle
    │   • fan-out task (tokio): BufSinkHandle → WsEvent::Audio { slot:2, … } (25 ms cadence)
    ▼
  [WS]   CH_AUDIO binary: [0x03 0x00][u16 slot][u32 seq][u16 rate][i16 samples …]
    ▼
  [UI]   parse_audio → (slot, seq, rate_hz, samples) → if targeted slot + not muted:
          Audio::push_samples → AudioBufferSourceNode (chained)
    │    + Shared mirrored from the `setvrx` response
    ▼
  [UI]   canvas: draw_vrx_passband (per-slot, panadapter + waterfall), vrx panel reflects
           the targeted slot's VrxState
  ```

### 16.8 Per-slot fan-out (multiple active receivers)

The EP6 pump now feeds **every** active slot, not just RX1. This is the
"support all 4 receivers" plumbing:

* **Ownership.** `Hl2` owns one
  [`BasebandFanout`](hl2/src/receiver/fanout.rs) (`Hl2.inner.fanout`) — a
  `BTreeMap<slot, Arc<Mutex<BasebandRing>>>`. Slot 1 is seeded at construction
  so the radio's anchor receiver always has a ring from the first frame.
* **Tune = register.** `tune(slot, freq)` calls
  `fanout.register_slot(slot)` (idempotent, same `Arc` kept on re-tune);
  `tune(slot, 0)` retires it via `unregister_slot`, re-seeding slot 1 if that
  empties the set. The active set is exactly `fanout.slots()`, in ascending
  slot order.
* **Parse = de-interleave.** Each frame the pump snapshots
  `(N, rings)` from the fan-out and calls
  `parse_receive_packet(&buf, N)` → `parse_baseband_chunk(…, N)`, splitting the
  payload into `per_rx[0..N]`. `N = fanout.rx_count()` is also the value in
  the C4 receiver-count bit of the baseline chunk (→ §16.1), so the wire and
  the de-interleave always agree.
* **Fan = one ring per slot.** For each `per_rx[p]` the pump pushes into
  `rings[p]` — position `p` is the p-th smallest active slot, matching the
  interleaved order (→ §16.1). So RX1→ring(slot1), RX2→ring(slot2), and if
  active = {1,3}, `per_rx[1]` lands in ring(slot3). Each ring is a drop-oldest
  [`BasebandRing`], so a slow demodulator on one slot never backpressures the
  socket or the other slots.
* **Readers address by slot.** `Hl2::baseband_ring(slot)` returns that
  slot's ring (falling back to slot 1's if it isn't active). The SSB CLI
  (`--slot`), the `hl2-api` vrx (`hub.rs` reads
  `ctrl.baseband_ring(c.slot)`), and any analysis tap all clone the `Arc<…>`
  for the slot they care about. Slot 1's ring is the backward-compatible
  default.

This makes the whole device a "4-receiver radio" at the baseband layer: the
gateware interleaves up to 4 DDC outputs, we split them, and any or all can be
demodulated in parallel. The §16.7 server pipeline now runs one `VrxTask`
(per-slot sink + fan-out + demod) per active slot — so all four can be
demodulated + streamed concurrently on one shared `CH_AUDIO` channel, routed by
the `slot` byte (§16.7 audio layout).

### 16.9 Follow-ups (not yet implemented)

* **Pinning N at START** — the EP6 interleave count is now tracked from tune
  state (→ §16.1). A further optimization would be to pin the radio's own
  receiver count at start (ADDR `0x00` high bits, or the start frame) so the
  gateware only emits one receiver and N is always 1 without host tracking.
  Not yet implemented; N=1 is the common (and currently assumed) case.
* **Multiple concurrent virtual receivers (implemented)** — the §16.7 pipeline
  is now **one `VrxTask` per active RX slot** (`Session.vrx:
  BTreeMap<u8, VrxTask>`), each with its own `BasebandRing` reader, demod
  thread, `BufSink` and 25 ms fan-out task. All of them share the single
  `CH_AUDIO` channel; the `slot` byte in the `AudioFrame` header
  ([§13](#13-websocket-serverclient-contract)) routes each block to the
  matching playback engine. `setvrx` / `setvrxoff` / `setvrxmute` are all
  per-slot, so one slot can be rebuilt, muted, or torn down without touching
  the others. The one shared resource is the UI's `AudioContext` (the
  browser keeps a single master gain for playback); the UI enforces
  one-audible-at-a-time by muting the other active slots (`setvrxmute`) when
  a slot is engaged — the server is still broadcasting the muted slots'
  demod + FT8 decode, but the fan-out discards so the bandwidth saving is
  real.
* **Shared slot ring, multi-reader peek (implemented)** — the per-slot
  [`BasebandRing`](hl2/src/receiver/baseband_ring.rs) is a bounded
  drop-oldest Vec ring with sequence numbers (default
  [`BASEBAND_RING_CAP`](hl2/src/receiver/baseband_ring.rs) = 8 192 complex
  samples ≈ 68 ms at 96 kSps). The pump remains the **sole writer**
  (`push`), but any number of readers follow the *same* ring from their own
  cursor (`peek(cursor, &mut out, max)` — non-destructive, copies the
  window instead of consuming it). This is what lets auto-decode attach a
   headless FT8, FT4, and JS8 decode pipeline per known in-window frequency to a
  slot that also has a live vrx: the pump writes each slot's stream once,
  and every downstream demod drains independently. A reader whose cursor has
  been dropped out by overflow (fell below `base_seq()`) resyncs the cursor
  to the oldest *still buffered* sample; a reader that merely caught up
  stays put. Both cases surface as `peek` returning 0 — the caller
  distinguishes them by comparing the cursor to `base_seq()` within the same
  locked critical section (a resync on idle would cause a double-read of
  already-consumed samples). The 3 demod-loop call sites
  ([`hl2/src/bin/main.rs`](hl2/src/bin/main.rs),
  [`api/src/hub.rs`](api/src/hub.rs) vrx + auto) all use this pattern.
  Fuzz-tested with a reference-deque model
  (`receiver::baseband_ring::tests::fuzz_push_peek_stays_in_buffer`).
* **Auto decode (implemented)** — `ClientCmd::AutoDecode { slot, enabled }`
  turns on **per-slot headless decode**: for every known in-window frequency
  (the module-level `KNOWN_FREQS` tables in
  [`hl2/src/receiver/ft8.rs`](hl2/src/receiver/ft8.rs) and
  [`hl2/src/receiver/js8/decoder.rs`](hl2/src/receiver/js8/decoder.rs)),
  `spawn_auto` in [`api/src/hub.rs`](api/src/hub.rs) builds a
  `VirtualReceiver` tuned to the slot NCO + offset, feeds it the slot's ring
  (sharing the write-once pump stream with the live vrx, §16.9 above), and
   runs the corresponding `Ft8` / `Js8` / `Ft4` decode task with a `DropSink` (no audio
   out). Decoded rows reuse the shared `WsEvent::Log` WS event — the
   log view is the same whether the decode came from a manual vrx or from an
   auto pipeline. `Session.auto: BTreeMap<u8, AutoTask>` is torn down on
  `Tune` (when the slot retunes, any auto pipelines on that slot are
  rebuilt) and on `Stop`. UI exposes an "Auto Decode" checkbox per slot with
  a live per-frequency readout (which bands are currently in the slot's EP6
  window) and locks the frequency stepper while enabled so tuning doesn't
  silently move the in-window bands. `auto_to_vrx_mode` in `hub.rs` maps
  `AutoMode` → `VrxMode` (the demod pipeline is the same — only the decode
  task differs).
* **DSP niceties** — a soft-clipper / true limiter in front of the existing
  RMS-AGC (the AGC clamps at ±54 dB of travel, which is plenty for voice but
  a 40 dB+ tone burst can still momentarily exceed 0 dBFS), and a real
  channel-select IIR if the FIR tap count ever needs to shrink.

### 16.10 FT8 (digital mode, implemented)

The virtual receiver supports a third mode, `VrxMode::Ft8`, alongside USB /
LSB (§16.3a), and two more, `VrxMode::Js8` (§16.11) and `VrxMode::Ft4` (§16.11).
All three are WSJT-family digital decoders. FT8 is decoded by the [`mfsk-core`](https://crates.io/crates/mfsk-core)
`Ft8` decoder (`hl2/src/receiver/ft8.rs`) — the same one WSJT-X uses — run
per 15 s slot against the demodulator's raw tap. FT4 uses the same `mfsk-core`
engine (`hl2/src/receiver/ft4.rs`) at a 7.5 s slot; JS8 runs its own LDPC
pipeline (`hl2/src/receiver/js8/`).

**Mode + sample rate.** `VrxMode::Ft8` selects a **12 kHz** baseband sample
rate (the mfsk-core FT8 reference rate), not the SSB audio path. The
`CH_AUDIO` stream still carries what the demod emits, so the mode is audibly
silent but the *decode* task is what the user waits on.

**Raw-sample tap.** The shared [`AudioEngine`](hl2/src/receiver/demod/engine.rs:41)
carries an optional `RawSampleTap` (an `Arc<dyn RawSampleTap>`,
[`demod/mod.rs:220`](hl2/src/receiver/demod/mod.rs:220)) installed in
`make_demod_tap` (demod/mod.rs:133). Its `emit`
([engine.rs:139](hl2/src/receiver/demod/engine.rs:139)) invokes the tap with
each pre-AGC, decimated `f32` block, so the FT8 decoder sees the same samples
the SSB/USB path hears, before the RMS-AGC rescales. `Ft8Tap`
(`hl2/src/receiver/ft8.rs`) is the concrete `RawSampleTap`: it appends into a
bounded ring behind `Arc<Mutex<SharedDecoder>>` (the `SharedDecoder` handle is
shared with the decode task). The `ReceiverConfig.tap` field is `Option<…>`;
`hl2 bin/main.rs` sets `None`, `hl2-api` sets the live `Ft8Tap`.

**Closed-slot decode.** `decode_closed_slot(&SharedDecoder, slot_ms)`
(`hl2/src/receiver/ft8.rs`) takes the most-recently-**completed**
15 s window (a multiple of 15 000 ms) that has not yet been decoded, runs the
mfsk-core FT8 batch decode over the trailing `FT8_SLOT_WINDOW_SAMPLES`
(180 000) samples of the ring — 15 s × 12 kHz — and returns the decoded rows
(`Ft8Message`). `shared()` in the same module hands out the
`Arc<Mutex<SharedDecoder>>` used by both the `Ft8Tap` (writer) and the decode
task (reader). The decode runs **off-lock** with the demod thread: `Ft8Tap::append`
is the only critical section and it is bounded by `Ft8Tap::BUFFER_CAP`
(≈ 20 s); the mfsk-core batch decode on `&i16` never blocks the demod.

`decode_window(&[i16], fmin, fmax, sync_min, max_cand)` is the pure function the
`closed_slot_for` path routes through — useful in tests and for a future
real-time path. `FT8_SLOT_MS = 15_000` and `FT8_SAMPLE_RATE_HZ = 12_000` are
`pub const` in the same module (cited in `PROTOCOL.md` here).

**Server task** (`hl2-api`, `api/src/hub.rs`). `spawn_vrx` branches on
`VrxMode`: in `Ft8` it builds a `shared()` decoder, wraps it as
`Ft8Tap` → `Arc<dyn RawSampleTap>` and puts it in `ReceiverConfig.tap`, and
spawns `spawn_ft8_decode_task` (`hub.rs`) — a 1 s ticker that computes
`closed_slot_for(now_ms)` and calls `decode_closed_slot`; on ≥ 1 row it
broadcasts `WsEvent::Log(DecodeLog { decodes })` to every connected websocket
(fired-and-forget: a lagged / gone fanout is a no-op — the slot is already
marked decoded, no retry). `VrxTask.stop` aborts that handle on teardown.

**Wire contract** (`hl2-common`):

| Item | Shape | Ref |
|------|-------|-----|
| `VrxMode::Ft8` | third `VrxMode` variant (the SSB variants are `Usb` / `Lsb`) | `lib.rs` |
| `VrxCfg.mode: VrxMode` | the requested mode (USB / LSB / FT8) | `lib.rs` |
| `VrxState.mode: VrxMode` | echo of the active mode (mirrored into the UI `<select>`) | `lib.rs` |
 | `DecodeRow { text, freq_hz, dt_sec, snr_db, slot_ms, vrx }` | the **one** decoded-message wire row shared by every digital mode (FT8 / FT4 / JS8). For FT8/FT4 `text` is the resolved 77-bit WSJT payload; for JS8 it is the decoder's display string. `vrx: VrxState` is the full snapshot (slot / offset / **mode** / bw / gain) of the receiver that demodulated it — the mode-of-record that replaced the per-mode `Ft8Decode`/`Ft4Decode`/`Js8Decode` types | `lib.rs:218` |
 | `DecodeLog { decodes: Vec<DecodeRow> }` | the batch, ≥ 0 rows; one envelope for **all** digital modes | `lib.rs:245` |
 | `WsEvent::Log` | the fanout event (replaces `WsEvent::Ft8Log` / `Ft4Log` / `Js8Log`); `to_message` (`api/src/ws.rs`) serializes to the wire below | `hub.rs` |

**Wire frame** (text, not binary) — one envelope for **all** digital modes:

```json
{"cmd":"log","data":[{"text":"CQ DE W1AW","freq_hz":153.5,"dt_sec":-0.42,
                      "snr_db":18.3,"slot_ms":1721395200000,
                      "vrx":{"slot":1,"offset_hz":0,"mode":"ft8","bw_hz":2600,"gain_db":0.0,"rate_hz":12000,"muted":false}}, …]}
```

The mode is **per-row**, carried in `data[..].vrx.mode` (`"ft8"` / `"ft4"` /
`"js8"`), so the same `"log"` command carries FT8, FT4, and JS8 rows
indistinguishably. An FT8 row is sent **at most once per completed 15 s slot**, and
**only when ≥ 1 row decodes**
(silent slots — no CRC-passing hit — are a no-op, not an empty-frame broadcast).
The `data` array is exactly the `DecodeLog.decodes` list. Each row carries the
`vrx` snapshot of the receiver that produced it, so a multi-slot setup shows
*exactly which (slot, offset, tuning)* demodulator decoded the message — the
UI stamps the row's **Rx** column from `vrx.slot` and the **Mode** column from
`vrx.mode`.

**UI panel** (`hl2-ui`, `ui/src/app.rs`). The vrx panel's `<select>` gains
`<option value="ft8">`. The panel is **per-slot**: the selected RX nav tab
drives `Shared.vrx_slot` and `Shared::refocus_vrx_panel` re-anchors the
sideband / bandwidth / gain editor to that slot's active [`VrxState`] (or the
band defaults: `lsb` below 10 MHz, `usb` at/above). On switch to a digital mode
the description line changes (§16.7 `vrx_desc`) and a single **decode log** table
renders in the panel — newest-first, capped at `FT8_LOG_CAP` (200 rows) — that
mixes every digital mode's rows. Each row has an **Rx** column stamped from
`vrx.slot` (so simultaneous decoders on different slots are distinguishable) and
a **Mode** column stamped from `vrx.mode`. The log is appended in
`on_text` (`ui/src/app.rs`, before the `ServerResponse` parse, because
`log` is not a `ServerResponse`) — `Envelope.data: Vec<DecodeRow>` matches
the wire shape exactly.

**End-to-end path.** `VrxMode::Ft8` → `spawn_vrx` → `ReceiverConfig.tap =
Ft8Tap(shared())` → demod `RawSampleTap::append` (writer, off the critical
section) → 1 s ticker `decode_closed_slot` (reader, off-lock) → each row stamped
with the slot's [`VrxState`] snapshot (`on_decodes`) → `WsEvent::Log` → every
tab's `on_text` → `decode_log` (bounded, newest-first) → the `decode_log_html`
table (§16.7 panel). The decode path is **per-slot, per-pipeline**: one decoder
per `VrxTask`, so two slots running FT8 decode independently (their `wsjtx`
slots are the same 15 s wall-clock window, but the demod + NCO offset differ).

### 16.11 JS8 (digital mode, all speeds, implemented)

JS8 (four speeds **A / B / C / E**) is implemented as a **native Rust
decoder** in `hl2/src/receiver/js8/` — a port of the `js8call` reference
(`JS8.cpp`, `js8a_*.f90`) — not a library dependency, like FT8's mfsk-core.
Concurrency shape: a raw-sample tap writing into one 60 s ring buffer, and
a 500 ms poller (`hl2-api`) that calls `js8_step` — which snapshots *any
speed whose cycle data is ready since its last re-arm* and decodes it
off-lock. There is no fixed 15 s slot grid: each speed has its own cycle
length (A 15 s, B 10 s, C 6 s, E 30 s) and a signal may straddle a cycle
boundary, so the reference decodes **as the data becomes ready**
(`decodeEnqueueReadyExperiment`) and re-arms each speed every ~1.5 s — that
is exactly what `js8_step` reproduces.

**Mode + sample rate.** `VrxMode::Js8` selects the same **12 kHz** baseband
path as FT8 (`make_demod_tap`, `hl2/src/receiver/demod/mod.rs` — the demod is the
FT8/USB pipeline; `Mode::Js8` is a mode flag that makes `hl2-api` attach the
JS8 tap instead of the FT8 tap). The `CH_AUDIO` stream still carries the
demod output.

**Signal model** (`hl2/src/receiver/js8/params.rs`, the `Mode` struct +
`MODES: [Mode; 4]`; constants mirror the reference's `ModeA/B/C/E` struct
bodies). All speeds share the frame structure — one 79-symbol signal
(21 Costas sync + 58 LDPC data symbols) — and differ only in speed and a
small set of tuned sync/codec constants:

| Speed | Cycle | nsps | nsps/s | Costas | astart |
|-------|-------|------|--------|--------|--------|
| A | 15 s | 1920 | 6.25 | ORIGINAL (FT8 arrays) | 0.5 s |
| B | 10 s | 1200 | 10 | MODIFIED | 0.2 s |
| C | 6 s | 600 | 20 | MODIFIED | 0.1 s |
| E | 30 s | 3840 | 3.125 | MODIFIED | 0.5 s |

One 79-symbol signal per cycle — three 7-symbol Costas sync blocks
(positions 0–6, 36–42, 72–78) bracketing two 29-symbol data blocks,
8-FSK (3 bits/symbol). Each data block is half of an LDPC(174,87) codeword
— 72 payload bits, packed per the JS8 varicode grammar (12-char varicode
+ command/grid, `msg.rs`), 3 frame-type bits, 12-bit CRC-12 — decoded by a
belief-propagation decoder (`ldpc.rs`). Transmit starts `astart` into the
cycle; the per-speed decode window is `nmax = ntxdur × 12 000` samples
(`Mode::nmax`, e.g. A = 180 000, B = 120 000, C = 72 000, E = 360 000).
Because all speeds consume the same 12 kHz samples, one rolling buffer
(`JS8_BUFFER_CAP = 60 s`, the reference's `JS8_RX_SAMPLE_SIZE`) serves all
four decodes; `js8_step` slices each speed's own trailing window from it.

**Decode loop** (`decoder.rs:426` `js8_step`). Each tick, for every speed
in `MODES` (`params.rs`): compute a `total_samples`-relative cycle clock and
decide **readiness** (the reference's `decodeEnqueueReadyExperiment` rule) —
a speed is ready once enough of its trailing window has landed since its own
last re-arm (`last_decode_k[4]`, one u64 per speed) to cover one full
`nmax`-length window AND we are within a 1.5 s window of either a cycle
start or a cycle end (so a signal straddling the boundary is still caught).
When ready, `js8_step` snapshots that speed's trailing `nmax` slice, runs the
three-pass loop (sync → decode → subtract, matching the reference's
`JS8.cpp` decode loop; passes 1–2 subtract each decoded signal to reveal
weaker interferers, pass 3 stops early if a pass improves nothing), re-arms
that speed, and emits its `Js8Message`s. Unready speeds are skipped — so all
four run concurrently from one buffer, each on its own clock.

**Pipeline** (module `hl2/src/receiver/js8/`):

| File | Role |
|------|------|
| `params.rs` | the `Mode` struct (all DSP/timing fields + derived `nmax`/`nfft1`/`costas`/`jz`/`astart`/`df`/`az`/`tstep`…), `MODES: [Mode; 4]` (A/B/C/E), `Mode::by_id`, shared frame consts (NN/NS/ND/N/K/NROWS/NFOS/NSSY) |
| `decoder.rs` | `Js8Decoder` ring (`JS8_BUFFER_CAP = 60 s`, `decoder.rs:50`), `Js8Tap` (`RawSampleTap`), `shared()` / `Js8SharedDecoder`, `js8_step` (readiness gate + re-arm, `decoder.rs:426`), `decode_window` (the three-pass loop) |
| `sync.rs` | `Spectra` (per-step spectrum, parameterized on `Mode`), `fit_baseline` (10th-percentile envelope, 5th-order polyfit, `BASELINE_MIN/MAX` band), `sync_pass` (Costas cross-correlation + `ASYNCMIN` scoring) |
| `decode.rs` | `Ctx::extract` (symbol extraction), `decode_candidate` (LDPC + CRC + frame unpack, SNR per the reference: `xbase = 10^(0.1*(savg_dB − 40))`, `SNR = max(10·log10(xsig/xbase − 1) − 32, −60)`), `subtract` (reference signal removal) |
| `ldpc.rs` / `ldpc_tables.rs` | BP decoder + parity-check matrix |
| `msg.rs` | varicode pack/unpack, commands, grid/callsign codes, 72-bit frame assembly |
| `frame.rs`, `jsc_map.rs` | frame grammar / JSC command map |

**Server task** (`hl2-api`, `api/src/hub.rs`). `spawn_vrx` branches on
`VrxMode`: in `Js8` it builds `js8_shared()`, wraps it as `Js8Tap` →
`Arc<dyn RawSampleTap>` into `ReceiverConfig.tap`, and spawns
`spawn_js8_decode` (`hub.rs:1887`) — a **500 ms** ticker (sliced into 5×100
ms sleeps so `stop` solves in ≤100 ms) calling `js8_step(&shared, now_ms)`.
On ≥ 1 frame it runs the shared decode tail `on_decodes` (`hub.rs:1724`) —
each `Js8Message` is read through the `DecodedMessage` trait (its
`impl DecodedMessage for Js8Message` supplies `display()` = the frame's
display string and `spot_fields()` = the JS8 caller/locator rules,
`decoder.rs`), rows are stamped with the receiver's [`VrxState`] into
`DecodeRow`, spot-able frames are appended to the PSK Reporter sink
(§16.12), and `WsEvent::Log(DecodeLog { decodes })` is broadcast.
`VrxTask.js8` holds the handle; `VrxTask::stop` aborts it.

**Wire contract** (`hl2-common`):

| Item | Shape | Ref |
|------|-------|-----|
| `VrxMode::Js8` | fourth `VrxMode` variant (wire name `js8`) | `lib.rs:164` |
| `DecodedMessage` | the trait every decoder implements; `Js8Message::display()` = the frame's display string, `Js8Message::spot_fields()` = the JS8 caller/locator rules (`decoder.rs`) | `lib.rs` |
| `DecodeRow { text, freq_hz, dt_sec, snr_db, slot_ms, vrx }` | the **one** decoded-message wire row (JS8's `text` = display string) | `lib.rs:218` |
| `WsEvent::Log` | the fanout event; `to_message` (`api/src/ws.rs`) serializes to the wire below | `hub.rs` |

**Wire frame** (text, not binary) — the **same** `"log"` envelope as FT8/FT4,
distinguished per-row by `data[..].vrx.mode = "js8"`:

```json
{"cmd":"log","data":[{"text":"N1MM: HEARTBEAT FN42",
                      "freq_hz":441.0,"dt_sec":-4.5,"snr_db":9.8,
                      "slot_ms":1721395200000,
                      "vrx":{"slot":1,"offset_hz":0,"mode":"js8","bw_hz":2600,
                             "gain_db":0.0,"rate_hz":12000,"muted":false}}, …]}
```

> The former structured JS8 columns (`kind`, `callsign`, `to`, `grid`, `cmd`,
> `num`, `submode`) are now folded into `text` (the decoder's display string),
> so the wire row has one uniform shape across modes. The speed tag that the
> old `submode` column carried is preserved inside that display string.

Sent **continuously** (500 ms tick), **only when ≥ 1 frame decodes on that
tick** — a given speed can fire at most once per ~1.5 s re-arm window (its
cycle boundary), so different speeds interleave on the wire.

**UI panel** (`hl2-ui`, `ui/src/app.rs`). The vrx panel's `<select>` gains
`<option value="js8">`; on switch to JS8 the description line states all
four speeds are decoded continuously, and the single **decode log** table
(newest-first, shared with FT8/FT4) renders this slot's JS8 rows — columns
**Rx** (from `vrx.slot`) / **Mode** (`vrx.mode`, via `vrx_mode_label`) /
Freq / Δt / SNR / Slot / **Text** (the display string); the empty state reads
*Waiting for a decoded message…*. Appended in `on_text` from the `log`
envelope, same shape as `ft8log` used to be.

**End-to-end path.** `VrxMode::Js8` → `spawn_vrx` → `ReceiverConfig.tap =
Js8Tap(js8_shared())` → demod `RawSampleTap::append` (writer) → 500 ms
ticker `js8_step` (reader, off-lock; readiness-gated, per-speed re-arm) →
 `on_decodes` stamping each row's [`VrxState`] + spot via `SpotSink`
 (mode tag = `JS8`) → `WsEvent::Log` → every tab's `on_text`
→ the `decode_log_html` table. Per-pipeline like FT8, but **per-speed continuous**
rather than per-slot.

### 16.12 PSK Reporter spot posting (FT8 / JS8 → pskreporter.info, implemented)

Decoded FT8, FT4, and JS8 rows are posted to [PSK Reporter](http://pskreporter.info) in
the shape wsjtx/js8call post them. Transport is
UDP to `report.pskreporter.info:4739` (wsjtx `PSKReporter.cpp:35-37`),
overridable with `PSKREP_ADDR`.

The spot selector per mode lives **inside the decoder** (`DecodedMessage::spot_fields`),
and the API has a single mode-agnostic bridge (`Pskrep: SpotSink`) that turns any
spot-able row into a wire `Spot` and enqueues it. No per-mode `add_*` methods remain.

**Crate `pskrep`** (no dependencies):

| Item | Location | Notes |
|------|----------|-------|
| `Spot` / `Station` | `pskrep/src/lib.rs` | wire-neutral spot / receiver-info records |
| `PskReporter` (queue + dedup) | `pskrep/src/lib.rs` | per-**callsign+band** TTL 300 s (5 min) — same DX on a different band is posted, same DX on the same band is not (multi-vrx friendly); cache prune 600 s, `max_pending` 2048, **≥ 49 MHz exempt from dedup** (wsjtx `PSKReporter.cpp:382`) |
| `wire::build_packets` | `pskrep/src/wire.rs` | IPFIX datagram builder: descriptors (PEN 30351, templates `0x50E2` receiver / `0x50E3` spot), 4-byte record padding, split at max payload (952 B UDP), per-packet `spot_count` for the sequence counter (wsjtx `PSKReporterIPFIX.cpp:324`) |
| `wire::__golden__` test | `pskrep/src/wire.rs` | byte-for-byte golden datagram vs. an independent re-derivation of the wsjtx layout |

**Hook `api/src/pskrep_hook.rs`**:

| Item | Location | Notes |
|------|----------|-------|
| `spot_fields` (FT8 / FT4) | `hl2/src/receiver/spot.rs` `wsjt_spot_fields` (called from each mode's `DecodedMessage::spot_fields`, `ft8.rs:134` / `ft4.rs:133`) | spot-selection port of wsjtx `pskPost` (`widgets/mainwindow.cpp:7565`): word1 = `CQ`/`QRZ` (with direction tag: `CQ DX`, `CQ NA`, `CQ 559`) or a bare call → **caller = word2, grid = word3** (`R` → word4), per `DecodedText::deCallAndGrid` (`Decoder/decodedtext.cpp:212`); post iff the grid passes `grid_is_square` (`common/src/lib.rs:366`, `[A-R]{2}[0-9]{2}([A-X]{2})?`) **or** the text contains ` CQ `; self-spot suppression when the text contains *both* our base callsign and our 4-char grid. FT8 and FT4 share this one grammar |
| `spot_fields` (JS8) | `hl2/src/receiver/js8/decoder.rs` `js8_spot_fields` (called from `Js8Message::spot_fields`, `decoder.rs:134`) | Mirrors js8call's spotting (mainwindow.cpp `logCallActivity` / `processSpots`): spot **any** frame with a sender, the locator being optional. Caller is the structured `callsign` (heartbeat / compound / compound-directed, or the directed `from`), else the leading `CALL:` word of a data frame's free text (js8call mainwindow.cpp:8292-8306). `locator` = the frame's `grid` when it is a valid square, else empty. Self-spot suppression: caller equals our base call **and** (grid equals our 4-char square, or no grid) |
| `build_spot` / `spot` | `api/src/pskrep_hook.rs` (`Pskrep::build_spot:98`, `impl SpotSink:115`) | The API's *single* spot bridge for **all** modes: reads each row through the `DecodedMessage` trait, calls `spot_fields(&spot_station)`, then computes the wire envelope from the mode-agnostic fields — `freq_hz = rf_hz + m.freq_hz()` (wsjtx `m_freqNominalPeriod + audioFrequency`, `mainwindow.cpp:7590`; `rf_hz` = the vrx slot's tune **plus NCO `offset_hz`** re-read at decode time) and `time_epoch = slot_ms/1000 + dt_sec` (the row's time column). `spot()` then enqueues into the PSK Reporter queue |
| `Pskrep` + `run_sender` | `api/src/pskrep_hook.rs` | **60 s** UDP loop: drain the shared queue → `build_packets` (descriptors on the next 3 reports at startup/reconnect, wsjtx `PSKReporter.cpp:84,152`) → send (splitting across multiple datagrams when > ~8 spots, or more with short grid) → advance sequence (wsjtx `PSKReporter.cpp:324`). Slower cadence than wsjtx's 1 s on purpose: a few virtual receivers (each running FT8 + FT4 + JS8) feed one shared queue, and we batch them into a friendly ≤1/min report rather than flood the collector. Runs on a dedicated std thread + current-thread tokio runtime (started in `api/src/main.rs` — Rocket builds state before its own runtime exists) |
| decode-task hook | `api/src/hub.rs` `on_decodes` (`hub.rs:1724`) | one `SpotSink::spot` per decoded row for **every** digital mode (wsjtx posts **every** decoded line that passes the selector, not just CQs); no per-mode `add_ft8`/`add_ft4`/`add_js8` anymore |

**On/off**: spot posting is **inactive unless `PSK_CALL` is set** (env);
`PSK_GRID` / `PSK_ANTENNA` / `PSK_RIG` fill the receiver-information
record. The decode broadcasts (`WsEvent::Log`) are unaffected.

