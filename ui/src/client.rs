//! A thin `WebSocket` wrapper for the browser.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{MessageEvent, WebSocket};

static NEXT_WS_ID: AtomicU64 = AtomicU64::new(1);

struct Cbs {
    open: Rc<dyn Fn()>,
    text: Rc<dyn Fn(&str)>,
    binary: Rc<dyn Fn(&[u8])>,
    close: Rc<dyn Fn()>,
    error: Rc<dyn Fn()>,
}

#[allow(dead_code)]
pub struct WsClient {
    id: u64,
    ws: WebSocket,
    _keepalive: Rc<RefCell<Vec<JsValue>>>,
}

impl WsClient {
    pub fn connect<O, F, B, C, E>(
        url: &str,
        on_open: O,
        on_text: F,
        on_binary: B,
        on_close: C,
        on_error: E,
    ) -> Result<Self, JsValue>
    where
        O: Fn() + 'static,
        F: Fn(&str) + 'static,
        B: Fn(&[u8]) + 'static,
        C: Fn() + 'static,
        E: Fn() + 'static,
    {
        let id = NEXT_WS_ID.fetch_add(1, Ordering::Relaxed);
        let ws = WebSocket::new(url)?;
        ws.set_binary_type(web_sys::BinaryType::Arraybuffer);

        let cbs = Rc::new(RefCell::new(Cbs {
            open: Rc::new(on_open),
            text: Rc::new(on_text),
            binary: Rc::new(on_binary),
            close: Rc::new(on_close),
            error: Rc::new(on_error),
        }));
        let keepalive: Rc<RefCell<Vec<JsValue>>> = Rc::new(RefCell::new(Vec::new()));

        // onopen: connection is up and ready for I/O.
        let cbs_open = cbs.clone();
        let keep = keepalive.clone();
        let open_closure: Closure<dyn FnMut()> = Closure::new(move || (cbs_open.borrow().open)());
        ws.set_onopen(Some(
            open_closure.as_ref().unchecked_ref::<js_sys::Function>(),
        ));
        keep.borrow_mut().push(open_closure.into_js_value());

        // onmessage: dispatch Text vs ArrayBuffer to the right callback.
        let cbs_msg = cbs.clone();
        let keep = keepalive.clone();
        let msg_closure: Closure<dyn FnMut(MessageEvent)> =
            Closure::new(move |ev: MessageEvent| {
                let data = ev.data();
                if data.is_string() {
                    (cbs_msg.borrow().text)(&data.as_string().unwrap());
                } else if let Ok(buf) = data.dyn_into::<js_sys::ArrayBuffer>() {
                    let u8 = js_sys::Uint8Array::new(&buf);
                    (cbs_msg.borrow().binary)(&u8.to_vec());
                }
            });
        ws.set_onmessage(Some(
            msg_closure.as_ref().unchecked_ref::<js_sys::Function>(),
        ));
        keep.borrow_mut().push(msg_closure.into_js_value());

        // onclose.
        let cbs_close = cbs.clone();
        let keep = keepalive.clone();
        let close_closure: Closure<dyn FnMut()> =
            Closure::new(move || (cbs_close.borrow().close)());
        ws.set_onclose(Some(
            close_closure.as_ref().unchecked_ref::<js_sys::Function>(),
        ));
        keep.borrow_mut().push(close_closure.into_js_value());

        // onerror.
        let cbs_err = cbs.clone();
        let keep = keepalive.clone();
        let err_closure: Closure<dyn FnMut()> = Closure::new(move || (cbs_err.borrow().error)());
        ws.set_onerror(Some(
            err_closure.as_ref().unchecked_ref::<js_sys::Function>(),
        ));
        keep.borrow_mut().push(err_closure.into_js_value());

        Ok(Self {
            id,
            ws,
            _keepalive: keepalive,
        })
    }

    /// The unique id assigned at construction. Lets the app detect that a
    /// late `onclose`/`onerror` belongs to a socket it has already replaced.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Send a JSON text message.
    pub fn send_text(&self, msg: &str) -> Result<(), JsValue> {
        self.ws.send_with_str(msg)
    }

    /// Close the connection cleanly.
    pub fn close(&self) {
        let _ = self.ws.close();
    }
}

/// Parse a binary spectrum payload. Layout:
///
/// ```text
/// [u16 LE channel][u16 LE bin_count][u16 LE mags (bin_count ×)]
/// ```
pub fn parse_binary(bytes: &[u8]) -> Result<(u16, Vec<u16>), String> {
    if bytes.len() < 4 {
        return Err(format!("binary payload too short: {} bytes", bytes.len()));
    }
    let channel = u16::from_le_bytes([bytes[0], bytes[1]]);
    let bin_count = u16::from_le_bytes([bytes[2], bytes[3]]);
    let expected = 4 + 2 * bin_count as usize;
    if bytes.len() < expected {
        return Err(format!(
            "binary payload expected {} bytes, got {}",
            expected,
            bytes.len()
        ));
    }
    let mut mags = Vec::with_capacity(bin_count as usize);
    for k in 0..bin_count as usize {
        let lo = 4 + 2 * k;
        mags.push(u16::from_le_bytes([bytes[lo], bytes[lo + 1]]));
    }
    Ok((channel, mags))
}

/// Parse a binary audio payload (on `CH_AUDIO`). Layout:
///
/// ```text
/// [u16 LE channel][u16 LE slot][u32 LE seq][u16 LE rate_hz][i16 LE samples (N × 2)]
/// ```
///
pub fn parse_audio(bytes: &[u8]) -> Option<(u16, u32, u16, Vec<i16>)> {
    // 2 (ch) + 2 (slot) + 4 (seq) + 2 (rate) = 10 header bytes minimum.
    if bytes.len() < 10 {
        return None;
    }
    let slot = u16::from_le_bytes([bytes[2], bytes[3]]);
    let seq = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let rate_hz = u16::from_le_bytes([bytes[8], bytes[9]]);
    let payload = &bytes[10..];
    if payload.len() % 2 != 0 {
        return None;
    }
    let mut samples = Vec::with_capacity(payload.len() / 2);
    let mut i = 0usize;
    while i + 1 < payload.len() {
        samples.push(i16::from_le_bytes([payload[i], payload[i + 1]]));
        i += 2;
    }
    Some((slot, seq, rate_hz, samples))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_binary_basic() {
        let mut bytes = vec![0x01, 0x00, 0x04, 0x00];
        for i in 0u16..4 {
            bytes.extend_from_slice(&(i * 1000).to_le_bytes());
        }
        let (ch, mags) = parse_binary(&bytes).unwrap();
        assert_eq!(ch, 1);
        assert_eq!(mags.len(), 4);
        assert_eq!(mags[1], 1000);
        assert_eq!(mags[3], 3000);
    }

    #[test]
    fn parse_binary_too_short() {
        assert!(parse_binary(&[0x01, 0x00]).is_err());
    }

    #[test]
    fn parse_audio_reads_slot_seq_rate_samples() {
        // ch=3, slot=2, seq=0x01020304, rate=4800, samples=[100, -200]
        let mut bytes = vec![3u8, 0];
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&0x0102_0304u32.to_le_bytes());
        bytes.extend_from_slice(&4_800u16.to_le_bytes());
        bytes.extend_from_slice(&100i16.to_le_bytes());
        bytes.extend_from_slice(&(-200i16).to_le_bytes());
        let (slot, seq, rate, samples) = parse_audio(&bytes).unwrap();
        assert_eq!(slot, 2);
        assert_eq!(seq, 0x0102_0304);
        assert_eq!(rate, 4_800);
        assert_eq!(samples, vec![100i16, -200]);
    }

    #[test]
    fn parse_audio_rejects_short_frames() {
        // 9 bytes = header minus one.
        assert!(parse_audio(&[0u8; 9]).is_none());
        // 10 bytes = header only, no samples: valid, empty.
        let (slot, seq, rate, samples) = parse_audio(&[0u8; 10]).unwrap();
        assert_eq!((slot, seq, rate), (0u16, 0u32, 0u16));
        assert!(samples.is_empty());
    }
}
