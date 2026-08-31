//! Web Audio playback for the virtual receiver (the `CH_AUDIO` stream).
//!
//! The server emits mono `i16` frames at a fixed sample rate (4 800 Hz). Rather
//! than a single long-lived `AudioBufferSourceNode` (which has to be
//! re-created whenever the stream rate changes), this module keeps a
//! `GainNode` master and, per incoming frame, appends a fresh
//! `AudioBufferSourceNode` scheduled to start right after the previous one.
//! The result is continuous audio even across the 25 ms server-side fan-out
//! ticks, and it naturally handles a fresh `AudioContext` on reconnect.
//!
#![allow(dead_code)] // not every helper is called on every code path yet

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::Float32Array;
use web_sys::{AudioBuffer, AudioBufferSourceNode, AudioContext, GainNode};

/// How far ahead of `current_time()` the next buffer is scheduled. Absorbs
/// underruns as a tiny silent lead instead of a past-time burst.
const LOOKAHEAD: f64 = 0.05; // 50 ms
// Duration of master-gain ramps (Off/On, volume, param-change dips).
const RAMP_SEC: f64 = 0.02; // 20 ms

#[allow(dead_code)]
struct Player {
    ctx: AudioContext,
    gain: GainNode,
    /// Next start time (in `ctx` seconds) for the upcoming buffer source.
    next_start: f64,
}

#[derive(Clone)]
pub struct Audio {
    inner: Rc<RefCell<Option<Player>>>,
    vol: Rc<RefCell<f32>>,
    muted: Rc<RefCell<bool>>,
}

impl Default for Audio {
    fn default() -> Self {
        Self {
            inner: Rc::new(RefCell::new(None)),
            vol: Rc::new(RefCell::new(0.8)),
            muted: Rc::new(RefCell::new(false)),
        }
    }
}

impl Audio {
    pub fn new() -> Self {
        Self::default()
    }

    /// Re-anchor a scheduled start so it is never in the Web Audio "past"
    fn clamp_next_start(next_start: f64, current_time: f64) -> f64 {
        next_start.max(current_time + LOOKAHEAD)
    }

    fn ramp_to(ctx: &AudioContext, gain: &GainNode, value: f32) {
        let now = ctx.current_time();
        let param = gain.gain();
        let _ = param.cancel_scheduled_values(now);
        let _ = param.linear_ramp_to_value_at_time(value, now + RAMP_SEC);
    }

    fn ensure(&self) -> bool {
        let mut guard = self.inner.borrow_mut();
        if let Some(p) = guard.as_mut() {
            let _ = p.ctx.resume();
            return true;
        }
        let ctx = match AudioContext::new() {
            Ok(c) => c,
            Err(_) => return false,
        };
        let gain = match ctx.create_gain() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let _ = gain.connect_with_audio_node(&ctx.destination());
        let level = if *self.muted.borrow() {
            0.0
        } else {
            *self.vol.borrow()
        };
        let _ = gain.gain().set_value(level);
        let now = ctx.current_time();
        let _ = ctx.resume();
        *guard = Some(Player {
            ctx,
            gain,
            next_start: now + LOOKAHEAD,
        });
        true
    }

    pub fn set_gain(&self, linear: f32) {
        *self.vol.borrow_mut() = linear;
        if *self.muted.borrow() {
            return; // master already silent; leave it — restore on unmute
        }
        let guard = self.inner.borrow();
        if let Some(p) = guard.as_ref() {
            if (p.gain.gain().value() - linear).abs() < 1e-6 {
                return; // already at this level (steady state) — skip the ramp
            }
            Self::ramp_to(&p.ctx, &p.gain, linear);
        }
    }

    pub fn mute(&self) {
        if *self.muted.borrow() {
            return; // already muted (steady state) — skip the ramp
        }
        *self.muted.borrow_mut() = true;
        if let Some(p) = self.inner.borrow().as_ref() {
            Self::ramp_to(&p.ctx, &p.gain, 0.0);
        }
    }

    pub fn is_muted(&self) -> bool {
        *self.muted.borrow()
    }

    pub fn unmute(&self) {
        *self.muted.borrow_mut() = false;
        let vol = *self.vol.borrow();
        if let Some(p) = self.inner.borrow_mut().as_mut() {
            p.next_start = p.ctx.current_time() + LOOKAHEAD;
            Self::ramp_to(&p.ctx, &p.gain, vol);
        }
        // Context not created yet? The first `push_samples` → `ensure()` seeds
        // the master from `vol` / `muted` when it is built.
    }

    /// Append a block of mono `i16` samples (already in i16 range) to the
    /// playback timeline at the given sample rate. The start time is
    /// re-anchored each frame so an underrun becomes a silent lead, not a
    /// past-time burst. Returns `true` if the samples were queued.
    pub fn push_samples(&self, samples: &[i16], rate_hz: u16) -> bool {
        if samples.is_empty() || rate_hz == 0 {
            return false;
        }
        if !self.ensure() {
            return false;
        }
        let mut guard = self.inner.borrow_mut();
        let Some(p) = guard.as_mut() else {
            return false;
        };

        // Copy the i16 frames into [-1, 1] for the AudioBuffer. This web-sys
        // version exposes `copy_to_channel` (write) and `get_channel_data`
        // (read), both taking/returning the full channel; we write via
        // `copy_to_channel` with a `Float32Array`.
        let n = samples.len();
        let float: Vec<f32> = samples.iter().map(|s| *s as f32 / 32768.0).collect();

        let ab: AudioBuffer = match p.ctx.create_buffer(1, n as u32, rate_hz as f32) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let arr = Float32Array::from(&float[..]);
        if ab
            .copy_to_channel_with_f32_array_and_start_in_channel(&arr, 0, 0)
            .is_err()
        {
            return false;
        }

        let src: AudioBufferSourceNode = match p.ctx.create_buffer_source() {
            Ok(s) => s,
            Err(_) => return false,
        };
        let _ = src.set_buffer(Some(&ab));
        let _ = src.connect_with_audio_node(&p.gain);

        let start = Self::clamp_next_start(p.next_start, p.ctx.current_time());
        let _ = src.start_with_when(start);
        p.next_start = start + (n as f64 / rate_hz as f64);
        true
    }

    /// Re-anchor the playback timeline to just-ahead-of-now (used across the
    /// server-side sideband/BW/gain rebuild gap) so the next frame starts with
    /// a clean head lead instead of a stale backlog. Does not alter the level.
    pub fn reset(&self) {
        if let Some(p) = self.inner.borrow_mut().as_mut() {
            p.next_start = p.ctx.current_time() + LOOKAHEAD;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_keeps_running_start_when_ahead() {
        let now = 100.0;
        let next = now + 0.1;
        assert_eq!(Audio::clamp_next_start(next, now), next);
    }

    #[test]
    fn clamp_reanchors_when_in_past() {
        let next = 99.5; // stale, behind the clock
        let now = 100.0;
        assert_eq!(Audio::clamp_next_start(next, now), now + LOOKAHEAD);
    }

    #[test]
    fn clamp_never_schedules_into_past() {
        // Even with `next_start` at 0, the clamp pushes past `now`.
        assert!(Audio::clamp_next_start(0.0, 50.0) > 50.0);
    }
}
