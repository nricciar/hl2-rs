//! 10× upsampler: 4 800 Hz demod audio → 48 kHz for the WM8731.
//!
//! A 2-tap linear interpolation (piecewise-linear) resampler. For each pair of
//! consecutive input samples it emits `RATE_RATIO` points on the straight line
//! between them. This is the cheapest resampler with acceptable HF roll-off for
//! a narrowband SSB/AM voice passband; it is deterministic, allocation-free,
//! and easy to test against a known ramp (PROTOCOL.md §16 audio).

/// Input rate (Hz) of the virtual-receiver audio (`AudioConfig::default`).
pub const SRC_RATE_HZ: u32 = 4_800;
/// Output rate (Hz) of the WM8731 / I2S stream.
pub const DST_RATE_HZ: u32 = 48_000;

/// Integer upsample factor. Must divide `DST_RATE_HZ` evenly.
pub const RATE_RATIO: usize = DST_RATE_HZ as usize / SRC_RATE_HZ as usize; // 10

/// A 2-tap linear 10× upsampler. Holds the previous input sample so that each
/// new input extends the interpolation line forward.
#[derive(Debug)]
pub struct Upsampler {
    prev: i16,
}

impl Upsampler {
    pub const fn new() -> Self {
        Self { prev: 0 }
    }

    /// Upsample `src` (input-rate `i16`) into `dst`, returning the number of
    /// output samples written. `dst` must hold at least
    /// `src.len() * RATE_RATIO` samples; any beyond that are untouched. On a
    /// length mismatch the write is clamped rather than panicking, so a
    /// slightly-off block can never overflow a caller-provided buffer.
    pub fn upsample(&mut self, src: &[i16], dst: &mut [i16]) -> usize {
        let mut o = 0usize;
        for &s in src {
            let prev = self.prev;
            let step = (s as i32 - prev as i32) / RATE_RATIO as i32;
            for k in 0..RATE_RATIO {
                let v = (prev as i32 + step * k as i32) as i16;
                if o < dst.len() {
                    dst[o] = v;
                }
                o += 1;
            }
            self.prev = s;
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_is_linear_over_full_ratio() {
        let mut u = Upsampler::new();
        // prev=0; feed a single step of +10 → expect 0,1,2,...,9 then land on 10.
        let mut dst = [0i16; RATE_RATIO];
        assert_eq!(u.upsample(&[10], &mut dst), RATE_RATIO);
        let expected: Vec<i16> = (0..RATE_RATIO as i16)
            .map(|k| 10 * k / RATE_RATIO as i16)
            .collect();
        assert_eq!(dst, expected.as_slice());
        // The next step holds at 10 at k==0 and ramps to 20 by the end.
        assert_eq!(u.prev, 10);
    }

    #[test]
    fn constant_signal_is_constant() {
        let mut u = Upsampler::new();
        u.upsample(&[500], &mut [0; 2]); // prime prev=500
        let mut dst = [7i16; RATE_RATIO];
        u.upsample(&[500], &mut dst);
        assert!(dst.iter().all(|&x| x == 500));
    }

    #[test]
    fn clamps_to_dst_capacity() {
        let mut u = Upsampler::new();
        let mut dst = [0i16; 3];
        // A 2-sample input would emit 20, but only 3 fit.
        assert_eq!(u.upsample(&[100, 200], &mut dst), 20);
        assert_eq!(dst, [0, 10, 20]);
    }

    #[test]
    fn ratio_is_exact_10() {
        assert_eq!(RATE_RATIO, 10);
    }
}
