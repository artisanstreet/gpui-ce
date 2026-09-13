use scheduler::Instant;
use std::{num::NonZeroU32, time::Duration};

/// A phase-preserving limit over display callbacks. Missed deadlines are
/// coalesced, never queued, and non-divisor limits do not round down to half
/// the display refresh rate.
#[derive(Default)]
pub(crate) struct FrameRateLimiter {
    rate: Option<NonZeroU32>,
    next: Option<Instant>,
}

impl FrameRateLimiter {
    pub(crate) fn set(&mut self, rate: Option<NonZeroU32>) {
        if self.rate != rate {
            self.rate = rate;
            self.next = None;
        }
    }

    pub(crate) fn rate(&self) -> Option<NonZeroU32> {
        self.rate
    }

    pub(crate) fn admit(&mut self, now: Instant) -> bool {
        let Some(rate) = self.rate else { return true };
        let interval = Duration::from_secs_f64(1.0 / f64::from(rate.get()));
        if let Some(next) = self.next {
            if now < next {
                return false;
            }
            self.next = Some(if now.duration_since(next) < interval {
                next + interval
            } else {
                now + interval
            });
        } else {
            self.next = Some(now + interval);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_divisor_limits_preserve_average_cadence() {
        for (display, rate) in [(165, 144), (240, 144), (240, 120), (165, 30), (240, 500)] {
            let mut limiter = FrameRateLimiter::default();
            limiter.set(NonZeroU32::new(rate));
            let start = Instant::now();
            let frames = (0..display * 10)
                .filter(|tick| {
                    limiter.admit(
                        start + Duration::from_secs_f64(f64::from(*tick) / f64::from(display)),
                    )
                })
                .count();
            assert!(
                (frames as i64 - i64::from(rate.min(display) * 10)).abs() <= 1,
                "{display}/{rate}: {frames}"
            );
        }
    }

    #[test]
    fn changes_and_idle_do_not_leave_stale_deadlines() {
        let mut limiter = FrameRateLimiter::default();
        let start = Instant::now();
        limiter.set(NonZeroU32::new(30));
        assert!(limiter.admit(start));
        assert!(!limiter.admit(start + Duration::from_millis(1)));
        let later = start + Duration::from_secs(10);
        assert!(limiter.admit(later));
        assert!(!limiter.admit(later + Duration::from_millis(1)));
        limiter.set(None);
        assert!(limiter.admit(later + Duration::from_millis(1)));
        limiter.set(NonZeroU32::new(240));
        assert!(limiter.admit(later + Duration::from_millis(1)));
    }
}
