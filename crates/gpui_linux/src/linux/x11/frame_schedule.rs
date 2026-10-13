use std::{
    cell::Cell,
    time::{Duration, Instant},
};

/// Coalesces invalidations and preserves cadence only while frames are wanted.
#[derive(Default)]
pub(crate) struct FrameSchedule {
    requested: Cell<bool>,
    last_frame: Cell<Option<Instant>>,
}

impl FrameSchedule {
    pub(crate) fn request(&self) -> bool {
        !self.requested.replace(true)
    }

    pub(crate) fn requested(&self) -> bool {
        self.requested.get()
    }

    pub(crate) fn deadline(&self, now: Instant, interval: Duration) -> Option<Instant> {
        self.requested.get().then(|| {
            self.last_frame
                .get()
                .map(|last| (last + interval).max(now))
                .unwrap_or(now)
        })
    }

    pub(crate) fn frame_started(&self, now: Instant) {
        self.requested.set(false);
        self.last_frame.set(Some(now));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_window_has_no_deadline() {
        let schedule = FrameSchedule::default();
        let now = Instant::now();
        let interval = Duration::from_millis(16);
        assert_eq!(schedule.deadline(now, interval), None);
        assert!(schedule.request());
        assert_eq!(schedule.deadline(now, interval), Some(now));
        schedule.frame_started(now);
        assert_eq!(schedule.deadline(now + interval, interval), None);
    }

    #[test]
    fn animation_requests_coalesce_and_preserve_cadence() {
        let schedule = FrameSchedule::default();
        let now = Instant::now();
        let interval = Duration::from_millis(16);
        schedule.request();
        schedule.frame_started(now);
        assert!(schedule.request());
        assert!(!schedule.request());
        assert_eq!(
            schedule.deadline(now + Duration::from_millis(3), interval),
            Some(now + interval)
        );
        schedule.frame_started(now + interval);
        assert!(!schedule.requested());
        assert!(schedule.request());
        assert_eq!(
            schedule.deadline(now + interval, interval),
            Some(now + interval * 2)
        );
    }

    #[test]
    fn wake_after_idle_has_no_delay_or_catch_up_frames() {
        let schedule = FrameSchedule::default();
        let now = Instant::now();
        let interval = Duration::from_millis(16);
        schedule.request();
        schedule.frame_started(now);
        let wake = now + Duration::from_secs(60);
        assert!(schedule.request());
        assert_eq!(schedule.deadline(wake, interval), Some(wake));
        schedule.frame_started(wake);
        assert_eq!(schedule.deadline(wake, interval), None);
    }
}
