//! Native D3D11 recovery work sleeps until a failure or display event requests it.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::frame_signal::FrameSignal;

#[derive(Default)]
pub(crate) struct DeviceRecoverySignal {
    wake: FrameSignal,
    invalidated: AtomicBool,
}

impl DeviceRecoverySignal {
    pub(crate) fn request_check(&self) {
        self.wake.request();
    }

    pub(crate) fn invalidate(&self) {
        self.invalidated.store(true, Ordering::Release);
        self.wake.request();
    }

    /// Consume demand before recovery starts. New failures during recovery
    /// remain queued, and no signal lock spans native calls or UI messages.
    pub(crate) fn wait(&self) -> Option<bool> {
        if !self.wake.wait() {
            return None;
        }
        let mut invalidated = false;
        self.wake
            .deliver(|| invalidated = self.invalidated.swap(false, Ordering::AcqRel))
            .then_some(invalidated)
    }

    pub(crate) fn close(&self) {
        self.wake.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, thread};

    #[test]
    fn invalidation_upgrades_a_queued_health_check() {
        let signal = DeviceRecoverySignal::default();
        signal.request_check();
        signal.invalidate();
        signal.invalidate();
        assert_eq!(signal.wait(), Some(true));
        signal.request_check();
        assert_eq!(signal.wait(), Some(false));
    }

    #[test]
    fn demand_during_recovery_remains_queued() {
        let signal = DeviceRecoverySignal::default();
        signal.invalidate();
        assert_eq!(signal.wait(), Some(true));
        signal.invalidate();
        assert_eq!(signal.wait(), Some(true));
        signal.request_check();
        assert_eq!(signal.wait(), Some(false));
    }

    #[test]
    fn close_releases_an_idle_recovery_worker() {
        let signal = Arc::new(DeviceRecoverySignal::default());
        let worker_signal = signal.clone();
        let worker = thread::spawn(move || worker_signal.wait());
        signal.close();
        assert_eq!(worker.join().unwrap(), None);
        signal.invalidate();
        assert_eq!(signal.wait(), None);
    }
}
