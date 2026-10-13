//! A coalesced frame request that sleeps until the UI has work.

use std::sync::{Condvar, Mutex};

#[derive(Default)]
pub(crate) struct FrameSignal {
    state: Mutex<State>,
    wake: Condvar,
}

#[derive(Default)]
struct State {
    requested: bool,
    closed: bool,
}

impl FrameSignal {
    pub(crate) fn request(&self) {
        let mut state = self.state.lock().unwrap();
        if !state.closed && !state.requested {
            state.requested = true;
            self.wake.notify_one();
        }
    }

    /// Leaves the request set during the vblank wait so further invalidations
    /// join the same frame instead of queuing another one.
    pub(crate) fn wait(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        while !state.requested && !state.closed {
            state = self.wake.wait(state).unwrap();
        }
        !state.closed
    }

    /// Deliver while holding the lock so closing a window cannot race with a
    /// redraw of its HWND after the handle has been destroyed and reused.
    pub(crate) fn deliver(&self, deliver: impl FnOnce()) -> bool {
        let mut state = self.state.lock().unwrap();
        state.requested = false;
        if state.closed {
            return false;
        }
        deliver();
        true
    }

    pub(crate) fn close(&self) {
        self.state.lock().unwrap().closed = true;
        self.wake.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, thread};

    #[test]
    fn requests_before_delivery_coalesce() {
        let signal = FrameSignal::default();
        signal.request();
        assert!(signal.wait());
        signal.request();
        signal.request();
        assert!(signal.deliver(|| {}));
        assert!(!signal.state.lock().unwrap().requested);

        signal.request();
        assert!(signal.wait());
        assert!(signal.deliver(|| {}));
        assert!(!signal.state.lock().unwrap().requested);
    }

    #[test]
    fn closing_wakes_an_idle_worker() {
        let signal = Arc::new(FrameSignal::default());
        let worker_signal = signal.clone();
        let worker = thread::spawn(move || worker_signal.wait());
        signal.close();
        assert!(!worker.join().unwrap());
    }

    #[test]
    fn closing_during_vblank_prevents_delivery() {
        let signal = FrameSignal::default();
        signal.request();
        assert!(signal.wait());
        signal.close();
        assert!(!signal.deliver(|| panic!("redraw after close")));
        signal.request();
        assert!(!signal.wait());
    }
}
