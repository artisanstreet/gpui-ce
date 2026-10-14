//! Bound retries without treating an attempted frame or a cache reset as recovery.

#[derive(Default)]
pub(super) struct FrameErrorState {
    consecutive_errors: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum FrameErrorAction {
    Continue,
    ResetResources,
    Abort,
}

impl FrameErrorState {
    pub(super) fn begin_frame(&mut self, has_error: bool) -> FrameErrorAction {
        if !has_error {
            return FrameErrorAction::Continue;
        }
        self.consecutive_errors = self.consecutive_errors.saturating_add(1);
        match self.consecutive_errors {
            6 => FrameErrorAction::ResetResources,
            11.. => FrameErrorAction::Abort,
            _ => FrameErrorAction::Continue,
        }
    }

    pub(super) fn count(&self) -> u32 {
        self.consecutive_errors
    }

    pub(super) fn frame_presented(&mut self, has_pending_error: bool, device_lost: bool) {
        // Submission/presentation can report validation errors synchronously.
        // Leave them for begin_frame and keep the streak if either failed.
        if !has_pending_error && !device_lost {
            self.consecutive_errors = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_validation_stops_after_one_resource_reset() {
        let mut state = FrameErrorState::default();
        for failure in 1..=11 {
            assert_eq!(
                state.begin_frame(true),
                match failure {
                    6 => FrameErrorAction::ResetResources,
                    11 => FrameErrorAction::Abort,
                    _ => FrameErrorAction::Continue,
                }
            );
            assert_eq!(state.count(), failure);

            // The cache-reset frame and subsequent skipped acquisitions have
            // no pending error. Neither one has presented a recovered frame.
            for _ in 0..3 {
                assert_eq!(state.begin_frame(false), FrameErrorAction::Continue);
                assert_eq!(state.count(), failure);
            }
        }
    }

    #[test]
    fn successful_frame_starts_a_new_recovery_budget() {
        let mut state = FrameErrorState::default();
        for _ in 0..6 {
            state.begin_frame(true);
        }
        state.frame_presented(false, false);
        assert_eq!(state.count(), 0);
        for _ in 0..5 {
            assert_eq!(state.begin_frame(true), FrameErrorAction::Continue);
        }
        assert_eq!(state.begin_frame(true), FrameErrorAction::ResetResources);
    }

    #[test]
    fn presentation_errors_and_device_loss_do_not_reset_the_streak() {
        let mut state = FrameErrorState::default();
        for _ in 0..5 {
            state.begin_frame(true);
        }
        state.frame_presented(true, false);
        assert_eq!(state.count(), 5);
        state.frame_presented(false, true);
        assert_eq!(state.count(), 5);
        assert_eq!(state.begin_frame(true), FrameErrorAction::ResetResources);
        state.frame_presented(true, true);
        assert_eq!(state.count(), 6);
        state.frame_presented(false, false);
        assert_eq!(state.count(), 0);
    }
}
