//! Snapshot transitions: the window keeps its last presented frame and
//! sweeps it away diagonally over the frames that follow, with a shimmer band
//! along the edge. Made for whole-window state changes such as a theme switch.
//!
//! The renderer composites the snapshot on every present while the
//! transition plays, so the window keeps presenting (like a pulsing quad)
//! without re-rendering any view.

use super::Window;
use crate::{Hsla, SnapshotTransitionFrame, hsla_to_rgba};
use scheduler::Instant;
use std::time::Duration;

/// How a [`Window::begin_snapshot_transition`] plays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnapshotTransition {
    /// How long the edge takes to sweep from the top-left corner past the
    /// bottom-right one.
    pub duration: Duration,
    /// Width of the shimmer band as a fraction of the window diagonal (e.g.
    /// 0.18).
    pub band: f32,
    /// Highlight colour added in the band (premultiplied-alpha blend over
    /// both frames); alpha = peak strength.
    pub shimmer: Hsla,
}

/// A transition the renderer holds a snapshot for.
pub(crate) struct ActiveSnapshotTransition {
    transition: SnapshotTransition,
    /// The first present after the snapshot, which the clock starts from, so
    /// a slow re-render of the changed state doesn't eat into the sweep.
    started: Option<Instant>,
}

/// Ease-in-out, `cubic-bezier(0.65, 0, 0.35, 1)`.
fn ease(t: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    // Solve x(s) = t for the bezier parameter s, then return y(s).
    let bezier = |s: f32, p1: f32, p2: f32| {
        let u = 1.0 - s;
        3.0 * u * u * s * p1 + 3.0 * u * s * s * p2 + s * s * s
    };
    let (mut low, mut high) = (0.0f32, 1.0f32);
    for _ in 0..24 {
        let mid = (low + high) * 0.5;
        if bezier(mid, 0.65, 0.35) < t {
            low = mid;
        } else {
            high = mid;
        }
    }
    bezier((low + high) * 0.5, 0.0, 1.0)
}

impl Window {
    /// Captures the currently presented frame and plays the transition over
    /// the following frames: the new frames draw normally and the old frame
    /// is composited over them, masked by a diagonal edge sweeping from the
    /// top-left corner to the bottom-right one, with a shimmer band along it.
    ///
    /// The caller changes state (e.g. the theme) right after calling this, in
    /// the same update. The window presents every frame until the transition
    /// ends, without re-rendering views. Starting one while another plays
    /// restarts from the frame drawn last; resizing the window ends it.
    /// Where the platform doesn't support transitions this does nothing, and
    /// the change shows at once. Callers should skip it under
    /// [`App::reduce_motion`](crate::App::reduce_motion).
    pub fn begin_snapshot_transition(&mut self, transition: SnapshotTransition) {
        self.snapshot_transition = None;
        if transition.duration.is_zero() {
            return;
        }
        if !self
            .platform_window
            .begin_snapshot_transition(&self.rendered_frame.scene)
        {
            return;
        }
        self.snapshot_transition = Some(ActiveSnapshotTransition {
            transition,
            started: None,
        });
        self.needs_present.set(true);
        self.invalidator.wake_platform();
    }

    /// Whether a snapshot transition is playing.
    pub fn snapshot_transition_active(&self) -> bool {
        self.snapshot_transition.is_some()
    }

    /// Ends the transition at once, as when the window resizes.
    pub(crate) fn end_snapshot_transition(&mut self) {
        if self.snapshot_transition.take().is_some() {
            self.needs_present.set(true);
        }
    }

    /// The transition frame the present at `now` composites, ending the
    /// transition once its duration has passed.
    pub(crate) fn snapshot_transition_frame(
        &mut self,
        now: Instant,
    ) -> Option<SnapshotTransitionFrame> {
        let active = self.snapshot_transition.as_mut()?;
        let started = *active.started.get_or_insert(now);
        let transition = active.transition;
        let elapsed = now.saturating_duration_since(started);
        if elapsed >= transition.duration {
            self.snapshot_transition = None;
            return None;
        }
        let rgba = hsla_to_rgba(transition.shimmer);
        Some(SnapshotTransitionFrame {
            progress: ease(elapsed.as_secs_f32() / transition.duration.as_secs_f32()),
            band: transition.band.max(0.0),
            shimmer: [rgba.red, rgba.green, rgba.blue, rgba.alpha],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ease;

    #[test]
    fn the_sweep_eases_in_and_out() {
        assert_eq!(ease(0.0), 0.0);
        assert!((ease(0.5) - 0.5).abs() < 1e-3);
        assert!((ease(1.0) - 1.0).abs() < 1e-3);
        // Slow at both ends, fast in the middle.
        assert!(ease(0.1) < 0.05);
        assert!(ease(0.9) > 0.95);
    }
}
