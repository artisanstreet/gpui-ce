//! Which pixels a present has to repaint.
//!
//! The swap chain flips between `buffer_count` back buffers, so the buffer a frame renders into
//! still holds the frame presented `buffer_count` presents earlier. A partial frame therefore
//! repaints its own damage plus everything the previous `buffer_count - 1` presents changed.
//! Blur filters read pixels around their bounds, so damage touching one grows to its whole
//! footprint. Free of Direct3D types, so these rules are testable on any host.

use std::collections::VecDeque;

/// A rectangle of whole device pixels, half-open: `left..right` by `top..bottom`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DamageRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl DamageRect {
    /// The smallest pixel rectangle covering the fractional rectangle at `x, y` of
    /// `width` by `height`: edges round outward, so partly covered pixels count.
    pub fn covering(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            left: x.floor() as i32,
            top: y.floor() as i32,
            right: (x + width).ceil() as i32,
            bottom: (y + height).ceil() as i32,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.right <= self.left || self.bottom <= self.top
    }

    pub fn area(&self) -> i64 {
        if self.is_empty() {
            0
        } else {
            (self.right - self.left) as i64 * (self.bottom - self.top) as i64
        }
    }

    /// The bounding rectangle of both; an empty side contributes nothing.
    pub fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Self {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }

    /// The overlap of both, empty when they do not overlap.
    pub fn intersect(self, other: Self) -> Self {
        Self {
            left: self.left.max(other.left),
            top: self.top.max(other.top),
            right: self.right.min(other.right),
            bottom: self.bottom.min(other.bottom),
        }
    }

    pub fn intersects(&self, other: &Self) -> bool {
        !self.intersect(*other).is_empty()
    }
}

/// What one present repaints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Repaint {
    /// The screen already shows this frame: no render, no present.
    Nothing,
    /// Only this rectangle; every pixel outside it is retained from the back buffer.
    Partial(DamageRect),
    /// The whole target.
    Full,
}

/// The facts about a scene that one present is planned from.
pub(crate) struct FrameDamage<'a> {
    pub generation: u64,
    /// The scene's damage against generation `generation - 1`; `None` means anything changed.
    pub scene: Option<&'a [DamageRect]>,
    /// Pulsing quads, which change on every present.
    pub pulses: &'a [DamageRect],
    /// Footprints of blur filters. Damage touching one grows to cover all of it: the blur reads
    /// the whole footprint and writes inside it.
    pub filters: &'a [DamageRect],
    pub width: i32,
    pub height: i32,
}

/// A planned present. Hand it back to [`DamageTracker::presented`] once it reached the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FramePlan {
    generation: u64,
    /// The pixels that differ from the previous present; `None` means all of them.
    changed: Option<DamageRect>,
    pub repaint: Repaint,
}

/// Remembers what recent presents changed, to plan the next one.
pub(crate) struct DamageTracker {
    buffer_count: usize,
    last_generation: Option<u64>,
    /// What each of the last `buffer_count - 1` presents changed, oldest first; `None` means the
    /// whole target.
    history: VecDeque<Option<DamageRect>>,
    /// The debug tint over the last present, which the next present paints over.
    overlay: Option<DamageRect>,
}

impl DamageTracker {
    pub fn new(buffer_count: usize) -> Self {
        Self {
            buffer_count,
            last_generation: None,
            history: VecDeque::with_capacity(buffer_count),
            overlay: None,
        }
    }

    /// Forgets what the back buffers hold, so the next `buffer_count` presents repaint
    /// everything. For anything that leaves them unknown: a resize, a lost device, recreated
    /// targets, a failed or unusual present.
    pub fn invalidate(&mut self) {
        self.last_generation = None;
        self.history.clear();
        self.overlay = None;
    }

    pub fn plan(&self, frame: &FrameDamage) -> FramePlan {
        let target = DamageRect {
            left: 0,
            top: 0,
            right: frame.width,
            bottom: frame.height,
        };
        let pulses = frame
            .pulses
            .iter()
            .fold(DamageRect::default(), |damage, pulse| damage.union(*pulse));
        let changed = match self.last_generation {
            Some(last) if last == frame.generation => Some(pulses),
            Some(last) if last.wrapping_add(1) == frame.generation => frame.scene.map(|scene| {
                scene
                    .iter()
                    .fold(pulses, |damage, rect| damage.union(*rect))
            }),
            _ => None,
        }
        .map(|changed| {
            spread(
                changed.union(self.overlay.unwrap_or_default()),
                frame.filters,
            )
            .intersect(target)
        });

        let repaint = match changed {
            None => Repaint::Full,
            Some(changed) if changed.is_empty() => Repaint::Nothing,
            // The back buffer has not seen enough presents for its age to be known.
            Some(_) if self.history.len() + 1 < self.buffer_count => Repaint::Full,
            Some(changed) => self
                .history
                .iter()
                .try_fold(changed, |stale, earlier| {
                    earlier.map(|earlier| stale.union(earlier))
                })
                .map(|stale| spread(stale, frame.filters).intersect(target))
                // Past half the target, scissoring saves little and one full pass is simpler.
                .filter(|stale| stale.area() * 2 <= target.area())
                .map_or(Repaint::Full, Repaint::Partial),
        };
        FramePlan {
            generation: frame.generation,
            changed,
            repaint,
        }
    }

    /// Records that `plan` reached the screen.
    pub fn presented(&mut self, plan: &FramePlan) {
        self.last_generation = Some(plan.generation);
        if plan.repaint == Repaint::Nothing {
            return;
        }
        self.overlay = None;
        self.history.push_back(plan.changed);
        while self.history.len() >= self.buffer_count.max(1) {
            self.history.pop_front();
        }
    }

    /// Records a debug tint drawn over `rect` of the last present. Those pixels differ from the
    /// scene, so the next present paints them over and later ones treat them as changed.
    pub fn overlaid(&mut self, rect: DamageRect) {
        if let Some(Some(last)) = self.history.back_mut() {
            *last = last.union(rect);
        }
        self.overlay = Some(rect);
    }
}

/// Grows `damage` until every filter footprint it touches lies wholly inside it.
fn spread(mut damage: DamageRect, filters: &[DamageRect]) -> DamageRect {
    loop {
        let grown = filters.iter().fold(damage, |grown, filter| {
            if grown.intersects(filter) {
                grown.union(*filter)
            } else {
                grown
            }
        });
        if grown == damage {
            return damage;
        }
        damage = grown;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUFFERS: usize = 3;

    fn rect(left: i32, top: i32, right: i32, bottom: i32) -> DamageRect {
        DamageRect {
            left,
            top,
            right,
            bottom,
        }
    }

    fn frame<'a>(
        generation: u64,
        scene: Option<&'a [DamageRect]>,
        pulses: &'a [DamageRect],
    ) -> FrameDamage<'a> {
        FrameDamage {
            generation,
            scene,
            pulses,
            filters: &[],
            width: 100,
            height: 100,
        }
    }

    /// Presents full frames until the back buffers' ages are known.
    fn settled(pulse: DamageRect) -> DamageTracker {
        let mut tracker = DamageTracker::new(BUFFERS);
        for _ in 0..BUFFERS {
            let plan = tracker.plan(&frame(1, None, &[pulse]));
            assert_eq!(plan.repaint, Repaint::Full);
            tracker.presented(&plan);
        }
        tracker
    }

    fn present(tracker: &mut DamageTracker, frame: &FrameDamage) -> Repaint {
        let plan = tracker.plan(frame);
        tracker.presented(&plan);
        plan.repaint
    }

    #[test]
    fn edges_round_outward_to_whole_pixels() {
        assert_eq!(DamageRect::covering(0.5, 0.5, 1.0, 1.0), rect(0, 0, 2, 2));
        assert_eq!(DamageRect::covering(2.0, 3.0, 4.0, 5.0), rect(2, 3, 6, 8));
        assert_eq!(
            DamageRect::covering(-1.5, -0.25, 1.0, 0.5),
            rect(-2, -1, 0, 1)
        );
        assert!(DamageRect::covering(4.0, 4.0, 0.0, 3.0).is_empty());
    }

    #[test]
    fn the_first_presents_repaint_everything_until_every_buffer_was_drawn() {
        let pulse = rect(10, 10, 20, 20);
        let mut tracker = DamageTracker::new(BUFFERS);
        // Frame one replaces whatever the buffers held; the next two render into buffers that
        // have never been drawn even though only the pulse changed.
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            Repaint::Full
        );
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            Repaint::Full
        );
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            Repaint::Full
        );
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            Repaint::Partial(pulse)
        );
    }

    #[test]
    fn a_pulse_only_present_repaints_the_pulses() {
        let pulses = [rect(10, 10, 20, 20), rect(30, 10, 40, 20)];
        let mut tracker = settled(pulses[0]);
        // The scene damage is ignored: it describes the step into this generation, which
        // earlier presents already showed.
        let scene = [rect(0, 0, 90, 90)];
        assert_eq!(
            present(&mut tracker, &frame(1, Some(&scene), &pulses)),
            Repaint::Partial(rect(10, 10, 40, 20))
        );
    }

    #[test]
    fn nothing_changed_skips_the_present() {
        let mut tracker = settled(rect(10, 10, 20, 20));
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[])),
            Repaint::Nothing
        );
        assert_eq!(
            present(&mut tracker, &frame(2, Some(&[]), &[])),
            Repaint::Nothing
        );
    }

    #[test]
    fn a_skipped_present_leaves_the_back_buffers_alone() {
        let pulse = rect(10, 10, 20, 20);
        let mut tracker = settled(pulse);
        let moved = rect(50, 50, 60, 60);
        assert_eq!(
            present(&mut tracker, &frame(2, Some(&[moved]), &[])),
            Repaint::Partial(pulse.union(moved))
        );
        // Without a flip the history must not age.
        assert_eq!(
            present(&mut tracker, &frame(2, None, &[])),
            Repaint::Nothing
        );
        assert_eq!(
            present(&mut tracker, &frame(3, Some(&[]), &[pulse])),
            Repaint::Partial(pulse.union(moved))
        );
    }

    #[test]
    fn the_next_generation_repaints_scene_damage_and_pulses() {
        let pulse = rect(10, 10, 20, 20);
        let mut tracker = settled(pulse);
        let scene = [rect(50, 50, 60, 60), rect(70, 50, 80, 55)];
        assert_eq!(
            present(&mut tracker, &frame(2, Some(&scene), &[pulse])),
            Repaint::Partial(rect(10, 10, 80, 60))
        );
    }

    #[test]
    fn unknown_or_skipped_generations_repaint_everything() {
        let pulse = rect(10, 10, 20, 20);
        let mut tracker = settled(pulse);
        assert_eq!(
            present(&mut tracker, &frame(2, None, &[pulse])),
            Repaint::Full
        );

        let mut tracker = settled(pulse);
        assert_eq!(
            present(&mut tracker, &frame(3, Some(&[]), &[pulse])),
            Repaint::Full
        );

        let mut tracker = settled(pulse);
        assert_eq!(
            present(&mut tracker, &frame(0, Some(&[]), &[pulse])),
            Repaint::Full
        );
    }

    #[test]
    fn each_present_repaints_what_its_back_buffer_missed() {
        let mut tracker = settled(rect(0, 0, 1, 1));
        let a = rect(10, 10, 20, 20);
        let b = rect(30, 30, 40, 40);
        let c = rect(50, 50, 60, 60);
        let d = rect(70, 70, 80, 80);
        // Buffer ages: each back buffer last showed the frame three presents ago.
        assert_eq!(
            present(&mut tracker, &frame(2, Some(&[a]), &[])),
            Repaint::Partial(rect(0, 0, 20, 20))
        );
        assert_eq!(
            present(&mut tracker, &frame(3, Some(&[b]), &[])),
            Repaint::Partial(rect(0, 0, 40, 40))
        );
        assert_eq!(
            present(&mut tracker, &frame(4, Some(&[c]), &[])),
            Repaint::Partial(rect(10, 10, 60, 60))
        );
        assert_eq!(
            present(&mut tracker, &frame(5, Some(&[d]), &[])),
            Repaint::Partial(rect(30, 30, 80, 80))
        );
    }

    #[test]
    fn a_full_present_repaints_every_buffer_once() {
        let pulse = rect(10, 10, 20, 20);
        let mut tracker = settled(pulse);
        assert_eq!(
            present(&mut tracker, &frame(2, None, &[pulse])),
            Repaint::Full
        );
        assert_eq!(
            present(&mut tracker, &frame(2, None, &[pulse])),
            Repaint::Full
        );
        assert_eq!(
            present(&mut tracker, &frame(2, None, &[pulse])),
            Repaint::Full
        );
        assert_eq!(
            present(&mut tracker, &frame(2, None, &[pulse])),
            Repaint::Partial(pulse)
        );
    }

    #[test]
    fn invalidating_repaints_everything_until_every_buffer_was_redrawn() {
        let pulse = rect(10, 10, 20, 20);
        let mut tracker = settled(pulse);
        tracker.invalidate();
        for _ in 0..BUFFERS {
            assert_eq!(
                present(&mut tracker, &frame(1, None, &[pulse])),
                Repaint::Full
            );
        }
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            Repaint::Partial(pulse)
        );
    }

    #[test]
    fn damage_over_most_of_the_target_repaints_everything() {
        let mut tracker = settled(rect(0, 0, 1, 1));
        assert_eq!(
            present(&mut tracker, &frame(2, Some(&[rect(0, 0, 100, 50)]), &[])),
            Repaint::Partial(rect(0, 0, 100, 50))
        );
        assert_eq!(
            present(&mut tracker, &frame(3, Some(&[rect(0, 0, 100, 51)]), &[])),
            Repaint::Full
        );
    }

    #[test]
    fn damage_is_clipped_to_the_target() {
        let mut tracker = settled(rect(0, 0, 1, 1));
        assert_eq!(
            present(&mut tracker, &frame(2, Some(&[rect(90, -5, 120, 10)]), &[])),
            Repaint::Partial(rect(0, 0, 100, 10))
        );
        // Damage wholly off the target changes nothing visible.
        let mut tracker = settled(rect(0, 0, 1, 1));
        assert_eq!(
            present(&mut tracker, &frame(2, Some(&[rect(100, 0, 120, 10)]), &[])),
            Repaint::Nothing
        );
    }

    #[test]
    fn damage_touching_a_filter_grows_to_its_footprint_and_onward() {
        let pulse = rect(10, 10, 12, 12);
        let tracker = settled(pulse);
        let filters = [
            rect(11, 11, 30, 30),
            rect(25, 25, 40, 40),
            rect(60, 60, 70, 70),
        ];
        let plan = tracker.plan(&FrameDamage {
            filters: &filters,
            ..frame(1, None, &[pulse])
        });
        // The first footprint overlaps the pulse, the second the first; the third stays clear.
        assert_eq!(plan.repaint, Repaint::Partial(rect(10, 10, 40, 40)));

        let plan = tracker.plan(&FrameDamage {
            filters: &[rect(12, 10, 30, 30)],
            ..frame(1, None, &[pulse])
        });
        // Touching edges share no pixel.
        assert_eq!(plan.repaint, Repaint::Partial(pulse));
    }

    #[test]
    fn older_damage_brought_in_by_the_back_buffer_also_spreads_through_filters() {
        // The back buffer missed a change under the filter two presents ago.
        let tracker = settled(rect(45, 45, 46, 46));
        let plan = tracker.plan(&FrameDamage {
            filters: &[rect(40, 40, 60, 60)],
            ..frame(2, Some(&[rect(90, 90, 91, 91)]), &[])
        });
        assert_eq!(plan.repaint, Repaint::Partial(rect(40, 40, 91, 91)));
    }

    #[test]
    fn a_debug_tint_is_painted_over_by_the_next_presents() {
        let pulse = rect(10, 10, 20, 20);
        let mut tracker = settled(pulse);
        let tint = rect(0, 0, 30, 30);
        let plan = tracker.plan(&frame(1, None, &[pulse]));
        tracker.presented(&plan);
        tracker.overlaid(tint);
        // The next present removes the tint even though only the pulse changed.
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            Repaint::Partial(tint)
        );
        // The buffer that showed the tint still needs it painted over when it comes back.
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            Repaint::Partial(tint)
        );
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            Repaint::Partial(tint)
        );
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            Repaint::Partial(pulse)
        );
    }
}
