//! Which pixels a present has to repaint.
//!
//! The swap chain flips between `buffer_count` back buffers, so the buffer a frame renders into
//! still holds the frame presented `buffer_count` presents earlier. A partial frame therefore
//! repaints its own damage plus everything the previous `buffer_count - 1` presents changed.
//! Damage is a short list of disjoint rectangles, so a pulse in one corner and a shimmer in the
//! other repaint two small regions rather than the box spanning both. Blur filters read pixels
//! around their bounds, so damage touching one grows to its whole footprint. Free of Direct3D
//! types, so these rules are testable on any host.

use std::collections::VecDeque;

/// The most rectangles a repaint is split into. Each costs the renderer a pass over the scene's
/// batches, so beyond this the cheapest pair merges.
pub(crate) const MAX_DAMAGE_RECTS: usize = 8;

/// Rectangles this close merge: the strip between them is cheaper to repaint than another pass.
const MERGE_GAP: i32 = 4;

/// A rectangle of whole device pixels, half-open: `left..right` by `top..bottom`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DamageRect {
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
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

    /// The pixels merging with `other` would repaint that neither covers.
    fn merge_cost(&self, other: &Self) -> i64 {
        self.union(*other).area() - self.area() - other.area()
    }

    /// Whether repainting the box around both is about as cheap as repainting them apart:
    /// they overlap, nearly touch, or their union is at most 30% larger than the two.
    fn merges_with(&self, other: &Self) -> bool {
        let gap_x = (other.left - self.right).max(self.left - other.right);
        let gap_y = (other.top - self.bottom).max(self.top - other.bottom);
        gap_x.max(gap_y) <= MERGE_GAP
            || self.union(*other).area() * 10 <= (self.area() + other.area()) * 13
    }
}

/// What one present repaints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Repaint {
    /// The screen already shows this frame: no render, no present.
    Nothing,
    /// Only these disjoint rectangles, at most [`MAX_DAMAGE_RECTS`], ordered top to bottom;
    /// every pixel outside them is retained from the back buffer.
    Partial(Vec<DamageRect>),
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FramePlan {
    generation: u64,
    /// The pixels that differ from the previous present; `None` means all of them.
    changed: Option<Vec<DamageRect>>,
    pub repaint: Repaint,
}

/// Remembers what recent presents changed, to plan the next one.
pub(crate) struct DamageTracker {
    buffer_count: usize,
    last_generation: Option<u64>,
    /// What each of the last `buffer_count - 1` presents changed, oldest first; `None` means the
    /// whole target.
    history: VecDeque<Option<Vec<DamageRect>>>,
    /// The debug tint over the last present, which the next present paints over.
    overlay: Vec<DamageRect>,
}

impl DamageTracker {
    pub fn new(buffer_count: usize) -> Self {
        Self {
            buffer_count,
            last_generation: None,
            history: VecDeque::with_capacity(buffer_count),
            overlay: Vec::new(),
        }
    }

    /// Forgets what the back buffers hold, so the next `buffer_count` presents repaint
    /// everything. For anything that leaves them unknown: a resize, a lost device, recreated
    /// targets, a failed or unusual present.
    pub fn invalidate(&mut self) {
        self.last_generation = None;
        self.history.clear();
        self.overlay.clear();
    }

    pub fn plan(&self, frame: &FrameDamage) -> FramePlan {
        let target = DamageRect {
            left: 0,
            top: 0,
            right: frame.width,
            bottom: frame.height,
        };
        let scene = match self.last_generation {
            Some(last) if last == frame.generation => Some(&[][..]),
            Some(last) if last.wrapping_add(1) == frame.generation => frame.scene,
            _ => None,
        };
        let changed = scene.map(|scene| {
            settle(
                scene
                    .iter()
                    .chain(frame.pulses)
                    .chain(&self.overlay)
                    .copied(),
                frame.filters,
                target,
            )
        });

        let repaint = match &changed {
            None => Repaint::Full,
            Some(changed) if changed.is_empty() => Repaint::Nothing,
            // The back buffer has not seen enough presents for its age to be known.
            Some(_) if self.history.len() + 1 < self.buffer_count => Repaint::Full,
            Some(changed) => self
                .history
                .iter()
                .try_fold(changed.clone(), |mut stale, earlier| {
                    stale.extend(earlier.as_ref()?);
                    Some(stale)
                })
                .map(|stale| settle(stale, frame.filters, target))
                // Past half the target, scissoring saves little and one full pass is simpler.
                .filter(|stale| {
                    stale.iter().map(DamageRect::area).sum::<i64>() * 2 <= target.area()
                })
                .map_or(Repaint::Full, Repaint::Partial),
        };
        FramePlan {
            generation: frame.generation,
            changed,
            repaint,
        }
    }

    /// Records that `plan` reached the screen.
    pub fn presented(&mut self, plan: FramePlan) {
        self.last_generation = Some(plan.generation);
        if plan.repaint == Repaint::Nothing {
            return;
        }
        self.overlay.clear();
        self.history.push_back(plan.changed);
        while self.history.len() >= self.buffer_count.max(1) {
            self.history.pop_front();
        }
    }

    /// Records a debug tint drawn over `rects` of the last present. Those pixels differ from the
    /// scene, so the next present paints them over and later ones treat them as changed.
    pub fn overlaid(&mut self, rects: &[DamageRect]) {
        if let Some(Some(last)) = self.history.back_mut() {
            last.extend(rects);
        }
        self.overlay = rects.to_vec();
    }
}

/// Reduces `damage` to at most [`MAX_DAMAGE_RECTS`] disjoint rectangles within `target`, ordered
/// top to bottom. Every filter footprint a rectangle touches lies wholly inside it, and
/// rectangles that are cheaper to repaint together merge.
fn settle(
    damage: impl IntoIterator<Item = DamageRect>,
    filters: &[DamageRect],
    target: DamageRect,
) -> Vec<DamageRect> {
    let mut pending: Vec<DamageRect> = damage.into_iter().collect();
    let mut settled: Vec<DamageRect> = Vec::with_capacity(MAX_DAMAGE_RECTS + 1);
    // Each step drops a rectangle, settles one, or merges two into one that goes back through
    // spreading, so the loop ends.
    while let Some(rect) = pending.pop() {
        let rect = spread(rect, filters).intersect(target);
        if rect.is_empty() {
            continue;
        }
        if let Some(ix) = settled.iter().position(|other| other.merges_with(&rect)) {
            pending.push(settled.swap_remove(ix).union(rect));
            continue;
        }
        settled.push(rect);
        if settled.len() > MAX_DAMAGE_RECTS {
            let (a, b) = cheapest_pair(&settled);
            let second = settled.swap_remove(b.max(a));
            let first = settled.swap_remove(b.min(a));
            pending.push(first.union(second));
        }
    }
    settled.sort_unstable();
    settled
}

/// The indices of the two rectangles whose merge repaints the fewest extra pixels.
fn cheapest_pair(rects: &[DamageRect]) -> (usize, usize) {
    let mut cheapest = (0, 1);
    let mut cost = i64::MAX;
    for a in 0..rects.len() {
        for b in a + 1..rects.len() {
            let merge_cost = rects[a].merge_cost(&rects[b]);
            if merge_cost < cost {
                cheapest = (a, b);
                cost = merge_cost;
            }
        }
    }
    cheapest
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

    fn partial(rects: &[DamageRect]) -> Repaint {
        let mut rects = rects.to_vec();
        rects.sort_unstable();
        Repaint::Partial(rects)
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
            tracker.presented(plan);
        }
        tracker
    }

    fn present(tracker: &mut DamageTracker, frame: &FrameDamage) -> Repaint {
        let plan = tracker.plan(frame);
        let repaint = plan.repaint.clone();
        tracker.presented(plan);
        repaint
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
            partial(&[pulse])
        );
    }

    #[test]
    fn a_pulse_only_present_repaints_the_pulses() {
        let pulses = [rect(10, 10, 20, 20), rect(60, 10, 70, 20)];
        let mut tracker = settled(pulses[0]);
        // The scene damage is ignored: it describes the step into this generation, which
        // earlier presents already showed.
        let scene = [rect(0, 0, 90, 90)];
        assert_eq!(
            present(&mut tracker, &frame(1, Some(&scene), &pulses)),
            partial(&pulses)
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
            partial(&[pulse, moved])
        );
        // Without a flip the history must not age.
        assert_eq!(
            present(&mut tracker, &frame(2, None, &[])),
            Repaint::Nothing
        );
        assert_eq!(
            present(&mut tracker, &frame(3, Some(&[]), &[pulse])),
            partial(&[pulse, moved])
        );
    }

    #[test]
    fn the_next_generation_repaints_scene_damage_and_pulses() {
        let pulse = rect(10, 10, 20, 20);
        let mut tracker = settled(pulse);
        let scene = [rect(50, 50, 60, 60), rect(62, 50, 72, 55)];
        // The two scene rectangles nearly touch and merge; the pulse stays apart.
        assert_eq!(
            present(&mut tracker, &frame(2, Some(&scene), &[pulse])),
            partial(&[pulse, rect(50, 50, 72, 60)])
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
    fn far_apart_damage_stays_apart() {
        // A sidebar dot and a status line across the window: their bounding box is most of it.
        let dot = rect(4, 10, 10, 16);
        let status = rect(40, 90, 98, 96);
        let mut tracker = settled(dot);
        assert_eq!(
            present(&mut tracker, &frame(2, Some(&[status]), &[dot])),
            partial(&[dot, status])
        );
    }

    #[test]
    fn overlapping_or_adjoining_damage_merges() {
        let tracker = settled(rect(0, 0, 1, 1));
        let plan = tracker.plan(&frame(
            2,
            Some(&[
                rect(10, 10, 30, 30),
                rect(20, 20, 40, 40),
                rect(43, 20, 50, 40),
                rect(70, 70, 80, 80),
            ]),
            &[],
        ));
        // The first two overlap and the third lies within the merge gap; the union of the
        // settled pulse and the first is far larger than both, so it stays apart.
        assert_eq!(
            plan.repaint,
            partial(&[rect(0, 0, 1, 1), rect(10, 10, 50, 40), rect(70, 70, 80, 80)])
        );
    }

    #[test]
    fn damage_whose_union_is_barely_larger_merges() {
        let tracker = settled(rect(0, 0, 1, 1));
        // Ten pixels apart, but stacked: the union adds only the strip between them.
        let plan = tracker.plan(&frame(
            2,
            Some(&[rect(20, 20, 80, 40), rect(20, 50, 80, 70)]),
            &[],
        ));
        assert_eq!(
            plan.repaint,
            partial(&[rect(0, 0, 1, 1), rect(20, 20, 80, 70)])
        );
    }

    #[test]
    fn beyond_the_cap_the_cheapest_pairs_merge() {
        let tracker = settled(rect(0, 0, 1, 1));
        // Ten dots in a row, far apart, plus the settled pulse.
        let dots: Vec<_> = (0..10)
            .map(|ix| rect(ix * 10, 50, ix * 10 + 2, 52))
            .collect();
        let plan = tracker.plan(&frame(2, Some(&dots), &[]));
        let Repaint::Partial(rects) = plan.repaint else {
            panic!("expected a partial repaint");
        };
        assert_eq!(rects.len(), MAX_DAMAGE_RECTS);
        for dot in &dots {
            assert!(
                rects.iter().any(|rect| rect.intersect(*dot) == *dot),
                "{dot:?} lost"
            );
        }
        for (ix, a) in rects.iter().enumerate() {
            for b in &rects[ix + 1..] {
                assert!(!a.intersects(b), "{a:?} overlaps {b:?}");
            }
        }
    }

    #[test]
    fn each_present_repaints_what_its_back_buffer_missed() {
        let start = rect(0, 0, 1, 1);
        let mut tracker = settled(start);
        let a = rect(10, 10, 20, 20);
        let b = rect(30, 30, 40, 40);
        let c = rect(50, 50, 60, 60);
        let d = rect(70, 70, 80, 80);
        // Each back buffer last showed the frame three presents ago.
        assert_eq!(
            present(&mut tracker, &frame(2, Some(&[a]), &[])),
            partial(&[start, a])
        );
        assert_eq!(
            present(&mut tracker, &frame(3, Some(&[b]), &[])),
            partial(&[start, a, b])
        );
        assert_eq!(
            present(&mut tracker, &frame(4, Some(&[c]), &[])),
            partial(&[a, b, c])
        );
        assert_eq!(
            present(&mut tracker, &frame(5, Some(&[d]), &[])),
            partial(&[b, c, d])
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
            partial(&[pulse])
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
            partial(&[pulse])
        );
    }

    #[test]
    fn damage_over_most_of_the_target_repaints_everything() {
        let mut tracker = settled(rect(0, 0, 1, 1));
        assert_eq!(
            present(&mut tracker, &frame(2, Some(&[rect(0, 0, 100, 49)]), &[])),
            partial(&[rect(0, 0, 100, 49)])
        );
        // Separate rectangles count by their total area.
        assert_eq!(
            present(
                &mut tracker,
                &frame(3, Some(&[rect(0, 0, 100, 30), rect(0, 70, 100, 92)]), &[])
            ),
            Repaint::Full
        );
    }

    #[test]
    fn damage_is_clipped_to_the_target() {
        let mut tracker = settled(rect(0, 0, 1, 1));
        assert_eq!(
            present(&mut tracker, &frame(2, Some(&[rect(90, -5, 120, 10)]), &[])),
            partial(&[rect(0, 0, 1, 1), rect(90, 0, 100, 10)])
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
        assert_eq!(plan.repaint, partial(&[rect(10, 10, 40, 40)]));

        let plan = tracker.plan(&FrameDamage {
            filters: &[rect(30, 10, 40, 30)],
            ..frame(1, None, &[pulse])
        });
        assert_eq!(plan.repaint, partial(&[pulse]));
    }

    #[test]
    fn rectangles_grown_into_each_other_by_a_filter_merge() {
        let tracker = settled(rect(0, 0, 1, 1));
        let filter = rect(30, 30, 70, 70);
        let plan = tracker.plan(&FrameDamage {
            filters: &[filter],
            ..frame(2, Some(&[rect(25, 25, 32, 32), rect(68, 68, 75, 75)]), &[])
        });
        assert_eq!(
            plan.repaint,
            partial(&[rect(0, 0, 1, 1), rect(25, 25, 75, 75)])
        );
    }

    #[test]
    fn older_damage_brought_in_by_the_back_buffer_also_spreads_through_filters() {
        // The back buffer missed a change under the filter two presents ago.
        let tracker = settled(rect(45, 45, 46, 46));
        let plan = tracker.plan(&FrameDamage {
            filters: &[rect(40, 40, 60, 60)],
            ..frame(2, Some(&[rect(90, 90, 91, 91)]), &[])
        });
        assert_eq!(
            plan.repaint,
            partial(&[rect(40, 40, 60, 60), rect(90, 90, 91, 91)])
        );
    }

    #[test]
    fn a_debug_tint_is_painted_over_by_the_next_presents() {
        let pulse = rect(10, 10, 20, 20);
        let mut tracker = settled(pulse);
        let tint = [rect(70, 70, 80, 80)];
        let plan = tracker.plan(&frame(1, None, &[pulse]));
        tracker.presented(plan);
        tracker.overlaid(&tint);
        // The next present removes the tint even though only the pulse changed.
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            partial(&[pulse, tint[0]])
        );
        // The buffers that showed or missed the tint still need it painted over.
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            partial(&[pulse, tint[0]])
        );
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            partial(&[pulse, tint[0]])
        );
        assert_eq!(
            present(&mut tracker, &frame(1, None, &[pulse])),
            partial(&[pulse])
        );
    }
}
