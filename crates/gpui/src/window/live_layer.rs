//! Live layers: cached views redrawn in place in the rendered frame.
//!
//! A contained frame (see [`super::contained`]) renders only the notified
//! cached views, but still rebuilds the frame around them. A cached view that
//! keeps re-rendering — an animation — is promoted to a *live layer*, the way
//! a browser promotes an animating element to its own compositing layer. A
//! live layer is drawn so that it can later be redrawn in place:
//!
//! - its paint is a scene region with draw orders of its own (see
//!   [`Scene::replace_repaint_region`](crate::Scene)), so its primitives form
//!   one run in each sorted vector;
//! - every per-frame stream it writes to — hitboxes, tooltips, dispatch
//!   nodes, accessed element states, mouse listeners, input handlers, cursor
//!   styles, tab stops, and the scene's vectors — holds slack after its
//!   contents: empty slots a later redraw can fill.
//!
//! When every notified view's cached view is a live layer, the window renders
//! each one alone into the empty scratch frame and transplants the result
//! into the layer's own slots of the rendered frame. Nothing else is copied,
//! shifted or diffed: stored ranges stay valid because nothing moves, damage
//! is the layer's old and new bounds, and the scene reports which vector
//! ranges changed. A layer whose new contents don't fit its slots, or that
//! touches what in-place editing can't (deferred draws, focus, text input),
//! falls back to the contained path, which redraws it with fresh slack.
//!
//! Views stay promoted for a while after they last re-rendered, then return
//! to ordinary cached views on a later full draw, so their draw orders can
//! batch with their neighbours' again.

use std::{any::TypeId, mem, ops::Range, sync::Arc};

use collections::FxHashSet;

use crate::{
    AnyElement, AnyView, App, Bounds, CursorStyle, DispatchNodeId, EntityId, EntityReads,
    GlobalElementId, Hitbox, HitboxBehavior, HitboxId, IntoElement as _, PaintIndex, Pixels,
    PrepaintStateIndex, View as _, ViewElementState, Window,
    scene::{Damage, RepaintId},
};

use super::{CursorStyleRequest, DrawPhase, contained::ContainedRedraw, contained::view_state};

/// Generations a live layer stays promoted after it last re-rendered.
const LIVE_LAYER_GENERATIONS: u64 = 600;

/// Line layouts in-place frames may leave in the text system's current frame
/// before a full draw clears them out.
const MAX_IN_PLACE_LINE_LAYOUTS: usize = 4096;

/// Empty slots a live layer keeps after `used` slots of a stream.
fn slack(used: usize) -> usize {
    used / 2 + 4
}

/// The key of empty accessed-element-state slots, which names no state.
struct NoElementState;

/// The outcome of [`Window::redraw_in_place`].
pub(crate) enum InPlace {
    /// Every hole was redrawn in place, changing `damage`.
    Done {
        damage: Damage,
        redrawn: Vec<EntityId>,
        dirty: FxHashSet<EntityId>,
    },
    /// Some hole couldn't be; the rest of the notified views must render on
    /// another path. Holes already redrawn in place are in the frame.
    Refused(FxHashSet<EntityId>),
}

impl Window {
    pub(crate) fn is_live_layer(&self, entity: EntityId) -> bool {
        self.live_layers.contains_key(&entity)
    }

    /// Promotes the cached view of `entity` to a live layer from its next
    /// draw on, or keeps it promoted.
    pub(crate) fn promote_live_layer(&mut self, entity: EntityId) {
        let generation = self.rendered_frame.scene.generation;
        self.live_layers.insert(entity, generation);
    }

    /// Demotes live layers that haven't re-rendered for a while. Called on
    /// full draws, which draw them as ordinary cached views again.
    pub(crate) fn demote_idle_live_layers(&mut self) {
        let generation = self.rendered_frame.scene.generation;
        self.live_layers
            .retain(|_, promoted| generation - *promoted < LIVE_LAYER_GENERATIONS);
    }

    pub(crate) fn allocate_repaint_id(&mut self) -> RepaintId {
        self.next_repaint_id += 1;
        RepaintId(self.next_repaint_id)
    }

    /// Renders a cached view's contents and prepaints them at `bounds`, as a
    /// cache miss does, leaving slack after them in a live layer.
    pub(crate) fn prepaint_cached_contents(
        &mut self,
        view: &AnyView,
        bounds: Bounds<Pixels>,
        live: bool,
        cx: &mut App,
    ) -> (AnyElement, EntityReads) {
        let refreshing = mem::replace(&mut self.refreshing, true);
        let start = self.prepaint_index();
        let result = cx.detect_accessed_entities(|cx| {
            let mut element = view.clone().render(self, cx).into_any_element();
            element.layout_as_root(bounds.size.into(), self, cx);
            element.prepaint_at(bounds.origin, self, cx);
            element
        });
        if live {
            self.reserve_prepaint_slack(&start);
        }
        self.refreshing = refreshing;
        result
    }

    /// Paints a cached view's freshly rendered contents. A live layer paints
    /// into a region of its own and leaves slack after its paint.
    pub(crate) fn paint_cached_contents(
        &mut self,
        element: &mut AnyElement,
        state: &mut ViewElementState,
        live: bool,
        cx: &mut App,
    ) {
        // A recording keeps paint output, not regions.
        let layer = (live && self.next_frame.scene.recording.is_none()).then(|| {
            *state
                .layer
                .get_or_insert_with(|| self.allocate_repaint_id())
        });
        let refreshing = mem::replace(&mut self.refreshing, true);
        let start = self.paint_index();
        state.context.tab_path = self.next_frame.tab_stops.current_path().into();
        if let Some(layer) = layer {
            self.next_frame.scene.begin_repaint_region(layer);
        }
        let ((), reads) = cx.detect_accessed_entities(|cx| element.paint(self, cx));
        if layer.is_some() {
            self.next_frame.scene.end_repaint_region();
            self.reserve_paint_slack(&start);
        }
        self.refreshing = refreshing;
        state.direct_reads.extend(reads.direct);
        state.accessed_entities.extend(reads.total);
    }

    /// Leaves empty slots after what was prepainted since `start`.
    fn reserve_prepaint_slack(&mut self, start: &PrepaintStateIndex) {
        let frame = &mut self.next_frame;
        let used = frame.hitboxes.len() - start.hitboxes_index;
        frame
            .hitboxes
            .extend((0..slack(used)).map(|_| Hitbox::tombstone()));
        let used = frame.tooltip_requests.len() - start.tooltips_index;
        frame
            .tooltip_requests
            .extend((0..slack(used)).map(|_| None));
        let used = frame.dispatch_tree.len() - start.dispatch_tree_index;
        frame.dispatch_tree.push_tombstones(slack(used));
        let used = frame.accessed_element_states.len() - start.accessed_element_states_index;
        frame
            .accessed_element_states
            .extend(std::iter::repeat_n(no_element_state(), slack(used)));
    }

    /// Leaves empty slots after what was painted since `start`. The scene
    /// region keeps its own slack.
    fn reserve_paint_slack(&mut self, start: &PaintIndex) {
        let frame = &mut self.next_frame;
        let used = frame.mouse_listeners.len() - start.mouse_listeners_index;
        frame.mouse_listeners.extend((0..slack(used)).map(|_| None));
        let used = frame.input_handlers.len() - start.input_handlers_index;
        frame.input_handlers.extend((0..slack(used)).map(|_| None));
        let used = frame.cursor_styles.len() - start.cursor_styles_index;
        frame
            .cursor_styles
            .extend((0..slack(used)).map(|_| CursorStyleRequest::tombstone()));
        let used = frame.accessed_element_states.len() - start.accessed_element_states_index;
        frame
            .accessed_element_states
            .extend(std::iter::repeat_n(no_element_state(), slack(used)));
        let used = frame.tab_stops.paint_index() - start.tab_handle_index;
        frame.tab_stops.skip(slack(used));
    }

    /// Redraws each planned hole in place when it is a live layer whose new
    /// contents fit its slots. Runs with the scratch frame empty, before the
    /// draw takes the platform's input handler back.
    pub(crate) fn redraw_in_place(&mut self, plan: ContainedRedraw, cx: &mut App) -> InPlace {
        let mut damage = Damage::default();
        let lines = self.text_system.layout_index().total();
        let dead_regions = self.rendered_frame.scene.repaint_regions.len()
            - self.rendered_frame.scene.region_indices.len();
        // In-place frames leave line layouts and dead regions behind; a
        // rebuild clears them out.
        if lines > MAX_IN_PLACE_LINE_LAYOUTS
            || dead_regions > self.rendered_frame.scene.region_indices.len() + 1024
        {
            return InPlace::Refused(plan.dirty);
        }
        let mut hovered = false;
        for hole in &plan.holes {
            match self.redraw_hole_in_place(hole.entity, &mut damage, cx) {
                Some(hover) => hovered |= hover,
                None => {
                    self.in_place_damage = Some(damage);
                    return InPlace::Refused(plan.dirty);
                }
            }
        }
        if hovered {
            let hit_test = self.rendered_frame.hit_test(self.mouse_position);
            if hit_test != self.mouse_hit_test {
                self.mouse_hit_test = hit_test;
                self.reset_cursor_style(cx);
            }
        }
        InPlace::Done {
            damage,
            redrawn: plan.holes.iter().map(|hole| hole.entity).collect(),
            dirty: plan.dirty,
        }
    }

    /// Redraws the live layer of `entity` in place. Returns whether its old
    /// or new hitboxes are under the mouse, or `None` if it can't be redrawn
    /// in place, having changed nothing.
    fn redraw_hole_in_place(
        &mut self,
        entity: EntityId,
        damage: &mut Damage,
        cx: &mut App,
    ) -> Option<bool> {
        let frame = &self.rendered_frame;
        let entry = frame.cached_views.get(&entity)?;
        let (global_id, node) = (entry.global_id.clone(), entry.node);
        let key = (global_id.clone(), TypeId::of::<ViewElementState>());
        let state = frame
            .element_states
            .get(&key)?
            .inner
            .downcast_ref::<Option<ViewElementState>>()?
            .as_ref()?;
        let region = *frame.scene.region_indices.get(&state.layer?)?;
        let (prepaint, paint) = (state.prepaint_range.clone(), state.paint_range.clone());
        let dispatch = prepaint.start.dispatch_tree_index..prepaint.end.dispatch_tree_index;
        if !self.is_live_layer(entity)
            || prepaint.start.deferred_draws_index != prepaint.end.deferred_draws_index
            || self
                .focus
                .and_then(|focus| frame.dispatch_tree.focusable_node_id(focus))
                .is_some_and(|focused| dispatch.contains(&focused.0))
            || frame.input_handlers
                [paint.start.input_handlers_index..paint.end.input_handlers_index]
                .iter()
                .any(Option::is_some)
            || self.taken_input_handler.is_some_and(|index| {
                (paint.start.input_handlers_index..paint.end.input_handlers_index).contains(&index)
            })
        {
            return None;
        }

        // Render the layer alone into the empty scratch frame.
        let mut state_box = self.rendered_frame.element_states.remove(&key).unwrap();
        let state = view_state(&mut state_box);
        let (view, bounds, layer) = (state.view.clone(), state.cache_key.bounds, state.layer?);
        let base = self.rendered_frame.scene.repaint_regions[region]
            .orders
            .start;
        self.invalidator.set_phase(DrawPhase::Prepaint);
        let saved = self.enter_cached_view(&global_id, state);
        let (mut element, reads) = self.with_rendered_view(entity, |window| {
            window.prepaint_cached_contents(&view, bounds, false, cx)
        });
        let prepaint_states = self.next_frame.accessed_element_states.len();
        self.invalidator.set_phase(DrawPhase::Paint);
        let tab_path = view_state(&mut state_box).context.tab_path.clone();
        self.next_frame.tab_stops.set_current_path(&tab_path);
        self.next_frame.scene.begin_repaint_region_at(layer, base);
        let ((), paint_reads) = self.with_rendered_view(entity, |window| {
            let refreshing = mem::replace(&mut window.refreshing, true);
            let result = cx.detect_accessed_entities(|cx| element.paint(window, cx));
            window.refreshing = refreshing;
            result
        });
        self.next_frame.scene.end_repaint_region();
        self.exit_cached_view(saved);
        self.invalidator.set_phase(DrawPhase::None);

        let scratch = &self.next_frame;
        let scene = &self.rendered_frame.scene;
        let fits = scratch.hitboxes.len()
            <= prepaint.end.hitboxes_index - prepaint.start.hitboxes_index
            && scratch.tooltip_requests.len()
                <= prepaint.end.tooltips_index - prepaint.start.tooltips_index
            && scratch.dispatch_tree.len() <= dispatch.len()
            && prepaint_states
                <= prepaint.end.accessed_element_states_index
                    - prepaint.start.accessed_element_states_index
            && scratch.accessed_element_states.len() - prepaint_states
                <= paint.end.accessed_element_states_index
                    - paint.start.accessed_element_states_index
            && scratch.mouse_listeners.len()
                <= paint.end.mouse_listeners_index - paint.start.mouse_listeners_index
            && scratch.input_handlers.len()
                <= paint.end.input_handlers_index - paint.start.input_handlers_index
            && scratch.cursor_styles.len()
                <= paint.end.cursor_styles_index - paint.start.cursor_styles_index
            && scratch.tab_stops.insertion_history.len()
                <= paint.end.tab_handle_index - paint.start.tab_handle_index
            && scratch.deferred_draws.is_empty()
            && scratch.focus.is_none()
            && scratch.input_handlers.iter().all(Option::is_none)
            && scratch.scene.repaint_regions[0].orders.start == base
            && scene.fits_repaint(region, &scratch.scene);
        if !fits {
            // Hand back the states the render took, and leave the frame be.
            let states = mem::take(&mut self.next_frame.element_states);
            self.rendered_frame.element_states.extend(states);
            self.rendered_frame.element_states.insert(key, state_box);
            self.next_frame.clear();
            return None;
        }

        // What the layer held before, to retire what it no longer holds.
        let frame = &mut self.rendered_frame;
        let old_states = frame.accessed_element_states[prepaint.start.accessed_element_states_index
            ..prepaint.end.accessed_element_states_index]
            .iter()
            .chain(
                &frame.accessed_element_states[paint.start.accessed_element_states_index
                    ..paint.end.accessed_element_states_index],
            )
            .cloned()
            .collect::<Vec<_>>();
        let old_hitboxes =
            &frame.hitboxes[prepaint.start.hitboxes_index..prepaint.end.hitboxes_index];
        let old_hitbox_ids = old_hitboxes
            .iter()
            .map(|hitbox| hitbox.id)
            .collect::<FxHashSet<_>>();
        let mut hovered = old_hitboxes
            .iter()
            .any(|hitbox| hitbox.hit_area.contains(&self.mouse_position));
        let mut old_regions = Vec::new();
        frame.scene.nested_region_ids(region, &mut old_regions);
        let pulses = frame.scene.region_pulses(region) || self.next_frame.scene.region_pulses(0);

        let scratch = &mut self.next_frame;
        let frame = &mut self.rendered_frame;
        hovered |= scratch
            .hitboxes
            .iter()
            .any(|hitbox| hitbox.hit_area.contains(&self.mouse_position));
        fill_slots(
            &mut frame.hitboxes,
            prepaint.start.hitboxes_index..prepaint.end.hitboxes_index,
            scratch.hitboxes.drain(..),
            Hitbox::tombstone,
        );
        fill_slots(
            &mut frame.tooltip_requests,
            prepaint.start.tooltips_index..prepaint.end.tooltips_index,
            scratch.tooltip_requests.drain(..),
            || None,
        );
        let replaced =
            frame
                .dispatch_tree
                .replace_slots(dispatch.clone(), &mut scratch.dispatch_tree, node);
        debug_assert!(replaced);
        let mut states = scratch.accessed_element_states.drain(..);
        fill_slots(
            &mut frame.accessed_element_states,
            prepaint.start.accessed_element_states_index
                ..prepaint.end.accessed_element_states_index,
            states.by_ref().take(prepaint_states),
            no_element_state,
        );
        fill_slots(
            &mut frame.accessed_element_states,
            paint.start.accessed_element_states_index..paint.end.accessed_element_states_index,
            states,
            no_element_state,
        );
        fill_slots(
            &mut frame.mouse_listeners,
            paint.start.mouse_listeners_index..paint.end.mouse_listeners_index,
            scratch.mouse_listeners.drain(..),
            || None,
        );
        fill_slots(
            &mut frame.cursor_styles,
            paint.start.cursor_styles_index..paint.end.cursor_styles_index,
            scratch.cursor_styles.drain(..),
            CursorStyleRequest::tombstone,
        );
        let replaced = frame.tab_stops.replace_slots(
            paint.start.tab_handle_index..paint.end.tab_handle_index,
            mem::take(&mut scratch.tab_stops.insertion_history),
            &tab_path,
        );
        debug_assert!(replaced);
        let replaced = frame
            .scene
            .replace_repaint_region(region, &mut scratch.scene, damage);
        debug_assert!(replaced);

        // Element states: the layer's new ones move in, and those it no
        // longer accessed go.
        let new_states = scratch
            .element_states
            .keys()
            .cloned()
            .collect::<FxHashSet<_>>();
        for old in old_states {
            if !new_states.contains(&old) {
                frame.element_states.remove(&old);
            }
        }
        frame
            .element_states
            .extend(mem::take(&mut scratch.element_states));

        // Cached views drawn inside the layer: their ranges and nodes were
        // relative to the scratch frame.
        let line_layouts = prepaint.start.line_layout_index.clone();
        for (nested, mut entry) in mem::take(&mut scratch.cached_views) {
            entry.node = DispatchNodeId(dispatch.start + entry.node.0);
            let key = (entry.global_id.clone(), TypeId::of::<ViewElementState>());
            if let Some(state) = frame
                .element_states
                .get_mut(&key)
                .and_then(|state| state.inner.downcast_mut::<Option<ViewElementState>>())
                .and_then(Option::as_mut)
            {
                let place = |index: &PrepaintStateIndex| PrepaintStateIndex {
                    hitboxes_index: prepaint.start.hitboxes_index + index.hitboxes_index,
                    tooltips_index: prepaint.start.tooltips_index + index.tooltips_index,
                    deferred_draws_index: prepaint.start.deferred_draws_index,
                    dispatch_tree_index: dispatch.start + index.dispatch_tree_index,
                    accessed_element_states_index: prepaint.start.accessed_element_states_index
                        + index.accessed_element_states_index,
                    line_layout_index: line_layouts.clone(),
                };
                let place_paint = |index: &PaintIndex| PaintIndex {
                    scene_index: paint.start.scene_index,
                    mouse_listeners_index: paint.start.mouse_listeners_index
                        + index.mouse_listeners_index,
                    input_handlers_index: paint.start.input_handlers_index
                        + index.input_handlers_index,
                    cursor_styles_index: paint.start.cursor_styles_index
                        + index.cursor_styles_index,
                    accessed_element_states_index: paint.start.accessed_element_states_index
                        + index.accessed_element_states_index
                        - prepaint_states,
                    tab_handle_index: paint.start.tab_handle_index + index.tab_handle_index,
                    line_layout_index: line_layouts.clone(),
                };
                state.prepaint_range =
                    place(&state.prepaint_range.start)..place(&state.prepaint_range.end);
                state.paint_range =
                    place_paint(&state.paint_range.start)..place_paint(&state.paint_range.end);
            }
            frame.cached_views.insert(nested, entry);
        }

        // Window controls aren't in any range.
        if !frame.window_control_hitboxes.is_empty() || !scratch.window_control_hitboxes.is_empty()
        {
            frame
                .window_control_hitboxes
                .retain(|(_, hitbox)| !old_hitbox_ids.contains(&hitbox.id));
            frame
                .window_control_hitboxes
                .append(&mut scratch.window_control_hitboxes);
        }
        #[cfg(any(test, feature = "test-support"))]
        frame.debug_bounds.extend(scratch.debug_bounds.drain());
        #[cfg(any(test, feature = "test-support"))]
        {
            self.frame_work.copied += prepaint.end.hitboxes_index - prepaint.start.hitboxes_index
                + dispatch.len()
                + paint.end.mouse_listeners_index
                - paint.start.mouse_listeners_index;
        }

        // Closures of regions the layer no longer holds go; new ones painted
        // this frame, and repaint from the next.
        for id in old_regions {
            self.retained_paints.remove(&id);
        }
        if pulses {
            self.rendered_scene_pulses = self
                .rendered_frame
                .scene
                .quads
                .iter()
                .any(|quad| quad.pulse.is_active());
        }

        let state = view_state(&mut state_box);
        state.accessed_entities = reads.total;
        state
            .accessed_entities
            .extend(paint_reads.total.iter().copied());
        state.direct_reads = reads.direct;
        state.direct_reads.extend(paint_reads.direct);
        cx.track_entities(
            self.handle,
            self.invalidator.clone(),
            &state.accessed_entities,
        );
        self.rendered_frame.element_states.insert(key, state_box);
        self.next_frame.clear();
        Some(hovered)
    }
}

/// Writes `values` into `slots`, filling the slots left over with `empty()`.
fn fill_slots<T>(
    vector: &mut [T],
    slots: Range<usize>,
    values: impl Iterator<Item = T>,
    empty: impl Fn() -> T,
) {
    let mut values = values.fuse();
    for slot in &mut vector[slots] {
        *slot = values.next().unwrap_or_else(&empty);
    }
}

fn no_element_state() -> (GlobalElementId, TypeId) {
    thread_local! {
        static NONE: GlobalElementId = GlobalElementId(Arc::from([]));
    }
    (NONE.with(Clone::clone), TypeId::of::<NoElementState>())
}

impl Hitbox {
    /// An empty slot of a live layer, which nothing hits.
    fn tombstone() -> Self {
        Hitbox {
            id: HitboxId::TOMBSTONE,
            bounds: Bounds::default(),
            content_mask: Default::default(),
            behavior: HitboxBehavior::Normal,
            hit_area: Bounds::default(),
        }
    }
}

impl CursorStyleRequest {
    /// An empty slot of a live layer, whose hitbox is never hovered.
    fn tombstone() -> Self {
        CursorStyleRequest {
            hitbox_id: Some(HitboxId::TOMBSTONE),
            style: CursorStyle::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use crate::{
        AnyWindowHandle, App, AppContext as _, Bounds, Context, Element, ElementId, Entity,
        GlobalElementId, Hsla, InspectorElementId, InteractiveElement as _, IntoElement, LayoutId,
        Modifiers, ParentElement as _, Pixels, Render, RequestFrameOptions,
        StatefulInteractiveElement as _, Style, StyleRefinement, Styled as _, TestAppContext,
        VisualTestContext, Window, WindowHandle, div, fill, hsla, px, red, size,
    };

    /// Recolors itself each frame, asking for the next one from paint, and
    /// paints `quads` quads.
    struct Tick {
        paints: Rc<Cell<usize>>,
        quads: Rc<Cell<usize>>,
        bounds: Rc<Cell<Bounds<Pixels>>>,
    }

    impl IntoElement for Tick {
        type Element = Self;

        fn into_element(self) -> Self {
            self
        }
    }

    impl Element for Tick {
        type RequestLayoutState = ();
        type PrepaintState = ();

        fn id(&self) -> Option<ElementId> {
            None
        }

        fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
            None
        }

        fn request_layout(
            &mut self,
            _: Option<&GlobalElementId>,
            _: Option<&InspectorElementId>,
            window: &mut Window,
            cx: &mut App,
        ) -> (LayoutId, ()) {
            let mut style = Style::default();
            style.size = size(px(40.).into(), px(10.).into());
            (window.request_layout(style, None, cx), ())
        }

        fn prepaint(
            &mut self,
            _: Option<&GlobalElementId>,
            _: Option<&InspectorElementId>,
            _: Bounds<Pixels>,
            _: &mut (),
            _: &mut Window,
            _: &mut App,
        ) {
        }

        fn paint(
            &mut self,
            _: Option<&GlobalElementId>,
            _: Option<&InspectorElementId>,
            bounds: Bounds<Pixels>,
            _: &mut (),
            _: &mut (),
            window: &mut Window,
            _: &mut App,
        ) {
            self.bounds.set(bounds);
            let paint = self.paints.get();
            self.paints.set(paint + 1);
            for _ in 0..self.quads.get() {
                window.paint_quad(fill(bounds, tick_color(paint)));
            }
            window.request_animation_frame();
        }
    }

    fn tick_color(paint: usize) -> Hsla {
        hsla((paint % 100) as f32 / 100.0, 0.8, 0.5, 1.0)
    }

    struct Ticker {
        paints: Rc<Cell<usize>>,
        quads: Rc<Cell<usize>>,
        bounds: Rc<Cell<Bounds<Pixels>>>,
        renders: Rc<Cell<usize>>,
    }

    impl Render for Ticker {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.renders.update(|renders| renders + 1);
            div().size_full().child(Tick {
                paints: self.paints.clone(),
                quads: self.quads.clone(),
                bounds: self.bounds.clone(),
            })
        }
    }

    /// A clickable row: a hitbox, a mouse listener and a quad.
    struct Row {
        index: usize,
        clicks: Rc<Cell<usize>>,
    }

    impl Render for Row {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let clicks = self.clicks.clone();
            div()
                .id(("row", self.index))
                .size_full()
                .bg(red())
                .on_click(move |_, _, _| clicks.update(|clicks| clicks + 1))
        }
    }

    struct Rows {
        ticker: Entity<Ticker>,
        rows: Vec<Entity<Row>>,
        renders: Rc<Cell<usize>>,
    }

    impl Render for Rows {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.renders.update(|renders| renders + 1);
            div()
                .size_full()
                .flex()
                .flex_col()
                .child(
                    self.ticker
                        .clone()
                        .cached(StyleRefinement::default().w(px(40.)).h(px(10.))),
                )
                .children(self.rows.iter().map(|row| {
                    row.clone()
                        .cached(StyleRefinement::default().w(px(100.)).h(px(4.)))
                }))
        }
    }

    struct Fixture {
        window: WindowHandle<Rows>,
        renders: Rc<Cell<usize>>,
        ticker_renders: Rc<Cell<usize>>,
        quads: Rc<Cell<usize>>,
        bounds: Rc<Cell<Bounds<Pixels>>>,
        clicks: Rc<Cell<usize>>,
    }

    fn open(cx: &mut TestAppContext, rows: usize) -> Fixture {
        let (renders, ticker_renders, quads, bounds, clicks) = (
            Rc::new(Cell::new(0)),
            Rc::new(Cell::new(0)),
            Rc::new(Cell::new(1)),
            Rc::new(Cell::new(Bounds::default())),
            Rc::new(Cell::new(0)),
        );
        let window = cx.open_window(size(px(400.), px(4000.)), {
            let (renders, ticker_renders, quads, bounds, clicks) = (
                renders.clone(),
                ticker_renders.clone(),
                quads.clone(),
                bounds.clone(),
                clicks.clone(),
            );
            move |_, cx| Rows {
                ticker: cx.new(|_| Ticker {
                    paints: Rc::new(Cell::new(0)),
                    quads,
                    bounds,
                    renders: ticker_renders,
                }),
                rows: (0..rows)
                    .map(|index| {
                        cx.new(|_| Row {
                            index,
                            clicks: clicks.clone(),
                        })
                    })
                    .collect(),
                renders,
            }
        });
        cx.run_until_parked();
        window
            .update(cx, |_, window, _| window.active.set(true))
            .unwrap();
        Fixture {
            window,
            renders,
            ticker_renders,
            quads,
            bounds,
            clicks,
        }
    }

    fn frame(cx: &mut TestAppContext, window: impl Into<AnyWindowHandle>) {
        cx.test_window(window.into())
            .simulate_frame_request(RequestFrameOptions::default());
        cx.run_until_parked();
    }

    /// The stream items one frame copied.
    fn copied(cx: &mut TestAppContext, fixture: &Fixture) -> usize {
        fixture
            .window
            .update(cx, |_, window, _| window.frame_work = Default::default())
            .unwrap();
        frame(cx, fixture.window);
        fixture
            .window
            .update(cx, |_, window, _| window.frame_work.copied)
            .unwrap()
    }

    fn ticker_colors(cx: &mut TestAppContext, fixture: &Fixture) -> Vec<Hsla> {
        let bounds = fixture.bounds.get();
        fixture
            .window
            .update(cx, |_, window, _| {
                let bounds = bounds.scale(window.scale_factor());
                window
                    .painted_quads()
                    .into_iter()
                    .filter(|quad| quad.bounds == bounds)
                    .map(|quad| quad.background.solid.into())
                    .collect()
            })
            .unwrap()
    }

    #[crate::test]
    fn in_place_frames_cost_the_same_beside_10_or_1000_rows(cx: &mut TestAppContext) {
        let mut costs = Vec::new();
        for rows in [10, 1000] {
            let fixture = open(cx, rows);
            let renders = fixture.renders.get();
            // The first contained frame reuses the frame around the ticker
            // and promotes it; the frames after redraw it in place.
            let contained = copied(cx, &fixture);
            let in_place = [copied(cx, &fixture), copied(cx, &fixture)];
            assert_eq!(
                fixture.renders.get(),
                renders,
                "the rows' parent never renders"
            );
            assert_eq!(in_place[0], in_place[1]);
            costs.push((contained, in_place[0]));
        }
        let [(contained_10, in_place_10), (contained_1000, in_place_1000)] = costs[..] else {
            unreachable!()
        };
        assert_eq!(
            in_place_10, in_place_1000,
            "an in-place frame copies the same beside 10 rows as beside 1000"
        );
        assert!(
            contained_1000 > 10 * contained_10 && contained_1000 > 20 * in_place_1000,
            "the contained frame copies everything around the hole: \
             {contained_10} at 10 rows, {contained_1000} at 1000; in place {in_place_1000}"
        );
    }

    #[crate::test]
    fn in_place_frames_damage_and_change_only_the_layer(cx: &mut TestAppContext) {
        let fixture = open(cx, 50);
        frame(cx, fixture.window);
        let mut colors = ticker_colors(cx, &fixture);
        for _ in 0..3 {
            let renders = fixture.ticker_renders.get();
            frame(cx, fixture.window);
            assert_eq!(fixture.ticker_renders.get(), renders + 1);
            let recolored = ticker_colors(cx, &fixture);
            assert_eq!(recolored.len(), 1);
            assert_ne!(recolored, colors);
            colors = recolored;
            let (damage, changed, quads) = fixture
                .window
                .update(cx, |_, window, _| {
                    let scene = &window.rendered_frame.scene;
                    (
                        scene.damage.clone(),
                        scene.changed_primitives.clone(),
                        scene.quads.len(),
                    )
                })
                .unwrap();
            let bounds = fixture.bounds.get().scale(2.0);
            let damage = damage.unwrap();
            assert_eq!(damage.len(), 1);
            assert!(bounds.is_contained_within(&damage[0]) || damage[0].intersects(&bounds));
            assert!(damage[0].size.height.0 < 30., "only the ticker: {damage:?}");
            let changed = changed.expect("an in-place frame reports what it changed");
            assert!(
                changed.quads.len() < 10 && changed.quads.end <= quads,
                "only the ticker's run of quads: {changed:?}"
            );
            assert!(changed.monochrome_sprites.is_empty());
        }
    }

    #[crate::test]
    fn a_layer_outgrowing_its_slots_redraws_on_the_contained_path(cx: &mut TestAppContext) {
        let fixture = open(cx, 20);
        frame(cx, fixture.window);
        frame(cx, fixture.window);
        let renders = fixture.renders.get();

        // Far more quads than the slack holds: the frame falls back, still
        // without rendering the parent, and the layer gets more slack.
        fixture.quads.set(40);
        frame(cx, fixture.window);
        assert_eq!(fixture.renders.get(), renders);
        assert_eq!(ticker_colors(cx, &fixture).len(), 40);
        let copied_after_growth = copied(cx, &fixture);
        assert_eq!(ticker_colors(cx, &fixture).len(), 40);
        assert!(
            copied_after_growth < 200,
            "in place again: {copied_after_growth}"
        );
    }

    #[crate::test]
    fn a_frame_edited_in_place_stays_reusable(cx: &mut TestAppContext) {
        let fixture = open(cx, 20);
        for _ in 0..4 {
            frame(cx, fixture.window);
        }
        // A full draw of the parent reuses every row and the ticker from the
        // frame the in-place frames edited.
        let renders = fixture.renders.get();
        fixture.window.update(cx, |_, _, cx| cx.notify()).unwrap();
        frame(cx, fixture.window);
        assert!(fixture.renders.get() > renders);
        assert_eq!(ticker_colors(cx, &fixture).len(), 1);
        for _ in 0..2 {
            frame(cx, fixture.window);
        }
        assert_eq!(ticker_colors(cx, &fixture).len(), 1);

        // Rows still take clicks, at their places.
        let mut visual = VisualTestContext::from_window(fixture.window.into(), cx);
        visual.simulate_click(
            crate::point(px(50.), px(10. + 4. * 7. + 2.)),
            Modifiers::none(),
        );
        assert_eq!(fixture.clicks.get(), 1);
    }
}
