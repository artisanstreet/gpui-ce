//! Contained re-renders: a notified view renders again inside its nearest
//! cached ancestor, and the rest of the frame is reused around it.
//!
//! A cached view is laid out from the style its parent gives it, never from
//! its contents, so nothing it renders can move anything outside it. When
//! every notified view sits inside a cached view (or is one), the window
//! draws a *contained frame*: it renders only those cached views again, at
//! the bounds and in the ambient state they were drawn with, and reuses the
//! rest of the previous frame around them — the same reuse a cached view gets
//! when its parent renders, applied to the whole frame with holes cut out.
//! Ancestors and siblings neither render nor lay out. Anything the reuse
//! can't reproduce soundly (a refresh, a dirty entity that isn't a drawn
//! view, a view drawn outside the main tree, accessibility, the inspector,
//! prompts, drags) takes the full path instead.

use std::{any::TypeId, mem, ops::Range};

use collections::{FxHashSet, hash_map::Entry};

use crate::{
    AnyElement, AnyImageCache, App, DispatchNodeId, EntityId, GlobalElementId, IntoElement as _,
    PaintIndex, Pixels, PrepaintStateIndex, TextColorMap, TextStyleRefinement, View as _,
    ViewElementState, Window,
};

use super::{DrawPhase, ElementStateBox, ScreenTransform};

/// The ambient state a cached view's subtree was drawn in, beyond what its
/// cache key holds, so a contained frame can draw it again in place.
pub(crate) struct CachedViewContext {
    text_style_stack: Vec<TextStyleRefinement>,
    image_cache_stack: Vec<AnyImageCache>,
    rem_size: Pixels,
    transform: ScreenTransform,
    /// Captured when the view paints.
    pub(crate) text_color_map: Option<TextColorMap>,
}

/// A cached view drawn in a frame: where to find its state, and the
/// dispatch node of its element, which sits just before its ranges.
pub(crate) struct CachedViewEntry {
    global_id: GlobalElementId,
    node: DispatchNodeId,
    /// Whether the entity was drawn as a cached view more than once, so
    /// which one to render again is unclear.
    ambiguous: bool,
}

/// The cached views a contained frame renders again, and the range of the
/// previous frame's main tree it reuses around them.
pub(crate) struct ContainedRedraw {
    holes: Vec<Hole>,
    root_prepaint: Range<PrepaintStateIndex>,
    root_paint: Range<PaintIndex>,
}

/// A cached view rendered again in a contained frame, with its ranges in the
/// previous frame and in the new one.
struct Hole {
    entity: EntityId,
    global_id: GlobalElementId,
    node: DispatchNodeId,
    old_prepaint: Range<PrepaintStateIndex>,
    old_paint: Range<PaintIndex>,
    new_node: DispatchNodeId,
    new_prepaint: Range<PrepaintStateIndex>,
    new_paint: Range<PaintIndex>,
    element: Option<AnyElement>,
}

impl Hole {
    fn state_key(&self) -> (GlobalElementId, TypeId) {
        (self.global_id.clone(), TypeId::of::<ViewElementState>())
    }
}

/// The ambient window state a contained frame swaps in to draw a hole.
struct SavedAmbience {
    text_style_stack: Vec<TextStyleRefinement>,
    image_cache_stack: Vec<AnyImageCache>,
    opacity: f32,
    pulse: crate::QuadPulse,
    transform: ScreenTransform,
    text_color_map: Option<TextColorMap>,
}

impl Window {
    /// Notes that the cached view of `entity` is drawn at `global_id` in the
    /// frame being built, under the element's current dispatch node.
    pub(crate) fn register_cached_view(&mut self, entity: EntityId, global_id: &GlobalElementId) {
        let node = self.next_frame.dispatch_tree.active_node_id().unwrap();
        match self.next_frame.cached_views.entry(entity) {
            Entry::Occupied(mut entry) => entry.get_mut().ambiguous = true,
            Entry::Vacant(entry) => {
                entry.insert(CachedViewEntry {
                    global_id: global_id.clone(),
                    node,
                    ambiguous: false,
                });
            }
        }
    }

    /// Captures the ambient state a cached view is about to render in.
    pub(crate) fn cached_view_context(&self) -> CachedViewContext {
        CachedViewContext {
            text_style_stack: self.text_style_stack.clone(),
            image_cache_stack: self.image_cache_stack.clone(),
            rem_size: self.rem_size(),
            transform: self.element_transform,
            text_color_map: None,
        }
    }

    /// Decides whether the views in `dirty` can render inside their cached
    /// ancestors, and if so which cached views to render again.
    pub(crate) fn plan_contained_redraw(
        &self,
        dirty: &FxHashSet<EntityId>,
        cx: &App,
    ) -> Option<ContainedRedraw> {
        if dirty.is_empty()
            || self.refreshing
            || self.a11y.is_active()
            || self.prompt.is_some()
            || cx.active_drag.is_some()
            || cx.mode.skip_drawing()
        {
            return None;
        }
        #[cfg(any(feature = "inspector", debug_assertions))]
        if self.inspector.is_some() {
            return None;
        }
        let frame = &self.rendered_frame;
        let (root_prepaint, root_paint) = frame.root_ranges.clone()?;
        let main_tree =
            root_prepaint.start.dispatch_tree_index..root_prepaint.end.dispatch_tree_index;
        // Outside every cached view, whatever read a notified entity renders
        // again, and that's the whole window.
        if !frame.root_reads.is_disjoint(dirty) {
            return None;
        }
        let state_of = |entity: &EntityId| {
            let entry = frame.cached_views.get(entity)?;
            let key = (entry.global_id.clone(), TypeId::of::<ViewElementState>());
            frame
                .element_states
                .get(&key)?
                .inner
                .downcast_ref::<Option<ViewElementState>>()?
                .as_ref()
        };

        let mut holes: Vec<Hole> = Vec::new();
        for view in dirty {
            // A notified entity that isn't a drawn view only matters to what
            // read it: the root didn't, and a cached view renders again only
            // when notified itself or when it encloses a notified view.
            let mut path = frame.dispatch_tree.view_path_reversed(*view).peekable();
            if path.peek().is_none() {
                continue;
            }
            // The nearest cached view at or above the notified one bounds what
            // its render can change, unless an enclosing cached view read
            // something notified: then that one renders again too.
            let mut entity = path
                .by_ref()
                .find(|view| frame.cached_views.contains_key(view))?;
            for ancestor in path {
                if let Some(state) = state_of(&ancestor)
                    && !state.direct_reads.is_disjoint(dirty)
                {
                    entity = ancestor;
                }
            }
            if holes.iter().any(|hole| hole.entity == entity) {
                continue;
            }
            let entry = &frame.cached_views[&entity];
            if entry.ambiguous {
                return None;
            }
            let state = state_of(&entity)?;
            let dispatch = state.prepaint_range.start.dispatch_tree_index
                ..state.prepaint_range.end.dispatch_tree_index;
            if !main_tree.contains(&entry.node.0) || dispatch.end > main_tree.end {
                return None;
            }
            holes.push(Hole {
                entity,
                global_id: entry.global_id.clone(),
                node: entry.node,
                old_prepaint: state.prepaint_range.clone(),
                old_paint: state.paint_range.clone(),
                new_node: entry.node,
                new_prepaint: Default::default(),
                new_paint: Default::default(),
                element: None,
            });
        }

        // Cached views nested in another being rendered render with it.
        holes.sort_by_key(|hole| hole.node.0);
        let mut outer_end = 0;
        holes.retain(|hole| {
            let nested = hole.node.0 < outer_end;
            if !nested {
                outer_end = hole.old_prepaint.end.dispatch_tree_index;
            }
            !nested
        });
        Some(ContainedRedraw {
            holes,
            root_prepaint,
            root_paint,
        })
    }

    /// Draws a contained frame: the planned cached views render again, and
    /// the previous frame's main tree is reused around them. Deferred draws
    /// and tooltips are drawn as in a full frame.
    pub(crate) fn draw_contained(&mut self, plan: ContainedRedraw, cx: &mut App) {
        let ContainedRedraw {
            mut holes,
            root_prepaint,
            root_paint,
        } = plan;
        self.invalidator.set_phase(DrawPhase::Prepaint);
        self.tooltip_bounds.take();

        let prepaint_start = self.prepaint_index();
        let mut cursor = root_prepaint.start.clone();
        for index in 0..holes.len() {
            let (done, rest) = holes.split_at_mut(index);
            let hole = &mut rest[0];
            let remap = prepaint_remap(&root_prepaint.start, &prepaint_start, done);
            self.reuse_prepaint_segment(cursor..hole.old_prepaint.start.clone(), &remap);
            hole.new_node = remap(hole.node);
            self.prepaint_hole(hole, cx);
            cursor = hole.old_prepaint.end.clone();
        }
        self.reuse_prepaint_segment(
            cursor..root_prepaint.end.clone(),
            &prepaint_remap(&root_prepaint.start, &prepaint_start, &holes),
        );
        let prepaint_end = self.prepaint_index();

        self.prepaint_deferred_draws(cx);
        let tooltip_element = self.prepaint_tooltip(cx);
        self.mouse_hit_test = self.next_frame.hit_test(self.mouse_position);

        self.invalidator.set_phase(DrawPhase::Paint);
        let paint_start = self.paint_index();
        let mut cursor = root_paint.start.clone();
        for hole in &mut holes {
            self.reuse_paint(cursor..hole.old_paint.start.clone());
            self.paint_hole(hole, cx);
            cursor = hole.old_paint.end.clone();
        }
        self.reuse_paint(cursor..root_paint.end.clone());
        let paint_end = self.paint_index();

        self.paint_deferred_draws(cx);
        if let Some(mut tooltip_element) = tooltip_element {
            tooltip_element.paint(self, cx);
        }

        self.carry_cached_views(
            &holes,
            (&root_prepaint.start, &prepaint_start),
            (&root_paint.start, &paint_start),
        );
        self.carry_window_control_hitboxes(&holes);
        self.next_frame.root_ranges = Some((prepaint_start..prepaint_end, paint_start..paint_end));
    }

    /// Like [`Self::reuse_prepaint`] for a stretch of the main tree that
    /// need not be a subtree: dispatch nodes whose parents lie before the
    /// stretch find them through `remap`.
    fn reuse_prepaint_segment(
        &mut self,
        range: Range<PrepaintStateIndex>,
        remap: &impl Fn(DispatchNodeId) -> DispatchNodeId,
    ) {
        self.next_frame.hitboxes.extend(
            self.rendered_frame.hitboxes[range.start.hitboxes_index..range.end.hitboxes_index]
                .iter()
                .cloned(),
        );
        self.next_frame.tooltip_requests.extend(
            self.rendered_frame.tooltip_requests
                [range.start.tooltips_index..range.end.tooltips_index]
                .iter_mut()
                .map(|request| request.take()),
        );
        self.next_frame.accessed_element_states.extend(
            self.rendered_frame.accessed_element_states[range.start.accessed_element_states_index
                ..range.end.accessed_element_states_index]
                .iter()
                .map(|(id, type_id)| (id.clone(), *type_id)),
        );
        self.text_system
            .reuse_layouts(range.start.line_layout_index..range.end.line_layout_index);
        if self.next_frame.dispatch_tree.reuse_segment(
            range.start.dispatch_tree_index..range.end.dispatch_tree_index,
            &mut self.rendered_frame.dispatch_tree,
            self.focus,
            remap,
        ) {
            self.next_frame.focus = self.focus;
        }
        self.next_frame.deferred_draws.extend(
            self.rendered_frame.deferred_draws
                [range.start.deferred_draws_index..range.end.deferred_draws_index]
                .iter()
                .map(|deferred_draw| super::DeferredDraw {
                    current_view: deferred_draw.current_view,
                    parent_node: remap(deferred_draw.parent_node),
                    element_id_stack: deferred_draw.element_id_stack.clone(),
                    text_style_stack: deferred_draw.text_style_stack.clone(),
                    content_mask: deferred_draw.content_mask,
                    rem_size: deferred_draw.rem_size,
                    priority: deferred_draw.priority,
                    element: None,
                    absolute_offset: deferred_draw.absolute_offset,
                    prepaint_range: deferred_draw.prepaint_range.clone(),
                    paint_range: deferred_draw.paint_range.clone(),
                }),
        );
    }

    /// Renders a hole's view again and prepaints it, as a cached view whose
    /// cache missed would, at its old bounds and in its old ambient state.
    fn prepaint_hole(&mut self, hole: &mut Hole, cx: &mut App) {
        let key = hole.state_key();
        let mut state_box = self
            .rendered_frame
            .element_states
            .remove(&key)
            .expect("planned holes have state");
        let state = view_state(&mut state_box);
        let (view, bounds) = (state.view.clone(), state.cache_key.bounds);

        self.next_frame.dispatch_tree.set_active_node(hole.new_node);
        let saved = self.enter_cached_view(&hole.global_id, state);
        let entity = hole.entity;
        let (element, reads) = self.with_rendered_view(entity, |window| {
            let refreshing = mem::replace(&mut window.refreshing, true);
            hole.new_prepaint.start = window.prepaint_index();
            let result = cx.detect_accessed_entities(|cx| {
                let mut element = view.render(window, cx).into_any_element();
                element.layout_as_root(bounds.size.into(), window, cx);
                element.prepaint_at(bounds.origin, window, cx);
                element
            });
            hole.new_prepaint.end = window.prepaint_index();
            window.refreshing = refreshing;
            result
        });
        self.exit_cached_view(saved);

        let state = view_state(&mut state_box);
        state.prepaint_range = hole.new_prepaint.clone();
        state.accessed_entities = reads.total;
        state.direct_reads = reads.direct;
        self.next_frame.element_states.insert(key, state_box);
        self.next_frame.cached_views.insert(
            entity,
            CachedViewEntry {
                global_id: hole.global_id.clone(),
                node: hole.new_node,
                ambiguous: false,
            },
        );
        hole.element = Some(element);
    }

    fn paint_hole(&mut self, hole: &mut Hole, cx: &mut App) {
        let key = hole.state_key();
        let mut element = hole.element.take().unwrap();
        let mut state_box = self.next_frame.element_states.remove(&key).unwrap();

        self.next_frame.dispatch_tree.set_active_node(hole.new_node);
        let saved = self.enter_cached_view(&hole.global_id, view_state(&mut state_box));
        let ((), reads) = self.with_rendered_view(hole.entity, |window| {
            let refreshing = mem::replace(&mut window.refreshing, true);
            hole.new_paint.start = window.paint_index();
            let result = cx.detect_accessed_entities(|cx| element.paint(window, cx));
            hole.new_paint.end = window.paint_index();
            window.refreshing = refreshing;
            result
        });
        self.exit_cached_view(saved);

        let state = view_state(&mut state_box);
        state.paint_range = hole.new_paint.clone();
        state.direct_reads.extend(reads.direct);
        state.accessed_entities.extend(reads.total);
        self.next_frame.element_states.insert(key, state_box);
    }

    /// Swaps in the ambient state a cached view was drawn in. Contained
    /// frames draw holes from the root, where every stack is empty.
    fn enter_cached_view(
        &mut self,
        global_id: &GlobalElementId,
        state: &ViewElementState,
    ) -> SavedAmbience {
        let context = &state.context;
        self.element_id_stack = global_id.0.iter().cloned().collect();
        self.content_mask_stack.push(state.cache_key.content_mask);
        self.rem_size_override_stack.push(context.rem_size);
        SavedAmbience {
            text_style_stack: mem::replace(
                &mut self.text_style_stack,
                context.text_style_stack.clone(),
            ),
            image_cache_stack: mem::replace(
                &mut self.image_cache_stack,
                context.image_cache_stack.clone(),
            ),
            opacity: mem::replace(&mut self.element_opacity, state.cache_key.inherited_opacity),
            pulse: mem::replace(&mut self.element_pulse, state.cache_key.inherited_pulse),
            transform: mem::replace(&mut self.element_transform, context.transform),
            text_color_map: mem::replace(&mut self.text_color_map, context.text_color_map.clone()),
        }
    }

    fn exit_cached_view(&mut self, saved: SavedAmbience) {
        self.element_id_stack.clear();
        self.content_mask_stack.pop();
        self.rem_size_override_stack.pop();
        self.text_style_stack = saved.text_style_stack;
        self.image_cache_stack = saved.image_cache_stack;
        self.element_opacity = saved.opacity;
        self.element_pulse = saved.pulse;
        self.element_transform = saved.transform;
        self.text_color_map = saved.text_color_map;
    }

    /// Carries the cached views reused around the holes into the new frame,
    /// moving their ranges to where the reuse put them: past each hole that
    /// precedes them, by how much that hole grew. Those inside holes were
    /// registered afresh by their render; those outside the main tree are
    /// dropped, so they take the full path if notified.
    fn carry_cached_views(
        &mut self,
        holes: &[Hole],
        prepaint_root: (&PrepaintStateIndex, &PrepaintStateIndex),
        paint_root: (&PaintIndex, &PaintIndex),
    ) {
        // Entries locate themselves in the previous frame.
        let Some((main_tree, _)) = self.rendered_frame.root_ranges.clone() else {
            return;
        };
        let main_tree = main_tree.start.dispatch_tree_index..main_tree.end.dispatch_tree_index;
        // The last hole wholly before a position, located by the dispatch
        // node it follows in the previous frame.
        let anchor = |node: usize| {
            holes
                .iter()
                .rev()
                .find(|hole| hole.old_prepaint.end.dispatch_tree_index <= node)
        };
        let shift_prepaint = |index: &PrepaintStateIndex, hole: Option<&Hole>| match hole {
            Some(hole) => index.shifted(&hole.old_prepaint.end, &hole.new_prepaint.end),
            None => index.shifted(prepaint_root.0, prepaint_root.1),
        };
        let shift_paint = |index: &PaintIndex, hole: Option<&Hole>| match hole {
            Some(hole) => index.shifted(&hole.old_paint.end, &hole.new_paint.end),
            None => index.shifted(paint_root.0, paint_root.1),
        };

        let previous = mem::take(&mut self.rendered_frame.cached_views);
        for (entity, entry) in &previous {
            let node = entry.node.0;
            if !main_tree.contains(&node)
                || holes.iter().any(|hole| {
                    (hole.old_prepaint.start.dispatch_tree_index
                        ..hole.old_prepaint.end.dispatch_tree_index)
                        .contains(&node)
                        || hole.entity == *entity
                })
            {
                continue;
            }
            let key = (entry.global_id.clone(), TypeId::of::<ViewElementState>());
            let Some(state) = self
                .rendered_frame
                .element_states
                .get_mut(&key)
                .and_then(|state| state.inner.downcast_mut::<Option<ViewElementState>>())
                .and_then(Option::as_mut)
            else {
                continue;
            };
            let start = anchor(node);
            let end = anchor(state.prepaint_range.end.dispatch_tree_index);
            state.prepaint_range = shift_prepaint(&state.prepaint_range.start, start)
                ..shift_prepaint(&state.prepaint_range.end, end);
            state.paint_range = shift_paint(&state.paint_range.start, start)
                ..shift_paint(&state.paint_range.end, end);
            let (old_anchor, new_anchor) = match start {
                Some(hole) => (&hole.old_prepaint.end, &hole.new_prepaint.end),
                None => prepaint_root,
            };
            let node = DispatchNodeId(
                node - old_anchor.dispatch_tree_index + new_anchor.dispatch_tree_index,
            );
            match self.next_frame.cached_views.entry(*entity) {
                // Also drawn inside a hole: two places now.
                Entry::Occupied(mut fresh) => fresh.get_mut().ambiguous = true,
                Entry::Vacant(vacant) => {
                    vacant.insert(CachedViewEntry {
                        global_id: entry.global_id.clone(),
                        node,
                        ambiguous: entry.ambiguous,
                    });
                }
            }
        }
        self.rendered_frame.cached_views = previous;
    }

    /// Window controls aren't part of any range; keep those of the reused
    /// tree, dropping the ones the holes registered before.
    fn carry_window_control_hitboxes(&mut self, holes: &[Hole]) {
        let redrawn = holes
            .iter()
            .flat_map(|hole| {
                self.rendered_frame.hitboxes
                    [hole.old_prepaint.start.hitboxes_index..hole.old_prepaint.end.hitboxes_index]
                    .iter()
                    .map(|hitbox| hitbox.id)
            })
            .collect::<FxHashSet<_>>();
        let kept = self
            .rendered_frame
            .window_control_hitboxes
            .iter()
            .filter(|(_, hitbox)| !redrawn.contains(&hitbox.id))
            .cloned()
            .collect::<Vec<_>>();
        self.next_frame.window_control_hitboxes.extend(kept);
    }
}

/// Maps a dispatch node outside the holes drawn so far to its id in the new
/// frame: everything between holes moved by as much as the hole before it.
fn prepaint_remap<'a>(
    old_root: &'a PrepaintStateIndex,
    new_root: &'a PrepaintStateIndex,
    holes: &'a [Hole],
) -> impl Fn(DispatchNodeId) -> DispatchNodeId + 'a {
    move |node| {
        let (old, new) = holes
            .iter()
            .rev()
            .find(|hole| hole.old_prepaint.end.dispatch_tree_index <= node.0)
            .map_or((old_root, new_root), |hole| {
                (&hole.old_prepaint.end, &hole.new_prepaint.end)
            });
        DispatchNodeId(node.0 - old.dispatch_tree_index + new.dispatch_tree_index)
    }
}

fn view_state(state: &mut ElementStateBox) -> &mut ViewElementState {
    state
        .inner
        .downcast_mut::<Option<ViewElementState>>()
        .and_then(Option::as_mut)
        .expect("a cached view's state")
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc, time::Duration};

    use crate::{
        Animation, AnimationExt as _, AnyWindowHandle, App, AppContext as _, Bounds, Context,
        Element, ElementId, Entity, FocusHandle, GlobalElementId, Hsla, InspectorElementId,
        InteractiveElement as _, IntoElement, LayoutId, Modifiers, ParentElement as _, Pixels,
        Render, RequestFrameOptions, ScaledPixels, StatefulInteractiveElement as _, Style,
        StyleRefinement, Styled as _, TestAppContext, VisualTestContext, Window, WindowHandle,
        blue, canvas, div, fill, hsla, point, px, red, size,
    };

    /// Counts what each frame renders, lays out and receives.
    #[derive(Default)]
    struct Probe {
        root_renders: Cell<usize>,
        root_layouts: Cell<usize>,
        sibling_renders: Cell<usize>,
        flicker_renders: Cell<usize>,
        flicker_paints: Cell<usize>,
        flicker_bounds: Cell<Bounds<Pixels>>,
        fade_renders: Cell<usize>,
        fade_bounds: Cell<Bounds<Pixels>>,
        button_bounds: Cell<Bounds<Pixels>>,
        inner_button_bounds: Cell<Bounds<Pixels>>,
        clicks: Cell<usize>,
        inner_clicks: Cell<usize>,
        keys: Cell<usize>,
        inner_keys: Cell<usize>,
    }

    /// A fixed-size element counting its layouts and, through `paint`,
    /// drawing whatever its owner wants.
    struct Probed {
        size: (Pixels, Pixels),
        layouts: Option<Rc<Probe>>,
        paint: Box<dyn FnMut(Bounds<Pixels>, &mut Window)>,
    }

    impl IntoElement for Probed {
        type Element = Self;

        fn into_element(self) -> Self {
            self
        }
    }

    impl Element for Probed {
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
            if let Some(probe) = &self.layouts {
                probe.root_layouts.update(|layouts| layouts + 1);
            }
            let mut style = Style::default();
            style.size = size(self.size.0.into(), self.size.1.into());
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
            (self.paint)(bounds, window);
        }
    }

    fn flicker_color(paint: usize) -> Hsla {
        hsla((paint % 100) as f32 / 100.0, 0.8, 0.5, 1.0)
    }

    /// Shimmer-like: recolors itself each frame by asking for the next one
    /// from paint. Also focusable and clickable inside the contained view.
    struct Flicker {
        probe: Rc<Probe>,
        focus: FocusHandle,
        /// Read by the root, which sizes the cached view from it.
        wide: bool,
    }

    impl Render for Flicker {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.probe.flicker_renders.update(|renders| renders + 1);
            let (probe, keys, clicks, button) = (
                self.probe.clone(),
                self.probe.clone(),
                self.probe.clone(),
                self.probe.clone(),
            );
            div()
                .id("flicker")
                .size_full()
                .flex()
                .track_focus(&self.focus)
                .on_key_down(move |_, _, _| keys.inner_keys.update(|keys| keys + 1))
                .child(Probed {
                    size: (px(40.), px(20.)),
                    layouts: None,
                    paint: Box::new(move |bounds, window| {
                        probe.flicker_bounds.set(bounds);
                        let paint = probe.flicker_paints.get();
                        probe.flicker_paints.set(paint + 1);
                        window.paint_quad(fill(bounds, flicker_color(paint)));
                        window.request_animation_frame();
                    }),
                })
                .child(
                    div()
                        .id("inner-button")
                        .size(px(20.))
                        .bg(blue())
                        .on_click(move |_, _, _| clicks.inner_clicks.update(|clicks| clicks + 1))
                        .child(
                            canvas(
                                move |bounds, _, _| button.inner_button_bounds.set(bounds),
                                |_, _, _, _| {},
                            )
                            .size_full(),
                        ),
                )
        }
    }

    /// A repeating `with_animation` fade.
    struct Fade {
        probe: Rc<Probe>,
    }

    impl Render for Fade {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.probe.fade_renders.update(|renders| renders + 1);
            let probe = self.probe.clone();
            div()
                .size_full()
                .bg(red())
                .child(
                    canvas(
                        move |bounds, _, _| probe.fade_bounds.set(bounds),
                        |_, _, _, _| {},
                    )
                    .size_full(),
                )
                .with_animation(
                    "fade",
                    Animation::new(Duration::from_secs(1)).repeat_synced(),
                    |fade, delta| fade.opacity(0.2 + 0.6 * delta),
                )
        }
    }

    /// A cached sibling with a button and a focusable element.
    struct Sibling {
        probe: Rc<Probe>,
        focus: FocusHandle,
        /// Read by the sibling alone.
        model: Entity<usize>,
    }

    impl Render for Sibling {
        fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            self.probe.sibling_renders.update(|renders| renders + 1);
            self.model.read(cx);
            let (keys, clicks, button) =
                (self.probe.clone(), self.probe.clone(), self.probe.clone());
            div()
                .size_full()
                .flex()
                .track_focus(&self.focus)
                .on_key_down(move |_, _, _| keys.keys.update(|keys| keys + 1))
                .child(
                    div()
                        .id("button")
                        .size(px(30.))
                        .bg(red())
                        .on_click(move |_, _, _| clicks.clicks.update(|clicks| clicks + 1))
                        .child(
                            canvas(
                                move |bounds, _, _| button.button_bounds.set(bounds),
                                |_, _, _, _| {},
                            )
                            .size_full(),
                        ),
                )
        }
    }

    struct Root {
        probe: Rc<Probe>,
        flicker: Option<Entity<Flicker>>,
        fade: Option<Entity<Fade>>,
        sibling: Entity<Sibling>,
        /// Whether the root sizes the flicker's cached view from its state.
        size_from_flicker: bool,
        /// Read by the root itself.
        model: Entity<usize>,
    }

    impl Render for Root {
        fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            self.probe.root_renders.update(|renders| renders + 1);
            self.model.read(cx);
            let mut root = div().size_full().flex().flex_row().child(Probed {
                size: (px(10.), px(10.)),
                layouts: Some(self.probe.clone()),
                paint: Box::new(|bounds, window| window.paint_quad(fill(bounds, blue()))),
            });
            if let Some(flicker) = &self.flicker {
                let wide = self.size_from_flicker && flicker.read(cx).wide;
                let width = if wide { 120. } else { 80. };
                root = root.child(
                    flicker
                        .clone()
                        .cached(StyleRefinement::default().w(px(width)).h(px(30.))),
                );
            }
            if let Some(fade) = &self.fade {
                root = root.child(
                    fade.clone()
                        .cached(StyleRefinement::default().w(px(30.)).h(px(30.))),
                );
            }
            root.child(
                self.sibling
                    .clone()
                    .cached(StyleRefinement::default().w(px(100.)).h(px(40.))),
            )
        }
    }

    fn open(cx: &mut TestAppContext, flicker: bool, fade: bool) -> (WindowHandle<Root>, Rc<Probe>) {
        let probe = Rc::new(Probe::default());
        let window = cx.open_window(size(px(400.), px(100.)), {
            let probe = probe.clone();
            move |_, cx| Root {
                flicker: flicker.then(|| {
                    cx.new(|cx| Flicker {
                        probe: probe.clone(),
                        focus: cx.focus_handle().tab_stop(true),
                        wide: false,
                    })
                }),
                fade: fade.then(|| {
                    cx.new(|_| Fade {
                        probe: probe.clone(),
                    })
                }),
                sibling: cx.new(|cx| Sibling {
                    probe: probe.clone(),
                    focus: cx.focus_handle().tab_stop(true),
                    model: cx.new(|_| 0),
                }),
                size_from_flicker: false,
                model: cx.new(|_| 0),
                probe,
            }
        });
        cx.run_until_parked();
        window
            .update(cx, |_, window, _| window.active.set(true))
            .unwrap();
        // Present once: damage holds against the last presented scene, and
        // the test app draws without presenting.
        frame(cx, window);
        (window, probe)
    }

    /// Delivers a frame: due animation callbacks notify their views, and the
    /// window draws.
    fn frame(cx: &mut TestAppContext, window: impl Into<AnyWindowHandle>) {
        cx.executor().advance_clock(Duration::from_millis(50));
        cx.test_window(window.into())
            .simulate_frame_request(RequestFrameOptions::default());
        cx.run_until_parked();
    }

    fn counts(probe: &Probe) -> (usize, usize, usize) {
        (
            probe.root_renders.get(),
            probe.root_layouts.get(),
            probe.sibling_renders.get(),
        )
    }

    fn quad_colors(
        cx: &mut TestAppContext,
        window: impl Into<AnyWindowHandle>,
        bounds: Bounds<Pixels>,
    ) -> Vec<Hsla> {
        cx.update_window(window.into(), |_, window, _| {
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

    fn damage(
        cx: &mut TestAppContext,
        window: impl Into<AnyWindowHandle>,
    ) -> Option<Vec<Bounds<ScaledPixels>>> {
        cx.update_window(window.into(), |_, window, _| {
            window.rendered_frame.scene.damage.clone()
        })
        .unwrap()
    }

    /// What damage reports for `bounds`: rounded out, one pixel of margin.
    fn damage_of(bounds: Bounds<Pixels>) -> Bounds<ScaledPixels> {
        let bounds = bounds.scale(2.0);
        Bounds::from_corners(
            point(
                ScaledPixels(bounds.left().0.floor() - 1.),
                ScaledPixels(bounds.top().0.floor() - 1.),
            ),
            point(
                ScaledPixels(bounds.right().0.ceil() + 1.),
                ScaledPixels(bounds.bottom().0.ceil() + 1.),
            ),
        )
    }

    #[crate::test]
    fn a_view_animating_from_paint_renders_alone_inside_its_cached_view(cx: &mut TestAppContext) {
        let (window, probe) = open(cx, true, false);
        let around = counts(&probe);
        let flicker = probe.flicker_bounds.get();
        let mut colors = quad_colors(cx, window, flicker);
        assert_eq!(colors.len(), 1);

        for frames in 1..=3 {
            let renders = probe.flicker_renders.get();
            frame(cx, window);
            assert_eq!(
                counts(&probe),
                around,
                "the root and the sibling neither render nor lay out"
            );
            assert_eq!(probe.flicker_renders.get(), renders + 1, "frame {frames}");
            let recolored = quad_colors(cx, window, flicker);
            assert_eq!(recolored.len(), 1);
            assert_ne!(recolored, colors);
            colors = recolored;
            assert_eq!(damage(cx, window), Some(vec![damage_of(flicker)]));
        }
    }

    #[crate::test]
    fn a_with_animation_fade_renders_alone_inside_its_cached_view(cx: &mut TestAppContext) {
        let (window, probe) = open(cx, false, true);
        let around = counts(&probe);
        let fade = probe.fade_bounds.get();
        let mut colors = quad_colors(cx, window, fade);
        assert_eq!(colors.len(), 1);

        for _ in 0..3 {
            let renders = probe.fade_renders.get();
            frame(cx, window);
            assert_eq!(counts(&probe), around);
            assert_eq!(probe.fade_renders.get(), renders + 1);
            let faded = quad_colors(cx, window, fade);
            assert_eq!(faded.len(), 1);
            assert_ne!(faded, colors, "the fade's opacity moves");
            colors = faded;
            assert_eq!(damage(cx, window), Some(vec![damage_of(fade)]));
        }
    }

    #[crate::test]
    fn a_notified_model_renders_only_what_the_full_path_would(cx: &mut TestAppContext) {
        let (window, probe) = open(cx, false, false);
        let (root_model, sibling_model) = window
            .update(cx, |root, _, cx| {
                (root.model.clone(), root.sibling.read(cx).model.clone())
            })
            .unwrap();

        // Read only inside a cached view that wasn't notified: a full frame
        // would reuse that view too, so nothing renders.
        let around = counts(&probe);
        sibling_model.update(cx, |_, cx| cx.notify());
        frame(cx, window);
        assert_eq!(counts(&probe), around);

        // Read by the root: the root renders again.
        root_model.update(cx, |_, cx| cx.notify());
        frame(cx, window);
        assert_eq!(probe.root_renders.get(), around.0 + 1);
        assert_eq!(
            probe.sibling_renders.get(),
            around.2,
            "the cached sibling is reused"
        );
    }

    #[crate::test]
    fn input_reaches_elements_inside_and_outside_contained_views(cx: &mut TestAppContext) {
        let (window, probe) = open(cx, true, true);
        let around = counts(&probe);
        for _ in 0..3 {
            frame(cx, window);
        }
        assert_eq!(counts(&probe), around, "only contained frames so far");

        // Clicks, inside and outside a re-rendered view.
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.simulate_click(probe.button_bounds.get().center(), Modifiers::none());
        visual.simulate_click(probe.inner_button_bounds.get().center(), Modifiers::none());
        assert_eq!((probe.clicks.get(), probe.inner_clicks.get()), (1, 1));

        // Clicking inside the flicker focused it; keys reach it there.
        let (flicker_focus, sibling_focus) = window
            .update(cx, |root, _, cx| {
                (
                    root.flicker.as_ref().unwrap().read(cx).focus.clone(),
                    root.sibling.read(cx).focus.clone(),
                )
            })
            .unwrap();
        let focused = |cx: &mut TestAppContext, handle: &FocusHandle| {
            window
                .update(cx, |_, window, _| handle.is_focused(window))
                .unwrap()
        };
        for _ in 0..2 {
            frame(cx, window);
        }
        assert!(focused(cx, &flicker_focus));
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.simulate_keystrokes("a");
        assert_eq!((probe.inner_keys.get(), probe.keys.get()), (1, 0));

        // Tab order runs on to the sibling outside, and back.
        window
            .update(cx, |_, window, cx| window.focus_next(cx))
            .unwrap();
        cx.run_until_parked();
        for _ in 0..2 {
            frame(cx, window);
        }
        assert!(focused(cx, &sibling_focus));
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.simulate_keystrokes("b");
        assert_eq!((probe.inner_keys.get(), probe.keys.get()), (1, 1));
        window
            .update(cx, |_, window, cx| window.focus_prev(cx))
            .unwrap();
        cx.run_until_parked();
        frame(cx, window);
        assert!(focused(cx, &flicker_focus));
    }

    #[crate::test]
    fn a_contained_view_that_resizes_itself_renders_its_parent_again(cx: &mut TestAppContext) {
        let (window, probe) = open(cx, true, false);
        window
            .update(cx, |root, _, cx| {
                root.size_from_flicker = true;
                cx.notify();
            })
            .unwrap();
        frame(cx, window);
        let narrow = probe.flicker_bounds.get();
        let root_renders = probe.root_renders.get();

        // The root sizes the cached view from the state the view changes, so
        // the root must render and lay out again.
        let flicker = window
            .update(cx, |root, _, _| root.flicker.clone().unwrap())
            .unwrap();
        flicker.update(cx, |flicker, cx| {
            flicker.wide = true;
            cx.notify();
        });
        frame(cx, window);
        assert!(probe.root_renders.get() > root_renders);
        let wide = probe.flicker_bounds.get();
        assert_eq!(narrow.size.width, px(40.));
        assert_eq!(wide.size.width, px(40.), "the probe itself keeps its size");
        let sibling = probe.button_bounds.get();
        assert_eq!(
            sibling.left(),
            px(10. + 120.),
            "the sibling moved past the widened view"
        );
        let damage = damage(cx, window).unwrap();
        assert!(
            damage
                .iter()
                .any(|region| sibling.scale(2.0).is_contained_within(region)
                    || region.intersects(&sibling.scale(2.0)))
        );

        // While the root reads the view's state, each of its frames renders
        // the root too.
        let root_renders = probe.root_renders.get();
        frame(cx, window);
        assert_eq!(probe.root_renders.get(), root_renders + 1);
    }

    /// Paints one to three stacked quads, a different count each frame, so
    /// everything after it in the frame moves.
    struct Inner {
        probe: Rc<NestedProbe>,
    }

    #[derive(Default)]
    struct NestedProbe {
        root_renders: Cell<usize>,
        outer_renders: Cell<usize>,
        inner_renders: Cell<usize>,
        animate: Cell<bool>,
        inner_bounds: Cell<Bounds<Pixels>>,
        after_bounds: Cell<Bounds<Pixels>>,
        after_clicks: Cell<usize>,
        after_keys: Cell<usize>,
    }

    impl Render for Inner {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let renders = self.probe.inner_renders.get() + 1;
            self.probe.inner_renders.set(renders);
            let probe = self.probe.clone();
            div().size_full().child(Probed {
                size: (px(60.), px(30.)),
                layouts: None,
                paint: Box::new(move |bounds, window| {
                    probe.inner_bounds.set(bounds);
                    for _ in 0..renders % 3 + 1 {
                        window.paint_quad(fill(bounds, flicker_color(renders)));
                    }
                    if probe.animate.get() {
                        window.request_animation_frame();
                    }
                }),
            })
        }
    }

    struct Outer {
        probe: Rc<NestedProbe>,
        inner: Entity<Inner>,
        focus: FocusHandle,
    }

    impl Render for Outer {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.probe.outer_renders.update(|renders| renders + 1);
            let (keys, clicks, bounds) =
                (self.probe.clone(), self.probe.clone(), self.probe.clone());
            div()
                .size_full()
                .flex()
                .child(
                    self.inner
                        .clone()
                        .cached(StyleRefinement::default().w(px(60.)).h(px(30.))),
                )
                .child(
                    div()
                        .id("after")
                        .size(px(20.))
                        .bg(blue())
                        .track_focus(&self.focus)
                        .on_key_down(move |_, _, _| keys.after_keys.update(|keys| keys + 1))
                        .on_click(move |_, _, _| clicks.after_clicks.update(|clicks| clicks + 1))
                        .child(
                            canvas(move |b, _, _| bounds.after_bounds.set(b), |_, _, _, _| {})
                                .size_full(),
                        ),
                )
        }
    }

    struct NestedRoot {
        probe: Rc<NestedProbe>,
        outer: Entity<Outer>,
    }

    impl Render for NestedRoot {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.probe.root_renders.update(|renders| renders + 1);
            div().size_full().flex().child(
                self.outer
                    .clone()
                    .cached(StyleRefinement::default().w(px(200.)).h(px(50.))),
            )
        }
    }

    #[crate::test]
    fn a_frame_reused_around_nested_holes_stays_reusable(cx: &mut TestAppContext) {
        let probe = Rc::new(NestedProbe::default());
        probe.animate.set(true);
        let window = cx.open_window(size(px(400.), px(100.)), {
            let probe = probe.clone();
            move |_, cx| NestedRoot {
                outer: cx.new(|cx| Outer {
                    inner: cx.new(|_| Inner {
                        probe: probe.clone(),
                    }),
                    focus: cx.focus_handle(),
                    probe: probe.clone(),
                }),
                probe,
            }
        });
        cx.run_until_parked();
        window
            .update(cx, |_, window, _| window.active.set(true))
            .unwrap();
        frame(cx, window);
        let around = (probe.root_renders.get(), probe.outer_renders.get());
        let quads = |cx: &mut TestAppContext| {
            let inner = probe.inner_bounds.get();
            quad_colors(cx, window, inner)
        };

        // Contained frames inside the cached outer view, each moving
        // everything after the inner one.
        for _ in 0..4 {
            let renders = probe.inner_renders.get();
            frame(cx, window);
            assert_eq!(
                (probe.root_renders.get(), probe.outer_renders.get()),
                around
            );
            assert_eq!(probe.inner_renders.get(), renders + 1);
            let renders = renders + 1;
            assert_eq!(quads(cx), vec![flicker_color(renders); renders % 3 + 1]);
        }

        // What follows the hole in the outer view still takes input.
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.simulate_click(probe.after_bounds.get().center(), Modifiers::none());
        assert_eq!(probe.after_clicks.get(), 1);
        frame(cx, window);
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.simulate_keystrokes("a");
        assert_eq!(probe.after_keys.get(), 1);

        // Stop animating, then render the root: the outer view is reused from
        // the ranges the contained frames moved.
        probe.animate.set(false);
        frame(cx, window);
        let (outer_renders, inner_renders) = (probe.outer_renders.get(), probe.inner_renders.get());
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        frame(cx, window);
        assert!(probe.root_renders.get() > around.0);
        assert_eq!(
            probe.outer_renders.get(),
            outer_renders,
            "the outer view is reused"
        );
        assert_eq!(
            quads(cx),
            vec![flicker_color(inner_renders); inner_renders % 3 + 1]
        );
        assert_eq!(
            quad_colors(cx, window, probe.after_bounds.get()),
            vec![blue()],
            "the rest of the reused outer view paints as before"
        );
        let mut visual = VisualTestContext::from_window(window.into(), cx);
        visual.simulate_click(probe.after_bounds.get().center(), Modifiers::none());
        visual.simulate_keystrokes("b");
        assert_eq!((probe.after_clicks.get(), probe.after_keys.get()), (2, 2));
    }
}
