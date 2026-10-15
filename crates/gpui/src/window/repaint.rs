//! Repaint-only frames: the invalidation level below re-rendering.
//!
//! See [`Window::paint_animated`] for the design.

use std::{mem, rc::Rc};

use crate::{
    App, AppContext as _, ColorExt, ContentMask, EntityId, MonochromeSprite, Pixels, QuadPulse,
    ScaledPixels, SubpixelSprite, TextColorMap, TransformationMatrix, Window,
    scene::{Damage, PrimitiveRanges, RecordedGlyph, RecordedPaint, RepaintId},
};

use super::{DrawPhase, ScreenTransform, glyph_color_spans};

/// Paint output captured by [`Window::record_paint`], for a retained paint
/// closure to paint again on later frames with [`Window::paint_recording`].
/// Cloning is cheap.
#[derive(Clone, Default)]
pub struct PaintRecording {
    paints: Rc<[RecordedPaint]>,
}

/// A closure [`Window::paint_animated`] keeps to repaint its region.
pub(crate) struct RetainedPaint {
    paint: Box<dyn FnMut(&mut Window, &mut App) -> bool>,
    context: PaintContext,
    /// Whether its last call asked to paint again on the next frame.
    animating: bool,
    /// Whether it painted inline during the draw in progress, so that draw
    /// doesn't repaint it a second time.
    pub(super) painted_this_draw: bool,
}

/// The paint state a retained closure was first called in, restored for
/// every repaint.
struct PaintContext {
    view: EntityId,
    content_mask: ContentMask<Pixels>,
    opacity: f32,
    pulse: QuadPulse,
    transform: ScreenTransform,
    text_color_map: Option<TextColorMap>,
    rem_size: Pixels,
}

impl Window {
    /// Paints `paint` now and, while it returns `true`, again on every
    /// following frame without rendering, laying out, prepainting or painting
    /// anything else. This is the invalidation level below
    /// [`Context::notify`](crate::Context::notify): a repaint of a region
    /// whose layout is unchanged.
    ///
    /// # Design
    ///
    /// Browsers separate layout, paint and composite invalidation; GPUI's only
    /// invalidation was "re-render this view", and a notified view dirties
    /// every ancestor, so one animated leaf rebuilt the whole window each
    /// frame. A retained paint closure is GPUI's paint invalidation:
    ///
    /// - What the closure paints becomes a *repaint region* of the scene. The
    ///   region stacks its primitives in draw orders reserved for it, above
    ///   everything painted before it and below everything painted after, so
    ///   replacing its primitives never reorders the rest of the scene,
    ///   whatever they cover. It takes one paint operation, so cached views
    ///   and deferred draws replaying the scene keep their indices.
    /// - A frame where only regions asked to repaint is a *repaint frame*:
    ///   the window calls each animating closure in the paint state captured
    ///   at its first call (content mask, opacity, pulse, transform, text color
    ///   map, rem size) and splices the result into the rendered scene in
    ///   place. Hitboxes, listeners, input handlers, tab stops, element state,
    ///   cursor styles and deferred draws are the previous frame's, untouched;
    ///   the cost is that of the closures: a region keeps slack in the
    ///   scene's vectors, so a repaint overwrites its own run, and only one
    ///   that outgrows its slack moves the rest of a vector.
    /// - [`Scene::damage`](crate::Scene::damage) on a repaint frame lists the
    ///   repainted regions' old and new bounds, widened through any blur that
    ///   samples them. Full draws compare the new scene with the old one.
    /// - A full draw keeps a region whose view it reuses from the cache, and
    ///   repaints it after the draw if it is animating; a view that renders
    ///   again paints a new region with a new closure.
    /// - Correctness first: a repaint that stacks deeper than the region's
    ///   reserved orders notifies the region's view instead, so the next frame
    ///   renders it in full.
    ///
    /// # Contract
    ///
    /// The closure only paints primitives, in window coordinates captured at
    /// the first call (copy the element's bounds in): quads, paths, text,
    /// sprites, recordings. It must not build elements, register listeners,
    /// hitboxes, cursor styles, input handlers or tab stops, or read element
    /// state: none of these are rebuilt by a repaint, and what a repaint
    /// registers is dropped. Derive the animated value inside the closure from
    /// the time or shared state, and return whether to paint again. Respect
    /// [`App::reduce_motion`] by returning `false`. Inside
    /// [`Self::record_paint`], or inside another retained closure, the
    /// closure paints once where it is called.
    ///
    /// Anything that changes layout must still go through
    /// [`Context::notify`](crate::Context::notify) or
    /// [`Self::request_animation_frame`]; paint-only motion — color, opacity,
    /// text shimmer, a spinner's rotation — belongs here. To animate content
    /// painted by child elements, paint the children once inside
    /// [`Self::record_paint`] and paint the recording from the closure.
    pub fn paint_animated(
        &mut self,
        mut paint: impl FnMut(&mut Window, &mut App) -> bool + 'static,
        cx: &mut App,
    ) {
        self.invalidator.debug_assert_paint();
        if self.next_frame.scene.recording.is_some() || self.painting_retained {
            paint(self, cx);
            return;
        }
        self.next_repaint_id += 1;
        let id = RepaintId(self.next_repaint_id);
        let context = PaintContext {
            view: self.current_view(),
            content_mask: self.content_mask(),
            opacity: self.element_opacity,
            pulse: self.element_pulse,
            transform: self.element_transform,
            text_color_map: self.text_color_map.clone(),
            rem_size: self.rem_size(),
        };
        self.next_frame.scene.begin_repaint_region(id);
        let animating = paint(self, cx);
        self.next_frame.scene.end_repaint_region();
        self.retained_paints.insert(
            id,
            RetainedPaint {
                paint: Box::new(paint),
                context,
                animating,
                painted_this_draw: true,
            },
        );
        if animating {
            self.platform_window.schedule_frame();
            self.invalidator.wake_platform();
        }
    }

    /// Runs `paint`, capturing what it paints instead of painting it. Paint
    /// the result with [`Self::paint_recording`], typically from a
    /// [`Self::paint_animated`] closure. Everything else `paint` does — mouse
    /// listeners, hitboxes, cursor styles — happens as usual.
    pub fn record_paint(&mut self, paint: impl FnOnce(&mut Self)) -> PaintRecording {
        self.invalidator.debug_assert_paint();
        let outer = self.next_frame.scene.recording.replace(Vec::new());
        paint(self);
        let recorded = mem::replace(&mut self.next_frame.scene.recording, outer);
        PaintRecording {
            paints: recorded.unwrap_or_default().into(),
        }
    }

    /// Paints a [`PaintRecording`] as it was recorded, clipped to the current
    /// content mask. Its glyphs take the current
    /// [`text color map`](Self::with_text_color_map), or the one in effect when
    /// they were recorded if there is none; nothing else is restyled.
    pub fn paint_recording(&mut self, recording: &PaintRecording) {
        self.invalidator.debug_assert_paint();
        let content_mask = self.content_mask().scale(self.scale_factor());
        let glyph_mask = self.snapped_content_mask();
        for paint in recording.paints.iter() {
            match paint {
                RecordedPaint::Primitive(primitive) => {
                    let mut primitive = primitive.clone();
                    let mask = primitive.content_mask_mut();
                    mask.bounds = mask.bounds.intersect(&content_mask.bounds);
                    self.next_frame.scene.insert_primitive(primitive);
                }
                RecordedPaint::Glyph(glyph) => {
                    let map = self
                        .text_color_map
                        .clone()
                        .or_else(|| glyph.color_map.clone());
                    let mask = ContentMask {
                        bounds: glyph.content_mask.bounds.intersect(&glyph_mask.bounds),
                    };
                    self.insert_glyph(glyph, mask, map.as_ref());
                }
            }
        }
    }

    /// Inserts a glyph's sprites, split into the spans `map` colors.
    pub(super) fn insert_glyph(
        &mut self,
        glyph: &RecordedGlyph,
        content_mask: ContentMask<ScaledPixels>,
        map: Option<&TextColorMap>,
    ) {
        let spans = glyph_color_spans(
            glyph.bounds,
            content_mask,
            self.scale_factor(),
            glyph.color,
            map,
        );
        for (content_mask, color) in spans {
            let color = color.opacity(glyph.opacity).into();
            if glyph.subpixel {
                self.next_frame.scene.insert_primitive(SubpixelSprite {
                    order: 0,
                    pad: 0,
                    bounds: glyph.bounds,
                    content_mask,
                    color,
                    tile: glyph.tile,
                    transformation: TransformationMatrix::unit(),
                });
            } else {
                self.next_frame.scene.insert_primitive(MonochromeSprite {
                    order: 0,
                    pad: 0,
                    bounds: glyph.bounds,
                    content_mask,
                    color,
                    tile: glyph.tile,
                    transformation: TransformationMatrix::unit(),
                });
            }
        }
    }

    /// Whether a retained paint asked to paint the next frame.
    pub(crate) fn repaint_pending(&self) -> bool {
        self.retained_paints.values().any(|paint| paint.animating)
    }

    /// Draws a repaint frame: animating regions repaint into the rendered
    /// scene, and nothing else is rendered, laid out or painted.
    pub(crate) fn repaint(&mut self, cx: &mut App) {
        #[cfg(feature = "frame-rate")]
        let repaint_start = scheduler::Instant::now();
        let mut damage = Damage::default();
        self.invalidator.set_phase(DrawPhase::Paint);
        let unrepaintable = self.repaint_animating_regions(&mut damage, cx);
        #[cfg(feature = "frame-rate")]
        self.repaint_frame_overlay(&mut damage);
        self.invalidator.set_phase(DrawPhase::None);
        self.rendered_frame.scene.generation += 1;
        let changed = mem::take(&mut self.rendered_frame.scene.replaced_primitives);
        self.settle_damage(Some(damage), Some(changed));
        self.needs_present.set(true);
        for view in unrepaintable {
            cx.notify(view);
        }
        #[cfg(feature = "frame-rate")]
        self.debug_frame_overlay
            .record_draw(repaint_start.elapsed());
        if self.verify_frames {
            self.verify_partial_frame("repaint", &[], &Default::default(), cx);
        }
    }

    /// Repaints the rendered scene's animating regions, except those the draw
    /// that built it painted inline, adding their bounds to `damage`. Returns
    /// the views of regions that outgrew their reserved orders, which must
    /// render again; their closures stop animating. Runs between frames, with
    /// `next_frame` empty: each region repaints into it, then moves over.
    pub(crate) fn repaint_animating_regions(
        &mut self,
        damage: &mut Damage,
        cx: &mut App,
    ) -> Vec<EntityId> {
        let mut unrepaintable = Vec::new();
        let mut pulses_changed = false;
        let mut due = self
            .retained_paints
            .iter()
            .filter(|(_, retained)| retained.animating && !retained.painted_this_draw)
            .filter_map(|(id, _)| Some((*self.rendered_frame.scene.region_indices.get(id)?, *id)))
            .collect::<Vec<_>>();
        // In paint order, which region indices follow within a draw.
        due.sort_unstable_by_key(|(index, _)| *index);
        for (index, id) in due {
            let base = self.rendered_frame.scene.repaint_regions[index]
                .orders
                .start;
            let mut retained = self.retained_paints.remove(&id).unwrap();
            self.next_frame.scene.begin_repaint_region_at(id, base);
            let RetainedPaint { paint, context, .. } = &mut retained;
            retained.animating = self.with_paint_context(context, |window| paint(window, cx));
            self.next_frame.scene.end_repaint_region();
            pulses_changed |= self.rendered_frame.scene.region_pulses(index)
                || self.next_frame.scene.region_pulses(0);
            if !self.rendered_frame.scene.replace_repaint_region(
                index,
                &mut self.next_frame.scene,
                damage,
            ) {
                retained.animating = false;
                unrepaintable.push(retained.context.view);
            }
            // Whatever else the closure registered is dropped here.
            self.next_frame.clear();
            self.retained_paints.insert(id, retained);
        }
        if pulses_changed {
            self.rendered_scene_pulses = self
                .rendered_frame
                .scene
                .quads
                .iter()
                .any(|quad| quad.pulse.is_active());
        }
        unrepaintable
    }

    /// Completes the damage and changed primitive ranges of a new rendered
    /// scene: damage widened through its filters and, if the scene before it
    /// was never presented, both joined with that scene's, so they hold
    /// against the last scene presented. `changed` is `None` for a scene
    /// built afresh.
    pub(crate) fn settle_damage(
        &mut self,
        damage: Option<Damage>,
        changed: Option<PrimitiveRanges>,
    ) {
        let unpresented_changes = self.changed_since_present.take();
        let changed = match (changed, unpresented_changes) {
            (Some(changed), _) if self.rendered_scene_presented => Some(changed),
            (Some(mut changed), Some(unpresented)) => {
                changed.extend(&unpresented);
                Some(changed)
            }
            _ => None,
        };
        self.rendered_frame.scene.changed_primitives = changed.clone();
        self.changed_since_present = changed;

        let unpresented = self.damage_since_present.take();
        let mut damage = match (damage, unpresented) {
            (Some(damage), _) if self.rendered_scene_presented => Some(damage),
            (Some(mut damage), Some(unpresented)) => {
                damage.merge(unpresented);
                Some(damage)
            }
            _ => None,
        };
        if let Some(damage) = &mut damage {
            self.rendered_frame.scene.add_filter_damage(damage);
        }
        self.rendered_frame.scene.damage = damage.as_ref().map(Damage::regions);
        self.damage_since_present = damage;
        self.rendered_scene_presented = false;
    }

    /// After a draw: forgets closures whose regions left the scene, and lets
    /// those painted inline repaint from the next frame on.
    pub(crate) fn retain_live_paints(&mut self) {
        let live = &self.rendered_frame.scene.region_indices;
        self.retained_paints.retain(|id, paint| {
            paint.painted_this_draw = false;
            live.contains_key(id)
        });
    }

    fn with_paint_context<R>(
        &mut self,
        context: &PaintContext,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        debug_assert!(self.content_mask_stack.is_empty());
        self.painting_retained = true;
        self.content_mask_stack.push(context.content_mask);
        self.rem_size_override_stack.push(context.rem_size);
        self.rendered_entity_stack.push(context.view);
        let opacity = mem::replace(&mut self.element_opacity, context.opacity);
        let pulse = mem::replace(&mut self.element_pulse, context.pulse);
        let transform = mem::replace(&mut self.element_transform, context.transform);
        let text_color_map = mem::replace(&mut self.text_color_map, context.text_color_map.clone());
        let result = f(self);
        self.text_color_map = text_color_map;
        self.element_transform = transform;
        self.element_pulse = pulse;
        self.element_opacity = opacity;
        self.rendered_entity_stack.pop();
        self.rem_size_override_stack.pop();
        self.content_mask_stack.pop();
        self.painting_retained = false;
        result
    }

    /// Repaints the frame-rate overlay's region, so its readout stays live on
    /// repaint frames.
    #[cfg(feature = "frame-rate")]
    pub(super) fn repaint_frame_overlay(&mut self, damage: &mut Damage) {
        let scene = &self.rendered_frame.scene;
        let Some(&index) = scene.region_indices.get(&RepaintId::FRAME_OVERLAY) else {
            return;
        };
        let base = scene.repaint_regions[index].orders.start;
        self.debug_frame_overlay.set_frame_demand(
            self.repaint_pending() || !self.next_frame_callbacks.borrow().is_empty(),
        );
        self.next_frame
            .scene
            .begin_repaint_region_at(RepaintId::FRAME_OVERLAY, base);
        self.debug_frame_overlay.paint(
            &mut self.next_frame.scene,
            self.viewport_size,
            self.scale_factor,
        );
        self.next_frame.scene.end_repaint_region();
        // The overlay stacks two deep, far within a region's orders.
        let replaced = self.rendered_frame.scene.replace_repaint_region(
            index,
            &mut self.next_frame.scene,
            damage,
        );
        debug_assert!(replaced);
        self.next_frame.scene.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc, time::Duration};

    use crate::{
        Animation, AnimationExt as _, AnyWindowHandle, App, AppContext as _, AtlasTextureId,
        AtlasTextureKind, AtlasTile, Bounds, ColorExt as _, Context, DevicePixels, Element,
        ElementId, Entity, GlobalElementId, Hsla, InspectorElementId, InteractiveElement as _,
        IntoElement, LayoutId, Modifiers, ParentElement as _, Pixels, Render, RequestFrameOptions,
        ScaledPixels, StatefulInteractiveElement as _, Style, StyleRefinement, Styled as _,
        TestAppContext, TileId, VisualTestContext, Window, WindowHandle, canvas, div, fill, hsla,
        point,
        prelude::FluentBuilder as _,
        px, red,
        scene::{RecordedGlyph, RecordedPaint},
        size,
    };

    use super::PaintRecording;

    /// Counts what each kind of frame work touches.
    #[derive(Default)]
    struct Probe {
        root_renders: Cell<usize>,
        glow_renders: Cell<usize>,
        glow_layouts: Cell<usize>,
        /// Calls of the glow's paint closure, which picks its color.
        glow_paints: Cell<usize>,
        /// How many overlapping quads each glow paint stacks.
        glow_depth: Cell<usize>,
        glow_bounds: Cell<Bounds<Pixels>>,
        bar_bounds: Cell<Bounds<Pixels>>,
        button_bounds: Cell<Bounds<Pixels>>,
        clicks: Cell<usize>,
    }

    fn glow_color(paint: usize) -> Hsla {
        hsla((paint % 100) as f32 / 100.0, 0.8, 0.5, 1.0)
    }

    /// A fixed-size element whose color changes every frame through a
    /// retained paint closure.
    struct Glow {
        width: Pixels,
        probe: Rc<Probe>,
    }

    impl IntoElement for Glow {
        type Element = Self;

        fn into_element(self) -> Self {
            self
        }
    }

    impl Element for Glow {
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
            self.probe.glow_layouts.update(|layouts| layouts + 1);
            let mut style = Style::default();
            style.size = size(self.width.into(), px(20.).into());
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
            cx: &mut App,
        ) {
            self.probe.glow_bounds.set(bounds);
            let probe = self.probe.clone();
            window.paint_animated(
                move |window, _| {
                    let paint = probe.glow_paints.get();
                    probe.glow_paints.set(paint + 1);
                    for _ in 0..probe.glow_depth.get().max(1) {
                        window.paint_quad(fill(bounds, glow_color(paint)));
                    }
                    true
                },
                cx,
            );
        }
    }

    struct Glowing {
        width: Pixels,
        probe: Rc<Probe>,
    }

    impl Render for Glowing {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.probe.glow_renders.update(|renders| renders + 1);
            div().p(px(10.)).child(Glow {
                width: self.width,
                probe: self.probe.clone(),
            })
        }
    }

    /// A bar that may grow, above a cached view holding the glow and a button.
    struct Root {
        probe: Rc<Probe>,
        glowing: Entity<Glowing>,
        grow_bar: bool,
    }

    impl Render for Root {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.probe.root_renders.update(|renders| renders + 1);
            let (bar_probe, button_probe, click_probe) =
                (self.probe.clone(), self.probe.clone(), self.probe.clone());
            let bar = div().h(px(20.)).w(px(10.)).bg(red()).child(
                canvas(
                    move |bounds, _, _| bar_probe.bar_bounds.set(bounds),
                    |_, _, _, _| {},
                )
                .size_full(),
            );
            let row = div()
                .flex()
                .flex_row()
                .child(
                    self.glowing
                        .clone()
                        .cached(StyleRefinement::default().w(px(200.)).h(px(40.))),
                )
                .child(
                    div()
                        .id("button")
                        .ml(px(20.))
                        .size(px(40.))
                        .bg(red())
                        .on_click(move |_, _, _| click_probe.clicks.update(|clicks| clicks + 1))
                        .child(
                            canvas(
                                move |bounds, _, _| button_probe.button_bounds.set(bounds),
                                |_, _, _, _| {},
                            )
                            .size_full(),
                        ),
                );
            div()
                .size_full()
                .flex()
                .flex_col()
                .map(|root| {
                    if self.grow_bar {
                        root.child(bar.with_animation(
                            "bar",
                            Animation::new(Duration::from_secs(1)).repeat_synced(),
                            |bar, delta| bar.w(px(10. + 100. * delta)),
                        ))
                    } else {
                        root.child(bar)
                    }
                })
                .child(row)
        }
    }

    fn open(cx: &mut TestAppContext, grow_bar: bool) -> (WindowHandle<Root>, Rc<Probe>) {
        let probe = Rc::new(Probe::default());
        let window = cx.open_window(size(px(400.), px(100.)), {
            let probe = probe.clone();
            move |_, cx| Root {
                glowing: cx.new(|_| Glowing {
                    width: px(60.),
                    probe: probe.clone(),
                }),
                probe,
                grow_bar,
            }
        });
        cx.run_until_parked();
        // Inactive windows throttle frames, which would skip the frames these
        // tests deliver.
        window
            .update(cx, |_, window, _| window.active.set(true))
            .unwrap();
        // Present once: damage holds against the last presented scene, and
        // the test app draws without presenting.
        frame(cx, window);
        (window, probe)
    }

    fn frame(cx: &mut TestAppContext, window: impl Into<AnyWindowHandle>) {
        cx.test_window(window.into())
            .simulate_frame_request(RequestFrameOptions::default());
        cx.run_until_parked();
    }

    /// The colors of the painted quads covering `bounds`.
    fn colors_at(
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

    /// What a repaint of `bounds` reports: rounded out, one pixel of margin.
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

    fn contains(damage: &[Bounds<ScaledPixels>], bounds: Bounds<Pixels>) -> bool {
        let bounds = bounds.scale(2.0);
        damage
            .iter()
            .any(|region| bounds.is_contained_within(region))
    }

    #[crate::test]
    fn repaint_frames_repaint_only_the_animated_element(cx: &mut TestAppContext) {
        let (window, probe) = open(cx, false);
        let glow = probe.glow_bounds.get();
        let renders = (probe.root_renders.get(), probe.glow_renders.get());
        let layouts = probe.glow_layouts.get();
        let mut colors = colors_at(cx, window, glow);
        assert_eq!(colors.len(), 1, "the glow paints one quad");
        assert!(
            window
                .update(cx, |_, window, _| window.awaits_next_frame())
                .unwrap(),
            "an animating region asks for the next frame"
        );

        for _ in 0..3 {
            let generation = window
                .update(cx, |_, window, _| window.rendered_frame.scene.generation)
                .unwrap();
            frame(cx, window);

            assert_eq!(
                (probe.root_renders.get(), probe.glow_renders.get()),
                renders,
                "a repaint renders no view"
            );
            assert_eq!(
                probe.glow_layouts.get(),
                layouts,
                "a repaint lays nothing out"
            );
            let repainted = colors_at(cx, window, glow);
            assert_eq!(repainted.len(), 1);
            assert_ne!(repainted, colors, "the glow's color moves every frame");
            colors = repainted;
            assert_eq!(
                damage(cx, window),
                Some(vec![damage_of(glow)]),
                "only the glow's pixels change"
            );
            window
                .update(cx, |_, window, _| {
                    assert_eq!(window.rendered_frame.scene.generation, generation + 1);
                    assert!(!window.invalidator.is_dirty());
                })
                .unwrap();
        }
    }

    #[crate::test]
    fn size_animations_lay_out_while_regions_repaint(cx: &mut TestAppContext) {
        let (window, probe) = open(cx, true);
        let glow = probe.glow_bounds.get();
        let glow_renders = probe.glow_renders.get();
        let layouts = probe.glow_layouts.get();
        let mut bar = probe.bar_bounds.get();
        let mut colors = colors_at(cx, window, glow);
        let button = probe.button_bounds.get();

        for _ in 0..3 {
            let root_renders = probe.root_renders.get();
            cx.executor().advance_clock(Duration::from_millis(100));
            frame(cx, window);

            // The bar's animation renders and lays out its view as before.
            assert_eq!(probe.root_renders.get(), root_renders + 1);
            let grown = probe.bar_bounds.get();
            assert!(
                grown.size.width > bar.size.width,
                "the bar is laid out wider"
            );
            bar = grown;
            // The cached glow is reused, yet keeps animating.
            assert_eq!(probe.glow_renders.get(), glow_renders);
            assert_eq!(probe.glow_layouts.get(), layouts);
            let repainted = colors_at(cx, window, glow);
            assert_eq!(repainted.len(), 1);
            assert_ne!(repainted, colors);
            colors = repainted;
            let damage = damage(cx, window).expect("a draw at the same size reports damage");
            assert!(
                contains(&damage, bar),
                "the bar's new extent is damaged: {damage:?} {bar:?}"
            );
            assert!(contains(&damage, glow), "the repainted glow is damaged");
            assert!(
                !damage
                    .iter()
                    .any(|region| region.intersects(&button.scale(2.0))),
                "the unchanged button is not: {damage:?}"
            );
        }
    }

    #[crate::test]
    fn a_resized_region_lays_out_again_and_repaints_at_its_new_size(cx: &mut TestAppContext) {
        let (window, probe) = open(cx, false);
        let narrow = probe.glow_bounds.get();
        let layouts = probe.glow_layouts.get();
        frame(cx, window);

        let glowing = window
            .update(cx, |root, _, _| root.glowing.clone())
            .unwrap();
        glowing.update(cx, |glowing, cx| {
            glowing.width = px(120.);
            cx.notify();
        });
        cx.run_until_parked();
        let wide = probe.glow_bounds.get();
        assert_eq!(wide.size.width, px(120.));
        assert_eq!(probe.glow_layouts.get(), layouts + 1);
        let damage_after_resize = damage(cx, window).unwrap();
        assert!(contains(&damage_after_resize, narrow));
        assert!(contains(&damage_after_resize, wide));

        frame(cx, window);
        assert_eq!(probe.glow_layouts.get(), layouts + 1);
        assert_eq!(colors_at(cx, window, wide).len(), 1);
        assert!(colors_at(cx, window, narrow).is_empty());
        assert_eq!(damage(cx, window), Some(vec![damage_of(wide)]));
    }

    #[crate::test]
    fn siblings_hit_test_after_repaint_frames(cx: &mut TestAppContext) {
        let (window, probe) = open(cx, false);
        let renders = probe.root_renders.get();
        for _ in 0..3 {
            frame(cx, window);
        }
        assert_eq!(probe.root_renders.get(), renders);

        let button = probe.button_bounds.get();
        let mut cx = VisualTestContext::from_window(window.into(), cx);
        cx.simulate_click(button.center(), Modifiers::none());
        assert_eq!(probe.clicks.get(), 1, "the button still receives its click");
        cx.simulate_click(point(px(5.), px(80.)), Modifiers::none());
        assert_eq!(probe.clicks.get(), 1, "a click elsewhere does not reach it");
    }

    #[crate::test]
    fn a_region_outgrowing_its_orders_renders_again(cx: &mut TestAppContext) {
        let (window, probe) = open(cx, false);
        let glow = probe.glow_bounds.get();
        let renders = probe.glow_renders.get();

        // Stacking deeper than the region's reserved orders cannot be
        // spliced in place: the scene keeps the last frame, and the view
        // renders again on the next.
        probe.glow_depth.set(300);
        frame(cx, window);
        assert_eq!(probe.glow_renders.get(), renders + 1);
        assert_eq!(colors_at(cx, window, glow).len(), 300);

        // The new region reserved enough orders to repaint in place again.
        // (The first of these frames also presents the test app's draw.)
        for _ in 0..2 {
            frame(cx, window);
        }
        assert_eq!(probe.glow_renders.get(), renders + 1);
        assert_eq!(colors_at(cx, window, glow).len(), 300);
        assert_eq!(damage(cx, window), Some(vec![damage_of(glow)]));
    }

    #[crate::test]
    fn full_draws_damage_only_what_changed(cx: &mut TestAppContext) {
        let (window, probe) = open(cx, false);
        let glow = probe.glow_bounds.get();
        frame(cx, window);

        // Re-rendering the root repaints none of its unchanged pixels; the
        // reused glow repaints because it animates.
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();
        assert_eq!(damage(cx, window), Some(vec![damage_of(glow)]));

        // A resize may move everything.
        cx.simulate_window_resize(window.into(), size(px(500.), px(100.)));
        cx.run_until_parked();
        assert_eq!(damage(cx, window), None);
    }

    #[cfg(feature = "frame-rate")]
    #[crate::test]
    fn the_frame_rate_overlay_stays_live_on_repaint_frames(cx: &mut TestAppContext) {
        let (window, probe) = open(cx, false);
        window
            .update(cx, |_, window, _| {
                window.set_debug_frame_overlay_mode(crate::DebugFrameOverlayMode::Full)
            })
            .unwrap();
        cx.run_until_parked();
        frame(cx, window);
        let renders = probe.root_renders.get();

        // Each present counts a frame, so the readout changes every frame.
        frame(cx, window);
        assert_eq!(probe.root_renders.get(), renders);
        let damage = damage(cx, window).unwrap();
        assert!(damage.contains(&damage_of(probe.glow_bounds.get())));
        assert!(
            damage
                .iter()
                .any(|region| region.left() > ScaledPixels(400.)),
            "the overlay in the top right repaints: {damage:?}"
        );
    }

    /// The shimmer pattern: children painted once into a recording, repainted
    /// each frame under a moving text color map.
    struct Shimmer {
        glyph: RecordedGlyph,
        phase: Rc<Cell<f32>>,
    }

    impl IntoElement for Shimmer {
        type Element = Self;

        fn into_element(self) -> Self {
            self
        }
    }

    impl Element for Shimmer {
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
            style.size = size(px(40.).into(), px(20.).into());
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
            cx: &mut App,
        ) {
            // The test text system rasterizes nothing, so the glyph is
            // recorded by hand beside a painted background.
            let mut recording =
                window.record_paint(|window| window.paint_quad(fill(bounds, red())));
            let mut paints = recording.paints.to_vec();
            paints.push(RecordedPaint::Glyph(self.glyph.clone()));
            recording = PaintRecording {
                paints: paints.into(),
            };
            let phase = self.phase.clone();
            window.paint_animated(
                move |window, _| {
                    let phase = phase.get();
                    window.with_text_color_map(
                        Some(Rc::new(move |_, color: Hsla| color.opacity(phase))),
                        |window| window.paint_recording(&recording),
                    );
                    true
                },
                cx,
            );
        }
    }

    struct ShimmerView {
        glyph: RecordedGlyph,
        phase: Rc<Cell<f32>>,
    }

    impl Render for ShimmerView {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().child(Shimmer {
                glyph: self.glyph.clone(),
                phase: self.phase.clone(),
            })
        }
    }

    #[crate::test]
    fn recorded_glyphs_take_the_repaint_time_color_map(cx: &mut TestAppContext) {
        let glyph = RecordedGlyph {
            bounds: Bounds::new(
                point(ScaledPixels(4.), ScaledPixels(4.)),
                size(ScaledPixels(8.), ScaledPixels(12.)),
            ),
            content_mask: Default::default(),
            color: hsla(0., 0., 0., 1.),
            opacity: 1.,
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: AtlasTextureKind::Monochrome,
                },
                tile_id: TileId(1),
                padding: 0,
                bounds: Bounds::new(
                    point(DevicePixels(0), DevicePixels(0)),
                    size(DevicePixels(8), DevicePixels(12)),
                ),
            },
            subpixel: false,
            color_map: None,
        };
        let glyph = RecordedGlyph {
            content_mask: crate::ContentMask {
                bounds: glyph.bounds,
            },
            ..glyph
        };
        let phase = Rc::new(Cell::new(0.25));
        let window = cx.open_window(size(px(100.), px(100.)), {
            let phase = phase.clone();
            move |_, _| ShimmerView { glyph, phase }
        });
        cx.run_until_parked();
        window
            .update(cx, |_, window, _| window.active.set(true))
            .unwrap();
        let glyph_alphas = |cx: &mut TestAppContext| {
            window
                .update(cx, |_, window, _| {
                    let scene = &window.rendered_frame.scene;
                    // Regions keep invisible slack behind what they paint.
                    (
                        scene
                            .monochrome_sprites
                            .iter()
                            .filter(|sprite| !sprite.bounds.is_empty())
                            .map(|sprite| sprite.color.a)
                            .collect::<Vec<_>>(),
                        scene
                            .quads
                            .iter()
                            .filter(|quad| !quad.bounds.is_empty())
                            .count(),
                    )
                })
                .unwrap()
        };
        assert_eq!(glyph_alphas(cx), (vec![0.25], 1));

        phase.set(0.75);
        frame(cx, window);
        assert_eq!(
            glyph_alphas(cx),
            (vec![0.75], 1),
            "the glyph recolors and the background stays"
        );
    }
}
