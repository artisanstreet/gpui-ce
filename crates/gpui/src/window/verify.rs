//! Frame verification: a safety net for partial frames.
//!
//! Contained, in-place and repaint frames rely on the views around what they
//! redraw being unchanged. A view that reads shared state which changed
//! without a notify breaks that silently: the full path would have rendered
//! it again in passing, a partial frame doesn't. With verification on, every
//! partial frame is followed by a full draw of the same invalidation, and the
//! two are compared: the paint in paint order (orders aside) and the
//! hitboxes. A difference is logged, naming the cached views whose output
//! differs and the damage the partial frame reported, and panics in tests.
//!
//! The contents of what the partial frame redrew, and of retained paints,
//! aren't compared: they render again in the full draw too, and animations
//! legitimately move between the two. Their placement is.
//!
//! Turn it on with `GPUI_VERIFY_FRAMES=1`, or [`Window::set_verify_frames`]
//! in tests. Off, it costs a flag check per partial frame.

use std::{any::TypeId, ops::Range};

use collections::{FxHashMap, FxHashSet};

use crate::{
    App, Bounds, ContentMask, EntityId, HitboxBehavior, HitboxId, Pixels, ViewElementState, Window,
    scene::{FlatPaint, RepaintId},
};

/// What verification compares between a partial frame and a full one.
struct FrameSnapshot {
    paint: Vec<FlatPaint>,
    hitboxes: Vec<(Bounds<Pixels>, ContentMask<Pixels>, HitboxBehavior)>,
    /// Each cached view's own paint, to name those that differ.
    views: FxHashMap<EntityId, Vec<FlatPaint>>,
}

impl Window {
    /// Verifies every partial frame against a full draw (see
    /// [`crate::window::verify`]); panics on a difference.
    #[cfg(any(test, feature = "test-support"))]
    pub fn set_verify_frames(&mut self, verify: bool) {
        self.verify_frames = verify;
    }

    /// Whether `GPUI_VERIFY_FRAMES=1` asks to verify partial frames.
    pub(crate) fn verify_frames_from_env() -> bool {
        std::env::var("GPUI_VERIFY_FRAMES").is_ok_and(|value| value == "1")
    }

    /// Draws the invalidation the partial frame just drew (`dirty`, with the
    /// cached views in `redrawn` redrawn) in full, and compares.
    pub(crate) fn verify_partial_frame(
        &mut self,
        kind: &str,
        redrawn: &[EntityId],
        dirty: &FxHashSet<EntityId>,
        cx: &mut App,
    ) {
        let layers = redrawn
            .iter()
            .filter_map(|view| self.view_state(*view)?.layer)
            .collect::<FxHashSet<_>>();
        let partial = self.snapshot(redrawn, &layers);
        let damage = self.rendered_frame.scene.damage.clone();

        for view in dirty {
            self.mark_view_dirty(*view);
        }
        self.drawing_full_for_verification = true;
        self.draw(cx).clear(cx);
        self.drawing_full_for_verification = false;
        let full = self.snapshot(redrawn, &layers);

        if partial.paint == full.paint && partial.hitboxes == full.hitboxes {
            return;
        }
        let mut views = partial
            .views
            .iter()
            .filter(|(view, paint)| full.views.get(view).is_some_and(|full| full != *paint))
            .map(|(view, paint)| (paint.len(), self.view_name(*view)))
            .collect::<Vec<_>>();
        // Innermost first.
        views.sort();
        let views = views.into_iter().map(|(_, name)| name).collect::<Vec<_>>();
        let culprit = if views.is_empty() {
            "outside every cached view".to_string()
        } else {
            format!("in {}", views.join(", "))
        };
        let message = format!(
            "frame verification: a {kind} frame painted differently from a full draw {culprit} \
             ({} vs {} paints, {} vs {} hitboxes); the {kind} frame's damage was {damage:?}. \
             A view outside what it redrew read state that changed without a notify.",
            partial.paint.len(),
            full.paint.len(),
            partial.hitboxes.len(),
            full.hitboxes.len(),
        );
        log::error!("{message}");
        #[cfg(any(test, feature = "test-support"))]
        panic!("{message}");
    }

    fn snapshot(&self, redrawn: &[EntityId], layers: &FxHashSet<RepaintId>) -> FrameSnapshot {
        let frame = &self.rendered_frame;
        let animated = self
            .retained_paints
            .keys()
            .copied()
            .collect::<FxHashSet<_>>();
        let excluded = redrawn
            .iter()
            .filter_map(|view| {
                let range = &self.view_state(*view)?.prepaint_range;
                Some(range.start.hitboxes_index..range.end.hitboxes_index)
            })
            .collect::<Vec<Range<usize>>>();
        let hitboxes = frame
            .hitboxes
            .iter()
            .enumerate()
            .filter(|(index, hitbox)| {
                hitbox.id != HitboxId::TOMBSTONE
                    && !excluded.iter().any(|range| range.contains(index))
            })
            .map(|(_, hitbox)| (hitbox.bounds, hitbox.content_mask, hitbox.behavior))
            .collect();
        let views = frame
            .cached_views
            .keys()
            .filter(|view| !redrawn.contains(view))
            .filter_map(|view| {
                let range = &self.view_state(*view)?.paint_range;
                let range = range.start.scene_index..range.end.scene_index;
                Some((*view, frame.scene.flatten(range, layers, &animated)))
            })
            .collect();
        FrameSnapshot {
            paint: frame.scene.flatten(0..frame.scene.len(), layers, &animated),
            hitboxes,
            views,
        }
    }

    fn view_state(&self, view: EntityId) -> Option<&ViewElementState> {
        let entry = self.rendered_frame.cached_views.get(&view)?;
        let key = (entry.global_id.clone(), TypeId::of::<ViewElementState>());
        self.rendered_frame
            .element_states
            .get(&key)?
            .inner
            .downcast_ref::<Option<ViewElementState>>()?
            .as_ref()
    }

    fn view_name(&self, view: EntityId) -> String {
        #[cfg(debug_assertions)]
        if let Some(name) = self.a11y.view_type_names.get(&view) {
            return format!("{name} ({view:?})");
        }
        format!("{view:?}")
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use crate::{
        AnyWindowHandle, AppContext as _, Context, Entity, IntoElement, ParentElement as _, Render,
        RequestFrameOptions, StyleRefinement, Styled as _, TestAppContext, Window, WindowHandle,
        blue, canvas, div, fill, hsla, px, red, size,
    };

    /// Recolors itself every frame by asking for the next one from paint.
    struct Ticker {
        paints: Rc<Cell<usize>>,
    }

    impl Render for Ticker {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let paints = self.paints.clone();
            canvas(
                |_, _, _| {},
                move |bounds, _, window, _| {
                    let paint = paints.get();
                    paints.set(paint + 1);
                    window.paint_quad(fill(bounds, hsla((paint % 10) as f32 / 10., 0.5, 0.5, 1.)));
                    window.request_animation_frame();
                },
            )
            .size_full()
        }
    }

    /// Reads a width it shares with the test, without being notified when
    /// it changes, and holds the ticker.
    struct Panel {
        width: Rc<Cell<f32>>,
        ticker: Entity<Ticker>,
    }

    impl Render for Panel {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div()
                .size_full()
                .flex()
                .child(div().h(px(10.)).w(px(self.width.get())).bg(red()))
                .child(
                    self.ticker
                        .clone()
                        .cached(StyleRefinement::default().size(px(20.))),
                )
        }
    }

    struct Root {
        panel: Entity<Panel>,
        /// Retained paints in the root itself repaint without rendering.
        animate_root: bool,
    }

    impl Render for Root {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let animate = self.animate_root;
            div()
                .size_full()
                .flex()
                .flex_col()
                .child(
                    self.panel
                        .clone()
                        .cached(StyleRefinement::default().w(px(200.)).h(px(40.))),
                )
                .child(
                    canvas(
                        |_, _, _| {},
                        move |bounds, _, window, cx| {
                            if animate {
                                let mut tick = 0;
                                window.paint_animated(
                                    move |window, _| {
                                        tick += 1;
                                        window.paint_quad(fill(
                                            bounds,
                                            hsla((tick % 10) as f32 / 10., 0.5, 0.5, 1.),
                                        ));
                                        true
                                    },
                                    cx,
                                );
                            } else {
                                window.paint_quad(fill(bounds, blue()));
                            }
                        },
                    )
                    .size(px(30.)),
                )
        }
    }

    fn open(
        cx: &mut TestAppContext,
        animate_root: bool,
        ticking: bool,
    ) -> (WindowHandle<Root>, Rc<Cell<f32>>) {
        let width = Rc::new(Cell::new(50.));
        let window = cx.open_window(size(px(400.), px(100.)), {
            let width = width.clone();
            move |_, cx| {
                let ticker = cx.new(|_| Ticker {
                    paints: Rc::new(Cell::new(0)),
                });
                Root {
                    panel: cx.new(|_| Panel { width, ticker }),
                    animate_root,
                }
            }
        });
        cx.run_until_parked();
        window
            .update(cx, |_, window, _| {
                window.active.set(true);
                window.set_verify_frames(true);
                if !ticking {
                    // Leave only the root's retained paint animating.
                    window.next_frame_callbacks.borrow_mut().clear();
                }
            })
            .unwrap();
        (window, width)
    }

    fn frame(cx: &mut TestAppContext, window: impl Into<AnyWindowHandle>) {
        cx.test_window(window.into())
            .simulate_frame_request(RequestFrameOptions::default());
        cx.run_until_parked();
    }

    #[crate::test]
    fn contained_and_in_place_frames_match_a_full_draw(cx: &mut TestAppContext) {
        let (window, _) = open(cx, false, true);
        // Contained, then in place, each checked against a full draw.
        for _ in 0..4 {
            frame(cx, window);
        }
    }

    #[crate::test]
    fn repaint_frames_match_a_full_draw(cx: &mut TestAppContext) {
        let (window, _) = open(cx, true, false);
        for _ in 0..3 {
            frame(cx, window);
        }
    }

    #[crate::test]
    #[should_panic(expected = "Panel")]
    fn verification_names_a_view_that_read_state_changed_without_a_notify(cx: &mut TestAppContext) {
        let (window, width) = open(cx, false, true);
        for _ in 0..2 {
            frame(cx, window);
        }
        // The panel would show this only once something renders it again.
        width.set(90.);
        frame(cx, window);
    }

    #[crate::test]
    fn without_verification_stale_state_stays_stale(cx: &mut TestAppContext) {
        let (window, width) = open(cx, false, true);
        window
            .update(cx, |_, window, _| window.set_verify_frames(false))
            .unwrap();
        for _ in 0..2 {
            frame(cx, window);
        }
        // Stale, but unverified frames don't look.
        width.set(90.);
        frame(cx, window);
        let bar_width = window
            .update(cx, |_, window, _| {
                window
                    .painted_quads()
                    .iter()
                    .filter(|quad| quad.background.solid == red().into())
                    .map(|quad| quad.bounds.size.width.0)
                    .find(|width| *width > 0.)
            })
            .unwrap();
        assert_eq!(bar_width, Some(100.), "the stale bar still shows");
    }
}
