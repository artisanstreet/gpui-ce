use std::time::Duration;

use crate::{
    AnyElement, App, Bounds, Element, ElementId, GlobalElementId, InspectorElementId, IntoElement,
    LayoutId, Pixels, QuadPulse, Window, pulse_clock_ms,
};

/// Breathes the opacity of an element's quads on the GPU.
///
/// Unlike [`AnimationExt::with_animation`](crate::AnimationExt::with_animation),
/// a pulse never re-renders its view: the window re-presents the rendered
/// scene each frame and the renderer evaluates the opacity. Use it for
/// repeating decorative motion that keeps its layout, like a working dot.
/// Only quads (backgrounds and borders) pulse; text and images hold still.
pub trait PulseExt: IntoElement + Sized {
    /// Pulses this element's quads from full opacity down to `min_opacity` and back
    /// once every `period`, starting at full opacity when it first appears.
    fn with_pulse(
        self,
        id: impl Into<ElementId>,
        period: Duration,
        min_opacity: f32,
    ) -> PulseElement<Self> {
        PulseElement {
            id: id.into(),
            element: Some(self),
            period_ms: u32::try_from(period.as_millis()).unwrap_or(u32::MAX),
            min_opacity: min_opacity.clamp(0.0, 1.0),
        }
    }
}

impl<E: IntoElement> PulseExt for E {}

/// An element whose quads breathe on the GPU. See [`PulseExt::with_pulse`].
pub struct PulseElement<E> {
    id: ElementId,
    element: Option<E>,
    period_ms: u32,
    min_opacity: f32,
}

impl<E: IntoElement + 'static> IntoElement for PulseElement<E> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl<E: IntoElement + 'static> Element for PulseElement<E> {
    type RequestLayoutState = (AnyElement, QuadPulse);
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        Some(self.id.clone())
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        // The start survives re-renders, so the breath never restarts mid-cycle.
        let start_ms = window.with_element_state(global_id.unwrap(), |start, _| {
            let start = start.unwrap_or_else(pulse_clock_ms);
            (start, start)
        });
        let pulse = if cx.reduce_motion() || self.period_ms == 0 {
            QuadPulse::default()
        } else {
            QuadPulse {
                start_ms,
                period_ms: self.period_ms,
                min_opacity: self.min_opacity,
                pad: 0,
            }
        };
        let mut element = self
            .element
            .take()
            .expect("should only be called once")
            .into_any_element();
        (element.request_layout(window, cx), (element, pulse))
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        (element, _): &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        element.prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        (element, pulse): &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        window.with_quad_pulse(*pulse, |window| element.paint(window, cx));
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::{
        Context, IntoElement, ParentElement, PulseExt, Render, Styled, TestAppContext, Window, div,
        px, red,
    };

    struct Dot;

    impl Render for Dot {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div()
                .size(px(100.0))
                .child(div().size(px(6.0)).bg(red()).with_pulse(
                    "dot",
                    Duration::from_millis(1600),
                    0.4,
                ))
        }
    }

    #[crate::test]
    fn a_pulsing_dot_animates_without_redrawing(cx: &mut TestAppContext) {
        let (_, cx) = cx.add_window_view(|_, _| Dot);
        cx.run_until_parked();
        let dirty = cx.update(|window, _| {
            let pulses: Vec<_> = window
                .rendered_frame
                .scene
                .quads
                .iter()
                .map(|quad| quad.pulse)
                .filter(|pulse| pulse.is_active())
                .collect();
            assert_eq!(pulses.len(), 1, "the dot's quad carries the pulse");
            assert_eq!(pulses[0].period_ms, 1600);
            assert_eq!(pulses[0].min_opacity, 0.4);
            assert!(window.rendered_scene_pulses);
            window.invalidator.is_dirty()
        });
        assert!(!dirty, "a pulse must not dirty the window to animate");
    }
}
