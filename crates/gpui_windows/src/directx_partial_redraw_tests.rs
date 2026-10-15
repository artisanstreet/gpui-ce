use super::path_pixel_tests::renderer;
use super::{DamageRect, DamageSources, DamageTracker, FrameDamage, Repaint};
use anyhow::Result;
use gpui::{
    BackdropFilter, Bounds, ContentMask, Corners, FilterBoundary, Quad, QuadPulse, ScaledFilter,
    ScaledPixels, Scene, WindowBackgroundAppearance, hsla, point, size,
};

const VIEWPORT: [u32; 2] = [96, 96];

fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(x), ScaledPixels(y)),
        size(ScaledPixels(width), ScaledPixels(height)),
    )
}

/// A dark window with a coloured square at `changed`; with `filters`, a glass panel blurs the
/// square's surroundings and a blurred group sits in a far corner. `extra` quads go on top.
fn scene(
    generation: u64,
    hue: f32,
    changed: Bounds<ScaledPixels>,
    filters: bool,
    extra: &[Quad],
) -> Scene {
    let full = rect(0.0, 0.0, VIEWPORT[0] as f32, VIEWPORT[1] as f32);
    let mask = ContentMask { bounds: full };
    let mut scene = Scene::default();
    scene.insert_primitive(Quad {
        bounds: full,
        content_mask: mask,
        background: hsla(0.6, 0.3, 0.15, 1.0).into(),
        ..Default::default()
    });
    scene.insert_primitive(Quad {
        order: 1,
        bounds: changed,
        content_mask: mask,
        background: hsla(hue, 1.0, 0.5, 1.0).into(),
        ..Default::default()
    });
    if filters {
        scene.insert_primitive(BackdropFilter {
            order: 2,
            bounds: rect(36.0, 20.0, 30.0, 30.0),
            content_mask: mask,
            corner_radii: Corners::all(ScaledPixels(4.0)),
            filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(4.0))],
            opacity: 1.0,
        });
        // Drawn over the glass: a partial repaint must not blur it back into the panel.
        scene.insert_primitive(Quad {
            order: 3,
            bounds: rect(44.0, 28.0, 10.0, 10.0),
            content_mask: mask,
            background: hsla(0.1, 0.9, 0.8, 0.7).into(),
            ..Default::default()
        });
        let boundary = FilterBoundary {
            order: 4,
            bounds: rect(4.0, 80.0, 10.0, 10.0),
            content_mask: mask,
            corner_radii: Corners::default(),
            filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(2.0))],
            opacity: 1.0,
            is_start: true,
        };
        scene.insert_primitive(boundary.clone());
        scene.insert_primitive(Quad {
            order: 5,
            bounds: rect(5.0, 81.0, 8.0, 8.0),
            content_mask: mask,
            background: hsla(0.3, 0.8, 0.6, 1.0).into(),
            ..Default::default()
        });
        scene.insert_primitive(FilterBoundary {
            order: 6,
            is_start: false,
            ..boundary
        });
    }
    for quad in extra {
        scene.insert_primitive(*quad);
    }
    scene.finish();
    scene.generation = generation;
    scene
}

#[test]
fn a_partial_repaint_over_the_previous_frame_matches_a_full_render() -> Result<()> {
    let background = WindowBackgroundAppearance::Transparent;
    for filters in [false, true] {
        let mut renderer = renderer(VIEWPORT)?;
        let before = scene(1, 0.0, rect(30.0, 30.0, 12.0, 12.0), filters, &[]);
        let moved = rect(33.0, 31.0, 12.0, 12.0);
        let mut after = scene(2, 0.33, moved, filters, &[]);
        after.damage = Some(vec![rect(30.0, 30.0, 15.0, 13.0)]);

        // Without a flip the back buffer keeps the previous render, so a single-buffer
        // tracker plans exactly what the renderer must repaint.
        let mut tracker = DamageTracker::new(1);
        let mut sources = DamageSources::default();
        let mut plan = |scene: &Scene| {
            sources.update(scene);
            let plan = tracker.plan(&FrameDamage {
                generation: scene.generation,
                scene: sources.scene.as_deref(),
                pulses: &sources.pulses,
                filters: &sources.filters,
                width: VIEWPORT[0] as i32,
                height: VIEWPORT[1] as i32,
            });
            tracker.presented(&plan);
            plan.repaint
        };
        assert_eq!(plan(&before), Repaint::Full);
        let repaint = plan(&after);
        let Repaint::Partial(region) = repaint else {
            panic!("expected a partial repaint, got {repaint:?}");
        };
        if filters {
            // The square sits under the glass panel's footprint, which repaints whole.
            assert!(
                region.left <= 36 - 12 && region.right >= 66 + 12,
                "{region:?}"
            );
        }

        renderer.render_to_image(&before, background)?;
        renderer.render(&after, background, repaint)?;
        let partial = renderer.read_render_target()?;
        let full = renderer.render_to_image(&after, background)?;
        for (x, y, pixel) in full.enumerate_pixels() {
            assert_eq!(
                partial.get_pixel(x, y),
                pixel,
                "partial repaint differs at ({x}, {y}), filters={filters}, region={region:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn pulse_only_presents_flip_with_the_pulse_as_their_dirty_rect() -> Result<()> {
    let background = WindowBackgroundAppearance::Transparent;
    let mut renderer = renderer(VIEWPORT)?;
    let pulsing = Quad {
        order: 7,
        bounds: rect(70.0, 74.0, 8.0, 8.0),
        content_mask: ContentMask {
            bounds: rect(0.0, 0.0, 74.0, 96.0),
        },
        background: hsla(0.9, 1.0, 0.6, 1.0).into(),
        pulse: QuadPulse {
            start_ms: 0,
            period_ms: 1200,
            min_opacity: 0.2,
            pad: 0,
        },
        ..Default::default()
    };
    let mut scene = scene(1, 0.0, rect(30.0, 30.0, 12.0, 12.0), true, &[pulsing]);
    // Every buffer of the flip chain is drawn whole once, then only the pulse.
    for frame in 0..6u32 {
        scene.pulse_clock_ms = frame * 16;
        renderer.draw(&scene, background)?;
    }
    // A present that failed or reported a status would have forgotten the buffers' contents.
    assert_eq!(
        renderer.plan_frame(&scene, background).repaint,
        Repaint::Partial(DamageRect {
            left: 70,
            top: 74,
            right: 74,
            bottom: 82,
        })
    );
    Ok(())
}
