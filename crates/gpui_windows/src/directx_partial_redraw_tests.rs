use super::path_pixel_tests::renderer;
use super::{DamageRect, DamageSources, DamageTracker, FrameDamage, Repaint, batch_bounds};
use anyhow::Result;
use gpui::{
    AtlasTextureId, AtlasTextureKind, AtlasTile, BackdropFilter, Bounds, ContentMask, Corners,
    FilterBoundary, PaddedBool32, PolychromeSprite, PrimitiveRanges, Quad, QuadPulse, ScaledFilter,
    ScaledPixels, Scene, Shadow, TileId, WindowBackgroundAppearance, hsla, point, size,
};

const VIEWPORT: [u32; 2] = [96, 96];

fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(x), ScaledPixels(y)),
        size(ScaledPixels(width), ScaledPixels(height)),
    )
}

/// A dark window with a coloured square at `changed`; with `glass`, a glass panel blurs the
/// square's surroundings, and with `group`, a blurred group sits in a far corner. `extra` quads
/// go on top.
fn scene(
    generation: u64,
    hue: f32,
    changed: Bounds<ScaledPixels>,
    glass: bool,
    group: bool,
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
    if glass {
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
    }
    if group {
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

/// Plans presenting `scenes` in turn. Without a flip the back buffer keeps the previous render,
/// so a single-buffer tracker plans exactly what the renderer must repaint over it.
fn plan_presents(scenes: &[&Scene]) -> Vec<Repaint> {
    let mut tracker = DamageTracker::new(1);
    let mut sources = DamageSources::default();
    scenes
        .iter()
        .map(|scene| {
            sources.update(scene);
            let plan = tracker.plan(&FrameDamage {
                generation: scene.generation,
                scene: sources.scene.as_deref(),
                pulses: &sources.pulses,
                filters: &sources.filters,
                previous_filters: &sources.previous_filters,
                width: VIEWPORT[0] as i32,
                height: VIEWPORT[1] as i32,
            });
            let repaint = plan.repaint.clone();
            tracker.presented(plan);
            repaint
        })
        .collect()
}

/// Renders `before` whole, then `after` as `repaint`, and checks the result against rendering
/// `after` whole.
fn assert_partial_matches_full(before: &Scene, after: &Scene, repaint: &Repaint) -> Result<()> {
    let background = WindowBackgroundAppearance::Transparent;
    let mut renderer = renderer(VIEWPORT)?;
    renderer.render_to_image(before, background)?;
    renderer.render(after, background, repaint)?;
    let partial = renderer.read_render_target()?;
    let full = renderer.render_to_image(after, background)?;
    for (x, y, pixel) in full.enumerate_pixels() {
        assert_eq!(
            partial.get_pixel(x, y),
            pixel,
            "partial repaint differs at ({x}, {y}), repaint={repaint:?}"
        );
    }
    Ok(())
}

#[test]
fn a_partial_repaint_over_the_previous_frame_matches_a_full_render() -> Result<()> {
    for filters in [false, true] {
        // A second change far from the first, like a sidebar dot beside a transcript line.
        let before = scene(
            1,
            0.0,
            rect(30.0, 30.0, 12.0, 12.0),
            filters,
            filters,
            &[corner(0.5)],
        );
        let moved = rect(33.0, 31.0, 12.0, 12.0);
        let mut after = scene(2, 0.33, moved, filters, filters, &[corner(0.8)]);
        after.damage = Some(vec![
            rect(30.0, 30.0, 15.0, 13.0),
            rect(88.0, 84.0, 6.0, 6.0),
        ]);

        let [first, repaint] = &plan_presents(&[&before, &after])[..] else {
            unreachable!();
        };
        assert_eq!(*first, Repaint::Full);
        let Repaint::Partial(regions) = repaint else {
            panic!("expected a partial repaint, got {repaint:?}");
        };
        assert_eq!(regions.len(), 2, "{regions:?}");
        if filters {
            // The square sits under the glass panel's footprint, which repaints whole.
            assert!(
                regions[0].left <= 36 - 12 && regions[0].right >= 66 + 12,
                "{regions:?}"
            );
        }

        assert_partial_matches_full(&before, &after, repaint)?;
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
    let mut scene = scene(1, 0.0, rect(30.0, 30.0, 12.0, 12.0), true, true, &[pulsing]);
    // Every buffer of the flip chain is drawn whole once, then only the pulse.
    for frame in 0..6u32 {
        scene.pulse_clock_ms = frame * 16;
        renderer.draw(&scene, background)?;
    }
    // A present that failed or reported a status would have forgotten the buffers' contents.
    assert_eq!(
        renderer.plan_frame(&scene, background).repaint,
        Repaint::Partial(vec![DamageRect {
            left: 70,
            top: 74,
            right: 74,
            bottom: 82,
        }])
    );
    Ok(())
}

#[test]
fn removing_a_blurred_group_repaints_the_blur_it_spread() -> Result<()> {
    let square = rect(30.0, 30.0, 12.0, 12.0);
    let before = scene(1, 0.0, square, false, true, &[]);
    let mut after = scene(2, 0.0, square, false, false, &[]);
    // What the window reports: the removed group and its child, padded by a pixel. The blur
    // drew up to 3 radii past the group's bounds.
    after.damage = Some(vec![rect(3.0, 79.0, 12.0, 12.0)]);
    let [_, repaint] = &plan_presents(&[&before, &after])[..] else {
        unreachable!();
    };
    assert!(matches!(repaint, Repaint::Partial(_)), "{repaint:?}");
    assert_partial_matches_full(&before, &after, repaint)
}

/// A small quad in the bottom-right corner, drawn last.
fn corner(hue: f32) -> Quad {
    Quad {
        order: 7,
        bounds: rect(88.0, 84.0, 6.0, 6.0),
        content_mask: ContentMask {
            bounds: rect(0.0, 0.0, 96.0, 96.0),
        },
        background: hsla(hue, 0.8, 0.5, 1.0).into(),
        ..Default::default()
    }
}

/// Renders `before` whole, then `after` whole over the instance buffers `before` left.
fn render_over(before: &Scene, after: &Scene) -> Result<image::RgbaImage> {
    let background = WindowBackgroundAppearance::Transparent;
    let mut renderer = renderer(VIEWPORT)?;
    renderer.render(before, background, &Repaint::Full)?;
    renderer.render(after, background, &Repaint::Full)?;
    renderer.read_render_target()
}

fn render_fresh(scene: &Scene) -> Result<image::RgbaImage> {
    renderer(VIEWPORT)?.render_to_image(scene, WindowBackgroundAppearance::Transparent)
}

#[test]
fn changed_ranges_upload_only_their_instances() -> Result<()> {
    let square = rect(30.0, 30.0, 12.0, 12.0);
    // Quads sort to [background, square, corner].
    let before = scene(1, 0.0, square, false, false, &[corner(0.5)]);
    let mut after = scene(2, 0.33, square, false, false, &[corner(0.8)]);

    after.changed_primitives = Some(PrimitiveRanges {
        quads: 1..3,
        ..Default::default()
    });
    assert!(render_over(&before, &after)? == render_fresh(&after)?);

    // Claiming only the square changed leaves the corner as uploaded before, which shows that
    // nothing outside the range was written.
    after.changed_primitives = Some(PrimitiveRanges {
        quads: 1..2,
        ..Default::default()
    });
    let stale_corner = scene(2, 0.33, square, false, false, &[corner(0.5)]);
    assert!(render_over(&before, &after)? == render_fresh(&stale_corner)?);
    Ok(())
}

#[test]
fn a_vector_outgrowing_its_buffer_uploads_whole() -> Result<()> {
    let square = rect(30.0, 30.0, 12.0, 12.0);
    let before = scene(1, 0.0, square, false, false, &[corner(0.5)]);
    // Past the quad buffer's initial 64 instances, so it regrows and starts empty.
    let dots: Vec<Quad> = (0..100)
        .map(|ix| Quad {
            order: 8 + ix,
            bounds: rect(
                (ix % 10) as f32 * 9.0,
                60.0 + (ix / 10) as f32 * 3.0,
                2.0,
                2.0,
            ),
            content_mask: ContentMask {
                bounds: rect(0.0, 0.0, 96.0, 96.0),
            },
            background: hsla(ix as f32 / 100.0, 0.8, 0.5, 1.0).into(),
            ..Default::default()
        })
        .collect();
    let mut after = scene(
        2,
        0.0,
        square,
        false,
        false,
        &[[corner(0.5)].as_slice(), &dots].concat(),
    );
    after.changed_primitives = Some(PrimitiveRanges {
        quads: 3..after.quads.len(),
        ..Default::default()
    });
    assert!(render_over(&before, &after)? == render_fresh(&after)?);
    Ok(())
}

#[test]
fn a_buffer_left_from_a_busy_scene_shrinks_and_still_draws_the_next_one() -> Result<()> {
    let background = WindowBackgroundAppearance::Transparent;
    let square = rect(30.0, 30.0, 12.0, 12.0);
    let dots: Vec<Quad> = (0..1000)
        .map(|ix| Quad {
            order: 8 + ix,
            bounds: rect((ix % 40) as f32 * 2.0, (ix / 40) as f32 * 2.0, 1.0, 1.0),
            content_mask: ContentMask {
                bounds: rect(0.0, 0.0, 96.0, 96.0),
            },
            background: hsla(ix as f32 / 1000.0, 0.8, 0.5, 1.0).into(),
            ..Default::default()
        })
        .collect();
    let busy = scene(1, 0.0, square, false, false, &dots);
    let calm = scene(2, 0.33, square, false, false, &[corner(0.8)]);
    let mut renderer = renderer(VIEWPORT)?;
    let backdate = |renderer: &mut super::DirectXRenderer| {
        renderer.pipelines.quad_pipeline.peak_since =
            std::time::Instant::now() - super::INSTANCE_BUFFER_SHRINK_AFTER;
    };
    renderer.render(&busy, background, &Repaint::Full)?;
    assert_eq!(renderer.pipelines.quad_pipeline.buffer_size, 1024);

    // The window that just closed saw the busy scene, so the buffer keeps its size.
    backdate(&mut renderer);
    renderer.render(&calm, background, &Repaint::Full)?;
    assert_eq!(renderer.pipelines.quad_pipeline.buffer_size, 1024);

    // A whole window of calm frames shrinks it. The new buffer starts empty, so this frame must
    // upload every quad even though it claims only the square changed.
    backdate(&mut renderer);
    let mut calm_again = scene(3, 0.66, square, false, false, &[corner(0.8)]);
    calm_again.changed_primitives = Some(PrimitiveRanges {
        quads: 1..2,
        ..Default::default()
    });
    renderer.render(&calm_again, background, &Repaint::Full)?;
    assert_eq!(renderer.pipelines.quad_pipeline.buffer_size, 64);
    assert!(renderer.read_render_target()? == render_fresh(&calm_again)?);
    Ok(())
}

#[test]
fn slack_instances_widen_no_batch() {
    let full = ContentMask {
        bounds: rect(0.0, 0.0, 96.0, 96.0),
    };
    let mut scene = Scene::default();
    let shadow = Shadow {
        order: 0,
        blur_radius: ScaledPixels(2.0),
        bounds: rect(40.0, 40.0, 10.0, 10.0),
        corner_radii: Corners::default(),
        content_mask: full,
        color: hsla(0.0, 0.0, 0.0, 0.5).into(),
        element_bounds: rect(40.0, 40.0, 10.0, 10.0),
        element_corner_radii: Corners::default(),
        inset: 0,
        pad: 0,
    };
    scene.insert_primitive(shadow);
    // Live layers reserve slots as primitives with empty bounds and masks.
    scene.insert_primitive(Shadow {
        order: 1,
        blur_radius: ScaledPixels(0.0),
        bounds: Bounds::default(),
        content_mask: ContentMask::default(),
        element_bounds: Bounds::default(),
        ..shadow
    });
    scene.insert_primitive(Quad {
        order: 2,
        bounds: rect(10.0, 12.0, 4.0, 4.0),
        content_mask: full,
        ..Default::default()
    });
    scene.insert_primitive(Quad {
        order: 3,
        ..Default::default()
    });
    scene.finish();
    let bounds: Vec<_> = scene
        .batches()
        .map(|batch| batch_bounds(&scene, batch))
        .collect();
    assert_eq!(
        bounds,
        [
            // The blur's gaussian tail reaches 3 radii out.
            Some(DamageRect {
                left: 34,
                top: 34,
                right: 56,
                bottom: 56,
            }),
            Some(DamageRect {
                left: 10,
                top: 12,
                right: 14,
                bottom: 16,
            }),
        ]
    );
}

#[test]
fn slack_sprites_naming_a_freed_atlas_texture_draw_nothing() -> Result<()> {
    let square = rect(30.0, 30.0, 12.0, 12.0);
    let plain = scene(1, 0.0, square, false, false, &[]);
    let mut slack = scene(1, 0.0, square, false, false, &[]);
    // A slot left by an image whose texture the atlas has since released: this renderer's atlas
    // holds no polychrome texture at all.
    slack.insert_primitive(PolychromeSprite {
        order: 9,
        pad: 0,
        grayscale: PaddedBool32::from(false),
        opacity: 1.0,
        bounds: Bounds::default(),
        content_mask: ContentMask::default(),
        corner_radii: Corners::default(),
        tile: AtlasTile {
            texture_id: AtlasTextureId {
                index: 3,
                kind: AtlasTextureKind::Polychrome,
            },
            tile_id: TileId(0),
            padding: 0,
            bounds: Bounds::default(),
        },
    });
    slack.finish();
    assert!(render_fresh(&slack)? == render_fresh(&plain)?);
    Ok(())
}
