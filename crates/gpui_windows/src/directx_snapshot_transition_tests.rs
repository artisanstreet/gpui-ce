use super::path_pixel_tests::renderer;
use super::{DirectXRenderer, Repaint};
use anyhow::Result;
use gpui::{
    Bounds, ContentMask, Quad, ScaledPixels, Scene, SnapshotTransitionFrame,
    WindowBackgroundAppearance, hsla, point, size,
};

const VIEWPORT: [u32; 2] = [96, 64];
const BACKGROUND: WindowBackgroundAppearance = WindowBackgroundAppearance::Opaque;

/// A window filled with one colour.
fn solid(generation: u64, hue: f32) -> Scene {
    let full = Bounds::new(
        point(ScaledPixels(0.0), ScaledPixels(0.0)),
        size(
            ScaledPixels(VIEWPORT[0] as f32),
            ScaledPixels(VIEWPORT[1] as f32),
        ),
    );
    let mut scene = Scene::default();
    scene.insert_primitive(Quad {
        bounds: full,
        content_mask: ContentMask { bounds: full },
        background: hsla(hue, 1.0, 0.5, 1.0).into(),
        ..Default::default()
    });
    scene.finish();
    scene.generation = generation;
    scene
}

fn frame(progress: f32, shimmer_alpha: f32) -> SnapshotTransitionFrame {
    SnapshotTransitionFrame {
        progress,
        band: 0.18,
        shimmer: [1.0, 1.0, 1.0, shimmer_alpha],
    }
}

/// Renders `scene` whole and composites its transition frame, as a present would, and reads the
/// back buffer instead of presenting it.
fn composited(renderer: &mut DirectXRenderer, scene: &Scene) -> Result<image::RgbaImage> {
    renderer.render(scene, BACKGROUND, &Repaint::Full)?;
    if let Some(frame) = &scene.snapshot_transition {
        renderer.composite_snapshot(frame)?;
    }
    renderer.read_render_target()
}

fn near(actual: &image::Rgba<u8>, expected: &image::Rgba<u8>) -> bool {
    actual
        .0
        .iter()
        .zip(expected.0)
        .all(|(a, e)| a.abs_diff(e) <= 2)
}

#[test]
fn the_old_frame_sweeps_away_from_the_top_left_corner() -> Result<()> {
    let mut renderer = renderer(VIEWPORT)?;
    let a = solid(1, 0.0);
    let b = solid(2, 0.6);
    let red = *renderer.render_to_image(&a, BACKGROUND)?.get_pixel(0, 0);
    let blue = *renderer.render_to_image(&b, BACKGROUND)?.get_pixel(0, 0);
    assert!(!near(&red, &blue));

    renderer.draw(&a, BACKGROUND)?;
    assert!(renderer.begin_snapshot_transition(&a, BACKGROUND)?);
    let (right, bottom) = (VIEWPORT[0] - 1, VIEWPORT[1] - 1);

    // At the start the snapshot still covers the whole window.
    let mut next = solid(2, 0.6);
    next.snapshot_transition = Some(frame(0.0, 0.0));
    let start = composited(&mut renderer, &next)?;
    assert!(
        near(start.get_pixel(0, 0), &red),
        "{:?}",
        start.get_pixel(0, 0)
    );
    assert!(near(start.get_pixel(right, bottom), &red));

    // Halfway the new frame shows top-left and the old one bottom-right.
    next.snapshot_transition = Some(frame(0.5, 0.0));
    let half = composited(&mut renderer, &next)?;
    assert!(
        near(half.get_pixel(0, 0), &blue),
        "{:?}",
        half.get_pixel(0, 0)
    );
    assert!(
        near(half.get_pixel(right, bottom), &red),
        "{:?}",
        half.get_pixel(right, bottom)
    );

    // The shimmer lightens the edge, which halfway runs through the centre.
    next.snapshot_transition = Some(frame(0.5, 0.6));
    let shimmering = composited(&mut renderer, &next)?;
    let edge = shimmering.get_pixel(VIEWPORT[0] / 2, VIEWPORT[1] / 2);
    let lightness = |pixel: &image::Rgba<u8>| pixel.0[..3].iter().map(|c| *c as u32).sum::<u32>();
    assert!(lightness(edge) > lightness(&red) + 150, "{edge:?}");
    assert!(near(shimmering.get_pixel(0, 0), &blue));

    // Presents during the transition succeed, and the first without a frame drops the snapshot.
    renderer.draw(&next, BACKGROUND)?;
    assert!(renderer.snapshot.is_some());
    next.snapshot_transition = None;
    renderer.draw(&next, BACKGROUND)?;
    assert!(renderer.snapshot.is_none());
    let done = composited(&mut renderer, &next)?;
    assert!(done.pixels().all(|pixel| near(pixel, &blue)));
    Ok(())
}

#[test]
fn resizing_ends_the_transition() -> Result<()> {
    let mut renderer = renderer(VIEWPORT)?;
    let a = solid(1, 0.0);
    renderer.draw(&a, BACKGROUND)?;
    assert!(renderer.begin_snapshot_transition(&a, BACKGROUND)?);
    renderer.resize(gpui::size(gpui::DevicePixels(80), gpui::DevicePixels(48)))?;
    assert!(renderer.snapshot.is_none());
    // A frame still carrying the transition presents as usual.
    let mut next = solid(2, 0.6);
    next.snapshot_transition = Some(frame(0.5, 0.5));
    renderer.draw(&next, BACKGROUND)
}
