use super::path_pixel_tests::renderer;
use anyhow::Result;
use gpui::{
    BackdropFilter, Bounds, ContentMask, Corners, FilterBoundary, Quad, ScaledFilter, ScaledPixels,
    Scene, WindowBackgroundAppearance, hsla, point, size,
};

fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(x), ScaledPixels(y)),
        size(ScaledPixels(width), ScaledPixels(height)),
    )
}

fn edge_scene(viewport: [u32; 2], horizontal: bool, hue: f32) -> Scene {
    let full = rect(0.0, 0.0, viewport[0] as f32, viewport[1] as f32);
    let mask = ContentMask { bounds: full };
    let mut scene = Scene::default();
    scene.insert_primitive(Quad {
        bounds: full,
        content_mask: mask,
        background: hsla(0.0, 0.0, 0.1, 1.0).into(),
        ..Default::default()
    });
    scene.insert_primitive(Quad {
        order: 1,
        bounds: if horizontal {
            rect(viewport[0] as f32 - 20.0, 0.0, 20.0, viewport[1] as f32)
        } else {
            rect(0.0, viewport[1] as f32 - 20.0, viewport[0] as f32, 20.0)
        },
        content_mask: mask,
        background: hsla(hue, 1.0, 0.5, 1.0).into(),
        ..Default::default()
    });
    scene.insert_primitive(BackdropFilter {
        order: 2,
        bounds: if horizontal {
            rect(-5.0, 20.0, 24.0, 24.0)
        } else {
            rect(20.0, -5.0, 24.0, 24.0)
        },
        content_mask: mask,
        corner_radii: Corners::all(ScaledPixels(3.0)),
        filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(5.3))],
        opacity: 0.8,
    });
    scene.finish();
    scene
}

#[test]
fn backdrop_blur_does_not_wrap_pixels_from_the_opposite_window_edge() -> Result<()> {
    for viewport in [[128, 96], [173, 117]] {
        let mut renderer = renderer(viewport)?;
        for horizontal in [false, true] {
            let red = renderer.render_to_image(
                &edge_scene(viewport, horizontal, 0.0),
                WindowBackgroundAppearance::Transparent,
            )?;
            let green = renderer.render_to_image(
                &edge_scene(viewport, horizontal, 0.33),
                WindowBackgroundAppearance::Transparent,
            )?;
            for y in 0..50 {
                for x in 0..50 {
                    assert_eq!(
                        red.get_pixel(x, y),
                        green.get_pixel(x, y),
                        "distant pixels wrapped into blur at ({x}, {y}), viewport={viewport:?}, horizontal={horizontal}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn content_blur_preserves_wgpu_pass_footprint_and_restores_rasterizer_state() -> Result<()> {
    use windows::Win32::Foundation::RECT;
    let mut renderer = renderer([80, 80])?;
    let full = rect(0.0, 0.0, 80.0, 80.0);
    let mask = ContentMask { bounds: full };
    let boundary = FilterBoundary {
        order: 0,
        bounds: rect(24.0, 24.0, 24.0, 24.0),
        content_mask: mask,
        corner_radii: Corners::default(),
        filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(4.0))],
        opacity: 1.0,
        is_start: true,
    };
    let mut scene = Scene::default();
    scene.insert_primitive(FilterBoundary {
        order: 0,
        is_start: true,
        ..boundary.clone()
    });
    // Overflow content deliberately covers the whole group. WGPU clears the
    // scratch outside its documented bounds.dilate(3*radius) pass footprint.
    scene.insert_primitive(Quad {
        order: 1,
        bounds: full,
        content_mask: mask,
        background: hsla(0.2, 0.7, 0.5, 1.0).into(),
        ..Default::default()
    });
    scene.insert_primitive(FilterBoundary {
        order: 2,
        is_start: false,
        ..boundary
    });
    scene.insert_primitive(Quad {
        order: 3,
        bounds: rect(1.0, 70.0, 6.0, 6.0),
        content_mask: mask,
        background: hsla(0.8, 0.7, 0.5, 1.0).into(),
        ..Default::default()
    });
    scene.finish();
    let context = renderer.devices.as_ref().unwrap().device_context.clone();
    let previous_state = unsafe { context.RSGetState()? };
    let previous_scissor = RECT {
        left: 2,
        top: 3,
        right: 7,
        bottom: 11,
    };
    unsafe {
        context.RSSetScissorRects(Some(&[previous_scissor]));
    }
    let image = renderer.render_to_image(&scene, WindowBackgroundAppearance::Transparent)?;
    assert_eq!(image.get_pixel(36, 36)[3], 255);
    assert_eq!(image.get_pixel(11, 36)[3], 0);
    assert!(
        (1..255).contains(&image.get_pixel(13, 13)[3]),
        "content blur lost the WGPU pass-edge fade"
    );
    assert_eq!(
        image.get_pixel(3, 72)[3],
        255,
        "blur scissor leaked into the following quad"
    );
    assert_eq!(unsafe { context.RSGetState()? }, previous_state);
    let mut restored = [RECT::default(); 16];
    let mut count = 16;
    unsafe {
        context.RSGetScissorRects(&mut count, Some(restored.as_mut_ptr()));
    }
    assert_eq!(count, 1);
    assert_eq!(restored[0], previous_scissor);
    Ok(())
}
