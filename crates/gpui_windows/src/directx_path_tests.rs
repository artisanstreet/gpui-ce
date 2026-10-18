use super::{
    BUFFER_COUNT, DamageSources, DamageTracker, DirectXAtlas, DirectXGlobalElements,
    DirectXRenderPipelines, DirectXRenderer, DirectXResources, FontInfo,
    resource_tests::test_devices,
};
use anyhow::{Context, Result};
use gpui::path_targets::path_batch_bounds;
use gpui::{
    Background, Bounds, ContentMask, DevicePixels, Path, ScaledPixels, Scene, hsla,
    linear_color_stop, linear_gradient, pattern_slash, point, px, size,
};
use std::sync::Arc;
use windows::Win32::{Foundation::HWND, Graphics::Direct3D11::*};

static FONT_INFO: FontInfo = FontInfo {
    gamma_ratios: [1.0; 4],
    grayscale_enhanced_contrast: 0.0,
    subpixel_enhanced_contrast: 0.0,
    is_bgr: false,
};

pub(super) fn renderer(viewport: [u32; 2]) -> Result<DirectXRenderer> {
    let devices = test_devices()?;
    let resources =
        DirectXResources::new(&devices, viewport[0], viewport[1], HWND::default(), false)?;
    Ok(DirectXRenderer {
        hwnd: HWND::default(),
        atlas: Arc::new(DirectXAtlas::new(&devices.device, &devices.device_context)),
        globals: DirectXGlobalElements::new(&devices.device)?,
        pipelines: DirectXRenderPipelines::new(&devices.device)?,
        resources: Some(resources),
        devices: Some(devices),
        direct_composition: None,
        font_info: &FONT_INFO,
        width: viewport[0],
        height: viewport[1],
        skip_draws: false,
        active_render_target: None,
        meshes: None,
        damage: DamageTracker::new(BUFFER_COUNT),
        damage_sources: DamageSources::default(),
        uploaded_generation: None,
        drawn_generation: None,
        background_appearance: None,
        snapshot: None,
    })
}

fn render_paths(
    renderer: &mut DirectXRenderer,
    paths: &[Path<ScaledPixels>],
    origin: [u32; 2],
    size: [u32; 2],
) -> Result<Vec<u8>> {
    renderer.pre_draw(&[0.0; 4], 0, None)?;
    let devices = renderer.devices.as_ref().unwrap();
    let resources = renderer.resources.as_mut().unwrap();
    resources.paths = None;
    resources.ensure_paths(&devices.device, Some(size))?;
    renderer.draw_paths_to_intermediate_at(paths, origin, None)?;
    renderer.draw_paths_from_intermediate(paths)?;
    let devices = renderer.devices.as_ref().unwrap();
    let resources = renderer.resources.as_ref().unwrap();
    readback(
        &devices.device,
        &devices.device_context,
        resources.render_target.as_ref().unwrap(),
    )
}

#[test]
fn compact_path_pixels_match_full_target_with_screen_space_effects() -> Result<()> {
    compare_path_pixels(false)
}

#[test]
#[ignore = "vkd3d 2.1 SM4.1/WARP reference; see directx_path_test_data/README.md"]
fn corrected_vkd3d_2_1_full_target_matches_reference() -> Result<()> {
    compare_path_pixels(true)
}

fn compare_path_pixels(check_vkd3d_reference: bool) -> Result<()> {
    for viewport in [[128, 128], [173, 117]] {
        let mut renderer = renderer(viewport)?;
        let gradient = linear_gradient(
            173.0,
            linear_color_stop(hsla(0.72, 0.1, 0.13, 0.9), 0.0),
            linear_color_stop(hsla(0.72, 0.1, 0.095, 0.7), 1.0),
        );
        let pattern = pattern_slash(hsla(0.4, 0.7, 0.6, 0.8), 2.3, 7.1);
        for (name, paths) in [
            ("gradient", vec![curved_path(gradient, 0.0)]),
            ("pattern", vec![curved_path(pattern, 0.0)]),
            (
                "overlap",
                vec![curved_path(gradient, 0.0), curved_path(pattern, 5.0)],
            ),
        ] {
            let [x, y, width, height] = path_batch_bounds(&paths, viewport);
            let compact_size = [width.next_power_of_two(), height.next_power_of_two()];
            assert!(compact_size[0] * compact_size[1] < viewport[0] * viewport[1]);
            let full = render_paths(&mut renderer, &paths, [0, 0], viewport)?;
            let compact = render_paths(&mut renderer, &paths, [x, y], compact_size)?;
            assert!(
                full.chunks_exact(4).filter(|p| p[3] > 0).count() > 200,
                "empty fixture"
            );
            assert!(
                full.chunks_exact(4).any(|p| p[3] > 0 && p[3] < 150),
                "missing alpha/AA"
            );
            assert_eq!(&compact[..4], &[0; 4], "escaped clip");
            let count = full.iter().zip(&compact).filter(|(a, b)| a != b).count();
            let max_delta = full
                .iter()
                .zip(&compact)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap();
            println!("{name} {viewport:?}: {count} changed bytes, maximum delta {max_delta}");
            if let Some(directory) = std::env::var_os("GPUI_NATIVE_PATH_TEST_OUTPUT") {
                let directory = std::path::PathBuf::from(directory);
                for (variant, pixels) in [("full", &full), ("compact", &compact)] {
                    std::fs::write(
                        directory.join(format!(
                            "{name}-{}x{}-{variant}.bgra",
                            viewport[0], viewport[1]
                        )),
                        pixels,
                    )?;
                }
            }
            assert_eq!(
                full, compact,
                "bounded target changed {name} pixels at {viewport:?}"
            );
            if check_vkd3d_reference {
                check_corrected_vkd3d_reference(name, viewport, &full)?;
            }
        }
    }
    Ok(())
}

fn check_corrected_vkd3d_reference(name: &str, viewport: [u32; 2], actual: &[u8]) -> Result<()> {
    macro_rules! fixture {
        ($name:literal) => {
            include_bytes!(concat!("directx_path_test_data/", $name, ".png")).as_slice()
        };
    }
    let (golden, rounding_mask, allowed_counts) = match (name, viewport) {
        ("gradient", [128, 128]) => (
            fixture!("gradient-128x128"),
            Some(fixture!("gradient-128x128-rounding-mask")),
            [0, 0, 0, 44],
        ),
        ("pattern", [128, 128]) => (
            fixture!("pattern-128x128"),
            Some(fixture!("pattern-128x128-rounding-mask")),
            [0, 31, 0, 165],
        ),
        ("overlap", [128, 128]) => (
            fixture!("overlap-128x128"),
            Some(fixture!("overlap-128x128-rounding-mask")),
            [0, 8, 0, 49],
        ),
        ("gradient", [173, 117]) => (fixture!("gradient-173x117"), None, [0, 0, 0, 0]),
        ("pattern", [173, 117]) => (fixture!("pattern-173x117"), None, [0, 0, 0, 0]),
        ("overlap", [173, 117]) => (fixture!("overlap-173x117"), None, [0, 0, 0, 0]),
        _ => unreachable!("missing corrected vkd3d reference"),
    };
    let decode = |bytes: &[u8]| -> Result<Vec<u8>> {
        let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)?.to_rgba8();
        assert_eq!(image.dimensions(), (viewport[0], viewport[1]));
        let mut bytes = image.into_raw();
        for pixel in bytes.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
        Ok(bytes)
    };
    let golden = decode(golden)?;
    let rounding_mask = rounding_mask.map(decode).transpose()?;
    assert_eq!(actual.len(), golden.len());
    let mut counts = [0; 4];
    for (index, (&old, &new)) in golden.iter().zip(actual).enumerate() {
        if old == new {
            continue;
        }
        let channel = index % 4;
        let x = index / 4 % viewport[0] as usize;
        let y = index / 4 / viewport[0] as usize;
        // Only the measured normalized-sampling rounding coordinates may differ.
        // This rejects displaced Bayer/pattern phases and any changed clip or AA edge.
        assert_eq!(
            rounding_mask.as_ref().map_or(0, |mask| mask[index]),
            255,
            "unexplained {name} reference change at ({x}, {y}), BGRA channel {channel}: {old}->{new}"
        );
        assert_eq!(
            old.abs_diff(new),
            1,
            "{name} reference change exceeds one code at ({x}, {y}), BGRA channel {channel}"
        );
        counts[channel] += 1;
    }
    println!(
        "{name} {viewport:?} corrected normalized-sampling differences by BGRA channel: {counts:?}"
    );
    for channel in 0..4 {
        assert_eq!(counts[channel], allowed_counts[channel]);
    }
    Ok(())
}

#[test]
fn path_targets_reuse_release_and_recreate_without_window_sized_allocation() -> Result<()> {
    let devices = test_devices()?;
    let mut resources = DirectXResources::new(&devices, 1518, 900, HWND::default(), false)?;
    assert!(resources.paths.is_none());
    resources.ensure_paths(&devices.device, Some([128, 64]))?;
    let first = resources.paths.as_ref().unwrap().texture.clone();
    resources.ensure_paths(&devices.device, Some([64, 64]))?;
    assert_eq!(resources.paths.as_ref().unwrap().texture, first);
    assert!(
        resources
            .ensure_paths(&devices.device, Some([u32::MAX, 64]))
            .is_err()
    );
    assert_eq!(
        resources.paths.as_ref().unwrap().texture,
        first,
        "failed growth must leave the current path resources intact"
    );
    resources.ensure_paths(&devices.device, Some([64, 192]))?;
    assert_eq!(resources.paths.as_ref().unwrap().size, [128, 192]);
    assert_ne!(resources.paths.as_ref().unwrap().texture, first);
    resources.ensure_paths(&devices.device, None)?;
    assert!(resources.paths.is_none());
    resources.ensure_paths(&devices.device, Some([64, 64]))?;
    resources.recreate_resources(&devices, 1518, 900)?;
    assert!(resources.paths.is_none());
    Ok(())
}

#[test]
fn resize_releases_active_backbuffer_after_incomplete_render() -> Result<()> {
    let mut renderer = renderer([128, 128])?;
    let devices = renderer.devices.as_ref().unwrap();
    let resources = renderer.resources.as_mut().unwrap();
    resources.ensure_paths(&devices.device, Some([64, 64]))?;
    // A fallible draw can exit before render's normal active-target cleanup.
    renderer.active_render_target = resources.render_target_view.clone();
    renderer.resize(size(DevicePixels(97), DevicePixels(65)))?;
    assert!(renderer.active_render_target.is_none());
    let resources = renderer.resources.as_ref().unwrap();
    assert!(resources.paths.is_none());
    assert_eq!(resources.viewport.Width, 97.0);
    assert_eq!(resources.viewport.Height, 65.0);
    let paths = [curved_path(hsla(0.4, 0.6, 0.5, 0.8).into(), 0.0)];
    let pixels = render_paths(&mut renderer, &paths, [26, 22], [64, 64])?;
    assert!(pixels.chunks_exact(4).any(|pixel| pixel[3] > 0));
    Ok(())
}

#[test]
fn ordinary_render_releases_path_targets_after_paths_disappear() -> Result<()> {
    let mut renderer = renderer([173, 117])?;
    let mut scene = Scene::default();
    scene.insert_primitive(curved_path(hsla(0.2, 0.8, 0.5, 0.7).into(), 0.0));
    scene.finish();
    let pixels = renderer.render_to_image(&scene, gpui::WindowBackgroundAppearance::Transparent)?;
    assert!(pixels.pixels().filter(|p| p[3] != 0).count() > 200);
    assert_eq!(
        renderer
            .resources
            .as_ref()
            .unwrap()
            .paths
            .as_ref()
            .unwrap()
            .size,
        [64, 64]
    );
    renderer.render_to_image(
        &Scene::default(),
        gpui::WindowBackgroundAppearance::Transparent,
    )?;
    assert!(renderer.resources.as_ref().unwrap().paths.is_none());
    Ok(())
}

fn curved_path(color: Background, offset: f32) -> Path<ScaledPixels> {
    let start = point(px(21.137 + offset), px(17.281));
    let mut path = Path::new(start);
    path.line_to(point(px(71.719 + offset), px(20.413)));
    path.curve_to(
        point(px(55.317 + offset), px(72.193)),
        point(px(82.719 + offset), px(67.431)),
    );
    path.line_to(point(px(24.619 + offset), px(65.137)));
    path.line_to(start);
    let mut path = path.scale(1.0);
    path.order = u32::from(offset != 0.0);
    path.color = color;
    path.content_mask = ContentMask {
        bounds: Bounds::new(
            point(ScaledPixels(27.13), ScaledPixels(23.17)),
            size(ScaledPixels(49.25), ScaledPixels(43.71)),
        ),
    };
    path
}

fn readback(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    target: &ID3D11Texture2D,
) -> Result<Vec<u8>> {
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe { target.GetDesc(&mut desc) };
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
    let mut staging = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut staging))? };
    let staging = staging.context("staging texture missing")?;
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    let mut result = vec![0; (desc.Width * desc.Height * 4) as usize];
    unsafe {
        context.CopyResource(&staging, target);
        context.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;
        for y in 0..desc.Height as usize {
            let row = std::slice::from_raw_parts(
                (mapped.pData as *const u8).add(y * mapped.RowPitch as usize),
                desc.Width as usize * 4,
            );
            result[y * desc.Width as usize * 4..(y + 1) * desc.Width as usize * 4]
                .copy_from_slice(row);
        }
        context.Unmap(&staging, 0);
    }
    Ok(result)
}
