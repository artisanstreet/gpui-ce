use super::path_pixel_tests::renderer;
use anyhow::Result;
use gpui::{
    BackdropFilter, Bounds, ContentMask, Corners, FilterBoundary, Quad, ScaledFilter, ScaledPixels,
    Scene, WindowBackgroundAppearance, hsla, point, size,
};

fn scene(content_depth: usize) -> Scene {
    let bounds = Bounds::new(
        point(ScaledPixels(11.0), ScaledPixels(9.0)),
        size(ScaledPixels(70.0), ScaledPixels(40.0)),
    );
    let mask = ContentMask {
        bounds: Bounds::new(
            point(ScaledPixels(0.0), ScaledPixels(0.0)),
            size(ScaledPixels(97.0), ScaledPixels(65.0)),
        ),
    };
    let mut scene = Scene::default();
    scene.insert_primitive(Quad {
        order: 0,
        bounds: mask.bounds,
        content_mask: mask,
        background: hsla(0.1, 0.2, 0.25, 0.9).into(),
        ..Default::default()
    });
    let mut order = 1;
    for _ in 0..content_depth {
        scene.insert_primitive(FilterBoundary {
            order,
            bounds,
            content_mask: mask,
            corner_radii: Corners::default(),
            filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(3.0))],
            opacity: 0.8,
            is_start: true,
        });
        order += 1;
    }
    scene.insert_primitive(Quad {
        order,
        bounds,
        content_mask: mask,
        background: hsla(0.7, 0.8, 0.6, 0.6).into(),
        ..Default::default()
    });
    order += 1;
    for _ in 0..content_depth {
        scene.insert_primitive(FilterBoundary {
            order,
            bounds,
            content_mask: mask,
            corner_radii: Corners::default(),
            filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(3.0))],
            opacity: 0.8,
            is_start: false,
        });
        order += 1;
    }
    scene.insert_primitive(BackdropFilter {
        order,
        bounds,
        content_mask: mask,
        corner_radii: Corners::default(),
        filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(2.0))],
        opacity: 0.7,
    });
    scene.finish();
    scene
}

#[test]
fn nested_content_groups_release_and_reappear_while_backdrop_blur_remains() -> Result<()> {
    let mut renderer = renderer([97, 65])?;
    let nested = scene(2);
    let first = renderer.render_to_image(&nested, WindowBackgroundAppearance::Transparent)?;
    assert!(first.pixels().any(|p| p[3] > 0));
    let blur = renderer.resources.as_ref().unwrap().blur.as_ref().unwrap();
    assert_eq!(blur.groups.len(), 2);
    let scene_color = blur.scene_color.clone();
    let ping = blur.ping.clone();
    let pong = blur.pong.clone();
    renderer.render_to_image(&scene(1), WindowBackgroundAppearance::Transparent)?;
    assert_eq!(
        renderer
            .resources
            .as_ref()
            .unwrap()
            .blur
            .as_ref()
            .unwrap()
            .groups
            .len(),
        2,
        "active content filters retain their high-water depth to avoid alternating-frame churn"
    );
    renderer.render_to_image(&scene(0), WindowBackgroundAppearance::Transparent)?;
    let blur = renderer.resources.as_ref().unwrap().blur.as_ref().unwrap();
    assert!(blur.groups.is_empty());
    assert!(blur.group_rtvs.is_empty());
    assert!(blur.group_srvs.is_empty());
    assert_eq!(blur.scene_color, scene_color);
    assert_eq!(blur.ping, ping);
    assert_eq!(blur.pong, pong);
    let restored = renderer.render_to_image(&nested, WindowBackgroundAppearance::Transparent)?;
    assert_eq!(
        restored, first,
        "recreated nested targets changed the image"
    );
    assert_eq!(
        renderer
            .resources
            .as_ref()
            .unwrap()
            .blur
            .as_ref()
            .unwrap()
            .groups
            .len(),
        2
    );
    Ok(())
}

#[test]
fn queued_content_group_work_survives_pool_release_and_recreation() -> Result<()> {
    use super::{BlurResources, resource_tests::test_devices};
    use anyhow::Context;
    use windows::Win32::Graphics::Direct3D11::{
        D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_TEXTURE2D_DESC,
        D3D11_USAGE_STAGING,
    };
    let devices = test_devices()?;
    let device = &devices.device;
    let context = &devices.device_context;
    let mut blur = BlurResources::new(device, 16, 8)?;
    blur.ensure_groups(device, 16, 8, 2)?;
    let mut deferred = None;
    unsafe { device.CreateDeferredContext(0, Some(&mut deferred))? };
    let deferred = deferred.context("deferred context missing")?;
    let mut readbacks = Vec::new();
    for (index, color) in [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
        .iter()
        .enumerate()
    {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { blur.groups[index].GetDesc(&mut desc) };
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
        let mut staging = None;
        unsafe { device.CreateTexture2D(&desc, None, Some(&mut staging))? };
        let staging = staging.context("staging missing")?;
        unsafe {
            deferred.ClearRenderTargetView(blur.group_rtvs[index].as_ref().unwrap(), color);
            deferred.CopyResource(&staging, &blur.groups[index]);
        }
        readbacks.push(staging);
    }
    let mut commands = None;
    unsafe { deferred.FinishCommandList(false, Some(&mut commands))? };
    let commands = commands.context("command list missing")?;
    drop(deferred);
    // No caller keeps the old group textures or views. The recorded commands
    // must own their dependencies until the immediate context executes them.
    blur.release_unused_groups(0);
    assert!(blur.groups.is_empty());
    blur.ensure_groups(device, 16, 8, 2)?;
    for view in &blur.group_rtvs {
        unsafe { context.ClearRenderTargetView(view.as_ref().unwrap(), &[0.0, 0.0, 1.0, 1.0]) };
    }
    unsafe { context.ExecuteCommandList(&commands, false) };
    for (texture, expected) in readbacks.iter().zip([[0, 0, 255, 255], [0, 255, 0, 255]]) {
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { context.Map(texture, 0, D3D11_MAP_READ, 0, Some(&mut mapped))? };
        let mut pixels = Vec::new();
        unsafe {
            for y in 0..8 {
                pixels.extend_from_slice(std::slice::from_raw_parts(
                    (mapped.pData as *const u8).add(y * mapped.RowPitch as usize),
                    16 * 4,
                ));
            }
            context.Unmap(texture, 0);
        }
        assert!(pixels.chunks_exact(4).all(|pixel| pixel == expected));
    }
    Ok(())
}
