use super::*;
use gpui::{Bounds, ContentMask, Corners, Mesh, MeshStyle, ScaledPixels, point, size};
use std::sync::Arc;
use windows::Win32::Foundation::HMODULE;

fn device() -> Result<(ID3D11Device, ID3D11DeviceContext)> {
    let mut device = None;
    let mut context = None;
    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_WARP,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )?;
    }
    Ok((
        device.context("WARP device missing")?,
        context.context("WARP context missing")?,
    ))
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

fn render(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    paints: &[PaintMesh],
) -> Result<Vec<u8>> {
    render_with_scratch_mode(device, context, paints, true)
}

fn render_with_scratch_mode(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    paints: &[PaintMesh],
    shared: bool,
) -> Result<Vec<u8>> {
    let mut renderers: Vec<_> = (0..if shared { 1 } else { paints.len() })
        .map(|_| MeshRenderer::new(device))
        .collect::<Result<_>>()?;
    if shared {
        renderers[0].prepare(device, context, paints, (WIDTH, HEIGHT))?;
    } else {
        for (renderer, paint) in renderers.iter_mut().zip(paints) {
            renderer.prepare(
                device,
                context,
                std::slice::from_ref(paint),
                (WIDTH, HEIGHT),
            )?;
        }
    }
    let target = texture(
        device,
        (WIDTH, HEIGHT),
        COLOR_FORMAT,
        1,
        D3D11_BIND_RENDER_TARGET,
    )?;
    let mut view = None;
    unsafe {
        device.CreateRenderTargetView(&target, None, Some(&mut view))?;
        context.ClearRenderTargetView(view.as_ref().unwrap(), &[0.0; 4]);
        context.OMSetRenderTargets(Some(&[view]), None);
        context.RSSetViewports(Some(&[D3D11_VIEWPORT {
            Width: WIDTH as f32,
            Height: HEIGHT as f32,
            MaxDepth: 1.0,
            ..Default::default()
        }]));
    }
    if shared {
        renderers[0].composite(context, 0..paints.len());
    } else {
        for renderer in renderers {
            renderer.composite(context, 0..1);
        }
    }
    let pixels = readback(device, context, &target)?;
    unsafe { context.OMSetRenderTargets(None, None) };
    Ok(pixels)
}

fn save(name: &str, pixels: &[u8]) -> Result<()> {
    if let Some(directory) = std::env::var_os("GPUI_NATIVE_MESH_TEST_OUTPUT") {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory)?;
        image::RgbaImage::from_raw(WIDTH, HEIGHT, pixels.to_vec())
            .unwrap()
            .save(directory.join(format!("{name}.png")))?;
    }
    Ok(())
}

const WIDTH: u32 = 128;
const HEIGHT: u32 = 96;

type Matrix = [[f32; 4]; 4];

/// Column-major product `a * b`.
fn multiply(a: Matrix, b: Matrix) -> Matrix {
    let mut out = [[0.0; 4]; 4];
    for (column, out_column) in out.iter_mut().enumerate() {
        for (row, cell) in out_column.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[k][row] * b[column][k]).sum();
        }
    }
    out
}

fn translation(x: f32, y: f32, z: f32) -> Matrix {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, y, z, 1.0],
    ]
}

fn rotation_x(degrees: f32) -> Matrix {
    let (s, c) = degrees.to_radians().sin_cos();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, s, 0.0],
        [0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn rotation_y(degrees: f32) -> Matrix {
    let (s, c) = degrees.to_radians().sin_cos();
    [
        [c, 0.0, -s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// A right-handed perspective projection into WebGPU's 0..1 depth range.
fn perspective(fov_degrees: f32, aspect: f32, near: f32, far: f32) -> Matrix {
    let f = 1.0 / (fov_degrees.to_radians() * 0.5).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

/// A unit cube with flat normals; the top face is red.
fn cube() -> Mesh {
    let mut vertices = Vec::new();
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
    ];
    for (normal, u, v) in faces {
        let corner =
            |a: f32, b: f32| [0, 1, 2].map(|i| 0.5 * normal[i] + 0.5 * a * u[i] + 0.5 * b * v[i]);
        let color = if normal[1] > 0.5 {
            [1.0, 0.0, 0.0, 1.0]
        } else {
            [0.0; 4]
        };
        for (a, b) in [
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        ] {
            vertices.push(MeshVertex {
                position: corner(a, b),
                normal,
                color,
            });
        }
    }
    Mesh::new(vertices)
}

fn paint(mesh: &Arc<Mesh>, edges: f32) -> PaintMesh {
    let bounds = Bounds::new(
        point(ScaledPixels(16.0), ScaledPixels(16.0)),
        size(ScaledPixels(96.0), ScaledPixels(64.0)),
    );
    let view = multiply(
        translation(0.0, 0.0, -3.2),
        multiply(rotation_x(25.0), rotation_y(35.0)),
    );
    PaintMesh {
        order: 0,
        bounds,
        content_mask: ContentMask {
            bounds: Bounds::new(
                point(ScaledPixels(0.0), ScaledPixels(0.0)),
                size(ScaledPixels(WIDTH as f32), ScaledPixels(HEIGHT as f32)),
            ),
        },
        corner_radii: Corners::all(ScaledPixels(12.0)),
        opacity: 1.0,
        mesh: Arc::clone(mesh),
        style: MeshStyle {
            view,
            projection: perspective(30.0, 96.0 / 64.0, 0.1, 100.0),
            base_color: [0.5, 0.5, 0.5],
            edge_color: [0.0, 0.0, 0.0, edges],
        },
    }
}

#[test]
fn shared_mesh_scratch_preserves_depth_transparency_and_composite_order() -> Result<()> {
    let (device, queue) = device()?;
    let mut first = paint(&Arc::new(cube()), 0.0);
    first.bounds.origin = point(ScaledPixels(4.0), ScaledPixels(8.0));
    let mut second = paint(&Arc::new(cube()), 1.0);
    second.bounds.origin = point(ScaledPixels(28.0), ScaledPixels(24.0));
    second.opacity = 0.6;
    second.style.view = multiply(
        translation(0.0, 0.0, -3.8),
        multiply(rotation_x(-30.0), rotation_y(-50.0)),
    );
    second.style.base_color = [0.1, 0.8, 0.2];
    let mut paints = [first, second];
    let forward = render_with_scratch_mode(&device, &queue, &paints, true)?;
    assert_eq!(
        forward,
        render_with_scratch_mode(&device, &queue, &paints, false)?
    );
    paints.reverse();
    let reverse = render_with_scratch_mode(&device, &queue, &paints, true)?;
    assert_eq!(
        reverse,
        render_with_scratch_mode(&device, &queue, &paints, false)?
    );
    save("overlap-forward", &forward)?;
    save("overlap-reverse", &reverse)?;
    assert_ne!(
        forward, reverse,
        "the fixture must exercise overlapping composites"
    );
    Ok(())
}

#[test]
fn meshes_draw_lit_on_a_transparent_rounded_viewport() -> Result<()> {
    let (device, queue) = device()?;
    let empty = render(
        &device,
        &queue,
        &[paint(&Arc::new(Mesh::new(Vec::new())), 0.0)],
    )?;
    assert!(empty.iter().all(|channel| *channel == 0));
    let mesh = Arc::new(cube());
    let pixels = render(&device, &queue, &[paint(&mesh, 0.0)])?;
    let pixel = |x: u32, y: u32| {
        let at = ((y * WIDTH + x) * 4) as usize;
        [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
    };
    let center = pixel(64, 48);
    assert_eq!(center[3], 255, "the cube covers the viewport center");
    assert!(center[0] > 40, "the cube is lit, not black: {center:?}");
    assert_eq!(pixel(4, 4)[3], 0, "outside the viewport stays clear");
    assert_eq!(pixel(20, 20)[3], 0, "the background stays transparent");
    // The top face is red: some pixel above the center is red-dominant.
    let red = (20..48).any(|y| {
        let [r, g, b, a] = pixel(64, y);
        a == 255 && r > g.saturating_add(40) && r > b.saturating_add(40)
    });
    assert!(red, "vertex colors reach the surface");

    save("lit-cube", &pixels)?;
    let wire = render(&device, &queue, &[paint(&mesh, 1.0)])?;
    save("wire-cube", &wire)?;
    let dark = |pixels: &[u8]| {
        pixels
            .chunks_exact(4)
            .filter(|p| p[3] == 255 && p[0] < 30 && p[1] < 30 && p[2] < 30)
            .count()
    };
    assert!(
        dark(&wire) > dark(&pixels) + 20,
        "wireframe edges draw: {} dark pixels vs {}",
        dark(&wire),
        dark(&pixels)
    );
    Ok(())
}

#[test]
fn rounded_corners_clip_a_mesh_that_fills_its_bounds() -> Result<()> {
    let (device, context) = device()?;
    let mut filled = paint(&Arc::new(cube()), 0.0);
    // Head-on from 0.9 units the front face spans about ±3.1 × ±4.7 in NDC, so
    // the cube covers every target pixel, the corners included.
    filled.style.view = translation(0.0, 0.0, -0.9);
    let pixels = render(&device, &context, &[filled])?;
    save("filled-corners", &pixels)?;
    let alpha = |x: u32, y: u32| pixels[((y * WIDTH + x) * 4 + 3) as usize];
    // The bounds span 16..112 × 16..80 with 12 px radii; these pixel centers
    // sit about 2.9 px outside each arc.
    for (x, y) in [(17, 17), (110, 17), (17, 78), (110, 78)] {
        assert_eq!(alpha(x, y), 0, "the rounded corner at ({x}, {y}) clips");
    }
    // Edge midpoints and a pixel just inside the top-left arc stay covered.
    for (x, y) in [(64, 17), (17, 48), (110, 48), (64, 78), (20, 20)] {
        assert_eq!(alpha(x, y), 255, "the mesh covers ({x}, {y})");
    }
    Ok(())
}

#[test]
fn geometry_and_targets_reuse_then_release_and_reappear() -> Result<()> {
    let (device, context) = device()?;
    let mut renderer = MeshRenderer::new(&device)?;
    assert!(renderer.targets.is_empty());
    assert!(renderer.scratch.is_empty());
    let mesh = Arc::new(cube());
    let mut paints = [paint(&mesh, 0.0), paint(&mesh, 1.0)];
    renderer.prepare(&device, &context, &paints, (WIDTH, HEIGHT))?;
    assert_eq!(renderer.geometry.len(), 1);
    assert_eq!(renderer.targets.len(), 2);
    assert_eq!(renderer.scratch.len(), 1);
    let geometry = renderer.geometry[&mesh.id()].buffer.clone();
    let first_resolve = renderer.targets[0].resolve.clone();
    let first_scratch = renderer.scratch[&(96, 64)].color.clone();
    let first_pixels = readback(&device, &context, &first_resolve)?;
    assert!(first_pixels.iter().any(|byte| *byte != 0));

    paints[0].style.base_color = [0.1, 0.8, 0.1];
    renderer.prepare(&device, &context, &paints, (WIDTH, HEIGHT))?;
    assert_eq!(renderer.geometry[&mesh.id()].buffer, geometry);
    assert_eq!(renderer.targets[0].resolve, first_resolve);
    assert_eq!(renderer.scratch[&(96, 64)].color, first_scratch);
    assert_ne!(readback(&device, &context, &first_resolve)?, first_pixels);

    paints[1].bounds.size = size(ScaledPixels(48.0), ScaledPixels(32.0));
    renderer.prepare(&device, &context, &paints, (WIDTH, HEIGHT))?;
    assert_eq!(renderer.scratch.len(), 2);
    renderer.prepare(&device, &context, &paints[1..], (WIDTH, HEIGHT))?;
    assert_eq!(renderer.targets.len(), 1);
    assert_eq!(renderer.targets[0].size, (48, 32));
    assert_eq!(renderer.scratch.len(), 1);
    let last_resolve = renderer.targets[0].resolve.clone();
    renderer.prepare(&device, &context, &[], (WIDTH, HEIGHT))?;
    assert!(renderer.targets.is_empty());
    assert!(renderer.geometry.is_empty());
    assert!(renderer.scratch.is_empty());
    // A submitted resolve remains readable after cached resources are dropped.
    assert!(
        readback(&device, &context, &last_resolve)?
            .iter()
            .any(|v| *v != 0)
    );
    renderer.prepare(&device, &context, &paints, (WIDTH, HEIGHT))?;
    assert_eq!(renderer.targets.len(), 2);
    assert_eq!(renderer.scratch.len(), 2);
    assert_ne!(renderer.geometry[&mesh.id()].buffer, geometry);
    Ok(())
}

#[test]
fn composite_respects_clip_and_premultiplied_opacity() -> Result<()> {
    let (device, context) = device()?;
    let mut painted = paint(&Arc::new(cube()), 0.0);
    let full = render(&device, &context, &[painted.clone()])?;
    painted.opacity = 0.5;
    painted.content_mask.bounds = Bounds::new(
        point(ScaledPixels(64.0), ScaledPixels(0.0)),
        size(ScaledPixels(64.0), ScaledPixels(HEIGHT as f32)),
    );
    let clipped = render(&device, &context, &[painted])?;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = ((y * WIDTH + x) * 4) as usize;
            if x < 64 {
                assert_eq!(&clipped[offset..offset + 4], &[0; 4]);
            } else {
                for channel in 0..4 {
                    let expected = full[offset + channel] as f32 * 0.5;
                    assert!((clipped[offset + channel] as f32 - expected).abs() <= 1.0);
                }
            }
        }
    }
    save("clip-opacity", &clipped)?;
    Ok(())
}

#[test]
fn depth_and_two_sided_lighting_do_not_depend_on_triangle_submission_order() -> Result<()> {
    let (device, context) = device()?;
    let cube = cube();
    let reversed = Mesh::new(
        cube.vertices()
            .chunks_exact(3)
            .rev()
            .flatten()
            .copied()
            .collect(),
    );
    let inverted_normals = Mesh::new(
        cube.vertices()
            .iter()
            .map(|vertex| MeshVertex {
                normal: vertex.normal.map(|component| -component),
                ..*vertex
            })
            .collect(),
    );
    let original = render(&device, &context, &[paint(&Arc::new(cube), 0.0)])?;
    assert_eq!(
        original,
        render(&device, &context, &[paint(&Arc::new(reversed), 0.0)])?
    );
    assert_eq!(
        original,
        render(
            &device,
            &context,
            &[paint(&Arc::new(inverted_normals), 0.0)]
        )?
    );
    Ok(())
}

#[test]
fn geometry_cache_evicts_unused_models_after_the_existing_grace_period() -> Result<()> {
    let (device, context) = device()?;
    let mut renderer = MeshRenderer::new(&device)?;
    let first = paint(&Arc::new(cube()), 0.0);
    let second = paint(&Arc::new(cube()), 1.0);
    renderer.prepare(
        &device,
        &context,
        &[first.clone(), second.clone()],
        (WIDTH, HEIGHT),
    )?;
    for _ in 0..EVICT_AFTER_FRAMES {
        renderer.prepare(
            &device,
            &context,
            std::slice::from_ref(&second),
            (WIDTH, HEIGHT),
        )?;
    }
    assert!(renderer.geometry.contains_key(&first.mesh.id()));
    renderer.prepare(
        &device,
        &context,
        std::slice::from_ref(&second),
        (WIDTH, HEIGHT),
    )?;
    assert!(!renderer.geometry.contains_key(&first.mesh.id()));
    assert!(renderer.geometry.contains_key(&second.mesh.id()));
    Ok(())
}

#[test]
fn mesh_passes_preserve_scene_and_batch_constant_bindings() -> Result<()> {
    let (device, context) = device()?;
    let existing = [
        Some(constant_buffer::<[f32; 4]>(&device)?),
        Some(constant_buffer::<[f32; 4]>(&device)?),
    ];
    unsafe {
        context.VSSetConstantBuffers(0, Some(&existing));
        context.PSSetConstantBuffers(0, Some(&existing));
    }
    render(&device, &context, &[paint(&Arc::new(cube()), 1.0)])?;
    let mut vertex = [None, None, None, None];
    let mut fragment = [None, None, None, None];
    unsafe {
        context.VSGetConstantBuffers(0, Some(&mut vertex));
        context.PSGetConstantBuffers(0, Some(&mut fragment));
    }
    assert_eq!(&vertex[..2], &existing);
    assert_eq!(&fragment[..2], &existing);
    assert!(vertex[2..].iter().all(Option::is_none));
    assert!(fragment[2..].iter().all(Option::is_none));
    assert!(unsafe { context.IAGetInputLayout() }.is_err());
    Ok(())
}
