//! Pixel coverage of the shipping mesh pass and composite.
use super::*;
use gpui::{ContentMask, Corners, Mesh, MeshStyle, MeshVertex, PaintMesh, block_on, point, size};
use std::{sync::mpsc, time::Duration};
use wgpu::util::DeviceExt as _;

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

fn device() -> Result<(wgpu::Device, wgpu::Queue)> {
    block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mesh pixel regression"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits())
                    .using_alignment(adapter.limits()),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(anyhow::Error::from)
    })
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

/// Renders `paints` through the mesh pass and composite; RGBA rows.
fn render(device: &wgpu::Device, queue: &wgpu::Queue, paints: &[PaintMesh]) -> Result<Vec<u8>> {
    let layouts = WgpuRenderer::create_bind_group_layouts(device, false);
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut meshes = MeshRenderer::new(device, &layouts.globals, format);
    let globals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh globals"),
        contents: bytemuck::bytes_of(&GlobalParams {
            viewport_size: [WIDTH as f32, HEIGHT as f32],
            premultiplied_alpha: 1,
            pad: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let gamma = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh gamma"),
        contents: bytemuck::bytes_of(&GammaParams {
            gamma_ratios: [0.0; 4],
            grayscale_enhanced_contrast: 0.0,
            subpixel_enhanced_contrast: 0.0,
            is_bgr: 0,
            _pad: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let globals_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh globals"),
        layout: &layouts.globals,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gamma.as_entire_binding(),
            },
        ],
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mesh pixels"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let bytes_per_row = WIDTH * 4;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mesh readback"),
        size: u64::from(bytes_per_row * HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    meshes.prepare(device, queue, &mut encoder, paints);
    let view = texture.create_view(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh pixels"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        meshes.composite(0..paints.len(), &globals_group, &mut pass);
    }
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::channel();
    output
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: Some(Duration::from_secs(10)),
    })?;
    receiver.recv_timeout(Duration::from_secs(10))??;
    let pixels = output.slice(..).get_mapped_range().to_vec();
    output.unmap();
    if let Some(path) = std::env::var_os("GPUI_MESH_TEST_DUMP") {
        // A binary PPM of the RGB channels, for eyeballing the render.
        let mut ppm = format!("P6 {WIDTH} {HEIGHT} 255\n").into_bytes();
        for pixel in pixels.chunks_exact(4) {
            ppm.extend_from_slice(&pixel[..3]);
        }
        std::fs::write(path, ppm)?;
    }
    Ok(pixels)
}

#[test]
fn meshes_draw_lit_on_a_transparent_rounded_viewport() -> Result<()> {
    let (device, queue) = device()?;
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
    assert_eq!(pixel(18, 18)[3], 0, "the rounded corner clips");
    assert_eq!(pixel(20, 20)[3], 0, "the background stays transparent");
    // The top face is red: some pixel above the center is red-dominant.
    let red = (20..48).any(|y| {
        let [r, g, b, a] = pixel(64, y);
        a == 255 && r > g.saturating_add(40) && r > b.saturating_add(40)
    });
    assert!(red, "vertex colors reach the surface");

    let wire = render(&device, &queue, &[paint(&mesh, 1.0)])?;
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
