//! Pixel regression for linear-gradient encoding and ordered dithering.
//!
//! Color-pipeline contract under test (see `shaders.wgsl`):
//! * gradient stops are gamma-encoded sRGB (`hsla_to_rgba`), decoded to
//!   linear light once, interpolated in the requested space (Oklab here),
//!   encoded once to sRGB, then stored in the non-sRGB UNORM targets that
//!   every other primitive (solids, sprites, shadows, text) already treats
//!   as gamma-encoded. All intermediate targets and resolves use that same
//!   UNORM format, so there is no other encode step to account for.
//! * a deterministic 4x4 Bayer dither of strictly less than +/-0.5 LSB is
//!   added in encoded space, pre-premultiplication, on the gradient path
//!   only. Flat fills never reach that branch and stay bit-exact.
//!
//! Uses the shipping quad shader and instance layout offscreen, mirroring
//! `shadow_tests.rs`. The reported case is the desktop face S900 -> S925
//! (dark Oklab vertical ramp, root `fff79dba`); the stops below are a dark
//! blue-grey pair with the same shape (narrow lightness delta, near-neutral
//! chroma) so the same banding appears without the fix.
use super::*;
use gpui::{
    BorderStyle, ContentMask, Corners, Edges, block_on, hsla, linear_color_stop,
    linear_gradient, point, size, solid_background,
};
use std::{sync::mpsc, time::Duration};
use wgpu::util::DeviceExt as _;

/// Dark ramp stops `(hue, saturation, lightness)`, shaped like S900 -> S925.
const TOP_STOP: (f32, f32, f32) = (0.79, 0.25, 0.10);
const BOTTOM_STOP: (f32, f32, f32) = (0.79, 0.22, 0.135);

#[test]
fn dark_oklab_ramp_is_single_encoded_and_dithered() -> Result<()> {
    let (device, queue) = test_device()?;
    let pipeline = quad_pipeline(&device);
    let globals = globals_group(&device, &pipeline, 64.0, 256.0);
    let (ht, st, lt) = TOP_STOP;
    let (hb, sb, lb) = BOTTOM_STOP;
    let background = linear_gradient(
        180.0,
        linear_color_stop(hsla(ht, st, lt, 1.0), 0.0),
        linear_color_stop(hsla(hb, sb, lb, 1.0), 1.0),
    );
    let pixels = render_quad_pixels(
        &device,
        &queue,
        &pipeline,
        &globals,
        full_quad(background, 64.0, 256.0),
        64,
        256,
    )?;
    let at = |x: usize, y: usize, c: usize| pixels[(y * 64 + x) * 4 + c];

    // High-precision (f64) CPU reference of the documented contract.
    let top_srgb = hsla_to_srgb(ht as f64, st as f64, lt as f64);
    let bottom_srgb = hsla_to_srgb(hb as f64, sb as f64, lb as f64);
    let top_ok = oklab_of_srgb(top_srgb);
    let bottom_ok = oklab_of_srgb(bottom_srgb);
    let reference = |y: usize, dithered: bool| -> [f64; 3] {
        let t = (y as f64 + 0.5) / 256.0;
        let ok = [
            top_ok[0] + (bottom_ok[0] - top_ok[0]) * t,
            top_ok[1] + (bottom_ok[1] - top_ok[1]) * t,
            top_ok[2] + (bottom_ok[2] - top_ok[2]) * t,
        ];
        let linear = linear_of_oklab(ok);
        let mut out = [0.0; 3];
        for (i, value) in out.iter_mut().enumerate() {
            let dither = if dithered {
                (bayer4(32, y as i32) + 0.5) / 16.0 - 0.5
            } else {
                0.0
            };
            *value =
                (linear_to_srgb(linear[i].clamp(0.0, 1.0)) + dither / 255.0).clamp(0.0, 1.0)
                    * 255.0;
        }
        out
    };

    // Sampled rows match the dithered reference within float/quant slack.
    for y in (0..256).step_by(8) {
        let expected = reference(y, true);
        for c in 0..3 {
            let got = at(32, y, c) as f64;
            assert!(
                (got - expected[c]).abs() <= 2.0,
                "row {y} channel {c}: got {got}, reference {}",
                expected[c]
            );
        }
    }
    // Endpoints equal the stops: a single encode, no missing/double gamma.
    for c in 0..3 {
        let top = at(32, 0, c) as f64;
        let bottom = at(32, 255, c) as f64;
        assert!(
            (top - top_srgb[c] * 255.0).abs() <= 2.0,
            "top endpoint channel {c}: got {top}, stop {}",
            top_srgb[c] * 255.0
        );
        assert!(
            (bottom - bottom_srgb[c] * 255.0).abs() <= 2.0,
            "bottom endpoint channel {c}: got {bottom}, stop {}",
            bottom_srgb[c] * 255.0
        );
    }
    // The dither is zero-mean: the column mean tracks the undithered ramp.
    let mean_green: f64 = (0..256).map(|y| at(32, y, 1) as f64).sum::<f64>() / 256.0;
    let mean_reference: f64 = (0..256).map(|y| reference(y, false)[1]).sum::<f64>() / 256.0;
    assert!(
        (mean_green - mean_reference).abs() <= 1.0,
        "ramp mean {mean_green} drifted from reference {mean_reference}"
    );
    // Continuity: no visible steps between adjacent rows.
    let column: Vec<u8> = (0..256).map(|y| at(32, y, 1)).collect();
    let max_step = column
        .windows(2)
        .map(|pair| pair[0].abs_diff(pair[1]))
        .max()
        .unwrap_or(0);
    assert!(
        max_step <= 2,
        "ramp steps by {max_step} codes between adjacent rows"
    );
    // Dither activity: without ordered dither this narrow ramp quantizes
    // into plateaus ~15 rows tall; the Bayer pattern keeps every run short.
    let mut max_run = 1;
    let mut run = 1;
    for pair in column.windows(2) {
        if pair[0] == pair[1] {
            run += 1;
            max_run = max_run.max(run);
        } else {
            run = 1;
        }
    }
    assert!(
        max_run <= 6,
        "flat plateau of {max_run} rows: gradient dither is missing"
    );
    let distinct: std::collections::BTreeSet<u8> = column.into_iter().collect();
    assert!(
        distinct.len() >= 8,
        "ramp covers only {} distinct codes",
        distinct.len()
    );

    // The gradient endpoint equals the same color as a solid fill: one
    // shared gamma contract, not a gradient-only variant.
    let solid_globals = globals_group(&device, &pipeline, 64.0, 64.0);
    let solid = render_quad_pixels(
        &device,
        &queue,
        &pipeline,
        &solid_globals,
        full_quad(solid_background(hsla(ht, st, lt, 1.0)), 64.0, 64.0),
        64,
        64,
    )?;
    for c in 0..3 {
        let gradient_top = at(32, 0, c) as f64;
        let flat = solid[(32 * 64 + 32) * 4 + c] as f64;
        assert!(
            (gradient_top - flat).abs() <= 2.0,
            "gradient endpoint {gradient_top} != solid fill {flat} on channel {c}"
        );
    }
    Ok(())
}

#[test]
fn solid_fill_stays_bit_exact() -> Result<()> {
    let (device, queue) = test_device()?;
    let pipeline = quad_pipeline(&device);
    let globals = globals_group(&device, &pipeline, 64.0, 64.0);
    let (h, s, l) = TOP_STOP;
    let pixels = render_quad_pixels(
        &device,
        &queue,
        &pipeline,
        &globals,
        full_quad(solid_background(hsla(h, s, l, 1.0)), 64.0, 64.0),
        64,
        64,
    )?;
    // Flat fills never touch the gradient dither branch: every pixel is
    // identical, so no noise is added to solid colors (or, by the same
    // gating, to sprite/text/blur paths, which share no code with it).
    let first = [pixels[0], pixels[1], pixels[2], pixels[3]];
    for y in 0..64 {
        for x in 0..64 {
            let pixel = [
                pixels[(y * 64 + x) * 4],
                pixels[(y * 64 + x) * 4 + 1],
                pixels[(y * 64 + x) * 4 + 2],
                pixels[(y * 64 + x) * 4 + 3],
            ];
            assert_eq!(
                pixel, first,
                "solid fill varies at ({x}, {y}): {pixel:?} vs {first:?}"
            );
        }
    }
    let expected = hsla_to_srgb(h as f64, s as f64, l as f64);
    for c in 0..3 {
        let got = first[c] as f64;
        assert!(
            (got - expected[c] * 255.0).abs() <= 1.0,
            "solid channel {c}: got {got}, reference {}",
            expected[c] * 255.0
        );
    }
    assert_eq!(first[3], 255);
    Ok(())
}

#[test]
fn transparent_gradient_edge_blends_monotonically() -> Result<()> {
    let (device, queue) = test_device()?;
    let pipeline = quad_pipeline(&device);
    let globals = globals_group(&device, &pipeline, 64.0, 256.0);
    let (h, s, l) = TOP_STOP;
    let background = linear_gradient(
        180.0,
        linear_color_stop(hsla(h, s, l, 1.0), 0.0),
        linear_color_stop(hsla(h, s, l, 0.0), 1.0),
    );
    let pixels = render_quad_pixels(
        &device,
        &queue,
        &pipeline,
        &globals,
        full_quad(background, 64.0, 256.0),
        64,
        256,
    )?;
    let alpha = |y: usize| pixels[(y * 64 + 32) * 4 + 3];
    assert!(
        alpha(0) >= 250,
        "opaque gradient edge lost coverage: {}",
        alpha(0)
    );
    assert!(
        alpha(255) <= 5,
        "transparent gradient edge stays visible: {}",
        alpha(255)
    );
    for y in 0..255 {
        assert!(
            alpha(y + 1) as u16 <= alpha(y) as u16 + 1,
            "alpha increases down the fade at row {y}: {} -> {}",
            alpha(y),
            alpha(y + 1)
        );
    }
    // Straight-alpha blending over a clear target stores rgb scaled by
    // coverage; the dithered source keeps that premultiplied ramp tight.
    let srgb = hsla_to_srgb(h as f64, s as f64, l as f64);
    for y in (0..256).step_by(8) {
        let coverage = alpha(y) as f64 / 255.0;
        for c in 0..3 {
            let got = pixels[(y * 64 + 32) * 4 + c] as f64;
            let expected = srgb[c] * 255.0 * coverage;
            assert!(
                (got - expected).abs() <= 4.0,
                "row {y} channel {c}: got {got}, premultiplied reference {expected}"
            );
        }
    }
    Ok(())
}

fn full_quad(background: Background, width: f32, height: f32) -> Quad {
    let bounds = |x, y, w, h| {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    };
    Quad {
        order: 0,
        border_style: BorderStyle::Solid,
        bounds: bounds(0.0, 0.0, width, height),
        content_mask: ContentMask {
            bounds: bounds(0.0, 0.0, width, height),
        },
        background,
        border_color: hsla(0.0, 0.0, 0.0, 0.0).into(),
        corner_radii: Corners::all(ScaledPixels(0.0)),
        border_widths: Edges::default(),
    }
}

fn test_device() -> Result<(wgpu::Device, wgpu::Queue)> {
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
                label: Some("gradient pixel regression"),
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

fn quad_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production quads"),
        source: wgpu::ShaderSource::Wgsl(STORAGE_BUFFER_SHADERS.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gradient pixel regression"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_quad"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_quad"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn globals_group(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    width: f32,
    height: f32,
) -> wgpu::BindGroup {
    let globals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gradient globals"),
        contents: bytemuck::bytes_of(&GlobalParams {
            viewport_size: [width, height],
            premultiplied_alpha: 0,
            pad: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gradient globals"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: globals.as_entire_binding(),
        }],
    })
}

fn render_quad_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    globals: &wgpu::BindGroup,
    quad: Quad,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    // SAFETY: every field, including padding from the repr(C) layout, is
    // initialized. This is the same instance upload the shipping renderer
    // uses.
    let bytes = unsafe { WgpuRenderer::instance_bytes(std::slice::from_ref(&quad)) };
    let instances = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gradient instance"),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let instances_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gradient instance"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: instances.as_entire_binding(),
        }],
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gradient pixels"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let bytes_per_row = width * 4;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gradient readback"),
        size: (bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let view = texture.create_view(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gradient pixels"),
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
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, globals, &[]);
        pass.set_bind_group(1, &instances_group, &[]);
        pass.draw(0..4, 0..1);
    }
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
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
    Ok(pixels)
}

/// Port of the shader's `hsla_to_rgba`, in f64 as the high-precision side
/// of the test contract.
fn hsla_to_srgb(h: f64, s: f64, l: f64) -> [f64; 3] {
    let h6 = h * 6.0;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (mut r, mut g, mut b) = (m, m, m);
    if h6 < 1.0 {
        r += c;
        g += x;
    } else if h6 < 2.0 {
        r += x;
        g += c;
    } else if h6 < 3.0 {
        g += c;
        b += x;
    } else if h6 < 4.0 {
        g += x;
        b += c;
    } else if h6 < 5.0 {
        r += x;
        b += c;
    } else {
        r += c;
        b += x;
    }
    [r, g, b]
}

fn srgb_to_linear_component(c: f64) -> f64 {
    if c < 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f64) -> f64 {
    if c < 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn oklab_of_srgb(srgb: [f64; 3]) -> [f64; 3] {
    let r = srgb_to_linear_component(srgb[0]);
    let g = srgb_to_linear_component(srgb[1]);
    let b = srgb_to_linear_component(srgb[2]);
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
    let l_ = l.powf(1.0 / 3.0);
    let m_ = m.powf(1.0 / 3.0);
    let s_ = s.powf(1.0 / 3.0);
    [
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ]
}

fn linear_of_oklab(oklab: [f64; 3]) -> [f64; 3] {
    let l_ = oklab[0] + 0.3963377774 * oklab[1] + 0.2158037573 * oklab[2];
    let m_ = oklab[0] - 0.1055613458 * oklab[1] - 0.0638541728 * oklab[2];
    let s_ = oklab[0] - 0.0894841775 * oklab[1] - 1.2914855480 * oklab[2];
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ]
}

/// Bit-exact mirror of the shader's `bayer4_threshold` (integer math).
fn bayer4(x: i32, y: i32) -> f64 {
    let qx = (x >> 1) & 1;
    let qy = (y >> 1) & 1;
    let lx = x & 1;
    let ly = y & 1;
    let inner = 2 * (lx ^ ly) + ly;
    let offset = 2 * (qx ^ qy) + qy;
    (4 * inner + offset) as f64
}
