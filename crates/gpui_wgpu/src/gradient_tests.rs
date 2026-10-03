//! Pixel regression for linear-gradient encoding and ordered dithering.
//!
//! Color-pipeline contract under test (see `shaders.wgsl`):
//! * `ColorSpace::Srgb` interpolates the gamma-encoded stops directly (that
//!   space's documented mode); `ColorSpace::Oklab` decodes the stops to
//!   linear light once, interpolates the converted values, maps back through
//!   linear light, and encodes once to sRGB. All targets (swapchain,
//!   path/scene/group/blur intermediates and resolves) use the same non-sRGB
//!   UNORM format, so there is no other encode step to account for.
//! * a deterministic 4x4 Bayer dither of strictly less than +/-0.5 LSB is
//!   added in encoded space, pre-premultiplication, on varying gradients
//!   only; identical stops skip it, so flat gradients stay uniform. Alpha is
//!   never dithered. Solid/sprite/text/blur paths never reach that branch.
//!
//! Precision note: the old conversions round-tripped at the endpoints, so
//! endpoint bytes were already stop-equal; the Oklab midpoint interpolation
//! was wrong, the sRGB path mixed double-encoded stops, and the narrow dark
//! ramp quantized into visible bands without dither. The tests below pin the
//! corrected midpoint, the encoded-sRGB midpoint, and the dither statistics.
//!
//! Uses the shipping quad shader and instance layout offscreen, mirroring
//! `shadow_tests.rs`. The ramp is the app's actual desktop face, S900 (top)
//! -> S925 (bottom, root `fff79dba`), derived from the canonical Oklch
//! constants through the app's own conversion chain
//! (`modules/ui/src/theme.rs`: `Oklch::to_srgb`, then `srgb_to_hsla`).
use super::*;
use gpui::{
    BorderStyle, ColorSpace, ContentMask, Corners, Edges, block_on, hsla,
    linear_color_stop, linear_gradient, point, size, solid_background,
};
use std::{sync::mpsc, time::Duration};
use wgpu::util::DeviceExt as _;

/// Canonical surface-ramp constants, `(lightness, chroma, hue)`.
const S900_OKLCH: (f64, f64, f64) = (0.21, 0.006, 285.885);
const S925_OKLCH: (f64, f64, f64) = (0.1755, 0.0055, 285.854);

#[test]
fn dark_oklab_ramp_matches_reference_with_zero_mean_dither() -> Result<()> {
    let (device, queue) = test_device()?;
    let pipeline = quad_pipeline(&device, wgpu::BlendState::ALPHA_BLENDING);
    let globals = globals_group(&device, &pipeline, 64.0, 256.0, 0);
    let (ht, st, lt) = surface_hsla(S900_OKLCH);
    let (hb, sb, lb) = surface_hsla(S925_OKLCH);
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
    // Endpoints round-trip to the stops: the single output encode inverts
    // the single input decode.
    assert!(
        (at(32, 0, 0) as f64 - top_srgb[0] * 255.0).abs() <= 2.0
            && (at(32, 0, 1) as f64 - top_srgb[1] * 255.0).abs() <= 2.0
            && (at(32, 0, 2) as f64 - top_srgb[2] * 255.0).abs() <= 2.0,
        "top endpoint drifted from S900"
    );
    assert!(
        (at(32, 255, 0) as f64 - bottom_srgb[0] * 255.0).abs() <= 2.0
            && (at(32, 255, 1) as f64 - bottom_srgb[1] * 255.0).abs() <= 2.0
            && (at(32, 255, 2) as f64 - bottom_srgb[2] * 255.0).abs() <= 2.0,
        "bottom endpoint drifted from S925"
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
    // Dither distribution: the Bayer pattern flips the rounding almost every
    // row on this sub-code-per-row slope; an undithered ramp would show ~8
    // transitions total (one per code boundary).
    let transitions = column.windows(2).filter(|pair| pair[0] != pair[1]).count();
    assert!(
        transitions >= 64,
        "only {transitions} row transitions: gradient dither is missing"
    );
    // Dither correctness: each full 4x4 Bayer period sums to exactly zero in
    // dither units, so every aligned block mean tracks the undithered
    // reference at the block center. Columns 32..36 are block-aligned.
    for block_row in [8, 24, 40, 56] {
        let mut sum = 0.0;
        for dy in 0..4 {
            for dx in 0..4 {
                sum += at(32 + dx, block_row * 4 + dy, 1) as f64;
            }
        }
        let mean = sum / 16.0;
        let center = (block_row * 4) as f64 + 2.0;
        let t = center / 256.0;
        let expected = undithered_oklab_green(top_ok, bottom_ok, t);
        assert!(
            (mean - expected).abs() <= 1.0,
            "block row {block_row} mean {mean} vs reference {expected}"
        );
    }
    // The dither is zero-mean overall as well.
    let mean_green: f64 = (0..256).map(|y| at(32, y, 1) as f64).sum::<f64>() / 256.0;
    let mean_reference: f64 = (0..256).map(|y| reference(y, false)[1]).sum::<f64>() / 256.0;
    assert!(
        (mean_green - mean_reference).abs() <= 1.0,
        "ramp mean {mean_green} drifted from reference {mean_reference}"
    );

    // The gradient endpoint equals the same color as a solid fill: one
    // shared gamma contract, not a gradient-only variant.
    let solid_globals = globals_group(&device, &pipeline, 64.0, 64.0, 0);
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
fn srgb_black_white_midpoint_is_encoded() -> Result<()> {
    let (device, queue) = test_device()?;
    let pipeline = quad_pipeline(&device, wgpu::BlendState::ALPHA_BLENDING);
    let globals = globals_group(&device, &pipeline, 64.0, 256.0, 0);
    let background = linear_gradient(
        180.0,
        linear_color_stop(hsla(0.0, 0.0, 0.0, 1.0), 0.0),
        linear_color_stop(hsla(0.0, 0.0, 1.0, 1.0), 1.0),
    )
    .color_space(ColorSpace::Srgb);
    let pixels = render_quad_pixels(
        &device,
        &queue,
        &pipeline,
        &globals,
        full_quad(background, 64.0, 256.0),
        64,
        256,
    )?;
    let green = |y: usize| pixels[(y * 64 + 32) * 4 + 1];
    // Encoded-sRGB interpolation puts the black/white midpoint near 128.
    // Linear-light interpolation would land near 188; the old double-encoded
    // mix landed near 55. The window below excludes both.
    for y in [127, 128] {
        assert!(
            (110..=150).contains(&green(y)),
            "midpoint row {y} is {}, expected encoded-sRGB ~128",
            green(y)
        );
    }
    // The ramp still spans the full range with unit slope.
    assert!(green(0) <= 2, "black endpoint lifted: {}", green(0));
    assert!(green(255) >= 253, "white endpoint sank: {}", green(255));
    for y in 120..136 {
        assert!(
            (118..=138).contains(&green(y)),
            "mid-band row {y} is {}",
            green(y)
        );
    }
    Ok(())
}

#[test]
fn equal_stop_gradient_matches_solid() -> Result<()> {
    let (device, queue) = test_device()?;
    let pipeline = quad_pipeline(&device, wgpu::BlendState::ALPHA_BLENDING);
    let globals = globals_group(&device, &pipeline, 64.0, 64.0, 0);
    let (h, s, l) = surface_hsla(S900_OKLCH);
    let flat = render_quad_pixels(
        &device,
        &queue,
        &pipeline,
        &globals,
        full_quad(
            linear_gradient(
                180.0,
                linear_color_stop(hsla(h, s, l, 1.0), 0.0),
                linear_color_stop(hsla(h, s, l, 1.0), 1.0),
            ),
            64.0,
            64.0,
        ),
        64,
        64,
    )?;
    let solid = render_quad_pixels(
        &device,
        &queue,
        &pipeline,
        &globals,
        full_quad(solid_background(hsla(h, s, l, 1.0)), 64.0, 64.0),
        64,
        64,
    )?;
    // Identical stops skip the dither branch, so the flat gradient is
    // spatially uniform: no noise is added to gradient-shaped solid fills.
    let first = [flat[0], flat[1], flat[2], flat[3]];
    for (i, byte) in flat.iter().enumerate() {
        assert_eq!(
            *byte,
            first[i % 4],
            "flat gradient varies at byte {i}: {byte} vs {}",
            first[i % 4]
        );
    }
    // ... and it matches the solid fill within f32 round-trip of the Oklab
    // conversions (the dither gate removes the noise term, not the math).
    for c in 0..4 {
        let gradient = first[c] as f64;
        let filled = solid[(32 * 64 + 32) * 4 + c] as f64;
        assert!(
            (gradient - filled).abs() <= 1.0,
            "flat gradient {gradient} != solid {filled} on channel {c}"
        );
    }
    Ok(())
}

#[test]
fn solid_fill_stays_bit_exact() -> Result<()> {
    let (device, queue) = test_device()?;
    let pipeline = quad_pipeline(&device, wgpu::BlendState::ALPHA_BLENDING);
    let globals = globals_group(&device, &pipeline, 64.0, 64.0, 0);
    let (h, s, l) = surface_hsla(S900_OKLCH);
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
fn transparent_gradient_edge_blends_monotonically_in_both_alpha_modes() -> Result<()> {
    let (device, queue) = test_device()?;
    let (h, s, l) = surface_hsla(S900_OKLCH);
    let background = linear_gradient(
        180.0,
        linear_color_stop(hsla(h, s, l, 1.0), 0.0),
        linear_color_stop(hsla(h, s, l, 0.0), 1.0),
    );
    // The app shell may composite straight (opaque window) or premultiplied
    // (transparent window); over a clear target both must store rgb scaled
    // by coverage with the same bytes.
    let modes = [
        (wgpu::BlendState::ALPHA_BLENDING, 0u32),
        (
            wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            1u32,
        ),
    ];
    let srgb = hsla_to_srgb(h as f64, s as f64, l as f64);
    for (blend, premultiplied) in modes {
        let pipeline = quad_pipeline(&device, blend);
        let globals = globals_group(&device, &pipeline, 64.0, 256.0, premultiplied);
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
            "opaque gradient edge lost coverage (premultiplied={premultiplied}): {}",
            alpha(0)
        );
        assert!(
            alpha(255) <= 5,
            "transparent gradient edge stays visible (premultiplied={premultiplied}): {}",
            alpha(255)
        );
        for y in 0..255 {
            assert!(
                alpha(y + 1) as u16 <= alpha(y) as u16 + 1,
                "alpha increases down the fade at row {y} (premultiplied={premultiplied}): {} -> {}",
                alpha(y),
                alpha(y + 1)
            );
        }
        for y in (0..256).step_by(8) {
            let coverage = alpha(y) as f64 / 255.0;
            for c in 0..3 {
                let got = pixels[(y * 64 + 32) * 4 + c] as f64;
                let expected = srgb[c] * 255.0 * coverage;
                assert!(
                    (got - expected).abs() <= 4.0,
                    "row {y} channel {c} (premultiplied={premultiplied}): got {got}, premultiplied reference {expected}"
                );
            }
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

fn quad_pipeline(device: &wgpu::Device, blend: wgpu::BlendState) -> wgpu::RenderPipeline {
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
                blend: Some(blend),
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
    premultiplied_alpha: u32,
) -> wgpu::BindGroup {
    let globals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gradient globals"),
        contents: bytemuck::bytes_of(&GlobalParams {
            viewport_size: [width, height],
            premultiplied_alpha,
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

/// App surface-ramp stop as GPUI `(hue, saturation, lightness)`, ported from
/// `modules/ui/src/theme.rs` (`Oklch::to_srgb` f64 Oklab constants with
/// clamped encode, then `srgb_to_hsla`).
fn surface_hsla(oklch: (f64, f64, f64)) -> (f32, f32, f32) {
    let (l, c, h_deg) = oklch;
    let radians = h_deg * std::f64::consts::TAU / 360.0;
    let a = c * radians.cos();
    let b = c * radians.sin();
    let l_ = l + 0.396_337_777_4 * a + 0.215_803_757_3 * b;
    let m_ = l - 0.105_561_345_8 * a - 0.063_854_172_8 * b;
    let s_ = l - 0.089_484_177_5 * a - 1.291_485_548_0 * b;
    let (big_l, big_m, big_s) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
    let lin_r = 4.076_741_662_1 * big_l - 3.307_711_591_3 * big_m + 0.230_969_929_2 * big_s;
    let lin_g = -1.268_438_004_6 * big_l + 2.609_757_401_1 * big_m - 0.341_319_396_5 * big_s;
    let lin_b = -0.004_196_086_3 * big_l - 0.703_418_614_7 * big_m + 1.707_614_701_0 * big_s;
    let (r, g, b) = (encode_app(lin_r), encode_app(lin_g), encode_app(lin_b));
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let lightness = (max + min) / 2.0;
    if delta == 0.0 {
        return (0.0, 0.0, lightness as f32);
    }
    let saturation = delta / (1.0 - (2.0 * lightness - 1.0).abs());
    let hue_sixth = if max == r {
        ((g - b) / delta).rem_euclid(6.0)
    } else if max == g {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    };
    (
        (hue_sixth / 6.0) as f32,
        saturation as f32,
        lightness as f32,
    )
}

/// App `encode`: sRGB transfer with sign-preserving clamp to `[0, 1]`.
fn encode_app(linear: f64) -> f64 {
    let magnitude = linear.abs();
    let encoded = if magnitude <= 0.003_130_8 {
        12.92 * magnitude
    } else {
        1.055 * magnitude.powf(1.0 / 2.4) - 0.055
    };
    let signed = if linear < 0.0 { -encoded } else { encoded };
    signed.clamp(0.0, 1.0)
}

/// Undithered Oklab-ramp reference for one green value at parameter `t`.
fn undithered_oklab_green(top_ok: [f64; 3], bottom_ok: [f64; 3], t: f64) -> f64 {
    let ok = [
        top_ok[0] + (bottom_ok[0] - top_ok[0]) * t,
        top_ok[1] + (bottom_ok[1] - top_ok[1]) * t,
        top_ok[2] + (bottom_ok[2] - top_ok[2]) * t,
    ];
    let linear = linear_of_oklab(ok);
    linear_to_srgb(linear[1].clamp(0.0, 1.0)).clamp(0.0, 1.0) * 255.0
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
