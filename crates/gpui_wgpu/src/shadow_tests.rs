//! Pixel coverage using the shipping shadow shader and instance layout.
use super::*;
use gpui::{ContentMask, Corners, block_on, hsla, point, size};
use std::{sync::mpsc, time::Duration};
use wgpu::util::DeviceExt as _;

#[test]
fn outer_shadows_do_not_fill_translucent_panels() -> Result<()> {
    let (device, queue) = block_on(async {
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
                label: Some("shadow pixel regression"),
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
    })?;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production shadows"),
        source: wgpu::ShaderSource::Wgsl(STORAGE_BUFFER_SHADERS.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shadow pixel regression"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_shadow"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_shadow"),
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
    });
    let globals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("shadow globals"),
        contents: bytemuck::bytes_of(&GlobalParams {
            viewport_size: [64.0, 64.0],
            premultiplied_alpha: 0,
            pad: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let globals_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow globals"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: globals.as_entire_binding(),
        }],
    });
    let bounds = |x, y, w, h| {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(w), ScaledPixels(h)),
        )
    };
    let mut shadow = Shadow {
        order: 0,
        blur_radius: ScaledPixels(0.0),
        // The shadow is translated and spread; the cutout must stay on the original element.
        bounds: bounds(15.0, 10.0, 40.0, 40.0),
        corner_radii: Corners::all(ScaledPixels(8.0)),
        content_mask: ContentMask {
            bounds: bounds(0.0, 0.0, 64.0, 64.0),
        },
        color: hsla(0.0, 0.0, 1.0, 0.5).into(),
        element_bounds: bounds(16.0, 16.0, 32.0, 32.0),
        element_corner_radii: Corners::all(ScaledPixels(8.0)),
        inset: 0,
        pad: 0,
    };
    for blur in [0.0, 3.0] {
        shadow.blur_radius = ScaledPixels(blur);
        let pixels = render_shadow_pixels(&device, &queue, &pipeline, &globals_group, shadow)?;
        let red = |x: usize, y: usize| pixels[(y * 64 + x) * 4];
        assert!(
            red(32, 32) <= 1,
            "outer shadow filled the card center with blur {blur}: {}",
            red(32, 32)
        );
        assert!(
            red(51, 32) > 20,
            "outer shadow disappeared beyond the border with blur {blur}"
        );
        assert!(
            red(18, 18) > 20,
            "cutout lost its rounded corner with blur {blur}"
        );
        assert_eq!(red(63, 63), 0);
    }
    shadow.inset = 1;
    shadow.blur_radius = ScaledPixels(0.0);
    shadow.bounds = bounds(22.0, 19.0, 26.0, 26.0);
    shadow.corner_radii = Corners::all(ScaledPixels(5.0));
    let pixels = render_shadow_pixels(&device, &queue, &pipeline, &globals_group, shadow)?;
    let red = |x: usize, y: usize| pixels[(y * 64 + x) * 4];
    assert!(red(17, 32) > 100, "inset shadow lost its inner edge");
    assert_eq!(red(32, 32), 0);
    assert_eq!(red(51, 32), 0, "inset shadow escaped its element");
    Ok(())
}

fn render_shadow_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    globals: &wgpu::BindGroup,
    shadow: Shadow,
) -> Result<Vec<u8>> {
    // SAFETY: every field, including both explicit padding fields, is initialized. This is
    // the same repr(C) instance upload used by the shipping renderer.
    let bytes = unsafe { WgpuRenderer::instance_bytes(std::slice::from_ref(&shadow)) };
    let instances = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("shadow instance"),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let instances_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shadow instance"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: instances.as_entire_binding(),
        }],
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow pixels"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shadow readback"),
        size: 64 * 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let view = texture.create_view(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow pixels"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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
                bytes_per_row: Some(256),
                rows_per_image: Some(64),
            },
        },
        wgpu::Extent3d {
            width: 64,
            height: 64,
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
