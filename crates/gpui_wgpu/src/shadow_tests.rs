//! Pixel coverage using the shipping shadow shader and instance layout.
use super::gpu_test_support::{PIXEL_FORMAT, render_pixels, test_device};
use super::*;
use gpui::{ContentMask, Corners, hsla, point, size};
use wgpu::util::DeviceExt as _;

#[test]
fn outer_shadows_do_not_fill_translucent_panels() -> Result<()> {
    let Some((device, queue)) = test_device("shadow pixel regression")? else {
        return Ok(());
    };
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
                format: PIXEL_FORMAT,
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
            pulse_clock_ms: 0,
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
    render_pixels(
        device,
        queue,
        device.create_command_encoder(&Default::default()),
        (64, 64),
        wgpu::Color::BLACK,
        |pass| {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, globals, &[]);
            pass.set_bind_group(1, &instances_group, &[]);
            pass.draw(0..4, 0..1);
        },
    )
}
