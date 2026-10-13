//! Compile optional pipeline groups only when a scene first uses them.
use super::{STORAGE_BUFFER_SHADERS, SUBPIXEL_SHADERS, WEBGL_SHADERS, WgpuBindGroupLayouts};

pub(super) struct WgpuPipelines {
    pub(super) quads: wgpu::RenderPipeline,
    pub(super) shadows: wgpu::RenderPipeline,
    pub(super) underlines: wgpu::RenderPipeline,
    pub(super) mono_sprites: wgpu::RenderPipeline,
    pub(super) subpixel_sprites: Option<wgpu::RenderPipeline>,
    pub(super) poly_sprites: wgpu::RenderPipeline,
    pub(super) path_pipelines: Option<PathPipelines>,
    #[allow(dead_code)]
    pub(super) surfaces: Option<wgpu::RenderPipeline>,
    // Capture only needs this copy/downsample pipeline, not the Gaussian blur shaders.
    pub(super) blur_downsample: Option<wgpu::RenderPipeline>,
    pub(super) blur_pipelines: Option<BlurPipelines>,
    // wgpu pipelines already retain this module internally. Reuse its Naga IR when an
    // optional group is compiled instead of parsing and retaining another complete copy.
    shader_module: wgpu::ShaderModule,
    color_target: wgpu::ColorTargetState,
    path_sample_count: u32,
}

pub(super) struct PathPipelines {
    pub(super) rasterization: wgpu::RenderPipeline,
    pub(super) composite: wgpu::RenderPipeline,
}

pub(super) struct BlurPipelines {
    pub(super) gaussian: wgpu::RenderPipeline,
    pub(super) composite: wgpu::RenderPipeline,
}

impl WgpuPipelines {
    pub(super) fn compiled_count(&self) -> usize {
        5 + usize::from(self.subpixel_sprites.is_some())
            + 2 * usize::from(self.path_pipelines.is_some())
            + usize::from(self.surfaces.is_some())
            + usize::from(self.blur_downsample.is_some())
            + 2 * usize::from(self.blur_pipelines.is_some())
    }

    pub(super) fn new(
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
        surface_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
        path_sample_count: u32,
        dual_source_blending: bool,
        uses_webgl_instance_data: bool,
    ) -> Self {
        // Diagnostic guard: verify the device actually has
        // DUAL_SOURCE_BLENDING. We have a crash report (ZED-5G1) where a
        // feature mismatch caused a wgpu-hal abort, but we haven't
        // identified the code path that produces the mismatch. This
        // guard prevents the crash and logs more evidence.
        // Remove this check once:
        // a) We find and fix the root cause, or
        // b) There are no reports of this warning appearing for some time.
        let device_has_feature = device
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING);
        if dual_source_blending && !device_has_feature {
            log::error!(
                "BUG: dual_source_blending flag is true but device does not \
                 have DUAL_SOURCE_BLENDING enabled (device features: {:?}). \
                 Falling back to mono text rendering. Please report this at \
                 https://github.com/zed-industries/zed/issues",
                device.features(),
            );
        }
        let dual_source_blending =
            dual_source_blending && device_has_feature && !uses_webgl_instance_data;

        let shader_source = if uses_webgl_instance_data {
            WEBGL_SHADERS
        } else {
            STORAGE_BUFFER_SHADERS
        };
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpui_shaders"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let subpixel_shader_module = if dual_source_blending {
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpui_subpixel_shaders"),
                source: wgpu::ShaderSource::Wgsl(SUBPIXEL_SHADERS.into()),
            }))
        } else {
            None
        };

        let blend_mode = match alpha_mode {
            wgpu::CompositeAlphaMode::PreMultiplied => {
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
            }
            _ => wgpu::BlendState::ALPHA_BLENDING,
        };

        let color_target = wgpu::ColorTargetState {
            format: surface_format,
            blend: Some(blend_mode),
            write_mask: wgpu::ColorWrites::ALL,
        };

        let quads = create_pipeline(
            device,
            "quads",
            "vs_quad",
            "fs_quad",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let shadows = create_pipeline(
            device,
            "shadows",
            "vs_shadow",
            "fs_shadow",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let underlines = create_pipeline(
            device,
            "underlines",
            "vs_underline",
            "fs_underline",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let mono_sprites = create_pipeline(
            device,
            "mono_sprites",
            "vs_mono_sprite",
            "fs_mono_sprite",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        let subpixel_sprites = if let Some(subpixel_module) = &subpixel_shader_module {
            let subpixel_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Src1,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc1,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            };

            Some(create_pipeline(
                device,
                "subpixel_sprites",
                "vs_subpixel_sprite",
                "fs_subpixel_sprite",
                &layouts.globals,
                &layouts.instances,
                Some(&layouts.texture),
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(subpixel_blend),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                1,
                subpixel_module,
            ))
        } else {
            None
        };

        let poly_sprites = create_pipeline(
            device,
            "poly_sprites",
            "vs_poly_sprite",
            "fs_poly_sprite",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(color_target.clone())],
            1,
            &shader_module,
        );

        Self {
            quads,
            shadows,
            underlines,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
            path_pipelines: None,
            surfaces: None,
            blur_downsample: None,
            blur_pipelines: None,
            shader_module,
            color_target,
            path_sample_count,
        }
    }

    pub(super) fn ensure_paths(&mut self, device: &wgpu::Device, layouts: &WgpuBindGroupLayouts) {
        if self.path_pipelines.is_some() {
            return;
        }
        let shader_module = &self.shader_module;
        let surface_format = self.color_target.format;
        let path_sample_count = self.path_sample_count;
        let path_rasterization = create_pipeline(
            device,
            "path_rasterization",
            "vs_path_rasterization",
            "fs_path_rasterization",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            path_sample_count,
            shader_module,
        );

        let paths_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let paths = create_pipeline(
            device,
            "paths",
            "vs_path",
            "fs_path",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(paths_blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            1,
            shader_module,
        );

        self.path_pipelines = Some(PathPipelines {
            rasterization: path_rasterization,
            composite: paths,
        });
    }

    pub(super) fn ensure_offscreen(
        &mut self,
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
        needs_blur: bool,
    ) {
        let shader_module = &self.shader_module;
        let surface_format = self.color_target.format;
        if self.blur_downsample.is_none() {
            // Blur pipelines all sample one texture into another; the downsample and gaussian passes
            // overwrite their (intermediate) target, while the composite blends over the scene.
            let no_blend_target = wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            };

            let blur_downsample = create_pipeline(
                device,
                "blur_downsample",
                "vs_blur_fullscreen",
                "fs_blur_downsample",
                &layouts.globals,
                &layouts.blur,
                None,
                wgpu::PrimitiveTopology::TriangleList,
                &[Some(no_blend_target)],
                1,
                shader_module,
            );

            self.blur_downsample = Some(blur_downsample);
        }
        if needs_blur && self.blur_pipelines.is_none() {
            // Blur pipelines all sample one texture into another; the downsample and gaussian passes
            // overwrite their (intermediate) target, while the composite blends over the scene.
            let no_blend_target = wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            };

            let blur = create_pipeline(
                device,
                "blur",
                "vs_blur_fullscreen",
                "fs_blur",
                &layouts.globals,
                &layouts.blur,
                None,
                wgpu::PrimitiveTopology::TriangleList,
                &[Some(no_blend_target)],
                1,
                shader_module,
            );

            // The blurred sample is premultiplied (blurring against the transparent, rgb=0 region
            // around the source scales rgb with the fading alpha), so the composite outputs
            // premultiplied and blends premultiplied — straight alpha blending would multiply rgb by
            // alpha a second time and darken the faded edges. Independent of the window's alpha mode.
            let premultiplied_target = wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            };
            let blur_composite = create_pipeline(
                device,
                "blur_composite",
                "vs_blur_composite",
                "fs_blur_composite",
                &layouts.globals,
                &layouts.blur,
                None,
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(premultiplied_target)],
                1,
                shader_module,
            );

            self.blur_pipelines = Some(BlurPipelines {
                gaussian: blur,
                composite: blur_composite,
            });
        }
    }

    pub(super) fn ensure_surfaces(
        &mut self,
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
    ) {
        if self.surfaces.is_some() {
            return;
        }
        let shader_module = &self.shader_module;
        let surfaces = create_pipeline(
            device,
            "surfaces",
            "vs_surface",
            "fs_surface",
            &layouts.globals,
            &layouts.surfaces,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(self.color_target.clone())],
            1,
            shader_module,
        );

        self.surfaces = Some(surfaces);
    }
}

#[allow(clippy::too_many_arguments)]
fn create_pipeline(
    device: &wgpu::Device,
    name: &str,
    vs_entry: &str,
    fs_entry: &str,
    globals_layout: &wgpu::BindGroupLayout,
    data_layout: &wgpu::BindGroupLayout,
    texture_layout: Option<&wgpu::BindGroupLayout>,
    topology: wgpu::PrimitiveTopology,
    color_targets: &[Option<wgpu::ColorTargetState>],
    sample_count: u32,
    module: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    let bind_group_layouts = [Some(globals_layout), Some(data_layout), texture_layout];
    let layout_count = if texture_layout.is_some() { 3 } else { 2 };
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{name}_layout")),
        bind_group_layouts: &bind_group_layouts[..layout_count],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vs_entry),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs_entry),
            targets: color_targets,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use crate::WgpuRenderer;
    use gpui::block_on;

    #[test]
    fn optional_pipelines_compile_on_demand_and_are_reused() -> anyhow::Result<()> {
        let (adapter, device) = block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await?;
            let (device, _) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("lazy pipeline regression"),
                    required_features: adapter.features() & wgpu::Features::DUAL_SOURCE_BLENDING,
                    required_limits: wgpu::Limits::downlevel_defaults()
                        .using_resolution(adapter.limits())
                        .using_alignment(adapter.limits()),
                    memory_hints: crate::WgpuContext::memory_hints(),
                    ..Default::default()
                })
                .await?;
            anyhow::Ok((adapter, device))
        })?;

        for (format, alpha_mode, webgl) in [
            (
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::CompositeAlphaMode::Opaque,
                false,
            ),
            (
                wgpu::TextureFormat::Bgra8Unorm,
                wgpu::CompositeAlphaMode::PreMultiplied,
                false,
            ),
            (
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::CompositeAlphaMode::Opaque,
                true,
            ),
        ] {
            let layouts = WgpuRenderer::create_bind_group_layouts(&device, webgl);
            let path_sample_count = if adapter
                .get_texture_format_features(format)
                .flags
                .sample_count_supported(4)
            {
                4
            } else {
                1
            };
            let mut pipelines = WgpuPipelines::new(
                &device,
                &layouts,
                format,
                alpha_mode,
                path_sample_count,
                device
                    .features()
                    .contains(wgpu::Features::DUAL_SOURCE_BLENDING),
                webgl,
            );
            assert!(pipelines.path_pipelines.is_none());
            assert!(pipelines.surfaces.is_none());
            assert!(pipelines.blur_downsample.is_none());
            assert!(pipelines.blur_pipelines.is_none());

            // A capture before any filter must not compile the Gaussian blur pipeline.
            pipelines.ensure_offscreen(&device, &layouts, false);
            let blit = pipelines
                .blur_downsample
                .clone()
                .expect("capture compiles blit");
            assert!(pipelines.blur_pipelines.is_none());
            pipelines.ensure_offscreen(&device, &layouts, true);
            assert_eq!(pipelines.blur_downsample.as_ref(), Some(&blit));
            let blur = pipelines.blur_pipelines.as_ref().unwrap().gaussian.clone();
            pipelines.ensure_offscreen(&device, &layouts, true);
            assert_eq!(pipelines.blur_pipelines.as_ref().unwrap().gaussian, blur);

            pipelines.ensure_paths(&device, &layouts);
            let path = pipelines
                .path_pipelines
                .as_ref()
                .unwrap()
                .rasterization
                .clone();
            pipelines.ensure_paths(&device, &layouts);
            assert_eq!(
                pipelines.path_pipelines.as_ref().unwrap().rasterization,
                path
            );

            pipelines.ensure_surfaces(&device, &layouts);
            let surface = pipelines
                .surfaces
                .clone()
                .expect("surface pipeline created");
            pipelines.ensure_surfaces(&device, &layouts);
            assert_eq!(pipelines.surfaces.as_ref(), Some(&surface));
        }
        Ok(())
    }
}
