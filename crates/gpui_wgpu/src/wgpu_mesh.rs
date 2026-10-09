//! Draws [`PaintMesh`] primitives: lit triangle meshes with a depth buffer.
//!
//! Each frame, before the main pass, every mesh in the scene renders into its
//! own offscreen target: 4× multisampled color and depth, resolved into a
//! single-sample texture the size of the mesh's bounds in device pixels.
//! In the main pass, a [`PrimitiveBatch::Meshes`] batch composites those
//! textures like images (content mask, rounded corners, opacity), so meshes
//! keep the scene's 2D draw order and never need a depth buffer there.
//!
//! Geometry uploads once per [`MeshId`] and stays while frames keep painting
//! it; a mesh unpainted for [`EVICT_AFTER_FRAMES`] frames is freed. Targets
//! are pooled by the mesh's index in the scene and resized as bounds change.
//!
//! [`PrimitiveBatch::Meshes`]: gpui::PrimitiveBatch::Meshes

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use gpui::{MeshId, PaintMesh};
use wgpu::util::DeviceExt as _;

/// The mesh pass, standalone.
const MESH_SHADERS: &str = include_str!("shaders_mesh.wgsl");

/// The composite, on the shared shader helpers and instance transport.
const COMPOSITE_SHADERS: &str = concat!(
    include_str!("shaders.wgsl"),
    include_str!("shaders_storage.wgsl"),
    include_str!("shaders_mesh_composite.wgsl"),
);

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Every WebGPU implementation supports 4× multisampling of both formats.
const SAMPLE_COUNT: u32 = 4;

/// Frames a cached mesh survives without being painted.
const EVICT_AFTER_FRAMES: u64 = 120;

/// The widest a target grows, per side, before the mesh renders smaller
/// and the composite scales it up.
const MAX_TARGET_SIDE: u32 = 4096;

/// Half the width of wireframe edges, in device pixels.
const EDGE_HALF_WIDTH: f32 = 0.6;

/// Floats per vertex: position, normal, color.
const VERTEX_FLOATS: usize = 3 + 3 + 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshUniforms {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    base_color: [f32; 4],
    edge_color: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeParams {
    bounds: [f32; 4],
    content_mask: [f32; 4],
    corner_radii: [f32; 4],
    params: [f32; 4],
}

/// Geometry uploaded for one [`MeshId`].
struct CachedGeometry {
    buffer: wgpu::Buffer,
    vertex_count: u32,
    last_painted: u64,
}

/// The offscreen target of one mesh in the scene.
struct Target {
    size: (u32, u32),
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    resolve_view: wgpu::TextureView,
    mesh_uniforms: wgpu::Buffer,
    mesh_bind_group: wgpu::BindGroup,
    composite_uniforms: wgpu::Buffer,
    composite_bind_group: wgpu::BindGroup,
}

/// Pipelines, cached geometry, and pooled targets for meshes.
pub(crate) struct MeshRenderer {
    mesh_pipeline: wgpu::RenderPipeline,
    mesh_layout: wgpu::BindGroupLayout,
    composite_pipeline: wgpu::RenderPipeline,
    composite_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// The format the composite pipeline draws into.
    scene_format: wgpu::TextureFormat,
    geometry: HashMap<MeshId, CachedGeometry>,
    targets: Vec<Target>,
    frame: u64,
}

impl MeshRenderer {
    /// Builds the pipelines; `globals_layout` is the scene's group 0 and
    /// `scene_format` the format of the targets the composite draws into.
    pub(crate) fn new(
        device: &wgpu::Device,
        globals_layout: &wgpu::BindGroupLayout,
        scene_format: wgpu::TextureFormat,
    ) -> Self {
        let mesh_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<MeshUniforms>() as u64),
                },
                count: None,
            }],
        });
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_composite_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<CompositeParams>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let mesh_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shaders"),
            source: wgpu::ShaderSource::Wgsl(MESH_SHADERS.into()),
        });
        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_pipeline_layout"),
            bind_group_layouts: &[Some(&mesh_layout)],
            immediate_size: 0,
        });
        let float = |offset: usize, components: usize, location: u32| wgpu::VertexAttribute {
            format: match components {
                3 => wgpu::VertexFormat::Float32x3,
                _ => wgpu::VertexFormat::Float32x4,
            },
            offset: (offset * 4) as u64,
            shader_location: location,
        };
        let attributes = [float(0, 3, 0), float(3, 3, 1), float(6, 4, 2)];
        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh"),
            layout: Some(&mesh_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mesh_module,
                entry_point: Some("vs_mesh"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: (VERTEX_FLOATS * 4) as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_module,
                entry_point: Some("fs_mesh"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Exported winding is unreliable; the shader lights both sides.
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let composite_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_composite_shaders"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADERS.into()),
        });
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mesh_composite_pipeline_layout"),
                bind_group_layouts: &[Some(globals_layout), Some(&composite_layout)],
                immediate_size: 0,
            });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_composite"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &composite_module,
                entry_point: Some("vs_mesh_composite"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_module,
                entry_point: Some("fs_mesh_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: scene_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mesh_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            mesh_pipeline,
            mesh_layout,
            composite_pipeline,
            composite_layout,
            sampler,
            scene_format,
            geometry: HashMap::new(),
            targets: Vec::new(),
            frame: 0,
        }
    }

    /// The format the composite pipeline was built for.
    pub(crate) fn scene_format(&self) -> wgpu::TextureFormat {
        self.scene_format
    }

    /// Renders every mesh into its offscreen target. Call once per frame
    /// before the main pass, with the scene's meshes in scene order.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        meshes: &[PaintMesh],
    ) {
        self.frame += 1;
        let max_side = device
            .limits()
            .max_texture_dimension_2d
            .min(MAX_TARGET_SIDE);
        self.targets.truncate(meshes.len());
        for (index, paint) in meshes.iter().enumerate() {
            let side = |length: f32| (length.ceil().max(1.0) as u32).min(max_side);
            let size = (
                side(paint.bounds.size.width.0),
                side(paint.bounds.size.height.0),
            );
            if self
                .targets
                .get(index)
                .is_none_or(|target| target.size != size)
            {
                let target = self.create_target(device, size);
                if index < self.targets.len() {
                    self.targets[index] = target;
                } else {
                    self.targets.push(target);
                }
            }
            let frame = self.frame;
            let geometry = self
                .geometry
                .entry(paint.mesh.id())
                .or_insert_with(|| upload(device, paint));
            geometry.last_painted = frame;

            let target = &self.targets[index];
            let style = &paint.style;
            let [r, g, b] = style.base_color;
            let uniforms = MeshUniforms {
                view: style.view,
                projection: style.projection,
                base_color: [r, g, b, 1.0],
                edge_color: style.edge_color,
                params: [EDGE_HALF_WIDTH, 0.0, 0.0, 0.0],
            };
            queue.write_buffer(&target.mesh_uniforms, 0, bytemuck::bytes_of(&uniforms));
            let bounds = |b: gpui::Bounds<gpui::ScaledPixels>| {
                [b.origin.x.0, b.origin.y.0, b.size.width.0, b.size.height.0]
            };
            let radii = paint.corner_radii;
            let composite = CompositeParams {
                bounds: bounds(paint.bounds),
                content_mask: bounds(paint.content_mask.bounds),
                corner_radii: [
                    radii.top_left.0,
                    radii.top_right.0,
                    radii.bottom_right.0,
                    radii.bottom_left.0,
                ],
                params: [paint.opacity, 0.0, 0.0, 0.0],
            };
            queue.write_buffer(
                &target.composite_uniforms,
                0,
                bytemuck::bytes_of(&composite),
            );

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.msaa_view,
                    resolve_target: Some(&target.resolve_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Discard,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.mesh_pipeline);
            pass.set_bind_group(0, &target.mesh_bind_group, &[]);
            pass.set_vertex_buffer(0, geometry.buffer.slice(..));
            pass.draw(0..geometry.vertex_count, 0..1);
        }
        let frame = self.frame;
        self.geometry
            .retain(|_, geometry| frame - geometry.last_painted <= EVICT_AFTER_FRAMES);
    }

    /// Composites the scene's meshes in `range`, prepared this frame.
    pub(crate) fn composite(
        &self,
        range: Range<usize>,
        globals: &wgpu::BindGroup,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, globals, &[]);
        for target in self.targets.get(range).unwrap_or_default() {
            pass.set_bind_group(1, &target.composite_bind_group, &[]);
            pass.draw(0..4, 0..1);
        }
    }

    fn create_target(&self, device: &wgpu::Device, size: (u32, u32)) -> Target {
        let texture = |label: &str, format: wgpu::TextureFormat, samples: u32, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let view =
            |texture: wgpu::Texture| texture.create_view(&wgpu::TextureViewDescriptor::default());
        let msaa_view = view(texture(
            "mesh_msaa",
            COLOR_FORMAT,
            SAMPLE_COUNT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ));
        let depth_view = view(texture(
            "mesh_depth",
            DEPTH_FORMAT,
            SAMPLE_COUNT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ));
        let resolve_view = view(texture(
            "mesh_resolve",
            COLOR_FORMAT,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        ));
        let uniform_buffer = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mesh_uniforms = uniform_buffer("mesh_uniforms", std::mem::size_of::<MeshUniforms>());
        let composite_uniforms = uniform_buffer(
            "mesh_composite_uniforms",
            std::mem::size_of::<CompositeParams>(),
        );
        let mesh_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_bind_group"),
            layout: &self.mesh_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: mesh_uniforms.as_entire_binding(),
            }],
        });
        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_composite_bind_group"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: composite_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&resolve_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        Target {
            size,
            msaa_view,
            depth_view,
            resolve_view,
            mesh_uniforms,
            mesh_bind_group,
            composite_uniforms,
            composite_bind_group,
        }
    }
}

/// Uploads a mesh's vertices as one interleaved vertex buffer.
fn upload(device: &wgpu::Device, paint: &PaintMesh) -> CachedGeometry {
    let vertices = paint.mesh.vertices();
    let data: Vec<f32> = vertices
        .iter()
        .flat_map(|vertex| {
            let [px, py, pz] = vertex.position;
            let [nx, ny, nz] = vertex.normal;
            let [r, g, b, a] = vertex.color;
            [px, py, pz, nx, ny, nz, r, g, b, a]
        })
        .collect();
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_vertices"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::VERTEX,
    });
    CachedGeometry {
        buffer,
        vertex_count: u32::try_from(vertices.len()).unwrap_or(u32::MAX),
        last_painted: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{COMPOSITE_SHADERS, MESH_SHADERS};

    fn validate(source: &str) {
        let module = naga::front::wgsl::parse_str(source).expect("shader should parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("shader should validate");
    }

    #[test]
    fn mesh_shaders_are_valid_wgsl() {
        validate(MESH_SHADERS);
        validate(COMPOSITE_SHADERS);
    }
}
