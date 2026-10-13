//! Opt-in accounting of allocations requested by GPUI, not driver heaps or process RSS.
use super::WgpuRenderer;
use crate::{wgpu_atlas::AtlasMemoryStats, wgpu_mesh::MeshMemoryStats};
use gpui::Scene;

pub(super) struct RendererMemoryDiagnostics {
    frame: u64,
    previous: Option<RendererMemoryStats>,
}

impl RendererMemoryDiagnostics {
    pub(super) fn from_env() -> Option<Box<Self>> {
        (std::env::var("GPUI_RENDERER_DIAGNOSTICS").as_deref() == Ok("1")).then(|| {
            Box::new(Self {
                frame: 0,
                previous: None,
            })
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
struct RendererMemoryStats {
    viewport: (u32, u32),
    path_sample_count: u32,
    instance_capacity_bytes: u64,
    uniform_buffer_bytes: u64,
    path_texture_bytes: u64,
    offscreen_texture_bytes: u64,
    filter_group_texture_bytes: u64,
    capture_buffer_bytes: u64,
    scene_primitive_capacity_bytes: usize,
    scene_path_vertex_capacity_bytes: usize,
    path_batch_bounds: Vec<[u32; 4]>,
    pipeline_count: usize,
    atlas: AtlasMemoryStats,
    meshes: MeshMemoryStats,
}

impl WgpuRenderer {
    pub(super) fn report_memory_allocations(&mut self, scene: &Scene) {
        // No counters, scans, clock reads, or logging in ordinary builds. Diagnostics
        // only run on frames already being presented and never schedule another frame.
        let Some(mut diagnostics) = self.memory_diagnostics.take() else {
            return;
        };
        diagnostics.frame += 1;
        let resources = self.resources();
        let texture_sum = |textures: &[&Option<wgpu::Texture>]| {
            textures
                .iter()
                .filter_map(|texture| texture.as_ref())
                .map(texture_bytes)
                .sum()
        };
        let stats = RendererMemoryStats {
            viewport: (self.surface_config.width, self.surface_config.height),
            path_sample_count: self.rendering_params.path_sample_count,
            instance_capacity_bytes: self.instance_data_capacity,
            uniform_buffer_bytes: resources.globals_buffer.size()
                + resources.surface_uniform_buffer.size()
                + resources.blur_params_buffer.size(),
            path_texture_bytes: texture_sum(&[
                &resources.path_intermediate_texture,
                &resources.path_msaa_texture,
            ]),
            offscreen_texture_bytes: texture_sum(&[
                &resources.scene_color_texture,
                &resources.blur_ping_texture,
                &resources.blur_pong_texture,
            ]),
            filter_group_texture_bytes: resources.group_textures.iter().map(texture_bytes).sum(),
            capture_buffer_bytes: self
                .pending_capture
                .borrow()
                .as_ref()
                .map_or(0, |capture| capture.staging.buffer.size()),
            scene_primitive_capacity_bytes: scene_primitive_capacity_bytes(scene),
            scene_path_vertex_capacity_bytes: scene
                .paths
                .iter()
                .map(|path| vec_capacity_bytes(&path.vertices))
                .sum(),
            path_batch_bounds: scene
                .batches()
                .filter_map(|batch| {
                    let gpui::PrimitiveBatch::Paths(range) = batch else {
                        return None;
                    };
                    Some(super::path_batch_bounds(
                        &scene.paths[range],
                        [self.surface_config.width, self.surface_config.height],
                    ))
                })
                .collect(),
            pipeline_count: resources.pipelines.compiled_count()
                + 2 * usize::from(resources.meshes.is_some()),
            atlas: self.atlas.memory_stats(),
            meshes: resources
                .meshes
                .as_ref()
                .map(|meshes| meshes.memory_stats())
                .unwrap_or_default(),
        };
        if diagnostics.previous.as_ref() != Some(&stats) {
            let requested_gpu_bytes = stats.instance_capacity_bytes
                + stats.uniform_buffer_bytes
                + stats.path_texture_bytes
                + stats.offscreen_texture_bytes
                + stats.filter_group_texture_bytes
                + stats.capture_buffer_bytes
                + stats.atlas.texture_bytes
                + stats.meshes.target_bytes
                + stats.meshes.vertex_buffer_bytes
                + stats.meshes.uniform_buffer_bytes;
            tracing::info!(
                target: "gpui::renderer_memory",
                frame = diagnostics.frame,
                adapter = self.adapter_info.name,
                backend = ?self.adapter_info.backend,
                device_type = ?self.adapter_info.device_type,
                driver = self.adapter_info.driver,
                driver_info = self.adapter_info.driver_info,
                requested_gpu_bytes,
                quads = scene.quads.len(),
                paths = scene.paths.len(),
                shadows = scene.shadows.len(),
                monochrome_sprites = scene.monochrome_sprites.len(),
                subpixel_sprites = scene.subpixel_sprites.len(),
                polychrome_sprites = scene.polychrome_sprites.len(),
                mesh_count = scene.meshes.len(),
                backdrop_filters = scene.backdrop_filters.len(),
                filter_boundaries = scene.filter_boundaries.len(),
                allocations = ?stats,
                "GPUI allocation state changed; GPU bytes exclude swapchain, driver heaps, pipeline/compiler memory and allocator padding"
            );
            if let Some(report) = resources.device.generate_allocator_report() {
                let block_sizes: Vec<_> = report.blocks.iter().map(|block| block.size).collect();
                tracing::info!(
                    target: "gpui::renderer_memory",
                    frame = diagnostics.frame,
                    allocator_live_bytes = report.total_allocated_bytes,
                    allocator_reserved_bytes = report.total_reserved_bytes,
                    allocator_block_sizes = ?block_sizes,
                    allocator = %format_args!("{report:.8?}"),
                    "GPU backend suballocation report; reserved bytes are not process RSS"
                );
            }
            diagnostics.previous = Some(stats);
        }
        self.memory_diagnostics = Some(diagnostics);
    }
}

// All GPUI-owned atlas and intermediate textures are uncompressed, single-mip 2D
// textures. Count texel storage from their actual descriptors, excluding driver padding.
pub(crate) fn texture_bytes(texture: &wgpu::Texture) -> u64 {
    u64::from(texture.width())
        * u64::from(texture.height())
        * u64::from(texture.depth_or_array_layers())
        * u64::from(texture.sample_count())
        * u64::from(texture.format().block_copy_size(None).unwrap_or(0))
}

fn vec_capacity_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity() * std::mem::size_of::<T>()
}

fn scene_primitive_capacity_bytes(scene: &Scene) -> usize {
    vec_capacity_bytes(&scene.quads)
        + vec_capacity_bytes(&scene.shadows)
        + vec_capacity_bytes(&scene.paths)
        + vec_capacity_bytes(&scene.underlines)
        + vec_capacity_bytes(&scene.monochrome_sprites)
        + vec_capacity_bytes(&scene.subpixel_sprites)
        + vec_capacity_bytes(&scene.polychrome_sprites)
        + vec_capacity_bytes(&scene.surfaces)
        + vec_capacity_bytes(&scene.meshes)
        + vec_capacity_bytes(&scene.backdrop_filters)
        + vec_capacity_bytes(&scene.filter_boundaries)
}
