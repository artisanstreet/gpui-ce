use std::{
    slice,
    sync::{Arc, OnceLock},
};

use anyhow::{Context, Result};
use gpui_util::ResultExt;
#[cfg(not(feature = "cosmic-text"))]
use windows::Win32::Graphics::DirectWrite::*;
use windows::{
    Win32::{
        Foundation::{HWND, RECT},
        Graphics::{
            Direct3D::*,
            Direct3D11::*,
            DirectComposition::*,
            Dxgi::{Common::*, *},
        },
    },
    core::{HSTRING, Interface},
};

use crate::directx_renderer::shader_resources::{RawShaderBytes, ShaderModule, ShaderTarget};
use crate::*;
use gpui::*;

use gpui::path_targets::{path_batch_bounds, path_target_size, retained_path_target_size};

/// The largest blur radius in a scene-space filter chain, in device pixels — used to size the
/// blur kernel and the dilated region the blur passes are scissored to.
///
/// The `match` is exhaustive on purpose: adding a [`ScaledFilter`] variant breaks it here,
/// forcing this backend to handle (or deliberately ignore) the new filter rather than silently
/// dropping it.
fn max_blur_radius(filters: &[ScaledFilter]) -> f32 {
    filters.iter().fold(0.0, |acc, filter| match filter {
        ScaledFilter::Blur(radius) => acc.max(radius.0),
    })
}

pub(crate) const DISABLE_DIRECT_COMPOSITION: &str = "GPUI_DISABLE_DIRECT_COMPOSITION";
const RENDER_TARGET_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;
// This configuration is used for MSAA rendering on paths only, and it's guaranteed to be supported by DirectX 11.
const PATH_MULTISAMPLE_COUNT: u32 = 4;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;

/// Number of content-filter (`filter`) nesting levels that get their own isolated group target.
/// Two covers the realistic "a blurred element inside another blurred element" case; deeper nests
/// render inline (unblurred at the inner level) rather than allocating unbounded VRAM. Must match
/// the wgpu backend's `MAX_FILTER_DEPTH` so nested blur renders consistently across platforms.
const MAX_FILTER_DEPTH: usize = 2;

fn filter_group_depth(boundaries: &[FilterBoundary]) -> usize {
    let mut depth = 0usize;
    let mut maximum = 0;
    for boundary in boundaries {
        if boundary.is_start {
            depth += 1;
            maximum = maximum.max(depth);
            if maximum == MAX_FILTER_DEPTH {
                break;
            }
        } else {
            depth = depth.saturating_sub(1);
        }
    }
    maximum
}

pub(crate) struct FontInfo {
    pub gamma_ratios: [f32; 4],
    pub grayscale_enhanced_contrast: f32,
    pub subpixel_enhanced_contrast: f32,
    pub is_bgr: bool,
}

pub(crate) struct DirectXRenderer {
    hwnd: HWND,
    atlas: Arc<DirectXAtlas>,
    devices: Option<DirectXRendererDevices>,
    resources: Option<DirectXResources>,
    globals: DirectXGlobalElements,
    pipelines: DirectXRenderPipelines,
    direct_composition: Option<DirectComposition>,
    font_info: &'static FontInfo,

    width: u32,
    height: u32,

    /// Whether we want to skip drawing due to device lost events.
    ///
    /// In that case we want to discard the first frame that we draw as we got reset in the middle of a frame
    /// meaning we lost all the allocated gpu textures and scene resources.
    skip_draws: bool,

    /// The render target currently bound for the main scene this frame (the offscreen
    /// `scene_color` when blur filters are present, a content-filter group texture inside such a
    /// group, or the swapchain otherwise). `draw_paths_to_intermediate` restores to this after
    /// its own pass so paths land on the correct target.
    active_render_target: Option<ID3D11RenderTargetView>,
    meshes: Option<crate::directx_mesh::MeshRenderer>,
}

/// Direct3D objects
#[derive(Clone)]
pub(crate) struct DirectXRendererDevices {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) dxgi_factory: IDXGIFactory6,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
    dxgi_device: Option<IDXGIDevice>,
    annotation: Option<ID3DUserDefinedAnnotation>,
}

struct DirectXResources {
    // Direct3D rendering objects
    swap_chain: IDXGISwapChain1,
    render_target: Option<ID3D11Texture2D>,
    render_target_view: Option<ID3D11RenderTargetView>,

    // One scratch pair shared by path batches, allocated only while paths are visible.
    paths: Option<PathTargets>,

    // Offscreen targets for blur filters (each is render-target + shader-resource).
    blur: Option<BlurResources>,

    // Cached viewport
    viewport: D3D11_VIEWPORT,
}

struct PathTargets {
    size: [u32; 2],
    viewport: D3D11_VIEWPORT,
    texture: ID3D11Texture2D,
    srv: Option<ID3D11ShaderResourceView>,
    msaa_texture: ID3D11Texture2D,
    msaa_view: Option<ID3D11RenderTargetView>,
}

impl PathTargets {
    fn new(device: &ID3D11Device, size: [u32; 2]) -> Result<Self> {
        let (texture, srv) = create_path_intermediate_texture(device, size[0], size[1])?;
        let (msaa_texture, msaa_view) =
            create_path_intermediate_msaa_texture_and_view(device, size[0], size[1])?;
        Ok(Self {
            size,
            viewport: D3D11_VIEWPORT {
                Width: size[0] as f32,
                Height: size[1] as f32,
                MaxDepth: 1.0,
                ..Default::default()
            },
            texture,
            srv,
            msaa_texture,
            msaa_view,
        })
    }
}

// Separate from batch/blur b1 and mesh b2/b3. All values are in device pixels.
#[repr(C)]
struct PathParams {
    origin: [f32; 2],
    size: [f32; 2],
}

/// Offscreen render targets used by the blur filters. The scene is rendered into `scene_color`
/// (so filters can sample it), `ping`/`pong` are half-resolution scratch for the separable
/// gaussian, and `groups` isolate content-filter (`filter`) subtrees — one per nesting level
/// (indexed by isolation depth), up to [`MAX_FILTER_DEPTH`], so nested content blurs isolate
/// correctly; deeper nests render inline.
struct BlurResources {
    #[cfg_attr(not(test), expect(dead_code))]
    scene_color: ID3D11Texture2D,
    scene_color_rtv: Option<ID3D11RenderTargetView>,
    scene_color_srv: Option<ID3D11ShaderResourceView>,
    #[cfg_attr(not(test), expect(dead_code))]
    ping: ID3D11Texture2D,
    ping_rtv: Option<ID3D11RenderTargetView>,
    ping_srv: Option<ID3D11ShaderResourceView>,
    #[cfg_attr(not(test), expect(dead_code))]
    pong: ID3D11Texture2D,
    pong_rtv: Option<ID3D11RenderTargetView>,
    pong_srv: Option<ID3D11ShaderResourceView>,
    // Kept alive for the lifetime of their views; indexed by isolation depth.
    groups: Vec<ID3D11Texture2D>,
    group_rtvs: Vec<Option<ID3D11RenderTargetView>>,
    group_srvs: Vec<Option<ID3D11ShaderResourceView>>,
}

impl BlurResources {
    fn new(device: &ID3D11Device, width: u32, height: u32) -> Result<Self> {
        let half_w = (width / 2).max(1);
        let half_h = (height / 2).max(1);
        let (scene_color, scene_color_rtv, scene_color_srv) =
            create_color_target(device, width, height)?;
        let (ping, ping_rtv, ping_srv) = create_color_target(device, half_w, half_h)?;
        let (pong, pong_rtv, pong_srv) = create_color_target(device, half_w, half_h)?;
        Ok(Self {
            scene_color,
            scene_color_rtv,
            scene_color_srv,
            ping,
            ping_rtv,
            ping_srv,
            pong,
            pong_rtv,
            pong_srv,
            groups: Vec::new(),
            group_rtvs: Vec::new(),
            group_srvs: Vec::new(),
        })
    }

    fn release_unused_groups(&mut self, content_depth: usize) {
        if content_depth == 0 {
            // Retain the high-water depth while content filters remain active,
            // but a persistent backdrop filter must not retain vanished groups.
            self.group_rtvs.clear();
            self.group_srvs.clear();
            self.groups.clear();
        }
    }

    fn ensure_groups(
        &mut self,
        device: &ID3D11Device,
        width: u32,
        height: u32,
        depth: usize,
    ) -> Result<()> {
        for _ in self.groups.len()..depth {
            let (texture, rtv, srv) = create_color_target(device, width, height)?;
            self.groups.push(texture);
            self.group_rtvs.push(rtv);
            self.group_srvs.push(srv);
        }
        Ok(())
    }
}

struct DirectXRenderPipelines {
    shadow_pipeline: PipelineState<Shadow>,
    quad_pipeline: PipelineState<Quad>,
    path_rasterization_pipeline: PipelineState<PathRasterizationSprite>,
    path_sprite_pipeline: PipelineState<PathSprite>,
    underline_pipeline: PipelineState<Underline>,
    mono_sprites: PipelineState<MonochromeSprite>,
    subpixel_sprites: PipelineState<SubpixelSprite>,
    poly_sprites: PipelineState<PolychromeSprite>,
    // Blur (backdrop-filter / filter). These don't use the generic PipelineState since they
    // sample a texture rather than read a structured instance buffer; their parameters live in
    // a dedicated constant buffer at register b1.
    blur_downsample_vertex: ID3D11VertexShader,
    blur_downsample_fragment: ID3D11PixelShader,
    blur_vertex: ID3D11VertexShader,
    blur_fragment: ID3D11PixelShader,
    blur_composite_vertex: ID3D11VertexShader,
    blur_composite_fragment: ID3D11PixelShader,
    blur_params_buffer: ID3D11Buffer,
    blur_blend_replace: ID3D11BlendState,
    blur_blend_composite: ID3D11BlendState,
    blur_rasterizer: ID3D11RasterizerState,
}

struct DirectXGlobalElements {
    global_params_buffer: Option<ID3D11Buffer>,
    batch_params_buffer: Option<ID3D11Buffer>,
    path_params_buffer: Option<ID3D11Buffer>,
    sampler: Option<ID3D11SamplerState>,
    blur_sampler: Option<ID3D11SamplerState>,
}

struct Annotation<'a>(&'a ID3DUserDefinedAnnotation);

impl<'a> Annotation<'a> {
    fn new(annotation: &'a ID3DUserDefinedAnnotation, label: HSTRING) -> Self {
        unsafe { annotation.BeginEvent(&label) };
        Self(annotation)
    }
}

impl Drop for Annotation<'_> {
    fn drop(&mut self) {
        unsafe { self.0.EndEvent() };
    }
}

struct DirectComposition {
    comp_device: IDCompositionDevice,
    comp_target: IDCompositionTarget,
    comp_visual: IDCompositionVisual,
}

impl DirectXRendererDevices {
    pub(crate) fn new(
        directx_devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        let DirectXDevices {
            adapter,
            dxgi_factory,
            device,
            device_context,
        } = directx_devices;
        let dxgi_device = if disable_direct_composition {
            None
        } else {
            Some(device.cast().context("Creating DXGI device")?)
        };
        let annotation = device_context.cast().ok();

        Ok(Self {
            adapter: adapter.clone(),
            dxgi_factory: dxgi_factory.clone(),
            device: device.clone(),
            device_context: device_context.clone(),
            dxgi_device,
            annotation,
        })
    }
}

impl DirectXRenderer {
    pub(crate) fn new(
        hwnd: HWND,
        directx_devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        if disable_direct_composition {
            log::info!("Direct Composition is disabled.");
        }

        let devices = DirectXRendererDevices::new(directx_devices, disable_direct_composition)
            .context("Creating DirectX devices")?;
        let atlas = Arc::new(DirectXAtlas::new(&devices.device, &devices.device_context));

        let resources = DirectXResources::new(&devices, 1, 1, hwnd, disable_direct_composition)
            .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectX global elements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectX render pipelines")?;

        let direct_composition = if disable_direct_composition {
            None
        } else {
            let composition = DirectComposition::new(devices.dxgi_device.as_ref().unwrap(), hwnd)
                .context("Creating DirectComposition")?;
            composition
                .set_swap_chain(&resources.swap_chain)
                .context("Setting swap chain for DirectComposition")?;
            Some(composition)
        };

        Ok(DirectXRenderer {
            hwnd,
            atlas,
            devices: Some(devices),
            resources: Some(resources),
            globals,
            pipelines,
            direct_composition,
            font_info: Self::get_font_info(),
            width: 1,
            height: 1,
            skip_draws: false,
            active_render_target: None,
            meshes: None,
        })
    }

    pub(crate) fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.atlas.clone()
    }

    fn pre_draw(&self, clear_color: &[f32; 4], pulse_clock_ms: u32) -> Result<()> {
        let resources = self.resources.as_ref().expect("resources missing");
        let device_context = &self
            .devices
            .as_ref()
            .expect("devices missing")
            .device_context;
        update_buffer(
            device_context,
            self.globals.global_params_buffer.as_ref().unwrap(),
            &[GlobalParams {
                gamma_ratios: self.font_info.gamma_ratios,
                viewport_size: [resources.viewport.Width, resources.viewport.Height],
                grayscale_enhanced_contrast: self.font_info.grayscale_enhanced_contrast,
                subpixel_enhanced_contrast: self.font_info.subpixel_enhanced_contrast,
                is_bgr: self.font_info.is_bgr as u32,
                pulse_clock_ms,
                _pad: [0; 2],
            }],
        )?;
        unsafe {
            device_context.ClearRenderTargetView(
                resources
                    .render_target_view
                    .as_ref()
                    .context("missing render target view")?,
                clear_color,
            );
            device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
            device_context.RSSetViewports(Some(slice::from_ref(&resources.viewport)));
            device_context
                .VSSetConstantBuffers(0, Some(slice::from_ref(&self.globals.global_params_buffer)));
            device_context
                .VSSetConstantBuffers(1, Some(slice::from_ref(&self.globals.batch_params_buffer)));
            device_context
                .PSSetConstantBuffers(0, Some(slice::from_ref(&self.globals.global_params_buffer)));
        }
        Ok(())
    }

    #[inline]
    fn present(&mut self) -> Result<()> {
        let result = unsafe {
            self.resources
                .as_ref()
                .expect("resources missing")
                .swap_chain
                .Present(0, DXGI_PRESENT(0))
        };
        result.ok().context("Presenting swap chain failed")
    }

    pub(crate) fn handle_device_lost(&mut self, directx_devices: &DirectXDevices) -> Result<()> {
        try_to_recover_from_device_lost(|| {
            self.handle_device_lost_impl(directx_devices)
                .context("DirectXRenderer handling device lost")
        })
    }

    fn handle_device_lost_impl(&mut self, directx_devices: &DirectXDevices) -> Result<()> {
        let disable_direct_composition = self.direct_composition.is_none();
        self.meshes = None;
        self.active_render_target = None;

        unsafe {
            #[cfg(debug_assertions)]
            if let Some(devices) = &self.devices {
                report_live_objects(&devices.device)
                    .context("Failed to report live objects after device lost")
                    .log_err();
            }

            self.resources.take();
            if let Some(devices) = &self.devices {
                devices.device_context.OMSetRenderTargets(None, None);
                devices.device_context.ClearState();
                devices.device_context.Flush();
                #[cfg(debug_assertions)]
                report_live_objects(&devices.device)
                    .context("Failed to report live objects after device lost")
                    .log_err();
            }

            self.direct_composition.take();
            self.devices.take();
        }

        let devices = DirectXRendererDevices::new(directx_devices, disable_direct_composition)
            .context("Recreating DirectX devices")?;
        let resources = DirectXResources::new(
            &devices,
            self.width,
            self.height,
            self.hwnd,
            disable_direct_composition,
        )
        .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectXGlobalElements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectXRenderPipelines")?;

        let direct_composition = if disable_direct_composition {
            None
        } else {
            let composition =
                DirectComposition::new(devices.dxgi_device.as_ref().unwrap(), self.hwnd)?;
            composition.set_swap_chain(&resources.swap_chain)?;
            Some(composition)
        };

        self.atlas
            .handle_device_lost(&devices.device, &devices.device_context);

        unsafe {
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
        }
        self.devices = Some(devices);
        self.resources = Some(resources);
        self.globals = globals;
        self.pipelines = pipelines;
        self.direct_composition = direct_composition;
        self.skip_draws = true;
        Ok(())
    }

    pub(crate) fn draw(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<()> {
        if self.skip_draws {
            // skip drawing this frame, we just recovered from a device lost event
            // and so likely do not have the textures anymore that are required for drawing
            return Ok(());
        }
        self.render(scene, background_appearance)?;
        self.present()
    }

    /// Clear the render target for `background_appearance` and encode every
    /// primitive batch of `scene` into it, without presenting. Shared by
    /// [`draw`](Self::draw) (which then presents) and
    /// [`render_to_image`](Self::render_to_image) (which reads the target back
    /// instead), so the two cannot drift.
    fn render(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<()> {
        if !scene.meshes.is_empty() || self.meshes.is_some() {
            let devices = self.devices.as_ref().context("devices missing")?;
            if self.meshes.is_none() {
                self.meshes = Some(crate::directx_mesh::MeshRenderer::new(&devices.device)?);
            }
            self.meshes.as_mut().unwrap().prepare(
                &devices.device,
                &devices.device_context,
                &scene.meshes,
                (self.width, self.height),
            )?;
        }
        self.pre_draw(
            &match background_appearance {
                WindowBackgroundAppearance::Opaque => [1.0f32; 4],
                _ => [0.0f32; 4],
            },
            scene.pulse_clock_ms,
        )?;
        self.upload_scene_buffers(scene)?;
        {
            let devices = self.devices.as_ref().context("devices missing")?;
            let resources = self.resources.as_mut().context("resources missing")?;
            // The previous frame may have left the scratch resolve bound to a shader.
            unsafe {
                devices
                    .device_context
                    .VSSetShaderResources(0, Some(&[None]));
                devices
                    .device_context
                    .PSSetShaderResources(0, Some(&[None]));
            }
            resources.ensure_paths(
                &devices.device,
                path_target_size(scene, [self.width, self.height]),
            )?;
        }

        // Only route through the offscreen scene texture when the scene contains blur filters;
        // otherwise render straight to the swapchain exactly as before.
        let use_offscreen =
            !scene.backdrop_filters.is_empty() || !scene.filter_boundaries.is_empty();
        let content_depth = filter_group_depth(&scene.filter_boundaries);
        let resources = self.resources.as_mut().context("resources missing")?;
        if use_offscreen {
            let device = &self.devices.as_ref().context("devices missing")?.device;
            if resources.blur.is_none() {
                resources.blur = Some(BlurResources::new(device, self.width, self.height)?);
            }
            resources.blur.as_mut().unwrap().ensure_groups(
                device,
                self.width,
                self.height,
                content_depth,
            )?;
        } else {
            // pre_draw rebound the swapchain, and blur passes unbind their SRVs.
            // The immediate context retains anything needed by pending GPU work.
            resources.blur = None;
        }

        // Clone the views we need (AddRef) so the loop can rebind render targets without holding a
        // borrow of `self` across the `&mut self` draw_* calls.
        let (scene_rtv, scene_srv, group_rtvs, group_srvs, swapchain_rtv) = {
            let r = self.resources.as_ref().context("resources missing")?;
            if let Some(blur) = r.blur.as_ref() {
                (
                    blur.scene_color_rtv.clone(),
                    blur.scene_color_srv.clone(),
                    blur.group_rtvs.clone(),
                    blur.group_srvs.clone(),
                    r.render_target_view.clone(),
                )
            } else {
                (
                    None,
                    None,
                    Vec::new(),
                    Vec::new(),
                    r.render_target_view.clone(),
                )
            }
        };
        let ctx = self
            .devices
            .as_ref()
            .context("devices missing")?
            .device_context
            .clone();

        if use_offscreen {
            unsafe {
                if let Some(rtv) = scene_rtv.as_ref() {
                    ctx.ClearRenderTargetView(rtv, &[0.0; 4]);
                }
                ctx.OMSetRenderTargets(Some(slice::from_ref(&scene_rtv)), None);
            }
            self.active_render_target = scene_rtv;
        } else {
            self.active_render_target = swapchain_rtv.clone();
        }

        // Current target for the main scene + a parent stack for content-filter groups.
        let mut current_rtv = self.active_render_target.clone();
        let mut current_srv = if use_offscreen {
            scene_srv.clone()
        } else {
            None
        };
        // (parent_rtv, parent_srv, isolated)
        let mut filter_stack: Vec<(
            Option<ID3D11RenderTargetView>,
            Option<ID3D11ShaderResourceView>,
            bool,
        )> = Vec::new();

        let annotation = self
            .devices
            .as_ref()
            .and_then(|devices| devices.annotation.clone())
            .filter(|annotation| unsafe { annotation.GetStatus().as_bool() });
        for batch in scene.batches() {
            let _annotation = annotation
                .as_ref()
                .map(|annotation| Annotation::new(annotation, HSTRING::from(batch.label())));
            match batch {
                PrimitiveBatch::Shadows(range) => self.draw_shadows(range.start, range.len()),
                PrimitiveBatch::Quads(range) => self.draw_quads(range.start, range.len()),
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    self.draw_paths_to_intermediate(paths)?;
                    self.draw_paths_from_intermediate(paths)
                }
                PrimitiveBatch::Underlines(range) => self.draw_underlines(range.start, range.len()),
                PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                    self.draw_monochrome_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::SubpixelSprites { texture_id, range } => {
                    self.draw_subpixel_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                    self.draw_polychrome_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(&scene.surfaces[range]),
                PrimitiveBatch::Meshes(range) => {
                    self.meshes.as_ref().context("mesh renderer missing")?.composite(&ctx, range);
                    Ok(())
                },
                PrimitiveBatch::BackdropFilters(range) => {
                    let result = (|| {
                        for filter in &scene.backdrop_filters[range] {
                            self.dx_blur_and_composite(
                                &current_srv,
                                &current_rtv,
                                filter.bounds,
                                filter.content_mask.bounds,
                                corner_radii_array(filter.corner_radii),
                                max_blur_radius(&filter.filters),
                                filter.opacity,
                                true,
                            )?;
                        }
                        Ok::<(), anyhow::Error>(())
                    })();
                    // Restore the current target for subsequent batches.
                    unsafe {
                        ctx.OMSetRenderTargets(Some(slice::from_ref(&current_rtv)), None);
                    }
                    result
                }
                PrimitiveBatch::FilterBoundary(ix) => {
                    let boundary = scene.filter_boundaries[ix].clone();
                    if boundary.is_start {
                        // Each isolated nesting level uses its own group target from the pool
                        // (indexed by current isolation depth). Beyond the pool size
                        // (MAX_FILTER_DEPTH) deeper filters render inline without isolation rather
                        // than corrupting an outer group.
                        let depth = filter_stack.iter().filter(|entry| entry.2).count();
                        if depth < group_rtvs.len() {
                            filter_stack.push((current_rtv.clone(), current_srv.clone(), true));
                            current_rtv = group_rtvs[depth].clone();
                            current_srv = group_srvs[depth].clone();
                            self.active_render_target = current_rtv.clone();
                            unsafe {
                                if let Some(rtv) = current_rtv.as_ref() {
                                    ctx.ClearRenderTargetView(rtv, &[0.0; 4]);
                                }
                                ctx.OMSetRenderTargets(Some(slice::from_ref(&current_rtv)), None);
                            }
                        } else {
                            filter_stack.push((current_rtv.clone(), current_srv.clone(), false));
                        }
                        Ok(())
                    } else if let Some((parent_rtv, parent_srv, isolated)) = filter_stack.pop() {
                        let result = if isolated {
                            self.dx_blur_and_composite(
                                &current_srv,
                                &parent_rtv,
                                boundary.bounds,
                                boundary.content_mask.bounds,
                                corner_radii_array(boundary.corner_radii),
                                max_blur_radius(&boundary.filters),
                                boundary.opacity,
                                false,
                            )
                        } else {
                            Ok(())
                        };
                        current_rtv = parent_rtv;
                        current_srv = parent_srv;
                        self.active_render_target = current_rtv.clone();
                        unsafe {
                            ctx.OMSetRenderTargets(Some(slice::from_ref(&current_rtv)), None);
                        }
                        result
                    } else {
                        Ok(())
                    }
                }
            }
            .with_context(|| {
                format!(
                    "scene too large:\
                    {} paths, {} shadows, {} quads, {} underlines, {} mono, {} subpixel, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.subpixel_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                )
            })?;
        }
        // Present the offscreen scene by blitting it into the swapchain.
        if use_offscreen {
            self.dx_blit(&scene_srv, &swapchain_rtv)?;
        }
        self.active_render_target = None;
        if let Some(blur) = self.resources.as_mut().and_then(|r| r.blur.as_mut()) {
            // Every blur pass and the final scene blit are now encoded. D3D11
            // retains resource dependencies of any pending GPU commands.
            blur.release_unused_groups(content_depth);
        }
        Ok(())
    }

    /// Render `scene` to an offscreen CPU image **without presenting** so
    /// the window need never be shown or visible (the macOS headless path
    /// goes through MetalRenderer; this is the Windows analogue). Draws into
    /// the existing render target, copies it into a `D3D11_USAGE_STAGING`
    /// texture, maps it, and converts BGRA to RGBA.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn render_to_image(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<image::RgbaImage> {
        // A pending device-lost recovery (`skip_draws`) leaves the atlas holding
        // tile references from the previous device; drawing before the forced
        // re-render rebuilds them panics in `DirectXAtlasState::texture`.
        anyhow::ensure!(
            !self.skip_draws,
            "render_to_image unavailable while recovering from a lost device"
        );
        self.render(scene, background_appearance)?;

        let devices = self.devices.as_ref().context("devices missing")?;
        let device = &devices.device;
        let context = &devices.device_context;
        let resources = self.resources.as_ref().context("resources missing")?;
        let render_target = resources
            .render_target
            .as_ref()
            .context("render target missing")?;

        // A CPU-readable copy of the render target.
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { render_target.GetDesc(&mut desc) };
        let width = desc.Width;
        let height = desc.Height;
        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            MipLevels: 1,
            ArraySize: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..desc
        };
        let mut staging: Option<ID3D11Texture2D> = None;
        unsafe { device.CreateTexture2D(&staging_desc, None, Some(&mut staging))? };
        let staging = staging.context("creating staging texture")?;
        unsafe { context.CopyResource(&staging, render_target) };

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { context.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))? };
        let row_bytes = (width as usize) * 4;
        let mut pixels = vec![0u8; row_bytes * height as usize];
        // SAFETY: `Map` succeeded, so `pData` points at `RowPitch * height`
        // readable bytes for as long as the mapping is held, and `RowPitch >=
        // row_bytes` (it only ever adds trailing padding). `pixels` is sized
        // `row_bytes * height`, so every copy stays in bounds on both sides,
        // and the regions cannot overlap (`pixels` is a fresh allocation).
        unsafe {
            let src = mapped.pData as *const u8;
            for row in 0..height as usize {
                let s = src.add(row * mapped.RowPitch as usize);
                let d = pixels.as_mut_ptr().add(row * row_bytes);
                std::ptr::copy_nonoverlapping(s, d, row_bytes);
            }
            context.Unmap(&staging, 0);
        }
        // The render target is BGRA; image::RgbaImage expects RGBA.
        for px in pixels.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        image::RgbaImage::from_raw(width, height, pixels)
            .context("Failed to build RgbaImage from staging readback")
    }

    pub(crate) fn resize(&mut self, new_size: Size<DevicePixels>) -> Result<()> {
        let width = new_size.width.0.max(1) as u32;
        let height = new_size.height.0.max(1) as u32;
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.width = width;
        self.height = height;

        // Clear the render target before resizing
        let devices = self.devices.as_ref().context("devices missing")?;
        unsafe { devices.device_context.OMSetRenderTargets(None, None) };
        // A failed render may still retain its backbuffer view. DXGI requires
        // every reference to the old buffers to be released before ResizeBuffers.
        self.active_render_target = None;
        let resources = self.resources.as_mut().context("resources missing")?;
        resources.render_target.take();
        resources.render_target_view.take();

        // Resizing the swap chain requires a call to the underlying DXGI adapter, which can return the device removed error.
        // The app might have moved to a monitor that's attached to a different graphics device.
        // When a graphics device is removed or reset, the desktop resolution often changes, resulting in a window size change.
        // But here we just return the error, because we are handling device lost scenarios elsewhere.
        unsafe {
            resources
                .swap_chain
                .ResizeBuffers(
                    BUFFER_COUNT as u32,
                    width,
                    height,
                    RENDER_TARGET_FORMAT,
                    DXGI_SWAP_CHAIN_FLAG(0),
                )
                .context("Failed to resize swap chain")?;
        }

        resources.recreate_resources(devices, width, height)?;

        unsafe {
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
        }

        Ok(())
    }

    fn upload_scene_buffers(&mut self, scene: &Scene) -> Result<()> {
        let devices = self.devices.as_ref().context("devices missing")?;

        if !scene.shadows.is_empty() {
            self.pipelines.shadow_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.shadows,
            )?;
        }

        if !scene.quads.is_empty() {
            self.pipelines.quad_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.quads,
            )?;
        }

        if !scene.underlines.is_empty() {
            self.pipelines.underline_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.underlines,
            )?;
        }

        if !scene.monochrome_sprites.is_empty() {
            self.pipelines.mono_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.monochrome_sprites,
            )?;
        }

        if !scene.subpixel_sprites.is_empty() {
            self.pipelines.subpixel_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.subpixel_sprites,
            )?;
        }

        if !scene.polychrome_sprites.is_empty() {
            self.pipelines.poly_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.polychrome_sprites,
            )?;
        }

        Ok(())
    }

    fn draw_shadows(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.shadow_pipeline.draw_range(
            &devices.device_context,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            start as u32,
            len as u32,
        )
    }

    fn draw_quads(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.quad_pipeline.draw_range(
            &devices.device_context,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            start as u32,
            len as u32,
        )
    }

    fn draw_paths_to_intermediate(&mut self, paths: &[Path<ScaledPixels>]) -> Result<()> {
        let [x, y, _, _] = path_batch_bounds(paths, [self.width, self.height]);
        self.draw_paths_to_intermediate_at(paths, [x, y])
    }

    fn draw_paths_to_intermediate_at(
        &mut self,
        paths: &[Path<ScaledPixels>],
        origin: [u32; 2],
    ) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let targets = resources.paths.as_ref().context("path targets missing")?;
        update_buffer(
            &devices.device_context,
            self.globals.path_params_buffer.as_ref().unwrap(),
            &[PathParams {
                origin: origin.map(|value| value as f32),
                size: [targets.size[0] as f32, targets.size[1] as f32],
            }],
        )?;
        unsafe {
            devices
                .device_context
                .VSSetShaderResources(0, Some(&[None]));
            devices
                .device_context
                .PSSetShaderResources(0, Some(&[None]));
            devices
                .device_context
                .VSSetConstantBuffers(4, Some(slice::from_ref(&self.globals.path_params_buffer)));
            devices
                .device_context
                .PSSetConstantBuffers(4, Some(slice::from_ref(&self.globals.path_params_buffer)));
            devices
                .device_context
                .RSSetViewports(Some(slice::from_ref(&targets.viewport)));
        }
        // Clear intermediate MSAA texture
        unsafe {
            devices
                .device_context
                .ClearRenderTargetView(targets.msaa_view.as_ref().unwrap(), &[0.0; 4]);
            // Set intermediate MSAA texture as render target
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&targets.msaa_view)), None);
        }

        // Collect all vertices and sprites for a single draw call
        let mut vertices = Vec::new();

        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationSprite {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.clipped_bounds(),
            }));
        }

        self.pipelines.path_rasterization_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            &vertices,
        )?;

        self.pipelines.path_rasterization_pipeline.draw(
            &devices.device_context,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            vertices.len() as u32,
            1,
        )?;

        // Resolve MSAA to non-MSAA intermediate texture
        unsafe {
            devices.device_context.ResolveSubresource(
                &targets.texture,
                0,
                &targets.msaa_texture,
                0,
                RENDER_TARGET_FORMAT,
            );
            // Restore the active render target (the offscreen scene/group target when blurring,
            // otherwise the swapchain) so the path sprites land on the correct surface.
            let restore_target = if self.active_render_target.is_some() {
                &self.active_render_target
            } else {
                &resources.render_target_view
            };
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(restore_target)), None);
            devices
                .device_context
                .RSSetViewports(Some(slice::from_ref(&resources.viewport)));
        }

        Ok(())
    }

    fn draw_paths_from_intermediate(&mut self, paths: &[Path<ScaledPixels>]) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites = if paths.last().unwrap().order == first_path.order {
            paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect::<Vec<_>>()
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            vec![PathSprite { bounds }]
        };

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        self.pipelines.path_sprite_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            &sprites,
        )?;

        // Draw the sprites with the path texture
        self.pipelines.path_sprite_pipeline.draw_with_texture(
            &devices.device_context,
            slice::from_ref(
                &resources
                    .paths
                    .as_ref()
                    .context("path targets missing")?
                    .srv,
            ),
            slice::from_ref(&self.globals.sampler),
            sprites.len() as u32,
        )
    }

    fn draw_underlines(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.underline_pipeline.draw_range(
            &devices.device_context,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            start as u32,
            len as u32,
        )
    }

    fn draw_monochrome_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.mono_sprites.draw_range_with_texture(
            &devices.device_context,
            &texture_view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_subpixel_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.subpixel_sprites.draw_range_with_texture(
            &devices.device_context,
            &texture_view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_polychrome_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.poly_sprites.draw_range_with_texture(
            &devices.device_context,
            &texture_view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_surfaces(&mut self, surfaces: &[PaintSurface]) -> Result<()> {
        if surfaces.is_empty() {
            return Ok(());
        }
        Ok(())
    }

    /// Run a single blur pass: a full-screen (or composite) draw sampling `source_srv` into
    /// `target_rtv`, with `params` in the blur constant buffer (b1).
    #[allow(clippy::too_many_arguments)]
    fn dx_blur_pass(
        &self,
        vertex: &ID3D11VertexShader,
        fragment: &ID3D11PixelShader,
        blend: &ID3D11BlendState,
        target_rtv: &Option<ID3D11RenderTargetView>,
        source_srv: &Option<ID3D11ShaderResourceView>,
        params: BlurParams,
        viewport: &D3D11_VIEWPORT,
        scissor: Option<RECT>,
        topology: D3D_PRIMITIVE_TOPOLOGY,
        vertex_count: u32,
        clear: bool,
    ) -> Result<()> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let ctx = &devices.device_context;
        update_buffer(ctx, &self.pipelines.blur_params_buffer, &[params])?;
        let null_srv: [Option<ID3D11ShaderResourceView>; 1] = [None];
        let blur_params = [Some(self.pipelines.blur_params_buffer.clone())];
        unsafe {
            // Unbind any SRV at slot 0 so the target texture isn't simultaneously bound as input.
            ctx.PSSetShaderResources(0, Some(&null_srv));
            if clear {
                ctx.ClearRenderTargetView(
                    target_rtv.as_ref().context("blur target view missing")?,
                    &[0.0; 4],
                );
            }
            ctx.OMSetRenderTargets(Some(slice::from_ref(target_rtv)), None);
            ctx.RSSetViewports(Some(slice::from_ref(viewport)));
            ctx.IASetPrimitiveTopology(topology);
            ctx.VSSetShader(vertex, None);
            ctx.PSSetShader(fragment, None);
            ctx.VSSetConstantBuffers(0, Some(slice::from_ref(&self.globals.global_params_buffer)));
            ctx.PSSetConstantBuffers(0, Some(slice::from_ref(&self.globals.global_params_buffer)));
            ctx.VSSetConstantBuffers(1, Some(&blur_params));
            ctx.PSSetConstantBuffers(1, Some(&blur_params));
            ctx.PSSetSamplers(0, Some(slice::from_ref(&self.globals.blur_sampler)));
            ctx.PSSetShaderResources(0, Some(slice::from_ref(source_srv)));
            ctx.OMSetBlendState(blend, None, 0xFFFFFFFF);
            let previous_rasterizer = scissor.and_then(|_| ctx.RSGetState().ok());
            let mut previous_scissors = [RECT::default(); 16];
            let mut previous_scissor_count = previous_scissors.len() as u32;
            if let Some(scissor) = scissor {
                ctx.RSGetScissorRects(
                    &mut previous_scissor_count,
                    Some(previous_scissors.as_mut_ptr()),
                );
                ctx.RSSetState(&self.pipelines.blur_rasterizer);
                ctx.RSSetScissorRects(Some(slice::from_ref(&scissor)));
            }
            ctx.DrawInstanced(vertex_count, 1, 0, 0);
            if scissor.is_some() {
                ctx.RSSetState(previous_rasterizer.as_ref());
                ctx.RSSetScissorRects(Some(&previous_scissors[..previous_scissor_count as usize]));
            }
            // Unbind the source so the target can be rebound as a render target next.
            ctx.PSSetShaderResources(0, Some(&null_srv));
        }
        Ok(())
    }

    /// Blur `source_srv` (full-resolution) using the half-res ping/pong textures and composite the
    /// result into `target_rtv`, clipped to `bounds`/`corner_radii`/`content_mask` and modulated
    /// by `opacity`. Shared by the backdrop and content-filter paths.
    #[allow(clippy::too_many_arguments)]
    fn dx_blur_and_composite(
        &self,
        source_srv: &Option<ID3D11ShaderResourceView>,
        target_rtv: &Option<ID3D11RenderTargetView>,
        bounds: Bounds<ScaledPixels>,
        content_mask: Bounds<ScaledPixels>,
        corner_radii: [f32; 4],
        blur_radius: f32,
        opacity: f32,
        // Backdrop clips to the rounded rect; content (`filter`) bleeds past its bounds.
        clip_rounded: bool,
    ) -> Result<()> {
        // Sigma is halved because the blur runs at half resolution.
        let sigma = (blur_radius * 0.5).max(0.0);
        if sigma <= 0.0 {
            return Ok(());
        }
        // Span ±3σ. If that needs more than 32 taps, spread the taps apart (tap_step > 1) rather
        // than truncating the kernel — keeps very large radii from clipping. Matches wgpu.
        let ideal_taps = (3.0 * sigma).ceil();
        let tap_count = ideal_taps.clamp(1.0, 32.0);
        let tap_step = (ideal_taps / tap_count).max(1.0);
        // Content blur bleeds ~3·radius past the box, so its composite quad covers a dilated rect.
        let composite_bounds = if clip_rounded {
            bounds
        } else {
            bounds.dilate(ScaledPixels(3.0 * blur_radius))
        };
        let half_w = (self.width / 2).max(1);
        let half_h = (self.height / 2).max(1);
        // Match the shipping WGPU pass footprint, including the transparent
        // outside pixels sampled by later passes of a content filter.
        let dilation = 3.0 * blur_radius;
        let x0 = (((bounds.origin.x.0 - dilation) * 0.5).floor().max(0.0) as u32).min(half_w);
        let y0 = (((bounds.origin.y.0 - dilation) * 0.5).floor().max(0.0) as u32).min(half_h);
        let x1 = ((((bounds.origin.x.0 + bounds.size.width.0 + dilation) * 0.5)
            .ceil()
            .max(0.0) as u32)
            .min(half_w))
        .max(x0);
        let y1 = ((((bounds.origin.y.0 + bounds.size.height.0 + dilation) * 0.5)
            .ceil()
            .max(0.0) as u32)
            .min(half_h))
        .max(y0);
        if x0 == x1 || y0 == y1 {
            return Ok(());
        }
        let scissor = RECT {
            left: x0 as i32,
            top: y0 as i32,
            right: x1 as i32,
            bottom: y1 as i32,
        };
        let half_vp = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: half_w as f32,
            Height: half_h as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let (full_vp, ping_rtv, ping_srv, pong_rtv, pong_srv) = {
            let r = self.resources.as_ref().context("resources missing")?;
            let blur = r.blur.as_ref().context("blur resources missing")?;
            (
                r.viewport,
                blur.ping_rtv.clone(),
                blur.ping_srv.clone(),
                blur.pong_rtv.clone(),
                blur.pong_srv.clone(),
            )
        };

        // Downsample source -> ping, then separable gaussian ping -> pong -> ping.
        self.dx_blur_pass(
            &self.pipelines.blur_downsample_vertex,
            &self.pipelines.blur_downsample_fragment,
            &self.pipelines.blur_blend_replace,
            &ping_rtv,
            source_srv,
            BlurParams {
                downsample: 1.0,
                ..Default::default()
            },
            &half_vp,
            Some(scissor),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            3,
            true,
        )?;
        self.dx_blur_pass(
            &self.pipelines.blur_vertex,
            &self.pipelines.blur_fragment,
            &self.pipelines.blur_blend_replace,
            &pong_rtv,
            &ping_srv,
            BlurParams {
                direction: [1.0 / half_w as f32, 0.0],
                sigma,
                tap_count,
                tap_step,
                ..Default::default()
            },
            &half_vp,
            Some(scissor),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            3,
            true,
        )?;
        self.dx_blur_pass(
            &self.pipelines.blur_vertex,
            &self.pipelines.blur_fragment,
            &self.pipelines.blur_blend_replace,
            &ping_rtv,
            &pong_srv,
            BlurParams {
                direction: [0.0, 1.0 / half_h as f32],
                sigma,
                tap_count,
                tap_step,
                ..Default::default()
            },
            &half_vp,
            Some(scissor),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            3,
            true,
        )?;
        // Composite the blurred result into the target (preserving its contents).
        self.dx_blur_pass(
            &self.pipelines.blur_composite_vertex,
            &self.pipelines.blur_composite_fragment,
            &self.pipelines.blur_blend_composite,
            target_rtv,
            &ping_srv,
            BlurParams {
                bounds: composite_bounds,
                content_mask,
                corner_radii,
                opacity,
                clip_rounded: if clip_rounded { 1.0 } else { 0.0 },
                ..Default::default()
            },
            &full_vp,
            None,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            4,
            false,
        )?;
        Ok(())
    }

    /// Copy the offscreen scene texture into the swapchain render target.
    fn dx_blit(
        &self,
        source_srv: &Option<ID3D11ShaderResourceView>,
        target_rtv: &Option<ID3D11RenderTargetView>,
    ) -> Result<()> {
        let full_vp = self
            .resources
            .as_ref()
            .context("resources missing")?
            .viewport;
        self.dx_blur_pass(
            &self.pipelines.blur_downsample_vertex,
            &self.pipelines.blur_downsample_fragment,
            &self.pipelines.blur_blend_replace,
            target_rtv,
            source_srv,
            BlurParams::default(),
            &full_vp,
            None,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            3,
            true,
        )
    }

    pub(crate) fn gpu_specs(&self) -> Result<GpuSpecs> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let desc = unsafe { devices.adapter.GetDesc1() }?;
        let is_software_emulated = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
        let device_name = String::from_utf16_lossy(&desc.Description)
            .trim_matches(char::from(0))
            .to_string();
        let driver_name = match desc.VendorId {
            0x10DE => "NVIDIA Corporation".to_string(),
            0x1002 => "AMD Corporation".to_string(),
            0x8086 => "Intel Corporation".to_string(),
            id => format!("Unknown Vendor (ID: {:#X})", id),
        };
        let driver_version = match desc.VendorId {
            0x10DE => nvidia::get_driver_version(),
            0x1002 => amd::get_driver_version(),
            // For Intel and other vendors, we use the DXGI API to get the driver version.
            _ => dxgi::get_driver_version(&devices.adapter),
        }
        .context("Failed to get gpu driver info")
        .log_err()
        .unwrap_or("Unknown Driver".to_string());
        Ok(GpuSpecs {
            is_software_emulated,
            device_name,
            driver_name,
            driver_info: driver_version,
        })
    }

    pub(crate) fn get_font_info() -> &'static FontInfo {
        static CACHED_FONT_INFO: OnceLock<FontInfo> = OnceLock::new();
        CACHED_FONT_INFO.get_or_init(|| {
            #[cfg(feature = "cosmic-text")]
            {
                let parameters = gpui::FontRenderingParameters::from_env();
                FontInfo {
                    gamma_ratios: parameters.gamma_ratios,
                    grayscale_enhanced_contrast: parameters.grayscale_enhanced_contrast,
                    subpixel_enhanced_contrast: parameters.subpixel_enhanced_contrast,
                    is_bgr: false,
                }
            }
            #[cfg(not(feature = "cosmic-text"))]
            unsafe {
                let factory: IDWriteFactory5 =
                    DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED).unwrap();
                let render_params: IDWriteRenderingParams1 =
                    factory.CreateRenderingParams().unwrap().cast().unwrap();
                FontInfo {
                    gamma_ratios: gpui::get_gamma_correction_ratios(render_params.GetGamma()),
                    grayscale_enhanced_contrast: render_params.GetGrayscaleEnhancedContrast(),
                    subpixel_enhanced_contrast: render_params.GetEnhancedContrast(),
                    is_bgr: render_params.GetPixelGeometry() == DWRITE_PIXEL_GEOMETRY_BGR,
                }
            }
        })
    }

    pub(crate) fn mark_drawable(&mut self) {
        self.skip_draws = false;
    }
}

impl DirectXResources {
    fn ensure_paths(&mut self, device: &ID3D11Device, required: Option<[u32; 2]>) -> Result<()> {
        let current = self.paths.as_ref().map(|targets| targets.size);
        let retained = retained_path_target_size(current, required);
        if current != retained {
            self.paths = retained
                .map(|size| PathTargets::new(device, size))
                .transpose()?;
        }
        Ok(())
    }

    pub fn new(
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
        hwnd: HWND,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        let swap_chain = if disable_direct_composition {
            create_swap_chain(&devices.dxgi_factory, &devices.device, hwnd, width, height)?
        } else {
            create_swap_chain_for_composition(
                &devices.dxgi_factory,
                &devices.device,
                width,
                height,
            )?
        };

        let (render_target, render_target_view, viewport) =
            create_resources(devices, &swap_chain, width, height)?;
        set_rasterizer_state(&devices.device, &devices.device_context)?;

        Ok(Self {
            swap_chain,
            render_target: Some(render_target),
            render_target_view,
            paths: None,
            blur: None,
            viewport,
        })
    }

    #[inline]
    fn recreate_resources(
        &mut self,
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let (render_target, render_target_view, viewport) =
            create_resources(devices, &self.swap_chain, width, height)?;
        self.render_target = Some(render_target);
        self.render_target_view = render_target_view;
        self.paths = None;
        self.blur = None;
        self.viewport = viewport;
        Ok(())
    }
}

impl DirectXRenderPipelines {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        let shadow_pipeline = PipelineState::new(
            device,
            "shadow_pipeline",
            ShaderModule::Shadow,
            4,
            create_blend_state(device)?,
        )?;
        let quad_pipeline = PipelineState::new(
            device,
            "quad_pipeline",
            ShaderModule::Quad,
            64,
            create_blend_state(device)?,
        )?;
        let path_rasterization_pipeline = PipelineState::new(
            device,
            "path_rasterization_pipeline",
            ShaderModule::PathRasterization,
            32,
            create_blend_state_for_path_rasterization(device)?,
        )?;
        let path_sprite_pipeline = PipelineState::new(
            device,
            "path_sprite_pipeline",
            ShaderModule::PathSprite,
            4,
            create_blend_state_for_path_sprite(device)?,
        )?;
        let underline_pipeline = PipelineState::new(
            device,
            "underline_pipeline",
            ShaderModule::Underline,
            4,
            create_blend_state(device)?,
        )?;
        let mono_sprites = PipelineState::new(
            device,
            "monochrome_sprite_pipeline",
            ShaderModule::MonochromeSprite,
            512,
            create_blend_state(device)?,
        )?;
        let subpixel_sprites = PipelineState::new(
            device,
            "subpixel_sprite_pipeline",
            ShaderModule::SubpixelSprite,
            512,
            create_blend_state_for_subpixel_rendering(device)?,
        )?;
        let poly_sprites = PipelineState::new(
            device,
            "polychrome_sprite_pipeline",
            ShaderModule::PolychromeSprite,
            16,
            create_blend_state(device)?,
        )?;

        let blur_downsample_vertex = create_vertex_shader(
            device,
            RawShaderBytes::new(ShaderModule::BlurDownsample, ShaderTarget::Vertex)?.as_bytes(),
        )?;
        let blur_downsample_fragment = create_fragment_shader(
            device,
            RawShaderBytes::new(ShaderModule::BlurDownsample, ShaderTarget::Fragment)?.as_bytes(),
        )?;
        let blur_vertex = create_vertex_shader(
            device,
            RawShaderBytes::new(ShaderModule::Blur, ShaderTarget::Vertex)?.as_bytes(),
        )?;
        let blur_fragment = create_fragment_shader(
            device,
            RawShaderBytes::new(ShaderModule::Blur, ShaderTarget::Fragment)?.as_bytes(),
        )?;
        let blur_composite_vertex = create_vertex_shader(
            device,
            RawShaderBytes::new(ShaderModule::BlurComposite, ShaderTarget::Vertex)?.as_bytes(),
        )?;
        let blur_composite_fragment = create_fragment_shader(
            device,
            RawShaderBytes::new(ShaderModule::BlurComposite, ShaderTarget::Fragment)?.as_bytes(),
        )?;
        let blur_params_buffer = create_constant_buffer::<BlurParams>(device)?.unwrap();
        let blur_blend_replace = create_blend_state_no_blend(device)?;
        // Premultiplied (One / InvSrcAlpha) — the composite outputs a premultiplied blurred sample;
        // straight-alpha blending would darken the faded edges.
        let blur_blend_composite = create_blend_state_for_path_rasterization(device)?;

        Ok(Self {
            shadow_pipeline,
            quad_pipeline,
            path_rasterization_pipeline,
            path_sprite_pipeline,
            underline_pipeline,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
            blur_downsample_vertex,
            blur_downsample_fragment,
            blur_vertex,
            blur_fragment,
            blur_composite_vertex,
            blur_composite_fragment,
            blur_params_buffer,
            blur_blend_replace,
            blur_blend_composite,
            blur_rasterizer: create_rasterizer_state(device, true)?,
        })
    }
}

impl DirectComposition {
    pub fn new(dxgi_device: &IDXGIDevice, hwnd: HWND) -> Result<Self> {
        let comp_device = get_comp_device(dxgi_device)?;
        let comp_target = unsafe { comp_device.CreateTargetForHwnd(hwnd, true) }?;
        let comp_visual = unsafe { comp_device.CreateVisual() }?;

        Ok(Self {
            comp_device,
            comp_target,
            comp_visual,
        })
    }

    pub fn set_swap_chain(&self, swap_chain: &IDXGISwapChain1) -> Result<()> {
        unsafe {
            self.comp_visual.SetContent(swap_chain)?;
            self.comp_target.SetRoot(&self.comp_visual)?;
            self.comp_device.Commit()?;
        }
        Ok(())
    }
}

impl DirectXGlobalElements {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        let global_params_buffer = create_constant_buffer::<GlobalParams>(device)?;
        let batch_params_buffer = create_constant_buffer::<BatchParams>(device)?;
        let path_params_buffer = create_constant_buffer::<PathParams>(device)?;

        let (sampler, blur_sampler) = unsafe {
            let mut desc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_WRAP,
                AddressV: D3D11_TEXTURE_ADDRESS_WRAP,
                AddressW: D3D11_TEXTURE_ADDRESS_WRAP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_ALWAYS,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: D3D11_FLOAT32_MAX,
            };
            let mut output = None;
            device.CreateSamplerState(&desc, Some(&mut output))?;
            // Blur samples the scene and must clamp at its edges, matching WGPU.
            // Atlas sampling keeps its existing address mode.
            desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
            desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
            desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            let mut blur_sampler = None;
            device.CreateSamplerState(&desc, Some(&mut blur_sampler))?;
            (output, blur_sampler)
        };

        Ok(Self {
            global_params_buffer,
            batch_params_buffer,
            path_params_buffer,
            sampler,
            blur_sampler,
        })
    }
}

#[derive(Debug, Default)]
#[repr(C)]
struct GlobalParams {
    gamma_ratios: [f32; 4],
    viewport_size: [f32; 2],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    pulse_clock_ms: u32,
    _pad: [u32; 2],
}

/// Mirrors the `BlurParams` cbuffer (register b1) in `shaders.hlsl`. 80 bytes (a multiple of 16,
/// as constant buffers require). Updated per blur pass via `update_buffer`.
#[repr(C)]
#[derive(Clone, Copy)]
struct BlurParams {
    bounds: Bounds<ScaledPixels>,
    content_mask: Bounds<ScaledPixels>,
    corner_radii: [f32; 4],
    direction: [f32; 2],
    sigma: f32,
    opacity: f32,
    tap_count: f32,
    /// 1.0 clips the composite to the rounded rect (backdrop); 0.0 lets content blur bleed past
    /// its bounds like CSS `filter: blur`.
    clip_rounded: f32,
    /// 1.0 = snapped 2:1 box downsample (anchor the half-res grid to a fixed 2px grid at the
    /// origin, so a stationary element blurs identically at every window size); 0.0 = 1:1 copy
    /// (the scene blit, which must not downsample). Downsample pass only.
    downsample: f32,
    /// Spacing between taps in pixels (gaussian passes only); >1 lets `tap_count` taps span very
    /// large radii without truncating the gaussian, matching the wgpu backend.
    tap_step: f32,
}

impl Default for BlurParams {
    fn default() -> Self {
        BlurParams {
            bounds: Bounds::default(),
            content_mask: Bounds::default(),
            corner_radii: [0.0; 4],
            direction: [0.0, 0.0],
            sigma: 0.0,
            opacity: 1.0,
            tap_count: 0.0,
            clip_rounded: 0.0,
            downsample: 0.0,
            tap_step: 0.0,
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
struct BatchParams {
    start_index: u32,
    _padding: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<BatchParams>() == 16);

struct PipelineState<T> {
    label: &'static str,
    vertex: ID3D11VertexShader,
    fragment: ID3D11PixelShader,
    buffer: ID3D11Buffer,
    buffer_size: usize,
    view: Option<ID3D11ShaderResourceView>,
    blend_state: ID3D11BlendState,
    _marker: std::marker::PhantomData<T>,
}

impl<T> PipelineState<T> {
    fn new(
        device: &ID3D11Device,
        label: &'static str,
        shader_module: ShaderModule,
        buffer_size: usize,
        blend_state: ID3D11BlendState,
    ) -> Result<Self> {
        let vertex = {
            let raw_shader = RawShaderBytes::new(shader_module, ShaderTarget::Vertex)?;
            create_vertex_shader(device, raw_shader.as_bytes())?
        };
        let fragment = {
            let raw_shader = RawShaderBytes::new(shader_module, ShaderTarget::Fragment)?;
            create_fragment_shader(device, raw_shader.as_bytes())?
        };
        let buffer = create_buffer(device, std::mem::size_of::<T>(), buffer_size)?;
        let view = create_buffer_view(device, &buffer)?;

        Ok(PipelineState {
            label,
            vertex,
            fragment,
            buffer,
            buffer_size,
            view,
            blend_state,
            _marker: std::marker::PhantomData,
        })
    }

    fn update_buffer(
        &mut self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
        data: &[T],
    ) -> Result<()> {
        if self.buffer_size < data.len() {
            let element_size = std::mem::size_of::<T>();
            let required_size = std::mem::size_of_val(data);
            anyhow::ensure!(
                required_size <= MAX_INSTANCE_BUFFER_SIZE,
                "{} buffer needs {required_size} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}",
                self.label
            );
            let new_buffer_size = data
                .len()
                .next_power_of_two()
                .min(MAX_INSTANCE_BUFFER_SIZE / element_size);
            log::debug!(
                "Updating {} buffer size from {} to {}",
                self.label,
                self.buffer_size,
                new_buffer_size
            );
            let buffer = create_buffer(device, std::mem::size_of::<T>(), new_buffer_size)?;
            let view = create_buffer_view(device, &buffer)?;
            self.buffer = buffer;
            self.view = view;
            self.buffer_size = new_buffer_size;
        }
        update_buffer(device_context, &self.buffer, data)
    }

    fn draw(
        &self,
        device_context: &ID3D11DeviceContext,
        topology: D3D_PRIMITIVE_TOPOLOGY,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<()> {
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            topology,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.DrawInstanced(vertex_count, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_with_texture(
        &self,
        device_context: &ID3D11DeviceContext,
        texture: &[Option<ID3D11ShaderResourceView>],
        sampler: &[Option<ID3D11SamplerState>],
        instance_count: u32,
    ) -> Result<()> {
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.PSSetSamplers(0, Some(sampler));
            device_context.VSSetShaderResources(0, Some(texture));
            device_context.PSSetShaderResources(0, Some(texture));

            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_range(
        &self,
        device_context: &ID3D11DeviceContext,
        batch_params_buffer: &ID3D11Buffer,
        first_instance: u32,
        instance_count: u32,
    ) -> Result<()> {
        anyhow::ensure!(
            first_instance as usize + instance_count as usize <= self.buffer_size,
            "DirectX instance range exceeds the {} buffer",
            self.label
        );
        update_batch_start(device_context, batch_params_buffer, first_instance)?;
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_range_with_texture(
        &self,
        device_context: &ID3D11DeviceContext,
        texture: &[Option<ID3D11ShaderResourceView>],
        batch_params_buffer: &ID3D11Buffer,
        sampler: &[Option<ID3D11SamplerState>],
        first_instance: u32,
        instance_count: u32,
    ) -> Result<()> {
        anyhow::ensure!(
            first_instance as usize + instance_count as usize <= self.buffer_size,
            "DirectX instance range exceeds the {} buffer",
            self.label
        );
        update_batch_start(device_context, batch_params_buffer, first_instance)?;
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.PSSetSamplers(0, Some(sampler));
            device_context.VSSetShaderResources(0, Some(texture));
            device_context.PSSetShaderResources(0, Some(texture));
            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PathRasterizationSprite {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
}

impl Drop for DirectXRenderer {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        if let Some(devices) = &self.devices {
            report_live_objects(&devices.device).ok();
        }
    }
}

#[inline]
fn get_comp_device(dxgi_device: &IDXGIDevice) -> Result<IDCompositionDevice> {
    Ok(unsafe { DCompositionCreateDevice(dxgi_device)? })
}

fn create_swap_chain_for_composition(
    dxgi_factory: &IDXGIFactory6,
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: RENDER_TARGET_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT as u32,
        // Composition SwapChains only support the DXGI_SCALING_STRETCH Scaling.
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_PREMULTIPLIED,
        Flags: 0,
    };
    Ok(unsafe { dxgi_factory.CreateSwapChainForComposition(device, &desc, None)? })
}

fn create_swap_chain(
    dxgi_factory: &IDXGIFactory6,
    device: &ID3D11Device,
    hwnd: HWND,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    use windows::Win32::Graphics::Dxgi::DXGI_MWA_NO_ALT_ENTER;

    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: RENDER_TARGET_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT as u32,
        Scaling: DXGI_SCALING_NONE,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_IGNORE,
        Flags: 0,
    };
    let swap_chain =
        unsafe { dxgi_factory.CreateSwapChainForHwnd(device, hwnd, &desc, None, None) }?;
    unsafe { dxgi_factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER) }?;
    Ok(swap_chain)
}

#[inline]
fn create_resources(
    devices: &DirectXRendererDevices,
    swap_chain: &IDXGISwapChain1,
    width: u32,
    height: u32,
) -> Result<(
    ID3D11Texture2D,
    Option<ID3D11RenderTargetView>,
    D3D11_VIEWPORT,
)> {
    let (render_target, render_target_view) =
        create_render_target_and_its_view(swap_chain, &devices.device)?;
    let viewport = D3D11_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: width as f32,
        Height: height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    Ok((render_target, render_target_view, viewport))
}

#[inline]
/// Flatten a `Corners` into the `[tl, tr, br, bl]` order expected by the blur composite shader.
fn corner_radii_array(corners: Corners<ScaledPixels>) -> [f32; 4] {
    [
        corners.top_left.0,
        corners.top_right.0,
        corners.bottom_right.0,
        corners.bottom_left.0,
    ]
}

fn create_render_target_and_its_view(
    swap_chain: &IDXGISwapChain1,
    device: &ID3D11Device,
) -> Result<(ID3D11Texture2D, Option<ID3D11RenderTargetView>)> {
    let render_target: ID3D11Texture2D = unsafe { swap_chain.GetBuffer(0) }?;
    let mut render_target_view = None;
    unsafe { device.CreateRenderTargetView(&render_target, None, Some(&mut render_target_view))? };
    Ok((render_target, render_target_view))
}

#[inline]
fn create_path_intermediate_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11ShaderResourceView>)> {
    let texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };

    let mut shader_resource_view = None;
    unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut shader_resource_view))? };

    Ok((texture, Some(shader_resource_view.unwrap())))
}

/// Create a color texture usable as both a render target and a shader resource, returning both
/// views. Used for the blur offscreen targets.
#[inline]
fn create_color_target(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(
    ID3D11Texture2D,
    Option<ID3D11RenderTargetView>,
    Option<ID3D11ShaderResourceView>,
)> {
    let texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width.max(1),
            Height: height.max(1),
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };
    let mut rtv = None;
    unsafe { device.CreateRenderTargetView(&texture, None, Some(&mut rtv))? };
    let mut srv = None;
    unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut srv))? };
    Ok((texture, rtv, srv))
}

#[inline]
fn create_path_intermediate_msaa_texture_and_view(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11RenderTargetView>)> {
    let msaa_texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: PATH_MULTISAMPLE_COUNT,
                Quality: D3D11_STANDARD_MULTISAMPLE_PATTERN.0 as u32,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };
    let mut msaa_view = None;
    unsafe { device.CreateRenderTargetView(&msaa_texture, None, Some(&mut msaa_view))? };
    Ok((msaa_texture, Some(msaa_view.unwrap())))
}

#[inline]
fn set_rasterizer_state(device: &ID3D11Device, device_context: &ID3D11DeviceContext) -> Result<()> {
    let state = create_rasterizer_state(device, false)?;
    unsafe { device_context.RSSetState(&state) };
    Ok(())
}

fn create_rasterizer_state(device: &ID3D11Device, scissor: bool) -> Result<ID3D11RasterizerState> {
    let desc = D3D11_RASTERIZER_DESC {
        FillMode: D3D11_FILL_SOLID,
        CullMode: D3D11_CULL_NONE,
        FrontCounterClockwise: false.into(),
        DepthBias: 0,
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: 0.0,
        DepthClipEnable: true.into(),
        ScissorEnable: scissor.into(),
        MultisampleEnable: true.into(),
        AntialiasedLineEnable: false.into(),
    };
    let rasterizer_state = unsafe {
        let mut state = None;
        device.CreateRasterizerState(&desc, Some(&mut state))?;
        state.unwrap()
    };
    Ok(rasterizer_state)
}

// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ns-d3d11-d3d11_blend_desc
#[inline]
fn create_blend_state(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_subpixel_rendering(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC1_COLOR;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC1_COLOR;
    // It does not make sense to draw transparent subpixel-rendered text, since it cannot be meaningfully alpha-blended onto anything else.
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    desc.RenderTarget[0].RenderTargetWriteMask =
        D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8 & !D3D11_COLOR_WRITE_ENABLE_ALPHA.0 as u8;

    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_path_rasterization(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    // If the feature level is set to greater than D3D_FEATURE_LEVEL_9_3, the display
    // device performs the blend in linear space, which is ideal.
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_path_sprite(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    // If the feature level is set to greater than D3D_FEATURE_LEVEL_9_3, the display
    // device performs the blend in linear space, which is ideal.
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

/// A blend state that overwrites the target (no blending) — used for the blur downsample and
/// gaussian passes.
#[inline]
fn create_blend_state_no_blend(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = false.into();
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_vertex_shader(device: &ID3D11Device, bytes: &[u8]) -> Result<ID3D11VertexShader> {
    unsafe {
        let mut shader = None;
        device.CreateVertexShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
fn create_fragment_shader(device: &ID3D11Device, bytes: &[u8]) -> Result<ID3D11PixelShader> {
    unsafe {
        let mut shader = None;
        device.CreatePixelShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
fn create_constant_buffer<T>(device: &ID3D11Device) -> Result<Option<ID3D11Buffer>> {
    const { assert!(std::mem::size_of::<T>() != 0 && std::mem::size_of::<T>().is_multiple_of(16)) };
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: std::mem::size_of::<T>() as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: 0,
        StructureByteStride: 0,
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) }?;
    Ok(buffer)
}

#[inline]
fn create_buffer(
    device: &ID3D11Device,
    element_size: usize,
    buffer_size: usize,
) -> Result<ID3D11Buffer> {
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: (element_size * buffer_size) as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32,
        StructureByteStride: element_size as u32,
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) }?;
    Ok(buffer.unwrap())
}

#[inline]
fn create_buffer_view(
    device: &ID3D11Device,
    buffer: &ID3D11Buffer,
) -> Result<Option<ID3D11ShaderResourceView>> {
    let mut view = None;
    unsafe { device.CreateShaderResourceView(buffer, None, Some(&mut view)) }?;
    Ok(view)
}

#[inline]
fn update_buffer<T>(
    device_context: &ID3D11DeviceContext,
    buffer: &ID3D11Buffer,
    data: &[T],
) -> Result<()> {
    unsafe {
        let mut dest = std::mem::zeroed();
        device_context.Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut dest))?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), dest.pData as _, data.len());
        device_context.Unmap(buffer, 0);
    }
    Ok(())
}

#[inline]
fn update_batch_start(
    device_context: &ID3D11DeviceContext,
    buffer: &ID3D11Buffer,
    first_instance: u32,
) -> Result<()> {
    update_buffer(
        device_context,
        buffer,
        &[BatchParams {
            start_index: first_instance,
            _padding: [0; 3],
        }],
    )?;
    // Blur passes bind their parameters at VS b1. Ordinary indexed batches must
    // restore this binding before reading their first-instance index.
    unsafe {
        device_context.VSSetConstantBuffers(1, Some(&[Some(buffer.clone())]));
    }
    Ok(())
}

#[inline]
fn set_pipeline_state(
    device_context: &ID3D11DeviceContext,
    buffer_view: &[Option<ID3D11ShaderResourceView>],
    topology: D3D_PRIMITIVE_TOPOLOGY,
    vertex_shader: &ID3D11VertexShader,
    fragment_shader: &ID3D11PixelShader,
    blend_state: &ID3D11BlendState,
) {
    unsafe {
        device_context.VSSetShaderResources(1, Some(buffer_view));
        device_context.PSSetShaderResources(1, Some(buffer_view));
        device_context.IASetPrimitiveTopology(topology);
        device_context.VSSetShader(vertex_shader, None);
        device_context.PSSetShader(fragment_shader, None);
        device_context.OMSetBlendState(blend_state, None, 0xFFFFFFFF);
    }
}

#[cfg(debug_assertions)]
fn report_live_objects(device: &ID3D11Device) -> Result<()> {
    let debug_device: ID3D11Debug = device.cast()?;
    unsafe {
        debug_device.ReportLiveDeviceObjects(D3D11_RLDO_DETAIL)?;
    }
    Ok(())
}

const BUFFER_COUNT: usize = 3;

pub(crate) mod shader_resources {
    use anyhow::Result;

    #[cfg(not(gpui_precompiled_shaders))]
    use windows::{
        Win32::Graphics::Direct3D::{
            Fxc::{D3DCOMPILE_DEBUG, D3DCOMPILE_SKIP_OPTIMIZATION, D3DCompileFromFile},
            ID3DBlob,
        },
        core::{HSTRING, PCSTR},
    };

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub(crate) enum ShaderModule {
        Quad,
        Shadow,
        Underline,
        PathRasterization,
        PathSprite,
        MonochromeSprite,
        SubpixelSprite,
        PolychromeSprite,
        #[cfg(not(feature = "cosmic-text"))]
        EmojiRasterization,
        BlurDownsample,
        Blur,
        BlurComposite,
        Mesh,
        MeshComposite,
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub(crate) enum ShaderTarget {
        Vertex,
        Fragment,
    }

    pub(crate) struct RawShaderBytes<'t> {
        inner: &'t [u8],

        #[cfg(not(gpui_precompiled_shaders))]
        _blob: ID3DBlob,
    }

    impl<'t> RawShaderBytes<'t> {
        pub(crate) fn new(module: ShaderModule, target: ShaderTarget) -> Result<Self> {
            #[cfg(gpui_precompiled_shaders)]
            {
                Ok(Self::from_bytes(module, target))
            }
            #[cfg(not(gpui_precompiled_shaders))]
            {
                let blob = build_shader_blob(module, target)?;
                let inner = unsafe {
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer() as *const u8,
                        blob.GetBufferSize(),
                    )
                };
                Ok(Self { inner, _blob: blob })
            }
        }

        pub(crate) fn as_bytes(&'t self) -> &'t [u8] {
            self.inner
        }

        #[cfg(gpui_precompiled_shaders)]
        fn from_bytes(module: ShaderModule, target: ShaderTarget) -> Self {
            let bytes = match module {
                ShaderModule::Quad => match target {
                    ShaderTarget::Vertex => QUAD_VERTEX_BYTES,
                    ShaderTarget::Fragment => QUAD_FRAGMENT_BYTES,
                },
                ShaderModule::Shadow => match target {
                    ShaderTarget::Vertex => SHADOW_VERTEX_BYTES,
                    ShaderTarget::Fragment => SHADOW_FRAGMENT_BYTES,
                },
                ShaderModule::Underline => match target {
                    ShaderTarget::Vertex => UNDERLINE_VERTEX_BYTES,
                    ShaderTarget::Fragment => UNDERLINE_FRAGMENT_BYTES,
                },
                ShaderModule::PathRasterization => match target {
                    ShaderTarget::Vertex => PATH_RASTERIZATION_VERTEX_BYTES,
                    ShaderTarget::Fragment => PATH_RASTERIZATION_FRAGMENT_BYTES,
                },
                ShaderModule::PathSprite => match target {
                    ShaderTarget::Vertex => PATH_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => PATH_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::MonochromeSprite => match target {
                    ShaderTarget::Vertex => MONOCHROME_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => MONOCHROME_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::SubpixelSprite => match target {
                    ShaderTarget::Vertex => SUBPIXEL_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => SUBPIXEL_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::PolychromeSprite => match target {
                    ShaderTarget::Vertex => POLYCHROME_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => POLYCHROME_SPRITE_FRAGMENT_BYTES,
                },
                #[cfg(not(feature = "cosmic-text"))]
                ShaderModule::EmojiRasterization => match target {
                    ShaderTarget::Vertex => EMOJI_RASTERIZATION_VERTEX_BYTES,
                    ShaderTarget::Fragment => EMOJI_RASTERIZATION_FRAGMENT_BYTES,
                },
                ShaderModule::BlurDownsample => match target {
                    ShaderTarget::Vertex => BLUR_DOWNSAMPLE_VERTEX_BYTES,
                    ShaderTarget::Fragment => BLUR_DOWNSAMPLE_FRAGMENT_BYTES,
                },
                ShaderModule::Blur => match target {
                    ShaderTarget::Vertex => BLUR_VERTEX_BYTES,
                    ShaderTarget::Fragment => BLUR_FRAGMENT_BYTES,
                },
                ShaderModule::BlurComposite => match target {
                    ShaderTarget::Vertex => BLUR_COMPOSITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => BLUR_COMPOSITE_FRAGMENT_BYTES,
                },
                ShaderModule::Mesh => match target {
                    ShaderTarget::Vertex => MESH_VERTEX_BYTES,
                    ShaderTarget::Fragment => MESH_FRAGMENT_BYTES,
                },
                ShaderModule::MeshComposite => match target {
                    ShaderTarget::Vertex => MESH_COMPOSITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => MESH_COMPOSITE_FRAGMENT_BYTES,
                },
            };
            Self { inner: bytes }
        }
    }

    #[cfg(not(gpui_precompiled_shaders))]
    pub(super) fn build_shader_blob(entry: ShaderModule, target: ShaderTarget) -> Result<ID3DBlob> {
        unsafe {
            use windows::Win32::Graphics::{
                Direct3D::ID3DInclude, Hlsl::D3D_COMPILE_STANDARD_FILE_INCLUDE,
            };

            let shader_name = match entry {
                #[cfg(not(feature = "cosmic-text"))]
                ShaderModule::EmojiRasterization => "color_text_raster.hlsl",
                ShaderModule::Mesh | ShaderModule::MeshComposite => "shaders_mesh.hlsl",
                _ => "shaders.hlsl",
            };

            let entry = format!(
                "{}_{}\0",
                entry.as_str(),
                match target {
                    ShaderTarget::Vertex => "vertex",
                    ShaderTarget::Fragment => "fragment",
                }
            );
            let target = match target {
                ShaderTarget::Vertex => "vs_4_1\0",
                ShaderTarget::Fragment => "ps_4_1\0",
            };

            let mut compile_blob = None;
            let mut error_blob = None;
            let shader_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(&format!("src/{}", shader_name))
                .canonicalize()?;

            let entry_point = PCSTR::from_raw(entry.as_ptr());
            let target_cstr = PCSTR::from_raw(target.as_ptr());

            // really dirty trick because winapi bindings are unhappy otherwise
            let include_handler = &std::mem::transmute::<usize, ID3DInclude>(
                D3D_COMPILE_STANDARD_FILE_INCLUDE as usize,
            );

            let ret = D3DCompileFromFile(
                &HSTRING::from(shader_path.to_str().unwrap()),
                None,
                include_handler,
                entry_point,
                target_cstr,
                D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
                0,
                &mut compile_blob,
                Some(&mut error_blob),
            );
            if ret.is_err() {
                let Some(error_blob) = error_blob else {
                    return Err(anyhow::anyhow!("{ret:?}"));
                };

                let error_string =
                    std::ffi::CStr::from_ptr(error_blob.GetBufferPointer() as *const i8)
                        .to_string_lossy();
                log::error!("Shader compile error: {}", error_string);
                return Err(anyhow::anyhow!("Compile error: {}", error_string));
            }
            Ok(compile_blob.unwrap())
        }
    }

    #[cfg(gpui_precompiled_shaders)]
    include!(concat!(env!("OUT_DIR"), "/shaders_bytes.rs"));
    #[cfg(gpui_precompiled_shaders)]
    include!(concat!(env!("OUT_DIR"), "/mesh_shaders_bytes.rs"));

    #[cfg(not(gpui_precompiled_shaders))]
    impl ShaderModule {
        pub fn as_str(self) -> &'static str {
            match self {
                ShaderModule::Quad => "quad",
                ShaderModule::Shadow => "shadow",
                ShaderModule::Underline => "underline",
                ShaderModule::PathRasterization => "path_rasterization",
                ShaderModule::PathSprite => "path_sprite",
                ShaderModule::MonochromeSprite => "monochrome_sprite",
                ShaderModule::SubpixelSprite => "subpixel_sprite",
                ShaderModule::PolychromeSprite => "polychrome_sprite",
                #[cfg(not(feature = "cosmic-text"))]
                ShaderModule::EmojiRasterization => "emoji_rasterization",
                ShaderModule::BlurDownsample => "blur_downsample",
                ShaderModule::Blur => "blur",
                ShaderModule::BlurComposite => "blur_composite",
                ShaderModule::Mesh => "mesh",
                ShaderModule::MeshComposite => "mesh_composite",
            }
        }
    }
}

mod nvidia {
    use std::{
        ffi::CStr,
        os::raw::{c_char, c_int, c_uint},
    };

    use anyhow::Result;
    use windows::{Win32::System::LibraryLoader::GetProcAddress, core::s};

    use crate::with_dll_library;

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L180
    const NVAPI_SHORT_STRING_MAX: usize = 64;

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L235
    #[allow(non_camel_case_types)]
    type NvAPI_ShortString = [c_char; NVAPI_SHORT_STRING_MAX];

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L447
    #[allow(non_camel_case_types)]
    type NvAPI_SYS_GetDriverAndBranchVersion_t = unsafe extern "C" fn(
        driver_version: *mut c_uint,
        build_branch_string: *mut NvAPI_ShortString,
    ) -> c_int;

    pub(super) fn get_driver_version() -> Result<String> {
        #[cfg(target_pointer_width = "64")]
        let nvidia_dll_name = s!("nvapi64.dll");
        #[cfg(target_pointer_width = "32")]
        let nvidia_dll_name = s!("nvapi.dll");

        with_dll_library(nvidia_dll_name, |nvidia_dll| unsafe {
            let nvapi_query_addr = GetProcAddress(nvidia_dll, s!("nvapi_QueryInterface"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get nvapi_QueryInterface address"))?;
            let nvapi_query: extern "C" fn(u32) -> *mut () = std::mem::transmute(nvapi_query_addr);

            // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_interface.h#L41
            let nvapi_get_driver_version_ptr = nvapi_query(0x2926aaad);
            if nvapi_get_driver_version_ptr.is_null() {
                anyhow::bail!("Failed to get NVIDIA driver version function pointer");
            }
            let nvapi_get_driver_version: NvAPI_SYS_GetDriverAndBranchVersion_t =
                std::mem::transmute(nvapi_get_driver_version_ptr);

            let mut driver_version: c_uint = 0;
            let mut build_branch_string: NvAPI_ShortString = [0; NVAPI_SHORT_STRING_MAX];
            let result = nvapi_get_driver_version(
                &mut driver_version as *mut c_uint,
                &mut build_branch_string as *mut NvAPI_ShortString,
            );

            if result != 0 {
                anyhow::bail!(
                    "Failed to get NVIDIA driver version, error code: {}",
                    result
                );
            }
            let major = driver_version / 100;
            let minor = driver_version % 100;
            let branch_string = CStr::from_ptr(build_branch_string.as_ptr());
            Ok(format!(
                "{}.{} {}",
                major,
                minor,
                branch_string.to_string_lossy()
            ))
        })
    }
}

mod amd {
    use std::os::raw::{c_char, c_int, c_void};

    use anyhow::Result;
    use windows::{Win32::System::LibraryLoader::GetProcAddress, core::s};

    use crate::with_dll_library;

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L145
    const AGS_CURRENT_VERSION: i32 = (6 << 22) | (3 << 12);

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L204
    // This is an opaque type, using struct to represent it properly for FFI
    #[repr(C)]
    struct AGSContext {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct AGSGPUInfo {
        pub driver_version: *const c_char,
        pub radeon_software_version: *const c_char,
        pub num_devices: c_int,
        pub devices: *mut c_void,
    }

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L429
    #[allow(non_camel_case_types)]
    type agsInitialize_t = unsafe extern "C" fn(
        version: c_int,
        config: *const c_void,
        context: *mut *mut AGSContext,
        gpu_info: *mut AGSGPUInfo,
    ) -> c_int;

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L436
    #[allow(non_camel_case_types)]
    type agsDeInitialize_t = unsafe extern "C" fn(context: *mut AGSContext) -> c_int;

    pub(super) fn get_driver_version() -> Result<String> {
        #[cfg(target_pointer_width = "64")]
        let amd_dll_name = s!("amd_ags_x64.dll");
        #[cfg(target_pointer_width = "32")]
        let amd_dll_name = s!("amd_ags_x86.dll");

        with_dll_library(amd_dll_name, |amd_dll| unsafe {
            let ags_initialize_addr = GetProcAddress(amd_dll, s!("agsInitialize"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get agsInitialize address"))?;
            let ags_deinitialize_addr = GetProcAddress(amd_dll, s!("agsDeInitialize"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get agsDeInitialize address"))?;

            let ags_initialize: agsInitialize_t = std::mem::transmute(ags_initialize_addr);
            let ags_deinitialize: agsDeInitialize_t = std::mem::transmute(ags_deinitialize_addr);

            let mut context: *mut AGSContext = std::ptr::null_mut();
            let mut gpu_info: AGSGPUInfo = AGSGPUInfo {
                driver_version: std::ptr::null(),
                radeon_software_version: std::ptr::null(),
                num_devices: 0,
                devices: std::ptr::null_mut(),
            };

            let result = ags_initialize(
                AGS_CURRENT_VERSION,
                std::ptr::null(),
                &mut context,
                &mut gpu_info,
            );
            if result != 0 {
                anyhow::bail!("Failed to initialize AMD AGS, error code: {}", result);
            }

            // Vulkan actually returns this as the driver version
            let software_version = if !gpu_info.radeon_software_version.is_null() {
                std::ffi::CStr::from_ptr(gpu_info.radeon_software_version)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown Radeon Software Version".to_string()
            };

            let driver_version = if !gpu_info.driver_version.is_null() {
                std::ffi::CStr::from_ptr(gpu_info.driver_version)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown Radeon Driver Version".to_string()
            };

            ags_deinitialize(context);
            Ok(format!("{} ({})", software_version, driver_version))
        })
    }
}

#[cfg(test)]
mod resource_tests {
    use super::{
        BlurResources, DirectXRendererDevices, DirectXResources, MAX_FILTER_DEPTH,
        filter_group_depth,
    };
    use anyhow::{Context, Result};
    use gpui::{Bounds, ContentMask, Corners, FilterBoundary, ScaledFilter, ScaledPixels};
    use windows::{
        Win32::{
            Foundation::{HMODULE, HWND},
            Graphics::{
                Direct3D::D3D_DRIVER_TYPE_WARP,
                Direct3D11::{
                    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
                    D3D11CreateDevice, ID3D11Texture2D,
                },
                Dxgi::{IDXGIAdapter1, IDXGIDevice, IDXGIFactory6},
            },
        },
        core::Interface,
    };

    pub(super) fn test_devices() -> Result<DirectXRendererDevices> {
        let mut device = None;
        let mut device_context = None;
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
                Some(&mut device_context),
            )?;
        }
        let device = device.context("WARP device missing")?;
        let dxgi_device: IDXGIDevice = device.cast()?;
        let adapter: IDXGIAdapter1 = unsafe { dxgi_device.GetAdapter()? }.cast()?;
        let dxgi_factory = unsafe { adapter.GetParent::<IDXGIFactory6>()? };
        Ok(DirectXRendererDevices {
            adapter,
            dxgi_factory,
            device,
            device_context: device_context.context("WARP device context missing")?,
            dxgi_device: Some(dxgi_device),
            annotation: None,
        })
    }

    fn texture_size(texture: &ID3D11Texture2D) -> (u32, u32) {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { texture.GetDesc(&mut desc) };
        (desc.Width, desc.Height)
    }

    fn boundary(is_start: bool) -> FilterBoundary {
        FilterBoundary {
            order: 0,
            bounds: Bounds::default(),
            content_mask: ContentMask::default(),
            corner_radii: Corners::default(),
            filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(4.0))],
            opacity: 1.0,
            is_start,
        }
    }

    #[test]
    fn content_group_depth_counts_nesting_instead_of_sibling_count() {
        let depth = |markers: &[bool]| {
            filter_group_depth(&markers.iter().copied().map(boundary).collect::<Vec<_>>())
        };
        assert_eq!(depth(&[]), 0);
        assert_eq!(depth(&[false]), 0);
        assert_eq!(depth(&[true, false, true, false, true, false]), 1);
        assert_eq!(depth(&[true, true, false, false]), 2);
        assert_eq!(
            depth(&[true, true, true, false, false, false]),
            MAX_FILTER_DEPTH
        );
    }

    #[test]
    fn blur_groups_allocate_only_required_depth_and_reuse_existing_textures() -> Result<()> {
        let devices = test_devices()?;
        let mut blur = BlurResources::new(&devices.device, 65, 33)?;
        assert_eq!(texture_size(&blur.scene_color), (65, 33));
        assert_eq!(texture_size(&blur.ping), (32, 16));
        assert_eq!(texture_size(&blur.pong), (32, 16));
        assert!(blur.groups.is_empty());
        assert!(blur.group_rtvs.is_empty());
        assert!(blur.group_srvs.is_empty());

        blur.ensure_groups(&devices.device, 65, 33, 0)?;
        assert!(blur.groups.is_empty());
        blur.ensure_groups(&devices.device, 65, 33, 1)?;
        let first_group = blur.groups[0].clone();
        let first_rtv = blur.group_rtvs[0].clone();
        let first_srv = blur.group_srvs[0].clone();
        assert_eq!(texture_size(&first_group), (65, 33));
        for requested_depth in [1, 0, 1] {
            blur.ensure_groups(&devices.device, 65, 33, requested_depth)?;
            assert_eq!(blur.groups.len(), 1);
            assert_eq!(blur.groups[0], first_group);
            assert_eq!(blur.group_rtvs[0], first_rtv);
            assert_eq!(blur.group_srvs[0], first_srv);
        }

        let markers = [true, true, true, false, false, false].map(boundary);
        blur.ensure_groups(&devices.device, 65, 33, filter_group_depth(&markers))?;
        assert_eq!(blur.groups.len(), MAX_FILTER_DEPTH);
        assert_eq!(blur.group_rtvs.len(), MAX_FILTER_DEPTH);
        assert_eq!(blur.group_srvs.len(), MAX_FILTER_DEPTH);
        assert_eq!(blur.groups[0], first_group);
        for ix in 0..MAX_FILTER_DEPTH {
            let rtv_resource: ID3D11Texture2D =
                unsafe { blur.group_rtvs[ix].as_ref().unwrap().GetResource()? }.cast()?;
            let srv_resource: ID3D11Texture2D =
                unsafe { blur.group_srvs[ix].as_ref().unwrap().GetResource()? }.cast()?;
            assert_eq!(rtv_resource, blur.groups[ix]);
            assert_eq!(srv_resource, blur.groups[ix]);
        }
        Ok(())
    }

    #[test]
    fn resource_recreation_discards_blur_targets_until_needed_again() -> Result<()> {
        let devices = test_devices()?;
        // Composition swapchains do not require an HWND or a visible window.
        let mut resources = DirectXResources::new(&devices, 65, 33, HWND::default(), false)?;
        assert!(resources.blur.is_none());
        let mut blur = BlurResources::new(&devices.device, 65, 33)?;
        blur.ensure_groups(&devices.device, 65, 33, 1)?;
        let old_scene = blur.scene_color.clone();
        let old_group_view = blur.group_rtvs[0].clone().unwrap();
        resources.blur = Some(blur);

        resources.recreate_resources(&devices, 65, 33)?;
        assert!(resources.blur.is_none());
        let replacement = BlurResources::new(&devices.device, 65, 33)?;
        assert_ne!(replacement.scene_color, old_scene);
        assert!(replacement.groups.is_empty());
        resources.blur = Some(replacement);
        drop(resources);

        // Views held by already encoded work remain valid when the owner is dropped.
        let retained_group: ID3D11Texture2D = unsafe { old_group_view.GetResource()? }.cast()?;
        assert_eq!(texture_size(&retained_group), (65, 33));
        let mut replacement = DirectXResources::new(&devices, 97, 49, HWND::default(), false)?;
        assert!(replacement.blur.is_none());
        replacement.blur = Some(BlurResources::new(&devices.device, 97, 49)?);
        let blur = replacement.blur.as_ref().unwrap();
        assert_eq!(texture_size(&blur.scene_color), (97, 49));
        assert_eq!(texture_size(&blur.ping), (48, 24));
        assert!(blur.groups.is_empty());
        Ok(())
    }
}

mod dxgi {
    use windows::{
        Win32::Graphics::Dxgi::{IDXGIAdapter1, IDXGIDevice},
        core::Interface,
    };

    pub(super) fn get_driver_version(adapter: &IDXGIAdapter1) -> anyhow::Result<String> {
        let number = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID as _) }?;
        Ok(format!(
            "{}.{}.{}.{}",
            number >> 48,
            (number >> 32) & 0xFFFF,
            (number >> 16) & 0xFFFF,
            number & 0xFFFF
        ))
    }
}

#[cfg(test)]
#[path = "directx_path_tests.rs"]
mod path_pixel_tests;

#[cfg(test)]
#[path = "directx_blur_lifecycle_tests.rs"]
mod blur_lifecycle_tests;

#[cfg(test)]
#[path = "directx_blur_regression_tests.rs"]
mod blur_regression_tests;
