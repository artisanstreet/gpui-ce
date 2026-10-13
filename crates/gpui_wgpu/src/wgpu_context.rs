#[cfg(not(target_family = "wasm"))]
use anyhow::Context as _;
#[cfg(not(target_family = "wasm"))]
use gpui_util::ResultExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::TextureFormat;

pub struct WgpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    backend: WgpuBackend,
    dual_source_blending: bool,
    color_texture_format: wgpu::TextureFormat,
    device_lost: Arc<AtomicBool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WgpuBackend {
    BrowserWebGpu,
    Gl,
    Native(wgpu::Backend),
}

#[cfg(target_family = "wasm")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WebBackendPreference {
    #[default]
    Auto,
    WebGpu,
    WebGl,
}

#[cfg(target_family = "wasm")]
pub struct PreparedWebGraphics {
    pub context: WgpuContext,
    pub surface: wgpu::Surface<'static>,
}

/// wgpu-core refuses to create a surface when neither the instance nor the surface
/// target carries a display handle, and `SurfaceTarget::Canvas` always passes `None`.
/// The WebGL2 backend never reads the handle (WebGPU bypasses wgpu-core entirely), so
/// a unit web display handle on the instance satisfies the check.
#[cfg(target_family = "wasm")]
#[derive(Debug)]
struct WebDisplaySource;

#[cfg(target_family = "wasm")]
impl raw_window_handle::HasDisplayHandle for WebDisplaySource {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Ok(raw_window_handle::DisplayHandle::web())
    }
}

#[derive(Clone, Copy)]
pub struct CompositorGpuHint {
    pub vendor_id: u32,
    pub device_id: u32,
}

#[cfg(not(target_family = "wasm"))]
type AdapterPriority = (u8, u8, u8, u8);

#[cfg(not(target_family = "wasm"))]
struct NativeBackend {
    surface: wgpu::Surface<'static>,
    instance: wgpu::Instance,
}

#[cfg(not(target_family = "wasm"))]
struct NativeAdapter {
    adapter: wgpu::Adapter,
    backend_index: usize,
    priority: AdapterPriority,
}

/// Extra wgpu features and limits that an application can request on top of
/// gpui's baseline.  Pass an instance to the platform via
/// [`gpui::App::set_gpu_requirements`] *before* opening any windows.
#[derive(Clone, Debug, Default)]
pub struct WgpuDeviceRequirements {
    /// Additional [`wgpu::Features`] to enable.  These are OR-ed with gpui's
    /// own required features.
    pub features: wgpu::Features,
    /// Additional [`wgpu::Limits`] to request.  Each field is merged by taking
    /// `max(gpui_limit, app_limit)` for upper-bound limits and
    /// `min(gpui_limit, app_limit)` for alignment/lower-bound limits.
    pub limits: wgpu::Limits,
}

impl WgpuContext {
    pub(crate) fn memory_hints() -> wgpu::MemoryHints {
        // UI resources are smaller than game workloads. Keep allocator growth
        // bounded; textures larger than a block still receive dedicated storage.
        wgpu::MemoryHints::Manual {
            suballocated_device_memory_block_size: 4 * 1024 * 1024..16 * 1024 * 1024,
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn new(
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
        compositor_gpu: Option<CompositorGpuHint>,
        extra_requirements: Option<&WgpuDeviceRequirements>,
    ) -> anyhow::Result<Self> {
        Self::new_with_options(instance, surface, compositor_gpu, false, extra_requirements)
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn new_rejecting_software(
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
        compositor_gpu: Option<CompositorGpuHint>,
        extra_requirements: Option<&WgpuDeviceRequirements>,
    ) -> anyhow::Result<Self> {
        Self::new_with_options(instance, surface, compositor_gpu, true, extra_requirements)
    }

    #[cfg(not(target_family = "wasm"))]
    fn new_with_options(
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
        compositor_gpu: Option<CompositorGpuHint>,
        reject_software: bool,
        extra_requirements: Option<&WgpuDeviceRequirements>,
    ) -> anyhow::Result<Self> {
        let device_id_filter = device_id_filter();

        // Select an adapter by actually testing surface configuration with the real device.
        // This is the only reliable way to determine compatibility on hybrid GPU systems.
        let (adapter, device, queue, dual_source_blending, color_texture_format) =
            gpui::block_on(Self::select_adapter_and_device(
                &instance,
                device_id_filter,
                surface,
                compositor_gpu.as_ref(),
                reject_software,
                extra_requirements,
            ))?;

        Ok(Self::from_selected_device(
            instance,
            adapter,
            device,
            queue,
            dual_source_blending,
            color_texture_format,
        ))
    }

    #[cfg(not(target_family = "wasm"))]
    fn from_selected_device(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        dual_source_blending: bool,
        color_texture_format: TextureFormat,
    ) -> Self {
        let device_lost = Arc::new(AtomicBool::new(false));
        device.set_device_lost_callback({
            let device_lost = Arc::clone(&device_lost);
            move |reason, message| {
                log::error!("wgpu device lost: reason={reason:?}, message={message}");
                if reason != wgpu::DeviceLostReason::Destroyed {
                    device_lost.store(true, Ordering::Relaxed);
                }
            }
        });

        log::info!(
            "Selected GPU adapter: {:?} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        let backend = WgpuBackend::Native(adapter.get_info().backend);
        Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            backend,
            dual_source_blending,
            color_texture_format,
            device_lost,
        }
    }

    /// Creates a surface and context while releasing every unused backend.
    ///
    /// As in the renderer's existing raw-handle surface creation, the caller
    /// keeps the native window alive until the returned surface is dropped.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn prepare_for_window<W>(
        window: &W,
        compositor_gpu: Option<CompositorGpuHint>,
        reject_software: bool,
        extra_requirements: Option<&WgpuDeviceRequirements>,
    ) -> anyhow::Result<(Self, wgpu::Surface<'static>)>
    where
        W: raw_window_handle::HasWindowHandle
            + raw_window_handle::HasDisplayHandle
            + std::fmt::Debug
            + Send
            + Sync
            + Clone
            + 'static,
    {
        let requested = native_backends(wgpu::Backends::from_env());
        let device_id_filter = device_id_filter();
        let gl_priority_bound = undiscovered_gl_priority(device_id_filter, compositor_gpu.as_ref());

        gpui::block_on(async {
            let mut backends = Vec::new();
            let mut adapters = Vec::new();
            // Match wgpu-core's stable enumeration order for equal priorities.
            for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
                if requested.contains(backend) {
                    Self::discover_native_backend(
                        window,
                        backend,
                        device_id_filter,
                        compositor_gpu.as_ref(),
                        &mut backends,
                        &mut adapters,
                    )
                    .await;
                }
            }
            let mut gl_pending = requested.contains(wgpu::Backends::GL);
            loop {
                adapters.sort_by_key(|candidate| candidate.priority);
                if gl_pending
                    && must_discover_gl(adapters.first().map(|c| c.priority), gl_priority_bound)
                {
                    gl_pending = false;
                    Self::discover_native_backend(
                        window,
                        wgpu::Backends::GL,
                        device_id_filter,
                        compositor_gpu.as_ref(),
                        &mut backends,
                        &mut adapters,
                    )
                    .await;
                    continue;
                }
                if adapters.is_empty() {
                    anyhow::bail!("No GPU adapter found that can configure the display surface");
                }
                let candidate = adapters.remove(0);
                let info = candidate.adapter.get_info();
                if reject_software && info.device_type == wgpu::DeviceType::Cpu {
                    log::info!(
                        "Skipping software renderer: {} ({:?})",
                        info.name,
                        info.backend
                    );
                    continue;
                }
                match Self::try_adapter_with_surface(
                    &candidate.adapter,
                    &backends[candidate.backend_index].surface,
                    extra_requirements,
                )
                .await
                {
                    Ok((device, queue, dual_source_blending, color_texture_format)) => {
                        let selected = backends.swap_remove(candidate.backend_index);
                        // Adapter handles and surfaces keep their instances alive.
                        // Release all three before constructing renderer resources.
                        drop(adapters);
                        drop(backends);
                        let context = Self::from_selected_device(
                            selected.instance,
                            candidate.adapter,
                            device,
                            queue,
                            dual_source_blending,
                            color_texture_format,
                        );
                        return Ok((context, selected.surface));
                    }
                    Err(error) => log::info!(
                        "Adapter {} ({:?}) failed: {}, trying next...",
                        info.name,
                        info.backend,
                        error
                    ),
                }
            }
        })
    }

    #[cfg(not(target_family = "wasm"))]
    async fn discover_native_backend<W>(
        window: &W,
        backend: wgpu::Backends,
        device_id_filter: Option<u32>,
        compositor_gpu: Option<&CompositorGpuHint>,
        backends: &mut Vec<NativeBackend>,
        adapters: &mut Vec<NativeAdapter>,
    ) where
        W: raw_window_handle::HasWindowHandle
            + raw_window_handle::HasDisplayHandle
            + std::fmt::Debug
            + Send
            + Sync
            + Clone
            + 'static,
    {
        let instance = Self::instance_for_backends(Box::new(window.clone()), backend);
        let surface = window
            .window_handle()
            .map_err(anyhow::Error::from)
            .and_then(|handle| {
                // The renderer retains the native window for the surface's lifetime.
                unsafe {
                    instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                        raw_display_handle: None,
                        raw_window_handle: handle.as_raw(),
                    })
                }
                .map_err(anyhow::Error::from)
            });
        let surface = match surface {
            Ok(surface) => surface,
            Err(error) => {
                log::info!("Backend {backend:?} could not create a surface: {error}");
                return;
            }
        };
        let backend_index = backends.len();
        let discovered = instance.enumerate_adapters(backend).await;
        for adapter in discovered {
            let info = adapter.get_info();
            log::info!("Found adapter {} ({:?})", info.name, info.backend);
            adapters.push(NativeAdapter {
                priority: adapter_priority(&info, device_id_filter, compositor_gpu),
                adapter,
                backend_index,
            });
        }
        backends.push(NativeBackend { surface, instance });
    }

    #[cfg(target_family = "wasm")]
    pub async fn new_web(
        canvas: &web_sys::HtmlCanvasElement,
        preference: WebBackendPreference,
    ) -> anyhow::Result<PreparedWebGraphics> {
        Self::new_web_with_backend(canvas, preference).await
    }

    #[cfg(target_family = "wasm")]
    #[allow(clippy::arc_with_non_send_sync)]
    async fn new_web_with_backend(
        canvas: &web_sys::HtmlCanvasElement,
        preference: WebBackendPreference,
    ) -> anyhow::Result<PreparedWebGraphics> {
        let backends = match preference {
            WebBackendPreference::Auto => wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            WebBackendPreference::WebGpu => wgpu::Backends::BROWSER_WEBGPU,
            WebBackendPreference::WebGl => wgpu::Backends::GL,
        };
        let descriptor = wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: Some(Box::new(WebDisplaySource)),
        };
        let instance = if preference == WebBackendPreference::Auto {
            wgpu::util::new_instance_with_webgpu_detection(descriptor).await
        } else {
            wgpu::Instance::new(descriptor)
        };
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|error| {
                anyhow::anyhow!("Failed to create browser graphics surface: {error}")
            })?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|error| {
                anyhow::anyhow!(
                    "Failed to request a {preference:?} adapter compatible with the canvas: {error}"
                )
            })?;
        let adapter_info = adapter.get_info();
        let backend = match adapter_info.backend {
            wgpu::Backend::BrowserWebGpu => WgpuBackend::BrowserWebGpu,
            wgpu::Backend::Gl => WgpuBackend::Gl,
            backend => {
                anyhow::bail!(
                    "Browser graphics initialization selected unexpected backend {backend:?}"
                )
            }
        };

        let device_lost = Arc::new(AtomicBool::new(false));
        let (device, queue, dual_source_blending, color_texture_format) =
            Self::create_device(&adapter, None).await?;
        device.set_device_lost_callback({
            let device_lost = Arc::clone(&device_lost);
            move |reason, message| {
                log::error!("wgpu device lost: reason={reason:?}, message={message}");
                if reason != wgpu::DeviceLostReason::Destroyed {
                    device_lost.store(true, Ordering::Relaxed);
                }
            }
        });
        log::info!(
            "Browser graphics initialized: requested={preference:?}, selected={backend:?}, \
             adapter={:?}, limits={:?}, dual_source_blending={dual_source_blending}",
            adapter_info.name,
            device.limits(),
        );

        let context = Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            backend,
            dual_source_blending,
            color_texture_format,
            device_lost,
        };
        Ok(PreparedWebGraphics { context, surface })
    }

    async fn create_device(
        adapter: &wgpu::Adapter,
        extra_requirements: Option<&WgpuDeviceRequirements>,
    ) -> anyhow::Result<(wgpu::Device, wgpu::Queue, bool, TextureFormat)> {
        let dual_source_blending = adapter
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING);

        let mut required_features = wgpu::Features::empty();
        if dual_source_blending {
            required_features |= wgpu::Features::DUAL_SOURCE_BLENDING;
        } else {
            log::warn!(
                "Dual-source blending not available on this GPU. \
                Subpixel text antialiasing will be disabled."
            );
        }

        let color_atlas_texture_format = Self::select_color_texture_format(adapter)?;
        #[cfg(target_family = "wasm")]
        let mut required_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
            wgpu::Limits::downlevel_webgl2_defaults()
                .using_resolution(adapter.limits())
                .using_alignment(adapter.limits())
        } else {
            wgpu::Limits::downlevel_defaults()
                .using_resolution(adapter.limits())
                .using_alignment(adapter.limits())
        };
        #[cfg(not(target_family = "wasm"))]
        let mut required_limits = wgpu::Limits::downlevel_defaults()
            .using_resolution(adapter.limits())
            .using_alignment(adapter.limits());

        // Merge application-requested requirements.
        if let Some(reqs) = extra_requirements {
            required_features |= reqs.features;
            required_limits = required_limits.or_better_values_from(&reqs.limits);
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gpui_device"),
                required_features,
                required_limits,
                memory_hints: Self::memory_hints(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create wgpu device: {e}"))?;

        Ok((
            device,
            queue,
            dual_source_blending,
            color_atlas_texture_format,
        ))
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn instance(display: Box<dyn wgpu::wgt::WgpuHasDisplayHandle>) -> wgpu::Instance {
        Self::instance_for_backends(display, native_backends(wgpu::Backends::from_env()))
    }

    #[cfg(not(target_family = "wasm"))]
    fn instance_for_backends(
        display: Box<dyn wgpu::wgt::WgpuHasDisplayHandle>,
        backends: wgpu::Backends,
    ) -> wgpu::Instance {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: Some(display),
        })
    }

    pub fn check_compatible_with_surface(&self, surface: &wgpu::Surface<'_>) -> anyhow::Result<()> {
        let caps = surface.get_capabilities(&self.adapter);
        if caps.formats.is_empty() {
            let info = self.adapter.get_info();
            anyhow::bail!(
                "Adapter {:?} (backend={:?}, device={:#06x}) is not compatible with the \
                 display surface for this window.",
                info.name,
                info.backend,
                info.device,
            );
        }
        Ok(())
    }

    /// Select an adapter and create a device, testing that the surface can actually be configured.
    /// This is the only reliable way to determine compatibility on hybrid GPU systems, where
    /// adapters may report surface compatibility via get_capabilities() but fail when actually
    /// configuring (e.g., NVIDIA reporting Vulkan Wayland support but failing because the
    /// Wayland compositor runs on the Intel GPU).
    #[cfg(not(target_family = "wasm"))]
    async fn select_adapter_and_device(
        instance: &wgpu::Instance,
        device_id_filter: Option<u32>,
        surface: &wgpu::Surface<'_>,
        compositor_gpu: Option<&CompositorGpuHint>,
        reject_software: bool,
        extra_requirements: Option<&WgpuDeviceRequirements>,
    ) -> anyhow::Result<(
        wgpu::Adapter,
        wgpu::Device,
        wgpu::Queue,
        bool,
        TextureFormat,
    )> {
        let mut adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).await;

        if adapters.is_empty() {
            anyhow::bail!("No GPU adapters found");
        }

        if let Some(device_id) = device_id_filter {
            log::info!("ZED_DEVICE_ID filter: {:#06x}", device_id);
        }

        // Sort adapters into a single priority order. Tiers (from highest to lowest):
        //
        // 1. ZED_DEVICE_ID match — explicit user override
        // 2. Compositor GPU match — the GPU the display server is rendering on
        // 3. Device type (Discrete > Integrated > Other > Virtual > Cpu).
        //    "Other" ranks above "Virtual" because OpenGL seems to count as "Other".
        // 4. Backend — prefer Vulkan/Metal/Dx12 over GL/etc.
        adapters.sort_by_key(|adapter| {
            adapter_priority(&adapter.get_info(), device_id_filter, compositor_gpu)
        });

        // Log all available adapters (in sorted order)
        log::info!("Found {} GPU adapter(s):", adapters.len());
        for adapter in &adapters {
            let info = adapter.get_info();
            log::info!(
                "  - {} (vendor={:#06x}, device={:#06x}, backend={:?}, type={:?})",
                info.name,
                info.vendor,
                info.device,
                info.backend,
                info.device_type,
            );
        }

        // Test each adapter by creating a device and configuring the surface
        for adapter in adapters {
            let info = adapter.get_info();

            if reject_software && info.device_type == wgpu::DeviceType::Cpu {
                log::info!(
                    "Skipping software renderer: {} ({:?})",
                    info.name,
                    info.backend
                );
                continue;
            }

            log::info!("Testing adapter: {} ({:?})...", info.name, info.backend);

            match Self::try_adapter_with_surface(&adapter, surface, extra_requirements).await {
                Ok((device, queue, dual_source_blending, color_atlas_texture_format)) => {
                    log::info!(
                        "Selected GPU (passed configuration test): {} ({:?})",
                        info.name,
                        info.backend
                    );
                    return Ok((
                        adapter,
                        device,
                        queue,
                        dual_source_blending,
                        color_atlas_texture_format,
                    ));
                }
                Err(e) => {
                    log::info!(
                        "  Adapter {} ({:?}) failed: {}, trying next...",
                        info.name,
                        info.backend,
                        e
                    );
                }
            }
        }

        anyhow::bail!("No GPU adapter found that can configure the display surface")
    }

    /// Try to use an adapter with a surface by creating a device and testing configuration.
    /// Returns the device and queue if successful, allowing them to be reused.
    #[cfg(not(target_family = "wasm"))]
    async fn try_adapter_with_surface(
        adapter: &wgpu::Adapter,
        surface: &wgpu::Surface<'_>,
        extra_requirements: Option<&WgpuDeviceRequirements>,
    ) -> anyhow::Result<(wgpu::Device, wgpu::Queue, bool, TextureFormat)> {
        let caps = surface.get_capabilities(adapter);
        if caps.formats.is_empty() {
            anyhow::bail!("no compatible surface formats");
        }
        if caps.alpha_modes.is_empty() {
            anyhow::bail!("no compatible alpha modes");
        }

        let (device, queue, dual_source_blending, color_atlas_texture_format) =
            Self::create_device(adapter, extra_requirements).await?;
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let test_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: 64,
            height: 64,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &test_config);

        let error = error_scope.pop().await;
        if let Some(e) = error {
            anyhow::bail!("surface configuration failed: {e}");
        }

        Ok((
            device,
            queue,
            dual_source_blending,
            color_atlas_texture_format,
        ))
    }

    fn select_color_texture_format(adapter: &wgpu::Adapter) -> anyhow::Result<wgpu::TextureFormat> {
        let required_usages = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let bgra_features = adapter.get_texture_format_features(wgpu::TextureFormat::Bgra8Unorm);
        let rgba_features = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba8Unorm);
        #[cfg(target_family = "wasm")]
        if adapter.get_info().backend == wgpu::Backend::Gl
            && rgba_features.allowed_usages.contains(required_usages)
        {
            return Ok(wgpu::TextureFormat::Rgba8Unorm);
        }
        if bgra_features.allowed_usages.contains(required_usages) {
            return Ok(wgpu::TextureFormat::Bgra8Unorm);
        }
        if rgba_features.allowed_usages.contains(required_usages) {
            let info = adapter.get_info();
            log::warn!(
                "Adapter {} ({:?}) does not support Bgra8Unorm atlas textures with usages {:?}; \
                 falling back to Rgba8Unorm atlas textures.",
                info.name,
                info.backend,
                required_usages,
            );
            return Ok(wgpu::TextureFormat::Rgba8Unorm);
        }

        let info = adapter.get_info();
        Err(anyhow::anyhow!(
            "Adapter {} ({:?}, device={:#06x}) does not support a usable color atlas texture \
             format with usages {:?}. Bgra8Unorm allowed usages: {:?}; \
             Rgba8Unorm allowed usages: {:?}.",
            info.name,
            info.backend,
            info.device,
            required_usages,
            bgra_features.allowed_usages,
            rgba_features.allowed_usages,
        ))
    }
    pub fn backend(&self) -> WgpuBackend {
        self.backend
    }

    pub fn uses_webgl_instance_data(&self) -> bool {
        matches!(self.backend, WgpuBackend::Gl) && cfg!(target_family = "wasm")
    }

    pub fn supports_dual_source_blending(&self) -> bool {
        self.dual_source_blending
    }

    pub fn color_texture_format(&self) -> wgpu::TextureFormat {
        self.color_texture_format
    }

    /// Returns true if the GPU device was lost (e.g., due to driver crash, suspend/resume).
    /// When this returns true, the context should be recreated.
    pub fn device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Relaxed)
    }

    /// Returns a clone of the device_lost flag for sharing with renderers.
    pub(crate) fn device_lost_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.device_lost)
    }
}

#[cfg(not(target_family = "wasm"))]
fn native_backends(requested: Option<wgpu::Backends>) -> wgpu::Backends {
    #[cfg(not(target_os = "windows"))]
    let defaults = wgpu::Backends::VULKAN | wgpu::Backends::GL;
    #[cfg(target_os = "windows")]
    let defaults = wgpu::Backends::DX12;
    #[cfg(not(target_os = "windows"))]
    let supported = defaults;
    #[cfg(target_os = "windows")]
    let supported = wgpu::Backends::DX12 | wgpu::Backends::VULKAN;
    requested
        .map(|backends| backends & supported)
        .filter(|backends| !backends.is_empty())
        .unwrap_or(defaults)
}

#[cfg(not(target_family = "wasm"))]
fn device_id_filter() -> Option<u32> {
    match std::env::var("ZED_DEVICE_ID") {
        Ok(value) => parse_pci_id(&value)
            .context("Failed to parse device ID from `ZED_DEVICE_ID` environment variable")
            .log_err(),
        Err(std::env::VarError::NotPresent) => None,
        error => {
            error
                .context("Failed to read value of `ZED_DEVICE_ID` environment variable")
                .log_err();
            None
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn adapter_priority(
    info: &wgpu::AdapterInfo,
    device_id_filter: Option<u32>,
    compositor_gpu: Option<&CompositorGpuHint>,
) -> AdapterPriority {
    let device_known = info.device != 0;
    let user_override = match device_id_filter {
        Some(id) if device_known && info.device == id => 0,
        _ => 1,
    };
    let compositor_match = match compositor_gpu {
        Some(hint)
            if device_known && info.vendor == hint.vendor_id && info.device == hint.device_id =>
        {
            0
        }
        _ => 1,
    };
    let device_type = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::Other => 2,
        wgpu::DeviceType::VirtualGpu => 3,
        wgpu::DeviceType::Cpu => 4,
    };
    let backend = match info.backend {
        wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Dx12 => 0,
        _ => 1,
    };
    (user_override, compositor_match, device_type, backend)
}

#[cfg(not(target_family = "wasm"))]
fn undiscovered_gl_priority(
    device_id_filter: Option<u32>,
    compositor_gpu: Option<&CompositorGpuHint>,
) -> AdapterPriority {
    // Conservatively allow an undiscovered GL adapter to match every requested
    // GPU and be discrete. Today's wgpu GL reports device=0 and infers only
    // Integrated/Other/CPU, but correctness must not depend on those details.
    (
        u8::from(device_id_filter.is_none()),
        u8::from(compositor_gpu.is_none()),
        0,
        1,
    )
}

#[cfg(not(target_family = "wasm"))]
fn must_discover_gl(best_known: Option<AdapterPriority>, gl_bound: AdapterPriority) -> bool {
    best_known.is_none_or(|priority| priority > gl_bound)
}

#[cfg(not(target_family = "wasm"))]
fn parse_pci_id(id: &str) -> anyhow::Result<u32> {
    let mut id = id.trim();

    if id.starts_with("0x") || id.starts_with("0X") {
        id = &id[2..];
    }
    let is_hex_string = id.chars().all(|c| c.is_ascii_hexdigit());
    let is_4_chars = id.len() == 4;
    anyhow::ensure!(
        is_4_chars && is_hex_string,
        "Expected a 4 digit PCI ID in hexadecimal format"
    );

    u32::from_str_radix(id, 16).context("parsing PCI ID as hex")
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;

    fn adapter_info(
        backend: wgpu::Backend,
        device_type: wgpu::DeviceType,
        vendor: u32,
        device: u32,
    ) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: "selection fixture".into(),
            vendor,
            device,
            device_type,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend,
            subgroup_min_size: 4,
            subgroup_max_size: 128,
            transient_saves_memory: false,
        }
    }

    #[test]
    fn lazy_gl_discovery_preserves_hardware_priority_and_failed_adapter_fallback() {
        use wgpu::{Backend::*, DeviceType::*};
        let discrete = adapter_priority(&adapter_info(Vulkan, DiscreteGpu, 1, 7), None, None);
        let software = adapter_priority(&adapter_info(Vulkan, Cpu, 1, 8), None, None);
        let gl_hardware = adapter_priority(&adapter_info(Gl, IntegratedGpu, 1, 0), None, None);
        let gl_software = adapter_priority(&adapter_info(Gl, Cpu, 1, 0), None, None);
        let bound = undiscovered_gl_priority(None, None);

        assert!(!must_discover_gl(Some(discrete), bound));
        // If the discrete device cannot configure the surface, the next known
        // candidate is software. Discover GL before attempting that candidate.
        assert!(must_discover_gl(Some(software), bound));
        assert!(must_discover_gl(None, bound));
        assert!(gl_hardware < software);
        assert!(software < gl_software);
        // Remain conservative even though current wgpu GL never reports Discrete.
        let integrated = adapter_priority(&adapter_info(Vulkan, IntegratedGpu, 1, 9), None, None);
        assert!(must_discover_gl(Some(integrated), bound));
    }

    #[test]
    fn adapter_priority_preserves_explicit_and_compositor_gpu_choices() {
        use wgpu::{Backend::*, DeviceType::*};
        let discrete = adapter_info(Vulkan, DiscreteGpu, 1, 7);
        let software = adapter_info(Vulkan, Cpu, 1, 8);
        let compositor = CompositorGpuHint {
            vendor_id: 1,
            device_id: 8,
        };
        assert!(
            adapter_priority(&software, Some(8), None) < adapter_priority(&discrete, Some(8), None)
        );
        assert!(
            adapter_priority(&software, None, Some(&compositor))
                < adapter_priority(&discrete, None, Some(&compositor))
        );
        // An explicit user choice takes precedence over the compositor hint.
        assert!(
            adapter_priority(&discrete, Some(7), Some(&compositor))
                < adapter_priority(&software, Some(7), Some(&compositor))
        );
        let unknown_gl = adapter_info(Gl, IntegratedGpu, 1, 0);
        assert_eq!(adapter_priority(&unknown_gl, Some(0), None).0, 1);
    }

    #[test]
    fn undiscovered_gl_bound_covers_all_device_types_and_gpu_matches() {
        use wgpu::{Backend::*, DeviceType::*};
        let types = [DiscreteGpu, IntegratedGpu, Other, VirtualGpu, Cpu];
        let compositor = CompositorGpuHint {
            vendor_id: 1,
            device_id: 7,
        };
        for filter in [None, Some(0), Some(7), Some(9)] {
            for hint in [None, Some(&compositor)] {
                let bound = undiscovered_gl_priority(filter, hint);
                for device_type in types {
                    for vendor in [0, 1, 2] {
                        for device in [0, 7, 9] {
                            let gl = adapter_priority(
                                &adapter_info(Gl, device_type, vendor, device),
                                filter,
                                hint,
                            );
                            assert!(bound <= gl);
                            for native_type in types {
                                let native = adapter_priority(
                                    &adapter_info(Vulkan, native_type, vendor, device),
                                    filter,
                                    hint,
                                );
                                if !must_discover_gl(Some(native), bound) {
                                    assert!(native <= gl);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn explicit_backend_requests_keep_existing_supported_filtering() {
        use wgpu::Backends;
        assert_eq!(native_backends(Some(Backends::VULKAN)), Backends::VULKAN);
        assert_eq!(
            native_backends(Some(Backends::VULKAN | Backends::METAL)),
            Backends::VULKAN
        );
        assert_eq!(
            native_backends(Some(Backends::METAL)),
            native_backends(None)
        );
        assert_eq!(
            native_backends(Some(Backends::empty())),
            native_backends(None)
        );
        #[cfg(not(target_os = "windows"))]
        {
            assert_eq!(native_backends(Some(Backends::GL)), Backends::GL);
            assert_eq!(native_backends(None), Backends::VULKAN | Backends::GL);
        }
        #[cfg(target_os = "windows")]
        {
            assert_eq!(native_backends(None), Backends::DX12);
            assert_eq!(
                native_backends(Some(Backends::VULKAN | Backends::DX12)),
                Backends::VULKAN | Backends::DX12
            );
        }
    }

    #[test]
    fn test_parse_device_id() {
        assert!(parse_pci_id("0xABCD").is_ok());
        assert!(parse_pci_id("ABCD").is_ok());
        assert!(parse_pci_id("abcd").is_ok());
        assert!(parse_pci_id("1234").is_ok());
        assert!(parse_pci_id("123").is_err());
        assert_eq!(
            parse_pci_id(&format!("{:x}", 0x1234)).unwrap(),
            parse_pci_id(&format!("{:X}", 0x1234)).unwrap(),
        );

        assert_eq!(
            parse_pci_id(&format!("{:#x}", 0x1234)).unwrap(),
            parse_pci_id(&format!("{:#X}", 0x1234)).unwrap(),
        );
    }
}
