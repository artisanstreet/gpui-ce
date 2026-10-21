use anyhow::{Context, Result};
use gpui_util::ResultExt;
use itertools::Itertools;
use windows::Win32::{
    Foundation::{CloseHandle, HMODULE, WAIT_OBJECT_0},
    Graphics::{
        Direct3D::{
            D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
        },
        Direct3D11::{
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_DEBUG,
            D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS,
            D3D11_SDK_VERSION, D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext,
        },
        Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_DEBUG,
            DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory6, IDXGIFactory7,
        },
    },
    System::Threading::{CreateEventW, INFINITE, WaitForSingleObject},
};

pub(crate) fn try_to_recover_from_device_lost<T>(mut f: impl FnMut() -> Result<T>) -> Result<T> {
    (0..5)
        .map(|i| {
            if i > 0 {
                // Add a small delay before retrying
                std::thread::sleep(std::time::Duration::from_millis(100 + i * 10));
            }
            f()
        })
        .find_or_last(Result::is_ok)
        .unwrap()
        .context("DirectXRenderer failed to recover from lost device after multiple attempts")
}

#[derive(Clone)]
pub(crate) struct DirectXDevices {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) dxgi_factory: IDXGIFactory6,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
}

impl DirectXDevices {
    pub(crate) fn new() -> Result<Self> {
        let debug_layer_available = check_debug_layer_available();
        let dxgi_factory =
            get_dxgi_factory(debug_layer_available).context("Creating DXGI factory")?;
        let (adapter, device, device_context, feature_level) =
            get_adapter(&dxgi_factory, debug_layer_available).context("Getting DXGI adapter")?;
        match feature_level {
            D3D_FEATURE_LEVEL_11_1 => {
                log::info!("Created device with Direct3D 11.1 feature level.")
            }
            D3D_FEATURE_LEVEL_11_0 => {
                log::info!("Created device with Direct3D 11.0 feature level.")
            }
            D3D_FEATURE_LEVEL_10_1 => {
                log::info!("Created device with Direct3D 10.1 feature level.")
            }
            _ => unreachable!(),
        }

        Ok(Self {
            adapter,
            dxgi_factory,
            device,
            device_context,
        })
    }
}

#[inline]
fn check_debug_layer_available() -> bool {
    #[cfg(debug_assertions)]
    {
        use windows::Win32::Graphics::Dxgi::{DXGIGetDebugInterface1, IDXGIInfoQueue};

        unsafe { DXGIGetDebugInterface1::<IDXGIInfoQueue>(0) }
            .log_err()
            .is_some()
    }
    #[cfg(not(debug_assertions))]
    {
        false
    }
}

#[inline]
fn get_dxgi_factory(debug_layer_available: bool) -> Result<IDXGIFactory6> {
    let factory_flag = if debug_layer_available {
        DXGI_CREATE_FACTORY_DEBUG
    } else {
        #[cfg(debug_assertions)]
        log::warn!(
            "Failed to get DXGI debug interface. DirectX debugging features will be disabled."
        );
        DXGI_CREATE_FACTORY_FLAGS::default()
    };
    unsafe { Ok(CreateDXGIFactory2(factory_flag)?) }
}

/// Hardware adapters come first: the software rasterizer (Microsoft Basic
/// Render Driver) is only for when no GPU can make a device, such as while a
/// display driver is being installed.
#[inline]
fn get_adapter(
    dxgi_factory: &IDXGIFactory6,
    debug_layer_available: bool,
) -> Result<(
    IDXGIAdapter1,
    ID3D11Device,
    ID3D11DeviceContext,
    D3D_FEATURE_LEVEL,
)> {
    let (software, hardware): (Vec<_>, Vec<_>) =
        adapters(dxgi_factory).partition(|adapter| is_software(adapter));
    for adapter in hardware.into_iter().chain(software) {
        let gpu_name = adapter_name(&adapter);
        // Check to see whether the adapter supports Direct3D 11 and create
        // the device if it does.
        let mut context: Option<ID3D11DeviceContext> = None;
        let mut feature_level = D3D_FEATURE_LEVEL::default();
        match get_device(
            &adapter,
            Some(&mut context),
            Some(&mut feature_level),
            debug_layer_available,
        ) {
            Ok(device) => {
                if is_software(&adapter) {
                    log::warn!(
                        "Using GPU: {gpu_name} (software rendering: no hardware adapter made a device)"
                    );
                } else {
                    log::info!("Using GPU: {gpu_name}");
                }
                return Ok((adapter, device, context.unwrap(), feature_level));
            }
            Err(error) => log::warn!("Skipping GPU {gpu_name}: {error:#}"),
        }
    }
    Err(anyhow::anyhow!(
        "No DXGI adapter could create a Direct3D 11 device"
    ))
}

fn adapters(dxgi_factory: &IDXGIFactory6) -> impl Iterator<Item = IDXGIAdapter1> + '_ {
    (0..).map_while(move |index| unsafe { dxgi_factory.EnumAdapters1(index) }.ok())
}

fn adapter_name(adapter: &IDXGIAdapter1) -> String {
    unsafe { adapter.GetDesc1() }.map_or_else(
        |_| "<unknown>".to_owned(),
        |desc| {
            String::from_utf16_lossy(&desc.Description)
                .trim_matches(char::from(0))
                .to_string()
        },
    )
}

pub(crate) fn is_software(adapter: &IDXGIAdapter1) -> bool {
    unsafe { adapter.GetDesc1() }
        .is_ok_and(|desc| desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0)
}

/// Whether a hardware adapter can make a device now. A device made while the
/// GPU was away (a driver update or reset) lands on the software rasterizer,
/// which never reports itself lost; this is how recovery knows to move back.
pub(crate) fn hardware_device_available() -> bool {
    let Ok(factory) = get_dxgi_factory(false) else {
        return false;
    };
    adapters(&factory)
        .filter(|adapter| !is_software(adapter))
        .any(|adapter| get_device(&adapter, None, None, false).is_ok())
}

/// Calls `changed` whenever a GPU adapter is added or removed, until it
/// returns false. Display messages alone do not reliably follow a display
/// driver coming back after an install.
pub(crate) fn watch_adapter_changes(changed: impl Fn() -> bool + Send + 'static) {
    let watch = move || -> Result<()> {
        let factory: IDXGIFactory7 =
            unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS::default()) }
                .context("Creating DXGI factory")?;
        let event = unsafe { CreateEventW(None, false, false, None) }
            .context("Creating the adapter change event")?;
        let registered = unsafe { factory.RegisterAdaptersChangedEvent(event) }
            .context("Registering for adapter changes");
        if registered.is_ok() {
            while unsafe { WaitForSingleObject(event, INFINITE) } == WAIT_OBJECT_0 && changed() {}
        }
        unsafe { CloseHandle(event) }.log_err();
        registered.map(drop)
    };
    std::thread::Builder::new()
        .name("GpuAdapters".to_owned())
        .spawn(move || {
            watch().context("Watching GPU adapter changes").log_err();
        })
        .context("Starting the GPU adapter watcher")
        .log_err();
}

#[inline]
fn get_device(
    adapter: &IDXGIAdapter1,
    context: Option<*mut Option<ID3D11DeviceContext>>,
    feature_level: Option<*mut D3D_FEATURE_LEVEL>,
    debug_layer_available: bool,
) -> Result<ID3D11Device> {
    let mut device: Option<ID3D11Device> = None;
    let device_flags = if debug_layer_available {
        D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_DEBUG
    } else {
        D3D11_CREATE_DEVICE_BGRA_SUPPORT
    };
    unsafe {
        D3D11CreateDevice(
            adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            device_flags,
            // 4x MSAA is required for Direct3D Feature Level 10.1 or better
            Some(&[
                D3D_FEATURE_LEVEL_11_1,
                D3D_FEATURE_LEVEL_11_0,
                D3D_FEATURE_LEVEL_10_1,
            ]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            feature_level,
            context,
        )?;
    }
    let device = device.unwrap();
    let mut data = D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS::default();
    unsafe {
        device
            .CheckFeatureSupport(
                D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS,
                &mut data as *mut _ as _,
                std::mem::size_of::<D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS>() as u32,
            )
            .context("Checking GPU device feature support")?;
    }
    if data
        .ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x
        .as_bool()
    {
        Ok(device)
    } else {
        Err(anyhow::anyhow!(
            "Required feature StructuredBuffer is not supported by GPU/driver"
        ))
    }
}
