#![cfg(target_os = "windows")]

mod clipboard;
#[cfg(not(feature = "wgpu"))]
mod damage;
mod destination_list;
#[cfg(not(feature = "wgpu"))]
mod device_recovery;
mod direct_manipulation;
#[cfg(all(not(feature = "wgpu"), not(feature = "cosmic-text")))]
mod direct_write;
#[cfg(not(feature = "wgpu"))]
mod directx_atlas;
#[cfg(not(feature = "wgpu"))]
mod directx_devices;
#[cfg(not(feature = "wgpu"))]
mod directx_mesh;
#[cfg(not(feature = "wgpu"))]
mod directx_renderer;
mod dispatcher;
mod display;
mod events;
mod frame_signal;
mod keyboard;
mod monitor_vsync;
mod platform;
mod system_notifications;
mod system_settings;
mod taskbar_badge;
mod util;
mod window;
mod wrapper;

pub(crate) use clipboard::*;
pub(crate) use destination_list::*;
#[cfg(not(feature = "wgpu"))]
pub(crate) use device_recovery::DeviceRecoverySignal;
#[cfg(all(not(feature = "wgpu"), not(feature = "cosmic-text")))]
pub(crate) use direct_write::*;
#[cfg(not(feature = "wgpu"))]
pub(crate) use directx_atlas::*;
#[cfg(not(feature = "wgpu"))]
pub(crate) use directx_devices::*;
#[cfg(not(feature = "wgpu"))]
pub(crate) use directx_renderer::*;
pub(crate) use dispatcher::*;
pub(crate) use display::*;
pub(crate) use events::*;
pub(crate) use keyboard::*;
pub(crate) use platform::*;
pub(crate) use system_notifications::*;
pub(crate) use system_settings::*;
pub(crate) use util::*;
pub(crate) use window::*;
pub(crate) use wrapper::*;

pub use platform::WindowsPlatform;

pub(crate) use windows::Win32::Foundation::HWND;
