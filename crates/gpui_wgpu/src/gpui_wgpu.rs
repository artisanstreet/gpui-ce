//! Cosmic text shaping, with an optional WGPU renderer.
//!
//! Native Windows uses the text system without the `renderer` feature. Platforms
//! that draw with WGPU enable it; direct users get it through the default features.

mod cosmic_text_system;
#[cfg(feature = "renderer")]
mod wgpu_atlas;
#[cfg(feature = "renderer")]
mod wgpu_context;
#[cfg(feature = "renderer")]
mod wgpu_mesh;
#[cfg(feature = "renderer")]
mod wgpu_renderer;

pub use cosmic_text_system::*;
#[cfg(feature = "renderer")]
pub use wgpu;
#[cfg(feature = "renderer")]
pub use wgpu_atlas::*;
#[cfg(feature = "renderer")]
pub use wgpu_context::*;
#[cfg(feature = "renderer")]
pub use wgpu_renderer::{GpuContext, WgpuRenderer, WgpuSurfaceConfig};
