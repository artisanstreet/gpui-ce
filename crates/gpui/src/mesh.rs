//! Lit triangle meshes painted by the GPU.
//!
//! A [`Mesh`] is triangle-soup geometry: three [`MeshVertex`] per triangle,
//! each with its own normal and optional color. The renderer uploads a mesh
//! once, keyed by its [`MeshId`], and keeps it while frames keep painting
//! it, so turning a model only changes its [`MeshStyle`].
//!
//! [`crate::Window::paint_mesh`] draws a mesh into element bounds: the
//! renderer rasterizes it offscreen with a depth buffer and multisampling,
//! then composites the result like an image, so clipping, rounded corners,
//! opacity, and draw order behave as they do for every other primitive. The
//! background stays transparent.
//!
//! Only the wgpu renderer draws meshes; check [`crate::Window::supports_meshes`]
//! before relying on one, and draw a fallback otherwise.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Identifies a [`Mesh`]'s geometry to the renderer's cache.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MeshId(pub usize);

/// One triangle corner.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct MeshVertex {
    /// Position in model space.
    pub position: [f32; 3],
    /// Unit normal in model space.
    pub normal: [f32; 3],
    /// Linear RGB and a flag in alpha: `1.0` uses this color, `0.0` uses the
    /// style's [`MeshStyle::base_color`].
    pub color: [f32; 4],
}

/// Triangle-soup geometry: every three vertices form one triangle.
#[derive(Debug)]
pub struct Mesh {
    id: MeshId,
    vertices: Vec<MeshVertex>,
}

impl Mesh {
    /// Geometry from `vertices`; a trailing partial triangle is dropped.
    pub fn new(mut vertices: Vec<MeshVertex>) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        vertices.truncate(vertices.len() / 3 * 3);
        Self {
            id: MeshId(NEXT_ID.fetch_add(1, Ordering::Relaxed)),
            vertices,
        }
    }

    /// The id the renderer caches this geometry under.
    pub fn id(&self) -> MeshId {
        self.id
    }

    /// Three vertices per triangle.
    pub fn vertices(&self) -> &[MeshVertex] {
        &self.vertices
    }
}

/// How one paint of a mesh looks. Matrices are column-major (each inner
/// array is a column), as WGSL lays out `mat4x4<f32>`.
///
/// The camera sits at the view-space origin looking down -Z. Lighting is a
/// fixed studio rig in view space (a warm key from the upper left, a cool
/// fill from the right, a rim from behind, and a sky-to-ground ambient), so
/// a model looks lit from every side. Faces are two-sided.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshStyle {
    /// Model space to view space. Normals use its upper 3×3, so keep it a
    /// rotation, translation, and uniform scale.
    pub view: [[f32; 4]; 4],
    /// View space to clip space.
    pub projection: [[f32; 4]; 4],
    /// Linear RGB of vertices that carry no color of their own.
    pub base_color: [f32; 3],
    /// Linear RGB and opacity of triangle edges; opacity `0.0` draws none.
    pub edge_color: [f32; 4],
}
