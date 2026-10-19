//! Lit meshes rendered into independent resolves with shared MSAA/depth scratch.
//! Composite at the mesh batch's scene position to retain 2D ordering and filters.

use std::{
    collections::HashMap,
    mem::size_of,
    ops::Range,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use gpui::{MeshId, MeshVertex, PaintMesh};
use windows::{
    Win32::Graphics::{Direct3D::*, Direct3D11::*, Dxgi::Common::*},
    core::s,
};

const COLOR_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
const SAMPLE_COUNT: u32 = 4;
const MAX_TARGET_SIDE: u32 = 4096;
const EVICT_AFTER_FRAMES: u64 = 120;
const MAX_VERTEX_BYTES: usize = 256 * 1024 * 1024;
/// The multisampled color and depth scratch is only drawn through: once rendered, a mesh
/// composites from its resolve. Renders this close together (a drag, an orbit) keep the scratch;
/// a lone render, or one this long after the last, frees it. A 1120x600 viewer's 4x color and
/// depth held 21 MB for as long as the model was on screen.
const SCRATCH_KEEP: Duration = Duration::from_secs(1);

use crate::directx_renderer::shader_resources::{RawShaderBytes, ShaderModule, ShaderTarget};

#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
struct MeshUniforms {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
    base_color: [f32; 4],
    edge_color: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
struct CompositeParams {
    bounds: [f32; 4],
    content_mask: [f32; 4],
    corner_radii: [f32; 4],
    params: [f32; 4],
    viewport: [f32; 4],
}

const _: () = {
    assert!(size_of::<MeshVertex>() == 24);
    assert!(std::mem::offset_of!(MeshVertex, normal) == 12);
    assert!(std::mem::offset_of!(MeshVertex, color) == 20);
    assert!(size_of::<MeshUniforms>() == 176);
    assert!(size_of::<CompositeParams>() == 80);
};

struct Geometry {
    buffer: Option<ID3D11Buffer>,
    vertex_count: u32,
    last_painted: u64,
}

struct Target {
    size: (u32, u32),
    /// What `resolve` holds; a frame asking for the same skips the render.
    rendered: Option<(MeshId, MeshUniforms)>,
    resolve: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
    uniforms: ID3D11Buffer,
    composite: ID3D11Buffer,
}

struct Scratch {
    color: ID3D11Texture2D,
    color_view: ID3D11RenderTargetView,
    depth_view: ID3D11DepthStencilView,
    last_used: u64,
}

pub(crate) struct MeshRenderer {
    vertex: ID3D11VertexShader,
    fragment: ID3D11PixelShader,
    composite_vertex: ID3D11VertexShader,
    composite_fragment: ID3D11PixelShader,
    input_layout: ID3D11InputLayout,
    depth_state: ID3D11DepthStencilState,
    rasterizer: ID3D11RasterizerState,
    composite_blend: ID3D11BlendState,
    sampler: ID3D11SamplerState,
    geometry: HashMap<MeshId, Geometry>,
    targets: Vec<Target>,
    scratch: HashMap<(u32, u32), Scratch>,
    frame: u64,
    last_prepare: Option<Instant>,
}

impl MeshRenderer {
    pub(crate) fn new(device: &ID3D11Device) -> Result<Self> {
        let vertex = RawShaderBytes::new(ShaderModule::Mesh, ShaderTarget::Vertex)?;
        let fragment = RawShaderBytes::new(ShaderModule::Mesh, ShaderTarget::Fragment)?;
        let composite_vertex =
            RawShaderBytes::new(ShaderModule::MeshComposite, ShaderTarget::Vertex)?;
        let composite_fragment =
            RawShaderBytes::new(ShaderModule::MeshComposite, ShaderTarget::Fragment)?;
        let vertex_shader = |bytes: &[u8]| -> Result<ID3D11VertexShader> {
            let mut shader = None;
            unsafe { device.CreateVertexShader(bytes, None, Some(&mut shader))? };
            shader.context("mesh vertex shader missing")
        };
        let pixel_shader = |bytes: &[u8]| -> Result<ID3D11PixelShader> {
            let mut shader = None;
            unsafe { device.CreatePixelShader(bytes, None, Some(&mut shader))? };
            shader.context("mesh pixel shader missing")
        };
        let mut input_layout = None;
        let elements = [
            (s!("POSITION"), DXGI_FORMAT_R32G32B32_FLOAT, 0),
            (s!("NORMAL"), DXGI_FORMAT_R16G16B16A16_SNORM, 12),
            (s!("COLOR"), DXGI_FORMAT_R8G8B8A8_UNORM, 20),
        ]
        .map(|(name, format, offset)| D3D11_INPUT_ELEMENT_DESC {
            SemanticName: name,
            Format: format,
            AlignedByteOffset: offset,
            InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
            ..Default::default()
        });
        unsafe { device.CreateInputLayout(&elements, vertex.as_bytes(), Some(&mut input_layout))? };
        let mut depth_state = None;
        unsafe {
            device.CreateDepthStencilState(
                &D3D11_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D11_COMPARISON_LESS,
                    ..Default::default()
                },
                Some(&mut depth_state),
            )?;
        }
        let mut rasterizer = None;
        unsafe {
            device.CreateRasterizerState(
                &D3D11_RASTERIZER_DESC {
                    FillMode: D3D11_FILL_SOLID,
                    CullMode: D3D11_CULL_NONE,
                    DepthClipEnable: true.into(),
                    MultisampleEnable: true.into(),
                    ..Default::default()
                },
                Some(&mut rasterizer),
            )?;
        }
        let mut blend_desc = D3D11_BLEND_DESC::default();
        blend_desc.RenderTarget[0] = D3D11_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            SrcBlend: D3D11_BLEND_ONE,
            DestBlend: D3D11_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D11_BLEND_OP_ADD,
            SrcBlendAlpha: D3D11_BLEND_ONE,
            DestBlendAlpha: D3D11_BLEND_INV_SRC_ALPHA,
            BlendOpAlpha: D3D11_BLEND_OP_ADD,
            RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
        };
        let mut composite_blend = None;
        unsafe { device.CreateBlendState(&blend_desc, Some(&mut composite_blend))? };
        let mut sampler = None;
        unsafe {
            device.CreateSamplerState(
                &D3D11_SAMPLER_DESC {
                    Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                    AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                    ComparisonFunc: D3D11_COMPARISON_NEVER,
                    MaxLOD: f32::MAX,
                    ..Default::default()
                },
                Some(&mut sampler),
            )?;
        }
        Ok(Self {
            vertex: vertex_shader(vertex.as_bytes())?,
            fragment: pixel_shader(fragment.as_bytes())?,
            composite_vertex: vertex_shader(composite_vertex.as_bytes())?,
            composite_fragment: pixel_shader(composite_fragment.as_bytes())?,
            input_layout: input_layout.context("mesh input layout missing")?,
            depth_state: depth_state.context("mesh depth state missing")?,
            rasterizer: rasterizer.context("mesh rasterizer missing")?,
            composite_blend: composite_blend.context("mesh blend state missing")?,
            sampler: sampler.context("mesh sampler missing")?,
            geometry: HashMap::new(),
            targets: Vec::new(),
            scratch: HashMap::new(),
            frame: 0,
            last_prepare: None,
        })
    }

    /// Frees the scratch once [`SCRATCH_KEEP`] has passed since the last render.
    pub(crate) fn trim_idle(&mut self) {
        if self
            .last_prepare
            .is_some_and(|last| last.elapsed() >= SCRATCH_KEEP)
        {
            self.scratch.clear();
        }
    }

    pub(crate) fn clear(&mut self) {
        self.geometry.clear();
        self.targets.clear();
        self.scratch.clear();
    }

    pub(crate) fn prepare(
        &mut self,
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        meshes: &[PaintMesh],
        viewport: (u32, u32),
    ) -> Result<()> {
        if meshes.is_empty() {
            self.clear();
            return Ok(());
        }
        let _restore = MeshStateRestore(context);
        self.frame += 1;
        self.targets.truncate(meshes.len());
        let mut drew = false;
        for (index, paint) in meshes.iter().enumerate() {
            let side = |length: f32| (length.ceil().max(1.0) as u32).min(MAX_TARGET_SIDE);
            let size = (
                side(paint.bounds.size.width.0),
                side(paint.bounds.size.height.0),
            );
            if self
                .targets
                .get(index)
                .is_none_or(|target| target.size != size)
            {
                let target = Target::new(device, size)?;
                if index < self.targets.len() {
                    self.targets[index] = target;
                } else {
                    self.targets.push(target);
                }
            }
            if let std::collections::hash_map::Entry::Vacant(entry) =
                self.geometry.entry(paint.mesh.id())
            {
                entry.insert(Geometry::new(device, paint)?);
            }
            let geometry = self.geometry.get_mut(&paint.mesh.id()).unwrap();
            geometry.last_painted = self.frame;
            let style = &paint.style;
            let uniforms = MeshUniforms {
                view: style.view,
                projection: style.projection,
                base_color: [
                    style.base_color[0],
                    style.base_color[1],
                    style.base_color[2],
                    1.0,
                ],
                edge_color: style.edge_color,
                params: [0.6, 0.0, 0.0, 0.0],
            };
            let bounds = |b: gpui::Bounds<gpui::ScaledPixels>| {
                [b.origin.x.0, b.origin.y.0, b.size.width.0, b.size.height.0]
            };
            let corners = paint.corner_radii;
            let composite = CompositeParams {
                bounds: bounds(paint.bounds),
                content_mask: bounds(paint.content_mask.bounds),
                corner_radii: [
                    corners.top_left.0,
                    corners.top_right.0,
                    corners.bottom_right.0,
                    corners.bottom_left.0,
                ],
                params: [paint.opacity, 0.0, 0.0, 0.0],
                viewport: [viewport.0 as f32, viewport.1 as f32, 0.0, 0.0],
            };
            // These repr(C) records contain only f32 arrays, with no padding.
            unsafe {
                context.UpdateSubresource(
                    &self.targets[index].composite,
                    0,
                    None,
                    &composite as *const _ as _,
                    0,
                    0,
                );
            }
            // Moving or clipping the viewer only changes the composite: the resolve already
            // holds this view of this mesh.
            let wanted = (paint.mesh.id(), uniforms);
            if self.targets[index].rendered == Some(wanted) {
                continue;
            }
            drew = true;
            if let std::collections::hash_map::Entry::Vacant(entry) = self.scratch.entry(size) {
                entry.insert(Scratch::new(device, size)?);
            }
            let scratch = self.scratch.get_mut(&size).unwrap();
            scratch.last_used = self.frame;
            let target = &mut self.targets[index];
            target.rendered = Some(wanted);
            unsafe {
                context.UpdateSubresource(
                    &target.uniforms,
                    0,
                    None,
                    &uniforms as *const _ as _,
                    0,
                    0,
                );
                context.PSSetShaderResources(0, Some(&[None]));
                context.ClearRenderTargetView(&scratch.color_view, &[0.0; 4]);
                context.ClearDepthStencilView(&scratch.depth_view, D3D11_CLEAR_DEPTH.0, 1.0, 0);
                context.OMSetRenderTargets(
                    Some(&[Some(scratch.color_view.clone())]),
                    &scratch.depth_view,
                );
                context.OMSetDepthStencilState(&self.depth_state, 0);
                context.OMSetBlendState(None, None, u32::MAX);
                context.RSSetState(&self.rasterizer);
                context.RSSetViewports(Some(&[D3D11_VIEWPORT {
                    Width: size.0 as f32,
                    Height: size.1 as f32,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]));
                context.IASetInputLayout(&self.input_layout);
                context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                context.VSSetShader(&self.vertex, None);
                context.PSSetShader(&self.fragment, None);
                context.VSSetConstantBuffers(2, Some(&[Some(target.uniforms.clone())]));
                context.PSSetConstantBuffers(2, Some(&[Some(target.uniforms.clone())]));
                if geometry.vertex_count != 0 {
                    context.IASetVertexBuffers(
                        0,
                        1,
                        Some(&geometry.buffer),
                        Some(&(size_of::<MeshVertex>() as u32)),
                        Some(&0),
                    );
                    context.Draw(geometry.vertex_count, 0);
                }
                context.OMSetRenderTargets(None, None);
                context.ResolveSubresource(&target.resolve, 0, &scratch.color, 0, COLOR_FORMAT);
            }
        }
        if !drew {
            self.trim_idle();
            self.geometry
                .retain(|_, geometry| self.frame - geometry.last_painted <= EVICT_AFTER_FRAMES);
            return Ok(());
        }
        let now = Instant::now();
        let rendering_again = self
            .last_prepare
            .is_some_and(|last| now.duration_since(last) < SCRATCH_KEEP);
        self.last_prepare = Some(now);
        if rendering_again {
            self.scratch
                .retain(|_, target| target.last_used == self.frame);
        } else {
            // The immediate context keeps what queued work still needs.
            self.scratch.clear();
        }
        self.geometry
            .retain(|_, geometry| self.frame - geometry.last_painted <= EVICT_AFTER_FRAMES);
        Ok(())
    }

    /// The caller already bound the current scene or content-filter target and viewport.
    pub(crate) fn composite(&self, context: &ID3D11DeviceContext, range: Range<usize>) {
        unsafe {
            context.IASetInputLayout(None);
            context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
            context.VSSetShader(&self.composite_vertex, None);
            context.PSSetShader(&self.composite_fragment, None);
            context.OMSetDepthStencilState(None, 0);
            context.OMSetBlendState(&self.composite_blend, None, u32::MAX);
            context.PSSetSamplers(0, Some(&[Some(self.sampler.clone())]));
            for target in &self.targets[range] {
                context.VSSetConstantBuffers(3, Some(&[Some(target.composite.clone())]));
                context.PSSetConstantBuffers(3, Some(&[Some(target.composite.clone())]));
                context.PSSetShaderResources(0, Some(&[Some(target.view.clone())]));
                context.Draw(4, 0);
            }
            context.PSSetShaderResources(0, Some(&[None]));
            context.VSSetConstantBuffers(3, Some(&[None]));
            context.PSSetConstantBuffers(3, Some(&[None]));
        }
    }
}

impl Geometry {
    fn new(device: &ID3D11Device, paint: &PaintMesh) -> Result<Self> {
        let vertices = paint.mesh.vertices();
        let bytes = std::mem::size_of_val(vertices);
        anyhow::ensure!(
            bytes <= MAX_VERTEX_BYTES,
            "mesh vertex buffer exceeds 256 MiB"
        );
        let mut buffer = None;
        if !vertices.is_empty() {
            unsafe {
                device.CreateBuffer(
                    &D3D11_BUFFER_DESC {
                        ByteWidth: bytes as u32,
                        Usage: D3D11_USAGE_IMMUTABLE,
                        BindFlags: D3D11_BIND_VERTEX_BUFFER.0 as u32,
                        ..Default::default()
                    },
                    Some(&D3D11_SUBRESOURCE_DATA {
                        pSysMem: vertices.as_ptr() as _,
                        ..Default::default()
                    }),
                    Some(&mut buffer),
                )?;
            }
        }
        Ok(Self {
            buffer,
            vertex_count: vertices.len() as u32,
            last_painted: 0,
        })
    }
}

impl Target {
    fn new(device: &ID3D11Device, size: (u32, u32)) -> Result<Self> {
        let resolve = texture(device, size, COLOR_FORMAT, 1, D3D11_BIND_SHADER_RESOURCE)?;
        let mut view = None;
        unsafe { device.CreateShaderResourceView(&resolve, None, Some(&mut view))? };
        Ok(Self {
            size,
            rendered: None,
            resolve,
            view: view.context("mesh resolve view missing")?,
            uniforms: constant_buffer::<MeshUniforms>(device)?,
            composite: constant_buffer::<CompositeParams>(device)?,
        })
    }
}

impl Scratch {
    fn new(device: &ID3D11Device, size: (u32, u32)) -> Result<Self> {
        let color = texture(
            device,
            size,
            COLOR_FORMAT,
            SAMPLE_COUNT,
            D3D11_BIND_RENDER_TARGET,
        )?;
        let depth = texture(
            device,
            size,
            DXGI_FORMAT_D32_FLOAT,
            SAMPLE_COUNT,
            D3D11_BIND_DEPTH_STENCIL,
        )?;
        let mut color_view = None;
        let mut depth_view = None;
        unsafe {
            device.CreateRenderTargetView(&color, None, Some(&mut color_view))?;
            device.CreateDepthStencilView(&depth, None, Some(&mut depth_view))?;
        }
        Ok(Self {
            color,
            color_view: color_view.context("mesh color view missing")?,
            depth_view: depth_view.context("mesh depth view missing")?,
            last_used: 0,
        })
    }
}

fn constant_buffer<T>(device: &ID3D11Device) -> Result<ID3D11Buffer> {
    let mut buffer = None;
    unsafe {
        device.CreateBuffer(
            &D3D11_BUFFER_DESC {
                ByteWidth: size_of::<T>() as u32,
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                ..Default::default()
            },
            None,
            Some(&mut buffer),
        )?;
    }
    buffer.context("mesh constant buffer missing")
}

fn texture(
    device: &ID3D11Device,
    size: (u32, u32),
    format: DXGI_FORMAT,
    samples: u32,
    bind: D3D11_BIND_FLAG,
) -> Result<ID3D11Texture2D> {
    let mut texture = None;
    unsafe {
        device.CreateTexture2D(
            &D3D11_TEXTURE2D_DESC {
                Width: size.0,
                Height: size.1,
                MipLevels: 1,
                ArraySize: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: samples,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: bind.0 as u32,
                ..Default::default()
            },
            None,
            Some(&mut texture),
        )?;
    }
    texture.context("mesh texture missing")
}

#[cfg(test)]
#[path = "directx_mesh_tests.rs"]
mod tests;

// Fallible texture/geometry allocation can interrupt prepare between mesh passes.
// Always release the mesh input/depth bindings before a later ordinary scene draw.
struct MeshStateRestore<'a>(&'a ID3D11DeviceContext);
impl Drop for MeshStateRestore<'_> {
    fn drop(&mut self) {
        unsafe {
            self.0.OMSetRenderTargets(None, None);
            self.0.IASetInputLayout(None);
            self.0
                .IASetVertexBuffers(0, 1, Some(&None), Some(&0), Some(&0));
            self.0.VSSetConstantBuffers(2, Some(&[None]));
            self.0.PSSetConstantBuffers(2, Some(&[None]));
            self.0.OMSetDepthStencilState(None, 0);
        }
    }
}
