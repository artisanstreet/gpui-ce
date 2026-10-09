// --- mesh composite --- //
//
// Draws a mesh's resolved offscreen target into the scene at the mesh's
// bounds, clipped to its content mask and rounded corners and scaled by the
// element's opacity. The target is premultiplied, so this outputs
// premultiplied color and blends premultiplied whatever the window's alpha
// mode (as the blur composite does).

struct MeshCompositeParams {
    bounds: Bounds,
    content_mask: Bounds,
    corner_radii: Corners,
    // x: opacity.
    params: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mesh_composite: MeshCompositeParams;
@group(1) @binding(1) var t_mesh: texture_2d<f32>;
@group(1) @binding(2) var s_mesh: sampler;

struct MeshCompositeVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) texture_position: vec2<f32>,
    @location(3) clip_distances: vec4<f32>,
}

@vertex
fn vs_mesh_composite(@builtin(vertex_index) vertex_id: u32) -> MeshCompositeVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    var out = MeshCompositeVarying();
    out.position = to_device_position(unit_vertex, mesh_composite.bounds);
    out.texture_position = unit_vertex;
    out.clip_distances = distance_from_clip_rect(
        unit_vertex,
        mesh_composite.bounds,
        mesh_composite.content_mask,
    );
    return out;
}

@fragment
fn fs_mesh_composite(input: MeshCompositeVarying) -> @location(0) vec4<f32> {
    if (any(input.clip_distances < vec4<f32>(0.0))) {
        return vec4<f32>(0.0);
    }
    let sample = textureSampleLevel(t_mesh, s_mesh, input.texture_position, 0.0);
    let distance = quad_sdf(input.position.xy, mesh_composite.bounds, mesh_composite.corner_radii);
    return sample * (mesh_composite.params.x * saturate(0.5 - distance));
}
