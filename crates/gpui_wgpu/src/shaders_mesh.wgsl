// Lit mesh pass: draws one `Mesh` into its own multisampled offscreen target
// with a depth buffer. The composite (shaders_mesh_composite.wgsl) then draws
// that target into the scene like an image.
//
// The camera sits at the view-space origin looking down -Z. Lighting is a
// fixed studio rig in view space, so a model reads well from every side:
// a warm key from the upper left, a cool fill from the right, a rim that
// lights grazing edges from behind, and a sky-to-ground ambient. Faces are
// two-sided: a normal facing away from the eye flips.
//
// Output is sRGB-encoded and premultiplied (alpha is 1 where the mesh
// covers a sample and 0 elsewhere), so the multisample resolve yields
// premultiplied, antialiased edges.

struct MeshUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    // rgb: linear color of uncolored vertices.
    base_color: vec4<f32>,
    // rgb: linear edge color; a: edge opacity, 0 for no edges.
    edge_color: vec4<f32>,
    // x: half the edge width, in pixels.
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> mesh: MeshUniforms;

struct MeshVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // rgb: linear color; a: 1 to use it, 0 to use the base color.
    @location(2) color: vec4<f32>,
}

struct MeshVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) view_position: vec3<f32>,
    @location(1) view_normal: vec3<f32>,
    @location(2) albedo: vec3<f32>,
    // One-hot per triangle corner; its screen-space derivative turns the
    // distance to the nearest edge into pixels.
    @location(3) barycentric: vec3<f32>,
}

@vertex
fn vs_mesh(input: MeshVertexInput, @builtin(vertex_index) vertex_index: u32) -> MeshVarying {
    let view_position = mesh.view * vec4<f32>(input.position, 1.0);
    var out: MeshVarying;
    out.position = mesh.projection * view_position;
    out.view_position = view_position.xyz;
    out.view_normal = (mesh.view * vec4<f32>(input.normal, 0.0)).xyz;
    out.albedo = select(mesh.base_color.rgb, input.color.rgb, input.color.a > 0.5);
    // Geometry is a triangle soup, so the vertex index names the corner.
    let corner = vertex_index % 3u;
    out.barycentric = vec3<f32>(
        f32(corner == 0u),
        f32(corner == 1u),
        f32(corner == 2u),
    );
    return out;
}

const KEY_COLOR: vec3<f32> = vec3<f32>(1.0, 0.96, 0.9);
const FILL_COLOR: vec3<f32> = vec3<f32>(0.26, 0.29, 0.34);
const RIM_COLOR: vec3<f32> = vec3<f32>(0.32, 0.32, 0.34);
const SKY_COLOR: vec3<f32> = vec3<f32>(0.3, 0.31, 0.34);
const GROUND_COLOR: vec3<f32> = vec3<f32>(0.12, 0.11, 0.1);

fn mesh_linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let clamped = clamp(linear, vec3<f32>(0.0), vec3<f32>(1.0));
    let low = clamped * 12.92;
    let high = 1.055 * pow(clamped, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, clamped <= vec3<f32>(0.0031308));
}

@fragment
fn fs_mesh(input: MeshVarying) -> @location(0) vec4<f32> {
    let eye = normalize(-input.view_position);
    var normal = input.view_normal;
    if (dot(normal, normal) < 1e-20) {
        normal = eye;
    }
    normal = normalize(normal);
    if (dot(normal, eye) < 0.0) {
        normal = -normal;
    }

    let key_direction = normalize(vec3<f32>(-0.45, 0.65, 0.6));
    let fill_direction = normalize(vec3<f32>(0.7, 0.05, 0.5));
    let rim_direction = normalize(vec3<f32>(0.15, 0.45, -0.9));
    let key_half = normalize(key_direction + vec3<f32>(0.0, 0.0, 1.0));

    let sky = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(GROUND_COLOR, SKY_COLOR, sky);
    let key = max(dot(normal, key_direction), 0.0);
    let fill = max(dot(normal, fill_direction), 0.0);
    let grazing = 1.0 - clamp(dot(normal, eye), 0.0, 1.0);
    let rim = max(dot(normal, rim_direction), 0.0) * grazing;
    let light = ambient + KEY_COLOR * key + FILL_COLOR * fill + RIM_COLOR * rim;
    let specular = pow(max(dot(normal, key_half), 0.0), 48.0) * 0.22 * sqrt(min(key, 1.0));
    var color = input.albedo * light + KEY_COLOR * specular;

    if (mesh.edge_color.a > 0.0) {
        let pixels = input.barycentric / max(fwidth(input.barycentric), vec3<f32>(1e-6));
        let distance = min(pixels.x, min(pixels.y, pixels.z));
        let cover = clamp(mesh.params.x + 0.5 - distance, 0.0, 1.0) * mesh.edge_color.a;
        color = mix(color, mesh.edge_color.rgb, cover);
    }

    return vec4<f32>(mesh_linear_to_srgb(color), 1.0);
}
