// The matrices use the public MeshStyle column-major representation.
cbuffer MeshUniforms : register(b2) {
    column_major float4x4 mesh_view;
    column_major float4x4 mesh_projection;
    float4 mesh_base_color;
    float4 mesh_edge_color;
    float4 mesh_params;
};

struct MeshInput {
    float3 position : POSITION;
    float4 normal : NORMAL;
    // rgb: sRGB-encoded color; a: 1 to use it, 0 to use the base color.
    float4 color : COLOR;
};
struct MeshVarying {
    float4 position : SV_Position;
    float3 view_position : TEXCOORD0;
    float3 view_normal : TEXCOORD1;
    float3 albedo : TEXCOORD2;
    float3 barycentric : TEXCOORD3;
};

float3 mesh_srgb_to_linear(float3 color) {
    float3 low = color / 12.92;
    float3 high = pow((color + 0.055) / 1.055, 2.4);
    return float3(color.r <= 0.04045 ? low.r : high.r,
                  color.g <= 0.04045 ? low.g : high.g,
                  color.b <= 0.04045 ? low.b : high.b);
}

MeshVarying mesh_vertex(MeshInput input, uint vertex_id : SV_VertexID) {
    float4 view_position = mul(mesh_view, float4(input.position, 1.0));
    MeshVarying output;
    output.position = mul(mesh_projection, view_position);
    output.view_position = view_position.xyz;
    output.view_normal = mul(mesh_view, float4(input.normal.xyz, 0.0)).xyz;
    output.albedo = input.color.a > 0.5 ? mesh_srgb_to_linear(input.color.rgb) : mesh_base_color.rgb;
    uint corner = vertex_id % 3;
    output.barycentric = float3(corner == 0, corner == 1, corner == 2);
    return output;
}

float3 mesh_linear_to_srgb(float3 value) {
    float3 color = saturate(value);
    float3 low = color * 12.92;
    float3 high = 1.055 * pow(color, 1.0 / 2.4) - 0.055;
    return float3(color.r <= 0.0031308 ? low.r : high.r,
                  color.g <= 0.0031308 ? low.g : high.g,
                  color.b <= 0.0031308 ? low.b : high.b);
}

float4 mesh_fragment(MeshVarying input) : SV_Target {
    const float3 key_color = float3(1.0, 0.96, 0.9);
    float3 eye = normalize(-input.view_position);
    float3 normal = input.view_normal;
    if (dot(normal, normal) < 1e-20) normal = eye;
    normal = normalize(normal);
    if (dot(normal, eye) < 0.0) normal = -normal;
    float3 key_direction = normalize(float3(-0.45, 0.65, 0.6));
    float3 fill_direction = normalize(float3(0.7, 0.05, 0.5));
    float3 rim_direction = normalize(float3(0.15, 0.45, -0.9));
    float3 key_half = normalize(key_direction + float3(0.0, 0.0, 1.0));
    float sky = saturate(normal.y * 0.5 + 0.5);
    float3 ambient = lerp(float3(0.12, 0.11, 0.1), float3(0.3, 0.31, 0.34), sky);
    float key = max(dot(normal, key_direction), 0.0);
    float fill = max(dot(normal, fill_direction), 0.0);
    float grazing = 1.0 - saturate(dot(normal, eye));
    float rim = max(dot(normal, rim_direction), 0.0) * grazing;
    float3 light = ambient + key_color * key + float3(0.26, 0.29, 0.34) * fill
        + float3(0.32, 0.32, 0.34) * rim;
    float specular = pow(max(dot(normal, key_half), 0.0), 48.0) * 0.22 * sqrt(min(key, 1.0));
    float3 color = input.albedo * light + key_color * specular;
    if (mesh_edge_color.a > 0.0) {
        float3 pixels = input.barycentric / max(fwidth(input.barycentric), 1e-6);
        float distance = min(pixels.x, min(pixels.y, pixels.z));
        float cover = saturate(mesh_params.x + 0.5 - distance) * mesh_edge_color.a;
        color = lerp(color, mesh_edge_color.rgb, cover);
    }
    return float4(mesh_linear_to_srgb(color), 1.0);
}

// Separate slots leave the scene's globals (b0) and batch/blur parameters (b1) intact.
cbuffer MeshCompositeParams : register(b3) {
    float4 mesh_bounds;
    float4 mesh_clip;
    float4 mesh_corners;
    float4 mesh_composite_params;
    float4 mesh_viewport;
};
Texture2D<float4> mesh_texture : register(t0);
SamplerState mesh_sampler : register(s0);

struct CompositeVarying {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 clip_distances : TEXCOORD1;
};

CompositeVarying mesh_composite_vertex(uint vertex_id : SV_VertexID) {
    float2 unit = float2(vertex_id & 1, 0.5 * (vertex_id & 2));
    float2 position = mesh_bounds.xy + unit * mesh_bounds.zw;
    CompositeVarying output;
    output.position = float4(2.0 * position.x / mesh_viewport.x - 1.0,
                            1.0 - 2.0 * position.y / mesh_viewport.y, 0.0, 1.0);
    output.uv = unit;
    output.clip_distances = float4(position - mesh_clip.xy,
                                    mesh_clip.xy + mesh_clip.zw - position);
    return output;
}

float4 mesh_composite_fragment(CompositeVarying input) : SV_Target {
    if (any(input.clip_distances < 0.0)) return 0.0;
    float2 center_to_point = input.position.xy - (mesh_bounds.xy + mesh_bounds.zw * 0.5);
    float radius = center_to_point.x < 0.0
        ? (center_to_point.y < 0.0 ? mesh_corners.x : mesh_corners.w)
        : (center_to_point.y < 0.0 ? mesh_corners.y : mesh_corners.z);
    float2 q = abs(center_to_point) - mesh_bounds.zw * 0.5 + radius;
    float distance = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
    return mesh_texture.SampleLevel(mesh_sampler, input.uv, 0.0)
        * (mesh_composite_params.x * saturate(0.5 - distance));
}
