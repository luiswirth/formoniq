// Line-field G-buffer pass: rasterize the surface, writing per fragment the
// screen-projected tangent direction, the nodal magnitude, a coverage mask, the
// world position (so the LIC noise is sampled in object space and sticks to the
// surface), and a Lambert shade term. The fullscreen LIC pass consumes these;
// see `lic.wgsl`.

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Only `viewport` is read here (to turn an NDC direction into a pixel one);
// the rest of the struct is shared with the LIC pass.
struct Lic {
    viewport: vec2<f32>,
    noise_scale: f32,
    omega: f32,
    time: f32,
    contrast: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(1) @binding(0)
var<uniform> lic: Lic;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) value: f32,
    @location(3) direction: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) magnitude: f32,
    // Tangent direction in NDC, as the difference of two projected points: not
    // unit and not linear under perspective, but good enough per fragment for a
    // v1 screen-space LIC. Normalized to a pixel direction in the fragment.
    @location(3) ndc_dir: vec2<f32>,
};

struct GBufferOut {
    @location(0) dir_mag: vec4<f32>,
    @location(1) pos_shade: vec4<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let clip0 = camera.view_proj * vec4<f32>(model.position, 1.0);
    // A small step along the world tangent, projected; the magnitude cancels
    // when the fragment normalizes, so any small epsilon recovers the tangent's
    // screen direction.
    let eps = 1e-3;
    let clip1 = camera.view_proj * vec4<f32>(model.position + eps * model.direction, 1.0);
    out.clip_position = clip0;
    out.world_pos = model.position;
    out.world_normal = model.normal;
    out.magnitude = model.value;
    out.ndc_dir = clip1.xy / clip1.w - clip0.xy / clip0.w;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOut {
    var out: GBufferOut;

    // NDC-to-pixel: x scales by half the viewport width, y flips (NDC up vs.
    // framebuffer down) and scales by half the height. Direction only, so the
    // shared 0.5 factor drops out under the normalize.
    let px_dir = vec2<f32>(in.ndc_dir.x * lic.viewport.x, -in.ndc_dir.y * lic.viewport.y);
    let len = length(px_dir);
    var sdir = vec2<f32>(0.0, 0.0);
    if (len > 1e-8) {
        sdir = px_dir / len;
    }

    let n = normalize(in.world_normal);
    // Two-sided Lambert against a fixed light, plus ambient: the surface is a
    // backdrop for the LIC, so the shading only needs to read as curvature.
    let light = normalize(vec3<f32>(0.4, 0.6, 1.0));
    let shade = 0.35 + 0.65 * abs(dot(n, light));

    out.dir_mag = vec4<f32>(sdir, in.magnitude, 1.0);
    out.pos_shade = vec4<f32>(in.world_pos, shade);
    return out;
}
