// Wireframe shader — draws mesh edges as screen-space-thick lines.
//
// A `LineList` primitive rasterizes at a fixed, backend-defined 1px in wgpu
// (there is no portable line-width control), which all but disappears once
// the supersampled scene pass is box-filtered down to the swapchain. Instead
// each edge is drawn as an instanced quad: two triangles fanned from its two
// endpoints, offset perpendicular to the edge in screen space by a fixed
// pixel half-width. The two endpoints arrive as two separate per-instance
// vertex buffers (see `Vertex::desc_endpoint_a`/`desc_endpoint_b` in
// `mesh.rs`) bound side by side, since a `step_mode: Vertex` buffer only ever
// exposes the current vertex to the shader, never a neighbor -- both ends of
// the same edge have to be visible to one invocation to compute the
// perpendicular.

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Same standing-wave displacement as the fill shader, so the wireframe tracks
// the displaced surface instead of the flat rest mesh. A line field sets the
// wave amplitude to zero, so its wireframe stays on the undisplaced surface.
struct Wave {
    time: f32,
    amplitude: f32,
    omega: f32,
    _pad: f32,
};
@group(1) @binding(0)
var<uniform> wave: Wave;

// The supersampled render-target size (so a pixel half-width means the same
// thing regardless of window size or the supersampling factor) and the
// line's half-width in that same pixel space.
struct Screen {
    viewport: vec2<f32>,
    half_width_px: f32,
    _pad: f32,
};
@group(2) @binding(0)
var<uniform> screen: Screen;

struct EndpointA {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) value: f32,
    @location(4) max_displacement: f32,
};
struct EndpointB {
    @location(5) position: vec3<f32>,
    @location(6) normal: vec3<f32>,
    @location(7) value: f32,
    @location(9) max_displacement: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// The same normal-displaced clip position the fill/scalar shader computes for
// a single vertex, reused here per endpoint.
fn displaced_clip(position: vec3<f32>, normal: vec3<f32>, value: f32, max_displacement: f32) -> vec4<f32> {
    let osc = cos(wave.omega * wave.time);
    let raw = wave.amplitude * osc * value;
    let capped = clamp(raw, -max_displacement, max_displacement);
    return camera.view_proj * vec4<f32>(position + capped * normal, 1.0);
}

@vertex
fn vs_main(a: EndpointA, b: EndpointB, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let clip_a = displaced_clip(a.position, a.normal, a.value, a.max_displacement);
    let clip_b = displaced_clip(b.position, b.normal, b.value, b.max_displacement);

    // The quad's two triangles (A-, B-, B+) and (A-, B+, A+): six corners,
    // each a (which endpoint is "self", which side of the edge) pair.
    var self_is_b = array<bool, 6>(false, true, true, false, true, false);
    var side = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let is_b = self_is_b[vertex_index];
    let self_clip = select(clip_a, clip_b, is_b);
    let other_clip = select(clip_b, clip_a, is_b);

    // The edge's direction in screen (pixel) space, and its perpendicular:
    // the offset every corner is pushed along, scaled by `self_clip.w` so it
    // survives the perspective divide as a constant on-screen width
    // regardless of depth.
    let ndc_self = self_clip.xy / self_clip.w;
    let ndc_other = other_clip.xy / other_clip.w;
    let px_delta = (ndc_other - ndc_self) * screen.viewport;
    let dir = px_delta / max(length(px_delta), 1e-6);
    let perp_px = vec2<f32>(-dir.y, dir.x) * screen.half_width_px;
    let perp_ndc = perp_px / screen.viewport;

    var out: VertexOutput;
    out.clip_position = self_clip;
    out.clip_position.x += perp_ndc.x * side[vertex_index] * self_clip.w;
    out.clip_position.y += perp_ndc.y * side[vertex_index] * self_clip.w;
    // Small depth bias: pull edges closer in NDC so they draw on top of faces.
    out.clip_position.z -= 0.0001;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0); // black edges
}
