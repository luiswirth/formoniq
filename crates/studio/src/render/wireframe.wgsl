// Wireframe shader — draws mesh edges as lines of constant *world-space*
// thickness (not constant screen pixels, so the line neither vanishes when
// zoomed out nor swamps the mesh when zoomed in).
//
// A `LineList` primitive rasterizes at a fixed, backend-defined 1px in wgpu
// (there is no portable line-width control), which all but disappears once
// the supersampled scene pass is box-filtered down to the swapchain. Instead
// each edge is drawn as an instanced quad: two triangles fanned from its two
// endpoints, offset in world space perpendicular to both the edge and the
// view direction (so the quad stays roughly screen-facing, the way a
// billboard does) by a fixed world-space half-width. The two endpoints
// arrive as two separate per-instance vertex buffers (see
// `Vertex::desc_endpoint_a`/`desc_endpoint_b` in `mesh.rs`) bound side by
// side, since a `step_mode: Vertex` buffer only ever exposes the current
// vertex to the shader, never a neighbor -- both ends of the same edge have
// to be visible to one invocation to compute the perpendicular.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
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

// The line's world-space half-width: a fixed fraction of the scene's own
// extent (set on the Rust side), not a pixel count, so it reads the same
// whether the mesh is zoomed to fill the screen or shrunk to a corner of it.
struct Wireframe {
    half_width_world: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
@group(2) @binding(0)
var<uniform> wireframe: Wireframe;

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

// The same normal-displaced world position the fill/scalar shader computes
// for a single vertex (before that shader's own projection), reused here per
// endpoint.
fn displaced_world(position: vec3<f32>, normal: vec3<f32>, value: f32, max_displacement: f32) -> vec3<f32> {
    let osc = cos(wave.omega * wave.time);
    let raw = wave.amplitude * osc * value;
    let capped = clamp(raw, -max_displacement, max_displacement);
    return position + capped * normal;
}

@vertex
fn vs_main(a: EndpointA, b: EndpointB, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let world_a = displaced_world(a.position, a.normal, a.value, a.max_displacement);
    let world_b = displaced_world(b.position, b.normal, b.value, b.max_displacement);

    // One perpendicular per *edge*, not per endpoint: computed once from the
    // edge midpoint's view direction and shared by both of the edge's
    // corners, so the quad is a parallelogram of constant width. Deriving it
    // separately at each endpoint's own (slightly different) view direction
    // instead tapers the quad into a wedge whenever the edge is long enough,
    // relative to the camera distance, for the two ends to look back at the
    // camera along visibly different directions -- exactly the coarse
    // triforce mesh's few, large triangles.
    let edge_vec = world_b - world_a;
    let edge_dir = edge_vec / max(length(edge_vec), 1e-6);
    let mid = 0.5 * (world_a + world_b);
    let view_vec = camera.eye.xyz - mid;
    let view_dir = view_vec / max(length(view_vec), 1e-6);
    var perp = cross(edge_dir, view_dir);
    let perp_len = length(perp);
    // Edge pointing straight at the camera: no well-defined screen-facing
    // perpendicular, but the quad also projects to ~zero screen length there,
    // so any fallback direction is invisible anyway.
    perp = select(perp / max(perp_len, 1e-6), vec3<f32>(1.0, 0.0, 0.0), perp_len < 1e-6);

    // The quad's two triangles (A-, B-, B+) and (A-, B+, A+): six corners,
    // each a (which endpoint is "self", which side of the edge) pair.
    var self_is_b = array<bool, 6>(false, true, true, false, true, false);
    var side = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let self_world = select(world_a, world_b, self_is_b[vertex_index]);
    let offset_world = self_world + perp * wireframe.half_width_world * side[vertex_index];

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(offset_world, 1.0);
    // Small depth bias, scaled by `w` so it survives the perspective divide
    // as a constant NDC offset regardless of depth: pulls edges toward the
    // camera so they draw on top of the coplanar face beneath them instead of
    // z-fighting it.
    out.clip_position.z -= 0.0005 * out.clip_position.w;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0); // black edges
}
