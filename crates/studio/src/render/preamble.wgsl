// Shared WGSL preamble, prepended to every shader body in this module (see
// `render::shader_source`). It declares only types, pipeline-overridable
// constants and pure functions -- never a binding, since the group a uniform
// sits in is a property of the pipeline that uses it, not of the value.
//
// Each struct here is the WGSL side of a `#[repr(C)]` Rust uniform of the same
// name in `uniform.rs`; the two must stay byte-identical.

// The supersampling factor per axis every scene pass renders at, set from Rust
// (`render::SSAA_SCALE`) as a pipeline constant rather than duplicated as a
// literal: the downsample's box filter has to be exactly the factor the targets
// were allocated at, and an `override` is what makes the two one number.
override SSAA_SCALE: i32 = 2;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    // World-space eye position, `w` unused: the billboard construction below
    // needs it directly, not only the matrix it feeds into.
    eye: vec4<f32>,
};

// Standing-wave animation of an eigenmode: $u(t) = a cos(omega t)$. A surface
// vertex is additively displaced along its own outward normal times its scalar
// value (the classical membrane picture). A field with no eigenvalue has
// `omega = 0` and `amplitude = 0`, so the same code leaves it static.
struct Wave {
    time: f32,
    amplitude: f32,
    omega: f32,
    _pad: f32,
};

// Colormap normalization range for the field on display.
struct Bounds {
    min_val: f32,
    max_val: f32,
    _pad1: f32,
    _pad2: f32,
};

// A segment pass's world-space half-width: a fixed fraction of the scene's own
// extent (set on the Rust side), not a pixel count, so a line reads the same
// thickness whether the mesh fills the screen or sits in a corner of it.
// Padded with scalars, not a `vec3<f32>`: a vec3 carries 16-byte alignment,
// which would push the struct to 32 bytes against the uniform's 16.
struct SegmentWidth {
    half_width_world: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

// The standing wave's instantaneous phase factor $cos(omega t)$: the one
// oscillator the displacement, the colormap pulse and the streamline fade all
// read, so they cannot drift out of phase.
fn wave_osc(wave: Wave) -> f32 {
    return cos(wave.omega * wave.time);
}

// The wave-displaced world position of one vertex, clamped to that vertex's own
// curvature cap so a thin or sharply curved feature never folds through its
// focal point.
fn wave_displace(wave: Wave, position: vec3<f32>, normal: vec3<f32>, value: f32, max_displacement: f32) -> vec3<f32> {
    let raw = wave.amplitude * wave_osc(wave) * value;
    let capped = clamp(raw, -max_displacement, max_displacement);
    return position + capped * normal;
}

fn colormap(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let r = 0.5 + 0.5 * sin(x * 6.28 - 1.57);
    let g = 0.5 + 0.5 * sin(x * 6.28 - 1.57 + 2.09);
    let b = 0.5 + 0.5 * sin(x * 6.28 - 1.57 - 2.09);
    return vec3<f32>(r, g, b);
}

fn colormap_in(bounds: Bounds, value: f32) -> vec3<f32> {
    return colormap((value - bounds.min_val) / (bounds.max_val - bounds.min_val));
}

// A segment (a mesh edge, a traced streamline step) drawn with constant
// world-space thickness. A `LineList` primitive rasterizes at a fixed,
// backend-defined 1px in wgpu -- there is no portable line-width control -- and
// all but disappears once the supersampled scene pass is filtered down. Instead
// each segment is an instanced quad: two triangles fanned from its two
// endpoints, offset perpendicular to both the segment and the view direction,
// so the quad stays screen-facing the way a billboard does.
//
// One perpendicular per *segment*, from the midpoint's view direction, shared
// by both corners so the quad is a parallelogram of constant width. Deriving it
// separately at each endpoint's own (slightly different) view direction instead
// tapers the quad into a wedge whenever the segment is long enough, relative to
// the camera distance, for the two ends to look back at the camera along
// visibly different directions -- exactly the coarse triforce mesh's few, large
// triangles.
fn billboard_perp(world_a: vec3<f32>, world_b: vec3<f32>, eye: vec3<f32>) -> vec3<f32> {
    let seg_vec = world_b - world_a;
    let seg_dir = seg_vec / max(length(seg_vec), 1e-6);
    let mid = 0.5 * (world_a + world_b);
    let view_vec = eye - mid;
    let view_dir = view_vec / max(length(view_vec), 1e-6);
    let perp = cross(seg_dir, view_dir);
    let perp_len = length(perp);
    // Segment pointing straight at the camera: no well-defined screen-facing
    // perpendicular, but the quad also projects to ~zero screen length there,
    // so any fallback direction is invisible anyway.
    return select(perp / max(perp_len, 1e-6), vec3<f32>(1.0, 0.0, 0.0), perp_len < 1e-6);
}

// The quad's two triangles (A-, B-, B+) and (A-, B+, A+) make six corners, each
// a (which endpoint is "self", which side of the segment) pair; these two
// functions are the halves of that pair, indexed by a `0..6` non-indexed draw.
fn billboard_is_b(vertex_index: u32) -> bool {
    var table = array<bool, 6>(false, true, true, false, true, false);
    return table[vertex_index];
}

fn billboard_side(vertex_index: u32) -> f32 {
    var table = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    return table[vertex_index];
}

fn billboard_corner(world_a: vec3<f32>, world_b: vec3<f32>, perp: vec3<f32>, half_width: f32, vertex_index: u32) -> vec3<f32> {
    let self_world = select(world_a, world_b, billboard_is_b(vertex_index));
    return self_world + perp * half_width * billboard_side(vertex_index);
}

// Small depth bias, scaled by `w` so it survives the perspective divide as a
// constant NDC offset regardless of depth: pulls a segment toward the camera so
// it draws on top of the coplanar face beneath it instead of z-fighting it.
fn depth_biased(clip: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(clip.xy, clip.z - 0.0005 * clip.w, clip.w);
}
