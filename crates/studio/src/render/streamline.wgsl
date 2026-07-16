// Streamline ribbon shader — draws the traced integral curves of a grade-1 line
// field as camera-facing ribbons of constant world-space thickness.
//
// The geometry is the wireframe's billboard quad (see `wireframe.wgsl`): each
// streamline segment (a consecutive pair of traced samples) is an instance,
// expanded into two triangles offset perpendicular to both the segment and the
// view direction, the perpendicular computed once per segment from its midpoint.
// What differs is the fragment: the ribbon is drawn as dark ink, faded toward
// the standing-wave node by the same $cos(sqrt(lambda) t)$ envelope the surface
// and the LIC use, and tapered to nothing at each curve's ends so a streamline
// fades in rather than starting with a hard cap. The samples are already on the
// (undisplaced) surface, so there is no wave displacement here.
//
// The ink is deliberately not the shared colormap. The division of labour is
// that the surface carries the magnitude and the curves carry the direction, so
// colormapping a ribbon would restate what the fill beneath it already says --
// and restate it invisibly: that colormap is three sinusoids at 120 degree phase
// offsets, whose channels sum to a constant, so it is iso-luminant. A ribbon
// tinted by the same magnitude as the fill under it therefore matches the
// backdrop in hue *and* luminance. Against a backdrop of constant mid luminance,
// a near-black ink is the one choice that separates everywhere.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Shares the wave clock with the surface: `omega`/`time` fade the ribbons at the
// node; `amplitude` is unused (the samples carry no displacement).
struct Wave {
    time: f32,
    amplitude: f32,
    omega: f32,
    _pad: f32,
};
@group(1) @binding(0)
var<uniform> wave: Wave;

// The ribbon's world-space half-width, a fraction of the scene extent, like the
// wireframe.
// Padded with scalars, not a `vec3<f32>`: a vec3 carries 16-byte alignment, which
// would push the struct to 32 bytes against the uniform's 16.
struct Streamline {
    half_width_world: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
@group(2) @binding(0)
var<uniform> stream: Streamline;

struct EndpointA {
    @location(0) position: vec3<f32>,
    @location(1) taper: f32,
};
struct EndpointB {
    @location(2) position: vec3<f32>,
    @location(3) taper: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) taper: f32,
};

// The ink: dark enough to separate from any colormap sample, not pure black, so
// the curves read as drawn on the surface rather than as holes cut through it.
const INK = vec3<f32>(0.05, 0.05, 0.07);

@vertex
fn vs_main(a: EndpointA, b: EndpointB, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let world_a = a.position;
    let world_b = b.position;

    // One perpendicular per segment, from the midpoint's view direction, shared
    // by both corners so the quad is a parallelogram of constant width -- the
    // same construction (and the same reasoning) as the wireframe's.
    let seg_vec = world_b - world_a;
    let seg_dir = seg_vec / max(length(seg_vec), 1e-6);
    let mid = 0.5 * (world_a + world_b);
    let view_vec = camera.eye.xyz - mid;
    let view_dir = view_vec / max(length(view_vec), 1e-6);
    var perp = cross(seg_dir, view_dir);
    let perp_len = length(perp);
    perp = select(perp / max(perp_len, 1e-6), vec3<f32>(1.0, 0.0, 0.0), perp_len < 1e-6);

    var self_is_b = array<bool, 6>(false, true, true, false, true, false);
    var side = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let self_world = select(world_a, world_b, self_is_b[vertex_index]);
    let offset_world = self_world + perp * stream.half_width_world * side[vertex_index];

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(offset_world, 1.0);
    // Same depth bias as the wireframe: pull the ribbon toward the camera so it
    // draws over the coplanar surface beneath it instead of z-fighting.
    out.clip_position.z -= 0.0005 * out.clip_position.w;
    out.taper = select(a.taper, b.taper, self_is_b[vertex_index]);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Fade toward the standing-wave node, where the field vanishes and the
    // curves are meaningless -- but never fully, since the integral curves of a
    // standing mode are the same set at every phase and blinking them out
    // entirely would read as the geometry changing.
    let env = abs(cos(wave.omega * wave.time));
    let alpha = in.taper * mix(0.25, 1.0, env);
    return vec4<f32>(INK, alpha);
}
