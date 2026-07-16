// Streamline ribbons: the traced integral curves of a grade-1 line field, drawn
// with the preamble's billboard-quad construction -- the same technique the
// wireframe uses, one instance per traced segment. The samples are already on
// the (undisplaced) surface, so there is no wave displacement here.
//
// What differs from the wireframe is the fragment: the ribbon is dark ink,
// faded toward the standing-wave node by the same $cos(sqrt(lambda) t)$
// envelope the surface uses, and tapered to nothing at each curve's ends so a
// streamline fades in rather than starting with a hard cap.
//
// The ink is deliberately not the shared colormap. The division of labour is
// that the surface carries the magnitude and the curves carry the direction, so
// colormapping a ribbon would restate what the fill beneath it already says --
// and restate it invisibly: that colormap is three sinusoids at 120 degree phase
// offsets, whose channels sum to a constant, so it is iso-luminant. A ribbon
// tinted by the same magnitude as the fill under it therefore matches the
// backdrop in hue *and* luminance. Against a backdrop of constant mid luminance,
// a near-black ink is the one choice that separates everywhere.

@group(0) @binding(0) var<uniform> camera: CameraUniform;
// Shares the wave clock with the surface: `omega`/`time` fade the ribbons at the
// node; `amplitude` is unused (the samples carry no displacement).
@group(1) @binding(0) var<uniform> wave: Wave;
@group(2) @binding(0) var<uniform> width: SegmentWidth;

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
    let perp = billboard_perp(a.position, b.position, camera.eye.xyz);
    let corner = billboard_corner(a.position, b.position, perp, width.half_width_world, vertex_index);

    var out: VertexOutput;
    out.clip_position = depth_biased(camera.view_proj * vec4<f32>(corner, 1.0));
    out.taper = select(a.taper, b.taper, billboard_is_b(vertex_index));
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Fade toward the standing-wave node, where the field vanishes and the
    // curves are meaningless -- but never fully, since the integral curves of a
    // standing mode are the same set at every phase and blinking them out
    // entirely would read as the geometry changing.
    let env = abs(wave_osc(wave));
    let alpha = in.taper * mix(0.25, 1.0, env);
    return vec4<f32>(INK, alpha);
}
