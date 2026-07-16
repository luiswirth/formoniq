// Segment marks: the mesh's 1-skeleton, a line field's traced integral curves,
// and a 1-manifold's own cells -- one shader, because they are one technique
// (the preamble's billboard-quad construction, one instance per segment) at
// different ink and width.
//
// The two endpoints arrive as two separate per-instance vertex buffers
// (`SegmentBatch::layouts` in `item.rs`) bound side by side, since a
// `step_mode: Vertex` buffer only ever exposes the current vertex to the
// shader, never a neighbor -- both ends of the same segment have to be visible
// to one invocation to compute the perpendicular.
//
// The ink is deliberately not the shared colormap. The division of labour is
// that the surface carries the magnitude and the marks carry the geometry, so
// colormapping a ribbon would restate what the fill beneath it already says --
// and restate it invisibly: that colormap is three sinusoids at 120 degree phase
// offsets, whose channels sum to a constant, so it is iso-luminant. A ribbon
// tinted by the same magnitude as the fill under it therefore matches the
// backdrop in hue *and* luminance. Against a backdrop of constant mid luminance,
// a near-black ink is the one choice that separates everywhere.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> material: SegmentMaterial;

struct EndpointA {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) max_displacement: f32,
    @location(3) taper: f32,
    @location(8) value: f32,
};
struct EndpointB {
    @location(4) position: vec3<f32>,
    @location(5) normal: vec3<f32>,
    @location(6) max_displacement: f32,
    @location(7) taper: f32,
    @location(9) value: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) taper: f32,
};

@vertex
fn vs_main(a: EndpointA, b: EndpointB, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // The same standing-wave displacement the fill applies, so the wireframe
    // tracks the displaced surface instead of the flat rest mesh. A mark that
    // does not sit on a displaced surface -- a traced streamline, a curve's own
    // cells -- carries a zero normal, and the displacement is the identity on it.
    let osc = wave_osc(frame, material.wave_omega);
    let world_a = wave_displace(material.wave_amplitude, osc, a.position, a.normal, a.value, a.max_displacement);
    let world_b = wave_displace(material.wave_amplitude, osc, b.position, b.normal, b.value, b.max_displacement);
    let perp = billboard_perp(world_a, world_b, frame.eye.xyz);
    let corner = billboard_corner(world_a, world_b, perp, material.half_width_world, vertex_index);

    var out: VertexOutput;
    out.clip_position = depth_biased(frame.view_proj * vec4<f32>(corner, 1.0));
    out.taper = select(a.taper, b.taper, billboard_is_b(vertex_index));
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Fade toward the standing-wave node, where the field vanishes and the
    // curves are meaningless -- but never fully, since the integral curves of a
    // standing mode are the same set at every phase and blinking them out
    // entirely would read as the geometry changing. A mark with no such story to
    // tell passes `fade_floor = 1`, and the envelope is constant.
    let env = abs(wave_osc(frame, material.wave_omega));
    let alpha = material.color.a * in.taper * mix(material.fade_floor, 1.0, env);
    return vec4<f32>(material.color.rgb, alpha);
}
