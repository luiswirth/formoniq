// Segment marks: the mesh's 1-skeleton, a 1-manifold's own cells, and a line
// field's traced curves -- one shader, because they are one technique (the
// preamble's billboard-quad construction, one instance per segment) at different
// ink and width. A billboard because a 1-dimensional mark has no plane to lie in
// -- an edge is shared by two faces, a space curve belongs to none -- so the
// quad is turned to face the camera to stay a visible, constant-width line from
// any angle. The arrow glyph, which does have a plane (its surface cell), is not
// here: it is a flat mark, in `glyph.wgsl`.
//
// The two endpoints arrive as two separate per-instance vertex buffers
// (`SegmentBatch::layouts` in `item.rs`) bound side by side, since a
// `step_mode: Vertex` buffer only ever exposes the current vertex to the
// shader, never a neighbor -- both ends of the same segment have to be visible
// to one invocation to compute the perpendicular.
//
// A segment mark has two inks, chosen by `material.colored`. Off, it is the
// fixed geometry color: the structural skeleton (the old wireframe), and the
// halo/core outline `display.rs` draws for a ribbon that must separate from a
// field it crosses (a wider light segment, a narrower dark one on top -- the
// standard cartographic outline). On, it is the shared colormap of the field
// traced onto the mark's own simplices (`colormap_sample`), the same palette
// the fill reads: a colored k-skeleton is one mark at a different ink, not a
// second pass. Visibility and coloring are independent -- a hidden skeleton is
// simply dropped from the draw list.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> material: SegmentMaterial;

struct EndpointA {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) max_displacement: f32,
    @location(3) opacity: f32,
    // The displacement height (the shared nodal recovery: continuous, per
    // vertex), and the colormap value (the trace on this mark's own simplex:
    // per edge). Height rides the wave; color reads the field.
    @location(8) height: f32,
    @location(10) color_value: f32,
};
struct EndpointB {
    @location(4) position: vec3<f32>,
    @location(5) normal: vec3<f32>,
    @location(6) max_displacement: f32,
    @location(7) opacity: f32,
    @location(9) height: f32,
    @location(11) color_value: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) opacity: f32,
    // Where the fragment sits across the ribbon: -1 at one edge to 1 at the
    // other, in units of the material's half-width. The one coordinate the ink
    // profile is a function of.
    @location(1) across: f32,
    // The field value interpolated along the segment, for the colormap ink.
    @location(2) color_value: f32,
};

@vertex
fn vs_main(a: EndpointA, b: EndpointB, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // The same standing-wave displacement the fill applies, so the wireframe
    // tracks the displaced surface instead of the flat rest mesh. A mark that
    // does not sit on a displaced surface -- a curve's own cells -- carries a
    // zero normal, and the displacement is the identity on it.
    let osc = wave_osc(frame, material.wave_omega);
    let world_a = wave_displace(material.wave_amplitude, osc, a.position, a.normal, a.height, a.max_displacement);
    let world_b = wave_displace(material.wave_amplitude, osc, b.position, b.normal, b.height, b.max_displacement);
    let perp = billboard_perp(world_a, world_b, frame.view_dir.xyz);
    let corner = billboard_corner(world_a, world_b, perp, material.half_width_world, vertex_index);
    let biased_corner = depth_biased_corner(corner, frame.view_dir.xyz, material.half_width_world);

    var out: VertexOutput;
    out.clip_position = frame.view_proj * vec4<f32>(biased_corner, 1.0);
    out.opacity = select(a.opacity, b.opacity, billboard_is_b(vertex_index));
    out.across = billboard_side(vertex_index);
    // The quad's two ends carry A's and B's values, so the rasterizer
    // interpolates the field linearly along the segment.
    out.color_value = select(a.color_value, b.color_value, billboard_is_b(vertex_index));
    return out;
}

// A mark occludes the medium exactly to the extent it is opaque, and a depth
// buffer can only say "fully" or "not at all". Two separate conditions decide
// it, and conflating them costs either the silhouette or the occlusion:
//
// `SILHOUETTE_EPS` cuts the quad's *shape*. A billboard is far wider than the
// mark drawn inside it, and that transparent margin must stay out of the depth
// buffer or it carves a rectangular hole in the medium around every ribbon. The
// threshold is near zero on purpose: the one-pixel `fwidth` gradient that
// resolves the edge is kept intact, so the depth written overshoots the visible
// mark by well under a pixel and the antialiasing is untouched.
//
// `OPAQUE_ENOUGH` asks whether the mark is see-through at all -- a ribbon faded
// toward a standing-wave node, a tracer tapered to its tail. Those should not
// occlude, because one can see through them, and the fog behind correctly shows
// through too.
const SILHOUETTE_EPS: f32 = 0.02;
const OPAQUE_ENOUGH: f32 = 0.5;

struct FsOut {
    @location(0) color: vec4<f32>,
    // A ribbon's ink is a fixed material color, alpha-blended -- never above 1,
    // so it never asks the tone curve for anything. See `display_transform`'s
    // note on `unbounded_mask`.
    @location(1) unbounded: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> FsOut {
    // Fade toward the standing-wave node, where the field vanishes and the
    // curves are meaningless -- but never fully, since the integral curves of a
    // standing mode are the same set at every phase and blinking them out
    // entirely would read as the geometry changing. A mark with no such story to
    // tell passes `fade_floor = 1`, and the envelope is constant.
    let env = abs(wave_osc(frame, material.wave_omega));

    // The quad is cut down to the ribbon's silhouette, one screen pixel of
    // gradient wide, so the edge is resolved rather than aliased. `fwidth` is
    // `across`'s rate of change in screen space, so the softening is a pixel
    // whatever the mark's world width and however the quad is foreshortened.
    let edge = fwidth(in.across);
    let ink = 1.0 - smoothstep(1.0 - edge, 1.0 + edge, abs(in.across));

    // The ink: the field's colormap where the mark reflects a field, else the
    // fixed geometry color. The colormap pulses with the standing wave in step
    // with the fill, and stays in [0, 1] (never above 1, so the tone curve is
    // untouched -- `unbounded = 0`), exactly as the geometry ink does.
    let field_rgb = colormap_sample(
        material.min_val, material.max_val, material.diverging,
        in.color_value * wave_osc(frame, material.wave_omega),
    );
    let rgb = select(material.color.rgb, field_rgb, material.colored > 0.5);

    let envelope = in.opacity * mix(material.fade_floor, 1.0, env);
    var out: FsOut;
    if (ink < SILHOUETTE_EPS || material.color.a * envelope < OPAQUE_ENOUGH) {
        discard;
    }
    out.color = vec4<f32>(rgb, material.color.a * envelope * ink);
    out.unbounded = 0.0;
    return out;
}
