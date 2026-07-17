// Segment marks: the mesh's 1-skeleton, a line field's traced integral curves,
// a 1-manifold's own cells, and the arrow glyphs of a line field -- one shader,
// because they are one technique (the preamble's billboard-quad construction,
// one instance per segment) at different ink, width and taper.
//
// The arrow is the taper and nothing else: the quad, the billboard and the
// vertex path are the ribbon's, and the head is a function the fragment cuts the
// quad's silhouette down to. Everything the mark needs is therefore material
// data, which is what keeps a fourth mark from being a fourth pipeline.
//
// The two endpoints arrive as two separate per-instance vertex buffers
// (`SegmentBatch::layouts` in `item.rs`) bound side by side, since a
// `step_mode: Vertex` buffer only ever exposes the current vertex to the
// shader, never a neighbor -- both ends of the same segment have to be visible
// to one invocation to compute the perpendicular.
//
// The ink is deliberately not the shared colormap. The division of labour is
// that the surface carries the magnitude and the marks carry the geometry, so
// colormapping a ribbon would restate what the fill beneath it already says.
// Neither colormap this preamble offers is iso-luminant (viridis runs dark to
// bright, the diverging map runs dark-blue to white to dark-red), so no single
// fixed ink separates from every sample of either -- a near-black ribbon still
// vanishes into viridis's low end, a white one into the diverging map's
// midpoint. `display.rs` draws each ribbon as two `Segments` items sharing one
// batch: a wider light halo first, a narrower dark core on top -- an outline,
// the standard cartographic answer to a line crossing a field it has no fixed
// contrast with, and correct against either colormap without knowing which.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> material: SegmentMaterial;

struct EndpointA {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) max_displacement: f32,
    @location(3) opacity: f32,
    @location(8) value: f32,
};
struct EndpointB {
    @location(4) position: vec3<f32>,
    @location(5) normal: vec3<f32>,
    @location(6) max_displacement: f32,
    @location(7) opacity: f32,
    @location(9) value: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) opacity: f32,
    // Where the fragment sits in the quad: `along` runs 0 at A to 1 at B, and
    // `across` runs -1 to 1 in units of the material's half-width. The two
    // coordinates the ink profile is a function of.
    @location(1) along: f32,
    @location(2) across: f32,
};

// The mark's half-width at `along`, as a fraction of the material's own: the
// shaft's constant fraction until the head begins, then the head's linear taper
// from its full-width base to a point at the tip.
//
// This is the whole of what makes a segment an arrow, and it is why an arrow is
// not a second pipeline: a plain segment passes `head_length_fraction = 0` and
// `shaft_width_fraction = 1`, and the expression collapses to a constant 1 --
// the full quad, exactly the mark this shader drew before there were arrows.
fn segment_profile(material: SegmentMaterial, along: f32, blend_half: f32) -> f32 {
    let head = max(material.head_length_fraction, 1e-6);
    let head_start = 1.0 - material.head_length_fraction;
    let taper = clamp((1.0 - along) / head, 0.0, 1.0);
    // Blended, not switched, across `head_start`: the taper reaches 1 there
    // (the head's full-width base) while the shaft sits at
    // `shaft_width_fraction` on the other side, so a hard `select` is a genuine
    // discontinuity in the profile. `fwidth` below differentiates across it,
    // and one row of fragments would see the whole shaft-to-head jump as its
    // screen-space derivative -- the antialiasing edge computed from that
    // blows open into a visible seam exactly at the join.
    //
    // `blend_half` is `fwidth(along)`, the caller's one-pixel footprint in
    // `along`, not a fraction of the head's length: sizing it from the
    // geometry would round the corner into a visible curve at any zoom where
    // that fraction covers more than a pixel. Tied to the pixel instead, the
    // blend is exactly the antialiasing this shader already does at every
    // other edge -- a corner smoothed over the one pixel it occupies, never
    // wider, so it reads as sharp at any zoom.
    let blend = smoothstep(head_start - blend_half, head_start + blend_half, along);
    return mix(material.shaft_width_fraction, taper, blend);
}

@vertex
fn vs_main(a: EndpointA, b: EndpointB, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // The same standing-wave displacement the fill applies, so the wireframe
    // tracks the displaced surface instead of the flat rest mesh. A mark that
    // does not sit on a displaced surface -- a glyph ribbon, a curve's own
    // cells -- carries a zero normal, and the displacement is the identity on it.
    let osc = wave_osc(frame, material.wave_omega);
    let world_a = wave_displace(material.wave_amplitude, osc, a.position, a.normal, a.value, a.max_displacement);
    let world_b = wave_displace(material.wave_amplitude, osc, b.position, b.normal, b.value, b.max_displacement);
    let perp = billboard_perp(world_a, world_b, frame.eye.xyz);
    let corner = billboard_corner(world_a, world_b, perp, material.half_width_world, vertex_index);
    let biased_corner = depth_biased_corner(corner, frame.eye.xyz, material.half_width_world);

    var out: VertexOutput;
    out.clip_position = frame.view_proj * vec4<f32>(biased_corner, 1.0);
    out.opacity = select(a.opacity, b.opacity, billboard_is_b(vertex_index));
    out.along = select(0.0, 1.0, billboard_is_b(vertex_index));
    out.across = billboard_side(vertex_index);
    return out;
}

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

    // The quad is cut down to the mark's own silhouette, one screen pixel of
    // gradient wide, so the edge is resolved rather than aliased. `fwidth` is
    // the profile's rate of change in screen space, so the softening is a pixel
    // whatever the mark's world width and however the quad is foreshortened --
    // and on a plain segment, whose profile is a constant 1, it antialiases the
    // straight edge the quad already had.
    let profile = segment_profile(material, in.along, fwidth(in.along));
    let edge = fwidth(in.across) + fwidth(profile);
    let dist = abs(in.across) - profile;
    let ink = 1.0 - smoothstep(-edge, edge, dist);

    // The outline is the same silhouette again, `outline_width_fraction`
    // further out: a second boundary at a fixed world-space remove from the
    // first; `ink` and `rim` are the fraction covered inside each, together
    // in the same units `dist` already is. Zero draws no rim -- `rim` and
    // `ink` coincide, and the ink-only path below is exact, not a case split.
    let rim = 1.0 - smoothstep(-edge, edge, dist - material.outline_width_fraction);

    // Standard alpha compositing of the ink (the material's own color) over
    // the rim (opaque black), both already faded by the envelope and the
    // mark's own opacity -- so the rim fades out with the ink it outlines
    // rather than staying a solid black skeleton once the ink above it is
    // gone.
    let envelope = in.opacity * mix(material.fade_floor, 1.0, env);
    let ink_a = material.color.a * envelope * ink;
    let rim_a = envelope * max(rim - ink, 0.0);
    let alpha = ink_a + rim_a * (1.0 - ink_a);
    let rgb = material.color.rgb * (ink_a / max(alpha, 1e-5));

    var out: FsOut;
    out.color = vec4<f32>(rgb, alpha);
    out.unbounded = 0.0;
    return out;
}
