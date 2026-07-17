// Segment marks: the mesh's 1-skeleton, a line field's traced integral curves,
// a 1-manifold's own cells, and the arrow glyphs of a line field -- one shader,
// because they are one technique (the preamble's billboard-quad construction,
// one instance per segment) at different ink, width and taper.
//
// The arrow is a silhouette the fragment cuts the quad down to, and nothing
// else: the quad, the billboard and the vertex path are the ribbon's. The mark
// is the region inside a signed distance to the arrow polygon -- a shaft
// rectangle fused to a head triangle -- so the outline is that same distance a
// fixed world-space band further out, uniform on every edge, both barbs and the
// tip alike. A plain segment is the same construction with a zero-length head:
// the polygon degenerates to the full rectangle, exactly the mark this shader
// drew before there were arrows, which is what keeps a fourth mark from being a
// fourth pipeline.
//
// The quad is inflated outward by that band so the rim has room on every side --
// past the tip, behind the tail, beyond the widest barb. A mark with no outline
// (a mesh edge, a ribbon) inflates by zero and is untouched.
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
    // The fragment's position in the arrow's own world frame: `xy.x` runs along
    // the segment from the tail at 0 to the tip at `seg_len`, `xy.y` across it,
    // both in world units so the signed distance below is Euclidean and its
    // outline band is one constant world width on every edge.
    @location(1) xy: vec2<f32>,
    @location(2) seg_len: f32,
};

// The signed distance from `p` to the arrow polygon (iq's `sdPolygon`),
// negative inside. Seven vertices: the shaft rectangle
// (half-width `shaft_hw`, from the tail at x = 0 to the head base at `x_head`)
// fused to the head triangle (full half-width `hw` at the base, a point at the
// tip `x = seg_len`). A zero-length head collapses `x_head` onto the tip and the
// polygon degenerates to the plain rectangle; the `max(dot(e,e), ...)` guards
// the zero-length edges that leaves behind.
fn sd_arrow(p: vec2<f32>, seg_len: f32, hw: f32, head_len: f32, shaft_frac: f32) -> f32 {
    let x_head = seg_len - head_len;
    let shaft_hw = shaft_frac * hw;
    var v = array<vec2<f32>, 7>(
        vec2<f32>(0.0, -shaft_hw),
        vec2<f32>(x_head, -shaft_hw),
        vec2<f32>(x_head, -hw),
        vec2<f32>(seg_len, 0.0),
        vec2<f32>(x_head, hw),
        vec2<f32>(x_head, shaft_hw),
        vec2<f32>(0.0, shaft_hw),
    );
    var d = dot(p - v[0], p - v[0]);
    var s = 1.0;
    for (var i = 0u; i < 7u; i = i + 1u) {
        let j = select(i - 1u, 6u, i == 0u);
        let e = v[j] - v[i];
        let w = p - v[i];
        let proj = w - e * clamp(dot(w, e) / max(dot(e, e), 1e-12), 0.0, 1.0);
        d = min(d, dot(proj, proj));
        let c = vec3<bool>((p.y >= v[i].y), (p.y < v[j].y), (e.x * w.y > e.y * w.x));
        if (all(c) || all(!c)) {
            s = -s;
        }
    }
    return s * sqrt(d);
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

    let seg_vec = world_b - world_a;
    let seg_len = max(length(seg_vec), 1e-6);
    let seg_dir = seg_vec / seg_len;

    // The quad is grown outward by the rim's own width (with slack for its
    // antialiased outer edge) so the outline has room on every side -- past the
    // tip, behind the tail, beyond the widest barb. A mark with no outline
    // inflates by zero and keeps the exact quad it had.
    let outline_world = material.outline_width_fraction * material.half_width_world;
    let margin = outline_world * 1.5;
    let a_ext = world_a - seg_dir * margin;
    let b_ext = world_b + seg_dir * margin;
    let half_w = material.half_width_world + margin;

    let perp = billboard_perp(a_ext, b_ext, frame.eye.xyz);
    let corner = billboard_corner(a_ext, b_ext, perp, half_w, vertex_index);
    let biased_corner = depth_biased_corner(corner, frame.eye.xyz, half_w);

    var out: VertexOutput;
    out.clip_position = frame.view_proj * vec4<f32>(biased_corner, 1.0);
    out.opacity = select(a.opacity, b.opacity, billboard_is_b(vertex_index));
    // The corner's coordinates in the arrow frame, exact under linear
    // interpolation: the tail edge sits at x = 0 and the tip at x = seg_len,
    // with the inflated ends reaching to -margin and seg_len + margin.
    let x = select(-margin, seg_len + margin, billboard_is_b(vertex_index));
    out.xy = vec2<f32>(x, billboard_side(vertex_index) * half_w);
    out.seg_len = seg_len;
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

    // The quad is cut down to the arrow's own silhouette, one screen pixel of
    // gradient wide, so the edge is resolved rather than aliased. `fwidth` is
    // the distance's rate of change in screen space, so the softening is a pixel
    // whatever the mark's world width and however the quad is foreshortened --
    // and on a plain segment, whose polygon is the full rectangle, it
    // antialiases the straight edge the quad already had.
    let head_len = material.head_length_fraction * in.seg_len;
    let dist = sd_arrow(in.xy, in.seg_len, material.half_width_world, head_len, material.shaft_width_fraction);
    let edge = max(fwidth(dist), 1e-6);
    let ink = 1.0 - smoothstep(-edge, edge, dist);

    // The outline is the same silhouette again, a fixed world width further out:
    // a second boundary at `outline_world` from the first; `ink` and `rim` are
    // the fraction covered inside each, in the same world units `dist` already
    // is. Zero draws no rim -- `rim` and `ink` coincide, and the ink-only path
    // below is exact, not a case split.
    let outline_world = material.outline_width_fraction * material.half_width_world;
    let rim = 1.0 - smoothstep(-edge, edge, dist - outline_world);

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
