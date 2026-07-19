// The arrow glyphs of a line field: the field read pointwise on each cell's
// barycentric lattice, drawn as a flat mark lying in the surface cell. Unlike a
// segment (`segments.wgsl`) an arrow has a plane -- its cell's -- so it is not
// billboarded; its geometry is a quad in that plane, baked final, and this
// shader only carves the arrow out of it and clips it to the cell.
//
// The mark is the region inside a signed distance to the arrow polygon -- a
// shaft rectangle fused to a head triangle -- so the outline is that same
// distance a fixed world band further out, uniform on every edge, both barbs and
// the tip alike. Because the quad lies in the cell, each corner's barycentric
// coordinate is known at bake time and interpolated here for free, so the clip
// that keeps the arrow inside the cell it was sampled in is a single
// `min(weight) < 0`.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> material: GlyphMaterial;

struct GlyphVertex {
    @location(0) position: vec3<f32>,
    @location(1) arrow_xy: vec2<f32>,
    @location(2) length: f32,
    @location(3) opacity: f32,
    @location(4) cell_bary: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) opacity: f32,
    // The fragment in the arrow's own world frame: `xy.x` runs along the field
    // from the tail at 0 to the tip at `length`, `xy.y` across it, both world
    // units so the signed distance below is Euclidean.
    @location(1) xy: vec2<f32>,
    @location(2) length: f32,
    // The fragment's barycentric coordinate within its cell, padded to four with
    // ones. Interpolated exactly, since the quad lies in the cell's plane and
    // barycentric coordinates are affine in position.
    @location(3) cell_bary: vec4<f32>,
};

// The signed distance from `p` to the arrow polygon (iq's `sdPolygon`), negative
// inside. Seven vertices: the shaft rectangle (half-width `shaft_hw`, from the
// tail at x = 0 to the head base at `x_head`) fused to the head triangle (full
// half-width `hw` at the base, a point at the tip `x = length`). The
// `max(dot(e,e), ...)` guards the zero-length edges a degenerate head would
// leave behind.
fn sd_arrow(p: vec2<f32>, length: f32, hw: f32, head_len: f32, shaft_frac: f32) -> f32 {
    let x_head = length - head_len;
    let shaft_hw = shaft_frac * hw;
    var v = array<vec2<f32>, 7>(
        vec2<f32>(0.0, -shaft_hw),
        vec2<f32>(x_head, -shaft_hw),
        vec2<f32>(x_head, -hw),
        vec2<f32>(length, 0.0),
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
fn vs_main(vertex: GlyphVertex) -> VertexOutput {
    // The arrow lies flat in the cell, coplanar with the fill, so it is nudged
    // toward the camera to draw over the surface instead of z-fighting it -- the
    // same bias the wireframe takes, tied to the mark's own world scale.
    let biased = depth_biased_corner(vertex.position, frame.view_dir.xyz, material.width_fraction * vertex.length);

    var out: VertexOutput;
    out.clip_position = frame.view_proj * vec4<f32>(biased, 1.0);
    out.opacity = vertex.opacity;
    out.xy = vertex.arrow_xy;
    out.length = vertex.length;
    out.cell_bary = vertex.cell_bary;
    return out;
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) unbounded: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> FsOut {
    // Fade toward the standing-wave node, where the field vanishes and an arrow
    // is meaningless -- but never fully, since the lattice of a standing mode is
    // the same set at every phase. A static field passes `fade_floor = 1`.
    let env = abs(wave_osc(frame, material.wave_omega));

    // The quad is cut down to the arrow's silhouette, one screen pixel of
    // gradient wide, so the edge is resolved rather than aliased.
    let head_len = material.head_length_fraction * in.length;
    let half_width = material.width_fraction * in.length;
    let dist = sd_arrow(in.xy, in.length, half_width, head_len, material.shaft_width_fraction);
    let edge = max(fwidth(dist), 1e-6);
    let ink = 1.0 - smoothstep(-edge, edge, dist);

    // The outline is the same silhouette a fixed world width further out. Zero
    // draws no rim -- `rim` and `ink` coincide, and the ink-only path below is
    // exact, not a case split.
    let outline_world = material.outline_width_fraction * half_width;
    let rim = 1.0 - smoothstep(-edge, edge, dist - outline_world);

    // Clip the arrow to its cell: discard where any interpolated barycentric
    // weight is negative, i.e. past the facet where the section the arrow reports
    // has no value. Softened over its own one-pixel footprint so the cut is
    // antialiased like every other edge.
    let min_bary = min(min(in.cell_bary.x, in.cell_bary.y), min(in.cell_bary.z, in.cell_bary.w));
    let clip_edge = max(fwidth(min_bary), 1e-6);
    let clip = smoothstep(-clip_edge, clip_edge, min_bary);

    // Standard alpha compositing of the ink (the material's own color) over the
    // rim (opaque black), both faded by the envelope, the mark's opacity and the
    // clip -- so the rim fades out with the ink it outlines rather than staying a
    // black skeleton once the ink above it is gone.
    let envelope = in.opacity * clip * mix(material.fade_floor, 1.0, env);
    let ink_a = material.color.a * envelope * ink;
    let rim_a = envelope * max(rim - ink, 0.0);
    let alpha = ink_a + rim_a * (1.0 - ink_a);
    let rgb = material.color.rgb * (ink_a / max(alpha, 1e-5));

    var out: FsOut;
    out.color = vec4<f32>(rgb, alpha);
    out.unbounded = 0.0;
    return out;
}
