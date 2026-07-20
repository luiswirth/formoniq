// The point pass: the 0-skeleton, one instanced billboard *circle* per mesh
// vertex. The 0-dimensional sibling of the segment pass -- the same view-plane
// billboard, the same two inks (the structural geometry color, or the field's
// colormap where `colored` is on), a disc instead of a ribbon. A vertex has no
// plane and no length, so it is a screen-facing circle of constant world radius,
// the honest picture of a point. It reuses `SegmentMaterial`: a point mark and a
// segment mark are the same material at a different primitive.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> material: SegmentMaterial;

struct Point {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) max_displacement: f32,
    @location(3) opacity: f32,
    @location(8) height: f32,
    @location(10) color_value: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) opacity: f32,
    // The quad coordinate in [-1, 1]^2; the disc is the fragments with
    // length(uv) <= 1, so the square quad is cut to a circle in the fragment.
    @location(1) uv: vec2<f32>,
    @location(2) color_value: f32,
};

// The six corners of a unit quad (two triangles) as (±1, ±1).
fn quad_corner(vertex_index: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    return corners[vertex_index];
}

@vertex
fn vs_main(p: Point, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // The same standing-wave displacement the fill and the segments apply, so
    // the points track the displaced surface. A point not on a displaced surface
    // carries a zero normal, and the displacement is the identity on it.
    let osc = wave_osc(frame, material.wave_omega);
    let world = wave_displace(material.wave_amplitude, osc, p.position, p.normal, p.height, p.max_displacement);

    // A billboard frame in the view plane: two axes perpendicular to the
    // camera's forward axis, aligned to the image plane the way the segment's
    // perpendicular is, so the disc reads as a constant-size circle under both
    // perspective and orthographic projection. The reference up flips near the
    // pole, where the forward axis is nearly vertical and the cross would vanish.
    let view_dir = frame.view_dir.xyz;
    let ref_up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(view_dir.y) > 0.99);
    let right = normalize(cross(ref_up, view_dir));
    let up = normalize(cross(view_dir, right));

    let uv = quad_corner(vertex_index);
    let corner = world + (right * uv.x + up * uv.y) * material.half_width_world;
    var out: VertexOutput;
    out.clip_position = frame.view_proj * vec4<f32>(corner, 1.0);
    out.opacity = p.opacity;
    out.uv = uv;
    out.color_value = p.color_value;
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
    @location(1) unbounded: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> FsOut {
    let env = abs(wave_osc(frame, material.wave_omega));

    // The square quad cut to a disc, one screen pixel of gradient at the rim so
    // the circle is resolved rather than aliased -- `fwidth` in the quad's own
    // coordinate, so the softening is a pixel however the disc is sized.
    let r = length(in.uv);
    let edge = fwidth(r);
    let mask = 1.0 - smoothstep(1.0 - edge, 1.0 + edge, r);

    // The ink: the field's colormap where the mark reflects a field, else the
    // fixed geometry color -- the same choice, and the same `colormap_sample`,
    // the segment mark makes. Kept in [0, 1] (`unbounded = 0`).
    let field_rgb = colormap_sample(
        material.min_val, material.max_val, material.diverging,
        in.color_value * wave_osc(frame, material.wave_omega),
    );
    let rgb = select(material.color.rgb, field_rgb, material.colored > 0.5);

    let envelope = in.opacity * mix(material.fade_floor, 1.0, env);
    var out: FsOut;
    if (mask < SILHOUETTE_EPS || material.color.a * envelope < OPAQUE_ENOUGH) {
        discard;
    }
    out.color = vec4<f32>(rgb, material.color.a * envelope * mask);
    out.unbounded = 0.0;
    return out;
}
