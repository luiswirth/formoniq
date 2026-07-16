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

// Where and when the scene is seen from: bound at group 0 by every pipeline.
// Time is the frame's; the frequency it is multiplied by belongs to the field on
// display, and so arrives in that item's material.
struct Frame {
    view_proj: mat4x4<f32>,
    // World-space eye position, `w` unused: the billboard construction below
    // needs it directly, not only the matrix it feeds into.
    eye: vec4<f32>,
    time: f32,
    // Three scalars, not a `vec3<f32>`: a WGSL vector is 16-aligned, so a
    // `vec3` here would start a new 16-byte slot past `time` rather than fill
    // out its one -- padding that pads the struct wider instead of closing it.
    // The Rust mirror is `[f32; 3]`, aligned as its element, and these match it.
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

// How a filled surface is drawn: its colormap normalization range, and the
// standing-wave animation of an eigenmode, $u(t) = a cos(omega t)$ -- a vertex
// additively displaced along its own outward normal times its scalar value (the
// classical membrane picture). A field with no eigenvalue has `omega = 0` and
// `amplitude = 0`, so the same code leaves it static.
// `diverging` is 1.0 for a signed field (colormap centered at its normalized
// midpoint, `min_val`/`max_val` symmetric about zero) and 0.0 for an unsigned
// magnitude (sequential, dark-to-bright low-to-high). Material data, not a
// global shader choice: which one applies is a property of the field, decided
// once in `display.rs` alongside the range itself.
struct SurfaceMaterial {
    min_val: f32,
    max_val: f32,
    wave_amplitude: f32,
    wave_omega: f32,
    diverging: f32,
};

// How a segment mark is drawn: its ink (`rgb` plus a base opacity), its
// world-space half-width -- a fixed fraction of the scene's own extent, set on
// the Rust side, not a pixel count, so a line reads the same thickness whether
// the mesh fills the screen or sits in a corner of it -- the opacity it fades to
// at the standing wave's node, and the wave it rides.
struct SegmentMaterial {
    color: vec4<f32>,
    half_width_world: f32,
    fade_floor: f32,
    wave_amplitude: f32,
    wave_omega: f32,
};

// The standing wave's instantaneous phase factor $cos(omega t)$: the one
// oscillator the displacement, the colormap pulse and the segment fade all read,
// so they cannot drift out of phase.
fn wave_osc(frame: Frame, omega: f32) -> f32 {
    return cos(omega * frame.time);
}

// The wave-displaced world position of one vertex, clamped to that vertex's own
// curvature cap so a thin or sharply curved feature never folds through its
// focal point. A vertex with a zero normal -- a mark that sits on no surface --
// is left where it is by the same expression.
fn wave_displace(amplitude: f32, osc: f32, position: vec3<f32>, normal: vec3<f32>, value: f32, max_displacement: f32) -> vec3<f32> {
    let raw = amplitude * osc * value;
    let capped = clamp(raw, -max_displacement, max_displacement);
    return position + capped * normal;
}

// Perceptually uniform sequential map (a polynomial fit to viridis), for an
// unsigned magnitude: dark, low-saturation at 0, rising in lightness and
// saturation to a bright yellow at 1. Monotone in luminance, which is exactly
// what lets a single fixed ink separate from it everywhere -- unlike the
// three-sinusoid rainbow this replaces, whose channels summed to a constant
// and made it iso-luminant.
fn viridis(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let c0 = vec3<f32>(0.2777273272, 0.0054073445, 0.3340998053);
    let c1 = vec3<f32>(0.1050930431, 1.4046135299, 1.3845901626);
    let c2 = vec3<f32>(-0.3308618287, 0.2148475595, 0.0950951630);
    let c3 = vec3<f32>(-4.6342304990, -5.7991009734, -19.3324409563);
    let c4 = vec3<f32>(6.2282699363, 14.1799333668, 56.6905526007);
    let c5 = vec3<f32>(4.7763849977, -13.7451453777, -65.3530326334);
    let c6 = vec3<f32>(-5.4354558559, 4.6458526122, 26.3124352496);
    return c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * c6)))));
}

// Diverging map for a zero-centered signed field: blue for negative, off-white
// at the midpoint, red for positive -- so the sign of the field is legible at
// a glance, not just its magnitude.
fn diverging(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let blue = vec3<f32>(0.230, 0.299, 0.754);
    let white = vec3<f32>(0.865, 0.865, 0.865);
    let red = vec3<f32>(0.706, 0.016, 0.150);
    return select(mix(white, red, (x - 0.5) * 2.0), mix(blue, white, x * 2.0), x < 0.5);
}

// Saturation boost applied to every colormap sample: pushes each channel away
// from the sample's own luminance, so the fill reads as fully saturated ink
// rather than the paler blend a straight polynomial fit gives near the middle
// of either map's range.
const SATURATION_BOOST: f32 = 1.6;

fn saturate_color(color: vec3<f32>) -> vec3<f32> {
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    return clamp(mix(vec3<f32>(luminance), color, SATURATION_BOOST), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn colormap_in(material: SurfaceMaterial, value: f32) -> vec3<f32> {
    let t = (value - material.min_val) / (material.max_val - material.min_val);
    return saturate_color(select(viridis(t), diverging(t), material.diverging > 0.5));
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

// Nudges a corner toward the camera by a small multiple of the mark's own
// half-width, so it draws on top of the coplanar face beneath it instead of
// z-fighting it. Biasing in world space, by an amount tied to the mark's own
// scale, keeps this an actual small distance regardless of how far the object
// is from the camera -- a fixed NDC-space offset does not: under a
// perspective projection the same NDC delta corresponds to an ever larger
// eye-space distance the farther out it is applied, and past some distance it
// exceeds the real depth gap between a surface's near and far side, letting
// the far side's wireframe win the depth test against the near side's fill.
//
// The multiple has to clear float32 depth-buffer roundoff, not just be
// "small": at a typical camera distance the perspective divide compresses a
// world-space nudge of one half-width into only a handful of representable
// depth steps, indistinguishable from the matrix multiply's own rounding
// error -- which is z-fighting again, just at a much smaller and harder-to-see
// bias than the old NDC-constant one. A handful of half-widths is still far
// below any real front/back depth separation except within an imperceptible
// sliver at the silhouette itself, where a biased line necessarily wins
// against a true depth gap that shrinks continuously to zero.
fn depth_biased_corner(corner: vec3<f32>, eye: vec3<f32>, half_width: f32) -> vec3<f32> {
    let to_eye = eye - corner;
    let dir = to_eye / max(length(to_eye), 1e-6);
    return corner + dir * (4.0 * half_width);
}
