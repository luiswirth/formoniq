// Shared WGSL preamble, prepended to every shader body in this module (see
// `render::shader_source`). It declares only types, pipeline-overridable
// constants and pure functions -- never a binding, since the group a uniform
// sits in is a property of the pipeline that uses it, not of the value.
//
// Each struct here is the WGSL side of a `#[repr(C)]` Rust mirror of the same
// name -- a uniform in `uniform.rs`, or a storage element declared beside the
// pass that owns it; the two must stay byte-identical.

// `SSAA_SCALE`, the supersampling factor per axis the downsample's box filter
// divides by, is NOT declared here. It is baked into the downsample shader as a
// plain `const` from the same Rust `ssaa` that sizes the targets (see
// `render::ssaa_prelude`): WebGPU on WebKit fails to specialize a
// pipeline-overridable `override` constant ("Vertex library failed creation"),
// and the factor is fixed for a pipeline's life regardless.

// Where and when the scene is seen from: bound at group 0 by every pipeline.
// Time is the frame's; the frequency it is multiplied by belongs to the field on
// display, and so arrives in that item's material.
struct Frame {
    view_proj: mat4x4<f32>,
    // The camera's world-space forward axis, `w` unused: the billboard
    // construction below aligns to the view plane (perpendicular to this), which
    // is what makes a mark parallel to the image plane under both projections --
    // a finite eye point would be the wrong reference under an orthographic one,
    // whose rays are parallel and whose projection has no vanishing point.
    view_dir: vec4<f32>,
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
// `deposit_floor`/`deposit_gain` are the flow's illumination of the surface:
// the fill's radiance is the colormap times `floor + gain * D`, with $D$ the
// deposit atlas sample. Hue stays the field's (the data); the trails carry
// *luminance*, in unbounded radiance, so a dense filament exceeds 1 and
// blooms. A field with no deposit passes floor 1 and gain 0, which is the
// identity -- "no trails" is arithmetic, not a pipeline.
struct SurfaceMaterial {
    min_val: f32,
    max_val: f32,
    wave_amplitude: f32,
    wave_omega: f32,
    diverging: f32,
    deposit_floor: f32,
    deposit_gain: f32,
    colored: f32,
};

// How a segment mark is drawn: its ink (`rgb` plus a base opacity), its
// world-space half-width -- a fixed fraction of the mesh's own mean edge length,
// set on the Rust side, not a pixel count, so a line reads the same thickness
// whether the mesh fills the screen or sits in a corner of it, and the same
// thickness relative to the edges it traces however finely the mesh is
// refined -- the opacity it fades to at the standing wave's node, and the wave
// it rides.
struct SegmentMaterial {
    color: vec4<f32>,
    half_width_world: f32,
    fade_floor: f32,
    wave_amplitude: f32,
    wave_omega: f32,
    min_val: f32,
    max_val: f32,
    diverging: f32,
    colored: f32,
};

// How an arrow glyph is drawn: a flat mark in its surface cell, not a billboard.
// Every dimension is a proportion of the arrow's own length, which the bake
// carries per glyph and sizes from the cell -- so the mark is one shape scaled
// by its cell, self-similar at any refinement, and no world size appears here.
struct GlyphMaterial {
    color: vec4<f32>,
    width_fraction: f32,
    fade_floor: f32,
    wave_omega: f32,
    head_length_fraction: f32,
    shaft_width_fraction: f32,
    outline_width_fraction: f32,
    _pad0: f32,
    _pad1: f32,
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
    let luma = luminance(color);
    return clamp(mix(vec3<f32>(luma), color, SATURATION_BOOST), vec3<f32>(0.0), vec3<f32>(1.0));
}

// The shared colormap: normalize a value into its range and read the palette,
// diverging (centered) for a signed field, sequential for a magnitude. One
// function for every mark that reflects a field -- the fill, and any colored
// skeleton -- so they cannot drift apart.
fn colormap_sample(min_val: f32, max_val: f32, is_diverging: f32, value: f32) -> vec3<f32> {
    let t = (value - min_val) / (max_val - min_val);
    return saturate_color(select(viridis(t), diverging(t), is_diverging > 0.5));
}

fn colormap_in(material: SurfaceMaterial, value: f32) -> vec3<f32> {
    return colormap_sample(material.min_val, material.max_val, material.diverging, value);
}

// A segment (a mesh edge, a curve's cell) drawn with constant world-space
// thickness. A `LineList` primitive rasterizes at a fixed, backend-defined 1px
// in wgpu -- there is no portable line-width control -- and all but disappears
// once the supersampled scene pass is filtered down. Instead each segment is an
// instanced quad: two triangles fanned from its two endpoints, offset
// perpendicular to both the segment and the camera's forward axis, so the quad
// lies in the view plane -- parallel to the image plane -- the way a billboard
// does.
//
// Aligned to the view plane (the constant forward axis), not turned to face the
// eye point. The two agree under perspective up to a slight per-position tilt,
// but under an orthographic projection they do not: its rays are parallel and it
// has no vanishing point, so a quad facing a finite eye tilts by a different
// amount at every position and never lies flat in the image. The forward axis is
// one direction for the whole scene, so the quad is also a parallelogram of
// constant width rather than a wedge -- correct under both projections with no
// branch between them.
fn billboard_perp(world_a: vec3<f32>, world_b: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let seg_vec = world_b - world_a;
    let seg_dir = seg_vec / max(length(seg_vec), 1e-6);
    let perp = cross(seg_dir, view_dir);
    let perp_len = length(perp);
    // Segment pointing straight along the view: no well-defined in-plane
    // perpendicular, but the quad also projects to ~zero screen length there, so
    // any fallback direction is invisible anyway.
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

// Nudges a corner toward the camera, along the view direction, by a small
// multiple of the mark's own half-width, so it draws on top of the coplanar
// face beneath it instead of z-fighting it. Coplanar is the whole difficulty:
// the true depth gap is zero, so something has to break the tie, and it may as
// well be a distance the mark already has.
//
// Along the view axis, not toward the eye point: under an orthographic
// projection the two differ, and a nudge toward a finite eye would move a corner
// laterally -- shifting the mark off the fill it outlines -- rather than purely
// in depth. Along the forward axis the whole nudge is depth.
//
// World space, and tied to the mark's own scale, so the nudge is a fixed small
// distance rather than a fixed small *depth* -- an NDC-space offset would stand
// for an ever larger eye-space distance the farther out it is applied, and past
// some distance would exceed a surface's own front-to-back gap, letting the far
// side's wireframe win against the near side's fill.
//
// The multiple is not a precision threshold: the depth buffer is reversed-Z
// float, whose resolution is roughly uniform in $z$, so a world-space nudge
// survives the divide at any zoom. It only has to beat coplanarity, and stay
// far below any real front/back separation -- which it does everywhere except
// an imperceptible sliver at the silhouette, where that separation shrinks
// continuously to zero and a biased line necessarily wins.
fn depth_biased_corner(corner: vec3<f32>, view_dir: vec3<f32>, half_width: f32) -> vec3<f32> {
    return corner - view_dir * (4.0 * half_width);
}

// A particle, i.e. a `MeshPoint`, plus what respawning it needs.
//
// `vec4` carries the weights for every intrinsic dimension the ambient reaches:
// a triangle uses three components and leaves `w` at zero, a tetrahedron uses
// all four. Not a padded 3-vector -- the fourth slot is a real barycentric
// weight one dimension up, which is why a surface and a solid share this buffer
// without a branch.
//
// `life` counts down in frame steps and `epoch` counts respawns: integers, so a
// particle's whole future is a function of `(id, epoch)` and the pass is
// reproducible frame for frame. An exporter that steps from a fixed seed to
// reach its instant gets the trajectory the viewer showed.
struct Particle {
    lambda: vec4<f32>,
    cell: u32,
    life: u32,
    epoch: u32,
    _pad: u32,
};

// The per-cell topology half of the bake: the cell across the facet opposite
// vertex $i$, or `NO_NEIGHBOUR` where the manifold has boundary.
struct Cell {
    neighbour: vec4<u32>,
};

struct AdvectParams {
    particle_count: u32,
    seed_count: u32,
    // A respawned particle lives `life_min + hash % life_spread` frame steps.
    // The spread is not cosmetic: with one shared lifetime the entire
    // population expires together and the field visibly breathes.
    life_min: u32,
    life_spread: u32,
    // $d$: a frame's step is $2^d$ ticks, and the bake carries levels $0..=d$.
    // The residual crossing error is one tick, so this is the exponent that
    // buys exactness -- at $d = 16$ the error is below `f32` epsilon, which is
    // the only precision this pass has to begin with.
    depth: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// The deposit atlas passes' one uniform: the atlas side in texels, the splat
// footprint radius (texels), the ink density (deposited energy per *texel of
// path* -- see `fs_splat` for why arc length, not time, is the measure), the
// survival factor of one step's exponential fade, and the advection's dyadic
// depth (which names the whole-step flow level the splat's own displacement is
// read off). All per *step*, never per second: the deposit is stepped state,
// and its determinism contract is the advection's own.
struct DepositParams {
    atlas_size: f32,
    radius: f32,
    energy: f32,
    decay: f32,
    depth: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// Relative luminance (Rec. 709), the one place those weights are written down.
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// How the scene's radiance is brought to the display: the exposure it is read
// at, how far its overflow is allowed to spill, and which curve closes it.
//
// A uniform rather than pipeline `override`s, unlike `SSAA_SCALE`: an override
// is baked when the pipeline is built, so toggling one would mean a second
// pipeline or a rebuild. `SSAA_SCALE` earns that -- it sizes the targets, so it
// cannot change without reallocating them anyway -- and these do not.
struct Post {
    exposure: f32,
    // How much of the blurred glow is added back. Zero disables bloom by
    // arithmetic, which is what lets the frame graph skip the chain without the
    // result depending on whether it did.
    bloom_intensity: f32,
    // 1.0 for the filmic curve, 0.0 for a hard clamp.
    tonemap: f32,
    _pad0: f32,
};

// The filmic curve: Narkowicz's rational fit to the shape of the ACES
// RRT+ODT. Not ACES itself -- a cheap lookalike of its S-curve.
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// The one crossing from unbounded linear radiance to a bounded display.
//
// The scene target is float, so a filament where a hundred specks overlap
// really does carry a hundred times one speck's light. Something has to decide
// what that means in $[0, 1]$, and these are the only two honest answers:
//
// - **Clamp** is what an 8-bit target did implicitly, and is therefore not an
//   approximation of the old renderer but exactly it -- everything at or below
//   1 is untouched, everything above is white. The whole range above 1 is lost.
// - **The curve** keeps that range, at a price that is unavoidable rather than
//   a flaw in this particular fit: $[0, 1]$ is already fully spent by the marks
//   that live there, so headroom above 1 *must* take range from below it. Every
//   mark shifts -- mids up, highlights down, hues skewed as the channels
//   compress at different rates.
//
// Which is why the choice is exposed rather than decided here. There is no
// right answer, only the question of whether the dynamic range or the palette
// matters more for what is being looked at.
//
// That question only *arises* for a mark that can actually overflow. A
// colormapped fill and a segment ribbon are clamped to `[0, 1]` by
// construction -- see the same invariant `bloom.wgsl`'s prefilter turns on --
// so clamp is not a mode for them, it is the identity, and running the curve
// over them regardless is what skewed the heatmap's palette for no gain: the
// curve was never asked whether there was anything above 1 to reconcile.
// `unbounded_mask` is that question, answered per pixel by the marks that
// drew there rather than guessed from the radiance's own value -- a value near
// 1 cannot tell a saturated fill from a faint overlap of particles, but the
// scene pass that wrote it already knows. It is a coverage fraction rather
// than a bit so a particle's own antialiased edge, or its blend with a fill
// beneath it, crosses over smoothly instead of at a hard seam.
fn display_transform(post: Post, radiance: vec3<f32>, unbounded_mask: f32) -> vec3<f32> {
    let exposed = radiance * post.exposure;
    let clamped = clamp(exposed, vec3<f32>(0.0), vec3<f32>(1.0));
    let curved = select(clamped, aces(exposed), post.tonemap > 0.5);
    return mix(clamped, curved, unbounded_mask);
}

// How a volumetric field is marched: the box the grid occupies in world space,
// the colormap range the transfer function reads through, and the two scalars
// that turn a field value into a medium.
//
// `inv_view_proj` unprojects a pixel into a world-space ray, and it is the
// *inverse* of the frame's own matrix rather than an eye position and a
// frustum, because that one construction serves a perspective and an
// orthographic camera alike: unproject the same pixel at two depths and the
// segment between them is the ray, converging or parallel as the projection
// dictates. An eye point would be the wrong primitive under an orthographic
// projection, which has none.
//
// `density` is the absorption coefficient at full normalized value, in units of
// inverse world length, so the medium's opacity is a property of how far light
// travels through it and not of how many steps the march happens to take.
// `emission` scales the radiance the medium gives off, in the same normalized
// units, keeping the colormap's hue and letting its brightness be the knob.
struct VolumeMaterial {
    inv_view_proj: mat4x4<f32>,
    // The grid's world-space minimum corner and extent, `w` unused.
    origin: vec4<f32>,
    size: vec4<f32>,
    // The affine decoding of a texel back into a field value: the bake stores
    // the scalar normalized into `[0, 1]`, so `value = value_min + texel *
    // value_range`. Kept here rather than folded into the colormap range so the
    // two stay separable -- the encoding is the texture's, the range is the
    // field's.
    value_min: f32,
    value_range: f32,
    min_val: f32,
    max_val: f32,
    diverging: f32,
    density: f32,
    emission: f32,
    // World-space distance between consecutive samples along a ray.
    step_size: f32,
    // The standing wave, exactly as the fill carries it: the medium's value is
    // $u(t) = u cos(omega t)$. A solid has no surface to displace along a
    // normal, so the wave shows as the *pulse* alone -- the fog thinning to
    // nothing as the cosine crosses zero and refilling with the opposite sign.
    // A field with no eigenvalue has `wave_omega = 0`, and $cos(0) = 1$ leaves
    // it static through the same arithmetic.
    wave_omega: f32,
    // Seconds into that wave. A frame fact rather than a material choice, like
    // `inv_view_proj`, and filled by the renderer from the same `FrameView`.
    time: f32,
    _pad0: f32,
    _pad1: f32,
};
