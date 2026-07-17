// Scalar surface fill: the per-corner cell-local value (a density, or a line
// field's magnitude) colormapped, displaced along the vertex normal as a
// standing wave, and lit by the deposit atlas where a particle population trails
// one.
//
// Four vertex streams, mirroring the bake's static/per-field split: the corner
// stream is a function of the mesh alone, the value and height streams of the
// field on it (the discontinuous colormap and the continuous displacement
// height respectively), and the deposit coordinate of the mesh's atlas layout.
// The deposit
// coordinate is interpolated plainly -- the map (cell, bary) -> atlas texel is
// affine per triangle, so the rasterizer's interpolation of the three corner
// values *is* that map at every fragment, and no per-fragment lookup exists.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> material: SurfaceMaterial;
@group(2) @binding(0) var deposit_tex: texture_2d<f32>;
@group(2) @binding(1) var deposit_samp: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) max_displacement: f32,
    @location(3) value: f32,
    @location(4) deposit_uv: vec2<f32>,
    // The displacement height, per mesh vertex (so a shared vertex stays
    // single-valued and the surface does not tear), distinct from `value`, the
    // per-corner cell-local colormap scalar.
    @location(5) height: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) value: f32,
    @location(1) deposit_uv: vec2<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.value = model.value;
    out.deposit_uv = model.deposit_uv;
    let osc = wave_osc(frame, material.wave_omega);
    let position = wave_displace(material.wave_amplitude, osc, model.position, model.normal, model.height, model.max_displacement);
    out.clip_position = frame.view_proj * vec4<f32>(position, 1.0);
    return out;
}

struct FsOut {
    @location(0) color: vec4<f32>,
    // Nonzero exactly where the deposit lifts the radiance above 1: the
    // colormap itself is clamped to `[0, 1]`, so without trails the fill never
    // asks the tone curve for anything -- see `display_transform`'s note on
    // `unbounded_mask`. With them, a dense filament overflows on purpose, and
    // this is what routes it through the curve (and the bloom) rather than a
    // clamp. A coverage fraction, so the crossing is smooth.
    @location(1) unbounded: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> FsOut {
    // Pulse the color by the same standing-wave factor that displaces the
    // surface: an eigenmode's crest and trough swing through the (symmetric)
    // colormap in sync with the up/down motion. A non-eigenmode has omega = 0,
    // so cos(0) = 1 leaves the color static.
    let ink = colormap_in(material, in.value * wave_osc(frame, material.wave_omega));
    // The flow's illumination: trails brighten the surface under them, in
    // radiance, before the tone map. Hue stays the colormap's -- the data --
    // and the deposit carries luminance only. Floor 1, gain 0 is the identity:
    // a field with no trails is the plain fill by arithmetic.
    let trail = textureSample(deposit_tex, deposit_samp, in.deposit_uv).r;
    let lift = material.deposit_floor + material.deposit_gain * trail;
    var out: FsOut;
    out.color = vec4<f32>(ink * lift, 1.0);
    out.unbounded = clamp(lift - 1.0, 0.0, 1.0);
    return out;
}
