// Scalar surface fill: the per-vertex value (a 0-form, or a line field's nodal
// magnitude) colormapped, and displaced along the vertex normal as a standing
// wave.
//
// Two vertex buffers, mirroring the bake's static/per-field split: the position
// stream is a function of the mesh alone, the value stream of the field on it.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> material: SurfaceMaterial;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) max_displacement: f32,
    @location(3) value: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) value: f32,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.value = model.value;
    let osc = wave_osc(frame, material.wave_omega);
    let position = wave_displace(material.wave_amplitude, osc, model.position, model.normal, model.value, model.max_displacement);
    out.clip_position = frame.view_proj * vec4<f32>(position, 1.0);
    return out;
}

struct FsOut {
    @location(0) color: vec4<f32>,
    // A colormapped fill is clamped to `[0, 1]` by the colormap itself, and so
    // never overflows -- see `display_transform`'s note on `unbounded_mask`.
    // Always zero: the fill never asks the tone curve for anything.
    @location(1) unbounded: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> FsOut {
    // Pulse the color by the same standing-wave factor that displaces the
    // surface: an eigenmode's crest and trough swing through the (symmetric)
    // colormap in sync with the up/down motion. A non-eigenmode has omega = 0,
    // so cos(0) = 1 leaves the color static.
    var out: FsOut;
    out.color = vec4<f32>(colormap_in(material, in.value * wave_osc(frame, material.wave_omega)), 1.0);
    out.unbounded = 0.0;
    return out;
}
