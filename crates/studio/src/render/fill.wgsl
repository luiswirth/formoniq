// Scalar surface fill: the per-vertex value (a 0-form, or a line field's nodal
// magnitude) colormapped, and displaced along the vertex normal as a standing
// wave.

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> wave: Wave;
@group(2) @binding(0) var<uniform> bounds: Bounds;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) value: f32,
    @location(3) max_displacement: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) value: f32,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.value = model.value;
    let position = wave_displace(wave, model.position, model.normal, model.value, model.max_displacement);
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Pulse the color by the same standing-wave factor that displaces the
    // surface: an eigenmode's crest and trough swing through the (symmetric)
    // colormap in sync with the up/down motion. A non-eigenmode has omega = 0,
    // so cos(0) = 1 leaves the color static.
    return vec4<f32>(colormap_in(bounds, in.value * wave_osc(wave)), 1.0);
}
