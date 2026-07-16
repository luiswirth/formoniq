// Scalar-field surface: a 0-form (or a top form starred to one) colored by a
// colormap and displaced along its normal as a standing wave.

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) value: f32,
    @location(4) max_displacement: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) value: f32,
};

// Standing-wave animation of an eigenmode: $u(t) = a cos(omega t)$. A surface
// vertex is additively displaced along its own outward normal times its scalar
// value (the classical membrane picture).
struct Wave {
    time: f32,
    amplitude: f32,
    omega: f32,
    _pad: f32,
};
@group(2) @binding(0)
var<uniform> wave: Wave;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.value = model.value;
    let osc = cos(wave.omega * wave.time);
    let raw = wave.amplitude * osc * model.value;
    let capped = clamp(raw, -model.max_displacement, model.max_displacement);
    let position = model.position + capped * model.normal;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    return out;
}

struct Bounds {
    min_val: f32,
    max_val: f32,
    _pad1: f32,
    _pad2: f32,
};
@group(1) @binding(0)
var<uniform> bounds: Bounds;

// A simple turbo/viridis-like colormap approximation for 0-1
fn colormap(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let r = 0.5 + 0.5 * sin(x * 6.28 - 1.57);
    let g = 0.5 + 0.5 * sin(x * 6.28 - 1.57 + 2.09);
    let b = 0.5 + 0.5 * sin(x * 6.28 - 1.57 - 2.09);
    return vec3<f32>(r, g, b);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Pulse the color by the same standing-wave factor that displaces the
    // surface: an eigenmode's crest and trough swing through the (symmetric)
    // colormap in sync with the up/down motion. A non-eigenmode has omega = 0,
    // so cos(0) = 1 leaves the color static.
    let osc = cos(wave.omega * wave.time);
    let value = in.value * osc;
    let t = (value - bounds.min_val) / (bounds.max_val - bounds.min_val);
    let color = colormap(t);
    return vec4<f32>(color, 1.0);
}
