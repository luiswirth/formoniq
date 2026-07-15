// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) value: f32,
    @location(2) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) value: f32,
};

// Standing-wave displacement of an eigenmode: $u(t) = a cos(omega t)$ at each
// vertex, moved along its own outward normal -- so `omega` and `amplitude` are
// a property of the mode being shown, not a fixed animation.
struct Wave {
    time: f32,
    amplitude: f32,
    omega: f32,
    _pad: f32,
};
@group(2) @binding(0)
var<uniform> wave: Wave;

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.value = model.value;
    let displacement = wave.amplitude * model.value * cos(wave.omega * wave.time);
    let position = model.position + displacement * model.normal;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    return out;
}

// Fragment shader

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
    let t = (in.value - bounds.min_val) / (bounds.max_val - bounds.min_val);
    let color = colormap(t);
    return vec4<f32>(color, 1.0);
}
