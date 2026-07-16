// Wireframe shader — draws edges as solid dark lines.
// Reuses the same camera uniform and vertex layout as the main shader.

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) value: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// Same standing-wave displacement as the fill shader, so the wireframe tracks
// the displaced surface instead of the flat rest mesh. A line field sets the
// wave amplitude to zero, so its wireframe stays on the undisplaced surface.
struct Wave {
    time: f32,
    amplitude: f32,
    omega: f32,
    _pad: f32,
};
@group(1) @binding(0)
var<uniform> wave: Wave;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let osc = cos(wave.omega * wave.time);
    let displacement = model.value * model.normal;
    let position = model.position + wave.amplitude * osc * displacement;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    // Small depth bias: pull edges closer in NDC so they draw on top of faces.
    out.clip_position.z -= 0.0001;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0); // black edges
}
