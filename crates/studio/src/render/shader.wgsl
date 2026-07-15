// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) value: f32,
    @location(2) displacement: vec3<f32>,
    @location(3) is_glyph: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) value: f32,
    @location(1) is_glyph: f32,
};

// Standing-wave animation of an eigenmode: $u(t) = a cos(omega t)$. A surface
// vertex is *additively* displaced along its own outward normal times its
// scalar value (the classical membrane picture). A glyph vertex has no such
// deflection reading -- it already is a vector -- so it is instead
// *multiplicatively* scaled from its arrow's root, which stays fixed at the
// sample point: at cos(omega t) = 1 it is the full static arrow, at 0 it
// collapses to the root, at -1 it is the mirrored (flipped) arrow, scaled
// further by the sample's magnitude relative to the field's peak so a weak
// part of the mode swings less than a strong one. See `Vertex::displacement`
// / `Vertex::value` (render/mesh.rs) for what each mark bakes in.
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
    out.is_glyph = model.is_glyph;
    let osc = cos(wave.omega * wave.time);
    var position: vec3<f32>;
    if (model.is_glyph > 0.5) {
        let root = model.displacement;
        let relative_magnitude = model.value;
        position = root + relative_magnitude * osc * (model.position - root);
    } else {
        position = model.position + wave.amplitude * osc * model.displacement;
    }
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    if (model.is_glyph > 0.5) {
        // A glyph sits exactly in its cell's tangent plane, coincident with
        // the surface underneath it: nudge it toward the camera in clip
        // space (view-independent, unlike a world-space lift) so it never
        // z-fights the canvas it's drawn on top of -- the same trick the
        // wireframe pipeline uses to stay above the fill.
        out.clip_position.z -= 0.0001;
    }
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
    if (in.is_glyph > 0.5) {
        // Direction-only: a glyph's magnitude already colors the surface
        // it's drawn on, so the glyph itself stays flat and neutral rather
        // than doubling up on the same reading in a second convention.
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    let t = (in.value - bounds.min_val) / (bounds.max_val - bounds.min_val);
    let color = colormap(t);
    return vec4<f32>(color, 1.0);
}
