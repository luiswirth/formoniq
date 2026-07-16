// Box-filter downsample of the supersampled scene color target into the render
// target: the antialiasing step, applied once after the scene passes have
// finished drawing (see `render::SSAA_SCALE` for why this replaces MSAA). The
// factor is the preamble's `SSAA_SCALE` override, set from that same Rust
// constant at pipeline creation, so the filter cannot disagree with the
// resolution the targets were allocated at.

@group(0) @binding(0) var scene_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base = vec2<i32>(in.clip_position.xy) * SSAA_SCALE;
    var sum = vec4<f32>(0.0);
    for (var dy: i32 = 0; dy < SSAA_SCALE; dy = dy + 1) {
        for (var dx: i32 = 0; dx < SSAA_SCALE; dx = dx + 1) {
            sum = sum + textureLoad(scene_tex, base + vec2<i32>(dx, dy), 0);
        }
    }
    return sum / f32(SSAA_SCALE * SSAA_SCALE);
}
