// Box-filter downsample of the supersampled scene color target into the
// swapchain: the antialiasing step for both the direct fill/wireframe path
// and the G-buffer/LIC path, applied uniformly after either has finished
// drawing (see `lib.rs`'s `SSAA_SCALE` for why this replaces MSAA).
//
// `SCALE` must match `SSAA_SCALE` in `lib.rs` -- there is no shared constant
// between Rust and WGSL, so the two are kept in sync by hand.
const SCALE: i32 = 2;

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
    let base = vec2<i32>(in.clip_position.xy) * SCALE;
    var sum = vec4<f32>(0.0);
    for (var dy: i32 = 0; dy < SCALE; dy = dy + 1) {
        for (var dx: i32 = 0; dx < SCALE; dx = dx + 1) {
            sum = sum + textureLoad(scene_tex, base + vec2<i32>(dx, dy), 0);
        }
    }
    return sum / f32(SCALE * SCALE);
}
