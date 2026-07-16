// Fullscreen line-integral convolution of the line-field G-buffer.
//
// Per covered pixel, integrate the projected tangent direction forward and
// backward a fixed screen-space arclength, sampling a 3D noise texture at the
// *world* position read back from the G-buffer at each step. Keying the noise
// on object space (not screen space) is what fixes the streamlines to the
// surface. The walk itself is in screen space, so the pattern is not fully
// view-independent -- a fully object/texture-space LIC is the later refinement.
//
// The line field itself is static: $ker$ and $sharp$ are scale-invariant, so
// the standing wave $u(t) = cos(sqrt(lambda) t) phi$ never rotates or travels
// the lines. Only the magnitude *envelope* animates -- it swings the tint
// through the colormap and fades the strokes out at the zero crossing, where
// the field momentarily vanishes.

struct Lic {
    viewport: vec2<f32>,
    noise_scale: f32,
    omega: f32,
    time: f32,
    contrast: f32,
    _pad0: f32,
    _pad1: f32,
};

struct Bounds {
    min_val: f32,
    max_val: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var gbuf_dir_mag: texture_2d<f32>;
@group(0) @binding(1) var gbuf_pos_shade: texture_2d<f32>;
@group(0) @binding(2) var gbuf_sampler: sampler;
@group(1) @binding(0) var noise_tex: texture_3d<f32>;
@group(1) @binding(1) var noise_sampler: sampler;
@group(2) @binding(0) var<uniform> lic: Lic;
@group(3) @binding(0) var<uniform> bounds: Bounds;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.uv = vec2<f32>(x, y);
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

fn colormap(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    let r = 0.5 + 0.5 * sin(x * 6.28 - 1.57);
    let g = 0.5 + 0.5 * sin(x * 6.28 - 1.57 + 2.09);
    let b = 0.5 + 0.5 * sin(x * 6.28 - 1.57 - 2.09);
    return vec3<f32>(r, g, b);
}

fn noise_at(world_pos: vec3<f32>) -> f32 {
    return textureSampleLevel(noise_tex, noise_sampler, world_pos * lic.noise_scale, 0.0).r;
}

const BACKGROUND: vec3<f32> = vec3<f32>(0.1, 0.1, 0.1);
const STEPS: i32 = 24;
const STEP_PX: f32 = 1.2;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir_mag = textureSampleLevel(gbuf_dir_mag, gbuf_sampler, in.uv, 0.0);
    let coverage = dir_mag.w;
    if (coverage < 0.5) {
        return vec4<f32>(BACKGROUND, 1.0);
    }

    let sdir = dir_mag.xy;
    let magnitude = dir_mag.z;
    let pos_shade = textureSampleLevel(gbuf_pos_shade, gbuf_sampler, in.uv, 0.0);
    let shade = pos_shade.w;

    // Line-integral convolution: average the noise along the streamline through
    // this pixel, sampling in object space so the pattern is locked to the
    // surface. Steps that leave the surface (coverage drops) are dropped rather
    // than smeared across the silhouette.
    let texel = 1.0 / lic.viewport;
    var sum = noise_at(pos_shade.xyz);
    var weight = 1.0;
    for (var i: i32 = 1; i <= STEPS; i = i + 1) {
        let off = sdir * (f32(i) * STEP_PX) * texel;
        let uv_f = in.uv + off;
        let uv_b = in.uv - off;
        let g_f = textureSampleLevel(gbuf_dir_mag, gbuf_sampler, uv_f, 0.0);
        let g_b = textureSampleLevel(gbuf_dir_mag, gbuf_sampler, uv_b, 0.0);
        if (g_f.w > 0.5) {
            sum = sum + noise_at(textureSampleLevel(gbuf_pos_shade, gbuf_sampler, uv_f, 0.0).xyz);
            weight = weight + 1.0;
        }
        if (g_b.w > 0.5) {
            sum = sum + noise_at(textureSampleLevel(gbuf_pos_shade, gbuf_sampler, uv_b, 0.0).xyz);
            weight = weight + 1.0;
        }
    }
    let lic_raw = sum / weight;
    // Contrast-stretch about the noise mean of 0.5, then push toward the
    // extremes with a smoothstep: the along-line average otherwise regresses to
    // grey, and the two together turn faint streaks into crisp light/dark
    // strokes.
    let stretched = clamp((lic_raw - 0.5) * lic.contrast + 0.5, 0.0, 1.0);
    let l = smoothstep(0.15, 0.85, stretched);

    // The standing-wave envelope: `osc` swings the tint through the symmetric
    // bounds $[-m, m]$ (the colormap pulses through its midpoint and flips as
    // the cosine crosses zero), and `env` fades the strokes toward a flat mid
    // tone as the field vanishes at the zero crossing -- there is no field to
    // draw lines for at that instant.
    let osc = cos(lic.omega * lic.time);
    let env = abs(osc);
    let signed = magnitude * osc;
    let t = (signed - bounds.min_val) / (bounds.max_val - bounds.min_val);
    let tint = colormap(t);

    let stroke = mix(0.5, l, env);
    // Wide brightness swing so the strokes dominate the read: dark valleys near
    // black, bright ridges pushed slightly over the tint, which still shows
    // through as the streaks' hue.
    let color = tint * shade * (0.08 + 1.25 * stroke);
    return vec4<f32>(color, 1.0);
}
