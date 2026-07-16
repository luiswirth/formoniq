// Fullscreen line-integral convolution of the line-field G-buffer.
//
// Per covered pixel, advect the projected tangent field forward and backward a
// fixed screen-space arclength -- re-reading the field at each step and turning
// with it, so the kernel follows the curved streamline rather than shooting off
// along the center pixel's tangent -- and sample a 3D noise texture at the
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

// One half of the convolution: advect from `start_uv` along `seed_dir`,
// accumulating the object-space noise at each covered step. Forward Euler in
// screen space, one step per `STEP_PX`. The G-buffer direction is unit but
// unsigned (a *line* field), so each step flips it to agree with the travel
// direction -- otherwise the walk could reverse at a pixel whose stored sign
// disagrees and fold the streamline back on itself. A step off the surface
// (coverage drops) ends the walk rather than smearing across the silhouette.
// Returns the running (sum, weight); the caller adds the shared center sample.
fn march(start_uv: vec2<f32>, seed_dir: vec2<f32>, texel: vec2<f32>) -> vec2<f32> {
    var pos = start_uv;
    var dir = seed_dir;
    var sum = 0.0;
    var weight = 0.0;
    for (var i: i32 = 0; i < STEPS; i = i + 1) {
        pos = pos + dir * STEP_PX * texel;
        let g = textureSampleLevel(gbuf_dir_mag, gbuf_sampler, pos, 0.0);
        if (g.w < 0.5) {
            break;
        }
        var d = g.xy;
        if (dot(d, dir) < 0.0) {
            d = -d;
        }
        // Keep the previous heading through a momentarily degenerate direction
        // (a field zero) rather than stalling on a null step.
        if (dot(d, d) > 1e-8) {
            dir = d;
        }
        let wp = textureSampleLevel(gbuf_pos_shade, gbuf_sampler, pos, 0.0).xyz;
        sum = sum + noise_at(wp);
        weight = weight + 1.0;
    }
    return vec2<f32>(sum, weight);
}

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
    // this pixel, advecting the field in each direction so the kernel follows
    // the line's curvature. The noise is sampled in object space, locking the
    // pattern to the surface. The two half-marches share the center sample.
    let texel = 1.0 / lic.viewport;
    let fwd = march(in.uv, sdir, texel);
    let bwd = march(in.uv, -sdir, texel);
    let sum = noise_at(pos_shade.xyz) + fwd.x + bwd.x;
    let weight = 1.0 + fwd.y + bwd.y;
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
