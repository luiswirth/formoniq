// The resolve: the supersampled scene target filtered down into the caller's
// render target, with the bloom composited in and the whole tone mapped. The
// last pass, and the only one that writes the render target directly.
//
// Three jobs in one pass because they are one operation: the scene target is
// linear HDR and the render target is `[0, 1]` sRGB, so this is the crossing
// between the two, and everything that must happen in radiance -- filtering,
// adding light -- happens before the curve that leaves it.

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var bloom_sampler: sampler;
// The unbounded-coverage mask the scene pass wrote alongside `scene_tex`: see
// `MASK_FORMAT` and `display_transform`'s note on `unbounded_mask`.
@group(0) @binding(3) var mask_tex: texture_2d<f32>;
@group(1) @binding(0) var<uniform> post: Post;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base = vec2<i32>(in.clip_position.xy) * SSAA_SCALE;

    // The box filter, weighted by $1 \/ (1 + "luma")$ rather than evenly.
    //
    // An even mean was right while the target was 8-bit and everything in it was
    // bounded by 1. In HDR it is not: one subsample of an additive filament can
    // be a hundred times its neighbours, so it alone decides the pixel, and as
    // the particle drifts sub-pixel between frames the pixel flickers -- the
    // firefly. Weighting each subsample down by its own brightness is what makes
    // the mean stable under that motion. It costs a slight understatement of a
    // genuinely isolated highlight, which the bloom then puts back as glow.
    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    var alpha = 0.0;
    var mask = 0.0;
    for (var dy: i32 = 0; dy < SSAA_SCALE; dy = dy + 1) {
        for (var dx: i32 = 0; dx < SSAA_SCALE; dx = dx + 1) {
            let coord = base + vec2<i32>(dx, dy);
            let texel = textureLoad(scene_tex, coord, 0);
            let weight = 1.0 / (1.0 + luminance(texel.rgb));
            sum += texel.rgb * weight;
            weight_sum += weight;
            alpha += texel.a;
            // Plain mean, not luminance-weighted: this is a coverage fraction,
            // not radiance, so every subsample counts equally regardless of
            // how bright it is.
            mask += textureLoad(mask_tex, coord, 0).r;
        }
    }
    let resolved = sum / max(weight_sum, 1e-6);
    let unbounded_mask = mask / f32(SSAA_SCALE * SSAA_SCALE);

    // The glow is sampled, not loaded: the chain is at half the scene's
    // resolution and below, so bilinear across it is what keeps the halo smooth
    // rather than blocky.
    let glow = textureSample(bloom_tex, bloom_sampler, in.uv).rgb;

    // The glow is added to the *radiance*, before the curve: the bloom chain
    // read this same untone-mapped target, and adding its light afterwards --
    // or tone mapping before the chain -- would leave the threshold nothing
    // above 1 to find and silently make bloom a no-op.
    let mapped = display_transform(post, resolved + glow * post.bloom_intensity, unbounded_mask);
    return vec4<f32>(mapped, alpha / f32(SSAA_SCALE * SSAA_SCALE));
}
