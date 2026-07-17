// Bloom: the light that spills out of whatever is brighter than the display can
// show. See `bloom.rs`.
//
// Three stages over a halving mip chain -- prefilter the scene into the top
// level, box-downsample to the bottom, tent-upsample additively back up -- which
// is a wide, cheap Gaussian: each level doubles the radius for a quarter of the
// pixels, so a blur spanning half the screen costs barely more than one spanning
// a few texels.
//
// **What blooms is what overflows, and that is not a rule this code enforces.**
// The threshold keeps only what exceeds the display range. A colormapped fill is
// clamped to $[0, 1]$ and cannot; a black wireframe cannot; the additively
// blended particles are the only mark that accumulates past 1, and they do it
// exactly where they pile up. So the physical criterion (this is more light than
// can be shown) and the intended one (the particle density is the thing that
// glows) coincide, and no pass needs to know which mark it is looking at.

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.uv = vec2<f32>(x, y);
    out.clip = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

// Where the roll-off starts, and how softly. Across the knee the contribution
// rises quadratically rather than switching on, so a filament brightening into
// range fades in instead of popping.
//
// **The knee is what selects the mark, so it is the knee that must clear 1.0 --
// not the threshold.** The contribution ramps from `THRESHOLD - KNEE`, not from
// `THRESHOLD`, so the invariant is
//
// $ "THRESHOLD" - "KNEE" >= 1 $
//
// and it is load-bearing rather than a tuning preference. Everything below 1.0
// is a mark that fits in the display and is not emitting: a colormapped fill
// (clamped to $[0, 1]$ by `saturate_color`), a black wireframe, a white
// glyph. Only the additively blended particles exceed 1.0. A knee reaching
// under 1.0 therefore does not soften the selection, it *breaks* it -- at
// `KNEE = 0.6` against `THRESHOLD = 1.0` the ramp opened at 0.4 and the fill's
// own bright end bled a sixth of itself into the glow, haloing the heatmap into
// its neighbours.
//
// These put the ramp at exactly $[1.0, 1.6]$: nothing displayable glows, and
// what overflows glows in proportion to its overflow.
const THRESHOLD: f32 = 1.3;
const KNEE: f32 = 0.3;

fn prefilter(color: vec3<f32>) -> vec3<f32> {
    let brightness = max(color.r, max(color.g, color.b));
    var soft = brightness - THRESHOLD + KNEE;
    soft = clamp(soft, 0.0, 2.0 * KNEE);
    soft = soft * soft / (4.0 * KNEE + 1e-6);
    let contribution = max(soft, brightness - THRESHOLD) / max(brightness, 1e-6);
    return color * contribution;
}

// A 2x2 tap of the source at half its resolution. Each `textureSample` sits on a
// texel corner, so bilinear filtering makes one fetch average four texels: the
// sixteen texels of the footprint cost four fetches.
fn box_4(uv: vec2<f32>, texel: vec2<f32>) -> vec3<f32> {
    let a = textureSample(source, source_sampler, uv + texel * vec2<f32>(-1.0, -1.0)).rgb;
    let b = textureSample(source, source_sampler, uv + texel * vec2<f32>(1.0, -1.0)).rgb;
    let c = textureSample(source, source_sampler, uv + texel * vec2<f32>(-1.0, 1.0)).rgb;
    let d = textureSample(source, source_sampler, uv + texel * vec2<f32>(1.0, 1.0)).rgb;
    return (a + b + c + d) * 0.25;
}

// The scene, thresholded, into the top of the chain.
//
// The four taps are averaged by Karis's weighting, $w = 1 \/ (1 + "luma")$,
// rather than evenly. A single speck far brighter than its neighbours would
// otherwise dominate its own 2x2 and, as the particle moves sub-pixel between
// frames, flicker the whole blurred halo it seeds -- the firefly. Weighting by
// the inverse of brightness is what makes the average stable under that motion,
// at the cost of slightly understating a genuinely isolated highlight.
@fragment
fn fs_prefilter(in: VsOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(source, 0));
    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    for (var i = 0; i < 4; i++) {
        let tap = textureSample(source, source_sampler, in.uv + texel * offsets[i]).rgb;
        let weight = 1.0 / (1.0 + luminance(tap));
        sum += tap * weight;
        weight_sum += weight;
    }
    return vec4<f32>(prefilter(sum / max(weight_sum, 1e-6)), 1.0);
}

@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(source, 0));
    return vec4<f32>(box_4(in.uv, texel), 1.0);
}

// A 3x3 tent back up into the level above, blended additively by the pipeline:
// each level adds its own, wider blur to the one below it, and the sum over the
// chain is the wide Gaussian no single pass computes.
@fragment
fn fs_upsample(in: VsOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(source, 0));
    var sum = vec3<f32>(0.0);
    sum += textureSample(source, source_sampler, in.uv + texel * vec2<f32>(-1.0, -1.0)).rgb * 1.0;
    sum += textureSample(source, source_sampler, in.uv + texel * vec2<f32>(0.0, -1.0)).rgb * 2.0;
    sum += textureSample(source, source_sampler, in.uv + texel * vec2<f32>(1.0, -1.0)).rgb * 1.0;
    sum += textureSample(source, source_sampler, in.uv + texel * vec2<f32>(-1.0, 0.0)).rgb * 2.0;
    sum += textureSample(source, source_sampler, in.uv).rgb * 4.0;
    sum += textureSample(source, source_sampler, in.uv + texel * vec2<f32>(1.0, 0.0)).rgb * 2.0;
    sum += textureSample(source, source_sampler, in.uv + texel * vec2<f32>(-1.0, 1.0)).rgb * 1.0;
    sum += textureSample(source, source_sampler, in.uv + texel * vec2<f32>(0.0, 1.0)).rgb * 2.0;
    sum += textureSample(source, source_sampler, in.uv + texel * vec2<f32>(1.0, 1.0)).rgb * 1.0;
    return vec4<f32>(sum / 16.0, 1.0);
}
