// The deposit atlas passes: fade, then splat. See `deposit.rs`, and
// `crate::deposit` for the layout the block table encodes.
//
// The atlas is manifold state -- a texel is a barycentric lattice point of one
// cell -- so no camera, no depth and no frame uniform appear anywhere here.
// Both passes render at the atlas's own resolution, once per advection step,
// and the ping-pong is what makes the fade a pure function: new = decay * old,
// read from one texture, written to the other, plus this step's splats on top.
//
// Determinism is per *step*, not per frame: a frame that owes several steps
// records this pair several times, so the accumulated trail is a function of
// the step count alone -- the same contract the advection itself keeps, and
// what lets a window and an exporter agree on the picture.

@group(0) @binding(0) var<uniform> deposit: DepositParams;
@group(1) @binding(0) var previous_tex: texture_2d<f32>;
@group(1) @binding(1) var previous_samp: sampler;
@group(2) @binding(0) var<storage, read> particles: array<Particle>;
// Per cell: block origin (x, y) and lattice resolution, `w` unused. Cell
// order, so a particle's own `cell` indexes it directly.
@group(2) @binding(1) var<storage, read> blocks: array<vec4<u32>>;
// $e^(M h 2^k)$ per cell and level, exactly as `advect.wgsl` reads it. Only
// the top level -- one whole step -- is read here, for the splat's own
// displacement.
@group(2) @binding(2) var<storage, read> flows: array<mat4x4<f32>>;

// The continuous texel coordinate of a point of the manifold: the block
// formula $O_c + R_c (lambda_0, lambda_1)$ -- the same affine map the fill's
// interpolated corner attribute extends, so the two sides cannot disagree
// about where a point of a cell lives in the atlas.
fn atlas_texel(cell: u32, lambda: vec4<f32>) -> vec2<f32> {
    let block = blocks[cell];
    return vec2<f32>(block.xy) + f32(block.z) * lambda.xy;
}

// Texel coordinates to clip space, y flipped so that texel (0, 0) is written
// at texture coordinate (0, 0): the one place the two conventions meet.
fn texel_to_clip(texel: vec2<f32>) -> vec4<f32> {
    let ndc = texel / deposit.atlas_size * 2.0 - 1.0;
    return vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
}

struct FadeOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_fade(@builtin(vertex_index) vertex_index: u32) -> FadeOut {
    // The standard fullscreen triangle.
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: FadeOut;
    out.clip = vec4<f32>(corners[vertex_index], 0.0, 1.0);
    return out;
}

// Exponential decay, one step's worth. `textureLoad` rather than a sampled
// read: the fade is texel-to-texel, and filtering here would diffuse the
// trail a little every step -- a blur nobody asked for, compounding.
@fragment
fn fs_fade(in: FadeOut) -> @location(0) vec4<f32> {
    let old = textureLoad(previous_tex, vec2<i32>(in.clip.xy), 0).r;
    return vec4<f32>(deposit.decay * old, 0.0, 0.0, 1.0);
}

struct SplatOut {
    @builtin(position) clip: vec4<f32>,
    // Position within the quad in units of the radius: the Gaussian's own
    // coordinate, exactly as the head speck's.
    @location(0) offset: vec2<f32>,
    // This step's path length in texels: the arc-length measure the fragment
    // scales its ink by.
    @location(1) step_texels: f32,
};

// One quad per particle, in atlas space: the particle's footprint, deposited
// where it *is*. The radius is in texels -- and texel density is uniform per
// metric area by construction, so a fixed texel radius is a fixed world
// radius, which is the world-space discipline every mark here keeps.
//
// The step's displacement is one more mat-vec by the cell's own whole-step
// flow, taken in *atlas* coordinates -- the same derivation the head speck's
// motion blur makes in ambient ones, and exact for the same reason: the flow
// matrix is the field, not a history of it. A crossing within the step is not
// accounted for, which understates the path only near a facet, at the scale
// of one splat.
@vertex
fn vs_splat(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance: u32) -> SplatOut {
    let particle = particles[instance];
    let center = atlas_texel(particle.cell, particle.lambda);
    let flow = flows[particle.cell * (deposit.depth + 1u) + deposit.depth];
    let next = atlas_texel(particle.cell, flow * particle.lambda);
    let u = select(-1.0, 1.0, billboard_is_b(vertex_index));
    let v = billboard_side(vertex_index);
    let corner = center + vec2<f32>(u, v) * deposit.radius;

    var out: SplatOut;
    out.clip = texel_to_clip(corner);
    out.offset = vec2<f32>(u, v);
    out.step_texels = distance(center, next);
    return out;
}

// The same procedural Gaussian profile the head speck draws with, here in
// texel units: a hard stamp would alias against the lattice as the particle
// crosses texel boundaries, and the smooth footprint is what makes the laid
// trail continuous rather than beaded. Additively blended, so overlapping
// footprints accumulate into density.
const SPLAT_FALLOFF: f32 = 3.5;

// Ink by *arc length*, not by time: the splat scales by the texels the
// particle moves this step. The trail's subject is motion, so a stationary
// particle -- a zero of the field -- deposits exactly nothing rather than
// piling a bright static dot that says nothing. It also cancels the classical
// occupancy bias by construction: a slow particle re-inks its texel more
// steps at proportionally less ink each, so a streak carries uniform ink per
// texel of its length whatever the speed that drew it.
@fragment
fn fs_splat(in: SplatOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.offset, in.offset);
    let gaussian = exp(-SPLAT_FALLOFF * r2);
    return vec4<f32>(deposit.energy * in.step_texels * gaussian, 0.0, 0.0, 1.0);
}
