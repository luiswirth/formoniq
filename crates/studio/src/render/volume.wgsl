// The volume march: a field on a solid drawn as a participating medium. See
// `volume.rs`.
//
// A codimension-zero manifold has no surface to rasterize, so the eye ray
// integrates through it instead. The emission-absorption integral
//
//   $L = integral_0^infinity c(s) sigma(s) exp(-integral_0^s sigma(u) dif u) dif s$
//
// is accumulated front-to-back, which is why this needs no sorting and no
// order-independent-transparency machinery: the ray *is* the depth order. That
// is the whole reason a medium beats drawing the interior faces translucent,
// where the blend order is the mesh's and the brightness tracks the
// triangulation rather than the field.
//
// The march is clamped by the depth buffer, so opaque geometry (the boundary
// skeleton, a line field's ribbons) correctly occludes the fog behind it while
// the fog in front of it still fogs it. Depth is *read*, never written: a medium
// has no surface to put in the depth buffer.

@group(0) @binding(0) var<uniform> volume_material: VolumeMaterial;

@group(1) @binding(0) var volume: texture_3d<f32>;
@group(1) @binding(1) var volume_sampler: sampler;

// The scene's depth, in its own group: it is reallocated whenever the target
// resizes, so it cannot sit in the batch's bind group beside a texture that
// outlives the resize.
@group(2) @binding(0) var scene_depth: texture_depth_2d;


// The ceiling on samples per ray. The step size is chosen from the grid so a
// ray crossing it takes far fewer than this; the bound exists so a grazing ray
// through a long thin box cannot stall the frame, and it is a *budget*, not the
// sampling rate. A ray that exhausts it terminates early, leaving the far side
// of the medium unintegrated -- visible as a soft cut, and preferable to a
// dropped frame.
const MAX_STEPS: i32 = 512;

// Below this the medium has absorbed essentially everything and the remaining
// samples cannot change the pixel: 1/255 is the display's own resolution.
const MIN_TRANSMITTANCE: f32 = 0.004;

// Below this a sample is empty space and contributes nothing worth a colormap
// evaluation. The mesh does not fill its bounding box and the field vanishes
// over much of what it does, so this skip is most of the march.
const MIN_OCCUPANCY: f32 = 0.002;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.ndc = vec2<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0);
    out.clip = vec4<f32>(out.ndc, 0.0, 1.0);
    return out;
}

// The world-space point a pixel at `ndc` and reversed-Z depth `z` unprojects to.
fn unproject(m: VolumeMaterial, ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let p = m.inv_view_proj * vec4<f32>(ndc, z, 1.0);
    return p.xyz / p.w;
}

// The ray's entry and exit parameters through the grid's axis-aligned box, as
// `vec2(t_near, t_far)`; empty when `t_near > t_far`. The standard slab test,
// with the division left to run into infinities on an axis-parallel ray: the
// min/max of an infinity and a finite bound is the finite one, so a ray exactly
// parallel to a face needs no special case.
fn slab(m: VolumeMaterial, origin: vec3<f32>, dir: vec3<f32>) -> vec2<f32> {
    let inv = 1.0 / dir;
    let a = (m.origin.xyz - origin) * inv;
    let b = (m.origin.xyz + m.size.xyz - origin) * inv;
    let lo = min(a, b);
    let hi = max(a, b);
    let t_near = max(max(lo.x, lo.y), lo.z);
    let t_far = min(min(hi.x, hi.y), hi.z);
    return vec2<f32>(t_near, t_far);
}

// The field value at a world position, decoded from the texture's normalized
// storage. Outside the grid the sampler's clamped edge would repeat the boundary
// texel, so the caller must only ask inside the slab.
fn field_at(m: VolumeMaterial, x: vec3<f32>) -> f32 {
    let uvw = (x - m.origin.xyz) / m.size.xyz;
    let texel = textureSampleLevel(volume, volume_sampler, uvw, 0.0).r;
    return m.value_min + texel * m.value_range;
}

// How much of the display range a value occupies, which is what drives both the
// absorption and the emission. Magnitude, not signed value: a diverging field's
// two lobes are equally present as medium, and only the *color* distinguishes
// them. Normalizing against whichever end of the range is further from zero
// keeps that symmetric.
fn occupancy(m: VolumeMaterial, value: f32) -> f32 {
    let scale = max(abs(m.min_val), abs(m.max_val));
    return clamp(abs(value) / max(scale, 1e-12), 0.0, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let m = volume_material;

    // The ray, from two unprojections of the same pixel. Reversed-Z puts the
    // near plane at 1 and the far plane at 0.
    let near = unproject(m, in.ndc, 1.0);
    let far = unproject(m, in.ndc, 0.0);
    let dir = normalize(far - near);

    var span = slab(m, near, dir);
    span.x = max(span.x, 0.0);

    // The opaque geometry in front of the fog, as a distance along the ray. The
    // cleared depth is the far plane, which unprojects to the far end of the
    // frustum and therefore clamps nothing.
    let pixel = vec2<i32>(in.clip.xy);
    let depth = textureLoad(scene_depth, pixel, 0);
    let hit = unproject(m, in.ndc, depth);
    span.y = min(span.y, dot(hit - near, dir));

    if (span.x >= span.y) {
        discard;
    }

    // Front-to-back accumulation in radiance. `transmittance` is what survives
    // to the eye; the loop stops when nothing meaningfully does, which is a
    // performance early-out and not a change to the integral.
    var radiance = vec3<f32>(0.0);
    var transmittance = 1.0;
    let dt = m.step_size;
    var t = span.x + dt * 0.5;
    for (var i = 0; i < MAX_STEPS; i = i + 1) {
        if (t >= span.y || transmittance < MIN_TRANSMITTANCE) {
            break;
        }
        // The standing wave, applied to the sampled value before anything reads it,
    // so the palette and the opacity pulse together the way the fill's do.
    let value = field_at(m, near + dir * t) * cos(m.wave_omega * m.time);
        let a = occupancy(m, value);
        // Empty space is the common case inside the box (the mesh does not fill
        // its own bounding box, and the field vanishes over much of what it
        // does), so skipping it is most of the frame's speed.
        if (a > MIN_OCCUPANCY) {
            let sigma = m.density * a;
            let absorbed = 1.0 - exp(-sigma * dt);
            let color = colormap_sample(m.min_val, m.max_val, m.diverging, value);
            radiance = radiance + transmittance * absorbed * color * m.emission;
            transmittance = transmittance * (1.0 - absorbed);
        }
        t = t + dt;
    }

    // Premultiplied: the pass blends `One`/`OneMinusSrcAlpha`, which is exactly
    // "the fog over what was already there".
    return vec4<f32>(radiance, 1.0 - transmittance);
}
