// The particle mark: one screen-facing speck per advected particle, additively
// blended. See `particles.rs`, and `advect.wgsl` for the pass that moves them.
//
// The particle buffer is read here as *storage*, indexed by the instance, not
// bound as a vertex stream: the advection writes barycentric weights and a cell
// index, which are not what a rasterizer interpolates, and the ambient position
// is derived rather than stored. Nothing writes a position anywhere.
//
// **Everything a speck is shaded by is derived, not stored.** A particle's
// velocity is one more mat-vec by the cell's own flow -- where it will be, less
// where it is -- so its speed, its heading and its motion blur cost one matrix
// multiply and no memory at all. That is what a trail buffer would have bought
// at 32 bytes a sample, and it is exact rather than sampled: the flow matrix is
// the field, not a history of it.

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> material: ParticleMaterial;
@group(2) @binding(0) var<storage, read> particles: array<Particle>;
// The ambient position of each cell's local vertices, indexed
// `4 * cell + vertex`.
@group(2) @binding(1) var<storage, read> cell_positions: array<vec4<f32>>;
// $e^(M h 2^k)$ per cell and level, as `advect.wgsl` reads it. Only the top
// level -- one whole step -- is read here.
@group(2) @binding(2) var<storage, read> flows: array<mat4x4<f32>>;
@group(2) @binding(3) var<uniform> params: AdvectParams;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    // Position within the quad, in units of the radius: the Gaussian's own
    // coordinate.
    @location(0) offset: vec2<f32>,
    // The speck's speed as a fraction of the field's peak: the one scalar the
    // fragment tints and dims by.
    @location(1) speed: f32,
};

// The particle's ambient position: its cell's affine parametrization applied to
// the weights it carries, $p = sum_i lambda_i P_i$.
//
// This is the *only* place a particle acquires a position, and it happens after
// the advection is over. An unused weight of a sub-maximal cell is zero, so the
// sum runs to four regardless of the intrinsic dimension.
fn ambient_position(cell: u32, lambda: vec4<f32>) -> vec3<f32> {
    var position = vec3<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        position += lambda[i] * cell_positions[4u * cell + i].xyz;
    }
    return position;
}

// How far this particle travels in one step, in ambient space.
//
// The flow's own top level is $e^(M Delta t)$, so this is the exact chord of the
// next step rather than a difference quotient: the speck is shaded by where the
// field actually takes it. A crossing within the step is not accounted for --
// the neighbour's $M$ is a different matrix -- which understates the motion only
// near a facet, at the scale of one speck.
fn step_displacement(particle: Particle) -> vec3<f32> {
    let flow = flows[particle.cell * (params.depth + 1u) + params.depth];
    let next = flow * particle.lambda;
    return ambient_position(particle.cell, next) - ambient_position(particle.cell, particle.lambda);
}

// The speck's own frame: a screen-facing quad, elongated along its motion.
//
// This is motion blur, not a trail. A speck that moves several radii in a frame
// and is drawn round reads as a jumping dot; drawn stretched along its own
// chord it reads as the direction it is going, which is the whole of what the
// eye is here to see. The stretch is derived from the same displacement the
// tint uses, so the two cannot disagree about which way the field points.
fn speck_corner(
    center: vec3<f32>,
    displacement: vec3<f32>,
    radius: f32,
    stretch: f32,
    vertex_index: u32,
) -> vec3<f32> {
    let view_vec = frame.eye.xyz - center;
    let view_dir = view_vec / max(length(view_vec), 1e-6);

    // The motion's component in the billboard plane. A speck moving straight at
    // the camera has none, and is round -- which is correct: it covers no
    // screen distance to blur along.
    let motion = displacement - view_dir * dot(displacement, view_dir);
    let motion_len = length(motion);

    var along = vec3<f32>(0.0);
    var across = vec3<f32>(0.0);
    if motion_len > 1e-9 {
        along = motion / motion_len;
        across = cross(view_dir, along);
    } else {
        // Still, or moving at the camera: any screen-facing frame will do.
        var hint = vec3<f32>(0.0, 1.0, 0.0);
        if abs(view_dir.y) > 0.999 {
            hint = vec3<f32>(1.0, 0.0, 0.0);
        }
        along = normalize(cross(hint, view_dir));
        across = cross(view_dir, along);
    }

    let u = select(-1.0, 1.0, billboard_is_b(vertex_index));
    let v = billboard_side(vertex_index);
    return center + along * (u * radius * stretch) + across * (v * radius);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance: u32) -> VsOut {
    let particle = particles[instance];
    let center = ambient_position(particle.cell, particle.lambda);
    let displacement = step_displacement(particle);
    let speed = clamp(length(displacement) / max(material.speed_scale, 1e-12), 0.0, 1.0);

    // A slow speck is smaller as well as dimmer. Both, rather than brightness
    // alone: at the size these are drawn, area reads as intensity anyway, and
    // letting the two agree keeps a near-critical point from reading as a
    // uniform haze of full-sized dim dots.
    let radius = material.radius_world * mix(0.55, 1.0, speed);
    let stretch = 1.0 + material.stretch * speed;
    let corner = speck_corner(center, displacement, radius, stretch, vertex_index);
    // Particles ride *on* the surface they advect over, so they are coplanar
    // with the fill and would z-fight it. The same world-space bias the
    // wireframe uses, for the same reason and at the mark's own scale.
    let biased = depth_biased_corner(corner, frame.eye.xyz, material.radius_world);

    var out: VsOut;
    out.clip = frame.view_proj * vec4<f32>(biased, 1.0);
    out.offset = vec2<f32>(
        select(-1.0, 1.0, billboard_is_b(vertex_index)),
        billboard_side(vertex_index),
    );
    out.speed = speed;
    return out;
}

// A Gaussian speck, not a disc: a hard-edged circle aliases at this size and a
// texture would only blur it at a fixed resolution. The falloff is procedural,
// so it is exact at any zoom and costs no fetch.
const FALLOFF: f32 = 3.5;

// The ink of the slowest particles, relative to the material's own: a cool, dim
// ember, where the material's color is the fast end. Speed therefore runs cool
// and dark to warm and bright -- a *luminance* ramp, monotone, so the specks
// separate from the viridis fill beneath at every value, which is exactly the
// property the preamble's own note on that map turns on. Hue is the surface's
// business; the particles carry motion.
const SLOW_INK: vec3<f32> = vec3<f32>(0.22, 0.35, 0.72);

struct FsOut {
    @location(0) color: vec4<f32>,
    // The one mark that overflows: additive blending is what lets a filament
    // of a hundred overlapping specks carry a hundred times one speck's light,
    // so this is where the tone curve's question -- is there anything above 1
    // to reconcile -- is actually live. See `display_transform`'s note on
    // `unbounded_mask`.
    //
    // Written as this speck's own coverage, not a flat 1: the mask target
    // blends with `Max`, so a texel a hundred specks pile onto still reads as
    // exactly the coverage of the one that covers it most, and the speck's own
    // antialiased edge crosses into the surface beneath it smoothly rather
    // than at a hard ring.
    @location(1) unbounded: f32,
};

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let r2 = dot(in.offset, in.offset);
    let gaussian = exp(-FALLOFF * r2);
    let ink = mix(SLOW_INK, material.color.rgb, in.speed);
    // Brightness rises faster than the speed itself, so the fast filaments read
    // as the structure and the slow bulk as the medium they run through. With
    // additive blending an overlapped filament goes past white on its own.
    let intensity = material.color.a * mix(0.25, 1.0, in.speed * in.speed);
    // Additive: the blend state multiplies by alpha and sums, so overlapping
    // specks accumulate into density rather than occluding one another, and no
    // depth sort is needed over the whole population.
    let alpha = gaussian * intensity;
    var out: FsOut;
    out.color = vec4<f32>(ink, alpha);
    out.unbounded = alpha;
    return out;
}
