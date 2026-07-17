// Lagrangian advection of a grade-1 field's integral curves, one thread per
// particle.
//
// A particle is a point of the simplicial manifold and nothing more: the cell
// it sits in, and its barycentric weights there. It never has an ambient
// position while it moves -- the draw pass computes one, this pass does not --
// and it never leaves the atlas, so there is no reprojection and no ambient
// point location anywhere in the loop.
//
// **No exterior calculus reaches this shader.** Within a cell the sharped
// Whitney field is affine, so the flow of $lambda$ is the *linear* system
// $dot(lambda) = M lambda$ with $M$ constant per cell, and at a fixed step its
// solution is exact: $lambda(t + h) = e^(M h) lambda(t)$. The bake evaluates the
// Whitney basis, applies the sharp and exponentiates, all in `f64` on the CPU
// where it is already tested. What crosses to the GPU is a matrix per cell per
// dyadic level, and every operation below is a mat-vec. There is no integrator
// here to carry an order or a truncation error.
//
// A facet crossing is the same operation again. Barycentric weights are
// non-negative exactly inside their own cell, so a negative component *is* the
// exit test -- exact, from the state already being integrated, with no search.
// The `Transition` into the neighbouring chart is a relabelling of those
// weights, hence a linear map, hence another mat-vec by a permutation matrix.
// Step and crossing are one instruction.
//
// **Time is an integer.** A frame's step is $2^d$ *ticks*, and the bake supplies
// $e^(M h 2^k)$ for every level $k <= d$ over the tick $h = Delta t \/ 2^d$, so
// any whole number of ticks is reachable as the product of the levels its binary
// expansion names. That is what makes the crossing time findable without a
// per-particle matrix exponential: the shader multiplies only matrices the bake
// already produced. The alternative -- Newton on the exit time, which converges
// far faster and whose derivative $dot(lambda) = M lambda$ is free -- needs
// $e^(M t)$ at an arbitrary $t$, and that is the one thing this split forbids.
//
// The exit time is not available in closed form. Weights sum to one for all
// time, so $bb(1)^T M = 0$ and $M$ carries a zero eigenvalue alongside those of
// the cell's own field matrix; a component of $lambda(t)$ is therefore
// $a + b e^(mu_1 t) + c e^(mu_2 t)$, an exponential polynomial in two distinct
// rates. Solvable when a rate degenerates, not in general -- and a closed-form
// branch for the degenerate cells would be a special case in exactly the sense
// the design rejects, with a per-particle eigendecomposition as its price.

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
// $e^(M h 2^k)$ for each cell and level, indexed `cell * (depth + 1) + level`.
@group(0) @binding(2) var<storage, read> flows: array<mat4x4<f32>>;
// The `Transition` matrices, indexed `4 * cell + facet`: the relabelling of
// barycentric weights into the chart across that facet.
@group(0) @binding(3) var<storage, read> transitions: array<mat4x4<f32>>;
// Where a particle is born. Sampled by the metric volume of the cells on the
// CPU, so the seeding density is a property of the manifold and not of its
// triangulation.
@group(0) @binding(4) var<storage, read> seeds: array<Particle>;
@group(0) @binding(5) var<uniform> params: AdvectParams;

const NO_NEIGHBOUR: u32 = 0xffffffffu;
const NO_FACET: u32 = 4u;
// A step spends its ticks over at most this many cells. A curve that would
// cross more has met a feature far finer than the step, and stopping short
// leaves it a tick behind rather than letting one thread spin the dispatch.
const MAX_CROSSINGS: u32 = 8u;

// Integer hash (`lowbias32`): the whole pass's randomness, so that randomness
// is a pure function of the particle and its epoch rather than of a clock.
fn hash_u32(value: u32) -> u32 {
    var h = value;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}

fn flow(cell: u32, level: u32) -> mat4x4<f32> {
    return flows[cell * (params.depth + 1u) + level];
}

fn inside(lambda: vec4<f32>) -> bool {
    return all(lambda >= vec4<f32>(0.0));
}

// The facet the step left through. Weights start the scan at zero, so the
// unused component of a sub-maximal dimension -- exactly zero, never negative
// -- cannot select a facet that does not exist.
//
// The *most* negative weight. Over a single tick that is the facet the curve
// left through; the bisection above is what keeps it a single tick, and hence
// what makes "most negative" and "first crossed" the same facet.
fn exit_facet(lambda: vec4<f32>) -> u32 {
    var worst = 0.0;
    var facet = NO_FACET;
    for (var i = 0u; i < 4u; i++) {
        if lambda[i] < worst {
            worst = lambda[i];
            facet = i;
        }
    }
    return facet;
}

struct Advance {
    lambda: vec4<f32>,
    ticks: u32,
};

// The longest whole-tick advance that leaves the particle inside its cell,
// together with the state there.
//
// The exit time is bracketed by its binary expansion, most significant level
// first: offer each level in turn, take it when the particle survives it, drop
// it when it does not. Descending, that is the binary search, and it terminates
// one tick short of the crossing -- so the state returned is still in the cell
// whose $M$ produced it, which is the invariant the whole pass rests on.
//
// The cost is one mat-vec per level, and the levels are the bake's, never an
// exponential taken here.
fn advance_within_cell(cell: u32, lambda: vec4<f32>, budget: u32) -> Advance {
    var reached = Advance(lambda, 0u);
    var level = params.depth + 1u;
    while level > 0u {
        level -= 1u;
        let span = 1u << level;
        if reached.ticks + span > budget {
            continue;
        }
        let trial = flow(cell, level) * reached.lambda;
        if inside(trial) {
            reached.lambda = trial;
            reached.ticks += span;
        }
    }
    return reached;
}

// The particle relabelled into the chart across `facet`.
//
// The overshoot is one tick, so clamping it away and renormalizing lands the
// particle on the facet it left through, which is where the transition is
// defined. That projection is the pass's only inexactness and it is $O(h)$: the
// flow inside a cell carries no error at all, and the tick is the bake's
// $Delta t \/ 2^d$.
fn cross_facet(particle: Particle, facet: u32) -> Particle {
    var crossed = particle;
    let clamped = max(particle.lambda, vec4<f32>(0.0));
    let on_facet = clamped / dot(clamped, vec4<f32>(1.0));
    crossed.lambda = transitions[4u * particle.cell + facet] * on_facet;
    crossed.cell = cells[particle.cell].neighbour[facet];
    return crossed;
}

fn respawn(id: u32, epoch: u32) -> Particle {
    let draw = hash_u32(id ^ hash_u32(epoch));
    var born = seeds[draw % params.seed_count];
    born.epoch = epoch + 1u;
    born.life = params.life_min + hash_u32(draw) % params.life_spread;
    return born;
}

// One frame's step, spending $2^d$ ticks across as many cells as the curve
// visits. Spending the remainder in the *new* cell is the point of counting
// ticks rather than crossings: a particle that stopped at each facet would lose
// the rest of its step there, and would therefore travel slower the more often
// it crossed -- an apparent speed that tracks the triangulation instead of the
// field.
//
// Respawning is not upkeep either. A field with divergence carries its
// particles into the sinks and leaves the rest of the manifold bare within
// seconds, so a finite life is what keeps the population a sample of the
// manifold rather than of the field's attractors.
@compute @workgroup_size(64)
fn advect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    if id >= params.particle_count {
        return;
    }

    var particle = particles[id];
    var ticks_left = 1u << params.depth;

    for (var crossing = 0u; crossing < MAX_CROSSINGS; crossing++) {
        let reached = advance_within_cell(particle.cell, particle.lambda, ticks_left);
        particle.lambda = reached.lambda;
        ticks_left -= reached.ticks;
        if ticks_left == 0u {
            break;
        }

        // The bisection stopped one tick short of the facet, so this tick is
        // the one that leaves the cell, and taking it is what names the facet.
        particle.lambda = flow(particle.cell, 0u) * particle.lambda;
        ticks_left -= 1u;
        particle = cross_facet(particle, exit_facet(particle.lambda));
        if particle.cell == NO_NEIGHBOUR {
            break;
        }
    }

    // Reaching the boundary ends a life early: the curve has left the manifold,
    // and there is no chart on the other side to continue it in.
    let escaped = particle.cell == NO_NEIGHBOUR;
    if escaped || particle.life == 0u {
        particle = respawn(id, particle.epoch);
    } else {
        particle.life -= 1u;
    }

    particles[id] = particle;
}
