//! The bake behind the particle advection: everything `advect.wgsl` needs to
//! flow a point of the simplicial manifold along a grade-1 field, and nothing
//! it does not.
//!
//! All of the exterior calculus lives here, in `f64`, on the CPU. What crosses
//! to the GPU is linear algebra on a simplex: a matrix per cell per dyadic
//! level, a permutation matrix per facet, a neighbour table, and a list of
//! points to be born at. The shader has no Whitney basis, no metric and no
//! integrator, and that asymmetry is the design -- the flow is *exact* at every
//! level, so there is nothing left for the shader to approximate except the
//! instant a curve leaves its cell.
//!
//! **The generator.** Inside a cell the sharped reduced field $V$ is affine, so
//! it is the barycentric interpolation of its own values at the cell's
//! vertices: $V = sum_i lambda_i V_i$, i.e. $V = hat(V) lambda$ with $hat(V)$
//! the $n times (n+1)$ matrix of those values in the chart's local frame. The
//! weights move with it, $dot(lambda) = (dif lambda \/ dif x) V$, and
//! $dif lambda \/ dif x$ is the reference chart's own constant
//! [`ref_difbarys`]. So the whole per-cell dynamic is the linear generator
//!
//! $ M = (dif lambda \/ dif x) hat(V), quad dot(lambda) = M lambda. $
//!
//! Its columns sum to zero, because $bb(1)^T dif lambda \/ dif x = 0$ -- which
//! is the statement that the flow preserves $sum_i lambda_i = 1$, and hence
//! that a particle stays a point of the manifold rather than drifting off the
//! affine hull it lives on.
//!
//! **The levels are one exponential and then squaring.** A frame's step is
//! $2^d$ ticks of $h = Delta t \/ 2^d$, and the shader reaches any whole number
//! of ticks by multiplying the levels its binary expansion names. Those levels
//! are $e^(M h 2^k) = (e^(M h))^(2^k)$, so the bake exponentiates *once* per
//! cell and squares $d$ times. The nesting is exact and free.
//!
//! **The padding is a dimension, not a filler.** Every matrix is emitted
//! $4 times 4$ and every weight vector `vec4`, so one buffer serves an
//! intrinsic dimension up to the ambient's own $3$: a triangle uses the leading
//! $3 times 3$ block, a tetrahedron all of it. The trailing block is *zero*,
//! never identity -- which is why the exponential is taken on the honest
//! $(n+1) times (n+1)$ generator and padded afterwards. Padding first would
//! exponentiate a zero block into a one, and an unused weight would spring to
//! life.

use common::{gramian::RiemannianMetric, linalg::nalgebra::Matrix};
use ddf::{cochain::Cochain, whitney::interpolant::WhitneyInterpolant};
use manifold::{
  atlas::{local2bary, ref_difbarys, ref_vertices, ChartExt, Local, MeshPoint},
  geometry::{coord::mesh::MeshCoords, metric::geometry::Geometry},
  topology::{complex::Complex, handle::SimplexIdx, simplex::Simplex},
  Dim,
};

use crate::scene::reduced_form;

/// The ambient dimension's own bound on the intrinsic one, and hence on the
/// barycentric weights: a `vec4` holds the weights of any cell an observer in
/// $RR^3$ can be inside of.
const MAX_VERTICES: usize = 4;

/// A facet with no cell on the other side: the manifold has boundary there.
/// Mirrors `NO_NEIGHBOUR` in `advect.wgsl`.
pub const NO_NEIGHBOUR: u32 = u32::MAX;

/// A $4 times 4$ matrix in WGSL's column-major order.
pub type Mat4 = [f32; 16];

/// Where a particle is born: a cell and its barycentric weights there.
#[derive(Clone, Copy, Debug)]
pub struct Seed {
  pub cell: u32,
  pub bary: [f32; 4],
}

/// Everything `advect.wgsl` reads, in the order it indexes it.
pub struct AdvectBake {
  /// $d$: a frame's step is $2^d$ ticks, and there are $d + 1$ levels per cell.
  pub depth: u32,
  /// $e^(M h 2^k)$, indexed `cell * (depth + 1) + level`.
  pub flows: Vec<Mat4>,
  /// Per cell, the neighbour across the facet opposite each local vertex.
  pub neighbours: Vec<[u32; 4]>,
  /// The `Transition` relabellings, indexed `4 * cell + facet`.
  pub transitions: Vec<Mat4>,
  pub seeds: Vec<Seed>,
}

impl AdvectBake {
  /// Bake the field `W cochain` (reduced to grade 1) for a frame step of
  /// `step`, resolved to $2^"depth"$ ticks, with `seed_count` birth sites.
  ///
  /// `step` is in the field's own time: a particle in a region where
  /// $|V|_g = v$ travels $v dot "step"$ per frame. Turning that into something
  /// the object's own size makes sense of is the display's business, not this
  /// one's.
  pub fn new(
    topology: &Complex,
    coords: &MeshCoords,
    cochain: &Cochain,
    step: f64,
    depth: u32,
    seed_count: usize,
  ) -> Self {
    let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);
    let dim = topology.dim();
    let tick = step / f64::from(1u32 << depth);

    let mut flows = Vec::with_capacity(topology.cells().len() * (depth as usize + 1));
    let mut neighbours = Vec::with_capacity(topology.cells().len());
    let mut transitions = Vec::with_capacity(topology.cells().len() * MAX_VERTICES);

    for cell in topology.cells().handle_iter() {
      let metric = coords.cell_metric(cell);
      let generator = flow_generator(&interpolant, cell.idx(), &metric);

      // One exponential, then `depth` squarings: level $k$ is $(e^(M h))^(2^k)$
      // by the flow's own semigroup law, so the higher levels cost a multiply
      // each rather than an exponential each.
      let mut level = (&generator * tick).exp();
      for _ in 0..=depth {
        flows.push(pad_mat4(&level));
        level = &level * &level;
      }

      let mut cell_neighbours = [NO_NEIGHBOUR; MAX_VERTICES];
      for (opposite, slot) in cell_neighbours.iter_mut().enumerate().take(dim + 1) {
        let Some(neighbour) = facet_neighbour(topology, cell.idx(), opposite) else {
          transitions.push(pad_mat4(&Matrix::zeros(dim + 1, dim + 1)));
          continue;
        };
        *slot = neighbour.kidx() as u32;
        let transition = cell.chart().transition_to(neighbour.chart());
        transitions.push(pad_mat4(transition.bary_map()));
      }
      // A cell of sub-maximal dimension has fewer facets than the buffer has
      // slots; the missing ones are boundary, which is what they mean.
      for _ in (dim + 1)..MAX_VERTICES {
        transitions.push(pad_mat4(&Matrix::zeros(dim + 1, dim + 1)));
      }
      neighbours.push(cell_neighbours);
    }

    Self {
      depth,
      flows,
      neighbours,
      transitions,
      seeds: seeds(topology, coords, seed_count),
    }
  }
}

/// The field's peak magnitude $max |V|_g$ over the cell barycenters.
///
/// What a display divides by to turn "so many object-radii per second" into the
/// field time the bake wants: the field's own units are the cochain's, and the
/// only scale a viewer can read is the object's. A field that vanishes
/// everywhere gives zero, and the caller decides what a still field looks like.
///
/// Sampled at barycenters rather than integrated: the peak of an affine field on
/// a simplex is at a vertex, so this underestimates slightly -- which sets the
/// speed a hair low and never a hair high, the safe direction for a mark whose
/// whole legibility is that it is slow enough.
pub fn peak_speed(topology: &Complex, coords: &MeshCoords, cochain: &Cochain) -> f64 {
  let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);
  topology
    .cells()
    .handle_iter()
    .map(|cell| {
      let metric = coords.cell_metric(cell);
      let point = MeshPoint::barycenter(cell.idx());
      reduced_form(interpolant.eval(&point), &metric).norm(&metric)
    })
    .fold(0.0, f64::max)
}

/// The field's *mean* magnitude over the manifold, area-weighted: the
/// barycenter samples of [`peak_speed`], averaged by the cells' metric volume
/// instead of maximized.
///
/// What the deposit's ink calibration divides by: with splats inked by arc
/// length, the equilibrium trail brightness is set by the *average* speed of
/// the population, not the peak, and the average is the field's own
/// area-weighted mean -- an exact quantity of the field, not a tuned ratio.
pub fn mean_speed(topology: &Complex, coords: &MeshCoords, cochain: &Cochain) -> f64 {
  let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);
  let (mut weighted, mut total) = (0.0, 0.0);
  for cell in topology.cells().handle_iter() {
    let metric = coords.cell_metric(cell);
    let weight = metric.det_sqrt();
    let point = MeshPoint::barycenter(cell.idx());
    weighted += weight * reduced_form(interpolant.eval(&point), &metric).norm(&metric);
    total += weight;
  }
  if total > 0.0 {
    weighted / total
  } else {
    0.0
  }
}

/// The generator $M = (dif lambda \/ dif x) hat(V)$ of one cell.
///
/// $hat(V)$ is read off the *reference vertices*, which is not a sampling
/// choice: the sharped reduced field is affine on the cell, and an affine map
/// on a simplex is exactly the barycentric interpolation of its vertex values.
/// The $n + 1$ evaluations are the field, not an approximation of it.
fn flow_generator(
  interpolant: &WhitneyInterpolant,
  cell: SimplexIdx,
  metric: &RiemannianMetric,
) -> Matrix {
  let dim = cell.dim();
  let vertices = ref_vertices(dim);
  let mut vertex_field = Matrix::zeros(dim, dim + 1);
  for i in 0..=dim {
    let local = Local::new(vertices.column(i).into_owned());
    let point = MeshPoint::new(cell, local2bary(&local));
    let form = reduced_form(interpolant.eval(&point), metric);
    vertex_field.set_column(i, form.sharp(metric).coeffs());
  }
  ref_difbarys(dim) * vertex_field
}

/// The cell across the facet opposite local vertex `opposite`, if any.
fn facet_neighbour<'a>(
  topology: &'a Complex,
  cell: SimplexIdx,
  opposite: usize,
) -> Option<manifold::topology::handle::SimplexRef<'a>> {
  let dim = cell.dim();
  if dim == 0 {
    // A point has no facet to cross, so every direction is boundary. The
    // degenerate case answers itself.
    return None;
  }
  let handle = cell.handle(topology);
  let facet_vertices: Vec<usize> = handle
    .simplex()
    .vertices
    .iter()
    .enumerate()
    .filter(|&(i, _)| i != opposite)
    .map(|(_, &v)| v)
    .collect();
  let facet = topology
    .skeleton(dim - 1)
    .handle_by_simplex(&Simplex::new(facet_vertices));
  facet.cells().find(|c| c.idx() != cell)
}

/// `seed_count` birth sites, distributed by the cells' own metric volume.
///
/// The cell is chosen by stratified inversion of the cumulative volume, which
/// is deterministic and exactly proportional -- no rejection, no alias table,
/// no clock. The weight is $sqrt(det g)$ rather than the volume proper because
/// the $1 \/ n!$ every cell of one dimension shares cancels in the normalization.
///
/// Weighting by volume rather than by cell is what keeps the particle density a
/// property of the manifold: seeding per cell would put as many particles in a
/// sliver as in a large triangle, and the population would then be a picture of
/// the triangulation.
fn seeds(topology: &Complex, coords: &MeshCoords, seed_count: usize) -> Vec<Seed> {
  let dim = topology.dim();
  let mut cumulative = Vec::with_capacity(topology.cells().len());
  let mut total = 0.0;
  for cell in topology.cells().handle_iter() {
    total += coords.cell_metric(cell).det_sqrt();
    cumulative.push(total);
  }
  if total <= 0.0 || cumulative.is_empty() {
    return Vec::new();
  }

  (0..seed_count)
    .map(|i| {
      let target = (i as f64 + 0.5) / seed_count as f64 * total;
      let cell = cumulative
        .partition_point(|&c| c < target)
        .min(cumulative.len() - 1);
      Seed {
        cell: cell as u32,
        bary: uniform_bary(dim, i as u32),
      }
    })
    .collect()
}

/// A point drawn uniformly from the reference simplex, as barycentric weights.
///
/// The spacings of $n$ sorted uniforms on $[0, 1]$ are a `Dirichlet(1, ..., 1)`
/// draw, which is the uniform distribution on the simplex -- not the normalized
/// uniforms, which crowd the barycenter.
fn uniform_bary(dim: Dim, index: u32) -> [f32; 4] {
  let mut cuts: Vec<f64> = (0..dim)
    .map(|k| uniform_from_hash(index.wrapping_mul(0x9e37_79b9) ^ k as u32))
    .collect();
  cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());

  let mut bary = [0.0f32; 4];
  let mut previous = 0.0;
  for (i, &cut) in cuts.iter().enumerate() {
    bary[i] = (cut - previous) as f32;
    previous = cut;
  }
  bary[dim] = (1.0 - previous) as f32;
  bary
}

/// The same integer hash `advect.wgsl` uses (`lowbias32`), mapped into
/// $[0, 1)$: the seeding is reproducible, and reproducible by the same rule on
/// both sides.
fn uniform_from_hash(value: u32) -> f64 {
  let mut h = value;
  h ^= h >> 16;
  h = h.wrapping_mul(0x7feb_352d);
  h ^= h >> 15;
  h = h.wrapping_mul(0x846c_a68b);
  h ^= h >> 16;
  f64::from(h) / f64::from(u32::MAX)
}

/// A square matrix embedded in the leading block of a $4 times 4$, the rest
/// zero, in WGSL's column-major order.
fn pad_mat4(matrix: &Matrix) -> Mat4 {
  let n = matrix.nrows();
  let mut out = [0.0f32; 16];
  for column in 0..n {
    for row in 0..n {
      out[column * 4 + row] = matrix[(row, column)] as f32;
    }
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use manifold::geometry::coord::mesh::MeshCoords;

  /// The generator preserves $sum_i lambda_i = 1$: its columns sum to zero, so
  /// $bb(1)^T dot(lambda) = 0$ and a particle never leaves the affine hull its
  /// weights live on. This is the law the whole pass rests on -- if it failed,
  /// the flow would carry points off the manifold -- and it holds in every
  /// dimension the ambient admits.
  #[test]
  fn generator_columns_sum_to_zero() {
    for dim in 1..=3 {
      let topology = Complex::standard(dim);
      let coords = MeshCoords::standard(dim);
      let cochain = Cochain::constant(1.0, topology.skeleton_raw(1));
      let interpolant = WhitneyInterpolant::new(cochain, &topology);
      for cell in topology.cells().handle_iter() {
        let metric = coords.cell_metric(cell);
        let generator = flow_generator(&interpolant, cell.idx(), &metric);
        for column in 0..generator.ncols() {
          let sum: f64 = generator.column(column).sum();
          assert!(
            sum.abs() < 1e-10,
            "dim {dim}: column {column} sums to {sum}"
          );
        }
      }
    }
  }

  /// Uniform barycentric draws are barycentric: non-negative and summing to one,
  /// at every dimension the ambient reaches, with the unused slots left at zero.
  #[test]
  fn uniform_bary_is_barycentric() {
    for dim in 0..=3 {
      for index in 0..64 {
        let bary = uniform_bary(dim, index);
        let sum: f32 = bary.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "dim {dim}: weights sum to {sum}");
        assert!(bary.iter().all(|&w| w >= 0.0), "dim {dim}: negative weight");
        assert!(bary[dim + 1..].iter().all(|&w| w == 0.0));
      }
    }
  }
}
