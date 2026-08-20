//! The piecewise-affine atlas of the simplicial manifold.
//!
//! The cells of the complex are an atlas: each cell is a chart, and the
//! transition maps between overlapping cells are the affine gluings of the
//! shared faces ([`Transition`]). The atlas is therefore piecewise affine
//!, a statement about the maps, which needs no metric. (Give it one, and the
//! simplicial manifold it presents is piecewise flat: curvature vanishes on
//! the cell interiors and concentrates on the codimension-2 hinges. That is a
//! statement about the geometry, and it is not this module's business.)
//!
//! A [`Chart`] is a cell, top-dimensional by construction, since a face
//! carries no chart. A point of the simplicial manifold is thus intrinsically a
//! pair $(K, lambda)$ of a chart and the barycentric coordinates within it, a
//! [`MeshPoint`]. Integration over a cell is quadrature over its chart, whose
//! nodes are such points ([`SimplexQuadRule`]).
//!
//! The chart's own structure, the reference vertices, the barycentric
//! differentials, the volume, depends on the dimension alone and not on the
//! cell: every chart of the atlas is the same chart up to the labeling of its
//! vertices. That is why the `unit_*` functions below take a [`Dim`] and no cell,
//! and it is why any per-cell quantity fixed by the reference chart is computed
//! once on the reference cell and reused on every cell of the mesh. What differs
//! between charts is the labeling, and the labeling is exactly what a
//! [`Transition`] is made of.
//!
//! Barycentric is the right chart: it is symmetric in the vertices, affine, and
//! needs neither a metric nor an embedding. Everything in this module is pure
//! affine combinatorics of the unit simplex, which is why it sits below
//! both the coordinate (extrinsic) and the metric layer, not inside either.
//!
//! The atlas determines the bundles too, and not only the points. A chart
//! identifies its cell with the reference simplex, so the tangent space at a
//! point is $RR^n$ in that chart's frame and the exterior powers over it are
//! fibers of a bundle fixed by the atlas alone ([`bundle`]). That is why the
//! trace onto a face and a face's tangent blade live here rather than wherever
//! a geometry does: neither asks for one.
//!
//! # The two coordinate systems of a chart
//!
//! A chart carries two coordinate systems, related by dropping the redundant
//! zeroth weight:
//!
//! - [`Bary`]: barycentric $lambda in RR^(n+1)$ with $sum_i lambda_i = 1$, the
//!   symmetric one;
//! - [`Local`]: cartesian $x in RR^n$ with $x_i = lambda_(i+1)$ and
//!   $lambda_0 = 1 - sum_i x_i$, the one in which the reference frame, and
//!   hence the value of a section, is expressed.
//!
//! Both are distinct from the [`Ambient`](coorder::Ambient) coordinates of
//! an embedding, and the [`coorder::CoordSpace`] tags keep the
//! three from being confused for one another.

pub mod bundle;
pub mod chart;
pub mod point;
pub mod quadrature;
pub mod refine;
pub mod simplex_coords;
pub mod transition;

pub use bundle::{FaceTrace, face_tangent_blade};
pub use chart::{Chart, ChartExt};
pub use point::{BARY_EPS, MeshPoint};
pub use quadrature::SimplexQuadRule;
pub use refine::{UnitRefinement, unit_refinement};
pub use simplex_coords::SimplexCoords;
pub use transition::Transition;

use crate::Dim;

use crate::linalg::{Matrix, RowVector, Vector};
use coorder::{CoordSpace, Coords, CoordsRef};
use multiindex::{Combination, Composition, factorial_f64};

/// The barycentric coordinate space of a chart: the affine weights
/// $lambda in RR^(n+1)$, $sum_i lambda_i = 1$.
pub enum Barycentric {}
impl CoordSpace for Barycentric {
  const NAME: &'static str = "bary";
}

/// The cartesian coordinate space of a chart: $x in RR^n$, the reference frame
/// in which the value of a section at a [`MeshPoint`] is expressed.
pub enum LocalCartesian {}
impl CoordSpace for LocalCartesian {
  const NAME: &'static str = "local";
}

/// Barycentric coordinates within a cell: $n+1$ affine weights summing to one.
pub type Bary = Coords<Barycentric>;
pub type BaryRef<'a> = CoordsRef<'a, Barycentric>;

/// Local (cartesian) coordinates within a cell chart.
pub type Local = Coords<LocalCartesian>;
pub type LocalRef<'a> = CoordsRef<'a, LocalCartesian>;

/// The volume of the unit $n$-simplex, $1 \/ n!$.
///
/// A property of the chart, not of the geometry: it is the factor by which a
/// chart integral scales, and the metric enters only through the further factor
/// $sqrt(abs(det g))$ (see `cell_volume`).
pub fn unit_simplex_volume(dim: impl Into<Dim>) -> f64 {
  let dim = dim.into();
  factorial_f64(dim.index()).recip()
}

pub fn bary2local<'a>(bary: impl Into<BaryRef<'a>>) -> Local {
  Local::new(bary.into().view().rows_range(1..).into_owned())
}
pub fn local2bary<'a>(local: impl Into<LocalRef<'a>>) -> Bary {
  let local = local.into();
  let bary0 = 1.0 - local.view().sum();
  Bary::new(local.view().insert_row(0, bary0))
}

/// Whether the barycentric weights lie in the closed reference cell, rather
/// than in the affine extension of the chart beyond it.
///
/// The weights sum to one: that is what makes them barycentric, so the
/// closed cell is cut out by their nonnegativity alone, and the upper bound
/// $lambda_i <= 1$ is implied rather than tested. Nonnegativity is tested up to
/// [`BARY_EPS`], because the weights vanishing on a face are only ever
/// floating-point zero, and a point of a face is a point of the cell.
pub fn is_bary_inside<'a>(bary: impl Into<BaryRef<'a>>) -> bool {
  let bary = bary.into();
  debug_assert!(
    approx::relative_eq!(bary.view().sum(), 1.0, epsilon = 1e-9),
    "Barycentric weights must sum to one."
  );
  bary.view().iter().all(|&b| b >= -BARY_EPS)
}

pub fn barycenter_bary(dim: impl Into<Dim>) -> Bary {
  let dim = dim.into();
  Bary::from_element((dim + 1).index(), ((dim + 1).index() as f64).recip())
}
pub fn barycenter_local(dim: impl Into<Dim>) -> Local {
  let dim = dim.into();
  Local::from_element(dim.index(), ((dim + 1).index() as f64).recip())
}

/// The $i$-th barycentric coordinate function evaluated in local coordinates.
pub fn unit_bary<'a>(ivertex: usize, local: impl Into<LocalRef<'a>>) -> f64 {
  let local = local.into();
  assert!(ivertex <= local.dim());
  if ivertex == 0 {
    1.0 - local.view().sum()
  } else {
    local[ivertex - 1]
  }
}

/// The differential $dif lambda_i$ of a barycentric coordinate function, a
/// constant covector in the reference frame.
pub fn unit_difbary(dim: impl Into<Dim>, ivertex: usize) -> RowVector {
  let dim = dim.into();
  assert!(ivertex <= dim);
  if ivertex == 0 {
    RowVector::from_element(dim.index(), -1.0)
  } else {
    let mut v = RowVector::zeros(dim.index());
    v[ivertex - 1] = 1.0;
    v
  }
}

/// The differential of the barycentric coordinate map
/// $lambda: RR^n -> RR^(n+1)$ of the unit simplex: the rows are the
/// constant covectors $dif lambda_i$.
///
/// Metric-free, and the same for every cell, any form built from the
/// barycentric differentials is therefore constant on the cell and evaluable
/// intrinsically, with no geometry at all.
pub fn unit_difbarys(dim: impl Into<Dim>) -> Matrix {
  let dim = dim.into();
  let mut difbarys = Matrix::zeros((dim + 1).index(), dim.index());
  difbarys.row_mut(0).fill(-1.0);
  for i in 0..dim.index() {
    difbarys[(i + 1, i)] = 1.0;
  }
  difbarys
}

/// The Gramian of the barycentric coordinate functions in $L^2(hat(K))$, at
/// unit volume:
/// $Q_(v w) = 1 \/ vol(hat(K)) integral_(hat(K)) lambda_v lambda_w
/// = (1 + delta_(v w)) \/ ((n+1)(n+2))$.
///
/// Identity plus rank one, $Q = (I + bb(1) bb(1)^top) \/ ((n+1)(n+2))$, hence
/// positive definite. Metric-free like every reference datum: this is the whole
/// polynomial content of the affine chart, the factor a mass matrix carries
/// alongside the inner product on $Lambda^k$, and the one a higher-order space
/// replaces by the barycentric moments of higher degree.
///
/// A bare matrix, not a metric: the $L^2$ inner product of the barycentric
/// functions is a datum of the reference chart, and this crate never learns
/// what a metric is.
pub fn unit_bary_gramian(dim: impl Into<Dim>) -> Matrix {
  let nvertices = (dim.into() + 1).index();
  let scale = ((nvertices * (nvertices + 1)) as f64).recip();
  let mut gramian = Matrix::from_element(nvertices, nvertices, scale);
  gramian.fill_diagonal(2.0 * scale);
  gramian
}

/// The local coordinates of the vertices of the unit $n$-simplex, as the
/// columns of $[0 | I_n]$: the origin and the standard basis.
pub fn unit_vertices(dim: impl Into<Dim>) -> Matrix {
  let dim = dim.into();
  let mut vertices = Matrix::zeros(dim.index(), (dim + 1).index());
  for i in 0..dim.index() {
    vertices[(i, i + 1)] = 1.0;
  }
  vertices
}

/// The barycentric lattice of the unit $n$-simplex at refinement $R$: the
/// weights whose parts are whole multiples of $1 \/ R$, as integer numerators.
///
/// $ L_R^n = { k in NN_0^(n+1) : sum_i k_i = R }, quad lambda = k \/ R $
///
/// The compositions of $R$ into $n + 1$ parts, hence $binom(R + n, n)$ points,
/// in the colex order of [`Composition::all`](multiindex::Composition::all). The integers are the primitive and
/// the weights the wrapper ([`unit_lattice_bary`]): a lattice point is an exact
/// combinatorial object, and the two properties below are identities on the
/// integers that would only be approximate equalities on the weights.
///
/// Affine, hence metric-free and embedding-free, hence a function of [`Dim`]
/// alone, every chart of the atlas carries the same lattice, and a cell's
/// share of it is uniform in the chart no matter the cell's size or shape. It
/// is not uniform in any metric, and on a manifold with no global coordinates
/// there is nothing else for "uniform" to mean.
///
/// Two properties are what make it worth having:
///
/// - It closes on the faces. A point with $k_i = 0$ lies on the face
///   opposite vertex $i$, and the sub-lattice there is $L_R^(n-1)$ at the same
///   $R$. Two cells sharing a facet therefore agree on the lattice points of it
///   up to the vertex labeling, which is exactly a [`Transition`], so the
///   agreement is combinatorial and needs no spatial tolerance.
/// - It extends [`unit_vertices`]. $R = 1$ is the vertex set, in the same
///   order; $R$ refines it from there. $R = 0$ is not a refinement and admits no
///   point ($lambda = k \/ 0$), so $R >= 1$. The barycenter is a lattice point
///   only when $(n+1) | R$.
pub fn unit_lattice(dim: impl Into<Dim>, refinement: usize) -> impl Iterator<Item = Vec<usize>> {
  let dim = dim.into();
  assert!(
    refinement >= 1,
    "A lattice needs a refinement of at least one."
  );
  Composition::all((dim + 1).index(), refinement).map(Composition::into_parts)
}

/// The lattice points strictly inside the reference cell: $k_i >= 1$ for every
/// $i$, so the point lies on no face.
///
/// $ mono(L)_R^n = { k in L_R^n : k > 0 } = 1 + L_(R - n - 1)^n $
///
/// The shift is the enumeration, an interior point is an arbitrary point
/// with one unit already spent on each part, so there are $binom(R - 1, n)$ of
/// them, and no separate combinatorics. $R = n + 1$ spends every unit and leaves
/// the barycenter alone; below that the interior is empty, which is the honest
/// answer rather than an error: a refinement too coarse to have an inside has
/// none.
///
/// This, not [`unit_lattice`], is what a per-cell sample set wants, and for a
/// mathematical reason rather than to dodge the double-count on a shared facet:
/// a section is only chart-independent in its tangential part, so at a point
/// of a facet the two incident charts genuinely disagree and the value there is
/// not the cell's to report. The open cell is where a section has a value at
/// all.
pub fn unit_lattice_interior(
  dim: impl Into<Dim>,
  refinement: usize,
) -> impl Iterator<Item = Vec<usize>> {
  let dim = dim.into();
  refinement
    .checked_sub((dim + 1).index())
    .into_iter()
    .flat_map(move |rest| Composition::all((dim + 1).index(), rest))
    .map(|k| k.parts().iter().map(|k| k + 1).collect())
}

/// A lattice point's barycentric weights, $lambda = k \/ R$: the one passage
/// from the integer primitive to the affine one.
///
/// The numerators must sum to the refinement, which is what makes the weights
/// affine, and is checked under `debug_assertions`.
pub fn lattice_bary(numerators: &[usize], refinement: usize) -> Bary {
  debug_assert_eq!(
    numerators.iter().sum::<usize>(),
    refinement,
    "a lattice point's parts sum to the refinement"
  );
  let scale = (refinement as f64).recip();
  Bary::new(Vector::from_iterator(
    numerators.len(),
    numerators.iter().map(|&k| k as f64 * scale),
  ))
}

/// [`unit_lattice_interior`] as barycentric weights.
pub fn unit_lattice_interior_bary(
  dim: impl Into<Dim>,
  refinement: usize,
) -> impl Iterator<Item = Bary> {
  unit_lattice_interior(dim, refinement).map(move |k| lattice_bary(&k, refinement))
}

/// [`unit_lattice`] as barycentric weights, $lambda = k \/ R$.
pub fn unit_lattice_bary(dim: impl Into<Dim>, refinement: usize) -> impl Iterator<Item = Bary> {
  unit_lattice(dim, refinement).map(move |k| lattice_bary(&k, refinement))
}

/// The spanning vectors $v_i = e_(p_i) - e_(p_0)$ of a face of the reference
/// cell, in the cell's reference frame, as the columns of an $n times k$ matrix.
///
/// Pure affine combinatorics of the local vertex positions: the face of a cell
/// needs no coordinates of its own, and on a manifold without an embedding
/// there are none to be had.
pub fn unit_face_spanning_vectors(cell_dim: impl Into<Dim>, positions: &Combination) -> Matrix {
  let cell_dim = cell_dim.into();
  let vertices = unit_vertices(cell_dim);
  let base = vertices.column(positions.index_at(0));
  let mut spanning = Matrix::zeros(cell_dim.index(), positions.card() - 1);
  for (i, position) in positions.iter().skip(1).enumerate() {
    spanning.set_column(i, &(vertices.column(position) - base));
  }
  spanning
}

/// The barycentric coordinates, within a cell, of a point given by its
/// barycentric coordinates on a face: scatter the face's weights onto the local
/// vertex positions of the face, zero elsewhere.
pub fn face_bary_to_cell_bary<'a>(
  cell_dim: impl Into<Dim>,
  positions: &Combination,
  face_bary: impl Into<BaryRef<'a>>,
) -> Bary {
  let cell_dim = cell_dim.into();
  let face_bary = face_bary.into();
  assert_eq!(
    face_bary.dim(),
    positions.card(),
    "Wrong number of weights."
  );
  let mut bary = Vector::zeros((cell_dim + 1).index());
  for (i, position) in positions.iter().enumerate() {
    bary[position] = face_bary[i];
  }
  Bary::new(bary)
}
