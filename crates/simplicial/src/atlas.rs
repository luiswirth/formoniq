//! The piecewise-affine atlas of the simplicial manifold.
//!
//! The cells of the complex are an atlas: each cell is a chart, and the
//! transition maps between overlapping cells are the affine gluings of the
//! shared faces ([`Transition`]). The atlas is therefore **piecewise affine**
//! -- a statement about the maps, which needs no metric. (Give it one, and the
//! simplicial manifold it presents is piecewise *flat*: curvature vanishes on
//! the cell interiors and concentrates on the codimension-2 hinges. That is a
//! statement about the geometry, and it is not this module's business.)
//!
//! A [`Chart`] *is* a cell -- top-dimensional by construction, since a face
//! carries no chart. A point of the simplicial manifold is thus intrinsically a
//! pair $(K, lambda)$ of a chart and the barycentric coordinates within it -- a
//! [`MeshPoint`]. Integration over a cell is quadrature over its chart, whose
//! nodes are such points ([`SimplexQuadRule`]).
//!
//! The chart's own structure -- the reference vertices, the barycentric
//! differentials, the volume -- depends on the dimension alone and not on the
//! cell: every chart of the atlas is the *same* chart up to the labelling of its
//! vertices. That is why the `ref_*` functions below take a [`Dim`] and no cell,
//! and it is why the element matrices of FEEC are computed once on the reference
//! cell and reused on every cell of the mesh. What differs between charts is the
//! labelling, and the labelling is exactly what a [`Transition`] is made of.
//!
//! Barycentric is the right chart: it is symmetric in the vertices, affine, and
//! needs neither a metric nor an embedding. Everything in this module is pure
//! affine combinatorics of the reference simplex, which is why it sits below
//! both the coordinate (extrinsic) and the metric layer, not inside either.
//!
//! # The two coordinate systems of a chart
//!
//! A chart carries two coordinate systems, related by dropping the redundant
//! zeroth weight:
//!
//! - [`Bary`]: barycentric $lambda in RR^(n+1)$ with $sum_i lambda_i = 1$, the
//!   symmetric one;
//! - [`Local`]: cartesian $x in RR^n$ with $x_i = lambda_(i+1)$ and
//!   $lambda_0 = 1 - sum_i x_i$, the one in which the reference frame -- and
//!   hence the value of a section -- is expressed.
//!
//! Both are distinct from the [`Ambient`](coorder::Ambient) coordinates of
//! an embedding, and the [`coorder::CoordSpace`] tags keep the
//! three from being confused for one another.

pub mod chart;
pub mod point;
pub mod quadrature;
pub mod transition;

pub use chart::{Chart, ChartExt};
pub use point::{MeshPoint, BARY_EPS};
pub use quadrature::SimplexQuadRule;
pub use transition::Transition;

use crate::Dim;

use crate::linalg::{Matrix, RowVector, Vector};
use coorder::{CoordSpace, Coords, CoordsRef};
use multiindex::{compositions, factorial_f64, Combination};

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

/// The volume of the reference $n$-simplex, $1 \/ n!$.
///
/// A property of the chart, not of the geometry: it is the factor by which a
/// chart integral scales, and the metric enters only through the further factor
/// $sqrt(det g)$ (see [`cell_volume`](crate::geometry::cell_volume)).
pub fn refsimp_vol(dim: Dim) -> f64 {
  factorial_f64(dim).recip()
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
/// The weights sum to one -- that is what makes them barycentric -- so the
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

pub fn barycenter_bary(dim: Dim) -> Bary {
  Bary::from_element(dim + 1, ((dim + 1) as f64).recip())
}
pub fn barycenter_local(dim: Dim) -> Local {
  Local::from_element(dim, ((dim + 1) as f64).recip())
}

/// The $i$-th barycentric coordinate function evaluated in local coordinates.
pub fn ref_bary<'a>(ivertex: usize, local: impl Into<LocalRef<'a>>) -> f64 {
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
pub fn ref_difbary(dim: Dim, ivertex: usize) -> RowVector {
  assert!(ivertex <= dim);
  if ivertex == 0 {
    RowVector::from_element(dim, -1.0)
  } else {
    let mut v = RowVector::zeros(dim);
    v[ivertex - 1] = 1.0;
    v
  }
}

/// The differential of the barycentric coordinate map
/// $lambda: RR^n -> RR^(n+1)$ of the reference simplex: the rows are the
/// constant covectors $dif lambda_i$.
///
/// Metric-free, and the same for every cell -- this is the whole reason the
/// Whitney forms can be evaluated intrinsically, with no geometry at all.
pub fn ref_difbarys(dim: Dim) -> Matrix {
  let mut difbarys = Matrix::zeros(dim + 1, dim);
  difbarys.row_mut(0).fill(-1.0);
  for i in 0..dim {
    difbarys[(i + 1, i)] = 1.0;
  }
  difbarys
}

/// The local coordinates of the vertices of the reference $n$-simplex, as the
/// columns of $[0 | I_n]$: the origin and the standard basis.
pub fn ref_vertices(dim: Dim) -> Matrix {
  let mut vertices = Matrix::zeros(dim, dim + 1);
  for i in 0..dim {
    vertices[(i, i + 1)] = 1.0;
  }
  vertices
}

/// The barycentric lattice of the reference $n$-simplex at refinement $R$: the
/// weights whose parts are whole multiples of $1 \/ R$, as integer numerators.
///
/// $ L_R^n = { k in NN_0^(n+1) : sum_i k_i = R }, quad lambda = k \/ R $
///
/// The compositions of $R$ into $n + 1$ parts, hence $binom(R + n, n)$ points,
/// in the colex order of [`compositions`]. The integers are the primitive and
/// the weights the wrapper ([`ref_lattice_bary`]): a lattice point is an exact
/// combinatorial object, and the two properties below are identities on the
/// integers that would only be approximate equalities on the weights.
///
/// Affine, hence metric-free and embedding-free, hence a function of [`Dim`]
/// alone -- every chart of the atlas carries the *same* lattice, and a cell's
/// share of it is uniform in the chart no matter the cell's size or shape. It
/// is not uniform in any metric, and on a manifold with no global coordinates
/// there is nothing else for "uniform" to mean.
///
/// Two properties are what make it worth having:
///
/// - **It closes on the faces.** A point with $k_i = 0$ lies on the face
///   opposite vertex $i$, and the sub-lattice there *is* $L_R^(n-1)$ at the same
///   $R$. Two cells sharing a facet therefore agree on the lattice points of it
///   up to the vertex labelling -- which is exactly a [`Transition`] -- so the
///   agreement is combinatorial and needs no spatial tolerance.
/// - **It extends [`ref_vertices`].** $R = 1$ *is* the vertex set, in the same
///   order; $R$ refines it from there. $R = 0$ is not a refinement and admits no
///   point ($lambda = k \/ 0$), so $R >= 1$. The barycenter is a lattice point
///   only when $(n+1) | R$.
pub fn ref_lattice(dim: Dim, refinement: usize) -> impl Iterator<Item = Vec<usize>> {
  assert!(
    refinement >= 1,
    "A lattice needs a refinement of at least one."
  );
  compositions(dim + 1, refinement)
}

/// The lattice points strictly inside the reference cell: $k_i >= 1$ for every
/// $i$, so the point lies on no face.
///
/// $ mono(L)_R^n = { k in L_R^n : k > 0 } = 1 + L_(R - n - 1)^n $
///
/// The shift *is* the enumeration -- an interior point is an arbitrary point
/// with one unit already spent on each part -- so there are $binom(R - 1, n)$ of
/// them, and no separate combinatorics. $R = n + 1$ spends every unit and leaves
/// the barycenter alone; below that the interior is empty, which is the honest
/// answer rather than an error: a refinement too coarse to have an inside has
/// none.
///
/// This, not [`ref_lattice`], is what a per-cell sample set wants, and for a
/// mathematical reason rather than to dodge the double-count on a shared facet:
/// a section is only chart-independent in its *tangential* part, so at a point
/// of a facet the two incident charts genuinely disagree and the value there is
/// not the cell's to report. The open cell is where a section has a value at
/// all.
pub fn ref_lattice_interior(dim: Dim, refinement: usize) -> impl Iterator<Item = Vec<usize>> {
  refinement
    .checked_sub(dim + 1)
    .into_iter()
    .flat_map(move |rest| compositions(dim + 1, rest))
    .map(|k| k.into_iter().map(|k| k + 1).collect())
}

/// [`ref_lattice_interior`] as barycentric weights.
pub fn ref_lattice_interior_bary(dim: Dim, refinement: usize) -> impl Iterator<Item = Bary> {
  let scale = (refinement as f64).recip();
  ref_lattice_interior(dim, refinement).map(move |k| {
    Bary::new(Vector::from_iterator(
      k.len(),
      k.into_iter().map(|k| k as f64 * scale),
    ))
  })
}

/// [`ref_lattice`] as barycentric weights, $lambda = k \/ R$.
pub fn ref_lattice_bary(dim: Dim, refinement: usize) -> impl Iterator<Item = Bary> {
  let scale = (refinement as f64).recip();
  ref_lattice(dim, refinement).map(move |k| {
    Bary::new(Vector::from_iterator(
      k.len(),
      k.into_iter().map(|k| k as f64 * scale),
    ))
  })
}

/// The spanning vectors $v_i = e_(p_i) - e_(p_0)$ of a face of the reference
/// cell, in the cell's reference frame, as the columns of an $n times k$ matrix.
///
/// Pure affine combinatorics of the local vertex positions: the face of a cell
/// needs no coordinates of its own, and on a manifold without an embedding
/// there are none to be had.
pub fn ref_face_spanning_vectors(cell_dim: Dim, positions: &Combination) -> Matrix {
  let vertices = ref_vertices(cell_dim);
  let base = vertices.column(positions.index_at(0));
  let mut spanning = Matrix::zeros(cell_dim, positions.card() - 1);
  for (i, position) in positions.iter().skip(1).enumerate() {
    spanning.set_column(i, &(vertices.column(position) - base));
  }
  spanning
}

/// The barycentric coordinates, within a cell, of a point given by its
/// barycentric coordinates on a face: scatter the face's weights onto the local
/// vertex positions of the face, zero elsewhere.
pub fn face_bary_to_cell_bary<'a>(
  cell_dim: Dim,
  positions: &Combination,
  face_bary: impl Into<BaryRef<'a>>,
) -> Bary {
  let face_bary = face_bary.into();
  assert_eq!(
    face_bary.dim(),
    positions.card(),
    "Wrong number of weights."
  );
  let mut bary = Vector::zeros(cell_dim + 1);
  for (i, position) in positions.iter().enumerate() {
    bary[position] = face_bary[i];
  }
  Bary::new(bary)
}

#[cfg(test)]
mod test {
  use super::*;

  use approx::assert_relative_eq;
  use multiindex::binomial;

  /// The lattice has $binom(R + n, n)$ points, each a composition of $R$.
  #[test]
  fn lattice_is_a_composition_set() {
    for dim in 0..=4 {
      for refinement in 1..=5 {
        let lattice: Vec<_> = ref_lattice(dim, refinement).collect();
        assert_eq!(lattice.len(), binomial(refinement + dim, dim));
        for point in &lattice {
          assert_eq!(point.len(), dim + 1);
          assert_eq!(point.iter().sum::<usize>(), refinement);
        }
        let unique: std::collections::HashSet<_> = lattice.iter().collect();
        assert_eq!(unique.len(), lattice.len());
      }
    }
  }

  /// $L_1^n$ *is* the vertex set of the reference cell, in the order
  /// [`ref_vertices`] places it: the lattice extends the vertices rather than
  /// merely containing them.
  #[test]
  fn lattice_at_refinement_one_is_the_vertices() {
    for dim in 0..=4 {
      let vertices = ref_vertices(dim);
      for (ivertex, bary) in ref_lattice_bary(dim, 1).enumerate() {
        assert_relative_eq!(bary2local(&bary).view(), &vertices.column(ivertex));
      }
    }
  }

  /// The weights are barycentric, and every lattice point is a point of the
  /// closed cell.
  #[test]
  fn lattice_bary_lies_in_the_cell() {
    for dim in 0..=4 {
      for refinement in 1..=5 {
        for bary in ref_lattice_bary(dim, refinement) {
          assert_relative_eq!(bary.view().sum(), 1.0);
          assert!(is_bary_inside(&bary));
        }
      }
    }
  }

  /// The lattice closes on the faces: the points vanishing on vertex `i` are,
  /// with that weight dropped, exactly the facet's own lattice at the same $R$.
  /// This is what lets two cells agree on a shared facet combinatorially.
  #[test]
  fn lattice_restricts_to_the_facet_lattice() {
    for dim in 1..=4 {
      for refinement in 1..=5 {
        let facet: std::collections::HashSet<_> = ref_lattice(dim - 1, refinement).collect();
        for ivertex in 0..=dim {
          let restricted: std::collections::HashSet<_> = ref_lattice(dim, refinement)
            .filter(|k| k[ivertex] == 0)
            .map(|mut k| {
              k.remove(ivertex);
              k
            })
            .collect();
          assert_eq!(restricted, facet);
        }
      }
    }
  }

  /// The interior is exactly the lattice minus every face: $binom(R-1, n)$
  /// points, each on no face, and none of the boundary ones missed.
  #[test]
  fn lattice_interior_is_the_lattice_off_the_faces() {
    for dim in 0..=4 {
      for refinement in 1..=6 {
        let interior: Vec<_> = ref_lattice_interior(dim, refinement).collect();
        let expected: Vec<_> = ref_lattice(dim, refinement)
          .filter(|k| k.iter().all(|&k| k >= 1))
          .collect();
        assert_eq!(interior, expected);
        assert_eq!(
          interior.len(),
          refinement.checked_sub(1).map_or(0, |r| binomial(r, dim))
        );
      }
    }
  }

  /// The base case of the interior: $R = n + 1$ spends one unit on each part and
  /// leaves the barycenter alone, and below that there is no inside to have.
  #[test]
  fn lattice_interior_bottoms_out_at_the_barycenter() {
    for dim in 0..=4 {
      for refinement in 0..=dim {
        assert_eq!(ref_lattice_interior(dim, refinement).count(), 0);
      }
      let base: Vec<_> = ref_lattice_interior_bary(dim, dim + 1).collect();
      assert_eq!(base.len(), 1);
      assert_relative_eq!(base[0].view(), barycenter_bary(dim).view());
    }
  }

  /// The two chart coordinate systems are mutually inverse.
  #[test]
  fn bary_local_roundtrip() {
    for dim in 0..=4 {
      let local = Local::from_iterator(dim, (0..dim).map(|i| 0.1 * (i + 1) as f64));
      let bary = local2bary(&local);
      assert_relative_eq!(bary.sum(), 1.0, epsilon = 1e-12);
      assert_relative_eq!(bary2local(&bary).vector(), local.vector(), epsilon = 1e-12);
    }
  }

  /// The rows of the barycentric differential are the individual $dif lambda_i$,
  /// and they sum to zero: $sum_i lambda_i = 1$ is constant.
  #[test]
  fn ref_difbarys_rows_are_difbary_and_sum_to_zero() {
    for dim in 0..=4 {
      let difbarys = ref_difbarys(dim);
      for ivertex in 0..=dim {
        assert_relative_eq!(
          difbarys.row(ivertex).into_owned(),
          ref_difbary(dim, ivertex)
        );
      }
      assert_relative_eq!(difbarys.row_sum(), RowVector::zeros(dim));
    }
  }

  /// The barycentric coordinate functions are dual to the reference vertices,
  /// $lambda_i (e_j) = delta_(i j)$, and the differentials are their gradients.
  #[test]
  fn ref_barys_are_dual_to_ref_vertices() {
    for dim in 0..=4 {
      let vertices = ref_vertices(dim);
      for (j, vertex) in vertices.column_iter().enumerate() {
        let local = Local::new(vertex.into_owned());
        for i in 0..=dim {
          let expected = f64::from(i == j);
          assert_relative_eq!(ref_bary(i, &local), expected, epsilon = 1e-12);
        }
      }
    }
  }

  /// A face's spanning vectors are the differences of the reference vertices it
  /// selects, and a point of the face has the face's weights on those positions.
  #[test]
  fn face_bary_scatters_onto_the_face() {
    let cell_dim = 3;
    for face_dim in 0..=cell_dim {
      for positions in multiindex::combinations(cell_dim + 1, face_dim + 1) {
        let face_bary = barycenter_bary(face_dim);
        let bary = face_bary_to_cell_bary(cell_dim, &positions, &face_bary);

        assert_relative_eq!(bary.sum(), 1.0, epsilon = 1e-12);
        for i in 0..=cell_dim {
          let on_face = positions.iter().any(|p| p == i);
          assert_eq!(bary[i] != 0.0, on_face);
        }

        // The face's barycenter, expressed in the cell chart, is the mean of the
        // reference vertices the face selects.
        let vertices = ref_vertices(cell_dim);
        let spanning = ref_face_spanning_vectors(cell_dim, &positions);
        assert_eq!(spanning.ncols(), face_dim);
        let mean = positions
          .iter()
          .map(|p| vertices.column(p).into_owned())
          .sum::<Vector>()
          / (face_dim + 1) as f64;
        assert_relative_eq!(bary2local(&bary).vector(), &mean, epsilon = 1e-12);
      }
    }
  }
}
