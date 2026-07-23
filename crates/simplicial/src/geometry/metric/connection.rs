//! The Levi-Civita connection of a Regge manifold: parallel transport across
//! the facets, and the hinge holonomy that is its curvature.
//!
//! A connection is what identifies the tangent spaces at two points, which the
//! smooth structure alone does not: $T_p M$ and $T_q M$ are different vector
//! spaces with no canonical isomorphism between them. Given a metric there is
//! exactly one connection that is compatible with it (transport preserves the
//! inner product) and torsion-free -- the **Levi-Civita** connection -- and on
//! a Regge manifold it is not extra data at all, but a function of the squared
//! edge lengths.
//!
//! Piecewise flatness collapses the general machinery into finite linear
//! algebra, with no ODE and no Christoffel symbol anywhere:
//!
//! - **Inside a cell there is nothing to transport.** A cell is flat and
//!   affine, so its chart's frame identifies all its tangent spaces at once,
//!   exactly as in $RR^n$. The connection has no interior degrees of freedom.
//! - **All of the content sits on the facets.** Two cells sharing a facet
//!   $sigma$ are two frames, and there is a unique isometry carrying one to the
//!   other that fixes $sigma$ and puts the cells on opposite sides of it: the
//!   *unfolding* of the pair into one flat frame. That is [`Transport`], and
//!   the whole connection is one such map per interior facet.
//! - **Curvature lives on the hinges, and only there.** A loop in the dual
//!   graph that avoids the codimension-2 skeleton bounds a region that unfolds
//!   flat, so its transports telescope to the identity. What survives is the
//!   holonomy around a *hinge* (a [`Ridge`]), an isometry fixing the hinge's
//!   own tangent space and rotating the $2$-plane normal to it by the **deficit
//!   angle** $epsilon_h = 2 pi - sum_(K supset h) theta_(K,h)$.
//!
//! Curvature is a $2$-form, so it is measured against area and never against a
//! length or a volume; a distribution supported on an $(n-2)$-simplex is
//! precisely a curvature $2$-form concentrated transversally to it. That is why
//! the hinges are the ridges, and it is why the deficit angle is the whole of
//! Regge curvature: vertices in 2D, edges in 3D, triangles in 4D, one mechanism.
//!
//! # Why the connection exists at all
//!
//! Two cells are not free to disagree arbitrarily on the facet they share: each
//! reads that facet's metric off the *same* squared edge lengths, so the two
//! restrictions agree ([`MeshLengthsSq::simplex_metric`] consults no containing
//! cell). Conformity of the geometry is what makes the gluing well posed, and a
//! bag of unrelated per-cell metrics would admit no connection at all. This is
//! the payoff of edge lengths being the primitive rather than
//! [`CellGramians`](super::CellGramians).
//!
//! # Frames, and what a transport matrix is
//!
//! Everything is expressed in each chart's own local cartesian frame -- the
//! basis $e_i = v_i - v_0$ of the cell's spanning vectors, in which
//! [`MeshLengthsSq::cell_metric`] is the Gramian. These frames are not
//! orthonormal, deliberately: orthonormalizing each cell would be an arbitrary
//! gauge choice per cell, whereas the local frame is canonical given the
//! chart. So a transport is not literally a matrix of $O(p,q)$ but an isometry
//! *between* two inner-product spaces,
//!
//! $ T^transpose g_(K') T = g_K, $
//!
//! which is the same statement without a choice in it. A holonomy, whose source
//! and target are one chart, does land in the isometry group $O(g_K)$ of that
//! chart -- conjugate to $O(p,q)$, with the conjugation being exactly the gauge
//! that was not fixed.

use super::mesh::MeshLengthsSq;
use crate::{
  Dim,
  atlas::{Chart, ChartExt, ref_face_spanning_vectors, ref_vertices},
  topology::{
    handle::SimplexIdx,
    role::{Cell, Ridge, roles},
  },
};

use crate::linalg::{Matrix, Vector};
use gramian::Metric;
use multiindex::Combination;

/// Below this the $g$-norm of a would-be normal counts as null, and the
/// transverse direction fails to be a direction: the facet's induced metric is
/// degenerate and no isometry across it exists.
const NULL_EPS: f64 = 1e-12;

/// Parallel transport between two charts: the linear isometry identifying the
/// frame of one cell with the frame of another.
///
/// $ T_(K' arrow.l K): (RR^n, g_K) -> (RR^n, g_(K')), quad
///   T^transpose g_(K') T = g_K $
///
/// A vector of the manifold expressed in the source chart's frame, expressed
/// instead in the target's. Transports compose along a path
/// ([`then`](Self::then)) and invert along its reversal, so they are a functor
/// from the path groupoid of the dual graph into the isometries -- which is
/// the whole content of a connection.
#[derive(Debug, Clone)]
pub struct Transport {
  source: SimplexIdx,
  target: SimplexIdx,
  matrix: Matrix,
}

impl Transport {
  /// The transport of a chart to itself: the identity, since a cell is flat
  /// and its frame already identifies all of its tangent spaces.
  pub fn identity(chart: Chart) -> Self {
    let dim = chart.dim().index();
    Self {
      source: chart.idx(),
      target: chart.idx(),
      matrix: Matrix::identity(dim, dim),
    }
  }

  pub fn source(&self) -> SimplexIdx {
    self.source
  }
  pub fn target(&self) -> SimplexIdx {
    self.target
  }
  pub fn dim(&self) -> Dim {
    self.source.dim()
  }
  /// The matrix of the transport, in the local frames of the two charts.
  pub fn matrix(&self) -> &Matrix {
    &self.matrix
  }
  pub fn into_matrix(self) -> Matrix {
    self.matrix
  }
  /// Whether source and target are the same chart.
  pub fn is_identity(&self) -> bool {
    self.source == self.target
  }

  /// The same vector, in the target chart's frame.
  pub fn apply(&self, vector: &Vector) -> Vector {
    &self.matrix * vector
  }

  /// The reverse transport, which is the inverse: transport along the reversed
  /// path. An isometry is invertible, so this is total.
  pub fn inverse(&self) -> Self {
    Self {
      source: self.target,
      target: self.source,
      matrix: self
        .matrix
        .clone()
        .try_inverse()
        .expect("a transport is an isometry, hence invertible"),
    }
  }

  /// Transport along the concatenation of two paths, $T_"next" compose T$.
  ///
  /// Panics if the paths do not meet: this one's target must be the next
  /// one's source.
  pub fn then(&self, next: &Self) -> Self {
    assert_eq!(
      self.target, next.source,
      "Transports compose only along a connected path."
    );
    Self {
      source: self.source,
      target: next.target,
      matrix: &next.matrix * &self.matrix,
    }
  }
}

/// The Levi-Civita connection, read off the Regge primitive.
impl MeshLengthsSq {
  /// Parallel transport from one chart into an adjacent one, across the facet
  /// they share.
  ///
  /// The unique linear isometry $(RR^n, g_K) -> (RR^n, g_(K'))$ that restricts
  /// on the shared facet to the change of frame [`Transition::differential`]
  /// and carries the direction *out of* the source to the direction *into* the
  /// target -- that is, the unfolding of the two cells into one flat frame,
  /// rather than the folding of one onto the other. The tangential condition
  /// fixes it on an $(n-1)$-dimensional subspace, isometry fixes the remaining
  /// direction up to sign, and the side condition picks the sign.
  ///
  /// Identity when source and target are the same chart. `None` when the two
  /// are not adjacent, and `None` on the two degeneracies that make the
  /// isometry not exist: a facet whose induced metric is degenerate (its normal
  /// direction is null, so there is nothing to normalize), and two cells whose
  /// metrics disagree in signature across the facet.
  ///
  /// [`Transition::differential`]: crate::atlas::Transition::differential
  pub fn transport(&self, source: Chart, target: Chart) -> Option<Transport> {
    if source == target {
      return Some(Transport::identity(source));
    }
    // A 0-manifold has no facets, hence no adjacency; the total accessor is
    // what says so, rather than a test on the dimension.
    source.complex().role_skeleton::<roles::Facet>()?;

    let dim = source.dim();
    let shared = source.facets().find(|facet| {
      let (a, b) = facet.adjacent_cells();
      a == target || b == Some(target)
    })?;

    let source_positions = shared.simplex().relative_to(source.simplex());
    let target_positions = shared.simplex().relative_to(target.simplex());

    // The tangent space of the facet, in each frame. The two are related by the
    // transition differential, which *is* the change of frame there -- the
    // metric-free half of the transport, already carried by the atlas.
    let source_tangents = ref_face_spanning_vectors(dim, &source_positions);
    let target_tangents = source.transition_to(target).differential() * &source_tangents;

    // Leaving the source through the facet is entering the target through it.
    let source_metric = self.cell_metric(source);
    let target_metric = self.cell_metric(target);
    let leaving = outward_normal(&source_metric, dim, &source_positions)?;
    let entering = -outward_normal(&target_metric, dim, &target_positions)?;

    // An isometry cannot exist if the transverse direction is spacelike on one
    // side and timelike on the other.
    if source_metric.norm_sq(&leaving).signum() != target_metric.norm_sq(&entering).signum() {
      return None;
    }

    let from = append_column(&source_tangents, &leaving);
    let to = append_column(&target_tangents, &entering);
    Some(Transport {
      source: source.idx(),
      target: target.idx(),
      matrix: to * from.try_inverse()?,
    })
  }

  /// Parallel transport along a path of charts, each adjacent to the next.
  ///
  /// The ordered product of the facet transports. `None` on an empty path, or
  /// wherever a consecutive pair is not adjacent.
  pub fn transport_along(&self, path: &[Chart]) -> Option<Transport> {
    let mut transport = Transport::identity(*path.first()?);
    for step in path.windows(2) {
      transport = transport.then(&self.transport(step[0], step[1])?);
    }
    Some(transport)
  }

  /// The holonomy around a hinge: transport once around the fan of cells
  /// meeting the ridge, back into the chart it started in.
  ///
  /// This is the curvature of the Regge manifold, in its integral form. It
  /// fixes the hinge's own tangent space pointwise and acts on the $2$-plane
  /// normal to it as a rotation by the deficit angle -- so it is the identity
  /// exactly where the manifold is flat, and away from the hinges every loop
  /// contracts and there is nothing else it could be.
  ///
  /// `None` on a boundary hinge, whose fan is open rather than closed and
  /// around which there is no loop to transport along.
  pub fn holonomy(&self, hinge: Ridge) -> Option<Transport> {
    if hinge.is_boundary() {
      return None;
    }
    let fan = hinge.fan();
    let closed: Vec<Chart> = fan.iter().copied().chain(std::iter::once(fan[0])).collect();
    self.transport_along(&closed)
  }

  /// The rotation angle of the hinge holonomy, in $[0, pi]$.
  ///
  /// Read off the trace: the holonomy fixes the $(n-2)$-dimensional tangent
  /// space of the hinge and rotates its normal plane, so
  /// $tr H = (n - 2) + 2 cos epsilon$. This is the magnitude of the deficit
  /// angle and not its sign, the sign being an orientation of the normal plane
  /// that the trace cannot see; [`Self::deficit_angle`] carries the signed
  /// value.
  ///
  /// Meaningful on a Riemannian geometry, where the normal plane is definite
  /// and the holonomy is a genuine rotation. On an indefinite signature the
  /// holonomy may be a boost, whose invariant is a rapidity rather than an
  /// angle, and the returned value is then not it.
  pub fn holonomy_angle(&self, hinge: Ridge) -> Option<f64> {
    let holonomy = self.holonomy(hinge)?;
    let dim = holonomy.dim().index() as f64;
    let cos = 0.5 * (holonomy.matrix().trace() - (dim - 2.0));
    Some(cos.clamp(-1.0, 1.0).acos())
  }

  /// The interior dihedral angle a cell subtends at a hinge: the angle between
  /// the two facets of the cell that contain the ridge.
  ///
  /// $ cos theta_(K,h) = - angle.l hat(n)_1, hat(n)_2 angle.r_g $
  ///
  /// with $hat(n)_i$ the outward unit normals of those two facets, both read in
  /// the cell's own frame from its Regge metric. In 2D, where a hinge is a
  /// vertex, this is the corner angle of the triangle there.
  ///
  /// `None` when either facet's normal is null. Like
  /// [`Self::holonomy_angle`], an angle is the Riemannian reading; on an
  /// indefinite signature the quantity is a Lorentzian dihedral angle and the
  /// arccosine is not it.
  pub fn dihedral_angle(&self, cell: Cell, hinge: Ridge) -> Option<f64> {
    let dim = cell.dim();
    let metric = self.cell_metric(cell);
    let mut normals = cell
      .facets()
      .filter(|facet| hinge.simplex().is_subsimplex_of(facet.simplex()))
      .map(|facet| {
        let positions = facet.simplex().relative_to(cell.simplex());
        outward_normal(&metric, dim, &positions)
      });
    let first = normals.next().expect("a cell has two facets at a hinge")?;
    let second = normals.next().expect("a cell has two facets at a hinge")?;
    Some((-metric.inner(&first, &second)).clamp(-1.0, 1.0).acos())
  }

  /// The deficit angle at a hinge: Regge curvature, as a scalar.
  ///
  /// $ epsilon_h = 2 pi - sum_(K supset h) theta_(K,h) $
  ///
  /// The shortfall by which the cells around the hinge fail to close up flat --
  /// zero exactly when they do. On a boundary hinge the fan is open and the
  /// closing target is $pi$ rather than $2 pi$, which folds the boundary's own
  /// extrinsic bending into the same scalar rather than tracking it apart.
  ///
  /// Unlike [`Self::holonomy`] this needs no ordering of the fan: a sum is
  /// commutative where a product of transports is not. It is signed, where
  /// the holonomy's trace is not, and the two agree in magnitude.
  pub fn deficit_angle(&self, hinge: Ridge) -> Option<f64> {
    let target = if hinge.is_boundary() {
      std::f64::consts::PI
    } else {
      std::f64::consts::TAU
    };
    let sum: f64 = hinge
      .get()
      .cells()
      .map(|cell| self.dihedral_angle(cell, hinge))
      .sum::<Option<f64>>()?;
    Some(target - sum)
  }
}

/// The $g$-unit normal of a facet within a cell, signed to point out of it.
///
/// The normal direction is the $g$-orthogonal complement of the facet's tangent
/// space, which is one-dimensional and non-degenerate whenever the facet's
/// induced metric is; `None` is the null case, where there is no unit vector to
/// be had. Outwardness is the sign of the pairing with the vector to the
/// opposite vertex, which is non-zero precisely because that vector is
/// transverse.
fn outward_normal(metric: &Metric, dim: Dim, positions: &Combination) -> Option<Vector> {
  let tangents = ref_face_spanning_vectors(dim, positions);
  let normal = unit_normal(metric, &tangents)?;

  let opposite = (0..=dim.index())
    .find(|&p| !positions.contains(p))
    .expect("a facet omits exactly one vertex of its cell");
  let vertices = ref_vertices(dim);
  let inward = vertices.column(opposite) - vertices.column(positions.index_at(0));

  Some(if metric.inner(&normal, &inward) > 0.0 {
    -normal
  } else {
    normal
  })
}

/// A vector spanning the $g$-orthogonal complement of the columns of
/// `tangents`, normalized to $g$-norm $plus.minus 1$.
///
/// The complement is the kernel of $B^transpose g$, obtained as the eigenvector
/// of least eigenvalue of the normal equations -- which stays an $n times n$
/// eigenproblem in every dimension, the empty tangent space of a $1$-manifold
/// included. `None` when the direction is null and admits no normalization.
fn unit_normal(metric: &Metric, tangents: &Matrix) -> Option<Vector> {
  let conditions = tangents.transpose() * metric.vector_gramian().matrix();
  let normal_equations = conditions.transpose() * &conditions;

  let eigen = na::SymmetricEigen::new(normal_equations);
  let least = (0..eigen.eigenvalues.len())
    .min_by(|&a, &b| {
      eigen.eigenvalues[a]
        .abs()
        .partial_cmp(&eigen.eigenvalues[b].abs())
        .unwrap()
    })
    .expect("a cell of a manifold has at least one direction");
  let normal = eigen.eigenvectors.column(least).into_owned();

  let norm_sq = metric.norm_sq(&normal);
  (norm_sq.abs() > NULL_EPS).then(|| normal / norm_sq.abs().sqrt())
}

/// A matrix with one further column appended on the right.
fn append_column(matrix: &Matrix, column: &Vector) -> Matrix {
  let mut extended = matrix.clone().insert_column(matrix.ncols(), 0.0);
  extended.set_column(matrix.ncols(), column);
  extended
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{
    Dim,
    geometry::coord::simplex::SimplexRefExt,
    mesher::cartesian::CartesianGrid,
    topology::{complex::Complex, role::roles},
  };

  use approx::assert_relative_eq;

  fn adjacent_pairs(complex: &Complex) -> Vec<(Chart<'_>, Chart<'_>)> {
    complex
      .facets()
      .handle_iter()
      .filter_map(|facet| {
        let (a, b) = facet.adjacent_cells();
        b.map(|b| (a, b))
      })
      .collect()
  }

  /// Metric compatibility: $T^transpose g_(K') T = g_K$. Transport is an
  /// isometry between the two frames, which is what makes it Levi-Civita
  /// rather than merely a change of basis, and is the condition that pins the
  /// transverse direction up to sign.
  #[test]
  fn transport_is_an_isometry() {
    for dim in (1..=3usize).map(Dim::from) {
      let (complex, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&complex);

      for (source, target) in adjacent_pairs(&complex) {
        let transport = lengths.transport(source, target).unwrap();
        let pullback = lengths
          .cell_metric(target)
          .vector_gramian()
          .pullback(transport.matrix());
        assert_relative_eq!(
          pullback.matrix(),
          lengths.cell_metric(source).vector_gramian().matrix(),
          epsilon = 1e-9
        );
      }
    }
  }

  /// The two cells induce the *same* metric on the facet they share -- they
  /// read it off the same edge lengths -- which is why an isometry across the
  /// facet exists at all, and why the connection is a function of the Regge
  /// primitive and of nothing else.
  #[test]
  fn shared_facet_metric_agrees() {
    for dim in (1..=3usize).map(Dim::from) {
      let (complex, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&complex);

      for facet in complex.facets().handle_iter() {
        let (a, b) = facet.adjacent_cells();
        let Some(b) = b else { continue };
        let facet_metric = |cell: Chart| {
          let positions = facet.simplex().relative_to(cell.simplex());
          let tangents = ref_face_spanning_vectors(cell.dim(), &positions);
          lengths
            .cell_metric(cell)
            .vector_gramian()
            .pullback(&tangents)
        };
        assert_relative_eq!(
          facet_metric(a).matrix(),
          facet_metric(b).matrix(),
          epsilon = 1e-12
        );
      }
    }
  }

  /// Against an embedding: on a mesh realized in $RR^n$ the unfolding of two
  /// adjacent cells is the identity of the ambient space, so the transport in
  /// the local frames must be $A_(K')^(-1) A_K$.
  ///
  /// This is the theorem that pins the construction, sign and all -- a
  /// reflection would satisfy the isometry law just as well and fail here --
  /// and it is a *test* rather than the definition precisely because the
  /// definition may not consult an embedding.
  #[test]
  fn transport_unfolds_an_embedded_mesh() {
    for dim in (1..=3usize).map(Dim::from) {
      let (complex, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&complex);

      for (source, target) in adjacent_pairs(&complex) {
        let source_frame = source.coord_simplex(&coords).linear_transform();
        let target_frame = target.coord_simplex(&coords).linear_transform();
        let expected = target_frame.try_inverse().unwrap() * source_frame;

        assert_relative_eq!(
          lengths.transport(source, target).unwrap().matrix(),
          &expected,
          epsilon = 1e-9
        );
      }
    }
  }

  /// Transport reverses along the reversed path, and a chart transports to
  /// itself as the identity: the path groupoid, discretely.
  #[test]
  fn transport_is_functorial() {
    for dim in (1..=3usize).map(Dim::from) {
      let (complex, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&complex);

      for (source, target) in adjacent_pairs(&complex) {
        let there = lengths.transport(source, target).unwrap();
        let back = lengths.transport(target, source).unwrap();
        assert_relative_eq!(back.matrix(), there.inverse().matrix(), epsilon = 1e-9);

        let roundtrip = there.then(&back);
        assert!(roundtrip.is_identity());
        assert_relative_eq!(
          roundtrip.matrix(),
          &Matrix::identity(dim.index(), dim.index()),
          epsilon = 1e-9
        );
      }
    }
  }

  /// A flat manifold has trivial holonomy around every hinge: the cells of a
  /// triangulated box unfold into one frame, so every loop of the dual graph
  /// telescopes. The base case of curvature, and the one every mesh with a
  /// coordinate realization must pass.
  #[test]
  fn flat_mesh_has_no_holonomy() {
    for dim in (2..=3usize).map(Dim::from) {
      let (complex, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&complex);

      let ridges = complex.role_skeleton::<roles::Ridge>().unwrap();
      for hinge in ridges.handle_iter() {
        let Some(holonomy) = lengths.holonomy(hinge) else {
          continue;
        };
        assert_relative_eq!(
          holonomy.matrix(),
          &Matrix::identity(dim.index(), dim.index()),
          epsilon = 1e-8
        );
        assert_relative_eq!(lengths.holonomy_angle(hinge).unwrap(), 0.0, epsilon = 1e-6);
        assert_relative_eq!(lengths.deficit_angle(hinge).unwrap(), 0.0, epsilon = 1e-9);
      }
    }
  }

  /// The two readings of Regge curvature agree: the rotation angle of the
  /// holonomy is the magnitude of the deficit angle. One is an ordered product
  /// of isometries around the fan, the other a commutative sum of dihedral
  /// angles, and they are the same number.
  #[test]
  fn holonomy_angle_is_the_deficit_angle() {
    let (complex, coords) = crate::mesher::sphere::mesh_sphere_surface(2);
    let lengths = coords.to_edge_lengths_sq(&complex);

    let ridges = complex.role_skeleton::<roles::Ridge>().unwrap();
    let mut curved = 0;
    for hinge in ridges.handle_iter() {
      let Some(angle) = lengths.holonomy_angle(hinge) else {
        continue;
      };
      let deficit = lengths.deficit_angle(hinge).unwrap();
      assert_relative_eq!(angle, deficit.abs(), epsilon = 1e-8);
      if deficit.abs() > 1e-6 {
        curved += 1;
      }
    }
    assert!(curved > 0, "a sphere is not flat");
  }

  /// Gauss-Bonnet on a closed surface: $sum_h epsilon_h = 2 pi chi$, the
  /// deficit angles summing to the Euler characteristic with no area, no
  /// refinement limit and no tolerance around a smooth quantity. On a sphere
  /// $chi = 2$.
  ///
  /// The theorem that says the deficit angle *is* curvature rather than
  /// merely resembling it.
  #[test]
  fn deficit_angles_sum_to_the_euler_characteristic() {
    for refinement in 1..=3 {
      let (complex, coords) = crate::mesher::sphere::mesh_sphere_surface(refinement);
      let lengths = coords.to_edge_lengths_sq(&complex);

      let ridges = complex.role_skeleton::<roles::Ridge>().unwrap();
      let total: f64 = ridges
        .handle_iter()
        .map(|hinge| lengths.deficit_angle(hinge).unwrap())
        .sum();

      let euler: i64 = (0..=complex.dim().index())
        .map(|k| {
          let n = complex.skeleton(k).len() as i64;
          if k % 2 == 0 { n } else { -n }
        })
        .sum();
      assert_relative_eq!(total, std::f64::consts::TAU * euler as f64, epsilon = 1e-9);
    }
  }

  /// In 2D a hinge is a vertex and a dihedral angle is a corner angle: the
  /// general construction reproduces the elementary one it generalizes.
  #[test]
  fn dihedral_angle_in_2d_is_the_corner_angle() {
    let (complex, coords) = CartesianGrid::new_unit(Dim::new(2), 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&complex);

    for cell in complex.cells().handle_iter() {
      let metric = lengths.cell_metric(cell);
      for (position, &vertex) in cell.simplex().vertices.iter().enumerate() {
        let hinge = crate::topology::handle::SimplexIdx::new(Dim::ZERO, vertex)
          .handle(&complex)
          .role::<roles::Ridge>();
        let (a, b) = ((position + 1) % 3, (position + 2) % 3);
        assert_relative_eq!(
          lengths.dihedral_angle(cell, hinge).unwrap(),
          metric.vector_gramian().vertex_angle(position, a, b),
          epsilon = 1e-9
        );
      }
    }
  }
}
