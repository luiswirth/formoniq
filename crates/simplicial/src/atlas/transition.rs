//! The transition maps of the piecewise-affine atlas.
//!
//! Two cells are two charts, and they overlap in the face they share. On that
//! overlap the same point of the manifold has two representations, one per
//! chart, and the map relating them is the **transition map**
//! $psi_(K' K): hat(K) supset.eq sigma -> sigma subset.eq hat(K')$.
//!
//! It is the affine gluing of the shared face, and it is pure combinatorics: a
//! vertex of the mesh has one barycentric weight, and each chart merely lists
//! the vertices in a different place. So the transition is the $0\/1$ matrix
//! $P$ that relabels the weights,
//!
//! $lambda'_j = cases(lambda_i & "if the" j"-th vertex of" K' "is the" i"-th of" K, 0 & "otherwise")$
//!
//! and it is defined precisely where the weights it must discard vanish -- on
//! the shared face. Metric-free, coordinate-free, exact.
//!
//! This is what makes the atlas an atlas, and it is what the reference-frame
//! implementation of the de Rham map rests on: the integral of a form over a
//! face may be computed in *either* adjacent chart because the two answers are
//! related by $psi$, and the pairing with the face's tangent blade is invariant
//! under it.

use super::{ref_difbarys, Bary, Chart, MeshPoint, BARY_EPS};
use crate::{topology::handle::SimplexIdx, Dim};

use crate::linalg::{Matrix, Vector};
use multiindex::Combination;

/// The transition map between the charts of two cells, on their overlap.
///
/// Degenerate cases are not special cases: cells that share nothing give a
/// transition with an empty overlap, on which [`apply`](Self::apply) is nowhere
/// defined, and a cell with itself gives the identity.
#[derive(Debug, Clone)]
pub struct Transition {
  source: SimplexIdx,
  target: SimplexIdx,
  /// $P$: the $(n+1) times (n+1)$ relabelling of barycentric weights, with a
  /// zero row for each vertex only the target has and a zero column for each
  /// vertex only the source has.
  bary_map: Matrix,
}

impl Transition {
  /// The transition from `source` into `target`.
  ///
  /// That the two are charts -- and hence cells -- is the [`Chart`] type's
  /// business, not this one's. What remains to check is that they are charts of
  /// the *same* atlas.
  pub fn new(source: Chart, target: Chart) -> Self {
    assert!(
      std::ptr::eq(source.complex(), target.complex()),
      "Charts of two different atlases have no transition."
    );
    let dim = source.dim();

    let source_vertices = &source.cell().simplex().vertices;
    let target_vertices = &target.cell().simplex().vertices;

    let mut bary_map = Matrix::zeros(dim + 1, dim + 1);
    for (j, vertex) in target_vertices.iter().enumerate() {
      if let Ok(i) = source_vertices.binary_search(vertex) {
        bary_map[(j, i)] = 1.0;
      }
    }

    Self {
      source: source.idx(),
      target: target.idx(),
      bary_map,
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

  /// $P$: the relabelling of the barycentric weights.
  pub fn bary_map(&self) -> &Matrix {
    &self.bary_map
  }

  /// The local vertex positions, in the *source* chart, of the vertices shared
  /// with the target: the overlap of the two charts, as a face of the source.
  pub fn overlap_positions(&self) -> Combination {
    Combination::from_increasing((0..=self.dim()).filter(|&i| self.bary_map.column(i).sum() != 0.0))
  }

  /// Whether the transition is the identity: source and target are the same
  /// chart.
  pub fn is_identity(&self) -> bool {
    self.source == self.target
  }

  /// The reverse transition $psi_(K K')$, which is the inverse of this one on
  /// the overlap.
  pub fn inverse(&self) -> Self {
    Self {
      source: self.target,
      target: self.source,
      bary_map: self.bary_map.transpose(),
    }
  }

  /// The same point of the manifold, in the target chart.
  ///
  /// `None` when the point is not in the overlap: the weights the relabelling
  /// would discard -- those on vertices the target does not have -- must vanish,
  /// and that is exactly the statement that the point lies on the shared face.
  pub fn apply(&self, point: &MeshPoint) -> Option<MeshPoint> {
    assert_eq!(
      point.cell_idx(),
      self.source,
      "Point is in the wrong chart."
    );

    let discarded: f64 = (0..=self.dim())
      .filter(|&i| self.bary_map.column(i).sum() == 0.0)
      .map(|i| point.bary()[i].abs())
      .sum();
    if discarded > BARY_EPS {
      return None;
    }

    let bary: Vector = &self.bary_map * point.bary().view();
    Some(MeshPoint::new(self.target, Bary::new(bary)))
  }

  /// The differential $dif psi$ of the transition, in the local (cartesian)
  /// coordinates of the two charts.
  ///
  /// Constant -- the transition is affine -- and metric-free. It is
  /// $dif psi = S P Lambda$, where $Lambda$ is the barycentric differential
  /// [`ref_difbarys`] of the source and $S$ drops the redundant zeroth weight of
  /// the target.
  ///
  /// **It is the differential of $psi$ only on the tangent space of the
  /// overlap**, which is all $psi$ is defined on. Transverse to the shared face
  /// the matrix is whatever the affine formula extends to, and means nothing.
  /// This is why only the *tangential* part of a section is chart-independent,
  /// and it is the precise reason the de Rham map is well defined while a
  /// pointwise value of a Whitney form is not.
  pub fn differential(&self) -> Matrix {
    let dim = self.dim();
    let drop_zeroth = self.bary_map.view_range(1.., ..);
    drop_zeroth * ref_difbarys(dim)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{
    atlas::{barycenter_bary, ChartExt, MeshPoint},
    gen::cartesian::CartesianMeshInfo,
    geometry::coord::simplex::SimplexRefExt,
    topology::complex::Complex,
  };

  use approx::assert_relative_eq;

  /// The transition of a chart with itself is the identity map.
  #[test]
  fn self_transition_is_the_identity() {
    for dim in 1..=3 {
      let complex = Complex::standard(dim);
      let cell = complex.cells().handle_iter().next().unwrap();

      let transition = cell.chart().transition_to(cell.chart());
      assert!(transition.is_identity());
      assert_relative_eq!(transition.bary_map(), &Matrix::identity(dim + 1, dim + 1));
      assert_relative_eq!(transition.differential(), Matrix::identity(dim, dim));

      let point = MeshPoint::barycenter(cell.idx());
      let mapped = transition.apply(&point).unwrap();
      assert_eq!(mapped, point);
    }
  }

  /// Every pair of adjacent cells, with the barycenter of the facet they share:
  /// the setting in which a transition is defined.
  fn adjacent_pairs(complex: &Complex) -> Vec<(Chart<'_>, Chart<'_>, MeshPoint)> {
    let dim = complex.dim();
    let mut pairs = Vec::new();
    for facet in complex.skeleton(dim - 1).handle_iter() {
      let cells: Vec<_> = facet.cells().collect();
      for (i, &source) in cells.iter().enumerate() {
        for &target in &cells[i + 1..] {
          let positions = facet.simplex().relative_to(source.simplex());
          let point = source
            .chart()
            .point_on_face(&positions, &barycenter_bary(dim - 1));
          pairs.push((source.chart(), target.chart(), point));
        }
      }
    }
    pairs
  }

  /// A point of the overlap, carried into the neighbouring chart and back, is
  /// the point one started with: the transitions of an atlas are invertible on
  /// the overlap, and the two directions are mutually inverse.
  #[test]
  fn transition_roundtrip_on_the_overlap() {
    for dim in 1..=3 {
      let (complex, _) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();

      for (source, target, point) in adjacent_pairs(&complex) {
        let transition = source.transition_to(target);
        let there = transition.apply(&point).expect("point is on the overlap");
        assert_eq!(there.cell_idx(), target.idx());

        let back = transition.inverse().apply(&there).unwrap();
        assert_eq!(back.cell_idx(), source.idx());
        assert_relative_eq!(back.bary().view(), point.bary().view(), epsilon = 1e-12);
      }
    }
  }

  /// Off the overlap there is no transition: a point in the interior of a cell
  /// has no representation in any other chart.
  #[test]
  fn no_transition_off_the_overlap() {
    for dim in 1..=3 {
      let (complex, _) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();

      for (source, target, _) in adjacent_pairs(&complex) {
        let interior = source.barycenter();
        assert!(source.transition_to(target).apply(&interior).is_none());
      }
    }
  }

  /// $psi_(K'' K') compose psi_(K' K) = psi_(K'' K)$: the cocycle condition, on
  /// the triple overlap where all three charts see the point.
  ///
  /// This is the coherence law of an atlas -- the statement that the charts
  /// describe *one* manifold and not three.
  #[test]
  fn transition_cocycle() {
    for dim in 2..=3 {
      let (complex, _) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();

      // A vertex of the mesh lies in the overlap of every cell around it.
      for vertex in complex.vertices().handle_iter() {
        let cells: Vec<_> = vertex.cells().collect();
        for &first in &cells {
          let positions = vertex.simplex().relative_to(first.simplex());
          let point = first.chart().point_on_face(&positions, &barycenter_bary(0));

          for &second in &cells {
            for &third in &cells {
              let direct = first
                .chart()
                .transition_to(third.chart())
                .apply(&point)
                .unwrap();
              let composed = first
                .chart()
                .transition_to(second.chart())
                .apply(&point)
                .and_then(|mid| second.chart().transition_to(third.chart()).apply(&mid))
                .unwrap();
              assert_eq!(direct, composed);
            }
          }
        }
      }
    }
  }

  /// The differential of the transition is the change of frame between the two
  /// charts, on the tangent space of the overlap.
  ///
  /// Checked against an embedding, which both charts parametrize: a tangent
  /// vector of the shared face, pushed into the ambient space through either
  /// chart, is the same ambient vector. $A_(K') dif psi = A_K$ on $T sigma$.
  #[test]
  fn differential_is_the_change_of_frame() {
    for dim in 2..=3 {
      let (complex, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();

      for facet in complex.skeleton(dim - 1).handle_iter() {
        let cells: Vec<_> = facet.cells().collect();
        for (i, &source) in cells.iter().enumerate() {
          for &target in &cells[i + 1..] {
            let differential = source.chart().transition_to(target.chart()).differential();

            // The tangent space of the shared facet, in the source chart.
            let positions = facet.simplex().relative_to(source.simplex());
            let tangents = crate::atlas::ref_face_spanning_vectors(dim, &positions);

            let source_frame = source.coord_simplex(&coords).linear_transform();
            let target_frame = target.coord_simplex(&coords).linear_transform();

            assert_relative_eq!(
              &target_frame * (&differential * &tangents),
              &source_frame * &tangents,
              epsilon = 1e-12
            );
          }
        }
      }
    }
  }
}
