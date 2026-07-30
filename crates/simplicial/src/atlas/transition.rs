//! The transition maps of the piecewise-affine atlas.
//!
//! Two cells are two charts, and they overlap in the face they share. On that
//! overlap the same point of the manifold has two representations, one per
//! chart, and the map relating them is the transition map
//! $psi_(K' K): hat(K) supset.eq sigma -> sigma subset.eq hat(K')$.
//!
//! It is the affine gluing of the shared face, and it is pure combinatorics: a
//! vertex of the mesh has one barycentric weight, and each chart merely lists
//! the vertices in a different place. So the transition is the $0\/1$ matrix
//! $P$ that relabels the weights,
//!
//! $lambda'_j = cases(lambda_i & "if the" j"-th vertex of" K' "is the" i"-th of" K, 0 & "otherwise")$
//!
//! and it is defined precisely where the weights it must discard vanish, on
//! the shared face. Metric-free, coordinate-free, exact.
//!
//! This is what makes the atlas an atlas, and on the barycentric weights it is
//! exact. On the *fibers* it is exact only where $psi$ itself is, and that
//! asymmetry is the point. Carrying a value of the exterior bundle from one
//! chart to the other ([`Transition::pullback`], [`Transition::pushforward`])
//! goes through $dif psi$, which is the true change of frame on the tangent
//! space of the overlap and an artifact of the affine formula off it. So a
//! transported value is determined only in its tangential part
//! ([`Transition::overlap_trace`]), a quantity two charts agree on is a
//! tangential one, and anything claiming to be chart-independent owes a
//! transition argument.

use super::{BARY_EPS, Bary, Chart, MeshPoint, unit_difbarys};
use crate::{Dim, topology::handle::SimplexIdx};

use super::bundle::FaceTrace;
use crate::linalg::{Matrix, Vector};
use multialgebra::{Slot, Tensor, tensor::Transport};
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
  /// $P$: the $(n+1) times (n+1)$ relabeling of barycentric weights, with a
  /// zero row for each vertex only the target has and a zero column for each
  /// vertex only the source has.
  bary_map: Matrix,
}

impl Transition {
  /// The transition from `source` into `target`.
  ///
  /// That the two are charts, and hence cells, is the [`Chart`] type's
  /// business, not this one's. What remains to check is that they are charts of
  /// the same atlas.
  pub fn new(source: Chart, target: Chart) -> Self {
    assert!(
      source.belongs_to(target.complex()),
      "Charts of two different atlases have no transition."
    );
    let dim = source.dim();

    let source_vertices = &source.simplex().vertices;
    let target_vertices = &target.simplex().vertices;

    let mut bary_map = Matrix::zeros((dim + 1).index(), (dim + 1).index());
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

  /// $P$: the relabeling of the barycentric weights.
  pub fn bary_map(&self) -> &Matrix {
    &self.bary_map
  }

  /// The local vertex positions, in the source chart, of the vertices shared
  /// with the target: the overlap of the two charts, as a face of the source.
  pub fn overlap_positions(&self) -> Combination {
    Combination::from_increasing(
      (0..=self.dim().index()).filter(|&i| self.bary_map.column(i).sum() != 0.0),
    )
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
  /// `None` when the point is not in the overlap: the weights the relabeling
  /// would discard, those on vertices the target does not have, must vanish,
  /// and that is exactly the statement that the point lies on the shared face.
  pub fn apply(&self, point: &MeshPoint) -> Option<MeshPoint> {
    assert_eq!(
      point.cell_idx(),
      self.source,
      "Point is in the wrong chart."
    );

    let discarded: f64 = (0..=self.dim().index())
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
  /// Constant, the transition is affine, and metric-free. It is
  /// $dif psi = S P Lambda$, where $Lambda$ is the barycentric differential
  /// [`unit_difbarys`] of the source and $S$ drops the redundant zeroth weight of
  /// the target.
  ///
  /// It is the differential of $psi$ only on the tangent space of the
  /// overlap, which is all $psi$ is defined on. Transverse to the shared face
  /// the matrix is whatever the affine formula extends to, and means nothing.
  /// This is why only the tangential part of a section is chart-independent,
  /// and it is the precise reason the de Rham map is well defined while a
  /// pointwise form value is not.
  pub fn differential(&self) -> Matrix {
    let dim = self.dim();
    let drop_zeroth = self.bary_map.view_range(1.., ..);
    drop_zeroth * unit_difbarys(dim)
  }

  /// The functor of [`differential`](Self::differential) on a fixed tensor
  /// shape, materialized once and applied to many values.
  ///
  /// A [`Transport`] carries no variance of its own, so this one object is both
  /// the pullback of [`Self::pullback`] and the pushforward of
  /// [`Self::pushforward`]; the value being transported decides which.
  pub fn transport(&self, slots: &[Slot]) -> Transport {
    Transport::new(slots, &self.differential())
  }

  /// The value in the source chart's frame of a covariant fiber value given in
  /// the target's: $psi^* omega$.
  ///
  /// Defined for every $omega$, a pullback needing no invertibility, but
  /// *meaningful* only tangentially: $dif psi$ is the change of frame on
  /// $T sigma$ for the overlap $sigma$, and off $T sigma$ it is whatever the
  /// affine formula extends to. So $tr_sigma (psi^* omega) = tr_sigma omega$,
  /// the traces taken in the two charts, while the remaining components of
  /// $psi^* omega$ are an artifact of the extension: any other route between the
  /// same two charts produces different ones. That is the precise content of
  /// "only the tangential part of a section is chart-independent", and
  /// [`overlap_trace`](Self::overlap_trace) is what takes it.
  pub fn pullback(&self, value: &Tensor) -> Tensor {
    value.pullback(&self.differential())
  }

  /// The value in the target chart's frame of a contravariant fiber value given
  /// in the source's: $psi_* v$.
  ///
  /// The other variance of the same map, and it inherits the same caveat from
  /// the other side: it is the change of frame on $Lambda^bullet$ of the shared
  /// face's tangent space, and a value with a component transverse to that face
  /// is carried by the affine extension of $psi$, which describes nothing.
  pub fn pushforward(&self, value: &Tensor) -> Tensor {
    value.pushforward(&self.differential())
  }

  /// The trace onto the overlap, taken in the source chart: the projection onto
  /// the part of a fiber value the two charts share.
  pub fn overlap_trace(&self, grade: impl Into<crate::Degree>) -> FaceTrace {
    FaceTrace::new(self.dim(), &self.overlap_positions(), grade)
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::Dim;
  use crate::mesher::grid::CartesianTopology;
  use crate::{
    atlas::{ChartExt, MeshPoint, barycenter_bary},
    topology::complex::Complex,
  };

  use crate::atlas::bundle::{FaceTrace, face_tangent_blade};
  use crate::topology::handle::SimplexRef;
  use multialgebra::{ExteriorGrade, Tensor, Vector, exterior_dim};

  use approx::assert_relative_eq;

  /// The transition of a chart with itself is the identity map.
  #[test]
  fn self_transition_is_the_identity() {
    for dim in (1..=3usize).map(Dim::from) {
      let complex = Complex::unit(dim);
      let cell = complex.cells().handle_iter().next().unwrap();

      let transition = cell.transition_to(cell);
      assert!(transition.is_identity());
      assert_relative_eq!(
        transition.bary_map(),
        &Matrix::identity((dim + 1).index(), (dim + 1).index())
      );
      assert_relative_eq!(
        transition.differential(),
        Matrix::identity(dim.index(), dim.index())
      );

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
          let point = source.point_on_face(&positions, &barycenter_bary(dim - 1));
          pairs.push((source, target, point));
        }
      }
    }
    pairs
  }

  /// A point of the overlap, carried into the neighboring chart and back, is
  /// the point one started with: the transitions of an atlas are invertible on
  /// the overlap, and the two directions are mutually inverse.
  #[test]
  fn transition_roundtrip_on_the_overlap() {
    for dim in (1..=3usize).map(Dim::from) {
      let complex = CartesianTopology::cube(dim, 2).triangulate();

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
    for dim in (1..=3usize).map(Dim::from) {
      let complex = CartesianTopology::cube(dim, 2).triangulate();

      for (source, target, _) in adjacent_pairs(&complex) {
        let interior = source.barycenter();
        assert!(source.transition_to(target).apply(&interior).is_none());
      }
    }
  }

  /// $psi_(K'' K') compose psi_(K' K) = psi_(K'' K)$: the cocycle condition, on
  /// the triple overlap where all three charts see the point.
  ///
  /// This is the coherence law of an atlas, the statement that the charts
  /// describe one manifold and not three.
  #[test]
  fn transition_cocycle() {
    for dim in (2..=3usize).map(Dim::from) {
      let complex = CartesianTopology::cube(dim, 2).triangulate();

      // A vertex of the mesh lies in the overlap of every cell around it.
      for vertex in complex.vertices().handle_iter() {
        let cells: Vec<_> = vertex.cells().collect();
        for &first in &cells {
          let positions = vertex.simplex().relative_to(first.simplex());
          let point = first.point_on_face(&positions, &barycenter_bary(Dim::new(0)));

          for &second in &cells {
            for &third in &cells {
              let direct = first.transition_to(third).apply(&point).unwrap();
              let composed = first
                .transition_to(second)
                .apply(&point)
                .and_then(|mid| second.transition_to(third).apply(&mid))
                .unwrap();
              assert_eq!(direct, composed);
            }
          }
        }
      }
    }
  }

  /// An arbitrary, nowhere-vanishing form of the given shape.
  fn test_form(dim: Dim, grade: ExteriorGrade) -> Tensor {
    let n = exterior_dim(dim, grade);
    Tensor::multiform(
      Vector::from_iterator(n, (0..n).map(|i| 0.7 * (i as f64) - 1.3)),
      dim,
      grade,
    )
  }

  /// Every face of dimension at least one, with the charts that contain it:
  /// the setting in which the fibers over an overlap can be compared.
  fn shared_faces(complex: &Complex) -> Vec<(SimplexRef<'_>, Vec<Chart<'_>>)> {
    let mut faces = Vec::new();
    for face_dim in Dim::ONE.range_to_inclusive(complex.dim()) {
      for face in complex.skeleton(face_dim).handle_iter() {
        let cells = face.cells().collect();
        faces.push((face, cells));
      }
    }
    faces
  }

  /// The tangent blade of a shared face transforms by $Lambda^d (dif psi)$: the
  /// pushforward of the blade computed in one chart is the blade computed in
  /// the other.
  ///
  /// The vector side of the agreement, and the sharpest form of it: the blade
  /// spans the whole of $Lambda^d (T tau)$, so the transition is pinned on the
  /// tangential part with nothing left over.
  #[test]
  fn tangent_blade_transforms_by_the_transition_differential() {
    for dim in (2..=3usize).map(Dim::from) {
      let complex = CartesianTopology::cube(dim, 2).triangulate();

      for (face, cells) in shared_faces(&complex) {
        for (i, &source) in cells.iter().enumerate() {
          for &target in &cells[i + 1..] {
            let here = face_tangent_blade(dim, &face.simplex().relative_to(source.simplex()));
            let there = face_tangent_blade(dim, &face.simplex().relative_to(target.simplex()));

            assert_relative_eq!(
              source.transition_to(target).pushforward(&here).components(),
              there.components(),
              epsilon = 1e-12
            );
          }
        }
      }
    }
  }

  /// Two charts sharing a face agree on the tangential part of a fiber value:
  /// $tr_tau (psi^* omega) = tr_tau omega$, the traces taken in the respective
  /// charts.
  ///
  /// The form side of the same fact, and the one that makes an integral over a
  /// face well defined regardless of which adjacent chart computes it.
  ///
  /// The equality is not vacuous, and the test says so: the two chart
  /// representations $psi^* omega$ and $omega$ genuinely differ, and it is only
  /// after tracing that they agree.
  #[test]
  fn charts_agree_on_the_tangential_part() {
    let mut disagreements = 0;
    for dim in (2..=3usize).map(Dim::from) {
      let complex = CartesianTopology::cube(dim, 2).triangulate();

      for (face, cells) in shared_faces(&complex) {
        let face_dim = face.dim();
        for (i, &source) in cells.iter().enumerate() {
          for &target in &cells[i + 1..] {
            let transition = source.transition_to(target);
            let here = face.simplex().relative_to(source.simplex());
            let there = face.simplex().relative_to(target.simplex());

            for grade in Dim::ZERO.range_to_inclusive(face_dim) {
              let value = test_form(dim, grade);
              let pulled = transition.pullback(&value);

              assert_relative_eq!(
                FaceTrace::new(dim, &here, grade)
                  .apply(&pulled)
                  .components(),
                FaceTrace::new(dim, &there, grade)
                  .apply(&value)
                  .components(),
                epsilon = 1e-12
              );

              if !pulled.eq_epsilon(&value, 1e-9) {
                disagreements += 1;
              }
            }
          }
        }
      }
    }
    assert!(
      disagreements > 0,
      "The charts must genuinely disagree off the tangential part."
    );
  }

  /// The cocycle law on the fibers:
  /// $psi_(K' K)^* compose psi_(K'' K')^* = psi_(K'' K)^*$, traced onto the
  /// face all three charts share. Pullbacks compose contravariantly, so a route
  /// through a third chart is the same map as the direct one.
  ///
  /// The law holds where the composite means anything, which is the triple
  /// overlap and no further. Off it the two routes are affine extensions of maps
  /// that describe nothing, and the test asserts that they do differ there: the
  /// untraced values disagree, which is what it looks like for the
  /// non-tangential part of a fiber value to be an artifact of the route rather
  /// than data of the manifold.
  ///
  /// This is the fiber-level counterpart of the cocycle on points, and it is
  /// what makes a tangential quantity well defined over the whole manifold
  /// rather than merely between neighbors.
  #[test]
  fn fiber_cocycle_on_the_triple_overlap() {
    let mut disagreements = 0;
    for dim in (2..=3usize).map(Dim::from) {
      let complex = CartesianTopology::cube(dim, 2).triangulate();

      for (face, cells) in shared_faces(&complex) {
        for &first in &cells {
          let positions = face.simplex().relative_to(first.simplex());
          for &second in &cells {
            for &third in &cells {
              let (a, b) = (first.transition_to(second), second.transition_to(third));
              let c = first.transition_to(third);

              for grade in Dim::ZERO.range_to_inclusive(face.dim()) {
                let value = test_form(dim, grade);
                let routed = a.pullback(&b.pullback(&value));
                let direct = c.pullback(&value);
                let trace = FaceTrace::new(dim, &positions, grade);

                assert_relative_eq!(
                  trace.apply(&routed).components(),
                  trace.apply(&direct).components(),
                  epsilon = 1e-12
                );

                if !routed.eq_epsilon(&direct, 1e-9) {
                  disagreements += 1;
                }
              }
            }
          }
        }
      }
    }
    assert!(
      disagreements > 0,
      "The two routes must differ off the triple overlap."
    );
  }
}
