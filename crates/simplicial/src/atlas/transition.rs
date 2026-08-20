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
