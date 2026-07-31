//! The tangent bundle of the piecewise-affine manifold, and the exterior
//! bundles over it.
//!
//! A chart identifies its cell with the reference simplex, so the tangent space
//! at a point of a cell *is* $RR^n$ read in that chart's frame, and the fiber
//! over the point is $Lambda^k (RR^n)$ or its dual $Lambda^k (RR^n)^*$. Nothing
//! but the atlas enters: no metric, no embedding. The bundle is therefore chart
//! data of a piece with the charts themselves, and a value of it is expressible
//! wherever a [`MeshPoint`](super::MeshPoint) is.
//!
//! What a chart does not give is a frame over a *face*, since only a cell is a
//! chart. A face has a tangent space all the same, spanned by its edges in the
//! frame of any cell containing it, and the inclusion $iota_tau: tau
//! arrow.hook K$ read in that frame is the whole content of this module, in its
//! two variances: [`face_tangent_blade`] is $Lambda^d$ of it on the vector side,
//! [`FaceTrace`] its pullback on the form side.
//!
//! A value in the fiber is expressed in *some* chart, and there the atlas
//! reasserts itself: two charts containing the same point are related by a
//! [`Transition`](super::Transition), and only the part of a value tangential
//! to their overlap is carried between them. See
//! [`Transition::pullback`](super::Transition::pullback).

use super::unit_face_spanning_vectors;
use crate::Dim;

use multialgebra::{
  ExteriorGrade, Tensor, Variance,
  tensor::{Transport, one_alternating},
};
use multiindex::Combination;

/// The tangent blade $v_1 wedge dots.c wedge v_d$ of a face of a cell, written
/// in the cell's frame: the [`Tensor::blade_of`] its spanning vectors, hence a
/// generator of the one-dimensional $Lambda^d (T tau)$.
///
/// The face is named by its local vertex positions within the cell, and the
/// blade carries the orientation that vertex order gives it. Metric-free and
/// coordinate-free: the spanning vectors are differences of the reference
/// vertices, so a face has them whatever geometry the complex carries, and none
/// at all if it carries none.
pub fn face_tangent_blade(cell_dim: impl Into<Dim>, positions: &Combination) -> Tensor {
  Tensor::blade_of(
    &unit_face_spanning_vectors(cell_dim, positions),
    Variance::Contravariant,
  )
}

/// The trace $tr_tau = iota_tau^*: Lambda^k (T^* K) -> Lambda^k (T^* tau)$ onto
/// a face of a cell, named by the face's local vertex positions: the pullback
/// along the inclusion, hence $Lambda^k$ of the face's spanning vectors.
///
/// Metric-free, being a pullback, and covariant-only: a pullback is the
/// contravariant action of the functor, so it runs this way and no other, and
/// the vector-side counterpart at the face's own grade is
/// [`face_tangent_blade`].
///
/// The result lives on the face's own tangent space, so the normal components
/// are not discarded but absent: there is no space left for them. Trivial above
/// the face's dimension, where $Lambda^k (T^* tau) = 0$.
///
/// Materialized once and applied at every point, $Lambda^k$ of the inclusion
/// being reference data of the chart rather than of the cell.
#[derive(Debug, Clone)]
pub struct FaceTrace {
  /// The functor of $iota_tau$ on one covariant alternating slot, materialized.
  transport: Transport,
  face_dim: Dim,
  grade: ExteriorGrade,
}

impl FaceTrace {
  pub fn new(
    cell_dim: impl Into<Dim>,
    positions: &Combination,
    grade: impl Into<ExteriorGrade>,
  ) -> Self {
    let (cell_dim, grade) = (cell_dim.into(), grade.into());
    let face_dim = Dim::from(positions.card() - 1);
    let inclusion = unit_face_spanning_vectors(cell_dim, positions);
    Self {
      transport: Transport::new(
        &one_alternating(grade, Variance::Covariant, cell_dim),
        &inclusion,
      ),
      face_dim,
      grade,
    }
  }

  pub fn face_dim(&self) -> Dim {
    self.face_dim
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.grade
  }

  /// $tr_tau omega$, valued on the face's tangent space: the pullback along the
  /// inclusion, which is what a trace is.
  pub fn apply(&self, form: &Tensor) -> Tensor {
    assert_eq!(
      form.grade(),
      self.grade,
      "Trace applied at the wrong grade."
    );
    self.transport.pullback(form)
  }

  /// The single coefficient of a trace at the face's own dimension, $k = d$,
  /// where $Lambda^d (T^* tau)$ is one-dimensional.
  ///
  /// At that grade the trace is the duality pairing of the form with the face's
  /// [`face_tangent_blade`], which is what integrating a $k$-form over a
  /// $k$-simplex reduces to: the top-grade case of the trace, not a
  /// construction of its own.
  pub fn top_coefficient(&self, form: &Tensor) -> f64 {
    assert_eq!(
      self.grade, self.face_dim,
      "The top coefficient exists only at the face's own dimension."
    );
    self.apply(form).as_scalar()
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::topology::simplex::Simplex;

  use multialgebra::{Vector, tensor::pairing};
  use multiindex::combinations;

  use approx::assert_relative_eq;

  /// An arbitrary, nowhere-vanishing form of the given shape.
  fn test_form(dim: Dim, grade: ExteriorGrade) -> Tensor {
    let n = multialgebra::exterior_dim(dim, grade);
    Tensor::multiform(
      Vector::from_iterator(n, (0..n).map(|i| 0.7 * (i as f64) - 1.3)),
      dim,
      grade,
    )
  }

  /// At the face's own grade the trace is the duality pairing with the face's
  /// tangent blade: the two variances of one inclusion are adjoint.
  #[test]
  fn top_grade_trace_is_the_tangent_blade_pairing() {
    for dim in (1..=4).map(Dim::from) {
      for face_dim in dim.range_inclusive() {
        for positions in combinations(dim.index() + 1, face_dim.index() + 1) {
          let form = test_form(dim, face_dim);
          let trace = FaceTrace::new(dim, &positions, face_dim);
          let blade = face_tangent_blade(dim, &positions);
          assert_relative_eq!(
            trace.top_coefficient(&form),
            pairing(&form, &blade),
            epsilon = 1e-12
          );
        }
      }
    }
  }

  /// The trace is functorial on the face poset,
  /// $tr_(rho subset tau) compose tr_(tau subset K) = tr_(rho subset K)$,
  /// which is the pullback of a composite inclusion being the composite of the
  /// pullbacks. It is what lets a trace be taken in any order down a chain of
  /// faces, and hence what makes the value on a face independent of the route
  /// taken down to it.
  #[test]
  fn traces_compose_along_a_chain_of_faces() {
    for dim in (0..=4).map(Dim::from) {
      let cell = Simplex::unit(dim);
      for tau_positions in combinations(dim.index() + 1, dim.index().max(1)) {
        let tau = cell.select(tau_positions);
        for rho_positions in combinations(tau.nvertices(), tau.nvertices().div_ceil(2)) {
          // rho inside K is the composite of the two monotone inclusions.
          let direct_positions = tau_positions.select(rho_positions);
          let rho_dim = Dim::from(rho_positions.card() - 1);

          for grade in Dim::ZERO.range_to_inclusive(rho_dim) {
            let form = test_form(dim, grade);

            let stepwise = FaceTrace::new(tau.dim(), &rho_positions, grade)
              .apply(&FaceTrace::new(dim, &tau_positions, grade).apply(&form));
            let direct = FaceTrace::new(dim, &direct_positions, grade).apply(&form);

            assert_relative_eq!(stepwise.components(), direct.components(), epsilon = 1e-12);
          }
        }
      }
    }
  }
}
