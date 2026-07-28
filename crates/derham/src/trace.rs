//! The trace of a form onto a face of a cell.

use {
  exterior::{Dim, ExteriorGrade, MultiForm, exterior_power},
  multiindex::Combination,
  simplicial::{atlas::ref_face_spanning_vectors, linalg::Matrix},
};

/// The trace $tr_tau = iota_tau^*: Lambda^k (T^* K) -> Lambda^k (T^* tau)$ onto
/// a face of a cell, identified by the face's local vertex positions: the
/// pullback along the inclusion, hence $Lambda^k$ of the face's spanning
/// vectors.
///
/// Metric-free, being a pullback, and covariant-only for the same reason a
/// [`Pullback`](crate::section::Pullback) is: a form pulls back along an
/// inclusion where a multivector pushes forward along it. The vector-side
/// counterpart is
/// [`face_tangent_blade`](crate::project::face_tangent_blade), and the two are
/// adjoint under the duality pairing.
///
/// The result lives on the face's *own* tangent space, of dimension
/// $d = dim tau$, so the normal components are not discarded but **absent**:
/// there is no longer a space for them to inhabit. That is what "only the
/// tangential part of a form is chart-independent" says as a type rather than
/// as a caveat, and why the de Rham map is well defined on a face while a
/// pointwise value is not.
///
/// Trivial below the grade: a face of dimension $d < k$ has $Lambda^k (T^* tau)
/// = 0$, and the trace lands in that zero space rather than failing.
///
/// The trace is a linear map, materialized once and applied at every point,
/// because $Lambda^k$ of the inclusion is reference data of the cell dimension
/// and the face's positions alone.
#[derive(Debug, Clone)]
pub struct FaceTrace {
  /// $Lambda^k (iota_tau)^T$, of shape $binom(d, k) times binom(n, k)$.
  map: Matrix,
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
    let inclusion = ref_face_spanning_vectors(cell_dim, positions);
    Self {
      map: exterior_power(&inclusion, grade).transpose(),
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

  /// $tr_tau omega$, valued on the face's tangent space.
  pub fn apply(&self, form: &MultiForm) -> MultiForm {
    assert_eq!(
      form.grade(),
      self.grade,
      "Trace applied at the wrong grade."
    );
    MultiForm::new(&self.map * form.coeffs(), self.face_dim, self.grade)
  }

  /// The single coefficient of a trace at the face's *top* grade, $k = d$,
  /// where $Lambda^d (T^* tau)$ is one-dimensional.
  ///
  /// This is the integrand of the de Rham map: $tr_tau omega$ at top grade is
  /// the duality pairing of $omega$ with the face's tangent blade, so
  /// integrating a $k$-form over a $k$-simplex is the top-grade case of the
  /// trace and not a construction of its own.
  pub fn top_coefficient(&self, form: &MultiForm) -> f64 {
    assert_eq!(
      self.grade, self.face_dim,
      "The top coefficient exists only at the face's own dimension."
    );
    self.apply(form).coeffs()[0]
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::project::face_tangent_blade;

  use exterior::Vector;
  use multiindex::combinations;
  use simplicial::topology::simplex::Simplex;

  use approx::assert_relative_eq;

  fn test_form(dim: Dim, grade: ExteriorGrade) -> MultiForm {
    let n = exterior::exterior_dim(dim, grade);
    MultiForm::new(
      Vector::from_iterator(n, (0..n).map(|i| 0.7 * (i as f64) - 1.3)),
      dim,
      grade,
    )
  }

  /// At the face's own grade the trace is the duality pairing with the face's
  /// tangent blade, which is the de Rham map's integrand: integrating a
  /// $k$-form over a $k$-simplex is the top-grade case of the trace.
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
            form.pairing(&blade),
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
  /// faces, and it is the reason the de Rham map on a face does not depend on
  /// the route taken to reach it.
  #[test]
  fn traces_compose_along_a_chain_of_faces() {
    for dim in (0..=4).map(Dim::from) {
      let cell = Simplex::standard(dim);
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

            assert_relative_eq!(stepwise.coeffs(), direct.coeffs(), epsilon = 1e-12);
          }
        }
      }
    }
  }
}
