//! The exterior bundle's face trace: functoriality on the face poset, and
//! its agreement with the tangent-blade pairing at top grade.

use approx::assert_relative_eq;
use multialgebra::{Tensor, Vector, tensor::pairing};
use multiindex::combinations;
use simplicial::Dim;
use simplicial::atlas::bundle::{FaceTrace, face_tangent_blade};
use simplicial::topology::simplex::Simplex;

/// An arbitrary, nowhere-vanishing form of the given shape.
fn test_form(dim: Dim, grade: multialgebra::ExteriorGrade) -> Tensor {
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
