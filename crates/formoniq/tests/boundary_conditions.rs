//! Manufactured-solution tests for boundary conditions.
//!
//! The linear solution $u(x) = x_1$ lies exactly in the Whitney 0-form
//! space and all loads are affine, so both the Dirichlet and the Neumann
//! discretizations reproduce it up to solver tolerance -- validating the
//! affine lifting, the trace complex geometry and the natural boundary
//! load.

extern crate nalgebra as na;

use common::linalg::{faer::FaerCholesky, nalgebra::CsrMatrix};
use ddf::derham::derham_map;
use exterior::field::DiffFormClosure;
use formoniq::{assemble, bc, operators::SourceElVec, whitney_complex::WhitneyComplex};
use manifold::gen::cartesian::CartesianMeshInfo;

use approx::assert_relative_eq;

/// Inhomogeneous essential (Dirichlet) BC by affine lifting:
/// $-Delta u = 0$ on the unit cube with $"tr" u = x_1$ has the exact
/// solution $u = x_1$, which lies in the FE space.
#[test]
fn inhomogeneous_dirichlet_reproduces_linear_solution() {
  for dim in 1..=3 {
    let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let fes = WhitneyComplex::new(&topology, &metric);
    let boundary = fes.boundary().unwrap();

    let exact = DiffFormClosure::coordinate_component(0, dim);
    let exact_cochain = derham_map(&exact, &topology, &coords, None);

    let boundary_values = boundary.trace_cochain(&exact_cochain);
    let laplace = CsrMatrix::from(&fes.codif_dif(0));
    let rhs = common::linalg::nalgebra::Vector::zeros(fes.ndofs(0));

    let solution =
      bc::solve_with_essential_bc(&fes.relative(), &boundary, laplace, &rhs, &boundary_values);

    assert_relative_eq!(solution.coeffs, exact_cochain.coeffs, epsilon = 1e-10);
  }
}

/// Inhomogeneous natural (Neumann) BC via the boundary load:
/// $-Delta u + u = x_1$ on the unit cube with flux data
/// $h = diff u \/ diff n = plus.minus 1$ on the faces $x_1 = 1, 0$ has the
/// exact solution $u = x_1$, which lies in the FE space.
#[test]
fn inhomogeneous_neumann_reproduces_linear_solution() {
  for dim in 1..=3 {
    let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let fes = WhitneyComplex::new(&topology, &metric);
    let boundary = fes.boundary().unwrap();

    let exact = DiffFormClosure::coordinate_component(0, dim);
    let exact_cochain = derham_map(&exact, &topology, &coords, None);

    // System: (grad u, grad v) + (u, v).
    let system =
      CsrMatrix::from(&fes.codif_dif(0)) + CsrMatrix::from(&fes.mass(0));

    // Source load (u, v) side: f = x_1. The integrand f phi_i is
    // quadratic, so an order-3 quadrature keeps it exact.
    let source = DiffFormClosure::coordinate_component(0, dim);
    let qr = manifold::geometry::coord::quadrature::SimplexQuadRule::order3(dim);
    let mut rhs = assemble::assemble_galvec(
      &topology,
      &metric,
      SourceElVec::new(&source, &coords, dim, Some(qr)),
    );

    // Natural boundary load: h = du/dn = -1 on x_1 = 0, +1 on x_1 = 1,
    // 0 on the remaining faces.
    let flux = DiffFormClosure::scalar(
      |p| {
        if p[0] <= 1e-12 {
          -1.0
        } else if p[0] >= 1.0 - 1e-12 {
          1.0
        } else {
          0.0
        }
      },
      dim,
    );
    rhs += bc::neumann_load(&boundary, &coords, &flux);

    let solution = FaerCholesky::new(system).solve(&rhs);

    assert_relative_eq!(solution, exact_cochain.coeffs, epsilon = 1e-9);
  }
}
