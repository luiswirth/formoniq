//! Manufactured-solution tests for boundary conditions.
//!
//! The linear solution $u(x) = x_1$ lies exactly in the Whitney 0-form
//! space and all loads are affine, so both the Dirichlet and the Neumann
//! discretizations reproduce it up to solver tolerance -- validating the
//! affine lifting, the trace complex geometry and the natural boundary
//! load.

extern crate nalgebra as na;

use common::linalg::{faer::FaerCholesky, nalgebra::CsrMatrix};
use ddf::{cochain::Cochain, derham::derham_map, section::CoordFieldExt};
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
    let whitney = WhitneyComplex::new(&topology, &metric);
    let boundary = whitney.boundary().unwrap();

    let exact = DiffFormClosure::coordinate_component(0, dim);
    let exact_cochain = derham_map(&exact.pullback_on(&topology, &coords), &topology, 1);

    let boundary_values = boundary.trace_cochain(&exact_cochain);
    let laplace = CsrMatrix::from(&whitney.codif_dif(0));
    let rhs = common::linalg::nalgebra::Vector::zeros(whitney.ndofs(0));

    let solution = bc::solve_with_essential_bc(
      &whitney.relative(),
      &boundary,
      laplace,
      &rhs,
      &boundary_values,
    );

    assert_relative_eq!(solution.coeffs(), exact_cochain.coeffs(), epsilon = 1e-10);
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
    let whitney = WhitneyComplex::new(&topology, &metric);
    let boundary = whitney.boundary().unwrap();

    let exact = DiffFormClosure::coordinate_component(0, dim);
    let exact_cochain = derham_map(&exact.pullback_on(&topology, &coords), &topology, 1);

    // System: (grad u, grad v) + (u, v).
    let system = CsrMatrix::from(&whitney.codif_dif(0)) + CsrMatrix::from(&whitney.mass(0));

    // Source load (u, v) side: f = x_1. The integrand f phi_i is
    // quadratic, so an order-3 quadrature keeps it exact.
    let source = DiffFormClosure::coordinate_component(0, dim);
    let source = source.pullback_on(&topology, &coords);
    let qr = manifold::geometry::coord::quadrature::SimplexQuadRule::degree(dim, 3);
    let mut rhs =
      assemble::assemble_galvec(&topology, &metric, SourceElVec::new(&source, Some(qr)));

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
    // The boundary data is a field on the boundary manifold, reached from the
    // ambient flux by pullback against the trace coordinates.
    let boundary_coords = boundary.boundary_complex().trace_coords(&coords);
    let flux = flux.pullback_on(boundary.topology(), &boundary_coords);
    rhs += bc::neumann_load(&boundary, &flux, None);

    let solution = FaerCholesky::new(system).solve(&rhs);

    assert_relative_eq!(solution, exact_cochain.coeffs(), epsilon = 1e-9);
  }
}

/// Mixed boundary conditions: Dirichlet on the faces $x_1 = 0, 1$
/// (where $u = x_1$ is 0 resp. 1), natural on the remaining faces
/// (where $diff u \/ diff n = 0$, homogeneous: do nothing).
/// The exact solution $u = x_1$ is reproduced.
#[test]
fn mixed_dirichlet_neumann_reproduces_linear_solution() {
  use manifold::geometry::coord::simplex::SimplexCoords;

  for dim in 2..=3 {
    let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);

    // Partition of the boundary facets by their barycenter.
    let dirichlet_facets: Vec<_> = topology
      .boundary_facets()
      .into_iter()
      .filter(|facet| {
        let facet_coords =
          SimplexCoords::from_simplex_and_coords(facet.handle(&topology).simplex(), &coords);
        let x = facet_coords.barycenter()[0];
        x <= 1e-12 || x >= 1.0 - 1e-12
      })
      .collect();
    let gamma_dirichlet = whitney.boundary_part(dirichlet_facets);

    let exact = DiffFormClosure::coordinate_component(0, dim);
    let exact_cochain = derham_map(&exact.pullback_on(&topology, &coords), &topology, 1);
    let boundary_values = gamma_dirichlet.trace_cochain(&exact_cochain);

    let laplace = CsrMatrix::from(&whitney.codif_dif(0));
    let rhs = common::linalg::nalgebra::Vector::zeros(whitney.ndofs(0));

    let solution = bc::solve_with_essential_bc(
      &whitney.relative_to(&gamma_dirichlet),
      &gamma_dirichlet,
      laplace,
      &rhs,
      &boundary_values,
    );

    assert_relative_eq!(solution.coeffs(), exact_cochain.coeffs(), epsilon = 1e-10);
  }
}

/// Robin boundary condition $diff u \/ diff n + alpha "tr" u = h$ with
/// $alpha = 1$ and $h$ manufactured from $u = x_1$. In 1d on the whole
/// boundary; in higher dimensions on the faces $x_1 = 0, 1$ (where $h$ is
/// per-face constant), combined with Dirichlet on the remaining faces --
/// all three condition kinds in one problem. The exact solution is
/// reproduced.
#[test]
fn robin_reproduces_linear_solution() {
  use manifold::geometry::coord::simplex::SimplexCoords;

  for dim in 1..=3 {
    let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);

    let exact = DiffFormClosure::coordinate_component(0, dim);
    let exact_cochain = derham_map(&exact.pullback_on(&topology, &coords), &topology, 1);

    let alpha = 1.0;
    // h = du/dn + alpha u: constant on each Robin face.
    let robin_data = DiffFormClosure::scalar(
      move |p| {
        if p[0] <= 1e-12 {
          -1.0 + alpha * 0.0
        } else {
          1.0 + alpha * 1.0
        }
      },
      dim,
    );

    let is_x_facet = |facet: &manifold::topology::handle::SimplexIdx| {
      let facet_coords =
        SimplexCoords::from_simplex_and_coords(facet.handle(&topology).simplex(), &coords);
      let x = facet_coords.barycenter()[0];
      x <= 1e-12 || x >= 1.0 - 1e-12
    };
    let (robin_facets, dirichlet_facets): (Vec<_>, Vec<_>) =
      topology.boundary_facets().into_iter().partition(is_x_facet);

    let gamma_robin = whitney.boundary_part(robin_facets);
    let system =
      CsrMatrix::from(&whitney.codif_dif(0)) + alpha * bc::boundary_mass(&gamma_robin, 0);
    let boundary_coords = gamma_robin.boundary_complex().trace_coords(&coords);
    let robin_data = robin_data.pullback_on(gamma_robin.topology(), &boundary_coords);
    let rhs = bc::neumann_load(&gamma_robin, &robin_data, None);

    let solution = if dirichlet_facets.is_empty() {
      // 1d: pure Robin.
      Cochain::new(0, FaerCholesky::new(system).solve(&rhs))
    } else {
      let gamma_dirichlet = whitney.boundary_part(dirichlet_facets);
      let boundary_values = gamma_dirichlet.trace_cochain(&exact_cochain);
      bc::solve_with_essential_bc(
        &whitney.relative_to(&gamma_dirichlet),
        &gamma_dirichlet,
        system,
        &rhs,
        &boundary_values,
      )
    };

    assert_relative_eq!(solution.coeffs(), exact_cochain.coeffs(), epsilon = 1e-9);
  }
}
