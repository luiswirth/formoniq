//! Manufactured solutions for the boundary conditions.
//!
//! The linear solution $u(x) = x_1$ lies exactly in the Whitney 0-form space
//! and all loads are affine, so a discretization that imposes its boundary
//! condition correctly reproduces it up to solver tolerance. That pins the
//! affine lifting of essential data, the geometry of the trace complex, the
//! natural boundary load and the boundary mass.

extern crate nalgebra as na;

use derham::{Cochain, project::derham_map, section::CoordFieldExt};
use formoniq::linalg::faer::FaerCholesky;
use formoniq::{
  bc,
  galerkin::GalerkinVector,
  whitney_complex::{HilbertComplex, WhitneyComplex},
};
use glatt::field::DiffFormClosure;
use regge::coord::simplex::simplex_coords;
use regge::mesher::cartesian::CartesianGrid;
use regge::subcomplex::SubcomplexExt;
use simplicial::{Dim, linalg::Vector};

use approx::assert_relative_eq;

/// Inhomogeneous essential (Dirichlet) BC by affine lifting:
/// $-Delta u = 0$ on the unit cube with $"tr" u = x_1$ has the exact
/// solution $u = x_1$, which lies in the FE space.
#[test]
fn inhomogeneous_dirichlet_reproduces_linear_solution() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);
    let boundary = whitney.boundary().unwrap();

    let exact = DiffFormClosure::coord_component(0, dim);
    let exact_cochain = derham_map(&exact.pullback_on(&topology, &coords), &topology, 1);

    let boundary_values = boundary.trace_cochain(&exact_cochain);
    let laplace = whitney.dif_both(1);
    let rhs = GalerkinVector::new(Dim::ZERO, Vector::zeros(whitney.ndofs(Dim::ZERO)));

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

/// Robin boundary condition $partial u \/ partial n + alpha "tr" u = h$ with
/// $alpha = 1$ and $h$ manufactured from $u = x_1$. In 1d on the whole
/// boundary. In higher dimensions on the faces $x_1 = 0, 1$ (where $h$ is
/// per-face constant), combined with Dirichlet on the remaining faces,
/// all three condition kinds in one problem. The exact solution is
/// reproduced.
#[test]
fn robin_reproduces_linear_solution() {
  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);

    let exact = DiffFormClosure::coord_component(0, dim);
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

    let is_x_facet = |facet: &simplicial::topology::role::Facet<'_>| {
      let facet_coords = simplex_coords(facet.simplex(), &coords);
      let x = facet_coords.barycenter()[0];
      x <= 1e-12 || x >= 1.0 - 1e-12
    };
    let (robin_facets, dirichlet_facets): (Vec<_>, Vec<_>) =
      topology.boundary_facets().into_iter().partition(is_x_facet);

    let gamma_robin = whitney.boundary_part(robin_facets);
    let system = whitney.dif_both(1) + alpha * bc::boundary_mass(&gamma_robin, Dim::ZERO);
    let boundary_coords = gamma_robin.boundary_complex().trace_coords(&coords);
    let robin_data = robin_data.pullback_on(gamma_robin.topology(), &boundary_coords);
    let rhs = bc::neumann_load(&gamma_robin, &robin_data, None);

    let solution = if dirichlet_facets.is_empty() {
      // 1d: pure Robin.
      Cochain::new(Dim::ZERO, FaerCholesky::new(system).solve(rhs.coeffs()))
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
