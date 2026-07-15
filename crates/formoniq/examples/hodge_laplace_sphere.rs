//! Hodge-Laplace source problem for a 1-form on the 2-sphere, with the data
//! stated in the continuum's own $(theta, phi)$ chart and pulled onto the mesh
//! through the sphere [`Parametrization`].
//!
//! This is the payoff of separating the continuum $M$ from the simplicial
//! approximation $M_h$: a manufactured $k >= 1$ solution on a *curved* manifold,
//! written in intrinsic curvilinear coordinates, to measure the solver against.
//! Before the split there was no way onto the mesh but the affine cell chart, so
//! a 1-form in spherical coordinates could not be pulled through it and the
//! grade-1 apparatus stayed unvalidated on curved geometry.
//!
//! The manufactured solution is the exact 1-form $u = dif f$ with $f = cos theta
//! = z$, a degree-1 spherical harmonic. Since $Delta$ commutes with $dif$ and
//! $f$ is an eigenfunction of the scalar Laplacian, $Delta_1 u = Delta_1 dif f =
//! dif Delta_0 f = lambda dif f = lambda u$ with $lambda = ell(ell + 1) = 2$: so
//! $u$ is a 1-form eigenform, the load is $Delta u = 2 u$, and $H^1(S^2) = 0$
//! leaves no harmonic part to fix. In $(theta, phi)$, $u = -sin theta dif theta$.
//!
//! Run by hand; read the convergence rate off the table.

extern crate nalgebra as na;

use {
  common::util::algebraic_convergence_rate,
  continuum::{field::DiffFormClosure, parametrization::Parametrization},
  ddf::section::CoordFieldExt,
  formoniq::{
    assemble::assemble_galvec, fe::fe_l2_error, operators::SourceElVec, problems::hodge_laplace,
    whitney_complex::WhitneyComplex,
  },
  manifold::dim3::mesh_sphere_surface,
};

fn main() {
  tracing_subscriber::fmt::init();

  let grade = 1;

  // The unit sphere as a continuum. `sphere(2)` gives the hyperspherical
  // forward map with its closed-form nearest-point chart; the axis convention is
  // $x_1 = cos phi_1$, so the polar angle is $phi_1 = u[0]$.
  let sphere = Parametrization::sphere(2, 1.0);

  // u = d(cos phi_1) = -sin(phi_1) d(phi_1), and its Hodge Laplacian Delta u = 2 u.
  let solution_exact = DiffFormClosure::one_form(|u| na::dvector![-u[0].sin(), 0.0], 2);
  let load = DiffFormClosure::one_form(|u| na::dvector![-2.0 * u[0].sin(), 0.0], 2);

  println!(
    "| {:>2} | {:>7} | {:>8} | {:>7} |",
    "k", "ncells", "L2 err", "L2 conv",
  );

  let mut errors = Vec::new();
  for irefine in 0..=6 {
    let (topology, coords) = mesh_sphere_surface(irefine).into_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let source = assemble_galvec(
      &topology,
      &metric,
      SourceElVec::new(&load.pullback_through(&topology, &coords, &sphere), None),
    );

    let (_, galsol, _) = hodge_laplace::solve_hodge_laplace_source(
      &WhitneyComplex::new(&topology, &metric),
      source,
      grade,
    );

    let solution = solution_exact.pullback_through(&topology, &coords, &sphere);
    let error = fe_l2_error(&galsol, &solution, &topology, &metric);
    let conv = errors.last().map_or(f64::INFINITY, |&prev| {
      algebraic_convergence_rate(error, prev)
    });
    errors.push(error);

    let ncells = topology.cells().len();
    println!("| {irefine:>2} | {ncells:>7} | {error:<8.2e} | {conv:>7.2} |");
  }
}
