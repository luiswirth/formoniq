//! The three maps from $L^2 Lambda^k$ into the Whitney space, and the error
//! between them.
//!
//! A differential form on the manifold reaches the discrete space by one of
//! three routes, and they are genuinely different maps:
//!
//! - $W: C^k -> cal(W) Lambda^k$, the Whitney interpolation
//!   ([`WhitneyInterpolant`]): the reconstruction of a form from a cochain, the
//!   right inverse of $R$.
//! - $R: L^2 Lambda^k -> C^k$, the de Rham map
//!   ([`derham_map`](derham::project::derham_map)): integration over the
//!   simplices. Canonical and metric-free, and a cochain map,
//!   $R compose dif = dif compose R$, which is exactly why the discrete
//!   complex inherits the cohomology of the continuous one. It needs the
//!   traces to exist, so it is not defined on all of $L^2 Lambda^k$.
//! - $P_h: L^2 Lambda^k -> cal(W) Lambda^k$, the $L^2$ projection
//!   ([`l2_projection`]): the best approximation in the energy norm, defined on
//!   all of $L^2 Lambda^k$ but not commuting with $dif$, and requiring a
//!   global mass solve rather than local integration.
//!
//! $R$ is the one the theory is built on; $P_h$ is the one that is optimal in
//! norm. Neither dominates the other, and the discrete complex is exact only
//! through $R$.

use {
  crate::linalg::faer::FaerLu,
  derham::{Cochain, interpolate::interpolant::WhitneyInterpolant, section::Section},
  iterative::{Jacobi, StopCriterion, krylov::cg},
  metric::tensor::inner,
  regge::{cell_volume, lengths::mesh::MeshLengthsSq},
  simplicial::{
    atlas::{MeshPoint, SimplexQuadRule},
    topology::complex::Complex,
  },
};

use crate::{
  galerkin::LinearForm,
  operators::SourceForm,
  whitney_complex::{HilbertComplex, WhitneyComplex},
};

/// The $L^2 Lambda^k$ error $norm(omega - W c)_(L^2)$ between an exact form on
/// the manifold and the Whitney reconstruction of a cochain.
///
/// Intrinsic: the pointwise difference is measured in the reference frame of
/// each cell by the induced inner product $Lambda^k g^(-1)$ of that cell's
/// metric. On a curved (embedded) mesh this is the only correct thing to do,
/// the flat ambient Gramian would measure the wrong norm.
pub fn fe_l2_error<F: Section>(
  fe_cochain: &Cochain,
  exact: &F,
  topology: &Complex,
  geometry: &MeshLengthsSq,
) -> f64 {
  let dim = topology.dim();
  let qr = SimplexQuadRule::degree(dim, 3);
  let fe_whitney = WhitneyInterpolant::new(fe_cochain.clone(), topology);

  let error_sq: f64 = topology
    .cells()
    .handle_iter()
    .map(|cell| {
      let metric = geometry.cell_metric(cell);
      let error_pointwise = |point: &MeshPoint| {
        let error = exact.at(point) - fe_whitney.at(point);
        inner(&error, &error, &metric)
      };
      qr.integrate_cell(cell, &error_pointwise, cell_volume(&metric))
    })
    .sum();

  error_sq.sqrt()
}

/// The $L^2$ projection $P_h omega$ of a form onto the Whitney space: the
/// solution of the mass system
///
/// $M c = b, quad b_sigma = integral_M inner(omega, W_sigma) vol$
///
/// i.e. the best approximation in the $L^2 Lambda^k$ norm, characterized by
/// Galerkin orthogonality $inner(omega - P_h omega, v) = 0$ for all
/// $v in cal(W) Lambda^k$.
///
/// Unlike the de Rham map this is defined for any $L^2$ form, but it does not
/// commute with $dif$ and it costs a global solve. See the module docs.
pub fn l2_projection<F: Sync + Section>(
  field: &F,
  whitney: WhitneyComplex,
  qr: Option<SimplexQuadRule>,
) -> Cochain {
  let grade = field.grade();
  let mass = whitney.mass(grade);
  let load = SourceForm::new(field, qr).assemble(whitney.topology(), whitney.geometry());

  // The mass is SPD only on a Riemannian geometry. There conjugate gradients
  // solves it far faster than a factorization, the mass is well conditioned
  // ($kappa = O(1)$, mesh-independent), so a fixed handful of Jacobi-CG
  // iterations suffices, with no fill. On an indefinite signature the mass is
  // symmetric non-degenerate but not definite, where CG does not apply and LU
  // carries the solve, keeping the projection total over every signature.
  let riemannian = whitney
    .topology()
    .cells()
    .handle_iter()
    .all(|cell| whitney.geometry().cell_metric(cell).is_riemannian());
  // The mass solve is the Riesz crossing back, $u = M^(-1) ell$: the load is a
  // functional on the Whitney space and the projection is an element of it.
  // A Krylov method works in coefficients, so the grading is spent here.
  let load = load.into_coeffs();
  let coeffs = if riemannian {
    cg(
      &mass,
      &Jacobi::new(&mass),
      &load,
      StopCriterion::rtol(1e-12),
    )
    .0
  } else {
    FaerLu::new(mass).solve(&load)
  };
  Cochain::new(grade, coeffs)
}
