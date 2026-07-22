//! The three maps from $L^2 Lambda^k$ into the Whitney space, and the error
//! between them.
//!
//! A differential form on the manifold reaches the discrete space by one of
//! three routes, and they are genuinely different maps:
//!
//! - $W: C^k -> cal(W) Lambda^k$, the Whitney *interpolation*
//!   ([`WhitneyInterpolant`]): the reconstruction of a form from a cochain, the
//!   right inverse of $R$.
//! - $R: L^2 Lambda^k -> C^k$, the *de Rham map*
//!   ([`derham_map`](derham::project::derham_map)): integration over the
//!   simplices. Canonical and metric-free, and a cochain map --
//!   $R compose dif = dif compose R$ -- which is exactly why the discrete
//!   complex inherits the cohomology of the continuous one. It needs the
//!   traces to exist, so it is not defined on all of $L^2 Lambda^k$.
//! - $P_h: L^2 Lambda^k -> cal(W) Lambda^k$, the $L^2$ *projection*
//!   ([`l2_projection`]): the best approximation in the energy norm, defined on
//!   all of $L^2 Lambda^k$ but *not* commuting with $dif$, and requiring a
//!   global mass solve rather than local integration.
//!
//! $R$ is the one the theory is built on; $P_h$ is the one that is optimal in
//! norm. Neither dominates the other, and the discrete complex is exact only
//! through $R$.

use {
  crate::linalg::faer::FaerLu,
  derham::{cochain::Cochain, interpolate::interpolant::WhitneyInterpolant, section::Section},
  exterior::{Covariant, multiform_gramian},
  iterative::{Jacobi, StopCriterion, krylov::cg},
  simplicial::{
    atlas::{MeshPoint, SimplexQuadRule},
    geometry::{cell_volume, metric::mesh::MeshLengthsSq},
    linalg::{CsrMatrix, Vector},
    topology::complex::Complex,
  },
};

use crate::{assemble::assemble_galvec, operators::SourceElVec, whitney_complex::WhitneyComplex};

/// The $L^2 Lambda^k$ error $norm(omega - W c)_(L^2)$ between an exact form on
/// the manifold and the Whitney reconstruction of a cochain.
///
/// Intrinsic: the pointwise difference is measured in the reference frame of
/// each cell by the induced inner product $Lambda^k g^(-1)$ of that cell's
/// metric. On a curved (embedded) mesh this is the only correct thing to do --
/// the flat ambient Gramian would measure the wrong norm.
pub fn fe_l2_error<F: Section<Covariant>>(
  fe_cochain: &Cochain,
  exact: &F,
  topology: &Complex,
  geometry: &MeshLengthsSq,
) -> f64 {
  let dim = topology.dim();
  let grade = fe_cochain.grade();
  let qr = SimplexQuadRule::degree(dim, 3);
  let fe_whitney = WhitneyInterpolant::new(fe_cochain.clone(), topology);

  let error_sq: f64 = topology
    .cells()
    .handle_iter()
    .map(|cell| {
      let metric = geometry.cell_metric(cell);
      let inner = multiform_gramian(&metric, grade);
      let error_pointwise =
        |point: &MeshPoint| inner.norm_sq((exact.at(point) - fe_whitney.at(point)).coeffs());
      qr.integrate_cell(cell.idx(), &error_pointwise, cell_volume(&metric))
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
pub fn l2_projection<F: Sync + Section<Covariant>>(
  field: &F,
  whitney: WhitneyComplex,
  qr: Option<SimplexQuadRule>,
) -> Cochain {
  let grade = field.grade();
  let mass = CsrMatrix::from(&whitney.mass(grade));
  let load: Vector = assemble_galvec(
    whitney.topology(),
    whitney.geometry(),
    SourceElVec::new(field, qr),
  );

  // The mass is SPD only on a Riemannian geometry. There conjugate gradients
  // solves it far faster than a factorization -- the mass is well conditioned
  // ($kappa = O(1)$, mesh-independent), so a fixed handful of Jacobi-CG
  // iterations suffices, with no fill. On an indefinite signature the mass is
  // symmetric non-degenerate but not definite, where CG does not apply and LU
  // carries the solve, keeping the projection total over every signature.
  let riemannian = whitney
    .topology()
    .cells()
    .handle_iter()
    .all(|cell| whitney.geometry().cell_metric(cell).is_riemannian());
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

#[cfg(test)]
mod test {
  use super::*;

  use derham::section::CoordFieldExt;
  use glatt::field::DiffFormClosure;
  use simplicial::mesher::cartesian::CartesianGrid;

  use crate::linalg::faer::FaerCholesky;
  use approx::assert_relative_eq;

  /// $P_h compose W = id$: the $L^2$ projection is the identity on the Whitney
  /// space, since a discrete form is its own best approximation.
  ///
  /// The sharpest available check that the Hodge mass matrix and the source
  /// load are the same bilinear form seen from two sides.
  #[test]
  fn l2_projection_reproduces_whitney_forms() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &lengths);

      for grade in 0..=dim {
        let ndofs = topology.nsimplices(grade);
        let cochain = Cochain::new(
          grade,
          Vector::from_iterator(ndofs, (0..ndofs).map(|i| ((i % 7) as f64) - 3.0)),
        );

        let field = WhitneyInterpolant::new(cochain.clone(), &topology);
        let qr = SimplexQuadRule::degree(dim, 3);
        let projected = l2_projection(&field, whitney, Some(qr));

        assert_relative_eq!(projected.coeffs(), cochain.coeffs(), epsilon = 1e-9);
      }
    }
  }

  /// A form that lies in the Whitney space is reproduced exactly by the
  /// projection, and hence has zero $L^2$ error against it: $W$ and $P_h$
  /// agree wherever both are exact.
  #[test]
  fn l2_error_vanishes_on_the_discrete_space() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &lengths);

      // A globally affine 0-form lies in the Whitney 0-form space.
      let exact = DiffFormClosure::coord_component(0, dim);
      let exact = exact.pullback_on(&topology, &coords);
      let projected = l2_projection(&exact, whitney, Some(SimplexQuadRule::degree(dim, 3)));

      let error = fe_l2_error(&projected, &exact, &topology, &lengths);
      assert!(error < 1e-9, "dim={dim} error={error}");
    }
  }

  /// The Whitney mass matrix is SPD on a Riemannian geometry, so the mass solve
  /// $M c = b$ is a genuine target for conjugate gradients. This pins that the
  /// iterative solve agrees with the direct Cholesky factorization to solver
  /// tolerance, swept over dimension and grade --- the correctness half of
  /// wiring `iterative` against a real FEEC operator.
  #[test]
  fn cg_mass_solve_matches_cholesky() {
    use iterative::{Jacobi, StopCriterion, krylov::cg};

    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
      let lengths = coords.to_edge_lengths_sq(&topology);
      let whitney = WhitneyComplex::new(&topology, &lengths);

      for grade in 0..=dim {
        let mass = CsrMatrix::from(&whitney.mass(grade));
        let n = mass.nrows();
        let b = Vector::from_fn(n, |i, _| ((i % 5) as f64 - 2.0) * 0.5);

        let direct = FaerCholesky::new(mass.clone()).solve(&b);
        let (iter, report) = cg(&mass, &Jacobi::new(&mass), &b, StopCriterion::rtol(1e-12));

        assert!(report.converged, "dim={dim} grade={grade} did not converge");
        assert!(
          (&iter - &direct).norm() < 1e-9,
          "dim={dim} grade={grade}: cg vs cholesky differ by {}",
          (&iter - &direct).norm()
        );
      }
    }
  }

  /// Bench, not an assertion: on a well-conditioned mass matrix ($kappa = O(1)$,
  /// mesh-independent) Jacobi-CG converges in a fixed handful of iterations, so
  /// it competes with the direct factorizations without their fill. Run with
  /// `cargo test -p formoniq --release bench_mass_solve -- --nocapture --ignored`.
  #[test]
  #[ignore = "timing bench, run explicitly with --nocapture"]
  fn bench_mass_solve() {
    use iterative::{Jacobi, StopCriterion, krylov::cg};
    use std::time::Instant;

    let dim = 3;
    let (topology, coords) = CartesianGrid::new_unit(dim, 12).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for grade in 0..=dim {
      let mass = CsrMatrix::from(&whitney.mass(grade));
      let n = mass.nrows();
      let b = Vector::from_fn(n, |i, _| ((i % 5) as f64 - 2.0) * 0.5);

      let t = Instant::now();
      let x_lu = FaerLu::new(mass.clone()).solve(&b);
      let t_lu = t.elapsed();

      let t = Instant::now();
      let x_ch = FaerCholesky::new(mass.clone()).solve(&b);
      let t_ch = t.elapsed();

      let precond = Jacobi::new(&mass);
      let t = Instant::now();
      let (x_cg, report) = cg(&mass, &precond, &b, StopCriterion::rtol(1e-10));
      let t_cg = t.elapsed();

      eprintln!(
        "grade {grade}: n={n:>6}  LU {t_lu:>10.2?}  Chol {t_ch:>10.2?}  \
         CG(Jacobi) {t_cg:>10.2?} in {} iters   (agree {:.1e})",
        report.iters,
        (&x_cg - &x_ch).norm().max((&x_lu - &x_ch).norm()),
      );
    }
  }
}
