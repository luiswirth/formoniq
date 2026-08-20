//! A geometric [`VCycle`] on the 1D Poisson problem: it contracts the
//! error uniformly in $h$, standalone and as a CG preconditioner.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

mod common;

use common::{DenseInverse, csr, dense_solve};
use iterative::krylov::cg;
use iterative::{ApproxInverse, CsrMatrix, Identity, Jacobi, Level, StopCriterion, VCycle, Vector};
use na::DMatrix;

/// The 1D Dirichlet Laplacian $"tridiag"(-1, 2, -1)$ on `n` interior points,
/// scaled by $h^(-2) = (n+1)^2$: the model second-order SPD operator whose
/// condition number grows like $h^(-2)$, so an $h$-independent iteration count
/// is a nontrivial claim.
fn laplacian_1d(n: usize) -> DMatrix<f64> {
  let h2_inv = ((n + 1) * (n + 1)) as f64;
  DMatrix::from_fn(n, n, |i, j| {
    if i == j {
      2.0 * h2_inv
    } else if i.abs_diff(j) == 1 {
      -h2_inv
    } else {
      0.0
    }
  })
}

/// Linear-interpolation prolongation from a coarse grid of `2^level - 1`
/// interior points to the fine grid of `2^(level+1) - 1`: each coarse point
/// maps to the fine point at the same location (weight 1) and its two fine
/// neighbors (weight 1/2). Its transpose is full-weighting restriction.
fn interpolation_1d(coarse: usize) -> CsrMatrix {
  let fine = 2 * coarse + 1;
  let mut coo = nas::CooMatrix::new(fine, coarse);
  for jc in 0..coarse {
    let jf = 2 * jc + 1; // coarse point jc sits at fine index 2 jc + 1
    coo.push(jf, jc, 1.0);
    coo.push(jf - 1, jc, 0.5);
    coo.push(jf + 1, jc, 0.5);
  }
  CsrMatrix::from(&coo)
}

/// Build a V-cycle for the 1D Laplacian tower of `levels` grids, Galerkin
/// coarse operators $A_c = P^T A P$, damped-Jacobi smoothing. Returns the
/// cycle and the finest operator.
fn poisson_vcycle(
  levels: usize,
  sweeps: usize,
) -> (VCycle<Jacobi, DenseInverse>, CsrMatrix, DMatrix<f64>) {
  // Finest grid has 2^levels - 1 points; coarsen by exact bisection.
  let fine_pts = (1 << levels) - 1;
  let a_fine_dense = laplacian_1d(fine_pts);
  let a_fine = csr(&a_fine_dense);

  let mut ops = vec![a_fine.clone()];
  let mut prolongs = Vec::new();
  for _ in 1..levels {
    let coarse_pts = (ops.last().unwrap().nrows() - 1) / 2;
    let p = interpolation_1d(coarse_pts);
    let pt = p.transpose();
    let a_coarse = &pt * &(ops.last().unwrap() * &p);
    prolongs.push(p);
    ops.push(a_coarse);
  }

  let level_structs: Vec<Level<Jacobi>> = (0..levels - 1)
    .map(|i| {
      let p = prolongs[i].clone();
      Level::new(ops[i].clone(), Jacobi::weighted(&ops[i], 2.0 / 3.0), p)
    })
    .collect();

  let coarsest = &ops[levels - 1];
  let coarse = DenseInverse::new(&DMatrix::from(coarsest));
  let cycle = VCycle::new(level_structs, coarse, sweeps);
  (cycle, a_fine, a_fine_dense)
}

/// The V-cycle used as a standalone stationary iteration contracts the error at
/// a rate bounded well below one, and that rate is essentially independent of
/// the mesh, the property that distinguishes multigrid from a one-level
/// smoother. Measured as the asymptotic residual reduction factor on two grids
/// an octave apart.
#[test]
fn vcycle_contracts_uniformly_in_h() {
  let factor = |levels: usize| -> f64 {
    let (cycle, a, _) = poisson_vcycle(levels, 2);
    let n = a.nrows();
    let x_true = Vector::from_fn(n, |i, _| (i as f64 + 1.0).sin());
    let b = &a * &x_true;
    // Stationary V-cycle iteration; read the reduction factor once transients
    // have died out.
    let mut x = Vector::zeros(n);
    let mut prev = f64::INFINITY;
    let mut factor = 0.0;
    for _ in 0..30 {
      x += cycle.apply(&(&b - &a * &x));
      let res = (&b - &a * &x).norm();
      factor = res / prev;
      prev = res;
      if res < 1e-11 * b.norm() {
        break;
      }
    }
    factor
  };
  let coarse = factor(4); // 15 points
  let fine = factor(7); //  127 points
  assert!(coarse < 0.4, "two-grid rate {coarse} not a contraction");
  assert!(fine < 0.4, "finer rate {fine} not a contraction");
  // h-independence: the rate must not degrade as the mesh is refined.
  assert!(
    fine < coarse + 0.1,
    "rate degraded under refinement: {coarse} -> {fine}"
  );
}

/// A V-cycle preconditions CG: it is self-adjoint, so the bound accepts it, and
/// MG-CG reaches the same solution as plain CG in far fewer iterations, that
/// count staying bounded as the mesh refines.
#[test]
fn vcycle_preconditioned_cg_is_mesh_independent() {
  let solve = |levels: usize| -> (usize, usize) {
    let (cycle, a, a_dense) = poisson_vcycle(levels, 2);
    let n = a.nrows();
    let x_true = Vector::from_fn(n, |i, _| (i as f64 - 3.0).tanh());
    let b = &a * &x_true;
    let stop = StopCriterion::rtol(1e-10);
    let (x_mg, rep_mg) = cg(&a, &cycle, &b, stop);
    let (_, rep_plain) = cg(&a, &Identity::new(n), &b, stop);
    assert!(rep_mg.converged);
    assert!(
      (x_mg - dense_solve(&a_dense, &b)).norm() < 1e-7,
      "MG-CG solution wrong at levels = {levels}"
    );
    (rep_mg.iters, rep_plain.iters)
  };
  let (mg_coarse, plain_coarse) = solve(4);
  let (mg_fine, plain_fine) = solve(7);
  // MG-CG iterations stay flat while plain CG grows with the condition number.
  assert!(mg_fine <= mg_coarse + 2, "{mg_coarse} -> {mg_fine} grew");
  assert!(
    plain_fine > 2 * plain_coarse,
    "plain CG should grow with h: {plain_coarse} -> {plain_fine}"
  );
  assert!(mg_fine < plain_fine / 4, "MG not beating plain CG");
}
