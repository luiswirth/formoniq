//! The geometric multigrid V-cycle, one [`ApproxInverse`] built from a hierarchy
//! of levels.
//!
//! A single V-cycle is the composition, on each level from fine to coarse and
//! back, of a smoother $S approx A^(-1)$ (any [`ApproxInverse`], a few Jacobi
//! sweeps in the minimal case) with a coarse-grid correction: restrict the
//! residual, solve the coarser problem recursively, prolong the correction. At
//! the coarsest level a direct solve $C$ replaces the recursion. It is exactly
//! the [`Stationary`](crate::stationary) iteration with $B$ the cycle itself, so
//! it plays every role the crate's approximate inverses do: iterated alone it is
//! a solver, wrapped in [`cg`](crate::krylov::cg) a preconditioner.
//!
//! Its reason to exist over a one-level smoother is $h$-independence: the smoother
//! damps the high-frequency error a level resolves, the coarse correction handles
//! the low-frequency error it cannot see, and together they contract the whole
//! spectrum at a rate bounded below one uniformly in the mesh size. That
//! uniformity is what a stationary or a Jacobi-preconditioned Krylov iteration
//! lacks, and it is the property the cycle is validated against.
//!
//! The cycle here is geometric and generic: it asks only for the assembled
//! operator, a smoother, and the intergrid transfer matrices on each level. What
//! those transfers are, for FEEC, the Whitney prolongation and its
//! adjoint, is the consumer's business, supplied as plain
//! [`CsrMatrix`]es. This crate stays backend-free and knows nothing of meshes or
//! forms.

use crate::{ApproxInverse, CsrMatrix, Field, SelfAdjoint, Vector, adjoint, stationary};

/// One level of the hierarchy: its operator, its smoother, and the transfer to
/// the next-coarser level.
///
/// The transfer is the prolongation $P: V_"coarse" -> V_"fine"$, the
/// inclusion of the coarser space into this one, and the restriction is its
/// adjoint $P^H$, formed here rather than supplied. That the two are adjoint is
/// half of what makes a symmetric cycle self-adjoint, so it is a property of
/// the level and not a promise a caller keeps. The operator is this level's
/// $A$, and `smoother` any approximate inverse of it. The coarsest level
/// carries no transfer: it is the [`VCycle`]'s coarse solver, not a `Level`.
pub struct Level<S, T = f64> {
  operator: CsrMatrix<T>,
  smoother: S,
  prolong: CsrMatrix<T>,
  restrict: CsrMatrix<T>,
}

impl<S, T: Field> Level<S, T> {
  /// A level from its operator, smoother and the prolongation from the coarser
  /// level below it.
  pub fn new(operator: CsrMatrix<T>, smoother: S, prolong: CsrMatrix<T>) -> Self {
    debug_assert_eq!(
      operator.nrows(),
      operator.ncols(),
      "operator must be square"
    );
    debug_assert_eq!(
      prolong.nrows(),
      operator.nrows(),
      "prolongation maps into this level"
    );
    let restrict = adjoint(&prolong);
    Self {
      operator,
      smoother,
      prolong,
      restrict,
    }
  }
}

/// A multigrid V-cycle as an approximate inverse of the finest-level operator.
///
/// The levels run finest first; below the last one sits the coarse solver `C`,
/// an [`ApproxInverse`] of the coarsest operator (a direct factorization in the
/// minimal case, the exact inverse). One [`apply`](ApproxInverse::apply) runs a
/// single V-cycle: `sweeps` smoothing steps down each level, the recursion,
/// then the same number back up.
///
/// The cycle is symmetric, one count and not two, because the down-sweep and
/// the up-sweep are then mutual adjoints and the whole cycle inherits the
/// self-adjointness of its smoother. That is what lets it precondition
/// [`cg`](crate::krylov::cg), and unequal counts would produce a cycle whose
/// [`SelfAdjoint`] marker is false.
///
/// With no levels at all it degrades to the coarse solver alone, the totality
/// base case, a hierarchy of one grid being a plain direct solve with no
/// special-casing.
pub struct VCycle<S, C, T = f64> {
  levels: Vec<Level<S, T>>,
  coarse: C,
  sweeps: usize,
}

impl<T: Field, S: ApproxInverse<Space = Vector<T>>, C: ApproxInverse<Space = Vector<T>>>
  VCycle<S, C, T>
{
  /// A V-cycle over the levels, with `sweeps` smoothing steps on the way down
  /// and the same number on the way up.
  pub fn new(levels: Vec<Level<S, T>>, coarse: C, sweeps: usize) -> Self {
    Self {
      levels,
      coarse,
      sweeps,
    }
  }

  /// One V-cycle starting at level `i`, returning the approximate solution of
  /// `operator x = r` on that level.
  fn cycle(&self, i: usize, r: &Vector<T>) -> Vector<T> {
    let Some(level) = self.levels.get(i) else {
      return self.coarse.apply(r);
    };
    let mut x = Vector::zeros(level.operator.nrows());
    let smooth = |x: &mut Vector<T>| {
      stationary::sweeps(&level.operator, &level.smoother, r, x, self.sweeps);
    };
    smooth(&mut x);
    let residual = r - &level.operator * &x;
    let coarse_residual = &level.restrict * &residual;
    let correction = self.cycle(i + 1, &coarse_residual);
    x += &level.prolong * &correction;
    smooth(&mut x);
    x
  }
}

impl<T: Field, S: ApproxInverse<Space = Vector<T>>, C: ApproxInverse<Space = Vector<T>>>
  ApproxInverse for VCycle<S, C, T>
{
  type Space = Vector<T>;
  fn dim(&self) -> usize {
    self
      .levels
      .first()
      .map_or_else(|| self.coarse.dim(), |l| l.operator.nrows())
  }
  fn apply(&self, r: &Vector<T>) -> Vector<T> {
    self.cycle(0, r)
  }
}

/// Self-adjoint exactly when the smoother and the coarse solver are: the
/// down-sweep and the up-sweep are mutual adjoints, since the cycle carries one
/// sweep count, and the coarse correction $P C R = P C P^H$ is self-adjoint
/// whenever $C$ is, since a [`Level`] forms its restriction as the adjoint of
/// its prolongation.
///
/// Both of those are structural, so this marker rests on the markers of its
/// parts and on nothing a caller has to remember. As everywhere in the crate,
/// positive definiteness stays the constructor's promise.
impl<T: Field, S: SelfAdjoint<Space = Vector<T>>, C: SelfAdjoint<Space = Vector<T>>> SelfAdjoint
  for VCycle<S, C, T>
{
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testutil::{DenseInverse, csr, dense_solve};
  use crate::{Identity, Jacobi, StopCriterion, krylov::cg};
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
}
