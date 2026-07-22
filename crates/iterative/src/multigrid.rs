//! The geometric multigrid V-cycle, one [`ApproxInverse`] built from a hierarchy
//! of levels.
//!
//! A single V-cycle is the composition, on each level from fine to coarse and
//! back, of a *smoother* $S approx A^(-1)$ (any [`ApproxInverse`], a few Jacobi
//! sweeps in the minimal case) with a coarse-grid *correction*: restrict the
//! residual, solve the coarser problem recursively, prolong the correction. At
//! the coarsest level a direct solve $C$ replaces the recursion. It is exactly
//! the [`Stationary`](crate::stationary) iteration with $B$ the cycle itself, so
//! it plays every role the crate's approximate inverses do: iterated alone it is
//! a solver, wrapped in [`cg`](crate::krylov::cg) a preconditioner.
//!
//! Its reason to exist over a one-level smoother is $h$-independence: the smoother
//! damps the high-frequency error a level resolves, the coarse correction handles
//! the low-frequency error it cannot see, and together they contract the whole
//! spectrum at a rate bounded below one *uniformly in the mesh size*. That
//! uniformity is what a stationary or a Jacobi-preconditioned Krylov iteration
//! lacks, and it is the property the cycle is validated against.
//!
//! The cycle here is geometric and generic: it asks only for the assembled
//! operator, a smoother, and the intergrid transfer matrices on each level. What
//! those transfers *are* --- for FEEC, the Whitney prolongation and its
//! transpose --- is the consumer's business, supplied as plain
//! [`CsrMatrix`]es; this crate stays backend-free and knows nothing of meshes or
//! forms.

use crate::{ApproxInverse, CsrMatrix, SelfAdjoint, Vector};

/// One level of the hierarchy: its operator, its smoother, and the transfers to
/// the next-coarser level.
///
/// `prolong` is $P: RR^(n_"coarse") -> RR^(n_"fine")$, the inclusion of the
/// coarser space into this one; `restrict` is $R: RR^(n_"fine") ->
/// RR^(n_"coarse")$, typically $P^T$. The operator is this level's $A$, and
/// `smoother` any approximate inverse of it. The coarsest level carries no
/// transfers --- it is the [`VCycle`]'s coarse solver, not a `Level`.
pub struct Level<S> {
  operator: CsrMatrix,
  smoother: S,
  prolong: CsrMatrix,
  restrict: CsrMatrix,
}

impl<S> Level<S> {
  /// A level from its operator, smoother and the two transfers to the coarser
  /// level below it.
  pub fn new(operator: CsrMatrix, smoother: S, prolong: CsrMatrix, restrict: CsrMatrix) -> Self {
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
    debug_assert_eq!(
      restrict.ncols(),
      operator.nrows(),
      "restriction maps out of this level"
    );
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
/// single V-cycle: `pre` smoothing sweeps down each level, the recursion, then
/// `post` sweeps back up.
///
/// With no levels at all it degrades to the coarse solver alone --- the totality
/// base case, a hierarchy of one grid being a plain direct solve with no
/// special-casing.
pub struct VCycle<S, C> {
  levels: Vec<Level<S>>,
  coarse: C,
  pre: usize,
  post: usize,
}

impl<S: ApproxInverse, C: ApproxInverse> VCycle<S, C> {
  /// A V-cycle with independent pre- and post-smoothing counts.
  pub fn new(levels: Vec<Level<S>>, coarse: C, pre: usize, post: usize) -> Self {
    Self {
      levels,
      coarse,
      pre,
      post,
    }
  }

  /// A symmetric V-cycle: `sweeps` pre-smoothing and the same number of
  /// post-smoothing steps.
  ///
  /// Equal counts with a self-adjoint smoother and $R = P^T$ make the whole
  /// cycle self-adjoint, the condition under which it may precondition
  /// [`cg`](crate::krylov::cg) --- see the [`SelfAdjoint`] impl.
  pub fn symmetric(levels: Vec<Level<S>>, coarse: C, sweeps: usize) -> Self {
    Self::new(levels, coarse, sweeps, sweeps)
  }

  /// One V-cycle starting at level `i`, returning the approximate solution of
  /// `operator x = r` on that level.
  fn cycle(&self, i: usize, r: &Vector) -> Vector {
    let Some(level) = self.levels.get(i) else {
      return self.coarse.apply(r);
    };
    let mut x = Vector::zeros(level.operator.nrows());
    smooth(&level.operator, &level.smoother, r, &mut x, self.pre);
    let residual = r - &level.operator * &x;
    let coarse_residual = &level.restrict * &residual;
    let correction = self.cycle(i + 1, &coarse_residual);
    x += &level.prolong * &correction;
    smooth(&level.operator, &level.smoother, r, &mut x, self.post);
    x
  }
}

/// Refine `x` toward solving `a x = r` by `sweeps` stationary steps of the
/// smoother, continuing from the incoming `x` rather than restarting --- which
/// is what post-smoothing after the coarse correction needs.
fn smooth<S: ApproxInverse>(a: &CsrMatrix, s: &S, r: &Vector, x: &mut Vector, sweeps: usize) {
  for _ in 0..sweeps {
    let residual = r - a * &*x;
    *x += s.apply(&residual);
  }
}

impl<S: ApproxInverse, C: ApproxInverse> ApproxInverse for VCycle<S, C> {
  fn dim(&self) -> usize {
    self
      .levels
      .first()
      .map_or_else(|| self.coarse.dim(), |l| l.operator.nrows())
  }
  fn apply(&self, r: &Vector) -> Vector {
    self.cycle(0, r)
  }
}

/// Self-adjoint when the smoother and coarse solver are and the cycle is
/// symmetric ($"pre" = "post"$, with $R = P^T$ on every level). A symmetric
/// V-cycle of a symmetric operator is a symmetric operator: the down-sweep and
/// up-sweep are mutual adjoints and the coarse correction $P C R = P C P^T$ is
/// self-adjoint whenever $C$ is.
///
/// As everywhere in the crate, self-adjointness is the marker and positive
/// definiteness the constructor's promise --- here also that `pre == post`,
/// which [`VCycle::symmetric`] guarantees. It is what lets a V-cycle precondition
/// [`cg`](crate::krylov::cg).
impl<S: SelfAdjoint, C: SelfAdjoint> SelfAdjoint for VCycle<S, C> {}
