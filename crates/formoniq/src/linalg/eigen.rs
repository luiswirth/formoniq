use super::faer::FaerLu;
use simplicial::linalg::{CooMatrix, CsrMatrix, Matrix, Vector};

use std::fmt;

/// Why [`sparse_shift_invert_eigen`] could not produce the wanted eigenpairs.
///
/// Not every failure is a bug in the caller's pencil: exceeding the iteration
/// budget says only that this budget was too small, which an interactive
/// caller can report and move on from.
#[derive(Debug, Clone, PartialEq)]
pub enum EigenError {
  /// $B$'s positive part is trivial, so the pencil has no finite eigenvalue.
  NoFiniteEigenvalue,
  /// $A - "shift" B$ stayed singular, or too ill-conditioned to solve with,
  /// under every perturbation of the shift tried.
  SingularPencil { shift: f64 },
  /// The restart budget ran out with the worst wanted pair still at
  /// `residual`.
  NotConverged { shift: f64, residual: f64 },
}
impl fmt::Display for EigenError {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    match self {
      Self::NoFiniteEigenvalue => write!(f, "B has no positive-seminorm direction"),
      Self::SingularPencil { shift } => write!(
        f,
        "A - {shift}*B is singular under every shift perturbation"
      ),
      Self::NotConverged { shift, residual } => {
        write!(
          f,
          "no convergence near shift {shift}; worst residual {residual:e}"
        )
      }
    }
  }
}
impl std::error::Error for EigenError {}

/// The `k` eigenpairs of $A x = lambda B x$ ($A$, $B$ symmetric, $B$ PSD,
/// possibly singular) nearest `shift`, ascending by eigenvalue, with
/// $B$-orthonormal eigenvectors. Fewer than `k` when $B$'s positive part
/// cannot support `k` independent directions.
///
/// Spectral-transformation Lanczos (Ericsson-Ruhe): the wanted eigenpairs of
/// the pencil are the extremal eigenpairs of $T = (A - "shift" B)^(-1) B$,
/// which is self-adjoint in the $B$-seminorm even for singular $B$ — the
/// structure that makes the mixed Hodge-Laplace pencil's rank-deficient mass
/// block harmless, since null directions of $B$ carry zero seminorm and are
/// never mistaken for an eigenvector. Block-started with block size $k$, so an
/// eigenvalue of multiplicity up to $k$ — the harmonic space at `shift = 0` —
/// is resolved in full rather than collapsed onto one direction. Restarted
/// from the best Ritz vectors on hitting the Krylov cap, with full
/// reorthogonalization throughout.
///
/// A raw Ritz vector's $ker B$ component is arbitrary, the $B$-inner products
/// the recurrence computes being blind to it. One more application of $T$
/// recovers it through $M$'s off-diagonal coupling.
pub fn sparse_shift_invert_eigen(
  a: &CsrMatrix,
  b: &CsrMatrix,
  shift: f64,
  k: usize,
) -> Result<(Vector, Matrix), EigenError> {
  let n = a.nrows();
  assert_eq!(a.nrows(), a.ncols());
  assert_eq!(b.nrows(), n);
  assert_eq!(b.ncols(), n);

  let k = k.min(n);
  if k == 0 {
    return Ok((Vector::zeros(0), Matrix::zeros(n, 0)));
  }

  let a_norm = inf_norm(a);
  let b_norm = inf_norm(b);
  let (lu, used_shift) = factor_with_retry(a, b, shift, a_norm)?;

  let target_dim = (4 * k).max(2 * k + 20).min(n);
  let (mut basis, mut bbasis) = seed_block(n, k, b)?;

  // The basis overhangs `target_dim` by a block's worth of still-unexpanded
  // leftovers, and a restart reseeds with at most `2k` — a bound independent
  // of how many cycles have run.
  let proj_cap = (target_dim + 2 * k).min(n);
  let mut proj = Matrix::zeros(proj_cap, proj_cap);
  let mut dim = 0usize;

  // The backward error accepted per pair: the returned `x` is exact for a
  // pencil perturbed by this much relative to $||A||$ and $||B||$. The floor
  // is set by the shift-invert solve, accurate to $kappa(A - sigma B)
  // epsilon$; on the mixed KKT pencil that conditioning grows like $h^(-2)$,
  // so a tighter demand sits below the achievable backward error and only
  // spins restarts.
  const RESIDUAL_TOL: f64 = 1e-9;
  const MAX_RESTART_CYCLES: usize = 100;

  let mut worst = f64::INFINITY;
  for _cycle in 0..=MAX_RESTART_CYCLES {
    // Breakdown on one vector leaves the block's other, still-unexpanded
    // siblings independently valid; only running out of vectors to expand is
    // true exhaustion of the reachable invariant subspace.
    while dim < target_dim && dim < basis.len() {
      expand(&mut basis, &mut bbasis, &mut proj, dim, &lu, b);
      dim += 1;
    }
    let exhausted = dim < target_dim;

    let block = proj.view_range(0..dim, 0..dim).into_owned();
    let (theta, s) = dense_self_adjoint_eigen(&block);

    // Largest |theta| <=> lambda = used_shift + 1/theta closest to the shift.
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&i, &j| theta[j].abs().total_cmp(&theta[i].abs()));

    let take = k.min(dim);
    let pairs: Vec<(f64, Vector, f64)> = {
      let ritz = |idx: usize| -> (f64, Vector, f64) {
        let lambda = used_shift + 1.0 / theta[idx];
        let y = combine(&basis, |l| s[(l, idx)], dim);
        let raw_res = residual(a, b, lambda, &y, a_norm, b_norm);
        // Refine only when the raw pair misses: on a nonsingular $B$ it is
        // already the answer, and one $T$-application on a converged pair only
        // reintroduces the solve's own error.
        if raw_res <= RESIDUAL_TOL {
          return (lambda, y, raw_res);
        }
        let mut x = lu.solve(&(b * &y));
        let bnorm = x.dot(&(b * &x)).sqrt();
        if bnorm > 0.0 {
          x /= bnorm;
        }
        let res = residual(a, b, lambda, &x, a_norm, b_norm);
        (lambda, x, res)
      };
      order[..take].iter().map(|&i| ritz(i)).collect()
    };

    worst = pairs.iter().map(|p| p.2).fold(0.0_f64, f64::max);

    // Exhaustion means the Krylov space reachable from this block is complete:
    // nothing further is available, so return what is in hand rather than
    // looping — total on the degenerate boundary.
    if worst <= RESIDUAL_TOL || exhausted {
      let mut pairs = pairs;
      pairs.sort_by(|p, q| p.0.total_cmp(&q.0));
      let eigenvals = Vector::from_iterator(pairs.len(), pairs.iter().map(|p| p.0));
      let eigenvecs = Matrix::from_fn(n, pairs.len(), |i, j| pairs[j].1[i]);
      return Ok((eigenvals, eigenvecs));
    }

    // Restart from the best Ritz vectors alone, letting the same expansion
    // loop rediscover the projected operator and fresh residual directions.
    // Dropping the old leftovers is what keeps the basis's overhang past `dim`
    // bounded across arbitrarily many cycles.
    let keep = (2 * k).min(target_dim.saturating_sub(1)).max(1);

    let mut new_basis = Vec::with_capacity(keep);
    let mut new_bbasis = Vec::with_capacity(keep);
    for &idx in &order[..keep] {
      new_basis.push(combine(&basis, |l| s[(l, idx)], dim));
      new_bbasis.push(combine(&bbasis, |l| s[(l, idx)], dim));
    }

    basis = new_basis;
    bbasis = new_bbasis;
    proj = Matrix::zeros(proj_cap, proj_cap);
    dim = 0;
  }

  Err(EigenError::NotConverged {
    shift,
    residual: worst,
  })
}

/// Applies $T = M^(-1) B$ to `basis[idx]`, reorthogonalizes against the whole
/// basis (not just up to `idx`: siblings past it still await their own
/// expansion), fills row/column `idx` of `proj`, and appends the
/// $B$-normalized residual — unless its $B$-seminorm has broken down, in which
/// case `proj` is still complete for `idx` and nothing is appended.
fn expand(
  basis: &mut Vec<Vector>,
  bbasis: &mut Vec<Vector>,
  proj: &mut Matrix,
  idx: usize,
  lu: &FaerLu,
  b: &CsrMatrix,
) {
  let mut w = lu.solve(&bbasis[idx]);

  let h = b_orthogonalize(&mut w, basis, bbasis);
  for (j, &hj) in h.iter().enumerate() {
    proj[(j, idx)] = hj;
    proj[(idx, j)] = hj;
  }

  let bw = b * &w;
  let beta_sq = w.dot(&bw);
  const BREAKDOWN_TOL_SQ: f64 = 1e-20;
  if beta_sq <= BREAKDOWN_TOL_SQ {
    return;
  }
  let beta = beta_sq.sqrt();
  basis.push(&w / beta);
  bbasis.push(&bw / beta);
}

/// `bs` mutually $B$-orthonormal deterministic pseudo-random seed vectors
/// (fewer, if $B$'s positive part cannot support `bs` independent directions):
/// the starting block, whose width is the largest eigenvalue multiplicity the
/// solve can resolve in one pass.
fn seed_block(
  n: usize,
  bs: usize,
  b: &CsrMatrix,
) -> Result<(Vec<Vector>, Vec<Vector>), EigenError> {
  let mut basis = Vec::with_capacity(bs);
  let mut bbasis = Vec::with_capacity(bs);
  let mut seed = 0u64;
  while basis.len() < bs && seed < bs as u64 * 32 + 32 {
    let mut v = Vector::from_iterator(n, (0..n).map(|i| pseudo_random(seed, i as u64)));
    seed += 1;
    b_orthogonalize(&mut v, &basis, &bbasis);
    let bv = b * &v;
    let norm_sq: f64 = v.dot(&bv);
    const SEED_TOL: f64 = 1e-24;
    if norm_sq > SEED_TOL {
      let norm = norm_sq.sqrt();
      basis.push(&v / norm);
      bbasis.push(&bv / norm);
    }
  }
  if basis.is_empty() {
    return Err(EigenError::NoFiniteEigenvalue);
  }
  Ok((basis, bbasis))
}

/// Two passes of modified Gram-Schmidt ("twice is enough") against a
/// $B$-orthonormal basis, returning the accumulated coefficients: the
/// projection of `v` onto the basis, which is the row of the projected operator
/// where the caller wants one.
fn b_orthogonalize(v: &mut Vector, basis: &[Vector], bbasis: &[Vector]) -> Vec<f64> {
  let mut coeffs = vec![0.0; basis.len()];
  for _pass in 0..2 {
    for (j, (vj, bvj)) in basis.iter().zip(bbasis).enumerate() {
      let c = v.dot(bvj);
      coeffs[j] += c;
      v.axpy(-c, vj, 1.0);
    }
  }
  coeffs
}

/// A deterministic splitmix64-style fill, so the solve is reproducible without
/// pulling in a `rand` dependency for this one internal use.
fn pseudo_random(seed: u64, index: u64) -> f64 {
  let mut z = seed
    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
    .wrapping_add(index.wrapping_mul(0xD1B5_4A32_D192_ED03))
    .wrapping_add(0x9E37_79B9_7F4A_7C15);
  z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
  z ^= z >> 31;
  (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn combine(vecs: &[Vector], coeff: impl Fn(usize) -> f64, dim: usize) -> Vector {
  let mut out = Vector::zeros(vecs[0].nrows());
  for (l, v) in vecs.iter().enumerate().take(dim) {
    out.axpy(coeff(l), v, 1.0);
  }
  out
}

/// The backward error of the pair $(lambda, x)$: the relative size of the
/// smallest perturbation of $(A, B)$ making it exact.
fn residual(
  a: &CsrMatrix,
  b: &CsrMatrix,
  lambda: f64,
  x: &Vector,
  a_norm: f64,
  b_norm: f64,
) -> f64 {
  let ax = a * x;
  let bx = b * x;
  let r = &ax - lambda * &bx;
  let xnorm = x.norm();
  let scale = a_norm * xnorm + lambda.abs() * b_norm * xnorm;
  if scale > 0.0 {
    r.norm() / scale
  } else {
    r.norm()
  }
}

fn inf_norm(m: &CsrMatrix) -> f64 {
  let mut row_sums = vec![0.0; m.nrows()];
  for (r, _, &v) in m.triplet_iter() {
    row_sums[r] += v.abs();
  }
  row_sums.into_iter().fold(0.0_f64, f64::max)
}

/// The shift `attempt` tries: the requested one first, then a geometrically
/// growing perturbation alternating in sign, so the search stays centered on
/// the target instead of drifting off one side.
fn perturbed_shift(shift: f64, eps0: f64, attempt: usize) -> f64 {
  if attempt == 0 {
    return shift;
  }
  let step = eps0 * 2f64.powi(((attempt - 1) / 2) as i32);
  if attempt % 2 == 1 {
    shift + step
  } else {
    shift - step
  }
}

/// $M = A - "shift" B$, factored via sparse LU. Retries with a perturbed shift
/// on a singular factorization — generic only when `shift` lands exactly on an
/// eigenvalue, which `shift = 0` does whenever the pencil has a harmonic
/// space.
fn factor_with_retry(
  a: &CsrMatrix,
  b: &CsrMatrix,
  shift: f64,
  a_norm: f64,
) -> Result<(FaerLu, f64), EigenError> {
  const MAX_ATTEMPTS: usize = 16;
  let eps0 = f64::EPSILON.sqrt() * a_norm.max(1.0);
  for attempt in 0..MAX_ATTEMPTS {
    let current_shift = perturbed_shift(shift, eps0, attempt);
    let m = shifted_matrix(a, b, current_shift);
    if let Some(lu) = FaerLu::try_new(m.clone()) {
      // A shift landing near an eigenvalue leaves an M that factors but
      // amplifies noise into the near-null direction. The forward residual
      // cannot see this (M nearly annihilates that direction); a round trip
      // can, recovering a probe to about $kappa(M) epsilon$ — so this rejects
      // $kappa(M) gt.tilde 10^10$, past which the solve's error swamps the
      // eigenvector it is meant to produce.
      let probe = Vector::from_iterator(
        m.nrows(),
        (0..m.nrows()).map(|i| pseudo_random(!0, i as u64)),
      );
      let rhs = &m * &probe;
      let resolved = lu.solve(&rhs);
      if (&resolved - &probe).norm() <= 1e-6 * probe.norm().max(1.0) {
        return Ok((lu, current_shift));
      }
    }
  }
  Err(EigenError::SingularPencil { shift })
}

fn shifted_matrix(a: &CsrMatrix, b: &CsrMatrix, shift: f64) -> CsrMatrix {
  let mut coo = CooMatrix::new(a.nrows(), a.ncols());
  for (r, c, &v) in a.triplet_iter() {
    coo.push(r, c, v);
  }
  for (r, c, &v) in b.triplet_iter() {
    coo.push(r, c, -shift * v);
  }
  CsrMatrix::from(&coo)
}

fn dense_self_adjoint_eigen(m: &Matrix) -> (Vector, Matrix) {
  let n = m.nrows();
  let fm = faer::Mat::from_fn(n, n, |i, j| m[(i, j)]);
  let eig = fm.self_adjoint_eigen(faer::Side::Lower).unwrap();
  let vals = eig.S().column_vector();
  let vecs = eig.U();
  let eigenvals = Vector::from_iterator(n, (0..n).map(|i| *vals.get(i)));
  let eigenvecs = Matrix::from_fn(n, n, |i, j| *vecs.get(i, j));
  (eigenvals, eigenvecs)
}
