use super::faer::FaerLu;
use super::nalgebra::{CooMatrix, CsrMatrix, Matrix, Vector};

/// The `k` eigenpairs of $A x = lambda B x$ ($A$, $B$ symmetric, $B$ PSD,
/// possibly singular) nearest `shift`, ascending by eigenvalue, with
/// $B$-orthonormal eigenvectors.
///
/// Spectral-transformation Lanczos (Ericsson-Ruhe): the wanted eigenpairs of
/// the original pencil are the extremal eigenpairs of the shift-inverted
/// operator $T = (A - "shift" B)^(-1) B$, which is self-adjoint in the
/// $B$-seminorm $angle.l x, y angle.r_B = x^top B y$ even though $B$ may be
/// singular — the exact structure that makes the mixed Hodge-Laplace pencil's
/// rank-deficient mass block harmless: null directions of $B$ carry zero
/// seminorm and are never mistaken for an eigenvector. Block-started, with
/// block size $k$, so a degenerate eigenvalue of multiplicity up to $k$ — the
/// harmonic space at `shift = 0` is exactly this case — is resolved in full,
/// not collapsed onto a single direction the way plain single-vector Lanczos
/// would. Thick-restarted once the Krylov dimension hits a cap, by simply
/// re-seeding a fresh expansion from the best Ritz vectors found so far:
/// self-adjointness of $T$ makes this warm-started restart rediscover the
/// same projected operator (and fresh residual directions) without a
/// hand-derived arrowhead correction. Full reorthogonalization throughout,
/// since the wanted eigenvalues cluster near the shift by construction.
///
/// Returned eigenvectors are $B$-seminorm-orthonormal directly whenever $B$
/// is (locally) nonsingular there; on a singular $B$, one more application of
/// $T$ recovers the correct null-space component before normalizing — the
/// null space is invisible to every $B$-inner-product the Lanczos recurrence
/// computes, so a raw Ritz combination leaves it arbitrary, and $T$'s own
/// action (via $M$'s off-diagonal coupling) is what fixes it.
///
/// Panics if $A - "shift" B$ stays singular (or so ill-conditioned that a
/// solve is numerically meaningless) after retrying with a perturbed shift,
/// or if convergence is not reached within the restart budget — diagnostic
/// panics matching this module's other primitives ([`FaerLu`],
/// `self_adjoint_eigen`), not a `Result`, since no caller here would
/// meaningfully recover.
pub fn sparse_shift_invert_eigen(
  a: &CsrMatrix,
  b: &CsrMatrix,
  shift: f64,
  k: usize,
) -> (Vector, Matrix) {
  let n = a.nrows();
  assert_eq!(a.nrows(), a.ncols());
  assert_eq!(b.nrows(), n);
  assert_eq!(b.ncols(), n);

  let k = k.min(n);
  if k == 0 {
    return (Vector::zeros(0), Matrix::zeros(n, 0));
  }

  let a_norm = inf_norm(a);
  let b_norm = inf_norm(b);
  let (lu, used_shift) = factor_with_retry(a, b, shift, a_norm);

  let target_dim = (4 * k).max(2 * k + 20).min(n);
  let (mut basis, mut bbasis) = seed_block(n, k, b);

  // `proj` needs headroom beyond `target_dim`: the basis transiently
  // overhangs it by a block's worth of still-unexpanded leftover vectors
  // (the steady-state queue a block-seeded expansion carries), and every
  // restart reseeds with at most `2k` vectors — a fixed bound independent of
  // how many restart cycles have run.
  let proj_cap = (target_dim + 2 * k).min(n);
  let mut proj = Matrix::zeros(proj_cap, proj_cap);
  let mut dim = 0usize;

  // Tight enough for the law tests' oracle comparisons, loose enough to be
  // reachable on an indefinite mixed-KKT pencil: driving for one more order
  // of magnitude there just cycles restarts without further improvement
  // until rounding accumulates into instability.
  const RESIDUAL_TOL: f64 = 1e-9;
  const MAX_RESTART_CYCLES: usize = 100;

  for _cycle in 0..=MAX_RESTART_CYCLES {
    // Breakdown on one vector (no residual left to extend the basis with)
    // does not mean the whole process is exhausted: a block-seeded or
    // restarted basis has other, still-unexpanded siblings whose own
    // diagonal/cross terms are independently valid and still need filling.
    // Only running out of existing vectors to expand is true exhaustion.
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
    // The raw Ritz combination is already a good eigenvector whenever B is
    // (locally) nonsingular. But the B-seminorm-orthonormal basis only ever
    // sees a vector's B-inner products, so on a singular B its null-space
    // component is arbitrary — exactly the sigma-component of a mixed
    // Hodge-Laplace eigenvector. Fix it up only when needed, with one more
    // application of T = M^(-1) B: T's own action fills in the null
    // component correctly via M's off-diagonal coupling. Refining
    // unconditionally would be wrong here too — within a degenerate cluster
    // (already well resolved) it collapses distinct Ritz directions onto each
    // other, since T acts as a near-identical scalar on the whole cluster.
    let ritz = |idx: usize| -> (f64, Vector) {
      let lambda = used_shift + 1.0 / theta[idx];
      let y = combine(&basis, |l| s[(l, idx)], dim);
      let raw_res = residual(a, b, lambda, &y, a_norm, b_norm);
      if raw_res <= RESIDUAL_TOL {
        return (lambda, y);
      }
      let mut x = lu.solve(&(b * &y));
      let bnorm = (x.dot(&(b * &x))).sqrt();
      if bnorm > 0.0 {
        x /= bnorm;
      }
      (lambda, x)
    };

    let converged = order[..take].iter().all(|&i| {
      let (lambda, x) = ritz(i);
      residual(a, b, lambda, &x, a_norm, b_norm) <= RESIDUAL_TOL
    });

    // Exhaustion means the Krylov space reachable from this starting block is
    // provably complete: there is nothing further the algorithm could do, so
    // the Ritz pairs in hand are returned regardless of the residual check —
    // total on the degenerate boundary rather than looping forever.
    if converged || exhausted {
      let mut pairs: Vec<(f64, Vector)> = order[..take].iter().map(|&i| ritz(i)).collect();
      pairs.sort_by(|p, q| p.0.total_cmp(&q.0));
      let eigenvals = Vector::from_iterator(pairs.len(), pairs.iter().map(|p| p.0));
      let eigenvecs = Matrix::from_fn(n, pairs.len(), |i, j| pairs[j].1[i]);
      return (eigenvals, eigenvecs);
    }

    // Thick restart: warm-start the next cycle from just the best Ritz
    // vectors found so far, and let the same expansion loop rediscover the
    // projected operator (and fresh residual directions) from scratch. Not
    // carrying the old leftover residual block forward keeps the basis's
    // transient overhang past `dim` bounded by a fixed multiple of `k` across
    // arbitrarily many restart cycles, rather than growing with each one.
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

  panic!(
    "sparse_shift_invert_eigen: exceeded {MAX_RESTART_CYCLES} restart cycles \
     converging {k} eigenpair(s) near shift {shift}"
  );
}

/// Applies $T = M^(-1) B$ to `basis[idx]`, reorthogonalizes against every
/// vector in `basis` so far (not just up to `idx`: a block-seeded or
/// restarted basis has siblings past `idx` still awaiting their own
/// expansion), and appends the $B$-normalized residual. Fills row/column
/// `idx` of `proj` unconditionally; returns `None` (breakdown, no vector
/// appended) when the residual's $B$-seminorm signals the reachable
/// invariant subspace is exhausted.
fn expand(
  basis: &mut Vec<Vector>,
  bbasis: &mut Vec<Vector>,
  proj: &mut Matrix,
  idx: usize,
  lu: &FaerLu,
  b: &CsrMatrix,
) -> Option<f64> {
  let mut w = lu.solve(&bbasis[idx]);

  let mut h = vec![0.0; basis.len()];
  // Two passes of modified Gram-Schmidt ("twice is enough") in the B-inner
  // product.
  for _pass in 0..2 {
    for (j, (vj, bvj)) in basis.iter().zip(bbasis.iter()).enumerate() {
      let c = w.dot(bvj);
      h[j] += c;
      w.axpy(-c, vj, 1.0);
    }
  }
  for (j, &hj) in h.iter().enumerate() {
    proj[(j, idx)] = hj;
    proj[(idx, j)] = hj;
  }

  let bw = b * &w;
  let beta_sq = w.dot(&bw);
  const BREAKDOWN_TOL_SQ: f64 = 1e-20;
  if beta_sq <= BREAKDOWN_TOL_SQ {
    return None;
  }
  let beta = beta_sq.sqrt();
  basis.push(&w / beta);
  bbasis.push(&bw / beta);
  Some(beta)
}

/// `bs` mutually $B$-orthonormal deterministic pseudo-random seed vectors
/// (fewer, if $B$'s positive part cannot support `bs` independent
/// directions): the starting block whose width sets the maximum eigenvalue
/// multiplicity the solve can resolve in one pass.
fn seed_block(n: usize, bs: usize, b: &CsrMatrix) -> (Vec<Vector>, Vec<Vector>) {
  let mut basis = Vec::with_capacity(bs);
  let mut bbasis = Vec::with_capacity(bs);
  let mut seed = 0u64;
  while basis.len() < bs && seed < bs as u64 * 32 + 32 {
    let mut v = Vector::from_iterator(n, (0..n).map(|i| pseudo_random(seed, i as u64)));
    seed += 1;
    for _pass in 0..2 {
      for (vj, bvj) in basis.iter().zip(bbasis.iter()) {
        let c: f64 = v.dot(bvj);
        v.axpy(-c, vj, 1.0);
      }
    }
    let bv = b * &v;
    let norm_sq: f64 = v.dot(&bv);
    const SEED_TOL: f64 = 1e-24;
    if norm_sq > SEED_TOL {
      let norm = norm_sq.sqrt();
      basis.push(&v / norm);
      bbasis.push(&bv / norm);
    }
  }
  assert!(
    !basis.is_empty(),
    "sparse_shift_invert_eigen: B has no reachable positive-seminorm direction; \
     the pencil has no finite eigenvalue to seek"
  );
  (basis, bbasis)
}

/// A deterministic splitmix64-style fill, so the solve is reproducible
/// without pulling in a `rand` dependency for this one internal use.
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

/// $M = A - "shift" B$, factored via sparse LU. Retries with a
/// scale-perturbed shift on singular factorization — generic only when
/// `shift` lands exactly on an eigenvalue, which `shift = 0` does whenever
/// the pencil has a harmonic space.
fn factor_with_retry(a: &CsrMatrix, b: &CsrMatrix, shift: f64, a_norm: f64) -> (FaerLu, f64) {
  const MAX_ATTEMPTS: usize = 16;
  let eps0 = f64::EPSILON.sqrt() * a_norm.max(1.0);
  let mut current_shift = shift;
  for attempt in 0..MAX_ATTEMPTS {
    let m = shifted_matrix(a, b, current_shift);
    if let Some(lu) = FaerLu::try_new(m.clone()) {
      // A merely non-singular-in-exact-arithmetic M can still be
      // catastrophically ill-conditioned in floating point — `shift`
      // landing so close to an eigenvalue that `sp_lu` "succeeds" but
      // amplifies noise into the near-null direction. The forward residual
      // `M x - rhs` stays small even then (M nearly annihilates that
      // direction), so it can't see the problem; round-tripping a probe
      // vector through solve and checking it comes back unchanged can.
      let probe = Vector::from_iterator(
        m.nrows(),
        (0..m.nrows()).map(|i| pseudo_random(!0, i as u64)),
      );
      let rhs = &m * &probe;
      let resolved = lu.solve(&rhs);
      let stable = (&resolved - &probe).norm() <= 1e-6 * probe.norm().max(1.0);
      if stable {
        return (lu, current_shift);
      }
    }
    current_shift = shift + eps0 * 2f64.powi(attempt as i32);
  }
  panic!(
    "sparse_shift_invert_eigen: A - shift*B remained singular after \
     {MAX_ATTEMPTS} perturbations of shift {shift}"
  );
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

#[cfg(test)]
mod test {
  use super::sparse_shift_invert_eigen;
  use crate::linalg::nalgebra::{CooMatrix, CsrMatrix, Matrix, Vector};

  fn symmetric(n: usize, f: impl Fn(usize, usize) -> f64) -> Matrix {
    Matrix::from_fn(n, n, |i, j| f(i.min(j), i.max(j)))
  }

  fn csr(m: &Matrix) -> CsrMatrix {
    m.into()
  }

  /// Every returned pair solves the pencil: $A x = lambda B x$.
  #[test]
  fn pairs_solve_the_pencil() {
    let n = 6;
    let a = symmetric(n, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
    // SPD: diagonally dominant with a positive diagonal.
    let b = symmetric(n, |i, j| if i == j { n as f64 } else { 0.3 });

    for nev in 1..=n {
      let (vals, vecs) = sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.0, nev);
      for k in 0..vals.len() {
        let x = vecs.column(k).into_owned();
        let residual = (&a * &x - vals[k] * (&b * &x)).norm();
        assert!(residual < 1e-9, "nev={nev} k={k} residual={residual:e}");
      }
    }
  }

  /// With $B = I$ the pencil is the standard symmetric eigenproblem, so the
  /// eigenvalues nearest $0$ must match a dense symmetric EVD oracle.
  #[test]
  fn matches_symmetric_evd_oracle() {
    let n = 7;
    let a = symmetric(n, |i, j| ((i * 5 + j * 2) % 13) as f64 - 6.0);
    let id = Matrix::identity(n, n);

    let mut oracle: Vec<f64> = a.clone().symmetric_eigenvalues().iter().copied().collect();
    oracle.sort_by(|x, y| x.abs().total_cmp(&y.abs()));

    for nev in 1..=n {
      let (vals, _) = sparse_shift_invert_eigen(&csr(&a), &csr(&id), 0.0, nev);
      let mut got: Vec<f64> = vals.iter().copied().collect();
      let mut want = oracle[..nev].to_vec();
      got.sort_by(f64::total_cmp);
      want.sort_by(f64::total_cmp);
      for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-8, "nev={nev}: got {g} want {w}");
      }
    }
  }

  /// Eigenvectors are $B$-orthonormal.
  #[test]
  fn eigenvectors_are_b_orthonormal() {
    let n = 6;
    let a = symmetric(n, |i, j| if i == j { 2.0 * n as f64 } else { 0.5 });
    let b = symmetric(n, |i, j| if i == j { n as f64 } else { 0.3 });

    let (_, vecs) = sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.0, n);
    for k in 0..vecs.ncols() {
      for l in 0..vecs.ncols() {
        let vk = vecs.column(k).into_owned();
        let vl = vecs.column(l).into_owned();
        let ip = vk.dot(&(&b * &vl));
        let want = if k == l { 1.0 } else { 0.0 };
        assert!((ip - want).abs() < 1e-8, "k={k} l={l} got {ip} want {want}");
      }
    }
  }

  /// A singular, indefinite-adjacent $B$ (the mixed-formulation regime): a
  /// null direction of $B$ is never selected, and the finite pairs returned
  /// still solve the pencil, even shifted away from $0$.
  #[test]
  fn excludes_the_null_space_of_b() {
    let n = 5;
    let a = symmetric(n, |i, j| ((i * 3 + j * 7) % 11) as f64 - 5.0);
    // PSD but rank-deficient: the last coordinate carries no mass.
    let b = Matrix::from_fn(n, n, |i, j| if i == j && i + 1 < n { 1.0 } else { 0.0 });

    let (vals, vecs) = sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.1, n - 1);
    assert_eq!(vals.len(), n - 1);
    for k in 0..vals.len() {
      assert!(vals[k].is_finite());
      let x: Vector = vecs.column(k).into_owned();
      let residual = (&a * &x - vals[k] * (&b * &x)).norm();
      assert!(residual < 1e-8, "k={k} residual={residual:e}");
    }
  }

  /// The $1 times 1$ (and, via `k = 0`, empty) base case — the $0$-manifold's
  /// Hodge-Laplace pencil.
  #[test]
  fn solves_the_scalar_pencil() {
    let a = Matrix::from_element(1, 1, 3.0);
    let b = Matrix::from_element(1, 1, 4.0);
    let (vals, vecs) = sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.0, 3);
    assert_eq!(vals.len(), 1);
    assert!((vals[0] - 3.0 / 4.0).abs() < 1e-9);
    let x: Vector = vecs.column(0).into_owned();
    assert!(
      (x.dot(&(&b * &x)) - 1.0).abs() < 1e-9,
      "eigenvector is B-normalized"
    );
    assert!((&a * &x - vals[0] * (&b * &x)).norm() < 1e-9);

    let (vals0, vecs0) = sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.0, 0);
    assert_eq!(vals0.len(), 0);
    assert_eq!(vecs0.ncols(), 0);
  }

  /// A degenerate cluster of multiplicity `m` sitting exactly at the shift —
  /// the harmonic-space case, `shift = 0` on a pencil where $A$ itself is
  /// singular — is fully resolved (not collapsed to one direction), forcing
  /// the shift-retry path every time.
  #[test]
  fn resolves_a_degenerate_cluster_at_the_shift() {
    let m = 3;
    let rest = 4;
    let n = m + rest;
    // B = I. A has an m-fold zero eigenvalue (the first m basis vectors) and
    // `rest` nonzero eigenvalues elsewhere, mixed by a fixed orthonormal-ish
    // change of basis so the cluster isn't axis-aligned with the seed.
    let b = Matrix::identity(n, n);
    let diag = Matrix::from_fn(n, n, |i, j| {
      if i != j || i < m {
        0.0
      } else {
        (i - m + 1) as f64
      }
    });
    // A small fixed rotation mixing coordinates, so the zero-eigenspace is
    // not simply the first m standard basis vectors.
    let rot = Matrix::from_fn(n, n, |i, j| ((i * 5 + j * 3 + 1) % 7) as f64 - 3.0);
    let rot = rot.clone() + rot.transpose() + Matrix::identity(n, n) * (2.0 * n as f64);
    let a = &rot * &diag * &rot.transpose();
    let a = (&a + a.transpose()) * 0.5;

    let (vals, vecs) = sparse_shift_invert_eigen(&csr(&a), &csr(&b), 0.0, m);
    assert_eq!(vals.len(), m);
    for &v in vals.iter() {
      assert!(v.abs() < 1e-6, "expected a near-zero eigenvalue, got {v}");
    }
    for k in 0..m {
      let x = vecs.column(k).into_owned();
      let residual = (&a * &x - vals[k] * &x).norm();
      assert!(residual < 1e-6, "k={k} residual={residual:e}");
    }
    // The m recovered eigenvectors are linearly independent (B-orthonormal),
    // so together they span the whole zero-eigenspace rather than repeating
    // one direction.
    let gram = vecs.transpose() * &vecs;
    let dev = (&gram - Matrix::identity(m, m)).norm();
    assert!(
      dev < 1e-6,
      "recovered directions are not mutually independent: {gram}"
    );
  }

  /// The 1D Dirichlet Laplacian (tridiagonal, $B = I$), whose eigenvalues have
  /// the closed form $lambda_j = 2 - 2 cos(j pi \/ (N + 1))$ — an oracle with
  /// no dense EVD needed at any size, including $N$ in the low thousands,
  /// where dense QZ has no business running.
  #[test]
  fn handles_a_large_sparse_pencil() {
    let n = 3000;
    let nev = 5;

    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
      coo.push(i, i, 2.0);
      if i + 1 < n {
        coo.push(i, i + 1, -1.0);
        coo.push(i + 1, i, -1.0);
      }
    }
    let a = CsrMatrix::from(&coo);
    let mut ident = CooMatrix::new(n, n);
    for i in 0..n {
      ident.push(i, i, 1.0);
    }
    let b = CsrMatrix::from(&ident);

    let (vals, vecs) = sparse_shift_invert_eigen(&a, &b, 0.0, nev);
    assert_eq!(vals.len(), nev);
    for k in 0..nev {
      let x = vecs.column(k).into_owned();
      let residual = (&a * &x - vals[k] * (&b * &x)).norm();
      assert!(residual < 1e-6, "k={k} residual={residual:e}");

      let want = 2.0 - 2.0 * (((k + 1) as f64 * std::f64::consts::PI) / (n as f64 + 1.0)).cos();
      assert!(
        (vals[k] - want).abs() < 1e-6,
        "k={k}: got {} want {want}",
        vals[k]
      );
    }
  }
}
