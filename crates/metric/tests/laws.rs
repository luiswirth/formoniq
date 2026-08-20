//! The metric's own laws: the pullback, the dual, the signature, and the
//! two hypotheses [`Metric::new_checked`] verifies.

use metric::{CausalType, Metric};
use multialgebra::{Matrix, Variance, Vector};
use std::f64::consts::FRAC_PI_2;

#[test]
fn euclidean_angles_and_norms() {
  let g = Metric::euclidean(2);
  let e0 = Vector::from_column_slice(&[1.0, 0.0]);
  let e1 = Vector::from_column_slice(&[0.0, 1.0]);

  assert!((g.norm(&e0) - 1.0).abs() < 1e-12);
  assert!((g.angle(&e0, &e1) - FRAC_PI_2).abs() < 1e-12);
  assert!(g.angle_cos(&e0, &e1).abs() < 1e-12);
  assert!(g.angle(&e0, &e0).abs() < 1e-12);
}

#[test]
fn nonstandard_metric_angle_matches_definition() {
  // A metric that stretches the second axis. The coordinate axes stay
  // g-orthogonal, but a diagonal vector no longer bisects them.
  let g = Metric::new(
    Variance::Covariant,
    Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 4.0]),
  );
  let v = Vector::from_column_slice(&[1.0, 1.0]);
  let w = Vector::from_column_slice(&[1.0, 0.0]);

  // g(v, w) = 1, |v|_g = sqrt(5), |w|_g = 1.
  assert!((g.inner(&v, &w) - 1.0).abs() < 1e-12);
  assert!((g.norm(&v) - 5.0_f64.sqrt()).abs() < 1e-12);
  assert!((g.angle_cos(&v, &w) - 1.0 / 5.0_f64.sqrt()).abs() < 1e-12);
}

// 2 on the diagonal, 1 off it: SPD with eigenvalues `dim+1` (once) and 1.
fn spd(dim: usize) -> Metric {
  let mut m = Matrix::from_element(dim, dim, 1.0);
  for i in 0..dim {
    m[(i, i)] = 2.0;
  }
  Metric::new(Variance::Covariant, m)
}

// A deterministic full-column-rank `nrows x ncols` matrix (ncols <= nrows):
// unit lower-triangular columns, injective, so the pullback stays s.p.d.
fn full_col_rank(nrows: usize, ncols: usize) -> Matrix {
  Matrix::from_fn(nrows, ncols, |i, j| {
    if i == j {
      1.0
    } else if i > j {
      0.5
    } else {
      0.0
    }
  })
}

fn close(a: &Matrix, b: &Matrix) {
  assert_eq!(a.shape(), b.shape());
  assert!((a - b).amax() < 1e-9, "{a} != {b}");
}

/// The pullback is functorial: pulling $g$ back along a composite $A B$
/// equals pulling first along $A$, then along $B$,
/// $(A B)^* g = B^* (A^* g)$, i.e. $(A B)^top g (A B) = B^top (A^top g A) B$.
/// Swept over the full grade/dimension range including the degenerate
/// zero-column maps, where the pulled-back metric is the empty $0 times 0$
/// Gramian.
#[test]
fn pullback_is_functorial() {
  for n in 0..=4 {
    let g = spd(n);
    for k in 0..=n {
      let a = full_col_rank(n, k);
      for m in 0..=k {
        let b = full_col_rank(k, m);
        let composite = &a * &b;
        let lhs = g.pullback(&composite);
        let rhs = g.pullback(&a).pullback(&b);
        close(lhs.matrix(), rhs.matrix());
      }
    }
  }
}

/// The pullback is literally $J^top G J$, and pulling back along the identity
/// changes nothing.
#[test]
fn pullback_matches_definition_and_fixes_identity() {
  for n in 1..=4 {
    let g = spd(n);
    let j = full_col_rank(n, n);
    close(g.pullback(&j).matrix(), &(j.transpose() * g.matrix() * &j));
    close(g.pullback(&Matrix::identity(n, n)).matrix(), g.matrix());
  }
}

/// $g$ and $g^(-1)$ are one datum: `dual` is an involution that flips the
/// variance, and `measuring` returns the same metric whichever of the two it
/// is handed. That is what makes a caller unable to pick the wrong one.
#[test]
fn the_dual_is_an_involution_and_measuring_is_side_blind() {
  for n in 1..=4 {
    let g = spd(n);
    let inverse = g.dual();

    assert_eq!(g.variance(), Variance::Covariant);
    assert_eq!(inverse.variance(), Variance::Contravariant);
    close(inverse.dual().matrix(), g.matrix());
    close(&(g.matrix() * inverse.matrix()), &Matrix::identity(n, n));

    // Vectors are measured by g, covectors by g^-1, from either side.
    for slot in [Variance::Contravariant, Variance::Covariant] {
      close(g.measuring(slot).matrix(), inverse.measuring(slot).matrix());
    }
    close(g.measuring(Variance::Contravariant).matrix(), g.matrix());
    close(g.measuring(Variance::Covariant).matrix(), inverse.matrix());
  }
}

/// Taking the dual and pulling back do not commute in general, which is why
/// `pullback` preserves the variance instead of inverting: $(J^* g)^(-1)$ is
/// $J^(-1) g^(-1) J^(-top)$, equal to $J^* (g^(-1))$ only when $J$ is
/// orthogonal. They agree on the identity, and the general failure is what the
/// doc contract warns about.
#[test]
fn the_dual_and_the_pullback_do_not_commute() {
  for n in 1..=4 {
    let g = spd(n);
    let j = full_col_rank(n, n);

    close(
      g.pullback(&Matrix::identity(n, n)).dual().matrix(),
      g.dual().pullback(&Matrix::identity(n, n)).matrix(),
    );

    let pull_then_dual = g.pullback(&j).dual();
    let dual_then_pull = g.dual().pullback(&j);
    assert_eq!(pull_then_dual.variance(), Variance::Contravariant);
    assert_eq!(dual_then_pull.variance(), Variance::Contravariant);
    if n > 1 {
      assert!(
        (pull_then_dual.matrix() - dual_then_pull.matrix()).amax() > 1e-9,
        "the two orders must differ, or the law is vacuous"
      );
    }
  }
}

/// The flat models carry their signature by construction: signature
/// $(p, q)$, determinant sign $(-1)^q$, unit volume factor, swept over
/// every signature up to dimension 4, the Euclidean $q = 0$ and the empty
/// $0 times 0$ form included.
#[test]
fn flat_model_signatures() {
  for dim in 0..=4 {
    for q in 0..=dim {
      let p = dim - q;
      let g = Metric::pseudo_euclidean(p, q);
      assert_eq!(g.signature(), (p, q));
      assert_eq!(g.is_riemannian(), q == 0);
      if dim > 0 {
        assert_eq!(g.det().signum(), (-1.0f64).powi(q as i32));
      }
      assert!((g.det_sqrt() - 1.0).abs() < 1e-12);
    }
  }
}

/// Sylvester's law of inertia: the signature is invariant under congruence,
/// i.e. under pullback along any invertible map.
#[test]
fn signature_is_congruence_invariant() {
  for dim in 1..=4 {
    for q in 0..=dim {
      let g = Metric::pseudo_euclidean(dim - q, q);
      let j = full_col_rank(dim, dim);
      assert_eq!(g.pullback(&j).signature(), (dim - q, q));
    }
  }
}

/// The causal trichotomy on Minkowski space: the time axis is timelike, the
/// space axes spacelike, the light-cone diagonal null, and the magnitude
/// is never NaN, on either side of the cone.
#[test]
fn minkowski_causal_types() {
  let eta = Metric::minkowski(4);
  assert_eq!(eta.signature(), (3, 1));

  let e0 = Vector::from_column_slice(&[1.0, 0.0, 0.0, 0.0]);
  let e1 = Vector::from_column_slice(&[0.0, 1.0, 0.0, 0.0]);
  let light = Vector::from_column_slice(&[1.0, 1.0, 0.0, 0.0]);

  assert_eq!(eta.causal_type(&e0), CausalType::Timelike);
  assert_eq!(eta.causal_type(&e1), CausalType::Spacelike);
  assert_eq!(eta.causal_type(&light), CausalType::Null);

  assert!((eta.norm_sq(&e0) - -1.0).abs() < 1e-12);
  assert!((eta.norm(&e0) - 1.0).abs() < 1e-12);
  assert!((eta.norm(&light) - 0.0).abs() < 1e-12);

  let metric = Metric::minkowski(4);
  assert_eq!(metric.signature(), (3, 1));
  assert_eq!(metric.causal_type(&e0), CausalType::Timelike);
  close(metric.dual().matrix(), metric.matrix());
}

/// The two hypotheses, each violated on its own, are each rejected, and a
/// matrix meeting both is accepted. Stated through the checking constructor
/// rather than as a panic, so the law holds in either build profile: the
/// plain constructor takes the contract on trust and asserts only in debug.
#[test]
fn a_metric_is_exactly_a_symmetric_nondegenerate_matrix() {
  let degenerate = Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
  let nonsymmetric = Matrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]);
  let good = Matrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);

  for bad in [degenerate, nonsymmetric] {
    assert!(Metric::new_checked(Variance::Covariant, bad.clone()).is_none());
    // The predicate and the constructor decide the same question.
    assert!(!Metric::new(Variance::Covariant, bad).is_valid());
  }

  let metric = Metric::new_checked(Variance::Covariant, good).unwrap();
  assert!(metric.is_valid());
}
