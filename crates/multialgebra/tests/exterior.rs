//! The exterior-algebra laws, on a single alternating slot.

use approx::assert_relative_eq;
use multialgebra::tensor::{one_alternating, wedge_pairing};
use multialgebra::{Matrix, Tensor, Variance, Vector, exterior_bases, exterior_dim};
use multiindex::Sign;

fn probe_element(dim: usize, grade: usize, seed: usize, variance: Variance) -> Tensor {
  Tensor::new(
    one_alternating(grade, variance, dim),
    Vector::from_fn(exterior_dim(dim, grade), |i, _| {
      ((seed + 5 * i) % 7) as f64 - 3.0
    }),
  )
}

/// $iota_v$ is an antiderivation of degree $-1$.
#[test]
fn interior_product_antiderivation() {
  for dim in 2..=4 {
    let vector = Tensor::line(
      Vector::from_fn(dim, |i, _| (i + 1) as f64),
      Variance::Contravariant,
    );
    for grade_a in 1..dim {
      for grade_b in 1..=(dim - grade_a) {
        let alpha = probe_element(dim, grade_a, 5, Variance::Covariant);
        let beta = probe_element(dim, grade_b, 6, Variance::Covariant);

        let lhs = alpha.wedge(&beta).interior_product(&vector);
        let rhs = alpha.interior_product(&vector).wedge(&beta)
          + Sign::from_parity(grade_a).as_f64() * alpha.wedge(&beta.interior_product(&vector));
        assert_relative_eq!(lhs.components(), rhs.components(), epsilon = 1e-12);
      }
    }
  }
}

/// The wedge pairing is graded-symmetric,
/// $chevron.l beta, alpha chevron.r = (-1)^(k(n-k)) chevron.l alpha, beta chevron.r$,
/// and nondegenerate.
///
/// Nondegeneracy is the content: it is what makes $Lambda^(n-k)$ the dual of
/// $Lambda^k$ with no inner product chosen, which is Poincare duality on the
/// algebra. Checked as the pairing matrix against the basis being invertible.
#[test]
fn the_wedge_pairing_is_graded_symmetric_and_nondegenerate() {
  for dim in 1..=4 {
    for grade in 0..=dim {
      let complement = dim - grade;
      let alpha = probe_element(dim, grade, 3, Variance::Covariant);
      let beta = probe_element(dim, complement, 5, Variance::Covariant);
      let sign = Sign::from_parity(grade * complement).as_f64();

      assert_relative_eq!(
        wedge_pairing(&beta, &alpha),
        sign * wedge_pairing(&alpha, &beta),
        epsilon = 1e-12
      );

      let (rows, cols) = (exterior_dim(dim, grade), exterior_dim(dim, complement));
      let matrix = Matrix::from_fn(rows, cols, |i, j| {
        let ei: Tensor = Tensor::from_blade_signed(
          dim,
          Sign::Pos,
          exterior_bases(dim, grade).nth(i).unwrap(),
          Variance::Covariant,
        );
        let ej: Tensor = Tensor::from_blade_signed(
          dim,
          Sign::Pos,
          exterior_bases(dim, complement).nth(j).unwrap(),
          Variance::Covariant,
        );
        wedge_pairing(&ei, &ej)
      });
      assert_eq!(rows, cols);
      assert!(
        matrix.determinant().abs() > 1e-12,
        "dim {dim} grade {grade}: the wedge pairing is degenerate"
      );
    }
  }
}
