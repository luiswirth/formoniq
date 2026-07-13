//! Quadrature on the reference simplex, in every dimension.
//!
//! One principled construction covers all dimensions and orders: the
//! Grundmann-Möller rules (SIAM J. Numer. Anal. 15, 1978), a single closed
//! formula giving symmetric rules exact for polynomials of any odd degree
//! on the n-simplex.

use super::{mesh::MeshCoords, simplex::SimplexCoords, CoordRef};
use crate::{
  topology::{complex::Complex, handle::SimplexRef},
  Dim,
};

use common::{
  combo::{binomial, factorial_f64},
  linalg::nalgebra::{Matrix, Vector},
};

/// A quadrature rule defined on the reference simplex.
///
/// Can be used to integrate functions defined on the reference simplex.
pub struct SimplexQuadRule {
  /// Points in local coordinates.
  points: Matrix,
  /// Normalized weights that sum to 1.
  weights: Vector,
}

impl SimplexQuadRule {
  /// The Grundmann-Möller rule of index `s` on the `dim`-simplex:
  /// exact for polynomials of degree $d = 2s + 1$, in every dimension.
  ///
  /// $integral_Delta f approx sum_(i=0)^s w_i sum_(|beta| = s-i)
  ///   f((2 beta + bb(1)) \/ (d + n - 2i))$
  /// over barycentric lattice points $beta in NN_0^(n+1)$, with
  /// $w_i = (-1)^i 2^(-2s) (d + n - 2i)^d \/ (i! (d + n - i)!)$.
  pub fn grundmann_moeller(dim: Dim, s: usize) -> Self {
    let n = dim;
    let d = 2 * s + 1;

    let npoints: usize = (0..=s).map(|i| binomial(s - i + n, n)).sum();
    let mut points = Matrix::zeros(n, npoints);
    let mut weights = Vector::zeros(npoints);

    let mut ipoint = 0;
    for i in 0..=s {
      let denominator = (d + n - 2 * i) as f64;
      let weight =
        (-1.0f64).powi(i as i32) * 2.0f64.powi(-2 * (s as i32)) * denominator.powi(d as i32)
          / (factorial_f64(i) * factorial_f64(d + n - i));

      for beta in compositions(s - i, n + 1) {
        // Barycentric point (2 beta + 1) / denominator;
        // local coordinates drop the 0th barycentric coordinate.
        for icomp in 0..n {
          points[(icomp, ipoint)] = (2 * beta[icomp + 1] + 1) as f64 / denominator;
        }
        weights[ipoint] = weight;
        ipoint += 1;
      }
    }

    // The formula integrates over the unnormalized reference simplex of
    // volume 1/n!; normalize the weights to sum to 1.
    weights /= weights.sum();

    Self { points, weights }
  }

  /// The (minimal-index) Grundmann-Möller rule exact for polynomials of the
  /// given degree.
  pub fn degree(dim: Dim, degree: usize) -> Self {
    Self::grundmann_moeller(dim, degree / 2)
  }
}

/// All $beta in NN_0^"nparts"$ with $|beta| = "total"$.
fn compositions(total: usize, nparts: usize) -> Vec<Vec<usize>> {
  if nparts == 1 {
    return vec![vec![total]];
  }
  (0..=total)
    .flat_map(|first| {
      compositions(total - first, nparts - 1)
        .into_iter()
        .map(move |mut rest| {
          rest.insert(0, first);
          rest
        })
    })
    .collect()
}

impl SimplexQuadRule {
  pub fn dim(&self) -> Dim {
    self.points.nrows()
  }
  pub fn npoints(&self) -> usize {
    self.points.ncols()
  }
  /// Uses a local coordinate function `f`.
  pub fn integrate_local<F>(&self, f: &F, vol: f64) -> f64
  where
    F: Fn(CoordRef) -> f64,
  {
    let mut integral = 0.0;
    for i in 0..self.npoints() {
      integral += self.weights[i] * f(self.points.column(i));
    }
    vol * integral
  }

  /// Uses a global coordinate function `f`.
  pub fn integrate_coord<F>(&self, f: &F, coords: &SimplexCoords) -> f64
  where
    F: Fn(CoordRef) -> f64,
  {
    self.integrate_local(
      &|local_coord| f(coords.local2global(local_coord).as_view()),
      coords.vol(),
    )
  }

  /// Uses a global coordinate function `f`.
  pub fn integrate_mesh<F>(&self, f: &F, complex: &Complex, coords: &MeshCoords) -> f64
  where
    F: Fn(CoordRef, SimplexRef) -> f64,
  {
    let mut integral = 0.0;
    for cell in complex.cells().handle_iter() {
      let cell_coords = SimplexCoords::from_simplex_and_coords(cell.simplex(), coords);
      integral += self.integrate_coord(&|x| f(x, cell), &cell_coords);
    }
    integral
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::geometry::refsimp_vol;

  use approx::assert_abs_diff_eq;
  use itertools::Itertools;

  /// The exact integral of a barycentric monomial over the reference simplex:
  /// $integral_Delta lambda^alpha = n! alpha_0 ! dots.c alpha_n ! /
  /// (n + |alpha|)! dot vol$.
  fn exact_barycentric_monomial(alpha: &[usize]) -> f64 {
    let n = alpha.len() - 1;
    let total: usize = alpha.iter().sum();
    let numerator: f64 =
      factorial_f64(n) * alpha.iter().map(|&a| factorial_f64(a)).product::<f64>();
    numerator / factorial_f64(n + total) * refsimp_vol(n)
  }

  /// Grundmann-Möller integrates every barycentric monomial of degree
  /// <= 2s + 1 exactly, in every dimension.
  #[test]
  fn grundmann_moeller_is_exact_on_polynomials() {
    for dim in 0..=4 {
      for s in 0..=3 {
        let quadrule = SimplexQuadRule::grundmann_moeller(dim, s);
        let max_degree = 2 * s + 1;
        for degree in 0..=max_degree {
          for alpha in compositions(degree, dim + 1) {
            let monomial = |p: CoordRef| -> f64 {
              let bary0 = 1.0 - p.sum();
              let mut value = bary0.powi(alpha[0] as i32);
              for icomp in 0..dim {
                value *= p[icomp].powi(alpha[icomp + 1] as i32);
              }
              value
            };
            let computed = quadrule.integrate_local(&monomial, refsimp_vol(dim));
            let exact = exact_barycentric_monomial(&alpha);
            assert_abs_diff_eq!(computed, exact, epsilon = 1e-12);
          }
        }
      }
    }
  }

  /// The degree constructor guarantees the requested exactness.
  #[test]
  fn degree_constructor_is_exact() {
    for dim in 1..=3 {
      for degree in 0..=5 {
        let quadrule = SimplexQuadRule::degree(dim, degree);
        let monomial = |p: CoordRef| p[0].powi(degree as i32);
        let alpha: Vec<usize> = std::iter::once(0)
          .chain(std::iter::once(degree))
          .chain(std::iter::repeat_n(0, dim - 1))
          .collect_vec();
        let computed = quadrule.integrate_local(&monomial, refsimp_vol(dim));
        let exact = exact_barycentric_monomial(&alpha);
        assert_abs_diff_eq!(computed, exact, epsilon = 1e-12);
      }
    }
  }
}
