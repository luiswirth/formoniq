//! Quadrature on the charts of the atlas, in every dimension.
//!
//! One principled construction covers all dimensions and orders: the
//! Grundmann-Möller rules (SIAM J. Numer. Anal. 15, 1978), a single closed
//! formula giving symmetric rules exact for polynomials of any odd degree on the
//! n-simplex.
//!
//! The nodes are **barycentric**, which is what the rules natively are and what
//! makes them chart data rather than coordinate data: a quadrature rule needs no
//! embedding and no metric, only the reference cell. Integrating over a cell of
//! the manifold therefore hands the integrand [`MeshPoint`]s -- points of the
//! manifold, in that cell's chart -- and the geometry enters through nothing but
//! the scalar volume factor.

use super::{Bary, BaryRef, MeshPoint};
use crate::{Dim, topology::handle::SimplexIdx};

use crate::linalg::{Vector, VectorView};
use multiindex::{Combination, Composition, binomial, factorial_f64};

/// A quadrature rule on the reference simplex, with barycentric nodes.
pub struct SimplexQuadRule {
  dim: Dim,
  /// The nodes, in barycentric coordinates.
  points: Vec<Bary>,
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
    let mut points = Vec::with_capacity(npoints);
    let mut weights = Vector::zeros(npoints);

    let mut ipoint = 0;
    for i in 0..=s {
      let denominator = (d + n - 2 * i) as f64;
      let weight =
        (-1.0f64).powi(i as i32) * 2.0f64.powi(-2 * (s as i32)) * denominator.powi(d as i32)
          / (factorial_f64(i) * factorial_f64(d + n - i));

      for beta in Composition::all(n + 1, s - i) {
        points.push(Bary::from_iterator(
          n + 1,
          beta
            .parts()
            .iter()
            .map(|&b| (2 * b + 1) as f64 / denominator),
        ));
        weights[ipoint] = weight;
        ipoint += 1;
      }
    }

    // The formula integrates over the unnormalized reference simplex of
    // volume 1/n!; normalize the weights to sum to 1.
    weights /= weights.sum();

    Self {
      dim,
      points,
      weights,
    }
  }

  /// The (minimal-index) Grundmann-Möller rule exact for polynomials of the
  /// given degree.
  pub fn degree(dim: Dim, degree: usize) -> Self {
    Self::grundmann_moeller(dim, degree / 2)
  }
}

impl SimplexQuadRule {
  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn npoints(&self) -> usize {
    self.weights.len()
  }
  /// The nodes, in barycentric coordinates of the reference cell.
  pub fn points(&self) -> impl ExactSizeIterator<Item = BaryRef<'_>> {
    self.points.iter().map(Bary::as_view)
  }
  /// The normalized weights, summing to one.
  pub fn weights(&self) -> VectorView<'_> {
    self.weights.as_view()
  }

  /// $vol dot sum_i w_i f(lambda_i)$: the quadrature of a function of the
  /// barycentric coordinates of the reference cell, scaled by the volume of the
  /// domain it stands for.
  ///
  /// The one place the weights are actually summed; every other method here is
  /// this one with the nodes placed somewhere.
  pub fn integrate_ref<F>(&self, f: &F, vol: f64) -> f64
  where
    F: Fn(BaryRef) -> f64,
  {
    let integral: f64 = self
      .points()
      .zip(self.weights.iter())
      .map(|(bary, weight)| weight * f(bary))
      .sum();
    vol * integral
  }

  /// $integral_K f$ over a cell of the manifold, of a function of the points of
  /// the manifold: the nodes are [`MeshPoint`]s in the chart of that cell.
  ///
  /// The volume of the cell is the only thing the geometry contributes -- for a
  /// metric $g$ it is `cell_volume(g)`, and for the chart alone it is
  /// [`refsimp_vol`](super::refsimp_vol).
  pub fn integrate_cell<F>(&self, cell: SimplexIdx, f: &F, vol: f64) -> f64
  where
    F: Fn(&MeshPoint) -> f64,
  {
    assert_eq!(
      self.dim,
      cell.dim(),
      "Quadrature rule of the wrong dimension."
    );
    self.integrate_ref(&|bary| f(&MeshPoint::new(cell, bary.to_coords())), vol)
  }

  /// $integral_sigma f$ over a face of a cell, of a function of the points of
  /// the manifold, in the chart of that **cell**: the nodes are the face's
  /// quadrature nodes, scattered onto the face's local vertex positions.
  ///
  /// A face carries no chart of its own, so this is the only way to integrate
  /// over one -- and the answer does not depend on which supporting cell is
  /// used, because the two differ by a [`Transition`](super::Transition).
  pub fn integrate_face<F>(&self, cell: SimplexIdx, positions: &Combination, f: &F, vol: f64) -> f64
  where
    F: Fn(&MeshPoint) -> f64,
  {
    assert_eq!(
      self.dim + 1,
      positions.card(),
      "Face of the wrong dimension."
    );
    self.integrate_ref(&|bary| f(&MeshPoint::on_face(cell, positions, bary)), vol)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::atlas::refsimp_vol;

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
          for alpha in Composition::all(dim + 1, degree) {
            let monomial = |bary: BaryRef| -> f64 {
              (0..=dim)
                .map(|i| bary[i].powi(alpha.parts()[i] as i32))
                .product()
            };
            let computed = quadrule.integrate_ref(&monomial, refsimp_vol(dim));
            let exact = exact_barycentric_monomial(alpha.parts());
            assert_abs_diff_eq!(computed, exact, epsilon = 1e-12);
          }
        }
      }
    }
  }

  /// The nodes are barycentric: their weights sum to one, and so do the
  /// quadrature weights.
  #[test]
  fn nodes_and_weights_are_barycentric() {
    for dim in 0..=4 {
      for s in 0..=3 {
        let quadrule = SimplexQuadRule::grundmann_moeller(dim, s);
        assert_abs_diff_eq!(quadrule.weights().sum(), 1.0, epsilon = 1e-12);
        for bary in quadrule.points() {
          assert_abs_diff_eq!(bary.view().sum(), 1.0, epsilon = 1e-12);
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
        let monomial = |bary: BaryRef| bary[1].powi(degree as i32);
        let alpha: Vec<usize> = std::iter::once(0)
          .chain(std::iter::once(degree))
          .chain(std::iter::repeat_n(0, dim - 1))
          .collect_vec();
        let computed = quadrule.integrate_ref(&monomial, refsimp_vol(dim));
        let exact = exact_barycentric_monomial(&alpha);
        assert_abs_diff_eq!(computed, exact, epsilon = 1e-12);
      }
    }
  }
}
