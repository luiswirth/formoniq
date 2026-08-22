//! [`SimplexLengthsSq`]: the Regge representation of a cell's metric, on any
//! signature. The representation is faithful, the squared lengths are the
//! metric's components in the edge basis, and the Cayley-Menger determinant
//! is what decides degeneracy.

use approx::assert_relative_eq;
use metric::Metric;
use multialgebra::tensor::pairing;
use multiindex::Dim;
use regge::lengths::simplex::{SimplexLengthsSq, unit_edge_squares};
use simplicial::linalg::Matrix;
use simplicial::topology::simplex::edge_index;

/// Cayley-Menger decides degeneracy, and it decides it scale-invariantly:
/// $"CM"(s) = det g \/ (n!)^2$ vanishes exactly when the induced metric is
/// singular, and the test is against the simplex's own volume scale
/// $"diam"^n$, so a uniform scaling leaves it alone. Collapsing two vertices
/// onto each other makes two rows of the distance matrix agree, hence
/// $"CM" = 0$, at every scale and in every dimension.
#[test]
fn degeneracy_is_scale_invariant() {
  for dim in (2..=4usize).map(Dim::from) {
    for scale in [1e-4, 1.0, 1e4] {
      let mut lengths = SimplexLengthsSq::unit(dim);
      *lengths.vector_mut() *= scale * scale;
      assert!(!lengths.is_degenerate());

      lengths.vector_mut()[edge_index(1, 2)] = 0.0;
      assert!(lengths.is_degenerate());
    }
  }
}

/// Squared edge lengths are the components of the metric in the basis dual to
/// the edge squares, $s_e = angle.l g, u_e dot.circle u_e angle.r$, and the
/// change of basis is faithful: [`SimplexLengthsSq::from_metric`] and
/// [`SimplexLengthsSq::metric`] are inverse.
///
/// The polarization identity of those two is that change of basis, and this
/// is what says so rather than asserting it in prose. Swept over every
/// signature, since the pairing is metric-free and so must hold on all of
/// them, and over the flat models pulled back to non-diagonal form, where a
/// representation reading only a diagonal would lose the off-diagonal part.
#[test]
fn the_squared_lengths_are_the_metric_paired_with_the_edge_squares() {
  for dim in (1..=4usize).map(Dim::from) {
    let shear = Matrix::from_fn(dim.index(), dim.index(), |i, j| {
      if i == j {
        1.0
      } else if i > j {
        ((2 * i + 3 * j) % 4) as f64 / 8.0
      } else {
        0.0
      }
    });

    for q in 0..=dim.index() {
      let flat = Metric::pseudo_euclidean(dim.index() - q, q);
      let lengths = SimplexLengthsSq::from_metric(&flat);
      for (iedge, square) in unit_edge_squares(dim).iter().enumerate() {
        assert_relative_eq!(
          lengths[iedge],
          pairing(&flat.tensor(), square),
          epsilon = 1e-12
        );
      }

      for metric in [flat.clone(), flat.pullback(&shear)] {
        let regge = SimplexLengthsSq::from_metric(&metric);
        assert_relative_eq!(regge.metric().matrix(), metric.matrix(), epsilon = 1e-12);
        assert_eq!(regge.metric().signature(), (dim.index() - q, q));
      }
    }
  }
}
