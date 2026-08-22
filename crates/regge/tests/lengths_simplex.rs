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

/// from_metric and metric are inverse, on every
/// signature, the flat models pulled back to non-diagonal form included.
/// The Regge representation loses nothing of a pseudo-Riemannian metric.
#[test]
fn metric_tensor_roundtrip() {
  for dim in (1..=4usize).map(Dim::from) {
    let lengths_sq = SimplexLengthsSq::unit(dim);
    let roundtrip = SimplexLengthsSq::from_metric(&lengths_sq.metric());
    assert_relative_eq!(lengths_sq.vector(), roundtrip.vector(), epsilon = 1e-12);

    for q in 0..=dim.index() {
      let j = Matrix::from_fn(dim.index(), dim.index(), |i, jj| {
        if i == jj {
          1.0
        } else if i > jj {
          ((2 * i + 3 * jj) % 4) as f64 / 8.0
        } else {
          0.0
        }
      });
      let g = Metric::pseudo_euclidean(dim.index() - q, q).pullback(&j);
      let regge = SimplexLengthsSq::from_metric(&g);
      assert_relative_eq!(regge.metric().matrix(), g.matrix(), epsilon = 1e-12);
      assert_eq!(regge.metric().signature(), (dim.index() - q, q));
    }
  }
}

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
/// the edge squares: $s_e = angle.l g, u_e dot.circle u_e angle.r$.
///
/// The polarization identity of [`SimplexLengthsSq::metric`] and
/// [`SimplexLengthsSq::from_metric`] is that change of basis, and this is
/// what says so rather than asserting it in prose. Swept over every signature,
/// since the pairing is metric-free and so must hold on all of them.
#[test]
fn the_squared_lengths_are_the_metric_paired_with_the_edge_squares() {
  for dim in (1..=4usize).map(Dim::from) {
    for q in 0..=dim.index() {
      let metric = Metric::pseudo_euclidean(dim.index() - q, q);
      let lengths = SimplexLengthsSq::from_metric(&metric);
      for (iedge, square) in unit_edge_squares(dim).iter().enumerate() {
        assert_relative_eq!(
          lengths[iedge],
          pairing(&metric.tensor(), square),
          epsilon = 1e-12
        );
      }
    }
  }
}
