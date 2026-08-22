//! The grade reduction: the one reading of a $k$-form as a scalar or a
//! vector that a viewer and an exporter must share, and the density a cochain
//! carries on its own simplex.

use derham::Cochain;
use derham::reduce::{scalarize, trace_value};
use metric::{Metric, tensor::TensorExt};
use multialgebra::Tensor;
use nalgebra as na;
use simplicial::{Sign, atlas::Bary, linalg::Vector};

/// [`scalarize`] reads a $k$-form through the star, and the two branches of
/// that reading are the interior and the extremes of the grade range.
///
/// Off the extremes it is the magnitude, which is Hodge-invariant,
/// $|omega|_g = |star omega|_g$, the star being an isometry on a Riemannian
/// metric: a field and its reduction read the same scalar, so which side of
/// $k <-> n-k$ a mark happens to hold cannot change the color on screen.
///
/// At the extremes it is signed, and signed for different reasons: a
/// $0$-form is a scalar (metric-free, no orientation involved), while an
/// $n$-form is a pseudoscalar whose sign is the coherent orientation's.
/// Flipping that orientation negates the readout and nothing else, which is
/// precisely invariant 6's gauge acting on the picture.
#[test]
fn scalarize_is_hodge_invariant_and_signed_at_the_extremes() {
  for dim in 1..=4 {
    let metric = Metric::euclidean(dim);

    for grade in 1..dim {
      let ncoeffs = Tensor::<f64>::multiform_zero(dim, grade).components().len();
      for i in 0..ncoeffs {
        let mut coeffs = na::DVector::zeros(ncoeffs);
        coeffs[i] = 2.0;
        let form = Tensor::multiform(coeffs, dim, grade);
        let starred = form.clone().star(&metric, Sign::Pos);
        let (direct, reduced) = (
          scalarize(form, &metric, None),
          scalarize(starred, &metric, None),
        );
        assert!((direct - reduced).abs() < 1e-12, "{direct} != {reduced}");
      }
    }

    let zero_form = Tensor::multiform(na::dvector![-1.0], dim, 0);
    assert!((scalarize(zero_form, &metric, None) + 1.0).abs() < 1e-12);

    let top = Tensor::multiform(na::dvector![1.0], dim, dim);
    let pos = scalarize(top.clone(), &metric, Some(Sign::Pos));
    let neg = scalarize(top, &metric, Some(Sign::Neg));
    assert!((pos + neg).abs() < 1e-12);
    assert!((pos.abs() - 1.0).abs() < 1e-12);
  }
}

/// On the diagonal $d = k$ the trace-reduced value is the cochain density
/// $c_tau \/ vol_g (tau)$, constant across the simplex however the point is
/// chosen, the one degree of freedom the lowest-order element carries there.
///
/// The anchoring number of the reduction: a volume computed from the edge
/// lengths, against a value the reduction reaches through the star and the
/// trace. Single-valued with no averaging, since the trace onto a
/// $k$-simplex reads only that simplex's own DOF.
#[test]
fn trace_diagonal_is_cochain_density() {
  use regge::{cell_volume, coord::mesh::unit_coord_complex};
  for n in 1..=3 {
    let (topology, coords) = unit_coord_complex(n);
    let geometry = coords.to_edge_lengths_sq(&topology);
    for k in 1..=n {
      let ndofs = topology.nsimplices(k);
      let cochain = Cochain::new(
        k,
        Vector::from_iterator(ndofs, (0..ndofs).map(|i| (i + 1) as f64)),
      );
      for tau in topology.skeleton(k).handle_iter() {
        // Magnitude of the density. Its sign, where it has one, is governed by
        // orientation, not the point on the simplex, which is what this pins.
        let expected = (cochain[tau] / cell_volume(&geometry.simplex_metric(tau))).abs();
        for shift in [0.0, 0.13] {
          let mut w = Vector::from_element(k + 1, (1.0 - shift) / (k + 1) as f64);
          w[0] += shift;
          let value = trace_value(&topology, &geometry, &cochain, tau, &Bary::new(w));
          assert!((value.abs() - expected).abs() < 1e-9, "n={n} k={k}");
        }
      }
    }
  }
}
