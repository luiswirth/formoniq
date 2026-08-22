//! Embedding a [`FlatQuotient`]: the Clifford embedding is isometric in every
//! dimension it needs, and a twisted quotient needs one more and is curved.

use multiindex::Dim;
use regge::lengths::LengthsSq;
use regge::mesher::quotient::FlatQuotient;
use regge::mesher::quotient_embed::{equivariant, is_isometric};
use simplicial::linalg::Vector;

/// The Clifford embedding is an isometry: the edge lengths it induces are
/// the flat quotient's own, in every dimension.
///
/// This is the law that ties the two constructions together, and it checks
/// both the generator's intrinsic geometry and the coordinate-to-lengths
/// bridge against each other, neither standing in for the other.
#[test]
fn the_clifford_embedding_is_isometric() {
  for dim in (1..=3usize).map(Dim::from) {
    let quotient = FlatQuotient::unit_torus(dim, 4);
    assert!(is_isometric(&quotient));

    let (complex, intrinsic) = quotient.triangulate();
    let coords = equivariant(&quotient, 1.0 + f64::EPSILON);
    assert_eq!(coords.dim().index(), 2 * dim.index());

    let induced = coords.to_edge_lengths_sq(&complex);
    for (a, b) in intrinsic.iter().zip(induced.iter()) {
      assert!((a - b).abs() < 1e-9, "dim {dim}: {a} vs {b}");
    }
  }
}

/// The twisted quotients land where the mathematics says they must: the
/// Möbius band and the Klein bottle both in $RR^4$, neither isometrically.
/// The Klein bottle has no $RR^3$ embedding at all, so the ambient count is
/// not a matter of taste.
#[test]
fn twisted_quotients_need_four_dimensions_and_are_not_isometric() {
  for quotient in [
    FlatQuotient::moebius(1.0, 0.4, 4),
    FlatQuotient::klein(Vector::from_element(2, 1.0), 4),
  ] {
    assert!(!is_isometric(&quotient));

    let (complex, intrinsic) = quotient.triangulate();
    let coords = equivariant(&quotient, 3.0);
    assert_eq!(coords.dim(), 4);

    let induced = coords.to_edge_lengths_sq(&complex);
    assert!(
      intrinsic
        .iter()
        .zip(induced.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6),
      "a twisted embedding is curved, so it cannot reproduce the flat lengths"
    );
    // Curved, but still a faithful realization: no edge collapses.
    assert!(induced.iter().all(|l| l > 0.0));
  }
}
