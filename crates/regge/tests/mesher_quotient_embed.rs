//! Embedding a [`FlatQuotient`]: the Clifford embedding is isometric in
//! every dimension it needs, a twisted quotient needs one more and is
//! curved, and the embedding separates the identified vertices.

use multiindex::Dim;
use regge::lengths::LengthsSq;
use regge::mesher::quotient::{FlatQuotient, Identification};
use regge::mesher::quotient_embed::{donut_r3, equivariant, is_isometric, moebius_r3};
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

/// An open axis is carried as itself, so a slab (no identification at all) is
/// embedded isometrically too, in $RR^d$ rather than $RR^(2d)$. The
/// degenerate end of the family, where the quotient is the grid.
#[test]
fn an_unidentified_slab_embeds_isometrically_in_its_own_dimension() {
  let quotient = FlatQuotient::new(
    Vector::from_element(2, 1.0),
    vec![Identification::Open, Identification::Open],
    3,
  );
  let (complex, intrinsic) = quotient.triangulate();
  let coords = equivariant(&quotient, 2.0);
  assert_eq!(coords.dim(), 2);

  let induced = coords.to_edge_lengths_sq(&complex);
  for (a, b) in intrinsic.iter().zip(induced.iter()) {
    assert!((a - b).abs() < 1e-9);
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

/// The embedding descends to the quotient: identified vertices are one
/// vertex, so the coordinates are single-valued, and distinct vertices stay
/// distinct. This is what "equivariant" buys, and it is the property the
/// half-angle frame exists to provide.
#[test]
fn the_embedding_separates_the_vertices() {
  for quotient in [
    FlatQuotient::unit_torus(Dim::new(2), 4),
    FlatQuotient::moebius(1.0, 0.4, 4),
    FlatQuotient::klein(Vector::from_element(2, 1.0), 4),
  ] {
    let coords = equivariant(&quotient, 3.0);
    let matrix = coords.matrix();
    for i in 0..quotient.nvertices() {
      for j in (i + 1)..quotient.nvertices() {
        let separation = (matrix.column(i) - matrix.column(j)).norm();
        assert!(separation > 1e-6, "vertices {i} and {j} coincide");
      }
    }
  }
}

/// The $RR^3$ pictures are three-dimensional, injective, and, the point
/// worth asserting, not isometric: their induced lengths are a
/// different manifold from the flat quotient that produced the topology.
#[test]
fn the_r3_pictures_are_curved() {
  let torus = FlatQuotient::unit_torus(Dim::new(2), 4);
  let strip = FlatQuotient::moebius(1.0, 0.4, 4);
  let cases = [
    (&torus, donut_r3(&torus, 0.4)),
    (&strip, moebius_r3(&strip, 2.0)),
  ];
  for (quotient, coords) in cases {
    assert_eq!(coords.dim(), 3);
    let (complex, intrinsic) = quotient.triangulate();
    let induced = coords.to_edge_lengths_sq(&complex);
    assert!(induced.iter().all(|l| l > 0.0));
    assert!(
      intrinsic
        .iter()
        .zip(induced.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6),
      "an RR^3 realization of a flat surface is curved"
    );
  }
}
