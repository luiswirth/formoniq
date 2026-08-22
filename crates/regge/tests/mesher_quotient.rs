//! [`FlatQuotient`]: the flat torus, Möbius band and Klein bottle as
//! identified grids, their topology (Betti numbers, orientability), their
//! geometry (uniform, seam-free), and the ordering guarantee an identification
//! does or does not carry.

use multiindex::{Dim, binomial};
use regge::lengths::LengthsSq;
use regge::lengths::mesh::MeshLengthsSq;
use regge::mesher::quotient::{FlatQuotient, Identification};
use simplicial::linalg::Vector;
use simplicial::topology::complex::Complex;

fn shape_classes(complex: &Complex, lengths: &MeshLengthsSq) -> usize {
  let mut classes: Vec<Vec<u64>> = complex
    .cells()
    .handle_iter()
    .map(|cell| {
      let mut ls: Vec<f64> = lengths
        .simplex_lengths_sq(*cell)
        .vector()
        .iter()
        .copied()
        .collect();
      ls.sort_by(|a, b| a.partial_cmp(b).unwrap());
      let max = *ls.last().unwrap();
      ls.iter().map(|l| (l / max * 1e6).round() as u64).collect()
    })
    .collect();
  classes.sort_unstable();
  classes.dedup();
  classes.len()
}

/// A translational identification preserves the generator's Kuhn chain
/// order as a face-consistent ordering, and refining in it keeps the quotient
/// self-similar: one shape class. Translation carries the Kuhn tiling of one
/// box onto the tiling of the next, chain order included, so the seam is
/// indistinguishable from the interior.
#[test]
fn the_chain_order_survives_a_translational_identification() {
  for dim in (1..=3usize).map(Dim::from) {
    let (complex, lengths, ordering) = FlatQuotient::unit_torus(dim, 3).triangulate_ordered();
    let ordering = ordering.expect("the tiling matches across the seam");
    assert_eq!(shape_classes(&complex, &lengths), 1);

    let sub = complex.refine_with(&ordering, 2);
    let fine_lengths = lengths.refine(&sub, &complex);
    assert_eq!(shape_classes(sub.complex(), &fine_lengths), 1);
  }
}

/// A reflecting identification does not, and the generator says so rather
/// than handing back an ordering that lies.
///
/// The Kuhn triangulation of a box is not reflection-invariant: mirroring an
/// axis exchanges the diagonal, so the two sides of a twisted seam emit
/// incompatible chain orders on the face they share. The quotient is still
/// conforming, the exchanged diagonal is interior to a box, never on a
/// shared face, which is why the topology above is correct, but invariant 7
/// asks for more than conformity, and the Kuhn order cannot supply it here.
/// Refinement still works through the colex ordering. What is lost is the
/// guarantee that a refinement tower stays self-similar.
///
/// Recovering it needs a reflection-invariant triangulation of the box, not a
/// repair of this one.
#[test]
fn a_reflecting_identification_admits_no_kuhn_chain_order() {
  for quotient in [
    FlatQuotient::moebius(1.0, 1.0, 3),
    FlatQuotient::klein(Vector::from_element(2, 1.0), 3),
  ] {
    let (_, _, ordering) = quotient.triangulate_ordered();
    assert!(
      ordering.is_none(),
      "a reflecting seam cannot be face-consistent in the Kuhn order"
    );
  }
}

/// The flat torus is closed (no boundary) and carries the cohomology of
/// $T^d$: Betti numbers $b_k = binom(d, k)$, Euler characteristic $0$, for
/// every dimension.
#[test]
fn torus_topology() {
  for dim in (1..=3usize).map(Dim::from) {
    let (complex, lengths) = FlatQuotient::unit_torus(dim, 3).triangulate();

    assert!(!complex.has_boundary(), "dim {dim}: torus is boundaryless");
    assert_eq!(
      complex.nsimplices(Dim::new(0)),
      3usize.pow(dim.index() as u32)
    );

    let betti = complex.betti_numbers();
    let expected = (0..=dim.index())
      .map(|k| binomial(dim.index(), k))
      .collect::<Vec<_>>();
    assert_eq!(betti, expected, "dim {dim}: Betti numbers of T^d");
    assert_eq!(complex.euler_characteristic(), 0);
    assert!(
      complex.orientation().is_some(),
      "dim {dim}: T^d is orientable"
    );

    // The geometry is flat and uniform: every edge is spacelike, and the
    // shortest edges are the axis steps of length 1/n.
    assert!(lengths.iter().all(|s| s > 0.0));
    assert!((lengths.mesh_width_min() - 1.0 / 3.0).abs() < 1e-12);
  }
}

/// The Möbius band: a non-orientable surface with boundary, homotopy
/// equivalent to its core circle, so $b_0 = b_1 = 1$, $b_2 = 0$ and
/// $chi = 0$.
#[test]
fn moebius_topology() {
  for ncells in 3..=5 {
    let (complex, lengths) = FlatQuotient::moebius(1.0, 1.0, ncells).triangulate();

    assert!(complex.has_boundary(), "the band has a boundary circle");
    assert_eq!(complex.betti_numbers(), vec![1, 1, 0]);
    assert_eq!(complex.euler_characteristic(), 0);
    assert!(
      complex.orientation().is_none(),
      "the Möbius band is non-orientable"
    );
    assert!(lengths.iter().all(|s| s > 0.0));
  }
}

/// The Klein bottle: closed and non-orientable, so no fundamental class and
/// $b_2 = 0$. Over $RR$ the $ZZ_2$ torsion of $H_1$ is invisible, leaving
/// $b_0 = b_1 = 1$ and $chi = 0$.
#[test]
fn klein_topology() {
  for ncells in 3..=5 {
    let (complex, _) = FlatQuotient::klein(Vector::from_element(2, 1.0), ncells).triangulate();

    assert!(!complex.has_boundary(), "the Klein bottle is closed");
    assert_eq!(complex.betti_numbers(), vec![1, 1, 0]);
    assert_eq!(complex.euler_characteristic(), 0);
    assert!(
      complex.orientation().is_none(),
      "the Klein bottle is non-orientable"
    );
  }
}

/// Every identification leaves the geometry flat and uniform: the quotient is
/// a relabeling, so its edge lengths are exactly the grid's, seam included.
#[test]
fn identification_does_not_move_the_geometry() {
  let reference = {
    let (_, lengths) = FlatQuotient::new(
      Vector::from_element(2, 1.0),
      vec![Identification::Open, Identification::Open],
      4,
    )
    .triangulate();
    let mut ls: Vec<u64> = lengths.iter().map(|l| (l * 1e9).round() as u64).collect();
    ls.sort_unstable();
    ls.dedup();
    ls
  };
  for quotient in [
    FlatQuotient::unit_torus(Dim::new(2), 4),
    FlatQuotient::moebius(1.0, 1.0, 4),
    FlatQuotient::klein(Vector::from_element(2, 1.0), 4),
  ] {
    let (_, lengths) = quotient.triangulate();
    let mut ls: Vec<u64> = lengths.iter().map(|l| (l * 1e9).round() as u64).collect();
    ls.sort_unstable();
    ls.dedup();
    assert_eq!(ls, reference, "the seam introduced a new length");
  }
}
