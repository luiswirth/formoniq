//! The coherent orientation across interior facets: the induced boundary
//! orientations cancel, reversal is an involution, and the return type is an
//! `Option` because a Möbius band admits none.

mod common;

use common::two_sphere;
use simplicial::Dim;
use simplicial::Sign;
use simplicial::topology::complex::Complex;
use simplicial::topology::role::{Cell, roles};
use simplicial::topology::simplex::Simplex;
use simplicial::topology::skeleton::Skeleton;

fn complex_of(cells: &[&[usize]]) -> Complex {
  Complex::from_cells(Skeleton::new(
    cells
      .iter()
      .map(|c| Simplex::from_word(c.to_vec()).1)
      .collect(),
  ))
}

/// The defining law, checked directly on the boundary operator: for every
/// interior facet the two induced orientations cancel.
fn assert_coherent(complex: &Complex) {
  let orientation = complex.orientation().expect("orientable");
  // The total accessor: a 0-complex has no facets, so the law is vacuous
  // rather than a case to exclude.
  let Some(facets) = complex.role_skeleton::<roles::Facet>() else {
    return;
  };
  for facet in facets.handle_iter() {
    let (a, b) = facet.adjacent_cells();
    let Some(b) = b else { continue };
    let induced = |cell: Cell| {
      cell
        .get()
        .boundary()
        .find(|(_, sub)| sub.idx() == facet.idx())
        .unwrap()
        .0
    };
    assert_eq!(
      (orientation.sign(a) * induced(a)).other(),
      orientation.sign(b) * induced(b),
      "induced orientations must cancel on an interior facet"
    );
  }
}

/// The unit simplex, at every dimension including the 0-complex whose
/// constraint set is empty.
#[test]
fn unit_simplex_is_orientable() {
  for dim in (0..=4usize).map(Dim::from) {
    let complex = Complex::unit(dim);
    assert!(complex.is_orientable());
    assert_coherent(&complex);
  }
}

/// A Möbius band: a triangulated strip glued with a flip. The smallest
/// non-orientable surface, and the reason the return type is an `Option`.
#[test]
fn moebius_band_is_not_orientable() {
  // Five quads around a strip, the last glued to the first with the two
  // boundary vertices exchanged.
  let mut cells: Vec<&[usize]> = Vec::new();
  let quads: [[usize; 4]; 5] = [
    [0, 1, 2, 3],
    [2, 3, 4, 5],
    [4, 5, 6, 7],
    [6, 7, 8, 9],
    // the flip: 0 and 1 swapped relative to the untwisted gluing
    [8, 9, 1, 0],
  ];
  let mut owned: Vec<Vec<usize>> = Vec::new();
  for q in quads {
    owned.push(vec![q[0], q[1], q[2]]);
    owned.push(vec![q[1], q[2], q[3]]);
  }
  for c in &owned {
    cells.push(c);
  }
  let complex = complex_of(&cells);
  assert!(!complex.is_orientable());
  assert!(complex.orientation().is_none());
}

/// Orientability is per component, and a disconnected complex is orientable
/// exactly when each component is.
#[test]
fn disconnected_components_are_oriented_independently() {
  let complex = complex_of(&[&[0, 1, 2], &[3, 4, 5]]);
  assert!(complex.is_orientable());
  assert_coherent(&complex);
  assert_eq!(complex.orientation().unwrap().signs().len(), 2);
}

/// Reversal is an involution and stays coherent: the other generator.
#[test]
fn reversal_is_an_involution() {
  let complex = two_sphere();
  let orientation = complex.orientation().unwrap();
  assert_eq!(&orientation.reversed().reversed(), orientation);
  assert!(
    orientation
      .reversed()
      .signs()
      .iter()
      .zip(orientation.signs())
      .all(|(a, b)| a != b)
  );
}

/// The sphere is orientable, and its colex frames genuinely disagree, the
/// orientation is doing work, not returning all-`Pos`.
#[test]
fn sphere_is_orientable_and_not_trivially_signed() {
  let complex = two_sphere();
  assert!(complex.is_orientable());
  assert_coherent(&complex);
  let signs = complex.orientation().unwrap().signs();
  assert!(
    signs.contains(&Sign::Neg),
    "colex order is not already coherent on the sphere"
  );
}
