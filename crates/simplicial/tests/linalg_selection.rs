//! A [`Selection`] of coordinates: restriction and scatter are transposes,
//! and a selection partitions the space with its complement.

use simplicial::linalg::{Matrix, Selection};

/// Restriction and extension by zero are transposes, and compose to the
/// identity on the subspace.
#[test]
fn the_two_maps_are_transposes_and_split() {
  let selection = Selection::new(6, vec![1, 2, 5]);
  let restriction = Matrix::from(&selection.restriction());

  let scattered = Matrix::from_column_slice(6, 1, &selection.scatter(&[2.0, 3.0, 5.0]));
  assert_eq!(
    scattered,
    Matrix::from_column_slice(6, 1, &[0.0, 2.0, 3.0, 0.0, 0.0, 5.0])
  );
  assert_eq!(
    restriction.transpose() * Matrix::from_column_slice(3, 1, &[2.0, 3.0, 5.0]),
    scattered
  );
  assert_eq!(
    &restriction * &restriction.transpose(),
    Matrix::identity(3, 3)
  );
}

/// A selection and its complement partition the space, and each is the
/// other's complement.
#[test]
fn a_selection_and_its_complement_partition_the_space() {
  for total in 0..=5 {
    for mask in 0..1u32 << total {
      let selection = Selection::new(total, (0..total).filter(|i| mask >> i & 1 == 1).collect());
      let complement = selection.complement();

      assert_eq!(selection.len() + complement.len(), total);
      assert_eq!(complement.complement(), selection);
      for coordinate in 0..total {
        assert!(
          selection.position(coordinate).is_some() ^ complement.position(coordinate).is_some()
        );
      }
    }
  }
}

/// `excluding` is the complement of the excluded coordinates, and the empty
/// and full selections are the degenerate ends of that.
#[test]
fn excluding_bottoms_out_at_the_empty_and_full_selections() {
  let full = Selection::excluding(4, []);
  assert_eq!(full.indices(), [0, 1, 2, 3]);
  let empty = Selection::excluding(4, 0..4);
  assert!(empty.is_empty());
  assert_eq!(empty.restriction().nrows(), 0);
  assert_eq!(empty.scatter::<f64>(&[]), vec![0.0; 4]);
}
