//! A sub-basis of a coordinate space, and the two maps it is.

use super::CooMatrix;

use num_traits::Zero;

/// A choice of coordinates out of `total`, in increasing order: a sub-basis of
/// $R^"total"$, hence a subspace together with a splitting of it.
///
/// It is two maps at once, and both are needed wherever one is. Reading a
/// vector of the whole space in the subspace is the restriction
/// $R^"total" -> R^"len"$ ([`position`](Self::position) coordinatewise,
/// [`restriction`](Self::restriction) as a matrix); writing one back out is the
/// extension by zero $R^"len" -> R^"total"$ ([`scatter`](Self::scatter)), the
/// section that splits it.
///
/// Selecting is monotone, so the two carry no sign and compose to the identity
/// on the subspace. What they do not compose to is the identity on the whole
/// space: the other way round they are the projection along the complement,
/// which is the content of [`complement`](Self::complement) being a second
/// selection rather than the same one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
  total: usize,
  /// Per coordinate of the whole space, its position in the selection.
  position: Vec<Option<usize>>,
  indices: Vec<usize>,
}

impl Selection {
  /// The selected coordinates, given in increasing order.
  ///
  /// # Panics
  /// If they are not increasing or one lies outside the space.
  pub fn new(total: usize, indices: Vec<usize>) -> Self {
    assert!(
      indices.windows(2).all(|w| w[0] < w[1]),
      "a selection is increasing"
    );
    assert!(
      indices.last().is_none_or(|&last| last < total),
      "a selected coordinate must lie in the space"
    );
    let mut position = vec![None; total];
    for (place, &coordinate) in indices.iter().enumerate() {
      position[coordinate] = Some(place);
    }
    Self {
      total,
      position,
      indices,
    }
  }

  /// Every coordinate but the excluded ones, which is how a relative complex
  /// selects the simplices not in the subcomplex it is relative to.
  pub fn excluding(total: usize, excluded: impl IntoIterator<Item = usize>) -> Self {
    let mut kept = vec![true; total];
    for coordinate in excluded {
      kept[coordinate] = false;
    }
    Self::new(
      total,
      (0..total).filter(|&coordinate| kept[coordinate]).collect(),
    )
  }

  /// The dimension of the space selected from.
  pub fn total(&self) -> usize {
    self.total
  }
  /// The dimension of the subspace: how many coordinates are selected.
  pub fn len(&self) -> usize {
    self.indices.len()
  }
  pub fn is_empty(&self) -> bool {
    self.indices.is_empty()
  }
  /// The selected coordinates, increasing.
  pub fn indices(&self) -> &[usize] {
    &self.indices
  }
  /// The position of a coordinate within the selection, `None` if it is not
  /// selected.
  pub fn position(&self, coordinate: usize) -> Option<usize> {
    self.position[coordinate]
  }

  /// The coordinates this one leaves out, as a selection of the same space:
  /// the complementary summand.
  pub fn complement(&self) -> Self {
    Self::excluding(self.total, self.indices.iter().copied())
  }

  /// A vector on the selection, extended by zero to the whole space.
  ///
  /// # Panics
  /// If the vector does not have one entry per selected coordinate.
  pub fn scatter<T: Clone + Zero>(&self, selected: &[T]) -> Vec<T> {
    assert_eq!(
      selected.len(),
      self.len(),
      "one entry per selected coordinate"
    );
    let mut full = vec![T::zero(); self.total];
    for (&coordinate, value) in self.indices.iter().zip(selected) {
      full[coordinate] = value.clone();
    }
    full
  }

  /// The restriction $R^"total" -> R^"len"$ as a matrix: one $1$ per selected
  /// coordinate, and [`scatter`](Self::scatter) is its transpose.
  pub fn restriction(&self) -> CooMatrix {
    let mut matrix = CooMatrix::new(self.len(), self.total);
    for (place, &coordinate) in self.indices.iter().enumerate() {
      matrix.push(place, coordinate, 1.0);
    }
    matrix
  }
}

#[cfg(test)]
mod test {
  use super::*;

  use crate::linalg::Matrix;

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
}
