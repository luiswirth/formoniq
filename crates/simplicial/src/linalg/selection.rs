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
