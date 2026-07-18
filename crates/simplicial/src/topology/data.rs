//! Data attached to simplices, keyed by their id.
//!
//! The mesh is an entity-component store in disguise: the simplices are
//! entities (identified by [`SimplexIdx`]), and everything defined *on* them
//! -- vertex coordinates, edge lengths, a cochain's coefficients, per-cell
//! metrics, markers -- is columnar data keyed by that id. This module holds
//! that idea as two traits and their dense implementations.
//!
//! - [`SkeletonData`] is data over the simplices of a *single* grade. It needs
//!   no [`Complex`](super::complex::Complex): it works on a bare skeleton,
//!   including the cell skeleton a complex is derived from.
//! - [`ComplexData`] is data over *all* grades at once, mirroring the complex's
//!   own graded storage.
//!
//! They are traits, not structs, because the backing storage varies: scalar or
//! metric columns are plain `Vec`s (returning `&T`), while vertex coordinates
//! are the columns of a matrix (returning a *view*). The associated
//! `Item<'_>` accommodates both. [`SkeletonVec`]/[`ComplexVec`] are the
//! `Vec`-backed implementations, and additionally support `store[id]` indexing
//! for the common owned-data case.

use super::handle::{KSimplexIdx, SimplexIdx, SimplexRef};
use crate::Dim;

use std::ops::{Index, IndexMut};

/// Data over the simplices of a single grade, read by `kidx`.
///
/// The columnar building block of the mesh: vertex coordinates (grade 0), edge
/// lengths (grade 1), a k-cochain, and per-cell metrics (grade n) are all
/// `SkeletonData` over their grade.
pub trait SkeletonData {
  /// A read view of one datum; `&T` for owned columns, a column view for
  /// matrix-backed ones.
  type Item<'a>
  where
    Self: 'a;

  fn grade(&self) -> Dim;
  fn len(&self) -> usize;
  fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// The datum of the `kidx`-th simplex of this grade.
  fn at(&self, kidx: KSimplexIdx) -> Self::Item<'_>;

  /// The datum at a simplex id, checking the grade matches.
  #[track_caller]
  fn at_id(&self, id: SimplexIdx) -> Self::Item<'_> {
    debug_assert_eq!(
      id.dim,
      self.grade(),
      "indexed grade-{} data with a grade-{} simplex",
      self.grade(),
      id.dim
    );
    self.at(id.kidx)
  }
  /// The datum at a simplex ref.
  fn at_ref(&self, simplex: SimplexRef) -> Self::Item<'_> {
    self.at_id(simplex.idx())
  }
}

/// Data over the simplices of every grade, read by [`SimplexIdx`].
pub trait ComplexData {
  type Item<'a>
  where
    Self: 'a;

  fn dim(&self) -> Dim;
  fn at(&self, id: SimplexIdx) -> Self::Item<'_>;
  fn at_ref(&self, simplex: SimplexRef) -> Self::Item<'_> {
    self.at(simplex.idx())
  }
}

/// The `Vec`-backed [`SkeletonData`]: dense owned data, one `T` per simplex of
/// a grade. Supports `array[id]` / `array[kidx]` indexing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SkeletonVec<T> {
  grade: Dim,
  values: Vec<T>,
}
impl<T> SkeletonVec<T> {
  pub fn new(grade: Dim, values: Vec<T>) -> Self {
    Self { grade, values }
  }
  pub fn from_fn(grade: Dim, len: usize, f: impl FnMut(KSimplexIdx) -> T) -> Self {
    Self::new(grade, (0..len).map(f).collect())
  }
  pub fn values(&self) -> &[T] {
    &self.values
  }
  pub fn values_mut(&mut self) -> &mut [T] {
    &mut self.values
  }
  pub fn into_values(self) -> Vec<T> {
    self.values
  }
  pub fn iter(&self) -> std::slice::Iter<'_, T> {
    self.values.iter()
  }
}
impl<T> SkeletonData for SkeletonVec<T> {
  type Item<'a>
    = &'a T
  where
    Self: 'a;
  fn grade(&self) -> Dim {
    self.grade
  }
  fn len(&self) -> usize {
    self.values.len()
  }
  fn at(&self, kidx: KSimplexIdx) -> &T {
    &self.values[kidx]
  }
}

impl<T> Index<KSimplexIdx> for SkeletonVec<T> {
  type Output = T;
  fn index(&self, kidx: KSimplexIdx) -> &T {
    &self.values[kidx]
  }
}
impl<T> IndexMut<KSimplexIdx> for SkeletonVec<T> {
  fn index_mut(&mut self, kidx: KSimplexIdx) -> &mut T {
    &mut self.values[kidx]
  }
}
impl<T> Index<SimplexIdx> for SkeletonVec<T> {
  type Output = T;
  #[track_caller]
  fn index(&self, id: SimplexIdx) -> &T {
    debug_assert_eq!(id.dim, self.grade, "grade mismatch");
    &self.values[id.kidx]
  }
}
impl<T> IndexMut<SimplexIdx> for SkeletonVec<T> {
  #[track_caller]
  fn index_mut(&mut self, id: SimplexIdx) -> &mut T {
    debug_assert_eq!(id.dim, self.grade, "grade mismatch");
    &mut self.values[id.kidx]
  }
}
impl<T> Index<SimplexRef<'_>> for SkeletonVec<T> {
  type Output = T;
  fn index(&self, simplex: SimplexRef<'_>) -> &T {
    &self[simplex.idx()]
  }
}

/// The `Vec`-backed [`ComplexData`]: one [`SkeletonVec`] per grade.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComplexVec<T> {
  grades: Vec<SkeletonVec<T>>,
}
impl<T> ComplexVec<T> {
  pub fn new(grades: Vec<SkeletonVec<T>>) -> Self {
    for (dim, array) in grades.iter().enumerate() {
      assert_eq!(array.grade(), dim, "grade array is out of order");
    }
    Self { grades }
  }
  pub fn grade(&self, dim: Dim) -> &SkeletonVec<T> {
    &self.grades[dim]
  }
  pub fn grade_mut(&mut self, dim: Dim) -> &mut SkeletonVec<T> {
    &mut self.grades[dim]
  }
  pub fn grades(&self) -> impl Iterator<Item = &SkeletonVec<T>> {
    self.grades.iter()
  }
}
impl<T> ComplexData for ComplexVec<T> {
  type Item<'a>
    = &'a T
  where
    Self: 'a;
  fn dim(&self) -> Dim {
    self.grades.len() - 1
  }
  fn at(&self, id: SimplexIdx) -> &T {
    &self.grades[id.dim][id.kidx]
  }
}
impl<T> Index<SimplexIdx> for ComplexVec<T> {
  type Output = T;
  fn index(&self, id: SimplexIdx) -> &T {
    &self.grades[id.dim][id.kidx]
  }
}
impl<T> IndexMut<SimplexIdx> for ComplexVec<T> {
  fn index_mut(&mut self, id: SimplexIdx) -> &mut T {
    &mut self.grades[id.dim][id.kidx]
  }
}
impl<T> Index<SimplexRef<'_>> for ComplexVec<T> {
  type Output = T;
  fn index(&self, simplex: SimplexRef<'_>) -> &T {
    &self[simplex.idx()]
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn skeleton_vec_indexing() {
    let edges = SkeletonVec::new(1, vec![10.0, 20.0, 30.0]);
    assert_eq!(edges[2usize], 30.0);
    assert_eq!(edges[SimplexIdx::new(1, 1)], 20.0);
    assert_eq!(edges.at_id(SimplexIdx::new(1, 0)), &10.0);
    assert_eq!(edges.grade(), 1);
    assert_eq!(edges.len(), 3);
  }

  #[test]
  fn complex_vec_indexing() {
    let data = ComplexVec::new(vec![
      SkeletonVec::new(0, vec![1, 2, 3]),
      SkeletonVec::new(1, vec![4, 5]),
    ]);
    assert_eq!(data[SimplexIdx::new(0, 2)], 3);
    assert_eq!(data[SimplexIdx::new(1, 0)], 4);
    assert_eq!(data.at(SimplexIdx::new(1, 1)), &5);
    assert_eq!(data.dim(), 1);
  }
}
