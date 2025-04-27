use common::combo::{binomial, nsubpermutations, sort_signed, Sign};

use super::VertexIdx;
use crate::Dim;

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
  pub vertices: Vec<VertexIdx>,
}
impl Simplex {
  pub fn new(vertices: Vec<VertexIdx>) -> Self {
    Self { vertices }
  }
  pub fn standard(dim: Dim) -> Self {
    Self::new((0..dim + 1).collect())
  }
  pub fn single(v: usize) -> Self {
    Self::new(vec![v])
  }

  pub fn with_sign(self, sign: Sign) -> SignedSimplex {
    SignedSimplex::new(self, sign)
  }

  pub fn nvertices(&self) -> usize {
    self.vertices.len()
  }
  pub fn dim(&self) -> Dim {
    self.nvertices() - 1
  }
}

impl Simplex {
  pub fn set_eq(&self, other: &Self) -> bool {
    self.clone().sorted() == other.clone().sorted()
  }
  pub fn is_permutation_of(&self, other: &Self) -> bool {
    self.set_eq(other)
  }
  pub fn permutations(&self) -> impl Iterator<Item = SignedSimplex> {
    self
      .subsets(self.dim())
      .enumerate()
      .map(|(i, simp)| SignedSimplex::new(simp, Sign::from_parity(i)))
  }

  pub fn is_subset_of(&self, other: &Self) -> bool {
    self.iter().all(|v| other.contains(v))
  }
  pub fn is_superset_of(&self, other: &Self) -> bool {
    other.is_subset_of(self)
  }
  pub fn subsets(&self, sub_dim: Dim) -> impl Iterator<Item = Self> {
    itertools::Itertools::permutations(self.clone().into_iter(), sub_dim + 1).map(Self::from)
  }

  pub fn is_subsequence_of(&self, sup: &Self) -> bool {
    let mut sup_iter = sup.iter();
    self.iter().all(|item| sup_iter.any(|x| x == item))
  }
  pub fn is_supersequence_of(&self, other: &Self) -> bool {
    other.is_subsequence_of(self)
  }
  /// The subsequence-simplices of this simplex.
  ///
  /// These are ordered lexicographically w.r.t. the local vertex indices.
  pub fn subsequences(&self, sub_dim: Dim) -> impl Iterator<Item = Self> {
    itertools::Itertools::combinations(self.clone().into_iter(), sub_dim + 1).map(Self::from)
  }
  pub fn boundary(&self) -> impl Iterator<Item = SignedSimplex> {
    let mut sign = Sign::from_parity(self.nvertices() - 1);
    self.subsequences(self.dim() - 1).map(move |simp| {
      let this_sign = sign;
      sign.flip();
      SignedSimplex::new(simp, this_sign)
    })
  }
  pub fn supersequences(
    &self,
    super_dim: Dim,
    root: &Self,
  ) -> impl Iterator<Item = Self> + use<'_> {
    root
      .subsequences(super_dim)
      .filter(|sup| self.is_subsequence_of(sup))
  }

  /// Computes local vertex numbers relative to sup.
  pub fn relative_to(&self, sup: &Self) -> Simplex {
    let local = self
      .iter()
      .map(|iglobal| {
        sup
          .iter()
          .position(|iother| iglobal == iother)
          .expect("Not a subset.")
      })
      .collect();
    Simplex::new(local)
  }
}

impl Simplex {
  pub fn iter(&self) -> std::iter::Copied<std::slice::Iter<'_, usize>> {
    self.vertices.iter().copied()
  }
}
impl IntoIterator for Simplex {
  type Item = usize;
  type IntoIter = std::vec::IntoIter<Self::Item>;
  fn into_iter(self) -> Self::IntoIter {
    self.vertices.into_iter()
  }
}

impl From<Vec<usize>> for Simplex {
  fn from(vertices: Vec<usize>) -> Self {
    Self::new(vertices)
  }
}
impl From<Simplex> for Vec<usize> {
  fn from(simp: Simplex) -> Self {
    simp.vertices
  }
}
impl<const N: usize> From<[usize; N]> for Simplex {
  fn from(vertices: [usize; N]) -> Self {
    Self::new(vertices.to_vec())
  }
}
impl<const N: usize> TryFrom<Simplex> for [usize; N] {
  type Error = Simplex;
  fn try_from(simp: Simplex) -> Result<Self, Self::Error> {
    simp.vertices.try_into().map_err(Simplex::new)
  }
}

impl std::ops::Index<usize> for Simplex {
  type Output = VertexIdx;
  fn index(&self, index: usize) -> &Self::Output {
    &self.vertices[index]
  }
}
impl std::ops::IndexMut<usize> for Simplex {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    &mut self.vertices[index]
  }
}

impl Simplex {
  pub fn is_sorted(&self) -> bool {
    self.vertices.is_sorted()
  }
  pub fn sort(&mut self) {
    self.vertices.sort_unstable()
  }
  pub fn sorted(mut self) -> Self {
    self.sort();
    self
  }
  pub fn sort_signed(&mut self) -> Sign {
    sort_signed(&mut self.vertices)
  }

  /// Orientation relative to sorted permutation.
  pub fn orientation_rel_sorted(&self) -> Sign {
    self.clone().sort_signed()
  }
  pub fn orientation_eq(&self, other: &Self) -> bool {
    assert!(self.set_eq(other));
    self.orientation_rel_sorted() == other.orientation_rel_sorted()
  }

  pub fn swap(&mut self, i: usize, j: usize) {
    self.vertices.swap(i, j)
  }
  pub fn contains(&self, ivertex: VertexIdx) -> bool {
    self.vertices.contains(&ivertex)
  }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct SignedSimplex {
  pub simplex: Simplex,
  pub sign: Sign,
}
impl SignedSimplex {
  pub fn new(simplex: Simplex, sign: Sign) -> Self {
    Self { simplex, sign }
  }
}

pub fn standard_subsimps(dim_cell: Dim, dim_sub: Dim) -> impl Iterator<Item = Simplex> {
  Simplex::standard(dim_cell).subsequences(dim_sub)
}
pub fn graded_subsimps(dim_cell: Dim) -> impl Iterator<Item = impl Iterator<Item = Simplex>> {
  (0..=dim_cell).map(move |d| standard_subsimps(dim_cell, d))
}

pub fn nsubsequence_simplices(dim_cell: Dim, dim_sub: Dim) -> usize {
  binomial(dim_cell + 1, dim_sub + 1)
}
pub fn nsubset_simplices(dim_cell: Dim, dim_sub: Dim) -> usize {
  nsubpermutations(dim_cell + 1, dim_sub + 1)
}

pub fn nedges(dim_cell: Dim) -> usize {
  nsubsequence_simplices(dim_cell, 1)
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::topology::simplex::nsubsequence_simplices;

  use itertools::Itertools;

  #[test]
  fn subsequence() {
    for dim in 0..=4 {
      let simp = Simplex::standard(dim);
      for sub_dim in 0..=dim {
        let subs = simp.subsequences(sub_dim).collect_vec();
        assert_eq!(subs.len(), nsubsequence_simplices(dim, sub_dim));
        assert!(subs.iter().all(|sub| sub.is_subset_of(&simp)));
        assert!(subs.iter().all(|sub| sub.is_subsequence_of(&simp)));
      }
    }
  }

  #[test]
  fn subset() {
    for dim in 0..=4 {
      let simp = Simplex::standard(dim);
      for sub_dim in 0..=dim {
        let subs = simp.subsets(sub_dim).collect_vec();
        assert_eq!(subs.len(), nsubset_simplices(dim, sub_dim));
        assert!(subs.iter().all(|sub| sub.is_subset_of(&simp)));
        let nsubsequences = subs
          .iter()
          .filter(|sub| sub.is_subsequence_of(&simp))
          .count();
        assert_eq!(nsubsequences, nsubsequence_simplices(dim, sub_dim));
      }
    }
  }
}
