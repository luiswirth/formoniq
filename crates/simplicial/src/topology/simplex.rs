use crate::linalg::Matrix;
use multiindex::{binomial, combinations, Combination, Sign};

use super::VertexIdx;
use crate::Dim;

/// An abstract simplex: a strictly increasing list of vertex indices.
///
/// Always the canonical (sorted) representative of its vertex set;
/// orientation is not encoded in the ordering but carried explicitly as a
/// [`Sign`] (see [`SignedSimplex`]).
///
/// Combinatorially, a mesh simplex is a monotone injection of the local
/// positions ${0, dots, k}$ into the vertex alphabet: all sign combinatorics
/// (boundary, subsimplices) happens positionally in [`Combination`] and is
/// mapped through the vertex list by [`Self::select`].
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Simplex {
  pub vertices: Vec<VertexIdx>,
}
impl Simplex {
  pub fn new(vertices: Vec<VertexIdx>) -> Self {
    assert!(
      vertices.windows(2).all(|w| w[0] < w[1]),
      "Simplex vertices must be strictly increasing."
    );
    Self { vertices }
  }
  /// Canonicalize an arbitrarily ordered vertex list into the sign of its
  /// permutation and the sorted simplex.
  ///
  /// Panics on repeated vertices (degenerate simplex).
  pub fn from_word(mut vertices: Vec<VertexIdx>) -> (Sign, Self) {
    let sign = multiindex::sort_signed(&mut vertices);
    let simplex = Self::new(vertices);
    (sign, simplex)
  }
  pub fn standard(dim: Dim) -> Self {
    Self::new((0..=dim).collect())
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
  pub fn contains(&self, ivertex: VertexIdx) -> bool {
    self.vertices.binary_search(&ivertex).is_ok()
  }
}

/// The positional combinatorics, mapped through the vertex alphabet.
impl Simplex {
  /// The subsimplex at the given positions: the image of a combination of
  /// positions under the monotone vertex map.
  pub fn select(&self, positions: Combination) -> Self {
    Self::new(positions.iter().map(|p| self.vertices[p]).collect())
  }

  /// The local positions of this simplex's vertices within a supersimplex.
  pub fn relative_to(&self, sup: &Self) -> Combination {
    Combination::from_increasing(
      self
        .iter()
        .map(|v| sup.vertices.binary_search(&v).expect("Not a subsimplex.")),
    )
  }

  pub fn is_subsimplex_of(&self, sup: &Self) -> bool {
    let mut sup_iter = sup.iter();
    self.iter().all(|v| sup_iter.any(|s| s == v))
  }
  pub fn is_supersimplex_of(&self, sub: &Self) -> bool {
    sub.is_subsimplex_of(self)
  }

  /// The subsimplices of the given dimension, in colexicographic order of
  /// their local positions.
  pub fn subsimps(&self, sub_dim: Dim) -> impl Iterator<Item = Self> + use<'_> {
    combinations(self.nvertices(), sub_dim + 1).map(|positions| self.select(positions))
  }

  /// The boundary $diff sigma = sum_i (-1)^i (sigma without v_i)$:
  /// alternating positional deletions.
  pub fn boundary(&self) -> impl Iterator<Item = SignedSimplex> + use<'_> {
    Combination::full(self.nvertices())
      .deletions()
      .map(|(sign, _, positions)| self.select(positions).with_sign(sign))
  }

  pub fn supersimps<'a>(
    &'a self,
    super_dim: Dim,
    root: &'a Self,
  ) -> impl Iterator<Item = Self> + 'a {
    root
      .subsimps(super_dim)
      .filter(|sup| self.is_subsimplex_of(sup))
  }
}

impl Simplex {
  pub fn iter(&self) -> std::iter::Copied<std::slice::Iter<'_, usize>> {
    self.vertices.iter().copied()
  }
}

/// Simplices are ordered **colexicographically**: compare from the largest
/// vertex downward. For same-cardinality simplices this is the order of their
/// colex rank, so sorting a skeleton by this order is the canonical numbering.
impl Ord for Simplex {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.vertices.iter().rev().cmp(other.vertices.iter().rev())
  }
}
impl PartialOrd for Simplex {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    Some(self.cmp(other))
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
impl From<Combination> for Simplex {
  /// A local simplex: the combination's indices as vertices.
  fn from(combination: Combination) -> Self {
    Self::new(combination.iter().collect())
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

/// The subsimplices of the standard simplex: local vertex sets,
/// in colexicographic order.
pub fn standard_subsimps(dim_cell: Dim, dim_sub: Dim) -> impl Iterator<Item = Combination> {
  combinations(dim_cell + 1, dim_sub + 1)
}
pub fn nsubsimplices(dim_cell: Dim, dim_sub: Dim) -> usize {
  binomial(dim_cell + 1, dim_sub + 1)
}
pub fn nedges(dim_cell: Dim) -> usize {
  nsubsimplices(dim_cell, 1)
}

/// $diff^k: Delta_k (Delta^n) -> Delta_(k-1) (Delta^n)$
///
/// Matrix of the boundary operator between the colexicographically ordered
/// subsimplices of the standard `dim_cell`-simplex.
///
/// Unaugmented: the boundary of vertices is zero, not the augmentation map
/// to the (-1)-simplex that `boundary_matrix` (below) computes.
pub fn standard_boundary_operator(dim_cell: Dim, dim_simp: Dim) -> Matrix {
  if dim_simp == 0 {
    return Matrix::zeros(0, dim_cell + 1);
  }
  boundary_matrix(dim_cell + 1, dim_simp + 1)
}

/// Matrix of the boundary operator
/// $diff: "colex-ordered card-subsets of" {0,..,n-1} -> "(card-1)-subsets"$,
/// built from the alternating deletions. Satisfies $diff compose diff = 0$.
///
/// Augmented: for card 1 this is the augmentation map onto the empty set
/// (the single (-1)-simplex of the reduced chain complex).
fn boundary_matrix(n: usize, card: usize) -> Matrix {
  if card == 0 {
    return Matrix::zeros(0, binomial(n, 0));
  }
  let mut matrix = Matrix::zeros(binomial(n, card - 1), binomial(n, card));
  for (isup, sup) in combinations(n, card).enumerate() {
    for (sign, _, sub) in sup.deletions() {
      matrix[(sub.rank(), isup)] = sign.as_f64();
    }
  }
  matrix
}

#[cfg(test)]
mod test {
  use super::*;

  use itertools::Itertools;

  /// $diff compose diff = 0$ for the combinatorial boundary matrices.
  #[test]
  fn boundary_matrix_squares_to_zero() {
    for n in 1..=6 {
      for card in 2..=n {
        let product = boundary_matrix(n, card - 1) * boundary_matrix(n, card);
        assert!(product.iter().all(|&v| v == 0.0));
      }
    }
  }

  #[test]
  fn subsimps() {
    for dim in 0..=4 {
      let simp = Simplex::standard(dim);
      for sub_dim in 0..=dim {
        let subs = simp.subsimps(sub_dim).collect_vec();
        assert_eq!(subs.len(), nsubsimplices(dim, sub_dim));
        assert!(subs.iter().all(|sub| sub.is_subsimplex_of(&simp)));
        assert!(subs
          .iter()
          .all(|sub| sub.relative_to(&simp) == Combination::from_increasing(sub.iter())));
      }
    }
  }

  #[test]
  fn from_word_orientation() {
    let (sign, simp) = Simplex::from_word(vec![2, 0, 1]);
    assert_eq!(sign, Sign::Pos);
    assert_eq!(simp, Simplex::from([0, 1, 2]));
    let (sign, _) = Simplex::from_word(vec![1, 0, 2]);
    assert_eq!(sign, Sign::Neg);
  }

  /// The boundary of the boundary is zero.
  #[test]
  fn boundary_of_boundary_cancels() {
    use std::collections::HashMap;
    let simp = Simplex::standard(3);
    let mut chain: HashMap<Simplex, i32> = HashMap::new();
    for face in simp.boundary() {
      for subface in face.simplex.boundary() {
        *chain.entry(subface.simplex).or_default() += (face.sign * subface.sign).as_i32();
      }
    }
    assert!(chain.values().all(|&c| c == 0));
  }
}
