use crate::linalg::Matrix;
use multiindex::{Combination, Sign, binomial, combinations};

use super::VertexIdx;
use crate::Dim;

/// An abstract simplex: a strictly increasing list of vertex indices.
///
/// Always the canonical (sorted) representative of its vertex set;
/// orientation is not encoded in the ordering but carried explicitly as a
/// [`Sign`] alongside it, as in [`Self::boundary`].
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
  pub fn unit(dim: impl Into<Dim>) -> Self {
    let dim = dim.into();
    Self::new((0..=dim.index()).collect())
  }
  pub fn single(v: usize) -> Self {
    Self::new(vec![v])
  }

  pub fn nvertices(&self) -> usize {
    self.vertices.len()
  }
  pub fn dim(&self) -> Dim {
    (self.nvertices() - 1).into()
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
  ///
  /// Total in the dimension: empty off the range $0 <= k <= dim sigma$, where
  /// there is no face to name. The empty simplex is not among them, so a
  /// vertex has no facets rather than one.
  pub fn subsimps(&self, sub_dim: Dim) -> impl Iterator<Item = Self> + use<'_> {
    sub_dim
      .index_in(self.dim())
      .into_iter()
      .flat_map(move |_| combinations(self.nvertices(), (sub_dim + 1).index()))
      .map(|positions| self.select(positions))
  }

  /// The boundary $diff sigma = sum_i (-1)^i (sigma without v_i)$:
  /// alternating positional deletions, each facet with the sign it carries.
  pub fn boundary(&self) -> impl Iterator<Item = (Sign, Self)> + use<'_> {
    Combination::full(self.nvertices())
      .deletions()
      .map(|(sign, _, positions)| (sign, self.select(positions)))
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

/// Simplices are ordered colexicographically: compare from the largest
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

/// The subsimplices of the unit simplex: local vertex sets,
/// in colexicographic order.
///
/// Total in the dimension, like [`Simplex::subsimps`] it mirrors: empty off
/// $0 <= k <= n$, where the unit cell has no face to name.
pub fn unit_subsimps(dim_cell: Dim, dim_sub: Dim) -> impl Iterator<Item = Combination> {
  dim_sub
    .index_in(dim_cell)
    .into_iter()
    .flat_map(move |_| combinations((dim_cell + 1).index(), (dim_sub + 1).index()))
}
/// How many there are, $binom(n+1, k+1)$, and zero off the range.
pub fn nsubsimplices(dim_cell: Dim, dim_sub: Dim) -> usize {
  dim_sub.index_in(dim_cell).map_or(0, |_| {
    binomial((dim_cell + 1).index(), (dim_sub + 1).index())
  })
}
pub fn nedges(dim_cell: Dim) -> usize {
  nsubsimplices(dim_cell, Dim::ONE)
}

/// $diff_k: Delta_k (hat(K)) -> Delta_(k-1) (hat(K))$, the boundary operator
/// between the colex-ordered subsimplices of the unit `dim_cell`-simplex, built
/// from the alternating positional deletions. Satisfies
/// $diff compose diff = 0$.
///
/// The reference-cell form of [`Complex::boundary_operator`], and the same
/// convention: unaugmented, so at grade $0$ it is the zero map into the zero
/// module rather than the augmentation onto the empty simplex.
///
/// [`Complex::boundary_operator`]: super::complex::Complex::boundary_operator
pub fn unit_boundary_operator(dim_cell: Dim, dim_simp: Dim) -> Matrix {
  let below = dim_simp - Dim::ONE;
  let mut matrix = Matrix::zeros(
    nsubsimplices(dim_cell, below),
    nsubsimplices(dim_cell, dim_simp),
  );
  for (icoface, coface) in unit_subsimps(dim_cell, dim_simp).enumerate() {
    // Off the range there is no face to scatter into: the deletions of a
    // vertex land on the empty simplex, which the complex does not carry.
    for (sign, _, face) in coface.deletions().filter(|_| below.in_range(dim_cell)) {
      matrix[(face.rank(), icoface)] = sign.as_f64();
    }
  }
  matrix
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::Dim;

  use itertools::Itertools;

  /// $diff compose diff = 0$ for the reference-cell boundary matrices, swept
  /// over every dimension and grade including the ends, where the operator is
  /// the zero map into or out of the zero module.
  #[test]
  fn unit_boundary_squares_to_zero() {
    for dim in (0..=5usize).map(Dim::from) {
      for grade in (dim + 1).range_inclusive() {
        let product = unit_boundary_operator(dim, grade - 1) * unit_boundary_operator(dim, grade);
        assert!(product.iter().all(|&v| v == 0.0), "dim {dim} grade {grade}");
      }
    }
  }

  #[test]
  fn subsimps() {
    for dim in (0..=4usize).map(Dim::from) {
      let simp = Simplex::unit(dim);
      for sub_dim in dim.range_inclusive() {
        let subs = simp.subsimps(sub_dim).collect_vec();
        assert_eq!(subs.len(), nsubsimplices(dim, sub_dim));
        assert!(subs.iter().all(|sub| sub.is_subsimplex_of(&simp)));
        assert!(
          subs
            .iter()
            .all(|sub| sub.relative_to(&simp) == Combination::from_increasing(sub.iter()))
        );
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
    let simp = Simplex::unit(Dim::new(3));
    let mut chain: HashMap<Simplex, i32> = HashMap::new();
    for (sign, face) in simp.boundary() {
      for (subsign, subface) in face.boundary() {
        *chain.entry(subface).or_default() += (sign * subsign).as_i32();
      }
    }
    assert!(chain.values().all(|&c| c == 0));
  }
}
