use crate::{Dim, VertexIdx};

use itertools::Itertools;
use num_integer::binomial;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SortedSimplex(Vec<VertexIdx>);
impl SortedSimplex {
  pub fn new(mut vertices: Vec<VertexIdx>) -> Self {
    vertices.sort();
    Self(vertices)
  }
  pub fn new_unchecked(vertices: Vec<VertexIdx>) -> Self {
    debug_assert!(vertices.is_sorted());
    Self(vertices)
  }
  pub fn vertex(v: VertexIdx) -> SortedSimplex {
    Self(vec![v])
  }
  pub fn edge(a: VertexIdx, b: VertexIdx) -> Self {
    if a < b {
      Self(vec![a, b])
    } else {
      Self(vec![b, a])
    }
  }

  pub fn dim(&self) -> Dim {
    self.nvertices() - 1
  }
  pub fn nvertices(&self) -> usize {
    self.0.len()
  }
  pub fn vertices(&self) -> &[VertexIdx] {
    &self.0
  }

  pub fn subsimplicies(&self, dim: Dim) -> impl Iterator<Item = Self> + '_ {
    // TODO: don't rely on internals of itertools for ordering -> use own implementation
    self
      .0
      .iter()
      .copied()
      .combinations(dim + 1)
      .map(Self::new_unchecked)
  }
}
/// Implements the subsimplex/subset partial order relation.
///
/// This implementation is efficent, since it can rely on the fact that the
/// vertices are sorted.
impl PartialOrd for SortedSimplex {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering as O;
    let mut is_le = true;
    let mut is_ge = true;

    let mut this = self.0.iter().peekable();
    let mut other = other.0.iter().peekable();
    while let (Some(self_v), Some(other_v)) = (this.peek(), other.peek()) {
      match self_v.cmp(other_v) {
        O::Equal => {
          this.next();
          other.next();
        }
        O::Less => {
          is_le = false;
          this.next();
        }
        O::Greater => {
          is_ge = false;
          other.next();
        }
      }
    }

    // Check if we have remaining elements.
    if this.next().is_some() {
      is_le = false;
    }
    if other.next().is_some() {
      is_ge = false;
    }

    match (is_le, is_ge) {
      (true, true) => Some(O::Equal),
      (true, false) => Some(O::Less),
      (false, true) => Some(O::Greater),
      _ => None,
    }
  }
}

pub fn factorial(num: usize) -> usize {
  (1..=num).product()
}

pub fn nsubsimplicies(dim: Dim, dim_sub: Dim) -> usize {
  let nvertices = dim + 1;
  let nvertices_sub = dim_sub + 1;
  binomial(nvertices, nvertices_sub)
}
pub fn nsubedges(dim: Dim) -> usize {
  nsubsimplicies(dim, 1)
}

/// performs a bubble sort and counts the number of swaps
pub fn sort_count_swaps<T: Ord>(a: &mut [T]) -> usize {
  let mut nswaps = 0;

  let mut n = a.len();
  if n > 0 {
    let mut swapped = true;
    while swapped {
      swapped = false;
      for i in 1..n {
        if a[i - 1] > a[i] {
          a.swap(i - 1, i);
          swapped = true;
          nswaps += 1;
        }
      }
      n -= 1;
    }
  }
  nswaps
}

/// Iterator implementation of the Steinhaus–Johnson–Trotter algorithm.
///
/// This iterator produces all permutations of a `Vec<T>`, where two consecutive
/// permutations differ only by a single swap of two adjacent elements. This property
/// ensures that even and odd permutations alternate.
pub struct Permutations<T: Clone> {
  vec: Vec<T>,
  idxs: Vec<usize>,
  dirs: Vec<Dir>,
  first: bool,
}

impl<T: Clone> Permutations<T> {
  pub fn new(vec: Vec<T>) -> Self {
    let n = vec.len();
    Permutations {
      vec,
      idxs: (0..n).collect(),
      dirs: vec![Dir::Neg; n],
      first: true,
    }
  }
}

impl<T: Clone> Iterator for Permutations<T> {
  type Item = Vec<T>;

  // TODO: clean up this implementation
  fn next(&mut self) -> Option<Self::Item> {
    if self.first {
      self.first = false;
      return Some(self.vec.clone());
    }

    let n = self.vec.len();
    let mut imobile = None;

    for i in 0..n {
      if ((self.dirs[i] == Dir::Neg && 0 < i && self.idxs[i - 1] < self.idxs[i])
        || (self.dirs[i] == Dir::Pos && i < n - 1 && self.idxs[i] > self.idxs[i + 1]))
        && (imobile.is_none() || self.idxs[i] > self.idxs[imobile.unwrap()])
      {
        imobile = Some(i);
      }
    }

    imobile.map(|imobile| {
      let iswap = if self.dirs[imobile] == Dir::Neg {
        imobile - 1
      } else {
        imobile + 1
      };

      self.idxs.swap(imobile, iswap);
      self.dirs.swap(imobile, iswap);

      for i in 0..n {
        if self.idxs[i] > self.idxs[iswap] {
          self.dirs[i] = -self.dirs[i];
        }
      }

      self.idxs.iter().map(|&i| self.vec[i].clone()).collect()
    })
  }
}

/// implementation detail for [`Permutation`]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dir {
  Pos = 1,
  Neg = -1,
}
impl std::ops::Neg for Dir {
  type Output = Dir;
  fn neg(self) -> Self::Output {
    match self {
      Self::Pos => Self::Neg,
      Self::Neg => Self::Pos,
    }
  }
}

#[cfg(test)]
mod test {
  use super::{sort_count_swaps, Permutations, SortedSimplex};

  #[test]
  fn sorted_simplex() {
    for dim in 0..5 {
      let nvertices = dim + 1;
      let simp = SortedSimplex::new((0..nvertices).collect());
      for sub_dim in 0..dim {
        assert!(simp.subsimplicies(sub_dim).all(|sub| sub < simp));
      }
      assert!(simp.subsimplicies(dim).all(|sub| sub == simp));
    }
  }

  #[test]
  fn permutation_and_sort() {
    for n in 0..5 {
      let vec: Vec<_> = (0..n).collect();
      let pers = Permutations::new(vec.clone());
      let mut max_nswaps = 0;
      for (i, p) in pers.enumerate() {
        let mut sorted = p.clone();
        let nswaps = sort_count_swaps(&mut sorted);
        max_nswaps = max_nswaps.max(nswaps);

        // must be sorted
        assert_eq!(vec, sorted);
        // permutation parity must alternate
        assert_eq!(i % 2, nswaps % 2);
      }

      // maximal number of swaps must be this
      if n > 0 {
        assert_eq!(max_nswaps, n * (n - 1) / 2);
      } else {
        assert_eq!(max_nswaps, 0);
      }
    }
  }
}
