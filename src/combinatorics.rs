pub mod orientation;
pub mod simplex;

pub use orientation::Orientation;
pub use simplex::{OrientedSimplex, OrderedSimplex, SortedSimplex};

use crate::Dim;

use num_integer::binomial;

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
  use super::{simplex::SortedSimplex, sort_count_swaps, Permutations};

  #[test]
  fn subs_order() {
    let dim = 2;
    let nvertices = dim + 1;
    let simp = SortedSimplex::new((0..nvertices).collect());
    let subs: Vec<_> = simp
      .subs(1)
      .into_iter()
      .map(|s| s.into_vertices())
      .collect();
    assert_eq!(subs, vec![&[0, 1], &[0, 2], &[1, 2]]);
  }

  #[test]
  fn sorted_simplex() {
    for dim in 0..5 {
      let nvertices = dim + 1;
      let simp = SortedSimplex::new((0..nvertices).collect());
      for sub_dim in 0..dim {
        assert!(simp.subs(sub_dim).into_iter().all(|sub| sub < simp));
      }
      assert!(simp.subs(dim).into_iter().all(|sub| sub == simp));
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
