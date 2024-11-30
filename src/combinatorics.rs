mod orientation;
mod vertplex;

pub use orientation::Orientation;
pub use vertplex::*;

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

/// Computes the lexicographic rank of a k-combination of {0,...,n-1}, where k = `combination.len()`.
pub fn rank_of_combination(combination: &[usize], n: usize) -> usize {
  let k = combination.len();

  let mut rank = 0;
  let mut iprefix = 0;
  for (i, &v) in combination.iter().enumerate() {
    for j in iprefix..v {
      rank += binomial(n - 1 - j, k - 1 - i);
    }
    iprefix = v + 1;
  }
  rank
}

/// Get the k-combination of {0,...,n-1} from its lexicographic rank.
pub fn combination_of_rank(mut rank: usize, n: usize, k: usize) -> Vec<usize> {
  let mut combination = Vec::with_capacity(k);
  let mut curr = 0;
  for i in 0..k {
    while rank >= binomial(n - 1 - curr, k - 1 - i) {
      rank -= binomial(n - 1 - curr, k - 1 - i);
      curr += 1;
    }
    combination.push(curr);
    curr += 1;
  }
  combination
}

pub fn generate_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
  let mut combinations = Vec::new();
  let mut combination: Vec<_> = (0..k).collect();

  loop {
    combinations.push(combination.clone());

    // Find the rightmost element that can be incremented
    let mut i = k;
    while i > 0 && combination[i - 1] == n - k + i - 1 {
      i -= 1;
    }

    // If all elements are at their maximum, we're done
    if i == 0 {
      break;
    }

    // Increment the current element
    combination[i - 1] += 1;

    // Reset the subsequent elements
    for j in i..k {
      combination[j] = combination[j - 1] + 1;
    }
  }

  combinations
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
  use crate::combinatorics::{
    combination_of_rank, generate_combinations, rank_of_combination, vertplex::CanonicalVertplex,
  };

  use super::{sort_count_swaps, Permutations};

  #[test]
  fn subs_order() {
    let dim = 2;
    let nvertices = dim + 1;
    let simp = CanonicalVertplex::new((0..nvertices).collect());
    let subs: Vec<_> = simp.subs(1).into_iter().map(|s| s.into_vec()).collect();
    assert_eq!(subs, vec![&[0, 1], &[0, 2], &[1, 2]]);
  }

  #[test]
  fn sorted_simplex() {
    for dim in 0..5 {
      let nvertices = dim + 1;
      let simp = CanonicalVertplex::new((0..nvertices).collect());
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

  #[test]
  fn combinations_of_4() {
    let n = 4;
    let combinations: [&[&[usize]]; 5] = [
      &[&[]],
      &[&[0], &[1], &[2], &[3]],
      &[&[0, 1], &[0, 2], &[0, 3], &[1, 2], &[1, 3], &[2, 3]],
      &[&[0, 1, 2], &[0, 1, 3], &[0, 2, 3], &[1, 2, 3]],
      &[&[0, 1, 2, 3]],
    ];
    for (k, &kcombinations) in combinations.iter().enumerate() {
      assert_eq!(generate_combinations(n, k), kcombinations);
    }
  }

  #[test]
  fn lexicographic_rank() {
    for n in 0..=5 {
      for k in 0..=n {
        let combinations = generate_combinations(n, k);
        for (rank, combination) in combinations.into_iter().enumerate() {
          let other_rank = rank_of_combination(&combination, n);
          assert_eq!(rank, other_rank);
          let other_combination = combination_of_rank(rank, n, k);
          assert_eq!(combination, other_combination);
        }
      }
    }
  }
}
