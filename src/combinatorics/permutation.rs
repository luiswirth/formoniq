use super::Sign;

pub fn factorial(num: usize) -> usize {
  (1..=num).product()
}

/// Iterator of alternating permutations.
///
/// Iterator implementation of the Steinhaus–Johnson–Trotter algorithm.
/// This iterator produces all permutations of a `Vec<T>`, where two consecutive
/// permutations differ only by a single swap of two adjacent elements. This
/// property ensures that the parity (even/odd) of the permutations alternate.
pub struct AlternatingPermutations<T: Clone> {
  vec: Vec<T>,
  idxs: Vec<usize>,
  dirs: Vec<Sign>,
  first: bool,
}

impl<T: Clone> AlternatingPermutations<T> {
  pub fn new(vec: Vec<T>) -> Self {
    let n = vec.len();
    let idxs = (0..n).collect();
    let dirs = vec![Sign::Neg; n];
    let first = true;
    Self {
      vec,
      idxs,
      dirs,
      first,
    }
  }
}

impl<T: Clone> Iterator for AlternatingPermutations<T> {
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
      if ((self.dirs[i] == Sign::Neg && 0 < i && self.idxs[i - 1] < self.idxs[i])
        || (self.dirs[i] == Sign::Pos && i < n - 1 && self.idxs[i] > self.idxs[i + 1]))
        && (imobile.is_none() || self.idxs[i] > self.idxs[imobile.unwrap()])
      {
        imobile = Some(i);
      }
    }

    imobile.map(|imobile| {
      let iswap = if self.dirs[imobile] == Sign::Neg {
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

#[cfg(test)]
mod test {
  use super::AlternatingPermutations;
  use crate::combinatorics::{sort_count_swaps, Sign};

  #[test]
  fn permutations_and_sort() {
    for n in 0..5 {
      let vec: Vec<_> = (0..n).collect();
      let pers = AlternatingPermutations::new(vec.clone());
      let mut max_nswaps = 0;
      for (i, p) in pers.enumerate() {
        let mut sorted = p.clone();
        let nswaps = sort_count_swaps(&mut sorted);
        max_nswaps = max_nswaps.max(nswaps);

        // must be sorted
        assert_eq!(vec, sorted);
        // permutation parity must alternate
        assert_eq!(Sign::from_parity(i), Sign::from_parity(nswaps));
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
