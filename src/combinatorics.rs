pub fn factorial(num: usize) -> usize {
  (1..=num).product()
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
  arr: Vec<T>,
  indices: Vec<usize>,
  directions: Vec<Dir>,
  first: bool,
}

impl<T: Clone> Permutations<T> {
  pub fn new(arr: Vec<T>) -> Self {
    let n = arr.len();
    Permutations {
      arr,
      indices: (0..n).collect(),
      directions: vec![Dir::Neg; n],
      first: true,
    }
  }
}

impl<T: Clone> Iterator for Permutations<T> {
  type Item = Vec<T>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.first {
      self.first = false;
      return Some(self.arr.clone());
    }

    let n = self.indices.len();
    let mut mobile_index: Option<usize> = None;

    for i in 0..n {
      if ((self.directions[i] == Dir::Neg && i > 0 && self.indices[i] > self.indices[i - 1])
        || (self.directions[i] == Dir::Pos && i < n - 1 && self.indices[i] > self.indices[i + 1]))
        && (mobile_index.is_none() || self.indices[i] > self.indices[mobile_index.unwrap()])
      {
        mobile_index = Some(i);
      }
    }

    if let Some(mobile_index) = mobile_index {
      let swap_with = if self.directions[mobile_index] == Dir::Neg {
        mobile_index - 1
      } else {
        mobile_index + 1
      };

      self.indices.swap(mobile_index, swap_with);
      self.directions.swap(mobile_index, swap_with);

      for i in 0..n {
        if self.indices[i] > self.indices[swap_with] {
          self.directions[i] = -self.directions[i];
        }
      }

      return Some(self.indices.iter().map(|&i| self.arr[i].clone()).collect());
    }

    None
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
  use crate::combinatorics::sort_count_swaps;

  use super::Permutations;

  #[test]
  fn permutation_and_sort() {
    for n in 0..5 {
      let arr: Vec<_> = (0..n).collect();
      let pers = Permutations::new(arr.clone());
      let mut max_nswaps = 0;
      for (i, p) in pers.enumerate() {
        let mut sorted = p.clone();
        let nswaps = sort_count_swaps(&mut sorted);

        // must be sorted
        assert_eq!(arr, sorted);
        // permutation parity must alternate
        assert_eq!(i % 2, nswaps % 2);

        max_nswaps = max_nswaps.max(nswaps);
      }

      // this is the maximal number of swaps
      if n > 0 {
        assert_eq!(max_nswaps, n * (n - 1) / 2);
      } else {
        assert_eq!(max_nswaps, 0);
      }
    }
  }
}
