use std::marker::PhantomData;

use super::index_set::variants::*;
use super::IndexSet;

type ComplexSetImpl<O> = IndexSet<Unspecified, O, Unsigned>;
pub struct NKComplex<B: Specified, O: Order> {
  /// Graded lexciographically ordered combinations of base.
  graded_sets: Vec<Vec<ComplexSetImpl<O>>>,
  _base: PhantomData<B>,
}

impl NKComplex<Local, Sorted> {
  /// Combinations of canonical base {0,...,n-1}
  pub fn canonical(n: usize) -> Self {
    let graded_sets = (0..=n)
      .map(|k| IndexSet::counting(n).subsets(k).collect())
      .collect();
    Self {
      graded_sets,
      _base: PhantomData,
    }
  }
}

impl<B: Specified, O: Order> NKComplex<B, O> {
  pub fn top(&self) -> &ComplexSetImpl<O> {
    &self.graded_sets.last().unwrap()[0]
  }
  pub fn graded_sets(&self) -> &[Vec<ComplexSetImpl<O>>] {
    &self.graded_sets
  }

  pub fn into_raw(self) -> Vec<Vec<Vec<usize>>> {
    self
      .graded_sets
      .into_iter()
      .map(|ksets| ksets.into_iter().map(|kset| kset.into_vec()).collect())
      .collect()
  }
}

#[cfg(test)]
mod test {
  use crate::combo::complex::NKComplex;

  #[test]
  fn complex4() {
    let n = 4;
    let computed = NKComplex::canonical(n).into_raw();
    let expected: [&[&[usize]]; 5] = [
      &[&[]],
      &[&[0], &[1], &[2], &[3]],
      &[&[0, 1], &[0, 2], &[0, 3], &[1, 2], &[1, 3], &[2, 3]],
      &[&[0, 1, 2], &[0, 1, 3], &[0, 2, 3], &[1, 2, 3]],
      &[&[0, 1, 2, 3]],
    ];
    assert_eq!(computed, expected);
  }

  // TODO: repair this test
  //#[test]
  //fn lexicographic_rank() {
  //  for n in 0..=5 {
  //    let complex = NKComplex::canonical(n);

  //    let mut rank = 0;
  //    for (k, kcombinations) in complex.graded_sets().iter().enumerate() {
  //      for kcombination in kcombinations {
  //        let other_rank = kcombination.rank();

  //        rank += 1;
  //      }
  //      assert_eq!(k, other_rank);
  //      let other_combination = combination_of_rank(k, n, k);
  //      assert_eq!(kcombinations, other_combination);
  //    }
  //  }
  //}
}
