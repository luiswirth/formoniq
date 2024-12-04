mod permutation;
mod sign;

use std::marker::PhantomData;

pub use permutation::*;
pub use sign::*;

pub use num_integer::binomial;

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct IndexSet<B: Base, O: Order, S: Signedness> {
  indices: Vec<usize>,
  base: B,
  ord: O,
  sign: S,
}

trait Base: Clone {}
impl Base for Unspecified {}
impl Base for Local {}
impl Base for Global {}
trait Specified: Base {
  fn n(&self) -> usize;
  fn indices(&self) -> Vec<usize>;
}
impl Specified for Local {
  fn n(&self) -> usize {
    self.0
  }
  fn indices(&self) -> Vec<usize> {
    (0..self.0).collect()
  }
}
impl Specified for Global {
  fn n(&self) -> usize {
    self.0.len()
  }
  fn indices(&self) -> Vec<usize> {
    self.0.clone()
  }
}

#[derive(Debug, Default, Clone, Copy)]
struct Unspecified;
#[derive(Debug, Clone, Copy)]
struct Local(usize);
#[derive(Debug, Clone)]
struct Global(Vec<usize>);

trait Order: Clone + Copy {}
impl Order for Sorted {}
impl Order for Ordered {}

#[derive(Debug, Default, Clone, Copy)]
struct Sorted;
#[derive(Debug, Default, Clone, Copy)]
struct Ordered;

trait Signedness: Clone + Copy {
  fn get_or_default(&self) -> Sign;
}

#[derive(Debug, Default, Clone, Copy)]
struct Unsigned;
#[derive(Debug, Default, Clone, Copy)]
struct Signed(Sign);
impl Signedness for Unsigned {
  fn get_or_default(&self) -> Sign {
    Sign::default()
  }
}
impl Signedness for Signed {
  fn get_or_default(&self) -> Sign {
    self.0
  }
}

// Algebraic Topology
pub type SimplexFace<B, O, S> = IndexSet<B, O, S>;

pub type LocalSimplex<O, S> = SimplexFace<Local, O, S>;
pub type RefSimplexFace = SimplexFace<Local, Sorted, Unsigned>;

pub type MeshContainerSimplex = SimplexFace<Unspecified, Ordered, Unsigned>;
pub type MeshSimplex<O, S> = SimplexFace<Global, O, S>;

// Exterior Algebra
pub type WedgeTerm<B, O, S> = IndexSet<B, O, S>;
pub type FormComponent<B, O, S> = WedgeTerm<B, O, S>;
pub type CanonicalFormComponent = FormComponent<Local, Sorted, Unsigned>;

impl<B: Base, O: Order, S: Signedness> IndexSet<B, O, S> {
  pub fn forget_sign(self) -> IndexSet<B, O, Unsigned> {
    IndexSet {
      indices: self.indices,
      base: self.base,
      ord: self.ord,
      sign: Unsigned,
    }
  }

  pub fn assume_sorted(self) -> IndexSet<B, Sorted, S> {
    debug_assert!(self.indices.is_sorted());
    IndexSet {
      indices: self.indices,
      base: self.base,
      ord: Sorted,
      sign: self.sign,
    }
  }
}

impl<B: Base, O: Order, S: Signedness> IndexSet<B, O, S> {
  pub fn k(&self) -> usize {
    self.indices.len()
  }

  /// Combinations on boundary (ksub=`self.len - 1`) with their respective (alternating) signs.
  ///
  /// For a simplex this is the resulting k-chain after applying the boundary operator.
  pub fn signed_boundary(&self) -> impl Iterator<Item = IndexSet<B, O, Signed>> {
    let k = self.k();
    let self_sign = self.sign.get_or_default();
    self
      .clone()
      .forget_sign()
      .subs(self.k() - 1)
      .enumerate()
      .map(move |(i, sub)| {
        let boundary_sign = Sign::from_parity(k - 1 - i);
        let sign = boundary_sign * self_sign;
        sub.with_sign(sign)
      })
  }

  #[allow(unreachable_code, unused_variables)]
  pub fn signed_permutations(&self) -> impl Iterator<Item = IndexSet<B, Ordered, Signed>> {
    let self_sign = self.sign.get_or_default();
    self
      .clone()
      .forget_sign()
      .permutations()
      .enumerate()
      .map(move |(i, sub)| {
        // TODO
        let permutation_sign: Sign = todo!();
        let sign = permutation_sign * self_sign;
        sub.with_sign(sign)
      })
  }
}

impl<B: Base, O: Order> IndexSet<B, O, Unsigned> {
  /// Combinations of length `len` of the indicies of `self`.
  ///
  /// For a simplex this gives the subsimplicies.
  pub fn subs(&self, ksub: usize) -> impl Iterator<Item = Self> {
    let base = self.base.clone();
    let ord = self.ord;
    // TODO: stop relying on implementation details of itertools
    itertools::Itertools::combinations(self.indices.clone().into_iter(), ksub).map(move |indices| {
      Self {
        indices,
        base: base.clone(),
        ord,
        sign: Unsigned,
      }
    })
  }

  pub fn permutations(&self) -> impl Iterator<Item = IndexSet<B, Ordered, Unsigned>> {
    let base = self.base.clone();
    itertools::Itertools::permutations(self.indices.clone().into_iter(), self.k()).map(
      move |indices| IndexSet {
        indices,
        base: base.clone(),
        ord: Ordered,
        sign: Unsigned,
      },
    )
  }
}

impl<B: Base, O: Order, S: Signedness> std::ops::Index<usize> for IndexSet<B, O, S> {
  type Output = usize;
  fn index(&self, index: usize) -> &Self::Output {
    &self.indices[index]
  }
}

/// Subset partial order relation.
impl<B: Base, S: Signedness> IndexSet<B, Sorted, S> {
  pub fn is_subset_of(&self, other: &Self) -> bool {
    self.subset_cmp(other).map(|o| o.is_le()).unwrap_or(false)
  }
  pub fn is_superset_of(&self, other: &Self) -> bool {
    self.subset_cmp(other).map(|o| o.is_ge()).unwrap_or(false)
  }

  pub fn subset_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering as O;
    let mut is_le = true;
    let mut is_ge = true;

    let mut this = self.iter().peekable();
    let mut other = other.iter().peekable();
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

/// Lexicographical comparisons.
impl<B: Base, O: Order, S: Signedness> IndexSet<B, O, S> {
  // Ignores differing k and only compares indicies lexicographically.
  pub fn pure_lexicographical_cmp(&self, other: &Self) -> std::cmp::Ordering {
    use std::cmp::Ordering as O;
    self
      .iter()
      .zip(other.iter())
      .find_map(|(a, b)| match a.cmp(b) {
        O::Equal => None,
        non_eq => Some(non_eq),
      })
      .unwrap_or(O::Equal)
  }

  // Compares indicies lexicographically, only when lengths are equal.
  pub fn partial_lexicographical_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    if self.k() == other.k() {
      Some(self.pure_lexicographical_cmp(other))
    } else {
      None
    }
  }

  /// First compares indicies lexicographically, then the lengths.
  pub fn lexicographical_cmp(&self, other: &Self) -> std::cmp::Ordering {
    self
      .pure_lexicographical_cmp(other)
      .then(self.k().cmp(&other.k()))
  }

  /// First compares lengths, then indicies lexicographically.
  pub fn graded_lexicographical_cmp(&self, other: &Self) -> std::cmp::Ordering {
    self
      .k()
      .cmp(&other.k())
      .then_with(|| self.pure_lexicographical_cmp(other))
  }
}

// constructors
impl IndexSet<Unspecified, Ordered, Unsigned> {
  pub fn new(indices: Vec<usize>) -> Self {
    Self {
      indices,
      ..Default::default()
    }
  }
}

impl IndexSet<Unspecified, Sorted, Unsigned> {
  pub fn none() -> Self {
    Self::default()
  }
  pub fn single(index: usize) -> Self {
    IndexSet::new(vec![index]).assume_sorted()
  }
  pub fn counting(n: usize) -> Self {
    IndexSet::new((0..n).collect()).assume_sorted()
  }
}

impl<B: Base> IndexSet<B, Ordered, Unsigned> {
  pub fn into_sorted(self) -> IndexSet<B, Sorted, Unsigned> {
    let mut indices = self.indices;
    indices.sort_unstable();
    IndexSet {
      indices,
      base: self.base,
      ord: Sorted,
      sign: Unsigned,
    }
  }
  pub fn into_sorted_signed(self) -> IndexSet<B, Sorted, Signed> {
    let mut indices = self.indices;
    let sign = sort_signed(&mut indices);
    IndexSet {
      indices,
      base: self.base,
      ord: Sorted,
      sign: Signed(sign),
    }
  }
}

impl<O: Order, S: Signedness> IndexSet<Unspecified, O, S> {
  pub fn with_local_base(self, n: usize) -> IndexSet<Local, O, S> {
    assert!(self.iter().all(|i| *i < n));
    IndexSet {
      indices: self.indices,
      base: Local(n),
      ord: self.ord,
      sign: self.sign,
    }
  }
}

impl<B: Specified, O: Order, S: Signedness> IndexSet<B, O, S> {
  pub fn n(&self) -> usize {
    self.base.n()
  }
}

impl<B: Specified> IndexSet<B, Sorted, Unsigned> {
  pub fn sups(&self, ksup: usize) -> impl Iterator<Item = Self> {
    let base = Self {
      indices: self.base.indices(),
      base: self.base.clone(),
      ord: self.ord,
      sign: self.sign,
    };
    let this = self.clone();
    base.subs(ksup).filter(move |sup| this.is_subset_of(&sup))
  }
}

impl IndexSet<Local, Sorted, Unsigned> {
  pub fn from_rank(n: usize, k: usize, mut rank: usize) -> Self {
    let nlast = n - 1;
    let klast = k - 1;

    let mut indices = Vec::with_capacity(k);
    let mut curr_idx = 0;
    for i in 0..k {
      loop {
        let binom = binomial(nlast - curr_idx, klast - i);
        if rank < binom {
          break;
        }
        rank -= binom;
        curr_idx += 1;
      }
      indices.push(curr_idx);
      curr_idx += 1;
    }

    Self {
      indices,
      base: Local(n),
      ord: Sorted,
      sign: Unsigned,
    }
  }

  pub fn rank(&self) -> usize {
    let n = self.n();
    let k = self.k();

    let mut rank = 0;
    let mut icurr = 0;
    for (i, &v) in self.iter().enumerate() {
      for j in icurr..v {
        rank += binomial(n - 1 - j, k - 1 - i);
      }
      icurr = v + 1;
    }
    rank
  }
}

// Conversions

impl<B: Base, O: Order, S: Signedness> IndexSet<B, O, S> {
  pub fn iter(&self) -> std::slice::Iter<usize> {
    self.indices.iter()
  }
  pub fn as_slice(&self) -> &[usize] {
    self.indices.as_slice()
  }
  pub fn into_vec(self) -> Vec<usize> {
    self.indices
  }
  pub fn into_array<const N: usize>(self) -> Result<[usize; N], Vec<usize>> {
    self.into_vec().try_into()
  }
}

impl From<Vec<usize>> for IndexSet<Unspecified, Ordered, Unsigned> {
  fn from(value: Vec<usize>) -> Self {
    Self::new(value)
  }
}
impl<const N: usize> From<[usize; N]> for IndexSet<Unspecified, Ordered, Unsigned> {
  fn from(value: [usize; N]) -> Self {
    Self::new(value.to_vec())
  }
}

impl<B: Base, O: Order, S: Signedness> IndexSet<B, O, S> {
  pub fn with_sign(self, sign: Sign) -> IndexSet<B, O, Signed> {
    IndexSet {
      indices: self.indices,
      base: self.base,
      ord: self.ord,
      sign: Signed(sign),
    }
  }
}

type ComplexSetImpl<O> = IndexSet<Unspecified, O, Unsigned>;
type ComplexSet<B, O> = IndexSet<B, O, Unsigned>;
pub struct NKComplex<B: Specified, O: Order> {
  /// Graded lexciographically ordered combinations of base.
  graded_sets: Vec<Vec<ComplexSetImpl<O>>>,
  _base: PhantomData<B>,
}

impl NKComplex<Local, Sorted> {
  /// Combinations of canonical base {0,...,n-1}
  pub fn canonical(n: usize) -> Self {
    let graded_sets = (0..=n)
      .map(|k| IndexSet::counting(n).subs(k).collect())
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
  use crate::combinatorics::NKComplex;

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
