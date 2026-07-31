//! The two families of multi-indices as one object.
//!
//! A [`Combination`](crate::Combination) is a strictly increasing word and a
//! [`Composition`](crate::Composition) a weakly increasing one, over the same
//! alphabet and of the same length. That is the entire difference between
//! them, and [`Repetition`] is the bit carrying it: forbidden gives the
//! subsets, allowed the multisets.
//!
//! Counting, ranking and enumeration are then one implementation with a
//! position-dependent offset, not two behind a shared signature. The offset is
//! the classical shift $w_i |-> w_i + i$ taking a weakly increasing word to a
//! strictly increasing one, and it appears here only inside the arithmetic,
//! never as a stored representation, which is the distinction the workspace
//! keeps between a theorem and a data structure.

use crate::{
  Sign, binomial,
  bits::{self, Bits},
  factorial,
};

/// The symbols of a monotone multi-index, inline up to a degree covering every
/// grade of a low-dimensional exterior algebra and a modest polynomial order.
///
/// Inline because these are built per basis element per quadrature point, where
/// a heap allocation would dominate the arithmetic. Spilling past the inline
/// capacity keeps the degree unbounded, which the symmetric side rests on.
pub type Symbols = tinyvec::TinyVec<[usize; 6]>;

/// Whether a multi-index may repeat a symbol.
///
/// Forbidden gives the strictly increasing words, the $binom(n, k)$ subsets;
/// allowed gives the weakly increasing ones, the $binom(n + k - 1, k)$
/// multisets. The two are otherwise the same object, and every operation here
/// is written once over both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Repetition {
  #[default]
  Forbidden,
  Allowed,
}

impl Repetition {
  /// The offset separating the two families: the amount by which the symbol at
  /// `position` is raised to make a weakly increasing word strictly
  /// increasing.
  ///
  /// Zero when repetition is forbidden, which is why one implementation covers
  /// both. Every other function in this module is the strictly increasing case
  /// composed with this.
  pub fn shift(self, position: usize) -> usize {
    match self {
      Repetition::Forbidden => 0,
      Repetition::Allowed => position,
    }
  }

  /// The sign a reordering of this many inversions carries: alternating when
  /// repetition is forbidden, trivial when it is allowed.
  ///
  /// The algebraic half of the distinction, where [`Self::shift`] is the
  /// counting half. Every signed operation reduces to it: a merge counts the
  /// inversions of the interleaving, a deletion those of moving one symbol to
  /// the front, a complement those against the rest of the alphabet.
  pub fn sign_of(self, inversions: usize) -> Sign {
    match self {
      Repetition::Forbidden => Sign::from_parity(inversions),
      Repetition::Allowed => Sign::Pos,
    }
  }

  /// The alphabet the shifted word ranges over: `nsymbols` unchanged when
  /// repetition is forbidden, widened by the shift when it is allowed.
  ///
  /// The counting of both families reduces to it. A monotone word of `degree`
  /// symbols is a strictly increasing one over this alphabet, so there are
  /// $binom("shifted", "degree")$ of them either way, which is $binom(n, k)$
  /// alternating and $binom(n + k - 1, k)$ symmetric.
  pub fn shifted_nsymbols(self, nsymbols: usize, degree: usize) -> usize {
    nsymbols + self.shift(degree.saturating_sub(1))
  }

  /// The number of monotone words of length `degree` over `nsymbols` symbols:
  /// $binom(n, k)$ forbidden and $binom(n + k - 1, k)$ allowed.
  ///
  /// One binomial, the family entering only through [`Self::shift`]. Total at
  /// the degenerate corners: the empty word is the unique word of degree zero
  /// over any alphabet, and no alphabet admits no others.
  pub fn count(self, nsymbols: usize, degree: usize) -> usize {
    binomial(self.shifted_nsymbols(nsymbols, degree), degree)
  }

  /// The colexicographic rank of a monotone word, its canonical index.
  ///
  /// The combinatorial number system $sum_i binom(w_i + "shift"(i), i + 1)$,
  /// applied to the shifted word. Independent of the alphabet size, so
  /// widening the alphabet renumbers nothing already there, the property
  /// that makes colex the workspace's convention.
  ///
  /// # Panics
  /// If the word is not monotone for this family.
  pub fn rank(self, word: &[usize]) -> usize {
    assert!(self.is_monotone(word), "word is not monotone for {self:?}");
    word
      .iter()
      .enumerate()
      .map(|(position, &symbol)| binomial(symbol + self.shift(position), position + 1))
      .sum()
  }

  /// The monotone word at the given colexicographic rank. Inverse to
  /// [`Self::rank`].
  ///
  /// The combinatorial number system read backwards: the highest position
  /// takes the largest symbol its binomial still fits under, and the remainder
  /// passes down. Greedy and exact, so it costs the degree rather than the
  /// rank.
  ///
  /// # Panics
  /// If the rank is not below [`Self::count`].
  pub fn word_from_rank(self, nsymbols: usize, degree: usize, rank: usize) -> Symbols {
    assert!(
      rank < self.count(nsymbols, degree),
      "rank out of range for {self:?}"
    );
    let mut remaining = rank;
    let mut shifted = Symbols::from_iter(std::iter::repeat_n(0, degree));
    for position in (0..degree).rev() {
      let mut symbol = position;
      while binomial(symbol + 1, position + 1) <= remaining {
        symbol += 1;
      }
      shifted[position] = symbol;
      remaining -= binomial(symbol, position + 1);
    }
    shifted
      .into_iter()
      .enumerate()
      .map(|(position, symbol)| symbol - self.shift(position))
      .collect()
  }

  /// Whether the word is monotone for this family: strictly increasing when
  /// repetition is forbidden, weakly increasing when it is allowed.
  pub fn is_monotone(self, word: &[usize]) -> bool {
    word.windows(2).all(|pair| match self {
      Repetition::Forbidden => pair[0] < pair[1],
      Repetition::Allowed => pair[0] <= pair[1],
    })
  }

  /// Every monotone word of length `degree` over `nsymbols` symbols, in
  /// colexicographic order, so the position in this iterator is
  /// [`Self::rank`].
  ///
  /// The successor runs on the shifted word, where both families are
  /// strictly increasing and the step is the same. The shift is undone on the
  /// way out. Empty when the alphabet cannot supply a word of that length,
  /// which for a forbidden repetition is any degree above `nsymbols` and for
  /// an allowed one only an empty alphabet.
  pub fn words(self, nsymbols: usize, degree: usize) -> impl Iterator<Item = Symbols> {
    let shifted_nsymbols = self.shifted_nsymbols(nsymbols, degree);
    let mut shifted: Symbols = (0..degree).collect();
    let mut exhausted = degree > shifted_nsymbols;

    std::iter::from_fn(move || {
      if exhausted {
        return None;
      }
      let current: Symbols = shifted
        .iter()
        .enumerate()
        .map(|(position, &symbol)| symbol - self.shift(position))
        .collect();
      // In place: enumeration is the innermost loop of the algebra above.
      exhausted = !advance_colex(&mut shifted, shifted_nsymbols);
      Some(current)
    })
  }
}

/// A monotone multi-index: a basis element of $Lambda^k$ or of $"Sym"^k$,
/// according to its [`Repetition`].
///
/// One type for both families, where [`Combination`](crate::Combination) and
/// [`Composition`](crate::Composition) are two, and one representation: the
/// bitset of the index's shifted word.
///
/// The shift $w_i |-> w_i + i$ takes a weakly increasing word to a strictly
/// increasing one, so in shifted form a multiset is a set, and a set of bounded
/// symbols is a bitset. Ranking, enumeration, deletion and the complement are
/// then the same bit operations for both families, which are consulted only for
/// the sign ([`Repetition::sign_of`]) and for unshifted symbols.
///
/// The alphabet is absent, as from [`Combination`](crate::Combination): a colex
/// rank does not depend on it. Only enumeration and the complement take one.
///
/// The width of the bitset is the type parameter, and it is a fact about the
/// machine alone: it bounds the *shifted* alphabet a value can hold and enters
/// no formula here. [`MonoIndex`] is this at the width the workspace reads,
/// which is what every constructor without a bitset argument needs, a default
/// type parameter applying in type position but not in inference.
///
/// `Default` is the empty alternating index, the unit of the merge monoid and
/// the sole basis element of the scalars.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct MonoIndexOver<B: Bits = DefaultBits> {
  repetition: Repetition,
  /// The set of shifted symbols, one bit each.
  shifted: B,
}

/// The backing width the workspace reads a multi-index at.
///
/// The bound is on the shifted alphabet $n + k - 1$, so the alternating side is
/// bounded by the dimension of the space and the symmetric side by the degree.
/// [`Composition`](crate::Composition) carries the unbounded-degree case outside
/// the bitset, so a wider default buys headroom nothing uses, and a narrower
/// backing measurably cheapens the index layer, the value being passed by
/// register rather than copied through the stack.
pub type DefaultBits = u64;

/// A monotone multi-index at the default width.
pub type MonoIndex = MonoIndexOver<DefaultBits>;

impl<B: Bits> MonoIndexOver<B> {
  /// From an ascending word.
  ///
  /// # Panics
  /// If the word is not monotone for this family, or reaches past the bitset
  /// once shifted.
  pub fn new(repetition: Repetition, word: impl IntoIterator<Item = usize>) -> Self {
    let mut shifted = B::ZERO;
    let mut previous: Option<usize> = None;
    for (position, symbol) in word.into_iter().enumerate() {
      assert!(
        previous.is_none_or(|last| match repetition {
          Repetition::Forbidden => last < symbol,
          Repetition::Allowed => last <= symbol,
        }),
        "word is not monotone for {repetition:?}"
      );
      previous = Some(symbol);
      shifted = shifted | B::singleton(symbol + repetition.shift(position));
    }
    Self {
      repetition,
      shifted,
    }
  }

  /// From the bitset of an already shifted word.
  pub(crate) fn from_shifted(repetition: Repetition, shifted: B) -> Self {
    Self {
      repetition,
      shifted,
    }
  }

  /// The empty index: the basis of the scalars $Lambda^0 = "Sym"^0 = RR$, for
  /// either family.
  pub fn empty(repetition: Repetition) -> Self {
    Self::from_shifted(repetition, B::ZERO)
  }

  /// A single symbol: a basis element of $Lambda^1 = "Sym"^1 = V$, where the
  /// two families coincide.
  pub fn single(repetition: Repetition, symbol: usize) -> Self {
    Self::from_shifted(repetition, B::singleton(symbol))
  }

  /// Canonicalize an arbitrarily ordered word into the sign of the sorting
  /// permutation and the monotone index underneath.
  ///
  /// `None` exactly when the family forbids the repetition the word contains:
  /// $v wedge v = 0$ on an alternating factor, an ordinary basis element on a
  /// symmetric one.
  pub fn from_word(
    repetition: Repetition,
    word: impl IntoIterator<Item = usize>,
  ) -> Option<(Sign, Self)> {
    let mut sorted = Symbols::new();
    let mut inversions = 0;
    for symbol in word {
      let position = sorted.iter().take_while(|&&s| s <= symbol).count();
      inversions += sorted.len() - position;
      if repetition == Repetition::Forbidden && sorted.contains(&symbol) {
        return None;
      }
      sorted.insert(position, symbol);
    }
    Some((
      repetition.sign_of(inversions),
      Self::new(repetition, sorted),
    ))
  }

  pub fn repetition(&self) -> Repetition {
    self.repetition
  }
  /// The degree $k$ of the $Lambda^k$ or $"Sym"^k$ this indexes: the length of
  /// the word, hence the number of set bits.
  pub fn degree(&self) -> usize {
    self.shifted.count_ones()
  }
  /// The raw shifted bitset.
  pub fn shifted_bits(&self) -> B {
    self.shifted
  }
  /// The shifted symbols, ascending: the set the representation is.
  pub fn shifted_iter(&self) -> impl Iterator<Item = usize> {
    bits::set_bits(self.shifted)
  }
  /// The symbols of the word, ascending, with the shift undone.
  pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
    self
      .shifted_iter()
      .enumerate()
      .map(move |(position, symbol)| symbol - self.repetition.shift(position))
  }
  /// The symbol at a position of the word.
  pub fn symbol(&self, position: usize) -> usize {
    self.iter().nth(position).expect("position out of range")
  }
  /// The word, ascending.
  pub fn word(&self) -> Symbols {
    self.iter().collect()
  }
  /// The multiplicity of a symbol: always $0$ or $1$ when repetition is
  /// forbidden.
  pub fn multiplicity(&self, symbol: usize) -> usize {
    self.iter().filter(|&s| s == symbol).count()
  }

  /// The size of the stabilizer of this index under the $S_k$ action on its
  /// positions, $alpha! = product_s m_s !$ over the multiplicities.
  ///
  /// The number of orderings of the word that give the word back, hence the
  /// multiplicity an unnormalized symmetrization carries. Always $1$ where
  /// repetition is forbidden, every multiplicity being at most one there.
  pub fn stabilizer(&self) -> usize {
    self
      .word()
      .chunk_by(|left, right| left == right)
      .map(|run| factorial(run.len()))
      .product()
  }

  /// The colexicographic rank, the canonical index into the basis.
  ///
  /// The combinatorial number system $sum_i binom(s_i, i+1)$ on the shifted
  /// symbols, one formula for both families because the shift is already in the
  /// representation. Independent of the alphabet, so widening it renumbers
  /// nothing.
  pub fn rank(&self) -> usize {
    self
      .shifted_iter()
      .enumerate()
      .map(|(position, symbol)| binomial(symbol, position + 1))
      .sum()
  }

  /// Inverse to [`Self::rank`]: the combinatorial number system read backwards,
  /// greedy from the top position down, so it costs the degree and not the
  /// rank.
  pub fn from_rank(repetition: Repetition, degree: usize, mut rank: usize) -> Self {
    let mut shifted = B::ZERO;
    for position in (1..=degree).rev() {
      let mut symbol = position - 1;
      while binomial(symbol + 1, position) <= rank {
        symbol += 1;
      }
      rank -= binomial(symbol, position);
      shifted = shifted | B::singleton(symbol);
    }
    Self::from_shifted(repetition, shifted)
  }

  /// Every index of this family and degree over `nsymbols` symbols, in colex.
  ///
  /// One enumeration for both: the shifted words are strictly increasing either
  /// way, so this walks the same bitset successor over an alphabet the shift
  /// widens.
  pub fn all(repetition: Repetition, nsymbols: usize, degree: usize) -> MonoIndicesOver<B> {
    MonoIndicesOver {
      repetition,
      next: (degree <= B::WIDTH).then(|| B::low_mask(degree)),
      remaining: repetition.count(nsymbols, degree),
    }
  }

  /// The next index of the same family and degree in colex order, `None` at the
  /// last one over the unbounded alphabet.
  ///
  /// Unbounded because a colex rank is: the successor of the last index inside
  /// ${0, dots, n-1}$ is the first index reaching past it, not the end of the
  /// enumeration. [`Self::all`] is this cut off at [`Repetition::count`].
  pub fn colex_successor(&self) -> Option<Self> {
    colex_successor_bits(self.shifted).map(|bits| Self::from_shifted(self.repetition, bits))
  }

  /// Merge two indices of the same family into one of the summed degree, with
  /// the sign of the interleaving: the wedge on an alternating factor, the
  /// monomial product on a symmetric one.
  ///
  /// `None` only where the merge is the zero of the algebra, so a symmetric
  /// merge is total: $x^alpha x^beta = x^(alpha + beta)$ never vanishes, where
  /// $e_I wedge e_J = 0$ whenever the two share an index.
  ///
  /// The alternating case is pure bit arithmetic, a disjointness test, an or
  /// and a popcount per symbol for the sign, the shift being zero there. The
  /// symmetric case walks the two words, the merged shift depending on
  /// positions in the result.
  ///
  /// # Panics
  /// If the two families differ.
  pub fn merge(&self, other: &Self) -> Option<(Sign, Self)> {
    assert_eq!(
      self.repetition, other.repetition,
      "a merge combines two factors of one parity"
    );
    match self.repetition {
      Repetition::Forbidden => {
        if !(self.shifted & other.shifted).is_empty() {
          return None;
        }
        let inversions: usize = other
          .shifted_iter()
          .map(|symbol| (self.shifted.shr_total(symbol).shr_total(1)).count_ones())
          .sum();
        Some((
          Sign::from_parity(inversions),
          Self::from_shifted(Repetition::Forbidden, self.shifted | other.shifted),
        ))
      }
      Repetition::Allowed => {
        let mut merged = B::ZERO;
        let (mut left, mut right) = (self.iter().peekable(), other.iter().peekable());
        for position in 0..self.degree() + other.degree() {
          let symbol = match (left.peek(), right.peek()) {
            (Some(&l), Some(&r)) if l <= r => {
              left.next();
              l
            }
            (_, Some(&r)) => {
              right.next();
              r
            }
            (Some(&l), None) => {
              left.next();
              l
            }
            (None, None) => unreachable!("the degrees count the symbols"),
          };
          merged = merged | B::singleton(symbol + position);
        }
        Some((Sign::Pos, Self::from_shifted(Repetition::Allowed, merged)))
      }
    }
  }

  /// Every single-symbol deletion, by position: the interior product on an
  /// alternating factor, the directional derivative on a symmetric one.
  ///
  /// Iterating positions rather than distinct symbols makes the two uniform. On
  /// an alternating factor the positions are the symbols and the sign
  /// alternates. On a symmetric one a symbol of multiplicity $m$ occupies $m$
  /// positions and yields the same reduced word $m$ times, which is the factor
  /// $alpha_i$ in $diff_i x^alpha = alpha_i x^(alpha - e_i)$.
  ///
  /// In shifted form this is one bit operation: drop the bit and slide
  /// everything above it down by the shift the removed position carried, one
  /// for a symmetric factor and zero for an alternating one.
  ///
  /// Total at the trivial end: the empty index has no deletions.
  pub fn deletions(&self) -> MonoDeletionsOver<B> {
    MonoDeletionsOver {
      index: *self,
      remaining: self.shifted,
      position: 0,
    }
  }

  /// The complement within ${0, dots, n-1}$ and the sign of
  /// $e_S wedge e_(S^c) = "sign" dot e_({0, dots, n-1})$: the combinatorics of
  /// the Hodge star.
  ///
  /// Alternating only: a symmetric factor has no top degree, hence no
  /// complement to take and no volume element to normalize against.
  ///
  /// # Panics
  /// If repetition is allowed.
  pub fn complement_signed(&self, nsymbols: usize) -> (Sign, Self) {
    assert_eq!(
      self.repetition,
      Repetition::Forbidden,
      "a symmetric factor has no top degree, so no complement"
    );
    let complement =
      Self::from_shifted(Repetition::Forbidden, !self.shifted & B::low_mask(nsymbols));
    let (sign, _) = self.merge(&complement).expect("the complement is disjoint");
    (sign, complement)
  }
}

impl<B: Bits> MonoIndexOver<B> {
  /// The same index read as a [`Combination`](crate::Combination), which it
  /// already is when repetition is forbidden: the shift is zero there, so the
  /// shifted word stored here is the set.
  ///
  /// # Panics
  /// If repetition is allowed, a multiset being no set of symbols.
  pub fn to_combination(&self) -> crate::combination::CombinationOver<B> {
    assert_eq!(
      self.repetition,
      Repetition::Forbidden,
      "a symmetric index is a multiset, which no bitset of symbols represents"
    );
    crate::combination::CombinationOver::from_bits(self.shifted)
  }
}

impl<B: Bits> std::fmt::Debug for MonoIndexOver<B> {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    match self.repetition {
      Repetition::Forbidden => f.debug_set().entries(self.iter()).finish(),
      Repetition::Allowed => f.debug_list().entries(self.iter()).finish(),
    }
  }
}

/// The colexicographic successor of a strictly increasing word held as a
/// bitset, or `None` at the last (Gosper's hack).
fn colex_successor_bits<B: Bits>(bits: B) -> Option<B> {
  if bits.is_empty() {
    return None;
  }
  let lowest = bits.trailing_zeros();
  let carried = bits.checked_add(bits.lowest())?;
  if carried.is_empty() {
    return None;
  }
  Some(carried | (bits ^ carried).shr_total(lowest + 2))
}

/// The colexicographic successor of a strictly increasing word over `nsymbols`
/// symbols, or `None` at the last.
///
/// Raise the lowest symbol that has room below its successor and reset
/// everything under it to the smallest word: colex orders by the largest
/// symbol first, so the low end is what turns over.
fn advance_colex(word: &mut [usize], nsymbols: usize) -> bool {
  let ceiling = |position: usize| {
    word
      .get(position + 1)
      .copied()
      .unwrap_or(nsymbols)
      .saturating_sub(1)
  };
  let Some(position) = (0..word.len()).find(|&position| word[position] < ceiling(position)) else {
    return false;
  };

  word[position] += 1;
  word[..position]
    .iter_mut()
    .enumerate()
    .for_each(|(index, symbol)| *symbol = index);
  true
}

/// Every [`MonoIndex`] of a family, degree and alphabet, in colex.
///
/// A named type rather than an opaque one so a caller dispatching between the
/// index families can hold either without boxing.
#[derive(Debug, Clone)]
pub struct MonoIndicesOver<B: Bits = DefaultBits> {
  repetition: Repetition,
  next: Option<B>,
  remaining: usize,
}

/// Every [`MonoIndex`] of a family, degree and alphabet, in colex.
pub type MonoIndices = MonoIndicesOver<DefaultBits>;

impl<B: Bits> Iterator for MonoIndicesOver<B> {
  type Item = MonoIndexOver<B>;
  fn next(&mut self) -> Option<Self::Item> {
    if self.remaining == 0 {
      return None;
    }
    let current = self.next?;
    self.next = colex_successor_bits(current);
    self.remaining -= 1;
    Some(MonoIndexOver::from_shifted(self.repetition, current))
  }
  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.remaining, Some(self.remaining))
  }
}
impl<B: Bits> ExactSizeIterator for MonoIndicesOver<B> {}

/// Every single-symbol deletion of a [`MonoIndex`], by position.
///
/// Owns the index, which is [`Copy`], so it carries no lifetime.
#[derive(Debug, Clone)]
pub struct MonoDeletionsOver<B: Bits = DefaultBits> {
  index: MonoIndexOver<B>,
  /// The shifted bits not yet visited. The lowest is the next position.
  remaining: B,
  position: usize,
}

/// Every single-symbol deletion of a [`MonoIndex`], by position.
pub type MonoDeletions = MonoDeletionsOver<DefaultBits>;

impl<B: Bits> Iterator for MonoDeletionsOver<B> {
  type Item = (Sign, usize, MonoIndexOver<B>);
  fn next(&mut self) -> Option<Self::Item> {
    if self.remaining.is_empty() {
      return None;
    }
    let bit = self.remaining.trailing_zeros();
    self.remaining = self.remaining.without_lowest();
    let position = self.position;
    self.position += 1;

    let repetition = self.index.repetition;
    let slide = repetition.shift(1);
    let below = self.index.shifted & B::low_mask(bit);
    let above = self.index.shifted.shr_total(bit).shr_total(1);
    Some((
      repetition.sign_of(position),
      bit - repetition.shift(position),
      MonoIndexOver::from_shifted(repetition, below | above.shl_total(bit + 1 - slide)),
    ))
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{Combination, Composition, Sign, combinations};

  /// Every law about a [`MonoIndexOver`] is a fact about the multi-index and
  /// not about the width of the bitset holding it, so each is checked at
  /// several backings. The narrow ones are what exercise the boundary a wide
  /// one never reaches.
  macro_rules! at_every_width {
    ($check:ident) => {{
      $check::<u16>();
      $check::<u32>();
      $check::<u64>();
      $check::<u128>();
    }};
  }

  /// The word of a composition: its symbol repeated with the multiplicity of
  /// each part, ascending.
  fn composition_word(composition: &Composition) -> Symbols {
    composition
      .parts()
      .iter()
      .enumerate()
      .flat_map(|(symbol, &multiplicity)| std::iter::repeat_n(symbol, multiplicity))
      .collect()
  }

  /// Enumeration and ranking are inverse, for both families: the position in
  /// [`Repetition::words`] is the word's [`Repetition::rank`], and the count
  /// is the number enumerated.
  #[test]
  fn rank_is_the_position_in_the_enumeration() {
    for repetition in [Repetition::Forbidden, Repetition::Allowed] {
      for nsymbols in 0..=5 {
        for degree in 0..=4 {
          let words: Vec<_> = repetition.words(nsymbols, degree).collect();
          assert_eq!(words.len(), repetition.count(nsymbols, degree));
          for (position, word) in words.iter().enumerate() {
            assert!(repetition.is_monotone(word));
            assert!(word.iter().all(|&symbol| symbol < nsymbols));
            assert_eq!(repetition.rank(word), position);
          }
        }
      }
    }
  }

  /// The word-level forbidden family and [`Combination`] agree exactly: same
  /// words, same order, same ranks.
  ///
  /// The two are different representations, an unbounded word and a bitset, so
  /// this is a theorem rather than a tautology. Order as well as content, since
  /// a rank is only meaningful against an enumeration.
  #[test]
  fn forbidden_repetition_is_the_combination() {
    for nsymbols in 0..=5 {
      for degree in 0..=4 {
        let unified: Vec<_> = Repetition::Forbidden.words(nsymbols, degree).collect();
        let existing: Vec<Symbols> = combinations(nsymbols, degree)
          .map(|combination| combination.iter().collect())
          .collect();
        assert_eq!(unified, existing, "n={nsymbols} k={degree}");

        for word in &unified {
          let combination = Combination::from_increasing(word.iter().copied());
          assert_eq!(Repetition::Forbidden.rank(word), combination.rank());
        }
      }
    }
  }

  /// A rank is independent of the alphabet size: widening the alphabet leaves
  /// every existing word where it was.
  ///
  /// This is what colex is for, and the formula shows it, the sum runs
  /// over the word and never mentions `nsymbols`. It holds for both families
  /// here, which is the substantive claim: the symmetric side is not a second
  /// convention that happens to agree: it is the same one.
  #[test]
  fn rank_does_not_depend_on_the_alphabet() {
    for repetition in [Repetition::Forbidden, Repetition::Allowed] {
      for degree in 0..=3 {
        for nsymbols in 0..=4 {
          for word in repetition.words(nsymbols, degree) {
            // The same word, found in a wider alphabet, keeps its position.
            for wider in nsymbols..=6 {
              let position = repetition
                .words(wider, degree)
                .position(|other| other == word);
              assert_eq!(position, Some(repetition.rank(&word)));
            }
          }
        }
      }
    }
  }

  /// The allowed family is [`Composition`]: same words, same order, same
  /// ranks.
  ///
  /// The counterpart of [`forbidden_repetition_is_the_combination`], and
  /// together they are the claim the module exists to make, both families
  /// enumerated and ranked by one implementation, differing only in the shift.
  ///
  /// [`forbidden_repetition_is_the_combination`]: self::forbidden_repetition_is_the_combination
  #[test]
  fn allowed_repetition_is_the_composition() {
    for nsymbols in 1..=5 {
      for degree in 0..=4 {
        let unified: Vec<_> = Repetition::Allowed.words(nsymbols, degree).collect();
        let existing: Vec<Symbols> = Composition::all(nsymbols, degree)
          .map(|composition| composition_word(&composition))
          .collect();
        assert_eq!(unified, existing, "n={nsymbols} k={degree}");

        for word in &unified {
          let composition = Composition::from_word(nsymbols, word);
          assert_eq!(Repetition::Allowed.rank(word), composition.rank());
        }
      }
    }
  }

  /// Ranking inverts the enumeration for both families, without walking it.
  #[test]
  fn word_from_rank_inverts_rank() {
    for repetition in [Repetition::Forbidden, Repetition::Allowed] {
      for nsymbols in 0..=5 {
        for degree in 0..=4 {
          for (position, word) in repetition.words(nsymbols, degree).enumerate() {
            assert_eq!(repetition.word_from_rank(nsymbols, degree, position), word);
          }
        }
      }
    }
  }

  /// [`MonoIndex`] reproduces [`Composition`] on the allowed side, and the
  /// merge is the monomial product $x^alpha x^beta = x^(alpha + beta)$: total,
  /// unsigned, and of the summed degree.
  #[test]
  fn the_allowed_index_is_the_composition() {
    at_every_width!(check);
    fn check<B: Bits>() {
      for nsymbols in 1..=4 {
        for degree in 0..=4 {
          for index in MonoIndexOver::<B>::all(Repetition::Allowed, nsymbols, degree) {
            let composition = Composition::from_word(nsymbols, &index.word());
            assert_eq!(index.rank(), composition.rank());

            for other in MonoIndexOver::<B>::all(Repetition::Allowed, nsymbols, 2) {
              let (sign, merged) = index
                .merge(&other)
                .expect("a monomial product never vanishes");
              assert_eq!(sign, Sign::Pos);
              assert_eq!(merged.degree(), index.degree() + other.degree());
              let expected = &composition + &Composition::from_word(nsymbols, &other.word());
              assert_eq!(Composition::from_word(nsymbols, &merged.word()), expected);
            }
          }
        }
      }
    }
  }

  /// The merge is graded-commutative, one law over both families:
  /// $b a = (-1)^(deg a deg b) a b$, which is antisymmetry of the wedge when
  /// repetition is forbidden and plain commutativity of monomials when it is
  /// allowed. The exponent is the same; only [`Repetition::sign_of`] differs.
  #[test]
  fn the_merge_is_graded_commutative() {
    at_every_width!(check);
    fn check<B: Bits>() {
      for repetition in [Repetition::Forbidden, Repetition::Allowed] {
        for degree_a in 0..=2 {
          for degree_b in 0..=2 {
            for a in MonoIndexOver::<B>::all(repetition, 4, degree_a) {
              for b in MonoIndexOver::<B>::all(repetition, 4, degree_b) {
                let sign = repetition.sign_of(degree_a * degree_b);
                match (a.merge(&b), b.merge(&a)) {
                  (None, None) => {}
                  (Some((sign_ab, ab)), Some((sign_ba, ba))) => {
                    assert_eq!(ab, ba);
                    assert_eq!(sign_ab, sign * sign_ba);
                  }
                  _ => panic!("the merge vanishes in only one order"),
                }
              }
            }
          }
        }
      }
    }
  }

  /// The merge is associative and the empty index is its unit, signs included:
  /// the graded monoid both families carry.
  #[test]
  fn the_merge_is_an_associative_monoid() {
    at_every_width!(check);
    fn check<B: Bits>() {
      for repetition in [Repetition::Forbidden, Repetition::Allowed] {
        let unit = MonoIndexOver::<B>::empty(repetition);
        for a in MonoIndexOver::<B>::all(repetition, 4, 2) {
          assert_eq!(a.merge(&unit), Some((Sign::Pos, a)));
          assert_eq!(unit.merge(&a), Some((Sign::Pos, a)));
          for b in MonoIndexOver::<B>::all(repetition, 4, 1) {
            for c in MonoIndexOver::<B>::all(repetition, 4, 1) {
              let left = a
                .merge(&b)
                .and_then(|(sign, ab)| ab.merge(&c).map(|(s, abc)| (sign * s, abc)));
              let right = b
                .merge(&c)
                .and_then(|(sign, bc)| a.merge(&bc).map(|(s, abc)| (sign * s, abc)));
              assert_eq!(left, right);
            }
          }
        }
      }
    }
  }

  /// Deleting twice cancels in pairs on an alternating factor,
  /// $iota_v^2 = 0 = diff compose diff$, and emphatically does not on a
  /// symmetric one, where the second derivative is symmetric rather than
  /// vanishing.
  ///
  /// A law asserting a quantity vanishes passes on an implementation returning
  /// zero for the wrong reason, so the same code path is checked to not vanish
  /// where it must not.
  #[test]
  fn double_deletion_vanishes_only_when_alternating() {
    at_every_width!(check);
    fn check<B: Bits>() {
      use std::collections::HashMap;
      for repetition in [Repetition::Forbidden, Repetition::Allowed] {
        for index in MonoIndexOver::<B>::all(repetition, 4, 3) {
          let mut chain: HashMap<Symbols, i32> = HashMap::new();
          for (sign_outer, _, once) in index.deletions() {
            for (sign_inner, _, twice) in once.deletions() {
              *chain.entry(twice.word()).or_default() += (sign_outer * sign_inner).as_i32();
            }
          }
          let vanishes = chain.values().all(|&coefficient| coefficient == 0);
          assert_eq!(vanishes, repetition == Repetition::Forbidden);
        }
      }
    }
  }

  /// A deletion is the derivation dual to the merge, at the level of indices:
  /// deleting a symbol from a merge hits one side or the other, with the Koszul
  /// sign on the alternating factor and none on the symmetric one. The Leibniz
  /// rule, before any coefficients enter.
  #[test]
  fn deletion_is_a_graded_derivation_of_the_merge() {
    at_every_width!(check);
    fn check<B: Bits>() {
      for repetition in [Repetition::Forbidden, Repetition::Allowed] {
        // `degree_a` odd is what exercises the Koszul sign: fixing it even
        // leaves the parity trivial and the law passes under any sign.
        for degree_a in 1..=3 {
          for degree_b in 1..=2 {
            for a in MonoIndexOver::<B>::all(repetition, 4, degree_a) {
              for b in MonoIndexOver::<B>::all(repetition, 4, degree_b) {
                let Some((sign_ab, ab)) = a.merge(&b) else {
                  continue;
                };
                // Deletions of the product, as a signed multiset keyed by the
                // deleted symbol and the resulting word.
                let mut from_product: std::collections::HashMap<(usize, Symbols), i32> =
                  std::collections::HashMap::new();
                for (sign, symbol, reduced) in ab.deletions() {
                  *from_product.entry((symbol, reduced.word())).or_default() +=
                    (sign_ab * sign).as_i32();
                }

                let mut from_leibniz: std::collections::HashMap<(usize, Symbols), i32> =
                  std::collections::HashMap::new();
                for (sign, symbol, reduced) in a.deletions() {
                  if let Some((merge_sign, whole)) = reduced.merge(&b) {
                    *from_leibniz.entry((symbol, whole.word())).or_default() +=
                      (sign * merge_sign).as_i32();
                  }
                }
                let parity = repetition.sign_of(a.degree());
                for (sign, symbol, reduced) in b.deletions() {
                  if let Some((merge_sign, whole)) = a.merge(&reduced) {
                    *from_leibniz.entry((symbol, whole.word())).or_default() +=
                      (parity * sign * merge_sign).as_i32();
                  }
                }

                from_product.retain(|_, coefficient| *coefficient != 0);
                from_leibniz.retain(|_, coefficient| *coefficient != 0);
                assert!(!from_product.is_empty(), "the law would hold vacuously");
                assert_eq!(from_product, from_leibniz);
              }
            }
          }
        }
      }
    }
  }

  /// An index may fill its backing exactly, and every operation stays total
  /// at that edge: the complement is empty, the colex successor is the end of
  /// the enumeration, and the deletions are the whole alphabet.
  ///
  /// The width is a ceiling on the representation and enters no formula, so
  /// the narrow backings are where this says something.
  #[test]
  fn an_index_may_span_the_full_width() {
    fn check<B: Bits>() {
      let full = MonoIndexOver::<B>::new(Repetition::Forbidden, 0..B::WIDTH);
      assert_eq!(full.degree(), B::WIDTH);
      assert_eq!(full.rank(), 0, "the full set is the first of its degree");
      assert_eq!(full.colex_successor(), None);

      let (sign, complement) = full.complement_signed(B::WIDTH);
      assert_eq!(sign, Sign::Pos);
      assert_eq!(complement.degree(), 0);

      let deleted: Vec<_> = full.deletions().map(|(_, symbol, _)| symbol).collect();
      assert_eq!(deleted, (0..B::WIDTH).collect::<Vec<_>>());
    }
    check::<u8>();
    check::<u16>();
    check::<u32>();
    check::<u64>();
    check::<u128>();
  }

  /// Past the width there is no representation, so construction refuses rather
  /// than wrapping into a different index.
  #[test]
  #[should_panic(expected = "index reaches past the bitset")]
  fn a_symbol_past_the_width_is_refused() {
    MonoIndexOver::<u8>::single(Repetition::Forbidden, u8::WIDTH);
  }

  /// Both families agree at the degenerate degrees, where there is no
  /// repetition to permit: one empty word at degree zero, and the alphabet
  /// itself at degree one.
  #[test]
  fn the_families_coincide_below_degree_two() {
    for nsymbols in 0..=5 {
      for degree in 0..=1 {
        let forbidden: Vec<_> = Repetition::Forbidden.words(nsymbols, degree).collect();
        let allowed: Vec<_> = Repetition::Allowed.words(nsymbols, degree).collect();
        assert_eq!(forbidden, allowed);
      }
    }
  }
}
