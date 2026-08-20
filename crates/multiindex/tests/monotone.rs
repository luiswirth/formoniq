//! The two monotone-word families, [`Repetition::Forbidden`] (combinations)
//! and [`Repetition::Allowed`] (compositions), as one enumeration.

use multiindex::{
  Bits, Combination, Composition, MonoIndexOver, Repetition, Sign, Symbols, combinations,
};

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

/// [`MonoIndexOver`] reproduces [`Composition`] on the allowed side, and the
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
