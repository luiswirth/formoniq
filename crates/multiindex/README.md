# multiindex

Combinatorics of finite index sets, in colexicographic order.

The central type is `MonoIndex`:
a monotone multi-index, weakly or strictly increasing according to whether it
may repeat a symbol, stored as the bitset of its *shifted* word.
The shift w_i ↦ w_i + i takes a weakly increasing word to a strictly
increasing one, so in shifted form a multiset is a set and one bitset serves
both families: ranking, enumeration, deletion and the complement are the same
bit operations either way, and the family is consulted only for the sign.
`Combination` is the strictly increasing case on its own, labeling a basis
element wherever ordered index subsets occur (a subset of vertices, a set of
covector indices, a corner of a cube), and `Composition` the weakly increasing
one, an exponent vector.
The crate enumerates and ranks everything in one convention, colex order,
via the combinatorial number system.

## What it provides

- Ranking and enumeration:
  - `rank()`, the colex rank, independent of ambient dimension, and its inverse `from_rank`
  - enumeration of all combinations of a given cardinality in colex order
  - the filtration property that the first binomial(n, k) of them are those inside {0, …, n−1}
- Signed algebra on index sets:
  - canonicalization of an arbitrary-order index word into permutation sign plus combination
  - signed disjoint union (the combinatorial wedge)
  - signed complement (the combinatorial Hodge star)
  - alternating single-element deletions (the combinatorial boundary, squaring to zero)
- Permutation parity:
  `Sign` as the sign group, and sorting that reports the parity of the sorting permutation.
- Weak compositions (stars and bars), enumerated through the same colex bijection.
- `cartesian`: positional (radix) multi-indices for tensor-product index sets.
  A radix-2 index is a cube corner, that is, a `Combination` of axes.

The bitset representation caps indices at 64 (`MAX_NINDICES`), asserted at construction.

## Correctness

The laws are tests, swept over cardinalities:
rank and unranking invert each other,
colex order coincides with bitset order,
enumeration is filtration-compatible,
the signed union is graded-antisymmetric,
the signed complement wedges to the top set,
and the alternating deletions square to zero.

## Place in the ecosystem

`multiindex` is the indexing layer of
[formoniq](https://github.com/luiswirth/formoniq),
a finite element exterior calculus (FEEC) engine:
`multialgebra` builds its basis blades and `simplicial` its simplices on `Combination`,
and the shared colex rank aligns their orderings.
The crate depends on neither.
It is pure combinatorics, usable on its own.

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
