# multialgebra

Multilinear algebra: the free tensor power, its exterior and symmetric
quotients, and tensor products of all three.

Λ^k and Sym^k are the two quotients of V^⊗k by a character of the symmetric
group. A character lands in an abelian group, so it factors through the
abelianization of S_k, which is Z/2 for k ≥ 2: the sign gives the exterior
power, whose basis is the k-element subsets and whose functor takes
determinants, and the trivial character gives the symmetric power, whose basis
is the degree-k monomials and whose functor takes permanents. Those two are
the whole list rather than a pair chosen out of many, and V^⊗k is the
unquotiented object sitting above them both.

This crate holds all three as one construction. Almost every operation is
written once over them, because the two quotients share a representation: the
shift w_i ↦ w_i + i takes a weakly increasing word to a strictly increasing
one, so in shifted form a multiset is a set and one bitset serves both. The
free power has no symmetry to exploit and so no such compression, and its
index is a word ranked in the radix. That the free family costs more is its
information content rather than a defect, and the type says which you are
paying for.

The Hodge star is the one operation that is not uniform: it needs a top degree
to complement against, which only an alternating slot has.

## What it provides

- `Symmetry` and `Factor`:
  whether a slot is free, alternating or symmetric, and the functor V^⊗k, Λ^k
  or Sym^k it is. A factor carries no dimension, being the functor rather than
  its value on a particular space, which is what lets one factor describe both
  ends of a rectangular map.
- `Variance` and `Slot`:
  whether a slot is built from V or from its dual, and the functor together
  with that side and the space it is over. A slot has an extent exactly as an
  axis of a dense array has a length, and a symmetry and a variance besides.
- `Tensor`:
  a list of slots, their strides, and the components. Components live on the
  product of the per-slot bases, first slot running fastest, which is colex on
  the per-slot ranks and makes an all-free tensor a dense array. Symmetry,
  variance and dimension are all per slot, so a rectangular map V* ⊗ W and the
  metric-free trace are both expressible.
- the algebra:
  `tensor` concatenates slots, `merge` collapses two of one symmetry into one
  of the summed degree, and `product` multiplies slot by slot with the Koszul
  sign. On a single alternating slot the last is the wedge.
- `contract` and `trace`:
  contraction into a slot, lowering its degree, and the contraction of two
  slots of dual variance against each other. Binary contraction between two
  tensors is `tensor` followed by one `trace` per pair, so multi-contraction
  is repeated tracing rather than a notion of its own.
- `transfer`:
  one degree moved between two slots. The exterior derivative and the Koszul
  operator are this operation in its two directions.
- the pairings:
  the duality pairing of dual variance, metric-free; and the wedge pairing
  Λ^k × Λ^(n-k) → ℝ, which needs only a top grade to land in and is
  nondegenerate. The Hodge star is what turns the second into the inner
  product, and that is where a metric enters.
- `exterior_power`, `symmetric_power` and `tensor_power`:
  the functor on a linear map. Sym^d(AB) = Sym^d(A) Sym^d(B) is the symmetric
  counterpart of Cauchy-Binet.
- `to_free`:
  the same tensor with every slot's symmetry forgotten. Λ and Sym are
  compressed representations of subspaces of the free power, and this is the
  map that says so; it is also the way out to code that knows only dense
  arrays.

## Coefficients

The coefficients are any commutative ring, not a field.
Every structure constant of the algebra is ±1 or a factorial,
and each of those is the image of an integer under the one ring map ℤ → R,
so nothing here divides:
the wedge, the contraction, both transfers, the pairings, the pushforward
and the functors above all run over ℤ,
where their laws are exact equalities rather than statements to a tolerance.

The exception is the operations that dualize a slot,
and it is the only one.
The stored basis is multiplicative, so the reciprocal basis element of a
symmetric slot is x^α / α!,
and over ℤ those span the divided power algebra rather than the symmetric one:
Sym^d(V)* ≅ Γ^d(V*), with equality only once the factorials are inverted.
So `from_reciprocal`, `evaluate` and the pullback ask for a `RationalAlgebra`,
a ring in which the positive integers are invertible,
and that is the mathematics rather than a limitation of the encoding.
Every α! is 1 on the alternating family,
where Λ^k(V)* ≅ Λ^k(V*) over any ring, so nothing purely exterior asks for it.

A ring is a `RationalAlgebra` by saying so.
Deriving it from the operations available would admit ℤ,
whose division operator truncates,
and that is exactly the kind of silently wrong answer the split exists to prevent.
`extend_scalars` is the map between rings, and it is natural:
every operation commutes with it, ℤ → ℚ → ℝ → ℂ being one operation read four ways.

What stays outside is Schur functors: a symmetry like the Riemann tensor's
R_ijkl = R_klij is cut out by a higher-dimensional irreducible representation
of S_k, not by a character, and its basis is no longer a word.

The combinatorics of the bases, colexicographic throughout, lives in
`multiindex`.
