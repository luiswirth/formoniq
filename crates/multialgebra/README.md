# multialgebra

Exterior and symmetric powers of a vector space, and tensor products of them.

Λ^k and Sym^k are the two quotients of the tensor algebra by the two ways
two adjacent factors can be made to commute:
antisymmetrically, giving the exterior power, whose basis is the k-element
subsets and whose functor takes determinants;
or symmetrically, giving the symmetric power, whose basis is the degree-k
monomials and whose functor takes permanents.
They are siblings, not one built on the other.

This crate treats them as such. A tensor is a list of factors, each a
Λ or a Sym of some degree, and almost every operation is the same on
both with a single sign differing between them. The exception is the
Hodge star, which needs a top degree, and so exists on an exterior factor
and nowhere else.

## What it provides

- `Parity`:
  whether a factor is alternating or symmetric.
  The one bit separating the two constructions.
- `Factor`:
  a single Λ^k or Sym^k, carrying its parity and degree.
- `symmetric_power`, and the exterior power beside it:
  the functor on a linear map. Sym^d(AB) = Sym^d(A) Sym^d(B) is the
  symmetric counterpart of Cauchy-Binet.
- the induced map on a tensor product:
  the Kronecker product of the per-factor induced maps.

The combinatorics of the bases, colexicographic throughout, lives in
`multiindex`: subsets for the alternating factors, weak compositions for
the symmetric ones.
