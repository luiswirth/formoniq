# metric

Pseudo-Riemannian metrics of arbitrary signature on a tangent space,
and the operations on multilinear algebra that need one.

## What it provides

- `Metric`:
  a non-degenerate symmetric bilinear form expressed in a basis,
  of arbitrary signature (p, q),
  hence an element of Sym²(V\*) when covariant and of Sym²(V) when contravariant.
  Riemannian (positive definite, q = 0) and Lorentzian metrics
  are one signature-parameterized type, not two code paths.
  It supports inner products, signed squared norms,
  magnitudes and (for definite forms) angles of vectors and of whole column families,
  the determinant and the volume factor √|det g|, the signature,
  and the pullback JᵀgJ along a linear map J
  (the metric a domain inherits by mapping its vectors through J and measuring with g).
  Flat models are built in:
  Euclidean, pseudo-Euclidean of any signature,
  and Minkowski (mostly-plus, time along the first basis vector).
- g and g⁻¹ as one datum.
  The variance says which of the two a value is,
  `dual` is the exact passage between them,
  and `measuring` asks for the one that measures a given variance.
  Nothing chooses between g and g⁻¹ by hand,
  and there is no stored pair to fall out of step:
  each determines the other, so either may be handed to an operation.
- `CausalType`:
  the timelike/null/spacelike trichotomy of a vector under an indefinite metric,
  classified from the sign of g(v, v).
  The signed squared norm is the primitive,
  and a magnitude alone never carries the causal character (and never yields NaN).
- The metric operations on a tensor:
  the induced metric on any exterior or symmetric power
  (one method over both families, taking determinants of the minors on Λᵏ
  and permanents on Symᵈ),
  the inner product, the norm, the Hodge star and the musical isomorphisms.
  Each slot is measured by its own variance,
  so a mixed tensor raises and lowers the right indices.

## Place in the ecosystem

`metric` is the metric layer of
[formoniq](https://github.com/luiswirth/formoniq),
a finite element exterior calculus (FEEC) engine.
It sits directly above `multialgebra`, and the boundary between them is the point:
the wedge, the contraction, the exterior derivative and both pairings need no metric,
while the inner product, the Hodge star and the musicals need nothing else.
An operation's crate therefore says what it depends on.
The crate has no notion of meshes:
this is a metric on a tangent space, and a metric on a mesh is `regge`'s.

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
