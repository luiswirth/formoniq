# gramian

Inner products expressed in a basis: Gram matrices and pseudo-Riemannian
metrics of arbitrary signature.

## What it provides

- `Gramian`: a non-degenerate symmetric matrix read as the inner-product
  tensor on an abstract basis, of arbitrary signature (p, q) — Riemannian
  (positive definite, q = 0) and Lorentzian inner products are one
  signature-parameterized type, not two code paths. It supports inner
  products, signed squared norms, magnitudes and (for definite forms) angles
  of vectors and of whole column families, the determinant and the volume
  factor √|det g|, the signature, inversion, and the pullback JᵀGJ along a
  linear map J (the inner product a domain inherits by mapping its vectors
  through J and measuring with G). Flat models are built in: Euclidean,
  pseudo-Euclidean of any signature, and Minkowski (mostly-plus, time along
  the first basis vector).
- `Metric`: a pseudo-Riemannian metric tensor of any signature, a Gramian
  paired with its inverse, so both g and
  g⁻¹ are available directly. Tangent vectors are measured with the vector
  Gramian, covectors with the covector Gramian, and the pullback recomputes
  the inverse rather than pushing it forward.
- `CausalType`: the timelike/null/spacelike trichotomy of a vector under an
  indefinite metric, classified from the sign of g(v, v) — the signed squared
  norm is the primitive, and a magnitude alone never carries the causal
  character (and never yields NaN).
- In distance-geometry terms: the Gramian of the spanning vectors of a simplex
  is its edge metric, and interior vertex angles follow from the law of cosines,
  without coordinates.

## Place in the ecosystem

`gramian` is the metric layer of
[formoniq](https://github.com/luiswirth/formoniq), a finite element exterior
calculus (FEEC) engine: `exterior` induces from it the inner products on
exterior powers (the signature is what turns the Hodge star Lorentzian), and
`simplicial` uses it as the per-cell metric of Regge geometry. The crate
itself is plain linear algebra over one basis at a time, with no notion of
meshes or exterior structure.

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
