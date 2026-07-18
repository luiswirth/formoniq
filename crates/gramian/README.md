# gramian

Inner products expressed in a basis: Gram matrices and Riemannian metrics.

## What it provides

- `Gramian`: a symmetric positive-definite matrix read as the inner-product
  tensor on an abstract basis (positive-definiteness asserted by Cholesky at
  construction). It supports inner products, norms and angles of vectors and of
  whole column families, the determinant and volume factor, inversion, and the
  pullback JᵀGJ along a linear map J (the inner product a domain inherits by
  mapping its vectors through J and measuring with G).
- `RiemannianMetric`: a Gramian paired with its inverse, so both g and g⁻¹ are
  available directly. Tangent vectors are measured with the vector Gramian,
  covectors with the covector Gramian, and the pullback recomputes the inverse
  rather than pushing it forward.
- In distance-geometry terms: the Gramian of the spanning vectors of a simplex
  is its edge metric, and interior vertex angles follow from the law of cosines,
  without coordinates.

## Place in the ecosystem

`gramian` is the metric layer of
[formoniq](https://github.com/luiswirth/formoniq), a finite element exterior
calculus (FEEC) engine: `exterior` induces from it the inner products on
exterior powers, and `simplicial` uses it as the per-cell metric of Regge
geometry. The crate itself is plain linear algebra over one basis at a time,
with no notion of meshes or exterior structure.

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
