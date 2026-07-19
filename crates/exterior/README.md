# exterior

The exterior algebra of a finite-dimensional real vector space, with
multivectors and multiforms on equal footing. Dimension and grade are runtime
values; nothing is specialized to 2D or 3D.

## What it provides

- `ExteriorElement<V: Variance>`: an element of the exterior power Λ^k, as
  coefficients on colexicographically ordered basis blades. The variance is a
  type parameter: `MultiVector` (contravariant, Λ^k V) and `MultiForm`
  (covariant, Λ^k V*), each the other's dual. It fixes the functorial direction,
  the duality pairing and the choice of g vs. g⁻¹ at the type level: multiforms
  pull back, multivectors push forward, and the wrong composition does not
  compile.
- Metric-free operations: the wedge product, the interior product (contraction,
  an antiderivation squaring to zero), the duality pairing between the two
  variances, and the exterior power Λ^k A of a linear map, the matrix of k×k
  minors.
- Metric operations, entering only through a `PseudoRiemannianMetric` of any
  signature (Riemannian and Lorentzian are one code path): induced inner
  products on Λ^k, the musical isomorphisms flat and sharp, and the Hodge star
  defined by α ∧ ⋆β = ⟨α, β⟩ vol.

## Correctness

The test suite states the laws that characterize the operations and sweeps them
over dimensions and grades: Cauchy-Binet functoriality Λ^k(AB) = (Λ^k A)(Λ^k B),
adjointness of pullback and pushforward, the signature-aware Hodge involution
⋆⋆ = (−1)^(k(n−k)) sgn(det g) (swept over every metric signature, with the
Lorentzian ⋆ on Minkowski space checked against its closed form), the
antiderivation identity for the interior product, and the musicals as mutual
inverses.

## Place in the ecosystem

`exterior` is the element-local algebra layer of
[formoniq](https://github.com/luiswirth/formoniq), a finite element exterior
calculus (FEEC) engine. It knows nothing of meshes, differential forms or PDEs:
it is plain multilinear algebra, usable on its own.

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
