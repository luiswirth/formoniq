# glatt

The smooth continuum manifold:
parametrizations, and analytic differential-form data on it.
(*Glatt* is German for smooth.)

## What it provides

- `Parametrization`:
  a smooth map φ from a coordinate domain into ambient space,
  together with the quantities it induces.
  The Jacobian is exact if supplied and finite-difference otherwise.
  The chart χ = φ⁻¹ ∘ r is exact if supplied
  and otherwise derived by Gauss-Newton as the orthogonal nearest-point projection onto the image,
  which makes the domain gap of an interpolating mesh O(h²) rather than O(h).
  The chart differential is the pseudo-inverse of the forward Jacobian,
  and the induced metric g = (dφ)ᵀ dφ is the pullback of the ambient Euclidean metric,
  computed from the Jacobian alone.
  Built-in parametrizations: sphere, ball, torus, and graph of a function.
- `CoordField`:
  analytic fields of exterior elements over a coordinate domain
  (exact solutions, sources, boundary data), with variance as a type parameter:
  covariant is a differential form, contravariant a multivector field.
  Closure-backed constructors cover the manufactured-solution workflow
  (scalar, one-form, vector field, coordinate components, radial fields).

A parametrization and a chart are inverse to each other:
the parametrization maps coordinates into the manifold,
the chart maps the manifold out to coordinates.
The crate keeps the distinction in its names:
the parametrization is the forward map φ, the chart its inverse.

## Place in the ecosystem

`glatt` is the continuum side of
[formoniq](https://github.com/luiswirth/formoniq),
a finite element exterior calculus (FEEC) engine.
It is a sibling of the mesh crate, not a layer of it:
a Regge mesh has no continuum, a continuum has no mesh,
and their one relation
(pulling continuum data onto a mesh, and the approximation error that costs)
lives in the crate above both.
`glatt` itself depends only on the exterior algebra,
and stands on its own as a small library
for parametrized smooth manifolds and analytic form fields.

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
