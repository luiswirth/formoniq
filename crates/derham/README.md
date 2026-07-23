# derham

The discrete de Rham complex:
differential forms on a simplicial manifold, without coordinates.

## What it provides

- `Cochain`:
  a discrete k-form, one coefficient per k-simplex,
  with the discrete exterior derivative as the coboundary.
  Arbitrary dimension and grade.
- Whitney interpolation `W`:
  the lowest-order Whitney forms, the basis of P⁻₁Λᵏ dual to the degrees of freedom,
  reconstructing from a cochain a form defined at every point of the manifold.
  Combinatorial: no coordinates, no metric.
- The de Rham map `R`:
  discretization of a form by integration over simplices, defined on any geometry or none.
- `Section`:
  a field over the simplicial manifold,
  evaluated at an intrinsic point (a cell plus barycentric weights)
  and valued in that chart's frame,
  with variance (form vs. multivector field) as a type parameter.
  Sections carry a lazy pointwise algebra: wedge, the musical isomorphisms, and the Hodge star.
- The bridge to the continuum:
  analytic data on a smooth manifold (from the `glatt` crate)
  is pulled back onto the mesh through a cell's parametrization and the continuum's chart,
  with a flat domain as the identity case.
  The pullback is implemented only for the covariant variance,
  so a multivector field cannot be pulled back.
  Ambient sampling of sections is provided for I/O and visualization
  and is not part of the core path.

## Correctness

The characterizing theorems are the test suite, swept over dimensions and grades:
Whitney's theorem R∘W = id (the DOF duality, checked entry by entry with signs),
the commuting properties R∘d = d∘R (Stokes) and d∘W = W∘d (the Whitney space is a subcomplex),
well-definedness of the de Rham map across the cells sharing a face,
and functoriality of the composite pullback on a curved manifold.

## Place in the ecosystem

`derham` is the discrete-forms layer of
[formoniq](https://github.com/luiswirth/formoniq),
a finite element exterior calculus (FEEC) engine:
the crate where the exterior algebra (`exterior`), the simplicial manifold (`simplicial`)
and the continuum (`glatt`) meet.
It assembles no global operators and solves nothing.
Those live one crate up.

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
