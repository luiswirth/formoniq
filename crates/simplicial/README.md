# simplicial

Simplicial complexes of arbitrary dimension, and the piecewise-affine atlas on
them. Metric-free throughout. Nothing is specialized to 2D or 3D, dimension is a
runtime value, and the degenerate cases (a point, a single cell, an empty
skeleton) run on the same code paths and return the trivial answer.

## The design

A mesh is a topology and a geometry, and the separation is mathematical rather
than a matter of taste: the boundary operator, incidence and homology are
metric-free facts, and no amount of combinatorics derives a metric. So a
geometry is a genuinely second input, and it lives in a crate of its own —
[regge](https://crates.io/crates/regge). Nothing here has a notion of length.

**Topology** is the combinatorial complex: incidence, orientation, navigation
(star, link, cofaces), boundary and coboundary operators, and exact integer
simplicial homology. Betti numbers, relative Betti numbers of the pair (K, ∂K),
the Euler characteristic and representative homology generators are computed by
exact rational arithmetic. The boundary of a complex is itself a first-class
complex, with the trace operator as a cochain map onto it.

**The atlas** is the piecewise-affine chart structure: barycentric charts on the
cells, affine transition maps between charts sharing a face (obeying the cocycle
law), an intrinsic notion of a point (a cell plus barycentric weights),
Grundmann-Möller quadrature exact to prescribed degree in every dimension, and
uniform (Freudenthal) refinement recording the affine map of each child. Affine,
not flat: flatness is about curvature and presupposes a metric, while the charts
are affine maps and need none.

**The Kuhn triangulation** of a box is here too, and that is not an oversight. It
is combinatorics: which simplices, in which vertex order, from the per-axis cell
counts alone. The vertex order is what makes uniform refinement compose, so it
belongs with the topology; placing the vertices in space is `regge`'s.

## Correctness

The test suite states laws and sweeps them over dimensions: ∂∘∂ = 0,
Euler-Poincaré, Poincaré duality on the sphere, Poincaré-Lefschetz for (K, ∂K),
the transition cocycle law, and exactness of quadrature on polynomials.

Every fixture is built combinatorially, which is the crate's own claim tested on
itself: a 2-sphere is the boundary of a tetrahedron, and the annulus is a grid
with its middle box removed *by index*. No test here needs a coordinate.

## I/O

CBOR serialization of the combinatorial structures, behind the `serde` feature.
Mesh formats are `regge`'s, since reading a mesh means reading coordinates too.

## Place in the ecosystem

`simplicial` is the topological layer of
[formoniq](https://github.com/luiswirth/formoniq), a finite element exterior
calculus (FEEC) engine. It knows nothing of differential forms or of geometry:
metrics and edge lengths live in `regge`, and cochains, Whitney forms and PDEs
in the crates above.
