# simplicial

Simplicial manifolds of arbitrary dimension, with geometry a separate input from
topology. Nothing is specialized to 2D or 3D; dimension is a runtime value, and
the degenerate cases (a point, a single cell, an empty skeleton) run on the same
code paths and return the trivial answer.

## The design

A mesh is three independent layers, each below the next.

**Topology** is the combinatorial complex: incidence, orientation, navigation
(star, link, cofaces), boundary and coboundary operators, and exact integer
simplicial homology. Betti numbers, relative Betti numbers of the pair (K, ∂K),
the Euler characteristic and representative homology generators are computed by
exact rational arithmetic. The boundary of a complex is itself a first-class
complex, with the trace operator as a cochain map onto it.

**The atlas** is the piecewise-affine chart structure, still metric-free:
barycentric charts on the cells, affine transition maps between charts sharing a
face (obeying the cocycle law), an intrinsic notion of a point (a cell plus
barycentric weights), Grundmann-Möller quadrature exact to prescribed degree in
every dimension, and uniform (Freudenthal) refinement recording the affine map
of each child.

**Geometry** enters as one concrete intrinsic currency: the signed squared edge
lengths on the 1-skeleton (positive spacelike, zero null, negative timelike — the
primitive Regge invented for general relativity, and the representation that keeps
every signature expressible), from which a per-cell pseudo-Riemannian metric of
any signature is derived. That metric is defined on every simplex, not only the
cells: an edge's length, a facet's area, a hinge's metric are read off the shared
edge data with no containing cell consulted. Coordinates and raw per-cell metric
tensors are sources rather than a separate abstraction — each converts to the
edge-length primitive at the API boundary, on equal footing — so an embedding (in
a flat ambient space of any signature, Euclidean by default, Minkowski for a
spacetime mesh) induces a metric but is never a prerequisite. Volumes, mesh
widths, shape regularity and Gaussian curvature by angle defect are computed from
edge data alone, on manifolds with no global coordinates (a flat torus, an
abstract Riemannian manifold, a coordinate-free simplicial spacetime). Extrinsic
quantities (mean curvature, a BVH point locator) sit downstream of the intrinsic
layer.

Distance geometry connects the metric representations: Cayley-Menger
realizability checks, and the exact conversion between squared edge lengths and metric
tensors in both directions. Refinement transports all three geometry
representations exactly.

## Correctness

The test suite states laws and sweeps them over dimensions: ∂∘∂ = 0,
Euler-Poincaré, Poincaré duality on the sphere, Poincaré-Lefschetz for (K, ∂K),
the transition cocycle law, exactness of quadrature on polynomials, and
Gauss-Bonnet on Regge data without any embedding.

## I/O

Gmsh `.msh` import is always available; CBOR serialization of every mesh
structure is behind the `serde` feature.

## Place in the ecosystem

`simplicial` is the mesh layer of
[formoniq](https://github.com/luiswirth/formoniq), a finite element exterior
calculus (FEEC) engine. It knows nothing of differential forms: cochains,
Whitney forms and PDEs live in the crates above it. The crate stands on its own
as a simplicial-topology and Regge-geometry library.

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
