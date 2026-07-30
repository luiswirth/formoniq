# simplicial

The simplicial manifold of arbitrary dimension: its topology, and the
piecewise-affine structure it carries. Metric-free throughout. Nothing is
specialized to 2D or 3D, dimension is a runtime value, and the degenerate cases
(a point, a single cell, an empty skeleton) run on the same code paths and
return the trivial answer.

The object is the manifold, and the complex is what it is built out of. Not
every complex is a manifold, so the crate does not pretend otherwise: the
manifold condition is a property to be *checked*, as far as it can be, and how
far that is is said below.

## The design

A mesh is a topology and a geometry, and the separation is mathematical rather
than a matter of taste: the boundary operator, incidence and homology are
metric-free facts, and no amount of combinatorics derives a metric. So a
geometry is a genuinely second input, and it lives in a crate of its own —
[regge](https://crates.io/crates/regge), which is Regge geometry *on* the
manifold this crate is. Nothing here has a notion of length.

**Topology** is the combinatorial complex: incidence, orientation, navigation
(star, link, cofaces), and the chain complex with its dual. Chains carry integer
coefficients and cochains real ones; the boundary ∂ and the coboundary d are one
signed incidence relation read in its two directions, adjoint under the
chain-cochain pairing. Simplicial homology is exact: Betti numbers, relative
Betti numbers of the pair (K, ∂K), the Euler characteristic and representative
homology generators are computed by exact rational arithmetic. The boundary of a
complex is itself a first-class complex, with the trace operator as a cochain
map onto it.

**The manifold condition** is checkable as far as it can be. A complex is
pure by construction, is verified to be a pseudomanifold (every facet in one
or two cells) when it is built, and can be asked whether it is a homology
manifold: every vertex link having the homology of a sphere, or being acyclic
on the boundary. That last rung is the strongest one there is, since
recognizing a PL sphere is undecidable above dimension four, and the code says
only what it checked.

**The atlas** is the piecewise-affine chart structure: barycentric charts on the
cells, affine transition maps between charts sharing a face (obeying the cocycle
law), an intrinsic notion of a point (a cell plus barycentric weights),
Grundmann-Möller quadrature exact to prescribed degree in every dimension, and
uniform (Freudenthal) refinement recording the affine map of each child. Affine,
not flat: flatness is about curvature and presupposes a metric, while the charts
are affine maps and need none.

**The bundles** come with the atlas, and not with a geometry. A chart identifies
its cell with the reference simplex, so the tangent space at a point is ℝⁿ read
in that chart's frame, and the exterior powers over it are fibers of a bundle
the atlas alone determines. What lives on those fibers is here: the tangent
blade of a face, the trace onto a face (the pullback along its inclusion), and
the action of a transition on a fiber value. The last one is where the atlas
bites. A transition is exact on points, but on fibers it is the true change of
frame only on the tangent space of the overlap, so only the tangential part of a
value is chart-independent, and a quantity claiming to be well defined on the
manifold owes a transition argument. The multilinear algebra itself is not
reimplemented here, it is
[multialgebra](https://crates.io/crates/multialgebra).

**The Kuhn triangulation** of a box is here too, and that is not an oversight. It
is combinatorics: which simplices, in which vertex order, from the per-axis cell
counts alone. The vertex order is what makes uniform refinement compose, so it
belongs with the topology; placing the vertices in space is `regge`'s.

## Correctness

The test suite states laws and sweeps them over dimensions: ∂∘∂ = 0,
Euler-Poincaré, Poincaré duality on the sphere, Poincaré-Lefschetz for (K, ∂K),
the transition cocycle law on points and again on fibers, functoriality of the
trace along a chain of faces, the agreement of two charts on the tangential part
of a fiber value, and exactness of quadrature on polynomials.

The laws that are meant to be sharp are checked to be sharp: where two charts
agree only tangentially, the test also asserts that they genuinely disagree
before the trace is taken, so a trivial implementation could not pass.

Every fixture is built combinatorially, which is the crate's own claim tested on
itself: a 2-sphere is the boundary of a tetrahedron, and the annulus is a grid
with its middle box removed *by index*. No test here needs a coordinate.

## I/O

CBOR serialization of the combinatorial structures, behind the `serde` feature.
Mesh formats are `regge`'s, since reading a mesh means reading coordinates too.

## Place in the ecosystem

`simplicial` is the topological layer of
[formoniq](https://github.com/luiswirth/formoniq), a finite element exterior
calculus (FEEC) engine. It knows nothing of geometry: metrics and edge lengths
live in `regge`, and the reading of a cochain as a discrete differential form,
Whitney forms and PDEs in the crates above. A form on a fiber is a different
matter, and it is here, because the bundle it lives in is atlas data.
