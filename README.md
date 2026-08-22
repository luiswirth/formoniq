# formoniq

A Finite Element Exterior Calculus (FEEC) engine in Rust.
Partial differential equations are formulated in the language of differential forms
and solved on simplicial pseudo-Riemannian manifolds of arbitrary dimension,
intrinsically, without reference to any coordinate embedding.

formoniq makes no distinction between Riemannian and Lorentzian geometry.
The metric carries its own signature, and a single assembly serves both regimes:
elliptic Hodge-Laplace problems on a Riemannian mesh,
and hyperbolic field theory on a Lorentzian spacetime mesh.
The geometry is Regge calculus on signed squared edge lengths
(positive spacelike, zero null, negative timelike),
which is the coordinate-free spacetime discretization Regge introduced for general relativity.
On a Minkowski mesh, Maxwell's equations solve as a covariant Hodge-Dirac operator
that is hyperbolic through the metric signature alone,
assembled from the Regge lengths with the embedding forgotten.

Research code, under active development.

An interactive build of the viewer runs in the browser, with no installation,
at [lwirth.com/formoniq-studio](https://lwirth.com/formoniq-studio):
meshes, cochains and the PDE solutions computed on them,
solved client-side via WebAssembly and WebGPU.

## What it does

- **Arbitrary dimension and form degree:**
  Both are runtime values, and nothing is specialized to 2D or 3D.
  The degenerate cases (the base dimension, the extremal grades, a one-element mesh)
  run on the same code paths as the interior ones
  and return the trivial answer rather than being excluded.
- **Three interchangeable geometry inputs:**
  Assembly consumes only the per-cell metric,
  provided as Regge signed squared edge lengths, raw metric tensors, or vertex coordinates.
  Nothing in the core path needs coordinates.
- **Any metric signature:**
  The metric is pseudo-Riemannian:
  Riemannian and Lorentzian geometry are one signature-parameterized type,
  the Hodge star reading the signature off the metric itself,
  which makes spacetime FEEC on Lorentzian (e.g. Minkowski) meshes
  the same machinery as elliptic problems on Riemannian ones.
- **Problems:**
  The Hodge-Laplace source and eigenvalue problems
  in the mixed Arnold, Falk and Winther formulation,
  with the harmonic space and gauge constraint handled explicitly.
  Maxwell's equations as the Hodge-Dirac evolution on the full de Rham complex,
  and the covariant form of the same operator solved directly on a Minkowski spacetime mesh,
  where hyperbolicity comes from the signature rather than a time-stepping loop.
  The heat and wave equations.
- **Structure-preserving time integration:**
  Symplectic Gauss-Legendre and an explicit Yee-style leapfrog conserving the discrete energy,
  and L-stable Radau IIA for the dissipative problems,
  on the singular-mass systems the mixed formulation produces.
- **Native, pure-Rust numerics:**
  Parallel assembly with rayon, and faer for the solves:
  a sparse LU for the indefinite saddle-point system,
  Cholesky for the constrained positive-definite systems,
  and a shift-invert Lanczos for the generalized eigenproblems.
  No external solver toolchain.

## Core crates

The workspace is a ladder of crates, each adding one thing to the ones below it.
Dependencies flow strictly downward,
and the separation of topology from geometry, and of intrinsic geometry from any embedding,
is enforced by the crate boundaries rather than by convention.

- **[`multiindex`](crates/multiindex/README.md)**:
  colexicographic combinatorics of finite index sets,
  with ranked combinations, signed index algebra and radix multi-indices.
- **[`coorder`](crates/coorder/README.md)**:
  affine coordinates tagged by the space they live in,
  so the maps between coordinate spaces are explicit and their confusion does not compile.
- **[`multialgebra`](crates/multialgebra/README.md)**:
  the free tensor power and its exterior and symmetric quotients as one construction,
  and tensor products of all three,
  metric-free throughout,
  with the variance of each slot deciding pullback against pushforward
  and which metric measures it.
- **[`metric`](crates/metric/README.md)**:
  pseudo-Riemannian metrics of arbitrary signature on a tangent space,
  and the operations needing one:
  the inner product, the Hodge star and the musical isomorphisms,
  with g and g⁻¹ one datum rather than a stored pair.
  Nothing to do with meshes.
- **[`simplicial`](crates/simplicial/README.md)**:
  the simplicial manifold: its topology and its piecewise-affine structure,
  the atlas and the bundle the atlas determines, metric-free throughout,
  with a geometry a genuinely separate input rather than a field on the mesh.
- **[`regge`](crates/regge/README.md)**:
  Regge geometry, which is geometry on that manifold:
  the metric of a piecewise-flat cell,
  with signed squared edge lengths as the primitive
  and coordinates or per-cell metrics as sources converting into them,
  so nothing in the core path requires an embedding.
- **[`glatt`](crates/glatt/README.md)**:
  the continuum manifold that the simplicial one approximates,
  with parametrizations and analytic differential-form data on them.
  It has never heard of a mesh.
- **[`derham`](crates/derham/README.md)**:
  discrete differential forms, everything about them:
  cochains read as forms, Whitney interpolation, the de Rham map,
  the degrees of freedom, and the road in from the continuum.
- **[`formoniq`](crates/formoniq/README.md)**:
  the FEM engine, holding assembly, boundary conditions, time integration,
  the solvers and the problem formulations.

Because each concept lives in the lowest crate that can express it,
the lower crates are self-contained mathematical objects rather than FEEC-internal plumbing,
and are usable on their own.
`multialgebra` is a multilinear-algebra library that knows nothing of meshes or PDEs.
`simplicial` is simplicial topology and the piecewise-affine structure on it
(boundary operators, homology, Betti numbers, charts),
and `regge` the discrete metric geometry that structure carries,
neither of which needs a differential form.
`multiindex` is colex-ranked combinatorics, `metric` is pseudo-Riemannian linear algebra,
and `glatt` is continuum differential geometry.
FEEC is what `derham` and `formoniq` build on top,
not something the layers below are entangled with.
Each core crate carries its own README.

## Off to the side

**[`iterative`](crates/iterative/README.md)** is deliberately not part of that ladder,
because it models no part of FEEC.
It sits below the mathematics: Krylov methods, preconditioners and smoothers
around one object, an approximate inverse.
It depends on nothing but sparse matrices
and would serve any PDE code equally well.

## Extrinsic output

The engine is intrinsic-first and needs no embedding.
Anything looked at needs one, because nothing reaches a screen or an interchange file
until a point has a position.
That carve-out is a boundary of each crate rather than a crate of its own:
what becomes extrinsic is still *of* the object it was intrinsic on,
so it lives with that object.
Mesh formats (Gmsh, Wavefront OBJ, the MDD point cache) are `regge::io`,
a mesh being a topology and its coordinates together;
the VTU written for ParaView is `derham::io`,
being a manifold and the discrete forms on it at once.

The reading a form gets on the way out is one rule and is shared, not duplicated.
A k-form and its Hodge dual are one datum, read at the grade min(k, n−k) where the pair is smallest:
a scalar density at 0, a tangent line field at 1.
That is `derham::reduce`, and an exporter's data array and a viewer's mark
are the same reading of the same field.

The interactive viewer is a separate project,
[formoniq-studio](https://github.com/luiswirth/formoniq-studio):
a wgpu/winit/egui application that runs natively and in the browser
via WebAssembly and WebGPU, with the solve running client-side.
It carries its own render reductions: a complex to wound triangles in R³, and the marks over them.
It depends on this repository and nothing here depends on it,
so the graphics stack is never in the engine's build.

## Origin

The first version was developed as the
[BSc thesis](https://github.com/luiswirth/bsc-thesis) of Luis Wirth at ETH Zürich,
supervised by Prof. Dr. Ralf Hiptmair.
It focused on the elliptic Hodge-Laplace problem with the first-order Whitney basis
([arXiv:2506.02429](https://arxiv.org/abs/2506.02429)).
The current version is a rebuild toward the more general library described above.

## Getting started

The engine is published on [crates.io](https://crates.io/crates/formoniq),
with documentation on [docs.rs](https://docs.rs/formoniq).
Depend on it with `cargo add formoniq`.

To build from source:

```sh
cargo test --workspace
cargo run --release --example source
```

The examples under `crates/formoniq/examples/` are the end-to-end demonstrations.
They report convergence rates and computed spectra, and are read by hand rather than asserted.

## Motivation

Classical finite element methods are usually written for a fixed dimension,
in explicit ambient coordinates,
with separate machinery for scalar and vector fields,
and with gradient, curl and divergence each treated on their own terms.
Conforming vector-valued elements, the Nédélec and Raviart-Thomas families,
were originally constructed case by case and are intricate to derive and implement.

FEEC, developed by Arnold, Falk and Winther,
replaces that patchwork with one construction over differential forms.
Gradient, curl and divergence are one exterior derivative d.
The scalar and vector Laplacians are one Hodge-Laplace operator Î = dÎ´ + Î´d.
Nodal, edge and face elements are Whitney forms at different degrees.
Once the vector calculus is gone,
the same construction works in any dimension and on domains of any topology.

The organizing idea is to discretize the whole de Rham complex at once
rather than each function space in isolation,
and to keep its structure exact under discretization:
the nilpotency d∘d = 0, the exactness relations, and the cohomology.
A discretization that preserves these is stable and convergent,
and reproduces the topology of the domain rather than recovering it approximately.
This is what "structure-preserving" means here,
and it is the reason FEEC is the standard framework
for constructing conforming finite element spaces for differential forms.

## Finite element families

The first-order Whitney forms, indexed only by form degree k,
recover the classical families as special cases:

- k = 0: Lagrange (nodal) elements
- k = 1: Nédélec edge elements
- k = n-1: Raviart-Thomas elements
- k = n: piecewise-constant discontinuous elements

Scalar and vector FEM, edge and face elements,
are one construction taken at different degrees
rather than four separately implemented families.
In the FEEC classification the Whitney space is the lowest-order trimmed polynomial space,
`WΛᵏ = P⁻₁Λᵏ`.
Higher-order `P⁻ᵣΛᵏ` elements are a direction being explored.

## Intrinsic geometry

Most finite element implementations assume an embedding:
the domain lives in Rá´º and geometry is read from vertex coordinates.
formoniq does not.
A domain is an abstract simplicial complex carrying a pseudo-Riemannian metric
supplied intrinsically, from Regge-style signed squared edge lengths,
from per-cell metric tensors, or, where an embedding happens to be available,
from vertex coordinates on equal footing with the other two.

Everything the solver needs is read from the metric alone:
lengths, areas, volumes, the Hodge star and the Hodge-Laplace operator,
all without a global coordinate.
The same code then runs on domains that have no global coordinates at all,
such as a flat torus or a manifold given only by its edge lengths.
Because the metric is the only geometric input,
the same formulation covers pseudo-Riemannian geometry of any signature without special cases,
the Hodge star reading the signature off the metric itself.
A Lorentzian metric on a 4D spacetime is one such signature:
Maxwell's equations run on it as the covariant Hodge-Dirac operator,
hyperbolic through the signature alone.

## Topology and cohomology

The Hodge-Laplace operator is singular.
Its kernel is the space of harmonic forms,
and by Hodge's theorem that kernel is isomorphic to the de Rham cohomology of the domain,
with dimension the Betti number βₖ, the number of k-dimensional holes.
Topology therefore governs solvability:
existence and uniqueness of the source problem hold only modulo the harmonics.

FEEC keeps this exact under discretization.
The simplicial homology of the mesh reproduces the cohomology of the continuum (de Rham's theorem),
so the discrete operator has a kernel of the right dimension
and the harmonic forms are computed explicitly.
On a torus the solver finds a two-dimensional space of harmonic 1-forms.
On a sphere it finds none.

The mixed formulation of Arnold, Falk and Winther makes this structure explicit.
It introduces the codifferential weakly,
so that only the exterior derivative appears in the discrete spaces
(finite element spaces conforming to both HÎáµ and its adjoint are hard to build),
carries the harmonic part as an unknown, and fixes the gauge u ⊥ ℋᵏ.
The result is a well-posed saddle-point system.

## Structure preservation

Two maps connect the continuous and discrete complexes.
The de Rham map R discretizes a form by integrating it over each simplex, giving a cochain.
The Whitney map W reconstructs a cochain into a piecewise-polynomial form by interpolation.
Both commute with the exterior derivative,
so the Whitney forms `WÎáµ` are a subcomplex of the de Rham complex
and the projection Πₕ = W∘R commutes with d.

On the discrete side the exterior derivative is purely topological:
the transpose of the signed incidence matrix of the mesh, with no metric involved,
dual to the simplicial boundary operator under the chain-cochain pairing.
The boundary squares to zero (∂∘∂ = 0) exactly as the exterior derivative does (d∘d = 0).

These identities are the test suite.
Nilpotency, Whitney's theorem R∘W = id, Stokes' theorem R∘d = d∘R,
the commuting-subcomplex property d∘W = W∘d,
the functoriality of the exterior power and the involution of the Hodge star
are stated as theorems and swept over all dimensions and grades.
The suite is a machine-checked statement of the mathematics
rather than a table of golden numbers.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE),
at your option.
