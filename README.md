# formoniq

A Finite Element Exterior Calculus (FEEC) engine in Rust. Partial differential
equations are formulated in the language of differential forms and solved on
simplicial Riemannian manifolds of arbitrary dimension, intrinsically, without
reference to any coordinate embedding.

Research code, under active development.

An interactive build of the viewer runs in the browser, with no installation, at
[lwirth.com/formoniq](https://lwirth.com/formoniq): meshes, cochains and the PDE
solutions computed on them, solved client-side via WebAssembly and WebGPU.

## What it does

- **Arbitrary dimension and form degree.** Both are runtime values, and nothing
  is specialized to 2D or 3D. The degenerate cases (the base dimension, the
  extremal grades, a one-element mesh) run on the same code paths as the interior
  ones and return the trivial answer rather than being excluded.
- **Three interchangeable geometry inputs.** Assembly consumes only the per-cell
  Riemannian metric, provided as Regge edge lengths, raw metric tensors, or
  vertex coordinates. Nothing in the core path needs coordinates.
- **Problems.** The Hodge-Laplace source and eigenvalue problems in the mixed
  Arnold, Falk and Winther formulation, with the harmonic space and gauge
  constraint handled explicitly. Maxwell's equations as the Hodge-Dirac evolution
  on the full de Rham complex. The heat and wave equations.
- **Structure-preserving time integration.** Symplectic Gauss-Legendre and an
  explicit Yee-style leapfrog conserving the discrete energy, and L-stable Radau
  IIA for the dissipative problems, on the singular-mass systems the mixed
  formulation produces.
- **Native, pure-Rust numerics.** Parallel assembly with rayon, and faer for the
  solves: a sparse LU for the indefinite saddle-point system, Cholesky for the
  constrained positive-definite systems, and a shift-invert Lanczos for the
  generalized eigenproblems. No external solver toolchain.

## Core crates

The workspace is a ladder of crates, each adding one thing to the ones below it.
Dependencies flow strictly downward, and the separation of topology from
geometry, and of intrinsic geometry from any embedding, is enforced by the crate
boundaries rather than by convention.

- **`multiindex`**: colexicographic combinatorics of finite index sets, with
  ranked combinations, signed index algebra and radix multi-indices.
- **`gramian`**: inner-product and metric structure in a basis, with Gram
  matrices, Riemannian metrics and the induced distance geometry.
- **`coorder`**: affine coordinates tagged by the space they live in, so the maps
  between coordinate spaces are explicit and their confusion does not compile.
- **`exterior`**: the exterior algebra, with variance (forms versus vectors)
  tracked at the type level, so pullback, the musical isomorphisms and the Hodge
  star take their direction and metric from the type rather than from a
  convention.
- **`simplicial`**: the simplicial manifold, with pure-combinatorial topology on
  one side and geometry entering only through a `Geometry` trait on the other.
- **`glatt`**: the continuum manifold that the simplicial one approximates, with
  parametrizations and analytic differential-form data on them.
- **`derham`**: discrete differential forms, where the exterior algebra, the mesh
  and the continuum meet. The home of cochains, Whitney interpolation and the de
  Rham map.
- **`formoniq`**: the FEM engine, holding assembly, boundary conditions, time
  integration, the solvers and the problem formulations.

Because each concept lives in the lowest crate that can express it, the lower
crates are self-contained mathematical objects rather than FEEC-internal
plumbing, and are usable on their own. `exterior` is an exterior-algebra library
that knows nothing of meshes or PDEs. `simplicial` carries combinatorial topology
(boundary operators, homology, Betti numbers) alongside intrinsic Regge geometry,
none of which needs a differential form. `multiindex` is colex-ranked
combinatorics, `gramian` is metric linear algebra, and `glatt` is continuum
differential geometry. FEEC is what `derham` and `formoniq` build on top, not
something the layers below are entangled with. Each core crate carries its own
README and is published on its own.

## Visualization

`studio` is the interactive viewer for the engine: a wgpu/winit/egui application
for inspecting meshes, simplicial manifolds, cochains and the PDE solutions
computed on them. It runs natively, and in the browser via WebAssembly and
WebGPU with the solve running client-side, so the same viewer is reachable
without a toolchain at [lwirth.com/formoniq](https://lwirth.com/formoniq). It
sits at the top of the stack, depends downward on `formoniq` and below, and is
kept off the core build's critical path (excluded from the workspace's default
members), so a core change is never gated on the graphics stack.

The engine is intrinsic-first and needs no embedding; a viewer needs one, because
nothing reaches the screen until a point has a position. `studio` is therefore
the deliberate consumer of that extrinsic carve-out. It is kept intrinsic as far
as it can be: a curve integrator, for instance, works in the barycentric charts
of the atlas and crosses between cells through their affine transition maps,
committing to an ambient position only at the last step. The embedding enters at
two named seams. The `Scene` carries the engine's own types (`Complex`,
`MeshCoords`, `Cochain`) rather than a lossy export, so coloring, displacement
and the choice of render mark stay decisions made on the real object. The bake
then reduces a complex to what a rasterizer draws: simplices of dimension ≤ 2
embedded in R³, with winding and position made explicit. Downstream of the bake
there are no FEEC types, only ambient geometry.

Ambient dimension is fixed at 3, the native space of the GPU, while intrinsic
dimension and form grade stay general within it. Two reductions carry this. Form
grade reduces to a render mark through the reduced grade min(k, n−k): a scalar
density coloring, a glyph or particle line field, a standing-wave displacement
height. Intrinsic dimension reduces to a render primitive min(n, 2): a surface to
wound triangles, a curve to segments, a point set to points, and a solid to the
2-simplices of its boundary. One segment pipeline then serves the wireframe
overlay, a line field's traced ribbons and a 1-manifold's own cells, distinguished
only by material data. `crates/studio/CLAUDE.md` documents the design in full.

## Origin

The first version was developed as the
[BSc thesis](https://github.com/luiswirth/bsc-thesis) of Luis Wirth at ETH
Zürich, supervised by Prof. Dr. Ralf Hiptmair. It focused on the elliptic
Hodge-Laplace problem with the first-order Whitney basis
([arXiv:2506.02429](https://arxiv.org/abs/2506.02429)). The current version is a
rebuild toward the more general library described above.

## Getting started

```sh
cargo test --workspace
cargo run --release --example source
```

The examples under `crates/formoniq/examples/` are the end-to-end
demonstrations. They report convergence rates and computed spectra, and are read
by hand rather than asserted.

## Motivation

Classical finite element methods are usually written for a fixed dimension, in
explicit ambient coordinates, with separate machinery for scalar and vector
fields, and with gradient, curl and divergence each treated on their own terms.
Conforming vector-valued elements, the Nédélec and Raviart-Thomas families, were
originally constructed case by case and are intricate to derive and implement.

FEEC, developed by Arnold, Falk and Winther, replaces that patchwork with one
construction over differential forms. Gradient, curl and divergence are one
exterior derivative d. The scalar and vector Laplacians are one Hodge-Laplace
operator Δ = dδ + δd. Nodal, edge and face elements are Whitney forms at
different degrees. Once the vector calculus is gone, the same construction works
in any dimension and on domains of any topology.

The organizing idea is to discretize the whole de Rham complex at once rather
than each function space in isolation, and to keep its structure exact under
discretization: the nilpotency d∘d = 0, the exactness relations, and the
cohomology. A discretization that preserves these is stable and convergent, and
reproduces the topology of the domain rather than recovering it approximately.
This is what "structure-preserving" means here, and it is the reason FEEC is the
standard framework for constructing conforming finite element spaces for
differential forms.

## Finite element families

The first-order Whitney forms, indexed only by form degree k, recover the
classical families as special cases:

- k = 0: Lagrange (nodal) elements
- k = 1: Nédélec edge elements
- k = n-1: Raviart-Thomas elements
- k = n: piecewise-constant discontinuous elements

Scalar and vector FEM, edge and face elements, are one construction taken at
different degrees rather than four separately implemented families. In the FEEC
classification the Whitney space is the lowest-order trimmed polynomial space,
`WΛᵏ = P⁻₁Λᵏ`; higher-order `P⁻ᵣΛᵏ` elements are a direction being explored.

## Intrinsic geometry

Most finite element implementations assume an embedding: the domain lives in Rᴺ
and geometry is read from vertex coordinates. formoniq does not. A domain is an
abstract simplicial complex carrying a Riemannian metric supplied intrinsically,
from Regge-style edge lengths, from per-cell metric tensors, or, where an
embedding happens to be available, from vertex coordinates on equal footing with
the other two.

Everything the solver needs is read from the metric alone: lengths, areas,
volumes, the Hodge star and the Hodge-Laplace operator, all without a global
coordinate. The same code then runs on domains that have no global coordinates
at all, such as a flat torus or a manifold given only by its edge lengths.
Because the metric is the only geometric input, the formulation extends without
special cases to pseudo-Riemannian geometry, for instance a Lorentzian metric on
a 4D spacetime, a direction being explored toward Maxwell's equations.

## Topology and cohomology

The Hodge-Laplace operator is singular. Its kernel is the space of harmonic
forms, and by Hodge's theorem that kernel is isomorphic to the de Rham
cohomology of the domain, with dimension the Betti number βₖ, the number of
k-dimensional holes. Topology therefore governs solvability: existence and
uniqueness of the source problem hold only modulo the harmonics.

FEEC keeps this exact under discretization. The simplicial homology of the mesh
reproduces the cohomology of the continuum (de Rham's theorem), so the discrete
operator has a kernel of the right dimension and the harmonic forms are computed
explicitly. On a torus the solver finds a two-dimensional space of harmonic
1-forms; on a sphere it finds none.

The mixed formulation of Arnold, Falk and Winther makes this structure explicit.
It introduces the codifferential weakly, so that only the exterior derivative
appears in the discrete spaces (finite element spaces conforming to both HΛᵏ and
its adjoint are hard to build), carries the harmonic part as an unknown, and
fixes the gauge u ⊥ ℋᵏ. The result is a well-posed saddle-point system.

## Structure preservation

Two maps connect the continuous and discrete complexes. The de Rham map R
discretizes a form by integrating it over each simplex, giving a cochain. The
Whitney map W reconstructs a cochain into a piecewise-polynomial form by
interpolation. Both commute with the exterior derivative, so the Whitney forms
`WΛᵏ` are a subcomplex of the de Rham complex and the projection Πₕ = W∘R
commutes with d.

On the discrete side the exterior derivative is purely topological: the transpose
of the signed incidence matrix of the mesh, with no metric involved, dual to the
simplicial boundary operator under the chain-cochain pairing. The boundary
squares to zero (∂∘∂ = 0) exactly as the exterior derivative does (d∘d = 0).

These identities are the test suite. Nilpotency, Whitney's theorem R∘W = id,
Stokes' theorem R∘d = d∘R, the commuting-subcomplex property d∘W = W∘d, the
functoriality of the exterior power and the involution of the Hodge star are
stated as theorems and swept over all dimensions and grades. The suite is a
machine-checked statement of the mathematics rather than a table of golden
numbers.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
