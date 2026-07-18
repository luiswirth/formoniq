# formoniq

A Finite Element Exterior Calculus (FEEC) engine in Rust. Partial differential
equations are formulated in the language of differential forms and solved on
simplicial Riemannian manifolds of arbitrary dimension, intrinsically, without
reference to any coordinate embedding.

Research code, under active development.

## The idea

Classical finite element methods are usually written for a fixed dimension, in
explicit ambient coordinates, with separate machinery for scalar and vector
fields. Gradient, curl and divergence are each treated on their own terms. FEEC,
developed by Arnold, Falk and Winther, replaces that patchwork with a single
construction. Gradient, curl and divergence become one exterior derivative. The
scalar and vector Laplacians become cases of one Hodge-Laplace operator. Nodal,
edge and face elements become Whitney forms at different degrees. Once the vector
calculus is gone, the whole theory works in any dimension and on domains of any
topology.

formoniq takes this a step beyond most implementations. It never assumes an
embedding. A domain is an abstract simplicial complex carrying a Riemannian
metric, and the metric is supplied intrinsically. It can come from Regge-style
edge lengths, from per-cell metric tensors, or, where an embedding happens to be
available, from vertex coordinates on equal footing with the other two.
Everything the solver needs it reads from the metric alone. Lengths, areas,
volumes, the Hodge star and the Hodge-Laplace operator are all determined
without a single global coordinate. The same code then runs on domains that have
no global coordinates at all, such as a flat torus, or an abstract manifold
given only by its edge lengths.

## What it does

- **Arbitrary dimension and arbitrary form degree.** Both are runtime values, and
  nothing is specialized to 2D or 3D. The degenerate cases, meaning the base
  dimension, the extremal grades and a one-element mesh, run on the same code
  paths as the interior ones and return the trivial answer instead of being
  excluded.
- **Coordinate-free geometry.** Assembly consumes only the per-cell Riemannian
  metric. An embedding, Regge edge lengths and raw metric tensors are three
  interchangeable ways to provide it. Nothing in the core path needs
  coordinates.
- **The de Rham complex, discretely.** Simplicial cochains, first-order Whitney
  forms, the de Rham integration map and the discrete exterior derivative, with
  the commuting-diagram and reconstruction identities checked as tests.
- **Structure preservation as law-based tests.** Nilpotency, Stokes' theorem,
  functoriality of the exterior power, the involution property of the Hodge star
  and the commuting-subcomplex property are stated as theorems and swept over all
  dimensions and grades. The test suite is a machine-checked statement of the
  mathematics rather than a table of golden numbers.
- **Problems.** The Hodge-Laplace source and eigenvalue problems in the mixed
  Arnold, Falk and Winther formulation, with the harmonic space and the gauge
  constraint handled explicitly. Maxwell's equations as the Hodge-Dirac evolution
  on the full de Rham complex. The heat and wave equations.
- **Native, pure-Rust numerics.** Parallel assembly with rayon, and faer for the
  solves. A sparse LU handles the indefinite saddle-point system, Cholesky the
  constrained positive-definite systems, and a spectral-transformation Lanczos
  shift-invert the generalized eigenproblems. There is no external solver
  toolchain.

## Why topology enters

The Hodge-Laplace operator has a kernel, the space of harmonic forms, and by
Hodge's theorem that kernel is isomorphic to the de Rham cohomology of the
domain. Its dimension is the Betti number, the number of holes of a given
dimension. Topology therefore governs whether the PDE is solvable, since
existence and uniqueness hold only modulo the harmonics. FEEC keeps this exact
under discretization. The simplicial homology of the mesh reproduces the
cohomology of the continuum, which is de Rham's theorem, so the discrete operator
has a kernel of the right dimension and the harmonic forms are computed instead
of lost. On a torus the solver finds a two-dimensional space of harmonic
1-forms. On a sphere it finds none.

## The finite element families, unified

The first-order Whitney forms, indexed only by form degree k, recover the
classical finite element families as special cases.

- k = 0: Lagrange (nodal) elements
- k = 1: Nédélec edge elements
- k = n-1: Raviart-Thomas elements
- k = n: piecewise-constant discontinuous elements

Scalar and vector FEM, edge and face elements, are one construction read off at
different degrees rather than four separately implemented families.

## Architecture

The workspace is a ladder of crates, each adding one thing to the ones below it.

- `multiindex`, `gramian`, `coorder`: combinatorial index structures, metric and
  inner-product structure, and typed affine coordinates.
- `exterior`: the exterior algebra, with variance (forms versus vectors) tracked
  at the type level so that pullback, the musical isomorphisms and the Hodge star
  are correct by construction.
- `simplicial`: the simplicial manifold, with pure-combinatorial topology on one
  side and geometry entering only through a `Geometry` trait on the other.
- `glatt`: the continuum manifold that the simplicial one approximates.
- `derham`: discrete differential forms, where exterior algebra, the mesh and the
  continuum meet. This is the home of cochains, Whitney interpolation and the de
  Rham map.
- `formoniq`: the FEM engine, holding assembly, boundary conditions, time
  integration, the solvers and the problem formulations.
- `studio`: a wgpu/winit/egui visualizer that runs natively and in the browser.

Dependencies flow strictly downward. The separation of topology from geometry,
and of intrinsic geometry from any embedding, is enforced by the crate
boundaries instead of being left to convention.

Because each concept lives in the lowest crate that can express it, the lower
crates are self-contained mathematical objects rather than FEEC-internal
plumbing, and are usable on their own. `exterior` is an exterior-algebra library
— wedge, interior product, Hodge star, the musical isomorphisms and
variance-typed pullback — that knows nothing of meshes or PDEs. `simplicial`
carries pure-combinatorial topology, including boundary operators and homology
with Betti numbers, alongside intrinsic Regge geometry, none of which needs a
differential form. `multiindex` is colex-ranked combinatorics, and `glatt` is
continuum differential geometry: parametrized manifolds and their induced
metrics. FEEC is what `derham` and `formoniq` build on top, not something the
layers below are entangled with.

## Getting started

```sh
cargo test --workspace
cargo run --release --example source
```

The examples under `crates/formoniq/examples/` are the end-to-end
demonstrations. They report convergence rates and computed spectra, and are read
by hand rather than asserted.

## Origin

The first version was developed as the
[BSc thesis](https://github.com/luiswirth/bsc-thesis) of Luis Wirth at ETH
Zürich, supervised by Prof. Dr. Ralf Hiptmair. It focused on the elliptic
Hodge-Laplace problem with the first-order Whitney basis
([arXiv:2506.02429](https://arxiv.org/abs/2506.02429)). The current version is a
rebuild toward the more general library described above.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
