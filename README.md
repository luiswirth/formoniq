# formoniq

A Rust library for Finite Element Exterior Calculus (FEEC): partial differential
equations formulated in terms of differential forms and solved on simplicial
Riemannian manifolds of arbitrary dimension, intrinsically and coordinate-free.

Research code, under active development.

## What it does

- **Arbitrary dimension and arbitrary form degree.** Dimension and grade are
  runtime values; nothing is specialized to 2D or 3D.
- **Coordinate-free geometry.** Assembly consumes only the Riemannian metric of
  each cell. An embedding is one way to supply that metric, on equal footing
  with Regge edge lengths or per-cell metric tensors — so the solver runs on
  manifolds that have no global coordinates at all.
- **The de Rham complex, discretely.** Cochains, first-order Whitney forms, the
  de Rham map, and the discrete exterior derivative, with the commuting-diagram
  properties covered by tests.
- **Problems.** Hodge-Laplace source and eigenvalue problems, Maxwell, heat,
  wave, Laplace-Beltrami. Parallel assembly; PETSc/SLEPc solver backend.

## Getting started

```sh
cargo test --workspace
cargo run --release --example hodge_laplace_source
```

The examples under `crates/formoniq/examples/` are the end-to-end demonstrations
and report convergence rates and spectra.

## Background

FEEC extends the finite element method using differential geometry and algebraic
topology. Expressing the problem with differential forms and (co-)chain complexes
preserves the topological and structural features of the continuous problem in
the discretization — which is what makes it well suited to the Hodge-Laplace
equation and to Maxwell's equations.

Conventional finite element methods rely on explicit coordinates for the
computational domain. A coordinate-free formulation is closer to the intrinsic
nature of differential geometry: representing the domain as a simplicial
Riemannian manifold, lengths, areas and volumes are determined by the metric
alone, with no embedding. The metric is also what defines the Hodge star, and
with it the Hodge-Laplace operator.

Rust is a good fit for this: safe by default, with performance comparable to
C/C++, and a type system expressive enough to encode mathematical structure —
variance, for instance, is a type parameter here, which makes pullback versus
pushforward and the choice of metric correct by construction rather than by
convention.

## Origin

The first version was developed as the
[BSc thesis](https://github.com/luiswirth/bsc-thesis) of Luis Wirth at
ETH Zurich, supervised by Prof. Dr. Ralf Hiptmair, focusing on the elliptic
Hodge-Laplace problem with the first-order Whitney basis. The current version is
a rebuild toward the more general library described above.

Paper: [arXiv:2506.02429](https://arxiv.org/abs/2506.02429)
