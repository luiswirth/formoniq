# formoniq

A Finite Element Exterior Calculus (FEEC) engine in Rust. Partial differential
equations are formulated in the language of differential forms and solved on
simplicial pseudo-Riemannian manifolds of arbitrary dimension, intrinsically,
without reference to any coordinate embedding.

Gradient, curl and divergence do not appear separately: they are one exterior
derivative. The scalar and vector Laplacians are cases of one Hodge-Laplace
operator. Dimension and form grade are runtime values, and nothing is
specialized to 2D or 3D.

## What it does

- **Coordinate-free assembly.** The Galerkin operators of the Whitney
  discretization of the L² de Rham complex, assembled in parallel from per-cell
  metrics alone. An embedding, Regge edge lengths and raw metric tensors are
  three interchangeable geometry inputs; the solvers run identically on all
  three, including on manifolds with no global coordinates.
- **The Hodge-Laplace problem.** Source and eigenvalue problems in the mixed
  formulation of Arnold, Falk and Winther, with the harmonic space (the
  cohomology of the domain) and the gauge constraint handled explicitly.
- **Boundary conditions as complexes.** Essential conditions restrict to the
  relative Whitney complex, natural conditions enter as boundary loads, Robin
  through the boundary mass. The solvers consume any Hilbert complex unchanged.
- **Structure-preserving evolution.** Maxwell as the Hodge-Dirac evolution on the
  full de Rham complex, integrated symplectically (Gauss-Legendre, energy
  conserved) or by an explicit Yee-style leapfrog; the Hodge heat and wave
  equations through the same mixed blocks.
- **Pure-Rust numerics.** Sparse LU and Cholesky, and a shift-invert block
  Lanczos eigensolver for generalized pencils with singular mass blocks, via
  faer. No external solver toolchain.

## Correctness

The test suite states the mathematics as laws and sweeps dimensions and grades:
the Dirac operator squares to the negative Hodge Laplacian, energy is conserved
for the hyperbolic problems and dissipates monotonically for the parabolic ones,
the L² projection reproduces Whitney forms, and assembly from edge lengths agrees
with assembly from metric tensors to machine precision. The examples (`source`,
`evp`, `heat`, `wave`, `dirac`) are the end-to-end convergence and spectrum
checks.

## Place in the ecosystem

`formoniq` is the top of a small stack of standalone crates:
[exterior](https://crates.io/crates/exterior) (exterior algebra),
[simplicial](https://crates.io/crates/simplicial) (simplicial topology and Regge
geometry), [glatt](https://crates.io/crates/glatt) (the smooth continuum) and
[derham](https://crates.io/crates/derham) (discrete differential forms). See the
[repository](https://github.com/luiswirth/formoniq) for the full picture.

## Origin

The first version was the BSc thesis of Luis Wirth at ETH Zürich, supervised by
Prof. Dr. Ralf Hiptmair ([arXiv:2506.02429](https://arxiv.org/abs/2506.02429)).

## License

Dual-licensed under either MIT or Apache-2.0, at your option.
