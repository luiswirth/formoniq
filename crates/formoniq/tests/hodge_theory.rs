//! Executable discrete Hodge theory.
//!
//! The dimension of the space of discrete harmonic k-forms (closed and
//! weakly coclosed cochains) equals the k-th Betti number: the geometry
//! (mass matrices) and the topology (boundary operators) of the library are
//! cross-validated against each other.
//!
//! With essential boundary conditions the same statement holds for the
//! relative complex of the pair $(K, diff K)$ and relative (co)homology.

extern crate nalgebra as na;

use common::linalg::nalgebra::{CooMatrix, CsrMatrix, Matrix};
use ddf::cochain::Cochain;
use formoniq::whitney_complex::{RelativeWhitneyComplex, WhitneyComplex};
use manifold::gen::cartesian::CartesianMeshInfo;

use approx::assert_relative_eq;

const RANK_TOL: f64 = 1e-8;

fn rank(m: &Matrix) -> usize {
  if m.is_empty() {
    0
  } else {
    m.rank(RANK_TOL)
  }
}

fn dense(csr: &CsrMatrix) -> Matrix {
  Matrix::from(&CooMatrix::from(csr))
}

/// Dimension of the discrete harmonic space
/// $frak(H)^k = ker dif_k sect ker (dif_(k-1)^T M_k)$:
/// closed and weakly coclosed k-cochains.
fn harmonic_space_dim(
  ndofs: usize,
  dif: Option<Matrix>,
  dif_prev: Option<Matrix>,
  mass: Matrix,
) -> usize {
  let coclosed = dif_prev.map(|d| d.transpose() * mass);
  let constraints: Vec<Matrix> = dif.into_iter().chain(coclosed).collect();
  let nrows = constraints.iter().map(|m| m.nrows()).sum();
  if nrows == 0 {
    return ndofs;
  }
  let mut stacked = Matrix::zeros(nrows, ndofs);
  let mut row = 0;
  for m in &constraints {
    stacked.view_mut((row, 0), (m.nrows(), ndofs)).copy_from(m);
    row += m.nrows();
  }
  ndofs - rank(&stacked)
}

/// Betti numbers of a cochain complex given by its dif matrices,
/// $b^k = dim ker dif_k - rank dif_(k-1)$.
fn cohomology_dim(difs: &[Matrix], ndofs: &[usize], k: usize) -> usize {
  let ker = ndofs[k] - rank(&difs[k]);
  let im_prev = if k > 0 { rank(&difs[k - 1]) } else { 0 };
  ker - im_prev
}

/// Discrete Hodge theorem on the n-cube: harmonics have the dimension of the
/// cohomology, $b^k = delta_(k 0)$.
#[test]
fn harmonics_are_cohomology_cube() {
  for dim in 1..=3 {
    let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let fes = WhitneyComplex::new(&topology, &metric);

    for k in 0..=dim {
      let dif = (k < dim).then(|| dense(&fes.dif(k)));
      let dif_prev = (k > 0).then(|| dense(&fes.dif(k - 1)));
      let mass = Matrix::from(&fes.mass(k));
      let harmonic_dim = harmonic_space_dim(fes.ndofs(k), dif, dif_prev, mass);

      let betti = topology.homology_dim(k);
      let expected = (k == 0) as usize;
      assert_eq!(betti, expected);
      assert_eq!(harmonic_dim, betti, "dim={dim} k={k}");
    }
  }
}

/// Discrete Hodge theorem on the sphere: $b^k = (1, 0, 1)$.
/// The mesh is closed, so the relative complex coincides with the full one.
#[test]
fn harmonics_are_cohomology_sphere() {
  let sphere = manifold::dim3::mesh_sphere_surface(1);
  let (topology, coords) = sphere.into_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  let fes = WhitneyComplex::new(&topology, &metric);
  let dim = topology.dim();

  let relative = fes.relative();
  for k in 0..=dim {
    assert_eq!(relative.ndofs(k), fes.ndofs(k));
  }

  for (k, expected) in [(0, 1), (1, 0), (2, 1)] {
    let dif = (k < dim).then(|| dense(&fes.dif(k)));
    let dif_prev = (k > 0).then(|| dense(&fes.dif(k - 1)));
    let mass = Matrix::from(&fes.mass(k));
    let harmonic_dim = harmonic_space_dim(fes.ndofs(k), dif, dif_prev, mass);

    assert_eq!(topology.homology_dim(k), expected);
    assert_eq!(harmonic_dim, expected, "k={k}");
  }
}

/// Discrete Hodge theorem for the pair $(K, diff K)$ on the n-cube:
/// relative harmonics have the dimension of the relative cohomology,
/// $b^k (K, diff K) = delta_(k n)$ (Lefschetz duality with $b_(n-k) = delta_(k n)$).
#[test]
fn relative_harmonics_are_relative_cohomology_cube() {
  for dim in 1..=3 {
    let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let fes = WhitneyComplex::new(&topology, &metric);
    let relative = fes.relative();

    let ndofs: Vec<_> = (0..=dim).map(|k| relative.ndofs(k)).collect();
    let difs: Vec<Matrix> = (0..=dim)
      .map(|k| {
        if k < dim {
          dense(&relative.dif(k))
        } else {
          Matrix::zeros(0, ndofs[k])
        }
      })
      .collect();

    for k in 0..=dim {
      let dif = (k < dim).then(|| difs[k].clone());
      let dif_prev = (k > 0).then(|| difs[k - 1].clone());
      let mass = Matrix::from(&relative.mass(k));
      let harmonic_dim = harmonic_space_dim(ndofs[k], dif, dif_prev, mass);

      let betti_rel = cohomology_dim(&difs, &ndofs, k);
      let expected = (k == dim) as usize;
      assert_eq!(betti_rel, expected, "dim={dim} k={k}");
      assert_eq!(harmonic_dim, betti_rel, "dim={dim} k={k}");
    }
  }
}

/// The inclusion $E: C^k (K, diff K) arrow.hook C^k (K)$ is a cochain map:
/// $D E_k = E_(k+1) dif_k$.
#[test]
fn relative_inclusion_is_cochain_map() {
  for dim in 1..=3 {
    let (topology, coords) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);
    let fes = WhitneyComplex::new(&topology, &metric);
    let relative = fes.relative();

    for k in 0..dim {
      let lhs = dense(&(fes.dif(k) * relative.inclusion(k)));
      let rhs = dense(&(relative.inclusion(k + 1) * relative.dif(k)));
      assert_relative_eq!(lhs, rhs);
    }
  }
}

/// Homogeneous essential BCs via the affine-lifting interface agree with
/// the direct solve on the relative complex.
#[test]
fn lifted_homogeneous_dirichlet_is_relative_solve() {
  use common::linalg::faer::FaerCholesky;
  use common::linalg::nalgebra::Vector;
  use exterior::field::DiffFormClosure;
  use formoniq::{assemble, bc, operators::SourceElVec};

  let dim = 2;
  let (topology, coords) = CartesianMeshInfo::new_unit(dim, 4).compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  let fes = WhitneyComplex::new(&topology, &metric);
  let boundary = fes.boundary().unwrap();

  let source = DiffFormClosure::constant_scalar(1.0, dim);
  let galvec =
    assemble::assemble_galvec(&topology, &metric, SourceElVec::new(&source, &coords, dim, None));

  // Affine-lifting path with zero boundary values.
  let zero_values = Cochain::new(0, Vector::zeros(boundary.ndofs(0)));
  let laplace = CsrMatrix::from(&fes.codif_dif(0));
  let sol_lifted =
    bc::solve_with_essential_bc(&fes.relative(), &boundary, laplace, &galvec, &zero_values);

  // Direct solve on C^0(K, dK), extended by zero.
  let relative: RelativeWhitneyComplex = fes.relative();
  let laplace_relative = CsrMatrix::from(&relative.codif_dif(0));
  let rhs_relative = relative.restrict(&Cochain::new(0, galvec));
  let sol_relative = FaerCholesky::new(laplace_relative).solve(rhs_relative.coeffs());
  let sol_relative = relative.extend_by_zero(&Cochain::new(0, sol_relative));

  assert_relative_eq!(sol_lifted.coeffs, sol_relative.coeffs, epsilon = 1e-10);
}
