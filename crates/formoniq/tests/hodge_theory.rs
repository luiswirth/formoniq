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
  let nrows = constraints.iter().map(na::Matrix::nrows).sum();
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
    let whitney = WhitneyComplex::new(&topology, &metric);

    for k in 0..=dim {
      let dif = (k < dim).then(|| dense(&whitney.dif(k)));
      let dif_prev = (k > 0).then(|| dense(&whitney.dif(k - 1)));
      let mass = Matrix::from(&whitney.mass(k));
      let harmonic_dim = harmonic_space_dim(whitney.ndofs(k), dif, dif_prev, mass);

      let betti = topology.homology_dim(k);
      let expected = usize::from(k == 0);
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
  let whitney = WhitneyComplex::new(&topology, &metric);
  let dim = topology.dim();

  let relative = whitney.relative();
  for k in 0..=dim {
    assert_eq!(relative.ndofs(k), whitney.ndofs(k));
  }

  for (k, expected) in [(0, 1), (1, 0), (2, 1)] {
    let dif = (k < dim).then(|| dense(&whitney.dif(k)));
    let dif_prev = (k > 0).then(|| dense(&whitney.dif(k - 1)));
    let mass = Matrix::from(&whitney.mass(k));
    let harmonic_dim = harmonic_space_dim(whitney.ndofs(k), dif, dif_prev, mass);

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
    let whitney = WhitneyComplex::new(&topology, &metric);
    let relative = whitney.relative();

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
      let expected = usize::from(k == dim);
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
    let whitney = WhitneyComplex::new(&topology, &metric);
    let relative = whitney.relative();

    for k in 0..dim {
      let lhs = dense(&(whitney.dif(k) * relative.inclusion(k)));
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
  use ddf::field::CoordFieldExt;
  use exterior::field::DiffFormClosure;
  use formoniq::{assemble, bc, operators::SourceElVec};

  let dim = 2;
  let (topology, coords) = CartesianMeshInfo::new_unit(dim, 4).compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  let whitney = WhitneyComplex::new(&topology, &metric);
  let boundary = whitney.boundary().unwrap();

  let source = DiffFormClosure::constant_scalar(1.0, dim);
  let source = source.pullback_on(&topology, &coords);
  let galvec = assemble::assemble_galvec(&topology, &metric, SourceElVec::new(&source, None));

  // Affine-lifting path with zero boundary values.
  let zero_values = Cochain::new(0, Vector::zeros(boundary.ndofs(0)));
  let laplace = CsrMatrix::from(&whitney.codif_dif(0));
  let sol_lifted = bc::solve_with_essential_bc(
    &whitney.relative(),
    &boundary,
    laplace,
    &galvec,
    &zero_values,
  );

  // Direct solve on C^0(K, dK), extended by zero.
  let relative: RelativeWhitneyComplex = whitney.relative();
  let laplace_relative = CsrMatrix::from(&relative.codif_dif(0));
  let rhs_relative = relative.restrict(&Cochain::new(0, galvec));
  let sol_relative = FaerCholesky::new(laplace_relative).solve(rhs_relative.coeffs());
  let sol_relative = relative.extend_by_zero(&Cochain::new(0, sol_relative));

  assert_relative_eq!(sol_lifted.coeffs(), sol_relative.coeffs(), epsilon = 1e-10);
}

/// The long exact sequence of the pair $(K, diff K)$,
///
/// $dots.c -> H^k (K, diff K) -> H^k (K) -> H^k (diff K) -> H^(k+1) (K, diff K) -> dots.c$
///
/// on an annulus (square with a square hole). Exactness forces the
/// alternating sum of all dimensions to vanish, and the three Betti
/// families have their known values: absolute $(1, 1, 0)$, relative
/// $(0, 1, 1)$ (Lefschetz duality) and boundary $(2, 2)$ (two circles).
/// Both the absolute and the relative harmonic spaces match -- including a
/// genuinely nontrivial harmonic 1-form around the hole.
#[test]
fn long_exact_sequence_of_the_pair_annulus() {
  use manifold::{
    geometry::coord::simplex::SimplexCoords,
    topology::{complex::Complex, skeleton::Skeleton},
  };

  // Annulus: 3x3 boxes with the middle box removed.
  let (square, coords) = CartesianMeshInfo::new_unit(2, 3).compute_coord_complex();
  let cells: Vec<_> = square
    .cells()
    .handle_iter()
    .filter(|cell| {
      let barycenter = SimplexCoords::from_simplex_and_coords(cell.simplex(), &coords).barycenter();
      let inside = |x: f64| 1.0 / 3.0 < x && x < 2.0 / 3.0;
      !(inside(barycenter[0]) && inside(barycenter[1]))
    })
    .map(|cell| cell.simplex().clone())
    .collect();
  let topology = Complex::from_cells(Skeleton::new(cells));
  let metric = coords.to_edge_lengths(&topology);
  let whitney = WhitneyComplex::new(&topology, &metric);
  let dim = topology.dim();

  // Absolute cohomology and harmonics.
  let mut betti_abs = Vec::new();
  for k in 0..=dim {
    let betti = topology.homology_dim(k);
    let dif = (k < dim).then(|| dense(&whitney.dif(k)));
    let dif_prev = (k > 0).then(|| dense(&whitney.dif(k - 1)));
    let mass = Matrix::from(&whitney.mass(k));
    assert_eq!(
      harmonic_space_dim(whitney.ndofs(k), dif, dif_prev, mass),
      betti,
      "absolute k={k}"
    );
    betti_abs.push(betti);
  }
  assert_eq!(betti_abs, vec![1, 1, 0]);

  // Relative cohomology and harmonics.
  let relative = whitney.relative();
  let ndofs_rel: Vec<_> = (0..=dim).map(|k| relative.ndofs(k)).collect();
  let difs_rel: Vec<Matrix> = (0..=dim)
    .map(|k| {
      if k < dim {
        dense(&relative.dif(k))
      } else {
        Matrix::zeros(0, ndofs_rel[k])
      }
    })
    .collect();
  let mut betti_rel = Vec::new();
  for k in 0..=dim {
    let betti = cohomology_dim(&difs_rel, &ndofs_rel, k);
    let dif = (k < dim).then(|| difs_rel[k].clone());
    let dif_prev = (k > 0).then(|| difs_rel[k - 1].clone());
    let mass = Matrix::from(&relative.mass(k));
    assert_eq!(
      harmonic_space_dim(ndofs_rel[k], dif, dif_prev, mass),
      betti,
      "relative k={k}"
    );
    betti_rel.push(betti);
  }
  assert_eq!(betti_rel, vec![0, 1, 1]);

  // Boundary cohomology: two circles.
  let boundary = topology.boundary_complex().unwrap();
  let betti_bdry: Vec<_> = (0..=boundary.dim())
    .map(|k| boundary.complex().homology_dim(k))
    .collect();
  assert_eq!(betti_bdry, vec![2, 2]);

  // Exactness of the long exact sequence forces the alternating sum of all
  // dimensions to vanish.
  let mut alternating_sum: i64 = 0;
  for k in 0..=dim {
    let bdry = if k <= boundary.dim() {
      betti_bdry[k] as i64
    } else {
      0
    };
    let sign = if k % 2 == 0 { 1 } else { -1 };
    alternating_sum += sign * (betti_rel[k] as i64 - betti_abs[k] as i64 + bdry);
  }
  assert_eq!(alternating_sum, 0);
}
