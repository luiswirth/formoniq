//! Executable discrete Hodge theory.
//!
//! The dimension of the space of discrete harmonic k-forms (closed and
//! weakly coclosed cochains) equals the k-th Betti number: the geometry
//! (mass matrices) and the topology (boundary operators) of the library are
//! cross-validated against each other.
//!
//! With essential boundary conditions the same statement holds for the
//! relative complex of the pair $(K, partial K)$ and relative (co)homology.

extern crate nalgebra as na;

use formoniq::whitney_complex::{HilbertComplex, WhitneyComplex};
use regge::coord::simplex::simplex_coords;
use regge::lengths::mesh::MeshLengthsSq;
use regge::mesher::cartesian::CartesianGrid;
use simplicial::{
  Dim,
  linalg::{CooMatrix, CsrMatrix, Matrix, Vector},
  topology::complex::Complex,
};

const RANK_TOL: f64 = 1e-8;

fn rank(m: &Matrix) -> usize {
  if m.is_empty() { 0 } else { m.rank(RANK_TOL) }
}

fn dense(csr: &CsrMatrix) -> Matrix {
  Matrix::from(&CooMatrix::from(csr))
}

/// Dimension of the discrete harmonic space
/// $frak(H)^k = ker dif_k inter ker (dif_(k-1)^T M_k)$:
/// closed and weakly coclosed k-cochains.
///
/// The two constraint blocks are stacked with no case for the extremal grades:
/// the complex is total in grade, so at the top $dif_k$ has no rows and at
/// grade $0$ neither does $dif_(k-1)^T M_k$, and a block of no rows constrains
/// nothing. Handing those the honest empty operator is what makes this one
/// statement over every grade.
fn harmonic_space_dim(ndofs: usize, dif: Matrix, dif_prev: Matrix, mass: Matrix) -> usize {
  let constraints = [dif, dif_prev.transpose() * mass];
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

/// The discrete Hodge theorem on one cochain complex: the harmonic space has
/// the dimension of the cohomology, at every grade, and both agree with the
/// Betti numbers of the space.
fn assert_hodge_theorem<C: HilbertComplex>(complex: &C, name: &str, expected: &[usize]) {
  let dim = Dim::from(expected.len() - 1);
  let ndofs: Vec<_> = dim.range_inclusive().map(|k| complex.ndofs(k)).collect();
  let difs: Vec<Matrix> = dim
    .range_inclusive()
    .map(|k| dense(&complex.dif(k)))
    .collect();

  for k in dim.range_inclusive() {
    let dif_prev = dense(&complex.dif(k - 1));
    let mass = Matrix::from(&complex.mass(k));
    let harmonic_dim =
      harmonic_space_dim(ndofs[k.index()], difs[k.index()].clone(), dif_prev, mass);
    let betti = cohomology_dim(&difs, &ndofs, k.index());

    assert_eq!(betti, expected[k.index()], "{name}, k = {k}: cohomology");
    assert_eq!(harmonic_dim, betti, "{name}, k = {k}: harmonics");
  }
}

/// The spaces the theorems below are stated over, each with the Betti
/// numbers of its two readings, absolute $b^k (K)$ and relative
/// $b^k (K, partial K)$.
///
/// The cube stages Poincaré--Lefschetz duality, $delta_(k 0)$ against
/// $delta_(k n)$. The sphere is closed, so its boundary subcomplex is empty
/// and the two readings coincide, which is a case the code cannot tell apart
/// rather than one it excludes: it is the only fixture whose $b^k$ is neither
/// concentrated in one grade nor forced by contractibility.
struct Fixture {
  name: String,
  topology: Complex,
  lengths: MeshLengthsSq,
  /// $b^k (K)$, the cohomology of the full complex.
  absolute: Vec<usize>,
  /// $b^k (K, partial K)$, the cohomology of the pair.
  relative: Vec<usize>,
}

fn fixtures() -> Vec<Fixture> {
  let mut fixtures = Vec::new();

  for dim in (1..=3).map(Dim::from) {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let absolute = dim.range_inclusive().map(|k| usize::from(k == 0)).collect();
    let relative = dim
      .range_inclusive()
      .map(|k| usize::from(k == dim))
      .collect();
    fixtures.push(Fixture {
      name: format!("cube^{dim}"),
      topology,
      lengths,
      absolute,
      relative,
    });
  }

  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(1);
  let lengths = coords.to_edge_lengths_sq(&topology);
  fixtures.push(Fixture {
    name: "sphere".to_string(),
    topology,
    lengths,
    absolute: vec![1, 0, 1],
    relative: vec![1, 0, 1],
  });

  fixtures
}

/// The discrete Hodge theorem, $frak(H)^k tilde.equals H^k$, on both readings
/// of every fixture.
///
/// The absolute side is cross-checked against [`Complex::betti_number`], so
/// the geometry (mass matrices) and the topology (boundary operators) of the
/// library are held against each other.
#[test]
fn harmonics_are_cohomology() {
  for Fixture {
    name,
    topology,
    lengths,
    absolute,
    relative,
  } in fixtures()
  {
    let whitney = WhitneyComplex::new(&topology, &lengths);

    for (k, &expected) in absolute.iter().enumerate() {
      assert_eq!(
        topology.betti_number(Dim::from(k)),
        expected,
        "{name}, k = {k}"
      );
    }

    assert_hodge_theorem(&whitney, &name, &absolute);
    assert_hodge_theorem(&whitney.relative(), &format!("{name}, relative"), &relative);
  }
}

/// The Hodge decomposition on one cochain complex: every cochain splits as
/// $omega = dif alpha + delta beta + h$, the three parts mutually
/// $L^2$-orthogonal and the last one harmonic.
fn assert_hodge_decomposition<C: HilbertComplex>(complex: &C, name: &str, grade: Dim) {
  let ndofs = complex.ndofs(grade);
  if ndofs == 0 {
    return;
  }
  let mass = Matrix::from(&complex.mass(grade));
  let dif = dense(&complex.dif(grade));
  let dif_prev = dense(&complex.dif(grade - 1));

  // A spanning set of each of the two exact subspaces: the image of $dif$ one
  // grade below, which is `dif_prev` itself, and the image of the
  // codifferential $delta = M^(-1) dif^T M$ one grade above, which is that of
  // $M^(-1) dif^T$.
  let exact = &dif_prev;
  let coexact = mass
    .clone()
    .cholesky()
    .expect("a Riemannian mass matrix is positive definite")
    .solve(&dif.transpose());

  // The $L^2$ projection onto the span of a set of cochains: the normal
  // equations $B^T M B c = B^T M omega$ in the mass inner product, solved
  // through the symmetric eigendecomposition of the Gram matrix, since a
  // spanning set that is not a basis makes them singular. The kept spectrum
  // is separated from the discarded one by many orders, so the threshold
  // reads a rank rather than choosing one.
  let project = |basis: &Matrix, omega: &Vector| {
    if basis.ncols() == 0 {
      return Vector::zeros(ndofs);
    }
    let gram = basis.transpose() * &mass * basis;
    let rhs = basis.transpose() * (&mass * omega);
    let eigen = gram.symmetric_eigen();
    let largest = eigen.eigenvalues.max();

    let coeffs = eigen
      .eigenvalues
      .iter()
      .enumerate()
      .filter(|&(_, &lambda)| lambda > 1e-10 * largest)
      .map(|(i, &lambda)| {
        let direction = eigen.eigenvectors.column(i);
        direction.dot(&rhs) / lambda * direction
      })
      .sum::<Vector>();
    basis * coeffs
  };

  let omega = Vector::from_fn(ndofs, |i, _| ((i % 7) as f64) - 3.0);
  let closed = project(exact, &omega);
  let coclosed = project(&coexact, &omega);
  let harmonic = &omega - &closed - &coclosed;

  let inner = |a: &Vector, b: &Vector| (a.transpose() * &mass * b)[(0, 0)];
  let scale = inner(&omega, &omega).sqrt();

  // Harmonic: closed and weakly coclosed.
  if dif.nrows() > 0 {
    let closure = (&dif * &harmonic).norm();
    assert!(
      closure < 1e-8 * scale,
      "{name}, k = {grade}: dif h is {closure}, scale {scale}"
    );
  }
  if dif_prev.ncols() > 0 {
    let coclosure = dif_prev.transpose() * (&mass * &harmonic);
    assert!(
      coclosure.norm() < 1e-8 * scale,
      "{name}, k = {grade}: delta h"
    );
  }

  // Mutually orthogonal, hence Pythagoras, which is what makes the splitting
  // unique rather than merely possible.
  let parts = [&closed, &coclosed, &harmonic];
  for (i, a) in parts.iter().enumerate() {
    for b in parts.iter().skip(i + 1) {
      assert!(
        inner(a, b).abs() < 1e-8 * scale * scale,
        "{name}, k = {grade}: the parts are not orthogonal"
      );
    }
  }
  let pythagoras: f64 = parts.iter().map(|part| inner(part, part)).sum();
  approx::assert_relative_eq!(
    pythagoras,
    inner(&omega, &omega),
    epsilon = 1e-8 * scale * scale
  );
}

/// The Hodge decomposition
/// $C^k = dif C^(k-1) plus.circle delta C^(k+1) plus.circle frak(H)^k$,
/// orthogonal in the $L^2$ inner product, on both readings of every fixture
/// and at every grade.
///
/// This is the theorem the dimension count above is a shadow of: the harmonic
/// space is not merely of the right size, it is what is left of an arbitrary
/// cochain once the exact and coexact parts are removed, and the removal is
/// unique because the sum is orthogonal.
#[test]
fn every_cochain_splits_into_exact_coexact_and_harmonic() {
  for Fixture {
    name,
    topology,
    lengths,
    ..
  } in fixtures()
  {
    let whitney = WhitneyComplex::new(&topology, &lengths);
    let relative = whitney.relative();

    for grade in topology.dim().range_inclusive() {
      assert_hodge_decomposition(&whitney, &name, grade);
      assert_hodge_decomposition(&relative, &format!("{name}, relative"), grade);
    }
  }
}

/// The long exact sequence of the pair $(K, partial K)$,
///
/// $dots.c -> H^k (K, partial K) -> H^k (K) -> H^k (partial K) -> H^(k+1) (K, partial K) -> dots.c$
///
/// on an annulus (square with a square hole). Exactness forces the
/// alternating sum of all dimensions to vanish, and the three Betti
/// families have their known values: absolute $(1, 1, 0)$, relative
/// $(0, 1, 1)$ (Lefschetz duality) and boundary $(2, 2)$ (two circles).
/// Both the absolute and the relative harmonic spaces match, including a
/// genuinely nontrivial harmonic 1-form around the hole.
#[test]
fn long_exact_sequence_of_the_pair_annulus() {
  use simplicial::topology::{complex::Complex, skeleton::Skeleton};

  // Annulus: 3x3 boxes with the middle box removed.
  let (square, coords) = CartesianGrid::new_unit(Dim::new(2), 3).triangulate();
  let cells: Vec<_> = square
    .cells()
    .handle_iter()
    .filter(|cell| {
      let barycenter = simplex_coords(cell.simplex(), &coords).barycenter();
      let inside = |x: f64| 1.0 / 3.0 < x && x < 2.0 / 3.0;
      !(inside(barycenter[0]) && inside(barycenter[1]))
    })
    .map(|cell| cell.simplex().clone())
    .collect();
  let topology = Complex::from_cells(Skeleton::new(cells));
  let metric = coords.to_edge_lengths_sq(&topology);
  let whitney = WhitneyComplex::new(&topology, &metric);
  let dim = topology.dim();

  // Absolute cohomology and harmonics.
  let mut betti_abs = Vec::new();
  for k in dim.range_inclusive() {
    let betti = topology.betti_number(k);
    let dif = dense(&whitney.dif(k));
    let dif_prev = dense(&whitney.dif(k - 1));
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
  let ndofs_rel: Vec<_> = dim.range_inclusive().map(|k| relative.ndofs(k)).collect();
  let difs_rel: Vec<Matrix> = dim
    .range_inclusive()
    .map(|k| dense(&relative.dif(k)))
    .collect();
  let mut betti_rel = Vec::new();
  for k in dim.range_inclusive() {
    let betti = cohomology_dim(&difs_rel, &ndofs_rel, k.index());
    let dif = difs_rel[k.index()].clone();
    let dif_prev = dense(&relative.dif(k - 1));
    let mass = Matrix::from(&relative.mass(k));
    assert_eq!(
      harmonic_space_dim(ndofs_rel[k.index()], dif, dif_prev, mass),
      betti,
      "relative k={k}"
    );
    betti_rel.push(betti);
  }
  assert_eq!(betti_rel, vec![0, 1, 1]);

  // Boundary cohomology: two circles.
  let boundary = topology.boundary_complex().unwrap();
  let betti_bdry: Vec<_> = boundary
    .dim()
    .range_inclusive()
    .map(|k| boundary.complex().betti_number(k))
    .collect();
  assert_eq!(betti_bdry, vec![2, 2]);

  // Exactness of the long exact sequence forces the alternating sum of all
  // dimensions to vanish.
  let mut alternating_sum: i64 = 0;
  for k in dim.range_inclusive() {
    let bdry = if k <= boundary.dim() {
      betti_bdry[k.index()] as i64
    } else {
      0
    };
    let sign = if k.index() % 2 == 0 { 1 } else { -1 };
    alternating_sum += sign * (betti_rel[k.index()] as i64 - betti_abs[k.index()] as i64 + bdry);
  }
  assert_eq!(alternating_sum, 0);
}
