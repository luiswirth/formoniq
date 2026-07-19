//! Verify FEEC Galerkin Matrices for scalar (negative) Laplacian
//! on tensor product meshes by comparing to Finite Difference Method (FDM).
//!
//! We look at the homogeneous Neumann problem $-Delta u = 1$ on $[0,n]^d$.
//! We discretize this problem on a tensor-product mesh with $n$ subdivisions,
//! resulting in $h=1$, making $h$-scaling irrelevant.
//! The Mesh vertices are ordered lexicographically.
//!
//! The equivalence holds on *interior* nodes only. There the Kuhn/Freudenthal
//! triangulation reproduces the FDM stencil exactly: the piecewise-affine FEM
//! and FDM give the same LSE, since both solve for values at the mesh vertices.
//! The LSE rows might be scaled differently between the two, so we convert the
//! LSE into a canonical normalized form, where the RHS is constantly one. This
//! makes the system matrices be the same for FEM and FDM. FDM is already in
//! normalized form; FEM needs to be normalized.
//!
//! At a *boundary* corner the two discretizations genuinely diverge and no
//! normalization repairs it: FEM computes a cotangent-type Laplacian on the
//! partial simplex star, which equals the tensor FDM graph Laplacian only up
//! to a global scalar in $d <= 2$, and not up to any row scaling in $d >= 3$.
//! So only the interior comparison is a theorem, and only it is tested here.

// NOTE: h-scaling for FDM and FEM (irrelevant, since $h=1$)
// FD: $1/h^2 P u = h f_p$ (independent of dimension)
// FEM: $h^(d-2) P u = h^d f_i$.
// P is the integer (graph) Laplacian matrix.
// $f_p$ is pointwise, while $f_i$ is integrated.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::whitney_complex::WhitneyComplex;
use simplicial::{
  gen::cartesian::CartesianMeshInfo,
  linalg::{Matrix, Vector},
  Dim,
};

use std::sync::LazyLock;

/// The Kronecker sum $A_1 oplus dots.c oplus A_d = sum_i I ox dots.c ox A_i ox dots.c ox I$
/// of square matrices: the generator of the tensor-product operator each
/// factor generates alone. Test-local: the one caller here builds the
/// separable FDM Laplacian from its 1D stencil, this way.
fn kronecker_sum<T>(mats: &[Matrix<T>]) -> Matrix<T>
where
  T: na::Scalar + num_traits::Zero + num_traits::One + na::ClosedMulAssign + na::ClosedAddAssign,
{
  assert!(!mats.is_empty());
  assert!(mats.iter().all(|m| m.nrows() == m.ncols()));

  let eyes: Vec<_> = mats
    .iter()
    .map(|m| Matrix::identity(m.nrows(), m.nrows()))
    .collect();

  let kron_size = mats.iter().map(na::Matrix::nrows).product::<usize>();
  let mut kron_sum = Matrix::zeros(kron_size, kron_size);
  for (dim, mat) in mats.iter().enumerate() {
    let eyes_before = eyes[..dim]
      .iter()
      .fold(Matrix::identity(1, 1), |prod, eye| prod.kronecker(eye));
    let eyes_after = eyes[dim + 1..]
      .iter()
      .fold(Matrix::identity(1, 1), |prod, eye| prod.kronecker(eye));

    let kron_prod = eyes_before.kronecker(mat).kronecker(&eyes_after);
    kron_sum += kron_prod;
  }

  kron_sum
}

/// A banded matrix with `values[i]` on the constant diagonal at `offsets[i]`.
/// Test-local: the one caller here builds the 1D FDM stencil matrix this way.
fn matrix_from_const_diagonals<T>(
  values: &[T],
  offsets: &[isize],
  nrows: usize,
  ncols: usize,
) -> Matrix<T>
where
  T: num_traits::Zero + na::Scalar + Copy,
{
  let mut matrix = Matrix::zeros(nrows, ncols);

  for (idiag, &offset) in offsets.iter().enumerate() {
    let [start_row, start_col] = if offset >= 0 {
      [0, offset as usize]
    } else {
      [(-offset) as usize, 0]
    };

    let mut r = start_row;
    let mut c = start_col;
    while r < nrows && c < ncols {
      matrix[(r, c)] = values[idiag];
      r += 1;
      c += 1;
    }
  }

  matrix
}

/// Handchecked integer (graph) Laplacian matrices on interior of mesh.
///
/// Unit d-cube mesh with a single subdivision. One dof in each corner.
/// Imaginary dofs outside of the mesh, such that all real dofs are on the interior and get full contributions.
/// Same as having periodic boundary conditions (no boundary at all).
#[rustfmt::skip]
static LAPLACE_MATRICES_INTERIOR: LazyLock<[Matrix<i32>; 4]> = LazyLock::new(|| [
  Matrix::from_row_slice(1, 1, &[
    2
  ]),
  Matrix::from_row_slice(2, 2, &[
     2,-1,
    -1, 2
  ]),
  // Famous 2D Poisson Matrix
  Matrix::from_row_slice(4, 4, &[
     4,-1,-1, 0,
    -1, 4, 0,-1,
    -1, 0, 4,-1,
     0,-1,-1, 4,
  ]),
  Matrix::from_row_slice(8, 8, &[
     6,-1,-1, 0,-1, 0, 0, 0,
    -1, 6, 0,-1, 0,-1, 0, 0,
    -1, 0, 6,-1, 0, 0,-1, 0,
     0,-1,-1, 6, 0, 0, 0,-1,
    -1, 0, 0, 0, 6,-1,-1, 0,
     0,-1, 0, 0,-1, 6, 0,-1,
     0, 0,-1, 0,-1, 0, 6,-1,
     0, 0, 0,-1, 0,-1,-1, 6,
  ]),
]);

/// Finite Difference gives Laplace Matrix.
#[test]
fn fdm_vs_handchecked_interior() {
  for (dim, handchecked) in LAPLACE_MATRICES_INTERIOR.iter().enumerate().skip(1) {
    let fdm = ndimensionalize_operator(|_| laplace_matrix_1d_interior(2), &vec![1; dim]);
    let diff = &fdm - handchecked;
    assert!(diff.iter().all(|&e| e == 0));
  }
}

fn ndimensionalize_operator<F>(f: F, vertex_counts: &[usize]) -> Matrix<i32>
where
  F: Fn(usize) -> Matrix<i32>,
{
  let lapls: Vec<_> = vertex_counts
    .iter()
    .map(|&nvertices| f(nvertices))
    .collect();
  kronecker_sum(&lapls)
}

fn laplace_matrix_1d_interior(nvertices: usize) -> Matrix<i32> {
  let stencil = [-1, 2, -1];
  matrix_from_const_diagonals(&stencil[..], &[-1, 0, 1], nvertices, nvertices)
}

#[test]
fn feec_vs_fdm_interior() {
  for nboxes_per_dim in 1..=3 {
    for dim in 1..=4 {
      let nvertices_per_dim = nboxes_per_dim + 1;
      let feec = feec_galmat_interior(dim, nboxes_per_dim);
      let fdm = ndimensionalize_operator(laplace_matrix_1d_interior, &vec![nvertices_per_dim; dim]);
      assert_eq!(feec, fdm, "dim={dim} nboxes_per_dim={nboxes_per_dim}");
    }
  }
}

/// Galmat from normalized LSE, where RHS galvec would be constant 1.
fn feec_galmat_interior(dim: Dim, mut nboxes_per_dim: usize) -> Matrix<i32> {
  nboxes_per_dim += 2;

  let full_galmat = feec_galmat_full(dim, nboxes_per_dim);

  let boundary_vertices =
    CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, nboxes_per_dim as f64)
      .boundary_vertices();
  let galmat = full_galmat
    .remove_columns_at(&boundary_vertices)
    .remove_rows_at(&boundary_vertices);

  cast_int(galmat)
}

/// Galmat from normalized LSE, where RHS galvec would be constant 1.
fn feec_galmat_full(dim: Dim, nboxes_axis: usize) -> Matrix {
  let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_axis, nboxes_axis as f64);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths_sq(&topology);
  let whitney = WhitneyComplex::new(&topology, &metric);
  let mut galmat = Matrix::from(&whitney.codif_dif(0));
  let mass = Matrix::from(&whitney.mass(0));
  let mut galvec = mass * Vector::from_element(topology.vertices().len(), 1.0);
  normalize_galerkin_lse(&mut galmat, &mut galvec);
  galmat
}

fn normalize_galerkin_lse(galmat: &mut Matrix, galvec: &mut Vector) {
  for (mut galmat_row, galvec_entry) in galmat.row_iter_mut().zip(galvec.iter_mut()) {
    galmat_row /= *galvec_entry;
    *galvec_entry = 1.0;
  }
}

fn cast_int(mat: Matrix) -> Matrix<i32> {
  const TOL: f64 = 10e-12;
  assert!(
    mat.iter().all(|e| (e - e.round()).abs() <= TOL),
    "Failed to round matrix:\n{mat:.2}"
  );
  // Round before casting: `try_cast` truncates toward zero, which would turn
  // an entry a few ulps below an integer into the integer beneath it --
  // inconsistent with the tolerance just asserted.
  mat.map(f64::round).try_cast().unwrap()
}
