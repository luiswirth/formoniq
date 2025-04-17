//! Verify FEEC Galerkin Matrices for scalar (negative) Laplacian
//! on tensor product meshes by comparing to Finite Difference Method (FDM).
//!
//! We look at the homogeneous Neumann problem $-Delta u = 1$ on $[0,n]^d$.
//! We discretize this problem on a tensor-product mesh with $n$ subdivisions,
//! resulting in $h=1$, making $h$-scaling irrelevant.
//! The Mesh vertices are ordered lexicographically.
//! Piecewise-linear FEM and FDM should give rise to the same LSE,
//! since both are solving for values at the mesh vertices.
//! The LSE rows might be scaled differently between the two,
//! so we convert the LSE into a canonical normalized form,
//! where the RHS is constantly one. This makes the system matrices
//! be the same for FEM and FDM. FDM is already in normalized form.
//! FEM needs to be normalized.

// NOTE: h-scaling for FDM and FEM (irrelevant, since $h=1$)
// FD: $1/h^2 P u = h f_p$ (independent of dimension)
// FEM: $h^(d-2) P u = h^d f_i$.
// P is the integer (graph) Laplacian matrix.
// $f_p$ is pointwise, while $f_i$ is integrated.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use common::linalg::nalgebra::{kronecker_sum, matrix_from_const_diagonals, Matrix, Vector};
use formoniq::{
  assemble,
  operators::{self},
};
use manifold::{gen::cartesian::CartesianMeshInfo, Dim};

use std::sync::LazyLock;

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

/// Handchecked integer (graph) Laplacian matrices on the boundary of the mesh.
///
/// Unit d-cube mesh with a single subdivision. One dof in each corner on the boundary.
#[rustfmt::skip]
static LAPLACE_MATRICES_BOUNDARY: LazyLock<[Matrix<i32>; 4]> = LazyLock::new(|| [
  Matrix::from_row_slice(1, 1, &[
    1
  ]),
  Matrix::from_row_slice(2, 2, &[
     1,-1,
    -1, 1
  ]),
  Matrix::from_row_slice(4, 4, &[
     2,-1,-1, 0,
    -1, 2, 0,-1,
    -1, 0, 2,-1,
     0,-1,-1, 2,
  ]),
  Matrix::from_row_slice(8, 8, &[
     3,-1,-1, 0,-1, 0, 0, 0,
    -1, 3, 0,-1, 0,-1, 0, 0,
    -1, 0, 3,-1, 0, 0,-1, 0,
     0,-1,-1, 3, 0, 0, 0,-1,
    -1, 0, 0, 0, 3,-1,-1, 0,
     0,-1, 0, 0,-1, 3, 0,-1,
     0, 0,-1, 0,-1, 0, 3,-1,
     0, 0, 0,-1, 0,-1,-1, 3,
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

/// Finite Difference gives Laplace Matrix.
#[test]
fn fdm_vs_handchecked_boundary() {
  for (dim, handchecked) in LAPLACE_MATRICES_BOUNDARY.iter().enumerate().skip(1) {
    let fdm = ndimensionalize_operator(|_| laplace_matrix_1d_boundary(), &vec![1; dim]);
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

fn laplace_matrix_1d_boundary() -> Matrix<i32> {
  laplace_matrix_1d_full(2)
}

/// Graph Laplacian for tensor-product mesh/graph.
fn laplace_matrix_1d_full(nvertices: usize) -> Matrix<i32> {
  assert!(nvertices >= 2);
  let mut mat = laplace_matrix_1d_interior(nvertices);
  mat[(0, 0)] = 1;
  mat[(0, 1)] = -1;
  mat[(nvertices - 1, nvertices - 2)] = -1;
  mat[(nvertices - 1, nvertices - 1)] = 1;
  mat
}

fn laplace_matrix_1d_interior(nvertices: usize) -> Matrix<i32> {
  let stencil = [-1, 2, -1];
  matrix_from_const_diagonals(&stencil[..], &[-1, 0, 1], nvertices, nvertices)
}

#[test]
fn feec_vs_fdm_interior() {
  let mut equal = true;
  // TODO: increase numbers, once performance allows
  for nboxes_per_dim in 1..=2 {
    for dim in 1..=4 {
      let nvertices_per_dim = nboxes_per_dim + 1;
      let feec = feec_galmat_interior(dim, nboxes_per_dim);
      let fdm = ndimensionalize_operator(laplace_matrix_1d_interior, &vec![nvertices_per_dim; dim]);
      equal &= compare_system_matrics(&feec, &fdm);
    }
  }
  assert!(equal);
}

// TODO: get this right!
//#[test]
#[allow(dead_code)]
fn feec_vs_fdm_boundary() {
  let mut equal = true;
  for dim in 1..=3 {
    let feec = feec_galmat_boundary(dim);
    let feec = cast_int(feec);
    let fdm = ndimensionalize_operator(|_| laplace_matrix_1d_boundary(), &vec![1; dim]);
    equal &= compare_system_matrics(&feec, &fdm);
  }
  assert!(equal);
}

/// Galmat from normalized LSE, where RHS galvec would be constant 1.
fn feec_galmat_interior(dim: Dim, mut nboxes_per_dim: usize) -> Matrix<i32> {
  nboxes_per_dim += 2;

  let full_galmat = feec_galmat_full(dim, nboxes_per_dim);

  // TODO: optimize!
  let boundary_vertices =
    CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, nboxes_per_dim as f64)
      .boundary_vertices();
  let galmat = full_galmat
    .remove_columns_at(&boundary_vertices)
    .remove_rows_at(&boundary_vertices);

  cast_int(galmat)
}

/// Galmat from normalized LSE, where RHS galvec would be constant 1.
fn feec_galmat_boundary(dim: Dim) -> Matrix {
  feec_galmat_full(dim, 1)
}

/// Galmat from normalized LSE, where RHS galvec would be constant 1.
fn feec_galmat_full(dim: Dim, nboxes_axis: usize) -> Matrix {
  let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_axis, nboxes_axis as f64);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  let galmat = assemble::assemble_galmat(
    &topology,
    &metric,
    operators::LaplaceBeltramiElmat::new(dim),
  );
  let mut galmat = Matrix::from(&galmat);
  let mass = assemble::assemble_galmat(&topology, &metric, operators::ScalarMassElmat);
  let mass = Matrix::from(&mass);
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

#[must_use]
fn compare_system_matrics(feec: &Matrix<i32>, fdm: &Matrix<i32>) -> bool {
  let diff = feec - fdm;
  let equal = diff.iter().all(|&e| e == 0);
  if !equal {
    println!("FEEC:\n{feec}");
    println!("FDM:\n{fdm}");
    println!("diff:\n{diff}");
    return false;
  }
  true
}

fn cast_int(mat: Matrix) -> Matrix<i32> {
  const TOL: f64 = 10e-12;
  assert!(
    mat.iter().all(|e| (e - e.round()).abs() <= TOL),
    "Failed to round matrix:\n{mat:.2}"
  );
  mat.try_cast().unwrap()
}
