//! Verify FEEC Galerkin Matrices for negative Laplacian
//! on tensor product meshes by comparing to other methods.
//!
//! Mesh nodes are ordered lexicographically.

// FD equations is always $1/h^2 P u = h f$
// FEM equation varies from dimension to dimension
// It is $h^(d-2) P u = h^d f$.
// P is the integer poisson matrix / graph laplacian.

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{
  assemble, fe,
  mesh::hyperbox::HyperBoxMeshInfo,
  space::FeSpace,
  util::{kronecker_sum, matrix_from_const_diagonals},
  Dim,
};

use std::{rc::Rc, sync::LazyLock};

/// Handchecked integer Poisson matrices (graph laplacian) on interior of mesh.
///
/// Unit d-cube mesh with a (real) dof in each corner.
/// Imaginary dofs outside of the mesh, such that all real dofs are on the interior and get full contributions.
/// This is the same as having periodic boundary conditions (no boundary at all).
#[rustfmt::skip]
static POISSON_MATRICES_INTERIOR: LazyLock<[na::DMatrix<i32>; 4]> = LazyLock::new(|| [
  na::DMatrix::from_row_slice(1, 1, &[
    2
  ]),
  na::DMatrix::from_row_slice(2, 2, &[
     2,-1,
    -1, 2
  ]),
  // Famous 2D Poisson Matrix
  na::DMatrix::from_row_slice(4, 4, &[
     4,-1,-1, 0,
    -1, 4, 0,-1,
    -1, 0, 4,-1,
     0,-1,-1, 4,
  ]),
  na::DMatrix::from_row_slice(8, 8, &[
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

/// Handchecked integer Poisson matrices (graph laplacian) for the boundary of the mesh.
///
/// Unit d-cube mesh, where the dofs in the corner are all on the boundary.
#[rustfmt::skip]
static POISSON_MATRICES_BOUNDARY: LazyLock<[na::DMatrix<i32>; 4]> = LazyLock::new(|| [
  na::DMatrix::from_row_slice(1, 1, &[
    1
  ]),
  na::DMatrix::from_row_slice(2, 2, &[
     1,-1,
    -1, 1
  ]),
  na::DMatrix::from_row_slice(4, 4, &[
     2,-1,-1, 0,
    -1, 2, 0,-1,
    -1, 0, 2,-1,
     0,-1,-1, 2,
  ]),
  na::DMatrix::from_row_slice(8, 8, &[
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

/// Finite Difference gives Poisson Matrix.
#[test]
fn fd_vs_handchecked_interior() {
  for (dim, handchecked) in POISSON_MATRICES_INTERIOR.iter().enumerate().skip(1) {
    let fd = ndimensionalize_operator(|_| poisson_matrix_1d_interior(2), &vec![1; dim]);
    let diff = &fd - handchecked;
    assert!(diff.iter().all(|&e| e == 0));
  }
}

/// Finite Difference gives Poisson Matrix.
#[test]
fn fd_vs_handchecked_boundary() {
  for (dim, handchecked) in POISSON_MATRICES_BOUNDARY.iter().enumerate().skip(1) {
    let fd = ndimensionalize_operator(|_| poisson_matrix_1d_boundary(), &vec![1; dim]);
    let diff = &fd - handchecked;
    assert!(diff.iter().all(|&e| e == 0));
  }
}

fn feec_galmat_interior(dim: Dim, mut nboxes_per_dim: usize) -> na::DMatrix<f64> {
  nboxes_per_dim += 2;

  let full_galmat = feec_galmat_full(dim, nboxes_per_dim);

  let removable_nodes =
    HyperBoxMeshInfo::new_unit_scaled(dim, nboxes_per_dim, nboxes_per_dim as f64).boundary_nodes();

  full_galmat
    .remove_columns_at(&removable_nodes)
    .remove_rows_at(&removable_nodes)
}

fn feec_galmat_boundary(dim: Dim) -> na::DMatrix<f64> {
  feec_galmat_full(dim, 1)
}

fn feec_galmat_full(dim: Dim, nboxes_per_dim: usize) -> na::DMatrix<f64> {
  let box_mesh = HyperBoxMeshInfo::new_unit_scaled(dim, nboxes_per_dim, nboxes_per_dim as f64);
  let coord_mesh = box_mesh.compute_coord_manifold();
  let mesh = Rc::new(coord_mesh.into_manifold());
  let space = FeSpace::new(mesh);
  assemble::assemble_galmat(&space, fe::laplacian_neg_elmat).to_nalgebra_dense()
}

#[test]
fn feec_vs_fd_interior() {
  for nboxes_per_dim in 1..=1 {
    for dim in 1..=4 {
      let nnodes_per_dim = nboxes_per_dim + 1;
      let feec = feec_galmat_interior(dim, nboxes_per_dim);
      let fd =
        ndimensionalize_operator(poisson_matrix_1d_interior, &vec![nnodes_per_dim; dim]).cast();
      compare_galmats(&feec, &fd);
    }
  }
}

// TODO: get this right for 3D!
#[test]
fn feec_vs_fd_boundary() {
  for dim in 1..=3 {
    // TODO: justify this scaling ! Is this even the right scaling???
    let feec = feec_galmat_boundary(dim) * dim as f64;
    let fd = ndimensionalize_operator(|_| poisson_matrix_1d_boundary(), &vec![1; dim]).cast();
    compare_galmats(&feec, &fd);
  }
}

fn ndimensionalize_operator<F>(f: F, node_counts: &[usize]) -> na::DMatrix<i32>
where
  F: Fn(usize) -> na::DMatrix<i32>,
{
  let lapls: Vec<_> = node_counts.iter().map(|&nnodes| f(nnodes)).collect();
  kronecker_sum(&lapls)
}

fn poisson_matrix_1d_boundary() -> na::DMatrix<i32> {
  poisson_matrix_1d_full(2)
}

/// Graph Laplacian for tensor-product mesh/graph.
fn poisson_matrix_1d_full(nnodes: usize) -> na::DMatrix<i32> {
  assert!(nnodes >= 2);
  let mut mat = poisson_matrix_1d_interior(nnodes);
  mat[(0, 0)] = 1;
  mat[(0, 1)] = -1;
  mat[(nnodes - 1, nnodes - 2)] = -1;
  mat[(nnodes - 1, nnodes - 1)] = 1;
  mat
}

fn poisson_matrix_1d_interior(nnodes: usize) -> na::DMatrix<i32> {
  let stencil = [-1, 2, -1];
  matrix_from_const_diagonals(&stencil[..], &[-1, 0, 1], nnodes, nnodes)
}

#[test]
fn feec_vs_fem_3d() {
  let dim = 3;

  let feec = feec_galmat_full(dim, 1);
  let fem = fem3d_galmat();
  compare_galmats(&feec, &fem);
}

fn fem3d_galmat() -> na::DMatrix<f64> {
  let vertices = vec![
    na::DVector::from_column_slice(&[0.0, 0.0, 0.0]),
    na::DVector::from_column_slice(&[1.0, 0.0, 0.0]),
    na::DVector::from_column_slice(&[0.0, 1.0, 0.0]),
    na::DVector::from_column_slice(&[1.0, 1.0, 0.0]),
    na::DVector::from_column_slice(&[0.0, 0.0, 1.0]),
    na::DVector::from_column_slice(&[1.0, 0.0, 1.0]),
    na::DVector::from_column_slice(&[0.0, 1.0, 1.0]),
    na::DVector::from_column_slice(&[1.0, 1.0, 1.0]),
  ];

  let tets = [
    vec![0, 1, 3, 7],
    vec![0, 1, 5, 7],
    vec![0, 2, 3, 7],
    vec![0, 2, 6, 7],
    vec![0, 4, 5, 7],
    vec![0, 4, 6, 7],
  ];
  let tet_vol = 1.0 / 6.0;

  let mut galmat = na::DMatrix::<f64>::zeros(vertices.len(), vertices.len());
  for tet_ivertices in &tets {
    let tet_vertices: Vec<_> = tet_ivertices.iter().map(|&i| vertices[i].clone()).collect();

    let ns: Vec<na::DVector<f64>> = (0..4)
      .map(|i| {
        let mut face = tet_vertices.clone();
        face.remove(i);

        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let b0 = &face[1] - &face[0];
        let b1 = &face[2] - &face[0];
        sign * b0.cross(&b1)
      })
      .collect();

    let mut elmat = na::DMatrix::zeros(4, 4);
    for i in 0..4 {
      for j in 0..4 {
        let ni = &ns[i];
        let nj = &ns[j];
        elmat[(i, j)] = ni.dot(nj);
      }
    }
    elmat *= 1.0 / (36.0 * tet_vol);

    for (ilocal, iglobal) in tet_ivertices.iter().copied().enumerate() {
      for (jlocal, jglobal) in tet_ivertices.iter().copied().enumerate() {
        galmat[(iglobal, jglobal)] += elmat[(ilocal, jlocal)];
      }
    }
  }

  galmat
}

fn compare_galmats(feec: &na::DMatrix<f64>, other: &na::DMatrix<f64>) {
  let diff = feec - other;
  let equal = diff.norm() <= 100.0 * f64::EPSILON;
  if !equal {
    let quotient = feec.component_div(other);
    println!("FEEC:\n{feec:.2}");
    println!("other:\n{other:.2}");
    println!("diff:\n{diff:.2}");
    println!("quotient:\n{quotient:.2}");
    panic!("FEEC and other disagree");
  }
}
