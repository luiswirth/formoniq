extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{assemble, fe, mesh::hyperbox::HyperBoxMeshInfo, space::FeSpace};

use std::rc::Rc;

#[test]
fn fe_vs_fd() {
  for dim in 1..=5 {
    let nboxes_per_dim = 1;
    let nnodes_per_dim = nboxes_per_dim + 1;
    let mesh_width = (nboxes_per_dim as f64).recip();

    let box_mesh = HyperBoxMeshInfo::new_unit(dim, nboxes_per_dim);
    let coord_mesh = box_mesh.compute_coord_manifold();
    let mesh = Rc::new(coord_mesh.into_manifold());
    let space = FeSpace::new(mesh);
    let fe_laplacian =
      assemble::assemble_galmat(&space, fe::laplacian_neg_elmat).to_nalgebra_dense();

    let fd_laplacian =
      1.0 / mesh_width.powi(2) * laplacian_nd_neumann(&vec![nnodes_per_dim; dim]).cast();

    let diff = &fe_laplacian - &fd_laplacian;
    println!("FE:\n{fe_laplacian:.3}");
    println!("FD:\n{fd_laplacian:.3}");
    println!("diff:\n{diff:.3}");
    assert!(
      diff.norm() < 10.0 * f64::EPSILON,
      "FE and FD disagree in d={dim}"
    );
  }
}

fn laplacian_nd_neumann(sizes: &[usize]) -> na::DMatrix<f64> {
  let lapls: Vec<_> = sizes
    .iter()
    .map(|&size| laplacian_1d_neumann(size))
    .collect();
  kronecker_sum(&lapls)
}

fn laplacian_1d_neumann(size: usize) -> na::DMatrix<f64> {
  let mut laplacian = laplacian_1d_dirichlet(size);

  if size > 1 {
    laplacian[(0, 0)] = 1.0;
    laplacian[(0, 1)] = -1.0;
    laplacian[(size - 1, size - 2)] = -1.0;
    laplacian[(size - 1, size - 1)] = 1.0;
  }

  laplacian
}

fn laplacian_1d_dirichlet(size: usize) -> na::DMatrix<f64> {
  let stencil = [-1.0, 2.0, -1.0];
  matrix_from_diagonals(&stencil[..], &[-1, 0, 1], size, size)
}

fn kronecker_sum(mats: &[na::DMatrix<f64>]) -> na::DMatrix<f64> {
  assert!(!mats.is_empty());
  assert!(mats.iter().all(|m| m.nrows() == m.ncols()));

  let eyes: Vec<_> = mats
    .iter()
    .map(|m| na::DMatrix::identity(m.nrows(), m.nrows()))
    .collect();

  let kron_size = mats.iter().map(|mat| mat.nrows()).product::<usize>();
  let mut kron_sum = na::DMatrix::zeros(kron_size, kron_size);
  for (dim, mat) in mats.iter().enumerate() {
    let eyes_before = eyes[..dim]
      .iter()
      .fold(na::DMatrix::identity(1, 1), |prod, eye| prod.kronecker(eye));
    let eyes_after = eyes[dim + 1..]
      .iter()
      .fold(na::DMatrix::identity(1, 1), |prod, eye| prod.kronecker(eye));

    let kron_prod = eyes_before.kronecker(mat).kronecker(&eyes_after);
    kron_sum += kron_prod;
  }

  kron_sum
}

fn matrix_from_diagonals(
  values: &[f64],
  offsets: &[isize],
  nrows: usize,
  ncols: usize,
) -> na::DMatrix<f64> {
  let mut matrix = na::DMatrix::zeros(nrows, ncols);

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

#[test]
fn galmat_paper_and_pen() {
  // This list of galmats can be trusted.
  // It was verified thorougly.
  // The mesh nodes are ordered lexicographically.
  #[rustfmt::skip]
  let expected_galmats = [
    na::DMatrix::from_row_slice(2, 2, &[
       1, -1,
      -1,  1
    ]).cast(),
    na::DMatrix::from_row_slice(4, 4, &[
       2, -1, -1, -0, 
      -1,  2, -0, -1, 
      -1, -0,  2, -1, 
       0, -1, -1,  2,
    ]).cast() / 2.0,
  ];

  for (i, expected) in expected_galmats.iter().enumerate() {
    let dim = i + 1;
    let box_mesh = HyperBoxMeshInfo::new_unit(dim, 1);
    let coord_mesh = box_mesh.compute_coord_manifold();
    let mesh = Rc::new(coord_mesh.into_manifold());
    let space = FeSpace::new(mesh);
    let computed = assemble::assemble_galmat(&space, fe::laplacian_neg_elmat).to_nalgebra_dense();
    let diff = &computed - expected;
    println!("computed:\n{computed:.3}");
    println!("expected:\n{expected:.3}");
    println!("diff:\n{diff:.3}");
    assert!(
      diff.norm() <= 10.0 * f64::EPSILON,
      "FE and handcomputed disagree in d={dim}"
    );
  }
}
