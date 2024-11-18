extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use formoniq::{assemble, fe, mesh::hyperbox::HyperBoxMeshInfo, space::FeSpace, Dim};

use std::rc::Rc;

const DIM: Dim = 3;

#[test]
fn feec_vs_fem3d() {
  let feec = feec_galmat(1);
  let fem = fem3d_galmat();
  let diff = feec - fem;
  let error = diff.norm();
  assert!(error <= 10e-9);
}

// TODO: generalize for arbitrary `nboxes_per_dim`
fn fem3d_galmat() -> na::DMatrix<f64> {
  let node_coords = vec![
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

  let mut galmat = na::DMatrix::<f64>::zeros(node_coords.len(), node_coords.len());
  for tet_ivertices in &tets {
    let tet_vertices: Vec<_> = tet_ivertices
      .iter()
      .map(|&i| node_coords[i].clone())
      .collect();

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

/// Galmat from normalized LSE, where RHS galvec would be constant 1.
fn feec_galmat(nboxes_per_dim: usize) -> na::DMatrix<f64> {
  let box_mesh = HyperBoxMeshInfo::new_unit_scaled(DIM, nboxes_per_dim, nboxes_per_dim as f64);
  let coord_mesh = box_mesh.compute_coord_manifold();
  let mesh = Rc::new(coord_mesh.into_manifold());
  let space = FeSpace::new(mesh.clone());
  assemble::assemble_galmat(&space, fe::laplacian_neg_elmat).to_nalgebra_dense()
}
