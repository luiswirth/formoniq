extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

use common::linalg::assert_mat_eq;
use formoniq::{assemble, operators};
use geometry::coord::manifold::cartesian::CartesianMeshInfo;
use topology::Dim;

const DIM: Dim = 3;

#[test]
fn feec_vs_fem3d() {
  for nboxes_per_dim in 1..=10 {
    let feec = feec_galmat(nboxes_per_dim);
    let fem = fem3d_galmat(nboxes_per_dim);
    assert_mat_eq(&feec, &fem, None);
  }
}

fn fem3d_galmat(nboxes_per_dim: usize) -> na::DMatrix<f64> {
  let nvertices_per_dim = nboxes_per_dim + 1;
  let nvertices = nvertices_per_dim.pow(DIM as u32);

  let h = (nboxes_per_dim as f64).recip();
  let box_vol = h.powi(DIM as i32);

  let mut vertex_coords = na::DMatrix::zeros(3, nvertices);
  for zvertex in 0..nvertices_per_dim {
    for yvertex in 0..nvertices_per_dim {
      for xvertex in 0..nvertices_per_dim {
        let ivertex = xvertex + nvertices_per_dim * (yvertex + nvertices_per_dim * zvertex);
        vertex_coords[(0, ivertex)] = h * xvertex as f64;
        vertex_coords[(1, ivertex)] = h * yvertex as f64;
        vertex_coords[(2, ivertex)] = h * zvertex as f64;
      }
    }
  }

  let tets_ivertices = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
  ];
  let tet_vol = box_vol / tets_ivertices.len() as f64;

  let mut galmat = na::DMatrix::<f64>::zeros(nvertices, nvertices);
  for zbox in 0..nboxes_per_dim {
    for ybox in 0..nboxes_per_dim {
      for xbox in 0..nboxes_per_dim {
        let mut box_ivertices = [0; 8];
        for k in 0..2 {
          for j in 0..2 {
            for i in 0..2 {
              let ivertex_local = i + 2 * (j + 2 * k);

              let xvertex = xbox + i;
              let yvertex = ybox + j;
              let zvertex = zbox + k;
              let ivertex_global =
                xvertex + nvertices_per_dim * (yvertex + nvertices_per_dim * zvertex);

              box_ivertices[ivertex_local] = ivertex_global;
            }
          }
        }

        for tet_ivertices in &tets_ivertices {
          let tet_ivertices = tet_ivertices.map(|i| box_ivertices[i]);
          let tet_vertices = tet_ivertices.map(|i| vertex_coords.column(i));

          let ns: Vec<na::DVector<f64>> = (0..tet_vertices.len())
            .map(|i| {
              let mut face = tet_vertices.to_vec();
              face.remove(i);

              let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
              let b0 = face[1] - face[0];
              let b1 = face[2] - face[0];
              sign * b0.cross(&b1)
            })
            .collect();

          let mut elmat = na::DMatrix::zeros(tet_vertices.len(), tet_vertices.len());
          for i in 0..tet_vertices.len() {
            for j in 0..tet_vertices.len() {
              let ni = &ns[i];
              let nj = &ns[j];
              elmat[(i, j)] = ni.dot(nj);
            }
          }
          elmat /= 36.0 * tet_vol;

          for (ilocal, iglobal) in tet_ivertices.iter().copied().enumerate() {
            for (jlocal, jglobal) in tet_ivertices.iter().copied().enumerate() {
              galmat[(iglobal, jglobal)] += elmat[(ilocal, jlocal)];
            }
          }
        }
      }
    }
  }

  galmat
}

fn feec_galmat(nboxes_per_dim: usize) -> na::DMatrix<f64> {
  let box_mesh = CartesianMeshInfo::new_unit(DIM, nboxes_per_dim);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  assemble::assemble_galmat(&topology, &metric, operators::LaplaceBeltramiElmat).to_nalgebra_dense()
}
