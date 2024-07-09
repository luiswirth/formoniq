use super::{Mesh, MeshSimplex, NodeId};
use crate::{util::factorial, Dim};

use itertools::Itertools;

pub fn from_facets(
  node_coords: na::DMatrix<f64>,
  facets: Vec<Vec<NodeId>>,
  compute_faces: bool,
) -> Mesh {
  let dim_intrinsic = facets[0].len() - 1;

  let mut simplicies = vec![Vec::new(); dim_intrinsic + 1];
  simplicies[0] = (0..node_coords.ncols())
    .map(|i| MeshSimplex::new(vec![i]))
    .collect();
  simplicies[dim_intrinsic] = facets.into_iter().map(MeshSimplex::new).collect();

  let mut face_relation: Vec<Vec<Vec<usize>>> = vec![Vec::new(); dim_intrinsic];

  if compute_faces {
    for child_dim in (0..dim_intrinsic).rev() {
      let parent_dim = child_dim + 1;
      let ([.., child_simps], [parent_simps, ..]) = simplicies.split_at_mut(parent_dim) else {
        unreachable!()
      };
      face_relation[parent_dim - 1] = vec![Vec::new(); parent_simps.len()];
      for (iparent_simp, parent_simp) in parent_simps.iter().enumerate() {
        for iv in 0..parent_simp.vertices.len() {
          let mut child_simp = parent_simp.vertices.clone();
          child_simp.remove(iv);
          let child_simp = MeshSimplex::new(child_simp);

          let ichild_simp = child_simps.iter().position(|f| *f == child_simp);
          let ichild_simp = ichild_simp.unwrap_or_else(|| {
            child_simps.push(child_simp);
            child_simps.len() - 1
          });
          face_relation[parent_dim - 1][iparent_simp].push(ichild_simp);
        }
      }
    }
  }

  Mesh {
    node_coords,
    simplicies,
    face_relation,
  }
}

pub fn linear_idx2cartesian_idx(mut idx: usize, d: Dim, dlen: usize) -> na::DVector<usize> {
  let mut coord = na::DVector::zeros(d);
  for icomp in 0..d {
    coord[icomp] = idx % dlen;
    idx /= dlen;
  }
  coord
}

pub fn cartesian_idx2linear_idx(coord: na::DVector<usize>, dlen: usize) -> usize {
  let d = coord.len();
  let mut idx = 0;
  for icomp in (0..d).rev() {
    idx *= dlen;
    idx += coord[icomp];
  }
  idx
}

pub fn linear_idx2cartesian_coords(
  idx: usize,
  d: Dim,
  dlen: usize,
  side_length: f64,
) -> na::DVector<f64> {
  linear_idx2cartesian_idx(idx, d, dlen).cast::<f64>() / (dlen - 1) as f64 * side_length
}

pub fn hypercube_mesh_nodes(d: usize, nsubdivisions: usize, side_length: f64) -> na::DMatrix<f64> {
  let nodes_per_dim = nsubdivisions + 1;
  let nnodes = nodes_per_dim.pow(d as u32);
  let mut nodes = na::DMatrix::zeros(d, nnodes);

  for (inode, mut coord) in nodes.column_iter_mut().enumerate() {
    coord.copy_from(&linear_idx2cartesian_coords(
      inode,
      d,
      nodes_per_dim,
      side_length,
    ));
  }

  nodes
}

/// Create a structured mesh of the unit hypercube $[0, 1]^d$.
pub fn hypercube_mesh(d: Dim, nsubdivisions: usize, side_length: f64) -> Mesh {
  let nodes_per_dim = nsubdivisions + 1;
  let ncubes = nsubdivisions.pow(d as u32);
  let nsimplicies = factorial(d) * ncubes;
  let mut simplicies = Vec::with_capacity(nsimplicies);

  for icube in 0..ncubes {
    let cube_coord = linear_idx2cartesian_idx(icube, d, nsubdivisions);

    simplicies.extend((0..d).permutations(d).map(|permut| {
      let mut vertices = Vec::with_capacity(d + 1);
      let mut vertex = cube_coord.clone();
      vertices.push(cartesian_idx2linear_idx(vertex.clone(), nodes_per_dim));
      for p in permut {
        vertex[p] += 1;
        vertices.push(cartesian_idx2linear_idx(vertex.clone(), nodes_per_dim));
      }
      vertices
    }));
  }

  let nodes = hypercube_mesh_nodes(d, nsubdivisions, side_length);
  from_facets(nodes, simplicies, false)
}

#[cfg(test)]
mod test {
  use super::hypercube_mesh;

  #[test]
  fn unit_cube_mesh() {
    let mesh = hypercube_mesh(3, 1, 1.0);
    #[rustfmt::skip]
    let node_coords = na::DMatrix::from_column_slice(3, 8, &[
      0., 0., 0.,
      1., 0., 0.,
      0., 1., 0.,
      1., 1., 0.,
      0., 0., 1.,
      1., 0., 1.,
      0., 1., 1.,
      1., 1., 1.,
    ]);
    assert_eq!(*mesh.node_coords(), node_coords);
    let expected_simplicies = vec![
      &[0, 1, 3, 7],
      &[0, 1, 5, 7],
      &[0, 2, 3, 7],
      &[0, 2, 6, 7],
      &[0, 4, 5, 7],
      &[0, 4, 6, 7],
    ];
    let computed_simplicies: Vec<_> = mesh.cells().iter().map(|c| c.vertices()).collect();
    assert_eq!(computed_simplicies, expected_simplicies);
  }
}
