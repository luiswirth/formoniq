use itertools::Itertools;

use crate::{util::factorial, Dim};

use super::{Mesh, MeshSimplex, NodeId};

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
  for icomp in 0..d {
    idx *= dlen;
    idx += coord[icomp];
  }
  idx
}

pub fn unit_hypercube_mesh_nodes(d: usize, nsubdivisions: usize) -> na::DMatrix<f64> {
  let nodes_per_dim = nsubdivisions + 1;
  let nnodes = nodes_per_dim.pow(d as u32);
  let mut nodes = na::DMatrix::zeros(d, nnodes);

  for (inode, mut coord) in nodes.column_iter_mut().enumerate() {
    coord.copy_from(&na::DVector::from_iterator(
      d,
      linear_idx2cartesian_idx(inode, d, nodes_per_dim)
        .into_iter()
        .map(|&c| c as f64 / (nodes_per_dim - 1) as f64),
    ));
  }

  nodes
}

/// Create a structured mesh of the unit hypercube $[0, 1]^d$.
pub fn unit_hypercube_mesh(d: Dim, nsubdivisions: usize) -> Mesh {
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

  let nodes = unit_hypercube_mesh_nodes(d, nsubdivisions);
  from_facets(nodes, simplicies, false)
}
