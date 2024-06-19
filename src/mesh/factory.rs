use super::{Mesh, MeshSimplex, NodeId};

pub fn from_facets(node_coords: na::DMatrix<f64>, facets: Vec<Vec<NodeId>>) -> Mesh {
  let dim_intrinsic = facets[0].len() - 1;

  let mut simplicies = vec![Vec::new(); dim_intrinsic + 1];
  simplicies[0] = (0..node_coords.ncols()).map(|i| vec![i]).collect();
  simplicies[dim_intrinsic] = facets
    .into_iter()
    .map(|mut s| {
      s.sort_unstable();
      s
    })
    .collect();

  let mut face_relation: Vec<Vec<Vec<usize>>> = vec![Vec::new(); dim_intrinsic];

  for child_dim in (0..dim_intrinsic).rev() {
    let parent_dim = child_dim + 1;
    let ([.., child_simps], [parent_simps, ..]) = simplicies.split_at_mut(parent_dim) else {
      unreachable!()
    };
    face_relation[parent_dim - 1] = vec![Vec::new(); parent_simps.len()];
    for (iparent_simp, parent_simp) in parent_simps.iter().enumerate() {
      for iv in 0..parent_simp.len() {
        let mut child_simp = parent_simp.clone();
        child_simp.remove(iv);

        let ichild_simp = child_simps
          .iter()
          // since indices are sorted we can just compare the vecs
          .position(|f| *f == child_simp);
        let ichild_simp = ichild_simp.unwrap_or_else(|| {
          child_simps.push(child_simp);
          child_simps.len() - 1
        });
        face_relation[parent_dim - 1][iparent_simp].push(ichild_simp);
      }
    }
  }

  let simplicies = simplicies
    .into_iter()
    .map(|dsimps| dsimps.into_iter().map(MeshSimplex::new).collect())
    .collect();

  Mesh {
    node_coords,
    simplicies,
    face_relation,
  }
}
