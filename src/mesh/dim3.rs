use super::coordinates::{CoordManifold, NodeCoords};
use crate::{
  combinatorics::{Orientation, OrientedSimplex},
  VertexIdx,
};

use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct TriangleSurface3D {
  triangles: Vec<[VertexIdx; 3]>,
  node_coords: na::Matrix3xX<f64>,
}
impl TriangleSurface3D {
  pub fn new(triangles: Vec<[VertexIdx; 3]>, node_coords: na::Matrix3xX<f64>) -> Self {
    Self {
      triangles,
      node_coords,
    }
  }
  pub fn triangles(&self) -> &[[VertexIdx; 3]] {
    &self.triangles
  }
  pub fn node_coords(&self) -> &na::Matrix3xX<f64> {
    &self.node_coords
  }
}

impl TriangleSurface3D {
  pub fn from_coord_manifold(mesh: CoordManifold) -> Self {
    assert!(mesh.dim_embedded() == 3, "Manifold is not embedded in 3D.");
    assert!(mesh.dim_intrinsic() == 2, "Manifold is not a surface.");

    let (cells, node_coords) = mesh.into_parts();

    let triangles = cells
      .into_iter()
      .map(|c| {
        let mut vs: [VertexIdx; 3] = c.vertices().to_vec().try_into().unwrap();
        if c.orientation().is_neg() {
          vs.swap(0, 1);
        }
        vs
      })
      .collect();

    let node_coords = na::try_convert(node_coords.into_inner()).unwrap();

    Self::new(triangles, node_coords)
  }

  pub fn into_coord_manifold(self) -> CoordManifold {
    let node_coords = NodeCoords::new(na::convert(self.node_coords));
    let cells = self
      .triangles
      .into_iter()
      .map(|t| OrientedSimplex::new(t.into(), Orientation::Pos))
      .collect();
    CoordManifold::new(cells, node_coords)
  }

  pub fn to_obj_string(&self) -> String {
    let mut string = String::new();
    for v in self.node_coords.column_iter() {
      writeln!(string, "v {:.6} {:.6} {:.6}", v.x, v.y, v.z).unwrap();
    }
    for t in &self.triangles {
      // .obj uses 1-indexing.
      writeln!(string, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1).unwrap();
    }
    string
  }

  pub fn displace_normal(&mut self, displacements: &[f64]) {
    let mut vertex_normals = vec![na::Vector3::zeros(); self.node_coords.ncols()];
    let mut vertex_triangle_counts = vec![0; self.node_coords.ncols()];
    for ivs in &self.triangles {
      let vs = ivs.map(|i| self.node_coords.column(i));
      let e0 = vs[1] - vs[0];
      let e1 = vs[2] - vs[0];
      let triangle_normal = e0.cross(&e1).normalize();
      for &iv in ivs {
        vertex_normals[iv] += triangle_normal;
        vertex_triangle_counts[iv] += 1;
      }
    }
    for (vertex_normal, count) in vertex_normals.iter_mut().zip(vertex_triangle_counts) {
      *vertex_normal /= count as f64;
    }
    for ((mut v, n), &d) in self
      .node_coords
      .column_iter_mut()
      .zip(vertex_normals)
      .zip(displacements)
    {
      v += d * n;
    }
  }
}
