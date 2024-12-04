use super::coordinates::{CoordManifold, NodeCoords};
use crate::{
  combo::{Sign, OrientedVertplex},
  VertexIdx,
};

use std::{collections::HashMap, fmt::Write, sync::LazyLock};

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
        let mut vs: [VertexIdx; 3] = c.as_slice().to_vec().try_into().unwrap();
        if c.superimposed_orient().is_neg() {
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
      .map(|t| OrientedVertplex::new(t.into(), Sign::Pos))
      .collect();
    CoordManifold::new(cells, node_coords)
  }

  pub fn to_obj_string(&self) -> String {
    let mut string = String::new();
    for v in self.node_coords.column_iter() {
      //writeln!(string, "v {:.6} {:.6} {:.6}", v.x, v.z, -v.y).unwrap();
      writeln!(string, "v {:.6} {:.6} {:.6}", v.x, v.y, v.z).unwrap();
    }
    for t in &self.triangles {
      // .obj uses 1-indexing.
      writeln!(string, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1).unwrap();
    }
    string
  }

  pub fn displace_normal<'a>(&mut self, displacements: impl Into<na::DVectorView<'a, f64>>) {
    let displacements = displacements.into();

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

/// Returns $[r, theta, phi]$ with $r in [0,oo), theta in [0,pi], phi in [0, tau)$
pub fn cartesian2spherical(p: na::Vector3<f64>) -> [f64; 3] {
  let r = p.norm();
  let theta = (p.z / r).acos(); // [0,pi]
  let phi = p.y.atan2(p.x); // [0,tau]
  [r, theta, phi]
}

// x = r sin ⁡ θ cos ⁡ φ , y = r sin ⁡ θ sin ⁡ φ , z = r cos ⁡ θ

/// Takes $(r, theta, phi)$ with $r in [0,oo), theta in [0,pi], phi in [0, tau)$
pub fn spherical2cartesian(r: f64, theta: f64, phi: f64) -> na::Vector3<f64> {
  let x = r * theta.sin() * phi.cos();
  let y = r * theta.sin() * phi.sin();
  let z = r * theta.cos();
  na::Vector3::new(x, y, z)
}

pub fn write_mdd_file(
  filename: &str,
  frames: &Vec<Vec<[f32; 3]>>,
  times: &[f32],
) -> std::io::Result<()> {
  use std::io::Write as _;

  let file = std::fs::File::create(filename)?;
  let mut writer = std::io::BufWriter::new(file);

  let nframes = frames.len() as u32;
  let nvertices = frames[0].len() as u32;

  // header
  writer.write_all(&nframes.to_be_bytes())?;
  writer.write_all(&nvertices.to_be_bytes())?;
  for &time in times {
    writer.write_all(&time.to_be_bytes())?;
  }

  // frame data
  for vertices in frames {
    for &vertex in vertices {
      //let vertex = [vertex[0], vertex[2], -vertex[1]];
      let vertex = [vertex[0], vertex[1], vertex[2]];
      for comp in vertex {
        writer.write_all(&comp.to_be_bytes())?;
      }
    }
  }

  Ok(())
}
/// Geodesic sphere from subdividing a icosahedron
pub fn mesh_sphere_surface(nsubdivisions: usize) -> TriangleSurface3D {
  let triangles = ICOSAHEDRON_SURFACE.triangles().to_vec();
  let node_coords = ICOSAHEDRON_SURFACE
    .node_coords()
    .column_iter()
    .map(|c| c.into_owned())
    .collect();

  let (triangles, node_coords) = subdivide(triangles, node_coords, nsubdivisions);

  let node_coords = na::Matrix3xX::from_columns(&node_coords);

  TriangleSurface3D::new(triangles, node_coords)
}

fn subdivide(
  triangles: Vec<[VertexIdx; 3]>,
  mut node_coords: Vec<na::Vector3<f64>>,
  depth: usize,
) -> (Vec<[VertexIdx; 3]>, Vec<na::Vector3<f64>>) {
  if depth == 0 {
    return (triangles, node_coords);
  }

  let mut midpoints = HashMap::new();

  let triangles = triangles
    .into_iter()
    .flat_map(|[v0, v1, v2]| {
      let v01 = get_midpoint(v0, v1, &mut node_coords, &mut midpoints);
      let v12 = get_midpoint(v1, v2, &mut node_coords, &mut midpoints);
      let v20 = get_midpoint(v2, v0, &mut node_coords, &mut midpoints);

      [
        [v0, v01, v20],
        [v1, v12, v01],
        [v2, v20, v12],
        [v01, v12, v20],
      ]
    })
    .collect();

  subdivide(triangles, node_coords, depth - 1)
}

fn get_midpoint(
  v0: usize,
  v1: usize,
  vertices: &mut Vec<na::Vector3<f64>>,
  midpoints: &mut HashMap<(usize, usize), usize>,
) -> usize {
  let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
  if let Some(&midpoint) = midpoints.get(&edge) {
    return midpoint;
  }

  let midpoint = ((vertices[v0] + vertices[v1]) / 2.0).normalize();
  vertices.push(midpoint);
  let index = vertices.len() - 1;
  midpoints.insert(edge, index);
  index
}

static ICOSAHEDRON_SURFACE: LazyLock<TriangleSurface3D> = LazyLock::new(|| {
  let phi = (1.0 + 5.0f64.sqrt()) / 2.0;

  #[rustfmt::skip]
  let vertices = [
    [-1.0, phi, 0.0],
    [ 1.0, phi, 0.0],
    [-1.0,-phi, 0.0],
    [ 1.0,-phi, 0.0],
    [ 0.0,-1.0, phi],
    [ 0.0, 1.0, phi],
    [ 0.0,-1.0,-phi],
    [ 0.0, 1.0,-phi],
    [ phi, 0.0,-1.0],
    [ phi, 0.0, 1.0],
    [-phi, 0.0,-1.0],
    [-phi, 0.0, 1.0],
  ];

  let vertices: Vec<_> = vertices
    .into_iter()
    .map(|v| na::Vector3::new(v[0], v[1], v[2]).normalize())
    .collect();
  let node_coords = na::Matrix3xX::from_columns(&vertices);

  let triangles = vec![
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
  ];

  TriangleSurface3D::new(triangles, node_coords)
});
