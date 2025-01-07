use super::CoordSkeleton;
use crate::coord::VertexCoords;

use topology::{simplex::Simplex, skeleton::ManifoldSkeleton};

use std::{collections::HashMap, fmt::Write, path::Path, sync::LazyLock};

pub type TriangleTopology = Vec<[usize; 3]>;
pub trait TriangleTopologyExt {
  fn try_from_skeleton(manifold: ManifoldSkeleton) -> Self;
}
impl TriangleTopologyExt for TriangleTopology {
  fn try_from_skeleton(skeleton: ManifoldSkeleton) -> Self {
    skeleton
      .into_simplex_iter()
      .map(|simp| simp.try_into().unwrap())
      .collect()
  }
}

pub type VertexCoords3d = VertexCoords<na::U3>;

#[derive(Debug, Clone)]
pub struct TriangleSurface3D {
  triangles: TriangleTopology,
  coords: VertexCoords3d,
}
impl TriangleSurface3D {
  pub fn new(triangles: TriangleTopology, coords: impl Into<VertexCoords3d>) -> Self {
    let coords = coords.into();
    Self { triangles, coords }
  }
  pub fn triangles(&self) -> &[[usize; 3]] {
    &self.triangles
  }
  pub fn vertex_coords(&self) -> &VertexCoords3d {
    &self.coords
  }
  pub fn vertex_coords_mut(&mut self) -> &mut VertexCoords3d {
    &mut self.coords
  }
}

impl TriangleSurface3D {
  pub fn from_coord_skeleton(skeleton: CoordSkeleton) -> Self {
    assert!(
      skeleton.dim_embedded() == 3,
      "Skeleton is not embedded in 3D."
    );
    assert!(
      skeleton.dim_intrinsic() == 2,
      "Skeleton is not a 2D surface."
    );
    let (skeleton, coords) = skeleton.into_parts();
    let triangles = TriangleTopology::try_from_skeleton(skeleton);
    let coords = coords.into_const_dim();

    Self::new(triangles, coords)
  }

  pub fn into_coord_skeleton(self) -> CoordSkeleton {
    let simps = self
      .triangles
      .into_iter()
      .map(|tria| Simplex::from(tria).into_sorted())
      .collect();
    let skeleton = ManifoldSkeleton::new(simps);
    let coords = self.coords.into_dyn_dim();
    CoordSkeleton::new(skeleton, coords)
  }

  pub fn to_obj_string(&self) -> String {
    let mut string = String::new();
    for v in self.coords.coord_iter() {
      writeln!(string, "v {:.6} {:.6} {:.6}", v.x, v.y, v.z).unwrap();
    }
    for t in &self.triangles {
      // .obj uses 1-indexing.
      writeln!(string, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1).unwrap();
    }
    string
  }

  pub fn from_obj_string(obj_string: &str) -> Self {
    let mut vertex_coords = Vec::new();
    let mut triangles = Vec::new();

    for line in obj_string.lines() {
      let line = line.trim();

      if let Some(coords) = line.strip_prefix("v ") {
        let coords: Vec<f64> = coords
          .split_whitespace()
          .map(|x| x.parse::<f64>().unwrap())
          .collect();
        assert!(coords.len() == 3);
        vertex_coords.push(na::Vector3::new(coords[0], coords[1], coords[2]));
      } else if let Some(indices) = line.strip_prefix("f ") {
        let indices: Vec<usize> = indices
          .split_whitespace()
          // .obj uses 1-indexing.
          .map(|x| x.parse::<usize>().unwrap() - 1)
          .collect();
        assert!(indices.len() == 3);
        triangles.push([indices[0], indices[1], indices[2]]);
      }
    }

    let vertex_coords = na::Matrix3xX::from_columns(&vertex_coords);
    let vertex_coords = VertexCoords::from(vertex_coords);
    Self::new(triangles, vertex_coords)
  }

  pub fn displace_normal<'a>(&mut self, displacements: impl Into<na::DVectorView<'a, f64>>) {
    let displacements = displacements.into();

    let mut vertex_normals = vec![na::Vector3::zeros(); self.coords.nvertices()];
    let mut vertex_triangle_counts = vec![0; self.coords.nvertices()];
    for ivs in &self.triangles {
      let vs = ivs.map(|i| self.coords.coord(i));
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
      .coords
      .coord_iter_mut()
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

/// Takes $(r, theta, phi)$ with $r in [0,oo), theta in [0,pi], phi in [0, tau)$
pub fn spherical2cartesian(r: f64, theta: f64, phi: f64) -> na::Vector3<f64> {
  let x = r * theta.sin() * phi.cos();
  let y = r * theta.sin() * phi.sin();
  let z = r * theta.cos();
  na::Vector3::new(x, y, z)
}

pub fn write_mdd_file(
  filename: impl AsRef<Path>,
  frames: &[Vec<[f32; 3]>],
  times: &[f32],
) -> std::io::Result<()> {
  use std::io::Write as _;

  let file = std::fs::File::create(filename)?;
  let mut writer = std::io::BufWriter::new(file);

  // header
  let nframes = frames.len() as u32;
  writer.write_all(&nframes.to_be_bytes())?;
  let nvertices = frames[0].len() as u32;
  writer.write_all(&nvertices.to_be_bytes())?;

  for &time in times {
    writer.write_all(&time.to_be_bytes())?;
  }
  for vertices in frames {
    for vertex in vertices {
      for comp in vertex {
        writer.write_all(&comp.to_be_bytes())?;
      }
    }
  }

  Ok(())
}

pub fn write_3dmesh_animation<'a, 'b>(
  coords_frames: impl IntoIterator<Item = &'a VertexCoords3d>,
  time_frames: impl IntoIterator<Item = f64>,
) where
  VertexCoords3d: std::fmt::Debug,
{
  let mdd_frames: Vec<Vec<[f32; 3]>> = coords_frames
    .into_iter()
    .map(|coords| {
      coords
        .coord_iter()
        .map(|col| [col.x as f32, col.y as f32, col.z as f32])
        .collect()
    })
    .collect();
  let time_frames: Vec<f32> = time_frames.into_iter().map(|t| t as f32).collect();

  write_mdd_file("out/sphere_wave.mdd", &mdd_frames, &time_frames).unwrap()
}

pub fn write_displacement_animation<'a>(
  base_surface: &TriangleSurface3D,
  displacements_frames: impl IntoIterator<Item = &'a na::DVector<f64>>,
  frame_times: impl IntoIterator<Item = f64>,
) {
  let coords_frames: Vec<_> = displacements_frames
    .into_iter()
    .map(|displacement| {
      let mut surface = base_surface.clone();
      surface.displace_normal(displacement);
      surface.coords
    })
    .collect();

  write_3dmesh_animation(&coords_frames, frame_times);
}

/// Geodesic sphere from subdividing a icosahedron
pub fn mesh_sphere_surface(nsubdivisions: usize) -> TriangleSurface3D {
  let triangles = ICOSAHEDRON_SURFACE.triangles().to_vec();
  let vertex_coords = ICOSAHEDRON_SURFACE
    .vertex_coords()
    .coord_iter()
    .map(|c| c.into_owned())
    .collect();

  let (triangles, vertex_coords) = subdivide(triangles, vertex_coords, nsubdivisions);

  TriangleSurface3D::new(triangles, vertex_coords.as_slice())
}

fn subdivide(
  triangles: Vec<[usize; 3]>,
  mut vertex_coords: Vec<na::Vector3<f64>>,
  depth: usize,
) -> (Vec<[usize; 3]>, Vec<na::Vector3<f64>>) {
  if depth == 0 {
    return (triangles, vertex_coords);
  }

  let mut midpoints = HashMap::new();

  let triangles = triangles
    .into_iter()
    .flat_map(|[v0, v1, v2]| {
      let v01 = get_midpoint(v0, v1, &mut vertex_coords, &mut midpoints);
      let v12 = get_midpoint(v1, v2, &mut vertex_coords, &mut midpoints);
      let v20 = get_midpoint(v2, v0, &mut vertex_coords, &mut midpoints);

      [
        [v0, v01, v20],
        [v1, v12, v01],
        [v2, v20, v12],
        [v01, v12, v20],
      ]
    })
    .collect();

  subdivide(triangles, vertex_coords, depth - 1)
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
  let vertex_coords = [
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

  let vertex_coords: Vec<_> = vertex_coords
    .into_iter()
    .map(|v| na::Vector3::new(v[0], v[1], v[2]).normalize())
    .collect();

  #[rustfmt::skip]
  let triangles = vec![
    [ 0,11, 5],
    [ 0, 5, 1],
    [ 0, 1, 7],
    [ 0, 7,10],
    [ 0,10,11],
    [ 1, 5, 9],
    [ 5,11, 4],
    [11,10, 2],
    [10, 7, 6],
    [ 7, 1, 8],
    [ 3, 9, 4],
    [ 3, 4, 2],
    [ 3, 2, 6],
    [ 3, 6, 8],
    [ 3, 8, 9],
    [ 4, 9, 5],
    [ 2, 4,11],
    [ 6, 2,10],
    [ 8, 6, 7],
    [ 9, 8, 1],
  ];

  TriangleSurface3D::new(triangles, vertex_coords.as_slice())
});
