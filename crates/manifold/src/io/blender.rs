use crate::topology::complex::Complex;
use crate::{dim3::TriangleSurface3D, geometry::coord::MeshVertexCoords};

use std::fmt::Write;
use std::path::Path;

pub fn coord_complex2obj(complex: &Complex, coords: &MeshVertexCoords) -> String {
  let surface = TriangleSurface3D::from_coord_skeleton(complex.cells().clone(), coords.clone());
  to_obj_string(&surface)
}

pub fn to_obj_string(surface: &TriangleSurface3D) -> String {
  let mut string = String::new();
  for v in surface.vertex_coords().coord_iter() {
    writeln!(string, "v {:.6} {:.6} {:.6}", v[0], v[1], v[2]).unwrap();
  }
  for t in surface.triangles() {
    // .obj uses 1-indexing.
    writeln!(string, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1).unwrap();
  }
  string
}

pub fn obj2coord_complex(obj_string: &str) -> (Complex, MeshVertexCoords) {
  let surface = from_obj_string(obj_string);
  surface.into_coord_complex()
}

pub fn from_obj_string(obj_string: &str) -> TriangleSurface3D {
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
      vertex_coords.push(na::dvector![coords[0], coords[1], coords[2]]);
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

  let vertex_coords = na::DMatrix::from_columns(&vertex_coords);
  let vertex_coords = MeshVertexCoords::from(vertex_coords);
  TriangleSurface3D::new(triangles, vertex_coords)
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
  coords_frames: impl IntoIterator<Item = &'a MeshVertexCoords>,
  time_frames: impl IntoIterator<Item = f64>,
) {
  let mdd_frames: Vec<Vec<[f32; 3]>> = coords_frames
    .into_iter()
    .map(|coords| {
      coords
        .coord_iter()
        .map(|col| [col[0] as f32, col[1] as f32, col[2] as f32])
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
      let (_, coords) = surface.into_parts();
      coords
    })
    .collect();

  write_3dmesh_animation(&coords_frames, frame_times);
}
