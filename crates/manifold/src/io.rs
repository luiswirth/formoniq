pub mod blender;
pub mod gmsh;

use std::{
  fs::File,
  io::{self, BufRead, BufReader, BufWriter},
  path::Path,
};

use crate::{
  geometry::coord::mesh::MeshCoords,
  topology::{complex::Complex, handle::SkeletonHandle, simplex::Simplex, skeleton::Skeleton},
  Dim,
};

pub fn save_skeleton_to_file(
  topology: &Complex,
  dim: Dim,
  path: impl AsRef<Path>,
) -> io::Result<()> {
  let file = File::create(path)?;
  let writer = BufWriter::new(file);
  write_skeleton(writer, topology.skeleton(dim))
}

pub fn write_skeleton<W: io::Write>(mut writer: W, skeleton: SkeletonHandle) -> io::Result<()> {
  for simplex in skeleton.handle_iter() {
    for vertex in simplex.vertices.iter() {
      write!(writer, "{vertex} ")?;
    }
    writeln!(writer)?;
  }
  Ok(())
}

pub fn read_skeleton_from_file(path: impl AsRef<Path>) -> io::Result<Skeleton> {
  let file = File::open(path)?;
  let reader = BufReader::new(file);

  let mut skeleton = Vec::new();
  for line in reader.lines() {
    let line = line?;
    let simplex: Vec<usize> = line
      .split_whitespace()
      .filter_map(|s| s.parse().ok())
      .collect();
    let simplex = Simplex::from(simplex);

    skeleton.push(simplex.sorted());
  }
  Ok(Skeleton::new(skeleton))
}

pub fn save_coords_to_file(coords: &MeshCoords, path: impl AsRef<Path>) -> io::Result<()> {
  let file = File::create(path)?;
  let writer = BufWriter::new(file);
  write_coords(writer, coords)
}

pub fn write_coords<W: io::Write>(mut writer: W, coords: &MeshCoords) -> io::Result<()> {
  for coord in coords.coord_iter() {
    for &comp in coord {
      write!(writer, "{comp:.6} ")?;
    }
    writeln!(writer)?;
  }
  Ok(())
}
