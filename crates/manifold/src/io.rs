pub mod blender;
pub mod gmsh;
pub mod vtk;

use std::{fs::File, io::BufWriter, path::Path};

use crate::{
  geometry::coord::MeshVertexCoords,
  topology::{complex::Complex, simplex::Simplex},
};

use multi_index::variants::IndexKind;

pub fn save_cells_to_file(topology: &Complex, path: impl AsRef<Path>) -> std::io::Result<()> {
  let file = File::create(path)?;
  let writer = BufWriter::new(file);
  write_simplicies(writer, topology.cells().set_iter())
}

pub fn write_simplicies<'a, W: std::io::Write, O: IndexKind + 'a>(
  mut writer: W,
  simplices: impl Iterator<Item = &'a Simplex<O>>,
) -> std::io::Result<()> {
  for simplex in simplices {
    for vertex in simplex.vertices.iter() {
      write!(writer, "{vertex} ")?;
    }
    writeln!(writer)?;
  }
  Ok(())
}

pub fn save_coords_to_file(
  coords: &MeshVertexCoords,
  path: impl AsRef<Path>,
) -> std::io::Result<()> {
  let file = File::create(path)?;
  let writer = BufWriter::new(file);
  write_coords(writer, coords)
}

pub fn write_coords<W: std::io::Write>(
  mut writer: W,
  coords: &MeshVertexCoords,
) -> std::io::Result<()> {
  for coord in coords.coord_iter() {
    for &comp in coord {
      write!(writer, "{comp:.6} ")?;
    }
    writeln!(writer)?;
  }
  Ok(())
}
