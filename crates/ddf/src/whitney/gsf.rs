use super::WhitneyLsf;
use crate::cochain::Cochain;

use exterior::{field::ExteriorField, MultiForm};
use manifold::{
  geometry::coord::{mesh::MeshCoords, simplex::SimplexHandleExt, CoordRef},
  topology::{complex::Complex, handle::SimplexHandle},
};

use std::{fs::File, io::BufWriter, path::Path};

pub fn whitney_form_eval<'a>(
  coord: impl Into<CoordRef<'a>>,
  cochain: &Cochain,
  mesh_cell: SimplexHandle,
  mesh_coords: &MeshCoords,
) -> MultiForm {
  let coord = coord.into();

  let cell_coords = mesh_cell.coord_simplex(mesh_coords);

  let dim_intrinsic = mesh_cell.dim();
  let grade = cochain.dim;

  let mut value = MultiForm::zero(dim_intrinsic, grade);
  for dof_simp in mesh_cell.mesh_subsimps(grade) {
    let local_dof_simp = dof_simp.relative_to(&mesh_cell);

    let lsf = WhitneyLsf::from_coords(cell_coords.clone(), local_dof_simp);
    let lsf_value = lsf.at_point(coord);
    let dof_value = cochain[dof_simp];
    value += dof_value * lsf_value;
  }
  value
}

pub fn whitney_form_eval_all_vertices(
  cochain: &Cochain,
  topology: &Complex,
  mesh_coords: &MeshCoords,
) -> Vec<Vec<MultiForm>> {
  topology
    .cells()
    .handle_iter()
    .map(|cell| {
      let cell_coords = cell.coord_simplex(mesh_coords);
      cell_coords
        .vertices
        .coord_iter()
        .map(|vertex_coord| whitney_form_eval(vertex_coord, cochain, cell, mesh_coords))
        .collect()
    })
    .collect()
}

pub fn save_evaluations_to_file(
  fe: &Cochain,
  topology: &Complex,
  coords: &MeshCoords,
  path: impl AsRef<Path>,
) -> std::io::Result<()> {
  let evals = whitney_form_eval_all_vertices(fe, topology, coords);
  let file = File::create(path)?;
  let writer = BufWriter::new(file);
  write_evaluations(writer, &evals)
}

pub fn write_evaluations<W: std::io::Write>(
  mut writer: W,
  cell_evals: &[Vec<MultiForm>],
) -> std::io::Result<()> {
  for cell_eval in cell_evals {
    writeln!(writer, "cell")?;
    for vertex_eval in cell_eval {
      for comp in vertex_eval.coeffs() {
        write!(writer, "{comp:.6} ")?;
      }
      writeln!(writer)?;
    }
    writeln!(writer)?;
  }
  Ok(())
}
