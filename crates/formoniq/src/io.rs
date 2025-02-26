use crate::fe::evaluate_fe_function_cell_vertices;

use {
  exterior::MultiForm,
  manifold::{geometry::coord::MeshVertexCoords, topology::complex::Complex},
  whitney::cochain::Cochain,
};

use std::{fs::File, io::BufWriter, path::Path};

pub fn save_evaluations_to_file(
  fe: &Cochain,
  topology: &Complex,
  coords: &MeshVertexCoords,
  path: impl AsRef<Path>,
) -> std::io::Result<()> {
  let evals = evaluate_fe_function_cell_vertices(fe, topology, coords);
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
