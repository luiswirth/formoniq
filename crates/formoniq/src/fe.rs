use crate::{
  assemble::assemble_galmat,
  operators::{CodifDifElmat, FeFunction, HodgeMassElmat},
  whitney::WhitneyForm,
};

use exterior::dense::{ExteriorField, MultiForm};
use geometry::{
  coord::{manifold::SimplexHandleExt, CoordRef, MeshVertexCoords},
  metric::MeshEdgeLengths,
};
use topology::complex::{dim::RelDimTrait, TopologyComplex};

pub fn l2_norm(fe: &FeFunction, topology: &TopologyComplex, geometry: &MeshEdgeLengths) -> f64 {
  let mass = assemble_galmat(topology, geometry, HodgeMassElmat(fe.dim)).to_nalgebra_csr();
  //fe.coeffs().transpose() * mass * fe.coeffs()
  ((mass.transpose() * fe.coeffs()).transpose() * fe.coeffs())
    .x
    .sqrt()
}

pub fn h1_norm(fe: &FeFunction, topology: &TopologyComplex, geometry: &MeshEdgeLengths) -> f64 {
  let difdif = assemble_galmat(topology, geometry, CodifDifElmat(fe.dim)).to_nalgebra_csr();
  //fe.coeffs().transpose() * difdif * fe.coeffs()
  ((difdif.transpose() * fe.coeffs()).transpose() * fe.coeffs())
    .x
    .sqrt()
}

pub fn evaluate_fe_function_at_coord<'a>(
  coord: impl Into<CoordRef<'a>>,
  fe: &FeFunction,
  topology: &TopologyComplex,
  coords: &MeshVertexCoords,
) -> MultiForm {
  let coord = coord.into();

  let dim = coords.dim();
  assert_eq!(coord.len(), dim);
  let grade = fe.dim.dim(topology.dim());

  // Find cell that contains coord.
  // WARN: very slow and inefficent
  let Some(cell) = topology
    .cells()
    .handle_iter()
    .find(|cell| cell.coord_simplex(coords).is_coord_inside(coord))
  else {
    return MultiForm::zero(dim, grade);
  };

  let mut fe_value = MultiForm::zero(topology.dim(), grade);
  for dof_simp in cell.subsimps(grade) {
    let local_dof_simp = cell
      .simplex_set()
      .global_to_local_subsimp(dof_simp.simplex_set());

    let dof_value =
      fe[dof_simp] * WhitneyForm::new(cell.coord_simplex(coords), local_dof_simp).at_point(coord);
    fe_value += dof_value;
  }

  fe_value
}

pub fn evaluate_fe_function_at_cell_barycenters(
  fe: &FeFunction,
  topology: &TopologyComplex,
  coords: &MeshVertexCoords,
) -> Vec<MultiForm> {
  let grade = fe.dim.dim(topology.dim());

  topology
    .cells()
    .handle_iter()
    .map(|cell| {
      let mut value = MultiForm::zero(topology.dim(), grade);
      for dof_simp in cell.subsimps(grade) {
        let local_dof_simp = cell
          .simplex_set()
          .global_to_local_subsimp(dof_simp.simplex_set());

        let barycenter = cell.coord_simplex(coords).barycenter();

        let dof_value = fe[dof_simp]
          * WhitneyForm::new(cell.coord_simplex(coords), local_dof_simp).at_point(&barycenter);
        value += dof_value;
      }
      value
    })
    .collect()
}

pub fn evaluate_fe_function_cell_vertices(
  fe: &FeFunction,
  topology: &TopologyComplex,
  coords: &MeshVertexCoords,
) -> Vec<Vec<MultiForm>> {
  let grade = fe.dim.dim(topology.dim());

  topology
    .cells()
    .handle_iter()
    .map(|cell| {
      cell
        .vertices()
        .map(|vertex| {
          let mut value = MultiForm::zero(topology.dim(), grade);
          for dof_simp in cell.subsimps(grade) {
            let local_dof_simp = cell
              .simplex_set()
              .global_to_local_subsimp(dof_simp.simplex_set());

            let dof_value = fe[dof_simp]
              * WhitneyForm::new(cell.coord_simplex(coords), local_dof_simp)
                .at_point(&coords.coord(vertex.kidx()));
            value += dof_value;
          }
          value
        })
        .collect()
    })
    .collect()
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
