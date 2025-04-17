use crate::{
  assemble::assemble_galmat,
  operators::{CodifDifElmat, HodgeMassElmat},
};

use {
  common::linalg::nalgebra::CsrMatrix,
  ddf::{cochain::Cochain, whitney::WhitneyPushforwardLsf},
  exterior::{field::ExteriorField, MultiForm},
  manifold::{
    geometry::{
      coord::{mesh::MeshCoords, simplex::SimplexHandleExt, CoordRef},
      metric::mesh::MeshLengths,
    },
    topology::complex::Complex,
  },
};

pub fn l2_norm(fe: &Cochain, topology: &Complex, geometry: &MeshLengths) -> f64 {
  let mass = assemble_galmat(
    topology,
    geometry,
    HodgeMassElmat::new(topology.dim(), fe.dim()),
  );
  let mass = CsrMatrix::from(&mass);
  //fe.coeffs().transpose() * mass * fe.coeffs()
  ((mass.transpose() * fe.coeffs()).transpose() * fe.coeffs())
    .x
    .sqrt()
}

pub fn h1_norm(fe: &Cochain, topology: &Complex, geometry: &MeshLengths) -> f64 {
  let codifdif = assemble_galmat(
    topology,
    geometry,
    CodifDifElmat::new(topology.dim(), fe.dim),
  );
  let codifdif = CsrMatrix::from(&codifdif);
  //fe.coeffs().transpose() * difdif * fe.coeffs()
  ((codifdif.transpose() * fe.coeffs()).transpose() * fe.coeffs())
    .x
    .sqrt()
}

pub fn reconstruct_at_coord<'a>(
  coord: impl Into<CoordRef<'a>>,
  cochain: &Cochain,
  topology: &Complex,
  coords: &MeshCoords,
) -> MultiForm {
  let coord = coord.into();

  let dim = coords.dim();
  assert_eq!(coord.len(), dim);
  let grade = cochain.dim;

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
  for dof_simp in cell.mesh_subsimps(grade) {
    let local_dof_simp = dof_simp.relative_to(&cell);

    let dof_value = cochain[dof_simp]
      * WhitneyPushforwardLsf::new(cell.coord_simplex(coords), local_dof_simp).at_point(coord);
    fe_value += dof_value;
  }

  fe_value
}

pub fn reconstruct_at_mesh_cells_barycenters(
  cochain: &Cochain,
  topology: &Complex,
  coords: &MeshCoords,
) -> Vec<MultiForm> {
  let grade = cochain.dim;

  topology
    .cells()
    .handle_iter()
    .map(|cell| {
      let mut value = MultiForm::zero(topology.dim(), grade);
      for dof_simp in cell.mesh_subsimps(grade) {
        let local_dof_simp = dof_simp.relative_to(&cell);

        let barycenter = cell.coord_simplex(coords).barycenter();

        let dof_value = cochain[dof_simp]
          * WhitneyPushforwardLsf::new(cell.coord_simplex(coords), local_dof_simp)
            .at_point(&barycenter);
        value += dof_value;
      }
      value
    })
    .collect()
}

pub fn reconstruct_at_mesh_cells_vertices(
  cochain: &Cochain,
  topology: &Complex,
  coords: &MeshCoords,
) -> Vec<Vec<MultiForm>> {
  let grade = cochain.dim();

  topology
    .cells()
    .handle_iter()
    .map(|cell| {
      cell
        .mesh_vertices()
        .map(|vertex| {
          let coord = coords.coord(vertex.kidx());

          // vertex value
          cell
            .mesh_subsimps(grade)
            .map(|dof_simp| {
              let coeff = cochain[dof_simp];

              let local_dof_simp = dof_simp.relative_to(&cell);
              let whitney = WhitneyPushforwardLsf::new(cell.coord_simplex(coords), local_dof_simp);

              // dof_value
              coeff * whitney.at_point(coord)
            })
            .sum()
        })
        .collect()
    })
    .collect()
}
