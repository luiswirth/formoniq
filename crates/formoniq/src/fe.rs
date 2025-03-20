use whitney::WhitneyCoordLsf;

use crate::{
  assemble::assemble_galmat,
  operators::{CodifDifElmat, HodgeMassElmat},
};

use {
  exterior::{field::ExteriorField, MultiForm},
  manifold::{
    geometry::{
      coord::{local::SimplexHandleExt, CoordRef, VertexCoords},
      metric::MeshEdgeLengths,
    },
    topology::complex::Complex,
  },
  whitney::cochain::Cochain,
};

pub fn l2_norm(fe: &Cochain, topology: &Complex, geometry: &MeshEdgeLengths) -> f64 {
  let mass = assemble_galmat(topology, geometry, HodgeMassElmat(fe.dim)).to_nalgebra_csr();
  //fe.coeffs().transpose() * mass * fe.coeffs()
  ((mass.transpose() * fe.coeffs()).transpose() * fe.coeffs())
    .x
    .sqrt()
}

pub fn h1_norm(fe: &Cochain, topology: &Complex, geometry: &MeshEdgeLengths) -> f64 {
  let difdif = assemble_galmat(topology, geometry, CodifDifElmat(fe.dim)).to_nalgebra_csr();
  //fe.coeffs().transpose() * difdif * fe.coeffs()
  ((difdif.transpose() * fe.coeffs()).transpose() * fe.coeffs())
    .x
    .sqrt()
}

pub fn reconstruct_at_coord<'a>(
  coord: impl Into<CoordRef<'a>>,
  cochain: &Cochain,
  topology: &Complex,
  coords: &VertexCoords,
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
  for dof_simp in cell.subsimps(grade) {
    let local_dof_simp = cell
      .simplex_set()
      .global_to_local_subsimp(dof_simp.simplex_set());

    let dof_value = cochain[dof_simp]
      * WhitneyCoordLsf::new(cell.coord_simplex(coords), local_dof_simp).at_point(coord);
    fe_value += dof_value;
  }

  fe_value
}

pub fn reconstruct_at_mesh_cells_barycenters(
  cochain: &Cochain,
  topology: &Complex,
  coords: &VertexCoords,
) -> Vec<MultiForm> {
  let grade = cochain.dim;

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

        let dof_value = cochain[dof_simp]
          * WhitneyCoordLsf::new(cell.coord_simplex(coords), local_dof_simp).at_point(&barycenter);
        value += dof_value;
      }
      value
    })
    .collect()
}

pub fn reconstruct_at_mesh_cells_vertices(
  cochain: &Cochain,
  topology: &Complex,
  coords: &VertexCoords,
) -> Vec<Vec<MultiForm>> {
  let grade = cochain.dim();

  topology
    .cells()
    .handle_iter()
    .map(|cell| {
      cell
        .vertices()
        .map(|vertex| {
          let coord = coords.coord(vertex.kidx());

          // vertex value
          cell
            .subsimps(grade)
            .map(|dof_simp| {
              let coeff = cochain[dof_simp];

              let local_dof_simp = cell
                .simplex_set()
                .global_to_local_subsimp(dof_simp.simplex_set());
              //dbg!(cell.simplex_set().vertices.indices());
              //dbg!(dof_simp.simplex_set().vertices.indices());
              //dbg!(&local_dof_simp.vertices.indices());
              let whitney = WhitneyCoordLsf::new(cell.coord_simplex(coords), local_dof_simp);

              // dof_value
              coeff * whitney.at_point(coord)
            })
            .sum()
        })
        .collect()
    })
    .collect()
}
