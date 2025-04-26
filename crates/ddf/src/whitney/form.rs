use super::lsf::WhitneyLsf;
use crate::cochain::Cochain;

use exterior::{field::ExteriorField, MultiForm};
use manifold::{
  geometry::coord::{mesh::MeshCoords, simplex::SimplexHandleExt, CoordRef},
  topology::handle::SimplexHandle,
};

pub struct WhitneyForm<'a> {
  cochain: &'a Cochain,
  mesh_cell: SimplexHandle<'a>,
  mesh_coords: &'a MeshCoords,
}
impl ExteriorField for WhitneyForm<'_> {
  fn dim_ambient(&self) -> exterior::Dim {
    self.mesh_coords.dim()
  }
  fn dim_intrinsic(&self) -> exterior::Dim {
    self.mesh_cell.dim()
  }
  fn grade(&self) -> exterior::ExteriorGrade {
    self.cochain.dim()
  }
  fn at_point<'a>(&self, coord: impl Into<CoordRef<'a>>) -> exterior::ExteriorElement {
    let coord = coord.into();

    let cell_coords = self.mesh_cell.coord_simplex(self.mesh_coords);

    let mut value = MultiForm::zero(self.dim_intrinsic(), self.grade());
    for dof_simp in self.mesh_cell.mesh_subsimps(self.grade()) {
      let local_dof_simp = dof_simp.relative_to(&self.mesh_cell);

      let lsf = WhitneyLsf::from_coords(cell_coords.clone(), local_dof_simp);
      let lsf_value = lsf.at_point(coord);
      let dof_value = self.cochain[dof_simp];
      value += dof_value * lsf_value;
    }
    value
  }
}
