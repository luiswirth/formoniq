use super::lsf::WhitneyLsf;
use crate::cochain::Cochain;

use exterior::{field::ExteriorField, MultiForm};
use manifold::{
  geometry::coord::{mesh::MeshCoords, simplex::SimplexHandleExt, CoordRef},
  topology::{complex::Complex, handle::SimplexHandle},
};

pub struct WhitneyForm<'a> {
  cochain: Cochain,
  complex: &'a Complex,
  mesh_coords: &'a MeshCoords,
}
impl<'a> WhitneyForm<'a> {
  pub fn new(cochain: Cochain, complex: &'a Complex, mesh_coords: &'a MeshCoords) -> Self {
    Self {
      cochain,
      complex,
      mesh_coords,
    }
  }

  pub fn dif(&self) -> Self {
    Self {
      cochain: self.cochain.dif(self.complex),
      complex: self.complex,
      mesh_coords: self.mesh_coords,
    }
  }
}
impl ExteriorField for WhitneyForm<'_> {
  fn dim_ambient(&self) -> exterior::Dim {
    self.mesh_coords.dim()
  }
  fn dim_intrinsic(&self) -> exterior::Dim {
    self.complex.dim()
  }
  fn grade(&self) -> exterior::ExteriorGrade {
    self.cochain.dim()
  }
  /// Global position
  fn at_point<'a>(&self, coord: impl Into<CoordRef<'a>>) -> exterior::ExteriorElement {
    let coord = coord.into();

    // WARN: This is slow!
    let cell = self
      .mesh_coords
      .find_cell_containing(self.complex, coord)
      .unwrap();
    let cell_coords = cell.coord_simplex(self.mesh_coords);

    let mut value = MultiForm::zero(self.dim_intrinsic(), self.grade());
    for dof_simp in cell.mesh_subsimps(self.grade()) {
      let local_dof_simp = dof_simp.relative_to(&cell);

      let lsf = WhitneyLsf::from_coords(cell_coords.clone(), local_dof_simp);
      let lsf_value = lsf.at_point(coord);
      let dof_value = self.cochain[dof_simp];
      value += dof_value * lsf_value;
    }
    value
  }
}
impl WhitneyForm<'_> {
  pub fn eval_known_cell<'a>(
    &self,
    cell: SimplexHandle,
    coord: impl Into<CoordRef<'a>>,
  ) -> exterior::ExteriorElement {
    let coord = coord.into();

    let cell_coords = cell.coord_simplex(self.mesh_coords);

    let mut value = MultiForm::zero(self.dim_intrinsic(), self.grade());
    for dof_simp in cell.mesh_subsimps(self.grade()) {
      let local_dof_simp = dof_simp.relative_to(&cell);

      let lsf = WhitneyLsf::from_coords(cell_coords.clone(), local_dof_simp);
      let lsf_value = lsf.at_point(coord);
      let dof_value = self.cochain[dof_simp];
      value += dof_value * lsf_value;
    }
    value
  }
}

pub struct DifWhitneyForm<'a> {
  cochain: &'a Cochain,
  complex: &'a Complex,
  mesh_coords: &'a MeshCoords,
}
impl<'a> DifWhitneyForm<'a> {
  pub fn new(cochain: &'a Cochain, complex: &'a Complex, mesh_coords: &'a MeshCoords) -> Self {
    Self {
      cochain,
      complex,
      mesh_coords,
    }
  }
}
impl ExteriorField for DifWhitneyForm<'_> {
  fn dim_ambient(&self) -> exterior::Dim {
    self.mesh_coords.dim()
  }
  fn dim_intrinsic(&self) -> exterior::Dim {
    self.complex.dim()
  }
  fn grade(&self) -> exterior::ExteriorGrade {
    self.cochain.dim()
  }
  /// Global position
  fn at_point<'a>(&self, coord: impl Into<CoordRef<'a>>) -> exterior::ExteriorElement {
    let coord = coord.into();

    // WARN: This is slow!
    let cell = self
      .mesh_coords
      .find_cell_containing(self.complex, coord)
      .unwrap();
    let cell_coords = cell.coord_simplex(self.mesh_coords);

    let mut value = MultiForm::zero(self.dim_intrinsic(), self.grade());
    for dof_simp in cell.mesh_subsimps(self.grade()) {
      let local_dof_simp = dof_simp.relative_to(&cell);

      let lsf = WhitneyLsf::from_coords(cell_coords.clone(), local_dof_simp);
      let dif_lsf_value = lsf.dif();
      let dof_value = self.cochain[dof_simp];
      value += dof_value * dif_lsf_value;
    }
    value
  }
}
