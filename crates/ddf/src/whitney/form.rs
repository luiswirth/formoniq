use super::lsf::WhitneyLsf;
use crate::cochain::Cochain;

use exterior::{field::ExteriorField, MultiForm};
use manifold::{
  geometry::coord::{locate::PointLocator, mesh::MeshCoords, simplex::SimplexRefExt, CoordRef},
  topology::{complex::Complex, handle::SimplexRef},
};

/// The Whitney interpolation $W c = sum_sigma c_sigma W_sigma$ of a cochain:
/// a piecewise linear differential form.
///
/// This is the FEEC representation formula: it reconstructs a genuine
/// differential form from a cochain, evaluable at any point. Attaching a
/// [`PointLocator`] with [`with_locator`](Self::with_locator) makes pointwise
/// evaluation logarithmic instead of a linear scan over all cells -- essential
/// for sampling the field on a grid.
pub struct WhitneyForm<'a> {
  cochain: Cochain,
  complex: &'a Complex,
  mesh_coords: &'a MeshCoords,
  locator: Option<&'a PointLocator>,
}
impl<'a> WhitneyForm<'a> {
  pub fn new(cochain: Cochain, complex: &'a Complex, mesh_coords: &'a MeshCoords) -> Self {
    Self {
      cochain,
      complex,
      mesh_coords,
      locator: None,
    }
  }

  /// Attach a prebuilt point locator to accelerate [`at_point`](ExteriorField::at_point).
  pub fn with_locator(mut self, locator: &'a PointLocator) -> Self {
    self.locator = Some(locator);
    self
  }

  /// The exterior derivative: since $W$ is a cochain map, this is the
  /// interpolation of the coboundary, $dif (W c) = W (dif c)$.
  pub fn dif(&self) -> Self {
    Self {
      cochain: self.cochain.dif(self.complex),
      complex: self.complex,
      mesh_coords: self.mesh_coords,
      locator: self.locator,
    }
  }

  pub fn eval_known_cell<'b>(
    &self,
    cell: SimplexRef,
    coord: impl Into<CoordRef<'b>>,
  ) -> MultiForm {
    let coord = coord.into();

    let cell_coords = cell.coord_simplex(self.mesh_coords);

    let mut value = MultiForm::zero(self.dim_ambient(), self.grade());
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

impl ExteriorField for WhitneyForm<'_> {
  fn dim_ambient(&self) -> exterior::Dim {
    self.mesh_coords.dim()
  }
  fn dim_intrinsic(&self) -> exterior::Dim {
    self.complex.dim()
  }
  fn grade(&self) -> exterior::ExteriorGrade {
    self.cochain.grade()
  }
  /// Evaluate the reconstructed form at a global point.
  ///
  /// With an attached [`PointLocator`] the containing cell is found in
  /// $O(log N)$; otherwise this falls back to a linear scan over all cells.
  fn at_point<'a>(&self, coord: impl Into<CoordRef<'a>>) -> MultiForm {
    let coord = coord.into();

    let cell = match self.locator {
      Some(locator) => locator.locate(coord).unwrap().cell.handle(self.complex),
      None => self
        .mesh_coords
        .find_cell_containing(self.complex, coord)
        .unwrap(),
    };
    self.eval_known_cell(cell, coord)
  }
}

#[cfg(test)]
mod test {
  use super::*;

  use common::linalg::nalgebra::Vector;
  use exterior::field::DiffFormClosure;
  use manifold::gen::cartesian::CartesianMeshInfo;

  use crate::derham::derham_map;

  /// The locator-accelerated `at_point` agrees exactly with the linear-scan
  /// fallback: same cell, same reconstructed form value.
  #[test]
  fn located_eval_matches_scan() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 3).compute_coord_complex();
      let locator = PointLocator::new(&topology, &coords);

      // A reconstructed 1-form from the de Rham projection of a linear field.
      let field = DiffFormClosure::one_form(|p| p.clone_owned(), dim);
      let cochain = derham_map(&field, &topology, &coords, 2);

      let scan = WhitneyForm::new(cochain.clone(), &topology, &coords);
      let fast = WhitneyForm::new(cochain, &topology, &coords).with_locator(&locator);

      let samples: usize = 4;
      let phase = [0.017, 0.113, 0.237];
      for flat in 0..samples.pow(dim as u32) {
        let x = Vector::from_iterator(
          dim,
          (0..dim).map(|d| {
            let i = flat / samples.pow(d as u32) % samples;
            0.05 + 0.9 * (i as f64 + phase[d]) / samples as f64
          }),
        );
        let a = scan.at_point(x.as_view());
        let b = fast.at_point(x.as_view());
        assert!(
          (a.coeffs() - b.coeffs()).norm() < 1e-12,
          "dim={dim} x={x:?}"
        );
      }
    }
  }
}
