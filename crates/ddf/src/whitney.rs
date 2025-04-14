use crate::LocalMultiForm;

use {
  common::{
    combo::{factorial, factorialf, Sign},
    linalg::nalgebra::Vector,
  },
  exterior::{field::ExteriorField, ExteriorGrade, MultiForm},
  manifold::{
    geometry::coord::{
      simplex::{is_bary_inside, local_to_bary_coords, SimplexCoords},
      AmbientCoordRef, LocalCoordRef,
    },
    topology::simplex::Simplex,
    Dim,
  },
};

pub fn difbary0_local(dim: Dim) -> LocalMultiForm {
  let coeffs = Vector::from_element(dim, -1.0);
  LocalMultiForm::line(coeffs)
}
pub fn difbary_local(ibary: usize, dim: Dim) -> LocalMultiForm {
  let nvertices = dim + 1;
  assert!(ibary < nvertices);
  if ibary == 0 {
    difbary0_local(dim)
  } else {
    let mut coeffs = Vector::zeros(dim);
    coeffs[ibary - 1] = 1.0;
    LocalMultiForm::line(coeffs)
  }
}
pub fn difbarys_local(dim: Dim) -> impl ExactSizeIterator<Item = LocalMultiForm> {
  let nvertices = dim + 1;
  (0..nvertices).map(move |ibary| difbary_local(ibary, dim))
}
pub fn difbary_wedge_local(dim: Dim) -> LocalMultiForm {
  MultiForm::wedge_big(difbarys_local(dim)).unwrap_or(MultiForm::one(dim))
}

pub fn bary0_local<'a>(coord: impl Into<LocalCoordRef<'a>>) -> f64 {
  let coord = coord.into();
  1.0 - coord.sum()
}

#[derive(Debug, Clone)]
pub struct WhitneyRefLsf {
  dim_cell: Dim,
  dof_simp: Simplex,
}
impl WhitneyRefLsf {
  pub fn new(dim_cell: Dim, dof_simp: Simplex) -> Self {
    Self { dim_cell, dof_simp }
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.dof_simp.dim()
  }
  /// The difbarys of the vertices of the DOF simplex.
  pub fn difbarys(&self) -> impl Iterator<Item = LocalMultiForm> + use<'_> {
    self
      .dof_simp
      .clone()
      .into_iter()
      .map(move |ibary| difbary_local(ibary, self.dim_cell))
  }

  /// d𝜆_i_0 ∧⋯∧̂ omit(d𝜆_i_iwedge) ∧⋯∧ d𝜆_i_dim
  pub fn wedge_term(&self, iterm: usize) -> LocalMultiForm {
    let dim_cell = self.dim_cell;
    let wedge = self
      .difbarys()
      .enumerate()
      // leave off i'th difbary
      .filter_map(|(pos, difbary)| (pos != iterm).then_some(difbary));
    MultiForm::wedge_big(wedge).unwrap_or(MultiForm::one(dim_cell))
  }
  pub fn wedge_terms(&self) -> impl ExactSizeIterator<Item = LocalMultiForm> + use<'_> {
    (0..self.dof_simp.nvertices()).map(move |iwedge| self.wedge_term(iwedge))
  }

  /// The constant exterior derivative of the Whitney LSF.
  pub fn dif(&self) -> LocalMultiForm {
    let dim = self.dim_cell;
    let grade = self.grade();
    if grade == dim {
      return MultiForm::zero(dim, grade + 1);
    }
    factorialf(grade + 1) * MultiForm::wedge_big(self.difbarys()).unwrap()
  }
}

impl ExteriorField for WhitneyRefLsf {
  fn dim(&self) -> exterior::Dim {
    self.dim_cell
  }
  fn grade(&self) -> ExteriorGrade {
    self.grade()
  }
  fn at_point<'a>(&self, coord_local: impl Into<LocalCoordRef<'a>>) -> LocalMultiForm {
    let barys = local_to_bary_coords(coord_local);
    assert!(is_bary_inside(&barys), "Point is outside cell.");

    let dim = self.dim_cell;
    let grade = self.grade();
    let mut form = MultiForm::zero(dim, grade);
    for (iterm, &vertex) in self.dof_simp.vertices.iter().enumerate() {
      let sign = Sign::from_parity(iterm);
      let wedge = self.wedge_term(iterm);

      let bary = barys[vertex];
      form += sign.as_f64() * bary * wedge;
    }
    (factorial(grade) as f64) * form
  }
}

pub struct WhitneyCoordLsf {
  pub cell_coords: SimplexCoords,
  pub ref_lsf: WhitneyRefLsf,
}
impl WhitneyCoordLsf {
  pub fn new(cell_coords: SimplexCoords, dof_simp: Simplex) -> Self {
    let ref_lsf = WhitneyRefLsf::new(cell_coords.dim_intrinsic(), dof_simp);
    Self {
      cell_coords,
      ref_lsf,
    }
  }
}
impl ExteriorField for WhitneyCoordLsf {
  fn dim(&self) -> exterior::Dim {
    self.ref_lsf.dim()
  }
  fn grade(&self) -> ExteriorGrade {
    self.ref_lsf.grade()
  }
  /// Pullback!
  fn at_point<'a>(&self, coord_global: impl Into<AmbientCoordRef<'a>>) -> MultiForm {
    let coord_local = self.cell_coords.global2local(coord_global);
    let value_local = self.ref_lsf.at_point(&coord_local);
    value_local.precompose_form(&self.cell_coords.linear_transform())
  }
}
