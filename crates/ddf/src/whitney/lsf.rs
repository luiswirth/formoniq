use crate::CoordSimplexExt;

use {
  common::combo::{factorial, factorialf, Combination, Sign},
  exterior::{field::ExteriorField, ExteriorGrade, MultiForm},
  manifold::{
    geometry::coord::{simplex::SimplexCoords, CoordRef},
    Dim,
  },
};

/// The Whitney local shape function $W_sigma$ of a DOF subsimplex:
///
/// $W_sigma = k! sum_i (-1)^i lambda_(sigma_i)
///   dif lambda_(sigma_0) wedge dots.c hat(dif lambda_(sigma_i)) dots.c wedge dif lambda_(sigma_k)$
///
/// built from the alternating deletions of the DOF vertex set: the same
/// boundary pattern as the simplicial $diff$.
#[derive(Debug, Clone)]
pub struct WhitneyLsf {
  cell_coords: SimplexCoords,
  /// The local vertex set of the DOF subsimplex.
  dof_simp: Combination,
}
impl WhitneyLsf {
  pub fn from_coords(cell_coords: SimplexCoords, dof_simp: Combination) -> Self {
    Self {
      cell_coords,
      dof_simp,
    }
  }
  pub fn standard(cell_dim: Dim, dof_simp: Combination) -> Self {
    Self::from_coords(SimplexCoords::standard(cell_dim), dof_simp)
  }
  pub fn grade(&self) -> ExteriorGrade {
    self.dof_simp.card() - 1
  }
  /// The difbarys of the vertices of the DOF simplex.
  pub fn difbarys(&self) -> impl Iterator<Item = MultiForm> + use<'_> {
    self
      .cell_coords
      .difbarys_ext()
      .into_iter()
      .enumerate()
      .filter_map(|(ibary, difbary)| self.dof_simp.contains(ibary).then_some(difbary))
  }

  /// $dif lambda_(sigma_0) wedge dots.c hat(dif lambda_(sigma_iterm)) dots.c wedge dif lambda_(sigma_k)$
  pub fn wedge_term(&self, iterm: usize) -> MultiForm {
    let dim_cell = self.cell_coords.dim_intrinsic();
    let wedge = self
      .difbarys()
      .enumerate()
      // leave off i'th difbary
      .filter_map(|(pos, difbary)| (pos != iterm).then_some(difbary));
    MultiForm::wedge_big(wedge).unwrap_or(MultiForm::one(dim_cell))
  }
  pub fn wedge_terms(&self) -> impl ExactSizeIterator<Item = MultiForm> + use<'_> {
    (0..self.dof_simp.card()).map(move |iwedge| self.wedge_term(iwedge))
  }

  /// The constant exterior derivative of the Whitney LSF,
  /// $dif W_sigma = (k+1)! dif lambda_(sigma_0) wedge dots.c wedge dif lambda_(sigma_k)$.
  pub fn dif(&self) -> MultiForm {
    let dim = self.cell_coords.dim_intrinsic();
    let grade = self.grade();
    if grade == dim {
      return MultiForm::zero(dim, grade + 1);
    }
    factorialf(grade + 1) * MultiForm::wedge_big(self.difbarys()).unwrap()
  }
}

impl ExteriorField for WhitneyLsf {
  fn dim_ambient(&self) -> exterior::Dim {
    self.cell_coords.dim_ambient()
  }
  fn dim_intrinsic(&self) -> exterior::Dim {
    self.cell_coords.dim_intrinsic()
  }
  fn grade(&self) -> ExteriorGrade {
    self.grade()
  }
  fn at_point<'a>(&self, coord: impl Into<CoordRef<'a>>) -> MultiForm {
    let barys = self.cell_coords.global2bary(coord);

    let dim = self.dim_ambient();
    let grade = self.grade();
    let mut form = MultiForm::zero(dim, grade);
    for (iterm, vertex) in self.dof_simp.iter().enumerate() {
      let sign = Sign::from_parity(iterm);
      let wedge = self.wedge_term(iterm);

      let bary = barys[vertex];
      form += sign.as_f64() * bary * wedge;
    }
    (factorial(grade) as f64) * form
  }
}

pub struct WhitneyPushforwardLsf {
  pub cell_coords: SimplexCoords,
  pub ref_lsf: WhitneyLsf,
}
impl WhitneyPushforwardLsf {
  pub fn new(cell_coords: SimplexCoords, dof_simp: Combination) -> Self {
    let ref_lsf = WhitneyLsf::standard(cell_coords.dim_intrinsic(), dof_simp);
    Self {
      cell_coords,
      ref_lsf,
    }
  }
}
impl ExteriorField for WhitneyPushforwardLsf {
  fn dim_ambient(&self) -> exterior::Dim {
    self.cell_coords.dim_ambient()
  }
  fn dim_intrinsic(&self) -> exterior::Dim {
    self.ref_lsf.dim_intrinsic()
  }
  fn grade(&self) -> ExteriorGrade {
    self.ref_lsf.grade()
  }
  fn at_point<'a>(&self, coord_global: impl Into<CoordRef<'a>>) -> MultiForm {
    let coord_ref = self.cell_coords.global2local(coord_global);
    let value_ref = self.ref_lsf.at_point(&coord_ref);
    // Pushforward: pullback along the inverse map.
    value_ref.precompose_form(&self.cell_coords.inv_linear_transform())
  }
}
