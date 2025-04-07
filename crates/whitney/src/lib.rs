extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod cochain;
pub mod io;

use {
  common::{
    combo::{factorial, Sign},
    sparse::CooMatrixExt,
  },
  exterior::{field::ExteriorField, ExteriorGrade, MultiForm, MultiVector},
  manifold::{
    geometry::coord::{
      local::{is_bary_inside, local_to_bary_coords, SimplexCoords},
      EmbeddingCoordRef, LocalCoordRef,
    },
    topology::{complex::Complex, simplex::Simplex},
    Dim,
  },
};

pub type LocalMultiForm = MultiForm;

pub trait ManifoldComplexExt {
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> nas::CooMatrix<f64>;
}
impl ManifoldComplexExt for Complex {
  /// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> nas::CooMatrix<f64> {
    self.boundary_operator(grade + 1).transpose()
  }
}

pub trait CoordSimplexExt {
  fn spanning_multivector(&self) -> MultiVector;
}
impl CoordSimplexExt for SimplexCoords {
  fn spanning_multivector(&self) -> MultiVector {
    let vectors = self.spanning_vectors();
    let vectors = vectors
      .column_iter()
      .map(|v| MultiVector::line(v.into_owned()));
    MultiVector::wedge_big(vectors).unwrap_or(MultiVector::one(self.dim_embedded()))
  }
}

pub fn difbary0_local(dim: Dim) -> LocalMultiForm {
  let coeffs = na::DVector::from_element(dim, -1.0);
  LocalMultiForm::line(coeffs)
}
pub fn difbary_local(ibary: usize, dim: Dim) -> LocalMultiForm {
  let nvertices = dim + 1;
  assert!(ibary < nvertices);
  if ibary == 0 {
    difbary0_local(dim)
  } else {
    let mut coeffs = na::DVector::zeros(dim);
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
    (factorial(grade + 1) as f64) * MultiForm::wedge_big(self.difbarys()).unwrap()
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
  fn at_point<'a>(&self, coord_global: impl Into<EmbeddingCoordRef<'a>>) -> MultiForm {
    let coord_local = self.cell_coords.global2local(coord_global);
    let value_local = self.ref_lsf.at_point(&coord_local);
    value_local.precompose_form(&self.cell_coords.linear_transform())
  }
}

#[cfg(test)]
mod test {
  use cochain::de_rahm_map_local;

  use super::*;

  use manifold::{
    geometry::coord::{local::SimplexHandleExt, MeshVertexCoords},
    topology::complex::Complex,
  };

  #[test]
  fn whitney_basis_property() {
    for dim in 0..=4 {
      let topology = Complex::standard(dim);
      let coords = MeshVertexCoords::standard(dim);

      for grade in 0..=dim {
        for dof_simp in topology.skeleton(grade).handle_iter() {
          let whitney_form = WhitneyRefLsf::new(dim, dof_simp.raw().clone());

          for other_simp in topology.skeleton(grade).handle_iter() {
            let are_same_simp = dof_simp == other_simp;
            let other_simplex = other_simp.coord_simplex(&coords);
            let discret = de_rahm_map_local(&whitney_form, &other_simplex);
            let expected = Sign::from_bool(are_same_simp).as_f64();
            let diff = (discret - expected).abs();
            const TOL: f64 = 10e-9;
            let equal = diff <= TOL;
            assert!(equal, "for: computed={discret} expected={expected}");
            if other_simplex.nvertices() >= 2 {
              let other_simplex_rev = other_simplex.clone().flipped_orientation();
              let discret_rev = de_rahm_map_local(&whitney_form, &other_simplex_rev);
              let expected_rev = Sign::Neg.as_f64() * are_same_simp as usize as f64;
              let diff_rev = (discret_rev - expected_rev).abs();
              let equal_rev = diff_rev <= TOL;
              assert!(
                equal_rev,
                "rev: computed={discret_rev} expected={expected_rev}"
              );
            }
          }
        }
      }
    }
  }
}
