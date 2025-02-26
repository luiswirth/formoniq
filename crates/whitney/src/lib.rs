extern crate nalgebra as na;

pub mod cochain;

use crate::cochain::Cochain;

use {
  common::sparse::SparseMatrix,
  exterior::{
    field::{DifferentialMultiForm, ExteriorField},
    variance, ExteriorElement, ExteriorGrade, MultiForm, MultiFormList, MultiVector,
  },
  manifold::{
    geometry::coord::{
      local::SimplexCoords, quadrature::barycentric_quadrature, CoordRef, MeshVertexCoords,
    },
    topology::{complex::Complex, simplex::Simplex},
    Dim,
  },
  multi_index::{factorial, sign::Sign, variants::SetOrder},
};

pub trait ManifoldComplexExt {
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> SparseMatrix;
}
impl ManifoldComplexExt for Complex {
  /// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> SparseMatrix {
    self.boundary_operator(grade + 1).transpose()
  }
}

pub trait CoordSimplexExt {
  fn difbary(&self, i: usize) -> MultiForm;
  fn difbarys(&self) -> Vec<MultiForm>;
  fn spanning_multivector(&self) -> MultiVector;
}
impl CoordSimplexExt for SimplexCoords {
  fn spanning_multivector(&self) -> MultiVector {
    let vectors = self.spanning_vectors();
    let vectors = vectors
      .column_iter()
      .map(|v| MultiVector::from_grade1(v.into_owned()));
    MultiVector::wedge_big(vectors).unwrap_or(MultiVector::one(self.dim_embedded()))
  }

  fn difbary(&self, i: usize) -> MultiForm {
    let gradbary = self.gradbary(i);
    MultiForm::from_grade1(gradbary)
  }

  fn difbarys(&self) -> Vec<MultiForm> {
    let gradbarys = self.gradbarys();
    gradbarys
      .column_iter()
      .map(|g| MultiForm::from_grade1(g.into_owned()))
      .collect()
  }
}

/// Discretize continuous coordinate-based differential k-form into
/// discrete k-cochain on CoordComplex via de Rham map (integration over k-simplex).
pub fn discretize_form_on_mesh(
  form: &impl DifferentialMultiForm,
  topology: &Complex,
  coords: &MeshVertexCoords,
) -> Cochain {
  let cochain = topology
    .skeleton(form.grade())
    .handle_iter()
    .map(|simp| SimplexCoords::from_simplex_and_coords(simp.simplex_set(), coords))
    .map(|simp| discretize_form_on_simplex(form, &simp))
    .collect::<Vec<_>>()
    .into();
  Cochain::new(form.grade(), cochain)
}

/// Approximates the integral of a differential k-form over a k-simplex,
/// by means of barycentric quadrature.
pub fn discretize_form_on_simplex(
  differential_form: &impl DifferentialMultiForm,
  simplex: &SimplexCoords,
) -> f64 {
  let multivector = simplex.spanning_multivector();
  let f = |coord: CoordRef| {
    differential_form
      .at_point(simplex.local_to_global_coord(coord).as_view())
      .on_multivector(&multivector)
  };
  let std_simp = SimplexCoords::standard(simplex.dim_intrinsic());
  barycentric_quadrature(&f, &std_simp)
}

/// Whitney Form on a coordinate complex.
///
/// Can be evaluated on local coordinates.
pub struct WhitneyForm<O: SetOrder> {
  cell_coords: SimplexCoords,
  associated_subsimp: Simplex<O>,
  difbarys: Vec<MultiForm>,
}
impl<O: SetOrder> WhitneyForm<O> {
  pub fn new(cell_coords: SimplexCoords, associated_subsimp: Simplex<O>) -> Self {
    let difbarys = associated_subsimp
      .vertices
      .iter()
      .map(|vertex| cell_coords.difbary(vertex))
      .collect();

    Self {
      cell_coords,
      associated_subsimp,
      difbarys,
    }
  }

  pub fn wedge_term(&self, iterm: usize) -> MultiForm {
    let wedge_terms = self
      .difbarys
      .iter()
      .enumerate()
      // leave off i'th difbary
      .filter_map(|(ipos, bary)| (ipos != iterm).then_some(bary.clone()));
    MultiForm::wedge_big(wedge_terms).unwrap_or(MultiForm::one(self.dim()))
  }

  pub fn wedge_terms(&self) -> MultiFormList {
    (0..self.difbarys.len())
      .map(|i| self.wedge_term(i))
      .collect()
  }

  /// The constant exterior derivative of the Whitney form.
  pub fn dif(&self) -> MultiForm {
    if self.grade() == self.dim() {
      return MultiForm::zero(self.dim(), self.grade() + 1);
    }
    let factorial = factorial(self.grade() + 1) as f64;
    let difbarys = self.difbarys.clone();
    factorial * MultiForm::wedge_big(difbarys).unwrap()
  }
}
impl<O: SetOrder> ExteriorField for WhitneyForm<O> {
  type Variance = variance::Co;
  fn dim(&self) -> Dim {
    self.cell_coords.dim_embedded()
  }
  fn grade(&self) -> ExteriorGrade {
    self.associated_subsimp.dim()
  }
  fn at_point<'a>(&self, coord_global: impl Into<CoordRef<'a>>) -> ExteriorElement<Self::Variance> {
    let coord_global = coord_global.into();
    assert_eq!(coord_global.len(), self.dim());
    let barys = self.cell_coords.global_to_bary_coord(coord_global);

    let dim = self.dim();
    let grade = self.grade();
    let mut form = MultiForm::zero(dim, grade);
    for (i, vertex) in self.associated_subsimp.vertices.iter().enumerate() {
      let sign = Sign::from_parity(i);
      let wedge = self.wedge_term(i);

      let bary = barys[vertex];
      form += sign.as_f64() * bary * wedge;
    }
    factorial(grade) as f64 * form
  }
}

#[cfg(test)]
mod test {
  use super::*;

  use {
    crate::discretize_form_on_simplex,
    manifold::{
      geometry::coord::{local::SimplexHandleExt, MeshVertexCoords},
      topology::complex::Complex,
    },
    multi_index::sign::Sign,
  };

  #[test]
  fn whitney_basis_property() {
    for dim in 0..=4 {
      let topology = Complex::standard(dim);
      let coords = MeshVertexCoords::standard(dim);

      let cell = topology.cells().get_by_kidx(0);
      let cell_coords = cell.coord_simplex(&coords);

      for grade in 0..=dim {
        for this_simp in topology.skeleton(grade).handle_iter() {
          let this_simpset = this_simp.simplex_set().clone();
          let whitney_form = WhitneyForm::new(cell_coords.clone(), this_simpset);

          for other_simplex in topology.skeleton(grade).handle_iter() {
            let are_equal = this_simp == other_simplex;
            let other_simplex = other_simplex.coord_simplex(&coords);
            let discret = discretize_form_on_simplex(&whitney_form, &other_simplex);
            let expected = Sign::Pos.as_f64() * are_equal as usize as f64;
            let diff = (discret - expected).abs();
            const TOL: f64 = 10e-9;
            let equal = diff <= TOL;
            assert!(equal, "for: computed={discret} expected={expected}");
            if other_simplex.nvertices() >= 2 {
              let other_simplex_rev = {
                let mut r = other_simplex.clone();
                r.flip_orientation();
                r
              };
              let discret_rev = discretize_form_on_simplex(&whitney_form, &other_simplex_rev);
              let expected_rev = Sign::Neg.as_f64() * are_equal as usize as f64;
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
