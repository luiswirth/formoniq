pub mod cochain;
pub mod io;
pub mod whitney;

use {
  common::linalg::nalgebra::{CooMatrix, CooMatrixExt},
  exterior::{ExteriorGrade, MultiForm, MultiVector},
  manifold::{geometry::coord::simplex::SimplexCoords, topology::complex::Complex},
};

pub type LocalMultiForm = MultiForm;

pub trait ManifoldComplexExt {
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> CooMatrix;
}
impl ManifoldComplexExt for Complex {
  /// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> CooMatrix {
    self.boundary_operator(grade + 1).transpose()
  }
}

pub trait CoordSimplexExt {
  fn difbarys_ext(&self) -> Vec<LocalMultiForm>;
  fn spanning_multivector(&self) -> MultiVector;
}
impl CoordSimplexExt for SimplexCoords {
  fn spanning_multivector(&self) -> MultiVector {
    let vectors = self.spanning_vectors();
    let vectors = vectors
      .column_iter()
      .map(|v| MultiVector::line(v.into_owned()));
    MultiVector::wedge_big(vectors).unwrap_or(MultiVector::one(self.dim_ambient()))
  }
  fn difbarys_ext(&self) -> Vec<LocalMultiForm> {
    self
      .difbarys()
      .row_iter()
      .map(|difbary| LocalMultiForm::line(difbary.transpose()))
      .collect()
  }
}

#[cfg(test)]
mod test {
  use crate::{cochain::integrate_form_simplex, whitney::WhitneyLsf};

  use common::combo::Sign;
  use manifold::{
    geometry::coord::{mesh::MeshCoords, simplex::SimplexHandleExt},
    topology::complex::Complex,
  };

  #[test]
  fn whitney_basis_property() {
    for dim in 0..=4 {
      let topology = Complex::standard(dim);
      let coords = MeshCoords::standard(dim);

      for grade in 0..=dim {
        for dof_simp in topology.skeleton(grade).handle_iter() {
          let whitney_form = WhitneyLsf::standard(dim, (*dof_simp).clone());

          for other_simp in topology.skeleton(grade).handle_iter() {
            let are_same_simp = dof_simp == other_simp;
            let other_simplex = other_simp.coord_simplex(&coords);
            let discret = integrate_form_simplex(&whitney_form, &other_simplex, None);
            let expected = are_same_simp as u8 as f64;
            let diff = (discret - expected).abs();
            const TOL: f64 = 10e-9;
            let equal = diff <= TOL;
            assert!(equal, "for: computed={discret} expected={expected}");
            if other_simplex.nvertices() >= 2 {
              let other_simplex_rev = other_simplex.clone().flipped_orientation();
              let discret_rev = integrate_form_simplex(&whitney_form, &other_simplex_rev, None);
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
