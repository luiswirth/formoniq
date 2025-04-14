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
}

#[cfg(test)]
mod test {
  use crate::{cochain::de_rahm_map_local, whitney::WhitneyRefLsf};

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
          let whitney_form = WhitneyRefLsf::new(dim, (*dof_simp).clone());

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
