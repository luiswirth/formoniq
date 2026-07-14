extern crate nalgebra as na;

pub mod cochain;
pub mod derham;
pub mod io;
pub mod whitney;

use {
  exterior::{exterior_power, MultiVector},
  manifold::geometry::coord::simplex::SimplexCoords,
};

pub trait CoordSimplexExt {
  fn spanning_multivector(&self) -> MultiVector;
}
impl CoordSimplexExt for SimplexCoords {
  /// The single blade $v_1 wedge dots.c wedge v_k$ of the spanning vectors:
  /// its coefficients are the $k$-minors, the single column
  /// $Lambda^k V in RR^(binom(n,k) times 1)$.
  fn spanning_multivector(&self) -> MultiVector {
    let vectors = self.spanning_vectors();
    let grade = self.dim_intrinsic();
    let coeffs = exterior_power(&vectors, grade).column(0).into_owned();
    MultiVector::new(coeffs, self.dim_ambient(), grade)
  }
}

#[cfg(test)]
mod test {
  use crate::{derham::integrate_form_simplex, whitney::lsf::WhitneyLsf};

  use common::combo::{Combination, Sign};
  use manifold::{
    geometry::coord::{mesh::MeshCoords, quadrature::SimplexQuadRule, simplex::SimplexHandleExt},
    topology::complex::Complex,
  };

  #[test]
  fn whitney_basis_property() {
    for dim in 0..=4 {
      let topology = Complex::standard(dim);
      let coords = MeshCoords::standard(dim);

      for grade in 0..=dim {
        for dof_simp in topology.skeleton(grade).handle_iter() {
          let whitney_form =
            WhitneyLsf::standard(dim, Combination::from_increasing(dof_simp.iter()));

          for other_simp in topology.skeleton(grade).handle_iter() {
            let are_same_simp = dof_simp == other_simp;
            let other_simplex = other_simp.coord_simplex(&coords);
            let qr = SimplexQuadRule::degree(grade, 1);
            let discret = integrate_form_simplex(&whitney_form, &other_simplex, &qr);
            let expected = f64::from(u8::from(are_same_simp));
            let diff = (discret - expected).abs();
            const TOL: f64 = 10e-9;
            let equal = diff <= TOL;
            assert!(equal, "for: computed={discret} expected={expected}");
            if other_simplex.nvertices() >= 2 {
              let other_simplex_rev = other_simplex.clone().flipped_orientation();
              let discret_rev = integrate_form_simplex(&whitney_form, &other_simplex_rev, &qr);
              let expected_rev = Sign::Neg.as_f64() * usize::from(are_same_simp) as f64;
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
