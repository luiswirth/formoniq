extern crate nalgebra as na;

pub mod cochain;
pub mod derham;
pub mod io;
pub mod section;
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
  use crate::{cochain::Cochain, derham::derham_map, whitney::interpolant::WhitneyInterpolant};

  use {
    common::linalg::nalgebra::Vector,
    manifold::{gen::cartesian::CartesianMeshInfo, topology::complex::Complex},
  };

  use approx::assert_relative_eq;

  /// $R compose W = id$: Whitney's theorem.
  ///
  /// The de Rham map is a left inverse of the Whitney interpolation, which is
  /// to say the Whitney forms are the basis dual to the degrees of freedom,
  /// $integral_tau W_sigma = delta_(sigma tau)$. Running it on every basis
  /// cochain checks that duality matrix entry by entry -- including the signs,
  /// since a DOF carries the orientation of its simplex.
  ///
  /// Both sides are intrinsic: no coordinates enter, only the topology.
  #[test]
  fn whitney_basis_property() {
    let standard = (0..=4).map(Complex::standard);
    let cartesian = (1..=3).map(|dim| {
      CartesianMeshInfo::new_unit(dim, 2)
        .compute_coord_complex()
        .0
    });

    for topology in standard.chain(cartesian) {
      let dim = topology.dim();
      for grade in 0..=dim {
        let ndofs = topology.nsimplices(grade);
        for idof in 0..ndofs {
          let mut coeffs = Vector::zeros(ndofs);
          coeffs[idof] = 1.0;
          let basis_cochain = Cochain::new(grade, coeffs);

          let whitney = WhitneyInterpolant::new(basis_cochain.clone(), &topology);
          let roundtrip = derham_map(&whitney, &topology, 1);

          assert_relative_eq!(roundtrip.coeffs(), basis_cochain.coeffs(), epsilon = 1e-9);
        }
      }
    }
  }
}
