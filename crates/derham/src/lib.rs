#![doc = include_str!("../README.md")]

extern crate nalgebra as na;

pub mod cochain;
pub mod interpolate;
pub mod io;
pub mod project;
pub mod prolongate;
pub mod section;

#[cfg(test)]
mod test {
  use crate::{
    cochain::Cochain, interpolate::interpolant::WhitneyInterpolant, project::derham_map,
  };
  use multiindex::Dim;

  use {
    simplicial::linalg::Vector,
    simplicial::{mesher::cartesian::CartesianGrid, topology::complex::Complex},
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
    let standard = (0..=4).map(Dim::from).map(Complex::standard);
    let cartesian = (1..=3).map(|dim| CartesianGrid::new_unit(dim, 2).triangulate().0);

    for topology in standard.chain(cartesian) {
      let dim = topology.dim();
      for grade in dim.range_inclusive() {
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
