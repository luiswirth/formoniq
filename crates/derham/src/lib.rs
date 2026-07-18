//! Discrete differential forms: the exterior algebra, over the simplicial
//! manifold.
//!
//! `derham` is the crate that *joins* [`exterior`] and [`simplicial`], and it exists
//! because neither may depend on the other. A [`Cochain`](cochain::Cochain) is
//! the discrete form; a [`Section`](section::Section) is the continuous field it
//! reconstructs to, evaluated at a [`MeshPoint`](simplicial::atlas::MeshPoint) and
//! valued in the reference frame of that point's chart. The two are related by
//! the [`project`] map and the [`interpolate`] interpolation, which are inverse in
//! the one direction that matters ($R compose W = id$) and cochain maps in both.

extern crate nalgebra as na;

pub mod cochain;
pub mod interpolate;
pub mod io;
pub mod project;
pub mod section;

#[cfg(test)]
mod test {
  use crate::{
    cochain::Cochain, interpolate::interpolant::WhitneyInterpolant, project::derham_map,
  };

  use {
    formoniq_linalg::nalgebra::Vector,
    simplicial::{gen::cartesian::CartesianMeshInfo, topology::complex::Complex},
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
