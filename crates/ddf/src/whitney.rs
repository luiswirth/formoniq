//! The Whitney interpolation $W: C^k -> L^2 Lambda^k$.
//!
//! The lowest-order finite element interpolation of cochains into piecewise
//! linear differential forms. Together with the de Rham map
//! $R: L^2 Lambda^k -> C^k$ (see [`crate::derham`]) it forms the pair of
//! cochain maps at the heart of FEEC, governed by the executable laws
//!
//! - $R compose W = id$: Whitney's theorem
//!   (test `whitney_basis_property` in [`crate`]).
//! - $dif compose W = W compose dif$: the Whitney forms are a subcomplex of
//!   the de Rham complex (test `whitney_interpolation_is_cochain_map`, below).
//! - $R compose dif = dif compose R$: Stokes' theorem
//!   (test `derham_map_is_cochain_map` in [`crate::derham`]).

pub mod form;
pub mod lsf;

#[cfg(test)]
mod test {
  use crate::{cochain::Cochain, whitney::form::WhitneyForm, whitney::lsf::WhitneyLsf};

  use {
    common::linalg::nalgebra::Vector,
    exterior::MultiForm,
    manifold::{
      geometry::coord::{mesh::MeshCoords, simplex::barycenter_local},
      topology::complex::Complex,
    },
  };

  /// $dif compose W = W compose dif$: Whitney interpolation is a cochain map.
  ///
  /// The exterior derivative of the interpolation of a cochain is the
  /// interpolation of its coboundary. Evaluated pointwise on the standard
  /// cell: $dif (W c) = sum_sigma c_sigma dif W_sigma$ against $W (dif c)$.
  #[test]
  fn whitney_interpolation_is_cochain_map() {
    for dim in 1..=3 {
      let topology = Complex::standard(dim);
      let coords = MeshCoords::standard(dim);
      let cell = topology.cells().handle_iter().next().unwrap();

      for grade in 0..dim {
        let ndofs = topology.nsimplices(grade);
        let cochain = Cochain::new(
          grade,
          Vector::from_iterator(ndofs, (0..ndofs).map(|i| (i + 1) as f64)),
        );

        // dif(W c) = sum_sigma c_sigma dif(W_sigma): elementwise constant.
        let mut dif_of_interpolation = MultiForm::zero(dim, grade + 1);
        for dof_simp in cell.faces(grade) {
          let lsf = WhitneyLsf::standard(dim, dof_simp.simplex().relative_to(cell.simplex()));
          dif_of_interpolation += cochain[dof_simp] * lsf.dif();
        }

        // W(dif c) evaluated anywhere in the cell.
        let dif_cochain = cochain.dif(&topology);
        let interpolation_of_dif = WhitneyForm::new(dif_cochain, &topology, &coords)
          .eval_known_cell(cell, &barycenter_local(dim));

        assert!(dif_of_interpolation.eq_epsilon(&interpolation_of_dif, 1e-12));
      }
    }
  }
}
