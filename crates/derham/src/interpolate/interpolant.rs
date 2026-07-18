use super::form::WhitneyForm;
use crate::cochain::Cochain;

use {
  exterior::MultiForm,
  simplicial::{
    atlas::MeshPoint,
    topology::{complex::Complex, simplex::standard_subsimps},
  },
};

/// The Whitney interpolation $W c = sum_sigma c_sigma W_sigma$ of a cochain: a
/// differential form on the simplicial manifold, affine on each cell -- an
/// element of the Whitney space $P^-_1 Lambda^k$.
///
/// This is the FEEC representation formula, reconstructing a genuine
/// differential form from a cochain. It is intrinsic -- the cochain and the
/// [`Complex`] are all it takes, since the [`WhitneyForm`]s are pure
/// combinatorics of the reference cell -- so the interpolant exists on a mesh
/// that carries only Regge edge lengths, or no geometry at all.
///
/// Evaluation in ambient coordinates is a strictly separate concern:
/// [`Sampler`](crate::section::Sampler).
pub struct WhitneyInterpolant<'a> {
  cochain: Cochain,
  complex: &'a Complex,
  /// The Whitney forms of the DOF subsimplices, in the colex order of their
  /// local vertex sets: the same order the faces of a cell come in.
  forms: Vec<WhitneyForm>,
}

impl<'a> WhitneyInterpolant<'a> {
  pub fn new(cochain: Cochain, complex: &'a Complex) -> Self {
    let forms = standard_subsimps(complex.dim(), cochain.grade())
      .map(|dof_simp| WhitneyForm::standard(complex.dim(), dof_simp))
      .collect();
    Self {
      cochain,
      complex,
      forms,
    }
  }

  pub fn cochain(&self) -> &Cochain {
    &self.cochain
  }
  pub fn complex(&self) -> &'a Complex {
    self.complex
  }

  /// The exterior derivative: since $W$ is a cochain map, this is the
  /// interpolation of the coboundary, $dif (W c) = W (dif c)$.
  pub fn dif(&self) -> Self {
    Self::new(self.cochain.dif(self.complex), self.complex)
  }

  /// The value at a point of the manifold, in the reference frame of its cell.
  pub fn eval(&self, point: &MeshPoint) -> MultiForm {
    let cell = point.chart(self.complex);
    let mut value = MultiForm::zero(self.complex.dim(), self.cochain.grade());
    for (dof_simp, form) in cell.faces(self.cochain.grade()).zip(&self.forms) {
      value += self.cochain[dof_simp] * form.at_bary(point.bary());
    }
    value
  }
}

#[cfg(test)]
mod test {
  use super::*;

  use crate::{cochain::Cochain, section::Section};

  use {
    exterior::MultiForm,
    simplicial::linalg::Vector,
    simplicial::{atlas::MeshPoint, topology::complex::Complex},
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
          let form = WhitneyForm::standard(dim, dof_simp.simplex().relative_to(cell.simplex()));
          dif_of_interpolation += cochain[dof_simp] * form.dif();
        }

        // W(dif c) evaluated anywhere in the cell.
        let interpolation_of_dif = WhitneyInterpolant::new(cochain, &topology)
          .dif()
          .at(&MeshPoint::barycenter(cell.idx()));

        assert!(dif_of_interpolation.eq_epsilon(&interpolation_of_dif, 1e-12));
      }
    }
  }
}
