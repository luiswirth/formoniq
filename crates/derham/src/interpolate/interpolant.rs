use super::form::WhitneyLsf;
use crate::Cochain;

use {
  multialgebra::Tensor,
  simplicial::{atlas::MeshPoint, topology::complex::Complex},
};

/// The Whitney interpolation $W c = sum_sigma c_sigma W_sigma$ of a cochain: a
/// differential form on the simplicial manifold, affine on each cell, an
/// element of the Whitney space $P^-_1 Lambda^k$.
///
/// This is the FEEC representation formula, reconstructing a genuine
/// differential form from a cochain. It is intrinsic, the cochain and the
/// [`Complex`] are all it takes, since the [`WhitneyLsf`]s are pure
/// combinatorics of the reference cell, so the interpolant exists on a mesh
/// that carries only Regge edge lengths, or no geometry at all.
///
/// Evaluation in ambient coordinates is a strictly separate concern:
/// [`Sampler`](crate::section::Sampler).
pub struct WhitneyInterpolant<'a> {
  cochain: Cochain,
  complex: &'a Complex,
  /// The Whitney forms of the DOF subsimplices, in the colex order of their
  /// local vertex sets: the same order the faces of a cell come in.
  forms: Vec<WhitneyLsf>,
}

impl<'a> WhitneyInterpolant<'a> {
  pub fn new(cochain: Cochain, complex: &'a Complex) -> Self {
    assert!(
      cochain.is_compatible_with(complex),
      "Cochain is not a cochain on this complex."
    );
    let forms = WhitneyLsf::basis(complex.dim(), cochain.grade()).collect();
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

  /// The exterior derivative: since $W$ is a cochain map: this is the
  /// interpolation of the coboundary, $dif (W c) = W (dif c)$.
  pub fn dif(&self) -> Self {
    Self::new(self.cochain.dif(self.complex), self.complex)
  }

  /// The value at a point of the manifold, in the reference frame of its cell.
  pub fn eval(&self, point: &MeshPoint) -> Tensor {
    let cell = point.chart(self.complex);
    let mut value = Tensor::multiform_zero(self.complex.dim(), self.cochain.grade());
    for (dof_simp, form) in cell.faces(self.cochain.grade()).zip(&self.forms) {
      value += self.cochain[dof_simp] * form.at_bary(point.bary());
    }
    value
  }
}
