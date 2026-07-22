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
    assert!(
      cochain.is_compatible_with(complex),
      "Cochain is not a cochain on this complex."
    );
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
  use multiindex::Dim;

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
    for dim in (1..=3).into_iter().map(Dim::from) {
      let topology = Complex::standard(dim);
      let cell = topology.cells().handle_iter().next().unwrap();

      for grade in dim.range() {
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

  /// $tr_tau compose W = W_tau compose tr_tau$: Whitney interpolation commutes
  /// with the trace onto a subsimplex.
  ///
  /// Pulling the reconstructed field back onto a face equals reconstructing, on
  /// that face as its own standard cell, the traced (restricted) cochain -- so
  /// the trace of a Whitney form is the Whitney form of the trace. Swept over
  /// every cell dimension, every grade, and every subsimplex whose dimension can
  /// still carry the grade ($d >= k$; below it the trace is the zero of the empty
  /// space and there is nothing to interpolate).
  #[test]
  fn whitney_trace_commutes() {
    use simplicial::atlas::{Bary, ref_face_spanning_vectors};

    for n in (1..=3).into_iter().map(Dim::from) {
      let complex = Complex::standard(n);
      let cell = complex.cells().handle_iter().next().unwrap();

      for k in n.range_inclusive() {
        let ndofs = complex.nsimplices(k);
        let cochain = Cochain::new(
          k,
          Vector::from_iterator(ndofs, (0..ndofs).map(|i| 0.5 * (i as f64) - 1.0)),
        );
        let interpolant = WhitneyInterpolant::new(cochain.clone(), &complex);

        for d in k.range_to_inclusive(n) {
          for tau in cell.faces(d) {
            let positions = tau.simplex().relative_to(cell.simplex());

            let weights: Vec<f64> = (0..=d.index()).map(|i| (i + 2) as f64).collect();
            let total: f64 = weights.iter().sum();
            let face_bary = Bary::new(Vector::from_iterator(
              d.index() + 1,
              weights.iter().map(|w| w / total),
            ));

            // tr_tau (W c): the ambient field pulled back along iota_tau.
            let ambient = interpolant.eval(&MeshPoint::on_face(cell.idx(), &positions, &face_bary));
            let traced_field = ambient.pullback(&ref_face_spanning_vectors(n, &positions));

            // W_tau (tr_tau c): the traced cochain interpolated on tau's own cell.
            let sub = Complex::standard(d);
            let sub_cell = sub.cells().handle_iter().next().unwrap();
            let field_of_trace = WhitneyInterpolant::new(cochain.trace(tau), &sub)
              .eval(&MeshPoint::new(sub_cell.idx(), face_bary.clone()));

            assert!(
              traced_field.eq_epsilon(&field_of_trace, 1e-12),
              "n={n} k={k} d={d}"
            );
          }
        }
      }
    }
  }
}
