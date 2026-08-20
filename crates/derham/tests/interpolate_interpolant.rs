//! Whitney interpolation: $dif compose W = W compose dif$ (the Whitney
//! space is a subcomplex) and the trace commutes with interpolation.

use derham::Cochain;
use derham::interpolate::form::WhitneyLsf;
use derham::interpolate::interpolant::WhitneyInterpolant;
use derham::section::Section;
use multialgebra::Tensor;
use multiindex::Dim;
use simplicial::atlas::MeshPoint;
use simplicial::linalg::Vector;
use simplicial::topology::complex::Complex;

/// $dif compose W = W compose dif$: Whitney interpolation is a cochain map.
///
/// The exterior derivative of the interpolation of a cochain is the
/// interpolation of its coboundary. Evaluated pointwise on the standard
/// cell: $dif (W c) = sum_sigma c_sigma dif W_sigma$ against $W (dif c)$.
#[test]
fn whitney_interpolation_is_cochain_map() {
  for dim in (1..=3).into_iter().map(Dim::from) {
    let topology = Complex::unit(dim);
    let cell = topology.cells().handle_iter().next().unwrap();

    for grade in dim.range() {
      let ndofs = topology.nsimplices(grade);
      let cochain = Cochain::new(
        grade,
        Vector::from_iterator(ndofs, (0..ndofs).map(|i| (i + 1) as f64)),
      );

      // dif(W c) = sum_sigma c_sigma dif(W_sigma): elementwise constant.
      let mut dif_of_interpolation = Tensor::multiform_zero(dim, grade + 1);
      for dof_simp in cell.faces(grade) {
        let form = WhitneyLsf::unit(dim, dof_simp.simplex().relative_to(cell.simplex()));
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
/// that face as its own reference cell, the traced (restricted) cochain, so
/// the trace of a Whitney form is the Whitney form of the trace. Swept over
/// every cell dimension, every grade, and every subsimplex whose dimension can
/// still carry the grade ($d >= k$; below it the trace is the zero of the empty
/// space and there is nothing to interpolate).
#[test]
fn whitney_trace_commutes() {
  use simplicial::atlas::{Bary, FaceTrace};

  for n in (1..=3).into_iter().map(Dim::from) {
    let complex = Complex::unit(n);
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
          let traced_field = FaceTrace::new(n, &positions, k).apply(&ambient);

          // W_tau (tr_tau c): the traced cochain interpolated on tau's own cell.
          let sub = Complex::unit(d);
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
