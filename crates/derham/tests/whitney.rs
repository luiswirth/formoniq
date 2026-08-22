//! The Whitney space: the lowest-order finite element of the de Rham complex.
//!
//! Its laws, in the order they build on each other: at grade $0$ the forms
//! are a partition of unity, the combinatorial normalization of the family is
//! the Koszul contraction $iota_bb(1) = diff$, the forms are dual to the
//! degrees of freedom ($R compose W = id$), and they commute with the
//! exterior derivative and with the trace onto a face.

use approx::assert_relative_eq;
use derham::Cochain;
use derham::interpolate::form::{WhitneyExpansion, WhitneyLsf};
use derham::interpolate::interpolant::WhitneyInterpolant;
use derham::project::derham_map;
use derham::section::Section;
use multialgebra::tensor::tensor_strides;
use multialgebra::{Tensor, exterior_dim};
use multiindex::{Dim, factorial_f64};
use regge::mesher::cartesian::CartesianGrid;
use simplicial::atlas::{Bary, MeshPoint};
use simplicial::linalg::{Matrix, Vector};
use simplicial::topology::complex::Complex;
use simplicial::topology::simplex::unit_boundary_operator;

/// A barycentric point of the reference $n$-cell, off the vertices and off the
/// barycenter, so a law is not read where terms cancel for the wrong reason.
fn probe_bary(dim: Dim) -> Bary {
  let nvertices = (dim + 1).index();
  let weights: Vec<f64> = (0..nvertices).map(|i| (i + 2) as f64).collect();
  let total: f64 = weights.iter().sum();
  Bary::new(Vector::from_iterator(
    nvertices,
    weights.iter().map(|w| w / total),
  ))
}

/// A cochain with distinct, non-degenerate entries on every DOF of a grade.
fn probe_cochain(complex: &Complex, grade: usize) -> Cochain {
  let ndofs = complex.nsimplices(grade);
  Cochain::new(
    grade,
    Vector::from_iterator(ndofs, (0..ndofs).map(|i| 0.5 * i as f64 - 1.3)),
  )
}

/// At grade $0$ the Whitney forms are the barycentric coordinates, hence a
/// partition of unity, $sum_i W_i = 1$ and $sum_i dif W_i = 0$.
///
/// The normalization anchor of the family: the constants lie in the space and
/// are reproduced exactly, which is what makes the interpolation converge at
/// all. It pins the scale of $W$ at the one grade where the $k!$ of the
/// Whitney formula is invisible, so a factor slipped into the grade-0 case
/// alone has nowhere to hide. The differentials summing to zero is the same
/// statement read one grade up, and is what lets $dif$ annihilate a constant.
#[test]
fn whitney_forms_are_a_partition_of_unity() {
  for dim in (0..=4).map(Dim::from) {
    let bary = probe_bary(dim);
    let mut sum = Tensor::multiform_zero(dim, 0);
    let mut sum_dif = Tensor::multiform_zero(dim, 1);
    for lsf in WhitneyLsf::basis(dim, 0) {
      sum += lsf.at_bary(&bary);
      sum_dif += lsf.dif();
    }
    assert_relative_eq!(sum.components(), &Vector::from_element(1, 1.0));
    assert_relative_eq!(
      sum_dif.components(),
      &Vector::zeros(dim.index()),
      epsilon = 1e-12
    );
  }
}

/// Summing the blocks of $C$ over the vertex index collapses the Koszul
/// contraction $kappa$ to $iota_bb(1)$, and $iota_bb(1)$ is the simplicial
/// boundary: the result is $k!$ times $diff$.
///
/// This is the $kappa$ half of a correspondence whose $dif$ half is Stokes,
/// $R compose dif = dif compose R$. The two operators of the exterior
/// algebra have the two operators of the chain complex as their shadows,
/// and forgetting the vertex weights is the map that takes one to the
/// other. It is why the deletion formula of a Whitney form and the boundary
/// of a simplex are the same combinatorics rather than an analogy, and the
/// $k!$ on the right is where the normalization of the whole family is fixed.
///
/// At grade 0 the collapse is the augmentation onto the empty simplex,
/// which [`unit_boundary_operator`] deliberately drops, so the law is read
/// there against the all-ones row it must be.
#[test]
fn koszul_collapses_to_the_boundary_operator() {
  for dim in (0..=4).map(Dim::from) {
    let nvertices = (dim + 1).index();
    for grade in 0..=dim.index() {
      let expansion = WhitneyExpansion::new(dim, grade);
      let matrix = expansion.matrix();
      let ndofs = expansion.dofs().len();
      let nblades = exterior_dim(nvertices, grade);

      let strides = tensor_strides(expansion.slots());
      let collapsed = Matrix::from_fn(nblades, ndofs, |blade, dof| {
        (0..nvertices)
          .map(|vertex| matrix[(blade * strides[0] + vertex * strides[1], dof)])
          .sum()
      });

      let scale = factorial_f64(grade);
      let expected = if grade == 0 {
        Matrix::from_element(1, ndofs, scale)
      } else {
        scale * unit_boundary_operator(dim, grade)
      };
      assert_relative_eq!(collapsed, expected);
    }
  }
}

/// $R compose W = id$: Whitney's theorem.
///
/// The de Rham map is a left inverse of the Whitney interpolation, which is
/// to say the Whitney forms are the basis dual to the degrees of freedom,
/// $integral_tau W_sigma = delta_(sigma tau)$. Running it on every basis
/// cochain checks that duality matrix entry by entry, including the signs,
/// since a DOF carries the orientation of its simplex.
///
/// Both sides are intrinsic: no coordinates enter, only the topology.
#[test]
fn derham_map_left_inverts_whitney_interpolation() {
  let standard = (0..=4).map(Dim::from).map(Complex::unit);
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

/// $dif compose W = W compose dif$: Whitney interpolation is a cochain map.
///
/// The exterior derivative of the interpolation of a cochain is the
/// interpolation of its coboundary, so the Whitney spaces of the grades form
/// a subcomplex of the de Rham complex. Evaluated pointwise on the standard
/// cell: $dif (W c) = sum_sigma c_sigma dif W_sigma$ against $W (dif c)$.
#[test]
fn whitney_interpolation_is_a_cochain_map() {
  for dim in (1..=3).map(Dim::from) {
    let topology = Complex::unit(dim);
    let cell = topology.cells().handle_iter().next().unwrap();

    for grade in dim.range() {
      let cochain = probe_cochain(&topology, grade.index());

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
/// the trace of a Whitney form is the Whitney form of the trace. It is the
/// conformity of the space: two cells sharing a face agree on it, which is
/// what makes the global interpolant a section at all. Swept over every cell
/// dimension, every grade, and every subsimplex whose dimension can still
/// carry the grade ($d >= k$; below it the trace is the zero of the empty
/// space and there is nothing to interpolate).
#[test]
fn whitney_interpolation_commutes_with_the_trace() {
  use simplicial::atlas::FaceTrace;

  for n in (1..=3).map(Dim::from) {
    let complex = Complex::unit(n);
    let cell = complex.cells().handle_iter().next().unwrap();

    for k in n.range_inclusive() {
      let cochain = probe_cochain(&complex, k.index());
      let interpolant = WhitneyInterpolant::new(cochain.clone(), &complex);

      for d in k.range_to_inclusive(n) {
        let face_bary = probe_bary(d);
        for tau in cell.faces(d) {
          let positions = tau.simplex().relative_to(cell.simplex());

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
