use exterior::{dense::MultiForm, ExteriorGrade};
use geometry::coord::{manifold::CoordSimplex, CoordRef};
use index_algebra::{
  binomial,
  combinators::IndexSubsets,
  factorial,
  sign::{sort_signed, Sign},
  variants::SetOrder,
  IndexSet,
};
use topology::{
  simplex::{Simplex, SimplexExt},
  Dim,
};

/// The constant exterior derivatives of the reference Whitney forms, given in
/// the k-form standard basis.
pub fn ref_difwhitneys(n: Dim, k: ExteriorGrade) -> na::DMatrix<f64> {
  let difk = k + 1;
  let difk_factorial = factorial(difk) as f64;

  let whitney_basis_size = binomial(n + 1, k + 1);
  let kform_basis_size = binomial(n, difk);

  let mut ref_difwhitneys = na::DMatrix::zeros(kform_basis_size, whitney_basis_size);
  for (whitney_comb_rank, whitney_comb) in IndexSubsets::canonical(n + 1, k + 1).enumerate() {
    if whitney_comb[0] == 0 {
      let kform_comb: Vec<_> = whitney_comb.iter().skip(1).map(|c| c - 1).collect();
      for i in 0..n {
        let mut kform_comb = kform_comb.clone();
        kform_comb.insert(0, i);
        let sign = sort_signed(&mut kform_comb);

        kform_comb.dedup();
        if kform_comb.len() != difk {
          continue;
        };

        let kform_comb = IndexSet::from(kform_comb).assume_sorted();
        let kform_comb_rank = kform_comb.lex_rank(n);
        ref_difwhitneys[(kform_comb_rank, whitney_comb_rank)] = -sign.as_f64() * difk_factorial;
      }
    } else {
      let kform_comb: Vec<_> = whitney_comb.iter().map(|c| c - 1).collect();
      let kform_comb = IndexSet::from(kform_comb).assume_sorted();
      let kform_comb_rank = kform_comb.lex_rank(n);
      ref_difwhitneys[(kform_comb_rank, whitney_comb_rank)] = difk_factorial;
    }
  }
  ref_difwhitneys
}

pub fn ref_bary(coord: CoordRef, vertex: usize) -> f64 {
  let dim = coord.len();
  assert!(vertex <= dim);
  if vertex == 0 {
    1.0 - coord.sum()
  } else {
    coord[vertex - 1]
  }
}

pub fn ref_difbary(dim: Dim, vertex: usize) -> MultiForm {
  assert!(vertex <= dim);
  let v = if vertex == 0 {
    na::DVector::from_element(dim, -1.0)
  } else {
    let mut v = na::DVector::zeros(dim);
    v[vertex - 1] = 1.0;
    v
  };
  MultiForm::from_grade1(v)
}

pub fn ref_whitney<O: SetOrder>(coord: CoordRef, simplex: &Simplex<O>) -> MultiForm {
  let dim = coord.len();
  let grade = simplex.dim();
  let mut form = MultiForm::zero(dim, grade);
  for (i, vertex) in simplex.iter().enumerate() {
    let wedge_terms = simplex
      .iter()
      .enumerate()
      .filter(|&(j, _)| j != i)
      .map(|(_, v)| ref_difbary(dim, v));
    let wedge = MultiForm::wedge_big(wedge_terms).unwrap_or(MultiForm::one(dim));

    let sign = Sign::from_parity(i);
    let bary = ref_bary(coord, vertex);
    form += sign.as_f64() * bary * wedge;
  }
  factorial(grade) as f64 * form
}

pub fn whitney_on_facet<O: SetOrder>(
  coord_global: CoordRef,
  facet: &CoordSimplex,
  whitney_simplex: &Simplex<O>,
) -> MultiForm {
  assert_eq!(coord_global.len(), facet.dim_embedded());

  // Pushforward of reference Whitney form.
  let transform = facet.affine_transform();
  let coord_ref = transform.try_apply_inverse(coord_global).unwrap();
  let form_ref = ref_whitney(coord_ref.as_view(), whitney_simplex);
  let linear_inv = transform.linear.try_inverse().unwrap();
  form_ref.precompose(&linear_inv)
}

#[cfg(test)]
mod test {
  use exterior::{dense::DifferentialMultiForm, manifold::discretize_form_on_simplex};
  use geometry::coord::manifold::{CoordComplex, SimplexHandleExt};

  use super::ref_whitney;

  #[test]
  fn whitney_basis_property() {
    for dim in 0..=4 {
      let complex = CoordComplex::reference(dim);
      for grade in 0..=dim {
        for this_simp in complex.topology().skeleton(grade).iter() {
          let this_simpset = this_simp.simplex_set().clone();
          let whitney_form = DifferentialMultiForm::from_coeff_fn(
            Box::new(move |coord| ref_whitney(coord.as_view(), &this_simpset).into_coeffs()),
            dim,
            grade,
          );
          for other_simplex in complex.topology().skeleton(grade).iter() {
            let computed =
              discretize_form_on_simplex(&whitney_form, &other_simplex.coord(complex.coords()));
            let expected = (this_simp == other_simplex) as u32 as f64;
            let diff = (computed - expected).abs();
            const TOL: f64 = 10e-9;
            let equal = diff <= TOL;
            assert!(equal, "computed={computed}\nexpected={expected}");
          }
        }
      }
    }
  }
}
