use exterior::{
  dense::{ExteriorElement, ExteriorField, MultiForm},
  manifold::CoordSimplexExt,
  variance, ExteriorGrade,
};
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

/// Whitney Form on the reference complex.
///
/// Can be evaluated on local coordinates.
pub struct WhitneyForm<O: SetOrder> {
  coord_facet: CoordSimplex,
  associated_subsimp: Simplex<O>,
}
impl<O: SetOrder> WhitneyForm<O> {
  pub fn new(coord_facet: CoordSimplex, associated_subsimp: Simplex<O>) -> Self {
    Self {
      coord_facet,
      associated_subsimp,
    }
  }
}
impl<O: SetOrder> ExteriorField for WhitneyForm<O> {
  type Variance = variance::Co;
  fn dim(&self) -> Dim {
    self.coord_facet.dim_intrinsic()
  }
  fn grade(&self) -> ExteriorGrade {
    self.associated_subsimp.dim()
  }
  fn at_point<'a>(&self, coord_local: impl Into<CoordRef<'a>>) -> ExteriorElement<Self::Variance> {
    let coord_local = coord_local.into();
    assert_eq!(coord_local.len(), self.dim());

    let barys = self.coord_facet.global_to_bary_coord(coord_local);
    let difbarys = self.coord_facet.difbarys();

    let dim = self.dim();
    let grade = self.grade();
    let mut form = MultiForm::zero(dim, grade);
    for (i, vertex) in self.associated_subsimp.iter().enumerate() {
      let wedge_terms = self
        .associated_subsimp
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != i)
        .map(|(_, v)| difbarys[v].clone());
      let wedge = MultiForm::wedge_big(wedge_terms).unwrap_or(MultiForm::one(dim));

      let sign = Sign::from_parity(i);
      let bary = barys[vertex];
      form += sign.as_f64() * bary * wedge;
    }
    factorial(grade) as f64 * form
  }
}

#[cfg(test)]
mod test {
  use exterior::manifold::discretize_form_on_simplex;
  use geometry::coord::manifold::{CoordComplex, SimplexHandleExt};

  use super::WhitneyForm;

  #[test]
  fn whitney_basis_property() {
    for dim in 0..=4 {
      let complex = CoordComplex::standard(dim);
      let coord_facet = complex
        .topology()
        .facets()
        .get_by_kidx(0)
        .coord_simplex(complex.coords());
      for grade in 0..=dim {
        for this_simp in complex.topology().skeleton(grade).iter() {
          let this_simpset = this_simp.simplex_set().clone();
          let whitney_form = WhitneyForm::new(coord_facet.clone(), this_simpset);
          for other_simplex in complex.topology().skeleton(grade).iter() {
            let computed = discretize_form_on_simplex(
              &whitney_form,
              &other_simplex.coord_simplex(complex.coords()),
            );
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
