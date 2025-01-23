use exterior::{
  dense::{ExteriorElement, ExteriorField, MultiForm, MultiFormList},
  manifold::CoordSimplexExt,
  variance, ExteriorGrade,
};
use geometry::coord::{manifold::CoordSimplex, CoordRef};
use index_algebra::{factorial, sign::Sign, variants::SetOrder};
use topology::{simplex::Simplex, Dim};

/// Whitney Form on a coordinate complex.
///
/// Can be evaluated on local coordinates.
pub struct WhitneyForm<O: SetOrder> {
  coord_facet: CoordSimplex,
  associated_subsimp: Simplex<O>,
  difbarys: Vec<MultiForm>,
}
impl<O: SetOrder> WhitneyForm<O> {
  pub fn new(coord_facet: CoordSimplex, associated_subsimp: Simplex<O>) -> Self {
    let difbarys = associated_subsimp
      .vertices
      .iter()
      .map(|vertex| coord_facet.difbary(vertex))
      .collect();

    Self {
      coord_facet,
      associated_subsimp,
      difbarys,
    }
  }

  pub fn wedge_term(&self, iterm: usize) -> MultiForm {
    let wedge_terms = self
      .difbarys
      .iter()
      .enumerate()
      // leave off i'th difbary
      .filter_map(|(ipos, bary)| (ipos != iterm).then_some(bary.clone()));
    MultiForm::wedge_big(wedge_terms).unwrap_or(MultiForm::one(self.dim()))
  }

  pub fn wedge_terms(&self) -> MultiFormList {
    (0..self.difbarys.len())
      .map(|i| self.wedge_term(i))
      .collect()
  }

  /// The constant exterior derivative of the Whitney form.
  pub fn dif(&self) -> MultiForm {
    if self.grade() == self.dim() {
      return MultiForm::zero(self.dim(), self.grade() + 1);
    }
    let factorial = factorial(self.grade() + 1) as f64;
    let difbarys = self.difbarys.clone();
    factorial * MultiForm::wedge_big(difbarys).unwrap()
  }
}
impl<O: SetOrder> ExteriorField for WhitneyForm<O> {
  type Variance = variance::Co;
  fn dim(&self) -> Dim {
    self.coord_facet.dim_embedded()
  }
  fn grade(&self) -> ExteriorGrade {
    self.associated_subsimp.dim()
  }
  fn at_point<'a>(&self, coord_global: impl Into<CoordRef<'a>>) -> ExteriorElement<Self::Variance> {
    let coord_global = coord_global.into();
    assert_eq!(coord_global.len(), self.dim());
    let barys = self.coord_facet.global_to_bary_coord(coord_global);

    let dim = self.dim();
    let grade = self.grade();
    let mut form = MultiForm::zero(dim, grade);
    for (i, vertex) in self.associated_subsimp.vertices.iter().enumerate() {
      let sign = Sign::from_parity(i);
      let wedge = self.wedge_term(i);

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
  use index_algebra::sign::Sign;

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
            let are_equal = this_simp == other_simplex;
            let other_simplex = other_simplex.coord_simplex(complex.coords());
            let discret = discretize_form_on_simplex(&whitney_form, &other_simplex);
            let expected = Sign::Pos.as_f64() * are_equal as usize as f64;
            let diff = (discret - expected).abs();
            const TOL: f64 = 10e-9;
            let equal = diff <= TOL;
            assert!(equal, "for: computed={discret} expected={expected}");
            if other_simplex.nvertices() >= 2 {
              let other_simplex_rev = {
                let mut r = other_simplex.clone();
                r.flip_orientation();
                r
              };
              let discret_rev = discretize_form_on_simplex(&whitney_form, &other_simplex_rev);
              let expected_rev = Sign::Neg.as_f64() * are_equal as usize as f64;
              let diff_rev = (discret_rev - expected_rev).abs();
              let equal_rev = diff_rev <= TOL;
              assert!(
                equal_rev,
                "rev: computed={discret_rev} expected={expected_rev}"
              );
            }
          }
        }
      }
    }
  }
}
