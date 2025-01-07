pub fn integrate_diff_form<F>(coord_complex: &CoordComplex, form: &DifferentialKForm<F>) -> Cochain
where
  F: Fn(CoordRef) -> KFormCoeffs,
{
  let cochain = self
    .complex
    .skeleton(form.rank())
    .keys()
    .map(|simp| CoordSimplex::from_vertplex(simp, &self.coords).integrate_diff_form(form))
    .collect::<Vec<_>>()
    .into();
  Cochain::new(form.rank(), cochain)
}

/// Approximates the integral of a differential k-form over a k-simplex,
/// by means of a vertex based (trapezoidal?) quadrature rule.
pub fn integrate_diff_form<F>(coord_simplex: &CoordSimplex, form: &DifferentialKForm<F>) -> f64
where
  F: Fn(CoordRef) -> KFormCoeffs,
{
  let kvector = self.spanning_kvector();
  let mut sum = 0.0;
  for vertex_coord in self.vertices.coord_iter() {
    sum += form.at_point(vertex_coord).on_kvector(&kvector);
  }
  self.vol() * sum / self.nvertices() as f64
}
