use crate::dense::{DifferentialMultiForm, MultiVector};

use geometry::coord::{
  manifold::{CoordComplex, CoordSimplex},
  CoordRef,
};
use topology::{complex::attribute::Cochain, Dim};

pub trait CoordSimplexExt {
  fn spanning_multivector(&self) -> MultiVector;
}
impl CoordSimplexExt for CoordSimplex {
  fn spanning_multivector(&self) -> MultiVector {
    let vectors = self.spanning_vectors();
    let vectors = vectors
      .column_iter()
      .map(|v| MultiVector::from_grade1(v.into_owned()));
    MultiVector::wedge_big(vectors).unwrap_or(MultiVector::one(self.dim_embedded()))
  }
}

/// Discretize continuous coordinate-based differential k-form into
/// discrete k-cochain on CoordComplex via de Rham map (integration over k-simplex).
pub fn discretize_form_on_mesh(
  form: &DifferentialMultiForm,
  complex: &CoordComplex,
) -> Cochain<Dim> {
  let cochain = complex
    .topology()
    .skeleton(form.grade())
    .iter()
    .map(|simp| CoordSimplex::from_simplex(simp.simplex_set(), complex.coords()))
    .map(|simp| discretize_form_on_simplex(form, &simp))
    .collect::<Vec<_>>()
    .into();
  Cochain::new(form.grade(), cochain)
}

/// Approximates the integral of a differential k-form over a k-simplex,
/// by means of a vertex based (trapezoidal?) quadrature rule.
pub fn discretize_form_on_simplex(
  differential_form: &DifferentialMultiForm,
  simplex: &CoordSimplex,
) -> f64 {
  let transform = simplex.affine_transform();
  let multivector = simplex.spanning_multivector();
  let f = |coord: CoordRef| {
    differential_form
      .at_point(transform.apply(coord).as_view())
      .on_multivector(&multivector)
  };
  let ref_simp = CoordSimplex::standard(simplex.dim_intrinsic());
  barycentric_quadrature(&f, &ref_simp)
}

/// Integrates affine linear functions exactly.
pub fn barycentric_quadrature<F>(f: &F, simplex: &CoordSimplex) -> f64
where
  F: Fn(CoordRef) -> f64,
{
  simplex.vol() * f(simplex.barycenter().as_view())
}

pub fn vertex_quadature<F>(f: &F, simplex: &CoordSimplex) -> f64
where
  F: Fn(CoordRef) -> f64,
{
  let sum: f64 = simplex.vertices.coord_iter().map(f).sum();
  simplex.vol() * sum / simplex.nvertices() as f64
}

pub fn edge_midpoint_quadrature<F>(f: &F, simplex: &CoordSimplex) -> f64
where
  F: Fn(CoordRef) -> f64,
{
  if simplex.dim_intrinsic() == 0 {
    return f(simplex.vertices.coord(0).as_view());
  }

  let sum: f64 = simplex
    .edges()
    .map(|edge| edge.barycenter())
    .map(|c| f(c.as_view()))
    .sum();
  simplex.vol() * sum / simplex.nvertices() as f64
}
