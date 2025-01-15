use geometry::coord::{
  manifold::{CoordComplex, CoordSimplex},
  CoordRef,
};
use topology::{complex::attribute::Cochain, Dim};

use crate::{
  dense::{KForm, KVector},
  ExteriorRank,
};

/// Discretize continuous coordinate-based differential k-form into
/// discrete k-cochain on CoordComplex via de Rham map (integration over k-simplex).
pub fn discretize_mesh<F>(
  form_field: &F,
  rank: ExteriorRank,
  complex: &CoordComplex,
) -> Cochain<Dim>
where
  F: Fn(CoordRef) -> KForm,
{
  let cochain = complex
    .topology()
    .skeleton(rank)
    .iter()
    .map(|simp| CoordSimplex::from_simplex(simp.simplex_set(), complex.coords()))
    .map(|simp| discretize_simplex(form_field, &simp))
    .collect::<Vec<_>>()
    .into();
  Cochain::new(rank, cochain)
}

/// Approximates the integral of a differential k-form over a k-simplex,
/// by means of a vertex based (trapezoidal?) quadrature rule.
pub fn discretize_simplex<F>(form_field: &F, simplex: &CoordSimplex) -> f64
where
  F: Fn(CoordRef) -> KForm,
{
  let vectors = simplex.spanning_vectors();
  let vectors = vectors
    .column_iter()
    .map(|v| KVector::from_rank1(v.into_owned()));

  let kvector = KVector::wedge_big(vectors).unwrap_or(KVector::one(simplex.dim_embedded()));

  let transform = simplex.affine_transform();
  let f = |coord: CoordRef| form_field(transform.apply(coord).as_view()).evaluate(&kvector);

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
