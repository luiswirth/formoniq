use geometry::coord::{
  manifold::{CoordComplex, CoordSimplex},
  Coord,
};
use topology::{complex::attribute::Cochain, Dim};

use crate::{
  dense::{KForm, KVector},
  ExteriorRank,
};

/// Discretize continuous coordinate-based differential k-form into
/// discrete k-cochain on CoordComplex via de Rham map (integration over k-simplex).
pub fn discretize_mesh<F>(form: &F, rank: ExteriorRank, complex: &CoordComplex) -> Cochain<Dim>
where
  F: Fn(Coord) -> KForm,
{
  let cochain = complex
    .topology()
    .skeleton(rank)
    .iter()
    .map(|simp| CoordSimplex::from_simplex(simp.simplex_set(), complex.coords()))
    .map(|simp| discretize_simplex(form, &simp))
    .collect::<Vec<_>>()
    .into();
  Cochain::new(rank, cochain)
}

/// Approximates the integral of a differential k-form over a k-simplex,
/// by means of a vertex based (trapezoidal?) quadrature rule.
pub fn discretize_simplex<F>(form: &F, simplex: &CoordSimplex) -> f64
where
  F: Fn(Coord) -> KForm,
{
  let vectors = simplex.spanning_vectors();
  let vectors = vectors
    .column_iter()
    .map(|v| KVector::from_1vector(v.into_owned()));
  let kvector = KVector::wedge_big(vectors);

  let mut sum = 0.0;
  for vertex_coord in simplex.vertices.coord_iter() {
    sum += form(vertex_coord.into_owned()).on_kvector(&kvector);
  }
  simplex.vol() * sum / simplex.nvertices() as f64
}
