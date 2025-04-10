use super::{simplex::SimplexCoords, CoordRef};

/// Integrates affine linear functions exactly.
pub fn barycentric_quadrature<F>(f: &F, simplex: &SimplexCoords) -> f64
where
  F: Fn(CoordRef) -> f64,
{
  simplex.vol() * f(simplex.barycenter().as_view())
}

pub fn vertex_quadature<F>(f: &F, simplex: &SimplexCoords) -> f64
where
  F: Fn(CoordRef) -> f64,
{
  let sum: f64 = simplex.vertices.coord_iter().map(f).sum();
  simplex.vol() * sum / simplex.nvertices() as f64
}
