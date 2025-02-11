use super::{local::SimplexCoords, CoordRef};

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

pub fn edge_midpoint_quadrature<F>(f: &F, simplex: &SimplexCoords) -> f64
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
