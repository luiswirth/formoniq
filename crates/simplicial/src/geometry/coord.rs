//! The extrinsic layer: a coordinate realization of the mesh.
//!
//! An embedding is *one* geometry among several (invariant 2), and everything
//! here is downstream of it: vertex coordinates, the affine parametrization of a
//! cell, point location. A manifold given by Regge edge lengths has none of it,
//! and the intrinsic geometry (invariant 2) must never ask for it.
//!
//! Ambient coordinates are therefore kept apart, by type, from the
//! [`Bary`](crate::atlas::Bary) and [`Local`](crate::atlas::Local) coordinates
//! of a chart: those are intrinsic and exist on every geometry, and the maps
//! between the two worlds are the parametrizations in
//! [`simplex`].

pub mod locate;
pub mod mesh;
pub mod simplex;

pub use coorder::{Ambient, Coord, CoordRef};

use crate::linalg::{RowVector, RowVectorView, Vector, VectorView};

use self::mesh::MeshCoords;
use super::{cell_volume, metric::Geometry};
use crate::topology::complex::Complex;

pub type TangentVector = Vector;
pub type TangentVectorRef<'a> = VectorView<'a>;

pub type CoTangentVector = RowVector;
pub type CoTangentVectorRef<'a> = RowVectorView<'a>;

/// The cotangent of the angle between two vectors, $cot theta = (u dot w) \/
/// |u times w|$, read off their dot product and the (dimension-agnostic) Gram
/// identity $|u times w|^2 = |u|^2 |w|^2 - (u dot w)^2$ rather than an actual
/// cross product -- so this works in an embedding of any dimension, not just
/// $RR^3$. The `max(0.0)` guards roundoff at a near-degenerate (collinear)
/// corner, where the identity's right-hand side can dip fractionally below
/// zero.
fn cot_angle(u: &Vector, w: &Vector) -> f64 {
  let dot = u.dot(w);
  let cross_sq = u.norm_squared() * w.norm_squared() - dot * dot;
  dot / cross_sq.max(0.0).sqrt()
}

/// Discrete mean curvature magnitude $|H(v)|$ at every vertex of an embedded
/// 2-dimensional simplicial surface, via the cotangent-Laplacian
/// mean-curvature-normal identity (Meyer, Desbrun, Schröder, Barr 2003)
/// $Delta_(L B) x (v) = (1)/(2A(v)) sum_(j in N(v)) (cot alpha_(v j) + cot
/// beta_(v j)) (x_j - x_v) = -2H(v) n(v)$: applying the Laplace-Beltrami
/// operator to the embedding's own coordinate function returns the mean
/// curvature normal directly, so no curvature-specific estimator is needed
/// beyond the mesh's own cotangent weights. Only the magnitude is returned --
/// the sign of $H$ needs an outward-normal convention this function has no
/// use for, since [`vertex_curvature_radius`] consumes it through $H^2$
/// alone.
///
/// Extrinsic, unlike [`super::vertex_gaussian_curvature`]: the cotangent
/// weight is read off the embedded edge vectors, not the metric alone, so
/// this lives on the `coord` side and needs an actual [`MeshCoords`], not
/// just a [`Geometry`]. $A(v)$ is the same barycentric lumped area
/// [`super::vertex_gaussian_curvature`] uses, so the two combine into
/// consistent principal curvatures in [`vertex_curvature_radius`].
pub fn vertex_mean_curvature(topology: &Complex, coords: &MeshCoords) -> Vec<f64> {
  assert_eq!(
    topology.dim(),
    2,
    "Mean curvature is a 2D-surface quantity."
  );
  let nvertices = topology.skeleton_raw(0).len();

  let mut areas = vec![0.0; nvertices];
  let mut laplacian = vec![Vector::zeros(coords.dim()); nvertices];
  for cell in topology.cells().handle_iter() {
    let verts = &cell.simplex().vertices;
    let p = [
      coords.coord(verts[0]),
      coords.coord(verts[1]),
      coords.coord(verts[2]),
    ];
    let vol = cell_volume(&coords.cell_metric(cell));
    for &v in verts {
      areas[v] += vol / 3.0;
    }
    for apex in 0..3 {
      let (o1, o2) = ((apex + 1) % 3, (apex + 2) % 3);
      let u = p[o1] - p[apex];
      let w = p[o2] - p[apex];
      let cot = cot_angle(&u, &w);
      let weighted = (p[o1] - p[o2]) * cot;
      laplacian[verts[o1]] += &weighted;
      laplacian[verts[o2]] -= &weighted;
    }
  }

  laplacian
    .into_iter()
    .zip(areas)
    .map(|(l, area)| l.norm() / (4.0 * area))
    .collect()
}

/// A vertex's local radius of curvature: $1 \/ max(|kappa_1|, |kappa_2|)$,
/// the distance along the normal at which the offset map $x |-> x + t n(x)$
/// first develops a fold -- a focal point of the normal congruence, where its
/// differential $I - t S$ ($S$ the shape operator) degenerates. This is the
/// principled bound on how far a surface may be displaced along its normal
/// before self-intersecting: exactly "as far as the surface is locally big"
/// at that point.
///
/// Recovered from the two curvature invariants via $kappa_(1,2) = H
/// plus.minus sqrt(H^2 - K)$, so $max_i |kappa_i| = |H| + sqrt(max(H^2 - K,
/// 0))$; the clamp guards the near-umbilic vertex where discretization noise
/// can push $H^2$ fractionally below $K$ though the true value is $0$.
///
/// Infinite at a flat vertex ($H = K = 0$) and, unconditionally, at a
/// boundary vertex: both [`super::vertex_gaussian_curvature`] and
/// [`vertex_mean_curvature`] are natural (Neumann) boundary quantities there,
/// not curvature -- provably so, since the coordinate function is exactly
/// linear yet $integral_diff.Omega phi_i thin n_x thin d s != 0$ for a
/// boundary test function even on a flat domain. Using either at the rim
/// would clamp displacement near a flat edge for no geometric reason, so a
/// caller relies on an independent upper bound there (e.g. the mesh's own
/// coordinate extent) rather than this curvature-based one.
pub fn vertex_curvature_radius(topology: &Complex, coords: &MeshCoords) -> Vec<f64> {
  let gauss = super::vertex_gaussian_curvature(topology, coords);
  let mean = vertex_mean_curvature(topology, coords);
  let boundary: std::collections::HashSet<usize> =
    topology.boundary_vertices().into_iter().collect();
  (0..gauss.len())
    .map(|v| {
      if boundary.contains(&v) {
        return f64::INFINITY;
      }
      let (k, h) = (gauss[v], mean[v]);
      let kappa_max = h + (h * h - k).max(0.0).sqrt();
      kappa_max.recip()
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The unit sphere has constant curvature $K = H^2 = 1$ and curvature
  /// radius $1$ everywhere. The discrete estimators recover $|H|$ to within
  /// the barycentric lumped area's discretization error (cruder than a mixed
  /// Voronoi area, but simpler and reused as-is for [`vertex_curvature_radius`]).
  /// That same area error enters $kappa_max = |H| + sqrt(max(H^2-K,0))$
  /// asymmetrically -- squared through $H^2$, linear through $K$ -- so the
  /// radius estimate carries a larger, but *conservative* (radius
  /// underestimated, never overestimated), bias than $H$ alone. Underestimating
  /// the safe radius is exactly the safe direction for a fold-safety cap, so
  /// this is loose on purpose, not a correctness gap; the exact Gauss-Bonnet
  /// identity elsewhere is what checks correctness.
  #[test]
  fn sphere_mean_curvature_and_radius_match_unit_radius() {
    let (topology, coords) = crate::gen::sphere::mesh_sphere_surface(3);
    let mean = vertex_mean_curvature(&topology, &coords);
    let radius = vertex_curvature_radius(&topology, &coords);
    for &h in &mean {
      assert!((h - 1.0).abs() < 0.2, "expected |H| ~ 1, got {h}");
    }
    for &r in &radius {
      assert!(
        r < 1.05 && r > 0.5,
        "expected curvature radius in (0.5, 1.05), got {r}"
      );
    }
  }

  /// A flat unit-square grid is developable: zero mean curvature at every
  /// interior vertex (a boundary vertex's raw $H$ is a natural boundary
  /// term, not curvature -- see [`vertex_curvature_radius`]), so the
  /// curvature radius is unbounded everywhere, boundary included -- curvature
  /// must never clamp displacement on a flat surface.
  #[test]
  fn flat_grid_has_unbounded_curvature_radius() {
    let (topology, coords) =
      crate::gen::cartesian::CartesianMeshInfo::new_unit(2, 4).compute_coord_complex();
    let coords = coords.embed_euclidean(3);
    let boundary: std::collections::HashSet<usize> =
      topology.boundary_vertices().into_iter().collect();
    let mean = vertex_mean_curvature(&topology, &coords);
    let radius = vertex_curvature_radius(&topology, &coords);
    for (v, &h) in mean.iter().enumerate() {
      if !boundary.contains(&v) {
        assert!(h.abs() < 1e-9, "expected interior H ~ 0, got {h}");
      }
    }
    for &r in &radius {
      assert!(
        r.is_infinite(),
        "expected an unbounded curvature radius, got {r}"
      );
    }
  }
}
