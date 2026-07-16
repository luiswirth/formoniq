//! The geometry a mesh carries: the metric layer, and the extrinsic coordinate
//! layer downstream of it.
//!
//! [`metric`] is the intrinsic one and the only one FEEC assembly consumes;
//! [`coord`] is an embedding, one [`Geometry`](metric::Geometry) implementor
//! among several. The dependency runs that way and not the other: an embedding
//! induces a metric, a metric induces no embedding.

pub mod coord;
pub mod metric;

use crate::{atlas::refsimp_vol, topology::complex::Complex};

use common::gramian::RiemannianMetric;

use self::metric::Geometry;

/// The volume of a cell carrying the given metric tensor,
/// $vol(K) = vol(hat(K)) sqrt(det g)$.
///
/// The chart contributes [`refsimp_vol`], the metric the factor
/// $sqrt(det g)$ -- the whole of the geometry, in one scalar.
pub fn cell_volume(metric: &RiemannianMetric) -> f64 {
  refsimp_vol(metric.dim()) * metric.det_sqrt()
}

/// Discrete Gaussian curvature at every vertex of a 2-dimensional simplicial
/// manifold, by the angle defect: $K(v) = (2 pi - sum_(f ni v) theta_f (v)) \/
/// A(v)$ at an interior vertex, or $(pi - sum_f theta_f (v)) \/ A(v)$ at a
/// boundary one -- the standard convention when a mesh has a rim, folding the
/// boundary's own geodesic curvature into $K$ rather than tracking it apart.
/// $A(v)$ is the barycentric lumped area $sum_(K ni v) "vol"(K) \/ 3$,
/// matching the mass-lumping convention `formoniq`'s own assembly uses
/// elsewhere.
///
/// Metric-only: works off any [`Geometry`], not just an embedding, since
/// [`Gramian::vertex_angle`](common::gramian::Gramian::vertex_angle) needs no
/// coordinates -- a Regge manifold given only as [`metric::mesh::MeshLengths`]
/// has a Gaussian curvature exactly as well as an embedded one. This is
/// Regge's curvature, concentrated at the codimension-2 hinges; in 2D the
/// hinges are vertices, which is the one case implemented here. Generalizing
/// to an $(n-2)$-dimensional hinge of an $n$-manifold needs dihedral angles
/// between codimension-1 facets, not corner angles between edges, and this
/// crate does not yet carry that computation -- fixed at 2D for the same
/// reason [`crate::gen::sphere`] is: the concept itself, not a shortcut, is
/// what is 2-dimensional here.
///
/// Exact, not an approximation of the smooth quantity: this is what
/// Gauss-Bonnet defines discrete curvature to be, with
/// $sum_v K(v) A(v) = 2 pi chi$ on a closed surface -- no refinement limit to
/// converge under.
pub fn vertex_gaussian_curvature(topology: &Complex, geometry: &impl Geometry) -> Vec<f64> {
  assert_eq!(
    topology.dim(),
    2,
    "Gaussian curvature is a 2D-surface quantity."
  );
  let nvertices = topology.skeleton_raw(0).len();
  let boundary: std::collections::HashSet<usize> =
    topology.boundary_vertices().into_iter().collect();

  let mut angle_sum = vec![0.0; nvertices];
  let mut areas = vec![0.0; nvertices];
  for cell in topology.cells().handle_iter() {
    let metric = geometry.cell_metric(cell);
    let vol = cell_volume(&metric);
    let verts = &cell.simplex().vertices;
    for m in 0..3 {
      let (a, b) = ((m + 1) % 3, (m + 2) % 3);
      angle_sum[verts[m]] += metric.vector_gramian().vertex_angle(m, a, b);
      areas[verts[m]] += vol / 3.0;
    }
  }

  (0..nvertices)
    .map(|v| {
      let target = if boundary.contains(&v) {
        std::f64::consts::PI
      } else {
        std::f64::consts::TAU
      };
      (target - angle_sum[v]) / areas[v]
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Gauss-Bonnet on the unit sphere ($chi = 2$): $sum_v K(v) A(v) = 4 pi$
  /// exactly, independent of the triangulation and of the area convention --
  /// a machine-checked identity, not a tolerance around a numerically
  /// approximated constant. Driven through [`metric::mesh::MeshLengths`], the
  /// Regge-only representation, to demonstrate this needs no embedding at
  /// all.
  #[test]
  fn sphere_gauss_bonnet_holds_exactly() {
    let (topology, coords) = crate::gen::sphere::mesh_sphere_surface(3);
    let lengths = coords.to_edge_lengths(&topology);
    let gauss = vertex_gaussian_curvature(&topology, &lengths);

    let nvertices = topology.skeleton_raw(0).len();
    let mut areas = vec![0.0; nvertices];
    for cell in topology.cells().handle_iter() {
      let vol = cell_volume(&lengths.cell_metric(cell));
      for &v in &cell.simplex().vertices {
        areas[v] += vol / 3.0;
      }
    }

    let total: f64 = gauss.iter().zip(&areas).map(|(k, a)| k * a).sum();
    assert!(
      (total - 4.0 * std::f64::consts::PI).abs() < 1e-9,
      "expected 4*pi, got {total}"
    );
  }
}
