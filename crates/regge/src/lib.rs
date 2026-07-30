#![doc = include_str!("../README.md")]

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod boundary;
pub mod coord;
pub mod lengths;
pub mod refine;

pub mod io;
pub mod mesher;

use metric::Metric;
use multiindex::Dim;
use simplicial::{atlas::unit_simplex_volume, topology::complex::Complex};

use self::lengths::mesh::MeshLengthsSq;

/// The volume of a cell carrying the given metric tensor,
/// $vol(K) = vol(hat(K)) sqrt(abs(det g))$.
///
/// The chart contributes [`unit_simplex_volume`], the metric the factor
/// $sqrt(abs(det g))$, the whole of the geometry, in one scalar: the same
/// formula on any signature, the absolute value doing nothing Riemannian-side.
pub fn cell_volume(metric: &Metric) -> f64 {
  unit_simplex_volume(metric.dim()) * metric.det_sqrt()
}

/// Discrete Gaussian curvature at every vertex of a 2-dimensional simplicial
/// manifold, by the angle defect: $K(v) = (2 pi - sum_(f ni v) theta_f (v)) \/
/// A(v)$ at an interior vertex, or $(pi - sum_f theta_f (v)) \/ A(v)$ at a
/// boundary one, the standard convention when a mesh has a rim, folding the
/// boundary's own geodesic curvature into $K$ rather than tracking it apart.
/// $A(v)$ is the barycentric lumped area $sum_(K ni v) "vol"(K) \/ 3$, the
/// standard mass-lumping convention.
///
/// Intrinsic: reads the Regge edge lengths, not an embedding, since
/// [`SimplexLengthsSq::vertex_angle`](lengths::simplex::SimplexLengthsSq::vertex_angle)
/// is a function of three of them and needs no coordinates, so a Regge
/// manifold given only as [`MeshLengthsSq`] has a Gaussian curvature exactly
/// as well as an embedded one, which is why the primitive is what this
/// consumes. This
/// Regge's curvature, concentrated at the codimension-2 hinges; in 2D the
/// hinges are vertices, which is the one case implemented here. Generalizing
/// to an $(n-2)$-dimensional hinge of an $n$-manifold needs dihedral angles
/// between codimension-1 facets, not corner angles between edges, and this
/// crate does not yet carry that computation, fixed at 2D for the same
/// reason [`crate::mesher::sphere`] is: the concept itself, not a shortcut, is
/// what is 2-dimensional here.
///
/// Exact, not an approximation of the smooth quantity: this is what
/// Gauss-Bonnet defines discrete curvature to be, with
/// $sum_v K(v) A(v) = 2 pi chi$ on a closed surface, no refinement limit to
/// converge under.
pub fn vertex_gaussian_curvature(topology: &Complex, geometry: &MeshLengthsSq) -> Vec<f64> {
  assert_eq!(
    topology.dim(),
    2,
    "Gaussian curvature is a 2D-surface quantity."
  );
  let nvertices = topology.skeleton_raw(Dim::ZERO).len();
  let boundary: std::collections::HashSet<usize> =
    topology.boundary_vertices().into_iter().collect();

  let mut angle_sum = vec![0.0; nvertices];
  let mut areas = vec![0.0; nvertices];
  for cell in topology.cells().handle_iter() {
    let lengths_sq = geometry.simplex_lengths_sq(cell.get());
    let vol = lengths_sq.vol();
    let verts = &cell.simplex().vertices;
    for m in 0..3 {
      let (a, b) = ((m + 1) % 3, (m + 2) % 3);
      angle_sum[verts[m]] += lengths_sq.vertex_angle(m, a, b);
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
  /// exactly, independent of the triangulation and of the area convention,
  /// a machine-checked identity, not a tolerance around a numerically
  /// approximated constant. Driven through [`lengths::mesh::MeshLengthsSq`], the
  /// Regge-only representation, to demonstrate this needs no embedding at
  /// all.
  #[test]
  fn sphere_gauss_bonnet_holds_exactly() {
    let (topology, coords) = crate::mesher::sphere::mesh_sphere_surface(3);
    let lengths = coords.to_edge_lengths_sq(&topology);
    let gauss = vertex_gaussian_curvature(&topology, &lengths);

    let nvertices = topology.skeleton_raw(Dim::ZERO).len();
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
