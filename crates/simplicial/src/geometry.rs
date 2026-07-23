//! The geometry a mesh carries: the metric layer, and the extrinsic coordinate
//! layer downstream of it.
//!
//! [`metric`] is the intrinsic one and the only one the manifold's geometry
//! actually rests on; [`coord`] is an embedding, one geometry source among
//! several. The dependency runs that way and not the other: an embedding
//! induces a metric, a metric induces no embedding.

pub mod coord;
pub mod metric;
pub mod refine;

use crate::{
  Dim,
  atlas::refsimp_vol,
  topology::{complex::Complex, role::roles},
};

use gramian::Metric;

use self::metric::mesh::MeshLengthsSq;

/// The volume of a cell carrying the given metric tensor,
/// $vol(K) = vol(hat(K)) sqrt(abs(det g))$.
///
/// The chart contributes [`refsimp_vol`], the metric the factor
/// $sqrt(abs(det g))$ -- the whole of the geometry, in one scalar: the same
/// formula on any signature, the absolute value doing nothing Riemannian-side.
pub fn cell_volume(metric: &Metric) -> f64 {
  refsimp_vol(metric.dim()) * metric.det_sqrt()
}

/// Discrete Gaussian curvature at every vertex of a 2-dimensional simplicial
/// manifold: the Regge deficit angle divided by the area it is spread over,
/// $K(v) = epsilon_v \/ A(v)$.
///
/// Nothing here is 2-dimensional except the packaging. The deficit angle
/// ([`MeshLengthsSq::deficit_angle`]) is Regge curvature at a hinge in any
/// dimension, and a vertex is what a hinge is when $n = 2$; what this adds is
/// the density, $A(v) = sum_(K in v) vol(K) \/ 3$, the barycentric lumped area
/// under the standard mass-lumping convention. Gaussian curvature is a
/// *scalar field*, so it needs that division, and only in 2D is the deficit's
/// hinge a point for the density to sit at.
///
/// Intrinsic: reads the Regge edge lengths, not an embedding -- a Regge
/// manifold given only as [`MeshLengthsSq`] has a Gaussian curvature exactly as
/// well as an embedded one, which is why the primitive is what this consumes.
///
/// Exact, not an approximation of the smooth quantity: this is what
/// Gauss-Bonnet defines discrete curvature to be, with
/// $sum_v K(v) A(v) = 2 pi chi$ on a closed surface -- no refinement limit to
/// converge under.
pub fn vertex_gaussian_curvature(topology: &Complex, geometry: &MeshLengthsSq) -> Vec<f64> {
  assert_eq!(
    topology.dim(),
    2,
    "Gaussian curvature is a 2D-surface quantity."
  );
  let nvertices = topology.skeleton_raw(Dim::ZERO).len();

  let mut areas = vec![0.0; nvertices];
  for cell in topology.cells().handle_iter() {
    let vol = cell_volume(&geometry.cell_metric(cell));
    for &vertex in &cell.simplex().vertices {
      areas[vertex] += vol / 3.0;
    }
  }

  let hinges = topology
    .role_skeleton::<roles::Ridge>()
    .expect("a 2-complex has ridges");
  hinges
    .handle_iter()
    .map(|hinge| {
      let deficit = geometry
        .deficit_angle(hinge)
        .expect("a Riemannian surface has dihedral angles");
      deficit / areas[hinge.kidx()]
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Dim;

  /// Gauss-Bonnet on the unit sphere ($chi = 2$): $sum_v K(v) A(v) = 4 pi$
  /// exactly, independent of the triangulation and of the area convention --
  /// a machine-checked identity, not a tolerance around a numerically
  /// approximated constant. Driven through [`metric::mesh::MeshLengthsSq`], the
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
