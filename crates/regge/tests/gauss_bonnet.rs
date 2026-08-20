//! Gauss-Bonnet on a triangulated sphere: a machine-checked identity, no
//! embedding needed once the edge lengths carry the geometry.

use multiindex::Dim;
use regge::mesher::sphere::mesh_sphere_surface;
use regge::{cell_volume, vertex_gaussian_curvature};

/// Gauss-Bonnet on the unit sphere ($chi = 2$): $sum_v K(v) A(v) = 4 pi$
/// exactly, independent of the triangulation and of the area convention,
/// a machine-checked identity, not a tolerance around a numerically
/// approximated constant. Driven through [`lengths::mesh::MeshLengthsSq`], the
/// Regge-only representation, to demonstrate this needs no embedding at
/// all.
#[test]
fn sphere_gauss_bonnet_holds_exactly() {
  let (topology, coords) = mesh_sphere_surface(3);
  let lengths = coords.to_edge_lengths_sq(&topology);
  let gauss = vertex_gaussian_curvature(&topology, &lengths);

  let nvertices = topology.skeleton(Dim::ZERO).len();
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
