//! Laws for [`realize::reach::vertex_reach`]: on the unit sphere the reach
//! is its radius, which the tangent-ball formula reproduces in closed form,
//! and a thin flat slab has infinite curvature radius yet reach half its
//! thickness, the non-local bottleneck the curvature half cannot see.

use nalgebra as na;
use realize::{bake::BakedMesh, reach::vertex_reach};
use regge::coord::{mesh::MeshCoords, vertex_curvature_radius};
use simplicial::{linalg::Vector, topology::complex::Complex};

type Vector3 = na::Vector3<f64>;

/// The vertex normal field the bake builds, which is what production hands
/// [`vertex_reach`].
fn normals_of(topology: &Complex, coords: &MeshCoords) -> Vec<Vector3> {
  BakedMesh::new(topology, coords)
    .positions
    .iter()
    .map(|v| Vector3::new(v.normal[0] as f64, v.normal[1] as f64, v.normal[2] as f64))
    .collect()
}

/// On the unit sphere the reach is the radius, and it is the curvature
/// half that says so: the medial axis is the center point. The tangent-ball
/// formula returns exactly $R$ for every pair on a sphere, so this also
/// checks the estimator against its one closed form.
#[test]
fn sphere_reach_is_its_radius() {
  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(3);
  let reach = vertex_reach(&topology, &coords, &normals_of(&topology, &coords), 10.0);
  for &r in &reach {
    assert!(r > 0.5 && r < 1.05, "expected reach ~ 1, got {r}");
  }
}

/// The half curvature cannot see. A thin flat slab has infinite curvature
/// radius on its faces: they are planes, and reach $t \/ 2$, because the
/// opposite face is what the offset runs into. This is the case that
/// collapses a mesh when a displacement is bounded by curvature alone: the
/// bound has to come from the thickness rather than from the (absent)
/// curvature.
#[test]
fn thin_slab_reach_is_half_its_thickness() {
  for &thickness in &[0.2, 0.05] {
    let (topology, coords) = slab(thickness);
    let curvature = vertex_curvature_radius(&topology, &coords);
    let reach = vertex_reach(&topology, &coords, &normals_of(&topology, &coords), 10.0);

    // The interior of a face is flat, so curvature alone would not bound it.
    let flat = curvature
      .iter()
      .filter(|r| r.is_infinite() || **r > 1.0)
      .count();
    assert!(flat > 0, "the slab's faces must be curvature-unbounded");

    let smallest = reach.iter().cloned().fold(f64::INFINITY, f64::min);
    let expected = thickness / 2.0;
    assert!(
      (smallest - expected).abs() < 0.2 * expected,
      "thickness {thickness}: expected reach ~ {expected}, got {smallest}"
    );
  }
}

/// A closed slab of the given thickness in $z$, triangulated on a coarse
/// grid: two parallel faces plus the four sides, wound as one closed surface.
fn slab(thickness: f64) -> (Complex, MeshCoords) {
  use simplicial::topology::{simplex::Simplex, skeleton::Skeleton};
  let n = 6;
  let half = thickness / 2.0;
  let mut points: Vec<Vector> = Vec::new();
  let index = |i: usize, j: usize, top: usize| top * (n + 1) * (n + 1) + j * (n + 1) + i;
  for top in 0..2 {
    let z = if top == 0 { -half } else { half };
    for j in 0..=n {
      for i in 0..=n {
        points.push(Vector::from_vec(vec![
          i as f64 / n as f64,
          j as f64 / n as f64,
          z,
        ]));
      }
    }
  }
  let mut quads: Vec<[usize; 4]> = Vec::new();
  for top in 0..2 {
    for j in 0..n {
      for i in 0..n {
        quads.push([
          index(i, j, top),
          index(i + 1, j, top),
          index(i + 1, j + 1, top),
          index(i, j + 1, top),
        ]);
      }
    }
  }
  // The four sides, closing the surface so it bounds a solid.
  for k in 0..n {
    quads.push([
      index(k, 0, 0),
      index(k + 1, 0, 0),
      index(k + 1, 0, 1),
      index(k, 0, 1),
    ]);
    quads.push([
      index(k, n, 0),
      index(k + 1, n, 0),
      index(k + 1, n, 1),
      index(k, n, 1),
    ]);
    quads.push([
      index(0, k, 0),
      index(0, k + 1, 0),
      index(0, k + 1, 1),
      index(0, k, 1),
    ]);
    quads.push([
      index(n, k, 0),
      index(n, k + 1, 0),
      index(n, k + 1, 1),
      index(n, k, 1),
    ]);
  }
  let cells = quads
    .into_iter()
    .flat_map(|q| {
      [
        Simplex::from_word(vec![q[0], q[1], q[2]]).1,
        Simplex::from_word(vec![q[0], q[2], q[3]]).1,
      ]
    })
    .collect();
  let complex = Complex::from_cells(Skeleton::new(cells));
  let coords = MeshCoords::from(simplicial::linalg::Matrix::from_columns(&points));
  (complex, coords)
}
