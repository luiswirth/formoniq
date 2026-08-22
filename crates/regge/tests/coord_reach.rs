//! Federer's reach of an embedded surface: the largest offset along the
//! normal that stays injective, the smaller of the curvature radius and half
//! the distance to a non-local bottleneck.

use nalgebra as na;
use regge::coord::{mesh::MeshCoords, reach::vertex_reach, vertex_curvature_radius};
use simplicial::{linalg::Vector, topology::complex::Complex};

type Vector3 = na::Vector3<f64>;

/// The normal field in closed form, so the law under test is the reach and
/// not a normal estimator. Only the line matters, never the sign.
type Normals = Box<dyn Fn(&[f64]) -> Vector3>;

/// The reach is the smaller of the two bounds, the curvature radius and half
/// the distance to a non-local bottleneck, on fixtures sitting on either side
/// of that minimum.
///
/// The unit sphere is curvature-limited: its medial axis is the center point,
/// so the reach is the radius, and the tangent-ball formula returns exactly
/// $R$ for every pair on a sphere. A thin flat slab is the other side: its
/// faces are planes, of infinite curvature radius, and the reach is half the
/// thickness, because the opposite face is what an offset runs into. That is
/// the case that collapses a mesh when a displacement is bounded by curvature
/// alone.
#[test]
fn the_reach_is_the_bottleneck_the_curvature_cannot_see() {
  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(3);
  // The outward normal of a sphere at a point is the point itself.
  let normal_of_sphere: Normals = Box::new(|c| Vector3::new(c[0], c[1], c[2]).normalize());
  // The sphere's window is wide below: the tangent-ball estimator reads the
  // reach off pairs of mesh vertices, so a coarse mesh underestimates it.
  let mut fixtures = vec![(
    "unit sphere".to_string(),
    topology,
    coords,
    normal_of_sphere,
    1.0,
    (0.5, 1.05),
    true,
  )];
  for thickness in [0.2, 0.05] {
    let (topology, coords) = slab(thickness);
    // Both faces are level sets of $z$, so the normal line is the $z$ axis;
    // the four sides are what the bottleneck has to be found in spite of.
    let normal_of_face: Normals = Box::new(|_| Vector3::z());
    fixtures.push((
      format!("slab of thickness {thickness}"),
      topology,
      coords,
      normal_of_face,
      thickness / 2.0,
      (0.8, 1.2),
      false,
    ));
  }

  for (name, topology, coords, normal, expected, (lo, hi), curvature_limited) in fixtures {
    let normals: Vec<Vector3> = coords
      .coord_iter()
      .map(|c| normal(c.view().as_slice()))
      .collect();

    let reach = vertex_reach(&topology, &coords, &normals, 10.0);
    let smallest = reach.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
      smallest >= lo * expected,
      "{name}: the reach fell below {expected}, to {smallest}"
    );
    assert!(
      smallest <= hi * expected,
      "{name}: the bound {expected} is never attained, the smallest reach is {smallest}"
    );

    // Which of the two bounds binds is what separates the fixtures: on the
    // sphere the curvature is the reach, on the slab it does not see it.
    let curvature = vertex_curvature_radius(&topology, &coords);
    let tightest = curvature.iter().copied().fold(f64::INFINITY, f64::min);
    if curvature_limited {
      assert!(
        (tightest - smallest).abs() < 1e-9,
        "{name}: the reach {smallest} is not the curvature radius {tightest}"
      );
    } else {
      // At a vertex of a face, where the curvature radius says nothing, the
      // reach is still half the thickness: the bound comes from the opposite
      // face and from nowhere else.
      let flat_reach = reach
        .iter()
        .zip(&curvature)
        .filter(|(_, radius)| **radius > 1.0)
        .map(|(reach, _)| *reach)
        .fold(f64::INFINITY, f64::min);
      assert!(
        (flat_reach - expected).abs() < 0.2 * expected,
        "{name}: a curvature-unbounded vertex has reach {flat_reach}, not {expected}"
      );
    }
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
