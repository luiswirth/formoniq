//! Laws for [`studio::display::amplitude_bound`]: at the chosen amplitude
//! no vertex displaces past its own reach on a thin flat slab, where
//! curvature alone would let the two faces pass through each other, and a
//! vanishing field is unconstrained rather than dividing by zero.

use formoniq_studio::display::amplitude_bound;
use nalgebra as na;
use regge::coord::vertex_curvature_radius;

/// The law the amplitude bound exists to enforce: at the chosen amplitude no
/// vertex displaces past its own reach, so the deformation stays an
/// embedding. Checked where it actually bites, a slab thin enough that the
/// bound is set by its thickness, and stated as the displacement, not as
/// the scalar, since that is the quantity that folds a surface.
#[test]
fn no_vertex_displaces_past_its_reach() {
  let thickness = 0.04;
  let (topology, coords) = slab(thickness);
  let cochain = derham::Cochain::new(
    0,
    na::DVector::from_fn(coords.nvertices(), |i, _| {
      // An arbitrary sign-changing field: what matters is that it is not
      // constant, so the bound is set somewhere specific.
      ((i as f64) * 0.7).sin()
    }),
  );
  let heights = realize::reduce::nodal_heights(&topology, &coords, &cochain);
  let peaks: Vec<f32> = heights.iter().map(|h| h.abs() as f32).collect();

  // The ceilings the bake itself derives, so the safety fraction is not
  // written down a second time here.
  let ceilings: Vec<f32> = realize::bake::BakedMesh::new(&topology, &coords)
    .positions
    .iter()
    .map(|v| v.max_displacement)
    .collect();

  let amplitude = amplitude_bound(ceilings.iter().copied(), &peaks);
  assert!(amplitude.is_finite(), "the bound must bind on a thin slab");

  for (i, (&peak, &ceiling)) in peaks.iter().zip(&ceilings).enumerate() {
    assert!(
      amplitude * peak <= ceiling + 1e-6,
      "vertex {i} displaces {} past its ceiling {ceiling}",
      amplitude * peak
    );
  }

  // And the bound is the thickness, not the curvature. The faces are flat,
  // so a curvature-only ceiling permits a displacement exceeding the slab's
  // own half-thickness, which is the two faces passing through each other,
  // stated as the concrete failure rather than as a ratio between bounds.
  let curvature: Vec<f32> = vertex_curvature_radius(&topology, &coords)
    .iter()
    .map(|r| (0.9 * r) as f32)
    .collect();
  let curvature_only = amplitude_bound(curvature.iter().copied(), &peaks);
  let deepest = curvature_only * peaks.iter().cloned().fold(0.0, f32::max);
  assert!(
    deepest > (thickness / 2.0) as f32,
    "curvature alone must permit a displacement of {deepest} past the \
     half-thickness {}, i.e. through the opposite face",
    thickness / 2.0
  );
  // The reach bound does not.
  assert!(amplitude * peaks.iter().cloned().fold(0.0, f32::max) <= (thickness / 2.0) as f32);
}

/// A field that vanishes everywhere constrains nothing, and the bound says so
/// rather than dividing by zero, the caller's aesthetic ceiling then
/// decides alone.
#[test]
fn a_vanishing_field_is_unconstrained() {
  assert_eq!(
    amplitude_bound([1.0f32, 2.0].into_iter(), &[0.0, 0.0]),
    f32::INFINITY
  );
}

fn slab(
  thickness: f64,
) -> (
  simplicial::topology::complex::Complex,
  regge::coord::mesh::MeshCoords,
) {
  use simplicial::linalg::{Matrix, Vector};
  use simplicial::topology::{complex::Complex, simplex::Simplex, skeleton::Skeleton};
  let n = 8;
  let half = thickness / 2.0;
  let idx = |i: usize, j: usize, top: usize| top * (n + 1) * (n + 1) + j * (n + 1) + i;
  let mut pts: Vec<Vector> = Vec::new();
  for top in 0..2 {
    let z = if top == 0 { -half } else { half };
    for j in 0..=n {
      for i in 0..=n {
        pts.push(Vector::from_vec(vec![
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
          idx(i, j, top),
          idx(i + 1, j, top),
          idx(i + 1, j + 1, top),
          idx(i, j + 1, top),
        ]);
      }
    }
  }
  for k in 0..n {
    quads.push([
      idx(k, 0, 0),
      idx(k + 1, 0, 0),
      idx(k + 1, 0, 1),
      idx(k, 0, 1),
    ]);
    quads.push([
      idx(k, n, 0),
      idx(k + 1, n, 0),
      idx(k + 1, n, 1),
      idx(k, n, 1),
    ]);
    quads.push([
      idx(0, k, 0),
      idx(0, k + 1, 0),
      idx(0, k + 1, 1),
      idx(0, k, 1),
    ]);
    quads.push([
      idx(n, k, 0),
      idx(n, k + 1, 0),
      idx(n, k + 1, 1),
      idx(n, k, 1),
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
  (
    Complex::from_cells(Skeleton::new(cells)),
    regge::coord::mesh::MeshCoords::from(Matrix::from_columns(&pts)),
  )
}
