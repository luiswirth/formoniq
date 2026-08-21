//! Laws for [`realize::volume::VolumeGrid`]: a constant 0-form samples to
//! its constant everywhere inside the mesh and to zero outside it, and the
//! grid is cubic-voxelled and covers the mesh, at every dimension.

use derham::Cochain;
use nalgebra as na;
use realize::volume::VolumeGrid;
use regge::{coord::locate::PointLocator, mesher::cartesian::CartesianGrid};

/// A constant 0-form samples to that constant everywhere inside the mesh
/// and to zero outside it: the interpolation is exact on $cal(W) Lambda^0$'s
/// own constants, so any deviation is the sampler's error and not the field's.
#[test]
fn constant_zero_form_samples_to_its_constant_inside_the_mesh() {
  for dim in 1..=3 {
    let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
    let cochain = Cochain::new(0, na::DVector::from_element(topology.nsimplices(0), 2.5));
    let locator = PointLocator::new(&topology, &coords);
    let grid = VolumeGrid::sample(&topology, &coords, &cochain, &locator);

    assert!((grid.peak - 2.5).abs() < 1e-6, "peak {} != 2.5", grid.peak);
    assert!(
      grid
        .values
        .iter()
        .all(|&v| v.abs() < 1e-6 || (v - 2.5).abs() < 1e-5)
    );
    // A solid fills its box, so most voxels must be inside it. Below that,
    // the locator is missing cells rather than the mesh being thin.
    if dim == 3 {
      let inside = grid.values.iter().filter(|v| v.abs() > 1e-6).count();
      assert!(
        inside * 2 > grid.values.len(),
        "only {inside} of {} voxels landed inside the solid",
        grid.values.len()
      );
    }
  }
}

/// The grid is cubic-voxelled and covers the mesh: the sampled box contains
/// every vertex, at every dimension including the degenerate ones a flat or
/// one-dimensional mesh gives.
#[test]
fn the_grid_covers_the_mesh_at_every_dimension() {
  for dim in 1..=3 {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let cochain = Cochain::new(0, na::DVector::zeros(topology.nsimplices(0)));
    let locator = PointLocator::new(&topology, &coords);
    let grid = VolumeGrid::sample(&topology, &coords, &cochain, &locator);

    for coord in coords.coord_iter() {
      for axis in 0..3 {
        let c = coord.get(axis).copied().unwrap_or(0.0) as f32;
        assert!(c >= grid.origin[axis] && c <= grid.origin[axis] + grid.size[axis]);
      }
    }
    assert!(grid.resolution.iter().all(|&r| r >= 1));
    assert_eq!(grid.values.len(), grid.resolution.iter().product::<usize>());
  }
}
