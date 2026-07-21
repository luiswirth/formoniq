//! The volume bake: a field on a solid, sampled onto a regular grid.
//!
//! A codimension-zero manifold has no surface to rasterize, so it is drawn as a
//! participating medium and the ray march needs the field as a texture. This is
//! the model half of that: pure sampling, no device and no buffer.
//!
//! The grid is legitimate here for a reason particular to $n = 3$ in ambient
//! $3$: a codimension-zero embedded manifold *is* an open subset of the ambient
//! space, so it carries one global coordinate system, and a voxel is indexed by
//! the manifold's own coordinate rather than by the camera. It is manifold
//! state, not screen state.
//!
//! Resampling is a *recovery* and is stated as one: $cal(W) Lambda^n$ is $P_0$,
//! genuinely discontinuous across cells, and trilinear filtering smooths exactly
//! the jump the surface bake tears open. An absorption integral averages along
//! the ray regardless, so the medium is not claiming pointwise values the way a
//! flat-shaded cell does -- but the concession is real and belongs here rather
//! than in a reader's surprise.

use coorder::Coord;
use derham::{cochain::Cochain, interpolate::interpolant::WhitneyInterpolant};
use simplicial::{
  Dim,
  atlas::MeshPoint,
  geometry::coord::{locate::PointLocator, mesh::MeshCoords},
  topology::complex::Complex,
};

use crate::scene::{reduction_sign, scalarize};

/// Voxels per axis, chosen so one voxel is about the mesh's own mean edge
/// length and clamped to what a texture upload should carry. Derived from the
/// object rather than asked of the reader, like every other mark scale: a
/// finer mesh earns a finer grid until the ceiling, and the ceiling is where
/// memory ($256^3$ scalars is 64 MB at `f32`) stops being reasonable on the web.
const MIN_RESOLUTION: usize = 32;
const MAX_RESOLUTION: usize = 128;

/// A scalar field sampled on a regular grid over the mesh's ambient bounding
/// box: what the ray march integrates.
///
/// Values outside the mesh are $0$, which is the physically right answer rather
/// than a sentinel -- empty space neither emits nor absorbs, so the medium ends
/// exactly where the manifold does and no explicit boundary is needed.
pub struct VolumeGrid {
  /// Voxels per axis, $x$ fastest.
  pub resolution: [usize; 3],
  /// Ambient position of the grid's minimum corner.
  pub origin: [f32; 3],
  /// Ambient extent of the whole grid, so a ray converts position to texture
  /// coordinate by one affine map.
  pub size: [f32; 3],
  /// The scalarized field at each voxel center, $0$ outside the mesh.
  pub values: Vec<f32>,
  /// The largest magnitude sampled, the scale the transfer function normalizes
  /// by. $0$ on a field that vanishes identically, which the caller must treat
  /// as "nothing to show" rather than dividing by.
  pub peak: f32,
}

impl VolumeGrid {
  /// Sample `cochain` over the mesh's bounding box, inverting the embedding
  /// through a locator the *mesh* owns.
  ///
  /// The locator is an argument rather than a local because building it is the
  /// expensive half by an order of magnitude, and it depends on nothing this
  /// call varies: a field switch re-samples, it does not re-triangulate.
  ///
  /// The scalar at a voxel is `scalarize` of the Whitney value there, read in
  /// the containing cell's own frame with that cell's metric -- the same
  /// reduction the surface marks use, so a field cannot mean one thing on a
  /// boundary face and another a millimetre inside it.
  pub fn sample(
    topology: &Complex,
    coords: &MeshCoords,
    cochain: &Cochain,
    locator: &PointLocator,
  ) -> Self {
    let (origin, size) = bounding_box(coords);
    let resolution = resolution_for(topology, coords, size);
    let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);
    let k = cochain.grade();
    let n = topology.dim();
    let ambient = coords.dim();

    let mut values = Vec::with_capacity(resolution.iter().product());
    let mut peak = 0.0f32;
    for iz in 0..resolution[2] {
      for iy in 0..resolution[1] {
        for ix in 0..resolution[0] {
          let x = voxel_center([ix, iy, iz], resolution, origin, size);
          // The probe carries the *mesh's* ambient dimension, not 3: a planar
          // mesh lives in R^2 and the locator would refuse a 3-vector.
          let probe = Coord::from_iterator(ambient, x.iter().copied().take(ambient));
          let value = locator.locate(&probe).map_or(0.0, |point| {
            sample_at(topology, coords, &interpolant, &point, k, n)
          });
          peak = peak.max(value.abs() as f32);
          values.push(value as f32);
        }
      }
    }

    Self {
      resolution,
      origin: origin.map(|c| c as f32),
      size: size.map(|c| c as f32),
      values,
      peak,
    }
  }
}

/// The scalar at one located point, in its cell's frame.
fn sample_at(
  topology: &Complex,
  coords: &MeshCoords,
  interpolant: &WhitneyInterpolant,
  point: &MeshPoint,
  k: usize,
  n: Dim,
) -> f64 {
  let cell = point.chart(topology);
  let signed = (k == n).then(|| reduction_sign(topology, cell, k));
  scalarize(
    interpolant.eval(point),
    &coords.simplex_metric(cell.get()),
    signed,
  )
}

/// The ambient minimum corner and extent of the mesh, padded by half a voxel on
/// each side so a boundary vertex is inside the grid rather than exactly on its
/// face.
fn bounding_box(coords: &MeshCoords) -> ([f64; 3], [f64; 3]) {
  let mut lo = [f64::INFINITY; 3];
  let mut hi = [f64::NEG_INFINITY; 3];
  for coord in coords.coord_iter() {
    for axis in 0..3 {
      let c = coord.get(axis).copied().unwrap_or(0.0);
      lo[axis] = lo[axis].min(c);
      hi[axis] = hi[axis].max(c);
    }
  }
  // An empty mesh leaves the bounds inverted; a degenerate axis (a flat mesh in
  // the z = 0 plane) leaves one of them zero. Both collapse to a unit box rather
  // than producing a division by zero downstream.
  let mut origin = [0.0; 3];
  let mut size = [0.0; 3];
  for axis in 0..3 {
    let extent = hi[axis] - lo[axis];
    if extent.is_finite() && extent > 0.0 {
      let pad = 0.02 * extent;
      origin[axis] = lo[axis] - pad;
      size[axis] = extent + 2.0 * pad;
    } else {
      origin[axis] = if lo[axis].is_finite() {
        lo[axis] - 0.5
      } else {
        -0.5
      };
      size[axis] = 1.0;
    }
  }
  (origin, size)
}

/// Voxels per axis: about one per mean edge length, equal on every axis so a
/// voxel is a cube and the medium is isotropic, clamped to the memory ceiling.
fn resolution_for(topology: &Complex, coords: &MeshCoords, size: [f64; 3]) -> [usize; 3] {
  let longest = size.iter().copied().fold(0.0, f64::max);
  let mean_edge = mean_edge_length(topology, coords);
  let per_axis = if mean_edge > 0.0 {
    (longest / mean_edge).ceil() as usize
  } else {
    MIN_RESOLUTION
  };
  let per_axis = per_axis.clamp(MIN_RESOLUTION, MAX_RESOLUTION);
  let voxel = longest / per_axis as f64;
  // Cubic voxels: a short axis gets proportionally fewer of them, never fewer
  // than one, so a flat mesh keeps a single layer instead of collapsing.
  size.map(|s| ((s / voxel).ceil() as usize).max(1))
}

fn mean_edge_length(topology: &Complex, coords: &MeshCoords) -> f64 {
  let nedges = topology.skeleton_raw(1).len();
  if nedges == 0 {
    return 0.0;
  }
  let lengths = coords.to_edge_lengths_sq(topology);
  let total: f64 = topology
    .skeleton(1)
    .handle_iter()
    .map(|edge| lengths.simplex_volume(edge))
    .sum();
  total / nedges as f64
}

fn voxel_center(
  index: [usize; 3],
  resolution: [usize; 3],
  origin: [f64; 3],
  size: [f64; 3],
) -> [f64; 3] {
  let mut x = [0.0; 3];
  for axis in 0..3 {
    let t = (index[axis] as f64 + 0.5) / resolution[axis] as f64;
    x[axis] = origin[axis] + t * size[axis];
  }
  x
}

#[cfg(test)]
mod tests {
  use super::*;
  use simplicial::mesher::cartesian::CartesianGrid;

  /// A constant 0-form samples to that constant everywhere *inside* the mesh
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
}
