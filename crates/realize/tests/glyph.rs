//! Laws for [`realize::glyph::bake_glyphs`]: an arrow is sized by its cell
//! and stays bounded across refinement, a solid's arrows lie on its boundary
//! surface and never in the volume, one instance is emitted per lattice
//! point with a direction and none for a directionless sample, and the
//! lattice tracks the mesh while the arrows track the field.

use derham::Cochain;
use nalgebra as na;
use realize::{
  glyph::{GLYPH_DIRECTION_FLOOR, bake_glyphs, lattice_size},
  surface::Surface,
};
use regge::mesher::cartesian::CartesianGrid;

/// The arrow is sized by the mesh, not by the object: its length is its
/// lattice's realized spacing, so it shrinks with the cells under refinement
/// rather than staying put while they shrink underneath it. Stated as a ratio
/// to the mean edge length, which is what must stay bounded.
///
/// Every other dimension of the arrow is a fraction of this length, applied
/// in the vertex shader, so self-similarity is structural and there is no
/// stored width that could disagree with it: this is the one number the
/// bake decides.
#[test]
fn an_arrow_is_sized_by_its_cell_at_every_refinement() {
  let mut ratios = Vec::new();
  for subdivisions in 1..=4 {
    let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(subdivisions);
    let cochain = Cochain::constant(1.0, topology.skeleton(1));
    let instances = bake_glyphs(&topology, &coords, &cochain, 0.06, 1.0);
    assert!(!instances.is_empty(), "the sweep must produce glyphs");

    let edge = coords.to_edge_lengths_sq(&topology).mesh_width_mean();
    let longest = instances
      .iter()
      .map(|g| f64::from(g.length))
      .fold(0.0, f64::max);
    assert!(longest > 0.0);
    ratios.push(longest / edge);
  }

  // Bounded above and below across the sweep: the arrow neither outgrows its
  // cell nor collapses relative to it. The band is wide because the lattice
  // refinement is an integer and steps as the cells cross the target spacing.
  for ratio in &ratios {
    assert!(
      *ratio > 0.05 && *ratio < 1.0,
      "arrow length is {ratio} of the mean edge across the sweep: {ratios:?}"
    );
  }
}

/// On a solid the arrows live on the boundary surface, never in the volume.
///
/// The failure this pins is not cosmetic. A tetrahedron has no plane for a
/// flat quad to lie in, so glyphing the cells of a $3$-manifold would give
/// arrows an arbitrary `across` axis, at points inside an opaque solid: the
/// mark evaluated on an object it cannot be a mark of. Stated as a
/// geometric fact rather than a count: every arrow's center lies on $diff M$,
/// so none is interior. Checked against the unit cube's faces, where being on
/// the boundary is exactly having a coordinate at 0 or 1.
#[test]
fn a_solids_arrows_lie_on_its_boundary_surface() {
  let (topology, coords) = CartesianGrid::new_unit(3, 2).triangulate();
  let surface = Surface::of(&topology, &coords);
  assert_eq!(
    surface.dim(&topology),
    2,
    "the render surface is a 2-manifold"
  );

  let cochain = Cochain::constant(1.0, topology.skeleton(1));
  let traced = surface
    .trace(&topology, &cochain)
    .expect("a 1-form traces onto the boundary");
  let instances = bake_glyphs(
    surface.complex(&topology),
    surface.coords(&coords),
    &traced,
    0.3,
    1.0,
  );
  assert!(!instances.is_empty(), "a solid must still get arrows");

  for glyph in &instances {
    let on_face = glyph
      .center
      .iter()
      .any(|&x| x.abs() < 1e-6 || (x - 1.0).abs() < 1e-6);
    assert!(
      on_face,
      "arrow at {:?} is interior to the solid, not on its boundary",
      glyph.center
    );
  }
}

/// One instance per lattice point where the field has a direction, six corners
/// generated per instance in the shader: the bake stores arrows, not corners.
#[test]
fn the_bake_emits_one_instance_per_lattice_point() {
  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(1);
  let cochain = Cochain::constant(1.0, topology.skeleton(1));
  let instances = bake_glyphs(&topology, &coords, &cochain, 0.06, 1.0);

  // The constant cochain has a direction everywhere on this mesh, so the
  // bound is attained and the two counts are equal.
  assert_eq!(instances.len(), lattice_size(&topology, &coords, 0.06));
}

/// A field with no direction gets no arrow, and roundoff is not a direction.
///
/// Both halves are the same statement at two scales. The zero field is the
/// exact case, where a heading would have to be invented outright. The scaled
/// one is the case that actually occurs: a field that ought to vanish arrives
/// at a few ulp of the peak instead, through the interpolation and the
/// pushforward, and its normalized direction is then pure noise. Drawn at full
/// opacity that noise is indistinguishable from data, which is why the floor
/// is a threshold on the bake rather than a fade in the shader.
///
/// The peak the floor is read against is the caller's, so the second half also
/// pins that the comparison is relative: the same cochain against a peak of
/// its own size is a field, against a much larger one it is noise.
#[test]
fn a_directionless_sample_gets_no_arrow() {
  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(2);
  let edges = topology.skeleton(1);

  let zero = Cochain::constant(0.0, edges);
  assert!(
    bake_glyphs(&topology, &coords, &zero, 0.06, 0.0).is_empty(),
    "the zero field points nowhere"
  );

  let one = Cochain::constant(1.0, edges);
  assert!(!bake_glyphs(&topology, &coords, &one, 0.06, 1.0).is_empty());
  let noise = Cochain::new(1, one.coeffs() * (0.1 * GLYPH_DIRECTION_FLOOR));
  assert!(
    bake_glyphs(&topology, &coords, &noise, 0.06, 1.0).is_empty(),
    "a field at a fraction of the floor is noise, not a direction"
  );
}

/// The lattice is a function of the mesh and the target spacing, the arrows
/// on it a function of the field: two instants of an evolving field give the
/// same count of arrows, pointing differently.
///
/// Both halves matter, and they fail in opposite directions. A count that
/// moved with the field would mean the arrows cannot be rewritten in place,
/// so a consumer holding a fixed buffer would have to rebuild it. Directions
/// that did not move would mean the arrows are not reading the field at all,
/// which is what a mark baked once from an initial condition looks like: it
/// is on screen, it is plausible, and it is the same picture forever.
#[test]
fn the_lattice_is_the_meshs_and_the_arrows_are_the_fields() {
  let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(2);
  let edges = topology.skeleton(1);
  let one = Cochain::constant(1.0, edges);
  // A different field on the same mesh, not a rescaling of the first: a
  // uniform factor would move no direction and prove nothing.
  let other = Cochain::new(
    1,
    na::DVector::from_iterator(edges.len(), (0..edges.len()).map(|i| (i as f64).sin())),
  );

  let baked = |cochain| bake_glyphs(&topology, &coords, cochain, 0.06, 1.0);
  let (a, b) = (baked(&one), baked(&other));
  assert!(!a.is_empty());
  assert_eq!(a.len(), b.len(), "the lattice does not move with the field");
  assert!(
    a.iter().zip(&b).any(|(x, y)| x.direction != y.direction),
    "the arrows do not read the field"
  );
}
