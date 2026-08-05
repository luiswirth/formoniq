//! The arrow glyphs of a reduced grade-1 field: the field read pointwise, on
//! the barycentric lattice of each cell.
//!
//! The second reading of the reduction the particles already carry: a particle
//! shows what the field does over time, a glyph what it is at a point, sampled
//! where the atlas places it rather than where a population respawns.
//!
//! The sample set is the full lattice
//! ([`simplicial::atlas::unit_lattice_bary`]), boundary included. The lattice
//! closes on the faces: a point on a shared facet has the same ambient position
//! from either incident cell (the two agree combinatorially, up to the
//! [`Transition`]'s vertex relabeling), so a glyph there is drawn twice at one
//! place rather than at two.
//!
//! [`Transition`]: simplicial::atlas::Transition
//!
//! Neither length nor opacity encodes magnitude. The mark carries the direction
//! and the fill beneath it carries the magnitude, which is why `segments.wgsl`
//! does not colormap them: scaling by $|V|$ would restate the fill and shrink
//! the field where it is small, which is where its direction is most worth
//! seeing. An arrow is therefore drawn at full opacity wherever it is drawn at
//! all, so a weak direction reads as well as a strong one.
//!
//! What a magnitude does decide is whether there is a direction at all. A
//! vanishing field points nowhere, and a field that vanishes only up to roundoff
//! points somewhere arbitrary, so a sample below [`GLYPH_DIRECTION_FLOOR`] times
//! the field's peak emits no arrow rather than an arrow with an invented
//! heading.
//!
//! A glyph is centered on its sample and sized to a fixed fraction
//! ([`GLYPH_LENGTH_FRACTION`]) of the lattice's tightest spacing (the shortest
//! edge over its [`glyph_refinement`], the spacing being anisotropic, see
//! [`cell_extent`]). Drawn tail-at-sample it would overshoot ahead of the point
//! and leave the ground behind it bare; centering splits the reach evenly.
//! The fraction leaves a gap between neighbors, so each arrow reads as its own
//! mark instead of fusing into a continuous line.

use coorder::Coord;
use derham::{Cochain, interpolate::interpolant::WhitneyInterpolant};
use metric::tensor::TensorExt;
use rayon::prelude::*;
use regge::coord::{mesh::MeshCoords, simplex::SimplexRefExt};
use simplicial::linalg::Vector;
use simplicial::{
  atlas::{MeshPoint, unit_lattice_bary},
  topology::complex::Complex,
};

use crate::{
  bake::{GlyphInstance, to_vec3},
  reduce::reduced_form,
};

/// The refinement is capped, not left to grow with the cell: an unbounded
/// ratio of world size to target spacing would let one degenerate huge cell
/// (a coarse mesh's single triangle) flood the scene with lattice points. This
/// is a worst-case bound on a per-cell count, not a global density control,
/// it never fires on a mesh whose cells are already commensurate with the
/// target spacing.
pub const GLYPH_REFINEMENT_MAX: usize = 8;

/// The arrow's length as a fraction of its lattice's realized spacing. Less than
/// one so neighboring arrows keep a gap rather than meeting tip-to-tail: at 2/3
/// the space between two collinear samples is a third empty, enough to read each
/// arrow as its own mark while still filling most of the room the lattice gives
/// it.
pub const GLYPH_LENGTH_FRACTION: f64 = 2.0 / 3.0;

/// The magnitude, relative to the field's peak, below which a sample is taken to
/// have no direction and gets no arrow.
///
/// Relative rather than absolute, because a magnitude carries the field's own
/// units and only a ratio is scale-free. The value sits far above double
/// roundoff, which the interpolation, the musical and the pushforward each
/// contribute to, and far below anything a field does where it is genuinely
/// nonzero: what it excludes is a direction computed out of noise, which is
/// arbitrary and moves from frame to frame while looking like data.
pub const GLYPH_DIRECTION_FLOOR: f64 = 1e-9;

/// The world-space diameter of a cell (greatest inter-vertex distance) and its
/// shortest edge (least). Cheap, since a cell has only `dim + 1` vertices.
///
/// The two answer different questions. The diameter is the cell's overall size,
/// and sets how many glyphs it earns ([`glyph_refinement`]). The shortest edge
/// is what the glyph length keys off, because the lattice spacing is
/// anisotropic: adjacent lattice points along an edge are that edge's length
/// over the refinement apart, so the spacing differs by direction and only the
/// shortest edge bounds it in every direction. An arrow sized to the diameter
/// overruns the shorter directions and meets its neighbors there, the right
/// isosceles reference cell, edges $1, 1, sqrt(2)$, is the visible case; sized
/// to the shortest edge it keeps its gap on any cell shape.
///
/// A `dim == 0` cell has no edge, so the shortest is `0.0`, no lattice to
/// space and no arrow to draw.
pub fn cell_extent(coord_simplex: &regge::coord::simplex::SimplexCoords) -> (f64, f64) {
  let vertices: Vec<_> = coord_simplex.coord_iter().collect();
  let (min, max) = vertices
    .iter()
    .enumerate()
    .flat_map(|(i, vi)| {
      vertices[i + 1..]
        .iter()
        .map(move |vj| (vi.view() - vj.view()).norm())
    })
    .fold((f64::INFINITY, 0.0_f64), |(mn, mx), d| {
      (mn.min(d), mx.max(d))
    });
  (if min.is_finite() { min } else { 0.0 }, max)
}

/// A barycentric weight vector as the four the glyph shader's cell clip
/// reads, padded with ones above the cell's intrinsic dimension. The pad has to
/// sit inside (the clip discards a fragment where any weight is negative), and
/// one is the safe interior value. A zero would clip every fragment on a cell of
/// dimension below three.
fn bary_clip4(bary: &Vector) -> [f32; 4] {
  let mut out = [1.0; 4];
  for (slot, &weight) in out.iter_mut().zip(bary.iter()) {
    *slot = weight as f32;
  }
  out
}

/// The refinement of the lattice a cell is glyphed on: chosen so the lattice's
/// world-space spacing matches `target_spacing`, not fixed at $n + 1$ (one
/// glyph, the barycenter). A Whitney form is affine (or, on a solved field,
/// higher-order) across the cell, not constant, so a single sample throws away
/// real intra-cell variation, the number of glyphs a cell earns has to come
/// from the cell's own size, not from the mesh's subdivision count.
///
/// $R approx "diameter" \/ "target spacing"$: the same object-intrinsic scale
/// every other mark uses, so it tracks the mesh's own detail (a coarse mesh's
/// big cells get many glyphs, a fine mesh's small cells collapse back to the
/// $n+1$ floor) without depending on the camera at all.
pub fn glyph_refinement(dim: simplicial::Dim, diameter: f64, target_spacing: f64) -> usize {
  let raw = if target_spacing > 0.0 {
    (diameter / target_spacing).round() as usize
  } else {
    0
  };
  raw.clamp((dim + 1).index(), GLYPH_REFINEMENT_MAX)
}

/// The number of lattice points [`bake_glyphs`] samples on this mesh at this
/// spacing, hence the greatest number of arrows any field on it can produce.
///
/// A function of the mesh and the spacing alone, which is exactly what makes it
/// the bound a consumer sizes a fixed buffer by: the arrows a given instant
/// produces are the samples where the field has a direction, a subset that moves
/// with the field, while the lattice under them does not move at all.
pub fn lattice_size(topology: &Complex, coords: &MeshCoords, target_spacing: f64) -> usize {
  topology
    .cells()
    .handle_iter()
    .map(|cell| {
      let (_, diameter) = cell_extent(&cell.coord_simplex(coords));
      unit_lattice_bary(
        cell.dim(),
        glyph_refinement(cell.dim(), diameter, target_spacing),
      )
      .count()
    })
    .sum()
}

/// The glyphs of a line field, baked as flat arrow quads: one arrow per lattice
/// point of each cell, centered on the point and lying in the cell's own plane.
///
/// `target_spacing` is the world spacing the per-cell lattice aims for (see
/// [`glyph_refinement`]); `peak` is the field's greatest magnitude, which the
/// direction floor is measured against. The arrow's proportions are not passed:
/// they are
/// fractions of its own length, applied in the vertex shader when it generates
/// the quad, so the bake decides only where each arrow is, which way it points
/// and how long it is.
///
/// The arrow lies in the cell rather than facing the camera, so its four corners
/// are final geometry and each carries its barycentric coordinate directly
/// (`global2bary`), which is what lets the fragment clip the
/// arrow to the cell it was sampled in for free. Six corners per glyph, the two
/// triangles of the quad, unindexed.
///
/// The glyphs sit on the undisplaced surface: the fragment biases them toward
/// the camera off it, the way the wireframe is, so they draw over the fill
/// rather than z-fighting it.
///
/// The complex passed here is the render surface, not the mesh (see
/// [`crate::surface::Surface`]), and the cochain is its trace. That is what an
/// arrow is: a mark lying in the manifold it is drawn on, which for a solid
/// is $diff M$ and never a tetrahedron. A cell of a $3$-manifold has no plane
/// for a flat quad to lie in, no determined perpendicular for its `across`
/// axis, and no side for the depth bias to lean toward, the frame built below
/// is well posed exactly because `cell` is at most a triangle. A volume glyph
/// is a different mark with a camera-facing frame, not this one run on tets.
pub fn bake_glyphs(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  target_spacing: f64,
  peak: f64,
) -> Vec<GlyphInstance> {
  let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);

  // Per cell, and the cells are independent: each reads the shared interpolant
  // and writes only its own arrows, so the walk is a map rather than a fold.
  topology
    .cells()
    .handle_iter()
    .collect::<Vec<_>>()
    .into_par_iter()
    .flat_map_iter(|cell| {
      let metric = coords.cell_metric(cell);
      let sign = crate::reduce::admitted_reduction_sign(topology, cell, cochain.grade());
      // The affine parametrization $psi_K: hat(K) -> RR^N$: its differential
      // pushes the sharped field out of the cell's tangent frame into the
      // ambient one, and `global2bary` reads a point back to the weights the
      // clip tests. The one place the embedding enters.
      let coord_simplex = cell.coord_simplex(coords);
      let (min_edge, diameter) = cell_extent(&coord_simplex);
      let refinement = glyph_refinement(cell.dim(), diameter, target_spacing);
      // A fixed fraction of the lattice's realized spacing in its tightest
      // direction (the shortest edge over the refinement), so neighboring
      // arrows keep a gap instead of meeting tip-to-tail on any cell shape,
      // not the diameter, which would only bound the spacing along the longest
      // direction.
      let length = GLYPH_LENGTH_FRACTION * min_edge / refinement as f64;
      let cell_verts: Vec<Vector> = coord_simplex
        .coord_iter()
        .map(|v| v.view().into_owned())
        .collect();

      let floor = GLYPH_DIRECTION_FLOOR * peak;

      unit_lattice_bary(cell.dim(), refinement)
        .filter_map(|bary| {
          let point = MeshPoint::new(cell.idx(), bary);
          let field = reduced_form(interpolant.eval(&point), &metric, sign).musical(&metric);
          // Sharped, hence contravariant, so it genuinely pushes forward: the
          // functor checks that where a bare matrix product would not.
          let ambient: Vector = field
            .pushforward(&coord_simplex.linear_transform())
            .components()
            .clone();
          let magnitude = ambient.norm();
          if magnitude <= floor {
            return None;
          }

          // The arrow's in-plane frame. The perpendicular is an in-plane vector
          // with the field component removed, so the whole arrow lies in the
          // cell.
          let direction = &ambient / magnitude;
          let across = cell_verts[1..]
            .iter()
            .map(|v| v - &cell_verts[0])
            .map(|edge| &edge - &direction * edge.dot(&direction))
            .find(|c| c.norm() > 1e-10)
            .map(|c| c.normalize())
            .unwrap_or_else(|| Vector::zeros(direction.len()));

          let center = coord_simplex.bary2global(point.bary()).view().into_owned();
          // `global2bary` is affine and the quad is planar, so the clip
          // coordinate over the whole arrow is its value here plus these two
          // gradients, exactly, not to first order. A unit step along each
          // frame axis is the gradient.
          let bary_at = |p: &Vector| {
            bary_clip4(
              &coord_simplex
                .global2bary(&Coord::new(p.clone()))
                .view()
                .into_owned(),
            )
          };
          let bary_center = bary_at(&center);
          let gradient = |axis: &Vector| {
            let moved = bary_at(&(&center + axis));
            let mut g = [0.0f32; 4];
            for i in 0..4 {
              g[i] = moved[i] - bary_center[i];
            }
            g
          };

          let c = to_vec3(&center);
          let d = to_vec3(&direction);
          let a = to_vec3(&across);
          Some(GlyphInstance {
            center: [c.x as f32, c.y as f32, c.z as f32],
            length: length as f32,
            direction: [d.x as f32, d.y as f32, d.z as f32],
            opacity: 1.0,
            across: [a.x as f32, a.y as f32, a.z as f32],
            magnitude: magnitude as f32,
            bary_center,
            bary_along: gradient(&direction),
            bary_across: gradient(&across),
          })
        })
        .collect::<Vec<_>>()
        .into_iter()
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

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
    use crate::surface::Surface;
    use regge::mesher::cartesian::CartesianGrid;

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
}
