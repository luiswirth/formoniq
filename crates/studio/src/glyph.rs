//! The arrow glyphs of a reduced grade-1 field: the field read pointwise, on
//! the barycentric lattice of each cell.
//!
//! The second reading of the same reduction the particles already carry. The
//! particles are the field's dynamics, and the glyphs are the field
//! *evaluated*, at points chosen by the atlas rather than by a population's
//! respawn. That is the one thing the particles cannot show: where a particle
//! tells you what the field does over time, a glyph tells you what it is at a
//! point, and a lattice of them tells you so at points that are placed by the
//! chart alone.
//!
//! **The sample set is the interior lattice**
//! ([`manifold::atlas::ref_lattice_interior`]) and not the full one. A section
//! is chart-independent only in its tangential part, so
//! on a shared facet the two incident charts genuinely disagree and the field
//! has no single value to glyph; the open cell is where it has one. The interior
//! is also what makes the sample set collision-free without a dedup, but that is
//! the consequence, not the reason.
//!
//! **The glyphs never encode magnitude in their length.** The mark carries the
//! direction and the fill beneath it carries the magnitude -- the same
//! division of labour the ribbons already keep, and the reason `segments.wgsl`
//! does not colormap them. Scaling a glyph by $|V|$ would restate the fill and
//! hide the field exactly where it is small, which is where its direction is
//! most worth seeing. What magnitude does control is *opacity*: a vanishing
//! field has no direction, so the glyph fades out rather than pointing
//! somewhere arbitrary.
//!
//! **A glyph's length is its lattice's own spacing, and it is centered on its
//! sample.** An arrow drawn tail-at-sample, head reaching forward, covers the
//! cell unevenly: overshooting ahead of the point and leaving the ground
//! behind it bare. Centering the arrow on its sample splits that reach evenly
//! both ways, and sizing it to exactly the lattice's own spacing (the cell's
//! diameter over its [`glyph_refinement`]) makes neighbouring arrows meet
//! without overlapping -- a woven look rather than a field of loose splinters.

use ddf::{cochain::Cochain, whitney::interpolant::WhitneyInterpolant};
use manifold::{
  atlas::{ref_lattice_interior_bary, MeshPoint},
  geometry::{
    coord::{mesh::MeshCoords, simplex::SimplexRefExt},
    metric::geometry::Geometry,
  },
  topology::complex::Complex,
};

use crate::{
  bake::{to_vec3, BakedVertex, SegmentVertex},
  scene::reduced_form,
};

/// The refinement is capped, not left to grow with the cell: an unbounded
/// ratio of world size to target spacing would let one degenerate huge cell
/// (a coarse mesh's single triangle) flood the scene with lattice points. This
/// is a worst-case bound on a per-cell count, not a global density control --
/// it never fires on a mesh whose cells are already commensurate with the
/// target spacing.
const GLYPH_REFINEMENT_MAX: usize = 8;

/// The world-space diameter of a cell: the greatest distance between any two
/// of its vertices. Cheap, since a cell has only `dim + 1` of them.
fn cell_diameter(coord_simplex: &manifold::geometry::coord::simplex::SimplexCoords) -> f64 {
  let vertices: Vec<_> = coord_simplex.coord_iter().collect();
  vertices
    .iter()
    .enumerate()
    .flat_map(|(i, vi)| {
      vertices[i + 1..]
        .iter()
        .map(move |vj| (vi.view() - vj.view()).norm())
    })
    .fold(0.0, f64::max)
}

/// The refinement of the lattice a cell is glyphed on: chosen so the lattice's
/// world-space spacing matches `target_spacing`, not fixed at $n + 1$ (one
/// glyph, the barycenter). A Whitney form is affine (or, on a solved field,
/// higher-order) across the cell, not constant, so a single sample throws away
/// real intra-cell variation -- the number of glyphs a cell earns has to come
/// from the cell's own size, not from the mesh's subdivision count.
///
/// $R approx "diameter" \/ "target spacing"$: the same object-intrinsic scale
/// every other mark uses, so it tracks the mesh's own detail (a coarse mesh's
/// big cells get many glyphs, a fine mesh's small cells collapse back to the
/// $n+1$ floor) without depending on the camera at all.
fn glyph_refinement(dim: manifold::Dim, diameter: f64, target_spacing: f64) -> usize {
  let raw = if target_spacing > 0.0 {
    (diameter / target_spacing).round() as usize
  } else {
    0
  };
  raw.clamp(dim + 1, GLYPH_REFINEMENT_MAX)
}

/// The glyphs of a line field, baked as segments: one arrow per interior lattice
/// point of each cell, centered on the point and running along the field.
///
/// `target_spacing` is the world spacing the per-cell lattice aims for (see
/// [`glyph_refinement`]), and doubles as the glyph length: `peak` is the
/// field's greatest magnitude, which the fade is measured against.
///
/// The glyphs sit on the undisplaced surface and carry no field value, so the
/// zero normal makes the standing-wave displacement the identity on them, the
/// same way a traced curve's samples do.
pub(crate) fn bake_glyphs(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  target_spacing: f64,
  peak: f64,
) -> (Vec<SegmentVertex>, Vec<f32>, Vec<[u32; 2]>) {
  let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);
  let mut vertices = Vec::new();
  let mut segments = Vec::new();

  for cell in topology.cells().handle_iter() {
    let metric = coords.cell_metric(cell);
    // The affine parametrization $psi_K: hat(K) -> RR^N$, whose differential
    // pushes the sharped vector out of the cell's own tangent frame into the
    // ambient one the renderer draws in: the one place the embedding enters.
    let coord_simplex = cell.coord_simplex(coords);
    let diameter = cell_diameter(&coord_simplex);
    let refinement = glyph_refinement(cell.dim(), diameter, target_spacing);
    // The lattice's *realized* spacing, not the target it was rounded and
    // clamped from -- what a neighbouring arrow's length must match for the
    // two to meet exactly.
    let length = diameter / refinement as f64;
    for bary in ref_lattice_interior_bary(cell.dim(), refinement) {
      let point = MeshPoint::new(cell.idx(), bary);
      let local = reduced_form(interpolant.eval(&point), &metric).sharp(&metric);
      let ambient = to_vec3(&coord_simplex.pushforward_vector(local.coeffs()));
      let magnitude = ambient.norm();

      // A field that vanishes here points nowhere, and the glyph that would
      // report a direction is drawn at zero opacity instead of being dropped:
      // the arrow still exists at the same lattice point, so the mark's own
      // geometry does not depend on the field's zero set.
      let center = to_vec3(&coord_simplex.bary2global(point.bary()).view().into_owned());
      let direction = if magnitude > 0.0 {
        ambient / magnitude
      } else {
        na::Vector3::zeros()
      };
      let opacity = if peak > 0.0 {
        (magnitude / peak).clamp(0.0, 1.0)
      } else {
        0.0
      };

      let base = vertices.len() as u32;
      let half = direction * (length / 2.0);
      for position in [center - half, center + half] {
        vertices.push(SegmentVertex {
          vertex: BakedVertex {
            position: [position.x as f32, position.y as f32, position.z as f32],
            normal: [0.0; 3],
            max_displacement: 0.0,
          },
          opacity: opacity as f32,
        });
      }
      segments.push([base, base + 1]);
    }
  }

  let attributes = vec![0.0; vertices.len()];
  (vertices, attributes, segments)
}
