//! The arrow glyphs of a reduced grade-1 field: the field read pointwise, on
//! the barycentric lattice of each cell.
//!
//! The third reading of the same reduction the streamlines and the particles
//! already carry. The curves are the field's geometry integrated, the particles
//! are its dynamics, and the glyphs are neither -- they are the field
//! *evaluated*, at points chosen by the atlas rather than by a tracer's seeding
//! or a population's respawn. That is the one thing the other two cannot show:
//! where a curve tells you what the field does over a distance, a glyph tells
//! you what it is at a point, and a lattice of them tells you so at points that
//! are placed by the chart alone.
//!
//! **The sample set is the interior lattice**
//! ([`manifold::atlas::ref_lattice_interior`]) and not the full one. A section
//! is chart-independent only in its tangential part, so
//! on a shared facet the two incident charts genuinely disagree and the field
//! has no single value to glyph; the open cell is where it has one. The interior
//! is also what makes the sample set collision-free without a dedup, but that is
//! the consequence, not the reason.
//!
//! **The glyphs are uniform in length.** The mark carries the direction and the
//! fill beneath it carries the magnitude -- the same division of labour the
//! ribbons already keep, and the reason `segments.wgsl` does not colormap them.
//! Scaling a glyph by $|V|$ would restate the fill and hide the field exactly
//! where it is small, which is where its direction is most worth seeing. What
//! magnitude does control is *opacity*: a vanishing field has no direction, so
//! the glyph fades out rather than pointing somewhere arbitrary.

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

/// The refinement of the lattice a cell is glyphed on, at intrinsic dimension
/// `dim`.
///
/// $R = n + 1$: the coarsest refinement with an interior at all, whose one point
/// is the barycenter. One glyph per cell, which is the honest default -- the
/// field's own resolution *is* the mesh, so a sample set finer than the cells
/// would draw detail the cochain does not have, and one coarser would skip cells
/// the mesh went to the trouble of resolving. Raising this refines the lattice
/// within each cell rather than changing anything else here.
fn glyph_refinement(dim: manifold::Dim) -> usize {
  dim + 1
}

/// The glyphs of a line field, baked as segments: one arrow per interior lattice
/// point of each cell, running from the point along the field.
///
/// `length` is the arrow's world length, uniform across the mark; `peak` is the
/// field's greatest magnitude, which the fade is measured against.
///
/// The glyphs sit on the undisplaced surface and carry no field value, so the
/// zero normal makes the standing-wave displacement the identity on them, the
/// same way a traced curve's samples do.
pub fn bake_glyphs(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  length: f64,
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
    for bary in ref_lattice_interior_bary(cell.dim(), glyph_refinement(cell.dim())) {
      let point = MeshPoint::new(cell.idx(), bary);
      let local = reduced_form(interpolant.eval(&point), &metric).sharp(&metric);
      let ambient = to_vec3(&coord_simplex.pushforward_vector(local.coeffs()));
      let magnitude = ambient.norm();

      // A field that vanishes here points nowhere, and the glyph that would
      // report a direction is drawn at zero opacity instead of being dropped:
      // the arrow still exists at the same lattice point, so the mark's own
      // geometry does not depend on the field's zero set.
      let tail = to_vec3(&coord_simplex.bary2global(point.bary()).view().into_owned());
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
      for position in [tail, tail + direction * length] {
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
