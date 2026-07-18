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
//! **The sample set is the full lattice**
//! ([`simplicial::atlas::ref_lattice_bary`]), boundary included. The lattice
//! closes on the faces: a point on a shared facet has the same ambient position
//! from either incident cell (the two agree combinatorially, up to the
//! [`Transition`]'s vertex relabelling), so a glyph there is drawn twice at one
//! place rather than at two. Where a section's value is single-valued matters
//! for what the arrow *reads*, but the grid the arrows sit on extends to the
//! boundary as naturally as into the interior, and the field is glyphed on all
//! of it.
//!
//! [`Transition`]: simplicial::atlas::Transition
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
//! both ways, and sizing it to a fixed fraction ([`GLYPH_LENGTH_FRACTION`]) of
//! the lattice's tightest spacing (the shortest edge over its
//! [`glyph_refinement`], since the spacing is anisotropic -- see [`cell_extent`])
//! keeps a gap between neighbours rather than letting them meet tip-to-tail --
//! each arrow reads as its own mark, filling most of the room the lattice gives
//! it without a field of arrows fused into continuous lines.

use common::{coord::Coord, linalg::nalgebra::Vector};
use derham::{cochain::Cochain, whitney::interpolant::WhitneyInterpolant};
use simplicial::{
  atlas::{ref_lattice_bary, MeshPoint},
  geometry::{
    coord::{mesh::MeshCoords, simplex::SimplexRefExt},
    metric::geometry::Geometry,
  },
  topology::complex::Complex,
};

use crate::{
  bake::{to_vec3, GlyphVertex},
  scene::reduced_form,
};

/// The refinement is capped, not left to grow with the cell: an unbounded
/// ratio of world size to target spacing would let one degenerate huge cell
/// (a coarse mesh's single triangle) flood the scene with lattice points. This
/// is a worst-case bound on a per-cell count, not a global density control --
/// it never fires on a mesh whose cells are already commensurate with the
/// target spacing.
const GLYPH_REFINEMENT_MAX: usize = 8;

/// The arrow's length as a fraction of its lattice's realized spacing. Less than
/// one so neighbouring arrows keep a gap rather than meeting tip-to-tail: at 2/3
/// the space between two collinear samples is a third empty, enough to read each
/// arrow as its own mark while still filling most of the room the lattice gives
/// it.
const GLYPH_LENGTH_FRACTION: f64 = 2.0 / 3.0;

/// The world-space diameter of a cell (greatest inter-vertex distance) and its
/// shortest edge (least). Cheap, since a cell has only `dim + 1` vertices.
///
/// The two answer different questions. The diameter is the cell's overall size,
/// and sets how many glyphs it earns ([`glyph_refinement`]). The shortest edge
/// is what the glyph *length* keys off, because the lattice spacing is
/// anisotropic: adjacent lattice points along an edge are that edge's length
/// over the refinement apart, so the spacing differs by direction and only the
/// shortest edge bounds it in every direction. An arrow sized to the diameter
/// overruns the shorter directions and meets its neighbours there -- the right
/// isosceles reference cell, edges $1, 1, sqrt(2)$, is the visible case; sized
/// to the shortest edge it keeps its gap on any cell shape.
///
/// A `dim == 0` cell has no edge, so the shortest is `0.0` -- no lattice to
/// space and no arrow to draw.
fn cell_extent(coord_simplex: &simplicial::geometry::coord::simplex::SimplexCoords) -> (f64, f64) {
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
/// one is the safe interior value; a zero would clip every fragment on a cell of
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
/// real intra-cell variation -- the number of glyphs a cell earns has to come
/// from the cell's own size, not from the mesh's subdivision count.
///
/// $R approx "diameter" \/ "target spacing"$: the same object-intrinsic scale
/// every other mark uses, so it tracks the mesh's own detail (a coarse mesh's
/// big cells get many glyphs, a fine mesh's small cells collapse back to the
/// $n+1$ floor) without depending on the camera at all.
fn glyph_refinement(dim: simplicial::Dim, diameter: f64, target_spacing: f64) -> usize {
  let raw = if target_spacing > 0.0 {
    (diameter / target_spacing).round() as usize
  } else {
    0
  };
  raw.clamp(dim + 1, GLYPH_REFINEMENT_MAX)
}

/// The glyphs of a line field, baked as flat arrow quads: one arrow per lattice
/// point of each cell, centered on the point and lying in the cell's own plane.
///
/// `target_spacing` is the world spacing the per-cell lattice aims for (see
/// [`glyph_refinement`]); `peak` is the field's greatest magnitude, which the
/// fade is measured against; `half_width` and `outline_width` are the arrow's
/// world dimensions, which size the quad the fragment carves the arrow from.
///
/// The arrow lies in the cell rather than facing the camera, so its four corners
/// are final geometry and each carries its barycentric coordinate directly
/// (`global2bary`) -- which is what lets the fragment clip the
/// arrow to the cell it was sampled in for free. Six corners per glyph, the two
/// triangles of the quad, unindexed.
///
/// The glyphs sit on the undisplaced surface: the fragment biases them toward
/// the camera off it, the way the wireframe is, so they draw over the fill
/// rather than z-fighting it.
pub(crate) fn bake_glyphs(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  target_spacing: f64,
  peak: f64,
  half_width: f64,
  outline_width: f64,
) -> Vec<GlyphVertex> {
  let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);
  let mut vertices = Vec::new();
  // The quad is grown past the arrow by the rim's own width, with slack for its
  // antialiased outer edge, so the outline has room on every side.
  let margin = outline_width * 1.5;

  for cell in topology.cells().handle_iter() {
    let metric = coords.cell_metric(cell);
    // The affine parametrization $psi_K: hat(K) -> RR^N$: its differential pushes
    // the sharped field out of the cell's tangent frame into the ambient one, and
    // `global2bary` reads a corner back to the weights the clip tests. The one
    // place the embedding enters.
    let coord_simplex = cell.coord_simplex(coords);
    let (min_edge, diameter) = cell_extent(&coord_simplex);
    let refinement = glyph_refinement(cell.dim(), diameter, target_spacing);
    // A fixed fraction of the lattice's realized spacing in its tightest
    // direction (the shortest edge over the refinement), so neighbouring arrows
    // keep a gap instead of meeting tip-to-tail on any cell shape -- not the
    // diameter, which would only bound the spacing along the longest direction.
    let length = GLYPH_LENGTH_FRACTION * min_edge / refinement as f64;
    let cell_verts: Vec<Vector> = coord_simplex
      .coord_iter()
      .map(|v| v.view().into_owned())
      .collect();

    for bary in ref_lattice_bary(cell.dim(), refinement) {
      let point = MeshPoint::new(cell.idx(), bary);
      let field = reduced_form(interpolant.eval(&point), &metric).sharp(&metric);
      let ambient: Vector = coord_simplex.pushforward_vector(field.coeffs());
      let magnitude = ambient.norm();
      let opacity = if peak > 0.0 {
        (magnitude / peak).clamp(0.0, 1.0)
      } else {
        0.0
      };

      // The arrow's in-plane frame. A field that vanishes here points nowhere,
      // and its arrow is invisible (opacity 0), so any direction does -- the
      // first spanning edge stands in. The perpendicular is an in-plane vector
      // with the field component removed, so the whole arrow lies in the cell.
      let direction = if magnitude > 0.0 {
        &ambient / magnitude
      } else {
        (&cell_verts[1] - &cell_verts[0]).normalize()
      };
      let perp = cell_verts[1..]
        .iter()
        .map(|v| v - &cell_verts[0])
        .map(|edge| &edge - &direction * edge.dot(&direction))
        .find(|c| c.norm() > 1e-10)
        .map(|c| c.normalize())
        .unwrap_or_else(|| Vector::zeros(direction.len()));

      let center = coord_simplex.bary2global(point.bary()).view().into_owned();
      let corner = |x: f64, y: f64| -> GlyphVertex {
        let world = &center + &direction * (x - length / 2.0) + &perp * y;
        let cell_bary = coord_simplex.global2bary(&Coord::new(world.clone()));
        let p = to_vec3(&world);
        GlyphVertex {
          position: [p.x as f32, p.y as f32, p.z as f32],
          arrow_xy: [x as f32, y as f32],
          length: length as f32,
          opacity: opacity as f32,
          cell_bary: bary_clip4(&cell_bary.view().into_owned()),
        }
      };

      let (x0, x1) = (-margin, length + margin);
      let (y0, y1) = (-(half_width + margin), half_width + margin);
      let (bl, br, tr, tl) = (
        corner(x0, y0),
        corner(x1, y0),
        corner(x1, y1),
        corner(x0, y1),
      );
      vertices.extend([bl, br, tr, bl, tr, tl]);
    }
  }

  vertices
}
