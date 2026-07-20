//! The scene's geometry and materials on the GPU: the reduction from a
//! [`Scene`] and a [`Selection`] to a [`DrawList`] the renderer can draw.
//!
//! This is the model-to-GPU layer, and it is deliberately not the windowed
//! one. It holds no window, no surface and no clock, so the interactive viewer
//! and a headless export build the *same* display from the same
//! `(Scene, Selection, extent)` and differ only in where the frame's time comes
//! from -- which is the renderer's argument, not this layer's business. A
//! material constructed here rather than at a caller is a material the two
//! cannot disagree on.
//!
//! The split within is by what a datum depends on: [`MeshDisplay`] is the bake
//! of the mesh, rebuilt only when the scene changes, and [`FieldDisplay`] is
//! what one field decides -- its materials, its own geometry, its attribute
//! stream. Switching modes rewrites the second alone.

use crate::bake::{self, BakedMesh};
use crate::deposit::DepositLayout;
use crate::render::{
  camera::Camera,
  deposit::DepositBatch,
  item::{DrawList, GlyphBatch, PointBatch, RenderItem, SegmentBatch, SurfaceBatch},
  particles::ParticleBatch,
  uniform::{GlyphMaterial, PostUniform, SegmentMaterial, SurfaceMaterial},
  GpuContext,
};
use crate::scene::Scene;
use crate::ui::{FieldView, MeshView, Post, Selection};

/// The exposure the scene's radiance is read at, before the display transform.
///
/// One, because the marks are already authored in display units: a colormap
/// lands in $[0, 1]$ by construction, and the particles' overflow is set by
/// their own ink and count. Exposure is the knob for when that stops being true.
const EXPOSURE: f32 = 1.0;

/// How much of the blurred glow is added back -- how far the light that will not
/// fit is allowed to spread, rather than how bright it is.
const BLOOM_INTENSITY: f32 = 0.85;

/// The display transform for a [`Post`] rung: the one place its meaning becomes
/// numbers.
///
/// The ladder is the model's, the strengths are the display's, and the renderer
/// sees neither -- only the uniform. Which is why bloom's "off" is an intensity
/// of zero rather than a flag: the resolve multiplies by it unconditionally, so
/// the frame graph can skip the chain without the image knowing.
pub(crate) fn post_uniform(post: Post) -> PostUniform {
  PostUniform {
    exposure: EXPOSURE,
    bloom_intensity: if post.blooms() { BLOOM_INTENSITY } else { 0.0 },
    tonemap: f32::from(post.tone_maps()),
    _pad0: 0.0,
  }
}

/// Peak standing-wave displacement, as a fraction of the scene's own coordinate
/// extent (its radius) -- an object-intrinsic scale, independent of how finely
/// the object is meshed. At this fraction a grade-$l$ eigenmode swells its
/// positive lobes to nearly twice the radius and pinches its negative lobes
/// almost to the center, so the deformed surface reads as the familiar
/// orbital-lobe shape rather than a faint ripple. Kept below 1 so a negative
/// lobe never overshoots the origin and inverts the surface.
const WAVE_AMPLITUDE_FRACTION: f32 = 0.9;

/// The stroke half-width shared by the 1-skeleton's segments and the
/// 0-skeleton's discs, as a fraction of the mesh's *mean edge length* (not the
/// scene's extent).
///
/// It draws the mesh's own edges and vertices, so its scale is the mesh's local
/// one ([`mean_edge_length`](simplicial::geometry::coord::mean_edge_length))
/// rather than the object's global one. Against the extent it read correctly at
/// one refinement only: refine the mesh and the cells shrink while the strokes
/// stay put, until the wireframe is a solid mass with no surface visible between
/// the lines. Against the edge length it is the same drawing at every resolution.
///
/// Shared by both skeletons on purpose: a disc of this radius is exactly as wide
/// as an edge is thick, so a vertex reads as a rounded node capping the edges
/// that meet it, and the two skeletons combine into one graph-style drawing --
/// nodes and links of a single stroke weight.
///
/// Still world-space, so a line reads the same whether the mesh fills the screen
/// or sits in a corner of it. Only which length it is a fraction *of* has
/// changed.
const SKELETON_WIDTH_FRACTION: f32 = 0.012;

/// The world-space spacing the glyph lattice aims for within a cell, on the
/// same object-intrinsic scale as every other mark's: not a fixed sample
/// count, but a target density, so a coarse mesh's large cells earn several
/// glyphs (see [`crate::glyph::glyph_refinement`]) and a fine mesh's small
/// ones collapse back to one without either being tuned by hand. Also the
/// glyph's own length -- each arrow is centered on its sample and sized to its
/// lattice's realized spacing, so neighbouring arrows meet rather than
/// overshooting past each other or leaving gaps (see `glyph.rs`).
const GLYPH_SPACING_FRACTION: f32 = 0.06;

/// The glyph's half-width as a fraction of the arrow's *own length*: the
/// arrowhead's, which its base spans in full, and which
/// [`GLYPH_SHAFT_WIDTH_FRACTION`] narrows the shaft down from.
///
/// A proportion, joining the head length and shaft width that already were
/// ones, so an arrow is a single shape scaled by the cell it sits in and the
/// whole mark is self-similar. The length is already cell-derived (the
/// lattice's realized spacing), so a world-space width was the one dimension
/// that did not follow the mesh: on a refined mesh the arrows shortened and
/// kept their width, thickening into stubs.
const GLYPH_WIDTH_FRACTION: f32 = 0.36;

/// The arrowhead's length as a fraction of the glyph's own, and the shaft's
/// half-width as a fraction of the head's base. A head a third of the arrow, on
/// a shaft a third of its width: the proportions of a drawn arrow, self-similar
/// at every glyph since both are fractions.
const GLYPH_HEAD_LENGTH_FRACTION: f32 = 0.36;
const GLYPH_SHAFT_WIDTH_FRACTION: f32 = 0.32;

/// The glyphs' ink: white, at reduced opacity so a lattice of them reads as a
/// field rather than as a wall of marks. Drawn over the fill.
const GLYPH_INK: [f32; 4] = [1.0, 1.0, 1.0, 0.75];

/// The outline's rim, as a fraction of the arrow's half-width -- so it is a
/// constant rim around the silhouette rather than one that thins toward the tip
/// along with the arrow it traces, and it scales with the arrow like every other
/// proportion. The color is fixed black in `segments.wgsl`: the one color that
/// separates from every sample of either colormap, the same reasoning the
/// shader's own doc gives for not colormapping the ink itself.
const GLYPH_OUTLINE_WIDTH_FRACTION: f32 = 0.27;

/// The opacity the glyphs of an eigenmode fade to at the standing wave's node,
/// where the field vanishes and an arrow is meaningless -- never fully, since
/// the glyph lattice of a standing mode is the same set at every phase, and
/// blinking it out entirely would read as the geometry changing.
const GLYPH_NODE_OPACITY: f32 = 0.25;

/// How fast the fastest particle crosses the object: scene radii per second, at
/// the field's peak magnitude.
///
/// Object-intrinsic, exactly as the glyph spacing is, so the flow reads
/// at the same pace whether the mesh is a unit sphere or an OBJ at arbitrary
/// scale, and regardless of the cochain's own units. Slow on purpose: a trail
/// laid a few pixels per step reads as flow, and one that jumps across the
/// screen reads as flicker. This is the knob that sets the pace of the flow.
const PARTICLE_SPEED_FRACTION: f64 = 0.10;

/// The advection's fixed rate, in steps per second of animation time.
///
/// The bake exponentiates one step's worth of field time, so the step is fixed
/// and only their *number* varies with elapsed time. Tied to the animation
/// clock rather than the frame rate: a slow frame takes more steps, not shorter
/// ones, and the flow runs at the same speed on any machine.
pub(crate) const STEPS_PER_SECOND: f32 = 60.0;

/// The dyadic resolution of one step: a step is $2^"depth"$ ticks, and a facet
/// crossing lands within one tick of the true crossing. At this depth that
/// residue is far below the `f32` the whole pass runs in, and the cost is one
/// mat-vec per level only on the steps that actually cross.
const ADVECT_DEPTH: u32 = 10;

/// How many particles, and how many distinct sites they are born at.
///
/// The count chooses the *collective* regime: no individual speck is meant to
/// be followed, and the picture is the density the population piles into and
/// the trails it lays in the deposit atlas -- which is why the radius and ink
/// above sit far below what a followable speck would want. The three are one
/// setting: move the count and they move with it.
///
/// The seeds are far fewer than the particles: a seed is a *place* to be born,
/// not a particle, and many pass through each over a session.
const PARTICLE_COUNT: u32 = 1_000_000;
const PARTICLE_SEEDS: usize = 65_536;

/// How long a laid trail survives, as the time for one deposit to decay to
/// $1\/e$. Together with [`PARTICLE_SPEED_FRACTION`] this *is* the streak
/// length: the fastest particles draw tails of about their speed times this,
/// $approx 0.15$ object radii, long enough to read as flow lines and short
/// enough not to smear the whole surface uniform.
const DEPOSIT_DECAY_SECONDS: f32 = 1.5;

/// The fill's brightness where no trail has passed, as a fraction of its plain
/// (un-deposited) radiance. Below 1 on purpose: the flow's illumination has to
/// take its contrast from somewhere, and dimming the still regions is what
/// makes the lit filaments read as light rather than as a paler colormap.
const DEPOSIT_FLOOR: f32 = 0.3;

/// The equilibrium trail brightness the gain is calibrated to: the lift
/// `floor + gain * D` at the *average* deposit density, were the population
/// spread uniformly. 1 makes the average trail exactly restore the plain
/// radiance -- so still regions sit at the floor, ordinary streaks at nominal,
/// and the filaments where the flow bunches overshoot 1 and bloom. The
/// calibration is arithmetic over the actual count, decay and atlas budget,
/// never a hand-tuned brightness.
const DEPOSIT_MEAN_LIFT: f32 = 1.0;

/// The advection steps that have elapsed by `time`.
///
/// The count is a function of the instant, not an accumulator, so it is the
/// same number for the window and for an exporter aiming at that instant --
/// which is what makes the two agree on where a particle is.
pub(crate) fn steps_at(time: f32) -> u32 {
  (time.max(0.0) * STEPS_PER_SECOND) as u32
}

/// The scene's coordinate extent: the largest distance of any vertex from the
/// mesh's own centroid -- its intrinsic radius, independent of where the mesh
/// sits in space. Measured about the centroid, not the origin, so a mesh
/// nowhere near the origin (a unit grid on $\[0,1\]^2$, an off-center loaded OBJ)
/// still reports its true size; an origin-centered unit sphere gives 1 either
/// way. Both the camera framing and the standing-wave amplitude scale off this,
/// so neither is tuned to the sphere.
pub(crate) fn scene_extent(scene: &Scene) -> f64 {
  scene_centroid_and_extent(scene).1
}

/// The mesh's own centroid, in the same 3-vector coordinates as
/// [`scene_extent`] -- the point the camera should target so an off-center
/// mesh (a unit grid on $\[0,1\]^2$, an off-center loaded OBJ) still ends up in
/// the middle of the view, not just correctly sized.
fn scene_centroid_and_extent(scene: &Scene) -> (na::DVector<f64>, f64) {
  let coords = &scene.coords;
  let n = coords.nvertices().max(1) as f64;
  let centroid = coords
    .coord_iter()
    .fold(na::DVector::zeros(3), |acc, c| acc + *c)
    / n;
  let extent = coords
    .coord_iter()
    .map(|c| (*c - &centroid).norm())
    .fold(0.0, f64::max)
    .max(1e-6);
  (centroid, extent)
}

/// The camera's natural starting orientation for a scene, derived purely from
/// its own coordinates -- not which `Demo` built it, so a future flat or 3D
/// scene gets the same sensible default without adding another `match` arm
/// here.
pub(crate) fn default_camera(scene: &Scene, aspect: f32) -> Camera {
  // Framing distance from the scene's own coordinate extent, not a constant
  // tuned for the sphere: an icosphere of radius 1 gives back exactly the
  // prior hardcoded 3.0, and a unit reference triangle frames itself too.
  let (centroid, extent) = scene_centroid_and_extent(scene);
  // A mesh flat in the z = 0 plane (the reference cell scenes: nothing has been
  // displaced off it yet) is looked down onto from straight above, along its own
  // normal and in parallel projection, rather than from the angled perspective
  // orbit a fully 3D shape like the sphere wants. Only the pose and the
  // projection differ; the controls are the same camera's either way.
  let z_extent = scene
    .coords
    .coord_iter()
    .map(|c| if c.len() > 2 { c[2].abs() } else { 0.0 })
    .fold(0.0, f64::max);
  let is_planar = z_extent < 1e-9 * extent;
  // Straight down ($theta = -pi/2$) is an ordinary pose, reachable and framed
  // like any other -- it is the pole the old `look_at` camera could not go to,
  // and having to stop short of it is what made top-down a mode of its own.
  // `yaw = pi/2` puts screen-right on world $+x$ (`Camera::right`), so a plane
  // seen from above keeps its own axes; the 3D default's $-pi/2$ is the
  // mirrored convention it has always had.
  let (yaw, pitch) = if is_planar {
    (std::f32::consts::FRAC_PI_2, -std::f32::consts::FRAC_PI_2)
  } else {
    (-std::f32::consts::FRAC_PI_2, 0.3)
  };

  let mut camera = Camera::new(aspect);
  camera.yaw = yaw;
  camera.pitch = pitch;
  camera.pivot_distance = 3.0 * extent as f32;
  camera.orthographic = is_planar;
  // The eye is the state and the pivot is derived, so the framing is stated
  // that way round: back off the centroid along the view by the framing
  // distance.
  let centroid = nalgebra::Point3::new(centroid[0] as f32, centroid[1] as f32, centroid[2] as f32);
  camera.eye = centroid - camera.forward() * camera.pivot_distance;
  // Fractions of the scene's own extent, not absolute constants: an OBJ loaded
  // at an arbitrary scale must clip and clamp at the same fraction a unit
  // sphere does, not at whatever constant happened to fit that sphere.
  camera.znear = 1e-3 * extent as f32;
  camera.zfar = 1e3 * extent as f32;
  camera
}

/// The scene's geometry on the GPU: what a mesh bakes to, and nothing a field
/// decides. Rebuilt when the scene changes, never when the field does.
pub(crate) struct MeshDisplay {
  /// The filled surface, absent for a bake with no fill (a curve, a point
  /// cloud), whose only mark is its segments.
  surface: Option<SurfaceBatch>,
  /// The 1-skeleton overlay of a surface, or a 1-manifold's own cells. Empty for
  /// a bake with neither, which draws nothing rather than being a case to
  /// exclude.
  segments: SegmentBatch,
  /// The 0-skeleton: one billboard disc per mesh vertex. Always present (every
  /// pure complex has a 0-skeleton), shown only when the view asks.
  points: PointBatch,
  /// The bake kept CPU-side, for the picking the camera's pivot needs. Held
  /// rather than rebuilt per pick: it is already what was uploaded, so a second
  /// bake could only disagree with what is on screen.
  baked: BakedMesh,
  /// The deposit atlas layout: a function of the mesh and its metric alone
  /// (like the bake's static half), so it lives here; the trail *state* over
  /// it is a field's, and is built by [`FieldDisplay`]. Empty away from
  /// intrinsic dimension 2.
  deposit_layout: DepositLayout,
}

impl MeshDisplay {
  pub(crate) fn build(device: &wgpu::Device, scene: &Scene) -> Self {
    let baked = BakedMesh::new(&scene.topology, &scene.coords);
    let deposit_layout = DepositLayout::new(&scene.topology, &scene.coords);
    let deposit_uvs = match &baked.cells {
      crate::bake::PrimBatch::Triangles(triangles) => {
        deposit_layout.corner_uvs(&scene.topology, triangles)
      }
      _ => Vec::new(),
    };
    let vertices = baked.segment_vertices();
    let heights = vec![0.0; vertices.len()];
    let segments = match &baked.cells {
      crate::bake::PrimBatch::Segments(cells) => cells.as_slice(),
      _ => &baked.wireframe,
    };
    let seg_zeros = vec![0.0f32; segments.len()];
    let vertex_zeros = vec![0.0f32; vertices.len()];
    Self {
      surface: SurfaceBatch::new(device, &baked, &deposit_uvs),
      segments: SegmentBatch::new(
        device,
        &vertices,
        &heights,
        [&seg_zeros, &seg_zeros],
        segments,
      ),
      points: PointBatch::new(device, &vertices, &vertex_zeros, &vertex_zeros),
      baked,
      deposit_layout,
    }
  }

  /// The 1-skeleton's edges (as pairs of mesh-vertex indices): the cells of a
  /// 1-manifold, the overlay otherwise. What the segment marks are drawn over,
  /// and what their per-edge trace colors are read on.
  pub(crate) fn segments(&self) -> &[[u32; 2]] {
    match &self.baked.cells {
      crate::bake::PrimBatch::Segments(cells) => cells,
      _ => &self.baked.wireframe,
    }
  }

  /// Where a world-space ray meets the mesh, as a distance along it. `None` on
  /// a miss, which every caller must have an answer for -- a curve and a point
  /// cloud have no surface to hit at all.
  pub(crate) fn raycast(&self, origin: na::Point3<f32>, dir: na::Vector3<f32>) -> Option<f32> {
    self.baked.raycast(origin.coords, dir)
  }

  /// The per-vertex displacement ceiling the bake derived from the mesh's
  /// reach: what [`safe_amplitude`] scales the field against.
  pub(crate) fn displacement_ceilings(&self) -> impl Iterator<Item = f32> + '_ {
    self.baked.positions.iter().map(|v| v.max_displacement)
  }

  /// The rendered triangles' cell-corner map, for reading a field per corner in
  /// its own cell (see [`crate::scene::surface_corner_values`]).
  pub(crate) fn cell_corners(&self) -> &[crate::bake::CellCorner] {
    &self.baked.cell_corners
  }

  /// Rebinds the mesh to a different field: one buffer write per stream, no
  /// rebake. The colormap value goes per corner (cell-local); the displacement
  /// height goes per mesh vertex to both the surface and the wireframe riding
  /// its wave.
  pub(crate) fn write_attributes(&self, queue: &wgpu::Queue, attributes: &FieldAttributes) {
    if let Some(surface) = &self.surface {
      surface.write_attributes(queue, &attributes.color, &attributes.surface_height);
    }
    self.segments.write_attributes(
      queue,
      &attributes.wire_height,
      [&attributes.segment_colors[0], &attributes.segment_colors[1]],
    );
    self
      .points
      .write_attributes(queue, &attributes.wire_height, &attributes.point_colors);
  }
}

/// The field streams a [`FieldDisplay`] hands back for the GPU. Split because
/// they answer different questions -- the honest (discontinuous) field readout,
/// the height the fill's cells ride, and the single-valued height the shared
/// 1-skeleton must ride -- and they coincide only for a genuine 0-form.
pub(crate) struct FieldAttributes {
  /// Per rendered corner (three per triangle), in the bake's triangle order.
  pub(crate) color: Vec<f32>,
  /// Per rendered corner, in the same order as [`Self::color`]: continuous
  /// where the field is, constant per cell where it is not
  /// ([`crate::scene::surface_corner_heights`]).
  pub(crate) surface_height: Vec<f32>,
  /// Per mesh vertex, for the segment marks: the 1-skeleton is shared and
  /// cannot tear, so it rides the continuous nodal recovery at every grade.
  pub(crate) wire_height: Vec<f32>,
  /// Per segment endpoint (two parallel arrays), for the 1-skeleton's colormap:
  /// the trace of the field on each edge ([`crate::scene::segment_colors`]).
  pub(crate) segment_colors: [Vec<f32>; 2],
  /// Per mesh vertex, for the 0-skeleton's colormap: the field's value there
  /// ([`crate::scene::point_colors`]).
  pub(crate) point_colors: Vec<f32>,
}

/// The largest *uniform* amplitude at which no vertex displaces past its own
/// reach: $A = min_v ("ceiling"_v \/ max_t |h_v (t)|)$.
///
/// A single global scalar, and that is the whole point. The per-vertex clamp in
/// the shader can only bound the displacement by discarding the field's shape
/// where it binds, and it binds at a *different* value at every vertex -- so on
/// a mesh whose reach varies (any real shape: thin ears, sharp creases, a
/// smooth belly) it flattens the mode in patches and leaves a visible seam
/// between clamped and unclamped neighbours. That is not a bound on the
/// deformation, it is a different deformation.
///
/// Scaling instead is the one operation an eigenmode is indifferent to: it is
/// defined up to a scalar, so a global $A$ changes nothing about *which* mode is
/// shown, only how far it swings. The displaced surface stays exactly $x + A f
/// (x) n(x)$ with the field's own shape intact, and the reach bound
/// ([`vertex_reach`](simplicial::geometry::coord::vertex_reach)) is what makes
/// that map an embedding -- no fold, no self-intersection.
///
/// The maximum is over the field's whole evolution, not its representative
/// frame: a trajectory's later frame can exceed its first, and an amplitude fit
/// to frame zero would clamp exactly when the wave gets interesting. A standing
/// wave rides $cos(sqrt(lambda) t) in [-1, 1]$, so its representative *is* its
/// peak.
///
/// With this in force the shader's clamp is a guard that does not fire in
/// normal operation, rather than the mechanism -- it stays for the case no
/// bound anticipates (a mesh whose reach the estimator could not resolve).
fn safe_amplitude(scene: &Scene, mesh: &MeshDisplay, selection: Selection) -> f32 {
  amplitude_bound(
    mesh.displacement_ceilings(),
    &peak_heights(scene, selection),
  )
}

/// The bound itself, over the per-vertex ceilings and peaks: `INFINITY` where
/// nothing constrains it (a field that vanishes, a mesh with no reach bound),
/// which composes correctly with the aesthetic ceiling the caller mins against.
fn amplitude_bound(ceilings: impl Iterator<Item = f32>, peaks: &[f32]) -> f32 {
  ceilings
    .zip(peaks)
    .filter(|(_, &peak)| peak > 1e-12)
    .map(|(ceiling, &peak)| ceiling / peak)
    .fold(f32::INFINITY, f32::min)
}

/// The peak $max_t |h_v (t)|$ of the displacement height at every vertex, over
/// the field's whole time evolution. The nodal (continuous) height throughout:
/// it is what the segments ride, and it bounds the per-cell rigid height too,
/// being an average of the same per-cell values.
fn peak_heights(scene: &Scene, selection: Selection) -> Vec<f32> {
  let nvertices = scene.coords.nvertices();
  let cochain = scene.field_cochain(selection);
  let mut peaks = vec![0.0f32; nvertices];
  let mut absorb = |cochain: &derham::cochain::Cochain| {
    let heights = crate::scene::nodal_heights(&scene.topology, &scene.coords, cochain);
    for (peak, h) in peaks.iter_mut().zip(heights) {
      *peak = peak.max(h.abs() as f32);
    }
  };
  match scene.field_time(selection) {
    crate::scene::FieldTime::Trajectory { frames, .. } => frames.iter().for_each(&mut absorb),
    _ => absorb(cochain),
  }
  peaks
}

/// How long a [`crate::scene::FieldTime::Trajectory`] takes to play through
/// once, in wall-clock seconds, before looping. A fixed span rather than the
/// solve's own final time: the physical $dif t$ is set by stability (CFL), not
/// by what reads well, so the playback maps that interval onto a watchable one.
pub(crate) const TRAJECTORY_LOOP_SECONDS: f64 = 6.0;

/// The two field streams for a cochain -- the per-corner cell-local colormap
/// value and the per-vertex continuous displacement height -- together with the
/// colormap's raw range. Extracted from [`FieldDisplay::build`] so a trajectory's
/// caller can recompute them per frame from an interpolated frame and rewrite
/// them into the mesh, which is the whole of scrubbing a trajectory.
pub(crate) fn field_attributes(
  topology: &simplicial::topology::complex::Complex,
  coords: &simplicial::geometry::coord::mesh::MeshCoords,
  cochain: &derham::cochain::Cochain,
  cell_corners: &[crate::bake::CellCorner],
  segments: &[[u32; 2]],
) -> (FieldAttributes, FieldRanges) {
  let colors = crate::scene::surface_corner_values(topology, coords, cochain, cell_corners);
  let surface_heights =
    crate::scene::surface_corner_heights(topology, coords, cochain, cell_corners);
  let wire_heights = crate::scene::nodal_heights(topology, coords, cochain);
  let segment_colors = crate::scene::segment_colors(topology, coords, cochain, segments);
  let point_colors = crate::scene::point_colors(topology, coords, cochain);
  // Each skeleton normalizes against its own values: a k-skeleton traces a
  // different-grade density on a different scale, so a shared range would
  // flatten one to read the other.
  let fill = crate::scene::corner_bounds(&colors);
  let segment =
    crate::scene::corner_bounds(&[segment_colors[0].clone(), segment_colors[1].clone()].concat());
  let point = crate::scene::corner_bounds(&point_colors);
  (
    FieldAttributes {
      color: bake::attributes(&colors),
      surface_height: bake::attributes(&surface_heights),
      wire_height: bake::attributes(&wire_heights),
      segment_colors: [
        bake::attributes(&segment_colors[0]),
        bake::attributes(&segment_colors[1]),
      ],
      point_colors: bake::attributes(&point_colors),
    },
    FieldRanges {
      fill,
      segment,
      point,
    },
  )
}

/// The colormap range of each skeleton that reflects the field, kept separate
/// because they measure different-grade densities on different scales.
pub(crate) struct FieldRanges {
  pub(crate) fill: (f32, f32),
  pub(crate) segment: (f32, f32),
  pub(crate) point: (f32, f32),
}

/// The material parameters for showing one field of a scene, and the geometry
/// only that field has: the one place a [`Selection`] turns into something
/// drawable. Everything here is static per field -- the renderer only re-times
/// it per frame.
///
/// The field's own half of the bake's vertex split -- its attribute stream --
/// is *returned* by [`Self::build`] rather than written from inside it: a
/// [`MeshDisplay`] is the static half, the two halves meet on the GPU, and
/// which of them does the writing is the caller's to see rather than a
/// constructor's to hide.
pub(crate) struct FieldDisplay {
  /// The arrow glyphs of a line field, `None` for a scalar field: the field
  /// evaluated -- at points the atlas places (the barycentric lattice of each
  /// cell, boundary included: see [`crate::glyph`]) rather than a tracer's
  /// seeding or a population's respawn.
  glyphs: Option<GlyphBatch>,
  /// The advected particles of a line field, absent for a scalar field and for
  /// a field that vanishes everywhere (which seeds nowhere).
  ///
  /// The same reduced grade-1 field the glyphs evaluate, read the other way:
  /// the glyphs are the field's value, standing still, and the particles are
  /// its dynamics, integrated on the GPU through the same charts and
  /// transitions.
  particles: Option<ParticleBatch>,
  /// The trails those particles lay down: the deposit atlas, stepped with the
  /// advection and read by the fill as illumination. Present exactly when the
  /// particles are and the mesh carries an atlas (intrinsic dimension 2).
  deposit: Option<DepositBatch>,
  surface: SurfaceMaterial,
  wireframe: SegmentMaterial,
  /// The 0-skeleton's material: a `SegmentMaterial` too, since a point mark and
  /// a segment mark differ only in primitive, not in what drives them.
  points: SegmentMaterial,
  glyph: GlyphMaterial,
}

impl FieldDisplay {
  /// The field's display and the attribute streams it decides -- which a caller
  /// writes into a [`MeshDisplay`] with [`MeshDisplay::write_attributes`].
  /// Returned rather than written here: see the type's own doc.
  pub(crate) fn build(
    ctx: &GpuContext,
    scene: &Scene,
    mesh: &MeshDisplay,
    selection: Selection,
    amplitude_scale: f32,
  ) -> (Self, FieldAttributes) {
    // The field is read once per rendered corner, each in its own cell, so a
    // reduced-grade Whitney form's discontinuity across cells reaches the
    // colormap intact -- a basis function's support ends on cell edges and does
    // not bleed into the cells it vanishes on. The displacement height is the
    // continuous nodal recovery of that same value, single-valued per vertex so
    // the surface does not tear. Both are functions of the field, computed once
    // here for either mark; the colormap range follows from the colors so it
    // spans exactly what is drawn.
    let cochain = match selection {
      Selection::Scalar(index) => &scene.fields[index].cochain,
      Selection::Line(index) => &scene.line_fields[index].cochain,
    };
    let (attributes, ranges) = field_attributes(
      &scene.topology,
      &scene.coords,
      cochain,
      mesh.cell_corners(),
      mesh.segments(),
    );
    let (raw_min, raw_max) = ranges.fill;

    // The mesh's own local length, which the marks that draw its features are
    // sized by -- as against `amplitude_scale`, the object's global extent,
    // which sizes what should read the same at any resolution. A mesh with no
    // edges has no local length, and falls back on the global one.
    let mesh_scale = {
      let mean = simplicial::geometry::coord::mean_edge_length(&scene.topology, &scene.coords);
      if mean > 0.0 {
        mean as f32
      } else {
        amplitude_scale
      }
    };

    let (glyphs, particles, speed_ratio, mut surface) = match selection {
      Selection::Scalar(index) => {
        let field = &scene.fields[index];

        let field_scale = raw_min.abs().max(raw_max.abs()).max(f32::EPSILON);
        // A field with no eigenvalue is not a standing-wave mode (e.g. a raw
        // Whitney basis function): no dispersion relation to animate at, so
        // the wave collapses to no displacement rather than a special case
        // here.
        let wave_omega = field.time.wave_omega();
        // Two ceilings, and the amplitude is whichever binds first. The
        // aesthetic one normalizes by the field's own peak so every mode
        // reaches the same displacement -- a fraction of the object's extent,
        // not its mesh width, so the lobes read at orbital scale regardless of
        // resolution. The geometric one is the mesh's reach, and it is what
        // keeps a shape with thin features from displacing through itself.
        let wave_amplitude = if field.time.animates() {
          let aesthetic = WAVE_AMPLITUDE_FRACTION * amplitude_scale / field_scale;
          aesthetic.min(safe_amplitude(scene, mesh, selection))
        } else {
          0.0
        };
        // An eigenmode's color pulses by $cos(sqrt(lambda) t)$ through zero, so
        // its colormap range is symmetric $[-s, s]$ about the midpoint -- the
        // same reasoning as the line field's tint. A static field keeps its own
        // asymmetric range.
        let (min_val, max_val) = if field.time.animates() {
          (-field_scale, field_scale)
        } else {
          (raw_min, raw_max)
        };

        (
          None,
          None,
          0.0,
          SurfaceMaterial {
            min_val,
            max_val,
            wave_amplitude,
            wave_omega,
            // Diverging exactly where the range above was widened to
            // symmetric: a signed eigenmode pulse, not an unsigned magnitude.
            diverging: f32::from(field.time.animates()),
            // The identity: a scalar field has no particles and lays no trail.
            deposit_floor: 1.0,
            deposit_gain: 0.0,
            // Overridden per frame from the 2-skeleton's coloring toggle.
            colored: 1.0,
          },
        )
      }
      Selection::Line(index) => {
        let field = &scene.line_fields[index];
        // An eigenmode's tint is the *signed* $|V| cos(sqrt(lambda) t)$, so its
        // colormap range is symmetric $[-m, m]$ about zero -- the pulse runs
        // through the midpoint and flips as the cosine crosses zero. A static
        // field has no such pulse (wave_omega below is 0, cos(0) = 1), so its
        // tint is the unsigned $|V|_g$ itself: using its true range instead of
        // widening to symmetric keeps the colormap from spending half its
        // span on negative values the field never takes. The glyphs are static
        // either way, so there is no geometric displacement --
        // `wave_amplitude` is 0 and only `wave_omega` (the tint clock) carries
        // the mode's frequency.
        let peak = raw_max.abs().max(raw_min.abs()).max(f32::EPSILON);
        let (min_val, max_val) = if field.time.animates() {
          (-peak, peak)
        } else {
          (raw_min, raw_max)
        };

        // The particles flow the same field, on the object's own clock: the
        // peak magnitude sets what "one step" is worth, so the fastest speck
        // covers `PARTICLE_SPEED_FRACTION` of the object's radius each second
        // whatever the cochain's units. A field that vanishes everywhere has no
        // scale to divide by and simply does not move.
        let peak = crate::advect::peak_speed(&scene.topology, &scene.coords, &field.cochain);

        // The same peak the advection normalizes by, read as the glyph fade's
        // reference: a glyph's opacity is its own magnitude against the field's
        // greatest, so the mark reports where the field is strong without
        // spending its length on it.
        let vertices = crate::glyph::bake_glyphs(
          &scene.topology,
          &scene.coords,
          &field.cochain,
          f64::from(GLYPH_SPACING_FRACTION * amplitude_scale),
          peak,
        );
        let glyphs = GlyphBatch::new(&ctx.device, &vertices);

        let particles = (peak > 0.0)
          .then(|| {
            let step = PARTICLE_SPEED_FRACTION * f64::from(amplitude_scale)
              / peak
              / f64::from(STEPS_PER_SECOND);
            let bake = crate::advect::AdvectBake::new(
              &scene.topology,
              &scene.coords,
              &field.cochain,
              step,
              ADVECT_DEPTH,
              PARTICLE_SEEDS,
            );
            ParticleBatch::new(&ctx.device, &bake, PARTICLE_COUNT)
          })
          .flatten();

        // The population's mean speed as a fraction of the peak the step is
        // normalized to: with splats inked by arc length, this is what the
        // equilibrium trail brightness actually scales with, and it is an
        // exact area-weighted quantity of the field, not a tuned ratio.
        let speed_ratio = if peak > 0.0 {
          crate::advect::mean_speed(&scene.topology, &scene.coords, &field.cochain) / peak
        } else {
          0.0
        };

        (
          Some(glyphs),
          particles,
          speed_ratio,
          // The surface is the same fill a scalar field gets, tinted by the
          // field's cell-local magnitude: the glyphs carry the direction, so the
          // surface has only the magnitude left to say.
          SurfaceMaterial {
            min_val,
            max_val,
            wave_amplitude: 0.0,
            wave_omega: field.time.wave_omega(),
            diverging: f32::from(field.time.animates()),
            // The identity until the deposit below exists; patched then.
            deposit_floor: 1.0,
            deposit_gain: 0.0,
            // Overridden per frame from the 2-skeleton's coloring toggle.
            colored: 1.0,
          },
        )
      }
    };

    // The trails, where both halves exist: a population to lay them and an
    // atlas to hold them. The decay is per step -- the deposit's determinism
    // contract is the advection's own -- and the fill's gain is *calibrated*,
    // not tuned: splats ink by arc length, so the equilibrium a uniformly
    // spread population reaches is count times one splat's texels times the
    // mean step length in texels times the decay's lifetime, over the texels
    // the atlas covers -- and at that mean the lift is [`DEPOSIT_MEAN_LIFT`].
    // Every factor is an exact quantity of the field, the population or the
    // layout; whatever the count or budget becomes, the picture keeps its
    // exposure.
    let decay = (-1.0 / (DEPOSIT_DECAY_SECONDS * STEPS_PER_SECOND)).exp();
    let deposit = particles.as_ref().and_then(|population| {
      let layout = &mesh.deposit_layout;
      let trails = DepositBatch::new(&ctx.device, layout, population, ADVECT_DEPTH, 1.0, decay)?;
      // The mean step length in texels: the peak-normalized step in world
      // units, scaled by the field's own mean/peak ratio, over the layout's
      // world size of one texel (uniform per area by construction).
      let step_world_peak =
        PARTICLE_SPEED_FRACTION * f64::from(amplitude_scale) / f64::from(STEPS_PER_SECOND);
      let total_area: f64 = scene
        .topology
        .cells()
        .handle_iter()
        .map(|cell| scene.coords.cell_metric(cell).det_sqrt() / 2.0)
        .sum();
      let used_texels = layout.used_texels().max(1) as f64;
      let world_per_texel = (total_area / used_texels).sqrt().max(f64::EPSILON);
      let mean_step_texels = step_world_peak * speed_ratio / world_per_texel;

      let lifetime_steps = f64::from(DEPOSIT_DECAY_SECONDS) * f64::from(STEPS_PER_SECOND);
      let mean_deposit = f64::from(population.count())
        * crate::render::deposit::splat_footprint_integral()
        * mean_step_texels
        * lifetime_steps
        / used_texels;
      surface.deposit_floor = DEPOSIT_FLOOR;
      surface.deposit_gain = ((f64::from(DEPOSIT_MEAN_LIFT) - f64::from(DEPOSIT_FLOOR))
        / mean_deposit.max(f64::EPSILON)) as f32;
      Some(trails)
    });

    // A skeleton's colormap range, by the same rule the fill uses: an eigenmode
    // pulses through zero, so its range is symmetric; a static field keeps its
    // own. Diverging where the trace is signed (values cross zero -- a 0-form, or
    // the manifold top form) or the mode pulses. One rule for every colored
    // skeleton, applied to each one's own range.
    let animates = scene.field_time(selection).animates();
    let colormap_range = |(raw_min, raw_max): (f32, f32)| -> (f32, f32, f32) {
      let (min_val, max_val) = if animates {
        let s = raw_min.abs().max(raw_max.abs()).max(f32::EPSILON);
        (-s, s)
      } else {
        (raw_min, raw_max)
      };
      (min_val, max_val, f32::from(animates || raw_min < 0.0))
    };
    let (segment_min, segment_max, segment_diverging) = colormap_range(ranges.segment);
    let (point_min, point_max, point_diverging) = colormap_range(ranges.point);

    let display = Self {
      glyphs,
      particles,
      deposit,
      surface,
      // The wireframe rides the surface's own wave, so it tracks the displaced
      // mesh rather than the flat rest one, and it has no node to fade at. Its
      // colormap range is the 1-skeleton's own (a diverging map when the trace
      // is signed -- a 0-form, or the pulse through zero -- else sequential);
      // `colored` is left off here and set per frame from the view toggle, since
      // whether the skeleton reflects the field is a view choice, not the
      // field's.
      wireframe: SegmentMaterial {
        color: [0.0, 0.0, 0.0, 1.0],
        half_width_world: SKELETON_WIDTH_FRACTION * mesh_scale,
        fade_floor: 1.0,
        wave_amplitude: surface.wave_amplitude,
        wave_omega: surface.wave_omega,
        min_val: segment_min,
        max_val: segment_max,
        diverging: segment_diverging,
        colored: 0.0,
      },
      // The 0-skeleton rides the same wave and clock; its disc is a few edge
      // fractions wide, and its colormap range is its own. `colored`, like the
      // wireframe's, is the view's to set per frame.
      points: SegmentMaterial {
        color: [0.0, 0.0, 0.0, 1.0],
        half_width_world: SKELETON_WIDTH_FRACTION * mesh_scale,
        fade_floor: 1.0,
        wave_amplitude: surface.wave_amplitude,
        wave_omega: surface.wave_omega,
        min_val: point_min,
        max_val: point_max,
        diverging: point_diverging,
        colored: 0.0,
      },
      // The glyphs share the surface's clock but not its displacement: the
      // samples sit on the undisplaced surface, so only the node fade reads the
      // mode. The outline rides the same material -- `glyph.wgsl` composites the
      // rim under the ink in one pass. Every dimension here is a proportion of
      // the arrow's own length, which the bake sets per cell, so the mark is
      // self-similar at any refinement and none of it is a world size.
      // The clip to the cell is intrinsic to the flat mark and needs no flag.
      glyph: GlyphMaterial {
        color: GLYPH_INK,
        width_fraction: GLYPH_WIDTH_FRACTION,
        fade_floor: GLYPH_NODE_OPACITY,
        wave_omega: surface.wave_omega,
        head_length_fraction: GLYPH_HEAD_LENGTH_FRACTION,
        shaft_width_fraction: GLYPH_SHAFT_WIDTH_FRACTION,
        outline_width_fraction: GLYPH_OUTLINE_WIDTH_FRACTION,
        _pad0: 0.0,
        _pad1: 0.0,
      },
    };
    (display, attributes)
  }

  /// The frame's items, in submission order: the surface writes depth, and the
  /// marks over it -- a line field's glyphs, then the wireframe -- only test
  /// against it, so they blend in the order given.
  ///
  /// The advected population is not among them: it is never on screen. It flows
  /// the field and lays its trail into the deposit atlas the surface reads, so
  /// it rides beside the items, stepped and never drawn.
  ///
  /// The two views enter the way the two objects do. A setting that is off drops
  /// an item rather than switching the frame graph, which the renderer cannot
  /// tell from a field that never had one; the `Option`s stay the structural
  /// truth beneath it, since a mark with no batch is unavailable whatever the
  /// bool says. Displacement is the one that drops nothing: it is a material,
  /// so its "off" is an amplitude of zero -- the same zero [`Self::build`]
  /// already gives a field with no eigenvalue, and the same shape as bloom's
  /// intensity of zero. Either way the toggle costs no branch below this line.
  pub(crate) fn draw_list<'a>(
    &'a self,
    mesh: &'a MeshDisplay,
    mesh_view: MeshView,
    field_view: FieldView,
  ) -> DrawList<'a> {
    // The wireframe rides the surface's wave, so the two amplitudes are one
    // setting; the glyphs sample the undisplaced surface and carry a zero
    // amplitude already.
    let (mut surface, mut wireframe, mut points) = (self.surface, self.wireframe, self.points);
    if !field_view.displacement {
      surface.wave_amplitude = 0.0;
      wireframe.wave_amplitude = 0.0;
      points.wave_amplitude = 0.0;
    }
    // Whether a skeleton reflects the field is a view choice, applied here
    // rather than baked into the material: the range is the field's, the mode
    // is the reader's. Uniform across the three skeletons, faces included.
    surface.colored = f32::from(mesh_view.skeleton(2).colored);
    wireframe.colored = f32::from(mesh_view.skeleton(1).colored);
    points.colored = f32::from(mesh_view.skeleton(0).colored);
    // The population is stepped only when the flow is shown, and the trail
    // follows it: without the population the atlas is neither stepped nor bound,
    // so the material reverts to the identity rather than dimming the fill to a
    // floor no trail will ever lift.
    let particles = self
      .particles
      .as_ref()
      .filter(|_| field_view.marks.particles);
    let deposit = self.deposit.as_ref().filter(|_| particles.is_some());
    if deposit.is_none() {
      surface.deposit_floor = 1.0;
      surface.deposit_gain = 0.0;
    }

    let mut items = Vec::new();
    if let Some(batch) = mesh
      .surface
      .as_ref()
      .filter(|_| mesh_view.skeleton(2).visible)
    {
      items.push(RenderItem::Surface(batch, surface));
    }
    if let Some(glyphs) = self.glyphs.as_ref().filter(|_| field_view.marks.glyphs) {
      items.push(RenderItem::Glyphs(glyphs, self.glyph));
    }
    if mesh_view.skeleton(1).visible {
      items.push(RenderItem::Segments(&mesh.segments, wireframe));
    }
    if mesh_view.skeleton(0).visible {
      items.push(RenderItem::Points(&mesh.points, points));
    }
    DrawList {
      items,
      particles,
      deposit,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use simplicial::geometry::coord::{vertex_curvature_radius, vertex_reach};

  /// The law the amplitude bound exists to enforce: at the chosen amplitude no
  /// vertex displaces past its own reach, so the deformation stays an
  /// embedding. Checked where it actually bites -- a slab thin enough that the
  /// bound is set by its thickness -- and stated as the displacement, not as
  /// the scalar, since that is the quantity that folds a surface.
  #[test]
  fn no_vertex_displaces_past_its_reach() {
    let thickness = 0.04;
    let (topology, coords) = slab(thickness);
    let cochain = derham::cochain::Cochain::new(
      0,
      na::DVector::from_fn(coords.nvertices(), |i, _| {
        // An arbitrary sign-changing field: what matters is that it is not
        // constant, so the bound is set somewhere specific.
        ((i as f64) * 0.7).sin()
      }),
    );
    let heights = crate::scene::nodal_heights(&topology, &coords, &cochain);
    let peaks: Vec<f32> = heights.iter().map(|h| h.abs() as f32).collect();

    let extent = 1.0;
    let reach = vertex_reach(&topology, &coords, extent);
    let ceilings: Vec<f32> = reach.iter().map(|r| (0.9 * r) as f32).collect();

    let amplitude = amplitude_bound(ceilings.iter().copied(), &peaks);
    assert!(amplitude.is_finite(), "the bound must bind on a thin slab");

    for (i, (&peak, &ceiling)) in peaks.iter().zip(&ceilings).enumerate() {
      assert!(
        amplitude * peak <= ceiling + 1e-6,
        "vertex {i} displaces {} past its ceiling {ceiling}",
        amplitude * peak
      );
    }

    // And the bound is the *thickness*, not the curvature. The faces are flat,
    // so a curvature-only ceiling permits a displacement exceeding the slab's
    // own half-thickness -- which is the two faces passing through each other,
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
  /// rather than dividing by zero -- the caller's aesthetic ceiling then
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
    simplicial::geometry::coord::mesh::MeshCoords,
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
      simplicial::geometry::coord::mesh::MeshCoords::from(Matrix::from_columns(&pts)),
    )
  }
}
