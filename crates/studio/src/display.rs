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
use crate::render::{
  camera::Camera,
  item::{DrawList, RenderItem, SegmentBatch, SurfaceBatch},
  particles::ParticleBatch,
  uniform::{ParticleMaterial, PostUniform, SegmentMaterial, SurfaceMaterial},
  GpuContext,
};
use crate::scene::Scene;
use crate::ui::{Marks, Post, Selection};

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

/// The wireframe's half-width as a fraction of the scene's own extent (its
/// radius) -- the same object-intrinsic scale the standing-wave displacement
/// uses -- rather than a fixed screen-pixel count, so the line reads the same
/// thickness whether the mesh is zoomed to fill the screen or shrunk to a corner
/// of it.
const WIREFRAME_WIDTH_FRACTION: f32 = 0.004;

/// The streamline ribbon's half-width, on the same object-intrinsic scale.
const STREAMLINE_WIDTH_FRACTION: f32 = 0.005;

/// The streamline ribbons' ink: pure white, single pass, no outline.
const STREAMLINE_INK: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// The arrow glyph's length, on the same object-intrinsic scale as every other
/// mark's: uniform across the mark, since the glyph carries the direction and
/// the fill beneath it carries the magnitude (see `glyph.rs`). Longer than the
/// ribbons are wide by an order of magnitude, because a glyph has to read as an
/// arrow -- a shaft and a head, both legible -- and not as a thick dot.
const GLYPH_LENGTH_FRACTION: f32 = 0.06;

/// The glyph's half-width: the arrowhead's, which its base spans in full, and
/// which [`GLYPH_SHAFT_WIDTH_FRACTION`] narrows the shaft down from.
const GLYPH_WIDTH_FRACTION: f32 = 0.012;

/// The arrowhead's length as a fraction of the glyph's own, and the shaft's
/// half-width as a fraction of the head's base. A head a third of the arrow, on
/// a shaft a third of its width: the proportions of a drawn arrow, self-similar
/// at every glyph since both are fractions.
const GLYPH_HEAD_LENGTH_FRACTION: f32 = 0.36;
const GLYPH_SHAFT_WIDTH_FRACTION: f32 = 0.32;

/// The glyphs' ink: the ribbons' white, at reduced opacity so a lattice of them
/// reads as a field rather than as a wall of marks. They are drawn under the
/// ribbons and over the fill.
const GLYPH_INK: [f32; 4] = [1.0, 1.0, 1.0, 0.75];

/// The opacity the ribbons of an eigenmode fade to at the standing wave's node,
/// where the field vanishes and the curves are meaningless -- never fully, since
/// the integral curves of a standing mode are the same set at every phase, and
/// blinking them out entirely would read as the geometry changing.
const STREAMLINE_NODE_OPACITY: f32 = 0.25;

/// The advected particles' radius, on the same object-intrinsic scale as every
/// other mark's width.
///
/// Read together with [`PARTICLE_COUNT`]: the two are one setting. What the eye
/// judges is the light per unit area, which is the count times the speck's own
/// area times its opacity -- so a count raised or lowered without moving these
/// gives a saturated wash or an empty one, and neither is a picture of the
/// field. This is the size at which a speck is individually legible, which is
/// the regime the count below chooses.
const PARTICLE_RADIUS_FRACTION: f32 = 0.0026;

/// The particles' ink: a warm white, additively blended, so overlapping specks
/// accumulate into a bright density rather than occluding one another.
///
/// Below full opacity because a speck is one sample among many and the
/// accumulation is what should read -- and because the target is HDR, so what a
/// pile-up exceeds 1.0 by is what blooms. Opacity here is therefore a *glow*
/// setting as much as a brightness one.
const PARTICLE_INK: [f32; 4] = [1.0, 0.86, 0.62, 0.10];

/// How fast the fastest particle crosses the object: scene radii per second, at
/// the field's peak magnitude.
///
/// Object-intrinsic, exactly as the streamline separation is, so the flow reads
/// at the same pace whether the mesh is a unit sphere or an OBJ at arbitrary
/// scale, and regardless of the cochain's own units. Slow on purpose: a speck
/// that moves a few pixels per frame reads as motion, and one that jumps across
/// the screen reads as flicker. This is the knob that makes a trail-less
/// particle legible at all.
const PARTICLE_SPEED_FRACTION: f64 = 0.10;

/// How many radii a peak-speed speck stretches along its own motion.
///
/// This is motion blur, not a trail: the elongation is derived from the speck's
/// own velocity in the vertex shader and stores nothing. A round speck moving
/// several radii per frame reads as a jumping dot; the same speck drawn along
/// its chord reads as a direction, which is what the mark is for.
const PARTICLE_STRETCH: f32 = 4.0;

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
/// A count where an individual speck can still be followed by eye, and the flow
/// is read from watching them move rather than from the density they pile into.
/// The two regimes are genuinely different pictures, not more or less of one:
/// far higher and no speck is legible, so only the accumulated density carries
/// anything, and the radius and opacity above have to come down to keep it from
/// saturating to a flat wash. Move the count and those two move with it.
///
/// The seeds are far fewer than the particles: a seed is a *place* to be born,
/// not a particle, and many pass through each over a session.
const PARTICLE_COUNT: u32 = 100_000;
const PARTICLE_SEEDS: usize = 16_384;

/// The advection steps that have elapsed by `time`.
///
/// The count is a function of the instant, not an accumulator, so it is the
/// same number for the window and for an exporter aiming at that instant --
/// which is what makes the two agree on where a particle is.
pub(crate) fn steps_at(time: f32) -> u32 {
  (time.max(0.0) * STEPS_PER_SECOND) as u32
}

/// Streamline separation, as a fraction of the scene extent (its radius) --
/// object-intrinsic, like the wave amplitude, so the line density is a property
/// of the object and not of the triangulation. The one knob that sets how dense
/// the evenly spaced curves are.
const STREAMLINE_SEPARATION_FRACTION: f32 = 0.09;

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
  /// The bake kept CPU-side, for the picking the camera's pivot needs. Held
  /// rather than rebuilt per pick: it is already what was uploaded, so a second
  /// bake could only disagree with what is on screen.
  baked: BakedMesh,
}

impl MeshDisplay {
  pub(crate) fn build(device: &wgpu::Device, scene: &Scene) -> Self {
    let baked = BakedMesh::new(&scene.topology, &scene.coords);
    let vertices = baked.segment_vertices();
    let values = vec![0.0; vertices.len()];
    let segments = match &baked.cells {
      crate::bake::PrimBatch::Segments(cells) => cells.as_slice(),
      _ => &baked.wireframe,
    };
    Self {
      surface: SurfaceBatch::new(device, &baked),
      segments: SegmentBatch::new(device, &vertices, &values, segments),
      baked,
    }
  }

  /// Where a world-space ray meets the mesh, as a distance along it. `None` on
  /// a miss, which every caller must have an answer for -- a curve and a point
  /// cloud have no surface to hit at all.
  pub(crate) fn raycast(&self, origin: na::Point3<f32>, dir: na::Vector3<f32>) -> Option<f32> {
    self.baked.raycast(origin.coords, dir)
  }

  /// Rebinds the mesh to a different field: one buffer write per stream, no
  /// rebake.
  pub(crate) fn write_attributes(&self, queue: &wgpu::Queue, attributes: &[f32]) {
    if let Some(surface) = &self.surface {
      surface.write_attributes(queue, attributes);
    }
    self.segments.write_attributes(queue, attributes);
  }
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
  /// The traced streamlines of a line field, `None` for a scalar field. Their
  /// presence *is* the line-field mark: there is no branch to pick.
  streamlines: Option<SegmentBatch>,
  /// The arrow glyphs of a line field, `None` for a scalar field: the same
  /// reduction read a third way. The ribbons integrate the field, the particles
  /// flow it, and the glyphs simply evaluate it -- at points the atlas places
  /// (the interior barycentric lattice of each cell) rather than a tracer's
  /// seeding or a population's respawn.
  glyphs: Option<SegmentBatch>,
  /// The advected particles of a line field, absent for a scalar field and for
  /// a field that vanishes everywhere (which seeds nowhere).
  ///
  /// The same reduced grade-1 field the streamlines trace, read the other way:
  /// the curves are the field's geometry, standing still, and the particles are
  /// its dynamics. Both are intrinsic -- one integrated on the CPU through the
  /// atlas, one on the GPU through the same charts and transitions -- so they
  /// are two marks on one reduction, not a mark and its screen-space twin.
  particles: Option<ParticleBatch>,
  surface: SurfaceMaterial,
  wireframe: SegmentMaterial,
  streamline: SegmentMaterial,
  glyph: SegmentMaterial,
  particle: ParticleMaterial,
}

impl FieldDisplay {
  /// The field's display and the attribute stream it decides, one scalar per
  /// baked vertex -- which a caller writes into a [`MeshDisplay`] with
  /// [`MeshDisplay::write_attributes`]. Returned rather than written here: see
  /// the type's own doc.
  pub(crate) fn build(
    ctx: &GpuContext,
    scene: &Scene,
    selection: Selection,
    amplitude_scale: f32,
  ) -> (Self, Vec<f32>) {
    let (streamlines, glyphs, particles, attributes, surface) = match selection {
      Selection::Scalar(index) => {
        let field = &scene.fields[index];
        let (raw_min, raw_max) = field.bounds();

        let field_scale = raw_min.abs().max(raw_max.abs()).max(f32::EPSILON);
        // A field with no eigenvalue is not a standing-wave mode (e.g. a raw
        // Whitney basis function): no dispersion relation to animate at, so
        // the wave collapses to no displacement rather than a special case
        // here.
        let wave_omega = field.eigenvalue.map_or(0.0, f64::sqrt) as f32;
        // Normalized by the field's own peak so every mode reaches the same peak
        // displacement -- a fraction of the object's extent, not its mesh width,
        // so the lobes read at orbital scale regardless of resolution.
        let wave_amplitude = if field.eigenvalue.is_some() {
          WAVE_AMPLITUDE_FRACTION * amplitude_scale / field_scale
        } else {
          0.0
        };
        // An eigenmode's color pulses by $cos(sqrt(lambda) t)$ through zero, so
        // its colormap range is symmetric $[-s, s]$ about the midpoint -- the
        // same reasoning as the line field's tint. A static field keeps its own
        // asymmetric range.
        let (min_val, max_val) = if field.eigenvalue.is_some() {
          (-field_scale, field_scale)
        } else {
          (raw_min, raw_max)
        };

        (
          None,
          None,
          None,
          bake::attributes(field.values()),
          SurfaceMaterial {
            min_val,
            max_val,
            wave_amplitude,
            wave_omega,
            // Diverging exactly where the range above was widened to
            // symmetric: a signed eigenmode pulse, not an unsigned magnitude.
            diverging: f32::from(field.eigenvalue.is_some()),
          },
        )
      }
      Selection::Line(index) => {
        let field = &scene.line_fields[index];
        // The integral curves of the true Whitney field, traced on the manifold
        // at a separation fixed to the object's own extent (not its mesh width).
        let d_sep = f64::from(STREAMLINE_SEPARATION_FRACTION * amplitude_scale);
        let traced =
          crate::streamline::trace(&scene.topology, &scene.coords, &field.cochain, d_sep);
        let (vertices, values, segments) = bake::bake_streamlines(&traced);
        let streamlines = SegmentBatch::new(&ctx.device, &vertices, &values, &segments);

        let (raw_min, raw_max) = field.bounds();
        // An eigenmode's tint is the *signed* $|V| cos(sqrt(lambda) t)$, so its
        // colormap range is symmetric $[-m, m]$ about zero -- the pulse runs
        // through the midpoint and flips as the cosine crosses zero. A static
        // field has no such pulse (wave_omega below is 0, cos(0) = 1), so its
        // tint is the unsigned $|V|_g$ itself: using its true range instead of
        // widening to symmetric keeps the colormap from spending half its
        // span on negative values the field never takes. The curves are static
        // either way, so there is no geometric displacement --
        // `wave_amplitude` is 0 and only `wave_omega` (the tint clock) carries
        // the mode's frequency.
        let peak = raw_max.abs().max(raw_min.abs()).max(f32::EPSILON);
        let (min_val, max_val) = if field.eigenvalue.is_some() {
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
        let (vertices, values, segments) = crate::glyph::bake_glyphs(
          &scene.topology,
          &scene.coords,
          &field.cochain,
          f64::from(GLYPH_LENGTH_FRACTION * amplitude_scale),
          peak,
        );
        let glyphs = SegmentBatch::new(&ctx.device, &vertices, &values, &segments);

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

        (
          Some(streamlines),
          Some(glyphs),
          particles,
          // The surface is the same fill a scalar field gets, tinted by the
          // field's nodal magnitude: the curves carry the direction, so the
          // surface has only the magnitude left to say.
          bake::attributes(&field.magnitude),
          SurfaceMaterial {
            min_val,
            max_val,
            wave_amplitude: 0.0,
            wave_omega: field.eigenvalue.map_or(0.0, f64::sqrt) as f32,
            diverging: f32::from(field.eigenvalue.is_some()),
          },
        )
      }
    };

    let display = Self {
      streamlines,
      glyphs,
      particles,
      surface,
      // The wireframe rides the surface's own wave, so it tracks the displaced
      // mesh rather than the flat rest one, and it has no node to fade at.
      wireframe: SegmentMaterial {
        color: [0.0, 0.0, 0.0, 1.0],
        half_width_world: WIREFRAME_WIDTH_FRACTION * amplitude_scale,
        fade_floor: 1.0,
        wave_amplitude: surface.wave_amplitude,
        wave_omega: surface.wave_omega,
        ..SegmentMaterial::PLAIN
      },
      // The ribbons share the wave's clock but not its displacement: the samples
      // sit on the undisplaced surface, so only the node fade reads the mode.
      streamline: SegmentMaterial {
        color: STREAMLINE_INK,
        half_width_world: STREAMLINE_WIDTH_FRACTION * amplitude_scale,
        fade_floor: STREAMLINE_NODE_OPACITY,
        wave_amplitude: 0.0,
        wave_omega: surface.wave_omega,
        ..SegmentMaterial::PLAIN
      },
      // The glyphs share the ribbons' clock and node fade -- they are the same
      // field, so they vanish where it does -- and differ only in the taper that
      // makes a segment an arrow.
      glyph: SegmentMaterial {
        color: GLYPH_INK,
        half_width_world: GLYPH_WIDTH_FRACTION * amplitude_scale,
        fade_floor: STREAMLINE_NODE_OPACITY,
        wave_amplitude: 0.0,
        wave_omega: surface.wave_omega,
        head_length_fraction: GLYPH_HEAD_LENGTH_FRACTION,
        shaft_width_fraction: GLYPH_SHAFT_WIDTH_FRACTION,
        _pad0: [0.0; 2],
      },
      // The speed normalization is the display's, not the bake's: the bake was
      // handed a step chosen so the *peak* covers
      // `PARTICLE_SPEED_FRACTION` of the object's radius each second, so the
      // peak's ambient displacement per step is that same fraction over the
      // step rate -- known here without measuring anything.
      particle: ParticleMaterial {
        color: PARTICLE_INK,
        radius_world: PARTICLE_RADIUS_FRACTION * amplitude_scale,
        speed_scale: PARTICLE_SPEED_FRACTION as f32 * amplitude_scale / STEPS_PER_SECOND,
        stretch: PARTICLE_STRETCH,
        _pad0: 0.0,
      },
    };
    (display, attributes)
  }

  /// The frame's items, in submission order: the surface writes depth, and the
  /// marks over it -- a line field's ribbons, its particles, then the wireframe
  /// -- only test against it, so they blend in the order given.
  ///
  /// `marks` drops items rather than switching the frame graph. A mark that is
  /// off is simply not in the list, which the renderer cannot tell from a field
  /// that never had one -- so the toggle costs no branch below this line.
  pub(crate) fn draw_list<'a>(&'a self, mesh: &'a MeshDisplay, marks: Marks) -> DrawList<'a> {
    let mut items = Vec::new();
    if let Some(surface) = &mesh.surface {
      items.push(RenderItem::Surface(surface, self.surface));
    }
    if let Some(glyphs) = self.glyphs.as_ref().filter(|_| marks.glyphs) {
      items.push(RenderItem::Segments(glyphs, self.glyph));
    }
    if let Some(streamlines) = self.streamlines.as_ref().filter(|_| marks.streamlines) {
      items.push(RenderItem::Segments(streamlines, self.streamline));
    }
    if let Some(particles) = self.particles.as_ref().filter(|_| marks.particles) {
      items.push(RenderItem::Particles(particles, self.particle));
    }
    items.push(RenderItem::Segments(&mesh.segments, self.wireframe));
    DrawList { items }
  }
}
