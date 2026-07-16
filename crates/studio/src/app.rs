//! The windowed viewer: winit event loop, wgpu surface, egui integration and
//! input handling. Consumes the gallery model (`gallery.rs`) and the panel
//! (`ui/panel.rs`), turns the current selection into a baked [`DrawList`], and
//! hands that to the [`Renderer`], which owns every pipeline.
//!
//! This is the one place a clock and a surface exist. The renderer takes time
//! as an argument, so a headless exporter can drive the same frame graph from a
//! frame counter instead.

use std::sync::Arc;

use wgpu::{Surface, SurfaceConfiguration};
use winit::{
  application::ApplicationHandler,
  event::*,
  event_loop::{ActiveEventLoop, EventLoop},
  window::{Window, WindowId},
};

use crate::bake::{self, BakedMesh};
use crate::demos::default_selection;
use crate::gallery::{Gallery, MeshSource, View};
use crate::render::{
  camera::Camera,
  item::{DrawList, RenderItem, SegmentBatch, SurfaceBatch},
  uniform::{SegmentMaterial, SurfaceMaterial},
  {FrameView, GpuContext, Renderer},
};
use crate::scene::Scene;
use crate::ui::panel::{Entry, PanelModel, Selection};

use egui_wgpu::{
  Renderer as EguiRenderer, RendererOptions as EguiRendererOptions, ScreenDescriptor,
};
use egui_winit::State as EguiWinitState;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

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

/// The streamline ribbons' ink: dark enough to separate from any colormap
/// sample, not pure black, so the curves read as drawn on the surface rather
/// than as holes cut through it. Deliberately not the colormap: the surface
/// carries the magnitude and the curves carry the direction, so tinting a ribbon
/// by the field would restate what the fill beneath it already says -- and
/// restate it invisibly, since that colormap is three sinusoids at 120 degree
/// phase offsets, whose channels sum to a constant, making it iso-luminant.
/// Against a backdrop of constant mid luminance, a near-black ink is the one
/// choice that separates everywhere.
const STREAMLINE_INK: [f32; 4] = [0.05, 0.05, 0.07, 1.0];

/// The opacity the ribbons of an eigenmode fade to at the standing wave's node,
/// where the field vanishes and the curves are meaningless -- never fully, since
/// the integral curves of a standing mode are the same set at every phase, and
/// blinking them out entirely would read as the geometry changing.
const STREAMLINE_NODE_OPACITY: f32 = 0.25;

/// Streamline separation, as a fraction of the scene extent (its radius) --
/// object-intrinsic, like the wave amplitude, so the line density is a property
/// of the object and not of the triangulation. The one knob that sets how dense
/// the evenly spaced curves are.
const STREAMLINE_SEPARATION_FRACTION: f32 = 0.09;

/// The scene's coordinate extent: the largest distance of any vertex from the
/// mesh's own centroid -- its intrinsic radius, independent of where the mesh
/// sits in space. Measured about the centroid, not the origin, so a mesh
/// nowhere near the origin (a unit grid on $[0,1]^2$, an off-center loaded OBJ)
/// still reports its true size; an origin-centered unit sphere gives 1 either
/// way. Both the camera framing and the standing-wave amplitude scale off this,
/// so neither is tuned to the sphere.
fn scene_extent(scene: &Scene) -> f64 {
  scene_centroid_and_extent(scene).1
}

/// The mesh's own centroid, in the same 3-vector coordinates as
/// [`scene_extent`] -- the point the camera should target so an off-center
/// mesh (a unit grid on $[0,1]^2$, an off-center loaded OBJ) still ends up in
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
fn default_camera(scene: &Scene, aspect: f32) -> Camera {
  // Framing distance from the scene's own coordinate extent, not a constant
  // tuned for the sphere: an icosphere of radius 1 gives back exactly the
  // prior hardcoded 3.0, and a unit reference triangle frames itself too.
  let (centroid, extent) = scene_centroid_and_extent(scene);
  // A mesh flat in the z = 0 plane (the reference cell scenes: nothing has
  // been displaced off it yet) is looked down onto from above, along its own
  // normal, in orthographic top-down mode, rather than the angled perspective
  // orbit tuned for a fully 3D shape like the sphere.
  let z_extent = scene
    .coords
    .coord_iter()
    .map(|c| if c.len() > 2 { c[2].abs() } else { 0.0 })
    .fold(0.0, f64::max);
  let is_planar = z_extent < 1e-9 * extent;
  // `|yaw| = pi/2` keeps the eye's x-offset at exactly zero (see the
  // `direction` formula in `Camera::build_view_projection_matrix`): only
  // `pitch` should change for a top-down-ish view, since any other yaw skews
  // the eye diagonally in x and rotates the on-screen framing away from the
  // mesh's own axes. The sign flips between the two defaults because it also
  // fixes the handedness of the screen's local `right` axis -- without it,
  // screen-right ends up world $-x$ instead of $+x$, mirroring the mesh
  // left-to-right.
  let pitch: f32 = if is_planar { -1.2 } else { 0.3 };
  let yaw: f32 = if is_planar { 1.57 } else { -1.57 };

  let mut camera = Camera::new(aspect);
  camera.target = nalgebra::Point3::new(centroid[0] as f32, centroid[1] as f32, centroid[2] as f32);
  camera.distance = 3.0 * extent as f32;
  camera.pitch = pitch;
  camera.yaw = yaw;
  camera.top_down = is_planar;
  camera
}

/// The scene's geometry on the GPU: what a mesh bakes to, and nothing a field
/// decides. Rebuilt when the scene changes, never when the field does.
struct MeshDisplay {
  /// The filled surface, absent for a bake with no fill (a curve, a point
  /// cloud), whose only mark is its segments.
  surface: Option<SurfaceBatch>,
  /// The 1-skeleton overlay of a surface, or a 1-manifold's own cells. Empty for
  /// a bake with neither, which draws nothing rather than being a case to
  /// exclude.
  segments: SegmentBatch,
}

impl MeshDisplay {
  fn build(device: &wgpu::Device, scene: &Scene) -> Self {
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
    }
  }

  /// Rebinds the mesh to a different field: one buffer write per stream, no
  /// rebake.
  fn write_attributes(&self, queue: &wgpu::Queue, attributes: &[f32]) {
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
struct FieldDisplay {
  /// The traced streamlines of a line field, `None` for a scalar field. Their
  /// presence *is* the line-field mark: there is no branch to pick.
  streamlines: Option<SegmentBatch>,
  surface: SurfaceMaterial,
  wireframe: SegmentMaterial,
  streamline: SegmentMaterial,
}

impl FieldDisplay {
  fn build(
    ctx: &GpuContext,
    mesh: &MeshDisplay,
    scene: &Scene,
    selection: Selection,
    amplitude_scale: f32,
  ) -> Self {
    let (streamlines, attributes, surface) = match selection {
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
          bake::attributes(field.values()),
          SurfaceMaterial {
            min_val,
            max_val,
            wave_amplitude,
            wave_omega,
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

        (
          Some(streamlines),
          // The surface is the same fill a scalar field gets, tinted by the
          // field's nodal magnitude: the curves carry the direction, so the
          // surface has only the magnitude left to say.
          bake::attributes(&field.magnitude),
          SurfaceMaterial {
            min_val,
            max_val,
            wave_amplitude: 0.0,
            wave_omega: field.eigenvalue.map_or(0.0, f64::sqrt) as f32,
          },
        )
      }
    };

    mesh.write_attributes(&ctx.queue, &attributes);

    Self {
      streamlines,
      surface,
      // The wireframe rides the surface's own wave, so it tracks the displaced
      // mesh rather than the flat rest one, and it has no node to fade at.
      wireframe: SegmentMaterial {
        color: [0.0, 0.0, 0.0, 1.0],
        half_width_world: WIREFRAME_WIDTH_FRACTION * amplitude_scale,
        fade_floor: 1.0,
        wave_amplitude: surface.wave_amplitude,
        wave_omega: surface.wave_omega,
      },
      // The ribbons share the wave's clock but not its displacement: the samples
      // sit on the undisplaced surface, so only the node fade reads the mode.
      streamline: SegmentMaterial {
        color: STREAMLINE_INK,
        half_width_world: STREAMLINE_WIDTH_FRACTION * amplitude_scale,
        fade_floor: STREAMLINE_NODE_OPACITY,
        wave_amplitude: 0.0,
        wave_omega: surface.wave_omega,
      },
    }
  }

  /// The frame's items, in submission order: the surface writes depth, and the
  /// marks over it -- a line field's ribbons, then the wireframe -- only test
  /// against it, so they blend in the order given.
  fn draw_list<'a>(&'a self, mesh: &'a MeshDisplay) -> DrawList<'a> {
    let mut items = Vec::new();
    if let Some(surface) = &mesh.surface {
      items.push(RenderItem::Surface(surface, self.surface));
    }
    if let Some(streamlines) = &self.streamlines {
      items.push(RenderItem::Segments(streamlines, self.streamline));
    }
    items.push(RenderItem::Segments(&mesh.segments, self.wireframe));
    DrawList { items }
  }
}

struct State<'a> {
  surface: Surface<'a>,
  ctx: GpuContext,
  config: SurfaceConfiguration,
  size: winit::dpi::PhysicalSize<u32>,
  renderer: Renderer,

  camera: Camera,
  mesh_display: MeshDisplay,
  display: FieldDisplay,

  /// The standing wave's phase origin, reset whenever the field changes so a
  /// newly selected mode starts at its crest. The renderer takes the elapsed
  /// time as an argument, so this clock lives here, in the windowed loop, and
  /// nowhere below.
  start_time: std::time::Instant,

  // Mouse state for orbit controls
  mouse_pressed: bool,
  last_mouse_pos: Option<winit::dpi::PhysicalPosition<f64>>,

  // The lazy, memoized per-grade loader: which view is current, which is
  // building in the background, and the cache of already-solved scenes. The
  // full scene it produces stays around (not just the field on display), so the
  // UI can switch which field is shown without a rebuild.
  gallery: Gallery,
  scene: Scene,
  selection: Selection,
  /// Fixed at scene-build time: the object's coordinate extent (its radius),
  /// which sets the standing-wave displacement scale for whichever field is on
  /// display and the world-space width of the segment marks -- an
  /// object-intrinsic length, so the lobes read at orbital scale on any mesh,
  /// however finely triangulated.
  amplitude_scale: f32,

  egui_ctx: egui::Context,
  egui_winit_state: EguiWinitState,
  egui_renderer: EguiRenderer,

  // The in-egui file browser for loading a custom OBJ mesh. Persistent (it
  // holds the browser's own navigation state across frames), updated every
  // frame inside the egui pass, and polled for a pick afterward. Native-only:
  // wasm has no filesystem to browse.
  #[cfg(not(target_arch = "wasm32"))]
  file_dialog: egui_file_dialog::FileDialog,
}

impl<'a> State<'a> {
  async fn new(window: Arc<Window>) -> State<'a> {
    let size = window.inner_size();
    let instance = wgpu::Instance::default();

    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
      })
      .await
      .unwrap();

    let (device, queue) = adapter
      .request_device(&wgpu::DeviceDescriptor::default())
      .await
      .unwrap();
    let ctx = GpuContext { device, queue };

    let config = surface
      .get_default_config(&adapter, size.width.max(1), size.height.max(1))
      .unwrap();
    surface.configure(&ctx.device, &config);

    let renderer = Renderer::new(&ctx, config.format);

    // Show the window immediately: the gallery opens on a cheap placeholder of
    // the starting mesh and builds the first grade's eigenmodes in the
    // background, swapping it in when it lands (`poll_view_load`), rather than
    // blocking the first frame on the solve.
    let (gallery, scene) = Gallery::new(MeshSource::START);
    let selection = default_selection(&scene);
    let amplitude_scale = scene_extent(&scene) as f32;
    let mesh_display = MeshDisplay::build(&ctx.device, &scene);
    let display = FieldDisplay::build(&ctx, &mesh_display, &scene, selection, amplitude_scale);

    let camera = default_camera(&scene, config.width as f32 / config.height as f32);

    let egui_ctx = egui::Context::default();
    let egui_winit_state = EguiWinitState::new(
      egui_ctx.clone(),
      egui::ViewportId::ROOT,
      &window,
      Some(window.scale_factor() as f32),
      None,
      None,
    );
    let egui_renderer =
      EguiRenderer::new(&ctx.device, config.format, EguiRendererOptions::default());

    Self {
      surface,
      ctx,
      config,
      size,
      renderer,
      camera,
      mesh_display,
      display,
      start_time: std::time::Instant::now(),
      mouse_pressed: false,
      last_mouse_pos: None,
      gallery,
      scene,
      selection,
      amplitude_scale,
      egui_ctx,
      egui_winit_state,
      egui_renderer,
      #[cfg(not(target_arch = "wasm32"))]
      file_dialog: egui_file_dialog::FileDialog::new()
        .add_file_filter_extensions("Wavefront OBJ", vec!["obj"])
        .default_file_filter("Wavefront OBJ"),
    }
  }

  /// Displays `selection` of the *current* scene, rebuilding exactly the pieces
  /// that depend on it and restarting the wave clock. Unconditional -- callers
  /// that only want to act on an actual change (the common case) go through
  /// [`Self::set_field`] instead.
  fn apply_field(&mut self, selection: Selection) {
    self.selection = selection;
    self.display = FieldDisplay::build(
      &self.ctx,
      &self.mesh_display,
      &self.scene,
      selection,
      self.amplitude_scale,
    );
    self.start_time = std::time::Instant::now();
  }

  /// Switches the displayed field within the current scene. Everything else
  /// (camera, pipelines, egui) stays untouched.
  fn set_field(&mut self, selection: Selection) {
    if selection == self.selection {
      return;
    }
    self.apply_field(selection);
  }

  /// Requests that `view` be shown. A cached view (an already-solved grade)
  /// installs instantly; an uncached one -- a grade's eigensolve -- runs
  /// on a background thread via `gallery`, and [`Self::poll_view_load`] installs
  /// the result once it lands, so this call never blocks the UI.
  fn set_view(&mut self, view: View) {
    if let Some(scene) = self.gallery.request(view) {
      self.install_scene(scene);
    }
  }

  /// Switches the gallery's mesh to a regenerable source and installs the
  /// placeholder it hands back (the new mesh, shown at once) while the current
  /// grade re-solves in the background. A no-op source change, or one under a
  /// view that ignores the mesh, installs nothing; a build failure is recorded
  /// on the gallery and shown in the panel, leaving the current mesh in place.
  fn set_mesh_source(&mut self, source: MeshSource) {
    match self.gallery.select_source(source) {
      Ok(Some(scene)) => self.install_scene(scene),
      Ok(None) => {}
      Err(error) => self.gallery.set_error(error),
    }
  }

  /// Loads a user-picked OBJ file as a custom mesh: reads it, parses it through
  /// the tolerant reader, and either installs it (re-solving the current grade)
  /// or records the parse/read error in the panel. Reading a file the user
  /// chose is not itself a side effect, so it needs no confirmation.
  #[cfg(not(target_arch = "wasm32"))]
  fn load_obj_path(&mut self, path: std::path::PathBuf) {
    let name = path.file_name().map_or_else(
      || "mesh.obj".to_string(),
      |s| s.to_string_lossy().into_owned(),
    );
    let loaded = std::fs::read_to_string(&path)
      .map_err(|e| e.to_string())
      .and_then(|obj| crate::io::obj::parse(&obj).map_err(|e| e.to_string()));
    match loaded {
      Ok(mesh) => {
        if let Some(scene) = self.gallery.load_custom(name, mesh) {
          self.install_scene(scene);
        }
      }
      Err(error) => self.gallery.set_error(format!("{name}: {error}")),
    }
  }

  /// Installs a scene the gallery just handed over -- a cache hit or a finished
  /// build -- opening it on its first mode. The selection and camera from the
  /// old scene are never reused here (unlike [`Self::set_field`]'s early-out):
  /// a selection valid in one grade can be out of range in another, and a
  /// camera tuned for the sphere is not a natural start for the flat reference
  /// cell, or vice versa.
  fn install_scene(&mut self, scene: Scene) {
    self.amplitude_scale = scene_extent(&scene) as f32;
    let selection = default_selection(&scene);
    self.scene = scene;
    self.mesh_display = MeshDisplay::build(&self.ctx.device, &self.scene);
    self.apply_field(selection);
    self.camera = default_camera(&self.scene, self.camera.aspect);
  }

  /// Non-blocking: installs a background build the frame it finishes. Called
  /// once per frame regardless of whether a build is in flight.
  fn poll_view_load(&mut self) {
    if let Some(scene) = self.gallery.poll() {
      self.install_scene(scene);
    }
  }

  fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
    if new_size.width > 0 && new_size.height > 0 {
      self.size = new_size;
      self.config.width = new_size.width;
      self.config.height = new_size.height;
      self.surface.configure(&self.ctx.device, &self.config);
      // The renderer reallocates its own targets from the size it is handed,
      // so there is nothing to resize here beyond the surface and the aspect.
      self.camera.aspect = self.config.width as f32 / self.config.height as f32;
    }
  }

  fn handle_input(&mut self, event: &WindowEvent) {
    match event {
      WindowEvent::MouseInput {
        state: button_state,
        button: winit::event::MouseButton::Left,
        ..
      } => {
        self.mouse_pressed = *button_state == ElementState::Pressed;
        if !self.mouse_pressed {
          self.last_mouse_pos = None;
        }
      }
      WindowEvent::CursorMoved { position, .. } => {
        if self.mouse_pressed {
          if let Some(last) = self.last_mouse_pos {
            let dx = (position.x - last.x) as f32;
            let dy = (position.y - last.y) as f32;
            if self.camera.top_down {
              // Drag-to-pan: the content follows the cursor, like dragging a
              // sheet of paper -- the opposite sense from orbit, where the
              // camera follows the cursor around the scene. World-per-pixel
              // comes from the same half-extent the orthographic frustum
              // itself is sized from, so panning and zooming agree on scale.
              let (_, half_height) = self.camera.ortho_half_extent();
              let world_per_pixel = 2.0 * half_height / self.size.height.max(1) as f32;
              self.camera.target.x -= dx * world_per_pixel;
              self.camera.target.y += dy * world_per_pixel;
            } else {
              self.camera.yaw += dx * 0.005;
              self.camera.pitch -= dy * 0.005;
              // Clamp pitch to avoid gimbal lock
              self.camera.pitch = self.camera.pitch.clamp(-1.5, 1.5);
            }
          }
          self.last_mouse_pos = Some(*position);
        }
      }
      WindowEvent::MouseWheel { delta, .. } => {
        let scroll = match delta {
          winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
          winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
        };
        self.camera.distance -= scroll;
        self.camera.distance = self.camera.distance.clamp(1.0, 100.0);
      }
      _ => {}
    }
  }

  /// Builds the panel's model snapshot, hands it to [`crate::ui::panel::panel`]
  /// inside the one egui pass, and applies the requested changes afterward.
  /// The model/response split keeps the panel itself a pure function of its
  /// input; only this method touches `self`.
  fn run_ui(&mut self, window: &Window) -> (Vec<egui::ClippedPrimitive>, egui::TexturesDelta, f32) {
    let ctx = self.egui_ctx.clone();
    let raw_input = self.egui_winit_state.take_egui_input(window);

    // What the panel reflects is the view being *loaded*, if any, so clicking a
    // grade highlights it and shows the spinner at once rather than a frame
    // after its solve lands. The displayed `self.scene` still belongs to the
    // previous view meanwhile, but the mode picker is hidden behind the spinner
    // until the new one arrives, so its entries are only read when they match.
    let shown_view = self.gallery.loading_view().unwrap_or(self.gallery.current);
    let mesh_source = self.gallery.mesh_source.clone();

    // The current scene's modes, as the picker needs them. Scalar and line
    // fields both feed in; for a single mesh grade exactly one list is
    // populated, so the pyramid is over that grade alone.
    let entries: Vec<Entry> = self
      .scene
      .fields
      .iter()
      .enumerate()
      .map(|(i, f)| Entry {
        selection: Selection::Scalar(i),
        grade: f.grade,
        eigenvalue: f.eigenvalue,
        dof_label: f.dof_label.as_deref(),
        name: f.name.as_str(),
      })
      .chain(
        self
          .scene
          .line_fields
          .iter()
          .enumerate()
          .map(|(i, f)| Entry {
            selection: Selection::Line(i),
            grade: f.grade,
            eigenvalue: f.eigenvalue,
            dof_label: f.dof_label.as_deref(),
            name: f.name.as_str(),
          }),
      )
      .collect();

    let model = PanelModel {
      shown_view,
      is_loading: self.gallery.is_loading(),
      loading_label: self.gallery.loading_view().map(View::label),
      last_mesh_grade: self.gallery.last_mesh_grade,
      max_grade: self.gallery.mesh.0.dim(),
      scene_dim: self.scene.topology.dim(),
      entries,
      mesh_source: mesh_source.clone(),
      mesh_error: self.gallery.error.clone(),
      selection: self.selection,
      top_down: self.camera.top_down,
    };

    // Borrow only the file dialog into the closure (native): the rest of the
    // panel is driven by `model`, so this disjoint field can be touched inside
    // the closure without conflicting with the `&mut self` calls after it. Its
    // own borrow ends with the closure, before those calls.
    #[cfg(not(target_arch = "wasm32"))]
    let file_dialog = &mut self.file_dialog;

    let mut response = None;
    let full_output = ctx.run_ui(raw_input, |ctx| {
      response = Some(crate::ui::panel::panel(ctx, &model));

      // The file browser draws as its own window and opens on the panel's
      // request; it must be updated within the frame, after the panel.
      #[cfg(not(target_arch = "wasm32"))]
      {
        if response.as_ref().unwrap().load_obj_clicked {
          file_dialog.pick_file();
        }
        file_dialog.update(ctx);
      }
    });
    let response = response.expect("the closure always runs exactly once");

    // A finished file pick, a mesh-source change and a view switch each replace
    // the field set and the camera's natural orientation wholesale, so the
    // `selection`/`top_down` picked this same frame belong to the view being
    // left and must not be applied afterward -- the mesh/view setters choose
    // both anew for what they land on. They are mutually exclusive in priority
    // (at most one such widget moves per frame anyway); a load or a mesh change
    // re-solves the current grade, so both precede a same-frame grade switch.
    #[cfg(not(target_arch = "wasm32"))]
    let loaded_file = match self.file_dialog.take_picked() {
      Some(path) => {
        self.load_obj_path(path);
        true
      }
      None => false,
    };
    #[cfg(target_arch = "wasm32")]
    let loaded_file = false;

    if !loaded_file {
      if response.requested_mesh_source != mesh_source {
        self.set_mesh_source(response.requested_mesh_source);
      } else if response.requested_view != shown_view {
        self.set_view(response.requested_view);
      } else {
        self.set_field(response.selection);
        self.camera.top_down = response.top_down;
      }
    }

    self
      .egui_winit_state
      .handle_platform_output(window, full_output.platform_output);
    let paint_jobs = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    (
      paint_jobs,
      full_output.textures_delta,
      full_output.pixels_per_point,
    )
  }

  fn render(&mut self, window: &Window) -> Result<(), ()> {
    // Ahead of `run_ui`: applying a finished background load before the panel
    // is built is what makes the field pickers reflect the new scene on the
    // very frame it lands, rather than one frame late.
    self.poll_view_load();
    let (paint_jobs, textures_delta, pixels_per_point) = self.run_ui(window);

    // Registered unconditionally, ahead of the surface-acquire early-returns
    // below: egui reports a texture delta exactly once, on the frame it
    // changes, so dropping it on a `Timeout`/`Occluded`/`Outdated` frame would
    // lose that texture (e.g. the font atlas) for the rest of the session.
    for (id, image_delta) in &textures_delta.set {
      self
        .egui_renderer
        .update_texture(&self.ctx.device, &self.ctx.queue, *id, image_delta);
    }

    let output = match self.surface.get_current_texture() {
      wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
      wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
        self.resize(self.size);
        return Ok(());
      }
      // `Occluded` here is normally moot -- `about_to_wait` already stops
      // chasing redraws once `WindowEvent::Occluded(true)` lands -- but
      // `Timeout`/`Validation` are surface trouble that isn't occlusion (e.g.
      // a transient GPU stall) and `about_to_wait` will immediately request
      // another frame regardless. A short sleep here is the backstop against
      // that turning into the same full-throttle retry spin, whatever its
      // cause.
      wgpu::CurrentSurfaceTexture::Timeout
      | wgpu::CurrentSurfaceTexture::Occluded
      | wgpu::CurrentSurfaceTexture::Validation => {
        // Blocking sleep is unavailable on the wasm32 target (the browser's
        // own event loop already paces redraws there); native is what can
        // spin a CPU core unbounded.
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(16));
        return Ok(());
      }
    };
    let view = output
      .texture
      .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = self
      .ctx
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Render Encoder"),
      });

    let items = self.display.draw_list(&self.mesh_display);
    let frame_view = FrameView {
      items: &items,
      camera: &self.camera,
      size: (self.config.width, self.config.height),
      time: self.start_time.elapsed().as_secs_f32(),
    };
    self
      .renderer
      .render(&self.ctx, &mut encoder, &view, &frame_view);

    // egui composites over the renderer's output, in the same submission: the
    // one thing the windowed wrapper draws that the frame graph does not.
    let screen_descriptor = ScreenDescriptor {
      size_in_pixels: [self.config.width, self.config.height],
      pixels_per_point,
    };
    let egui_cmd_buffers = self.egui_renderer.update_buffers(
      &self.ctx.device,
      &self.ctx.queue,
      &mut encoder,
      &paint_jobs,
      &screen_descriptor,
    );
    {
      let egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Egui Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
          view: &view,
          resolve_target: None,
          ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
          },
          depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
      });
      self.egui_renderer.render(
        &mut egui_pass.forget_lifetime(),
        &paint_jobs,
        &screen_descriptor,
      );
    }
    for id in &textures_delta.free {
      self.egui_renderer.free_texture(id);
    }

    self.ctx.queue.submit(
      egui_cmd_buffers
        .into_iter()
        .chain(std::iter::once(encoder.finish())),
    );
    output.present();

    Ok(())
  }
}

#[derive(Default)]
struct App<'a> {
  window: Option<Arc<Window>>,
  state: Option<State<'a>>,
  // Whether the window is currently fully covered/minimized/off-screen. While
  // occluded, `get_current_texture` can never succeed, so there is no vsync
  // to pace the render loop against -- unconditionally chasing another
  // `RedrawRequested` (the naive `about_to_wait` pattern) turns into an
  // unbounded busy-spin of GPU driver calls for as long as the window stays
  // hidden (locked screen, closed lid, switched away), which is exactly the
  // kind of sustained driver contention that can wedge the whole system, not
  // just this process. `WindowEvent::Occluded` is winit's own signal for
  // this and is what actually stops the spin, rather than a fixed sleep.
  occluded: bool,
}

impl<'a> ApplicationHandler for App<'a> {
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    if self.window.is_none() {
      let window = Arc::new(
        event_loop
          .create_window(Window::default_attributes())
          .unwrap(),
      );

      #[cfg(target_arch = "wasm32")]
      {
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
          .and_then(|win| win.document())
          .and_then(|doc| {
            let dst = doc.get_element_by_id("wasm-example")?;
            let canvas = web_sys::Element::from(window.canvas()?);
            dst.append_child(&canvas).ok()?;
            Some(())
          })
          .expect("Couldn't append canvas to document body.");
      }

      let state = pollster::block_on(State::new(window.clone()));
      self.window = Some(window);
      self.state = Some(state);
    }
  }

  fn window_event(
    &mut self,
    event_loop: &ActiveEventLoop,
    _window_id: WindowId,
    event: WindowEvent,
  ) {
    let (Some(window), Some(state)) = (&self.window, &mut self.state) else {
      return;
    };

    // Every event goes to egui first; camera/orbit controls only see what
    // egui didn't consume (e.g. a drag that started on a widget).
    let consumed = state
      .egui_winit_state
      .on_window_event(window, &event)
      .consumed;

    match event {
      WindowEvent::CloseRequested
      | WindowEvent::KeyboardInput {
        event:
          winit::event::KeyEvent {
            state: ElementState::Pressed,
            logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
            ..
          },
        ..
      } => event_loop.exit(),
      WindowEvent::Resized(physical_size) => {
        state.resize(physical_size);
      }
      WindowEvent::Occluded(occluded) => {
        self.occluded = occluded;
        if !occluded {
          window.request_redraw();
        }
      }
      WindowEvent::RedrawRequested => {
        let _ = state.render(window);
      }
      other => {
        if !consumed {
          state.handle_input(&other);
        }
      }
    }
  }

  fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
    // Not while occluded: see the field doc on `occluded`. The loop resumes
    // on its own once `WindowEvent::Occluded(false)` fires the next redraw.
    if self.occluded {
      return;
    }
    if let Some(window) = &self.window {
      window.request_redraw();
    }
  }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
  cfg_if::cfg_if! {
      if #[cfg(target_arch = "wasm32")] {
          std::panic::set_hook(Box::new(console_error_panic_hook::hook));
          console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
      } else {
          env_logger::init();
      }
  }

  let event_loop = EventLoop::new().unwrap();
  let mut app = App::default();
  let _ = event_loop.run_app(&mut app);
}
