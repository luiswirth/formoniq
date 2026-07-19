//! The windowed viewer: winit event loop, wgpu surface, egui integration and
//! input handling. Consumes the gallery model (`gallery.rs`) and the panel
//! (`ui.rs`), builds the current selection's display (`display.rs`), and
//! hands its draw list to the [`Renderer`], which owns every pipeline.
//!
//! What is here is what a *window* adds, and nothing else: the surface, the
//! event loop, the input, egui, and the clock. Everything from a [`Scene`] to a
//! draw list is `display.rs`, which a headless export drives identically -- the
//! two differ only in where the frame's time comes from, and time is the
//! renderer's argument.
//!
//! This is therefore the one place a clock and a surface exist.

use std::sync::Arc;

use wgpu::{Surface, SurfaceConfiguration};
use winit::{
  application::ApplicationHandler,
  event::*,
  event_loop::ActiveEventLoop,
  window::{Window, WindowId},
};
// The event loop is constructed only by the native `run`; the web build enters
// through `web::start`, which owns its own loop.
#[cfg(not(target_arch = "wasm32"))]
use winit::event_loop::EventLoop;

use crate::demos::default_selection;
use crate::display::{default_camera, scene_extent, FieldDisplay, MeshDisplay};
use crate::gallery::{presets, Gallery, MeshSource, Preset, Study};
use crate::render::{camera::Camera, ssaa_for_dpr, FrameView, GpuContext, Renderer};
use crate::scene::Scene;
use crate::ui::{Entry, PanelModel, Selection};

use egui_wgpu::{
  Renderer as EguiRenderer, RendererOptions as EguiRendererOptions, ScreenDescriptor,
};
use egui_winit::State as EguiWinitState;

/// The control shell's zoom relative to the window's own scale factor: every
/// egui widget is drawn at this fraction of its native point size, shrinking the
/// content-sized sidebars so they leave the viewport most of the frame. Below 1
/// because the default egui metrics are tuned for a full-window tool, not a
/// pair of sidebars flanking a 3D scene.
const UI_ZOOM: f32 = 1.25;

/// The most advection steps one frame will catch up on. At the advection's own
/// rate this is a few frames' worth, so ordinary jitter is absorbed and a long
/// stall is not.
const MAX_STEPS_PER_FRAME: u32 = 8;

/// Radians of rotation per pixel dragged, and the fraction of the distance to
/// the cursor's point one scroll notch closes. Zoom is multiplicative, so
/// approach is asymptotic: the step shrinks with the distance and the object
/// cannot be scrolled through.
const RADIANS_PER_PIXEL: f32 = 0.005;
const ZOOM_PER_NOTCH: f32 = 0.15;

/// Fly speed as a fraction of the scene's own extent per second, and the sprint
/// multiplier. Object-intrinsic like the framing: a mesh loaded at extent 1000
/// crosses itself in the same seconds a unit sphere does.
const FLY_EXTENTS_PER_SECOND: f32 = 0.5;
const SPRINT_FACTOR: f32 = 3.0;

/// Orthographic keyboard navigation: `WASD` pans this many view-heights per
/// second, `Space`/`Shift` zoom at this many e-folds per second. Screen-relative
/// rather than world-relative like the perspective fly, because a flat view has
/// a natural on-screen scale -- its half-extent -- and no depth to fly into, so
/// panning a fraction of the visible region per second is what keeps the feel
/// constant across zoom.
const PAN_SCREENS_PER_SECOND: f32 = 1.2;
const KEY_ZOOM_PER_SECOND: f32 = 2.5;

/// What a held mouse button means. The three are one rotation primitive and one
/// translation: orbit and look differ *only* in the center they rotate about
/// ([`Camera::rotate`]), and pan is the translation that keeps the pivot's depth
/// under the cursor.
#[derive(Clone, Copy)]
enum Drag {
  /// Rotate about where the *view axis* meets the mesh, picked when the drag
  /// began and held for its duration -- re-deriving it mid-drag would move the
  /// center the gesture is defined by, since the eye is what the gesture moves.
  ///
  /// Neither of the two tempting alternatives:
  ///
  /// *Not the point under the cursor.* That center is off the view axis, so the
  /// object swings across the viewport instead of spinning in place, and it
  /// moves with every press, so no two drags agree.
  ///
  /// *Not a depth along the axis* ([`Camera::pivot`]). That is the object's
  /// centroid at the default framing and follows a zoom, but it is a point in
  /// air: fly toward the mesh and the depth goes stale, leaving the center
  /// buried behind the surface and the orbit swinging about the inside of the
  /// object. Re-picking against the geometry is what makes flying and orbiting
  /// compose -- it is derived from what is actually there, so nothing it depends
  /// on can drift.
  ///
  /// The cost, paid knowingly: on a solid framed whole, the axis meets the
  /// *near* surface rather than the centroid, so the bulk swings a little more
  /// than a true turntable would.
  Orbit {
    pivot: nalgebra::Point3<f32>,
  },
  /// Rotate about the eye: the view turns in place.
  Look,
  Pan,
}

pub(crate) struct State {
  surface: Surface<'static>,
  ctx: GpuContext,
  config: SurfaceConfiguration,
  size: winit::dpi::PhysicalSize<u32>,
  renderer: Renderer,

  camera: Camera,
  mesh_display: MeshDisplay,
  display: FieldDisplay,

  /// What of the mesh is drawn, and how the selected field is read on it: the
  /// two objects on screen, settled separately because they are separate. Viewer
  /// state, not either object's -- everything is built whatever these say, so a
  /// toggle costs a draw call (or, for the displacement, a material field) and
  /// never a rebake.
  mesh_view: crate::ui::MeshView,
  field_view: crate::ui::FieldView,

  /// Which sidebars are open. Viewer state like the two views above: it
  /// rebuilds nothing, and which controls a reader wants in front of them is
  /// not a fact about the object.
  sidebars: crate::ui::Sidebars,

  /// How the scene's light reaches the display. Viewer state for the same
  /// reason: it rebuilds nothing, and it is a question about what is being
  /// looked for rather than about the object.
  post: crate::ui::Post,

  /// The standing wave's transport clock: the elapsed animation time the
  /// renderer is handed each frame, pausable from the transport bar and reset
  /// whenever the field changes so a newly selected mode starts at its crest.
  /// The renderer takes the time as an argument, so this clock lives here, in
  /// the windowed loop, and nowhere below.
  clock: WaveClock,

  /// Advection steps already taken by the particles on display.
  ///
  /// The renderer steps a population forward; it cannot evaluate one at an
  /// instant. So the window remembers how far it has stepped and asks only for
  /// the difference. Reset to the clock's current count whenever the field
  /// changes, because a fresh `FieldDisplay` is a fresh population sitting at
  /// its seeds -- not one that owes the whole elapsed history at once.
  steps_taken: u32,

  /// The drag in progress, if any: which gesture the held button means.
  drag: Option<Drag>,
  /// The cursor's position, tracked whether or not a button is down -- a press
  /// does not carry one, and a press is exactly when the pivot must be picked
  /// from under it.
  cursor: Option<winit::dpi::PhysicalPosition<f64>>,
  last_mouse_pos: Option<winit::dpi::PhysicalPosition<f64>>,

  /// The fingers currently touching the surface, in the order they landed.
  /// Touch has no hover -- a finger exists only while down -- so this is both
  /// the pointer state and the gesture's arity: one finger looks, two pinch.
  /// The mouse path is untouched by it; the two input sources are independent
  /// and a device that emits one never emits the other for the same motion.
  touches: Vec<(u64, winit::dpi::PhysicalPosition<f64>)>,
  /// The touch gesture's baseline from the previous move: the centroid every
  /// finger is measured against, and (for two-finger pinch) their mean spread
  /// about it. Held as deltas rather than absolutes so a finger landing or
  /// lifting only re-seats the baseline ([`State::reseat_touch`]) instead of
  /// jumping the camera by the discontinuity in the centroid.
  touch_centroid: Option<(f64, f64)>,
  touch_spread: Option<f64>,

  /// Physical keys currently held, for the fly controls (`WASD` +
  /// `Space`/`Shift`/`Ctrl`). Tracked independently of winit's own key-repeat
  /// so movement follows the frame rate, not the repeat rate.
  keys_held: std::collections::HashSet<winit::keyboard::KeyCode>,
  /// Wall-clock time of the last frame, for the fly controls' `dt`. Distinct
  /// from [`WaveClock`]: that one is the pausable animation clock the
  /// renderer is handed, this is real elapsed time, which movement must
  /// always follow even while the wave is paused.
  last_frame: web_time::Instant,

  // The lazy, memoized per-grade loader: which view is current, which is
  // building in the background, and the cache of already-solved scenes. The
  // full scene it produces stays around (not just the field on display), so the
  // UI can switch which field is shown without a rebuild.
  gallery: Gallery,
  scene: Scene,
  selection: Selection,
  /// The curated presets the browser lists, built once. A preset is a
  /// configuration of the two axes (`gallery.rs`), so it lives model-side, not
  /// in the panel.
  presets: Vec<Preset>,
  /// The field a just-requested preset means to open on, applied when its scene
  /// lands ([`Self::install_scene`]). `None` for any other install, which opens
  /// on the scene's own first mode.
  pending_selection: Option<Selection>,
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
  // frame inside the egui pass, and polled for a pick afterward. Native only:
  // the two file browsers back the desktop's local-file features, which the
  // web build does not carry.
  #[cfg(not(target_arch = "wasm32"))]
  file_dialog: egui_file_dialog::FileDialog,
  // A second, independent browser for the "Export PNG…" save dialog: separate
  // from the OBJ loader so each carries its own filter and default name, and so
  // a pick on one is never mistaken for the other.
  #[cfg(not(target_arch = "wasm32"))]
  export_dialog: egui_file_dialog::FileDialog,
}

impl State {
  pub(crate) async fn new(window: Arc<Window>) -> State {
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

    // Grant the adapter's real limits, not the WebGPU baseline
    // `DeviceDescriptor::default()` pins: that baseline caps
    // `max_texture_dimension_2d` at 8192, below the supersampled depth target a
    // Retina fullscreen window needs (window width times `DEFAULT_SSAA_SCALE`),
    // while Metal itself allows 16384.
    let (device, queue) = adapter
      .request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
      })
      .await
      .unwrap();
    let ctx = GpuContext { device, queue };

    let config = surface
      .get_default_config(&adapter, size.width.max(1), size.height.max(1))
      .unwrap();
    surface.configure(&ctx.device, &config);

    let renderer = Renderer::new(
      &ctx,
      config.format,
      ssaa_for_dpr(window.scale_factor() as f32),
    );

    // Show the window immediately: the gallery opens on a cheap placeholder of
    // the starting mesh and builds the starting preset's study in the
    // background, swapping it in when it lands (`poll_view_load`), rather than
    // blocking the first frame on the solve. The placeholder carries no fields
    // of that study yet, so the preset's own opening selection and marks are
    // applied when its scene arrives, through the same pending path a
    // browser-selected preset takes.
    let start = crate::gallery::start_preset();
    let (gallery, scene) = Gallery::new(&start);
    let selection = default_selection(&scene);
    let amplitude_scale = scene_extent(&scene) as f32;
    let mesh_display = MeshDisplay::build(&ctx.device, &scene);
    let (display, attributes) =
      FieldDisplay::build(&ctx, &scene, &mesh_display, selection, amplitude_scale);
    mesh_display.write_attributes(&ctx.queue, &attributes);

    let camera = default_camera(&scene, config.width as f32 / config.height as f32);

    let egui_ctx = egui::Context::default();
    // Downscale the whole control shell relative to the window: the panels size
    // to their content (a slider, the eigenmode pyramid, the readout row), so
    // shrinking every widget uniformly is what keeps the sidebars from eating
    // the viewport, rather than hand-tuning each one's width.
    egui_ctx.set_zoom_factor(UI_ZOOM);
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
      clock: WaveClock::new(),
      steps_taken: 0,
      mesh_view: crate::ui::MeshView::default(),
      field_view: crate::ui::FieldView {
        marks: start.marks.unwrap_or_default(),
        ..Default::default()
      },
      sidebars: crate::ui::Sidebars::default(),
      post: crate::ui::Post::default(),
      drag: None,
      cursor: None,
      last_mouse_pos: None,
      touches: Vec::new(),
      touch_centroid: None,
      touch_spread: None,
      keys_held: std::collections::HashSet::new(),
      last_frame: web_time::Instant::now(),
      gallery,
      scene,
      selection,
      presets: presets(),
      pending_selection: start.selection,
      amplitude_scale,
      egui_ctx,
      egui_winit_state,
      egui_renderer,
      #[cfg(not(target_arch = "wasm32"))]
      file_dialog: egui_file_dialog::FileDialog::new()
        .add_file_filter_extensions("Wavefront OBJ", vec!["obj"])
        .default_file_filter("Wavefront OBJ"),
      #[cfg(not(target_arch = "wasm32"))]
      export_dialog: egui_file_dialog::FileDialog::new()
        .add_save_extension("PNG image", "png")
        .default_save_extension("PNG image")
        .default_file_name("frame.png"),
    }
  }

  /// The advection steps this frame owes: how far the clock has run, less how
  /// far the population has already been stepped.
  ///
  /// Clamped, because the count is derived from a clock the loop does not
  /// control: a stall, a paused window or a debugger breakpoint would otherwise
  /// present a frame with thousands of dispatches to catch up on, and the catch-up
  /// would itself stall the next frame. Dropping the excess lets the flow fall
  /// behind the wave clock rather than freezing the viewer -- the honest failure
  /// for a mark whose whole job is to be watched.
  fn pending_steps(&mut self) -> u32 {
    let owed = crate::display::steps_at(self.clock.time()).saturating_sub(self.steps_taken);
    let steps = owed.min(MAX_STEPS_PER_FRAME);
    self.steps_taken += steps;
    steps
  }

  /// The current solve-time position of the displayed trajectory: the wave
  /// clock's wall-seconds mapped onto the trajectory's own `duration`, looping
  /// every [`crate::display::TRAJECTORY_LOOP_SECONDS`]. The physical $dif t$ is a
  /// stability choice, so playback maps the sampled interval onto a watchable one
  /// rather than running at solve speed.
  fn trajectory_solve_time(&self, duration: f64) -> f64 {
    if duration <= 0.0 {
      return 0.0;
    }
    let phase =
      (f64::from(self.clock.time()) / crate::display::TRAJECTORY_LOOP_SECONDS).rem_euclid(1.0);
    phase * duration
  }

  /// Re-bakes the displayed field's stream from its interpolated frame at the
  /// current clock time, when that field is a trajectory. A no-op otherwise:
  /// a static field and a standing wave are baked once and re-timed on the GPU,
  /// so only a sampled trajectory owes a per-frame CPU rewrite -- exactly the
  /// "scrubbing a trajectory rewrites only the field stream" the bake draws.
  fn rebake_trajectory(&self) {
    let time_model = self.scene.field_time(self.selection);
    let Some(duration) = time_model.duration() else {
      return;
    };
    let base = self.scene.field_cochain(self.selection);
    let cochain = time_model.frame_at(base, self.trajectory_solve_time(duration));
    let (attributes, _) = crate::display::field_attributes(
      &self.scene.topology,
      &self.scene.coords,
      &cochain,
      self.mesh_display.cell_corners(),
    );
    self
      .mesh_display
      .write_attributes(&self.ctx.queue, &attributes);
  }

  /// Displays `selection` of the *current* scene, rebuilding exactly the pieces
  /// that depend on it and restarting the wave clock. Unconditional -- callers
  /// that only want to act on an actual change (the common case) go through
  /// [`Self::set_field`] instead.
  fn apply_field(&mut self, selection: Selection) {
    self.selection = selection;
    let (display, attributes) = FieldDisplay::build(
      &self.ctx,
      &self.scene,
      &self.mesh_display,
      selection,
      self.amplitude_scale,
    );
    self
      .mesh_display
      .write_attributes(&self.ctx.queue, &attributes);
    self.display = display;
    self.clock.restart();
    // A fresh `FieldDisplay` is a fresh population, sitting at its seeds. It
    // owes nothing, and the restarted clock owes nothing either.
    self.steps_taken = 0;
  }

  /// Switches the displayed field within the current scene. Everything else
  /// (camera, pipelines, egui) stays untouched.
  fn set_field(&mut self, selection: Selection) {
    if selection == self.selection {
      return;
    }
    self.apply_field(selection);
  }

  /// Requests that `study` be shown on the current mesh. A cached pair (an
  /// already-solved grade) installs instantly; an uncached one -- a grade's
  /// eigensolve -- runs on a background thread via `gallery`, and
  /// [`Self::poll_view_load`] installs the result once it lands, so this call
  /// never blocks the UI.
  fn set_study(&mut self, study: Study) {
    if let Some(scene) = self.gallery.select_study(study) {
      self.install_scene(scene);
    }
  }

  /// Switches the gallery's mesh to a regenerable source and installs the scene
  /// it hands back -- a cache hit, or a placeholder on the new mesh shown at
  /// once while the current study re-solves in the background. A no-op source
  /// change installs nothing; a build failure is recorded on the gallery and
  /// shown in the panel, leaving the current mesh in place.
  fn set_mesh_source(&mut self, source: MeshSource) {
    match self.gallery.select_mesh(source) {
      Ok(Some(scene)) => self.install_scene(scene),
      Ok(None) => {}
      Err(error) => self.gallery.set_error(error),
    }
  }

  /// Applies a preset: sets both axes, the opening field and the marks it opens
  /// with, then installs the scene the gallery hands back (a cache hit or a
  /// placeholder on the new mesh). The preset's field is remembered in
  /// `pending_selection` and applied once the real scene lands. A build failure
  /// is recorded and shown.
  ///
  /// The marks apply at once rather than pending: they are a view setting, not
  /// a fact about the scene, so there is nothing to wait for. A preset that
  /// names none leaves the reader's current toggles alone -- only an explicit
  /// opening view overrides them.
  fn set_preset(&mut self, index: usize) {
    self.pending_selection = self.presets[index].selection;
    if let Some(marks) = self.presets[index].marks {
      self.field_view.marks = marks;
    }
    match self.gallery.select_preset(&self.presets[index]) {
      Ok(Some(scene)) => self.install_scene(scene),
      Ok(None) => {}
      Err(error) => self.gallery.set_error(error),
    }
  }

  /// Loads a user-picked OBJ file as a custom mesh: reads it, parses it through
  /// the tolerant reader, and either installs it (re-solving the current grade)
  /// or records the parse/read error in the panel. Reading a file the user
  /// chose is not itself a side effect, so it needs no confirmation.
  ///
  /// Native only: the web build has no local filesystem and offers no OBJ
  /// picker in the panel.
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
    // A preset's opening field, if one is pending and in range on this scene;
    // otherwise the scene's own first mode. A placeholder (one scalar field)
    // does not satisfy a line-field preset selection, so the pending choice is
    // kept until the real scene it belongs to lands.
    let selection = match self.pending_selection {
      Some(sel) if selection_in_range(&scene, sel) => {
        self.pending_selection = None;
        sel
      }
      _ => default_selection(&scene),
    };
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
      WindowEvent::KeyboardInput {
        event:
          winit::event::KeyEvent {
            physical_key: winit::keyboard::PhysicalKey::Code(code),
            state: key_state,
            repeat: false,
            ..
          },
        ..
      } => {
        match key_state {
          ElementState::Pressed => self.keys_held.insert(*code),
          ElementState::Released => self.keys_held.remove(code),
        };
      }
      // Held keys are meaningless once the window stops receiving key events
      // for them; without this a key released while the window was
      // unfocused would stay "held" forever.
      WindowEvent::Focused(false) => self.keys_held.clear(),
      WindowEvent::MouseInput {
        state: ElementState::Pressed,
        button,
        ..
      } => {
        self.drag = if self.camera.orthographic {
          // A flat, face-on view does not rotate: tumbling it would only tilt
          // the plane away from the observer, which is the one thing the
          // orthographic view exists to keep square. So every button pans and
          // the view stays a 2D map -- the one place the interaction reads the
          // projection, mirrored in the keyboard split below.
          Some(Drag::Pan)
        } else {
          match button {
            winit::event::MouseButton::Left => {
              let pivot = self.center_point();
              // The pivot depth is re-established here and nowhere else: it is
              // what pan and the orthographic frustum read as "the depth of what
              // I am looking at", and a press on the geometry is the one moment
              // that has a real answer for them.
              self.camera.pivot_distance = (pivot - self.camera.eye).norm();
              Some(Drag::Orbit { pivot })
            }
            winit::event::MouseButton::Right => Some(Drag::Look),
            winit::event::MouseButton::Middle => Some(Drag::Pan),
            _ => None,
          }
        };
        self.last_mouse_pos = self.cursor;
      }
      WindowEvent::MouseInput {
        state: ElementState::Released,
        ..
      } => {
        self.drag = None;
        self.last_mouse_pos = None;
      }
      WindowEvent::CursorMoved { position, .. } => {
        self.cursor = Some(*position);
        let (Some(drag), Some(last)) = (self.drag, self.last_mouse_pos) else {
          return;
        };
        let dx = (position.x - last.x) as f32;
        let dy = (position.y - last.y) as f32;
        self.last_mouse_pos = Some(*position);

        match drag {
          // One primitive, two centers -- the whole content of the
          // orbit/look distinction.
          // `pivot_distance` needs no update: a rigid rotation about the pivot
          // conserves the distance to it, which is the invariant that keeps the
          // framing steady across a drag.
          Drag::Orbit { pivot } => {
            self
              .camera
              .rotate(dx * RADIANS_PER_PIXEL, dy * RADIANS_PER_PIXEL, pivot);
          }
          Drag::Look => {
            let eye = self.camera.eye;
            self
              .camera
              .rotate(dx * RADIANS_PER_PIXEL, dy * RADIANS_PER_PIXEL, eye);
          }
          // The content follows the cursor, like dragging a sheet of paper, so
          // the eye moves against it. Scaled at the pivot's depth, which is
          // what makes the grabbed point track the cursor exactly.
          Drag::Pan => {
            let scale = self.camera.world_per_pixel(self.size.height);
            self.camera.eye += (self.camera.up() * dy - self.camera.right() * dx) * scale;
          }
        }
      }
      WindowEvent::MouseWheel { delta, .. } => {
        let scroll = match delta {
          winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
          winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
        };
        // The wheel zooms where the cursor points, under either projection.
        let focus = self.cursor_point();
        self.zoom_by(scroll * ZOOM_PER_NOTCH, focus);
      }
      WindowEvent::Touch(touch) => self.handle_touch(touch),
      _ => {}
    }
  }

  /// Touch navigation, the finger counterpart to the mouse split above and
  /// built on the same camera primitives -- one finger looks (pans, in the
  /// orthographic view that does not rotate), two fingers pinch to zoom and
  /// drag to pan. Nothing here is web-only: winit delivers `Touch` on every
  /// platform that has a touchscreen, so a touch laptop gets the same gestures
  /// the browser does, which is why it lives in the shared input path and not
  /// in `web.rs`.
  fn handle_touch(&mut self, touch: &winit::event::Touch) {
    use winit::event::TouchPhase;
    match touch.phase {
      TouchPhase::Started => {
        self.touches.push((touch.id, touch.location));
        self.reseat_touch();
      }
      TouchPhase::Ended | TouchPhase::Cancelled => {
        self.touches.retain(|(id, _)| *id != touch.id);
        self.reseat_touch();
      }
      TouchPhase::Moved => {
        if let Some(slot) = self.touches.iter_mut().find(|(id, _)| *id == touch.id) {
          slot.1 = touch.location;
        }
        self.apply_touch();
      }
    }
  }

  /// Recomputes the gesture baseline from the fingers currently down, called
  /// whenever their count changes. This is what keeps a landing or lifting
  /// finger from jolting the camera: the next move measures its delta against
  /// the configuration that produced it, not the one before it.
  fn reseat_touch(&mut self) {
    self.touch_centroid = self.touch_centroid_now();
    self.touch_spread = self.touch_spread_now();
  }

  /// The mean of the active finger positions, or `None` with no finger down.
  fn touch_centroid_now(&self) -> Option<(f64, f64)> {
    if self.touches.is_empty() {
      return None;
    }
    let n = self.touches.len() as f64;
    let (sx, sy) = self
      .touches
      .iter()
      .fold((0.0, 0.0), |(sx, sy), (_, p)| (sx + p.x, sy + p.y));
    Some((sx / n, sy / n))
  }

  /// The fingers' mean distance from their centroid: the pinch's one scalar.
  /// `None` below two fingers, where a spread is undefined and no pinch runs.
  fn touch_spread_now(&self) -> Option<f64> {
    if self.touches.len() < 2 {
      return None;
    }
    let (cx, cy) = self.touch_centroid_now()?;
    let n = self.touches.len() as f64;
    let spread = self
      .touches
      .iter()
      .map(|(_, p)| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt())
      .sum::<f64>()
      / n;
    Some(spread)
  }

  /// Turns the current finger configuration into a camera motion, relative to
  /// the baseline the last move (or [`Self::reseat_touch`]) left. One finger
  /// rotates about the eye -- the look gesture, or a pan in the orthographic
  /// view that declines rotation, mirroring the mouse. Two fingers pinch about
  /// their centroid and pan by its translation, the two composing the way a map
  /// does.
  fn apply_touch(&mut self) {
    let Some((cx, cy)) = self.touch_centroid_now() else {
      return;
    };
    // The centroid drives picking for the pinch focus, exactly as the cursor
    // does for the wheel: one finger's position is the pointer.
    self.cursor = Some(winit::dpi::PhysicalPosition::new(cx, cy));

    if let Some((px, py)) = self.touch_centroid {
      let (dx, dy) = ((cx - px) as f32, (cy - py) as f32);
      match self.touches.len() {
        1 if !self.camera.orthographic => {
          let eye = self.camera.eye;
          self
            .camera
            .rotate(dx * RADIANS_PER_PIXEL, dy * RADIANS_PER_PIXEL, eye);
        }
        _ => {
          // One finger in the orthographic view, or two in either -- both pan.
          let scale = self.camera.world_per_pixel(self.size.height);
          self.camera.eye += (self.camera.up() * dy - self.camera.right() * dx) * scale;
        }
      }
    }

    // The pinch rides on top of the two-finger pan: the ratio of spreads is the
    // multiplicative zoom the wheel already speaks, about the point the fingers
    // straddle.
    if let (Some(prev), Some(now)) = (self.touch_spread, self.touch_spread_now()) {
      if prev > f64::EPSILON && now > f64::EPSILON {
        let e_folds = (now / prev).ln() as f32;
        let focus = self.cursor_point();
        self.zoom_by(e_folds, focus);
      }
    }

    self.touch_centroid = Some((cx, cy));
    self.touch_spread = self.touch_spread_now();
  }

  /// Where a ray meets the mesh, or -- on a miss, and for a bake with no
  /// surface at all -- the point at the current pivot depth along it. The
  /// fallback is what keeps the gestures built on this total: pointing at empty
  /// space still yields a point, rather than making every caller conditional on
  /// having hit something.
  fn pick(
    &self,
    (origin, dir): (nalgebra::Point3<f32>, nalgebra::Vector3<f32>),
  ) -> nalgebra::Point3<f32> {
    let t = self
      .mesh_display
      .raycast(origin, dir)
      .unwrap_or(self.camera.pivot_distance);
    origin + dir * t
  }

  /// Where the view axis meets the mesh: the orbit's center.
  fn center_point(&self) -> nalgebra::Point3<f32> {
    self.pick(self.camera.ray(0.0, 0.0))
  }

  /// Where the cursor's ray meets the mesh: the zoom's focus.
  fn cursor_point(&self) -> nalgebra::Point3<f32> {
    self.pick(self.cursor_ray())
  }

  /// The cursor's ray, or the viewport center's when the cursor is outside the
  /// window.
  fn cursor_ray(&self) -> (nalgebra::Point3<f32>, nalgebra::Vector3<f32>) {
    let Some(pos) = self.cursor else {
      return self.camera.ray(0.0, 0.0);
    };
    let ndc_x = 2.0 * (pos.x as f32 / self.size.width.max(1) as f32) - 1.0;
    let ndc_y = 1.0 - 2.0 * (pos.y as f32 / self.size.height.max(1) as f32);
    self.camera.ray(ndc_x, ndc_y)
  }

  /// Zoom about `focus` by `e_folds`, multiplicatively: positive brings the eye
  /// nearer (`exp(-e_folds) < 1`).
  ///
  /// The one zoom primitive, shared by the wheel and the orthographic keys.
  /// Multiplicative, so the step is a fraction of the remaining distance and the
  /// approach is asymptotic -- the surface cannot be zoomed through, which is
  /// what makes a fixed clamp unnecessary. The pivot depth scales with it: that
  /// is the orthographic frustum's only sense of scale, so this is exactly what
  /// makes a parallel projection zoom at all rather than sit at a fixed
  /// magnification.
  fn zoom_by(&mut self, e_folds: f32, focus: nalgebra::Point3<f32>) {
    let factor = (-e_folds).exp();
    self.camera.eye = focus + (self.camera.eye - focus) * factor;
    self.camera.pivot_distance *= factor;
  }

  /// Keyboard navigation, dispatched by projection.
  ///
  /// The two views have different natural motions -- a perspective camera flies
  /// through the scene, an orthographic one slides across a flat view and zooms
  /// -- so this is where the interaction reads the projection, the keyboard
  /// counterpart to the mouse split at the button press. The primitives beneath
  /// (the eye translation, the multiplicative zoom) are shared; only which key
  /// drives which is chosen here.
  fn apply_movement(&mut self, dt: f32) {
    if self.keys_held.is_empty() {
      return;
    }
    if self.camera.orthographic {
      self.apply_ortho_movement(dt);
    } else {
      self.apply_fly_movement(dt);
    }
  }

  /// Orthographic navigation: `WASD` pans the flat view, `Space`/`Shift` zoom
  /// out/in, `Ctrl` sprints. No motion along the view -- a parallel projection
  /// has no depth to move into, so a "forward" key would change nothing but the
  /// clip planes.
  fn apply_ortho_movement(&mut self, dt: f32) {
    use winit::keyboard::KeyCode;
    let held = |code| self.keys_held.contains(&code);
    let axis = |pos, neg| f32::from(held(pos)) - f32::from(held(neg));
    let sprint = if held(KeyCode::ControlLeft) || held(KeyCode::ControlRight) {
      SPRINT_FACTOR
    } else {
      1.0
    };

    // Pan in the view plane, screen-relative: a fraction of the visible height
    // per second, so a keypress covers the same on-screen distance however far
    // zoomed -- the same scale the mouse pan reads, one from seconds and one
    // from pixels.
    let pan = self.camera.right() * axis(KeyCode::KeyD, KeyCode::KeyA)
      + self.camera.up() * axis(KeyCode::KeyW, KeyCode::KeyS);
    if let Some(direction) = pan.try_normalize(1e-6) {
      let (_, half_height) = self.camera.ortho_half_extent();
      let speed = PAN_SCREENS_PER_SECOND * 2.0 * half_height * sprint;
      self.camera.eye += direction * speed * dt;
    }

    // Zoom about the view center: `Space` out, `Shift` in. About the center and
    // not the cursor because a key is not a pointing gesture -- the wheel is the
    // one that zooms where you point.
    let zoom_in = f32::from(held(KeyCode::ShiftLeft) || held(KeyCode::ShiftRight));
    let e_folds = (zoom_in - f32::from(held(KeyCode::Space))) * KEY_ZOOM_PER_SECOND * sprint * dt;
    if e_folds != 0.0 {
      let focus = self.center_point();
      self.zoom_by(e_folds, focus);
    }
  }

  /// Fly the eye: `WASD` along the view, `Space`/`Shift` along world up, `Ctrl`
  /// to sprint.
  ///
  /// Forward is where the camera *looks*, pitch included -- not the yaw-only
  /// level flight a first-person game uses. That split exists to respect a
  /// ground plane, and a mesh in $RR^3$ has none: nothing here is standing on
  /// anything, so flight has no reason to stay level.
  ///
  /// Ascent is the exception and is deliberately *world* up, not the screen's:
  /// panning the view up-screen is what the middle button already does, so
  /// leaving `Space` to mean the one thing that survives any orientation --
  /// actually ascending -- is what keeps a pitched-over camera navigable.
  fn apply_fly_movement(&mut self, dt: f32) {
    use winit::keyboard::KeyCode;
    let held = |code| self.keys_held.contains(&code);
    let axis = |pos, neg| f32::from(held(pos)) - f32::from(held(neg));

    let delta = self.camera.forward() * axis(KeyCode::KeyW, KeyCode::KeyS)
      + self.camera.right() * axis(KeyCode::KeyD, KeyCode::KeyA)
      + crate::render::camera::WORLD_UP
        * (f32::from(held(KeyCode::Space))
          - f32::from(held(KeyCode::ShiftLeft) || held(KeyCode::ShiftRight)));

    let Some(direction) = delta.try_normalize(1e-6) else {
      return;
    };
    let sprint = if held(KeyCode::ControlLeft) || held(KeyCode::ControlRight) {
      SPRINT_FACTOR
    } else {
      1.0
    };
    self.camera.eye += direction * FLY_EXTENTS_PER_SECOND * self.amplitude_scale * sprint * dt;
  }

  /// Builds the panel's model snapshot, hands it to [`crate::ui::panel`]
  /// inside the one egui pass, and applies the requested changes afterward.
  /// The model/response split keeps the panel itself a pure function of its
  /// input; only this method touches `self`.
  fn run_ui(&mut self, window: &Window) -> (Vec<egui::ClippedPrimitive>, egui::TexturesDelta, f32) {
    let ctx = self.egui_ctx.clone();
    let raw_input = self.egui_winit_state.take_egui_input(window);

    // The two live axes drive the panel's highlighting, so clicking a study or
    // a grade reflects at once rather than a frame after its solve lands. The
    // displayed `self.scene` still belongs to the previous pair meanwhile, but
    // the mode picker is hidden behind the spinner until the new one arrives,
    // so its entries are only read when they match.
    let mesh_source = self.gallery.mesh_source.clone();
    let study = self.gallery.study.clone();

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
        eigenvalue: f.time.eigenvalue(),
        dof: f.dof.as_ref(),
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
            eigenvalue: f.time.eigenvalue(),
            dof: f.dof.as_ref(),
            name: f.name.as_str(),
          }),
      )
      .collect();

    // The displayed field's temporal model, for the transport readouts: an
    // eigenvalue gives the standing wave a frequency to report; a trajectory
    // gives a solve-time position within its duration. At most one is `Some`.
    let time_model = self.scene.field_time(self.selection);
    let eigenvalue = time_model.eigenvalue();
    let trajectory = time_model
      .duration()
      .map(|duration| (self.trajectory_solve_time(duration), duration));

    let model = PanelModel {
      sidebars: self.sidebars,
      mesh_source: mesh_source.clone(),
      study: study.clone(),
      is_loading: self.gallery.is_loading(),
      loading_label: self.gallery.loading_label(),
      last_grade: self.gallery.last_grade,
      max_grade: self.gallery.mesh.0.dim(),
      scene_dim: self.scene.topology.dim(),
      entries,
      mesh_error: self.gallery.error.clone(),
      selection: self.selection,
      mesh_view: self.mesh_view,
      field_view: self.field_view,
      offers: self.scene.offers(self.selection),
      post: self.post,
      orthographic: self.camera.orthographic,
      presets: &self.presets,
      eigenvalue,
      trajectory,
      playing: self.clock.playing,
      time: self.clock.time(),
    };

    // Borrow only the file dialogs into the closure: the rest of the panel is
    // driven by `model`, so these disjoint fields can be touched inside the
    // closure without conflicting with the `&mut self` calls after it. Their own
    // borrow ends with the closure, before those calls. Native only -- the web
    // panel raises neither dialog.
    #[cfg(not(target_arch = "wasm32"))]
    let file_dialog = &mut self.file_dialog;
    #[cfg(not(target_arch = "wasm32"))]
    let export_dialog = &mut self.export_dialog;

    let mut response = None;
    let full_output = ctx.run_ui(raw_input, |ui| {
      response = Some(crate::ui::panel(ui, &model));

      // Both browsers draw as their own windows and open on the panel's
      // request; each must be updated within the frame, after the panel. `Ui`
      // derefs to `Context`, which is what `FileDialog::update` takes.
      #[cfg(not(target_arch = "wasm32"))]
      {
        let response = response.as_ref().unwrap();
        if response.load_obj_clicked {
          file_dialog.pick_file();
        }
        file_dialog.update(ui);
        if response.export_png_clicked {
          export_dialog.save_file();
        }
        export_dialog.update(ui);
      }
    });
    let response = response.expect("the closure always runs exactly once");

    // A finished file pick, a preset, a mesh-source change and a study switch
    // each replace the field set and the camera's natural orientation
    // wholesale, so the `selection`/`orthographic` picked this same frame belong to
    // the pair being left and must not be applied afterward -- the setters
    // choose both anew for what they land on. They are mutually exclusive in
    // priority (at most one such widget moves per frame anyway); a preset sets
    // both axes at once, so it precedes the individual axis switches.
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
      if let Some(index) = response.requested_preset {
        self.set_preset(index);
      } else if response.requested_mesh != mesh_source {
        self.set_mesh_source(response.requested_mesh);
      } else if response.requested_study != study {
        self.set_study(response.requested_study);
      } else {
        self.set_field(response.selection);
        if response.orthographic && !self.camera.orthographic {
          self.camera.snap_top_down();
        }
        self.camera.orthographic = response.orthographic;
      }
    }

    // These are orthogonal to which field is shown -- pausing the wave, hiding a
    // mark, changing the display transform: none of them touches the pair, the
    // selection or a buffer. So they apply regardless of the branch above, where
    // being inside it would silently drop a toggle that happened to fire on the
    // same frame as a mesh or study change.
    self.mesh_view = response.mesh_view;
    self.field_view = response.field_view;
    self.sidebars = response.sidebars;
    self.post = response.post;
    self.clock.set_playing(response.playing);

    // A finished export pick is likewise orthogonal to the shown pair: it reads
    // the current frame and writes it, changing nothing on screen.
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(path) = self.export_dialog.take_picked() {
      self.export_current_png(&path);
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

  /// Writes the current view -- this field, this camera, this wave phase -- to
  /// `path` as a PNG still, at the window's own resolution. The same draw list
  /// and camera the window is rendering, handed to the headless path, so the
  /// still is exactly what is on screen (minus the egui panels). A write failure
  /// is surfaced in the panel rather than aborting the loop.
  ///
  /// Native only: there is no local filesystem to write to on the web.
  #[cfg(not(target_arch = "wasm32"))]
  fn export_current_png(&mut self, path: &std::path::Path) {
    let items = self
      .display
      .draw_list(&self.mesh_display, self.mesh_view, self.field_view);
    let frame = FrameView {
      items: &items,
      camera: &self.camera,
      size: (self.config.width, self.config.height),
      time: self.clock.time(),
      // The window's own particles, at the point they have already reached:
      // the still is what is on screen, so it steps them no further.
      steps: 0,
      post: crate::display::post_uniform(self.post),
    };
    if let Err(error) = crate::export::export_frame_png(&self.ctx, &frame, path) {
      self.gallery.set_error(error);
    }
  }

  fn render(&mut self, window: &Window) -> Result<(), ()> {
    // Ahead of `run_ui`: applying a finished background load before the panel
    // is built is what makes the field pickers reflect the new scene on the
    // very frame it lands, rather than one frame late.
    self.poll_view_load();

    let now = web_time::Instant::now();
    let dt = (now - self.last_frame).as_secs_f32();
    self.last_frame = now;
    self.apply_movement(dt);

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
        // A backstop against this turning into a full-throttle retry spin.
        // Native only: the web loop is paced by the browser's animation frame,
        // so there is no synchronous spin to throttle (and no thread to sleep).
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

    // A trajectory's field stream is re-baked from its current frame before the
    // draw list is built; a static field or standing wave rebakes nothing (the
    // GPU re-times those), so this is a no-op away from a trajectory.
    self.rebake_trajectory();

    // Taken before the draw list borrows `self`, not because the order matters
    // to the frame but because the step accounting is a mutation and the list
    // is a borrow of what it accounts for.
    let steps = self.pending_steps();
    let items = self
      .display
      .draw_list(&self.mesh_display, self.mesh_view, self.field_view);
    let frame_view = FrameView {
      items: &items,
      camera: &self.camera,
      size: (self.config.width, self.config.height),
      time: self.clock.time(),
      steps,
      post: crate::display::post_uniform(self.post),
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

/// The standing wave's transport clock: elapsed animation time, pausable
/// without losing the phase reached. Time advances only while `playing`; a
/// pause freezes the accumulated total and a resume starts a fresh wall-clock
/// span from it, so the wave holds still under the cursor and picks up exactly
/// where it stopped -- the phase is never discontinuous across a toggle.
struct WaveClock {
  playing: bool,
  /// Animation seconds accumulated before the current play span.
  base: f32,
  /// Wall-clock start of the current play span; unread while paused.
  anchor: web_time::Instant,
}

impl WaveClock {
  fn new() -> Self {
    Self {
      playing: true,
      base: 0.0,
      anchor: web_time::Instant::now(),
    }
  }

  /// The elapsed animation time to hand the renderer this frame.
  fn time(&self) -> f32 {
    if self.playing {
      self.base + self.anchor.elapsed().as_secs_f32()
    } else {
      self.base
    }
  }

  /// Plays or pauses without moving the phase: the time reached is folded into
  /// `base` and a new span begins, so `time()` is continuous across the change.
  fn set_playing(&mut self, playing: bool) {
    if playing == self.playing {
      return;
    }
    self.base = self.time();
    self.anchor = web_time::Instant::now();
    self.playing = playing;
  }

  /// Back to the crest: a newly selected mode starts at $t = 0$, keeping the
  /// current play/pause state.
  fn restart(&mut self) {
    self.base = 0.0;
    self.anchor = web_time::Instant::now();
  }
}

/// Whether `selection` indexes a field the scene actually has -- the guard a
/// preset's opening field goes through, since a placeholder (one scalar field)
/// cannot satisfy a line-field selection and a selection valid in one study can
/// be out of range in another.
fn selection_in_range(scene: &Scene, selection: Selection) -> bool {
  match selection {
    Selection::Scalar(i) => i < scene.fields.len(),
    Selection::Line(i) => i < scene.line_fields.len(),
  }
}

#[derive(Default)]
pub(crate) struct App {
  window: Option<Arc<Window>>,
  state: Option<State>,
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

impl ApplicationHandler for App {
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    if self.window.is_some() {
      return;
    }
    let window = Arc::new(
      event_loop
        .create_window(Window::default_attributes().with_title("formoniq-studio"))
        .unwrap(),
    );
    self.window = Some(window.clone());

    #[cfg(not(target_arch = "wasm32"))]
    {
      self.state = Some(pollster::block_on(State::new(window)));
    }
    // On the web, adapter/device/surface creation is async and must yield to
    // the browser rather than block. The web module mounts the canvas, drives
    // `State::new` to completion off the event loop, and parks the result in a
    // slot the loop drains in `about_to_wait`.
    #[cfg(target_arch = "wasm32")]
    crate::web::init_state(window);
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
      WindowEvent::CloseRequested => event_loop.exit(),
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
    // The web `State::new` completes asynchronously; drain its slot once it
    // lands so the first real frame can render. No-op once installed.
    #[cfg(target_arch = "wasm32")]
    if self.state.is_none() {
      self.state = crate::web::take_ready_state();
      // The renderer just landed. Its initial size races the async bootstrap
      // and the `Resized` that would correct it was dropped while `state` was
      // still `None` (see `web::canvas_surface_size`), so reconcile the surface
      // to the page's settled canvas size now rather than render 1x1 forever.
      if let (Some(state), Some(window)) = (self.state.as_mut(), self.window.as_ref()) {
        state.resize(crate::web::canvas_surface_size(window));
      }
    }
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

/// The native windowed entry point. The web build enters through
/// `web::start` instead, which owns the browser's async event loop.
#[cfg(not(target_arch = "wasm32"))]
pub async fn run() {
  env_logger::init();

  let event_loop = EventLoop::new().unwrap();
  let mut app = App::default();
  let _ = event_loop.run_app(&mut app);
}
