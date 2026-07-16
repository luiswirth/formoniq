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
  event_loop::{ActiveEventLoop, EventLoop},
  window::{Window, WindowId},
};

use crate::demos::default_selection;
use crate::display::{default_camera, scene_extent, FieldDisplay, MeshDisplay};
use crate::gallery::{presets, Gallery, MeshSource, Preset, Study};
use crate::render::{camera::Camera, FrameView, GpuContext, Renderer, DEFAULT_SSAA_SCALE};
use crate::scene::Scene;
use crate::ui::{Entry, PanelModel, Selection};

use egui_wgpu::{
  Renderer as EguiRenderer, RendererOptions as EguiRendererOptions, ScreenDescriptor,
};
use egui_winit::State as EguiWinitState;

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
  // frame inside the egui pass, and polled for a pick afterward.
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

    let renderer = Renderer::new(&ctx, config.format, DEFAULT_SSAA_SCALE);

    // Show the window immediately: the gallery opens on a cheap placeholder of
    // the starting mesh and builds the first grade's eigenmodes in the
    // background, swapping it in when it lands (`poll_view_load`), rather than
    // blocking the first frame on the solve.
    let (gallery, scene) = Gallery::new(MeshSource::START);
    let selection = default_selection(&scene);
    let amplitude_scale = scene_extent(&scene) as f32;
    let mesh_display = MeshDisplay::build(&ctx.device, &scene);
    let (display, attributes) = FieldDisplay::build(&ctx, &scene, selection, amplitude_scale);
    mesh_display.write_attributes(&ctx.queue, &attributes);

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
      presets: presets(),
      pending_selection: None,
      amplitude_scale,
      egui_ctx,
      egui_winit_state,
      egui_renderer,
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
    let (display, attributes) =
      FieldDisplay::build(&self.ctx, &self.scene, selection, self.amplitude_scale);
    self
      .mesh_display
      .write_attributes(&self.ctx.queue, &attributes);
    self.display = display;
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

  /// Applies a preset: sets both axes and the opening field, then installs the
  /// scene the gallery hands back (a cache hit or a placeholder on the new
  /// mesh). The preset's field is remembered in `pending_selection` and applied
  /// once the real scene lands. A build failure is recorded and shown.
  fn set_preset(&mut self, index: usize) {
    self.pending_selection = self.presets[index].selection;
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
        // A step and a clamp range as fractions of the scene's own extent, not
        // absolute constants: an OBJ loaded at extent 1000 must zoom and clamp
        // at the same fractions a unit sphere does, not snap into the mesh on
        // the first scroll because 100 was tuned for the sphere.
        self.camera.distance -= scroll * 0.1 * self.amplitude_scale;
        self.camera.distance = self
          .camera
          .distance
          .clamp(0.05 * self.amplitude_scale, 50.0 * self.amplitude_scale);
      }
      _ => {}
    }
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
      top_down: self.camera.top_down,
      presets: &self.presets,
    };

    // Borrow only the file dialog into the closure: the rest of the panel is
    // driven by `model`, so this disjoint field can be touched inside the
    // closure without conflicting with the `&mut self` calls after it. Its own
    // borrow ends with the closure, before those calls.
    let file_dialog = &mut self.file_dialog;

    let mut response = None;
    let full_output = ctx.run_ui(raw_input, |ctx| {
      response = Some(crate::ui::panel(ctx, &model));

      // The file browser draws as its own window and opens on the panel's
      // request; it must be updated within the frame, after the panel.
      if response.as_ref().unwrap().load_obj_clicked {
        file_dialog.pick_file();
      }
      file_dialog.update(ctx);
    });
    let response = response.expect("the closure always runs exactly once");

    // A finished file pick, a preset, a mesh-source change and a study switch
    // each replace the field set and the camera's natural orientation
    // wholesale, so the `selection`/`top_down` picked this same frame belong to
    // the pair being left and must not be applied afterward -- the setters
    // choose both anew for what they land on. They are mutually exclusive in
    // priority (at most one such widget moves per frame anyway); a preset sets
    // both axes at once, so it precedes the individual axis switches.
    let loaded_file = match self.file_dialog.take_picked() {
      Some(path) => {
        self.load_obj_path(path);
        true
      }
      None => false,
    };

    if !loaded_file {
      if let Some(index) = response.requested_preset {
        self.set_preset(index);
      } else if response.requested_mesh != mesh_source {
        self.set_mesh_source(response.requested_mesh);
      } else if response.requested_study != study {
        self.set_study(response.requested_study);
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
        // A backstop against this turning into a full-throttle retry spin.
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

pub async fn run() {
  env_logger::init();

  let event_loop = EventLoop::new().unwrap();
  let mut app = App::default();
  let _ = event_loop.run_app(&mut app);
}
