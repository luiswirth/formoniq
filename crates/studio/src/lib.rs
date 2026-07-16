extern crate nalgebra as na;

use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, RenderPipeline, Surface, SurfaceConfiguration};
use winit::{
  application::ApplicationHandler,
  event::*,
  event_loop::{ActiveEventLoop, EventLoop},
  window::{Window, WindowId},
};

use crate::render::{
  camera::{Camera, CameraUniform},
  mesh::{MeshBuffer, Vertex},
};
use crate::scene::Scene;

use egui_wgpu::{
  Renderer as EguiRenderer, RendererOptions as EguiRendererOptions, ScreenDescriptor,
};
use egui_winit::State as EguiWinitState;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub mod io;
pub mod mesh3d;
pub mod render;
pub mod scene;
pub mod ui;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
// The line-field G-buffer targets: `dir_mag` (screen tangent xy, magnitude,
// coverage) and `pos_shade` (world position, Lambert shade). Half-float so the
// world position survives for the LIC pass's object-space noise lookup.
const GBUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
// Edge length of the cubic object-space noise texture the LIC integrates.
const NOISE_SIZE: u32 = 64;
// How hard the along-line noise average is stretched back out of grey.
const LIC_CONTRAST: f32 = 5.0;
// Object-space noise frequency, in cycles per mesh width: fixes the streamline
// texture to the surface at a density that reads as lines rather than mush.
const NOISE_CYCLES_PER_MESH_WIDTH: f32 = 6.0;

// Icosphere subdivision depth. The Laplace-Beltrami eigensolve is dense in the
// vertex count, so keep this modest for an instant startup; bump for fidelity.
const SPHERE_SUBDIVISIONS: usize = 3;
const SPHERE_MODES: usize = 10;
// Which eigenmode colors the surface. Mode 0 is the constant harmonic; mode 4
// is the first grade-2 spherical harmonic.
const DISPLAY_MODE: usize = 4;
// Peak standing-wave displacement, as a multiple of the mesh's own width
// $h_max$ -- a mesh-intrinsic scale, not a sphere radius, so it stays
// meaningful on any mesh the scene is built from.
const WAVE_AMPLITUDE_FRACTION: f32 = 1.0;

/// Which built-in scene is on display: a picker in the UI switches this at
/// runtime, so this is a value, not a per-build constant -- the render path
/// already treats every `Scene` alike regardless of which of these built it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Demo {
  /// Laplace-Beltrami eigenmodes on the sphere.
  SphericalHarmonics,
  /// Every Whitney basis function of the reference triangle.
  WhitneyBasis,
}

impl Demo {
  const ALL: [Demo; 2] = [Demo::SphericalHarmonics, Demo::WhitneyBasis];

  fn label(self) -> &'static str {
    match self {
      Demo::SphericalHarmonics => "Spherical harmonics",
      Demo::WhitneyBasis => "Whitney basis (reference triangle)",
    }
  }
}

/// The starting scene when the viewer opens.
const DEMO: Demo = Demo::WhitneyBasis;

/// A rebuild of `T` running off the render thread, so a demo switch that
/// triggers a dense eigensolve (`Scene::spherical_harmonics`) never blocks the
/// UI. `poll` is non-blocking and yields the result exactly once, the frame it
/// arrives.
///
/// Wasm has no threads to spawn onto, so there `build` just runs eagerly and
/// the result is already waiting on the first `poll` -- the freeze that
/// motivates this on native is unavoidable there today, not silently
/// reintroduced.
struct PendingLoad<T> {
  rx: std::sync::mpsc::Receiver<T>,
}

impl<T: Send + 'static> PendingLoad<T> {
  fn spawn(build: impl FnOnce() -> T + Send + 'static) -> Self {
    let (tx, rx) = std::sync::mpsc::channel();
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::spawn(move || {
      let _ = tx.send(build());
    });
    #[cfg(target_arch = "wasm32")]
    let _ = tx.send(build());
    Self { rx }
  }

  fn poll(&self) -> Option<T> {
    self.rx.try_recv().ok()
  }
}

/// Which demo is on display, and which, if any, is being rebuilt in the
/// background to replace it. Kept free of any GPU type so it can be
/// unit-tested on its own -- the actual `Scene`/`Selection` it carries in
/// `State` are just its `T`.
///
/// Only one load is ever in flight. Requesting the demo already on display
/// cancels a pending switch away from it, rather than letting an abandoned
/// load land once it happens to finish; requesting a third demo while a
/// switch is already loading replaces that load outright. Either way, a
/// result can only ever be applied for the load `loading` currently names --
/// there is no path for a stale one to surface later.
struct SceneSwitcher<D, T> {
  current: D,
  loading: Option<(D, PendingLoad<T>)>,
}

impl<D: Copy + PartialEq, T: Send + 'static> SceneSwitcher<D, T> {
  fn new(current: D) -> Self {
    Self {
      current,
      loading: None,
    }
  }

  fn is_loading(&self) -> bool {
    self.loading.is_some()
  }

  fn request(&mut self, demo: D, build: impl FnOnce() -> T + Send + 'static) {
    if demo == self.current {
      self.loading = None;
      return;
    }
    if self.loading.as_ref().is_some_and(|(d, _)| *d == demo) {
      return;
    }
    self.loading = Some((demo, PendingLoad::spawn(build)));
  }

  /// `Some` exactly once, the frame a load completes, and commits `demo` as
  /// the new `current` in the same step.
  fn poll(&mut self) -> Option<T> {
    let result = self
      .loading
      .as_ref()
      .and_then(|(_, pending)| pending.poll())?;
    let (demo, _) = self.loading.take().unwrap();
    self.current = demo;
    Some(result)
  }
}

#[cfg(test)]
mod scene_switcher_tests {
  use super::SceneSwitcher;

  fn poll_until<D: Copy + PartialEq, T: Send + 'static>(switcher: &mut SceneSwitcher<D, T>) -> T {
    loop {
      if let Some(result) = switcher.poll() {
        return result;
      }
      std::thread::sleep(std::time::Duration::from_millis(1));
    }
  }

  #[test]
  fn request_delivers_the_built_value_and_commits_current() {
    let mut switcher = SceneSwitcher::new(0u32);
    switcher.request(1, || 42u32);
    let value = poll_until(&mut switcher);
    assert_eq!(value, 42);
    assert_eq!(switcher.current, 1);
    assert!(!switcher.is_loading());
  }

  #[test]
  fn requesting_the_current_demo_cancels_a_pending_switch() {
    let (gate_tx, gate_rx) = std::sync::mpsc::channel::<()>();
    let mut switcher = SceneSwitcher::new(0u32);
    switcher.request(1, move || {
      gate_rx.recv().ok();
      99u32
    });
    assert!(switcher.is_loading());

    // Switch back before the background build ever finishes.
    switcher.request(0, || unreachable!("current demo never rebuilds itself"));
    assert!(!switcher.is_loading());
    assert_eq!(switcher.current, 0);

    // Let the abandoned build finish; its result must never surface.
    gate_tx.send(()).ok();
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert_eq!(switcher.poll(), None);
    assert_eq!(switcher.current, 0);
  }

  #[test]
  fn requesting_a_third_demo_replaces_the_pending_load() {
    let (gate_tx, gate_rx) = std::sync::mpsc::channel::<()>();
    let mut switcher = SceneSwitcher::new(0u32);
    switcher.request(1, move || {
      gate_rx.recv().ok();
      1u32
    });
    switcher.request(2, || 2u32);

    let value = poll_until(&mut switcher);
    assert_eq!(value, 2);
    assert_eq!(switcher.current, 2);

    // The abandoned load for demo 1 must not surface once it finishes either.
    gate_tx.send(()).ok();
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert_eq!(switcher.poll(), None);
    assert_eq!(switcher.current, 2);
  }
}

/// The scene a demo builds, and the field it's shown on initially -- the one
/// place a `Demo` turns into a `Scene`, called both at startup and whenever
/// the UI switches it.
fn build_scene(demo: Demo) -> (Scene, Selection) {
  match demo {
    // Real formoniq output: Laplace-Beltrami eigenfunctions on the unit
    // sphere (discrete spherical harmonics), colored by one mode.
    Demo::SphericalHarmonics => (
      Scene::spherical_harmonics(SPHERE_SUBDIVISIONS, SPHERE_MODES),
      Selection::Scalar(DISPLAY_MODE),
    ),
    // The Whitney basis functions of the reference triangle: grade 0 (and the
    // top grade 2, starred to a density) color the surface; grade 1 is drawn
    // as a line field.
    Demo::WhitneyBasis => (Scene::whitney_basis(2), Selection::Scalar(0)),
  }
}

/// The camera's natural starting orientation for a scene, derived purely from
/// its own coordinates -- not which `Demo` built it, so a future flat or 3D
/// scene gets the same sensible default without adding another `match` arm
/// here.
fn default_camera(scene: &Scene, aspect: f32) -> Camera {
  // Framing distance from the scene's own coordinate extent, not a constant
  // tuned for the sphere: an icosphere of radius 1 gives back exactly the
  // prior hardcoded 3.0, and a unit reference triangle frames itself too.
  let extent = scene
    .coords
    .coord_iter()
    .map(|c| c.norm())
    .fold(0.0, f64::max)
    .max(1e-6);
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
  camera.target = nalgebra::Point3::origin();
  camera.distance = 3.0 * extent as f32;
  camera.pitch = pitch;
  camera.yaw = yaw;
  camera.top_down = is_planar;
  camera
}

fn create_depth_texture(device: &Device, config: &SurfaceConfiguration) -> wgpu::TextureView {
  let size = wgpu::Extent3d {
    width: config.width,
    height: config.height,
    depth_or_array_layers: 1,
  };
  let desc = wgpu::TextureDescriptor {
    label: Some("Depth Texture"),
    size,
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D2,
    format: DEPTH_FORMAT,
    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    view_formats: &[],
  };
  let texture = device.create_texture(&desc);
  texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// The two line-field G-buffer render targets, sized to the surface. Recreated
/// on resize alongside the depth texture.
fn create_gbuffer_textures(
  device: &Device,
  config: &SurfaceConfiguration,
) -> (wgpu::TextureView, wgpu::TextureView) {
  let make = |label: &str| {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
      label: Some(label),
      size: wgpu::Extent3d {
        width: config.width.max(1),
        height: config.height.max(1),
        depth_or_array_layers: 1,
      },
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: GBUFFER_FORMAT,
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
      view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
  };
  (make("G-buffer dir/mag"), make("G-buffer pos/shade"))
}

/// The LIC pass's binding of the two G-buffer views and their sampler. Rebuilt
/// whenever the views are (at startup and on resize).
fn create_gbuffer_bind_group(
  device: &Device,
  layout: &wgpu::BindGroupLayout,
  dir_view: &wgpu::TextureView,
  pos_view: &wgpu::TextureView,
  sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
  device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("gbuffer_tex_bind_group"),
    layout,
    entries: &[
      wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::TextureView(dir_view),
      },
      wgpu::BindGroupEntry {
        binding: 1,
        resource: wgpu::BindingResource::TextureView(pos_view),
      },
      wgpu::BindGroupEntry {
        binding: 2,
        resource: wgpu::BindingResource::Sampler(sampler),
      },
    ],
  })
}

/// A cubic bipolar (black/white) noise texture the LIC integrates in object
/// space. Binary rather than continuous on purpose: the along-line average of
/// smooth value noise barely leaves its mean, washing the streaks out, whereas
/// two-level noise carries maximal variance into the convolution so the
/// streamlines read as crisp light/dark lines. Trilinearly filtered on sample,
/// which softens the two levels back into antialiased strokes.
fn create_noise_texture(device: &Device, queue: &Queue) -> (wgpu::TextureView, wgpu::Sampler) {
  let n = NOISE_SIZE as usize;
  let mut data = vec![0u8; n * n * n];
  // A cheap integer hash (splitmix-ish) over the linear texel index, thresholded
  // to full black or full white: no crate, deterministic, maximal contrast.
  for (i, texel) in data.iter_mut().enumerate() {
    let mut h = i as u64 ^ 0x9e37_79b9_7f4a_7c15;
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    *texel = if h & 0x80 != 0 { 255 } else { 0 };
  }

  let texture = device.create_texture(&wgpu::TextureDescriptor {
    label: Some("LIC noise"),
    size: wgpu::Extent3d {
      width: NOISE_SIZE,
      height: NOISE_SIZE,
      depth_or_array_layers: NOISE_SIZE,
    },
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D3,
    format: wgpu::TextureFormat::R8Unorm,
    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    view_formats: &[],
  });
  queue.write_texture(
    wgpu::TexelCopyTextureInfo {
      texture: &texture,
      mip_level: 0,
      origin: wgpu::Origin3d::ZERO,
      aspect: wgpu::TextureAspect::All,
    },
    &data,
    wgpu::TexelCopyBufferLayout {
      offset: 0,
      bytes_per_row: Some(NOISE_SIZE),
      rows_per_image: Some(NOISE_SIZE),
    },
    wgpu::Extent3d {
      width: NOISE_SIZE,
      height: NOISE_SIZE,
      depth_or_array_layers: NOISE_SIZE,
    },
  );

  let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
  let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
    label: Some("LIC noise sampler"),
    address_mode_u: wgpu::AddressMode::Repeat,
    address_mode_v: wgpu::AddressMode::Repeat,
    address_mode_w: wgpu::AddressMode::Repeat,
    mag_filter: wgpu::FilterMode::Linear,
    min_filter: wgpu::FilterMode::Linear,
    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
    ..Default::default()
  });
  (view, sampler)
}

/// Which field of a scene is on display: its reduced grade decides the mark
/// ([`Scene`]'s own rule), and this is that choice's UI-facing form -- a
/// scalar field colors the surface with its own value; a line field colors the
/// surface with its nodal magnitude and draws line-integral convolution on top.
/// `PartialEq` so `egui::Ui::radio_value` can bind directly to it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Selection {
  Scalar(usize),
  Line(usize),
}

impl Selection {
  fn is_line(self) -> bool {
    matches!(self, Selection::Line(_))
  }
}

/// The mesh buffer and bounds/wave uniforms for showing one field of a
/// scene: the one place a [`Selection`] turns into pixels, called both at
/// startup and whenever the UI switches it.
struct FieldDisplay {
  mesh_buffer: MeshBuffer,
  field_min: f32,
  field_max: f32,
  wave_amplitude: f32,
  wave_omega: f32,
}

fn build_field_display(
  device: &Device,
  scene: &Scene,
  selection: Selection,
  mesh_width: f32,
) -> FieldDisplay {
  match selection {
    Selection::Scalar(index) => {
      let field = &scene.fields[index];
      let (raw_min, raw_max) = field.bounds();
      let mesh_buffer = MeshBuffer::new(device, &scene.topology, &scene.coords, field.values());

      let field_scale = raw_min.abs().max(raw_max.abs()).max(f32::EPSILON);
      // A field with no eigenvalue is not a standing-wave mode (e.g. a raw
      // Whitney basis function): no dispersion relation to animate at, so
      // the wave collapses to no displacement rather than a special case
      // here.
      let wave_omega = field.eigenvalue.map_or(0.0, f64::sqrt) as f32;
      let wave_amplitude = if field.eigenvalue.is_some() {
        WAVE_AMPLITUDE_FRACTION * mesh_width / field_scale
      } else {
        0.0
      };
      // An eigenmode's color pulses by $cos(sqrt(lambda) t)$ through zero, so
      // its colormap range is symmetric $[-s, s]$ about the midpoint -- the
      // same reasoning as the line field's tint. A static field keeps its own
      // asymmetric range.
      let (field_min, field_max) = if field.eigenvalue.is_some() {
        (-field_scale, field_scale)
      } else {
        (raw_min, raw_max)
      };

      FieldDisplay {
        mesh_buffer,
        field_min,
        field_max,
        wave_amplitude,
        wave_omega,
      }
    }
    Selection::Line(index) => {
      let field = &scene.line_fields[index];
      let mesh_buffer = MeshBuffer::from_line_field(device, &scene.topology, &scene.coords, field);
      // The tint is the signed magnitude $|V| cos(sqrt(lambda) t)$, so the
      // colormap range is symmetric $[-m, m]$ about zero: the pulse runs
      // through the colormap's midpoint and flips as the cosine crosses zero,
      // unlike an unsigned scalar that starts at 0. The LIC direction is
      // static, so there is no geometric displacement -- `wave_amplitude` is 0
      // and only `wave_omega` (the tint clock) carries the mode's frequency.
      let peak = field.max_magnitude().max(f64::from(f32::EPSILON)) as f32;
      let wave_omega = field.eigenvalue.map_or(0.0, f64::sqrt) as f32;

      FieldDisplay {
        mesh_buffer,
        field_min: -peak,
        field_max: peak,
        wave_amplitude: 0.0,
        wave_omega,
      }
    }
  }
}

struct State<'a> {
  surface: Surface<'a>,
  device: Device,
  queue: Queue,
  config: SurfaceConfiguration,
  size: winit::dpi::PhysicalSize<u32>,
  render_pipeline: RenderPipeline,
  wireframe_pipeline: RenderPipeline,
  depth_view: wgpu::TextureView,

  // Line-field path: an offscreen G-buffer pass feeds a fullscreen LIC pass.
  // Only exercised when the current selection is a line field; a scalar field
  // takes the direct fill path above.
  gbuffer_pipeline: RenderPipeline,
  lic_pipeline: RenderPipeline,
  gbuffer_dir_view: wgpu::TextureView,
  gbuffer_pos_view: wgpu::TextureView,
  gbuffer_sampler: wgpu::Sampler,
  gbuffer_tex_bind_group_layout: wgpu::BindGroupLayout,
  gbuffer_tex_bind_group: wgpu::BindGroup,
  noise_bind_group: wgpu::BindGroup,
  lic_buffer: wgpu::Buffer,
  lic_bind_group: wgpu::BindGroup,
  // The object-space noise frequency for the current scene, fixed from its own
  // mesh width so the streamlines read at a consistent density on any mesh.
  noise_scale: f32,

  camera: Camera,
  camera_uniform: CameraUniform,
  camera_buffer: wgpu::Buffer,
  camera_bind_group: wgpu::BindGroup,

  mesh_buffer: MeshBuffer,

  // kept alive to back bounds_bind_group's binding; never read directly
  #[allow(dead_code)]
  bounds_buffer: wgpu::Buffer,
  bounds_bind_group: wgpu::BindGroup,

  // Standing-wave animation: the mode's own frequency and a displacement
  // amplitude fixed at scene-build time; only `time` changes per frame.
  start_time: std::time::Instant,
  wave_amplitude: f32,
  wave_omega: f32,
  wave_buffer: wgpu::Buffer,
  wave_bind_group: wgpu::BindGroup,

  // Mouse state for orbit controls
  mouse_pressed: bool,
  last_mouse_pos: Option<winit::dpi::PhysicalPosition<f64>>,

  // The full scene stays around, not just the field on display, so the UI
  // can switch which one is shown -- a scene is exactly as general as the
  // picker built over it, regardless of how many fields it carries. Which
  // demo is `current` and which, if any, is loading in the background lives
  // in `scene_switcher`, not a bare `Demo` field.
  scene_switcher: SceneSwitcher<Demo, (Scene, Selection)>,
  scene: Scene,
  selection: Selection,
  // Fixed at scene-build time: the mesh's own edge-length scale, used to
  // normalize the standing-wave amplitude to whichever field is on display.
  mesh_width: f32,

  egui_ctx: egui::Context,
  egui_winit_state: EguiWinitState,
  egui_renderer: EguiRenderer,
}

/// Per-frame standing-wave state: $u(t) = "amplitude" dot "value" dot cos(omega t)$,
/// displacing each vertex along its own normal (see `shader.wgsl`/`wireframe.wgsl`).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveUniform {
  time: f32,
  amplitude: f32,
  omega: f32,
  _pad: f32,
}

/// Colormap normalization range for the field currently on display.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BoundsUniform {
  min_val: f32,
  max_val: f32,
  _pad1: f32,
  _pad2: f32,
}

/// Shared state for the G-buffer and LIC passes: the viewport (to project the
/// tangent to pixels and to step in texel units), the object-space noise
/// frequency, the tint clock $(omega, t)$ that swings the magnitude, and the
/// contrast the along-line average is stretched by.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LicUniform {
  viewport: [f32; 2],
  noise_scale: f32,
  omega: f32,
  time: f32,
  contrast: f32,
  _pad0: f32,
  _pad1: f32,
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

    let config = surface
      .get_default_config(&adapter, size.width.max(1), size.height.max(1))
      .unwrap();
    surface.configure(&device, &config);

    let demo = DEMO;
    let (scene, selection) = build_scene(demo);
    // The standing-wave amplitude is scaled by the mesh's own width, not a
    // sphere radius, so the displacement reads the same whether the mesh is a
    // sphere or anything else, and whichever field is currently on display.
    let mesh_width = scene
      .coords
      .to_edge_lengths(&scene.topology)
      .mesh_width_max() as f32;
    let display = build_field_display(&device, &scene, selection, mesh_width);
    let (field_min, field_max) = (display.field_min, display.field_max);
    let mesh_buffer = display.mesh_buffer;
    let wave_amplitude = display.wave_amplitude;
    let wave_omega = display.wave_omega;

    let camera = default_camera(&scene, config.width as f32 / config.height as f32);
    let mut camera_uniform = CameraUniform::new();
    camera_uniform.update_view_proj(&camera);

    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Camera Buffer"),
      contents: bytemuck::cast_slice(&[camera_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let camera_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::VERTEX,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        }],
        label: Some("camera_bind_group_layout"),
      });

    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &camera_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: camera_buffer.as_entire_binding(),
      }],
      label: Some("camera_bind_group"),
    });

    let bounds_uniform = BoundsUniform {
      min_val: field_min,
      max_val: field_max,
      _pad1: 0.0,
      _pad2: 0.0,
    };

    let bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Bounds Buffer"),
      contents: bytemuck::cast_slice(&[bounds_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bounds_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        }],
        label: Some("bounds_bind_group_layout"),
      });

    let bounds_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &bounds_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: bounds_buffer.as_entire_binding(),
      }],
      label: Some("bounds_bind_group"),
    });

    let start_time = std::time::Instant::now();
    let wave_uniform = WaveUniform {
      time: 0.0,
      amplitude: wave_amplitude,
      omega: wave_omega,
      _pad: 0.0,
    };

    let wave_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Wave Buffer"),
      contents: bytemuck::cast_slice(&[wave_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let wave_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
          binding: 0,
          // Also visible in the fragment stage: the scalar fill pulses its
          // colormap by the same $cos(sqrt(lambda) t)$ that displaces it.
          visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        }],
        label: Some("wave_bind_group_layout"),
      });

    let wave_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &wave_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: wave_buffer.as_entire_binding(),
      }],
      label: Some("wave_bind_group"),
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("Shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("render/shader.wgsl").into()),
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("Render Pipeline Layout"),
      bind_group_layouts: &[
        Some(&camera_bind_group_layout),
        Some(&bounds_bind_group_layout),
        Some(&wave_bind_group_layout),
      ],
      immediate_size: 0,
    });

    let depth_stencil = Some(wgpu::DepthStencilState {
      format: DEPTH_FORMAT,
      depth_write_enabled: Some(true),
      depth_compare: Some(wgpu::CompareFunction::Less),
      stencil: wgpu::StencilState::default(),
      bias: wgpu::DepthBiasState::default(),
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Render Pipeline"),
      layout: Some(&render_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),
        compilation_options: Default::default(),
        buffers: &[Vertex::desc()],
      },
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        compilation_options: Default::default(),
        targets: &[Some(wgpu::ColorTargetState {
          format: config.format,
          blend: Some(wgpu::BlendState::REPLACE),
          write_mask: wgpu::ColorWrites::ALL,
        })],
      }),
      primitive: wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        polygon_mode: wgpu::PolygonMode::Fill,
        unclipped_depth: false,
        conservative: false,
      },
      depth_stencil: depth_stencil.clone(),
      multisample: wgpu::MultisampleState::default(),
      multiview_mask: None,
      cache: None,
    });

    // Wireframe pipeline — LineList with camera-only bind group
    let wireframe_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("Wireframe Shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("render/wireframe.wgsl").into()),
    });

    let wireframe_pipeline_layout =
      device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Wireframe Pipeline Layout"),
        bind_group_layouts: &[
          Some(&camera_bind_group_layout),
          Some(&wave_bind_group_layout),
        ],
        immediate_size: 0,
      });

    let wireframe_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Wireframe Pipeline"),
      layout: Some(&wireframe_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &wireframe_shader,
        entry_point: Some("vs_main"),
        compilation_options: Default::default(),
        buffers: &[Vertex::desc()],
      },
      fragment: Some(wgpu::FragmentState {
        module: &wireframe_shader,
        entry_point: Some("fs_main"),
        compilation_options: Default::default(),
        targets: &[Some(wgpu::ColorTargetState {
          format: config.format,
          blend: Some(wgpu::BlendState::REPLACE),
          write_mask: wgpu::ColorWrites::ALL,
        })],
      }),
      primitive: wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::LineList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        polygon_mode: wgpu::PolygonMode::Fill,
        unclipped_depth: false,
        conservative: false,
      },
      depth_stencil,
      multisample: wgpu::MultisampleState::default(),
      multiview_mask: None,
      cache: None,
    });

    let depth_view = create_depth_texture(&device, &config);

    // Line-field path: G-buffer + fullscreen LIC.
    let (gbuffer_dir_view, gbuffer_pos_view) = create_gbuffer_textures(&device, &config);
    let (noise_view, noise_sampler) = create_noise_texture(&device, &queue);
    let noise_scale = NOISE_CYCLES_PER_MESH_WIDTH / mesh_width.max(f32::EPSILON);

    let lic_uniform = LicUniform {
      viewport: [config.width as f32, config.height as f32],
      noise_scale,
      omega: wave_omega,
      time: 0.0,
      contrast: LIC_CONTRAST,
      _pad0: 0.0,
      _pad1: 0.0,
    };
    let lic_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Lic Buffer"),
      contents: bytemuck::cast_slice(&[lic_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    // Shared by the G-buffer pass (which reads the viewport in its fragment
    // stage) and the LIC pass.
    let lic_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("lic_bind_group_layout"),
      entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
          ty: wgpu::BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      }],
    });
    let lic_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("lic_bind_group"),
      layout: &lic_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: lic_buffer.as_entire_binding(),
      }],
    });

    let gbuffer_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
      label: Some("G-buffer sampler"),
      address_mode_u: wgpu::AddressMode::ClampToEdge,
      address_mode_v: wgpu::AddressMode::ClampToEdge,
      address_mode_w: wgpu::AddressMode::ClampToEdge,
      mag_filter: wgpu::FilterMode::Nearest,
      min_filter: wgpu::FilterMode::Nearest,
      mipmap_filter: wgpu::MipmapFilterMode::Nearest,
      ..Default::default()
    });
    // The G-buffer carries a discontinuous direction and a world position, so
    // it is sampled nearest (non-filtering): interpolation across a silhouette
    // would blend a foreground tangent with a background one.
    let gbuffer_tex_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gbuffer_tex_bind_group_layout"),
        entries: &[
          wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
              sample_type: wgpu::TextureSampleType::Float { filterable: false },
              view_dimension: wgpu::TextureViewDimension::D2,
              multisampled: false,
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
              sample_type: wgpu::TextureSampleType::Float { filterable: false },
              view_dimension: wgpu::TextureViewDimension::D2,
              multisampled: false,
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
          },
        ],
      });
    let gbuffer_tex_bind_group = create_gbuffer_bind_group(
      &device,
      &gbuffer_tex_bind_group_layout,
      &gbuffer_dir_view,
      &gbuffer_pos_view,
      &gbuffer_sampler,
    );

    let noise_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("noise_bind_group_layout"),
        entries: &[
          wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
              sample_type: wgpu::TextureSampleType::Float { filterable: true },
              view_dimension: wgpu::TextureViewDimension::D3,
              multisampled: false,
            },
            count: None,
          },
          wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
          },
        ],
      });
    let noise_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("noise_bind_group"),
      layout: &noise_bind_group_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: wgpu::BindingResource::TextureView(&noise_view),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: wgpu::BindingResource::Sampler(&noise_sampler),
        },
      ],
    });

    let gbuffer_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("G-buffer Shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("render/gbuffer.wgsl").into()),
    });
    let gbuffer_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("G-buffer Pipeline Layout"),
      bind_group_layouts: &[
        Some(&camera_bind_group_layout),
        Some(&lic_bind_group_layout),
      ],
      immediate_size: 0,
    });
    let gbuffer_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("G-buffer Pipeline"),
      layout: Some(&gbuffer_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &gbuffer_shader,
        entry_point: Some("vs_main"),
        compilation_options: Default::default(),
        buffers: &[Vertex::desc()],
      },
      fragment: Some(wgpu::FragmentState {
        module: &gbuffer_shader,
        entry_point: Some("fs_main"),
        compilation_options: Default::default(),
        targets: &[
          Some(wgpu::ColorTargetState {
            format: GBUFFER_FORMAT,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
          }),
          Some(wgpu::ColorTargetState {
            format: GBUFFER_FORMAT,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
          }),
        ],
      }),
      primitive: wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        polygon_mode: wgpu::PolygonMode::Fill,
        unclipped_depth: false,
        conservative: false,
      },
      depth_stencil: Some(wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
      }),
      multisample: wgpu::MultisampleState::default(),
      multiview_mask: None,
      cache: None,
    });

    let lic_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("LIC Shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("render/lic.wgsl").into()),
    });
    let lic_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("LIC Pipeline Layout"),
      bind_group_layouts: &[
        Some(&gbuffer_tex_bind_group_layout),
        Some(&noise_bind_group_layout),
        Some(&lic_bind_group_layout),
        Some(&bounds_bind_group_layout),
      ],
      immediate_size: 0,
    });
    let lic_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("LIC Pipeline"),
      layout: Some(&lic_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &lic_shader,
        entry_point: Some("vs_main"),
        compilation_options: Default::default(),
        buffers: &[],
      },
      fragment: Some(wgpu::FragmentState {
        module: &lic_shader,
        entry_point: Some("fs_main"),
        compilation_options: Default::default(),
        targets: &[Some(wgpu::ColorTargetState {
          format: config.format,
          blend: Some(wgpu::BlendState::REPLACE),
          write_mask: wgpu::ColorWrites::ALL,
        })],
      }),
      primitive: wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        polygon_mode: wgpu::PolygonMode::Fill,
        unclipped_depth: false,
        conservative: false,
      },
      depth_stencil: None,
      multisample: wgpu::MultisampleState::default(),
      multiview_mask: None,
      cache: None,
    });

    let egui_ctx = egui::Context::default();
    let egui_winit_state = EguiWinitState::new(
      egui_ctx.clone(),
      egui::ViewportId::ROOT,
      &window,
      Some(window.scale_factor() as f32),
      None,
      None,
    );
    let egui_renderer = EguiRenderer::new(&device, config.format, EguiRendererOptions::default());

    Self {
      surface,
      device,
      queue,
      config,
      size,
      render_pipeline,
      wireframe_pipeline,
      depth_view,
      gbuffer_pipeline,
      lic_pipeline,
      gbuffer_dir_view,
      gbuffer_pos_view,
      gbuffer_sampler,
      gbuffer_tex_bind_group_layout,
      gbuffer_tex_bind_group,
      noise_bind_group,
      lic_buffer,
      lic_bind_group,
      noise_scale,
      camera,
      camera_uniform,
      camera_buffer,
      camera_bind_group,
      mesh_buffer,
      bounds_buffer,
      bounds_bind_group,
      start_time,
      wave_amplitude,
      wave_omega,
      wave_buffer,
      wave_bind_group,
      mouse_pressed: false,
      last_mouse_pos: None,
      scene_switcher: SceneSwitcher::new(demo),
      scene,
      selection,
      mesh_width,
      egui_ctx,
      egui_winit_state,
      egui_renderer,
    }
  }

  /// Displays `selection` of the *current* scene, rebuilding exactly the
  /// pieces that depend on it: the mesh buffer, the colormap bounds, and the
  /// standing-wave parameters. Unconditional -- callers that only want to
  /// act on an actual change (the common case) go through
  /// [`Self::set_field`] instead.
  fn apply_field(&mut self, selection: Selection) {
    self.selection = selection;
    let display = build_field_display(&self.device, &self.scene, selection, self.mesh_width);
    self.mesh_buffer = display.mesh_buffer;
    self.wave_amplitude = display.wave_amplitude;
    self.wave_omega = display.wave_omega;
    self.start_time = std::time::Instant::now();

    let bounds_uniform = BoundsUniform {
      min_val: display.field_min,
      max_val: display.field_max,
      _pad1: 0.0,
      _pad2: 0.0,
    };
    self.queue.write_buffer(
      &self.bounds_buffer,
      0,
      bytemuck::cast_slice(&[bounds_uniform]),
    );
  }

  /// Switches the displayed field within the current scene. Everything else
  /// (camera, pipelines, egui) stays untouched.
  fn set_field(&mut self, selection: Selection) {
    if selection == self.selection {
      return;
    }
    self.apply_field(selection);
  }

  /// Requests a switch to a different built-in scene. The rebuild --
  /// including, for [`Demo::SphericalHarmonics`], a dense eigensolve -- runs
  /// on a background thread via `scene_switcher` rather than blocking this
  /// call, so the UI stays responsive while it completes; [`Self::poll_scene_load`]
  /// is what actually applies the result once it lands.
  fn set_demo(&mut self, demo: Demo) {
    self.scene_switcher.request(demo, move || build_scene(demo));
  }

  /// Applies a scene that just finished loading (or, at startup, the initial
  /// one): a new topology, coordinates and field set, plus the camera's
  /// natural orientation for it. The selection and camera state from the old
  /// scene are never reused as-is here (unlike [`Self::set_field`]'s
  /// early-out) -- a selection valid in one scene can be out of range in
  /// another, and a camera tuned for a sphere is not a natural start for a
  /// flat reference cell, or vice versa.
  fn apply_new_scene(&mut self, scene: Scene, selection: Selection) {
    self.mesh_width = scene
      .coords
      .to_edge_lengths(&scene.topology)
      .mesh_width_max() as f32;
    self.noise_scale = NOISE_CYCLES_PER_MESH_WIDTH / self.mesh_width.max(f32::EPSILON);
    self.scene = scene;
    self.apply_field(selection);

    self.camera = default_camera(&self.scene, self.camera.aspect);
    self.update_camera_buffer();
  }

  /// Non-blocking: applies a scene switch requested through [`Self::set_demo`]
  /// exactly once, the frame its background build finishes. Called once per
  /// frame regardless of whether a load is in flight.
  fn poll_scene_load(&mut self) {
    if let Some((scene, selection)) = self.scene_switcher.poll() {
      self.apply_new_scene(scene, selection);
    }
  }

  fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
    if new_size.width > 0 && new_size.height > 0 {
      self.size = new_size;
      self.config.width = new_size.width;
      self.config.height = new_size.height;
      self.surface.configure(&self.device, &self.config);
      self.depth_view = create_depth_texture(&self.device, &self.config);
      let (dir_view, pos_view) = create_gbuffer_textures(&self.device, &self.config);
      self.gbuffer_dir_view = dir_view;
      self.gbuffer_pos_view = pos_view;
      self.gbuffer_tex_bind_group = create_gbuffer_bind_group(
        &self.device,
        &self.gbuffer_tex_bind_group_layout,
        &self.gbuffer_dir_view,
        &self.gbuffer_pos_view,
        &self.gbuffer_sampler,
      );

      self.camera.aspect = self.config.width as f32 / self.config.height as f32;
      self.camera_uniform.update_view_proj(&self.camera);
      self.queue.write_buffer(
        &self.camera_buffer,
        0,
        bytemuck::cast_slice(&[self.camera_uniform]),
      );
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
            self.update_camera_buffer();
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
        self.update_camera_buffer();
      }
      _ => {}
    }
  }

  fn update_camera_buffer(&mut self) {
    self.camera_uniform.update_view_proj(&self.camera);
    self.queue.write_buffer(
      &self.camera_buffer,
      0,
      bytemuck::cast_slice(&[self.camera_uniform]),
    );
  }

  /// Builds and tessellates the control panel: a scene picker, that scene's
  /// field picker, and the camera-mode toggle. Reads out of `self` into plain
  /// locals before entering the `egui::Context::run_ui` closure, since `ctx`
  /// (a clone of `self.egui_ctx`) is what the closure captures -- borrowing
  /// `self` itself inside it would conflict with the `&mut self` calls
  /// (`set_demo`/`set_field`) right after.
  fn run_ui(&mut self, window: &Window) -> (Vec<egui::ClippedPrimitive>, egui::TexturesDelta, f32) {
    let ctx = self.egui_ctx.clone();
    let raw_input = self.egui_winit_state.take_egui_input(window);

    let mut demo = self.scene_switcher.current;
    let loading = self.scene_switcher.is_loading();
    let field_names: Vec<&str> = self.scene.fields.iter().map(|f| f.name.as_str()).collect();
    let line_field_names: Vec<&str> = self
      .scene
      .line_fields
      .iter()
      .map(|f| f.name.as_str())
      .collect();
    let mut selection = self.selection;
    let mut top_down = self.camera.top_down;

    let full_output = ctx.run_ui(raw_input, |ctx| {
      egui::Window::new("Scene").show(ctx, |ui| {
        // Disabled while a switch is loading: picking a third demo mid-load
        // is allowed by `SceneSwitcher` itself, but a frozen picker is the
        // clearer signal that a rebuild (the sphere's dense eigensolve, in
        // particular) is already in flight.
        ui.add_enabled_ui(!loading, |ui| {
          for d in Demo::ALL {
            ui.radio_value(&mut demo, d, d.label());
          }
        });
        if loading {
          ui.label(format!("Loading {}...", demo.label()));
        }
        ui.separator();

        for (i, name) in field_names.iter().enumerate() {
          ui.radio_value(&mut selection, Selection::Scalar(i), *name);
        }
        if !line_field_names.is_empty() {
          ui.separator();
          for (i, name) in line_field_names.iter().enumerate() {
            ui.radio_value(&mut selection, Selection::Line(i), *name);
          }
        }

        ui.separator();
        ui.checkbox(&mut top_down, "Top-down (orthographic, drag to pan)");
      });
    });

    // A scene switch replaces the field set and the camera's natural
    // orientation wholesale, so the `selection`/`top_down` picked in this
    // same frame belong to the *old* scene and must not be applied
    // afterward -- `set_demo` already chooses both anew for the scene it
    // switches to, once the background load completes.
    if demo != self.scene_switcher.current {
      self.set_demo(demo);
    } else {
      self.set_field(selection);
      if top_down != self.camera.top_down {
        self.camera.top_down = top_down;
        self.update_camera_buffer();
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
    // is built is what makes the field/vector-field pickers reflect the new
    // scene on the very frame it lands, rather than one frame late.
    self.poll_scene_load();
    let (paint_jobs, textures_delta, pixels_per_point) = self.run_ui(window);

    // Registered unconditionally, ahead of the surface-acquire early-returns
    // below: egui reports a texture delta exactly once, on the frame it
    // changes, so dropping it on a `Timeout`/`Occluded`/`Outdated` frame would
    // lose that texture (e.g. the font atlas) for the rest of the session.
    for (id, image_delta) in &textures_delta.set {
      self
        .egui_renderer
        .update_texture(&self.device, &self.queue, *id, image_delta);
    }

    let wave_uniform = WaveUniform {
      time: self.start_time.elapsed().as_secs_f32(),
      amplitude: self.wave_amplitude,
      omega: self.wave_omega,
      _pad: 0.0,
    };
    self
      .queue
      .write_buffer(&self.wave_buffer, 0, bytemuck::cast_slice(&[wave_uniform]));

    // The LIC pass shares the tint clock with the wave: the direction is
    // static, so `omega`/`time` swing only the magnitude tint, never the lines.
    let lic_uniform = LicUniform {
      viewport: [self.config.width as f32, self.config.height as f32],
      noise_scale: self.noise_scale,
      omega: self.wave_omega,
      time: self.start_time.elapsed().as_secs_f32(),
      contrast: LIC_CONTRAST,
      _pad0: 0.0,
      _pad1: 0.0,
    };
    self
      .queue
      .write_buffer(&self.lic_buffer, 0, bytemuck::cast_slice(&[lic_uniform]));

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
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Render Encoder"),
      });

    // A line field takes the G-buffer + fullscreen-LIC path; a scalar field
    // takes the direct colormap fill. The mark differs because the reduced
    // grade does -- this is [`Scene`]'s min(k, n-k) rule surfacing at draw
    // time, not a dimension/grade special case in the core. Both then draw the
    // wireframe on top and hand off to egui.
    let clear_color = wgpu::Color {
      r: 0.1,
      g: 0.1,
      b: 0.1,
      a: 1.0,
    };
    if self.selection.is_line() {
      // G-buffer: the surface's screen tangent, magnitude, coverage, world
      // position and shade, into two offscreen targets and the shared depth.
      {
        let mut gbuffer_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
          label: Some("G-buffer Pass"),
          color_attachments: &[
            Some(wgpu::RenderPassColorAttachment {
              view: &self.gbuffer_dir_view,
              resolve_target: None,
              ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
              },
              depth_slice: None,
            }),
            Some(wgpu::RenderPassColorAttachment {
              view: &self.gbuffer_pos_view,
              resolve_target: None,
              ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
              },
              depth_slice: None,
            }),
          ],
          depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &self.depth_view,
            depth_ops: Some(wgpu::Operations {
              load: wgpu::LoadOp::Clear(1.0),
              store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
          }),
          timestamp_writes: None,
          occlusion_query_set: None,
          multiview_mask: None,
        });
        gbuffer_pass.set_pipeline(&self.gbuffer_pipeline);
        gbuffer_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        gbuffer_pass.set_bind_group(1, &self.lic_bind_group, &[]);
        gbuffer_pass.set_vertex_buffer(0, self.mesh_buffer.vertex_buffer.slice(..));
        gbuffer_pass.set_index_buffer(
          self.mesh_buffer.index_buffer.slice(..),
          wgpu::IndexFormat::Uint32,
        );
        gbuffer_pass.draw_indexed(0..self.mesh_buffer.num_indices, 0, 0..1);
      }
      // Fullscreen LIC: integrate the tangent, tint by the animated magnitude,
      // composite over the shaded surface, into the swapchain view.
      {
        let mut lic_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
          label: Some("LIC Pass"),
          color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
              load: wgpu::LoadOp::Clear(clear_color),
              store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
          })],
          depth_stencil_attachment: None,
          timestamp_writes: None,
          occlusion_query_set: None,
          multiview_mask: None,
        });
        lic_pass.set_pipeline(&self.lic_pipeline);
        lic_pass.set_bind_group(0, &self.gbuffer_tex_bind_group, &[]);
        lic_pass.set_bind_group(1, &self.noise_bind_group, &[]);
        lic_pass.set_bind_group(2, &self.lic_bind_group, &[]);
        lic_pass.set_bind_group(3, &self.bounds_bind_group, &[]);
        lic_pass.draw(0..3, 0..1);
      }
      // Wireframe over the LIC, reusing the G-buffer's depth so back edges stay
      // occluded.
      {
        let mut wire_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
          label: Some("Wireframe Pass"),
          color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: wgpu::Operations {
              load: wgpu::LoadOp::Load,
              store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
          })],
          depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &self.depth_view,
            depth_ops: Some(wgpu::Operations {
              load: wgpu::LoadOp::Load,
              store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
          }),
          timestamp_writes: None,
          occlusion_query_set: None,
          multiview_mask: None,
        });
        wire_pass.set_pipeline(&self.wireframe_pipeline);
        wire_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        wire_pass.set_bind_group(1, &self.wave_bind_group, &[]);
        wire_pass.set_vertex_buffer(0, self.mesh_buffer.vertex_buffer.slice(..));
        wire_pass.set_index_buffer(
          self.mesh_buffer.wireframe_index_buffer.slice(..),
          wgpu::IndexFormat::Uint32,
        );
        wire_pass.draw_indexed(0..self.mesh_buffer.num_wireframe_indices, 0, 0..1);
      }
    } else {
      let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
          view: &view,
          resolve_target: None,
          ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(clear_color),
            store: wgpu::StoreOp::Store,
          },
          depth_slice: None,
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
          view: &self.depth_view,
          depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store: wgpu::StoreOp::Store,
          }),
          stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
      });

      // Draw filled triangles
      render_pass.set_pipeline(&self.render_pipeline);
      render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
      render_pass.set_bind_group(1, &self.bounds_bind_group, &[]);
      render_pass.set_bind_group(2, &self.wave_bind_group, &[]);
      render_pass.set_vertex_buffer(0, self.mesh_buffer.vertex_buffer.slice(..));
      render_pass.set_index_buffer(
        self.mesh_buffer.index_buffer.slice(..),
        wgpu::IndexFormat::Uint32,
      );
      render_pass.draw_indexed(0..self.mesh_buffer.num_indices, 0, 0..1);

      // Draw wireframe edges on top
      render_pass.set_pipeline(&self.wireframe_pipeline);
      render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
      render_pass.set_bind_group(1, &self.wave_bind_group, &[]);
      render_pass.set_vertex_buffer(0, self.mesh_buffer.vertex_buffer.slice(..));
      render_pass.set_index_buffer(
        self.mesh_buffer.wireframe_index_buffer.slice(..),
        wgpu::IndexFormat::Uint32,
      );
      render_pass.draw_indexed(0..self.mesh_buffer.num_wireframe_indices, 0, 0..1);
    }

    let screen_descriptor = ScreenDescriptor {
      size_in_pixels: [self.config.width, self.config.height],
      pixels_per_point,
    };
    let egui_cmd_buffers = self.egui_renderer.update_buffers(
      &self.device,
      &self.queue,
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

    self.queue.submit(
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
