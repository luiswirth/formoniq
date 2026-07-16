//! The windowed viewer: winit event loop, wgpu surface, egui integration and
//! input handling. Consumes the gallery model (`gallery.rs`) and the panel
//! (`ui/panel.rs`), and owns every GPU resource -- pipelines, uniforms, the
//! frame graph -- driving one scene's worth of rendering per frame.

use std::sync::Arc;

use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, RenderPipeline, Surface, SurfaceConfiguration};
use winit::{
  application::ApplicationHandler,
  event::*,
  event_loop::{ActiveEventLoop, EventLoop},
  window::{Window, WindowId},
};

use crate::demos::default_selection;
use crate::gallery::{Gallery, MeshSource, View};
use crate::render::{
  camera::{Camera, CameraUniform},
  mesh::{MeshBuffer, Vertex},
  streamline::StreamEndpoint,
};
use crate::scene::Scene;
use crate::ui::panel::{Entry, LineMark, PanelModel, Selection};

use egui_wgpu::{
  Renderer as EguiRenderer, RendererOptions as EguiRendererOptions, ScreenDescriptor,
};
use egui_winit::State as EguiWinitState;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
// Supersampling factor applied per axis (so 4x the pixel count) to every scene
// pass -- the fill, the wireframe, and the G-buffer/LIC pair -- before a box
// filter downsamples back to the swapchain. MSAA was rejected for the
// line-field path: its resolve would blend the G-buffer's direction and world
// position across a silhouette, corrupting the very data the LIC pass reads
// back (this is also why the G-buffer sampler is nearest, not linear). A
// single supersampled offscreen target antialiases both paths uniformly with
// no such hazard, at the cost of the extra fill rate. Must match `SCALE` in
// `render/downsample.wgsl`, which cannot read a Rust constant.
const SSAA_SCALE: u32 = 2;

// The line-field G-buffer targets: `dir_mag` (screen tangent xy, magnitude,
// coverage) and `pos_shade` (world position, Lambert shade). Half-float so the
// world position survives for the LIC pass's object-space noise lookup.
const GBUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
// Edge length of the cubic object-space noise texture the LIC integrates.
const NOISE_SIZE: u32 = 64;
// How hard the along-line noise average is stretched back out of grey.
const LIC_CONTRAST: f32 = 5.0;
// Object-space noise frequency, in cycles per scene extent (its radius): fixes
// the streamline texture to the surface at a density that reads as lines rather
// than mush. Keyed on the object's own size ([`scene_extent`]), the same
// object-intrinsic scale the wave amplitude and wireframe width use -- never on
// the edge length, a property of the triangulation and not the object: refining
// the mesh would then drive the noise frequency past the pixel Nyquist and
// alias the streamlines into static.
const NOISE_CYCLES_PER_EXTENT: f32 = 10.0;
// Streamline separation and ribbon half-width, each as a fraction of the scene
// extent (its radius) -- object-intrinsic, like the wave amplitude and wireframe
// width, so the line density and thickness are properties of the object, not the
// triangulation. The separation is the one knob that sets how dense the evenly
// spaced curves are.
const STREAMLINE_SEPARATION_FRACTION: f32 = 0.09;
const STREAMLINE_WIDTH_FRACTION: f32 = 0.005;

// Peak standing-wave displacement, as a fraction of the scene's own coordinate
// extent (its radius) -- an object-intrinsic scale, independent of how finely
// the object is meshed. At this fraction a grade-$l$ eigenmode swells its
// positive lobes to nearly twice the radius and pinches its negative lobes
// almost to the center, so the deformed surface reads as the familiar
// orbital-lobe shape rather than a faint ripple. Kept below 1 so a negative
// lobe never overshoots the origin and inverts the surface.
const WAVE_AMPLITUDE_FRACTION: f32 = 0.9;

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

/// The supersampled offscreen resolution every scene pass renders at: the
/// swapchain's own size scaled by [`SSAA_SCALE`] per axis.
fn supersampled_size(config: &SurfaceConfiguration) -> (u32, u32) {
  (
    config.width.max(1) * SSAA_SCALE,
    config.height.max(1) * SSAA_SCALE,
  )
}

fn create_depth_texture(device: &Device, width: u32, height: u32) -> wgpu::TextureView {
  let size = wgpu::Extent3d {
    width,
    height,
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

/// The offscreen target every scene pass (the direct fill/wireframe path and
/// the LIC path alike) draws into, at the supersampled resolution -- the
/// downsample pass then box-filters this down into the swapchain view.
/// Recreated on resize alongside the depth texture.
fn create_scene_color_texture(
  device: &Device,
  format: wgpu::TextureFormat,
  width: u32,
  height: u32,
) -> wgpu::TextureView {
  let texture = device.create_texture(&wgpu::TextureDescriptor {
    label: Some("Scene Color Texture (supersampled)"),
    size: wgpu::Extent3d {
      width,
      height,
      depth_or_array_layers: 1,
    },
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D2,
    format,
    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    view_formats: &[],
  });
  texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// The bind group for the downsample pass's one texture binding. Rebuilt
/// whenever the scene color view is (at startup and on resize).
fn create_scene_color_bind_group(
  device: &Device,
  layout: &wgpu::BindGroupLayout,
  scene_color_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
  device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: Some("scene_color_bind_group"),
    layout,
    entries: &[wgpu::BindGroupEntry {
      binding: 0,
      resource: wgpu::BindingResource::TextureView(scene_color_view),
    }],
  })
}

/// The two line-field G-buffer render targets, at the supersampled resolution
/// so they line up texel-for-texel with the scene color target the LIC pass
/// writes into. Recreated on resize alongside the depth texture.
fn create_gbuffer_textures(
  device: &Device,
  width: u32,
  height: u32,
) -> (wgpu::TextureView, wgpu::TextureView) {
  let make = |label: &str| {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
      label: Some(label),
      size: wgpu::Extent3d {
        width,
        height,
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

/// The mesh buffer and bounds/wave uniforms for showing one field of a
/// scene: the one place a [`Selection`] turns into pixels, called both at
/// startup and whenever the UI switches it.
struct FieldDisplay {
  mesh_buffer: MeshBuffer,
  /// The traced streamlines of a line field, `None` for a scalar field. Built
  /// once here, since the field is static per mode; the render pass only
  /// re-tints it per frame.
  streamlines: Option<crate::render::streamline::StreamlineBuffer>,
  field_min: f32,
  field_max: f32,
  wave_amplitude: f32,
  wave_omega: f32,
}

fn build_field_display(
  device: &Device,
  scene: &Scene,
  selection: Selection,
  amplitude_scale: f32,
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
      let (field_min, field_max) = if field.eigenvalue.is_some() {
        (-field_scale, field_scale)
      } else {
        (raw_min, raw_max)
      };

      FieldDisplay {
        mesh_buffer,
        streamlines: None,
        field_min,
        field_max,
        wave_amplitude,
        wave_omega,
      }
    }
    Selection::Line(index) => {
      let field = &scene.line_fields[index];
      let mesh_buffer = MeshBuffer::from_line_field(device, &scene.topology, &scene.coords, field);
      // The integral curves of the true Whitney field, traced on the manifold
      // at a separation fixed to the object's own extent (not its mesh width).
      let d_sep = f64::from(STREAMLINE_SEPARATION_FRACTION * amplitude_scale);
      let traced = crate::streamline::trace(&scene.topology, &scene.coords, &field.cochain, d_sep);
      let streamlines = Some(crate::render::streamline::StreamlineBuffer::new(
        device, &traced,
      ));
      let (raw_min, raw_max) = field.bounds();
      // An eigenmode's tint is the *signed* $|V| cos(sqrt(lambda) t)$, so its
      // colormap range is symmetric $[-m, m]$ about zero -- the pulse runs
      // through the midpoint and flips as the cosine crosses zero. A static
      // field has no such pulse (wave_omega below is 0, cos(0) = 1), so its
      // tint is the unsigned $|V|_g$ itself: using its true range instead of
      // widening to symmetric keeps the colormap from spending half its
      // span on negative values the field never takes. The LIC direction is
      // static either way, so there is no geometric displacement --
      // `wave_amplitude` is 0 and only `wave_omega` (the tint clock) carries
      // the mode's frequency.
      let peak = raw_max.abs().max(raw_min.abs()).max(f32::EPSILON);
      let (field_min, field_max) = if field.eigenvalue.is_some() {
        (-peak, peak)
      } else {
        (raw_min, raw_max)
      };
      let wave_omega = field.eigenvalue.map_or(0.0, f64::sqrt) as f32;

      FieldDisplay {
        mesh_buffer,
        streamlines,
        field_min,
        field_max,
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

  // Antialiasing: every scene pass draws into this supersampled offscreen
  // target instead of the swapchain; the downsample pipeline then box-filters
  // it into the swapchain view, once, ahead of the egui pass (see
  // `SSAA_SCALE`).
  scene_color_view: wgpu::TextureView,
  downsample_pipeline: RenderPipeline,
  scene_color_bind_group_layout: wgpu::BindGroupLayout,
  scene_color_bind_group: wgpu::BindGroup,

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

  // The streamline mark for a line field: the evenly-spaced ribbon pipeline, the
  // traced curves of the current field (`None` for a scalar field or an empty
  // line field), and the per-frame width/bounds uniform. `line_mark` chooses
  // between this and the LIC path above.
  line_mark: LineMark,
  streamline_pipeline: RenderPipeline,
  streamline_buffer: Option<crate::render::streamline::StreamlineBuffer>,
  streamline_uniform_buffer: wgpu::Buffer,
  streamline_bind_group: wgpu::BindGroup,

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

  // The wireframe's world-space half-width, rewritten alongside `wave_buffer`
  // whenever `amplitude_scale` changes (a new mesh or view).
  wireframe_buffer: wgpu::Buffer,
  wireframe_bind_group: wgpu::BindGroup,

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
  // Fixed at scene-build time: the object's coordinate extent (its radius),
  // which sets the standing-wave displacement scale for whichever field is on
  // display and the LIC noise frequency (`noise_scale`) -- an object-intrinsic
  // length, so the lobes read at orbital scale and the streamlines at a fixed
  // density on any mesh, however finely triangulated.
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

/// The wireframe pipeline's world-space thickening state: the line
/// half-width, in the same world units the mesh's own coordinates are in.
/// See `wireframe.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WireframeUniform {
  half_width_world: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}

/// The streamline ribbon pipeline's per-frame state: the world-space ribbon
/// half-width, a fraction of the scene extent. The ribbons carry no colormap --
/// the surface beneath them does (see `streamline.wgsl`) -- so the fill's bounds
/// are not needed here.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct StreamlineUniform {
  half_width_world: f32,
  _pad: [f32; 3],
}

/// The wireframe's half-width as a fraction of the scene's own extent
/// ([`scene_extent`]) -- the same object-intrinsic scale
/// `WAVE_AMPLITUDE_FRACTION` uses for the standing-wave displacement --
/// rather than a fixed screen-pixel count, so the line reads the same
/// thickness whether the mesh is zoomed to fill the screen or shrunk to a
/// corner of it.
const WIREFRAME_WIDTH_FRACTION: f32 = 0.004;

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

    // Show the window immediately: the gallery opens on a cheap placeholder of
    // the starting mesh and builds the first grade's eigenmodes in the
    // background, swapping it in when it lands (`poll_view_load`), rather than
    // blocking the first frame on the solve.
    let (gallery, scene) = Gallery::new(MeshSource::START);
    let selection = default_selection(&scene);
    // The coordinate extent fixes both the standing-wave displacement and the
    // LIC noise frequency (see the `State` field): object-intrinsic, so neither
    // is tuned to the sphere.
    let amplitude_scale = scene_extent(&scene) as f32;
    let display = build_field_display(&device, &scene, selection, amplitude_scale);
    let (field_min, field_max) = (display.field_min, display.field_max);
    let mesh_buffer = display.mesh_buffer;
    let streamline_buffer = display.streamlines;
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

    // The wireframe's world-space half-width: rewritten alongside `amplitude_scale`
    // (a new mesh or view), so it stays a fixed fraction of the object's own
    // size rather than a screen-pixel count.
    let wireframe_uniform = WireframeUniform {
      half_width_world: WIREFRAME_WIDTH_FRACTION * amplitude_scale,
      _pad0: 0.0,
      _pad1: 0.0,
      _pad2: 0.0,
    };
    let wireframe_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Wireframe Width Buffer"),
      contents: bytemuck::cast_slice(&[wireframe_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let wireframe_width_bind_group_layout =
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
        label: Some("wireframe_width_bind_group_layout"),
      });
    let wireframe_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &wireframe_width_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: wireframe_buffer.as_entire_binding(),
      }],
      label: Some("wireframe_bind_group"),
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

    // Wireframe pipeline — instanced screen-space-thick quads, one per edge
    // (see `wireframe.wgsl`), reading each edge's two endpoints off two
    // parallel per-instance vertex buffers.
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
          Some(&wireframe_width_bind_group_layout),
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
        buffers: &[Vertex::desc_endpoint_a(), Vertex::desc_endpoint_b()],
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

    // Streamline ribbon pipeline: the billboard quads of `streamline.wgsl`, one
    // instance per traced segment, tinted by the field magnitude with alpha
    // blending so the ends taper and the node fade read over the surface.
    let streamline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("Streamline Shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("render/streamline.wgsl").into()),
    });
    let streamline_uniform = StreamlineUniform {
      half_width_world: STREAMLINE_WIDTH_FRACTION * amplitude_scale,
      _pad: [0.0; 3],
    };
    let streamline_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Streamline Uniform Buffer"),
      contents: bytemuck::cast_slice(&[streamline_uniform]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let streamline_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        }],
        label: Some("streamline_bind_group_layout"),
      });
    let streamline_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      layout: &streamline_bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: streamline_uniform_buffer.as_entire_binding(),
      }],
      label: Some("streamline_bind_group"),
    });
    let streamline_pipeline_layout =
      device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Streamline Pipeline Layout"),
        bind_group_layouts: &[
          Some(&camera_bind_group_layout),
          Some(&wave_bind_group_layout),
          Some(&streamline_bind_group_layout),
        ],
        immediate_size: 0,
      });
    let streamline_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Streamline Pipeline"),
      layout: Some(&streamline_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &streamline_shader,
        entry_point: Some("vs_main"),
        compilation_options: Default::default(),
        buffers: &[StreamEndpoint::desc_a(), StreamEndpoint::desc_b()],
      },
      fragment: Some(wgpu::FragmentState {
        module: &streamline_shader,
        entry_point: Some("fs_main"),
        compilation_options: Default::default(),
        targets: &[Some(wgpu::ColorTargetState {
          format: config.format,
          blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
      // The ribbons are an alpha-blended overlay on the surface: they test
      // against it (so a curve on the far side stays hidden) but must not write
      // depth. Writing it would let a ribbon, biased toward the camera and drawn
      // first, occlude the wireframe edge it lies along -- and depth-writing
      // translucent geometry is wrong regardless of what is drawn after.
      depth_stencil: Some(wgpu::DepthStencilState {
        depth_write_enabled: Some(false),
        ..depth_stencil.clone().unwrap()
      }),
      multisample: wgpu::MultisampleState::default(),
      multiview_mask: None,
      cache: None,
    });

    let (ss_width, ss_height) = supersampled_size(&config);
    let depth_view = create_depth_texture(&device, ss_width, ss_height);
    let scene_color_view = create_scene_color_texture(&device, config.format, ss_width, ss_height);

    // Line-field path: G-buffer + fullscreen LIC.
    let (gbuffer_dir_view, gbuffer_pos_view) =
      create_gbuffer_textures(&device, ss_width, ss_height);
    let (noise_view, noise_sampler) = create_noise_texture(&device, &queue);
    let noise_scale = NOISE_CYCLES_PER_EXTENT / amplitude_scale.max(f32::EPSILON);

    let lic_uniform = LicUniform {
      viewport: [ss_width as f32, ss_height as f32],
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

    // Downsample: the supersampling antialiasing step. A box filter over
    // `SSAA_SCALE^2` texels, applied to whichever scene pass just filled
    // `scene_color_view`, into the swapchain -- the one place both render
    // paths' output gets antialiased.
    let scene_color_bind_group_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_color_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::FRAGMENT,
          ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
          },
          count: None,
        }],
      });
    let scene_color_bind_group =
      create_scene_color_bind_group(&device, &scene_color_bind_group_layout, &scene_color_view);

    let downsample_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("Downsample Shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("render/downsample.wgsl").into()),
    });
    let downsample_pipeline_layout =
      device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Downsample Pipeline Layout"),
        bind_group_layouts: &[Some(&scene_color_bind_group_layout)],
        immediate_size: 0,
      });
    let downsample_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Downsample Pipeline"),
      layout: Some(&downsample_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &downsample_shader,
        entry_point: Some("vs_main"),
        compilation_options: Default::default(),
        buffers: &[],
      },
      fragment: Some(wgpu::FragmentState {
        module: &downsample_shader,
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
      scene_color_view,
      downsample_pipeline,
      scene_color_bind_group_layout,
      scene_color_bind_group,
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
      line_mark: LineMark::Streamlines,
      streamline_pipeline,
      streamline_buffer,
      streamline_uniform_buffer,
      streamline_bind_group,
      bounds_buffer,
      bounds_bind_group,
      start_time,
      wave_amplitude,
      wave_omega,
      wave_buffer,
      wave_bind_group,
      wireframe_buffer,
      wireframe_bind_group,
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

  /// Displays `selection` of the *current* scene, rebuilding exactly the
  /// pieces that depend on it: the mesh buffer, the colormap bounds, and the
  /// standing-wave parameters. Unconditional -- callers that only want to
  /// act on an actual change (the common case) go through
  /// [`Self::set_field`] instead.
  fn apply_field(&mut self, selection: Selection) {
    self.selection = selection;
    let display = build_field_display(&self.device, &self.scene, selection, self.amplitude_scale);
    self.mesh_buffer = display.mesh_buffer;
    self.streamline_buffer = display.streamlines;
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

    // The ribbons take their width from the current object extent, which changes
    // with the scene, so rewrite it here rather than only at startup.
    let streamline_uniform = StreamlineUniform {
      half_width_world: STREAMLINE_WIDTH_FRACTION * self.amplitude_scale,
      _pad: [0.0; 3],
    };
    self.queue.write_buffer(
      &self.streamline_uniform_buffer,
      0,
      bytemuck::cast_slice(&[streamline_uniform]),
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
    self.noise_scale = NOISE_CYCLES_PER_EXTENT / self.amplitude_scale.max(f32::EPSILON);
    let selection = default_selection(&scene);
    self.scene = scene;
    self.apply_field(selection);

    self.camera = default_camera(&self.scene, self.camera.aspect);
    self.update_camera_buffer();
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
      self.surface.configure(&self.device, &self.config);
      let (ss_width, ss_height) = supersampled_size(&self.config);
      self.depth_view = create_depth_texture(&self.device, ss_width, ss_height);
      self.scene_color_view =
        create_scene_color_texture(&self.device, self.config.format, ss_width, ss_height);
      self.scene_color_bind_group = create_scene_color_bind_group(
        &self.device,
        &self.scene_color_bind_group_layout,
        &self.scene_color_view,
      );
      let (dir_view, pos_view) = create_gbuffer_textures(&self.device, ss_width, ss_height);
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
      line_mark: self.line_mark,
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
        self.line_mark = response.line_mark;
        if response.top_down != self.camera.top_down {
          self.camera.top_down = response.top_down;
          self.update_camera_buffer();
        }
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
    self.poll_view_load();
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
    // `viewport` is the G-buffer/scene-color target's own (supersampled)
    // resolution, not the swapchain's -- it fixes the texel size the pass
    // steps in.
    let (ss_width, ss_height) = supersampled_size(&self.config);
    let lic_uniform = LicUniform {
      viewport: [ss_width as f32, ss_height as f32],
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

    // Rewritten every frame (cheap, and simpler than hunting down every place
    // `amplitude_scale` itself changes) rather than only once at startup, so
    // the wireframe's world-space half-width always tracks the scene
    // currently on display.
    let wireframe_uniform = WireframeUniform {
      half_width_world: WIREFRAME_WIDTH_FRACTION * self.amplitude_scale,
      _pad0: 0.0,
      _pad1: 0.0,
      _pad2: 0.0,
    };
    self.queue.write_buffer(
      &self.wireframe_buffer,
      0,
      bytemuck::cast_slice(&[wireframe_uniform]),
    );

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
    if self.selection.is_line() && self.line_mark == LineMark::Lic {
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
      // composite over the shaded surface, into the supersampled scene color
      // target (downsampled into the swapchain below, alongside the wireframe).
      {
        let mut lic_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
          label: Some("LIC Pass"),
          color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &self.scene_color_view,
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
            view: &self.scene_color_view,
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
        wire_pass.set_bind_group(2, &self.wireframe_bind_group, &[]);
        wire_pass.set_vertex_buffer(0, self.mesh_buffer.wireframe_a_buffer.slice(..));
        wire_pass.set_vertex_buffer(1, self.mesh_buffer.wireframe_b_buffer.slice(..));
        wire_pass.draw(0..6, 0..self.mesh_buffer.num_wireframe_edges);
      }
    } else {
      let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
          view: &self.scene_color_view,
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

      // A line field in streamline mode: the evenly-spaced ribbons over the
      // magnitude-tinted surface, before the wireframe. The surface fill above
      // is identical to a scalar field's, since a line field's `MeshBuffer`
      // carries its nodal magnitude as the fill value.
      if let Some(streamlines) = &self.streamline_buffer {
        if streamlines.num_segments > 0 {
          render_pass.set_pipeline(&self.streamline_pipeline);
          render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
          render_pass.set_bind_group(1, &self.wave_bind_group, &[]);
          render_pass.set_bind_group(2, &self.streamline_bind_group, &[]);
          render_pass.set_vertex_buffer(0, streamlines.endpoint_a.slice(..));
          render_pass.set_vertex_buffer(1, streamlines.endpoint_b.slice(..));
          render_pass.draw(0..6, 0..streamlines.num_segments);
        }
      }

      // Draw wireframe edges on top
      render_pass.set_pipeline(&self.wireframe_pipeline);
      render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
      render_pass.set_bind_group(1, &self.wave_bind_group, &[]);
      render_pass.set_bind_group(2, &self.wireframe_bind_group, &[]);
      render_pass.set_vertex_buffer(0, self.mesh_buffer.wireframe_a_buffer.slice(..));
      render_pass.set_vertex_buffer(1, self.mesh_buffer.wireframe_b_buffer.slice(..));
      render_pass.draw(0..6, 0..self.mesh_buffer.num_wireframe_edges);
    }

    // Antialiasing: box-filter the supersampled scene color target down into
    // the swapchain, once, regardless of which path above filled it.
    {
      let mut downsample_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Downsample Pass"),
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
      downsample_pass.set_pipeline(&self.downsample_pipeline);
      downsample_pass.set_bind_group(0, &self.scene_color_bind_group, &[]);
      downsample_pass.draw(0..3, 0..1);
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
