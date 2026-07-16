extern crate nalgebra as na;

use std::collections::HashMap;
use std::sync::Arc;

use exterior::ExteriorGrade;
use manifold::{geometry::coord::mesh::MeshCoords, topology::complex::Complex, Dim};
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

// Icosphere subdivision depth the gallery opens on. The Laplace-Beltrami
// eigensolve is dense in the vertex count, so keep this modest for an instant
// startup; the mesh slider goes up to `SPHERE_SUBDIVISIONS_MAX` for fidelity.
const SPHERE_SUBDIVISIONS: usize = 3;
// Upper end of the sphere refinement slider. The per-grade solve is dense
// ($O(n^3)$), and at grade 1 the edge count is what enters, so a step past this
// turns the background solve from seconds into minutes -- the cap keeps every
// reachable mesh solvable while the window stays responsive.
const SPHERE_SUBDIVISIONS_MAX: usize = 4;
// Cells per axis the unit-square grid opens on, and the upper end of its
// refinement slider. A grid this fine keeps the dense per-grade solve well
// inside the sphere's cost, so the same cap reasoning applies loosely.
const GRID_CELLS_DEFAULT: usize = 8;
const GRID_CELLS_MAX: usize = 20;
// Chosen so both grades close on a complete degeneracy shell: grade 0 fills
// $l = 0..=3$ ($sum (2l+1) = 16$) and grade 1 fills $l = 1, 2$
// ($6 + 10 = 16$), so the orbital pyramid the UI lays these out in has no
// half-built final row.
const SPHERE_MODES: usize = 16;
// Peak standing-wave displacement, as a fraction of the scene's own coordinate
// extent (its radius) -- an object-intrinsic scale, independent of how finely
// the object is meshed. At this fraction a grade-$l$ eigenmode swells its
// positive lobes to nearly twice the radius and pinches its negative lobes
// almost to the center, so the deformed surface reads as the familiar
// orbital-lobe shape rather than a faint ripple. Kept below 1 so a negative
// lobe never overshoots the origin and inverts the surface.
const WAVE_AMPLITUDE_FRACTION: f32 = 0.9;

/// The shared surface mesh the gallery solves its grades against, built once
/// so every per-grade eigensolve reuses it rather than remeshing.
type Mesh = (Complex, MeshCoords);

/// One of the CC0 surface meshes the studio ships as a built-in gallery,
/// embedded in the binary (see `assets/meshes`, and its `SOURCES.md` for
/// provenance). Chosen to span topology -- genus 0 and genus 1, down to a
/// 7-vertex torus -- so the harmonic (zero-eigenvalue) modes the gallery shows
/// at grade 1 range over $dim H^1 = 2 g$.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BuiltinMesh {
  /// Spot the cow (genus 0).
  Spot,
  /// Bob (genus 1).
  Bob,
  /// Blub the fish (genus 0); the largest, so the slowest to solve.
  Blub,
  /// The Császár polyhedron: a diagonal-free 7-vertex triangulation of the
  /// torus (genus 1).
  Csaszar,
}

impl BuiltinMesh {
  const ALL: [BuiltinMesh; 4] = [
    BuiltinMesh::Spot,
    BuiltinMesh::Bob,
    BuiltinMesh::Blub,
    BuiltinMesh::Csaszar,
  ];

  fn label(self) -> &'static str {
    match self {
      BuiltinMesh::Spot => "Spot (cow)",
      BuiltinMesh::Bob => "Bob",
      BuiltinMesh::Blub => "Blub (fish)",
      BuiltinMesh::Csaszar => "Császár torus",
    }
  }

  /// The embedded OBJ source of the mesh. Baked in with `include_str!`, so it
  /// travels with the binary and needs no filesystem at runtime (and works on
  /// wasm). If the git-LFS content was never fetched this is the LFS pointer
  /// text, which [`crate::io::obj::parse`] reports as an empty mesh rather than
  /// silently mis-loading.
  fn obj(self) -> &'static str {
    match self {
      BuiltinMesh::Spot => include_str!("../assets/meshes/spot.obj"),
      BuiltinMesh::Bob => include_str!("../assets/meshes/bob.obj"),
      BuiltinMesh::Blub => include_str!("../assets/meshes/blub.obj"),
      BuiltinMesh::Csaszar => include_str!("../assets/meshes/csaszar.obj"),
    }
  }
}

/// The chosen source of the surface mesh the eigenmode gallery solves on -- a
/// runtime input, not a fixed sphere. A generated family carries its refinement
/// (moved by a slider); a built-in or a user-loaded file carries none.
///
/// Nothing downstream distinguishes the variants: the eigensolve, the reduced
/// grade dispatch and the degeneracy clustering are all mesh-agnostic, and the
/// camera reads a curved surface (perspective orbit) from a flat one (top-down
/// orthographic) off the coordinates alone. The sphere's spherical harmonics
/// are then one mesh's spectrum among others, not a privileged case.
///
/// [`Self::Custom`] is not regenerable -- the loaded mesh lives in the gallery,
/// keyed by this descriptor for the picker -- so it is the one variant
/// [`Self::build`] cannot serve.
#[derive(Clone, PartialEq)]
enum MeshSource {
  /// An icosphere of the given subdivision depth.
  Sphere { subdivisions: usize },
  /// A triangulated unit square with the given number of cells per axis, a
  /// surface with boundary -- so its Hodge-Laplace spectrum is the natural
  /// (Neumann) one, and its degeneracies (a mode and its transpose share an
  /// eigenvalue) fill the pyramid's rows just as the sphere's multiplets do.
  Grid { cells_axis: usize },
  /// One of the embedded CC0 gallery meshes.
  Builtin(BuiltinMesh),
  /// A mesh the user loaded from an OBJ file, named for the picker.
  Custom { name: String },
}

impl MeshSource {
  /// The mesh the gallery opens on: the icosphere, matching the historical
  /// startup.
  const START: MeshSource = MeshSource::Sphere {
    subdivisions: SPHERE_SUBDIVISIONS,
  };

  fn label(&self) -> String {
    match self {
      MeshSource::Sphere { .. } => "Sphere".to_string(),
      MeshSource::Grid { .. } => "Grid".to_string(),
      MeshSource::Builtin(builtin) => builtin.label().to_string(),
      MeshSource::Custom { name } => name.clone(),
    }
  }

  /// Builds the mesh for a regenerable source. `Err` carries a human-readable
  /// reason (a malformed embedded asset -- e.g. an unfetched LFS pointer),
  /// which the caller surfaces without disturbing the current mesh.
  ///
  /// Panics on [`Self::Custom`]: a loaded mesh is installed directly and never
  /// rebuilt from its descriptor.
  fn build(&self) -> Result<Mesh, String> {
    match self {
      MeshSource::Sphere { subdivisions } => {
        Ok(manifold::gen::sphere::mesh_sphere_surface(*subdivisions))
      }
      MeshSource::Grid { cells_axis } => {
        let (topology, coords) =
          manifold::gen::cartesian::CartesianMeshInfo::new_unit(2, *cells_axis)
            .compute_coord_complex();
        // The renderer draws in 3D and reads the surface normal off the
        // embedding; the grid is planar in $RR^2$, so lift it into the $z = 0$
        // plane of $RR^3$, exactly as the flat reference-cell scenes do.
        Ok((topology, coords.embed_euclidean(3)))
      }
      MeshSource::Builtin(builtin) => {
        crate::io::obj::parse(builtin.obj()).map_err(|e| format!("{}: {e}", builtin.label()))
      }
      MeshSource::Custom { .. } => {
        unreachable!("a custom mesh is installed directly, never rebuilt from its descriptor")
      }
    }
  }
}

/// What the viewer is showing -- the unit a build, and its memoization, is
/// keyed on. A picker in the UI switches this at runtime; the render path
/// treats every `Scene` alike regardless of which `View` built it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum View {
  /// A single grade of the mesh's Hodge-Laplace eigenmodes. The expensive
  /// case: each grade is an eigensolve, run once and memoized, so only the
  /// grade actually being viewed is ever computed.
  MeshGrade(ExteriorGrade),
  /// Every Whitney basis function ("local shape function") of the reference
  /// triangle. Cheap to build.
  WhitneyBasis,
  /// Every Whitney basis function ("global shape function") of the triforce
  /// teaching mesh ([`crate::mesh3d::triforce`]): the same construction as
  /// [`View::WhitneyBasis`], but a DOF simplex's support now spans the
  /// several cells incident to it instead of a single reference cell. Cheap
  /// to build, and mesh-independent like the reference-cell basis.
  WhitneyBasisMesh,
}

impl View {
  /// The starting view when the viewer opens: grade 0 of the mesh.
  const START: View = View::MeshGrade(0);

  fn label(self) -> String {
    match self {
      View::MeshGrade(grade) => format!("Eigenmodes, grade {grade}"),
      View::WhitneyBasis => "Whitney basis (reference triangle)".to_string(),
      View::WhitneyBasisMesh => "Whitney basis (triforce mesh)".to_string(),
    }
  }
}

/// Builds the scene for `view`. For a mesh grade this runs that grade's dense
/// eigensolve against the shared `mesh` -- the expensive path the caller runs
/// on a background thread and memoizes; the Whitney basis is cheap and ignores
/// the mesh.
fn build_view(view: View, mesh: &Mesh) -> Scene {
  match view {
    View::MeshGrade(grade) => {
      let (topology, coords) = mesh;
      Scene::mesh_grade(topology, coords, grade, SPHERE_MODES)
    }
    View::WhitneyBasis => Scene::whitney_basis(2),
    View::WhitneyBasisMesh => {
      let (topology, coords) = crate::mesh3d::triforce();
      Scene::whitney_basis_mesh(topology, coords)
    }
  }
}

/// The field a freshly shown scene opens on: its first mode. A sphere grade
/// carries only one render mark, so exactly one of the two lists is nonempty; a
/// scene with neither (never produced here) falls back harmlessly to the first
/// scalar slot.
fn default_selection(scene: &Scene) -> Selection {
  if !scene.fields.is_empty() {
    Selection::Scalar(0)
  } else if !scene.line_fields.is_empty() {
    Selection::Line(0)
  } else {
    Selection::Scalar(0)
  }
}

/// A rebuild of `T` running off the render thread, so a solve that triggers an
/// eigensolve never blocks the UI. `poll` is non-blocking and yields the
/// result exactly once, the frame it arrives.
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

/// The gallery's lazy, memoized loader. Owns the shared surface mesh; each
/// view's scene is built at most once -- the expensive mesh grades on a
/// background thread -- and cached, so revisiting a grade is instant and only
/// the grades actually viewed are ever solved.
///
/// Kept free of any GPU type; `State` owns the `Scene` it produces. Only one
/// build is ever in flight: requesting a view while another is loading replaces
/// the pending build, and a landed result is only ever installed for the view
/// it was built for.
struct Gallery {
  /// Which mesh the current `mesh` was built from, so the UI can reflect the
  /// live choice and a re-selection of the same source is a no-op.
  mesh_source: MeshSource,
  mesh: Arc<Mesh>,
  current: View,
  /// The last mesh grade viewed, so toggling to the Whitney basis and back
  /// resumes that grade rather than resetting to 0.
  last_mesh_grade: ExteriorGrade,
  cache: HashMap<View, Scene>,
  loading: Option<(View, PendingLoad<Scene>)>,
  /// The last mesh-source failure (a malformed OBJ, an unfetched asset),
  /// surfaced in the panel until the next successful mesh change. `None` when
  /// the current mesh loaded cleanly.
  error: Option<String>,
}

impl Gallery {
  /// Opens on [`View::START`], returning the cheap placeholder scene to show at
  /// once while that view's real build runs in the background (delivered later
  /// by [`Self::poll`]). The placeholder shares the loader's mesh, so the
  /// window is up on the first frame without waiting on any eigensolve.
  fn new(source: MeshSource) -> (Self, Scene) {
    // The starting source is the icosphere, whose build is infallible.
    let mesh = Arc::new(source.build().expect("the starting mesh builds"));
    let placeholder = Scene::placeholder_on(mesh.0.clone(), mesh.1.clone());
    let mut gallery = Self {
      mesh_source: source,
      mesh,
      current: View::START,
      last_mesh_grade: 0,
      cache: HashMap::new(),
      loading: None,
      error: None,
    };
    gallery.spawn(View::START);
    (gallery, placeholder)
  }

  /// Switches to a regenerable source (a generated family or a built-in mesh),
  /// building its mesh first so a failure leaves the current one untouched.
  /// `Ok(Some(scene))` is the placeholder to show while the re-solve runs;
  /// `Ok(None)` is a no-op source change, or one under a view that ignores the
  /// mesh; `Err` is a build failure to surface.
  fn select_source(&mut self, source: MeshSource) -> Result<Option<Scene>, String> {
    if source == self.mesh_source {
      return Ok(None);
    }
    let mesh = source.build()?;
    Ok(self.install_mesh(source, mesh))
  }

  /// Installs a mesh loaded from an OBJ file as the [`MeshSource::Custom`]
  /// source, named for the picker. Unlike a regenerable source there is nothing
  /// to build -- the mesh is already in hand -- so this always takes effect.
  fn load_custom(&mut self, name: String, mesh: Mesh) -> Option<Scene> {
    self.install_mesh(MeshSource::Custom { name }, mesh)
  }

  /// Adopts `mesh` under `source` and re-solves the current grade against it.
  /// The per-grade eigensolves are all tied to the old mesh, so they are
  /// dropped from the cache (both Whitney-basis views are mesh-independent --
  /// the reference triangle and the triforce mesh are fixed, not the picker's
  /// mesh -- and kept); the current grade's solve is respawned in the
  /// background.
  ///
  /// Returns a solve-free placeholder on the new mesh to show at once while
  /// that solve runs -- or `None` when the current view does not depend on the
  /// mesh (the Whitney basis), where nothing visible changes and only the
  /// stored source is updated, taking effect the next time a mesh grade is
  /// shown.
  fn install_mesh(&mut self, source: MeshSource, mesh: Mesh) -> Option<Scene> {
    self.error = None;
    self.mesh_source = source;
    self.mesh = Arc::new(mesh);
    self
      .cache
      .retain(|view, _| !matches!(view, View::MeshGrade(_)));
    match self.current {
      View::MeshGrade(_) => {
        self.spawn(self.current);
        Some(Scene::placeholder_on(
          self.mesh.0.clone(),
          self.mesh.1.clone(),
        ))
      }
      View::WhitneyBasis | View::WhitneyBasisMesh => None,
    }
  }

  fn set_error(&mut self, error: String) {
    self.error = Some(error);
  }

  fn spawn(&mut self, view: View) {
    let mesh = self.mesh.clone();
    self.loading = Some((view, PendingLoad::spawn(move || build_view(view, &mesh))));
  }

  fn is_loading(&self) -> bool {
    self.loading.is_some()
  }

  fn loading_view(&self) -> Option<View> {
    self.loading.as_ref().map(|(view, _)| *view)
  }

  /// Requests that `view` be shown. Returns `Some(scene)` to install right now
  /// -- a cache hit, instant -- or `None` when the view is either already shown
  /// or now building in the background (a spinner shows until [`Self::poll`]
  /// delivers it).
  fn request(&mut self, view: View) -> Option<Scene> {
    if let View::MeshGrade(grade) = view {
      self.last_mesh_grade = grade;
    }
    if view == self.current && self.loading.is_none() {
      return None;
    }
    if let Some(scene) = self.cache.get(&view) {
      self.current = view;
      self.loading = None;
      return Some(scene.clone());
    }
    if !self.loading.as_ref().is_some_and(|(v, _)| *v == view) {
      self.spawn(view);
    }
    None
  }

  /// `Some` exactly once, the frame a background build lands: commits it as
  /// `current`, memoizes it, and hands it back to be installed.
  fn poll(&mut self) -> Option<Scene> {
    let (view, scene) = self
      .loading
      .as_ref()
      .and_then(|(view, pending)| pending.poll().map(|scene| (*view, scene)))?;
    self.loading = None;
    self.current = view;
    self.cache.insert(view, scene.clone());
    Some(scene)
  }
}

/// The scene's coordinate extent: the largest distance of any vertex from the
/// mesh's own centroid -- its intrinsic radius, independent of where the mesh
/// sits in space. Measured about the centroid, not the origin, so a mesh
/// nowhere near the origin (a unit grid on $[0,1]^2$, an off-center loaded OBJ)
/// still reports its true size; an origin-centered unit sphere gives 1 either
/// way. Both the camera framing and the standing-wave amplitude scale off this,
/// so neither is tuned to the sphere.
fn scene_extent(scene: &Scene) -> f64 {
  let coords = &scene.coords;
  let n = coords.nvertices().max(1) as f64;
  let centroid = coords
    .coord_iter()
    .fold(na::DVector::zeros(3), |acc, c| acc + *c)
    / n;
  coords
    .coord_iter()
    .map(|c| (*c - &centroid).norm())
    .fold(0.0, f64::max)
    .max(1e-6)
}

/// The camera's natural starting orientation for a scene, derived purely from
/// its own coordinates -- not which `Demo` built it, so a future flat or 3D
/// scene gets the same sensible default without adding another `match` arm
/// here.
fn default_camera(scene: &Scene, aspect: f32) -> Camera {
  // Framing distance from the scene's own coordinate extent, not a constant
  // tuned for the sphere: an icosphere of radius 1 gives back exactly the
  // prior hardcoded 3.0, and a unit reference triangle frames itself too.
  let extent = scene_extent(scene);
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

/// One mode of the currently shown scene, as the picker needs it: the field's
/// [`Selection`], its original grade (before the reduction to a render mark),
/// its eigenvalue (for the degeneracy layout), its DOF label (for the basis
/// grid) and its full name (for the hover). The render mark the selection
/// resolves to is decided elsewhere by the reduced grade; here a mode is just
/// a selectable cell.
struct Entry<'a> {
  selection: Selection,
  grade: ExteriorGrade,
  eigenvalue: Option<f64>,
  dof_label: Option<&'a str>,
  name: &'a str,
}

/// One degeneracy shell of an eigenmode list: a maximal run of consecutive
/// modes whose eigenvalues agree up to the clustering tolerance -- one
/// degenerate eigenspace. A row of the orbital pyramid.
struct Shell {
  /// A representative eigenvalue of the shell (its first member's), labelling
  /// the row.
  eigenvalue: f64,
  /// Indices into the entry list this shell was grouped from.
  members: Vec<usize>,
}

/// The relative gap above which two consecutive eigenvalues are taken to lie in
/// *different* degeneracy shells. Within a shell the discrete eigenvalues of a
/// degenerate eigenspace differ only by the mesh's small symmetry-breaking
/// error, far below this; between distinct shells they jump by an order-one
/// fraction, far above it.
const SHELL_REL_GAP: f64 = 0.3;

/// An absolute tolerance, as a fraction of the spectrum's scale, added to the
/// relative one so a cluster of (near-)zero modes -- a harmonic space, e.g. the
/// constant 0-mode or a flat torus's two 1-cocycles -- stays together instead
/// of splitting on numerical noise, where the relative gap alone has no scale.
const SHELL_ABS_FRAC: f64 = 1e-6;

/// Groups a list of eigenmodes into its degeneracy shells by clustering
/// consecutive near-equal eigenvalues.
///
/// The modes arrive sorted by eigenvalue; a run whose successive gaps stay
/// within [`SHELL_ABS_FRAC`]$dot lambda_max + $[`SHELL_REL_GAP`]$dot lambda$ is
/// one degenerate eigenspace -- a row of the pyramid. This reads the
/// organization straight off the spectrum, with no geometry: on $S^2$ the
/// near-equal clusters are exactly the $(2l+1)$ spherical-harmonic shells,
/// while on a generic mesh with no symmetry the eigenvalues are simple, every
/// gap exceeds the tolerance, and each row collapses to a single member ordered
/// by eigenvalue.
///
/// `None` if any field carries no eigenvalue (not an eigenmode scene, e.g. the
/// raw Whitney basis), where the caller keeps a flat list instead.
fn degeneracy_shells(eigenvalues: impl IntoIterator<Item = Option<f64>>) -> Option<Vec<Shell>> {
  let lambdas: Vec<f64> = eigenvalues.into_iter().collect::<Option<Vec<f64>>>()?;
  let scale = lambdas.iter().map(|l| l.abs()).fold(0.0, f64::max).max(1.0);
  let atol = SHELL_ABS_FRAC * scale;
  let mut shells: Vec<Shell> = Vec::new();
  for (idx, &lambda) in lambdas.iter().enumerate() {
    let same_shell =
      idx > 0 && lambda - lambdas[idx - 1] <= atol + SHELL_REL_GAP * lambdas[idx - 1].abs();
    if same_shell {
      shells.last_mut().unwrap().members.push(idx);
    } else {
      shells.push(Shell {
        eigenvalue: lambda,
        members: vec![idx],
      });
    }
  }
  Some(shells)
}

/// Renders the modes of the currently shown scene as a picker. Eigenmodes
/// (the harmonics) lay out as the orbital pyramid by degeneracy shell; raw
/// Whitney basis functions (LSFs and GSFs alike) lay out as a grid by grade
/// instead, since they carry a DOF label but no eigenvalue; anything carrying
/// neither -- not produced today, but the totality this dispatch is answering
/// to -- falls back to one flat list.
fn render_modes(ui: &mut egui::Ui, entries: &[Entry], selection: &mut Selection, n: Dim) {
  if let Some(shells) = degeneracy_shells(entries.iter().map(|e| e.eigenvalue)) {
    pyramid(ui, &shells, entries, selection);
  } else if entries.iter().all(|e| e.dof_label.is_some()) {
    grade_grid(ui, entries, selection, n);
  } else {
    for entry in entries {
      let selected = *selection == entry.selection;
      if ui.selectable_label(selected, entry.name).clicked() {
        *selection = entry.selection;
      }
    }
  }
}

/// Lays out a Whitney basis gallery (LSFs or GSFs) as one row per grade,
/// ordered $0..=n$ and labelled by [`grade_mark_label`], each row a wrapped
/// flow of DOF cells -- unlike the eigenmode pyramid there is no natural width
/// to center on, since a mesh's edge count need not match its vertex or face
/// count. Hovering a cell shows the basis function's full name.
fn grade_grid(ui: &mut egui::Ui, entries: &[Entry], selection: &mut Selection, n: Dim) {
  const CELL: [f32; 2] = [30.0, 22.0];
  for grade in 0..=n {
    let members: Vec<usize> = entries
      .iter()
      .enumerate()
      .filter(|(_, e)| e.grade == grade)
      .map(|(i, _)| i)
      .collect();
    if members.is_empty() {
      continue;
    }
    ui.label(grade_mark_label(grade, n));
    ui.horizontal_wrapped(|ui| {
      for idx in members {
        let entry = &entries[idx];
        let selected = *selection == entry.selection;
        let label = entry.dof_label.unwrap_or(entry.name);
        if ui
          .add_sized(CELL, egui::Button::selectable(selected, label))
          .on_hover_text(entry.name)
          .clicked()
        {
          *selection = entry.selection;
        }
      }
    });
  }
}

/// Lays out one grade's eigenmodes as the orbital pyramid: one centered row per
/// degeneracy shell, rows ordered by ascending eigenvalue and labelled by it,
/// each cell a mode selector labelled by its centered within-shell offset (the
/// magnetic index $m in -l..=l$ on the sphere's $2l+1$-fold grade-0 multiplet).
/// Hovering a cell shows the mode's full name and eigenvalue.
fn pyramid(ui: &mut egui::Ui, shells: &[Shell], entries: &[Entry], selection: &mut Selection) {
  // A fixed cell size makes the columns line up into a grid; the matching
  // gutters left and right keep each row's cells centered under `vertical_centered`.
  const CELL: [f32; 2] = [30.0, 22.0];
  const GUTTER: f32 = 56.0;
  ui.vertical_centered(|ui| {
    for shell in shells {
      ui.horizontal(|ui| {
        ui.add_sized(
          [GUTTER, CELL[1]],
          egui::Label::new(format!("λ = {:.1}", shell.eigenvalue)),
        );
        let n = shell.members.len() as isize;
        for (pos, &idx) in shell.members.iter().enumerate() {
          let m = pos as isize - (n - 1) / 2;
          let label = if m == 0 {
            "0".to_string()
          } else {
            format!("{m:+}")
          };
          let entry = &entries[idx];
          let selected = *selection == entry.selection;
          if ui
            .add_sized(CELL, egui::Button::selectable(selected, label))
            .on_hover_text(entry.name)
            .clicked()
          {
            *selection = entry.selection;
          }
        }
        ui.add_space(GUTTER);
      });
    }
  });
}

/// The render mark a grade-$k$ field is drawn with on an $n$-manifold, named
/// for the UI: its reduced grade $min(k, n-k)$ decides between a scalar density
/// and a tangent line field (discussion #101). Whether the reduction went
/// through a Hodge star ($k$ above the fold) is noted so the top-grade section
/// reads as a density arrived at by $star$, not a bare 0-form.
fn grade_mark_label(grade: ExteriorGrade, n: Dim) -> String {
  let reduced = grade.min(n - grade);
  let mark = match reduced {
    0 => "density",
    1 => "line field",
    _ => "sheet",
  };
  if grade == reduced {
    format!("grade {grade} · {mark}")
  } else {
    format!("grade {grade} · {mark} (⋆)")
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

  // The lazy, memoized per-grade loader: which view is current, which is
  // building in the background, and the cache of already-solved scenes. The
  // full scene it produces stays around (not just the field on display), so the
  // UI can switch which field is shown without a rebuild.
  gallery: Gallery,
  scene: Scene,
  selection: Selection,
  // Fixed at scene-build time: the mesh's own edge-length scale, which sets the
  // LIC noise frequency (`noise_scale`).
  mesh_width: f32,
  // Fixed at scene-build time: the object's coordinate extent (its radius),
  // which sets the standing-wave displacement scale for whichever field is on
  // display -- an object-intrinsic length, so the lobes read at orbital scale
  // on any mesh.
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
    // The mesh width fixes the LIC noise frequency; the coordinate extent fixes
    // the standing-wave displacement (see the two `State` fields). Both are
    // object-intrinsic, so neither is tuned to the sphere.
    let mesh_width = scene
      .coords
      .to_edge_lengths(&scene.topology)
      .mesh_width_max() as f32;
    let amplitude_scale = scene_extent(&scene) as f32;
    let display = build_field_display(&device, &scene, selection, amplitude_scale);
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
      gallery,
      scene,
      selection,
      mesh_width,
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
    self.mesh_width = scene
      .coords
      .to_edge_lengths(&scene.topology)
      .mesh_width_max() as f32;
    self.noise_scale = NOISE_CYCLES_PER_MESH_WIDTH / self.mesh_width.max(f32::EPSILON);
    self.amplitude_scale = scene_extent(&scene) as f32;
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

  /// Builds and tessellates the control panel: the family picker, the mesh's
  /// per-grade tabs, that grade's orbital-pyramid mode picker (or a spinner
  /// while it solves), and the camera-mode toggle. Reads out of `self` into
  /// plain locals before entering the `egui::Context::run_ui` closure, since
  /// `ctx` (a clone of `self.egui_ctx`) is what the closure captures --
  /// borrowing `self` inside it would conflict with the `&mut self` calls
  /// (`set_view`/`set_field`) right after.
  fn run_ui(&mut self, window: &Window) -> (Vec<egui::ClippedPrimitive>, egui::TexturesDelta, f32) {
    let ctx = self.egui_ctx.clone();
    let raw_input = self.egui_winit_state.take_egui_input(window);

    // What the panel reflects is the view being *loaded*, if any, so clicking a
    // grade highlights it and shows the spinner at once rather than a frame
    // after its solve lands. The displayed `self.scene` still belongs to the
    // previous view meanwhile, but the mode picker is hidden behind the spinner
    // until the new one arrives, so its entries are only read when they match.
    let shown_view = self.gallery.loading_view().unwrap_or(self.gallery.current);
    let is_loading = self.gallery.is_loading();
    let loading_label = self.gallery.loading_view().map(View::label);
    let last_mesh_grade = self.gallery.last_mesh_grade;
    let max_grade = self.gallery.mesh.0.dim();
    let scene_dim = self.scene.topology.dim();

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

    let mesh_source = self.gallery.mesh_source.clone();
    let mesh_error = self.gallery.error.clone();
    let mut requested_view = shown_view;
    let mut requested_mesh_source = mesh_source.clone();
    let mut selection = self.selection;
    let mut top_down = self.camera.top_down;

    // Borrow only the file dialog into the closure (native): the rest of the
    // panel is driven by the plain locals above, so this disjoint field can be
    // touched inside the closure without conflicting with the `&mut self` calls
    // after it. Its own borrow ends with the closure, before those calls.
    #[cfg(not(target_arch = "wasm32"))]
    let file_dialog = &mut self.file_dialog;

    let full_output = ctx.run_ui(raw_input, |ctx| {
      egui::Window::new("Gallery").show(ctx, |ui| {
        let on_eigenmodes = matches!(shown_view, View::MeshGrade(_));
        ui.horizontal(|ui| {
          if ui.selectable_label(on_eigenmodes, "Eigenmodes").clicked() {
            requested_view = View::MeshGrade(last_mesh_grade);
          }
          if ui
            .selectable_label(shown_view == View::WhitneyBasis, "LSF (reference cell)")
            .clicked()
          {
            requested_view = View::WhitneyBasis;
          }
          if ui
            .selectable_label(shown_view == View::WhitneyBasisMesh, "GSF (triforce mesh)")
            .clicked()
          {
            requested_view = View::WhitneyBasisMesh;
          }
        });
        ui.separator();

        if let View::MeshGrade(grade) = shown_view {
          // Mesh source: the surface the eigenmodes live on is a chosen input,
          // not a fixed sphere -- a generated family, a built-in gallery mesh,
          // or a loaded OBJ. Picking any remeshes and re-solves the current
          // grade (`requested_mesh_source`, applied after the closure).
          ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("mesh-source")
              .selected_text(requested_mesh_source.label())
              .show_ui(ui, |ui| {
                // A generated family resets to its default refinement when
                // first chosen; re-picking the current family keeps the
                // slider's value.
                let is_sphere = matches!(requested_mesh_source, MeshSource::Sphere { .. });
                if ui.selectable_label(is_sphere, "Sphere").clicked() && !is_sphere {
                  requested_mesh_source = MeshSource::Sphere {
                    subdivisions: SPHERE_SUBDIVISIONS,
                  };
                }
                let is_grid = matches!(requested_mesh_source, MeshSource::Grid { .. });
                if ui.selectable_label(is_grid, "Grid").clicked() && !is_grid {
                  requested_mesh_source = MeshSource::Grid {
                    cells_axis: GRID_CELLS_DEFAULT,
                  };
                }
                for builtin in BuiltinMesh::ALL {
                  let selected = requested_mesh_source == MeshSource::Builtin(builtin);
                  if ui.selectable_label(selected, builtin.label()).clicked() {
                    requested_mesh_source = MeshSource::Builtin(builtin);
                  }
                }
              });
            // Opens the in-egui file browser; the pick is retrieved after the
            // closure (`file_dialog.take_picked`). Native-only: wasm has no
            // filesystem to browse.
            #[cfg(not(target_arch = "wasm32"))]
            if ui.button("Load OBJ…").clicked() {
              file_dialog.pick_file();
            }
          });
          // A generated family carries a refinement slider; a built-in or
          // loaded mesh has none.
          match &mut requested_mesh_source {
            MeshSource::Sphere { subdivisions } => {
              ui.add(
                egui::Slider::new(subdivisions, 0..=SPHERE_SUBDIVISIONS_MAX).text("subdivisions"),
              );
            }
            MeshSource::Grid { cells_axis } => {
              ui.add(egui::Slider::new(cells_axis, 1..=GRID_CELLS_MAX).text("cells/axis"));
            }
            MeshSource::Builtin(_) | MeshSource::Custom { .. } => {}
          }
          if let Some(error) = &mesh_error {
            ui.colored_label(egui::Color32::LIGHT_RED, format!("⚠ {error}"));
          }
          ui.separator();

          // One tab per grade of the de Rham complex; every grade is solved and
          // shown, the top grade through its Hodge star just like grade 0.
          ui.horizontal(|ui| {
            for g in 0..=max_grade {
              if ui
                .selectable_label(g == grade, grade_mark_label(g, max_grade))
                .clicked()
              {
                requested_view = View::MeshGrade(g);
              }
            }
          });
          ui.separator();
        }

        if is_loading {
          ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(20.0));
            if let Some(label) = &loading_label {
              ui.label(format!("Solving {label}…"));
            }
          });
        } else {
          match shown_view {
            View::MeshGrade(_) => ui.label("rows: degeneracy shell (λ) · cells: order"),
            View::WhitneyBasis | View::WhitneyBasisMesh => {
              ui.label("rows: grade · cells: DOF simplex")
            }
          };
          render_modes(ui, &entries, &mut selection, scene_dim);
        }

        ui.separator();
        ui.checkbox(&mut top_down, "Top-down (orthographic, drag to pan)");
      });

      // The file browser draws as its own window; it must be updated within the
      // frame, after the panel that opens it.
      #[cfg(not(target_arch = "wasm32"))]
      file_dialog.update(ctx);
    });

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
      if requested_mesh_source != mesh_source {
        self.set_mesh_source(requested_mesh_source);
      } else if requested_view != shown_view {
        self.set_view(requested_view);
      } else {
        self.set_field(selection);
        if top_down != self.camera.top_down {
          self.camera.top_down = top_down;
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

#[cfg(test)]
mod tests {
  use super::*;

  fn shell_sizes(eigenvalues: &[f64]) -> Vec<usize> {
    degeneracy_shells(eigenvalues.iter().map(|&l| Some(l)))
      .unwrap()
      .iter()
      .map(|s| s.members.len())
      .collect()
  }

  /// The measured subdivision-3 icosphere grade-0 spectrum clusters into the
  /// $(2l+1)$ spherical-harmonic shells: the near-equal multiplets group, the
  /// order-one jumps between degrees split.
  #[test]
  fn sphere_spectrum_recovers_2l_plus_1_shells() {
    let spectrum = [0.00, 2.01, 2.01, 2.01, 6.07, 6.07, 6.07, 6.07, 6.07, 12.24];
    assert_eq!(shell_sizes(&spectrum), vec![1, 3, 5, 1]);
  }

  /// A near-zero harmonic space (a flat torus's two 1-cocycles) stays one shell
  /// rather than splitting on numerical noise, since the absolute tolerance
  /// carries a scale the relative gap alone lacks near zero.
  #[test]
  fn near_zero_harmonics_stay_one_shell() {
    let spectrum = [-1e-9, 2e-9, 4.0, 4.0];
    assert_eq!(shell_sizes(&spectrum), vec![2, 2]);
  }

  /// A generic simple spectrum -- no symmetry, no degeneracy -- degenerates the
  /// pyramid to one member per row, ordered by eigenvalue.
  #[test]
  fn simple_spectrum_gives_singletons() {
    let spectrum = [1.0, 2.5, 4.0, 6.0, 9.0];
    assert_eq!(shell_sizes(&spectrum), vec![1, 1, 1, 1, 1]);
  }

  /// A field carrying no eigenvalue (the raw Whitney basis) has no shell
  /// structure, so the caller falls back to a flat list.
  #[test]
  fn missing_eigenvalue_declines_to_shell() {
    assert!(degeneracy_shells([Some(1.0), None, Some(2.0)]).is_none());
  }
}
