//! The gallery model: which mesh and which view of it are shown, and the
//! lazy, memoized loader that builds a view's [`crate::scene::Scene`] --
//! possibly on a background thread. Free of any GPU type; the windowed
//! wrapper (`app.rs`) owns the `Scene` this produces.

use std::collections::HashMap;
use std::sync::Arc;

use exterior::ExteriorGrade;
use manifold::{geometry::coord::mesh::MeshCoords, topology::complex::Complex};

use crate::scene::Scene;

// Icosphere subdivision depth the gallery opens on. The Laplace-Beltrami
// eigensolve is dense in the vertex count, so keep this modest for an instant
// startup; the mesh slider goes up to `SPHERE_SUBDIVISIONS_MAX` for fidelity.
pub(crate) const SPHERE_SUBDIVISIONS: usize = 3;
// Upper end of the sphere refinement slider. The per-grade solve is dense
// ($O(n^3)$), and at grade 1 the edge count is what enters, so a step past this
// turns the background solve from seconds into minutes -- the cap keeps every
// reachable mesh solvable while the window stays responsive.
pub(crate) const SPHERE_SUBDIVISIONS_MAX: usize = 4;
// Cells per axis the unit-square grid opens on, and the upper end of its
// refinement slider. A grid this fine keeps the dense per-grade solve well
// inside the sphere's cost, so the same cap reasoning applies loosely.
pub(crate) const GRID_CELLS_DEFAULT: usize = 8;
pub(crate) const GRID_CELLS_MAX: usize = 20;

/// The shared surface mesh the gallery solves its grades against, built once
/// so every per-grade eigensolve reuses it rather than remeshing.
pub(crate) type Mesh = (Complex, MeshCoords);

/// One of the CC0 surface meshes the studio ships as a built-in gallery,
/// embedded in the binary (see `assets/meshes`, and its `SOURCES.md` for
/// provenance). Chosen to span topology -- genus 0 and genus 1, down to a
/// 7-vertex torus -- so the harmonic (zero-eigenvalue) modes the gallery shows
/// at grade 1 range over $dim H^1 = 2 g$.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BuiltinMesh {
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
  pub(crate) const ALL: [BuiltinMesh; 4] = [
    BuiltinMesh::Spot,
    BuiltinMesh::Bob,
    BuiltinMesh::Blub,
    BuiltinMesh::Csaszar,
  ];

  pub(crate) fn label(self) -> &'static str {
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
/// `build` cannot serve.
#[derive(Clone, PartialEq)]
pub enum MeshSource {
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
  pub const START: MeshSource = MeshSource::Sphere {
    subdivisions: SPHERE_SUBDIVISIONS,
  };

  pub(crate) fn label(&self) -> String {
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
  pub(crate) fn build(&self) -> Result<Mesh, String> {
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
pub enum View {
  /// A single grade of the mesh's Hodge-Laplace eigenmodes. The expensive
  /// case: each grade is an eigensolve, run once and memoized, so only the
  /// grade actually being viewed is ever computed.
  MeshGrade(ExteriorGrade),
  /// Every Whitney basis function ("local shape function") of the reference
  /// triangle. Cheap to build.
  WhitneyBasis,
  /// Every Whitney basis function ("global shape function") of the triforce
  /// teaching mesh ([`crate::demos::triforce`]): the same construction as
  /// [`View::WhitneyBasis`], but a DOF simplex's support now spans the
  /// several cells incident to it instead of a single reference cell. Cheap
  /// to build, and mesh-independent like the reference-cell basis.
  WhitneyBasisMesh,
  /// The constant/pure-curl/pure-div worked examples on the triforce mesh
  /// ([`Scene::whitney_examples`]): three grade-1 fields, each an explicit
  /// linear combination of GSFs rather than a single one-hot cochain. Shares
  /// the triforce mesh with [`View::WhitneyBasisMesh`] but is its own view,
  /// since a flat 3-entry list has no grade grouping to gain from being
  /// merged into that gallery.
  WhitneyExamplesMesh,
}

impl View {
  /// The starting view when the viewer opens: grade 0 of the mesh.
  pub const START: View = View::MeshGrade(0);

  pub(crate) fn label(self) -> String {
    match self {
      View::MeshGrade(grade) => format!("Eigenmodes, grade {grade}"),
      View::WhitneyBasis => "Whitney basis (reference triangle)".to_string(),
      View::WhitneyBasisMesh => "Whitney basis (triforce mesh)".to_string(),
      View::WhitneyExamplesMesh => "Whitney examples (triforce mesh)".to_string(),
    }
  }
}

/// A rebuild of `T` running off the render thread, so a solve that triggers an
/// eigensolve never blocks the UI. `poll` is non-blocking and yields the
/// result exactly once, the frame it arrives.
struct PendingLoad<T> {
  rx: std::sync::mpsc::Receiver<T>,
}

impl<T: Send + 'static> PendingLoad<T> {
  fn spawn(build: impl FnOnce() -> T + Send + 'static) -> Self {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
      let _ = tx.send(build());
    });
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
pub(crate) struct Gallery {
  /// Which mesh the current `mesh` was built from, so the UI can reflect the
  /// live choice and a re-selection of the same source is a no-op.
  pub(crate) mesh_source: MeshSource,
  pub(crate) mesh: Arc<Mesh>,
  pub(crate) current: View,
  /// The last mesh grade viewed, so toggling to the Whitney basis and back
  /// resumes that grade rather than resetting to 0.
  pub(crate) last_mesh_grade: ExteriorGrade,
  cache: HashMap<View, Scene>,
  loading: Option<(View, PendingLoad<Scene>)>,
  /// The last mesh-source failure (a malformed OBJ, an unfetched asset),
  /// surfaced in the panel until the next successful mesh change. `None` when
  /// the current mesh loaded cleanly.
  pub(crate) error: Option<String>,
}

impl Gallery {
  /// Opens on [`View::START`], returning the cheap placeholder scene to show at
  /// once while that view's real build runs in the background (delivered later
  /// by [`Self::poll`]). The placeholder shares the loader's mesh, so the
  /// window is up on the first frame without waiting on any eigensolve.
  pub(crate) fn new(source: MeshSource) -> (Self, Scene) {
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
  pub(crate) fn select_source(&mut self, source: MeshSource) -> Result<Option<Scene>, String> {
    if source == self.mesh_source {
      return Ok(None);
    }
    let mesh = source.build()?;
    Ok(self.install_mesh(source, mesh))
  }

  /// Installs a mesh loaded from an OBJ file as the [`MeshSource::Custom`]
  /// source, named for the picker. Unlike a regenerable source there is nothing
  /// to build -- the mesh is already in hand -- so this always takes effect.
  pub(crate) fn load_custom(&mut self, name: String, mesh: Mesh) -> Option<Scene> {
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
      View::WhitneyBasis | View::WhitneyBasisMesh | View::WhitneyExamplesMesh => None,
    }
  }

  pub(crate) fn set_error(&mut self, error: String) {
    self.error = Some(error);
  }

  fn spawn(&mut self, view: View) {
    let mesh = self.mesh.clone();
    self.loading = Some((
      view,
      PendingLoad::spawn(move || crate::demos::build_view(view, &mesh)),
    ));
  }

  pub(crate) fn is_loading(&self) -> bool {
    self.loading.is_some()
  }

  pub(crate) fn loading_view(&self) -> Option<View> {
    self.loading.as_ref().map(|(view, _)| *view)
  }

  /// Requests that `view` be shown. Returns `Some(scene)` to install right now
  /// -- a cache hit, instant -- or `None` when the view is either already shown
  /// or now building in the background (a spinner shows until [`Self::poll`]
  /// delivers it).
  pub(crate) fn request(&mut self, view: View) -> Option<Scene> {
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
  pub(crate) fn poll(&mut self) -> Option<Scene> {
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
