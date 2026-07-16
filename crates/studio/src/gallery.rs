//! The gallery model: which mesh and which study of it are shown, and the
//! lazy, memoized loader that builds a `(mesh, study)` pair's
//! [`crate::scene::Scene`] -- possibly on a background thread. Free of any GPU
//! type; the windowed wrapper (`app.rs`) owns the `Scene` this produces.
//!
//! The two axes are independent. What is shown is a point in the product
//! [`MeshSource`] × [`Study`]: every study runs on every mesh, and the cache,
//! the background load and the placeholder machinery all key on the pair. A
//! `Preset` is only a named point in that product together with the field it
//! opens on -- a value the browser fills the two axes from, never a build path
//! of its own.

use std::sync::Arc;

use ddf::cochain::Cochain;
use exterior::ExteriorGrade;
use manifold::{
  geometry::coord::mesh::{standard_coord_complex, MeshCoords},
  topology::{complex::Complex, simplex::Simplex},
  Dim,
};

use crate::scene::Scene;
use crate::ui::Selection;

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

// The reference cell the Whitney-basis study opens on, and the top of its
// dimension slider: the intrinsic dimensions the fixed ambient $RR^3$ embeds.
// A triangle (dim 2) matches the historical local-shape-function gallery.
pub const REFERENCE_CELL_DIM: Dim = 2;
pub(crate) const REFERENCE_CELL_DIM_MAX: Dim = 3;

// Hodge-Laplace modes an eigenmode study solves for by default. Chosen so both
// low grades close on a complete degeneracy shell on the sphere: grade 0 fills
// $l = 0..=3$ ($sum (2l+1) = 16$) and grade 1 fills $l = 1, 2$
// ($6 + 10 = 16$), so the orbital pyramid the UI lays these out in has no
// half-built final row.
pub const DEFAULT_NMODES: usize = 16;

/// The shared surface mesh a study solves against, built once so every
/// per-grade eigensolve reuses it rather than remeshing.
pub(crate) type Mesh = (Complex, MeshCoords);

/// One of the CC0 surface meshes the studio ships as a built-in gallery,
/// embedded in the binary (see `assets/meshes`, and its `SOURCES.md` for
/// provenance). Chosen to span topology -- genus 0 and genus 1 -- so the
/// harmonic (zero-eigenvalue) modes the gallery shows at grade 1 range over
/// $dim H^1 = 2 g$.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BuiltinMesh {
  /// Spot the cow (genus 0).
  Spot,
  /// Bob (genus 1).
  Bob,
  /// Blub the fish (genus 0); the largest, so the slowest to solve.
  Blub,
}

impl BuiltinMesh {
  pub const ALL: [BuiltinMesh; 3] = [BuiltinMesh::Spot, BuiltinMesh::Bob, BuiltinMesh::Blub];

  pub(crate) fn label(self) -> &'static str {
    match self {
      BuiltinMesh::Spot => "Spot (cow)",
      BuiltinMesh::Bob => "Bob",
      BuiltinMesh::Blub => "Blub (fish)",
    }
  }

  /// The mesh's name as a caller writes it: ASCII, lowercase, no spaces --
  /// what the picker's own label is not, since a label is prose and this is a
  /// token. Paired with [`Self::from_name`], so a mesh added to [`Self::ALL`]
  /// reaches the CLI without a second list to keep in step.
  pub fn name(self) -> &'static str {
    match self {
      BuiltinMesh::Spot => "spot",
      BuiltinMesh::Bob => "bob",
      BuiltinMesh::Blub => "blub",
    }
  }

  pub fn from_name(name: &str) -> Option<Self> {
    Self::ALL.into_iter().find(|m| m.name() == name)
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
    }
  }
}

/// The chosen source of the mesh a study runs on -- a runtime input, not a
/// fixed sphere. A generated family carries its refinement (moved by a slider);
/// a built-in, the reference cell, the triforce or a user-loaded file each
/// carry their own fixed geometry.
///
/// Nothing downstream distinguishes the variants: the eigensolve, the Whitney
/// reconstruction, the reduced-grade dispatch and the degeneracy clustering are
/// all mesh-agnostic, and the camera reads a curved surface (perspective orbit)
/// from a flat one (top-down orthographic) off the coordinates alone. The
/// sphere's spherical harmonics are then one mesh's spectrum among others, not
/// a privileged case.
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
  /// The standard reference cell of the given intrinsic dimension as a one-cell
  /// mesh. The Whitney-basis study on it is the local shape functions, so
  /// "local shape functions" is that composition, not a study of its own.
  /// `dim` ranges over $1..=3$, the intrinsic dimensions the fixed ambient
  /// $RR^3$ embeds.
  ReferenceCell { dim: Dim },
  /// The triforce teaching mesh ([`crate::demos::triforce`]): four cells around
  /// one interior vertex, flat in the $z = 0$ plane. The multi-cell counterpart
  /// of the reference cell -- the Whitney-basis study on it is the global shape
  /// functions.
  Triforce,
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
      MeshSource::ReferenceCell { .. } => "Reference cell".to_string(),
      MeshSource::Triforce => "Triforce".to_string(),
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
      MeshSource::ReferenceCell { dim } => {
        let (topology, coords) = standard_coord_complex(*dim);
        // A reference cell of `dim < 3` embeds as itself in the `z = 0` plane;
        // a no-op once `dim >= 3`.
        Ok((topology, coords.embed_euclidean((*dim).max(3))))
      }
      MeshSource::Triforce => Ok(crate::demos::triforce()),
      MeshSource::Builtin(builtin) => {
        crate::io::obj::parse(builtin.obj()).map_err(|e| format!("{}: {e}", builtin.label()))
      }
      MeshSource::Custom { .. } => {
        unreachable!("a custom mesh is installed directly, never rebuilt from its descriptor")
      }
    }
  }
}

/// What is computed on the mesh: the second axis of the platform. Parameters
/// live in the variant; a `Preset` fills them with concrete values, and the
/// inspector edits them. Every study builds on every [`MeshSource`], and the
/// build goes through the general [`Scene`] constructors -- there is no
/// per-study display path.
#[derive(Clone, PartialEq)]
pub enum Study {
  /// Hodge-Laplace eigenmodes of a single grade: the standing-wave normal
  /// modes $Delta u = lambda u$, solved once and memoized. The harmonic forms
  /// are its zero shell ($lambda = 0$), not a separate solver.
  Eigenmodes { grade: ExteriorGrade, nmodes: usize },
  /// One field per DOF simplex of every grade: the one-hot cochains, the
  /// Whitney basis functions of the mesh. A reference cell's local shape
  /// functions and a multi-cell mesh's global ones are the same construction,
  /// differing only in whether a DOF simplex's support is one cell or several.
  WhitneyBasis,
  /// A named list of explicit cochains -- the triforce worked examples today, a
  /// loaded cochain file later.
  Cochains(Vec<NamedCochain>),
  /// The grade-1 Hodge decomposition of a probe field, exposed as four
  /// switchable fields: the input and its exact, coexact and harmonic shells.
  /// The harmonic shell is nonzero exactly on a mesh with grade-1 homology, so
  /// this is the study that shows the topology of the surface directly.
  HodgeDecomposition,
}

impl Study {
  /// The study the viewer opens on: grade-0 eigenmodes.
  pub(crate) fn start() -> Study {
    Study::Eigenmodes {
      grade: 0,
      nmodes: DEFAULT_NMODES,
    }
  }

  pub(crate) fn label(&self) -> String {
    match self {
      Study::Eigenmodes { grade, .. } => format!("Eigenmodes, grade {grade}"),
      Study::WhitneyBasis => "Whitney basis".to_string(),
      Study::Cochains(_) => "Cochains".to_string(),
      Study::HodgeDecomposition => "Hodge decomposition".to_string(),
    }
  }

  /// Builds the study's scene on `mesh`. For an eigenmode grade this runs that
  /// grade's dense eigensolve -- the expensive path the caller runs on a
  /// background thread and memoizes; the Whitney basis and the explicit
  /// cochains are cheap.
  pub(crate) fn build(&self, mesh: &Mesh) -> Scene {
    let (topology, coords) = mesh;
    match self {
      Study::Eigenmodes { grade, nmodes } => Scene::mesh_grade(topology, coords, *grade, *nmodes),
      Study::WhitneyBasis => Scene::whitney_basis_mesh(topology.clone(), coords.clone()),
      Study::Cochains(specs) => Scene::cochains(topology.clone(), coords.clone(), specs),
      Study::HodgeDecomposition => Scene::hodge_decomposition(topology.clone(), coords.clone()),
    }
  }
}

/// A named entry of a [`Study::Cochains`] list: an explicit cochain kept as
/// data ([`CochainSpec`]) plus the name the picker shows it under.
#[derive(Clone, PartialEq)]
pub struct NamedCochain {
  pub name: String,
  pub spec: CochainSpec,
}

/// A concrete cochain kept as data rather than as a constructed vector, so a
/// preset builds one and the resolution against a mesh happens at build time.
#[derive(Clone, PartialEq)]
pub enum CochainSpec {
  /// A grade-1 cochain given as a coefficient per edge, each edge addressed by
  /// its ordered vertex pair $(v_0, v_1)$ with $v_0 < v_1$ -- the canonical
  /// (positively oriented) edge orientation -- so the value lands on the mesh's
  /// own edge regardless of that mesh's internal edge indexing.
  ByEdges(Vec<(usize, usize, f64)>),
}

impl CochainSpec {
  /// Resolves the spec to a cochain on `topology`.
  pub(crate) fn resolve(&self, topology: &Complex) -> Cochain {
    match self {
      CochainSpec::ByEdges(entries) => {
        let edges = topology.skeleton_raw(1);
        let mut coeffs = na::DVector::zeros(topology.nsimplices(1));
        for &(v0, v1, value) in entries {
          coeffs[edges.kidx_by_simplex(&Simplex::new(vec![v0, v1]))] = value;
        }
        Cochain::new(1, coeffs)
      }
    }
  }
}

/// A named point in the [`MeshSource`] × [`Study`] product, plus the field it
/// opens on. Selecting a preset sets the two axes and the selection;
/// everything afterward is the ordinary platform. A preset is a
/// *configuration*, never a code path of its own -- the moment a curated
/// example would need its own branch to build or display, it has stopped being
/// a preset.
pub(crate) struct Preset {
  pub(crate) name: &'static str,
  pub(crate) mesh: MeshSource,
  pub(crate) study: Study,
  /// The field the preset opens on, or `None` to open on the scene's first
  /// mode (the platform's own default).
  pub(crate) selection: Option<Selection>,
}

/// The curated first-wave presets, in browser order. Each is a configuration of
/// the general platform: a mesh, a study, and the field to open on.
pub(crate) fn presets() -> Vec<Preset> {
  vec![
    Preset {
      name: "Spherical harmonics",
      mesh: MeshSource::START,
      study: Study::start(),
      selection: None,
    },
    Preset {
      name: "Harmonic 1-forms",
      mesh: MeshSource::Builtin(BuiltinMesh::Bob),
      study: Study::Eigenmodes {
        grade: 1,
        nmodes: DEFAULT_NMODES,
      },
      selection: None,
    },
    Preset {
      name: "Whitney basis",
      mesh: MeshSource::ReferenceCell {
        dim: REFERENCE_CELL_DIM,
      },
      study: Study::WhitneyBasis,
      selection: None,
    },
    Preset {
      name: "Global shape functions",
      mesh: MeshSource::Triforce,
      study: Study::WhitneyBasis,
      selection: None,
    },
    Preset {
      name: "Constant / curl / div",
      mesh: MeshSource::Triforce,
      study: Study::Cochains(crate::demos::triforce_examples()),
      selection: Some(Selection::Line(0)),
    },
    Preset {
      // A genus-1 surface, so the harmonic shell is genuinely 2-dimensional
      // ($b_1 = 2$) rather than empty: the decomposition is the full
      // exact + coexact + harmonic, not the contractible Helmholtz special
      // case. Opens on the harmonic shell -- the component that sees the hole.
      name: "Hodge decomposition",
      mesh: MeshSource::Builtin(BuiltinMesh::Bob),
      study: Study::HodgeDecomposition,
      selection: Some(Selection::Line(3)),
    },
  ]
}

/// What the viewer is showing: the `(mesh, study)` pair a build and its
/// memoization key on. The unit the gallery caches and loads.
#[derive(Clone, PartialEq)]
pub(crate) struct Shown {
  pub(crate) mesh: MeshSource,
  pub(crate) study: Study,
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

/// The gallery's lazy, memoized loader. Owns the current mesh; each
/// `(mesh, study)` pair's scene is built at most once -- the expensive
/// eigensolves on a background thread -- and cached, so revisiting a pair is
/// instant and only the pairs actually viewed are ever solved.
///
/// Kept free of any GPU type; the windowed wrapper owns the `Scene` it
/// produces. Only one build is ever in flight: requesting a pair while another
/// is loading replaces the pending build, and a landed result is only ever
/// installed for the pair it was built for.
pub(crate) struct Gallery {
  /// The live mesh axis, and the mesh built from it. Updated the moment a mesh
  /// change is requested, so the panel reflects the choice at once.
  pub(crate) mesh_source: MeshSource,
  pub(crate) mesh: Arc<Mesh>,
  /// The live study axis.
  pub(crate) study: Study,
  /// The last eigenmode grade viewed, so toggling to the Whitney basis and back
  /// resumes that grade rather than resetting to 0.
  pub(crate) last_grade: ExteriorGrade,
  cache: Vec<(Shown, Scene)>,
  loading: Option<(Shown, PendingLoad<Scene>)>,
  /// The last mesh-source failure (a malformed OBJ, an unfetched asset),
  /// surfaced in the panel until the next successful mesh change. `None` when
  /// the current mesh loaded cleanly.
  pub(crate) error: Option<String>,
}

impl Gallery {
  /// Opens on the starting study of `source`, returning the cheap placeholder
  /// scene to show at once while that pair's real build runs in the background
  /// (delivered later by [`Self::poll`]). The placeholder shares the loader's
  /// mesh, so the window is up on the first frame without waiting on any
  /// eigensolve.
  pub(crate) fn new(source: MeshSource) -> (Self, Scene) {
    // The starting source is the icosphere, whose build is infallible.
    let mesh = Arc::new(source.build().expect("the starting mesh builds"));
    let placeholder = Scene::placeholder_on(mesh.0.clone(), mesh.1.clone());
    let mut gallery = Self {
      mesh_source: source,
      mesh,
      study: Study::start(),
      last_grade: 0,
      cache: Vec::new(),
      loading: None,
      error: None,
    };
    let shown = gallery.shown();
    gallery.spawn(shown);
    (gallery, placeholder)
  }

  /// The pair currently on the two live axes.
  pub(crate) fn shown(&self) -> Shown {
    Shown {
      mesh: self.mesh_source.clone(),
      study: self.study.clone(),
    }
  }

  fn cached(&self, shown: &Shown) -> Option<Scene> {
    self
      .cache
      .iter()
      .find(|(s, _)| s == shown)
      .map(|(_, scene)| scene.clone())
  }

  fn spawn(&mut self, shown: Shown) {
    let mesh = self.mesh.clone();
    let study = shown.study.clone();
    self.loading = Some((shown, PendingLoad::spawn(move || study.build(&mesh))));
  }

  /// Requests a build of the current pair against `self.mesh` (which the caller
  /// has already made match `self.mesh_source`). `Some(scene)` is a cache hit
  /// to install now; `None` means a background build is in flight (a spinner
  /// shows until [`Self::poll`] delivers it).
  fn request(&mut self) -> Option<Scene> {
    let shown = self.shown();
    if let Some(scene) = self.cached(&shown) {
      self.loading = None;
      return Some(scene);
    }
    if !self.loading.as_ref().is_some_and(|(s, _)| *s == shown) {
      self.spawn(shown);
    }
    None
  }

  /// Switches the study on the current mesh. `Some(scene)` is a cache hit to
  /// install now; `None` is a no-op (the study is already shown) or a build now
  /// running in the background.
  pub(crate) fn select_study(&mut self, study: Study) -> Option<Scene> {
    if study == self.study && self.loading.is_none() {
      return None;
    }
    if let Study::Eigenmodes { grade, .. } = &study {
      self.last_grade = *grade;
    }
    self.study = study;
    self.request()
  }

  /// Switches to a regenerable mesh source, building its mesh first so a
  /// failure leaves the current one untouched. `Ok(Some(scene))` installs now
  /// -- a cache hit, or a solve-free placeholder on the new mesh to show while
  /// its study re-solves; `Ok(None)` is a no-op source change; `Err` is a build
  /// failure to surface.
  pub(crate) fn select_mesh(&mut self, source: MeshSource) -> Result<Option<Scene>, String> {
    if source == self.mesh_source && self.loading.is_none() {
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

  /// Applies a preset: builds its mesh, sets the two axes, and requests the
  /// build. Returns a scene to install now (a cache hit, or a placeholder on
  /// the new mesh) or `None` while the pair solves. The preset's opening
  /// selection is the caller's to apply once the scene lands.
  pub(crate) fn select_preset(&mut self, preset: &Preset) -> Result<Option<Scene>, String> {
    let mesh = preset.mesh.build()?;
    if let Study::Eigenmodes { grade, .. } = &preset.study {
      self.last_grade = *grade;
    }
    self.study = preset.study.clone();
    Ok(self.install_mesh(preset.mesh.clone(), mesh))
  }

  /// Adopts `mesh` under `source` and requests the current study against it. A
  /// cache hit (this pair was built before) installs at once; otherwise the
  /// build is spawned and a solve-free placeholder on the new mesh is returned
  /// to show while it runs.
  fn install_mesh(&mut self, source: MeshSource, mesh: Mesh) -> Option<Scene> {
    self.error = None;
    self.mesh_source = source;
    self.mesh = Arc::new(mesh);
    let shown = self.shown();
    if let Some(scene) = self.cached(&shown) {
      self.loading = None;
      return Some(scene);
    }
    self.spawn(shown);
    Some(Scene::placeholder_on(
      self.mesh.0.clone(),
      self.mesh.1.clone(),
    ))
  }

  pub(crate) fn set_error(&mut self, error: String) {
    self.error = Some(error);
  }

  pub(crate) fn is_loading(&self) -> bool {
    self.loading.is_some()
  }

  /// The study label of the build in flight, for the spinner.
  pub(crate) fn loading_label(&self) -> Option<String> {
    self.loading.as_ref().map(|(shown, _)| shown.study.label())
  }

  /// `Some` exactly once, the frame a background build lands: memoizes it and
  /// hands it back to be installed.
  pub(crate) fn poll(&mut self) -> Option<Scene> {
    let (shown, scene) = self
      .loading
      .as_ref()
      .and_then(|(shown, pending)| pending.poll().map(|scene| (shown.clone(), scene)))?;
    self.loading = None;
    self.cache.push((shown, scene.clone()));
    Some(scene)
  }
}
