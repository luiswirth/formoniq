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

use std::path::PathBuf;
use std::sync::Arc;

use derham::cochain::Cochain;
use exterior::ExteriorGrade;
use simplicial::{
  geometry::coord::mesh::{standard_coord_complex, MeshCoords},
  topology::{complex::Complex, simplex::Simplex},
  Dim,
};

use crate::scene::Scene;
use crate::ui::{Marks, Selection};

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

// A time-dependent study samples its solution at this many steps over the
// solve's final time. Enough that the linear interpolation between frames reads
// as continuous motion at the trajectory's playback rate.
pub const DEFAULT_TRAJECTORY_STEPS: usize = 160;
// The heat flow's final time: long enough for the initial bump to diffuse and
// visibly decay on the unit-scale gallery meshes. The wave equation's, in the
// same units: several periods of the lowest modes, so the fronts propagate and
// reflect rather than barely stirring.
pub const HEAT_FINAL_TIME: f64 = 0.5;
pub const WAVE_FINAL_TIME: f64 = 12.0;

/// The shared surface mesh a study solves against, built once so every
/// per-grade eigensolve reuses it rather than remeshing.
pub(crate) type Mesh = (Complex, MeshCoords);

/// One of the surface meshes the studio ships, embedded in the binary (see
/// `assets/meshes`, and its `SOURCES.md` for provenance and topology).
///
/// The set is *not* written down here: `build.rs` enumerates the asset
/// directory and generates the table below, so a mesh dropped into
/// `assets/meshes` is selectable with no code to change, and there is no list
/// that can fall out of step with what actually ships. A handle is an index
/// into that table -- the meshes are fixed for the life of the binary, so an
/// index is a name, and comparing two is comparing which mesh they are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BuiltinMesh(usize);

/// How a shipped asset is read. The extension decides, in `build.rs`: the
/// gallery does not care which format a mesh arrived in, only that it becomes
/// a complex and its coordinates.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Format {
  Obj,
  Gmsh,
}

pub(crate) struct Builtin {
  name: &'static str,
  format: Format,
  /// The asset's bytes. If the git-LFS content was never fetched these are the
  /// LFS pointer text, which the readers report as an empty or malformed mesh
  /// rather than silently mis-loading.
  bytes: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/builtin_meshes.rs"));

impl BuiltinMesh {
  /// Every shipped mesh, in the table's own (name-sorted) order.
  pub fn all() -> impl ExactSizeIterator<Item = BuiltinMesh> {
    (0..BUILTINS.len()).map(BuiltinMesh)
  }

  fn entry(self) -> &'static Builtin {
    &BUILTINS[self.0]
  }

  /// The mesh's name as a caller writes it: the asset's file stem, which is
  /// ASCII, lowercase and space-free by the same convention that names the
  /// files. Paired with [`Self::from_name`], so a mesh reaches the CLI and the
  /// picker together, from the one table.
  pub fn name(self) -> &'static str {
    self.entry().name
  }

  /// The picker's label: the name, capitalized. Prose is not derivable from a
  /// filename, so the filename *is* the label -- renaming the asset is how a
  /// mesh is renamed, rather than a second table of display strings that the
  /// first one could disagree with.
  pub(crate) fn label(self) -> String {
    let name = self.name();
    let mut chars = name.chars();
    match chars.next() {
      Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
      None => String::new(),
    }
  }

  pub fn from_name(name: &str) -> Option<Self> {
    Self::all().find(|m| m.name() == name)
  }

  /// The mesh, parsed by whichever reader its format calls for.
  pub(crate) fn build(self) -> Result<(Complex, MeshCoords), String> {
    let entry = self.entry();
    match entry.format {
      Format::Obj => {
        let text = std::str::from_utf8(entry.bytes)
          .map_err(|e| format!("{}: not UTF-8 ({e})", self.label()))?;
        crate::io::obj::parse(text).map_err(|e| format!("{}: {e}", self.label()))
      }
      Format::Gmsh => Ok(simplicial::io::gmsh::gmsh2coord_complex(entry.bytes)),
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
  /// A mesh saved by the engine's own serialization: a directory holding
  /// `topology.cbor` ([`Complex::save`]) and `coords.cbor`
  /// ([`MeshCoords::save`]). The general escape hatch for anything the
  /// gallery does not generate itself -- a solve's output mesh, a mesh
  /// exported for a paper figure -- regenerable from the path alone, unlike
  /// [`Self::Custom`].
  File(PathBuf),
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
      MeshSource::Builtin(builtin) => builtin.label(),
      MeshSource::Custom { name } => name.clone(),
      MeshSource::File(path) => path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned()),
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
        Ok(simplicial::gen::sphere::mesh_sphere_surface(*subdivisions))
      }
      MeshSource::Grid { cells_axis } => {
        let (topology, coords) =
          simplicial::gen::cartesian::CartesianGrid::new_unit(2, *cells_axis).triangulate();
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
      MeshSource::Builtin(builtin) => builtin.build(),
      MeshSource::Custom { .. } => {
        unreachable!("a custom mesh is installed directly, never rebuilt from its descriptor")
      }
      MeshSource::File(path) => {
        let topology = Complex::load(path.join("topology.cbor"))
          .map_err(|e| format!("{}: {e}", path.display()))?;
        let coords = MeshCoords::load(path.join("coords.cbor"))
          .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok((topology, coords))
      }
    }
  }
}

/// What is computed on the mesh: the second axis of the platform. Parameters
/// live in the variant; a `Preset` fills them with concrete values, and the
/// inspector edits them. Every study builds on every [`MeshSource`], and the
/// build goes through the general [`Scene`] constructors -- there is no
/// per-study display path.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
  /// The heat flow $diff_t u = -Delta u$ of a localized bump, as a sampled
  /// trajectory: the parabolic smoothing of the Hodge-Laplacian shown directly
  /// rather than through its spectrum. `nsteps` samples over `final_time`.
  Heat { nsteps: usize, final_time: f64 },
  /// The wave equation $diff_(t t) u = -Delta u$ of a localized bump at rest, as
  /// a sampled trajectory: the hyperbolic counterpart of [`Self::Heat`], fronts
  /// propagating and reflecting off any boundary.
  Wave { nsteps: usize, final_time: f64 },
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
      Study::Heat { .. } => "Heat equation".to_string(),
      Study::Wave { .. } => "Wave equation".to_string(),
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
      Study::Heat { nsteps, final_time } => {
        Scene::heat(topology.clone(), coords.clone(), *nsteps, *final_time)
      }
      Study::Wave { nsteps, final_time } => {
        Scene::wave(topology.clone(), coords.clone(), *nsteps, *final_time)
      }
    }
  }
}

/// A named entry of a [`Study::Cochains`] list: an explicit cochain kept as
/// data ([`CochainSpec`]) plus the name the picker shows it under.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NamedCochain {
  pub name: String,
  pub spec: CochainSpec,
}

/// A concrete cochain kept as data rather than as a constructed vector, so a
/// preset builds one and the resolution against a mesh happens at build time.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CochainSpec {
  /// A grade-1 cochain given as a coefficient per edge, each edge addressed by
  /// its ordered vertex pair $(v_0, v_1)$ with $v_0 < v_1$ -- the canonical
  /// (positively oriented) edge orientation -- so the value lands on the mesh's
  /// own edge regardless of that mesh's internal edge indexing.
  ByEdges(Vec<(usize, usize, f64)>),
  /// A cochain saved by [`Cochain::save`], loaded by path -- the general
  /// counterpart to [`MeshSource::File`], for a cochain the gallery did not
  /// itself solve.
  File(PathBuf),
}

impl CochainSpec {
  /// Resolves the spec to a cochain on `topology`. Panics if a
  /// [`Self::File`] cochain fails to load or is incompatible with
  /// `topology` -- a build-time failure the caller surfaces the same way as
  /// any other malformed input, not a silent placeholder.
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
      CochainSpec::File(path) => {
        let cochain = Cochain::load(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(
          cochain.is_compatible_with(topology),
          "{}: cochain of grade {} does not match the mesh",
          path.display(),
          cochain.grade()
        );
        cochain
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
  /// The marks the preset opens with, or `None` for the platform default.
  /// Still configuration, not a code path: a preset may say which readings of
  /// its field it is worth first seeing together (a line field whose glyphs
  /// and flow complement each other), and everything after that first frame is
  /// the ordinary toggle.
  pub(crate) marks: Option<Marks>,
}

/// The preset the viewer opens on, and one of the curated set rather than a
/// startup configuration beside it: "where the viewer starts" is a choice of
/// point in the platform's product, which is exactly what a [`Preset`] is. Kept
/// as one constructor used from both places so the opening view and the entry
/// in the browser cannot drift apart.
///
/// The pure curl on the triforce, with both of a line field's marks: the glyphs
/// state what the field is at a point and the particles what it does over time,
/// and a rotational field is where seeing the two at once says most.
pub(crate) fn start_preset() -> Preset {
  curl_on_triforce()
}

fn curl_on_triforce() -> Preset {
  Preset {
    name: "Constant / curl / div",
    mesh: MeshSource::Triforce,
    study: Study::Cochains(crate::demos::triforce_examples()),
    // The second of the three (constant, curl, div), all grade 1 and so all
    // line fields, in the order `triforce_examples` builds them.
    selection: Some(Selection::Line(1)),
    marks: Some(Marks {
      glyphs: true,
      particles: true,
    }),
  }
}

/// The curated first-wave presets, in browser order. Each is a configuration of
/// the general platform: a mesh, a study, and the field to open on.
pub(crate) fn presets() -> Vec<Preset> {
  // A preset naming a shipped mesh looks it up by name, and is offered only if
  // that asset is present: the mesh set is the asset directory's, so a preset
  // over a mesh nobody shipped is not a configuration that exists.
  let bob = BuiltinMesh::from_name("bob");
  let mut presets = vec![
    Preset {
      name: "Spherical harmonics",
      mesh: MeshSource::START,
      study: Study::start(),
      selection: None,
      marks: None,
    },
    Preset {
      name: "Whitney basis",
      mesh: MeshSource::ReferenceCell {
        dim: REFERENCE_CELL_DIM,
      },
      study: Study::WhitneyBasis,
      selection: None,
      marks: None,
    },
    Preset {
      name: "Global shape functions",
      mesh: MeshSource::Triforce,
      study: Study::WhitneyBasis,
      selection: None,
      marks: None,
    },
    curl_on_triforce(),
    Preset {
      name: "Heat equation",
      mesh: MeshSource::START,
      study: Study::Heat {
        nsteps: DEFAULT_TRAJECTORY_STEPS,
        final_time: HEAT_FINAL_TIME,
      },
      selection: None,
      marks: None,
    },
    Preset {
      name: "Wave equation",
      mesh: MeshSource::START,
      study: Study::Wave {
        nsteps: DEFAULT_TRAJECTORY_STEPS,
        final_time: WAVE_FINAL_TIME,
      },
      selection: None,
      marks: None,
    },
  ];

  if let Some(bob) = bob {
    presets.insert(
      1,
      Preset {
        name: "Harmonic 1-forms",
        mesh: MeshSource::Builtin(bob),
        study: Study::Eigenmodes {
          grade: 1,
          nmodes: DEFAULT_NMODES,
        },
        selection: None,
        marks: None,
      },
    );
    presets.push(Preset {
      // A genus-1 surface, so the harmonic shell is genuinely 2-dimensional
      // ($b_1 = 2$) rather than empty: the decomposition is the full
      // exact + coexact + harmonic, not the contractible Helmholtz special
      // case. Opens on the harmonic shell -- the component that sees the hole.
      name: "Hodge decomposition",
      mesh: MeshSource::Builtin(bob),
      study: Study::HodgeDecomposition,
      selection: Some(Selection::Line(3)),
      marks: None,
    });
  }
  presets
}

/// What the viewer is showing: the `(mesh, study)` pair a build and its
/// memoization key on. The unit the gallery caches and loads.
#[derive(Clone, PartialEq)]
pub(crate) struct Shown {
  pub(crate) mesh: MeshSource,
  pub(crate) study: Study,
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
  loading: Option<(Shown, crate::solve::Pending)>,
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
  /// Opens on `preset`, so the starting point is the same configuration the
  /// browser offers rather than a second hardcoded pair beside it.
  pub(crate) fn new(preset: &Preset) -> (Self, Scene) {
    let mesh = Arc::new(preset.mesh.build().expect("the starting mesh builds"));
    let placeholder = Scene::placeholder_on(mesh.0.clone(), mesh.1.clone());
    let last_grade = match &preset.study {
      Study::Eigenmodes { grade, .. } => *grade,
      _ => 0,
    };
    let mut gallery = Self {
      mesh_source: preset.mesh.clone(),
      mesh,
      study: preset.study.clone(),
      last_grade,
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
    let request = crate::solve::SolveRequest::new(&self.mesh, shown.study.clone());
    self.loading = Some((shown, crate::solve::Pending::spawn(request)));
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
  ///
  /// Native only: the web build has no OBJ picker (see `app.rs`).
  #[cfg(not(target_arch = "wasm32"))]
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
    // The outcome carries only the fields; the mesh it was solved on is the
    // one still held here, so the scene is reassembled rather than returned.
    let (shown, scene) = self.loading.as_ref().and_then(|(shown, pending)| {
      pending
        .poll()
        .map(|outcome| (shown.clone(), outcome.into_scene(&self.mesh)))
    })?;
    self.loading = None;
    self.cache.push((shown, scene.clone()));
    Some(scene)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The opening preset resolves to the field it means. Its selection is an
  /// *index* into the scene's line fields, so it is exactly the kind of thing
  /// that goes quietly wrong when the study's cochains are reordered -- this
  /// pins it to the field's name instead, which is what the preset is really
  /// choosing.
  #[test]
  fn the_start_preset_opens_on_the_curl_field() {
    let preset = start_preset();
    let mesh = preset.mesh.build().expect("the starting mesh builds");
    let scene = preset.study.build(&mesh);
    let Some(Selection::Line(index)) = preset.selection else {
      panic!("the start preset opens on a line field");
    };
    assert_eq!(scene.line_fields[index].name, "pure curl");
    // Both marks, which is the whole reason this preset opens the viewer.
    let marks = preset.marks.expect("the start preset sets its marks");
    assert!(marks.glyphs && marks.particles);
  }

  /// Every shipped asset loads. This is what the generated table buys and what
  /// it risks: the picker now offers whatever is in `assets/meshes`, so a file
  /// dropped in with an unreadable body, or an extension whose reader cannot
  /// actually parse it, becomes a broken entry in the UI rather than a compile
  /// error. Building each one here is the check that the directory and the
  /// readers agree.
  ///
  /// A closed surface is not asserted -- a shipped mesh need not be closed --
  /// but a mesh with no cells is either an unfetched LFS pointer or a file that
  /// is not a mesh at all, and neither belongs in the picker.
  #[test]
  fn every_shipped_mesh_loads() {
    assert!(
      BuiltinMesh::all().len() > 0,
      "the asset directory must ship at least one mesh"
    );
    for builtin in BuiltinMesh::all() {
      let (topology, coords) = builtin
        .build()
        .unwrap_or_else(|e| panic!("{}: {e}", builtin.name()));
      assert!(
        topology.nsimplices(topology.dim()) > 0,
        "{}: built empty (unfetched LFS asset?)",
        builtin.name()
      );
      assert_eq!(
        coords.nvertices(),
        topology.nsimplices(0),
        "{}: coordinates and vertices disagree",
        builtin.name()
      );
    }
  }

  /// The names the CLI accepts are the picker's, and they are unique -- the
  /// file stems, so two assets differing only by extension would collide and
  /// `from_name` would silently resolve to whichever sorted first.
  #[test]
  fn shipped_mesh_names_are_unique_and_resolve() {
    let names: Vec<&str> = BuiltinMesh::all().map(|m| m.name()).collect();
    let unique: std::collections::HashSet<&&str> = names.iter().collect();
    assert_eq!(
      unique.len(),
      names.len(),
      "duplicate mesh names in {names:?}"
    );
    for builtin in BuiltinMesh::all() {
      assert_eq!(BuiltinMesh::from_name(builtin.name()), Some(builtin));
    }
  }

  /// The other preset that names a field explicitly. Its `Line(3)` is an index
  /// into the scene its study builds, so it breaks silently when the shells are
  /// reordered -- pinned here to the shell's name instead.
  ///
  /// Checked on `torus0.msh` rather than on its own mesh (Bob): what the index
  /// depends on is the shell ordering of
  /// `Study::HodgeDecomposition` and the surface's first Betti number, and the
  /// torus fixture has the same $b_1 = 2$ at 127 vertices instead of ~3000. It
  /// is a faithful stand-in for what is being tested, not a weaker one -- and
  /// it keeps Bob's harmonic solve out of the test suite, for the reason
  /// `assets/meshes/SOURCES.md` already records.
  #[test]
  fn the_hodge_preset_opens_on_the_harmonic_shell() {
    let hodge = presets()
      .into_iter()
      .find(|p| matches!(p.study, Study::HodgeDecomposition))
      .expect("a Hodge decomposition preset");
    let mesh =
      simplicial::io::gmsh::gmsh2coord_complex(include_bytes!("../assets/meshes/torus0.msh"));
    assert!(
      mesh.0.nsimplices(1) > 0,
      "torus fixture built empty (unfetched LFS asset?)"
    );
    let scene = hodge.study.build(&mesh);
    let Some(Selection::Line(index)) = hodge.selection else {
      panic!("the Hodge preset opens on a line field");
    };
    assert!(
      index < scene.line_fields.len(),
      "Hodge preset opens on line {index} of {}",
      scene.line_fields.len()
    );
    assert!(
      scene.line_fields[index].name.contains("harmonic"),
      "the Hodge preset opens on the harmonic shell, got {:?}",
      scene.line_fields[index].name
    );
  }
}
