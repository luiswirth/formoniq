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
  Dim,
  geometry::coord::mesh::{MeshCoords, standard_coord_complex},
  topology::{complex::Complex, simplex::Simplex},
};

use simplicial::mesher::quotient::Identification;

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
// Cells per axis the unit-cube grid opens on, and the upper end of its
// refinement slider. A grid this fine keeps the dense per-grade solve well
// inside the sphere's cost, so the same cap reasoning applies loosely.
pub(crate) const GRID_CELLS_DEFAULT: usize = 8;
pub(crate) const GRID_CELLS_MAX: usize = 20;

// Cells on the *longest* axis a quotient surface opens on, and the ends of its
// refinement slider. The shorter axes are scaled down from it, so the cells stay
// near equilateral rather than inheriting the fundamental domain's aspect ratio.
// Three is the floor the generator imposes on a closed axis.
pub const QUOTIENT_CELLS_DEFAULT: usize = 12;
pub(crate) const QUOTIENT_CELLS_MIN: usize = 3;
pub(crate) const QUOTIENT_CELLS_MAX: usize = 48;

// The donut's tube as a fraction of its revolution radius. The binding
// constraint is not self-intersection of the mesh -- the generator already
// bounds that -- but the *displaced* surface: a field's displacement is
// amplitude-bounded by the mesh's reach, which here is the tube radius, so a
// full-amplitude mode can grow the tube to twice its size. The hole, of radius
// $1 - t$, therefore has to survive a tube of $2 t$, which needs $t < 1\/3$.
// Above that the donut inflates shut and reads as a blob at every mode.
const DONUT_TUBE_RATIO: f64 = 0.25;
const MOEBIUS_RADIUS_SLACK: f64 = 2.0;
// The period of the swept axis, chosen so the revolution radius comes out near
// 1: the gallery's meshes are unit-scale, and the marks, the displacement bound
// and the camera framing are all fractions of an object's extent, so a surface
// an order of magnitude smaller reads wrong on every one of them.
const QUOTIENT_CIRCUMFERENCE: f64 = std::f64::consts::TAU;
// The band's width as a fraction of the revolution radius. Wide enough that
// the quasi-uniform resolution puts several cells across it -- a band one cell
// wide has no interior and nothing to show a field on -- and still well inside
// `MOEBIUS_RADIUS_SLACK`, which keeps the swept strip clear of its own axis.
const MOEBIUS_WIDTH_RATIO: f64 = 1.5;

// The intrinsic dimension the grid opens on, and the top of its dimension
// slider: the same $1..=3$ the reference cell spans, since both live in the
// fixed ambient $RR^3$. A square (dim 2) matches the historical planar grid.
pub const GRID_DIM_DEFAULT: usize = 2;
pub const GRID_DIM_MAX: usize = 3;

// The reference cell the Whitney-basis study opens on, and the top of its
// dimension slider: the intrinsic dimensions the fixed ambient $RR^3$ embeds.
// A triangle (dim 2) matches the historical local-shape-function gallery.
pub const REFERENCE_CELL_DIM: usize = 2;
pub const REFERENCE_CELL_DIM_MAX: usize = 3;

// Hodge-Laplace modes an eigenmode study solves for by default. Chosen so both
// low grades close on a complete degeneracy shell on the sphere: grade 0 fills
// $l = 0..=3$ ($sum (2l+1) = 16$) and grade 1 fills $l = 1, 2$
// ($6 + 10 = 16$), so the orbital pyramid the UI lays these out in has no
// half-built final row.
pub const DEFAULT_NMODES: usize = 16;
// The top of the eigenmode-count slider. The per-grade solve is dense
// ($O(n^3)$) in the shift-invert factorization but the projected subspace is
// what grows with the mode count, so a cap here keeps the background solve from
// widening past what stays interactive. Enough to close the sphere's $l = 0..=5$
// grade-0 pyramid ($sum_(l=0)^5 (2l+1) = 36$) with a full final row.
pub const EIGENMODES_NMODES_MIN: usize = 1;
pub const EIGENMODES_NMODES_MAX: usize = 36;

// A time-dependent study samples its solution at this many steps over the
// solve's final time. Enough that the linear interpolation between frames reads
// as continuous motion at the trajectory's playback rate.
pub const DEFAULT_TRAJECTORY_STEPS: usize = 160;
// The trajectory-sampling slider's range. The lower end still reads as motion
// under interpolation; the upper end is where the sampled frames stop earning
// their memory against the linear interpolant between them.
pub const TRAJECTORY_STEPS_MIN: usize = 10;
pub const TRAJECTORY_STEPS_MAX: usize = 400;
// The heat flow's final time: long enough for the initial bump to diffuse and
// visibly decay on the unit-scale gallery meshes. The wave equation's, in the
// same units: several periods of the lowest modes, so the fronts propagate and
// reflect rather than barely stirring.
pub const HEAT_FINAL_TIME: f64 = 0.5;
pub const WAVE_FINAL_TIME: f64 = 12.0;
// The final-time sliders' ranges, one per equation because the two evolve on
// different scales: the parabolic smoothing settles quickly, the hyperbolic
// fronts want several periods to propagate and reflect.
pub const HEAT_FINAL_TIME_MIN: f64 = 0.05;
pub const HEAT_FINAL_TIME_MAX: f64 = 2.0;
pub const WAVE_FINAL_TIME_MIN: f64 = 1.0;
pub const WAVE_FINAL_TIME_MAX: f64 = 30.0;

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
  // Constructed only by `build.rs`, from the extension of whatever is in the
  // asset directory -- so it goes unconstructed whenever no `.msh` ships, and
  // that is not a reason to drop the reader. The directory is the source of
  // truth for which meshes exist; the formats are what it may contain.
  #[allow(dead_code)]
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

/// Which flat quotient of the square, among those with an $RR^3$ realization.
///
/// Both are [`simplicial::mesher::quotient::FlatQuotient`]s -- one generator, two
/// per-axis identifications -- and both are drawn through the surfaces of
/// revolution of `mesher::quotient_embed`, the constructions that fit the fixed
/// ambient $RR^3$. The rest of the family does not fit and so is not offered:
/// the Klein bottle has no $RR^3$ embedding at all, and the isometric Clifford
/// realization of the torus needs $RR^4$.
///
/// **The surface the viewer shows is the curved one, not the flat quotient.**
/// A `MeshSource` produces coordinates, and every geometric quantity downstream
/// is induced by them, so the donut here carries the Gaussian curvature of a
/// torus of revolution -- positive on the outer rim, negative on the inner --
/// and its spectrum is that surface's, not the flat torus's. This is the
/// inversion of the parent's invariant 2 doing exactly what it says: the viewer
/// is extrinsic by necessity, and the honest reading is that these are
/// embedded surfaces which happen to be built by identification.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuotientSurface {
  /// The torus of revolution: both axes periodic. Orientable and closed, with
  /// the Betti numbers $1, 2, 1$ of $T^2$.
  Donut,
  /// The Möbius band: one axis twisted about an open fiber. **Non-orientable**,
  /// and the one mesh in the gallery that is -- so it is where the reduced-grade
  /// marks that need a coherent orientation are refused rather than drawn with
  /// a per-cell sign.
  Moebius,
}

impl QuotientSurface {
  pub(crate) fn label(self) -> &'static str {
    match self {
      Self::Donut => "Donut",
      Self::Moebius => "Möbius band",
    }
  }

  /// The name a caller writes, on the CLI and in a preset.
  pub fn name(self) -> &'static str {
    match self {
      Self::Donut => "donut",
      Self::Moebius => "moebius",
    }
  }

  pub fn from_name(name: &str) -> Option<Self> {
    [Self::Donut, Self::Moebius]
      .into_iter()
      .find(|s| s.name() == name)
  }

  fn build(self, cells_axis: usize) -> (Complex, MeshCoords) {
    use simplicial::mesher::{quotient::FlatQuotient, quotient_embed};
    let cells_axis = cells_axis.max(QUOTIENT_CELLS_MIN);
    match self {
      Self::Donut => {
        // Quasi-uniform: the tube's period is `DONUT_TUBE_RATIO` of the
        // sweep's, so a shared count would stretch every cell by that ratio.
        let quotient = FlatQuotient::quasi_uniform(
          simplicial::linalg::Vector::from_column_slice(&[
            QUOTIENT_CIRCUMFERENCE,
            QUOTIENT_CIRCUMFERENCE * DONUT_TUBE_RATIO,
          ]),
          vec![Identification::Periodic; 2],
          cells_axis,
        );
        let coords = quotient_embed::donut_r3(&quotient, DONUT_TUBE_RATIO);
        (quotient.triangulate().0, coords)
      }
      Self::Moebius => {
        let quotient = FlatQuotient::moebius(
          QUOTIENT_CIRCUMFERENCE,
          MOEBIUS_WIDTH_RATIO * QUOTIENT_CIRCUMFERENCE / std::f64::consts::TAU,
          cells_axis,
        );
        let coords = quotient_embed::moebius_r3(&quotient, MOEBIUS_RADIUS_SLACK);
        (quotient.triangulate().0, coords)
      }
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
  /// A triangulated unit cube of the given intrinsic dimension, with the given
  /// number of cells per axis: a manifold with boundary in arbitrary dimension.
  /// The Kuhn triangulation is dimension-general, so `dim = 1` is an interval,
  /// `dim = 2` a square of triangles, `dim = 3` a cube of tetrahedra -- one
  /// generator, no special case. `dim` ranges over $1..=3$, the intrinsic
  /// dimensions the fixed ambient $RR^3$ embeds.
  Grid { dim: usize, cells_axis: usize },
  /// The standard reference cell of the given intrinsic dimension as a one-cell
  /// mesh. The Whitney-basis study on it is the local shape functions, so
  /// "local shape functions" is that composition, not a study of its own.
  /// `dim` ranges over $1..=3$, the intrinsic dimensions the fixed ambient
  /// $RR^3$ embeds.
  ReferenceCell { dim: usize },
  /// A flat quotient of the square, realized in $RR^3$ as a surface of
  /// revolution: the two members of the family that fit in the fixed ambient.
  Quotient {
    surface: QuotientSurface,
    cells_axis: usize,
  },
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
      MeshSource::Quotient { surface, .. } => surface.label().to_string(),
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
  pub fn build(&self) -> Result<Mesh, String> {
    match self {
      MeshSource::Sphere { subdivisions } => Ok(simplicial::mesher::sphere::mesh_sphere_surface(
        *subdivisions,
      )),
      MeshSource::Grid { dim, cells_axis } => {
        let (topology, coords) =
          simplicial::mesher::cartesian::CartesianGrid::new_unit(*dim, *cells_axis).triangulate();
        // The renderer draws in 3D; a grid of intrinsic dimension below 3 lifts
        // into $RR^3$ (a curve or a $z = 0$ surface), a no-op once `dim >= 3`,
        // exactly as the flat reference-cell scenes do.
        Ok((topology, coords.embed_euclidean((*dim).max(3))))
      }
      MeshSource::ReferenceCell { dim } => {
        let (topology, coords) = standard_coord_complex(*dim);
        // A reference cell of `dim < 3` embeds as itself in the `z = 0` plane;
        // a no-op once `dim >= 3`.
        Ok((topology, coords.embed_euclidean((*dim).max(3))))
      }
      MeshSource::Quotient {
        surface,
        cells_axis,
      } => Ok(surface.build(*cells_axis)),
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
  Heat {
    grade: ExteriorGrade,
    nsteps: usize,
    final_time: f64,
  },
  /// The wave equation $diff_(t t) u = -Delta u$ of a localized bump at rest, as
  /// a sampled trajectory: the hyperbolic counterpart of [`Self::Heat`], fronts
  /// propagating and reflecting off any boundary.
  Wave {
    grade: ExteriorGrade,
    nsteps: usize,
    final_time: f64,
  },
}

impl Study {
  /// The study the viewer opens on: grade-0 eigenmodes.
  pub(crate) fn start() -> Study {
    Study::Eigenmodes {
      grade: Dim::ZERO,
      nmodes: DEFAULT_NMODES,
    }
  }

  /// The form grade the study is posed at, for the studies that are posed at
  /// one -- the eigenproblem and the two evolutions. `None` where the study
  /// spans every grade at once (the Whitney basis, the Hodge decomposition) or
  /// reads the grade off its data (an explicit cochain list).
  pub(crate) fn grade(&self) -> Option<ExteriorGrade> {
    match self {
      Study::Eigenmodes { grade, .. } | Study::Heat { grade, .. } | Study::Wave { grade, .. } => {
        Some(*grade)
      }
      Study::WhitneyBasis | Study::HodgeDecomposition | Study::Cochains(_) => None,
    }
  }

  pub(crate) fn label(&self) -> String {
    match self {
      Study::Eigenmodes { grade, .. } => format!("Eigenmodes, grade {grade}"),
      Study::WhitneyBasis => "Whitney basis".to_string(),
      Study::Cochains(_) => "Cochains".to_string(),
      Study::HodgeDecomposition => "Hodge decomposition".to_string(),
      Study::Heat { grade, .. } => format!("Heat equation, grade {grade}"),
      Study::Wave { grade, .. } => format!("Wave equation, grade {grade}"),
    }
  }

  /// Builds the study's scene on `mesh`. For an eigenmode grade this runs that
  /// grade's dense eigensolve -- the expensive path the caller runs on a
  /// background thread and memoizes; the Whitney basis and the explicit
  /// cochains are cheap.
  pub fn build(&self, mesh: &Mesh) -> Scene {
    let (topology, coords) = mesh;
    match self {
      Study::Eigenmodes { grade, nmodes } => Scene::mesh_grade(topology, coords, *grade, *nmodes),
      Study::WhitneyBasis => Scene::whitney_basis_mesh(topology.clone(), coords.clone()),
      Study::Cochains(specs) => Scene::cochains(topology.clone(), coords.clone(), specs),
      Study::HodgeDecomposition => Scene::hodge_decomposition(topology.clone(), coords.clone()),
      Study::Heat {
        grade,
        nsteps,
        final_time,
      } => Scene::heat(
        topology.clone(),
        coords.clone(),
        *grade,
        *nsteps,
        *final_time,
      ),
      Study::Wave {
        grade,
        nsteps,
        final_time,
      } => Scene::wave(
        topology.clone(),
        coords.clone(),
        *grade,
        *nsteps,
        *final_time,
      ),
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
  /// A one-line gloss of what the preset shows, for the browser's hover. It
  /// names the point in the product in the reader's terms -- the mathematics on
  /// the mesh -- so the curated set is legible before it is clicked.
  pub(crate) description: &'static str,
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
///
/// It opens on the default marks, which is the glyphs alone. The particles are
/// not a thing a preset should switch on for a reader: their cost is the same
/// on every mesh (see [`Marks`]), so no preset is in a position to decide the
/// reader can afford them.
pub(crate) fn start_preset() -> Preset {
  curl_on_triforce()
}

fn curl_on_triforce() -> Preset {
  Preset {
    name: "Constant / curl / div",
    description: "The three grade-1 cochains on the triforce: a constant field, a pure curl, a pure divergence",
    mesh: MeshSource::Triforce,
    study: Study::Cochains(crate::demos::triforce_examples()),
    // The second of the three (constant, curl, div), all grade 1 and so all
    // line fields, in the order `triforce_examples` builds them.
    selection: Some(Selection::Line(1)),
    marks: None,
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
      description: "Grade-0 Hodge-Laplace eigenmodes of the sphere: the spherical-harmonic multiplets, laid out as the orbital pyramid",
      mesh: MeshSource::START,
      study: Study::start(),
      selection: None,
      marks: None,
    },
    Preset {
      name: "Whitney basis",
      description: "The local shape functions: the Whitney basis on a single reference cell, one field per DOF simplex",
      mesh: MeshSource::ReferenceCell {
        dim: REFERENCE_CELL_DIM,
      },
      study: Study::WhitneyBasis,
      selection: None,
      marks: None,
    },
    Preset {
      name: "Global shape functions",
      description: "The Whitney basis on the multi-cell triforce: the global shape functions, whose support spans several cells",
      mesh: MeshSource::Triforce,
      study: Study::WhitneyBasis,
      selection: None,
      marks: None,
    },
    curl_on_triforce(),
    Preset {
      name: "Heat equation",
      description: "Parabolic smoothing of a localized bump on the sphere, sampled as a scrubbable trajectory",
      mesh: MeshSource::START,
      study: Study::Heat {
        grade: Dim::ZERO,
        nsteps: DEFAULT_TRAJECTORY_STEPS,
        final_time: HEAT_FINAL_TIME,
      },
      selection: None,
      marks: None,
    },
    Preset {
      name: "Wave equation",
      description: "Hyperbolic propagation and reflection of a bump on the sphere, sampled as a scrubbable trajectory",
      mesh: MeshSource::START,
      study: Study::Wave {
        grade: Dim::ZERO,
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
        description: "Grade-1 eigenmodes of a genus-1 surface: the harmonic 1-forms are its zero shell, and they see the hole",
        mesh: MeshSource::Builtin(bob),
        study: Study::Eigenmodes {
          grade: Dim::ONE,
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
      description: "A grade-1 field split into exact, coexact and harmonic parts on a genus-1 surface; opens on the harmonic shell",
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
    let last_grade = preset.study.grade().unwrap_or(Dim::ZERO);
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
    if let Some(grade) = study.grade() {
      self.last_grade = grade;
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
    if let Some(grade) = preset.study.grade() {
      self.last_grade = grade;
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

  /// The particles are opt-in, everywhere. Their cost does not scale with the
  /// mesh -- the population is a fixed count -- so there is no mesh on which
  /// assuming them is cheap, and a weak GPU should not spend it unasked. The
  /// glyphs stay on, so a line field still has a mark.
  #[test]
  fn the_particles_are_opt_in() {
    let marks = crate::ui::Marks::default();
    assert!(!marks.particles);
    assert!(marks.glyphs, "a line field must still carry a mark");
    // No preset overrides that: a preset is in no position to decide the
    // reader can afford them.
    for preset in presets() {
      if let Some(marks) = preset.marks {
        assert!(
          !marks.particles,
          "{} switches the particles on",
          preset.name
        );
      }
    }
  }

  /// Both quotient surfaces build, in $RR^3$, with the topology their gluing
  /// says -- and the Möbius band is the gallery's **non-orientable** mesh,
  /// which the donut is not.
  ///
  /// That contrast is the reason the band is offered at all: the reduced-grade
  /// reduction takes a coherent orientation wherever the Hodge star fires, so
  /// the band is the mesh on which that path is exercised rather than assumed.
  /// A picker entry that could not be non-orientable would not test it.
  #[test]
  fn the_quotient_surfaces_build_and_differ_in_orientability() {
    let cases = [
      (QuotientSurface::Donut, vec![1, 2, 1], false, true),
      (QuotientSurface::Moebius, vec![1, 1, 0], true, false),
    ];
    for (surface, betti, has_boundary, orientable) in cases {
      let source = MeshSource::Quotient {
        surface,
        cells_axis: QUOTIENT_CELLS_DEFAULT,
      };
      let (topology, coords) = source.build().expect("a generated mesh always builds");

      assert_eq!(coords.dim(), 3, "{}: the fixed ambient", surface.name());
      assert_eq!(topology.dim(), 2, "{}: a surface", surface.name());
      assert_eq!(topology.betti_numbers(), betti, "{}", surface.name());
      assert_eq!(topology.has_boundary(), has_boundary, "{}", surface.name());
      assert_eq!(
        topology.orientation().is_some(),
        orientable,
        "{}",
        surface.name()
      );
      // The label and the CLI name reach the picker and the command line from
      // the one enum, so neither can name a surface the other cannot.
      assert_eq!(source.label(), surface.label());
      assert_eq!(QuotientSurface::from_name(surface.name()), Some(surface));
    }
  }

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
    // And it opens on the default marks rather than choosing for the reader.
    assert!(preset.marks.is_none());
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
  /// Checked on the generated donut rather than on its own mesh (Bob): what the
  /// index depends on is the shell ordering of `Study::HodgeDecomposition` and
  /// the surface's first Betti number, and the donut has the same $b_1 = 2$ at
  /// a few dozen vertices instead of ~3000. It is a faithful stand-in for what
  /// is being tested, not a weaker one -- and it keeps Bob's harmonic solve out
  /// of the test suite.
  #[test]
  fn the_hodge_preset_opens_on_the_harmonic_shell() {
    let hodge = presets()
      .into_iter()
      .find(|p| matches!(p.study, Study::HodgeDecomposition))
      .expect("a Hodge decomposition preset");
    let mesh = MeshSource::Quotient {
      surface: QuotientSurface::Donut,
      cells_axis: QUOTIENT_CELLS_DEFAULT,
    }
    .build()
    .expect("a generated mesh always builds");
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
