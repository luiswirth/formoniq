use std::borrow::Cow;

use glatt::field::DiffFormClosure;
use gramian::Metric;
use simplicial::Sign;
use simplicial::linalg::Vector;

use crate::bake::CellCorner;
use crate::surface::Surface;
use crate::ui::Selection;
use derham::{
  cochain::Cochain, interpolate::interpolant::WhitneyInterpolant, project::derham_map,
  section::CoordFieldExt,
};
use exterior::{Blade, ExteriorGrade, MultiForm};
use simplicial::{
  Dim,
  atlas::{Bary, MeshPoint},
  geometry::coord::mesh::MeshCoords,
  topology::{
    complex::Complex,
    handle::{SimplexIdx, SimplexRef},
    role::Cell,
    simplex::Simplex,
  },
};

/// A renderable scene: a simplicial surface together with fields on it.
///
/// The seam between the formoniq engine and the viewer. A scene is produced
/// either by a live solve (the native path here) or, later, by deserializing a
/// cached bundle; the viewer neither knows nor cares which. It carries the
/// engine's own types — topology, coordinates, cochains — rather than a lossy
/// export format, so the coloring, displacement and mode selection stay
/// decisions of the viewer.
///
/// A field's grade decides the mark it is drawn with, not a per-scene choice,
/// and the rule is one line: reduce the $k$-form to its *reduced grade*
/// $min(k, n-k)$ via the Hodge star, then dispatch on that. A reduced grade of
/// 0 is a scalar density coloring the surface ([`ScalarField`]); a reduced
/// grade of 1 is a line field drawn with line-integral convolution
/// ([`LineField`]). Both marks exhaust $n <= 3$: only $n >= 4$ produces a
/// reduced grade $>= 2$, an $(n-k)$-dimensional sheet with no mark yet -- a
/// future mark, not a special case to route around.
#[derive(Clone)]
pub struct Scene {
  pub topology: Complex,
  pub coords: MeshCoords,
  /// The reduction the bake draws and every field mark is chosen against: the
  /// mesh itself below $n = 3$, its boundary $diff M$ for a solid. Built once
  /// with the scene, because the marks are filed against its dimension, not
  /// the parent's.
  pub(crate) surface: Surface,
  pub fields: Vec<ScalarField>,
  pub line_fields: Vec<LineField>,
}

/// How a field varies in time -- the temporal model the render clock reads, one
/// axis orthogonal to the render mark (which the reduced grade picks) and the
/// spatial cochain.
///
/// Three cases dissolving into one generality, not three mechanisms:
///
/// - [`Self::Static`] is a field with no clock.
/// - [`Self::StandingWave`] is the analytic special case
///   $u(t) = cos(sqrt(lambda) t) phi$: one spatial mode modulated by a scalar
///   the GPU evaluates in closed form, so the cochain is baked once and the
///   vertex shader re-times it (`wave_omega`, `wave_amplitude`).
/// - [`Self::Trajectory`] is the general sampled case: a time-indexed family of
///   cochains from a solve (heat, wave), with no closed form. It is interpolated
///   on the CPU and its field stream is re-baked per frame -- exactly the
///   "scrubbing a trajectory rewrites only the field stream" the bake anticipates.
///
/// The eigenmode is the degenerate one-mode-with-known-modulation point of the
/// trajectory, which is why the two share every display path below and differ
/// only in where the animation is evaluated. The previous `Option<f64>`
/// eigenvalue was already this axis in two states (`None`/`Some`); this names the
/// third.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum FieldTime {
  Static,
  StandingWave { eigenvalue: f64 },
  Trajectory { dt: f64, frames: Vec<Cochain> },
}

impl FieldTime {
  /// The Hodge-Laplace eigenvalue driving the analytic standing wave, when the
  /// field is one. `None` for a static field and for a sampled trajectory --
  /// neither is animated by a dispersion relation. What the UI reads for the
  /// degeneracy pyramid and the transport frequency readout.
  pub fn eigenvalue(&self) -> Option<f64> {
    match self {
      FieldTime::StandingWave { eigenvalue } => Some(*eigenvalue),
      _ => None,
    }
  }

  /// The GPU standing wave's angular frequency $omega = sqrt(lambda)$. Zero for
  /// anything the GPU does not modulate in closed form -- a static field, and a
  /// trajectory, whose height stream is rewritten per frame on the CPU instead
  /// (so $cos(0 dot.c t) = 1$ applies the current frame's height at full
  /// amplitude, statically).
  pub fn wave_omega(&self) -> f32 {
    self.eigenvalue().map_or(0.0, f64::sqrt) as f32
  }

  /// Whether the field animates the surface's displacement height at all -- an
  /// eigenmode riding $cos(sqrt(lambda) t)$, or a trajectory whose per-frame
  /// height moves. A static field does not, so it offers no displacement toggle
  /// and takes an asymmetric (non-diverging) colormap.
  pub fn animates(&self) -> bool {
    !matches!(self, FieldTime::Static)
  }

  /// Whether the field is a sampled trajectory, whose field stream the caller
  /// must re-bake per frame (the GPU has no closed form for it).
  pub fn is_trajectory(&self) -> bool {
    matches!(self, FieldTime::Trajectory { .. })
  }

  /// The trajectory's total solve-time span $T = dif t dot.c (N - 1)$, the
  /// interval the transport scrubs and the export samples. `None` for a field
  /// that is not a sampled trajectory.
  pub fn duration(&self) -> Option<f64> {
    match self {
      FieldTime::Trajectory { dt, frames } => Some(dt * frames.len().saturating_sub(1) as f64),
      _ => None,
    }
  }

  /// The field's cochain at solve-time `t`, linearly interpolated between the
  /// bracketing sampled frames -- lerping coefficients *is* lerping the Whitney
  /// field, since the interpolation is linear in them. For a static field or a
  /// standing wave the spatial cochain does not itself vary (the GPU modulates
  /// the standing wave), so `base` is returned unchanged. `t` is clamped to the
  /// sampled interval; the caller's own loop decides the wrap.
  pub fn frame_at<'a>(&'a self, base: &'a Cochain, t: f64) -> Cow<'a, Cochain> {
    match self {
      FieldTime::Trajectory { dt, frames } if frames.len() > 1 && *dt > 0.0 => {
        let last = frames.len() - 1;
        let x = (t / dt).clamp(0.0, last as f64);
        let i = x.floor() as usize;
        if i >= last {
          return Cow::Borrowed(&frames[last]);
        }
        let s = x - i as f64;
        let (a, b) = (frames[i].coeffs(), frames[i + 1].coeffs());
        Cow::Owned(Cochain::new(frames[i].grade(), a + (b - a) * s))
      }
      FieldTime::Trajectory { frames, .. } => Cow::Borrowed(&frames[0]),
      _ => Cow::Borrowed(base),
    }
  }
}

/// A named scalar field on the surface: the reduced-grade-0 mark, a density
/// coloring the surface and displacing it as a standing wave.
///
/// A grade-0 form is a density directly; a top-grade ($k = n$) form becomes one
/// by the pointwise Hodge star $star: Lambda^n -> Lambda^0$. Either way the
/// density is read per cell at draw time (`surface_corner_values`), not stored
/// -- so the top form's discontinuity across cells survives to the colormap
/// instead of being averaged away.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ScalarField {
  pub name: String,
  /// The grade $k$ of the form this field was reconstructed from, before the
  /// reduction to a density. A genuine 0-form keeps $k = 0$; a top form keeps
  /// $k = n$ even though it is drawn through its Hodge star. Carried so the
  /// gallery can organize by original grade, not by the render mark the reduced
  /// grade happens to share with grade 0.
  pub grade: ExteriorGrade,
  /// The field's spatial representative: the cochain the static readout, the
  /// colormap range and the initial frame are read from. For a
  /// [`FieldTime::Trajectory`] this is its first frame (the initial condition);
  /// the moving field is read through [`FieldTime::frame_at`] on [`Self::time`].
  pub cochain: Cochain,
  /// How this field varies in time: a static field, an eigenmode's standing
  /// wave, or a solve's sampled trajectory. Was an `Option<f64>` eigenvalue; see
  /// [`FieldTime`].
  pub time: FieldTime,
  /// The DOF simplex this field is dual to, when the field is a raw Whitney
  /// basis function. `None` for a solved field (an eigenmode, a trajectory),
  /// which has no single dual simplex. Lets the picker group basis functions by
  /// grade and label each cell by its DOF (via `dof_label`) without reparsing
  /// [`Self::name`]. Kept as the simplex, not its rendered label, so the DOF is
  /// a typed value the UI formats rather than a string the model commits to.
  pub dof: Option<Simplex>,
}

/// A named line field on the surface: the reduced-grade-1 mark, drawn as arrow
/// glyphs and advected particles.
///
/// A grade-1 (or, via the Hodge star, grade-$(n-1)$) form reduces to a genuine
/// tangent *line* field. Its (unsigned) magnitude $|V|_g$ is read per cell
/// (`surface_corner_values`) and tints the surface the marks are drawn on.
///
/// The glyphs are static: $ker$ and $sharp$ are scale-invariant, so the
/// standing wave $u(t) = cos(sqrt(lambda) t) phi$ leaves them fixed and swings
/// only the magnitude tint through zero. A single real eigenmode does not
/// travel, so the glyphs are never advected -- only the particles are, on the
/// object's own clock.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LineField {
  pub name: String,
  /// The grade $k$ of the form this field was reconstructed from, before the
  /// reduction to a line field. A grade-1 form keeps $k = 1$; a grade-$(n-1)$
  /// form keeps $k = n-1$ even though it is drawn through its Hodge star. See
  /// [`ScalarField::grade`].
  pub grade: ExteriorGrade,
  /// The original $k$-cochain, kept whole so the surface tint, the glyphs and
  /// the particles all read the *true* Whitney field $W c$ (via
  /// [`WhitneyInterpolant`]) cell by cell -- there is no per-vertex reduction,
  /// because a reduced-grade field has no single value at a shared vertex.
  pub cochain: Cochain,
  /// See [`ScalarField::time`].
  pub time: FieldTime,
  /// See [`ScalarField::dof`].
  pub dof: Option<Simplex>,
}

/// The vertex-tuple label of a DOF simplex, e.g. `013` for the face
/// $\{0, 1, 3\}$. Single-digit vertices concatenate; once any vertex reaches two
/// digits the tuple is comma-separated, so the label stays unambiguous on a mesh
/// with ten or more vertices. Purely a display of the typed [`ScalarField::dof`]
/// / [`LineField::dof`], computed at the UI boundary rather than stored.
pub(crate) fn dof_label(dof: &Simplex) -> String {
  let separator = if dof.vertices.iter().all(|&v| v < 10) {
    ""
  } else {
    ","
  };
  dof
    .vertices
    .iter()
    .map(ToString::to_string)
    .collect::<Vec<_>>()
    .join(separator)
}

/// The colormap range of a per-corner value stream, for normalization. Falls
/// back to a unit range on an empty or constant field so the viewer never
/// normalizes by a zero span.
pub(crate) fn corner_bounds(values: &[f64]) -> (f32, f32) {
  let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
  for &v in values {
    lo = lo.min(v);
    hi = hi.max(v);
  }
  if lo < hi {
    (lo as f32, hi as f32)
  } else {
    (-1.0, 1.0)
  }
}

/// The display metadata a reconstructed field carries regardless of which
/// render mark it lands in -- everything [`Scene::field`] needs beyond the
/// cochain itself, bundled so the two independent `Option`s
/// ([`ScalarField::eigenvalue`]/[`ScalarField::dof`]) don't turn the
/// constructor into an unreadable run of positional arguments.
struct FieldMeta {
  name: String,
  time: FieldTime,
  dof: Option<Simplex>,
}

/// What the selected field offers to be read with -- which of
/// [`crate::ui::FieldView`]'s settings are live.
///
/// The mesh side has no counterpart: every scene has geometry, so its settings
/// are always live and there is nothing to gate. Only the field is asked, and
/// the answer is its reduced grade's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct FieldOffers {
  /// Whether the field drives a standing wave. A reduced grade of 0 displaces
  /// the surface along its normal, and only an eigenmode has the dispersion
  /// relation to do it at: without an eigenvalue the amplitude is already zero,
  /// so the toggle would control nothing.
  pub(crate) displacement: bool,
  /// Whether the field has marks of its own: a reduced grade of 1 is a tangent
  /// line field, and the glyphs and particles are its two readings. A density
  /// has no mark beyond the surface it paints, which is the mesh's.
  pub(crate) marks: bool,
  /// Whether the field has an interior to march. A solid's field lives in a
  /// volume the boundary primitive cannot show, so the medium is offered
  /// exactly when the manifold is codimension-zero enough to have one -- an
  /// intrinsic-dimension question, not a grade one, which is why it is the only
  /// offer read off the complex rather than the selection.
  pub(crate) volume: bool,
}

impl FieldOffers {
  /// Whether the field offers anything at all -- false for a density that is no
  /// eigenmode (a raw Whitney basis function), whose whole rendering is the
  /// tint on the mesh's surface.
  pub(crate) fn any(self) -> bool {
    self.displacement || self.marks || self.volume
  }
}

impl Scene {
  /// The reduced grade's answer to what its field can be read with, and the one
  /// place it is asked outside the display: a selection already *is* the
  /// reduction (which list it indexes is which mark it landed in), so this
  /// reads it off rather than dispatching on grade a second time.
  pub(crate) fn offers(&self, selection: Selection) -> FieldOffers {
    let volume = self.topology.dim() >= 3;
    match selection {
      Selection::Scalar(index) => FieldOffers {
        displacement: self.fields[index].time.animates(),
        marks: false,
        volume,
      },
      Selection::Line(_) => FieldOffers {
        displacement: false,
        marks: true,
        volume,
      },
    }
  }

  /// Hodge-Laplace eigenmodes of a single grade -- standing-wave normal
  /// modes, $Delta u = lambda u$ -- of an arbitrary simplicial surface with
  /// the given geometry, filed into `fields` or `line_fields` through the
  /// same [`Self::field`] dispatch a raw Whitney basis function goes
  /// through: an eigenmode and a one-hot cochain differ only in where the
  /// cochain comes from (an eigensolve vs. a Kronecker delta), not in
  /// how it is reconstructed or displayed.
  ///
  /// A failed eigensolve contributes no fields and is reported on stderr: an
  /// iteration budget too small for one mesh is not a reason to take the
  /// viewer down.
  fn eigenmode_fields(
    topology: &Complex,
    surface: &Surface,
    coords: &MeshCoords,
    grade: ExteriorGrade,
    nmodes: usize,
    fields: &mut Vec<ScalarField>,
    line_fields: &mut Vec<LineField>,
  ) {
    use formoniq::{problems::elliptic::solve_evp, whitney_complex::WhitneyComplex};

    let metric = coords.to_edge_lengths_sq(topology);
    let solved = solve_evp(&WhitneyComplex::new(topology, &metric), grade, nmodes);
    let (eigenvals, _, eigenfuncs) = match solved {
      Ok(solved) => solved,
      Err(err) => {
        eprintln!("grade {grade} eigensolve failed: {err}");
        return;
      }
    };

    for (i, (&lambda, col)) in eigenvals.iter().zip(eigenfuncs.column_iter()).enumerate() {
      let name = format!("mode {i} (grade {grade}, lambda = {lambda:.2})");
      let cochain = Cochain::new(grade, col.into_owned());
      Self::field(
        topology,
        surface,
        FieldMeta {
          name,
          time: FieldTime::StandingWave { eigenvalue: lambda },
          dof: None,
        },
        cochain,
        fields,
        line_fields,
      );
    }
  }

  /// Grade-0 Hodge-Laplace eigenmodes of an arbitrary simplicial surface.
  /// Neither the mesh nor its embedding are assumed to be a sphere: any 2D
  /// `Complex` with a `MeshCoords` realization goes in, so a scene is exactly
  /// as general as the underlying eigensolve. The spherical harmonics are one
  /// instantiation of this ([`Self::spherical_harmonics`]), not a special
  /// case baked into the solve.
  pub fn eigenmodes(topology: Complex, coords: MeshCoords, nmodes: usize) -> Self {
    // Built before the fields are filed: the mark each one gets is chosen
    // against the surface's dimension, not the mesh's.
    let surface = Surface::of(&topology, &coords);
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    Self::eigenmode_fields(
      &topology,
      &surface,
      &coords,
      Dim::ZERO,
      nmodes,
      &mut fields,
      &mut line_fields,
    );
    Self {
      surface,
      topology,
      coords,
      fields,
      line_fields,
    }
  }

  /// Hodge-Laplace eigenmodes on the unit sphere — the discrete spherical
  /// harmonics — at *every* grade $0..=n$ of the de Rham complex, together in
  /// one scene, on an icosphere of the given subdivision depth.
  ///
  /// Every grade goes through the same `field` reduction: on the sphere
  /// ($n = 2$) grade 0 and the top grade 2 are scalar densities (grade 2 by its
  /// pointwise Hodge star, $star(f dvol) = f$), while grade 1 is a tangent line
  /// field. No grade is special-cased -- the extremal grade 2 runs on exactly
  /// the same code as grade 0, which is the point of the $min(k, n-k)$ dispatch
  /// (discussion #101).
  pub fn spherical_harmonics(nsubdivisions: usize, nmodes: usize) -> Self {
    use simplicial::mesher::sphere::mesh_sphere_surface;

    let (topology, coords) = mesh_sphere_surface(nsubdivisions);
    // Built before the fields are filed: the mark each one gets is chosen
    // against the surface's dimension, not the mesh's.
    let surface = Surface::of(&topology, &coords);
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    for grade in topology.dim().range_inclusive() {
      let (mut f, mut l) = Self::eigenmodes_grade(&topology, &coords, grade, nmodes);
      fields.append(&mut f);
      line_fields.append(&mut l);
    }
    Self {
      surface,
      topology,
      coords,
      fields,
      line_fields,
    }
  }

  /// The Hodge-Laplace eigenmodes of a *single* grade on a shared surface
  /// mesh: the fields one grade's eigensolve contributes, split by their
  /// render mark. The unit the gallery computes lazily and
  /// memoizes -- the mesh is built once and passed in, so switching grade pays
  /// only for that grade's solve, and only the first time it is viewed. The
  /// discrete spherical harmonics are one instantiation, the mesh being a
  /// sphere; nothing here assumes it.
  pub fn eigenmodes_grade(
    topology: &Complex,
    coords: &MeshCoords,
    grade: ExteriorGrade,
    nmodes: usize,
  ) -> (Vec<ScalarField>, Vec<LineField>) {
    // Built before the fields are filed: the mark each one gets is chosen
    // against the surface's dimension, not the mesh's.
    let surface = Surface::of(topology, coords);
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    Self::eigenmode_fields(
      topology,
      &surface,
      coords,
      grade,
      nmodes,
      &mut fields,
      &mut line_fields,
    );
    (fields, line_fields)
  }

  /// The bare icosphere carrying a single constant field: the mesh of
  /// [`Self::spherical_harmonics`] without its eigensolve.
  /// Stands in for the real scene so the viewer can show the sphere the instant
  /// the window opens, while the solve runs in the background and swaps the
  /// actual modes in when it lands. The lone field has no eigenvalue, so it is
  /// drawn as a plain, undeformed surface.
  pub fn sphere_placeholder(nsubdivisions: usize) -> Self {
    use simplicial::mesher::sphere::mesh_sphere_surface;

    let (topology, coords) = mesh_sphere_surface(nsubdivisions);
    Self::placeholder_on(topology, coords)
  }

  /// The same solve-free placeholder as [`Self::sphere_placeholder`], but on a
  /// mesh already in hand -- so the gallery can share one sphere mesh between
  /// the instant placeholder and the per-grade solves that follow, rather than
  /// meshing twice.
  pub fn placeholder_on(topology: Complex, coords: MeshCoords) -> Self {
    let nvertices = topology.skeleton_raw(0).len();
    let fields = vec![ScalarField {
      name: "loading...".to_string(),
      grade: Dim::ZERO,
      cochain: Cochain::new(Dim::ZERO, na::DVector::zeros(nvertices)),
      time: FieldTime::Static,
      dof: None,
    }];
    Self {
      surface: Surface::of(&topology, &coords),
      topology,
      coords,
      fields,
      line_fields: Vec::new(),
    }
  }

  /// A full [`Scene`] carrying one grade's Hodge-Laplace eigenmodes on the
  /// shared surface mesh: the mesh (cloned in) plus the fields that grade's
  /// eigensolve contributes. The display unit the gallery memoizes per grade.
  pub fn mesh_grade(
    topology: &Complex,
    coords: &MeshCoords,
    grade: ExteriorGrade,
    nmodes: usize,
  ) -> Self {
    let (fields, line_fields) = Self::eigenmodes_grade(topology, coords, grade, nmodes);
    Self {
      surface: Surface::of(topology, coords),
      topology: topology.clone(),
      coords: coords.clone(),
      fields,
      line_fields,
    }
  }

  /// Every Whitney basis function ("local shape function") of the standard
  /// reference cell of dimension `cell_dim` -- the single-cell case of the
  /// shared construction below, where every DOF simplex's support is the one
  /// cell itself.
  pub fn whitney_basis(cell_dim: impl Into<Dim>) -> Self {
    use simplicial::geometry::coord::mesh::standard_coord_complex;

    let cell_dim = cell_dim.into();
    let (topology, coords) = standard_coord_complex(cell_dim);
    // The renderer is 3D-only; a reference cell of `dim < 3` embeds as
    // itself in the `z = 0` plane, same as `bake.rs` does
    // for any other flat surface. A no-op once `cell_dim >= 3`.
    let coords = coords.embed_euclidean(cell_dim.max(Dim::new(3)));
    Self::whitney_basis_on(topology, coords)
  }

  /// Every Whitney basis function ("global shape function") of an arbitrary
  /// simplicial mesh -- the multi-cell case of the shared construction below,
  /// where a DOF simplex's support spans every cell incident to it, which is
  /// exactly where the LSF/GSF distinction shows up on screen: the same
  /// one-hot-cochain construction, just no longer confined to a single cell.
  pub fn whitney_basis_mesh(topology: Complex, coords: MeshCoords) -> Self {
    Self::whitney_basis_on(topology, coords)
  }

  /// A named list of explicit cochains on a mesh, each resolved from its
  /// [`crate::gallery::CochainSpec`] and reduced to its render mark through the
  /// same `field` dispatch every other field goes through -- a field
  /// here is a general linear combination, not confined to a single one-hot
  /// cochain. The worked triforce examples (a constant field, a pure-curl
  /// field and a pure-divergence field) are one such list; a loaded cochain
  /// file is a future one.
  pub fn cochains(
    topology: Complex,
    coords: MeshCoords,
    specs: &[crate::gallery::NamedCochain],
  ) -> Self {
    // Built before the fields are filed: the mark each one gets is chosen
    // against the surface's dimension, not the mesh's.
    let surface = Surface::of(&topology, &coords);
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    for named in specs {
      Self::field(
        &topology,
        &surface,
        FieldMeta {
          name: named.name.clone(),
          time: FieldTime::Static,
          dof: None,
        },
        named.spec.resolve(&topology),
        &mut fields,
        &mut line_fields,
      );
    }

    Self {
      surface,
      topology,
      coords,
      fields,
      line_fields,
    }
  }

  /// The Hodge decomposition of a probe field, as four switchable fields: the
  /// input $omega$ and its three $L^2$-orthogonal shells
  /// $omega = dif alpha + delta beta + h$ -- exact, coexact, and harmonic. The
  /// harmonic shell is what makes this more than the classical Helmholtz split:
  /// on a contractible mesh it vanishes, and on a genus-$g$ surface it is the
  /// $2g$-dimensional space the two independent cycles pair with, seen directly.
  ///
  /// The probe is a pulled-back ambient 1-form with a harmonic cycle mixed in
  /// (`hodge_probe_input`), so every mesh gets a non-trivial grade-1 form that
  /// exercises all three shells; the underlying `hodge_decompose` is itself
  /// dimension- and grade-general. A failed solve falls back to showing the
  /// input alone rather than taking the viewer down.
  pub fn hodge_decomposition(topology: Complex, coords: MeshCoords) -> Self {
    let input = hodge_probe_input(&topology, &coords);

    // Built before the fields are filed: the mark each one gets is chosen
    // against the surface's dimension, not the mesh's.
    let surface = Surface::of(&topology, &coords);
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();

    let named = match hodge_decompose(&topology, &coords, &input) {
      Ok(parts) => vec![
        ("ω input", input),
        ("dα exact", parts.exact),
        ("δβ coexact", parts.coexact),
        ("h harmonic", parts.harmonic),
      ],
      Err(err) => {
        eprintln!("grade-1 Hodge decomposition failed: {err}");
        vec![("ω input", input)]
      }
    };

    for (name, cochain) in named {
      Self::field(
        &topology,
        &surface,
        FieldMeta {
          name: name.to_string(),
          time: FieldTime::Static,
          dof: None,
        },
        cochain,
        &mut fields,
        &mut line_fields,
      );
    }

    Self {
      surface,
      topology,
      coords,
      fields,
      line_fields,
    }
  }

  /// Shared construction for [`Self::whitney_basis`] and
  /// [`Self::whitney_basis_mesh`]: one field per DOF simplex of every grade
  /// $0..=$ `topology.dim()`, each the reconstructed field of a one-hot
  /// cochain -- the basis function dual to a DOF simplex $sigma$ *is* the
  /// cochain $c_tau = delta_(sigma tau)$, so there is no separate "evaluate a
  /// Whitney form" code path here. The interpolant and the sharp musical
  /// isomorphism are exactly the general machinery a solved field (an
  /// eigenmode, or a future loaded cochain) goes through too. DOF simplices
  /// are named by their vertex tuple straight off `topology`'s own
  /// colexicographic skeleton order, which coincides with
  /// `standard_subsimps` on the single-cell reference complex.
  fn whitney_basis_on(topology: Complex, coords: MeshCoords) -> Self {
    let dim = topology.dim();
    // Built before the fields are filed: the mark each one gets is chosen
    // against the surface's dimension, not the mesh's.
    let surface = Surface::of(&topology, &coords);
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    for grade in dim.range_inclusive() {
      let ndofs = topology.nsimplices(grade);
      for (idof, dof_simp) in topology.skeleton_raw(grade).iter().enumerate() {
        let name = format!("W^{grade}_{{{}}}", dof_label(dof_simp));

        let mut coeffs = na::DVector::zeros(ndofs);
        coeffs[idof] = 1.0;
        let cochain = Cochain::new(grade, coeffs);

        Self::field(
          &topology,
          &surface,
          FieldMeta {
            name,
            time: FieldTime::Static,
            dof: Some(dof_simp.clone()),
          },
          cochain,
          &mut fields,
          &mut line_fields,
        );
      }
    }

    Self {
      surface,
      topology,
      coords,
      fields,
      line_fields,
    }
  }

  /// The heat flow $diff_t u = -kappa Delta u$ of a localized initial bump, as a
  /// single [`FieldTime::Trajectory`] field of grade `grade`: the sampled solution the
  /// transport scrubs and the surface re-bakes per frame. The bump diffuses and
  /// decays -- the parabolic smoothing of the Hodge-Laplacian, shown directly
  /// rather than through its spectrum.
  ///
  /// Mesh-agnostic: the boundary condition is carried entirely by which complex
  /// the flow runs on. The relative complex is the identity on a closed surface
  /// (sphere, Bob), where it is the free Neumann heat equation, and homogeneous
  /// essential (Dirichlet) on a mesh with boundary, holding the trace at zero
  /// (the interior bump is near zero there already). The same [`solve_heat`]
  /// serves both.
  ///
  /// [`solve_heat`]: formoniq::problems::heat::solve_heat
  pub fn heat(
    topology: Complex,
    coords: MeshCoords,
    grade: impl Into<ExteriorGrade>,
    nsteps: usize,
    final_time: f64,
  ) -> Self {
    let grade = grade.into();
    use formoniq::{problems::heat::solve_heat, whitney_complex::WhitneyComplex};

    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);
    let relative = whitney.relative();

    let initial = ambient_bump(&topology, &coords, grade);
    let source = Cochain::new(grade, na::DVector::zeros(whitney.ndofs(grade)));
    let dt = final_time / nsteps.max(1) as f64;
    let frames = solve_heat(&relative, grade, nsteps, dt, &initial, &source, 1.0);

    Self::trajectory_scene(topology, coords, initial, dt, frames)
  }

  /// The wave equation $diff_(t t) u = -Delta u$ of a localized initial bump at
  /// rest, as a single [`FieldTime::Trajectory`] field of grade `grade`. The bump splits
  /// and its fronts propagate, reflecting off any boundary -- the hyperbolic
  /// counterpart of [`Self::heat`], on the same initial data and the same
  /// mesh-agnostic footing (a closed mesh uses the identity inclusion).
  ///
  /// [`solve_wave`]: formoniq::problems::wave::solve_wave
  pub fn wave(
    topology: Complex,
    coords: MeshCoords,
    grade: impl Into<ExteriorGrade>,
    nsteps: usize,
    final_time: f64,
  ) -> Self {
    let grade = grade.into();
    use formoniq::{
      problems::wave::{WaveState, solve_wave},
      whitney_complex::WhitneyComplex,
    };

    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);

    let initial = ambient_bump(&topology, &coords, grade);
    let ndofs = whitney.ndofs(grade);
    let dt = final_time / nsteps.max(1) as f64;
    let times: Vec<f64> = (0..=nsteps).map(|k| k as f64 * dt).collect();
    let state = WaveState::new(initial.coeffs().clone(), na::DVector::zeros(ndofs));
    let force = Cochain::new(grade, na::DVector::zeros(ndofs));
    let frames = solve_wave(&whitney, grade, &times, state, force)
      .into_iter()
      .map(|s| Cochain::new(grade, s.pos))
      .collect();

    Self::trajectory_scene(topology, coords, initial, dt, frames)
  }

  /// Files a solved trajectory of any grade into a scene through the same `field`
  /// dispatch every other field goes through: the trajectory's first frame is
  /// its spatial representative, the sampled family its [`FieldTime`].
  fn trajectory_scene(
    topology: Complex,
    coords: MeshCoords,
    initial: Cochain,
    dt: f64,
    frames: Vec<Cochain>,
  ) -> Self {
    // Built before the fields are filed: the mark each one gets is chosen
    // against the surface's dimension, not the mesh's.
    let surface = Surface::of(&topology, &coords);
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    Self::field(
      &topology,
      &surface,
      FieldMeta {
        name: "trajectory".to_string(),
        time: FieldTime::Trajectory { dt, frames },
        dof: None,
      },
      initial,
      &mut fields,
      &mut line_fields,
    );
    Self {
      surface,
      topology,
      coords,
      fields,
      line_fields,
    }
  }

  /// The displayed field's temporal model, for the transport clock and the
  /// per-frame re-bake the caller drives.
  pub(crate) fn field_time(&self, selection: Selection) -> &FieldTime {
    match selection {
      Selection::Scalar(i) => &self.fields[i].time,
      Selection::Line(i) => &self.line_fields[i].time,
    }
  }

  /// The displayed field's spatial representative cochain -- the `base` a
  /// [`FieldTime::frame_at`] reads at each instant.
  pub(crate) fn field_cochain(&self, selection: Selection) -> &Cochain {
    match selection {
      Selection::Scalar(i) => &self.fields[i].cochain,
      Selection::Line(i) => &self.line_fields[i].cochain,
    }
  }

  /// Reconstructs a cochain as the render mark its *reduced grade*
  /// $min(k, n-k)$ calls for, and files it into `fields` or `line_fields`
  /// accordingly -- the one general entry point both a raw Whitney basis
  /// function ([`Self::whitney_basis`]) and a solved field arrive at.
  ///
  /// The Hodge star is what makes the dispatch total. A reduced grade of 0
  /// ($k = 0$ or $k = n$) is a scalar density, a reduced grade of 1 ($k = 1$ or
  /// $k = n-1$) a tangent line field, and a reduced grade $>= 2$ (only reachable
  /// at $n >= 4$) has no mark yet. The reduction is not applied here: the
  /// original cochain is stored whole, and the render mark reads it per cell at
  /// draw time (see [`surface_corner_values`]).
  ///
  /// **The grade reduces against the surface's dimension, not the mesh's**, and
  /// that is what makes the mark the mark of the thing on screen. A field on a
  /// solid is seen through its boundary, so the $n$ in $min(k, n-k)$ is
  /// $dim diff M$: a $2$-form on a $3$-manifold is a line field in the volume
  /// but the boundary's *top* form, hence a density, where it is actually
  /// drawn. Reducing against the parent would file it as arrows for a flux that
  /// has no direction on the surface carrying it.
  ///
  /// The exception is the grade that does not trace at all ($k = n$, where
  /// $C^k (diff M) = 0$): a volume density is not a surface quantity, so it
  /// reduces against the parent and is drawn by sampling the cells behind the
  /// boundary, until a volume mark exists to own it.
  fn field(
    topology: &Complex,
    surface: &Surface,
    meta: FieldMeta,
    cochain: Cochain,
    fields: &mut Vec<ScalarField>,
    line_fields: &mut Vec<LineField>,
  ) {
    let FieldMeta { name, time, dof } = meta;
    // A mode's sign is arbitrary, so it is pinned; a trajectory's is physical
    // (it solved from an initial condition), and its frames are what the
    // display reads, so flipping the representative alone would desync it.
    let cochain = if time.is_trajectory() {
      cochain
    } else {
      canonical_sign(cochain)
    };
    let k = cochain.grade();
    // The manifold the mark is *drawn on*. A grade that does not trace is a
    // volume quantity and keeps the parent's reduction (see the doc above).
    let n = if surface.traces(topology, k) {
      surface.dim(topology)
    } else {
      topology.dim()
    };

    // The reduction stars whenever $k > n-k$, and the star needs a global
    // volume form, which a non-orientable mesh does not have. The field is
    // then not drawable at all -- there is no orientation-independent density
    // or direction to show -- so it is refused here rather than rendered with
    // a per-cell sign that means nothing. Everything below the star is
    // unaffected and still files normally; the solver is unaffected either
    // way, since the gauge cancels inside the assembly.
    // Orientability is asked of the manifold whose volume form the star needs
    // -- the one `n` was just read from, so the check and the reduction cannot
    // disagree about which object they are talking about.
    if k > n - k && !surface.complex(topology).is_orientable() {
      eprintln!(
        "field '{name}' (grade {k} of {n}) needs the Hodge star to be drawn, \
         and the mesh is non-orientable: no global volume form, so it is skipped"
      );
      return;
    }

    match (k.min(n - k)).index() {
      0 => {
        // The original $k$-cochain is kept whole; the reduction to a density (a
        // pointwise Hodge star for $k = n$, the identity for $k = 0$) is read
        // per cell at draw time by [`surface_corner_values`], never averaged
        // into the stored field.
        fields.push(ScalarField {
          name,
          grade: k,
          cochain,
          time,
          dof,
        });
      }
      1 => {
        line_fields.push(LineField {
          name,
          grade: k,
          cochain,
          time,
          dof,
        });
      }
      _reduced => {
        // A reduced grade >= 2 (only reachable at n >= 4) has no render mark
        // yet -- files into no list rather than panicking the viewer.
      }
    }
  }
}

/// The reduced form at a point, in the reference frame of its cell: the Whitney
/// value $W c$ if its grade is already $<= n-k$, else its Hodge star, so the
/// result always has grade $min(k, n-k)$. The star is where -- and the only
/// place -- a metric enters the reduction.
///
/// `sign` is the cell's coherent orientation
/// ([`Orientation::sign`](simplicial::topology::orientation::Orientation::sign)),
/// and it is the *second* thing the star needs beyond the metric. A cell's
/// stored colex vertex order fixes a volume form only up to sign, so
/// $star: Lambda^n -> Lambda^0$ read cell by cell returns the density against
/// each cell's own arbitrary frame -- $plus.minus$ the true one, flipping
/// wherever colex disagrees with the manifold's orientation. Multiplying by
/// `sign` is what makes the reduced value comparable across cells, and hence
/// what makes a top-grade density or an $(n-1)$-form's direction mean anything
/// globally. Below the star the sign is irrelevant, which is why it costs
/// nothing to pass it always: [`reduction_sign`] returns `Pos` there.
pub(crate) fn reduced_form(form: MultiForm, metric: &Metric, sign: Sign) -> MultiForm {
  let n = form.dim();
  let k = form.grade();
  if k <= n - k {
    form
  } else {
    form.hodge_star(metric) * sign.as_f64()
  }
}

/// The scalar a form reduces to, for every mark that consumes one.
///
/// The one rule, total over grade and dimension: a $0$-form *is* a scalar and is
/// read signed and metric-free; the manifold's top form is a pseudoscalar and
/// becomes a scalar through $star$; everything else reduces by its magnitude
/// $|omega|_g$, the direction being the line-field mark's to carry.
///
/// `signed` is `Some` exactly when the form is the manifold's own top form *and*
/// a coherent orientation fixes its volume form, so holding one is the proof
/// invariant 6 demands: only then is a signed density comparable across cells.
/// The caller states that condition, because only the caller knows whether the
/// form's own dimension is the manifold's (the trace onto a face is top on the
/// face while carrying no global sign). `None` is the honest magnitude.
pub(crate) fn scalarize(form: MultiForm, metric: &Metric, signed: Option<Sign>) -> f64 {
  if form.grade() == 0 {
    return form.coeffs()[0];
  }
  match signed {
    Some(sign) => form.hodge_star(metric).coeffs()[0] * sign.as_f64(),
    None => form.norm(metric),
  }
}

/// Which natural operator a field is read through before it is reduced to a
/// scalar.
///
/// The scalar every scalar-consuming mark draws is `scalarize(F omega)`, and
/// this is $F$. Each variant is total over grade and dimension, degenerating
/// rather than being excluded: $dif omega = 0$ at $k = n$, and [`scalarize`]
/// takes the resulting top or bottom grade uniformly. The axis is deliberately
/// separate from the reduction -- the operator is metric-free, the reduction is
/// where the metric enters.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum Scalarization {
  /// $omega$ itself: where the field is, and the only reading that needs no
  /// derivative.
  #[default]
  Value,
  /// $dif omega$: where the field *varies*. At $k = n - 1$ this is the source
  /// density, a top form the reduction stars into a signed scalar -- so the
  /// classical divergence is not a case here but the same composite at one
  /// grade.
  Differential,
}

impl Scalarization {
  /// The cochain this reading actually draws. Metric-free: the coboundary is
  /// the simplicial exterior derivative, so nothing here consults a geometry.
  pub(crate) fn apply<'a>(self, cochain: &'a Cochain, topology: &Complex) -> Cow<'a, Cochain> {
    match self {
      Self::Value => Cow::Borrowed(cochain),
      Self::Differential => Cow::Owned(cochain.dif(topology)),
    }
  }

  pub(crate) const ALL: [Self; 2] = [Self::Value, Self::Differential];

  pub(crate) fn label(self) -> &'static str {
    match self {
      Self::Value => "value",
      Self::Differential => "differential",
    }
  }

  pub(crate) fn hover(self) -> &'static str {
    match self {
      Self::Value => "The field itself, |omega|: where it is",
      Self::Differential => "The exterior derivative, |d omega|: where it varies. Metric-free",
    }
  }
}

/// The orientation factor [`reduced_form`] needs on one cell: `Pos` when the
/// reduction is the identity (no star, so no volume form and no orientation),
/// otherwise the cell's coherent orientation.
///
/// Panics on a non-orientable complex, and that is sound rather than a lurking
/// edge case: a field whose reduction needs the star is *not filed* into the
/// scene at all unless the mesh is orientable (see [`Scene::field`]), so
/// holding a field that reaches here is already the proof that the orientation
/// exists. The refusal happens once, where the field is built, instead of at
/// every draw.
pub(crate) fn reduction_sign(topology: &Complex, cell: Cell, grade: ExteriorGrade) -> Sign {
  let n = topology.dim();
  if grade <= n - grade {
    return Sign::Pos;
  }
  topology
    .orientation()
    .expect("a starred field is only filed on an orientable mesh")
    .sign(cell)
}

/// The surface colormap scalar at every rendered triangle corner -- three per
/// triangle, in [`CellCorner`] order -- as the [`trace_value`] of the field on
/// the triangle's own 2-simplex (the fill is the 2-skeleton).
///
/// The trace is single-valued across the cells incident at the face by tangential
/// conformity, so no per-corner cell disambiguation and no averaging is needed:
/// a face the form vanishes on colors to zero because its trace *is* zero, and a
/// grade above 2 traces to zero on every face, leaving the fill black (its home
/// is volumetric, deferred). At $n = 2$ the face is the cell and this reproduces
/// the reduced-grade density exactly; at $n = 3$ it reads the face's own trace,
/// not a value borrowed from an incident tet.
pub(crate) fn surface_corner_values(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  cell_corners: &[CellCorner],
) -> Vec<f64> {
  let n = topology.dim();
  let mut values = Vec::with_capacity(3 * cell_corners.len());
  for cc in cell_corners {
    let cell = SimplexIdx::new(n, cc.cell).handle(topology);
    let mut positions = cc.local;
    positions.sort_unstable();
    let vertices = &cell.simplex().vertices;
    let face_simplex = Simplex::new(positions.iter().map(|&p| vertices[p]).collect());
    let face = topology.skeleton(2).handle_by_simplex(&face_simplex);
    for &ilocal in &cc.local {
      let corner = positions.iter().position(|&p| p == ilocal).unwrap();
      let mut weights = Vector::zeros(3);
      weights[corner] = 1.0;
      values.push(trace_value(
        topology,
        coords,
        cochain,
        face,
        &Bary::new(weights),
      ));
    }
  }
  values
}

/// The 1-skeleton colormap value at each segment's two endpoints, as two
/// parallel arrays (`[i]` is segment `i`'s two ends) -- the [`trace_value`] of
/// the field on each edge.
///
/// The $k = 1$ counterpart of [`surface_corner_values`]'s $k = 2$: the same
/// trace, on a different skeleton, at a different render primitive. Per edge
/// rather than per vertex because a grade-1 density differs between edges
/// sharing a vertex, and single-valued by conformity so no averaging enters. A
/// grade above 1 traces to zero on every edge (the reduction returns 0), so the
/// 1-skeleton of a flux field is uncolored, honestly.
pub(crate) fn segment_colors(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  segments: &[[u32; 2]],
) -> [Vec<f64>; 2] {
  let mut ends = [
    Vec::with_capacity(segments.len()),
    Vec::with_capacity(segments.len()),
  ];
  for &vpair in segments {
    let mut vs = [vpair[0] as usize, vpair[1] as usize];
    vs.sort_unstable();
    let edge = topology
      .skeleton(1)
      .handle_by_simplex(&Simplex::new(vs.to_vec()));
    for (end, &v) in vpair.iter().enumerate() {
      let corner = vs.iter().position(|&p| p == v as usize).unwrap();
      let mut weights = Vector::zeros(2);
      weights[corner] = 1.0;
      ends[end].push(trace_value(
        topology,
        coords,
        cochain,
        edge,
        &Bary::new(weights),
      ));
    }
  }
  ends
}

/// The 0-skeleton colormap value at every mesh vertex -- the [`trace_value`] of
/// the field on each 0-simplex.
///
/// The $k = 0$ member of the same family as [`segment_colors`] and
/// [`surface_corner_values`]. A vertex is the one skeleton simplex a field is
/// always single-valued on with no reduction: a 0-form reads its own value
/// there, and any higher grade traces to zero (a $k$-form has no restriction to
/// a point). Per vertex, not per incident cell -- the 0-form is continuous, so
/// there is nothing to average.
pub(crate) fn point_colors(topology: &Complex, coords: &MeshCoords, cochain: &Cochain) -> Vec<f64> {
  let bary = Bary::new(Vector::from_element(1, 1.0));
  topology
    .skeleton(0)
    .handle_iter()
    .map(|vertex| trace_value(topology, coords, cochain, vertex, &bary))
    .collect()
}

/// A cochain in canonical sign: the coefficient of largest magnitude is made
/// positive, ties broken by colex rank.
///
/// An eigenvector is defined only up to a scalar, and a solver may return
/// either sign on a whim -- so the same mode can come out red where it was blue
/// between two runs of the same scene, for reasons that are not the
/// mathematics. Pinning it is what makes a rendered field reproducible, and it
/// dominates the orientation gauge (which is already deterministic): the
/// field's own sign sits in front of the density either way.
///
/// A gauge fix, not a normalization: the magnitude is untouched, so the
/// colormap range still spans what the field actually is. The zero cochain is
/// its own canonical form, having no largest coefficient to orient by.
///
/// Applies to modes only. A trajectory's sign is *physical* -- it solved from
/// an initial condition -- and flipping it would be a lie about the solve, so
/// the caller excludes it.
fn canonical_sign(cochain: Cochain) -> Cochain {
  let pivot = cochain
    .coeffs()
    .iter()
    .copied()
    .enumerate()
    .max_by(|(_, a), (_, b)| {
      a.abs()
        .partial_cmp(&b.abs())
        .unwrap_or(std::cmp::Ordering::Equal)
    });
  match pivot {
    Some((_, c)) if c < 0.0 => Cochain::new(cochain.grade(), -cochain.coeffs()),
    _ => cochain,
  }
}

/// The surface's displacement height per rendered corner, by the strategy the
/// field's own continuity calls for -- the same reduction that picks the mark,
/// asked once more.
///
/// $cal(W) Lambda^0$ is $P_1$ and continuous, so a vertex has one value and the
/// nodal recovery below *is* the field: the surface displaces as one connected
/// sheet, exactly. $cal(W) Lambda^n$ is $P_0$: the reduced density is constant
/// on each cell and genuinely discontinuous across it, so there is no
/// continuous height to displace by, and the nodal average would invent one --
/// showing a $P_0$ field flat-shaded in color and smooth in shape, two
/// contradictory claims about one field in one frame. Instead each cell
/// displaces *rigidly*, by its own constant value.
///
/// **A rigidly displaced surface tears, and that is the point.** The cells
/// separate by exactly the jump in the density across their shared face, so the
/// discontinuity becomes visible space rather than being smoothed away, and the
/// surface visibly re-closes under refinement as the jump vanishes. It is the
/// displacement counterpart of reading the colormap per corner.
///
/// The direction stays the *vertex* normal, so a cell translates rather than
/// moving exactly along its own normal. On a resolved mesh the two differ by
/// the normal's variation across one cell. What this costs is stated in
/// [`reduced_form`]'s terms: $d_K n_K$ with the orientation-induced cell normal
/// would be invariant under the orientation gauge outright, whereas the
/// embedding's outward normal fixes that gauge only up to one *global* sign --
/// the same ambiguity an eigenvector already carries, and not the per-cell
/// scrambling that made the star wrong.
pub(crate) fn surface_corner_heights(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  cell_corners: &[CellCorner],
) -> Vec<f64> {
  let n = topology.dim();
  let k = cochain.grade();
  if k > n - k {
    // Discontinuous: the per-corner read is already constant on each cell, so
    // the honest colormap value and the rigid height are the same number.
    return surface_corner_values(topology, coords, cochain, cell_corners);
  }
  let nodal = nodal_heights(topology, coords, cochain);
  cell_corners
    .iter()
    .flat_map(|cc| {
      let vertices = SimplexIdx::new(n, cc.cell)
        .handle(topology)
        .simplex()
        .vertices
        .clone();
      cc.local.map(|ilocal| nodal[vertices[ilocal]])
    })
    .collect()
}

/// The per-vertex displacement height: the reduced field's nodal average over
/// the cells incident at each vertex.
///
/// Exact for a continuous field ($cal(W) Lambda^0$), where the incident cells
/// already agree and this is the identity on the DOFs; a smoothing recovery
/// wherever the reduction stars, which is why the surface does not use it there
/// (see [`surface_corner_heights`]). It stays the height of the *segment* marks
/// at every grade: the 1-skeleton is shared between cells and cannot tear
/// without duplicating it, so the wireframe rides the continuous recovery and
/// reads as the reference the fill's torn cells sit around.
pub(crate) fn nodal_heights(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
) -> Vec<f64> {
  let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);
  let nvertices = topology.skeleton_raw(0).len();
  let mut sum = vec![0.0; nvertices];
  let mut count = vec![0u32; nvertices];
  for cell in topology.cells().handle_iter() {
    let metric = coords.cell_metric(cell);
    for (ilocal, &v) in cell.simplex().vertices.iter().enumerate() {
      sum[v] += reduced_value(&interpolant, cell, &metric, ilocal);
      count[v] += 1;
    }
  }
  sum
    .into_iter()
    .zip(count)
    .map(|(s, c)| if c > 0 { s / f64::from(c) } else { 0.0 })
    .collect()
}

/// The reduced field's scalar readout at one cell vertex, in that cell's chart:
/// the signed density for a reduced grade of 0, the magnitude $|V|_g$ for a
/// reduced grade of 1 (the direction is the glyph and particle marks' to carry).
/// The one place the reduction is evaluated, shared by the per-corner colormap
/// and the per-vertex height so the two cannot drift.
fn reduced_value(
  interpolant: &WhitneyInterpolant,
  cell: Cell,
  metric: &Metric,
  ilocal: usize,
) -> f64 {
  let mut weights = na::DVector::zeros(cell.nvertices());
  weights[ilocal] = 1.0;
  let point = MeshPoint::new(cell.idx(), weights.into());
  let k = interpolant.cochain().grade();
  let signed = (k == cell.complex().dim()).then(|| reduction_sign(cell.complex(), cell, k));
  scalarize(interpolant.eval(&point), metric, signed)
}

/// The trace-reduced scalar of a field on a skeleton simplex: the one rule that
/// colors every $k$-skeleton alike. Pull the Whitney field back onto the simplex
/// ([`Cochain::trace`]) and reduce the traced form to a scalar with the
/// simplex's own metric.
///
/// The trace is exact by tangential ($H(dif)$) conformity, so it is
/// single-valued across the cells incident at a shared simplex -- no averaging,
/// no tearing. The trace of a grade-$k$ form onto a $d$-simplex is a $k$-form on
/// it, and $Lambda^k(tau) = 0$ for $d < k$: a form colors a skeleton *below* its
/// grade with an honest zero. On the diagonal $d = k$ the trace is the constant
/// top-form of density $c_tau \/ vol_g(tau)$, flat-shading the simplex by its
/// cochain density; above it ($d > k$) the trace varies and the norm reads the
/// magnitude.
///
/// The scalar is *signed* only where the sign is intrinsic, and its *magnitude*
/// otherwise -- because a $k$-cochain value ($k >= 1$) is defined relative to the
/// simplex's orientation, which here is the colex bookkeeping convention, so a
/// signed color would paint that artifact on the screen. Two cases escape it:
/// $k = 0$, where a vertex has trivial orientation and the value is a genuine
/// scalar; and $k = d = n$, the manifold's own top form, where the coherent
/// [`Complex::orientation`] fixes the global density -- consulted here exactly as
/// invariant 6 demands, and refused on a non-orientable mesh. Nothing fixes the
/// sign for $0 < k < n$: a manifold orientation induces *opposite*
/// co-orientations on an interior facet ($diff compose diff = 0$), so it cannot
/// reach the sub-top skeletons, and the honest reading there is the magnitude.
/// The direction a magnitude drops is not lost -- it lives in the line-field
/// mark, as a genuine vector.
pub(crate) fn trace_value(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  simplex: SimplexRef,
  bary: &Bary,
) -> f64 {
  let n = topology.dim();
  let d = simplex.dim();
  let k = cochain.grade();
  if k > d {
    return 0.0;
  }
  let sub = Complex::standard(d);
  let interpolant = WhitneyInterpolant::new(cochain.trace(simplex), &sub);
  let cell = sub.cells().handle_iter().next().unwrap();
  let form = interpolant.eval(&MeshPoint::new(cell.idx(), bary.clone()));
  // A top form is the manifold's own only on a cell ($d = n$); on a face it is
  // top for the face while no coherent orientation reaches it, so it reduces by
  // magnitude like every other grade.
  let signed = (k == n && d == n).then(|| reduction_sign(topology, simplex.role(), k));
  scalarize(form, &coords.simplex_metric(simplex), signed)
}

/// The three $L^2$-orthogonal shells of the discrete Hodge decomposition of a
/// $k$-cochain, $omega = "exact" + "coexact" + "harmonic"$, each a $k$-cochain
/// on the same complex.
pub(crate) struct HodgeParts {
  /// $dif alpha$, $alpha in cal(W) Lambda^(k-1)$: the exact (range-of-$dif$)
  /// component.
  pub(crate) exact: Cochain,
  /// $delta beta$, $beta in cal(W) Lambda^(k+1)$: the coexact
  /// (range-of-$delta$) component.
  pub(crate) coexact: Cochain,
  /// $h in cal(H)^k$: the harmonic component, the $L^2$-projection of $omega$
  /// onto the $b_k$-dimensional harmonic space.
  pub(crate) harmonic: Cochain,
}

/// The discrete Hodge decomposition of a $k$-cochain through the mixed
/// Hodge-Laplace source problem $Delta u = omega$ (absolute boundary
/// conditions, the full [`WhitneyComplex`]).
///
/// The mixed solve returns $(sigma, u, p)$ with $sigma = delta u$ weakly and
/// $p$ the harmonic projection of the load in harmonic-basis coordinates. Its
/// $u$-block reads $M(dif sigma + delta dif u + H p) = M omega$, so at the
/// coefficient level
/// $omega = underbrace(dif sigma, "exact") + underbrace(delta dif u, "coexact") + underbrace(H p, "harmonic")$
/// *exactly* -- the three shells sum back to $omega$ with no residual, and the
/// coexact shell is recovered as the remainder rather than by forming $delta$
/// explicitly. Their pairwise $L^2$-orthogonality is the content of the mixed
/// formulation.
///
/// [`WhitneyComplex`]: formoniq::whitney_complex::WhitneyComplex
pub(crate) fn hodge_decompose(
  topology: &Complex,
  coords: &MeshCoords,
  input: &Cochain,
) -> Result<HodgeParts, formoniq::linalg::eigen::EigenError> {
  use formoniq::{
    problems::elliptic::{solve_harmonics, solve_source},
    whitney_complex::WhitneyComplex,
  };
  use simplicial::linalg::CsrMatrix;

  let grade = input.grade();
  let metric = coords.to_edge_lengths_sq(topology);
  let complex = WhitneyComplex::new(topology, &metric);

  // The load vector of the source problem is the Riesz representation of the
  // functional $angle.l omega, dot.c angle.r$, i.e. $M_k omega$.
  let mass = CsrMatrix::from(&complex.mass(grade));
  let source_galvec = &mass * input.coeffs();

  let (sigma, _u, p) = solve_source(&complex, source_galvec, grade)?;
  let harmonics = solve_harmonics(&complex, grade)?;

  // exact $= dif sigma$. At grade 0 the $sigma in Lambda^(-1)$ space is empty,
  // so the exact shell is identically zero.
  let exact = if grade > 0 {
    let dif = CsrMatrix::from(&topology.coboundary_operator(grade - 1));
    &dif * sigma.coeffs()
  } else {
    Vector::zeros(input.coeffs().len())
  };
  // harmonic $= H p$, lifting the harmonic-basis coordinates back into
  // $cal(W) Lambda^k$. Zero-width when $b_k = 0$, so this is the zero cochain
  // on a contractible mesh.
  let harmonic = &harmonics * p.coeffs();
  // coexact as the exact-arithmetic remainder (see the doc comment).
  let coexact = input.coeffs() - &exact - &harmonic;

  Ok(HodgeParts {
    exact: Cochain::new(grade, exact),
    coexact: Cochain::new(grade, coexact),
    harmonic: Cochain::new(grade, harmonic),
  })
}

/// A deterministic, mesh-independent probe form for the Hodge decomposition:
/// the ambient 1-form $omega = -y dif x + x dif y + z dif z$ pulled onto the
/// mesh through the `derham` bridge, then de Rham mapped to a `grade`-cochain.
///
/// The swirl $-y dif x + x dif y$ is not closed (a coexact part) and threads
/// any handle enclosing the $z$-axis (a harmonic part); the $z dif z = dif(z^2\/2)$
/// contributes a manifestly exact part. Its pullback therefore lights up all
/// three shells on a genus-1 surface, and degrades gracefully (harmonic part
/// zero) on a contractible one.
///
/// The probe field the decomposition study actually splits: the ambient swirl
/// [`hodge_probe_form`] plus, on a mesh with grade-1 homology, an explicit copy
/// of a harmonic 1-form scaled to the swirl's magnitude.
///
/// The swirl alone supplies rich exact and coexact shells, but its periods
/// around the handles can vanish -- they do on the Császár torus, where a purely
/// ambient probe leaves the harmonic shell at numerical zero. Whether an ambient
/// field excites a cycle depends on how the handle happens to sit in space,
/// which is no basis for a teaching example. Injecting a harmonic generator
/// makes the field genuinely carry a topological cycle on *any* genus-$g$
/// surface, so the decomposition demonstrates all three shells regardless of
/// embedding -- and injecting the harmonic part is itself the point: the
/// decomposition returns it untouched, orthogonal to the two it did not put
/// there. On a contractible mesh the harmonic space is empty and nothing is
/// added.
pub(crate) fn hodge_probe_input(topology: &Complex, coords: &MeshCoords) -> Cochain {
  use formoniq::{problems::elliptic::solve_harmonics, whitney_complex::WhitneyComplex};
  use simplicial::linalg::CsrMatrix;

  let swirl = hodge_probe_form(topology, coords);
  let metric = coords.to_edge_lengths_sq(topology);
  let complex = WhitneyComplex::new(topology, &metric);
  let mass = CsrMatrix::from(&complex.mass(1));
  let m_norm = |v: &Vector| (&mass * v).dot(v).max(0.0).sqrt();

  match solve_harmonics(&complex, 1) {
    Ok(harmonics) if harmonics.ncols() > 0 => {
      let h0 = harmonics.column(0).clone_owned();
      let (swirl_norm, h0_norm) = (m_norm(swirl.coeffs()), m_norm(&h0));
      // Scale the injected cycle to the swirl's magnitude so neither swamps the
      // other; guard the degenerate zero-norm harmonic vector.
      let scale = if h0_norm > 1e-12 {
        swirl_norm / h0_norm
      } else {
        0.0
      };
      Cochain::new(1, swirl.coeffs() + scale * h0)
    }
    _ => swirl,
  }
}

/// The smooth part of the decomposition probe: the ambient 1-form
/// $omega = -y dif x + x dif y + z dif z$ pulled onto the mesh through the `derham`
/// bridge and de Rham mapped to a 1-cochain.
///
/// The swirl $-y dif x + x dif y$ is not closed, so it carries both an exact and
/// a coexact part; the $z dif z = dif(z^2\/2)$ makes the exact part manifestly
/// nonzero. It does *not* reliably carry a harmonic part -- whether an ambient
/// field has nonzero periods around a surface's handles depends on how those
/// handles sit in space, and on the Császár torus, for one, they vanish. The
/// harmonic shell is supplied separately by [`hodge_probe_input`].
fn hodge_probe_form(topology: &Complex, coords: &MeshCoords) -> Cochain {
  let n = coords.dim().index();
  let field = DiffFormClosure::one_form(
    move |p| {
      let x = p.vector();
      let mut omega = Vector::zeros(n);
      if n >= 2 {
        omega[0] = -x[1];
        omega[1] = x[0];
      }
      if n >= 3 {
        omega[2] = x[2];
      }
      omega
    },
    n,
  );
  let pulled = field.pullback_on(topology, coords);
  derham_map(&pulled, topology, 2)
}

/// A localized grade-$k$ initial condition for a time-dependent solve, defined
/// off the mesh's own coordinates: a Gaussian in ambient distance centered on
/// the vertex *nearest* the centroid, of width a fixed fraction of the
/// coordinate extent, times the first basis blade of grade `grade`. Pulled onto
/// the mesh through the `derham` bridge and de Rham mapped to a `grade`-cochain,
/// so it lands on any embedded mesh without assuming a shape.
///
/// Which blade carries the bump is a gauge of the ambient frame, not of the
/// mathematics: any nonzero constant $k$-covector gives the same construction,
/// and grade 0 (the empty blade) recovers the scalar bump exactly.
///
/// The nearest-to-centroid vertex, not the farthest: on a mesh with boundary
/// (the flat grid) the farthest vertex is a boundary corner, where a held
/// boundary would pin the bump instead of letting it diffuse; the nearest one is
/// interior, so its boundary trace is near zero and the flow is free. On a closed
/// mesh every vertex is on the surface, so the nearest merely also works where a
/// boundary exists.
fn ambient_bump(topology: &Complex, coords: &MeshCoords, grade: ExteriorGrade) -> Cochain {
  let n = coords.dim().index();
  let nvertices = coords.nvertices().max(1) as f64;
  let centroid = coords
    .coord_iter()
    .fold(Vector::zeros(n), |acc, c| acc + *c)
    / nvertices;

  let extent = coords
    .coord_iter()
    .map(|c| (*c - &centroid).norm())
    .fold(0.0, f64::max);
  let center = coords
    .coord_iter()
    .min_by(|a, b| {
      (**a - &centroid)
        .norm()
        .total_cmp(&(**b - &centroid).norm())
    })
    .map_or_else(|| centroid.clone(), |c| (*c).into_owned());
  let sigma = 0.25 * extent.max(1e-6);

  let blade = MultiForm::from_blade_signed(n, Sign::Pos, Blade::from_rank(grade.index(), 0));
  let field = DiffFormClosure::new(
    move |p| {
      let r2 = (p.vector() - &center).norm_squared();
      blade.clone() * (-r2 / (2.0 * sigma * sigma)).exp()
    },
    n,
    grade,
  );
  let pulled = field.pullback_on(topology, coords);
  derham_map(&pulled, topology, 2)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The medium is offered exactly where there is an interior to march, and
  /// that is an *intrinsic-dimension* question rather than a grade one: at
  /// $n <= 2$ the boundary primitive already draws the whole manifold, and at
  /// $n >= 3$ it cannot. Swept over both, and over every grade at each, since
  /// the answer must not depend on which field is selected.
  #[test]
  fn the_medium_is_offered_exactly_on_a_solid() {
    for dim in 1..=3 {
      let scene = Scene::whitney_basis(dim);
      let selections = (0..scene.fields.len())
        .map(Selection::Scalar)
        .chain((0..scene.line_fields.len()).map(Selection::Line));
      let mut asked = 0;
      for selection in selections {
        assert_eq!(
          scene.offers(selection).volume,
          dim >= 3,
          "dimension {dim} offered the wrong medium"
        );
        asked += 1;
      }
      assert!(asked > 0, "dimension {dim} produced no field to ask about");
    }
  }

  /// The magnitude branch of [`scalarize`] is Hodge-invariant:
  /// $|omega|_g = |star omega|_g$, the star being an isometry on a Riemannian
  /// metric. So a field and its reduction ([`reduced_form`]) read the *same*
  /// scalar, and which side of $k <-> n-k$ a mark happens to hold cannot change
  /// the color on screen. Swept over every dimension and every strictly
  /// intermediate grade, which is exactly where the magnitude branch applies.
  #[test]
  fn scalarize_is_hodge_invariant_off_the_extremal_grades() {
    for dim in 2..=4 {
      let metric = Metric::standard(dim);
      for grade in 1..dim {
        let ncoeffs = MultiForm::zero(dim, grade).coeffs().len();
        for i in 0..ncoeffs {
          let mut coeffs = na::DVector::zeros(ncoeffs);
          coeffs[i] = 2.0;
          let form = MultiForm::new(coeffs, dim, grade);
          let starred = form.clone().hodge_star(&metric);
          let (direct, reduced) = (
            scalarize(form, &metric, None),
            scalarize(starred, &metric, None),
          );
          assert!((direct - reduced).abs() < 1e-12, "{direct} != {reduced}");
        }
      }
    }
  }

  /// The extremal grades are the signed ones, and they are signed for different
  /// reasons: a $0$-form *is* a scalar (metric-free, no orientation involved),
  /// while an $n$-form is a pseudoscalar whose sign is the coherent
  /// orientation's. Flipping that orientation negates the readout and nothing
  /// else, which is precisely invariant 6's gauge acting on the picture.
  #[test]
  fn scalarize_is_signed_at_the_extremal_grades() {
    for dim in 1..=4 {
      let metric = Metric::standard(dim);
      let zero_form = MultiForm::new(na::dvector![-1.0], dim, 0);
      assert!((scalarize(zero_form, &metric, None) + 1.0).abs() < 1e-12);

      let top = MultiForm::new(na::dvector![1.0], dim, dim);
      let pos = scalarize(top.clone(), &metric, Some(Sign::Pos));
      let neg = scalarize(top, &metric, Some(Sign::Neg));
      assert!((pos + neg).abs() < 1e-12);
      assert!((pos.abs() - 1.0).abs() < 1e-12);
    }
  }

  /// On the diagonal $d = k$ the trace-colored value is the cochain density
  /// $c_tau \/ vol_g(tau)$, and constant across the simplex however the point is
  /// chosen -- the flat-shaded DOF the lowest-order element forces. Single-valued
  /// with no averaging: the trace onto a $k$-simplex reads only that simplex's
  /// own DOF.
  #[test]
  fn trace_diagonal_is_cochain_density() {
    use simplicial::geometry::{cell_volume, coord::mesh::standard_coord_complex};
    for n in 1..=3 {
      let (topology, coords) = standard_coord_complex(n);
      for k in 1..=n {
        let ndofs = topology.nsimplices(k);
        let cochain = Cochain::new(
          k,
          Vector::from_iterator(ndofs, (0..ndofs).map(|i| (i + 1) as f64)),
        );
        for tau in topology.skeleton(k).handle_iter() {
          // Magnitude of the density; its sign, where it has one, is governed by
          // orientation, not the point on the simplex, which is what this pins.
          let expected = (cochain[tau] / cell_volume(&coords.simplex_metric(tau))).abs();
          for shift in [0.0, 0.13] {
            let mut w = Vector::from_element(k + 1, (1.0 - shift) / (k + 1) as f64);
            w[0] += shift;
            let value = trace_value(&topology, &coords, &cochain, tau, &Bary::new(w));
            assert!((value.abs() - expected).abs() < 1e-9, "n={n} k={k}");
          }
        }
      }
    }
  }

  /// A top-grade field displaces its cells *rigidly*: the height is constant
  /// within each cell, which is what makes the fill tear along the field's own
  /// discontinuity instead of smoothing across it. Constant to zero spread,
  /// not approximately -- the Whitney top form is genuinely $P_0$.
  #[test]
  fn top_grade_displacement_is_constant_within_each_cell() {
    let scene = Scene::spherical_harmonics(1, 2);
    let baked = crate::bake::BakedMesh::new(&scene.topology, &scene.coords);
    let top = scene
      .fields
      .iter()
      .find(|f| f.grade == scene.topology.dim())
      .expect("a top-grade scalar field");
    let heights = surface_corner_heights(
      &scene.topology,
      &scene.coords,
      &top.cochain,
      &baked.cell_corners,
    );
    for corner in heights.chunks(3) {
      assert_eq!(
        corner[0], corner[1],
        "a cell's corners must share one rigid height"
      );
      assert_eq!(corner[0], corner[2]);
    }
  }

  /// A grade-0 field stays continuous: the corners a shared mesh vertex
  /// contributes to agree, so the surface displaces as one sheet and does not
  /// tear. The other half of the dispatch, and the reason it is a dispatch
  /// rather than a switch to rigid displacement everywhere.
  #[test]
  fn grade_zero_displacement_agrees_at_a_shared_vertex() {
    let scene = Scene::spherical_harmonics(1, 2);
    let baked = crate::bake::BakedMesh::new(&scene.topology, &scene.coords);
    let scalar = scene
      .fields
      .iter()
      .find(|f| f.grade == 0)
      .expect("a grade-0 field");
    let heights = surface_corner_heights(
      &scene.topology,
      &scene.coords,
      &scalar.cochain,
      &baked.cell_corners,
    );
    let mut seen: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for (cc, corner) in baked.cell_corners.iter().zip(heights.chunks(3)) {
      let vertices = SimplexIdx::new(scene.topology.dim(), cc.cell)
        .handle(&scene.topology)
        .simplex()
        .vertices
        .clone();
      for (slot, &ilocal) in cc.local.iter().enumerate() {
        let previous = seen.insert(vertices[ilocal], corner[slot]);
        if let Some(previous) = previous {
          assert!((previous - corner[slot]).abs() < 1e-12);
        }
      }
    }
  }

  /// The sign gauge is pinned, so a solver returning the opposite eigenvector
  /// renders the identical picture: the largest-magnitude coefficient comes out
  /// positive either way, and the magnitudes are untouched.
  #[test]
  fn canonical_sign_is_a_gauge_fix() {
    let c = Cochain::new(0, na::DVector::from_vec(vec![0.5, -2.0, 1.0]));
    let flipped = Cochain::new(0, -c.coeffs());
    assert_eq!(
      canonical_sign(c.clone()).coeffs(),
      canonical_sign(flipped).coeffs()
    );
    // The pivot is made positive, and nothing is rescaled.
    let fixed = canonical_sign(c);
    assert_eq!(fixed.coeffs()[1], 2.0);
    assert_eq!(fixed.coeffs()[0], -0.5);
    // The zero cochain is its own canonical form.
    let zero = Cochain::new(0, na::DVector::zeros(3));
    assert_eq!(canonical_sign(zero.clone()).coeffs(), zero.coeffs());
  }

  /// The harmonic top-grade form on a closed orientable surface is a multiple
  /// of the volume form, $h = c dvol$, so its reduction $star h = c$ is
  /// *constant* over the whole manifold. That makes the reduced readout of the
  /// $lambda = 0$ grade-2 mode on the sphere a law with an exact answer, and
  /// the sharpest available statement that the Hodge star is being taken
  /// against one global volume form rather than each cell's own.
  ///
  /// It is precisely the test the colex vertex order fails without a coherent
  /// orientation: the density comes out $plus.minus c$ cell by cell, and the
  /// nodal average of that collapses toward zero instead of reproducing $c$.
  #[test]
  fn harmonic_top_form_reduces_to_a_constant_density() {
    use formoniq::{problems::elliptic::solve_evp, whitney_complex::WhitneyComplex};

    let (topology, coords) = simplicial::mesher::sphere::mesh_sphere_surface(2);
    let lengths = coords.to_edge_lengths_sq(&topology);
    let (eigenvals, _, eigenfuncs) =
      solve_evp(&WhitneyComplex::new(&topology, &lengths), 2, 1).unwrap();
    // $b_2 = 1$ on the sphere: the lowest grade-2 mode is the harmonic one.
    assert!(eigenvals[0].abs() < 1e-8, "expected the harmonic mode");

    let cochain = Cochain::new(2, eigenfuncs.column(0).into_owned());
    let heights = nodal_heights(&topology, &coords, &cochain);
    let mean = heights.iter().sum::<f64>() / heights.len() as f64;
    assert!(mean.abs() > 1e-3, "the density must not cancel to zero");
    for h in heights {
      assert!(
        (h - mean).abs() / mean.abs() < 1e-10,
        "reduced harmonic density {h} is not the constant {mean}"
      );
    }
  }

  /// A trajectory's frame at an instant is the linear interpolation of its
  /// bracketing samples, clamped to the sampled interval: the interpolation is
  /// linear in the cochain coefficients, so it is exact at the samples and
  /// affine between them, and its duration is $dif t (N - 1)$.
  #[test]
  fn frame_at_interpolates_between_samples() {
    let frames = vec![
      Cochain::new(0, na::DVector::from_vec(vec![0.0, 0.0])),
      Cochain::new(0, na::DVector::from_vec(vec![2.0, -2.0])),
      Cochain::new(0, na::DVector::from_vec(vec![4.0, -4.0])),
    ];
    let time = FieldTime::Trajectory { dt: 0.5, frames };
    let base = Cochain::new(0, na::DVector::zeros(2));

    assert_eq!(time.duration(), Some(1.0));
    // Exact at the two endpoints of the sampled interval.
    assert_eq!(time.frame_at(&base, 0.0).coeffs()[0], 0.0);
    assert_eq!(time.frame_at(&base, 1.0).coeffs()[0], 4.0);
    // Affine at the quarter point of the first (dt = 0.5) interval: halfway.
    let mid = time.frame_at(&base, 0.25);
    assert!((mid.coeffs()[0] - 1.0).abs() < 1e-12);
    assert!((mid.coeffs()[1] + 1.0).abs() < 1e-12);
    // Clamped past the end rather than extrapolated.
    assert_eq!(time.frame_at(&base, 100.0).coeffs()[0], 4.0);
  }

  /// The heat flow of a localized bump is one grade-0 trajectory that decays:
  /// the parabolic Hodge-Laplacian damps the $L^2$ norm monotonically toward the
  /// held boundary, and the field animates (offers displacement) because it is a
  /// trajectory, not because it is an eigenmode. `nsteps` steps give `nsteps + 1`
  /// sampled frames.
  #[test]
  fn heat_trajectory_decays_and_animates() {
    let (topology, coords) = crate::gallery::MeshSource::Grid {
      dim: 2,
      cells_axis: 6,
    }
    .build()
    .unwrap();
    let scene = Scene::heat(topology, coords, 0, 20, 0.2);

    assert_eq!(scene.fields.len(), 1);
    assert!(scene.line_fields.is_empty());
    let FieldTime::Trajectory { frames, .. } = &scene.fields[0].time else {
      panic!("the heat flow is a trajectory");
    };
    assert_eq!(frames.len(), 21);
    let l2 = |c: &Cochain| c.coeffs().norm();
    assert!(
      l2(frames.last().unwrap()) < l2(&frames[0]),
      "the heat flow damps the bump"
    );
    assert!(scene.offers(Selection::Scalar(0)).displacement);
    assert!(scene.fields[0].time.eigenvalue().is_none());
  }

  /// The heat flow is total on a *closed* mesh, where there is no boundary to
  /// hold: `solve_heat` runs the free Neumann flow (the relative complex is the
  /// identity inclusion there) instead of panicking on an empty boundary
  /// subcomplex. The regression
  /// for the sphere preset, whose background solve otherwise never completes.
  /// Mass is conserved (the constant is the Neumann kernel), so the bump spreads
  /// rather than decaying to zero -- the peak drops while the total does not.
  #[test]
  fn heat_flow_is_total_on_a_closed_mesh() {
    use simplicial::mesher::sphere::mesh_sphere_surface;
    let (topology, coords) = mesh_sphere_surface(2);
    let scene = Scene::heat(topology, coords, 0, 10, 0.2);

    let FieldTime::Trajectory { frames, .. } = &scene.fields[0].time else {
      panic!("the heat flow is a trajectory");
    };
    assert_eq!(frames.len(), 11);
    let peak = |c: &Cochain| c.coeffs().amax();
    assert!(
      peak(frames.last().unwrap()) < peak(&frames[0]),
      "the closed-mesh heat flow spreads the bump"
    );
  }

  /// The wave equation of the same bump is one grade-0 trajectory that does not
  /// decay: the symplectic integrator conserves energy, so the $L^2$ norm stays
  /// bounded near its initial value rather than damping away. Like heat, it
  /// animates without being an eigenmode.
  #[test]
  fn wave_trajectory_is_a_bounded_animating_trajectory() {
    let (topology, coords) = crate::gallery::MeshSource::Grid {
      dim: 2,
      cells_axis: 6,
    }
    .build()
    .unwrap();
    let scene = Scene::wave(topology, coords, 0, 30, 4.0);

    let FieldTime::Trajectory { frames, .. } = &scene.fields[0].time else {
      panic!("the wave equation is a trajectory");
    };
    assert_eq!(frames.len(), 31);
    let l2 = |c: &Cochain| c.coeffs().norm();
    let initial = l2(&frames[0]);
    assert!(
      frames.iter().all(|f| l2(f) <= 4.0 * initial),
      "the conservative wave flow stays bounded"
    );
    assert!(scene.offers(Selection::Scalar(0)).displacement);
  }

  /// Both evolutions are posed at every grade of the de Rham complex, not just
  /// at 0: the parabolic flow damps the $L^2$ norm and the symplectic wave flow
  /// keeps it bounded, whatever grade the bump is a form of. Swept over every
  /// grade of a surface, the extremal ones included -- the top grade is where a
  /// grade-0-only construction (a scalar bump, a `ndofs(0)` source) would have
  /// gone wrong silently.
  #[test]
  fn both_evolutions_run_at_every_grade() {
    let (topology, coords) = crate::gallery::MeshSource::Grid {
      dim: 2,
      cells_axis: 4,
    }
    .build()
    .unwrap();
    let l2 = |c: &Cochain| c.coeffs().norm();

    // A trajectory files under the mark its *reduced* grade earns, so on a
    // surface grade 1 is a line field and 0 and 2 are densities; the single
    // field is read from whichever list it landed in.
    let only_field = |scene: &Scene| -> FieldTime {
      let times: Vec<FieldTime> = scene
        .fields
        .iter()
        .map(|f| f.time.clone())
        .chain(scene.line_fields.iter().map(|f| f.time.clone()))
        .collect();
      assert_eq!(times.len(), 1, "exactly one trajectory field");
      times.into_iter().next().unwrap()
    };

    for grade in topology.dim().range_inclusive() {
      let heat = Scene::heat(topology.clone(), coords.clone(), grade, 10, 0.2);
      let FieldTime::Trajectory { frames, .. } = &only_field(&heat) else {
        panic!("the heat flow is a trajectory at grade {grade}");
      };
      assert_eq!(frames.len(), 11);
      assert_eq!(frames[0].grade(), grade);
      assert!(
        l2(frames.last().unwrap()) <= l2(&frames[0]) + 1e-12,
        "the heat flow does not grow at grade {grade}"
      );

      let wave = Scene::wave(topology.clone(), coords.clone(), grade, 20, 2.0);
      let FieldTime::Trajectory { frames, .. } = &only_field(&wave) else {
        panic!("the wave equation is a trajectory at grade {grade}");
      };
      assert_eq!(frames[0].grade(), grade);
      let initial = l2(&frames[0]);
      assert!(
        frames.iter().all(|f| l2(f) <= 4.0 * initial + 1e-12),
        "the conservative wave flow stays bounded at grade {grade}"
      );
    }
  }

  /// What a field offers is its reduced grade's answer, and the answer is total
  /// on the range the ambient reaches: every field of the reference cell in
  /// every dimension is asked, and each gets the reading its reduction earns --
  /// a density the surface it paints (nothing of its own), a line field its
  /// three marks. Nothing is dropped and nothing is asked twice.
  #[test]
  fn offers_follow_the_reduced_grade_in_every_dimension() {
    for dim in 1..=3 {
      let scene = Scene::whitney_basis(dim);
      for index in 0..scene.fields.len() {
        let offers = scene.offers(Selection::Scalar(index));
        assert!(!offers.marks, "dim {dim}: a density has no mark of its own");
      }
      for index in 0..scene.line_fields.len() {
        let offers = scene.offers(Selection::Line(index));
        assert!(offers.marks, "dim {dim}: a line field offers its marks");
        assert!(
          !offers.displacement,
          "dim {dim}: a line field's curves are static -- there is no wave to ride"
        );
      }
    }
  }

  /// Displacement is offered exactly when there is a standing wave to toggle,
  /// which is what an eigenvalue is: the same distinction `FieldDisplay::build`
  /// makes when it hands a field with none an amplitude of zero. A raw Whitney
  /// basis function is that field, and a grade-0 eigenmode of the same cell
  /// complex is its counterpart -- so the two together are the rule, not one
  /// example of it.
  #[test]
  fn displacement_is_offered_exactly_to_a_standing_wave() {
    let basis = Scene::whitney_basis(2);
    for index in 0..basis.fields.len() {
      assert!(
        !basis.offers(Selection::Scalar(index)).displacement,
        "a Whitney basis function is no eigenmode: its amplitude is already zero"
      );
      assert!(!basis.offers(Selection::Scalar(index)).any());
    }

    let (topology, coords) = crate::gallery::MeshSource::Grid {
      dim: 2,
      cells_axis: 4,
    }
    .build()
    .unwrap();
    let (fields, _) = Scene::eigenmodes_grade(&topology, &coords, simplicial::Dim::ZERO, 4);
    let scene = Scene {
      surface: Surface::of(&topology, &coords),
      topology,
      coords,
      fields,
      line_fields: Vec::new(),
    };
    assert!(
      !scene.fields.is_empty(),
      "the grade-0 eigensolve produced modes"
    );
    for index in 0..scene.fields.len() {
      assert!(
        scene.offers(Selection::Scalar(index)).displacement,
        "a grade-0 eigenmode has a wave to ride"
      );
    }
  }

  /// The reference triangle's LSF gallery: one field per subsimplex of every
  /// grade, split into scalar densities (grades 0 and 2, the latter through
  /// the top-form Hodge star) and the grade-1 line field.
  #[test]
  fn whitney_basis_reference_triangle_has_one_field_per_subsimplex() {
    let scene = Scene::whitney_basis(2);
    assert_eq!(scene.fields.len(), 3 + 1); // 3 vertices, 1 face
    assert_eq!(scene.line_fields.len(), 3); // 3 edges
  }

  /// The triforce mesh's GSF gallery: same reduction, but every DOF simplex
  /// is now a global simplex of a 4-cell, 6-vertex, 9-edge mesh instead of a
  /// subsimplex of a single reference cell.
  #[test]
  fn whitney_basis_mesh_has_one_field_per_mesh_simplex() {
    let (topology, coords) = crate::demos::triforce();
    assert_eq!(topology.nsimplices(0), 6);
    assert_eq!(topology.nsimplices(1), 9);
    assert_eq!(topology.nsimplices(2), 4);

    let scene = Scene::whitney_basis_mesh(topology, coords);
    assert_eq!(scene.fields.len(), 6 + 4); // vertices, and faces via ⋆
    assert_eq!(scene.line_fields.len(), 9); // edges
  }

  /// The Hodge decomposition is a theorem, not a golden number. On a genus-1
  /// surface ($b_1 = 2$) the probe 1-form splits into three shells that sum
  /// back to it exactly, are pairwise $L^2$-orthogonal, and carry a genuinely
  /// nonzero harmonic component -- the part the two handle cycles pair with.
  /// The flat unit square is the contractible base case: the same solve,
  /// harmonic shell identically zero because there is no grade-1 homology to
  /// project onto. The harmonic dimension is read off the complex and must
  /// equal the surface's first Betti number in each case.
  ///
  /// Deliberately not Bob or another gallery mesh: those run to several
  /// thousand vertices, and the harmonic solve at that size dominates the
  /// entire workspace test suite's runtime. The generated donut is the genus-1
  /// case at a few dozen vertices, and being generated it needs no asset at
  /// all -- what it stands for is $b_1 = 2$, which its construction guarantees
  /// rather than a file's contents happening to have it.
  #[test]
  fn hodge_decomposition_splits_orthogonally() {
    use crate::gallery::MeshSource;
    use crate::gallery::{QUOTIENT_CELLS_DEFAULT, QuotientSurface};
    use formoniq::whitney_complex::{HilbertComplex, WhitneyComplex};
    use simplicial::linalg::CsrMatrix;

    let torus = || {
      MeshSource::Quotient {
        surface: QuotientSurface::Donut,
        cells_axis: QUOTIENT_CELLS_DEFAULT,
      }
      .build()
      .unwrap()
    };

    for (label, build, betti_1) in [
      (
        "Donut",
        &torus as &dyn Fn() -> (Complex, MeshCoords),
        2usize,
      ),
      (
        "Grid",
        &(|| {
          MeshSource::Grid {
            dim: 2,
            cells_axis: 4,
          }
          .build()
          .unwrap()
        }),
        0usize,
      ),
    ] {
      let (topology, coords) = build();
      assert!(
        topology.nsimplices(1) > 0,
        "{label}: mesh built empty (unfetched asset?)"
      );

      let input = hodge_probe_input(&topology, &coords);
      let parts = hodge_decompose(&topology, &coords, &input).expect("the decomposition solves");

      let metric = coords.to_edge_lengths_sq(&topology);
      let complex = WhitneyComplex::new(&topology, &metric);
      assert_eq!(
        complex.harmonic_dim(1),
        betti_1,
        "{label}: harmonic dimension is the first Betti number"
      );

      let mass = CsrMatrix::from(&complex.mass(1));
      let ip = |a: &Cochain, b: &Cochain| (&mass * b.coeffs()).dot(a.coeffs());

      // The three shells reconstruct the input exactly (LU residual aside).
      let sum = parts.exact.coeffs() + parts.coexact.coeffs() + parts.harmonic.coeffs();
      let residual = (&sum - input.coeffs()).norm();
      assert!(
        residual < 1e-8,
        "{label}: shells do not sum to input ({residual})"
      );

      // Pairwise orthogonal in the $L^2 Lambda^1$ inner product.
      let scale = ip(&input, &input).sqrt().max(1e-12);
      for (a, b, name) in [
        (&parts.exact, &parts.coexact, "exact·coexact"),
        (&parts.exact, &parts.harmonic, "exact·harmonic"),
        (&parts.coexact, &parts.harmonic, "coexact·harmonic"),
      ] {
        let cross = ip(a, b).abs() / (scale * scale);
        assert!(cross < 1e-6, "{label}: {name} not orthogonal ({cross})");
      }

      // The harmonic shell is nonzero exactly when the surface has grade-1
      // homology to carry it.
      let harmonic_frac = ip(&parts.harmonic, &parts.harmonic).sqrt() / scale;
      if betti_1 > 0 {
        assert!(
          harmonic_frac > 1e-6,
          "{label}: harmonic shell vanished ({harmonic_frac})"
        );
      } else {
        assert!(
          harmonic_frac < 1e-9,
          "{label}: spurious harmonic shell ({harmonic_frac})"
        );
      }
    }
  }

  /// The three worked examples are all grade-1 line fields (no scalar
  /// density), named for the picker -- and the edge-by-vertex-pair lookup in
  /// [`crate::gallery::CochainSpec::resolve`] found every edge of the
  /// triforce's coefficient table without panicking, which is the
  /// actual thing under test.
  #[test]
  fn triforce_cochains_are_three_named_line_fields() {
    let (topology, coords) = crate::demos::triforce();
    let scene = Scene::cochains(topology, coords, &crate::demos::triforce_examples());
    assert!(scene.fields.is_empty());
    let names: Vec<_> = scene.line_fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["constant field", "pure curl", "pure div"]);
  }

  /// The top-grade Whitney form stars to a constant density: on the flat
  /// reference triangle its pointwise Hodge star is the same nonzero scalar at
  /// every corner, so the surface renders as a flat color rather than blank.
  #[test]
  fn grade_top_whitney_basis_stars_to_a_constant_nonzero_density() {
    let scene = Scene::whitney_basis(2);
    let density = scene.fields.last().unwrap();
    let baked = crate::bake::BakedMesh::new(&scene.topology, &scene.coords);
    let colors = surface_corner_values(
      &scene.topology,
      &scene.coords,
      &density.cochain,
      &baked.cell_corners,
    );
    assert!(!colors.is_empty());
    assert!(colors.iter().all(|&v| v.abs() > 1e-9));
    assert!(colors.iter().all(|&v| (v - colors[0]).abs() < 1e-9));
  }

  /// The regression theorem for the surface tint: a Whitney basis field's
  /// support is exactly the cells its DOF simplex bounds. Read per corner in the
  /// corner's own cell, every cell the form vanishes on reads exactly zero at
  /// all three corners -- so a basis function no longer bleeds into cells that do
  /// not contain its DOF. A per-vertex tint could not state this: the DOF's
  /// endpoints carry a nonzero nodal value into every incident cell.
  #[test]
  fn whitney_basis_support_is_exactly_its_dof_cells() {
    let (topology, coords) = crate::demos::triforce();
    let scene = Scene::whitney_basis_mesh(topology, coords);
    let baked = crate::bake::BakedMesh::new(&scene.topology, &scene.coords);
    let n = scene.topology.dim();

    let basis = scene
      .fields
      .iter()
      .map(|f| (f.name.as_str(), &f.cochain))
      .chain(
        scene
          .line_fields
          .iter()
          .map(|f| (f.name.as_str(), &f.cochain)),
      );
    for (name, cochain) in basis {
      // The DOF simplex, as its vertex set, from the single nonzero cochain
      // entry (a Whitney basis function is dual to exactly one simplex).
      let idof = cochain
        .coeffs()
        .iter()
        .position(|&c| c.abs() > 0.5)
        .expect("a basis DOF");
      let dof_vertices = &scene
        .topology
        .skeleton_raw(cochain.grade())
        .simplex_by_kidx(idof)
        .vertices;

      let colors =
        surface_corner_values(&scene.topology, &scene.coords, cochain, &baked.cell_corners);
      for (cc, corners) in baked.cell_corners.iter().zip(colors.chunks_exact(3)) {
        let cell = SimplexIdx::new(n, cc.cell).handle(&scene.topology);
        let supported = dof_vertices
          .iter()
          .all(|v| cell.simplex().vertices.contains(v));
        assert!(
          supported || corners.iter().all(|&v| v.abs() < 1e-9),
          "field {name} tints a cell outside the support of its DOF {dof_vertices:?}",
        );
      }
    }
  }
}
