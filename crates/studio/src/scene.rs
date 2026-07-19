use std::borrow::Cow;

use glatt::field::DiffFormClosure;
use gramian::PseudoRiemannianMetric;
use simplicial::linalg::Vector;

use crate::bake::CellCorner;
use crate::ui::Selection;
use derham::{
  cochain::Cochain, interpolate::interpolant::WhitneyInterpolant, project::derham_map,
  section::CoordFieldExt,
};
use exterior::{ExteriorGrade, MultiForm};
use simplicial::{
  atlas::MeshPoint,
  geometry::{coord::mesh::MeshCoords, metric::geometry::Geometry},
  topology::{complex::Complex, handle::SimplexIdx, role::Cell, simplex::Simplex},
  Dim,
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
#[derive(Clone)]
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
#[derive(Clone)]
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
#[derive(Clone)]
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
}

impl FieldOffers {
  /// Whether the field offers anything at all -- false for a density that is no
  /// eigenmode (a raw Whitney basis function), whose whole rendering is the
  /// tint on the mesh's surface.
  pub(crate) fn any(self) -> bool {
    self.displacement || self.marks
  }
}

impl Scene {
  /// The reduced grade's answer to what its field can be read with, and the one
  /// place it is asked outside the display: a selection already *is* the
  /// reduction (which list it indexes is which mark it landed in), so this
  /// reads it off rather than dispatching on grade a second time.
  pub(crate) fn offers(&self, selection: Selection) -> FieldOffers {
    match selection {
      Selection::Scalar(index) => FieldOffers {
        displacement: self.fields[index].time.animates(),
        marks: false,
      },
      Selection::Line(_) => FieldOffers {
        displacement: false,
        marks: true,
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
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    Self::eigenmode_fields(&topology, &coords, 0, nmodes, &mut fields, &mut line_fields);
    Self {
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
    use simplicial::gen::sphere::mesh_sphere_surface;

    let (topology, coords) = mesh_sphere_surface(nsubdivisions);
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    for grade in 0..=topology.dim() {
      let (mut f, mut l) = Self::eigenmodes_grade(&topology, &coords, grade, nmodes);
      fields.append(&mut f);
      line_fields.append(&mut l);
    }
    Self {
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
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    Self::eigenmode_fields(
      topology,
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
    use simplicial::gen::sphere::mesh_sphere_surface;

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
      grade: 0,
      cochain: Cochain::new(0, na::DVector::zeros(nvertices)),
      time: FieldTime::Static,
      dof: None,
    }];
    Self {
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
  pub fn whitney_basis(cell_dim: Dim) -> Self {
    use simplicial::geometry::coord::mesh::standard_coord_complex;

    let (topology, coords) = standard_coord_complex(cell_dim);
    // The renderer is 3D-only; a reference cell of `dim < 3` embeds as
    // itself in the `z = 0` plane, same as `bake.rs` does
    // for any other flat surface. A no-op once `cell_dim >= 3`.
    let coords = coords.embed_euclidean(cell_dim.max(3));
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
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    for named in specs {
      Self::field(
        &topology,
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
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    for grade in 0..=dim {
      let ndofs = topology.nsimplices(grade);
      for (idof, dof_simp) in topology.skeleton_raw(grade).iter().enumerate() {
        let name = format!("W^{grade}_{{{}}}", dof_label(dof_simp));

        let mut coeffs = na::DVector::zeros(ndofs);
        coeffs[idof] = 1.0;
        let cochain = Cochain::new(grade, coeffs);

        Self::field(
          &topology,
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
      topology,
      coords,
      fields,
      line_fields,
    }
  }

  /// The heat flow $diff_t u = -kappa Delta u$ of a localized initial bump, as a
  /// single grade-0 [`FieldTime::Trajectory`] field: the sampled solution the
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
  pub fn heat(topology: Complex, coords: MeshCoords, nsteps: usize, final_time: f64) -> Self {
    use formoniq::{problems::heat::solve_heat, whitney_complex::WhitneyComplex};

    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);
    let relative = whitney.relative();

    let initial = ambient_bump(&topology, &coords);
    let source = Cochain::new(0, na::DVector::zeros(whitney.ndofs(0)));
    let dt = final_time / nsteps.max(1) as f64;
    let frames = solve_heat(&relative, 0, nsteps, dt, &initial, &source, 1.0);

    Self::trajectory_scene(topology, coords, initial, dt, frames)
  }

  /// The wave equation $diff_(t t) u = -Delta u$ of a localized initial bump at
  /// rest, as a single grade-0 [`FieldTime::Trajectory`] field. The bump splits
  /// and its fronts propagate, reflecting off any boundary -- the hyperbolic
  /// counterpart of [`Self::heat`], on the same initial data and the same
  /// mesh-agnostic footing (a closed mesh uses the identity inclusion).
  ///
  /// [`solve_wave`]: formoniq::problems::wave::solve_wave
  pub fn wave(topology: Complex, coords: MeshCoords, nsteps: usize, final_time: f64) -> Self {
    use formoniq::{
      problems::wave::{solve_wave, WaveState},
      whitney_complex::WhitneyComplex,
    };

    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);

    let initial = ambient_bump(&topology, &coords);
    let ndofs = whitney.ndofs(0);
    let dt = final_time / nsteps.max(1) as f64;
    let times: Vec<f64> = (0..=nsteps).map(|k| k as f64 * dt).collect();
    let state = WaveState::new(initial.coeffs().clone(), na::DVector::zeros(ndofs));
    let force = Cochain::new(0, na::DVector::zeros(ndofs));
    let frames = solve_wave(&whitney, 0, &times, state, force)
      .into_iter()
      .map(|s| Cochain::new(0, s.pos))
      .collect();

    Self::trajectory_scene(topology, coords, initial, dt, frames)
  }

  /// Files a solved grade-0 trajectory into a scene through the same `field`
  /// dispatch every other field goes through: the trajectory's first frame is
  /// its spatial representative, the sampled family its [`FieldTime`].
  fn trajectory_scene(
    topology: Complex,
    coords: MeshCoords,
    initial: Cochain,
    dt: f64,
    frames: Vec<Cochain>,
  ) -> Self {
    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    Self::field(
      &topology,
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
  fn field(
    topology: &Complex,
    meta: FieldMeta,
    cochain: Cochain,
    fields: &mut Vec<ScalarField>,
    line_fields: &mut Vec<LineField>,
  ) {
    let FieldMeta { name, time, dof } = meta;
    let n = topology.dim();
    let k = cochain.grade();
    match k.min(n - k) {
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
pub(crate) fn reduced_form(form: MultiForm, metric: &PseudoRiemannianMetric) -> MultiForm {
  let n = form.dim();
  let k = form.grade();
  if k <= n - k {
    form
  } else {
    form.hodge_star(metric)
  }
}

/// The surface colormap scalar at every rendered triangle corner, read *in the
/// corner's own cell* -- three per triangle, in [`CellCorner`] order.
///
/// This is the fix for the discontinuity the reduced-grade fields have: a
/// Whitney form is single-valued only in its *tangential* part across a face, so
/// incident cells disagree at a shared vertex and its support ends exactly on
/// cell edges. A per-vertex value shared between cells therefore leaks a basis
/// function's magnitude into cells outside its support (via the rasterizer's
/// interpolation across the shared vertex). Reading the field once per corner,
/// each in its own cell, keeps the support exact: a cell the form vanishes on
/// contributes three zeros and stays black.
///
/// A reduced grade of 0 takes the density (the signed 0-form coefficient); a
/// reduced grade of 1 takes the magnitude $|V|_g$, the intrinsic
/// chart-independent scalar (the direction is left to the glyph and particle
/// marks, which read the true Whitney field cell by cell already).
pub(crate) fn surface_corner_values(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  cell_corners: &[CellCorner],
) -> Vec<f64> {
  let n = topology.dim();
  let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);
  let mut values = Vec::with_capacity(3 * cell_corners.len());
  for cc in cell_corners {
    let cell = SimplexIdx::new(n, cc.cell).handle(topology).role();
    let metric = coords.cell_metric(cell);
    for &ilocal in &cc.local {
      values.push(reduced_value(&interpolant, cell, &metric, ilocal));
    }
  }
  values
}

/// The per-vertex displacement height: the reduced field's nodal average over
/// the cells incident at each vertex.
///
/// The colormap must be discontinuous (see [`surface_corner_values`]), but the
/// standing-wave *displacement* is a geometric height of one connected object,
/// and a shared vertex has one position -- displacing its incident cells' copies
/// apart would tear the surface (or the curve). So displacement takes this
/// continuous nodal recovery, exact for a genuine 0-form whose incident cells
/// already agree. It is a function of the field alone, not of the render
/// primitive, so it is defined at every dimension -- a 1-manifold has no fill to
/// average corners over, yet its curve still displaces.
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
  metric: &PseudoRiemannianMetric,
  ilocal: usize,
) -> f64 {
  let mut weights = na::DVector::zeros(cell.nvertices());
  weights[ilocal] = 1.0;
  let point = MeshPoint::new(cell.idx(), weights.into());
  let form = reduced_form(interpolant.eval(&point), metric);
  if form.grade() == 0 {
    form.coeffs()[0]
  } else {
    form.norm(metric)
  }
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
  let n = coords.dim();
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

/// A localized grade-0 initial condition for a time-dependent solve, defined off
/// the mesh's own coordinates: a Gaussian in ambient distance centered on the
/// vertex *nearest* the centroid, of width a fixed fraction of the coordinate
/// extent. Pulled onto the mesh through the `derham` bridge and de Rham mapped to a
/// 0-cochain, so it lands on any embedded surface without assuming a shape.
///
/// The nearest-to-centroid vertex, not the farthest: on a mesh with boundary
/// (the flat grid) the farthest vertex is a boundary corner, where a held
/// boundary would pin the bump instead of letting it diffuse; the nearest one is
/// interior, so its boundary trace is near zero and the flow is free. On a closed
/// mesh every vertex is on the surface, so the nearest merely also works where a
/// boundary exists.
fn ambient_bump(topology: &Complex, coords: &MeshCoords) -> Cochain {
  let n = coords.dim();
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

  let field = DiffFormClosure::scalar(
    move |p| {
      let r2 = (p.vector() - &center).norm_squared();
      (-r2 / (2.0 * sigma * sigma)).exp()
    },
    n,
  );
  let pulled = field.pullback_on(topology, coords);
  derham_map(&pulled, topology, 2)
}

#[cfg(test)]
mod tests {
  use super::*;

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
    let (topology, coords) = crate::gallery::MeshSource::Grid { cells_axis: 6 }
      .build()
      .unwrap();
    let scene = Scene::heat(topology, coords, 20, 0.2);

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
    use simplicial::gen::sphere::mesh_sphere_surface;
    let (topology, coords) = mesh_sphere_surface(2);
    let scene = Scene::heat(topology, coords, 10, 0.2);

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
    let (topology, coords) = crate::gallery::MeshSource::Grid { cells_axis: 6 }
      .build()
      .unwrap();
    let scene = Scene::wave(topology, coords, 30, 4.0);

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

    let (topology, coords) = crate::gallery::MeshSource::Grid { cells_axis: 4 }
      .build()
      .unwrap();
    let (fields, _) = Scene::eigenmodes_grade(&topology, &coords, 0, 4);
    let scene = Scene {
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
  /// entire workspace test suite's runtime. `torus0.msh` -- a genus-1 surface
  /// at 127 vertices, the coarsest level of a Gmsh refinement family -- is kept
  /// as a fixture solely for this test (see `assets/meshes/SOURCES.md`).
  #[test]
  fn hodge_decomposition_splits_orthogonally() {
    use crate::gallery::MeshSource;
    use formoniq::whitney_complex::{HilbertComplex, WhitneyComplex};
    use simplicial::io::gmsh::gmsh2coord_complex;
    use simplicial::linalg::CsrMatrix;

    let torus = || gmsh2coord_complex(include_bytes!("../assets/meshes/torus0.msh"));

    for (label, build, betti_1) in [
      (
        "torus0.msh",
        &torus as &dyn Fn() -> (Complex, MeshCoords),
        2usize,
      ),
      (
        "Grid",
        &(|| MeshSource::Grid { cells_axis: 4 }.build().unwrap()),
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
