use std::borrow::Cow;

use glatt::field::DiffFormClosure;
use regge::lengths::mesh::MeshLengthsSq;
use simplicial::Sign;
use simplicial::linalg::{Matrix, Vector};
use simplicial::topology::orientation::Orientation;

use crate::ui::Selection;
use derham::{Cochain, project::derham_map, section::CoordFieldExt};
use multialgebra::{Blade, ExteriorGrade, Tensor, Variance};
use realize::surface::Surface;
use regge::coord::mesh::MeshCoords;
use simplicial::{
  Dim,
  topology::{complex::Complex, simplex::Simplex},
};

/// A renderable scene: a simplicial surface together with fields on it.
///
/// The seam between the formoniq engine and the viewer. A scene is produced
/// either by a live solve (the native path here) or, later, by deserializing a
/// cached bundle. The viewer neither knows nor cares which. It carries the
/// engine's own types — topology, coordinates, cochains — rather than a lossy
/// export format, so the coloring, displacement and mode selection stay
/// decisions of the viewer.
///
/// A field's grade decides the mark it is drawn with, not a per-scene choice,
/// and the rule is one line: reduce the $k$-form to its reduced grade
/// $min(k, n-k)$ via the Hodge star, then dispatch on that. A reduced grade of
/// 0 is a scalar density coloring the surface ([`ScalarField`]); a reduced
/// grade of 1 is a line field drawn with line-integral convolution
/// ([`LineField`]). Both marks exhaust $n <= 3$: only $n >= 4$ produces a
/// reduced grade $>= 2$, an $(n-k)$-dimensional sheet with no mark yet, a
/// future mark, not a special case to route around.
#[derive(Clone)]
pub struct Scene {
  pub topology: Complex,
  pub coords: MeshCoords,
  /// The reduction the bake draws and every field mark is chosen against: the
  /// mesh itself below $n = 3$, its boundary $partial M$ for a solid. Built once
  /// with the scene, because the marks are filed against its dimension, not
  /// the parent's.
  pub(crate) surface: Surface,
  pub fields: Vec<ScalarField>,
  pub line_fields: Vec<LineField>,
}

/// How a field varies in time, the temporal model the render clock reads, one
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
///   on the CPU and its field stream is re-baked per frame, exactly the
///   "scrubbing a trajectory rewrites only the field stream" the bake anticipates.
///
/// The eigenmode is the degenerate one-mode-with-known-modulation point of the
/// trajectory, which is why the two share every display path below and differ
/// only in where the animation is evaluated.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum FieldTime {
  Static,
  StandingWave { eigenvalue: f64 },
  Trajectory { dt: f64, frames: Vec<Cochain> },
}

impl FieldTime {
  /// The Hodge-Laplace eigenvalue driving the analytic standing wave, when the
  /// field is one. `None` for a static field and for a sampled trajectory,
  /// neither is animated by a dispersion relation. What the UI reads for the
  /// degeneracy pyramid and the transport frequency readout.
  pub fn eigenvalue(&self) -> Option<f64> {
    match self {
      FieldTime::StandingWave { eigenvalue } => Some(*eigenvalue),
      _ => None,
    }
  }

  /// The GPU standing wave's angular frequency $omega = sqrt(lambda)$. Zero for
  /// anything the GPU does not modulate in closed form, a static field, and a
  /// trajectory, whose height stream is rewritten per frame on the CPU instead
  /// (so $cos(0 dot.c t) = 1$ applies the current frame's height at full
  /// amplitude, statically).
  pub fn wave_omega(&self) -> f32 {
    self.eigenvalue().map_or(0.0, f64::sqrt) as f32
  }

  /// Whether the field animates the surface's displacement height at all, an
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
  /// bracketing sampled frames, lerping coefficients is lerping the Whitney
  /// field, since the interpolation is linear in them. For a static field or a
  /// standing wave the spatial cochain does not itself vary (the GPU modulates
  /// the standing wave), so `base` is returned unchanged. `t` is clamped to the
  /// sampled interval. The caller's own loop decides the wrap.
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
/// density is read per rendered corner at draw time
/// (`realize::reduce::corner_values`), not stored, so the top form's
/// discontinuity across cells survives to the colormap instead of being
/// averaged away.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ScalarField {
  pub name: String,
  /// The grade $k$ of the form this field was reconstructed from, before the
  /// reduction to a density. A genuine 0-form keeps $k = 0$. A top form keeps
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
  /// wave, or a solve's sampled trajectory. See [`FieldTime`].
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
/// tangent line field. Its (unsigned) magnitude $|V|_g$ is read per cell
/// (`realize::reduce::corner_values`) and tints the surface the marks are drawn
/// on.
///
/// The glyphs are static: $ker$ and $sharp$ are scale-invariant, so the
/// standing wave $u(t) = cos(sqrt(lambda) t) phi$ leaves them fixed and swings
/// only the magnitude tint through zero. A single real eigenmode does not
/// travel, so the glyphs are never advected, only the particles are, on the
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
  /// the particles all read the true Whitney field $W c$ (via
  /// [`WhitneyInterpolant`](derham::interpolate::interpolant::WhitneyInterpolant)) cell by cell: there is no per-vertex reduction,
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

/// The render mark a grade-$k$ field on an $n$-manifold is drawn with: the
/// reduced grade $min(k, n-k)$, named.
///
/// The rule stated once, for the two questions that ask it. A field being filed
/// asks it of its own grade, and a grade tab asks it of a grade nothing has
/// been solved at yet, which is why the mark cannot always be read off a
/// [`Selection`] and has to be derivable from the grade alone.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mark {
  /// Reduced grade 0: a scalar density, coloring the surface.
  Density,
  /// Reduced grade 1: a tangent line field, drawn as glyphs and particles.
  LineField,
  /// Reduced grade $>= 2$, only reachable at $n >= 4$: an $(n-k)$-dimensional
  /// sheet, with no mark yet. A future mark, not a case to route around.
  Sheet,
}

impl Mark {
  pub(crate) fn of(grade: ExteriorGrade, n: Dim) -> Self {
    match (grade.min(n - grade)).index() {
      0 => Self::Density,
      1 => Self::LineField,
      _ => Self::Sheet,
    }
  }

  pub(crate) fn label(self) -> &'static str {
    match self {
      Self::Density => "density",
      Self::LineField => "line field",
      Self::Sheet => "sheet",
    }
  }
}

/// The display metadata a reconstructed field carries regardless of which
/// render mark it lands in, everything [`Scene::field`] needs beyond the
/// cochain itself, bundled so the two independent `Option`s
/// ([`ScalarField::time`]/[`ScalarField::dof`]) don't turn the
/// constructor into an unreadable run of positional arguments.
struct FieldMeta {
  name: String,
  time: FieldTime,
  dof: Option<Simplex>,
}

/// What the selected field offers to be read with, which of
/// `crate::ui::FieldView`'s settings are live.
///
/// The mesh side has no counterpart: every scene has geometry, so its settings
/// are always live and there is nothing to gate. Only the field is asked, and
/// the answer is its reduced grade's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FieldOffers {
  /// Whether the field drives a standing wave. A reduced grade of 0 displaces
  /// the surface along its normal, and only an eigenmode has the dispersion
  /// relation to do it at: without an eigenvalue the amplitude is already zero,
  /// so the toggle would control nothing.
  pub displacement: bool,
  /// Whether the field has marks of its own: a reduced grade of 1 is a tangent
  /// line field, and the glyphs and particles are its two readings. A density
  /// has no mark beyond the surface it paints, which is the mesh's.
  pub marks: bool,
  /// Whether the field has an interior to march. A solid's field lives in a
  /// volume the boundary primitive cannot show, so the medium is offered
  /// exactly when the manifold is codimension-zero enough to have one, an
  /// intrinsic-dimension question, not a grade one, which is why it is the only
  /// offer read off the complex rather than the selection.
  pub volume: bool,
}

impl FieldOffers {
  /// Whether the field offers anything at all, false for a density that is no
  /// eigenmode (a raw Whitney basis function), whose whole rendering is the
  /// tint on the mesh's surface.
  pub fn any(self) -> bool {
    self.displacement || self.marks || self.volume
  }
}

impl Scene {
  /// A scene on a mesh, carrying no fields yet: what every constructor below
  /// starts from and [`Self::file`] fills.
  ///
  /// The surface reduction is fixed here, before any field exists, because the
  /// mark a field gets is chosen against the surface's dimension and not the
  /// mesh's.
  pub(crate) fn on(topology: Complex, coords: MeshCoords) -> Self {
    Self {
      surface: Surface::of(&topology, &coords),
      topology,
      coords,
      fields: Vec::new(),
      line_fields: Vec::new(),
    }
  }

  /// The reduced grade's answer to what its field can be read with, and the one
  /// place it is asked outside the display: a selection already is the
  /// reduction (which list it indexes is which mark it landed in), so this
  /// reads it off rather than dispatching on grade a second time.
  pub fn offers(&self, selection: Selection) -> FieldOffers {
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

  /// Files the Hodge-Laplace eigenmodes of one grade: the standing-wave normal
  /// modes $Delta u = lambda u$ of the scene's own mesh, through the same
  /// [`Self::file`] dispatch a raw Whitney basis function goes through. An
  /// eigenmode and a one-hot cochain differ only in where the cochain comes
  /// from, an eigensolve against a Kronecker delta, not in how it is
  /// reconstructed or displayed.
  ///
  /// A failed eigensolve contributes no fields and is reported on stderr: an
  /// iteration budget too small for one mesh is not a reason to take the
  /// viewer down.
  fn file_eigenmodes(&mut self, grade: ExteriorGrade, nmodes: usize) {
    use formoniq::{problems::elliptic::solve_evp, whitney_complex::WhitneyComplex};

    let metric = self.coords.to_edge_lengths_sq(&self.topology);
    let solved = solve_evp(&WhitneyComplex::new(&self.topology, &metric), grade, nmodes);
    let (eigenvals, _, eigenfuncs) = match solved {
      Ok(solved) => solved,
      Err(err) => {
        eprintln!("grade {grade} eigensolve failed: {err}");
        return;
      }
    };

    for (i, (&lambda, col)) in eigenvals.iter().zip(eigenfuncs.column_iter()).enumerate() {
      self.file(
        FieldMeta {
          name: format!("mode {i} (grade {grade}, lambda = {lambda:.2})"),
          time: FieldTime::StandingWave { eigenvalue: lambda },
          dof: None,
        },
        Cochain::new(grade, col.into_owned()),
      );
    }
  }

  /// The solve-free placeholder on a mesh: a single flat field, so the viewer
  /// can show the geometry the instant it has one while the study solves in
  /// the background and swaps the real fields in when it lands. The lone field
  /// has no time, so it is drawn as a plain, undeformed surface.
  pub fn placeholder_on(topology: Complex, coords: MeshCoords) -> Self {
    let nvertices = topology.skeleton(0).len();
    let mut scene = Self::on(topology, coords);
    scene.fields.push(ScalarField {
      name: "loading...".to_string(),
      grade: Dim::ZERO,
      cochain: Cochain::new(Dim::ZERO, na::DVector::zeros(nvertices)),
      time: FieldTime::Static,
      dof: None,
    });
    scene
  }

  /// One grade's Hodge-Laplace eigenmodes, the standing-wave normal modes of
  /// an arbitrary simplicial manifold with the given geometry. The unit the
  /// gallery memoizes per grade, so switching grade pays for that grade's
  /// solve alone, and only the first time it is viewed.
  ///
  /// Nothing here assumes a sphere: the discrete spherical harmonics are this
  /// study on that mesh, and every grade runs the same code, the extremal ones
  /// included, which is the point of the $min(k, n-k)$ dispatch.
  pub fn eigenmodes(
    topology: &Complex,
    coords: &MeshCoords,
    grade: ExteriorGrade,
    nmodes: usize,
  ) -> Self {
    let mut scene = Self::on(topology.clone(), coords.clone());
    scene.file_eigenmodes(grade, nmodes);
    scene
  }

  /// Every Whitney basis function ("local shape function") of the standard
  /// reference cell of dimension `cell_dim`, the single-cell case of the
  /// shared construction below, where every DOF simplex's support is the one
  /// cell itself.
  pub fn whitney_basis(cell_dim: impl Into<Dim>) -> Self {
    use regge::coord::mesh::unit_coord_complex;

    let cell_dim = cell_dim.into();
    let (topology, coords) = unit_coord_complex(cell_dim);
    // The renderer is 3D-only. A reference cell of `dim < 3` embeds as
    // itself in the `z = 0` plane, same as `bake.rs` does
    // for any other flat surface. A no-op once `cell_dim >= 3`.
    let coords = coords.embed_euclidean(cell_dim.max(Dim::new(3)));
    Self::whitney_basis_on(topology, coords)
  }

  /// Every Whitney basis function ("global shape function") of an arbitrary
  /// simplicial mesh, the multi-cell case of the shared construction below,
  /// where a DOF simplex's support spans every cell incident to it, which is
  /// exactly where the LSF/GSF distinction shows up on screen: the same
  /// one-hot-cochain construction, not confined to a single cell.
  pub fn whitney_basis_mesh(topology: Complex, coords: MeshCoords) -> Self {
    Self::whitney_basis_on(topology, coords)
  }

  /// A named list of explicit cochains on a mesh, each resolved from its
  /// [`crate::gallery::CochainSpec`] and reduced to its render mark through the
  /// same `file` dispatch every other field goes through, a field
  /// here is a general linear combination, not confined to a single one-hot
  /// cochain. The worked triforce examples (a constant field, a pure-curl
  /// field and a pure-divergence field) are one such list; a loaded cochain
  /// file is a future one.
  pub fn cochains(
    topology: Complex,
    coords: MeshCoords,
    specs: &[crate::gallery::NamedCochain],
  ) -> Self {
    let mut scene = Self::on(topology, coords);
    for named in specs {
      let cochain = named.spec.resolve(&scene.topology);
      scene.file(
        FieldMeta {
          name: named.name.clone(),
          time: FieldTime::Static,
          dof: None,
        },
        cochain,
      );
    }
    scene
  }

  /// The Hodge decomposition of a probe field, as four switchable fields: the
  /// input $omega$ and its three $L^2$-orthogonal shells
  /// $omega = dif alpha + delta beta + h$, exact, coexact, and harmonic. The
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

    let mut scene = Self::on(topology, coords);
    for (name, cochain) in named {
      scene.file(
        FieldMeta {
          name: name.to_string(),
          time: FieldTime::Static,
          dof: None,
        },
        cochain,
      );
    }
    scene
  }

  /// Shared construction for [`Self::whitney_basis`] and
  /// [`Self::whitney_basis_mesh`]: one field per DOF simplex of every grade
  /// $0..=$ `topology.dim()`, each the reconstructed field of a one-hot
  /// cochain, the basis function dual to a DOF simplex $sigma$ is the
  /// cochain $c_tau = delta_(sigma tau)$, so there is no separate "evaluate a
  /// Whitney form" code path here. The interpolant and the sharp musical
  /// isomorphism are exactly the general machinery a solved field (an
  /// eigenmode, or a future loaded cochain) goes through too. DOF simplices
  /// are named by their vertex tuple straight off `topology`'s own
  /// colexicographic skeleton order, which coincides with
  /// `unit_subsimps` on the single-cell reference complex.
  fn whitney_basis_on(topology: Complex, coords: MeshCoords) -> Self {
    let mut scene = Self::on(topology, coords);
    for grade in scene.topology.dim().range_inclusive() {
      let ndofs = scene.topology.nsimplices(grade);
      let dofs: Vec<Simplex> = scene.topology.skeleton_raw(grade).iter().cloned().collect();
      for (idof, dof) in dofs.into_iter().enumerate() {
        let mut coeffs = na::DVector::zeros(ndofs);
        coeffs[idof] = 1.0;
        scene.file(
          FieldMeta {
            name: format!("W^{grade}_{}", dof_label(&dof)),
            time: FieldTime::Static,
            dof: Some(dof),
          },
          Cochain::new(grade, coeffs),
        );
      }
    }
    scene
  }

  /// The heat flow $partial_t u = -kappa Delta u$ of a localized initial bump, as a
  /// single [`FieldTime::Trajectory`] field of grade `grade`: the sampled solution the
  /// transport scrubs and the surface re-bakes per frame. The bump diffuses and
  /// decays, the parabolic smoothing of the Hodge-Laplacian, shown directly
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
    use formoniq::{
      problems::heat::solve_heat,
      whitney_complex::{HilbertComplex, WhitneyComplex},
    };

    let metric = coords.to_edge_lengths_sq(&topology);
    let whitney = WhitneyComplex::new(&topology, &metric);
    let relative = whitney.relative();

    let initial = ambient_bump(&topology, &coords, grade);
    let source = Cochain::new(grade, na::DVector::zeros(whitney.ndofs(grade)));
    let dt = final_time / nsteps.max(1) as f64;
    let frames = solve_heat(&relative, grade, nsteps, dt, &initial, &source, 1.0);

    Self::trajectory_scene(topology, coords, initial, dt, frames)
  }

  /// The wave equation $partial_(t t) u = -Delta u$ of a localized initial bump at
  /// rest, as a single [`FieldTime::Trajectory`] field of grade `grade`. The bump splits
  /// and its fronts propagate, reflecting off any boundary, the hyperbolic
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
      whitney_complex::{HilbertComplex, WhitneyComplex},
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

  /// Linear advection $partial_t omega + cal(L)_v omega = 0$ of a localized bump
  /// along a rotational velocity field, as a sampled trajectory: the transport
  /// counterpart of [`Self::heat`] and [`Self::wave`], on the same initial data
  /// and the same mesh-agnostic footing.
  ///
  /// The velocity is exactly divergence-free with
  /// single-valued facet flux, so grade 0 and top grade are conservative on any
  /// mesh. The scheme is central and dispersive
  /// ([`LieDerivative`](formoniq::operators::LieDerivative)), so what
  /// the trajectory shows next to the transport is the oscillation that costs.
  pub fn advection(
    topology: Complex,
    coords: MeshCoords,
    grade: impl Into<ExteriorGrade>,
    nsteps: usize,
    final_time: f64,
  ) -> Self {
    let grade = grade.into();
    use derham::{interpolate::interpolant::WhitneyInterpolant, section::SectionOps};
    use formoniq::problems::advection::{Transport, solve_transport};

    let metric = coords.to_edge_lengths_sq(&topology);
    // The star reads an orientation, so a non-orientable mesh has no velocity
    // to build and the field stands still rather than flipping across facets.
    let Some(orientation) = topology.orientation().cloned() else {
      return Self::placeholder_on(topology, coords);
    };
    let flux = mean_speed_flux(&topology, &coords, &metric, &orientation);
    let velocity = WhitneyInterpolant::new(flux, &topology)
      .hodge_star(&topology, &metric, &orientation)
      .musical(&topology, &metric);

    let initial = ambient_bump(&topology, &coords, grade);
    let dt = final_time / nsteps.max(1) as f64;
    let transport = Transport {
      grade,
      velocity: &velocity,
      quad_degree: 2,
    };
    let frames = solve_transport(&topology, &metric, &transport, nsteps, dt, &initial);

    Self::trajectory_scene(topology, coords, initial, dt, frames)
  }

  /// Files a solved trajectory of any grade into a scene through the same
  /// [`Self::file`] dispatch every other field goes through: the trajectory's
  /// first frame is its spatial representative, the sampled family its
  /// [`FieldTime`].
  fn trajectory_scene(
    topology: Complex,
    coords: MeshCoords,
    initial: Cochain,
    dt: f64,
    frames: Vec<Cochain>,
  ) -> Self {
    let mut scene = Self::on(topology, coords);
    scene.file(
      FieldMeta {
        name: "trajectory".to_string(),
        time: FieldTime::Trajectory { dt, frames },
        dof: None,
      },
      initial,
    );
    scene
  }

  /// The displayed field's temporal model, for the transport clock and the
  /// per-frame re-bake the caller drives.
  pub(crate) fn field_time(&self, selection: Selection) -> &FieldTime {
    match selection {
      Selection::Scalar(i) => &self.fields[i].time,
      Selection::Line(i) => &self.line_fields[i].time,
    }
  }

  /// The displayed field's spatial representative cochain, the `base` a
  /// [`FieldTime::frame_at`] reads at each instant.
  pub(crate) fn field_cochain(&self, selection: Selection) -> &Cochain {
    match selection {
      Selection::Scalar(i) => &self.fields[i].cochain,
      Selection::Line(i) => &self.line_fields[i].cochain,
    }
  }

  /// Every field of the scene, in the picker's flat order: the scalars, then
  /// the line fields.
  ///
  /// One order, stated once. It is what the mode picker lays out and what
  /// `--field N` indexes, and those two have to be the same order or a name on
  /// the command line means a different field than the one on screen.
  pub fn selections(&self) -> impl Iterator<Item = Selection> + use<'_> {
    (0..self.fields.len())
      .map(Selection::Scalar)
      .chain((0..self.line_fields.len()).map(Selection::Line))
  }

  /// The scene's fields as the mode picker reads them, in [`Self::selections`]'s
  /// order.
  ///
  /// The picker's rows and the flat index `--field N` names are then the same
  /// walk over the same order, so neither can be laid out against a convention
  /// the other does not share.
  pub(crate) fn entries(&self) -> Vec<crate::ui::Entry<'_>> {
    self
      .selections()
      .map(|selection| {
        let (name, grade, dof) = match selection {
          Selection::Scalar(i) => {
            let field = &self.fields[i];
            (field.name.as_str(), field.grade, field.dof.as_ref())
          }
          Selection::Line(i) => {
            let field = &self.line_fields[i];
            (field.name.as_str(), field.grade, field.dof.as_ref())
          }
        };
        crate::ui::Entry {
          selection,
          grade,
          eigenvalue: self.field_time(selection).eigenvalue(),
          dof,
          name,
        }
      })
      .collect()
  }

  /// Reconstructs a cochain as the render mark its reduced grade
  /// $min(k, n-k)$ calls for, and files it into the scene under the mark that
  /// grade names: the one general entry point both a raw Whitney basis function
  /// ([`Self::whitney_basis`]) and a solved field arrive at.
  ///
  /// The Hodge star is what makes the dispatch total. A reduced grade of 0
  /// ($k = 0$ or $k = n$) is a scalar density, a reduced grade of 1 ($k = 1$ or
  /// $k = n-1$) a tangent line field, and a reduced grade $>= 2$ (only reachable
  /// at $n >= 4$) has no mark yet. The reduction is not applied here: the
  /// original cochain is stored whole, and the render mark reads it per cell at
  /// draw time (see [`realize::reduce::corner_values`]).
  ///
  /// The grade reduces against the surface's dimension, not the mesh's, and
  /// that is what makes the mark the mark of the thing on screen. A field on a
  /// solid is seen through its boundary, so the $n$ in $min(k, n-k)$ is
  /// $dim partial M$: a $2$-form on a $3$-manifold is a line field in the volume
  /// but the boundary's top form, hence a density, where it is actually
  /// drawn. Reducing against the parent would file it as arrows for a flux that
  /// has no direction on the surface carrying it.
  ///
  /// The exception is the grade that does not trace at all ($k = n$, where
  /// $C^k (partial M) = 0$): a volume density is not a surface quantity, so it
  /// reduces against the parent and is drawn by sampling the cells behind the
  /// boundary, until a volume mark exists to own it.
  fn file(&mut self, meta: FieldMeta, cochain: Cochain) {
    let FieldMeta { name, time, dof } = meta;
    // A mode's sign is arbitrary, so it is pinned. A trajectory's is physical
    // (it solved from an initial condition), and its frames are what the
    // display reads, so flipping the representative alone would desync it.
    let cochain = if time.is_trajectory() {
      cochain
    } else {
      canonical_sign(cochain)
    };
    let k = cochain.grade();
    // The manifold the mark is drawn on, named once: a grade that does not
    // trace is a volume quantity and keeps the parent (see the doc above).
    // Both the reduction's $n$ and the orientability the star needs are read
    // off it, so the two cannot disagree about which object they are about.
    let topology = &self.topology;
    let drawn_on = if self.surface.traces(topology, k) {
      self.surface.complex(topology)
    } else {
      topology
    };
    let n = drawn_on.dim();

    // The reduction stars whenever $k > n-k$, and the star needs a global
    // volume form, which a non-orientable mesh does not have. The field is
    // then not drawable at all: there is no orientation-independent density
    // or direction to show, so it is refused here rather than rendered with
    // a per-cell sign that means nothing. Everything below the star is
    // unaffected and still files normally. The solver is unaffected either
    // way, since the gauge cancels inside the assembly.
    if k > n - k && !drawn_on.is_orientable() {
      eprintln!(
        "field '{name}' (grade {k} of {n}) needs the Hodge star to be drawn, \
         and the mesh is non-orientable: no global volume form, so it is skipped"
      );
      return;
    }

    match Mark::of(k, n) {
      Mark::Density => {
        // The original $k$-cochain is kept whole. The reduction to a density (a
        // pointwise Hodge star for $k = n$, the identity for $k = 0$) is read
        // per corner at draw time by `realize::reduce::corner_values`, never
        // averaged into the stored field.
        self.fields.push(ScalarField {
          name,
          grade: k,
          cochain,
          time,
          dof,
        });
      }
      Mark::LineField => {
        self.line_fields.push(LineField {
          name,
          grade: k,
          cochain,
          time,
          dof,
        });
      }
      Mark::Sheet => {
        // No render mark yet: it files into no list rather than panicking.
      }
    }
  }
}

/// Which natural operator a field is read through before it is reduced to a
/// scalar.
///
/// The scalar every scalar-consuming mark draws is `scalarize(F omega)`, and
/// this is $F$. Each variant is total over grade and dimension, degenerating
/// rather than being excluded: $dif omega = 0$ at $k = n$, and [`scalarize`]
/// takes the resulting top or bottom grade uniformly. The axis is deliberately
/// separate from the reduction: the operator is metric-free, the reduction is
/// where the metric enters.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum Scalarization {
  /// $omega$ itself: where the field is, and the only reading that needs no
  /// derivative.
  #[default]
  Value,
  /// $dif omega$: where the field varies. At $k = n - 1$ this is the source
  /// density, a top form the reduction stars into a signed scalar, so the
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

/// A cochain in canonical sign: the coefficient of largest magnitude is made
/// positive, ties broken by colex rank.
///
/// An eigenvector is defined only up to a scalar, and a solver may return
/// either sign on a whim, so the same mode can come out red where it was blue
/// between two runs of the same scene, for reasons that are not the
/// mathematics. Pinning it is what makes a rendered field reproducible, and it
/// dominates the orientation gauge (which is already deterministic): the
/// field's own sign sits in front of the density either way.
///
/// A gauge fix, not a normalization: the magnitude is untouched, so the
/// colormap range still spans what the field actually is. The zero cochain is
/// its own canonical form, having no largest coefficient to orient by.
///
/// Applies to modes only. A trajectory's sign is physical, it solved from
/// an initial condition, and flipping it would be a lie about the solve, so
/// the caller excludes it.
pub fn canonical_sign(cochain: Cochain) -> Cochain {
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

/// The three $L^2$-orthogonal shells of the discrete Hodge decomposition of a
/// $k$-cochain, $omega = "exact" + "coexact" + "harmonic"$, each a $k$-cochain
/// on the same complex.
pub struct HodgeParts {
  /// $dif alpha$, $alpha in cal(W) Lambda^(k-1)$: the exact (range-of-$dif$)
  /// component.
  pub exact: Cochain,
  /// $delta beta$, $beta in cal(W) Lambda^(k+1)$: the coexact
  /// (range-of-$delta$) component.
  pub coexact: Cochain,
  /// $h in cal(H)^k$: the harmonic component, the $L^2$-projection of $omega$
  /// onto the $b_k$-dimensional harmonic space.
  pub harmonic: Cochain,
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
/// exactly, the three shells sum back to $omega$ with no residual, and the
/// coexact shell is recovered as the remainder rather than by forming $delta$
/// explicitly. Their pairwise $L^2$-orthogonality is the content of the mixed
/// formulation.
///
/// [`WhitneyComplex`]: formoniq::whitney_complex::WhitneyComplex
pub fn hodge_decompose(
  topology: &Complex,
  coords: &MeshCoords,
  input: &Cochain,
) -> Result<HodgeParts, formoniq::linalg::eigen::EigenError> {
  use formoniq::{
    galerkin::GalerkinVector,
    problems::elliptic::{solve_harmonics, solve_source},
    whitney_complex::{HilbertComplex, WhitneyComplex},
  };

  let grade = input.grade();
  let metric = coords.to_edge_lengths_sq(topology);
  let complex = WhitneyComplex::new(topology, &metric);

  // The load vector of the source problem is the Riesz representation of the
  // functional $angle.l omega, dot.c angle.r$, i.e. $M_k omega$.
  let mass = complex.mass(grade);
  let source_galvec = GalerkinVector::new(grade, &mass * input.coeffs());

  let (sigma, _u, p) = solve_source(&complex, source_galvec, grade)?;
  let harmonics = solve_harmonics(&complex, grade)?;

  // exact $= dif sigma$. At grade 0 the $sigma in Lambda^(-1)$ space is empty,
  // so the exact shell is identically zero.
  let exact = if grade > 0 {
    let dif = simplicial::linalg::CsrMatrix::from(&topology.coboundary_operator(grade - 1));
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

/// The probe field the decomposition study splits: the ambient swirl
/// `hodge_probe_form` plus, on a mesh with grade-1 homology, an explicit copy
/// of a harmonic 1-form scaled to the swirl's magnitude.
///
/// The swirl alone supplies rich exact and coexact shells, but its periods
/// around the handles can vanish: they do on the Császár torus, where a purely
/// ambient probe leaves the harmonic shell at numerical zero. Whether an ambient
/// field excites a cycle depends on how the handle happens to sit in space,
/// which is no basis for a teaching example. Injecting a harmonic generator,
/// the one dual to the first homology cycle, makes the field genuinely carry a
/// topological cycle on any genus-$g$ surface, so the decomposition
/// demonstrates all three shells regardless of
/// embedding, and injecting the harmonic part is itself the point: the
/// decomposition returns it untouched, orthogonal to the two it did not put
/// there. On a contractible mesh the harmonic space is empty and nothing is
/// added.
pub fn hodge_probe_input(topology: &Complex, coords: &MeshCoords) -> Cochain {
  use formoniq::{
    harmonic::harmonics,
    problems::elliptic::solve_harmonics,
    whitney_complex::{HilbertComplex, WhitneyComplex},
  };

  let swirl = hodge_probe_form(topology, coords);
  let metric = coords.to_edge_lengths_sq(topology);
  let complex = WhitneyComplex::new(topology, &metric);
  let mass = complex.mass(1);
  let m_norm = |v: &Vector| (&mass * v).dot(v).max(0.0).sqrt();

  // The period-normalized reading: the injected cycle then threads one handle,
  // which is what makes the harmonic shell the decomposition returns legible as
  // that handle's circulation rather than as a mixture of the genus-many.
  let basis = match harmonics(&complex, 1) {
    Some(harmonics) => Ok(harmonics.integral),
    None => solve_harmonics(&complex, 1),
  };
  match basis {
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
/// nonzero. It does not reliably carry a harmonic part, whether an ambient
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

/// How many of the lowest eigenmodes to ask for: enough to cover the
/// degeneracy of the bottom eigenspace, which is the ambient dimension's worth
/// of rotations on a sphere and rarely more elsewhere. The zero shell is
/// separated by its eigenvalue rather than by counting.
const LOWEST_MODES: usize = 4;
const HARMONIC_EIGENVALUE: f64 = 1e-8;

/// A discretely divergence-free velocity's flux, as an $(n-1)$-cochain: the
/// smoothest closed one the mesh admits.
///
/// Closed is what conservation needs. $dif sigma = 0$ is the discrete
/// divergence, and reading the velocity off an $(n-1)$-form is the other half,
/// since the tangential part of one on a facet is its flux, which the Whitney
/// space's conformity makes single-valued between neighbors. Those are the two
/// conditions transport needs to conserve at the ends of the grade range.
///
/// Smoothest is what makes it look like transport. A field that is not Killing
/// shears, drawing a bump into filaments finer than the mesh holds, which a
/// central scheme turns into oscillation. A Killing field generally does not
/// exist: on a piecewise-flat manifold it would have to fix every hinge of
/// nonzero angle defect. The reachable ask is therefore the least-shearing
/// closed field, the minimizer of the Hodge-Laplace Rayleigh quotient over
/// closed forms.
///
/// The classical answers fall out of that one construction rather than being
/// special-cased. The harmonic forms are its zero shell, so a mesh whose
/// topology supplies them uses one, on a torus the circulation around a handle.
/// Where there are none the lowest exact mode takes over, on a sphere $dif$ of
/// the $ell = 1$ eigenfunction (the rigid rotation) at the eigenvalue
/// $ell (ell + 1) = 2$.
///
/// The lowest eigenspace is degenerate, threefold on a sphere and twofold on a
/// torus, so a member is chosen by projecting a fixed reference onto it rather
/// than by taking whichever vector the eigensolver returned first, which would
/// swing the flow's direction between refinements. The reference is a
/// coboundary, so the Hodge decomposition puts it orthogonal to the harmonic
/// shell and the projection is empty exactly where the topology supplies
/// harmonics, leaving the first harmonic generator, which the period
/// normalization ties to one specific handle. Either way the choice is a gauge,
/// every member of the shell being equally the smoothest closed field, and both
/// the reference and the periods buy reproducibility rather than correctness.
pub fn solenoidal_flux(topology: &Complex, coords: &MeshCoords, metric: &MeshLengthsSq) -> Cochain {
  let reference = ambient_blade_flux(topology, coords);
  let Some(space) = smoothest_closed_space(topology, metric) else {
    return reference;
  };
  project_onto(&reference, &space, topology, metric)
    .unwrap_or_else(|| Cochain::new(reference.grade(), space.column(0).into_owned()))
}

/// The lowest-eigenvalue closed $(n-1)$-forms: the harmonic shell where the
/// topology has one, else $dif$ of the lowest nonzero $(n-2)$-eigenmode, whose
/// eigenvalue it inherits because $dif$ commutes with the Laplacian.
pub fn smoothest_closed_space(topology: &Complex, metric: &MeshLengthsSq) -> Option<Matrix> {
  use formoniq::{
    harmonic::harmonics,
    problems::elliptic::{solve_evp, solve_harmonics},
    whitney_complex::WhitneyComplex,
  };
  let flux_grade = topology.dim() - 1;
  let whitney = WhitneyComplex::new(topology, metric);

  // The period-normalized reading, so that where the reference projects to
  // nothing and a column is displayed directly, that column is the circulation
  // around one handle rather than an arbitrary rotation of the shell. The
  // orthonormal reading spans the same space and serves where the projection
  // behind the periods is not well posed.
  let harmonic = match harmonics(&whitney, flux_grade) {
    Some(harmonics) => harmonics.integral,
    None => solve_harmonics(&whitney, flux_grade).ok()?,
  };
  if harmonic.ncols() > 0 {
    return Some(harmonic);
  }

  let potential_grade = flux_grade - 1;
  if potential_grade < 0 {
    return None;
  }
  let (values, _, modes) = solve_evp(&whitney, potential_grade, LOWEST_MODES).ok()?;
  let columns: Vec<Vector> = values
    .iter()
    .enumerate()
    .filter(|(_, value)| **value > HARMONIC_EIGENVALUE)
    .take(LOWEST_MODES)
    .map(|(i, _)| {
      Cochain::new(potential_grade, modes.column(i).into_owned())
        .dif(topology)
        .into_coeffs()
    })
    .collect();
  (!columns.is_empty()).then(|| Matrix::from_columns(&columns))
}

/// The $L^2$ projection of `reference` onto the span of `space`'s columns,
/// `None` if it lands on nothing there.
///
/// The emptiness test is in the mass norm the projection is taken in, against a
/// margin far above roundoff: where the reference is genuinely orthogonal to the
/// space, what survives is the eigensolver's residual, and a threshold close to
/// machine epsilon would pass that noise off as a field.
fn project_onto(
  reference: &Cochain,
  space: &Matrix,
  topology: &Complex,
  metric: &MeshLengthsSq,
) -> Option<Cochain> {
  use formoniq::{galerkin::BilinearForm, operators::WhitneyPairing};

  let grade = reference.grade();
  let mass = WhitneyPairing::mass(topology.dim(), grade).assemble(topology, metric);
  let weighted = &mass * space;
  let gram = space.transpose() * &weighted;
  let rhs = weighted.transpose() * reference.coeffs();

  let coeffs = gram.lu().solve(&rhs)?;
  let projected_norm_sq = coeffs.dot(&rhs);
  let reference_norm_sq = formoniq::linalg::quadratic_form_sparse(&mass, reference.coeffs());

  (projected_norm_sq > 1e-12 * reference_norm_sq).then(|| Cochain::new(grade, space * coeffs))
}

/// The de Rham map of the constant ambient $(n-1)$-form summing every blade: a
/// cocycle on any mesh at all, since a constant form is closed and both
/// pullback and the de Rham map commute with $dif$.
///
/// Summing the blades rather than picking one keeps it from vanishing where the
/// tangent planes annihilate that blade. It shears, so it serves as the
/// reference a smoother field is chosen against rather than as the velocity.
pub fn ambient_blade_flux(topology: &Complex, coords: &MeshCoords) -> Cochain {
  let ambient = coords.dim();
  let flux_grade = topology.dim() - 1;
  let coeffs = Vector::from_element(multialgebra::exterior_dim(ambient, flux_grade), 1.0);

  let form = DiffFormClosure::new(
    move |_| Tensor::multiform(coeffs.clone(), ambient, flux_grade),
    ambient,
    flux_grade,
  );
  derham_map(&form.pullback_on(topology, coords), topology, 1)
}

/// [`solenoidal_flux`] scaled to unit mean speed, so a unit of solve time is
/// a unit of distance traveled and the final time reads as a path length.
///
/// The mean and not the peak: a peak is one cell's outlier, and normalizing
/// against it leaves the field as a whole moving slower than the final time
/// says. Scaling a cochain is linear all the way to the velocity and leaves the
/// cocycle a cocycle.
fn mean_speed_flux(
  topology: &Complex,
  coords: &MeshCoords,
  metric: &MeshLengthsSq,
  orientation: &Orientation,
) -> Cochain {
  use derham::interpolate::interpolant::WhitneyInterpolant;
  use derham::section::{Section, SectionOps};
  use simplicial::atlas::ChartExt;

  let flux = solenoidal_flux(topology, coords, metric);
  let probe = WhitneyInterpolant::new(flux.clone(), topology)
    .hodge_star(topology, metric, orientation)
    .musical(topology, metric);

  let cells = topology.cells();
  // The magnitude of a vector field is its metric norm, not the Euclidean
  // length of its components in a cell's own frame: those agree only where the
  // cell's metric is the identity, so a stretched mesh would be scaled wrong.
  let mean: f64 = cells
    .handle_iter()
    .map(|cell| {
      metric::tensor::TensorExt::norm(
        &probe.at(&ChartExt::barycenter(cell)),
        &metric.cell_metric(cell),
      )
    })
    .sum::<f64>()
    / cells.len().max(1) as f64;

  if mean > 0.0 {
    Cochain::new(flux.grade(), flux.coeffs() / mean)
  } else {
    flux
  }
}

/// A localized grade-$k$ initial condition for a time-dependent solve, defined
/// off the mesh's own coordinates: a Gaussian in ambient distance centered on
/// the vertex nearest the centroid, of width a fixed fraction of the
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
/// boundary would pin the bump instead of letting it diffuse. The nearest one is
/// interior, so its boundary trace is near zero and the flow is free. On a closed
/// mesh every vertex is on the surface, so the nearest merely also works where a
/// boundary exists.
pub fn ambient_bump(topology: &Complex, coords: &MeshCoords, grade: ExteriorGrade) -> Cochain {
  let geometry = coords.to_edge_lengths_sq(topology);
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

  let bump = move |x: &Vector| {
    let r2 = (x - &center).norm_squared();
    (-r2 / (2.0 * sigma * sigma)).exp()
  };

  // At top grade the bump is a density $f vol$, and the volume form is the
  // manifold's own, not an ambient blade's pullback. The two agree where the
  // mesh fills its ambient space, and on a submanifold only this one is right:
  // the pullback of a fixed ambient $n$-blade is the projection of the tangent
  // multivector onto it, which changes sign over a curved surface.
  //
  // The cochain of an $n$-form is its integral over each cell, so the density
  // is written directly rather than through a section.
  if grade == topology.dim() {
    let coeffs = topology
      .cells()
      .handle_iter()
      .map(|cell| {
        let vertices = cell.get().simplex().clone();
        let barycenter: Vector = vertices
          .iter()
          .map(|v| coords.coord(v).into_owned())
          .fold(Vector::zeros(n), |acc, c| acc + c)
          / vertices.nvertices() as f64;
        bump(&barycenter) * regge::cell_volume(&geometry.cell_metric(cell))
      })
      .collect::<Vec<_>>();
    return Cochain::new(grade, Vector::from_vec(coeffs));
  }

  let blade = Tensor::from_blade_signed(
    n,
    Sign::Pos,
    Blade::from_rank(grade.index(), 0),
    Variance::Covariant,
  );
  let field = DiffFormClosure::new(move |p| blade.clone() * bump(p.vector()), n, grade);
  let pulled = field.pullback_on(topology, coords);
  derham_map(&pulled, topology, 2)
}
