use common::{gramian::RiemannianMetric, linalg::nalgebra::Vector};
use continuum::field::DiffFormClosure;
use ddf::{
  cochain::Cochain, derham::derham_map, section::CoordFieldExt,
  whitney::interpolant::WhitneyInterpolant,
};
use exterior::{ExteriorGrade, MultiForm};
use manifold::{
  atlas::MeshPoint,
  geometry::{
    coord::{mesh::MeshCoords, simplex::SimplexRefExt},
    metric::geometry::Geometry,
  },
  topology::complex::Complex,
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

/// A named scalar field on the surface: a 0-cochain, one value per vertex.
///
/// A grade-0 form is one directly; a top-grade ($k = n$) form becomes one by
/// the pointwise Hodge star $star: Lambda^n -> Lambda^0$, nodal-sampled to the
/// vertices. From here the two are indistinguishable -- the same density
/// colormap and the same normal-displacement standing wave.
#[derive(Clone)]
pub struct ScalarField {
  pub name: String,
  /// The grade $k$ of the form this field was reconstructed from, before the
  /// reduction to a density. A genuine 0-form keeps $k = 0$; a top form keeps
  /// $k = n$ even though it is drawn through its Hodge star. Carried so the
  /// gallery can organize by original grade, not by the render mark the reduced
  /// grade happens to share with grade 0.
  pub grade: ExteriorGrade,
  pub cochain: Cochain,
  /// The Hodge-Laplace eigenvalue $lambda$ this field is an eigenfunction of,
  /// when it is one. For the wave equation $partial_t^2 u + Delta u = 0$ this
  /// is the square of the mode's own temporal frequency, $omega = sqrt(lambda)$
  /// -- the standing wave for *this* mode, not a frequency picked by the
  /// viewer. `None` for a field that is not an eigenfunction (e.g. a raw
  /// Whitney basis function), which disables the standing-wave animation.
  pub eigenvalue: Option<f64>,
  /// The DOF simplex this field is dual to, as its vertex tuple (e.g.
  /// `"01"`), when the field is a raw Whitney basis function. `None` for a
  /// solved field (an eigenmode), which has no single dual simplex -- the
  /// same is-this-a-basis-function distinction [`Self::eigenvalue`] makes,
  /// mirrored: exactly one of the two is `Some` on any field this crate
  /// produces. Lets the picker group basis functions by grade and label each
  /// cell by its DOF without reparsing [`Self::name`].
  pub dof_label: Option<String>,
}

/// A named line field on the surface: the reduced-grade-1 mark, drawn with
/// line-integral convolution rather than sampled arrow glyphs.
///
/// A grade-1 (or, via the Hodge star, grade-$(n-1)$) form reduces to a genuine
/// tangent *line* field, drawn as its integral curves: the mark is the traced
/// streamlines of the true Whitney field, evenly spaced at a separation fixed
/// to the object's own extent. Its (unsigned) nodal magnitude is the same
/// per-vertex recovery a [`ScalarField`] uses, and tints the surface the curves
/// are drawn on.
///
/// The curves are static: $ker$ and $sharp$ are scale-invariant, so the
/// standing wave $u(t) = cos(sqrt(lambda) t) phi$ leaves the lines fixed and
/// swings only the magnitude tint through zero. A single real eigenmode does
/// not travel, so the curves are never advected.
#[derive(Clone)]
pub struct LineField {
  pub name: String,
  /// The grade $k$ of the form this field was reconstructed from, before the
  /// reduction to a line field. A grade-1 form keeps $k = 1$; a grade-$(n-1)$
  /// form keeps $k = n-1$ even though it is drawn through its Hodge star. See
  /// [`ScalarField::grade`].
  pub grade: ExteriorGrade,
  /// The original $k$-cochain this field was reduced from, kept whole so the
  /// viewer can reconstruct the *true* Whitney field $W c$ (via
  /// [`WhitneyInterpolant`]) rather than only the nodal average below. The
  /// streamline tracer integrates $((W c)|_"reduced")^sharp$ cell by cell; the
  /// nodal `magnitude` below is the coarser readout the surface tint uses.
  pub cochain: Cochain,
  /// Per-vertex nodal magnitude $|V|_g$, the intrinsic chart-independent scalar
  /// averaged across incident cells: what tints the surface, times
  /// $cos(sqrt(lambda) t)$ per frame so the sign flips through zero.
  pub magnitude: Vec<f64>,
  /// See [`ScalarField::eigenvalue`].
  pub eigenvalue: Option<f64>,
  /// See [`ScalarField::dof_label`].
  pub dof_label: Option<String>,
}

impl ScalarField {
  pub fn values(&self) -> &[f64] {
    self.cochain.coeffs().as_slice()
  }

  /// Value range across the field, for colormap normalization. Falls back to a
  /// unit range on an empty or constant field so the viewer never normalizes by
  /// a zero span.
  pub fn bounds(&self) -> (f32, f32) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in self.values() {
      lo = lo.min(v);
      hi = hi.max(v);
    }
    if lo < hi {
      (lo as f32, hi as f32)
    } else {
      (-1.0, 1.0)
    }
  }
}

/// The display metadata a reconstructed field carries regardless of which
/// render mark it lands in -- everything [`Scene::field`] needs beyond the
/// cochain itself, bundled so the two independent `Option`s
/// ([`ScalarField::eigenvalue`]/[`ScalarField::dof_label`]) don't turn the
/// constructor into an unreadable run of positional arguments.
struct FieldMeta {
  name: String,
  eigenvalue: Option<f64>,
  dof_label: Option<String>,
}

impl LineField {
  /// Magnitude range across the field, for colormap normalization -- the
  /// [`ScalarField::bounds`] of the nodal magnitude. Unsigned by
  /// construction ($|V|_g >= 0$), so a static field's true range starts at
  /// (or near) zero; the caller widens it to the symmetric $[-m, m]$ an
  /// animated tint needs, since $|V| cos(sqrt(lambda) t)$ swings negative
  /// even though $|V|$ itself never does.
  pub fn bounds(&self) -> (f32, f32) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in &self.magnitude {
      lo = lo.min(v);
      hi = hi.max(v);
    }
    if lo < hi {
      (lo as f32, hi as f32)
    } else {
      (-1.0, 1.0)
    }
  }
}

impl Scene {
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
    use formoniq::{
      problems::hodge_laplace::solve_hodge_laplace_evp, whitney_complex::WhitneyComplex,
    };

    let metric = coords.to_edge_lengths(topology);
    let solved = solve_hodge_laplace_evp(&WhitneyComplex::new(topology, &metric), grade, nmodes);
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
        coords,
        FieldMeta {
          name,
          eigenvalue: Some(lambda),
          dof_label: None,
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
    use manifold::gen::sphere::mesh_sphere_surface;

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
    use manifold::gen::sphere::mesh_sphere_surface;

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
      eigenvalue: None,
      dof_label: None,
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
    use manifold::geometry::coord::mesh::standard_coord_complex;

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
        &coords,
        FieldMeta {
          name: named.name.clone(),
          eigenvalue: None,
          dof_label: None,
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
        &coords,
        FieldMeta {
          name: name.to_string(),
          eigenvalue: None,
          dof_label: None,
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
        let label = dof_simp
          .vertices
          .iter()
          .map(|v| v.to_string())
          .collect::<String>();
        let name = format!("W^{grade}_{{{label}}}");

        let mut coeffs = na::DVector::zeros(ndofs);
        coeffs[idof] = 1.0;
        let cochain = Cochain::new(grade, coeffs);

        Self::field(
          &topology,
          &coords,
          FieldMeta {
            name,
            eigenvalue: None,
            dof_label: Some(label),
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

  /// Reconstructs a cochain as the render mark its *reduced grade*
  /// $min(k, n-k)$ calls for, and files it into `fields` or `line_fields`
  /// accordingly -- the one general entry point both a raw Whitney basis
  /// function ([`Self::whitney_basis`]) and a solved field arrive at.
  ///
  /// The Hodge star is what makes the dispatch total. A reduced grade of 0
  /// ($k = 0$ or $k = n$) is a scalar density: grade 0 needs no reconstruction
  /// (a Whitney 0-form is the barycentric-linear function through its vertex
  /// values, which the rasterizer already interpolates), while $k = n$ stars
  /// pointwise to a 0-form and is nodal-sampled. A reduced grade of 1 ($k = 1$
  /// or $k = n-1$) reduces to a genuine tangent line field. A reduced grade
  /// $>= 2$ is only reachable at $n >= 4$ and has no mark yet.
  fn field(
    topology: &Complex,
    coords: &MeshCoords,
    meta: FieldMeta,
    cochain: Cochain,
    fields: &mut Vec<ScalarField>,
    line_fields: &mut Vec<LineField>,
  ) {
    let FieldMeta {
      name,
      eigenvalue,
      dof_label,
    } = meta;
    let n = topology.dim();
    let k = cochain.grade();
    match k.min(n - k) {
      0 => {
        // k == 0: the coefficients already are the vertex values. k == n: the
        // pointwise Hodge star of the top form, nodal-sampled to a 0-cochain.
        let cochain = if k == 0 {
          cochain
        } else {
          let interpolant = WhitneyInterpolant::new(cochain, topology);
          Cochain::new(0, nodal_scalar_density(topology, coords, &interpolant))
        };
        fields.push(ScalarField {
          name,
          grade: k,
          cochain,
          eigenvalue,
          dof_label,
        });
      }
      1 => {
        let interpolant = WhitneyInterpolant::new(cochain, topology);
        let magnitude = nodal_line_magnitude(topology, coords, &interpolant);
        line_fields.push(LineField {
          name,
          grade: k,
          cochain: interpolant.cochain().clone(),
          magnitude,
          eigenvalue,
          dof_label,
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
pub(crate) fn reduced_form(form: MultiForm, metric: &RiemannianMetric) -> MultiForm {
  let n = form.dim();
  let k = form.grade();
  if k <= n - k {
    form
  } else {
    form.hodge_star(metric)
  }
}

/// The nodal average over incident cells of the reduced grade-1 field's
/// magnitude $|V|_g$, the intrinsic chart-independent scalar the surface is
/// tinted by. Unlike grade 0, a grade-1 field has no canonical value at a
/// vertex -- only the *tangential* part of a section is chart-independent there
/// (the atlas's own invariant), so incident cells genuinely disagree, and
/// averaging is the same nodal recovery classical FEM postprocessing uses for a
/// piecewise, not globally, single-valued quantity.
///
/// The metric is what makes this well defined: $|V|_g$ is a scalar, so the
/// average is of numbers, not of vectors in cell-local frames that no shared
/// frame relates. The field's *direction* has no such nodal recovery worth
/// keeping -- the streamline tracer reads the true Whitney field cell by cell
/// instead.
fn nodal_line_magnitude(
  topology: &Complex,
  coords: &MeshCoords,
  interpolant: &WhitneyInterpolant,
) -> Vec<f64> {
  let nvertices = topology.skeleton_raw(0).len();
  let mut mag_sum = vec![0.0; nvertices];
  let mut count = vec![0u32; nvertices];
  for cell in topology.cells().handle_iter() {
    let metric = coords.cell_metric(cell);
    // The affine parametrization $psi_K: hat(K) -> RR^N$ whose differential
    // pushes the sharped vector's coefficients (in the cell's own local
    // tangent basis, the basis `metric` is expressed in) out into the ambient
    // frame the renderer draws in -- the one place the embedding enters.
    let coord_simplex = cell.coord_simplex(coords);
    for (ilocal, &v) in cell.simplex().vertices.iter().enumerate() {
      let mut weights = na::DVector::zeros(cell.nvertices());
      weights[ilocal] = 1.0;
      let point = MeshPoint::new(cell.idx(), weights.into());
      let local = reduced_form(interpolant.eval(&point), &metric).sharp(&metric);
      let ambient = coord_simplex.pushforward_vector(local.coeffs());
      mag_sum[v] += to_vec3(&ambient).norm();
      count[v] += 1;
    }
  }
  mag_sum
    .into_iter()
    .zip(count)
    .map(|(s, c)| if c > 0 { s / f64::from(c) } else { 0.0 })
    .collect()
}

/// The nodal average over incident cells of the reduced grade-0 field (the
/// pointwise Hodge star of a top-grade form), one signed value per vertex.
fn nodal_scalar_density(
  topology: &Complex,
  coords: &MeshCoords,
  interpolant: &WhitneyInterpolant,
) -> Vector {
  let nvertices = topology.skeleton_raw(0).len();
  let mut sum = vec![0.0; nvertices];
  let mut count = vec![0u32; nvertices];
  for cell in topology.cells().handle_iter() {
    let metric = coords.cell_metric(cell);
    for (ilocal, &v) in cell.simplex().vertices.iter().enumerate() {
      let mut weights = na::DVector::zeros(cell.nvertices());
      weights[ilocal] = 1.0;
      let point = MeshPoint::new(cell.idx(), weights.into());
      let density = reduced_form(interpolant.eval(&point), &metric).coeffs()[0];
      sum[v] += density;
      count[v] += 1;
    }
  }
  Vector::from_iterator(
    nvertices,
    sum
      .into_iter()
      .zip(count)
      .map(|(s, c)| if c > 0 { s / f64::from(c) } else { 0.0 }),
  )
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
) -> Result<HodgeParts, common::linalg::eigen::EigenError> {
  use common::linalg::nalgebra::CsrMatrix;
  use formoniq::{
    problems::hodge_laplace::{solve_hodge_laplace_harmonics, solve_hodge_laplace_source},
    whitney_complex::WhitneyComplex,
  };

  let grade = input.grade();
  let metric = coords.to_edge_lengths(topology);
  let complex = WhitneyComplex::new(topology, &metric);

  // The load vector of the source problem is the Riesz representation of the
  // functional $angle.l omega, dot.c angle.r$, i.e. $M_k omega$.
  let mass = CsrMatrix::from(&complex.mass(grade));
  let source_galvec = &mass * input.coeffs();

  let (sigma, _u, p) = solve_hodge_laplace_source(&complex, source_galvec, grade)?;
  let harmonics = solve_hodge_laplace_harmonics(&complex, grade)?;

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
/// mesh through the `ddf` bridge, then de Rham mapped to a `grade`-cochain.
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
  use common::linalg::nalgebra::CsrMatrix;
  use formoniq::{
    problems::hodge_laplace::solve_hodge_laplace_harmonics, whitney_complex::WhitneyComplex,
  };

  let swirl = hodge_probe_form(topology, coords);
  let metric = coords.to_edge_lengths(topology);
  let complex = WhitneyComplex::new(topology, &metric);
  let mass = CsrMatrix::from(&complex.mass(1));
  let m_norm = |v: &Vector| (&mass * v).dot(v).max(0.0).sqrt();

  match solve_hodge_laplace_harmonics(&complex, 1) {
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
/// $omega = -y dif x + x dif y + z dif z$ pulled onto the mesh through the `ddf`
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

/// A nalgebra vector of ambient coordinates, zero-padded to 3D.
fn to_vec3(v: &Vector) -> na::Vector3<f64> {
  na::Vector3::new(
    v[0],
    if v.len() > 1 { v[1] } else { 0.0 },
    if v.len() > 2 { v[2] } else { 0.0 },
  )
}

#[cfg(test)]
mod tests {
  use super::*;

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

  /// The Hodge decomposition is a theorem, not a golden number. On Bob (a
  /// genus-1 surface, so $b_1 = 2$) the probe 1-form splits into three shells
  /// that sum back to it exactly, are pairwise $L^2$-orthogonal, and carry a
  /// genuinely nonzero harmonic component -- the part the two handle cycles pair
  /// with. The flat unit square is the contractible base case: the same solve,
  /// harmonic shell identically zero because there is no grade-1 homology to
  /// project onto. The harmonic dimension is read off the complex and must equal
  /// the surface's first Betti number in each case.
  #[test]
  fn hodge_decomposition_splits_orthogonally() {
    use crate::gallery::{BuiltinMesh, MeshSource};
    use common::linalg::nalgebra::CsrMatrix;
    use formoniq::whitney_complex::{HilbertComplex, WhitneyComplex};

    for (source, betti_1) in [
      (MeshSource::Builtin(BuiltinMesh::Bob), 2usize),
      (MeshSource::Grid { cells_axis: 4 }, 0usize),
    ] {
      let label = source.label();
      let (topology, coords) = source.build().unwrap();
      assert!(
        topology.nsimplices(2) > 0,
        "{label}: mesh built empty (unfetched asset?)"
      );

      let input = hodge_probe_input(&topology, &coords);
      let parts = hodge_decompose(&topology, &coords, &input).expect("the decomposition solves");

      let metric = coords.to_edge_lengths(&topology);
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
  /// [`crate::gallery::CochainSpec::resolve`] found every edge of
  /// `plot/in/triforce`'s coefficient table without panicking, which is the
  /// actual thing under test.
  #[test]
  fn triforce_cochains_are_three_named_line_fields() {
    let (topology, coords) = crate::demos::triforce();
    let scene = Scene::cochains(topology, coords, &crate::demos::triforce_examples());
    assert!(scene.fields.is_empty());
    let names: Vec<_> = scene.line_fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["constant field", "pure curl", "pure div"]);
  }
}
