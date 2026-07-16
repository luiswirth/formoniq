use common::{gramian::RiemannianMetric, linalg::nalgebra::Vector};
use ddf::{cochain::Cochain, whitney::interpolant::WhitneyInterpolant};
use exterior::{ExteriorGrade, MultiForm};
use manifold::{
  atlas::MeshPoint,
  geometry::{
    coord::{mesh::MeshCoords, simplex::SimplexRefExt},
    metric::geometry::Geometry,
  },
  topology::{complex::Complex, simplex::standard_subsimps},
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
}

/// A named line field on the surface: the reduced-grade-1 mark, drawn with
/// line-integral convolution rather than sampled arrow glyphs.
///
/// A grade-1 (or, via the Hodge star, grade-$(n-1)$) form reduces to a genuine
/// tangent *line* field. Its direction is a per-vertex unit ambient tangent
/// vector, nodal-averaged across incident cells and interpolated by the
/// rasterizer; its (unsigned) magnitude is the same per-vertex nodal recovery a
/// [`ScalarField`] uses, and drives both the tint and the standing-wave
/// amplitude. LIC is a continuous per-fragment texture, resolution-independent,
/// so unlike an arrow quiver there is no sample lattice to choose.
///
/// The direction is static: $ker$ and $sharp$ are scale-invariant, so the
/// standing wave $u(t) = cos(sqrt(lambda) t) phi$ leaves the lines fixed and
/// swings only the magnitude tint through zero. A single real eigenmode does
/// not travel, so the LIC is never advected.
#[derive(Clone)]
pub struct LineField {
  pub name: String,
  /// The grade $k$ of the form this field was reconstructed from, before the
  /// reduction to a line field. A grade-1 form keeps $k = 1$; a grade-$(n-1)$
  /// form keeps $k = n-1$ even though it is drawn through its Hodge star. See
  /// [`ScalarField::grade`].
  pub grade: ExteriorGrade,
  /// Per-vertex unit ambient tangent direction (zero where the field
  /// vanishes). Nodal-averaged across incident cells then normalized -- unlike
  /// grade 0, a grade-1 field has no canonical value at a vertex, since only
  /// the *tangential* part of a section is chart-independent there (the atlas's
  /// own invariant), so incident cells can genuinely disagree on the full
  /// vector; averaging is the same nodal recovery classical FEM postprocessing
  /// uses for a piecewise, not globally, single-valued quantity.
  pub direction: Vec<na::Vector3<f64>>,
  /// Per-vertex nodal magnitude $|V|_g$, the intrinsic chart-independent scalar
  /// averaged across incident cells: what tints the surface, times
  /// $cos(sqrt(lambda) t)$ per frame so the sign flips through zero.
  pub magnitude: Vec<f64>,
  /// See [`ScalarField::eigenvalue`].
  pub eigenvalue: Option<f64>,
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

impl LineField {
  /// Largest nodal magnitude, for tint normalization. Symmetric bounds
  /// $[-m, m]$ follow from it, since the animated tint $|V| cos(sqrt(lambda) t)$
  /// is signed even though $|V|$ is not.
  pub fn max_magnitude(&self) -> f64 {
    self.magnitude.iter().copied().fold(0.0, f64::max)
  }
}

impl Scene {
  /// Hodge-Laplace eigenmodes of a single grade -- standing-wave normal
  /// modes, $Delta u = lambda u$ -- of an arbitrary simplicial surface with
  /// the given geometry, filed into `fields` or `line_fields` through the
  /// same [`Self::field`] dispatch a raw Whitney basis function goes
  /// through: an eigenmode and a one-hot cochain differ only in where the
  /// cochain comes from (a dense eigensolve vs. a Kronecker delta), not in
  /// how it is reconstructed or displayed.
  ///
  /// The eigensolve is dense ($O(n^3)$ in the DOF count), so mesh resolution
  /// controls both fidelity and cost.
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
    let (eigenvals, _, eigenfuncs) =
      solve_hodge_laplace_evp(&WhitneyComplex::new(topology, &metric), grade, nmodes);

    for (i, (&lambda, col)) in eigenvals.iter().zip(eigenfuncs.column_iter()).enumerate() {
      let name = format!("mode {i} (grade {grade}, lambda = {lambda:.2})");
      let cochain = Cochain::new(grade, col.into_owned());
      Self::field(
        topology,
        coords,
        name,
        cochain,
        Some(lambda),
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
      let (mut f, mut l) = Self::spherical_harmonics_grade(&topology, &coords, grade, nmodes);
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

  /// The discrete spherical harmonics of a *single* grade on a shared sphere
  /// mesh: the fields one grade's (dense, $O(n^3)$) eigensolve contributes,
  /// split by their render mark. The unit the gallery computes lazily and
  /// memoizes -- the mesh is built once and passed in, so switching grade pays
  /// only for that grade's solve, and only the first time it is viewed.
  pub fn spherical_harmonics_grade(
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
  /// [`Self::spherical_harmonics`] without its (dense, $O(n^3)$) eigensolve.
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
    }];
    Self {
      topology,
      coords,
      fields,
      line_fields: Vec::new(),
    }
  }

  /// A full [`Scene`] carrying one grade's discrete spherical harmonics on the
  /// shared sphere mesh: the mesh (cloned in) plus the fields that grade's
  /// eigensolve contributes. The display unit the gallery memoizes per grade.
  pub fn sphere_grade(
    topology: &Complex,
    coords: &MeshCoords,
    grade: ExteriorGrade,
    nmodes: usize,
  ) -> Self {
    let (fields, line_fields) = Self::spherical_harmonics_grade(topology, coords, grade, nmodes);
    Self {
      topology: topology.clone(),
      coords: coords.clone(),
      fields,
      line_fields,
    }
  }

  /// Every Whitney basis function ("local shape function") of the standard
  /// reference cell of dimension `cell_dim`, visualized as nothing but a
  /// reconstructed field of a one-hot cochain: the basis function dual to a DOF
  /// simplex $sigma$ *is* the cochain $c_tau = delta_(sigma tau)$, so there is
  /// no separate "evaluate a Whitney form" code path here -- the interpolant
  /// and the sharp musical isomorphism are exactly the general machinery a
  /// solved field (an eigenmode, or a future loaded cochain) goes through too.
  /// One field per subsimplex of every grade $0..=$ `cell_dim`.
  pub fn whitney_basis(cell_dim: Dim) -> Self {
    use manifold::geometry::coord::mesh::standard_coord_complex;

    let (topology, coords) = standard_coord_complex(cell_dim);
    // The renderer is 3D-only; a reference cell of `dim < 3` embeds as
    // itself in the `z = 0` plane, same as `mesh3d::TriangleSurface3D` does
    // for any other flat surface. A no-op once `cell_dim >= 3`.
    let coords = coords.embed_euclidean(cell_dim.max(3));

    let mut fields = Vec::new();
    let mut line_fields = Vec::new();
    for grade in 0..=cell_dim {
      let ndofs = topology.nsimplices(grade);
      for (idof, dof_simp) in standard_subsimps(cell_dim, grade).enumerate() {
        let label = dof_simp.iter().map(|v| v.to_string()).collect::<String>();
        let name = format!("W^{grade}_{{{label}}}");

        let mut coeffs = na::DVector::zeros(ndofs);
        coeffs[idof] = 1.0;
        let cochain = Cochain::new(grade, coeffs);

        Self::field(
          &topology,
          &coords,
          name,
          cochain,
          None,
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
    name: String,
    cochain: Cochain,
    eigenvalue: Option<f64>,
    fields: &mut Vec<ScalarField>,
    line_fields: &mut Vec<LineField>,
  ) {
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
        });
      }
      1 => {
        let interpolant = WhitneyInterpolant::new(cochain, topology);
        let (direction, magnitude) = nodal_line_field(topology, coords, &interpolant);
        line_fields.push(LineField {
          name,
          grade: k,
          direction,
          magnitude,
          eigenvalue,
        });
      }
      reduced => todo!(
        "reduced grade {reduced} (n = {n}, k = {k}): an (n-k)-dimensional sheet \
         has no render mark yet; only reachable at n >= 4"
      ),
    }
  }
}

/// The reduced form at a point, in the reference frame of its cell: the Whitney
/// value $W c$ if its grade is already $<= n-k$, else its Hodge star, so the
/// result always has grade $min(k, n-k)$. The star is where -- and the only
/// place -- a metric enters the reduction.
fn reduced_form(form: MultiForm, metric: &RiemannianMetric) -> MultiForm {
  let n = form.dim();
  let k = form.grade();
  if k <= n - k {
    form
  } else {
    form.hodge_star(metric)
  }
}

/// The nodal average over incident cells of the reduced grade-1 field, sharped
/// and pushed forward into ambient coordinates: `direction` (normalized) and
/// `magnitude` ($|V|_g$) per vertex. See [`LineField`] for why a nodal average.
fn nodal_line_field(
  topology: &Complex,
  coords: &MeshCoords,
  interpolant: &WhitneyInterpolant,
) -> (Vec<na::Vector3<f64>>, Vec<f64>) {
  let nvertices = topology.skeleton_raw(0).len();
  let mut dir_sum = vec![na::Vector3::zeros(); nvertices];
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
      let vector = to_vec3(&ambient);
      mag_sum[v] += vector.norm();
      dir_sum[v] += vector;
      count[v] += 1;
    }
  }
  let direction = dir_sum
    .into_iter()
    .zip(&count)
    .map(|(d, &c)| {
      if c > 0 {
        d.try_normalize(0.0).unwrap_or_else(na::Vector3::zeros)
      } else {
        na::Vector3::zeros()
      }
    })
    .collect();
  let magnitude = mag_sum
    .into_iter()
    .zip(count)
    .map(|(s, c)| if c > 0 { s / f64::from(c) } else { 0.0 })
    .collect();
  (direction, magnitude)
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

/// A nalgebra vector of ambient coordinates, zero-padded to 3D.
fn to_vec3(v: &Vector) -> na::Vector3<f64> {
  na::Vector3::new(
    v[0],
    if v.len() > 1 { v[1] } else { 0.0 },
    if v.len() > 2 { v[2] } else { 0.0 },
  )
}
