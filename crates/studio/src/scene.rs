use ddf::{cochain::Cochain, whitney::interpolant::WhitneyInterpolant};
use manifold::{
  atlas::{Bary, MeshPoint},
  geometry::{coord::mesh::MeshCoords, metric::geometry::Geometry},
  topology::{complex::Complex, handle::SimplexRef, simplex::standard_subsimps},
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
/// A field's grade decides the mark it is drawn with, not a per-scene choice:
/// grade 0 sharps to a one-component multivector (a scalar) and colors the
/// surface ([`ScalarField`]); grade 1 sharps to a genuine tangent vector and
/// is drawn as an arrow ([`VectorField`]). Grades strictly between 1 and
/// `cell_dim - 1` (only possible once `cell_dim >= 4`) sharp to a
/// higher-arity multivector with no glyph implemented yet -- a future mark,
/// not a special case to work around.
pub struct Scene {
  pub topology: Complex,
  pub coords: MeshCoords,
  pub fields: Vec<ScalarField>,
  pub vector_fields: Vec<VectorField>,
}

/// A named scalar field on the surface: a 0-cochain, one value per vertex.
pub struct ScalarField {
  pub name: String,
  pub cochain: Cochain,
  /// The Hodge-Laplace eigenvalue $lambda$ this field is an eigenfunction of,
  /// when it is one. For the wave equation $partial_t^2 u + Delta u = 0$ this
  /// is the square of the mode's own temporal frequency, $omega = sqrt(lambda)$
  /// -- the standing wave for *this* mode, not a frequency picked by the
  /// viewer. `None` for a field that is not an eigenfunction (e.g. a raw
  /// Whitney basis function), which disables the standing-wave animation.
  pub eigenvalue: Option<f64>,
}

/// A named vector field on the surface, sampled at a set of points rather than
/// carried as a cochain: the musical isomorphism $sharp$ that turns a grade-1
/// [`exterior::MultiForm`] into a tangent [`exterior::MultiVector`] needs a
/// metric evaluated pointwise, so unlike a scalar field (exact from vertex
/// values alone, since Whitney 0-forms are barycentric-linear and the
/// rasterizer interpolates linearly too) a 1-form is reconstructed on a
/// lattice of sample points, one arrow per sample.
pub struct VectorField {
  pub name: String,
  pub samples: Vec<VectorSample>,
}

pub struct VectorSample {
  /// Ambient position of the sample point.
  pub position: na::Vector3<f64>,
  /// The sharped tangent vector at that point, in ambient coordinates.
  pub vector: na::Vector3<f64>,
}

impl VectorField {
  /// Largest arrow length, for glyph-size normalization.
  pub fn max_magnitude(&self) -> f64 {
    self
      .samples
      .iter()
      .map(|s| s.vector.norm())
      .fold(0.0, f64::max)
  }
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

impl Scene {
  /// Grade-0 Hodge-Laplace eigenmodes -- standing-wave normal modes, $Delta u =
  /// lambda u$ with $Delta = delta dif$ on functions, no multiplier block -- of
  /// an arbitrary simplicial surface with the given geometry.
  ///
  /// Neither the mesh nor its embedding are assumed to be a sphere: any 2D
  /// `Complex` with a `MeshCoords` realization goes in, so a scene is exactly
  /// as general as the underlying eigensolve. The spherical harmonics are one
  /// instantiation of this ([`Self::spherical_harmonics`]), not a special case
  /// baked into the solve.
  ///
  /// The eigensolve is dense ($O(n^3)$ in the vertex count $n$), so mesh
  /// resolution controls both fidelity and cost.
  pub fn eigenmodes(topology: Complex, coords: MeshCoords, nmodes: usize) -> Self {
    use formoniq::{
      problems::hodge_laplace::solve_hodge_laplace_evp, whitney_complex::WhitneyComplex,
    };

    let metric = coords.to_edge_lengths(&topology);
    let (eigenvals, _, eigenfuncs) =
      solve_hodge_laplace_evp(&WhitneyComplex::new(&topology, &metric), 0, nmodes);

    let fields = eigenvals
      .iter()
      .zip(eigenfuncs.column_iter())
      .enumerate()
      .map(|(i, (&lambda, col))| ScalarField {
        name: format!("mode {i} (lambda = {lambda:.2})"),
        cochain: Cochain::new(0, col.into_owned()),
        eigenvalue: Some(lambda),
      })
      .collect();

    Self {
      topology,
      coords,
      fields,
      vector_fields: Vec::new(),
    }
  }

  /// Laplace-Beltrami eigenfunctions on the unit sphere — the discrete
  /// spherical harmonics — on an icosphere of the given subdivision depth.
  pub fn spherical_harmonics(nsubdivisions: usize, nmodes: usize) -> Self {
    use manifold::gen::sphere::mesh_sphere_surface;

    let (topology, coords) = mesh_sphere_surface(nsubdivisions);
    Self::eigenmodes(topology, coords, nmodes)
  }

  /// Every Whitney basis function ("local shape function") of the standard
  /// reference cell of dimension `cell_dim`, visualized as nothing but a
  /// reconstructed field of a one-hot cochain: the basis function dual to a DOF
  /// simplex $sigma$ *is* the cochain $c_tau = delta_(sigma tau)$, so there is
  /// no separate "evaluate a Whitney form" code path here -- the interpolant
  /// and the sharp musical isomorphism are exactly the general machinery a
  /// solved field (an eigenmode, or a future loaded cochain) goes through too.
  /// One field per subsimplex of every grade $0..=$ `cell_dim`.
  pub fn whitney_basis(cell_dim: Dim, arrow_samples_per_cell_edge: usize) -> Self {
    use manifold::geometry::coord::mesh::standard_coord_complex;

    let (topology, coords) = standard_coord_complex(cell_dim);
    // The renderer is 3D-only; a reference cell of `dim < 3` embeds as
    // itself in the `z = 0` plane, same as `mesh3d::TriangleSurface3D` does
    // for any other flat surface. A no-op once `cell_dim >= 3`.
    let coords = coords.embed_euclidean(cell_dim.max(3));

    let mut fields = Vec::new();
    let mut vector_fields = Vec::new();
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
          arrow_samples_per_cell_edge,
          &mut fields,
          &mut vector_fields,
        );
      }
    }

    Self {
      topology,
      coords,
      fields,
      vector_fields,
    }
  }

  /// Reconstructs a cochain as the render mark its grade calls for, and files
  /// it into `fields` or `vector_fields` accordingly -- the one general entry
  /// point both a raw Whitney basis function ([`Self::whitney_basis`]) and a
  /// solved field arrive at.
  ///
  /// Grade 0 needs no reconstruction: a Whitney 0-form is exactly the
  /// barycentric-linear function through its vertex values, and the
  /// rasterizer already interpolates linearly, so the cochain's coefficients
  /// *are* the vertex values. Grade 1 sharps to a genuine tangent vector and
  /// is reconstructed via the [`WhitneyInterpolant`] on a barycentric lattice
  /// of every cell, one arrow per sample. Higher grades sharp to a
  /// higher-arity multivector, for which no glyph exists yet, and are
  /// skipped -- a future mark, not a special case to route around.
  #[allow(clippy::too_many_arguments)]
  fn field(
    topology: &Complex,
    coords: &MeshCoords,
    name: String,
    cochain: Cochain,
    eigenvalue: Option<f64>,
    arrow_samples_per_cell_edge: usize,
    fields: &mut Vec<ScalarField>,
    vector_fields: &mut Vec<VectorField>,
  ) {
    match cochain.grade() {
      0 => fields.push(ScalarField {
        name,
        cochain,
        eigenvalue,
      }),
      1 => {
        let interpolant = WhitneyInterpolant::new(cochain, topology);
        let lattice = barycentric_lattice(topology.dim(), arrow_samples_per_cell_edge);
        let samples = topology
          .cells()
          .handle_iter()
          .flat_map(|cell| {
            let metric = coords.cell_metric(cell);
            let interpolant = &interpolant;
            lattice.iter().map(move |bary| {
              let point = MeshPoint::new(cell.idx(), bary.clone());
              let vector = interpolant.eval(&point).sharp(&metric);
              let c = vector.coeffs();
              let vector = na::Vector3::new(c[0], if c.len() > 1 { c[1] } else { 0.0 }, 0.0);
              VectorSample {
                position: ambient_point(cell, coords, bary),
                vector,
              }
            })
          })
          .collect();
        vector_fields.push(VectorField { name, samples });
      }
      _ => {}
    }
  }
}

/// The ambient position of a barycentric point of `cell`: the affine
/// combination of its vertices' ambient coordinates. The extrinsic
/// counterpart of [`MeshPoint`], confined to the render layer like every
/// other use of an embedding.
fn ambient_point(cell: SimplexRef, coords: &MeshCoords, bary: &Bary) -> na::Vector3<f64> {
  let mut p = na::Vector3::zeros();
  for (i, &v) in cell.simplex().vertices.iter().enumerate() {
    let c = coords.coord(v);
    let ambient = na::Vector3::new(
      c[0],
      if c.len() > 1 { c[1] } else { 0.0 },
      if c.len() > 2 { c[2] } else { 0.0 },
    );
    p += bary[i] * ambient;
  }
  p
}

/// A uniform barycentric lattice on the standard `dim`-simplex: every
/// composition of `resolution` into `dim + 1` nonnegative parts, normalized to
/// weights summing to 1. Dimension-general -- unlike a hand-rolled triangle
/// grid, this is the same construction at every `dim`, including the
/// `resolution = 0` degenerate case (a single point, the barycenter).
fn barycentric_lattice(dim: Dim, resolution: usize) -> Vec<Bary> {
  fn recurse(
    remaining: usize,
    parts_left: usize,
    prefix: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
  ) {
    if parts_left == 1 {
      prefix.push(remaining);
      out.push(prefix.clone());
      prefix.pop();
      return;
    }
    for i in 0..=remaining {
      prefix.push(i);
      recurse(remaining - i, parts_left - 1, prefix, out);
      prefix.pop();
    }
  }
  let resolution = resolution.max(1);
  let mut compositions = Vec::new();
  recurse(resolution, dim + 1, &mut Vec::new(), &mut compositions);
  compositions
    .into_iter()
    .map(|weights| {
      na::DVector::from_iterator(
        dim + 1,
        weights.into_iter().map(|w| w as f64 / resolution as f64),
      )
      .into()
    })
    .collect()
}
