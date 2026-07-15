use ddf::cochain::Cochain;
use manifold::{geometry::coord::mesh::MeshCoords, topology::complex::Complex};

/// A renderable scene: a simplicial surface together with scalar fields on it.
///
/// The seam between the formoniq engine and the viewer. A scene is produced
/// either by a live solve (the native path here) or, later, by deserializing a
/// cached bundle; the viewer neither knows nor cares which. It carries the
/// engine's own types — topology, coordinates, cochains — rather than a lossy
/// export format, so the coloring, displacement and mode selection stay
/// decisions of the viewer.
pub struct Scene {
  pub topology: Complex,
  pub coords: MeshCoords,
  pub fields: Vec<ScalarField>,
}

/// A named scalar field on the surface: a 0-cochain, one value per vertex.
pub struct ScalarField {
  pub name: String,
  pub cochain: Cochain,
  /// The Hodge-Laplace eigenvalue $lambda$ this field is an eigenfunction of.
  /// For the wave equation $partial_t^2 u + Delta u = 0$ this is the square of
  /// the mode's own temporal frequency, $omega = sqrt(lambda)$ -- the standing
  /// wave for *this* mode, not a frequency picked by the viewer.
  pub eigenvalue: f64,
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
        eigenvalue: lambda,
      })
      .collect();

    Self {
      topology,
      coords,
      fields,
    }
  }

  /// Laplace-Beltrami eigenfunctions on the unit sphere — the discrete
  /// spherical harmonics — on an icosphere of the given subdivision depth.
  pub fn spherical_harmonics(nsubdivisions: usize, nmodes: usize) -> Self {
    use manifold::dim3::mesh_sphere_surface;

    let (topology, coords) = mesh_sphere_surface(nsubdivisions).into_coord_complex();
    Self::eigenmodes(topology, coords, nmodes)
  }
}
