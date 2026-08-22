//! The render readings of a reduced form: the colormap value at a rendered
//! corner, and the displacement height of the surface.
//!
//! The grade reduction itself is [`derham::reduce`]: a $k$-form read at its
//! reduced grade $min(k, n-k)$, a scalar density at 0 and a tangent line
//! field at 1. What is here is what a consumer does with it, per rendered
//! corner rather than per simplex.
//!
//! Where a field is single-valued decides how it is read. Only the
//! tangential part of a section is chart-independent, so a reduced-grade
//! Whitney form is discontinuous across cells and has no single value at a
//! shared vertex. A quantity on a skeleton simplex is therefore read through
//! the trace $i^*$ ([`trace_value`]), exact by $H(dif)$ conformity and so
//! single-valued with no averaging. A quantity read in a cell's own frame
//! (the nodal recovery of [`nodal_heights`]) is per cell and genuinely
//! disagrees with its neighbor.
//! Averaging the second into the first is a recovery, and presenting a recovery
//! as the field is the thing to avoid.

use derham::Cochain;
use derham::interpolate::interpolant::WhitneyInterpolant;
use derham::reduce::{admitted_reduction_sign, scalarize, trace_value};
use regge::coord::mesh::MeshCoords;
use simplicial::linalg::Vector;
use simplicial::{
  atlas::{Bary, MeshPoint},
  topology::{complex::Complex, simplex::Simplex},
};

/// The colormap range of a per-corner value stream, for normalization. Falls
/// back to a unit range on an empty or constant field so the viewer never
/// normalizes by a zero span.
pub fn corner_bounds(values: &[f64]) -> (f32, f32) {
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

/// The colormap scalar at every corner of a stream of render primitives, `N`
/// values per primitive in the primitive's own corner order: the one rule that
/// colors the fill, the wireframe and the points alike.
///
/// A primitive is the global vertex tuple of an $(N-1)$-simplex of the mesh,
/// wound in whatever order the rasterizer draws it, and its corner's value is
/// the [`trace_value`] of the field on that simplex at the corner's barycentric
/// indicator. So the triangles of the fill read the 2-skeleton, the segments of
/// the wireframe the 1-skeleton and the vertices the 0-skeleton, one body at
/// three grades rather than three techniques.
///
/// The trace is single-valued across the cells incident at a shared simplex by
/// tangential conformity, so no per-corner cell disambiguation and no averaging
/// enters: a simplex the form vanishes on colors to zero because its trace is
/// zero, and a grade above $N-1$ traces to zero on every one of them, leaving
/// that skeleton honestly uncolored. At $n = N-1$ the simplex is a cell and this
/// reproduces the reduced-grade density exactly; above it the simplex's own
/// trace is read, never a value borrowed from an incident cell.
pub fn corner_values<const N: usize>(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  primitives: impl IntoIterator<Item = [u32; N]>,
) -> Vec<f64> {
  let geometry = coords.to_edge_lengths_sq(topology);
  let mut values = Vec::new();
  for corners in primitives {
    // A `Simplex` is the colex-sorted vertex set while a primitive carries its
    // winding, so a corner is named by its vertex rather than by its position.
    let mut sorted = corners;
    sorted.sort_unstable();
    let simplex = topology
      .skeleton(N - 1)
      .handle_by_simplex(&Simplex::new(sorted.iter().map(|&v| v as usize).collect()));
    values.extend(corners.iter().map(|v| {
      let corner = sorted.iter().position(|u| u == v).unwrap();
      let mut weights = Vector::zeros(N);
      weights[corner] = 1.0;
      trace_value(topology, &geometry, cochain, simplex, &Bary::new(weights))
    }));
  }
  values
}

/// The surface's displacement height per rendered corner, by the strategy the
/// field's own continuity calls for, the same reduction that picks the mark,
/// asked once more.
///
/// $cal(W) Lambda^0$ is $P_1$ and continuous, so a vertex has one value and the
/// nodal recovery below is the field: the surface displaces as one connected
/// sheet, exactly. $cal(W) Lambda^n$ is $P_0$: the reduced density is constant
/// on each cell and genuinely discontinuous across it, so there is no
/// continuous height to displace by, and the nodal average would invent one,
/// showing a $P_0$ field flat-shaded in color and smooth in shape, two
/// contradictory claims about one field in one frame. Instead each cell
/// displaces rigidly, by its own constant value.
///
/// A rigidly displaced surface tears, and that is the point. The cells
/// separate by exactly the jump in the density across their shared face, so the
/// discontinuity becomes visible space rather than being smoothed away, and the
/// surface visibly re-closes under refinement as the jump vanishes. It is the
/// displacement counterpart of reading the colormap per corner.
///
/// The direction stays the vertex normal, so a cell translates rather than
/// moving exactly along its own normal. On a resolved mesh the two differ by
/// the normal's variation across one cell. What this costs is stated in
/// [`derham::reduce::reduced_form`]'s terms: $d_K n_K$ with the orientation-induced cell normal
/// would be invariant under the orientation gauge outright, whereas the
/// embedding's outward normal fixes that gauge only up to one global sign,
/// the same ambiguity an eigenvector already carries, and not the per-cell
/// scrambling that made the star wrong.
pub fn surface_corner_heights(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  triangles: &[[u32; 3]],
) -> Vec<f64> {
  let n = topology.dim();
  let k = cochain.grade();
  if k > n - k {
    // Discontinuous: the per-corner read is already constant on each cell, so
    // the honest colormap value and the rigid height are the same number.
    return corner_values(topology, coords, cochain, triangles.iter().copied());
  }
  let nodal = nodal_heights(topology, coords, cochain);
  triangles
    .iter()
    .flat_map(|t| t.map(|v| nodal[v as usize]))
    .collect()
}

/// The per-vertex displacement height: the reduced field's nodal average over
/// the cells incident at each vertex.
///
/// Exact for a continuous field ($cal(W) Lambda^0$), where the incident cells
/// already agree and this is the identity on the DOFs; a smoothing recovery
/// wherever the reduction stars, which is why the surface does not use it there
/// (see [`surface_corner_heights`]). It stays the height of the segment marks
/// at every grade: the 1-skeleton is shared between cells and cannot tear
/// without duplicating it, so the wireframe rides the continuous recovery and
/// reads as the reference the fill's torn cells sit around.
pub fn nodal_heights(topology: &Complex, coords: &MeshCoords, cochain: &Cochain) -> Vec<f64> {
  let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);
  let nvertices = topology.skeleton(0).len();
  let mut sum = vec![0.0; nvertices];
  let mut count = vec![0u32; nvertices];
  let k = cochain.grade();
  for cell in topology.cells().handle_iter() {
    let metric = coords.cell_metric(cell);
    // The readout in the cell's own chart: the signed density for a reduced
    // grade of 0, the magnitude for a reduced grade of 1, the direction there
    // being the glyph and particle marks' to carry.
    let signed = (k == topology.dim()).then(|| admitted_reduction_sign(topology, cell, k));
    for (ilocal, &v) in cell.simplex().vertices.iter().enumerate() {
      let mut weights = Vector::zeros(cell.nvertices());
      weights[ilocal] = 1.0;
      let point = MeshPoint::new(cell.idx(), weights.into());
      sum[v] += scalarize(interpolant.eval(&point), &metric, signed);
      count[v] += 1;
    }
  }
  sum
    .into_iter()
    .zip(count)
    .map(|(s, c)| if c > 0 { s / f64::from(c) } else { 0.0 })
    .collect()
}
