//! The grade reduction: a $k$-form read as the scalar or the vector a consumer
//! can eat.
//!
//! The rule is one line and it is total over grade and dimension: reduce the
//! $k$-form to its *reduced grade* $min(k, n-k)$ through the Hodge star, then
//! dispatch on that. A reduced grade of 0 is a scalar density, a reduced grade
//! of 1 a genuine tangent line field. Nothing here decides what to *do* with
//! the result, which is what lets a viewer's mark and a file's data array be
//! the same reading of the same field.
//!
//! **The star needs a global volume form, not just a metric.** Where it fires
//! ($k > n-k$) the reduction takes the cell's coherent orientation alongside
//! the metric: a cell's stored colex vertex order fixes a volume form only up
//! to sign, so a per-cell star returns $plus.minus$ the true density with the
//! sign flipping wherever colex disagrees with the manifold. That is the
//! parent's invariant 6, and it is why [`reduction_sign`] is a separate
//! argument rather than something the reduction helps itself to.
//!
//! **Where a field is single-valued decides how it is read.** Only the
//! tangential part of a section is chart-independent, so a reduced-grade
//! Whitney form is discontinuous across cells and has no single value at a
//! shared vertex. A quantity on a *skeleton* simplex is therefore read through
//! the **trace** $i^*$ ([`trace_value`]), exact by $H(dif)$ conformity and so
//! single-valued with no averaging; a quantity read in a *cell's own* frame
//! ([`reduced_value`]) is per cell and genuinely disagrees with its neighbour.
//! Averaging the second into the first is a recovery, and presenting a recovery
//! as the field is the thing to avoid.

use derham::{cochain::Cochain, interpolate::interpolant::WhitneyInterpolant};
use exterior::{ExteriorGrade, MultiForm};
use gramian::Metric;
use simplicial::linalg::Vector;
use simplicial::{
  Sign,
  atlas::{Bary, MeshPoint},
  geometry::coord::mesh::MeshCoords,
  topology::{
    complex::Complex,
    handle::{SimplexIdx, SimplexRef},
    role::Cell,
    simplex::Simplex,
  },
};

use crate::bake::CellCorner;

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
pub fn reduced_form(form: MultiForm, metric: &Metric, sign: Sign) -> MultiForm {
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
pub fn scalarize(form: MultiForm, metric: &Metric, signed: Option<Sign>) -> f64 {
  if form.grade() == 0 {
    return form.coeffs()[0];
  }
  match signed {
    Some(sign) => form.hodge_star(metric).coeffs()[0] * sign.as_f64(),
    None => form.norm(metric),
  }
}

/// The orientation factor [`reduced_form`] needs on one cell: `Pos` when the
/// reduction is the identity (no star, so no volume form and no orientation),
/// otherwise the cell's coherent orientation.
///
/// Panics on a non-orientable complex, and the contract that makes it sound is
/// the *caller's*: a field whose reduction needs the star must not be admitted
/// on a mesh with no coherent orientation, so holding a field that reaches here
/// is already the proof that the orientation exists. The refusal belongs where
/// the field is admitted, once, rather than at every draw -- which is why this
/// is a panic and not a `Result`.
///
/// A consumer that cannot make that promise up front asks
/// [`Complex::orientation`](simplicial::topology::complex::Complex::orientation)
/// itself and passes `None` to [`scalarize`], which is the honest magnitude.
/// What no consumer may do is star per cell against each cell's own colex
/// frame: that returns $plus.minus$ the true density with the sign flipping
/// wherever colex disagrees with the manifold, which is plausible on screen and
/// wrong.
pub fn reduction_sign(topology: &Complex, cell: Cell, grade: ExteriorGrade) -> Sign {
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
pub fn surface_corner_values(
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
pub fn segment_colors(
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
pub fn point_colors(topology: &Complex, coords: &MeshCoords, cochain: &Cochain) -> Vec<f64> {
  let bary = Bary::new(Vector::from_element(1, 1.0));
  topology
    .skeleton(0)
    .handle_iter()
    .map(|vertex| trace_value(topology, coords, cochain, vertex, &bary))
    .collect()
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
pub fn surface_corner_heights(
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
pub fn nodal_heights(topology: &Complex, coords: &MeshCoords, cochain: &Cochain) -> Vec<f64> {
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
pub fn reduced_value(
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
pub fn trace_value(
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

#[cfg(test)]
mod tests {
  use super::*;
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
}
