//! The grade reduction: a $k$-form and its Hodge dual read as one datum, at
//! the grade $min(k, n-k)$ where the pair is smallest.
//!
//! The rule is one line and it is total over grade and dimension: reduce the
//! $k$-form to its reduced grade $min(k, n-k)$ through the Hodge star, then
//! read it at that grade. A reduced grade of 0 is a scalar density, a reduced
//! grade of 1 a genuine tangent line field. It is the discrete counterpart of
//! the classical identification of an $(n-1)$-form with a vector field and an
//! $n$-form with a density, stated once for every $n$ and $k$ rather than
//! twice for $n = 3$.
//!
//! The star needs a global volume form, not just a metric. Where it fires
//! ($k > n-k$) the reduction takes the cell's coherent orientation alongside
//! the metric: a cell's stored colex vertex order fixes a volume form only up
//! to sign, so a per-cell star returns $plus.minus$ the true density with the
//! sign flipping wherever colex disagrees with the manifold. That is why
//! [`reduction_sign`] is a separate argument rather than something the
//! reduction helps itself to.
//!
//! Where a form is single-valued decides how it is read. Only the tangential
//! part of a section is chart-independent, so a reduced-grade Whitney form is
//! discontinuous across cells and has no single value at a shared vertex. A
//! quantity on a skeleton simplex is therefore read through the trace $i^*$
//! ([`trace_value`]), exact by $H(dif)$ conformity and so single-valued with
//! no averaging. A quantity read in a cell's own frame is per cell and
//! genuinely disagrees with its neighbor, and averaging the second into the
//! first is a recovery, not the form.

use metric::Metric;
use metric::tensor::TensorExt;
use multialgebra::{ExteriorGrade, Tensor};
use regge::lengths::mesh::MeshLengthsSq;
use simplicial::{
  Sign,
  atlas::{Bary, MeshPoint},
  topology::{complex::Complex, handle::SimplexRef, role::Cell},
};

use crate::{Cochain, interpolate::interpolant::WhitneyInterpolant};

/// The reduced form of a $k$-form in the frame it is given in: the form itself
/// if its grade is already $<= n-k$, else its Hodge star, so the result always
/// has grade $min(k, n-k)$. The star is where, and the only place, a metric
/// enters the reduction.
///
/// `sign` is the cell's coherent orientation
/// ([`Orientation::sign`](simplicial::topology::orientation::Orientation::sign)),
/// and it is the second thing the star needs beyond the metric. A cell's
/// stored colex vertex order fixes a volume form only up to sign, so
/// $star: Lambda^n -> Lambda^0$ read cell by cell returns the density against
/// each cell's own arbitrary frame, $plus.minus$ the true one, flipping
/// wherever colex disagrees with the manifold's orientation. Multiplying by
/// `sign` is what makes the reduced value comparable across cells, and hence
/// what makes a top-grade density or an $(n-1)$-form's direction mean anything
/// globally. Below the star the sign is irrelevant, which is why it costs
/// nothing to pass it always: [`reduction_sign`] returns `Pos` there.
pub fn reduced_form(form: Tensor, metric: &Metric, sign: Sign) -> Tensor {
  let n = form.dim();
  let k = form.grade();
  if k <= n - k {
    form
  } else {
    form.star(metric, sign)
  }
}

/// The scalar a form reduces to.
///
/// The one rule, total over grade and dimension: a $0$-form is a scalar and is
/// read signed and metric-free. The manifold's top form is a pseudoscalar and
/// becomes a scalar through $star$; everything else reduces by its magnitude
/// $|omega|_g$, the direction being carried by the reduced form itself.
///
/// `signed` is `Some` exactly when the form is the manifold's own top form and
/// a coherent orientation fixes its volume form, so holding one is the proof
/// invariant 6 demands: only then is a signed density comparable across cells.
/// The caller states that condition, because only the caller knows whether the
/// form's own dimension is the manifold's (the trace onto a face is top on the
/// face while carrying no global sign). `None` is the honest magnitude.
pub fn scalarize(form: Tensor, metric: &Metric, signed: Option<Sign>) -> f64 {
  if form.grade() == 0 {
    return form.as_scalar();
  }
  match signed {
    Some(sign) => form.star(metric, sign).as_scalar(),
    None => form.norm(metric),
  }
}

/// The orientation factor [`reduced_form`] needs on one cell: `Pos` when the
/// reduction is the identity (no star, so no volume form and no orientation),
/// otherwise the cell's coherent orientation.
///
/// `None` is the one case where the reduction has no sign to be read with: the
/// star fires and the complex is not orientable, so there is no global volume
/// form to read it against. What no caller may do then is star per cell
/// against each cell's own colex frame: that returns $plus.minus$ the true
/// value with the sign flipping wherever colex disagrees with the manifold. A
/// scalar still has the honest magnitude ([`scalarize`] with `None`); a
/// direction has no such reading and there is nothing to return.
pub fn reduction_sign(topology: &Complex, cell: Cell, grade: ExteriorGrade) -> Option<Sign> {
  let n = topology.dim();
  if grade <= n - grade {
    return Some(Sign::Pos);
  }
  Some(topology.orientation()?.sign(cell))
}

/// [`reduction_sign`] under the caller's standing promise that the form was
/// admitted on an orientable mesh.
///
/// A form whose reduction needs the star must not be admitted on a mesh with
/// no coherent orientation, so holding one that reaches a reduction is already
/// the proof that the orientation exists. The refusal belongs where the form
/// is admitted, once, rather than at every reduction, which is why this panics
/// rather than widening every caller's signature.
pub fn admitted_reduction_sign(topology: &Complex, cell: Cell, grade: ExteriorGrade) -> Sign {
  reduction_sign(topology, cell, grade)
    .expect("a starred form is only admitted on an orientable mesh")
}

/// The trace-reduced scalar of a discrete form on a skeleton simplex: pull the
/// Whitney form back onto the simplex ([`Cochain::trace`]) and reduce the
/// traced form to a scalar with the simplex's own metric.
///
/// The trace is exact by tangential ($H(dif)$) conformity, so it is
/// single-valued across the cells incident at a shared simplex, no averaging.
/// The trace of a grade-$k$ form onto a $d$-simplex is a $k$-form on it, and
/// $Lambda^k (tau) = 0$ for $d < k$: a form reads an honest zero on a skeleton
/// below its grade. On the diagonal $d = k$ the trace is the constant top form
/// of density $c_tau \/ vol_g (tau)$, the simplex's cochain coefficient per
/// unit volume; above it ($d > k$) the trace varies and the norm reads the
/// magnitude.
///
/// The scalar is signed only where the sign is intrinsic, and its magnitude
/// otherwise, because a $k$-cochain value ($k >= 1$) is defined relative to the
/// simplex's orientation, which here is the colex bookkeeping convention. Two
/// cases escape it: $k = 0$, where a vertex has trivial orientation and the
/// value is a genuine scalar; and $k = d = n$, the manifold's own top form,
/// where the coherent [`Complex::orientation`] fixes the global density.
/// Nothing fixes the sign for $0 < k < n$: a manifold orientation induces
/// opposite co-orientations on an interior facet ($diff compose diff = 0$), so
/// it cannot reach the sub-top skeletons, and the honest reading there is the
/// magnitude. The direction a magnitude drops is not lost, it is the reduced
/// form's ([`reduced_form`]).
///
/// The geometry enters as [`MeshLengthsSq`] rather than as an embedding
/// because the metric asked for is a subsimplex's, which the edge lengths
/// answer at every grade.
pub fn trace_value(
  topology: &Complex,
  geometry: &MeshLengthsSq,
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
  let sub = Complex::unit(d);
  let interpolant = WhitneyInterpolant::new(cochain.trace(simplex), &sub);
  let cell = sub.cells().handle_iter().next().unwrap();
  let form = interpolant.eval(&MeshPoint::new(cell.idx(), bary.clone()));
  // A top form is the manifold's own only on a cell ($d = n$). On a face it is
  // top for the face while no coherent orientation reaches it, so it reduces by
  // magnitude like every other grade.
  let signed = (k == n && d == n).then(|| admitted_reduction_sign(topology, simplex.role(), k));
  scalarize(form, &geometry.simplex_metric(simplex), signed)
}
