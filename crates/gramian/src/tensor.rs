//! The metric-dependent operations on a [`Tensor`].
//!
//! `multialgebra` is the metric-free half of the algebra: the wedge, the
//! contraction, the transfer, the duality and wedge pairings, all of which ask
//! for nothing but a vector space. Everything here needs a [`Metric`], and that
//! is the whole reason it lives one crate up rather than beside them.
//!
//! The split is invariant 5 made structural. An operation's crate now says what
//! it depends on, so a metric cannot leak into a signature that does not need
//! one by anyone forgetting to check.

use multialgebra::ExteriorGrade;
use multialgebra::{
  Factor, Matrix, Slot, Symmetry, Tensor, Variance,
  tensor::{Basis, Slots, factorwise_kronecker},
};
use multiindex::{MultiIndex, Sign};

use crate::{Gramian, Metric};

/// The Gramian measuring a slot of a variance: the metric tensor $g$ for the
/// contravariant side, its inverse $g^(-1)$ for the covariant one.
///
/// The choice is never to be made by hand.
pub fn variance_gramian(variance: Variance, metric: &Metric) -> &Gramian {
  match variance {
    Variance::Contravariant => metric.vector_gramian(),
    Variance::Covariant => metric.covector_gramian(),
  }
}

/// The Gramian measuring one slot, its variance choosing $g$ or $g^(-1)$.
///
/// # Panics
/// If the metric is not of the slot's own space.
pub fn slot_gramian(slot: &Slot, metric: &Metric) -> Gramian {
  assert_eq!(
    metric.dim(),
    slot.dim.index(),
    "a slot is measured by a metric of its own space"
  );
  Gramian::new_unchecked(
    slot
      .factor
      .induced_form(variance_gramian(slot.variance, metric).matrix()),
  )
}

/// The induced inner product on $times.circle_i F_i$: the Kronecker product of
/// the per-slot Gramians, in the same slot order as the components.
///
/// Each slot chooses $g$ or $g^(-1)$ by its own variance, so a mixed tensor is
/// measured correctly where a single global choice would not be.
pub fn tensor_gramian(slots: &[Slot], metric: &Metric) -> Gramian {
  let per_slot: Vec<Matrix> = slots
    .iter()
    .map(|slot| slot_gramian(slot, metric).matrix().clone())
    .collect();
  Gramian::new_unchecked(factorwise_kronecker(&per_slot))
}

/// The inner product on $Lambda^k$ induced by one on $V$:
/// $inner(e_I, e_J) = det [inner(e_i, e_j)]_(i in I, j in J)$.
pub fn multi_gramian(single: &Gramian, grade: impl Into<ExteriorGrade>) -> Gramian {
  Gramian::new_unchecked(Factor::alternating(grade).induced_form(single.matrix()))
}

/// The induced inner product on multivectors $Lambda^k V$: $Lambda^k g$.
pub fn multivector_gramian(metric: &Metric, grade: impl Into<ExteriorGrade>) -> Gramian {
  multi_gramian(metric.vector_gramian(), grade)
}

/// The induced inner product on multiforms $Lambda^k V^*$: $Lambda^k g^(-1)$.
pub fn multiform_gramian(metric: &Metric, grade: impl Into<ExteriorGrade>) -> Gramian {
  multi_gramian(metric.covector_gramian(), grade)
}

/// The metric inner product of two tensors of one shape.
///
/// A free function, not a method: an inner product is a bilinear map and
/// privileges neither argument, exactly as the metric-free pairings do not.
///
/// # Panics
/// If the shapes differ.
pub fn inner(left: &Tensor, right: &Tensor, metric: &Metric) -> f64 {
  assert_eq!(
    left.slots(),
    right.slots(),
    "an inner product is of one shape"
  );
  tensor_gramian(left.slots(), metric).inner(left.components(), right.components())
}

/// The metric operations on a [`Tensor`], as methods where the notation asks
/// for them.
///
/// A thin extension trait rather than free functions: the star and the musicals
/// are unary operators written postfix in the mathematics, and `omega.star(..)`
/// carries that where `star(&omega, ..)` would not. [`inner`] stays free,
/// being symmetric in its two arguments.
pub trait TensorExt {
  /// The induced inner product as a Gramian on this tensor's own basis.
  fn gramian(&self, metric: &Metric) -> Gramian;
  /// Magnitude $sqrt(abs(inner(v, v)))$, never NaN: on an indefinite metric the
  /// sign of [`inner`] carries the causal character separately.
  fn norm(&self, metric: &Metric) -> f64;
  /// The Hodge star on slot `which`, $star: Lambda^k -> Lambda^(n-k)$, defined
  /// by $alpha wedge star beta = inner(alpha, beta) vol$.
  ///
  /// The one operation not uniform over the slots: it needs a top degree to
  /// complement against, which only an alternating slot has.
  ///
  /// `orientation` is the handedness of this basis against the one the volume
  /// form is taken in, required because a metric alone does not determine a
  /// star. `Sign::Pos` is right for a standalone vector space and wrong for a
  /// mesh read cell by cell, each cell's frame being a gauge.
  ///
  /// # Panics
  /// If the slot is not alternating.
  fn hodge_star(&self, which: usize, metric: &Metric, orientation: Sign) -> Tensor;
  /// [`Self::hodge_star`] where there is only one slot to star.
  fn star(&self, metric: &Metric, orientation: Sign) -> Tensor;
  /// The musical isomorphism on every slot, each by its own variance: $flat$ on
  /// a contravariant slot, $sharp$ on a covariant one.
  ///
  /// Slot by slot rather than through one induced matrix, so a mixed tensor
  /// raises and lowers the right indices instead of applying one Gramian to all
  /// of them.
  fn musical(&self, metric: &Metric) -> Tensor;
}

impl TensorExt for Tensor {
  fn gramian(&self, metric: &Metric) -> Gramian {
    tensor_gramian(self.slots(), metric)
  }

  fn norm(&self, metric: &Metric) -> f64 {
    inner(self, self, metric).abs().sqrt()
  }

  fn hodge_star(&self, which: usize, metric: &Metric, orientation: Sign) -> Tensor {
    let slot = self.slots()[which];
    assert_eq!(
      slot.symmetry(),
      Symmetry::Alternating,
      "a symmetric or free slot has no top degree, so no Hodge star"
    );
    assert_eq!(metric.dim(), slot.dim.index());

    let gramian = variance_gramian(slot.variance, metric);
    let volume = multi_gramian(gramian, slot.dim).matrix()[(0, 0)]
      .abs()
      .sqrt();
    let weighted = self.apply_to_slot(which, &slot.factor.induced(gramian.matrix()));

    let mut starred_slots: Slots = self.slots().iter().copied().collect();
    starred_slots[which] = slot.with_degree(slot.dim - slot.degree());
    let mut starred = Tensor::zero(starred_slots);

    for (component, basis) in weighted.basis_iter() {
      let (sign, complement) = basis[which]
        .as_mono()
        .expect("a star is of an alternating slot")
        .complement_signed(slot.dim.index());
      let mut target: Basis = basis;
      target[which] = MultiIndex::Mono(complement);
      let flat = starred.flat_index(&target);
      starred.components_mut()[flat] = orientation.as_f64() * sign.as_f64() * component / volume;
    }
    starred
  }

  fn star(&self, metric: &Metric, orientation: Sign) -> Tensor {
    self.hodge_star(0, metric, orientation)
  }

  fn musical(&self, metric: &Metric) -> Tensor {
    let mut musical = self.clone();
    for which in 0..self.slots().len() {
      let slot = self.slots()[which];
      let matrix = slot
        .factor
        .induced(variance_gramian(slot.variance, metric).matrix());
      musical = musical.apply_to_slot(which, &matrix).dualize_slot(which);
    }
    musical
  }
}
