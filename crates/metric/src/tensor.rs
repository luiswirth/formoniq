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
  tensor::{Basis, Slots, apply_factorwise, factorwise_kronecker},
};
use multiindex::{MonoIndex, MultiIndex, Repetition, Sign};

use crate::Metric;

/// A metric read as the $"Sym"^2$ element it is, and back.
///
/// A symmetric bilinear form on $V$ is an element of $"Sym"^2(V^*)$, and one
/// on $V^*$ an element of $"Sym"^2(V)$, so the variance is the slot's variance
/// and nothing is chosen here. The matrix stays the representation; this is the
/// identity, computed on demand.
///
/// The reciprocal basis is a metric's home. A metric's components there are
/// its matrix entries, exactly, with no weight anywhere: $g_(i j)$ is the
/// component at $x^(e_i + e_j)$. That is not a convenience: it is what a
/// bilinear form is, an object that eats vectors, hence a dual one, and it is
/// why both directions here go through [`Tensor::reciprocal`] and
/// [`Tensor::from_reciprocal`] rather than touching the stored components. Read
/// in the multiplicative basis the same metric carries $1 \/ alpha!$ on its
/// diagonal, which is the factor the earlier form of this code spelled out by
/// hand.
impl Metric {
  /// The $"Sym"^2$ slot a metric of this space and variance occupies.
  fn sym2(&self) -> Slot {
    Slot::new(Factor::symmetric(2), self.variance(), self.dim())
  }

  /// This metric as an element of $"Sym"^2$ of its own space.
  ///
  /// The inverse of [`Self::from_tensor`], up to floating point.
  pub fn tensor(&self) -> Tensor {
    let slot = self.sym2();
    let reciprocal: Vec<f64> = slot
      .basis()
      .map(|index| {
        let symbols = index.word();
        self.matrix()[(symbols[0], symbols[1])]
      })
      .collect();
    Tensor::from_reciprocal(Slots::from_iter([slot]), reciprocal.into())
  }

  /// The metric a symmetric 2-tensor is.
  ///
  /// # Panics
  /// If the tensor is not a single symmetric slot of degree $2$, or if the form
  /// it carries is degenerate.
  pub fn from_tensor(tensor: &Tensor) -> Self {
    let [slot] = tensor.slots() else {
      panic!("a metric is one slot");
    };
    assert_eq!(
      slot.symmetry(),
      Symmetry::Symmetric,
      "a metric is symmetric"
    );
    assert_eq!(slot.degree().index(), 2, "a metric is of degree two");

    // One slot, so the flat component index is the basis element's colex rank.
    let reciprocal = tensor.reciprocal();
    let matrix = Matrix::from_fn(slot.dim.index(), slot.dim.index(), |i, j| {
      let index = MonoIndex::new(Repetition::Allowed, [i.min(j), i.max(j)]);
      reciprocal[index.rank()]
    });
    Self::new(slot.variance, matrix)
  }
}

/// The Gram matrix induced on $times.circle_i F_i$: the Kronecker product of the
/// per-slot metrics, in the same slot order as the components.
///
/// Each slot is measured by [`Metric::measuring`] against its own variance, so a
/// mixed tensor is measured correctly where a single global choice would not be.
///
/// A bare matrix, and not a [`Metric`], for exactly that reason. A mixed tensor's
/// Gramian has factors from $g$ and from $g^(-1)$, so no single variance
/// describes it, and a constructor that supplied one would be guessing. Nothing
/// about this object is a metric on the underlying space. It is the Gram matrix
/// of a basis, which is all an inner product needs.
///
/// Formed, and usually not what you want. The Gram matrix of a product is
/// the product of the Gram matrices, so measuring anything applies
/// [`per_slot_gramians`] slot by slot and never builds this: see
/// [`apply_factorwise`]. Reach for the formed matrix only where a matrix is the
/// deliverable.
pub fn tensor_gramian(slots: &[Slot], metric: &Metric) -> Matrix {
  factorwise_kronecker(&per_slot_gramians(slots, metric))
}

/// The Gram matrix of each slot separately, in slot order: the factors of
/// [`tensor_gramian`], and the form the measuring operations actually use.
///
/// Each slot is measured by [`Metric::measuring`] against its own variance,
/// which is the whole reason these stay apart rather than collapsing into one
/// induced metric: on a mixed tensor some factors come from $g$ and some from
/// $g^(-1)$.
pub fn per_slot_gramians(slots: &[Slot], metric: &Metric) -> Vec<Matrix> {
  slots
    .iter()
    .map(|slot| metric.on_slot(slot).matrix().clone())
    .collect()
}

/// The metric induced on multivectors $Lambda^k V$: $Lambda^k g$.
pub fn multivector_metric(metric: &Metric, grade: impl Into<ExteriorGrade>) -> Metric {
  metric
    .measuring(Variance::Contravariant)
    .induced(Factor::alternating(grade))
}

/// The metric induced on multiforms $Lambda^k V^*$: $Lambda^k g^(-1)$.
pub fn multiform_metric(metric: &Metric, grade: impl Into<ExteriorGrade>) -> Metric {
  metric
    .measuring(Variance::Covariant)
    .induced(Factor::alternating(grade))
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
  // The Gramian of a product is the product of the Gramians, so it is applied
  // slot by slot and never formed: see `tensor_gramian` for what that costs.
  let dims: Vec<usize> = left.slots().iter().map(Slot::multidim).collect();
  let measured = apply_factorwise(
    &per_slot_gramians(left.slots(), metric),
    &dims,
    right.components(),
  );
  left.components().dot(&measured)
}

/// The metric operations on a [`Tensor`], as methods where the notation asks
/// for them.
///
/// A thin extension trait rather than free functions: the star and the musicals
/// are unary operators written postfix in the mathematics, and `omega.star(..)`
/// carries that where `star(&omega, ..)` would not. [`inner`] stays free,
/// being symmetric in its two arguments.
pub trait TensorExt {
  /// The Gram matrix on this tensor's own basis, each slot measured by its own
  /// variance. Not a [`Metric`]: see [`tensor_gramian`].
  fn gramian(&self, metric: &Metric) -> Matrix;
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
  /// raises and lowers the right indices instead of applying one metric to all
  /// of them.
  fn musical(&self, metric: &Metric) -> Tensor;
}

impl TensorExt for Tensor {
  fn gramian(&self, metric: &Metric) -> Matrix {
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

    let measuring = metric.measuring(slot.variance);
    let volume = measuring.induced(Factor::alternating(slot.dim)).matrix()[(0, 0)]
      .abs()
      .sqrt();
    let weighted = self.apply_to_slot(which, &slot.factor.induced(measuring.matrix()));

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
        .induced(metric.measuring(slot.variance).matrix());
      musical = musical.apply_to_slot(which, &matrix).dualize_slot(which);
    }
    musical
  }
}
