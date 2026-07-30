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
use multiindex::{MultiIndex, Sign, Word};

use crate::Metric;

/// A metric read as the $"Sym"^2$ element it is, and back.
///
/// A symmetric bilinear form on $V$ *is* an element of $"Sym"^2(V^*)$, and one
/// on $V^*$ an element of $"Sym"^2(V)$, so the variance is the slot's variance
/// and nothing is chosen here. The matrix stays the representation; this is the
/// identity, computed on demand.
///
/// The two directions carry the **symmetric multiplicity**, and it is never
/// written as a number. The $"Sym"$ basis is unnormalized, $x^alpha$ summing
/// over all $k!$ orderings, so the orderings that coincide on a repeated symbol
/// are summed rather than assigned and $norm(x^alpha)^2 = alpha!$. A packed
/// component and its matrix entry therefore differ by $alpha!$, which for
/// $"Sym"^2$ is $2$ on the diagonal and $1$ off it. Rather than spell that out,
/// the weights are read off [`Factor::induced_form`] of the identity, whose
/// diagonal is exactly $alpha!$: one convention, stated in one place, and the
/// same code at any degree.
impl Metric {
  /// The $"Sym"^2$ basis of this metric's space, and the multiplicity $alpha!$
  /// of each of its elements.
  fn sym2(&self) -> (Slot, Vec<f64>) {
    let slot = Slot::new(Factor::symmetric(2), self.variance(), self.dim());
    let multiplicity = Factor::symmetric(2)
      .induced_form(&Matrix::identity(self.dim(), self.dim()))
      .diagonal()
      .iter()
      .copied()
      .collect();
    (slot, multiplicity)
  }

  /// This metric as an element of $"Sym"^2$ of its own space.
  ///
  /// The inverse of [`Self::from_tensor`], up to floating point.
  pub fn tensor(&self) -> Tensor {
    let (slot, multiplicity) = self.sym2();
    let components = slot
      .basis()
      .zip(&multiplicity)
      .map(|(index, alpha_factorial)| {
        let symbols = index.word();
        self.matrix()[(symbols[0], symbols[1])] / alpha_factorial
      })
      .collect::<Vec<_>>();
    Tensor::new(Slots::from_iter([slot]), components.into())
  }

  /// The metric a symmetric 2-tensor is.
  ///
  /// Reads the entries off [`Tensor::to_free`] rather than reapplying the
  /// multiplicity by hand, so the two directions cannot disagree about the
  /// convention: the free power *is* the matrix, laid out over $[n; 2]$.
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

    let dim = slot.dim.index();
    let free = tensor.to_free();
    let matrix = Matrix::from_fn(dim, dim, |i, j| {
      free.components()[free.flat_index(&[MultiIndex::Word(Word::new(dim, [i, j]))])]
    });
    Self::new(slot.variance, matrix)
  }
}

/// The metric induced on $times.circle_i F_i$: the Kronecker product of the
/// per-slot metrics, in the same slot order as the components.
///
/// Each slot is measured by [`Metric::measuring`] against its own variance, so
/// a mixed tensor is measured correctly where a single global choice would not
/// be.
pub fn tensor_metric(slots: &[Slot], metric: &Metric) -> Metric {
  let per_slot: Vec<Matrix> = slots
    .iter()
    .map(|slot| metric.on_slot(slot).matrix().clone())
    .collect();
  Metric::new_unchecked(Variance::Covariant, factorwise_kronecker(&per_slot))
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
  tensor_metric(left.slots(), metric).inner(left.components(), right.components())
}

/// The metric operations on a [`Tensor`], as methods where the notation asks
/// for them.
///
/// A thin extension trait rather than free functions: the star and the musicals
/// are unary operators written postfix in the mathematics, and `omega.star(..)`
/// carries that where `star(&omega, ..)` would not. [`inner`] stays free,
/// being symmetric in its two arguments.
pub trait TensorExt {
  /// The induced metric on this tensor's own basis.
  fn induced_metric(&self, metric: &Metric) -> Metric;
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
  fn induced_metric(&self, metric: &Metric) -> Metric {
    tensor_metric(self.slots(), metric)
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
