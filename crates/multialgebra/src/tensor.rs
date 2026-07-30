//! A tensor product of alternating and symmetric factors over one space.

use multiindex::{Degree, Dim, MonoIndex, MultiIndex, Sign, Word};

use crate::{Factor, Matrix, Slot, Symmetry, Variance, Vector, basis_multiplicity, induced};

/// The slots of a [`Tensor`], inline: one or two slots is the common case, and
/// a heap allocation per element in the assembly loop would cost more than the
/// arithmetic.
pub type Slots = tinyvec::TinyVec<[Slot; 4]>;
/// The stride of each factor in the flat component index.
pub type Strides = tinyvec::TinyVec<[usize; 4]>;
/// A basis element of a [`Tensor`]: one multi-index per factor.
pub type Basis = tinyvec::TinyVec<[MultiIndex; 2]>;

/// An element of $times.circle_i F_i (V)$, each $F_i$ an alternating or
/// symmetric power, all over one space of dimension $n$ and one [`Variance`].
///
/// Components live on the product of the per-factor bases -- a
/// [`Combination`](multiindex::Combination)-shaped index on an alternating
/// factor, a [`Composition`](multiindex::Composition)-shaped one on a symmetric
/// factor, both spelled [`MonoIndex`] -- in colexicographic order, with the
/// **last factor running fastest**.
///
/// The convention does not follow from functoriality, which holds under any
/// consistent reordering, so it is stated and tested as its own law.
///
/// Every operation is uniform over the factors, reading the symmetry only through
/// [`Factor`], except the Hodge star: a symmetric power has no top degree to
/// complement against.
#[derive(Debug, Clone)]
pub struct Tensor {
  slots: Slots,
  /// Derived from the slots, stored so that the single-slot case indexes as
  /// `components[rank]` with no arithmetic at all.
  strides: Strides,
  components: Vector,
}

/// The dimension of $times.circle_i F_i (RR^n)$: the product of the per-factor
/// dimensions.
///
/// Total at the trivial ends: an empty product of factors is the scalars, and a
/// factor naming a trivial space sends the whole product to zero.
pub fn tensor_dim(slots: &[Slot]) -> usize {
  slots.iter().map(Slot::multidim).product()
}

/// The degree a factor contributes to a Koszul sign: its own when alternating,
/// zero when symmetric, since a symmetric factor is even and commutes freely.
///
/// A degree off the range names the trivial space, which contributes nothing to
/// a sign and zeroes the product anyway, so it counts as zero rather than
/// trapping.
fn alternating_degree(slot: &Slot) -> usize {
  match slot.symmetry() {
    Symmetry::Alternating => slot.degree().get().max(0) as usize,
    // A symmetric slot is even, and a free one has no symmetry under a swap at
    // all, so neither contributes to a Koszul sign.
    Symmetry::Symmetric | Symmetry::Free => 0,
  }
}

/// The stride of each factor in the flat component index, last factor running
/// fastest: $s_i = product_(j > i) dim F_j$.
pub fn tensor_strides(slots: &[Slot]) -> Strides {
  let mut strides = Strides::from_iter(std::iter::repeat_n(1, slots.len()));
  for i in (0..slots.len().saturating_sub(1)).rev() {
    strides[i] = strides[i + 1] * slots[i + 1].multidim();
  }
  strides
}

impl Tensor {
  pub fn new(slots: impl Into<Slots>, components: Vector) -> Self {
    let slots = slots.into();
    assert_eq!(
      components.len(),
      tensor_dim(&slots),
      "component count must be the dimension of the tensor product"
    );
    let strides = tensor_strides(&slots);
    Self {
      slots,
      strides,
      components,
    }
  }

  pub fn zero(slots: impl Into<Slots>) -> Self {
    let slots = slots.into();
    let components = Vector::zeros(tensor_dim(&slots));
    Self::new(slots, components)
  }

  /// The scalar $lambda in RR$: the empty tensor product, of no slots.
  pub fn scalar(value: f64, _dim: impl Into<Dim>) -> Self {
    Self::new(Slots::new(), Vector::from_element(1, value))
  }

  /// The value of a scalar, the inverse of [`Self::scalar`].
  ///
  /// A tensor of no slots is a number, and this is that identification. Note
  /// that a tensor of *degree* zero in some slot is also one-dimensional, and
  /// this accepts it too: $Lambda^0 = "Sym"^0 = RR$, so the slots are there but
  /// the space they span is the ground field either way.
  ///
  /// # Panics
  /// If the tensor is not one-dimensional.
  pub fn as_scalar(&self) -> f64 {
    assert_eq!(
      self.components.len(),
      1,
      "a scalar is a one-dimensional tensor"
    );
    self.components[0]
  }

  /// A grade-one element of the space itself, where the two symmetries coincide.
  pub fn line(components: Vector, variance: Variance) -> Self {
    let dim = components.len();
    Self::new(Self::one_alternating(1, variance, dim), components)
  }

  /// A multiform: one covariant alternating slot, $Lambda^k V^*$.
  pub fn multiform(components: Vector, dim: impl Into<Dim>, grade: impl Into<Degree>) -> Self {
    Self::new(
      Self::one_alternating(grade, Variance::Covariant, dim),
      components,
    )
  }

  /// A multivector: one contravariant alternating slot, $Lambda^k V$.
  pub fn multivector(components: Vector, dim: impl Into<Dim>, grade: impl Into<Degree>) -> Self {
    Self::new(
      Self::one_alternating(grade, Variance::Contravariant, dim),
      components,
    )
  }

  /// The decomposable blade of a frame: the wedge
  /// $v_1 wedge dots.c wedge v_k$ of the columns of `frame`, in one alternating
  /// slot of the given variance.
  ///
  /// $Lambda^k$ of the frame read as a map $RR^k -> RR^n$, whose single column
  /// is the $binom(n,k)$ minors in colex, so this is functoriality rather than a
  /// fold of [`wedge`](Self::wedge) over the columns: the blade of a frame is
  /// the image of the generator of $Lambda^k (RR^k) tilde.eq RR$.
  ///
  /// Zero exactly when the columns are dependent, which is the statement that a
  /// blade decomposes a subspace, and total at $k = 0$, where the empty wedge is
  /// the unit.
  ///
  /// The variance is the frame's own: a frame of tangent vectors gives a
  /// multivector, one of covectors a multiform. Nothing here derives it
  /// (invariant 4), so it is stated.
  pub fn blade_of(frame: &Matrix, variance: Variance) -> Self {
    let (dim, grade) = (frame.nrows(), frame.ncols());
    Self::new(
      Self::one_alternating(grade, variance, dim),
      crate::exterior_power(frame, grade).column(0).into(),
    )
  }

  /// A single basis blade of one alternating slot, with the given sign.
  pub fn from_blade_signed(
    dim: impl Into<Dim>,
    sign: Sign,
    blade: crate::Blade,
    variance: Variance,
  ) -> Self {
    let (dim, grade) = (dim.into(), blade.card());
    let mut element = Self::zero(Self::one_alternating(grade, variance, dim));
    let rank = MonoIndex::from(blade).rank();
    element.components[rank] = sign.as_f64();
    element
  }

  /// The zero multiform of a grade: one covariant alternating slot.
  pub fn multiform_zero(dim: impl Into<Dim>, grade: impl Into<Degree>) -> Self {
    let (dim, grade) = (dim.into(), grade.into());
    Self::zero(Self::one_alternating(grade, Variance::Covariant, dim))
  }

  /// The zero multivector of a grade: one contravariant alternating slot.
  pub fn multivector_zero(dim: impl Into<Dim>, grade: impl Into<Degree>) -> Self {
    let (dim, grade) = (dim.into(), grade.into());
    Self::zero(Self::one_alternating(grade, Variance::Contravariant, dim))
  }

  /// The unit of the exterior algebra: one covariant slot at grade 0, holding 1.
  ///
  /// Distinct from [`Self::scalar`], which is the empty tensor product. Both
  /// are $RR$; they differ in arity, and only this one has a slot to wedge or
  /// star.
  pub fn one(dim: impl Into<Dim>) -> Self {
    Self::multiform(Vector::from_element(1, 1.0), dim, Degree::ZERO)
  }

  /// The shape of a single alternating slot.
  pub fn one_alternating(
    grade: impl Into<Degree>,
    variance: Variance,
    dim: impl Into<Dim>,
  ) -> Slots {
    uniform_slots([Factor::alternating(grade)], variance, dim)
  }

  pub fn slots(&self) -> &[Slot] {
    &self.slots
  }

  /// The same tensor with one slot's variance flipped, components untouched:
  /// $Lambda^k V$ relabelled as $Lambda^k V^*$ and back.
  ///
  /// Metric-free, and a relabelling rather than a map. Variance has no
  /// representational footprint ($dim Lambda^k (V) = dim Lambda^k (V^*)$), so
  /// there is nothing to compute; what changes is which space the components
  /// are read in. It is *not* a musical isomorphism, which applies $g$ or
  /// $g^(-1)$ and does need a metric: the musical is this relabelling composed
  /// with that application, and the two halves live in the two crates that
  /// need them.
  ///
  /// # Panics
  /// If `which` is out of range.
  pub fn dualize_slot(mut self, which: usize) -> Self {
    self.slots[which] = self.slots[which].dual();
    self
  }

  /// The variance shared by every slot, if they share one.
  ///
  /// `None` on a genuinely mixed tensor, and that is what makes functorial
  /// transport partial: covariant slots pull back and contravariant ones push
  /// forward, opposite directions, so a mixed tensor moves only along an
  /// isomorphism.
  pub fn variance(&self) -> Option<Variance> {
    let first = self.slots.first()?.variance;
    self
      .slots
      .iter()
      .all(|slot| slot.variance == first)
      .then_some(first)
  }

  /// Whether every slot has the given variance. Vacuously true of the scalar.
  pub fn is_variance(&self, variance: Variance) -> bool {
    self.slots.iter().all(|slot| slot.variance == variance)
  }

  /// The single slot, if this tensor has exactly one.
  ///
  /// The shape question every "is this really a Lambda^k" check reduces to,
  /// answered with the slot rather than a bool, so a caller that needs its
  /// symmetry, degree or variance does not look it up again.
  pub fn single(&self) -> Option<Slot> {
    match self.slots.as_slice() {
      [slot] => Some(*slot),
      _ => None,
    }
  }

  /// Every factor is alternating. Vacuously true of the scalar.
  pub fn is_alternating(&self) -> bool {
    self.slots.iter().all(Slot::is_alternating)
  }

  /// Every factor is symmetric. Vacuously true of the scalar.
  pub fn is_symmetric(&self) -> bool {
    self.slots.iter().all(Slot::is_symmetric)
  }

  /// An element of the exterior algebra $Lambda(V)$: exactly one factor, and it
  /// alternating.
  ///
  /// Stricter than [`Self::is_alternating`], and deliberately.
  /// $Lambda^a times.circle Lambda^b$ is alternating in both factors but lives
  /// in $Lambda(V) times.circle Lambda(V)$, a bigraded algebra whose product is
  /// not the wedge. It is [`Self::merge`] that collapses the two into one, and
  /// that map is the exterior algebra's multiplication.
  pub fn is_exterior(&self) -> bool {
    self.single().is_some_and(|factor| factor.is_alternating())
  }
  /// The space every slot is over.
  ///
  /// # Panics
  /// If the slots are over different spaces, where there is no one answer.
  /// Use [`Self::slots`] and read each slot's own `dim` instead.
  pub fn dim(&self) -> Dim {
    self
      .uniform_dim()
      .expect("this tensor's slots are over different spaces")
  }

  /// The space every slot is over, if they share one.
  ///
  /// `None` on a tensor spanning several spaces, which is what a rectangular
  /// map is. The scalar has no slots and so no space to name, and answers with
  /// the zero-dimensional one.
  pub fn uniform_dim(&self) -> Option<Dim> {
    let first = self.slots.first().map_or(Dim::ZERO, |slot| slot.dim);
    self
      .slots
      .iter()
      .all(|slot| slot.dim == first)
      .then_some(first)
  }
  pub fn strides(&self) -> &[usize] {
    &self.strides
  }
  pub fn components(&self) -> &Vector {
    &self.components
  }
  pub fn components_mut(&mut self) -> &mut Vector {
    &mut self.components
  }
  pub fn into_components(self) -> Vector {
    self.components
  }

  /// The components in the **reciprocal basis**: the basis dual to the one a
  /// tensor stores its components in.
  ///
  /// The stored basis is *multiplicative*, $x^alpha x^beta = x^(alpha + beta)$
  /// and $e_I wedge e_J = plus.minus e_(I union J)$ with unit coefficients,
  /// which is what lets $Lambda$ and $"Sym"$ be one construction. The price is
  /// that it is not self-dual on a symmetric slot: $x^alpha$ sums over all $k!$
  /// orderings, so $norm(x^alpha)^2 = alpha!$ and the reciprocal basis element
  /// is $x^alpha \/ alpha!$. Both bases are rational; only an *orthonormal* one
  /// would need $sqrt(alpha!)$, and none is used here.
  ///
  /// **Everything that dualizes a slot reads through this**: the duality
  /// pairing, and the pullback, which is the adjoint of the pushforward and so a
  /// pairing in disguise. The multiplicative operations use
  /// [`Self::components`] directly. On $Lambda^k$ the two coincide, every
  /// $alpha!$ being $1$, which is why a law swept over the alternating family
  /// alone says nothing about any of this.
  pub fn reciprocal(&self) -> Vector {
    self
      .components
      .component_mul(&basis_multiplicity(&self.slots, self.dim()))
  }

  /// A tensor from its components in the reciprocal basis: the inverse of
  /// [`Self::reciprocal`], and what a dualizing operation ends with.
  pub fn from_reciprocal(slots: impl Into<Slots>, reciprocal: Vector) -> Self {
    let slots = slots.into();
    let dim = slots.first().map_or(0, |slot| slot.dim.index());
    let components = reciprocal.component_div(&basis_multiplicity(&slots, dim));
    Self::new(slots, components)
  }

  /// The dimension of the space this tensor lives in.
  pub fn multidim(&self) -> usize {
    self.components.len()
  }

  /// The basis of each slot separately, in its family's own order.
  fn slot_bases(&self) -> Vec<Vec<MultiIndex>> {
    self
      .slots
      .iter()
      .map(|slot| slot.basis().collect())
      .collect()
  }

  /// The flat component index of a basis element: mixed radix over the
  /// per-factor colex ranks, $sum_i "rank"(alpha_i) s_i$.
  pub fn flat_index(&self, basis: &[MultiIndex]) -> usize {
    assert_eq!(basis.len(), self.slots.len());
    basis
      .iter()
      .zip(&self.strides)
      .map(|(index, stride)| index.rank() * stride)
      .sum()
  }

  /// Every basis element with its component, in the order the components are
  /// stored: colex per factor, last factor fastest.
  pub fn basis_iter(&self) -> impl Iterator<Item = (f64, Basis)> + '_ {
    let bases = self.slot_bases();
    let mut odometer = vec![0usize; self.slots.len()];
    self.components.iter().copied().map(move |component| {
      let basis: Basis = odometer
        .iter()
        .zip(&bases)
        .map(|(&position, factor_basis)| factor_basis[position])
        .collect();
      // Advance the last factor fastest, matching the stride convention.
      for (position, factor_basis) in odometer.iter_mut().zip(&bases).rev() {
        *position += 1;
        if *position < factor_basis.len() {
          break;
        }
        *position = 0;
      }
      (component, basis)
    })
  }

  /// The grade of a single-slot tensor.
  ///
  /// # Panics
  /// If the tensor does not have exactly one slot.
  pub fn grade(&self) -> Degree {
    self
      .single()
      .expect("a grade is of a single-slot tensor")
      .degree()
  }

  /// The wedge $wedge: Lambda^k times Lambda^l -> Lambda^(k+l)$, which is
  /// [`Self::product`] read on a single alternating slot.
  pub fn wedge(&self, other: &Self) -> Self {
    self.product(other)
  }

  /// The interior product $iota_v$ on the first slot: [`Self::contract`] where
  /// there is only one slot to contract.
  pub fn interior_product(&self, dual: &Self) -> Self {
    self.contract(0, dual)
  }

  /// The tensor product $times.circle$: factor lists concatenated, components
  /// the Kronecker product.
  ///
  /// Not the wedge: this concatenates, where [`Self::merge`] combines two
  /// like-symmetry factors into one of the summed degree. The wedge is their
  /// composition.
  pub fn tensor(&self, other: &Self) -> Self {
    let mut factors = self.slots.clone();
    factors.extend_from_slice(&other.slots);
    let components = self.components.kronecker(&other.components);
    Self::new(factors, components)
  }

  /// The product of the graded tensor-product algebra: two tensors of the same
  /// *shape* multiplied factor by factor, degrees adding.
  ///
  /// One operation over every shape: the wedge on a single alternating factor,
  /// the polynomial product on a single symmetric one, and on
  /// $"Sym"^r times.circle Lambda^k$ both at once,
  /// $(lambda^alpha dif lambda_I)(lambda^beta dif lambda_J)
  /// = lambda^(alpha + beta) dif lambda_I wedge dif lambda_J$.
  ///
  /// The sign is Koszul: the right factor $i$ commutes past every alternating
  /// factor of the left after it, contributing
  /// $(-1)^(deg b_i sum_(j > i) deg a_j)$. Symmetric factors are even.
  ///
  /// Factor by factor, where [`Self::tensor`] concatenates shapes, so the two
  /// compose. No intermediate is materialized.
  ///
  /// # Panics
  /// If the shapes differ in length or in any factor's symmetry.
  pub fn product(&self, other: &Self) -> Self {
    assert_eq!(
      self.slots.len(),
      other.slots.len(),
      "a product multiplies two tensors of one shape"
    );

    let mut slots = self.slots.clone();
    for (slot, right) in slots.iter_mut().zip(&other.slots) {
      assert_eq!(
        slot.symmetry(),
        right.symmetry(),
        "a product multiplies matching slots"
      );
      assert_eq!(
        slot.variance, right.variance,
        "a product multiplies slots of one variance"
      );
      *slot = slot.with_degree(slot.degree() + right.degree());
    }
    let mut product = Self::zero(slots);

    // The alternating degrees after each slot on the left: what the right slot
    // commutes past, hence the Koszul exponent.
    let mut trailing = vec![0usize; self.slots.len() + 1];
    for i in (0..self.slots.len()).rev() {
      trailing[i] = trailing[i + 1] + alternating_degree(&self.slots[i]);
    }

    self.product_into(other, &mut product, 0, 0, 0, 0, 1.0, &trailing);
    product
  }

  /// One factor of [`Self::product`], recursing into the next.
  ///
  /// Walks the pairs of per-factor basis elements, carrying the partial flat
  /// indices into both operands and the result. Nothing is allocated.
  #[allow(clippy::too_many_arguments)]
  fn product_into(
    &self,
    other: &Self,
    product: &mut Self,
    factor: usize,
    left_flat: usize,
    right_flat: usize,
    target_flat: usize,
    sign: f64,
    trailing: &[usize],
  ) {
    if factor == self.slots.len() {
      product.components[target_flat] +=
        sign * self.components[left_flat] * other.components[right_flat];
      return;
    }

    let (left, right) = (self.slots[factor], other.slots[factor]);
    let (left_stride, right_stride) = (self.strides[factor], other.strides[factor]);
    let target_stride = product.strides[factor];

    for (left_rank, left_index) in left.basis().enumerate() {
      let left_flat = left_flat + left_rank * left_stride;
      for (right_rank, right_index) in right.basis().enumerate() {
        let Some((merge_sign, merged)) = left_index.merge(&right_index) else {
          continue;
        };
        let koszul = alternating_degree(&right) * trailing[factor + 1];
        self.product_into(
          other,
          product,
          factor + 1,
          left_flat,
          right_flat + right_rank * right_stride,
          target_flat + merged.rank() * target_stride,
          sign * merge_sign.as_f64() * Sign::from_parity(koszul).as_f64(),
          trailing,
        );
      }
    }
  }

  /// Merge factor `first` with the one after it into a single factor of the
  /// summed degree: the wedge on two alternating factors, the polynomial
  /// product on two symmetric ones.
  ///
  /// One implementation for both, the symmetry entering only through
  /// [`MonoIndex::merge`]: signed and partial when alternating, unsigned and
  /// total when symmetric.
  ///
  /// # Panics
  /// If the two factors differ in symmetry.
  pub fn merge(&self, first: usize) -> Self {
    let (left, right) = (self.slots[first], self.slots[first + 1]);
    assert_eq!(
      left.symmetry(),
      right.symmetry(),
      "a merge combines two slots of one symmetry"
    );
    assert_eq!(
      left.variance, right.variance,
      "a merge combines two slots of one variance"
    );

    let mut merged_slots = self.slots.clone();
    merged_slots.remove(first + 1);
    merged_slots[first] = left.with_degree(left.degree() + right.degree());
    let mut merged = Self::zero(merged_slots);

    // The combinatorics runs once per pair of indices of the two touched
    // factors; the rest ride along as strides, counted as products rather than
    // by dividing the component count, so a factor naming the trivial space
    // leaves an empty sweep instead of a division by zero.
    let (left_dim, right_dim) = (left.multidim(), right.multidim());
    let inner = self.strides[first + 1];
    let outer = tensor_dim(&self.slots[..first]);
    let merged_dim = merged.slots[first].multidim();

    for (left_rank, left_index) in left.basis().enumerate() {
      for (right_rank, right_index) in right.basis().enumerate() {
        let Some((sign, product)) = left_index.merge(&right_index) else {
          continue;
        };
        let sign = sign.as_f64();
        let product_rank = product.rank();
        for block in 0..outer {
          let from = ((block * left_dim + left_rank) * right_dim + right_rank) * inner;
          let to = (block * merged_dim + product_rank) * inner;
          for offset in 0..inner {
            merged.components[to + offset] += sign * self.components[from + offset];
          }
        }
      }
    }
    merged
  }

  /// Contract factor `which` with a grade-one element of the dual variance:
  /// the interior product $iota_v$ on an alternating factor, the directional
  /// derivative $diff_v$ on a symmetric one.
  ///
  /// One implementation for both, through [`MonoIndex::deletions`]. On a
  /// symmetric factor, repeating it in every slot is evaluation,
  /// $p(x) = T(x, dots, x)$.
  ///
  /// Total at the trivial end: contracting a degree-zero factor lands in the
  /// trivial space, the empty index having no deletions.
  pub fn contract(&self, which: usize, dual: &Tensor) -> Self {
    assert_eq!(
      dual.slots[0].dim, self.slots[which].dim,
      "contraction is against an element of the slot's own space"
    );
    assert_eq!(dual.slots.len(), 1, "contraction is with a single slot");
    assert_eq!(
      dual.slots[0].degree(),
      1,
      "contraction is with a grade-1 element"
    );
    assert_eq!(
      dual.slots[0].variance,
      self.slots[which].variance.dual(),
      "contraction is against the dual variance"
    );

    let mut contracted_slots = self.slots.clone();
    contracted_slots[which] = contracted_slots[which].with_degree(self.slots[which].degree() - 1);
    let mut contracted = Self::zero(contracted_slots);

    // As in `merge`: the deletions of one index are enumerated once, and every
    // component sharing it is swept by stride arithmetic.
    let slot = self.slots[which];
    let basis_dim = slot.multidim();
    let inner = self.strides[which];
    let outer = tensor_dim(&self.slots[..which]);
    let reduced_dim = contracted.slots[which].multidim();

    for (rank, index) in slot.basis().enumerate() {
      for (sign, symbol, reduced) in index.deletions() {
        let weight = sign.as_f64() * dual.components[symbol];
        if weight == 0.0 {
          continue;
        }
        let reduced_rank = reduced.rank();
        for block in 0..outer {
          let from = (block * basis_dim + rank) * inner;
          let to = (block * reduced_dim + reduced_rank) * inner;
          for offset in 0..inner {
            contracted.components[to + offset] += weight * self.components[from + offset];
          }
        }
      }
    }
    contracted
  }

  /// Move one degree from one factor to another, summed over the basis of the
  /// underlying space: $sum_i (diff_(e_i) "on" F_"from") times.circle
  /// (e_i dot "on" F_"to")$.
  ///
  /// The exterior derivative and the Koszul operator are this one operation in
  /// its two directions. On $"Sym"^r times.circle Lambda^k$ in barycentric
  /// coordinates,
  ///
  ///   $dif (lambda^alpha dif lambda_I)
  ///     = sum_i alpha_i lambda^(alpha - e_i) dif lambda_i wedge dif lambda_I$
  ///
  /// is the transfer from the symmetric factor to the alternating one, and
  ///
  ///   $kappa (lambda^alpha dif lambda_I)
  ///     = sum_p (-1)^p lambda_(i_p) lambda^alpha dif lambda_(I without i_p)$
  ///
  /// is the transfer back. Both metric-free: the deletion supplies $alpha_i$ on
  /// a symmetric factor and the alternating sign on an alternating one, and the
  /// merge supplies the wedge sign one way and none the other.
  ///
  /// Total at the ends. Transferring out of a degree-zero factor has no
  /// deletions, and into a full alternating factor no room to merge, so both
  /// give the zero tensor.
  pub fn transfer(&self, from: usize, to: usize) -> Self {
    assert_ne!(from, to, "a transfer moves a degree between two factors");

    assert_eq!(
      self.slots[from].variance, self.slots[to].variance,
      "a transfer moves a degree within one variance"
    );

    let mut slots = self.slots.clone();
    slots[from] = slots[from].with_degree(self.slots[from].degree() - 1);
    slots[to] = slots[to].with_degree(self.slots[to].degree() + 1);
    let mut transferred = Self::zero(slots);

    for (component, basis) in self.basis_iter() {
      if component == 0.0 {
        continue;
      }
      for (deletion_sign, symbol, reduced) in basis[from].deletions() {
        let unit = match self.slots[to].symmetry().repetition() {
          Some(repetition) => MultiIndex::Mono(MonoIndex::single(repetition, symbol)),
          None => MultiIndex::Word(Word::new(self.slots[to].dim.index(), [symbol])),
        };
        let Some((merge_sign, grown)) = unit.merge(&basis[to]) else {
          continue;
        };
        let mut target: Basis = basis.clone();
        target[from] = reduced;
        target[to] = grown;
        let flat = transferred.flat_index(&target);
        transferred.components[flat] += (deletion_sign * merge_sign).as_f64() * component;
      }
    }
    transferred
  }

  /// The same tensor with every slot's symmetry forgotten: each $F^k$ becomes
  /// $V^(times.circle k)$, of the same degree and variance.
  ///
  /// The embedding of the quotients back into the free power, and it stays
  /// inside the algebra rather than handing back a bare array. $Lambda$ and
  /// $"Sym"$ *are* compressed representations of subspaces of
  /// $V^(times.circle k)$, and this is the map that says so. It is also the way
  /// out to code that knows only dense arrays: the strides of an all-free
  /// tensor are the radix, so the components are row-major over
  /// `[dim; total degree]` with no permutation.
  ///
  /// The entry at a word is that word's canonicalization, which
  /// [`MonoIndex::from_word`] already computes: the sign of the sorting
  /// permutation times the component of the sorted index, and zero where an
  /// alternating slot repeats a symbol.
  ///
  /// **Unnormalized**, matching [`Factor::induced_form`]: a basis element is
  /// $e_I = sum_sigma "sgn"(sigma) e_(i_(sigma(1))) times.circle dots.c$ with
  /// no $1\/k!$, so the orderings that coincide on a symmetric slot are summed
  /// rather than assigned.
  ///
  /// Costs $n^d$ against the packed dimension, so it is a boundary rather than
  /// something to compute in.
  pub fn to_free(&self) -> Self {
    let free: Slots = self
      .slots
      .iter()
      .map(|slot| {
        Slot::new(
          Factor::new(Symmetry::Free, slot.degree()),
          slot.variance,
          slot.dim,
        )
      })
      .collect();
    let mut expanded = Self::zero(free);
    let bases = expanded.slot_bases();
    let mut odometer = vec![0usize; self.slots.len()];

    for flat in 0..expanded.components.len() {
      let mut sign = 1.0;
      let mut source = Basis::new();
      for ((slot, position), slot_basis) in self.slots.iter().zip(&odometer).zip(&bases) {
        let index = &slot_basis[*position];
        match slot.symmetry().repetition() {
          None => source.push(*index),
          Some(repetition) => match MonoIndex::from_word(repetition, index.word()) {
            Some((canonical_sign, canonical)) => {
              // The orderings of the canonical word that give back this exact
              // one: its stabilizer, of size alpha!. On an alternating slot
              // every multiplicity is one and this is a no-op; on a symmetric
              // one it is the multiplicity the unnormalized symmetrization
              // carries, and dropping it costs exactly that factor.
              let stabilizer: usize = canonical
                .word()
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .map(|&symbol| multiindex::factorial(canonical.multiplicity(symbol)))
                .product();
              sign *= canonical_sign.as_f64() * stabilizer as f64;
              source.push(MultiIndex::Mono(canonical));
            }
            // A repeated symbol in an alternating slot: the zero of the
            // quotient, hence a zero of the embedding.
            None => {
              sign = 0.0;
              break;
            }
          },
        }
      }
      if sign != 0.0 {
        expanded.components[flat] += sign * self.components[self.flat_index(&source)];
      }
      for (position, slot_basis) in odometer.iter_mut().zip(&bases).rev() {
        *position += 1;
        if *position < slot_basis.len() {
          break;
        }
        *position = 0;
      }
    }
    expanded
  }

  /// Contract slot `i` against slot `j` of this tensor, summing over the shared
  /// basis: the trace.
  ///
  /// Metric-free, and that is the point. $"tr": V^* times.circle V -> RR$ needs
  /// no inner product, and reaching it by raising an index instead would drag a
  /// metric into an operation that does not depend on one.
  ///
  /// Both slots are consumed. The two must carry the same functor and opposite
  /// variance, which is what makes the sum the duality pairing rather than an
  /// inner product.
  ///
  /// # Panics
  /// If the slots coincide, carry different functors, or share a variance.
  pub fn trace(&self, i: usize, j: usize) -> Self {
    assert_ne!(i, j, "a trace contracts two distinct slots");
    let (left, right) = (self.slots[i], self.slots[j]);
    assert_eq!(
      left.factor, right.factor,
      "a trace contracts two slots of one functor"
    );
    assert_eq!(
      left.variance,
      right.variance.dual(),
      "a trace contracts dual variances, an inner product being the metric one"
    );

    let (low, high) = if i < j { (i, j) } else { (j, i) };
    let mut slots = self.slots.clone();
    slots.remove(high);
    slots.remove(low);
    let mut traced = Self::zero(slots);

    for (component, basis) in self.basis_iter() {
      if component == 0.0 || basis[i] != basis[j] {
        continue;
      }
      let mut target: Basis = basis;
      target.remove(high);
      target.remove(low);
      let flat = traced.flat_index(&target);
      traced.components[flat] += component;
    }
    traced
  }

  /// Contract slots of this tensor against slots of another, pair by pair.
  ///
  /// The general binary contraction, and it is [`Self::tensor`] followed by one
  /// [`Self::trace`] per pair: multi-contraction is repeated tracing, so there
  /// is no separate concept. Everything else is a special case, matrix
  /// composition and the duality pairing included.
  ///
  /// `pairs` names slots of `self` and of `other` by their own indices, before
  /// concatenation.
  ///
  /// Materializes the tensor product first, so it costs the product of the two
  /// dimensions. Fusing it is worthwhile where the traced slots dominate.
  pub fn contract_with(&self, other: &Self, pairs: &[(usize, usize)]) -> Self {
    let offset = self.slots.len();
    let mut contracted = self.tensor(other);
    let mut pending: Vec<(usize, usize)> = pairs
      .iter()
      .map(|&(mine, theirs)| (mine, theirs + offset))
      .collect();

    // Each trace removes two slots, so every surviving index above one of them
    // shifts down. Rebasing the pending pairs after each step makes the order
    // they are traced in irrelevant.
    while let Some((mine, theirs)) = pending.pop() {
      contracted = contracted.trace(mine, theirs);
      let (low, high) = (mine.min(theirs), mine.max(theirs));
      let rebase = |slot: &mut usize| {
        *slot -= usize::from(*slot > low) + usize::from(*slot > high);
      };
      for (mine, theirs) in &mut pending {
        rebase(mine);
        rebase(theirs);
      }
    }
    contracted
  }

  /// Evaluate a factor at a vector: contract it against that same vector once
  /// per unit of degree, $T(x, dots, x)$, consuming the factor.
  ///
  /// A symmetric factor of degree $r$ is a polynomial of degree $r$, so this is
  /// evaluation at a point, the same operation as feeding a tangent vector into
  /// an alternating factor.
  ///
  /// Repeated contraction fills the $r$ slots in every order and so overcounts
  /// by $r!$, which is divided out. On the monomial basis, $x^alpha |-> x^alpha (v)$.
  pub fn evaluate(&self, which: usize, vector: &Tensor) -> Self {
    let degree = self.slots[which].degree();
    let mut value = self.clone();
    for _ in 0..degree.index() {
      value = value.contract(which, vector);
    }
    value.components /= multiindex::factorial(degree.index()) as f64;

    let mut slots = value.slots.clone();
    slots.remove(which);
    Self::new(slots, value.components)
  }

  /// Apply a linear map to one slot, leaving the others alone:
  /// $id times.circle dots times.circle M times.circle dots times.circle id$.
  ///
  /// By stride arithmetic rather than by materializing the Kronecker product,
  /// so a single slot costs exactly one application of `matrix`.
  ///
  /// The map acts on the slot's *own* basis, not on the underlying space: it is
  /// already an induced map. [`Self::pullback`] and [`Self::pushforward`] are
  /// what take a map of the space.
  pub fn apply_to_slot(&self, which: usize, matrix: &Matrix) -> Self {
    let stride = self.strides[which];
    let source_dim = self.slots[which].multidim();
    assert_eq!(matrix.ncols(), source_dim);

    let mut slots = self.slots.clone();
    // The map may change that factor's dimension only by changing its degree,
    // which the caller states; here the shape is read off `matrix`.
    let target_dim = matrix.nrows();
    let outer = self.components.len() / (stride * source_dim);

    let mut components = Vector::zeros(outer * target_dim * stride);
    for before in 0..outer {
      for source in 0..source_dim {
        for after in 0..stride {
          let from = (before * source_dim + source) * stride + after;
          let value = self.components[from];
          if value == 0.0 {
            continue;
          }
          for target in 0..target_dim {
            let to = (before * target_dim + target) * stride + after;
            components[to] += matrix[(target, source)] * value;
          }
        }
      }
    }
    slots[which] = self.slots[which];
    Self::new(slots, components)
  }

  /// The variance every slot shares, or a panic naming why a mixed tensor has
  /// no functorial transport along an arbitrary map.
  fn transport_variance(&self, what: &str) -> Option<Variance> {
    assert!(
      self.slots.is_empty() || self.variance().is_some(),
      "a mixed tensor has no {what}: its covariant slots pull back and its \
       contravariant ones push forward, opposite directions, so it transports \
       only along an isomorphism"
    );
    self.variance()
  }

  pub fn eq_epsilon(&self, other: &Self, eps: f64) -> bool {
    self.slots == other.slots
      && (&self.components - &other.components).norm_squared() <= eps.powi(2)
  }
}

impl Tensor {
  /// Pushforward along $A: V -> W$: the covariant action of the functor.
  ///
  /// # Panics
  /// If any slot is covariant, or the slots are mixed.
  pub fn pushforward(&self, map: &Matrix) -> Self {
    assert_eq!(
      self.dim(),
      map.ncols(),
      "one map transports a tensor whose slots share a space"
    );
    if let Some(variance) = self.transport_variance("pushforward") {
      assert_eq!(
        variance,
        Variance::Contravariant,
        "a covariant tensor pulls back, it does not push forward"
      );
    }
    Transport::new(&self.slots, map).pushforward(self)
  }

  /// Pullback along $A: V -> W$: the contravariant action of the functor.
  ///
  /// Defined for any linear map, injective or not, which is what a uniform
  /// variance buys and a mixed tensor does not have.
  ///
  /// # Panics
  /// If any slot is contravariant, or the slots are mixed.
  pub fn pullback(&self, map: &Matrix) -> Self {
    assert_eq!(
      self.dim(),
      map.nrows(),
      "one map transports a tensor whose slots share a space"
    );
    if let Some(variance) = self.transport_variance("pullback") {
      assert_eq!(
        variance,
        Variance::Covariant,
        "a contravariant tensor pushes forward, it does not pull back"
      );
    }
    Transport::new(&self.slots, map).pullback(self)
  }
}

/// The maps a linear map $A: V -> W$ induces on a *fixed* tensor shape,
/// materialized once and applied many times.
///
/// $Lambda^k A$ and $"Sym"^r A$ cost a compound matrix to form and nothing to
/// apply, so wherever one map transports a whole family of tensors --- the
/// trace onto a face at every quadrature node, a refinement child's Jacobian
/// over every cell --- the functor is reference data and belongs outside the
/// loop. This is that cache, and it is *the same* code path as
/// [`Tensor::pullback`] and [`Tensor::pushforward`], which are one-shot uses of
/// it rather than a second implementation.
///
/// Uniform over the slots, so it transports a $"Sym"^r times.circle Lambda^k$
/// exactly as it does a bare multiform. That is the reason to have the object
/// at all rather than a bare $Lambda^k A$ matrix: a stored matrix applied to raw
/// components is the pullback only where the multiplicative basis is self-dual,
/// which is to say on the alternating family alone.
#[derive(Debug, Clone)]
pub struct Transport {
  /// The shape in the domain $V$.
  domain: Slots,
  /// The shape in the codomain $W$: the same factors over the other space.
  codomain: Slots,
  /// $times.circle_i F_i (A)$, the Kronecker product of the per-slot functors.
  induced: Matrix,
}

impl Transport {
  /// The functor of `map` on the given shape.
  ///
  /// Only the *factors* are read. The dimensions come from the map, and the
  /// variance from the tensor being transported, never from here: a transport
  /// carries a variance rather than imposing one, so the same object serves the
  /// pullback of a form and the pushforward of a vector, which is exactly what
  /// makes the two adjoint (invariant 4).
  pub fn new(slots: &[Slot], map: &Matrix) -> Self {
    Self {
      domain: slots.iter().map(|s| s.with_dim(map.ncols())).collect(),
      codomain: slots.iter().map(|s| s.with_dim(map.nrows())).collect(),
      induced: induced(slots, map),
    }
  }

  /// The shape this transport takes to [`Self::codomain`] under
  /// [`Self::pushforward`], and lands in under [`Self::pullback`].
  pub fn domain(&self) -> &[Slot] {
    &self.domain
  }
  /// The shape [`Self::pushforward`] lands in, and [`Self::pullback`] consumes.
  pub fn codomain(&self) -> &[Slot] {
    &self.codomain
  }
  /// $times.circle_i F_i (A)$ itself, for a caller assembling it into a larger
  /// matrix rather than applying it.
  pub fn matrix(&self) -> &Matrix {
    &self.induced
  }

  /// # Panics
  /// If the tensor's shape is not the domain shape, or it is not uniformly
  /// contravariant.
  pub fn pushforward(&self, tensor: &Tensor) -> Tensor {
    self.check(tensor, &self.domain, Variance::Contravariant, "pushforward");
    Tensor::new(
      Self::carrying(&self.codomain, tensor),
      &self.induced * tensor.components(),
    )
  }

  /// # Panics
  /// If the tensor's shape is not the codomain shape, or it is not uniformly
  /// covariant.
  pub fn pullback(&self, tensor: &Tensor) -> Tensor {
    self.check(tensor, &self.codomain, Variance::Covariant, "pullback");

    // A pullback dualizes: it is the adjoint of the pushforward, so it acts on
    // the reciprocal components and lands in the reciprocal basis of the domain.
    // On a symmetric slot that is what makes it the transpose of the functor
    // conjugated by alpha! rather than the bare transpose.
    Tensor::from_reciprocal(
      Self::carrying(&self.domain, tensor),
      self.induced.transpose() * tensor.reciprocal(),
    )
  }

  /// The target shape wearing the transported tensor's variance: the factors
  /// and dimensions are the transport's, the variance is the value's.
  fn carrying(shape: &Slots, tensor: &Tensor) -> Slots {
    shape
      .iter()
      .zip(tensor.slots())
      .map(|(slot, source)| Slot::new(slot.factor, source.variance, slot.dim))
      .collect()
  }

  fn check(&self, tensor: &Tensor, shape: &Slots, wanted: Variance, what: &str) {
    assert!(
      tensor
        .slots()
        .iter()
        .zip(shape)
        .all(|(a, b)| a.factor == b.factor && a.dim == b.dim)
        && tensor.slots().len() == shape.len(),
      "a transport applies to the shape it was built for"
    );
    if let Some(variance) = tensor.transport_variance(what) {
      assert_eq!(
        variance,
        wanted,
        "a {} tensor does not {what}",
        match wanted {
          Variance::Contravariant => "covariant",
          Variance::Covariant => "contravariant",
        }
      );
    }
  }
}

/// The metric-free duality pairing of two tensors of dual variance,
/// $angle.l omega, v angle.r$.
///
/// Slot for slot: same factor, opposite variance. That is what "dual" means
/// here, and it is checked rather than assumed, a pairing of two covariant
/// tensors being a different operation that needs a metric.
///
/// A free function, not a method: a pairing is a bilinear map on two spaces and
/// privileges neither argument. It is symmetric in them, `pairing(a, b)` being
/// `pairing(b, a)`, which method syntax would quietly deny.
///
/// One side is read in the **reciprocal basis** ([`Tensor::reciprocal`]), which
/// is what the pairing of a basis element with its dual means. Either side will
/// do and the answer is the same, which is the pairing's symmetry stated in the
/// implementation. A plain dot product of the stored components would be off by
/// $alpha!$ on every repeated symbol, and right on $Lambda^k$, where the two
/// bases coincide.
pub fn pairing(left: &Tensor, right: &Tensor) -> f64 {
  assert_eq!(
    left.slots().len(),
    right.slots().len(),
    "a pairing is slot for slot"
  );
  for (slot, other) in left.slots().iter().zip(right.slots()) {
    assert_eq!(
      *slot,
      other.dual(),
      "a pairing is against the dual variance, slot for slot"
    );
  }
  left.reciprocal().dot(right.components())
}

/// The wedge pairing $Lambda^k times Lambda^(n-k) -> RR$,
/// $(alpha, beta) |-> alpha wedge beta$ read as a multiple of the basis volume
/// element.
///
/// Poincare duality at the level of the algebra, and **metric-free**: it needs
/// only a top degree to land in and a basis to read the coefficient against,
/// where the Hodge star needs a metric as well. That is the whole difference
/// between the two dualities the exterior algebra carries, and it is why this
/// exists separately from the metric inner product, which lives one crate up.
///
/// Nondegenerate, so it identifies $Lambda^(n-k)$ with the dual of $Lambda^k$
/// without ever choosing an inner product. Antisymmetric up to the grading,
/// $angle.l beta, alpha angle.r = (-1)^(k(n-k)) angle.l alpha, beta angle.r$.
///
/// Both arguments must be single alternating slots of one variance whose grades
/// sum to the dimension.
///
/// # Panics
/// If either is not a single alternating slot, or the grades do not sum to $n$.
pub fn wedge_pairing(left: &Tensor, right: &Tensor) -> f64 {
  let (a, b) = (
    left
      .single()
      .expect("a wedge pairing is of single-slot tensors"),
    right
      .single()
      .expect("a wedge pairing is of single-slot tensors"),
  );
  assert!(
    a.is_alternating() && b.is_alternating(),
    "a wedge pairing needs a top degree, which a symmetric slot has not"
  );
  assert_eq!(
    a.variance, b.variance,
    "a wedge pairing multiplies within one variance"
  );
  assert_eq!(
    a.degree() + b.degree(),
    left.dim(),
    "a wedge pairing lands in the top grade"
  );

  let top = left.wedge(right);
  // The top grade is one-dimensional, so the product is a multiple of the basis
  // volume element and that multiple is the pairing.
  top.as_scalar()
}
/// Every factor at one variance: the shape of a uniform-variance tensor.
///
/// The common case by far, since a mixed tensor is what you build deliberately.
pub fn uniform_slots(
  factors: impl IntoIterator<Item = Factor>,
  variance: Variance,
  dim: impl Into<Dim>,
) -> Slots {
  let dim = dim.into();
  factors
    .into_iter()
    .map(|factor| Slot::new(factor, variance, dim))
    .collect()
}

/// [`uniform_slots`] at [`Variance::Covariant`]: forms, and everything built
/// from the dual.
pub fn covariant_slots(factors: impl IntoIterator<Item = Factor>, dim: impl Into<Dim>) -> Slots {
  uniform_slots(factors, Variance::Covariant, dim)
}

/// Combine per-slot matrices into one on the tensor's component basis: the
/// Kronecker product in the order the strides imply, last factor fastest.
///
/// The one place that order is written down for matrices, so a caller building
/// a weight slot by slot cannot disagree with [`tensor_strides`] about which
/// index runs fastest. Getting it backwards yields a matrix of the right
/// shape and the wrong meaning, which no shape check catches.
///
/// An empty product of slots is the scalars, on which the identity acts.
pub fn factorwise_kronecker(per_slot: &[Matrix]) -> Matrix {
  per_slot
    .iter()
    .cloned()
    .reduce(|acc, slot| acc.kronecker(&slot))
    .unwrap_or_else(|| Matrix::identity(1, 1))
}

impl std::ops::Add for Tensor {
  type Output = Self;
  fn add(mut self, other: Self) -> Self {
    self += other;
    self
  }
}
impl std::ops::AddAssign for Tensor {
  fn add_assign(&mut self, other: Self) {
    assert_eq!(self.slots, other.slots);
    self.components += other.components;
  }
}
impl std::ops::Sub for Tensor {
  type Output = Self;
  fn sub(mut self, other: Self) -> Self {
    self -= other;
    self
  }
}
impl std::ops::SubAssign for Tensor {
  fn sub_assign(&mut self, other: Self) {
    assert_eq!(self.slots, other.slots);
    self.components -= other.components;
  }
}
impl std::ops::Mul<f64> for Tensor {
  type Output = Self;
  fn mul(mut self, scalar: f64) -> Self {
    self *= scalar;
    self
  }
}
impl std::ops::MulAssign<f64> for Tensor {
  fn mul_assign(&mut self, scalar: f64) {
    self.components *= scalar;
  }
}
impl std::ops::Mul<Tensor> for f64 {
  type Output = Tensor;
  fn mul(self, tensor: Tensor) -> Tensor {
    tensor * self
  }
}
impl std::ops::Index<usize> for Tensor {
  type Output = f64;
  fn index(&self, index: usize) -> &f64 {
    &self.components[index]
  }
}
impl std::iter::Sum for Tensor {
  fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
    let mut sum = iter.next().expect("an empty sum has no shape to take");
    for tensor in iter {
      sum += tensor;
    }
    sum
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use approx::assert_relative_eq;
  use multiindex::{Degree, factorial};

  fn probe(nrows: usize, ncols: usize, seed: usize) -> Matrix {
    Matrix::from_fn(nrows, ncols, |i, j| {
      ((7 * i + 3 * j + 5 * seed + 1) % 11) as f64 - 5.0
    })
  }
  /// A map of full column rank, so a pulled-back metric stays non-degenerate
  /// and the law is tested on a metric rather than on a singular form.
  fn probe_map(nrows: usize, ncols: usize, seed: usize) -> Matrix {
    Matrix::from_fn(nrows, ncols, |i, j| {
      ((seed + 3 * i + 7 * j) % 5) as f64 / 5.0 + if i == j { 1.0 } else { 0.0 }
    })
  }
  /// A symmetric positive definite matrix: a bilinear form, which is all the
  /// induced form asks for.
  fn probe_metric(dim: usize) -> Matrix {
    let a = probe(dim, dim, 2);
    a.transpose() * &a + Matrix::identity(dim, dim)
  }

  /// The induced Gramian is the pullback of the Gramian: measuring the images
  /// under $F(A)$ is measuring the originals in the pulled-back metric,
  /// $"Gram"_F (A^T G A) = F(A)^T "Gram"_F (G) F(A)$.
  ///
  /// This is the law that decides the normalization, and it is stated on a
  /// *rectangular* map so the two ends cannot coincide and hide a factor. The
  /// alternating side is the familiar Cauchy-Binet statement; the symmetric one
  /// is its permanental counterpart, and it is what pins $"per"$ with nothing
  /// in front of it against the $"per" \/ d!$ that the symmetrized-tensor
  /// convention would give.
  #[test]
  fn the_induced_gramian_is_the_pullback_of_the_gramian() {
    for degree in 0..=3 {
      for symmetry in [Symmetry::Alternating, Symmetry::Symmetric] {
        for &(target, source) in &[(3, 2), (4, 3), (3, 3), (4, 2)] {
          let factor = Factor {
            symmetry,
            degree: degree.into(),
          };
          let map = probe_map(target, source, 1);
          let metric = probe_metric(target);

          let pulled = map.transpose() * (&metric) * &map;
          let induced = factor.induced(&map);
          assert_relative_eq!(
            &factor.induced_form(&pulled),
            &(induced.transpose() * &factor.induced_form(&metric) * &induced),
            epsilon = 1e-7
          );
        }
      }
    }
  }

  /// Under a Euclidean metric the alternating basis is orthonormal and the
  /// symmetric one orthogonal with $norm(x^alpha)^2 = alpha!$.
  ///
  /// The multiplicity a repeated slot carries, which is exactly what the
  /// alternating side lacks because it forbids the repetition. Not a loose
  /// normalization: the two follow from the one convention, and the previous
  /// law is what forces it.
  #[test]
  fn the_euclidean_gramian_reads_off_the_multiplicities() {
    for dim in 1..=4 {
      let euclidean = Matrix::identity(dim, dim);
      for degree in 0..=3 {
        let alternating = Factor::alternating(degree);
        let identity = Matrix::identity(alternating.multidim(dim), alternating.multidim(dim));
        assert_relative_eq!(&alternating.induced_form(&euclidean), &identity);

        let symmetric = Factor::symmetric(degree);
        let gramian = symmetric.induced_form(&euclidean);
        for (i, index) in symmetric.basis(dim).enumerate() {
          for (j, other) in symmetric.basis(dim).enumerate() {
            let expected = if i == j {
              (0..dim)
                .map(|symbol| factorial(index.multiplicity(symbol)))
                .product::<usize>() as f64
            } else {
              0.0
            };
            assert_relative_eq!(gramian[(i, j)], expected);
            let _ = &other;
          }
        }
      }
    }
  }

  /// On a single factor the product *is* the merge of the tensor, which is what
  /// ties the algebra structure back to the two primitives:
  /// $a b = "merge"_0 (a times.circle b)$.
  ///
  /// Stated where it is true. With one factor a side there is no reordering, so
  /// no Koszul sign, and the factorwise product and the seam merge coincide;
  /// with more factors they genuinely differ, and it is the factorwise one that
  /// is the algebra.
  #[test]
  fn on_one_factor_the_product_is_a_merge_of_a_tensor() {
    let dim = 3;
    for symmetry in [Symmetry::Alternating, Symmetry::Symmetric] {
      for left_degree in 0..=2 {
        for right_degree in 0..=2 {
          let build = |degree, seed| {
            let factors = covariant_slots(
              [Factor {
                symmetry,
                degree: Degree::from(degree),
              }],
              dim,
            );
            let len = tensor_dim(&factors);
            Tensor::new(
              factors,
              Vector::from_fn(len, |i, _| ((seed + 5 * i) % 7) as f64 - 3.0),
            )
          };
          let (a, b) = (build(left_degree, 1), build(right_degree, 2));
          let fused = a.product(&b);
          let composed = a.tensor(&b).merge(0);
          assert_eq!(fused.slots(), composed.slots());
          assert_relative_eq!(fused.components(), composed.components(), epsilon = 1e-12);
        }
      }
    }
  }

  /// The product is graded-commutative in the Koszul sense:
  /// $b a = (-1)^(abs(a) abs(b)) a b$, where the degree that counts is the
  /// *alternating* one, symmetric factors being even.
  ///
  /// Checked on a mixed shape, $"Sym" times.circle Lambda$, which is where the
  /// sign is a real claim: on one factor it is the wedge's antisymmetry, and on
  /// a purely symmetric shape it is plain commutativity, so neither alone
  /// exercises the rule.
  #[test]
  fn the_product_is_koszul_graded_commutative() {
    let dim = 3;
    for left in 0..=2 {
      for right in 0..=2 {
        let a = poly_form(dim, 1, left, 1);
        let b = poly_form(dim, 2, right, 2);
        let sign = Sign::from_parity(left * right).as_f64();
        assert_relative_eq!(
          a.product(&b).components(),
          &(sign * b.product(&a)).components(),
          epsilon = 1e-12
        );
      }
    }
  }

  /// The product is associative, and the scalar tensor is its unit.
  #[test]
  fn the_product_is_an_associative_algebra() {
    let dim = 3;
    let unit = Tensor::new(
      covariant_slots([Factor::symmetric(0), Factor::alternating(0)], dim),
      Vector::from_element(1, 1.0),
    );
    let a = poly_form(dim, 1, 1, 3);
    assert_relative_eq!(a.product(&unit).components(), a.components());
    assert_relative_eq!(unit.product(&a).components(), a.components());

    let b = poly_form(dim, 1, 1, 4);
    let c = poly_form(dim, 2, 1, 5);
    assert_relative_eq!(
      a.product(&b).product(&c).components(),
      a.product(&b.product(&c)).components(),
      epsilon = 1e-9
    );
  }

  /// A homogeneous polynomial form $"Sym"^r times.circle Lambda^k$ with
  /// deterministic components.
  fn poly_form(dim: usize, r: usize, k: usize, seed: usize) -> Tensor {
    let factors = covariant_slots([Factor::symmetric(r), Factor::alternating(k)], dim);
    let len = tensor_dim(&factors);
    Tensor::new(
      factors,
      Vector::from_fn(len, |i, _| ((seed + 5 * i) % 7) as f64 - 3.0),
    )
  }

  /// Transferring twice in the same direction vanishes, both ways round:
  /// $dif compose dif = 0$ and $kappa compose kappa = 0$.
  ///
  /// One law for two operators, which is the point of [`Tensor::transfer`] --
  /// they are the same operation in opposite directions, so nilpotency is one
  /// statement about it rather than two coincidences.
  #[test]
  fn transferring_twice_in_one_direction_vanishes() {
    for dim in 1..=4 {
      for r in 0..=3 {
        for k in 0..=dim {
          let form = poly_form(dim, r, k, 1);
          // Sym -> Lambda twice: the exterior derivative.
          let twice = form.transfer(0, 1).transfer(0, 1);
          assert_relative_eq!(twice.components().amax(), 0.0, epsilon = 1e-12);
          // Lambda -> Sym twice: the Koszul operator.
          let twice = form.transfer(1, 0).transfer(1, 0);
          assert_relative_eq!(twice.components().amax(), 0.0, epsilon = 1e-12);
        }
      }
    }
  }

  /// The Koszul homotopy formula: on homogeneous $"Sym"^r times.circle
  /// Lambda^k$, $dif kappa + kappa dif = (r + k) id$.
  ///
  /// The identity the whole polynomial de Rham complex rests on -- it is what
  /// makes that complex exact, and hence what the trimmed spaces
  /// $P^-_r Lambda^k$ are cut out by. Checking it here checks that both
  /// directions of the transfer carry the right signs and the right
  /// multiplicities, which no weaker law does: nilpotency alone passes on an
  /// operator scaled by anything.
  #[test]
  fn the_koszul_homotopy_formula_holds() {
    for dim in 1..=4 {
      for r in 0..=3 {
        for k in 0..=dim {
          let form = poly_form(dim, r, k, 2);
          let dif_then_koszul = form.transfer(0, 1).transfer(1, 0);
          let koszul_then_dif = form.transfer(1, 0).transfer(0, 1);
          let sum = dif_then_koszul + koszul_then_dif;
          let expected = (r + k) as f64 * form.clone();
          assert_relative_eq!(sum.components(), expected.components(), epsilon = 1e-9);
          if r + k > 0 {
            assert!(
              expected.components().amax() > 0.0,
              "the law would hold vacuously"
            );
          }
        }
      }
    }
  }

  /// What the pullback law does and does not pin.
  ///
  /// A *global* constant cancels: scaling the Gramian by any $c$ scales both
  /// sides of $"Gram"(A^* g) = F(A)^* "Gram"(g)$ equally, so functoriality
  /// alone leaves the overall factor free and cannot choose between $"per"$ and
  /// $"per" \/ d!$. What it does pin is the *shape* -- a normalization
  /// depending on the multi-index, such as the $alpha!$ that
  /// [`Factor::induced`] carries, breaks the law outright.
  ///
  /// The remaining constant is fixed by siblinghood rather than by this law:
  /// $Lambda$ and $"Sym"$ take the same construction, so they take the same
  /// factor, and $Lambda$'s is one. Recording the distinction here because a
  /// law that passes for a whole family of candidates is not evidence for any
  /// one of them.
  #[test]
  fn the_pullback_law_pins_the_shape_and_not_the_constant() {
    let factor = Factor::symmetric(2);
    let (target, source) = (4, 3);
    let map = probe_map(target, source, 1);
    let metric = probe_metric(target);
    let pulled = map.transpose() * (&metric) * &map;
    let induced = factor.induced(&map);

    let holds = |rescale: &dyn Fn(&Factor, &Matrix) -> Matrix| {
      let lhs = rescale(&factor, &pulled);
      let rhs = rescale(&factor, &metric);
      (lhs - induced.transpose() * rhs * &induced).amax() < 1e-7
    };

    // Any global constant survives, so the law cannot choose one.
    for constant in [1.0, 0.5, 1.0 / 6.0] {
      assert!(holds(&move |factor: &Factor, g: &Matrix| {
        factor.induced_form(g) * constant
      }));
    }
    // A per-index factor does not.
    assert!(!holds(&|factor: &Factor, g: &Matrix| {
      let scale: Vec<f64> = factor
        .basis(g.nrows())
        .map(|index| {
          (0..g.nrows())
            .map(|symbol| factorial(index.multiplicity(symbol)))
            .product::<usize>() as f64
        })
        .collect();
      Matrix::from_fn(scale.len(), scale.len(), |i, j| {
        factor.induced_form(g)[(i, j)] / scale[i]
      })
    }));
  }

  /// The two families give the same Gramian at degrees zero and one, where
  /// there is no symmetry to impose and $det$ and $"per"$ of a $1 times 1$
  /// block agree.
  #[test]
  fn the_gramians_coincide_below_degree_two() {
    for dim in 1..=4 {
      let metric = probe_metric(dim);
      for degree in 0..=1 {
        assert_relative_eq!(
          &Factor::alternating(degree).induced_form(&metric),
          &Factor::symmetric(degree).induced_form(&metric)
        );
      }
    }
  }

  /// [`factorwise_kronecker`] lays a per-factor matrix out in the order the
  /// strides imply: the entry at a pair of flat component indices is the
  /// product of the per-factor entries at the ranks those indices decompose
  /// into.
  ///
  /// The law tying the two conventions together. Reversing the Kronecker order
  /// produces a matrix of the same shape, symmetric and positive definite when
  /// the operands are, so nothing but this catches it.
  #[test]
  fn factorwise_kronecker_follows_the_stride_order() {
    for dim in 1..=3 {
      let factors = [Factor::symmetric(2), Factor::alternating(1)];
      let strides = tensor_strides(&covariant_slots(factors, dim));
      let dims: Vec<usize> = factors.iter().map(|f| f.multidim(dim)).collect();

      let per_factor: Vec<Matrix> = dims
        .iter()
        .enumerate()
        .map(|(f, &d)| Matrix::from_fn(d, d, |i, j| (1 + f + 3 * i + 7 * j) as f64))
        .collect();
      let combined = factorwise_kronecker(&per_factor);

      assert_eq!(combined.nrows(), dims.iter().product::<usize>());
      for row_a in 0..dims[0] {
        for row_b in 0..dims[1] {
          for col_a in 0..dims[0] {
            for col_b in 0..dims[1] {
              let row = row_a * strides[0] + row_b * strides[1];
              let col = col_a * strides[0] + col_b * strides[1];
              assert_eq!(
                combined[(row, col)],
                per_factor[0][(row_a, col_a)] * per_factor[1][(row_b, col_b)],
                "dim {dim}: the kronecker order disagrees with the strides"
              );
            }
          }
        }
      }
    }
  }

  /// An element of the exterior algebra is a single alternating factor, which
  /// is strictly stronger than being alternating in every factor: a tensor
  /// product of two exterior elements is not one.
  #[test]
  fn exterior_is_single_slot_not_merely_alternating() {
    let dim = 3;
    let single = Tensor::zero(covariant_slots([Factor::alternating(2)], dim));
    assert!(single.is_exterior());
    assert!(single.is_alternating());
    assert_eq!(single.single().map(|s| s.degree()), Some(Degree::from(2)));

    let paired = Tensor::zero(covariant_slots(
      [Factor::alternating(1), Factor::alternating(1)],
      dim,
    ));
    assert!(paired.is_alternating());
    assert!(
      !paired.is_exterior(),
      "two slots are not the exterior algebra"
    );
    assert!(paired.single().is_none());

    let mixed = Tensor::zero(covariant_slots(
      [Factor::symmetric(2), Factor::alternating(1)],
      dim,
    ));
    assert!(!mixed.is_alternating());
    assert!(!mixed.is_symmetric());
    assert!(!mixed.is_exterior());

    // The scalar is vacuously both, and is not a slot.
    let scalar = Tensor::scalar(1.0, dim);
    assert!(scalar.is_alternating() && scalar.is_symmetric());
    assert!(scalar.single().is_none());
  }
}
