//! The chain complex of the simplicial complex, and its dual.
//!
//! A [`Chain`] is an element of $C_k (K; R)$, a formal combination of the
//! oriented $k$-simplices; a [`Cochain`] is an element of $C^k (K; R) =
//! "Hom"(C_k, R)$, one coefficient per $k$-simplex. The [`pairing`] between
//! them is what makes the second the dual of the first, and under it the
//! boundary $partial$ and the coboundary $dif$ are adjoint, $chevron.l dif omega, c
//! chevron.r = chevron.l omega, partial c chevron.r$.
//!
//! Both are one type, [`FreeModule`], because both *are* the free $R$-module on
//! the $k$-simplices and nothing in the data distinguishes them. What
//! distinguishes them is [`Variance`], and it is stated rather than derived: a
//! chain transforms with the simplices and its differential lowers the grade, a
//! cochain transforms against them and its differential raises it. The
//! differential itself is written once. It is the signed incidence
//! [`Complex::incidences`] read in the direction the variance names, $partial$
//! scattering a coface's coefficient onto its faces and $dif$ gathering a
//! coface's coefficient from them, so the two operators are one traversal of
//! one relation rather than two implementations that must be kept in step.
//!
//! Variance being a type parameter here, where
//! [`multialgebra`](https://docs.rs/multialgebra)'s slot variance deliberately
//! is not, is the same rule applied to a different shape: there the datum is
//! per-slot over a runtime number of slots, here it is one datum for the whole
//! object and known statically, so the witness costs nothing and the omission
//! it would leave silent cannot arise.
//!
//! The coefficient ring is the other axis, and it is a parameter for the same
//! reason: the complex is defined over any ring, and this library genuinely
//! runs over two. Over $ZZ$ the (co)homology is computed exactly and a class
//! has representatives that are honest integer combinations; over $RR$ a
//! cochain is what an operator multiplies and a solver returns. The defaults
//! name the ring each side is usually asked for, so neither spelling carries a
//! parameter it does not care about, and no operation is defined for only one
//! ring. A ring map $R -> S$ carries the whole complex to another
//! ([`FreeModule::extend_scalars`]) and commutes with the differentials,
//! because an incidence coefficient is $plus.minus 1$ and every ring map fixes
//! those.
//!
//! All of this is topology. A cochain becomes a *discrete differential form*
//! only through the de Rham map, which integrates a form over each simplex and
//! needs a geometry; that reading, and the theorem that it is an isomorphism on
//! cohomology, belong to whoever supplies the geometry.

use super::{
  complex::Complex,
  data::SkeletonData,
  handle::{KSimplexIdx, SimplexIdx, SimplexRef, SkeletonRef},
};
use crate::linalg::Vector;
use crate::{Dim, Sign};

use num_traits::Zero;
use std::{marker::PhantomData, ops::Neg};

#[cfg(feature = "serde")]
use std::{io, path::Path};

/// The coefficient ring of a [`FreeModule`].
///
/// A blanket alias, so a ring is a coefficient ring by being one. The bounds
/// are what the operations actually consume, and they are less than a ring:
/// the differentials need only the additive group, since an incidence
/// coefficient is $plus.minus 1$ and is applied as an addition or a
/// subtraction. Multiplication enters in the [`pairing`] and in scaling alone.
pub trait Coefficient:
  na::Scalar
  + Zero
  + na::ClosedAddAssign
  + na::ClosedSubAssign
  + na::ClosedMulAssign
  + Neg<Output = Self>
{
}
impl<R> Coefficient for R where
  R: na::Scalar
    + Zero
    + na::ClosedAddAssign
    + na::ClosedSubAssign
    + na::ClosedMulAssign
    + Neg<Output = R>
{
}

/// Which of the two dual complexes a [`FreeModule`] belongs to: the direction
/// its differential runs, and which end of an incidence it reads.
///
/// The datum has no representational footprint, $C_k$ and $C^k$ being free
/// modules of the same rank on the same simplices, so it is stated by the type
/// and never inferred from a value.
pub trait Variance {
  /// The grade the differential lands in: one below for a chain, one above for
  /// a cochain.
  fn target(grade: Dim) -> Dim;
  /// The rung of [`Complex::incidences`] connecting the two grades, which is
  /// the lower of them.
  fn rung(grade: Dim) -> Dim;
  /// One incidence resolved into the coordinate read from and the coordinate
  /// written to.
  fn traverse(face: KSimplexIdx, coface: KSimplexIdx) -> (KSimplexIdx, KSimplexIdx);
}

/// The variance of a [`Chain`]: covariant, its differential the boundary
/// $partial_k: C_k -> C_(k-1)$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Homological;
impl Variance for Homological {
  fn target(grade: Dim) -> Dim {
    grade - 1
  }
  fn rung(grade: Dim) -> Dim {
    grade - 1
  }
  fn traverse(face: KSimplexIdx, coface: KSimplexIdx) -> (KSimplexIdx, KSimplexIdx) {
    (coface, face)
  }
}

/// The variance of a [`Cochain`]: contravariant, its differential the
/// coboundary $dif^k: C^k -> C^(k+1)$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cohomological;
impl Variance for Cohomological {
  fn target(grade: Dim) -> Dim {
    grade + 1
  }
  fn rung(grade: Dim) -> Dim {
    grade
  }
  fn traverse(face: KSimplexIdx, coface: KSimplexIdx) -> (KSimplexIdx, KSimplexIdx) {
    (face, coface)
  }
}

/// The free $R$-module on the $k$-simplices, of the variance `V`: one
/// coefficient per simplex of one grade, in colex order (indexed by
/// [`KSimplexIdx`]).
///
/// [`Chain`] and [`Cochain`] are its two instantiations. Pure combinatorics,
/// carrying no metric and no geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
  feature = "serde",
  serde(bound(
    serialize = "R: serde::Serialize",
    deserialize = "R: serde::Deserialize<'de>"
  ))
)]
pub struct FreeModule<V: Variance, R: Coefficient> {
  grade: Dim,
  coeffs: Vector<R>,
  #[cfg_attr(feature = "serde", serde(skip))]
  variance: PhantomData<V>,
}

/// An integer $k$-chain: a formal $ZZ$-combination $sum_sigma c_sigma sigma$ of
/// the $k$-simplices, an element of the chain group $C_k$.
///
/// $ZZ$ by default because that is the ring homology is computed over: the
/// classes are integral and a representative is an honest combination of
/// simplices, not a rounding of one.
pub type Chain<R = i64> = FreeModule<Homological, R>;

/// A $k$-cochain: one coefficient per $k$-simplex, an element of
/// $C^k = "Hom"(C_k, R)$, the dual of the chain group.
///
/// $RR$ by default because that is the ring the analysis runs over: an operator
/// multiplies a cochain and a solver returns one. Cohomology computed exactly
/// asks for `Cochain<i64>` instead, the same type over the other ring.
pub type Cochain<R = f64> = FreeModule<Cohomological, R>;

/// Add or subtract a coefficient according to a sign.
///
/// The only thing a differential does with the incidence: an incidence
/// coefficient is $plus.minus 1$ in every ring, so no ring element ever
/// represents the sign and no multiplication is performed.
fn accumulate<R: Coefficient>(target: &mut R, sign: Sign, value: &R) {
  match sign {
    Sign::Pos => *target += value.clone(),
    Sign::Neg => *target -= value.clone(),
  }
}

impl<V: Variance, R: Coefficient> FreeModule<V, R> {
  /// From the coefficients, one per $k$-simplex in colex order.
  pub fn new(grade: impl Into<Dim>, coeffs: Vector<R>) -> Self {
    Self {
      grade: grade.into(),
      coeffs,
      variance: PhantomData,
    }
  }
  /// From the coefficients as a plain vector, one per $k$-simplex in colex
  /// order.
  pub fn from_vec(grade: impl Into<Dim>, coeffs: Vec<R>) -> Self {
    Self::new(grade, Vector::from_vec(coeffs))
  }
  /// The constant assignment of a coefficient to every simplex of a skeleton.
  pub fn constant(value: R, skeleton: SkeletonRef) -> Self {
    Self::new(skeleton.dim(), Vector::from_element(skeleton.len(), value))
  }
  pub fn zero(skeleton: SkeletonRef) -> Self {
    Self::constant(R::zero(), skeleton)
  }
  /// From a function of the simplex, evaluated over the grade's skeleton in
  /// colex order.
  pub fn from_function<F>(f: F, grade: impl Into<Dim>, topology: &Complex) -> Self
  where
    F: FnMut(SimplexRef) -> R,
  {
    let grade = grade.into();
    let skeleton = topology.skeleton(grade);
    Self::new(
      grade,
      Vector::from_iterator(skeleton.len(), skeleton.handle_iter().map(f)),
    )
  }

  pub fn grade(&self) -> Dim {
    self.grade
  }
  pub fn coeffs(&self) -> &Vector<R> {
    &self.coeffs
  }
  pub fn coeffs_mut(&mut self) -> &mut Vector<R> {
    &mut self.coeffs
  }
  pub fn into_coeffs(self) -> Vector<R> {
    self.coeffs
  }
  pub fn len(&self) -> usize {
    self.coeffs.len()
  }
  pub fn is_empty(&self) -> bool {
    self.coeffs.is_empty()
  }

  /// The simplices carrying a nonzero coefficient, with that coefficient: the
  /// support.
  pub fn support(&self) -> impl Iterator<Item = (KSimplexIdx, &R)> {
    self.coeffs.iter().enumerate().filter(|(_, c)| !c.is_zero())
  }

  /// The differential of the complex: the boundary $partial$ on a chain, the
  /// coboundary $dif$ on a cochain.
  ///
  /// One traversal of the signed incidence [`Complex::incidences`], the
  /// variance deciding which of its two ends is read from and which is written
  /// to. Exact over any ring: the incidence coefficients are $plus.minus 1$, so
  /// $partial compose partial = 0$ and $dif compose dif = 0$ hold without rounding.
  ///
  /// Total at both ends. Off the range $0 <= k <= n$ the complex extends by the
  /// zero module, the incidence rung is empty and the target skeleton has no
  /// simplices, so the result is the empty element of that zero module rather
  /// than a panic, with no case distinction to make it so.
  pub fn differential(&self, topology: &Complex) -> Self {
    let target = V::target(self.grade);
    let mut coeffs = Vector::zeros(topology.nsimplices(target));
    for (sign, face, coface) in topology.incidences(V::rung(self.grade)) {
      let (from, to) = V::traverse(face, coface);
      accumulate(&mut coeffs[to], sign, &self.coeffs[from]);
    }
    Self::new(target, coeffs)
  }

  /// The image under a ring map $R -> S$: the extension of scalars
  /// $C(K; R) -> C(K; S)$.
  ///
  /// It commutes with the differential, since a ring map fixes $plus.minus 1$.
  /// The caller owes that `ring_map` is one; an arbitrary function of the
  /// coefficients is not, and its image is not a map of complexes.
  pub fn extend_scalars<S: Coefficient>(&self, ring_map: impl FnMut(&R) -> S) -> FreeModule<V, S> {
    FreeModule::new(
      self.grade,
      Vector::from_iterator(self.coeffs.len(), self.coeffs.iter().map(ring_map)),
    )
  }

  /// Whether this could live on `topology`: same grade, one coefficient per
  /// simplex of that grade.
  pub fn is_compatible_with(&self, topology: &Complex) -> bool {
    self.grade <= topology.dim() && self.len() == topology.skeleton(self.grade).len()
  }
}

impl<R: Coefficient> Chain<R> {
  /// The boundary $partial_k: C_k -> C_(k-1)$: the incidence relation scattered
  /// downward, the [`differential`](FreeModule::differential) of this variance.
  pub fn boundary(&self, topology: &Complex) -> Self {
    self.differential(topology)
  }
}

impl<R: Coefficient> Cochain<R> {
  /// The coboundary $dif^k: C^k -> C^(k+1)$: the incidence relation gathered
  /// upward, the [`differential`](FreeModule::differential) of this variance
  /// and the transpose of [`Chain::boundary`].
  pub fn dif(&self, topology: &Complex) -> Self {
    self.differential(topology)
  }

  /// The restriction to a subsimplex: the pullback along the inclusion
  /// $iota_tau: tau arrow.r.hook K$, as a cochain on `simplex` regarded as its
  /// own reference complex ([`Complex::unit`]).
  ///
  /// Combinatorially it is the reading of the cochain on the subsimplices of
  /// `simplex`: the [`faces`](SimplexRef::faces) come in the colex order of
  /// their local vertex positions, the same order the coefficients of the
  /// reference complex take, and colex is preserved under passing to a subset,
  /// so each face's coefficient is read off with no sign. A grade exceeding
  /// `simplex.dim()` has no faces of that grade and restricts to the empty
  /// cochain, which is how it stays total below the cochain's grade.
  ///
  /// A restriction, hence contravariant, which is why it is a cochain
  /// operation: a chain pushes forward along an inclusion rather than pulling
  /// back.
  pub fn trace(&self, simplex: SimplexRef) -> Self {
    let coeffs: Vec<R> = simplex
      .faces(self.grade())
      .map(|face| self[face].clone())
      .collect();
    Self::new(self.grade(), Vector::from_vec(coeffs))
  }

  #[cfg(feature = "serde")]
  pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()>
  where
    R: serde::Serialize,
  {
    crate::io::cbor::save_cbor(self, path)
  }
  #[cfg(feature = "serde")]
  pub fn load(path: impl AsRef<Path>) -> io::Result<Self>
  where
    R: serde::de::DeserializeOwned,
  {
    crate::io::cbor::load_cbor(path)
  }
}

/// The duality pairing $chevron.l omega, c chevron.r = sum_sigma omega_sigma
/// c_sigma$ of a cochain with a chain of the same grade over the same ring.
///
/// The pairing that makes $C^k$ the dual of $C_k$. Under it $partial$ and $dif$
/// are adjoint, $chevron.l dif omega, c chevron.r = chevron.l omega, partial c
/// chevron.r$, which is why the coboundary is the transpose of the boundary.
///
/// One ring, since a bilinear map is over one: a $ZZ$-chain meets an
/// $RR$-cochain by [`extending its scalars`](FreeModule::extend_scalars) first,
/// which names the ring map instead of leaving it implicit.
///
/// A free function, not a method: a pairing is a bilinear map on two spaces and
/// privileges neither of them.
///
/// # Panics
/// If the grades or the lengths disagree.
pub fn pairing<R: Coefficient>(cochain: &Cochain<R>, chain: &Chain<R>) -> R {
  assert_eq!(
    cochain.grade(),
    chain.grade(),
    "a pairing is between a cochain and a chain of one grade"
  );
  assert_eq!(
    cochain.coeffs().len(),
    chain.coeffs().len(),
    "a pairing is over one skeleton"
  );
  cochain
    .coeffs()
    .iter()
    .zip(chain.coeffs().iter())
    .fold(R::zero(), |acc, (coefficient, multiplicity)| {
      acc + coefficient.clone() * multiplicity.clone()
    })
}

/// A chain or a cochain is columnar data over one grade, read like any other:
/// the coefficient of a $k$-simplex, keyed by its id.
///
/// The storage stays an algebraic [`Vector`], not a
/// [`SkeletonVec`](super::data::SkeletonVec), because both are vectors: the
/// differential multiplies them and a solver returns one. The trait carries the
/// reading, the type keeps its own representation.
///
/// Shape alone does not make two such columns the same object. A geometry's
/// signed squared edge lengths are grade-1 columns too, but their datum is
/// quadratic in the edge tangent and so blind to its reversal, while a
/// cochain's is linear and changes sign with it. They sit on the two sides of
/// $Lambda^1 times.o Lambda^1 = Lambda^2 plus.o "Sym"^2$: the
/// coboundary acts on this one, and not on that one.
impl<V: Variance, R: Coefficient> SkeletonData for FreeModule<V, R> {
  type Item<'a>
    = &'a R
  where
    Self: 'a;
  fn grade(&self) -> Dim {
    self.grade
  }
  fn len(&self) -> usize {
    self.coeffs.len()
  }
  fn at(&self, kidx: KSimplexIdx) -> &R {
    &self.coeffs[kidx]
  }
}

impl<V: Variance, R: Coefficient> std::ops::Index<SimplexIdx> for FreeModule<V, R> {
  type Output = R;
  fn index(&self, idx: SimplexIdx) -> &Self::Output {
    assert_eq!(idx.dim(), self.grade());
    &self.coeffs[idx.kidx]
  }
}
impl<V: Variance, R: Coefficient> std::ops::IndexMut<SimplexIdx> for FreeModule<V, R> {
  fn index_mut(&mut self, idx: SimplexIdx) -> &mut Self::Output {
    assert_eq!(idx.dim(), self.grade());
    &mut self.coeffs[idx.kidx]
  }
}

impl<V: Variance, R: Coefficient> std::ops::Index<SimplexRef<'_>> for FreeModule<V, R> {
  type Output = R;
  fn index(&self, handle: SimplexRef<'_>) -> &Self::Output {
    assert_eq!(handle.dim(), self.grade());
    &self.coeffs[handle.kidx()]
  }
}
impl<V: Variance, R: Coefficient> std::ops::IndexMut<SimplexRef<'_>> for FreeModule<V, R> {
  fn index_mut(&mut self, idx: SimplexRef<'_>) -> &mut Self::Output {
    assert_eq!(idx.dim(), self.grade());
    &mut self.coeffs[idx.kidx()]
  }
}

impl<V: Variance, R: Coefficient> std::ops::Index<usize> for FreeModule<V, R> {
  type Output = R;
  fn index(&self, idx: usize) -> &Self::Output {
    &self.coeffs[idx]
  }
}

impl<V: Variance, R: Coefficient> std::ops::Mul<R> for FreeModule<V, R> {
  type Output = Self;
  fn mul(mut self, rhs: R) -> Self::Output {
    self *= rhs;
    self
  }
}
impl<V: Variance, R: Coefficient> std::ops::MulAssign<R> for FreeModule<V, R> {
  fn mul_assign(&mut self, rhs: R) {
    self.coeffs *= rhs;
  }
}
/// Scaling from the left, which coherence cannot state generically: an
/// `impl Mul<FreeModule<V, R>> for R` would be an implementation on a foreign
/// type parameter. The one ring the analysis scales by gets it concretely.
impl<V: Variance> std::ops::Mul<FreeModule<V, f64>> for f64 {
  type Output = FreeModule<V, f64>;
  fn mul(self, rhs: FreeModule<V, f64>) -> Self::Output {
    rhs * self
  }
}
impl<V: Variance, R: Coefficient> Neg for FreeModule<V, R> {
  type Output = Self;
  fn neg(self) -> Self::Output {
    Self::new(self.grade, -self.coeffs)
  }
}
impl<V: Variance, R: Coefficient> std::ops::AddAssign for FreeModule<V, R> {
  fn add_assign(&mut self, rhs: Self) {
    assert_eq!(self.grade, rhs.grade);
    self.coeffs += rhs.coeffs;
  }
}
impl<V: Variance, R: Coefficient> std::ops::Add for FreeModule<V, R> {
  type Output = Self;
  fn add(mut self, rhs: Self) -> Self::Output {
    self += rhs;
    self
  }
}
impl<V: Variance, R: Coefficient> std::ops::SubAssign for FreeModule<V, R> {
  fn sub_assign(&mut self, rhs: Self) {
    assert_eq!(self.grade, rhs.grade);
    self.coeffs -= rhs.coeffs;
  }
}
impl<V: Variance, R: Coefficient> std::ops::Sub for FreeModule<V, R> {
  type Output = Self;
  fn sub(mut self, rhs: Self) -> Self::Output {
    self -= rhs;
    self
  }
}
