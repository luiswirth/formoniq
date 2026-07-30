//! The chain complex of the simplicial complex, and its dual.
//!
//! A [`Chain`] is an element of $C_k$, a formal combination of the oriented
//! $k$-simplices; a [`Cochain`] is an element of $C^k = "Hom"(C_k, RR)$, one
//! coefficient per $k$-simplex. The [`pairing`] between them is what makes the
//! second the dual of the first, and under it the boundary $diff$ and the
//! coboundary $dif$ are adjoint, $angle.l dif omega, c angle.r = angle.l omega,
//! diff c angle.r$.
//!
//! Both differentials are the same datum, the signed incidence
//! [`Complex::incidences`], traversed in its two directions: $diff$ scatters a
//! coface's coefficient onto its faces, $dif$ gathers a coface's coefficient
//! from them. The two types differ in their coefficient ring, and only there:
//! chains carry $ZZ$, so homology is computed exactly, and cochains carry $RR$
//! and are stored as an algebraic [`Vector`], because that is what an operator
//! multiplies and a solver returns.
//!
//! All of this is topology. A cochain becomes a *discrete differential form*
//! only through the de Rham map, which integrates a form over each simplex and
//! needs a geometry; that reading, and the theorem that it is an isomorphism on
//! cohomology, belong to whoever supplies the geometry.

use super::{
  complex::Complex,
  data::SkeletonData,
  handle::{KSimplexIdx, SimplexIdx, SimplexRef},
  skeleton::Skeleton,
};
use crate::Dim;
use crate::linalg::Vector;

#[cfg(feature = "serde")]
use std::{io, path::Path};

/// An integer $k$-chain: a formal $ZZ$-combination $sum_sigma c_sigma sigma$ of
/// the k-simplices, coefficients in colex order (indexed by [`KSimplexIdx`]).
///
/// An element of the chain group $C_k$. Pure combinatorics, carrying no metric
/// and no geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chain {
  grade: Dim,
  coeffs: Vec<i64>,
}
impl Chain {
  /// A chain from its coefficients, one per $k$-simplex in colex order.
  pub fn new(grade: impl Into<Dim>, coeffs: Vec<i64>) -> Self {
    Self {
      grade: grade.into(),
      coeffs,
    }
  }

  pub fn grade(&self) -> Dim {
    self.grade
  }
  /// The coefficient of each k-simplex, in colex order.
  pub fn coeffs(&self) -> &[i64] {
    &self.coeffs
  }
  /// The simplices carrying a nonzero coefficient, with that coefficient: the
  /// support of the chain.
  pub fn support(&self) -> impl Iterator<Item = (KSimplexIdx, i64)> {
    self
      .coeffs
      .iter()
      .enumerate()
      .filter(|&(_, &c)| c != 0)
      .map(|(kidx, &c)| (kidx, c))
  }

  /// The boundary $diff_k: C_k -> C_(k-1)$: the incidence relation scattered
  /// downward, restricted to the chain's support.
  ///
  /// Exact over $ZZ$: the incidence coefficients are $plus.minus 1$, so the
  /// chain complex stays integral and $diff compose diff = 0$ holds without
  /// rounding.
  ///
  /// Total at the ends. Below grade zero there is nothing to bound, and the
  /// complex extends by zero either way, so the result is the empty chain
  /// rather than a panic.
  pub fn boundary(&self, topology: &Complex) -> Self {
    if self.grade == 0 {
      return Self::new(self.grade - 1, Vec::new());
    }
    let mut coeffs = vec![0i64; topology.nsimplices(self.grade - 1)];
    let skeleton = topology.skeleton(self.grade);
    for (kidx, multiplicity) in self.support() {
      for (sign, face) in skeleton.handle_by_kidx(kidx).boundary() {
        coeffs[face.kidx()] += sign.as_i32() as i64 * multiplicity;
      }
    }
    Self::new(self.grade - 1, coeffs)
  }
}

/// A $k$-cochain: one real coefficient per $k$-simplex of the skeleton.
///
/// An element of $C^k = "Hom"(C_k, RR)$, the dual of the chain group, hence a
/// vector space of dimension the number of $k$-simplices.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cochain {
  coeffs: Vector,
  grade: Dim,
}
impl Cochain {
  pub fn new(grade: impl Into<Dim>, coeffs: Vector) -> Self {
    Self {
      coeffs,
      grade: grade.into(),
    }
  }
  pub fn constant(value: f64, skeleton: &Skeleton) -> Self {
    let ncoeffs = skeleton.len();
    Self::new(skeleton.dim(), Vector::from_element(ncoeffs, value))
  }
  pub fn zero(skeleton: &Skeleton) -> Self {
    Self::constant(0.0, skeleton)
  }
  pub fn from_function<F>(f: F, grade: impl Into<Dim>, topology: &Complex) -> Self
  where
    F: FnMut(SimplexRef) -> f64,
  {
    let grade = grade.into();
    let skeleton = topology.skeleton(grade);
    let coeffs = Vector::from_iterator(skeleton.len(), skeleton.handle_iter().map(f));
    Self::new(grade, coeffs)
  }

  pub fn grade(&self) -> Dim {
    self.grade
  }
  pub fn coeffs(&self) -> &Vector {
    &self.coeffs
  }
  pub fn coeffs_mut(&mut self) -> &mut Vector {
    &mut self.coeffs
  }
  pub fn into_coeffs(self) -> Vector {
    self.coeffs
  }
  pub fn len(&self) -> usize {
    self.coeffs.len()
  }
  pub fn is_empty(&self) -> bool {
    self.coeffs.is_empty()
  }

  /// The coboundary $dif: C^k -> C^(k+1)$: the incidence relation gathered
  /// upward, the transpose of [`Chain::boundary`].
  ///
  /// Total at the top, where there are no cofaces and the image is the empty
  /// cochain of the zero space $C^(n+1)$.
  pub fn dif(&self, topology: &Complex) -> Self {
    let mut coeffs = Vector::zeros(topology.nsimplices(self.grade + 1));
    for (sign, face, coface) in topology.incidences(self.grade) {
      coeffs[coface] += sign.as_f64() * self.coeffs[face];
    }
    Self::new(self.grade + 1, coeffs)
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
  pub fn trace(&self, simplex: SimplexRef) -> Self {
    let coeffs: Vec<f64> = simplex.faces(self.grade).map(|face| self[face]).collect();
    Self::new(self.grade, Vector::from_vec(coeffs))
  }

  /// Whether this could be a cochain on `topology`: same grade, one
  /// coefficient per simplex of that grade.
  pub fn is_compatible_with(&self, topology: &Complex) -> bool {
    self.grade <= topology.dim() && self.len() == topology.skeleton(self.grade).len()
  }

  #[cfg(feature = "serde")]
  pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
    crate::io::cbor::save_cbor(self, path)
  }
  #[cfg(feature = "serde")]
  pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
    crate::io::cbor::load_cbor(path)
  }
}

/// The duality pairing $angle.l omega, c angle.r = sum_sigma omega_sigma
/// c_sigma$ of a cochain with a chain of the same grade.
///
/// The pairing that makes $C^k$ the dual of $C_k$. Under it $diff$ and $dif$
/// are adjoint, $angle.l dif omega, c angle.r = angle.l omega, diff c
/// angle.r$, which is why the coboundary is the transpose of the boundary.
///
/// A free function, not a method: a pairing is a bilinear map on two spaces and
/// privileges neither of them.
///
/// # Panics
/// If the grades or the lengths disagree.
pub fn pairing(cochain: &Cochain, chain: &Chain) -> f64 {
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
    .zip(chain.coeffs())
    .map(|(coefficient, multiplicity)| coefficient * *multiplicity as f64)
    .sum()
}

/// A chain is columnar data over one grade: the multiplicity of each
/// $k$-simplex, keyed by its id.
impl SkeletonData for Chain {
  type Item<'a>
    = &'a i64
  where
    Self: 'a;
  fn grade(&self) -> Dim {
    self.grade
  }
  fn len(&self) -> usize {
    self.coeffs.len()
  }
  fn at(&self, kidx: KSimplexIdx) -> &i64 {
    &self.coeffs[kidx]
  }
}

/// A cochain is columnar data over one grade, read like any other: the
/// coefficient of a $k$-simplex, keyed by its id.
///
/// The storage stays an algebraic [`Vector`], not a
/// [`SkeletonVec`](super::data::SkeletonVec), because a cochain is a vector:
/// the coboundary multiplies it and a solver returns it. The trait carries the
/// reading, the type keeps its own representation.
///
/// Shape alone does not make two such columns the same object. A geometry's
/// signed squared edge lengths are grade-1 columns too, but their datum is
/// quadratic in the edge tangent and so blind to its reversal, while a
/// cochain's is linear and changes sign with it. They sit on the two sides of
/// $Lambda^1 times.circle Lambda^1 = Lambda^2 plus.circle "Sym"^2$: the
/// coboundary acts on this one, and not on that one.
impl SkeletonData for Cochain {
  type Item<'a>
    = &'a f64
  where
    Self: 'a;
  fn grade(&self) -> Dim {
    self.grade
  }
  fn len(&self) -> usize {
    self.coeffs.len()
  }
  fn at(&self, kidx: KSimplexIdx) -> &f64 {
    &self.coeffs[kidx]
  }
}

impl std::ops::Index<SimplexIdx> for Cochain {
  type Output = f64;
  fn index(&self, idx: SimplexIdx) -> &Self::Output {
    assert_eq!(idx.dim(), self.grade());
    &self.coeffs[idx.kidx]
  }
}
impl std::ops::IndexMut<SimplexIdx> for Cochain {
  fn index_mut(&mut self, idx: SimplexIdx) -> &mut Self::Output {
    assert_eq!(idx.dim(), self.grade());
    &mut self.coeffs[idx.kidx]
  }
}

impl std::ops::Index<SimplexRef<'_>> for Cochain {
  type Output = f64;
  fn index(&self, handle: SimplexRef<'_>) -> &Self::Output {
    assert_eq!(handle.dim(), self.grade());
    &self.coeffs[handle.kidx()]
  }
}
impl std::ops::IndexMut<SimplexRef<'_>> for Cochain {
  fn index_mut(&mut self, idx: SimplexRef<'_>) -> &mut Self::Output {
    assert_eq!(idx.dim(), self.grade());
    &mut self.coeffs[idx.kidx()]
  }
}

impl std::ops::Index<usize> for Cochain {
  type Output = f64;
  fn index(&self, idx: usize) -> &Self::Output {
    &self.coeffs[idx]
  }
}

impl std::ops::Mul<f64> for Cochain {
  type Output = Cochain;
  fn mul(mut self, rhs: f64) -> Self::Output {
    self *= rhs;
    self
  }
}
impl std::ops::Mul<Cochain> for f64 {
  type Output = Cochain;
  fn mul(self, rhs: Cochain) -> Self::Output {
    rhs * self
  }
}
impl std::ops::MulAssign<f64> for Cochain {
  fn mul_assign(&mut self, rhs: f64) {
    self.coeffs *= rhs;
  }
}
impl std::ops::Neg for Cochain {
  type Output = Self;
  fn neg(self) -> Self::Output {
    Self::new(self.grade, -self.coeffs)
  }
}
impl std::ops::AddAssign for Cochain {
  fn add_assign(&mut self, rhs: Self) {
    assert_eq!(self.grade, rhs.grade);
    self.coeffs += rhs.coeffs;
  }
}
impl std::ops::Add for Cochain {
  type Output = Self;
  fn add(mut self, rhs: Self) -> Self::Output {
    self += rhs;
    self
  }
}
impl std::ops::SubAssign for Cochain {
  fn sub_assign(&mut self, rhs: Self) {
    assert_eq!(self.grade, rhs.grade);
    self.coeffs -= rhs.coeffs;
  }
}
impl std::ops::Sub for Cochain {
  type Output = Self;
  fn sub(mut self, rhs: Self) -> Self::Output {
    self -= rhs;
    self
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::mesher::grid::CartesianTopology;

  fn probe_complex(dim: usize) -> Complex {
    CartesianTopology::cube(dim, 2).triangulate()
  }

  /// A probe chain and cochain of a grade, with coefficients that are neither
  /// constant nor symmetric, so a law cannot pass by accident.
  fn probe_chain(topology: &Complex, grade: usize) -> Chain {
    let coeffs = (0..topology.nsimplices(grade))
      .map(|i| (i % 7) as i64 - 3)
      .collect();
    Chain::new(grade, coeffs)
  }
  fn probe_cochain(topology: &Complex, grade: usize) -> Cochain {
    Cochain::from_function(|s| ((s.kidx() % 5) as f64) - 2.0, grade, topology)
  }

  /// The boundary and the coboundary are adjoint under the chain-cochain
  /// pairing: $angle.l dif omega, c angle.r = angle.l omega, diff c angle.r$.
  ///
  /// The statement that makes $C^k$ the dual complex of $C_k$ rather than
  /// merely a vector space of the same dimension, and the reason the coboundary
  /// is the transpose of the boundary. It holds with no metric, no orientation
  /// and no geometry.
  #[test]
  fn the_boundary_and_the_coboundary_are_adjoint() {
    for dim in 1..=3 {
      let topology = probe_complex(dim);
      for grade in 0..dim {
        let cochain = probe_cochain(&topology, grade);
        let chain = probe_chain(&topology, grade + 1);

        let differentiated = pairing(&cochain.dif(&topology), &chain);
        let bounded = pairing(&cochain, &chain.boundary(&topology));

        assert!(
          differentiated.abs() > 1e-9,
          "dim {dim} grade {grade}: the law would hold vacuously"
        );
        assert!(
          (differentiated - bounded).abs() < 1e-9,
          "dim {dim} grade {grade}: {differentiated} != {bounded}"
        );
      }
    }
  }

  /// $diff compose diff = 0$ and $dif compose dif = 0$ are the same statement
  /// read through the pairing, so neither can hold while the other fails.
  ///
  /// Both are checked to be nonzero one step earlier, since a pairing that
  /// vanished for its own reasons would satisfy this without either operator
  /// being nilpotent.
  #[test]
  fn nilpotency_is_one_statement_on_both_sides() {
    for dim in 2..=3 {
      let topology = probe_complex(dim);
      for grade in 0..dim - 1 {
        let cochain = probe_cochain(&topology, grade);
        let chain = probe_chain(&topology, grade + 2);

        // One step must not already vanish, or both halves below hold for the
        // wrong reason. It is checked against a chain one grade up rather than
        // against the boundary: adjointness makes
        // $angle.l dif omega, diff c angle.r = angle.l dif dif omega, c angle.r$,
        // which is zero for the very reason being tested.
        assert!(
          pairing(&cochain.dif(&topology), &probe_chain(&topology, grade + 1)).abs() > 1e-9,
          "dim {dim} grade {grade}: one step already vanishes"
        );
        let twice_up = pairing(&cochain.dif(&topology).dif(&topology), &chain);
        let twice_down = pairing(&cochain, &chain.boundary(&topology).boundary(&topology));

        assert!(twice_up.abs() < 1e-9, "dim {dim} grade {grade}: dd != 0");
        assert!(twice_down.abs() < 1e-9, "dim {dim} grade {grade}: bb != 0");
      }
    }
  }

  /// The differentials agree with the assembled operator: $dif$ is the
  /// transpose of $diff$ as a matrix, and both are the same incidence the
  /// coefficient-wise sweeps read.
  #[test]
  fn the_differentials_agree_with_the_assembled_operators() {
    use crate::linalg::CsrMatrix;
    for dim in 1..=3 {
      let topology = probe_complex(dim);
      for grade in 0..=dim {
        let cochain = probe_cochain(&topology, grade);
        let assembled =
          CsrMatrix::from(&topology.coboundary_operator(grade.into())) * cochain.coeffs();
        assert_eq!(cochain.dif(&topology).coeffs(), &assembled);

        let chain = probe_chain(&topology, grade);
        let boundary = chain.boundary(&topology);
        let matrix = CsrMatrix::from(topology.boundary_operator(grade.into()));
        let applied = matrix
          * Vector::from_iterator(
            chain.coeffs().len(),
            chain.coeffs().iter().map(|&c| c as f64),
          );
        for (kidx, &coefficient) in boundary.coeffs().iter().enumerate() {
          assert_eq!(coefficient as f64, applied[kidx]);
        }
      }
    }
  }

  /// The pairing is bilinear and reads the coefficients it says it does.
  #[test]
  fn the_pairing_sums_over_the_simplices() {
    for dim in 1..=3 {
      let topology = probe_complex(dim);
      for grade in 0..=dim {
        let cochain = probe_cochain(&topology, grade);
        let chain = probe_chain(&topology, grade);

        let expected: f64 = chain
          .support()
          .map(|(kidx, multiplicity)| cochain.coeffs()[kidx] * multiplicity as f64)
          .sum();
        assert!((pairing(&cochain, &chain) - expected).abs() < 1e-12);
      }
    }
  }

  /// Both readings of a chain and of a cochain, as columnar data and by their
  /// own accessors, are one column.
  #[test]
  fn skeleton_data_reading_agrees_with_indexing() {
    for dim in 1..=3 {
      let topology = probe_complex(dim);
      for grade in 0..=dim {
        let cochain = probe_cochain(&topology, grade);
        let chain = probe_chain(&topology, grade);
        let skeleton = topology.skeleton(grade);

        assert_eq!(SkeletonData::grade(&cochain), skeleton.dim());
        assert_eq!(SkeletonData::len(&cochain), skeleton.len());
        assert_eq!(SkeletonData::grade(&chain), skeleton.dim());

        for simplex in skeleton.handle_iter() {
          assert_eq!(*cochain.at_ref(simplex), cochain[simplex.idx()]);
          assert_eq!(*chain.at_ref(simplex), chain.coeffs()[simplex.kidx()]);
        }
      }
    }
  }

  #[cfg(feature = "serde")]
  #[test]
  fn save_load_roundtrip_and_compatibility() {
    let topology = probe_complex(2);
    let cochain = Cochain::from_function(|s| s.kidx() as f64, 1, &topology);
    assert!(cochain.is_compatible_with(&topology));

    let path = std::env::temp_dir().join(format!("simplicial_test_{}.cbor", std::process::id()));
    cochain.save(&path).unwrap();
    let loaded = Cochain::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(loaded.grade(), cochain.grade());
    assert_eq!(loaded.coeffs(), cochain.coeffs());

    let other = CartesianTopology::cube(2, 5).triangulate();
    assert!(!loaded.is_compatible_with(&other));
  }
}
