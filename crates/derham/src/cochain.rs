use simplicial::linalg::{CsrMatrix, Vector};

use {
  multialgebra::ExteriorGrade,
  simplicial::{
    topology::skeleton::Skeleton,
    topology::{
      complex::Complex,
      data::SkeletonData,
      handle::{KSimplexIdx, SimplexIdx, SimplexRef},
      homology::Chain,
    },
  },
};

#[cfg(feature = "serde")]
use std::{io, path::Path};

/// The duality pairing $angle.l omega, c angle.r = sum_sigma omega_sigma
/// c_sigma$ of a cochain with a chain of the same grade.
///
/// The pairing that makes $C^k$ the dual of $C_k$, and the discrete reading of
/// $integral_c omega$: a cochain is the assignment of an integral to each
/// simplex, so summing against a chain's coefficients integrates over the
/// chain.
///
/// Metric-free, and so is its adjunction. $diff$ and $dif$ are adjoint here,
/// $angle.l dif omega, c angle.r = angle.l omega, diff c angle.r$, which is
/// Stokes' theorem in its discrete form and the reason the coboundary is the
/// transpose of the boundary.
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

/// A $k$-cochain: one real coefficient per $k$-simplex of the skeleton.
///
/// An element of the cochain space $C^k$, hence a vector space over the
/// simplices of a fixed grade.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Cochain {
  coeffs: Vector,
  grade: ExteriorGrade,
}
impl Cochain {
  pub fn new(grade: impl Into<ExteriorGrade>, coeffs: Vector) -> Self {
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
  pub fn from_function<F>(f: F, grade: impl Into<ExteriorGrade>, topology: &Complex) -> Self
  where
    F: FnMut(SimplexRef) -> f64,
  {
    let grade = grade.into();
    let skeleton = topology.skeleton(grade);
    let coeffs = Vector::from_iterator(skeleton.len(), skeleton.handle_iter().map(f));
    Self::new(grade, coeffs)
  }

  pub fn grade(&self) -> ExteriorGrade {
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

  /// The discrete exterior derivative $dif: C^k -> C^(k+1)$: the coboundary
  /// operator applied to this cochain's coefficients.
  pub fn dif(&self, topology: &Complex) -> Self {
    let dif_operator = CsrMatrix::from(&topology.coboundary_operator(self.grade()));
    Cochain::new(self.grade() + 1, dif_operator * self.coeffs())
  }

  /// The trace onto a subsimplex: the pullback of the discrete form along the
  /// inclusion $iota_tau: tau arrow.r.hook M$, as a cochain on `simplex`
  /// regarded as its own reference cell
  /// ([`Complex::unit`](simplicial::topology::complex::Complex::unit)).
  ///
  /// Metric-free (invariant 5: the trace is a pullback). Combinatorially it is
  /// the restriction of the cochain to the subsimplices of `simplex`: the
  /// [`faces`](SimplexRef::faces) come in the colex order of their local vertex
  /// positions, the same order the DOFs of the reference cell take, and colex is
  /// preserved under passing to a subset, so each face's coefficient is read off
  /// with no sign. A grade exceeding `simplex.dim()` has no faces of that grade
  /// and traces to the empty cochain, the zero of $Lambda^k(tau) = 0$, which
  /// is how the trace stays total below the form's grade.
  ///
  /// Whitney interpolation commutes with it, $tr_tau compose W = W_tau compose
  /// tr_tau$ (test `whitney_trace_commutes` in
  /// [`crate::interpolate`]): the Whitney field on the subsimplex is the
  /// interpolation of the traced cochain, so the trace of a Whitney form comes
  /// for free.
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
    simplicial::io::cbor::save_cbor(self, path)
  }
  #[cfg(feature = "serde")]
  pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
    simplicial::io::cbor::load_cbor(path)
  }
}

/// A cochain is columnar data over one grade, read like any other: the
/// coefficient of a $k$-simplex, keyed by its id.
///
/// The storage stays an algebraic [`Vector`], not a
/// [`SkeletonVec`](simplicial::topology::data::SkeletonVec), because a cochain
/// is a vector: the coboundary multiplies it and the mass matrices pair it.
/// The trait carries the reading, the type keeps its own representation.
///
/// Shape alone does not make two such columns the same object. Signed squared
/// edge lengths
/// ([`MeshLengthsSq`](regge::lengths::mesh::MeshLengthsSq)) are
/// grade-1 columns too, but their datum $g(t_e, t_e)$ is quadratic in the
/// edge tangent and so blind to its reversal, while a cochain's $integral_e
/// omega$ is linear and changes sign with it. They sit on the two sides of
/// $Lambda^1 times.circle Lambda^1 = Lambda^2 plus.circle "Sym"^2$: the
/// coboundary acts on this one, and not on that one.
impl SkeletonData for Cochain {
  type Item<'a>
    = &'a f64
  where
    Self: 'a;
  fn grade(&self) -> ExteriorGrade {
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
  use regge::mesher::cartesian::CartesianGrid;

  #[cfg(feature = "serde")]
  #[test]
  fn save_load_roundtrip_and_compatibility() {
    let (topology, _) = CartesianGrid::new_unit(2, 3).triangulate();
    let cochain = Cochain::from_function(|s| s.kidx() as f64, 1, &topology);
    assert!(cochain.is_compatible_with(&topology));

    let path = std::env::temp_dir().join(format!("formoniq_test_{}.cbor", std::process::id()));
    cochain.save(&path).unwrap();
    let loaded = Cochain::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(loaded.grade(), cochain.grade());
    assert_eq!(loaded.coeffs(), cochain.coeffs());

    let other = CartesianGrid::new_unit(2, 5).triangulate().0;
    assert!(!loaded.is_compatible_with(&other));
  }

  /// A cochain read as columnar data over its grade agrees with reading it as
  /// a cochain, at every grade of every dimension: the two are one column, not
  /// two views that could drift.
  #[test]
  fn skeleton_data_reading_agrees_with_cochain_indexing() {
    for dim in 1..=3 {
      let (topology, _) = CartesianGrid::new_unit(dim, 2).triangulate();
      for grade in 0..=dim {
        let cochain = Cochain::from_function(|s| s.kidx() as f64 + 0.5, grade, &topology);
        let skeleton = topology.skeleton(grade);

        assert_eq!(SkeletonData::grade(&cochain), skeleton.dim());
        assert_eq!(SkeletonData::len(&cochain), skeleton.len());

        for simplex in skeleton.handle_iter() {
          assert_eq!(*cochain.at_ref(simplex), cochain[simplex.idx()]);
        }
      }
    }
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

  /// The boundary and the exterior derivative are adjoint under the
  /// chain-cochain pairing: $angle.l dif omega, c angle.r
  /// = angle.l omega, diff c angle.r$.
  ///
  /// Discrete Stokes, and the statement that makes $C^k$ the dual complex of
  /// $C_k$ rather than merely a vector space of the same dimension. It is the
  /// reason the coboundary is the transpose of the boundary, so it holds with
  /// no metric, no orientation and no geometry.
  #[test]
  fn the_boundary_and_the_exterior_derivative_are_adjoint() {
    for dim in 1..=3 {
      let (topology, _) = CartesianGrid::new_unit(dim, 2).triangulate();
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
      let (topology, _) = CartesianGrid::new_unit(dim, 2).triangulate();
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

  /// The pairing is bilinear and reads the coefficients it says it does.
  #[test]
  fn the_pairing_sums_over_the_simplices() {
    for dim in 1..=3 {
      let (topology, _) = CartesianGrid::new_unit(dim, 2).triangulate();
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
}
