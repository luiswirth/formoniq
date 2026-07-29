//! Polynomial differential forms on the reference cell: $P_r Lambda^k$ and the
//! trimmed $P^-_r Lambda^k$.
//!
//! $P_r Lambda^k (hat(K)) = "Sym"^r times.circle Lambda^k$ over the $n+1$
//! barycentric coordinates. Barycentric coordinates make the coefficient factor
//! homogeneous, $sum_i lambda_i = 1$ supplying the missing variable, so
//! $dim P_r (RR^n) = binom(n+r, n) = dim "Sym"^r (RR^(n+1))$ and no graded layer
//! is needed above it.
//!
//! $dif$ and $kappa$ are one operation in opposite directions, moving a degree
//! between the two factors. Both metric-free.
//!
//! Everything is on the reference cell: every chart is the same chart up to
//! vertex labelling.

use multialgebra::Variance;
use multialgebra::{
  ExteriorGrade, Factor, Tensor,
  tensor::{Slots, covariant_slots},
};
use multiindex::{
  Combination, Degree, Dim, MonoIndex, MultiIndex, Repetition, binomial, factorial_f64,
};
use simplicial::atlas::{BaryRef, unit_difbarys};

/// The polynomial degree $r$ of a form in $P_r Lambda^k$.
pub type PolyDegree = Degree;

/// The index of the symmetric factor, and of the alternating one.
///
/// Which factor is acted on is the whole content of [`PolyForm::dif`] against
/// [`PolyForm::koszul`].
const COEFFICIENTS: usize = 0;
const BLADE: usize = 1;

/// An element of $P_r Lambda^k (hat(K))$: a polynomial differential $k$-form on
/// the reference cell, in barycentric coordinates.
///
/// The blade factor lives in the formal $Lambda^k (RR^(n+1))$ spanned by the
/// $dif lambda_i$, one dimension too many since $sum_i dif lambda_i = 0$. The
/// relation is not quotiented out here but applied where a value is taken, by
/// pulling back along [`unit_difbarys`], whose kernel it is. Keeping the formal
/// space keeps the combinatorics symmetric in the $n+1$ vertices.
#[derive(Debug, Clone)]
pub struct PolyForm {
  /// $"Sym"^r times.circle Lambda^k$ over the $n+1$ barycentric coordinates.
  tensor: Tensor,
}

impl PolyForm {
  /// The cell dimension $n$, one less than the number of barycentric
  /// coordinates.
  pub fn cell_dim(&self) -> Dim {
    self.tensor.dim() - 1
  }
  /// The polynomial degree $r$.
  pub fn degree(&self) -> PolyDegree {
    self.tensor.slots()[COEFFICIENTS].degree()
  }
  /// The form grade $k$.
  pub fn grade(&self) -> ExteriorGrade {
    self.tensor.slots()[BLADE].degree()
  }
  pub fn tensor(&self) -> &Tensor {
    &self.tensor
  }

  pub fn from_tensor(tensor: Tensor) -> Self {
    Self { tensor }
  }

  /// The shape of $P_r Lambda^k$ on an $n$-cell: both slots over the $n+1$
  /// barycentric coordinates, which is the space this lives in.
  fn slots(
    cell_dim: impl Into<Dim>,
    degree: impl Into<PolyDegree>,
    grade: impl Into<ExteriorGrade>,
  ) -> Slots {
    covariant_slots(
      [Factor::symmetric(degree), Factor::alternating(grade)],
      cell_dim.into() + 1,
    )
  }

  /// The zero form of the given degree and grade.
  pub fn zero(
    cell_dim: impl Into<Dim>,
    degree: impl Into<PolyDegree>,
    grade: impl Into<ExteriorGrade>,
  ) -> Self {
    let cell_dim = cell_dim.into();
    Self::from_tensor(Tensor::zero(Self::slots(cell_dim, degree, grade)))
  }

  /// The monomial-times-blade basis element
  /// $lambda^alpha dif lambda_I$.
  pub fn monomial(cell_dim: impl Into<Dim>, coefficients: MonoIndex, blade: Combination) -> Self {
    let cell_dim = cell_dim.into();
    let mut form = Self::zero(cell_dim, coefficients.degree(), blade.card());
    let index = [
      MultiIndex::Mono(coefficients),
      MultiIndex::Mono(MonoIndex::from(blade)),
    ];
    let flat = form.tensor.flat_index(&index);
    form.tensor.components_mut()[flat] = 1.0;
    form
  }

  /// The exterior derivative $dif: P_r Lambda^k -> P_(r-1) Lambda^(k+1)$.
  ///
  /// One degree moved from the coefficients into the blade. Metric-free, and
  /// exact rather than discretized.
  pub fn dif(&self) -> Self {
    Self::from_tensor(self.tensor.transfer(COEFFICIENTS, BLADE))
  }

  /// The Koszul operator $kappa: P_r Lambda^k -> P_(r+1) Lambda^(k-1)$.
  ///
  /// The same degree moved the other way: contraction with the position vector,
  /// which in barycentric coordinates is $lambda$ itself.
  ///
  /// $kappa^2 = 0$ and $dif kappa + kappa dif = (r + k) id$ on the homogeneous
  /// pieces, which is what cuts out the trimmed spaces.
  pub fn koszul(&self) -> Self {
    Self::from_tensor(self.tensor.transfer(BLADE, COEFFICIENTS))
  }

  /// The same form, written one degree higher.
  ///
  /// Multiplication by $sum_i lambda_i = 1$: the identity as a function, a
  /// change of representation as a tensor. Needed to compare forms of different
  /// polynomial degree, which $dif$ produces.
  pub fn raise(&self) -> Self {
    let nvertices = (self.cell_dim() + 1).index();
    let unit = Tensor::new(
      Self::slots(self.cell_dim(), 1, 0),
      multialgebra::Vector::from_element(nvertices, 1.0),
    );
    Self::from_tensor(self.tensor.product(&unit))
  }

  /// The components in a faithful representation of the form.
  ///
  /// The tensor is not faithful, $sum_i dif lambda_i = 0$. Pulling the blade
  /// factor back along [`unit_difbarys`] applies that relation and lands in
  /// $"Sym"^r times.circle Lambda^k (RR^n) = P_r Lambda^k$; the coefficient
  /// factor is already faithful.
  ///
  /// Two forms are equal exactly when these agree.
  pub fn faithful_components(&self) -> multialgebra::Vector {
    let pullback = multialgebra::exterior_power(&unit_difbarys(self.cell_dim()), self.grade());
    let blade_dim = pullback.ncols();
    let ncoefficients = self.tensor.components().len() / pullback.nrows().max(1);
    let reshaped = multialgebra::Matrix::from_fn(ncoefficients, pullback.nrows(), |i, j| {
      self.tensor.components()[i * pullback.nrows() + j]
    });
    let pulled = reshaped * &pullback;
    multialgebra::Vector::from_fn(ncoefficients * blade_dim, |i, _| {
      pulled[(i / blade_dim, i % blade_dim)]
    })
  }

  /// The value at a barycentric point, as a $k$-form in the reference frame.
  ///
  /// Evaluate the coefficient factor at $lambda$, then pull the remaining
  /// formal blade back along the barycentric differentials.
  pub fn at_bary<'a>(&self, bary: impl Into<BaryRef<'a>>) -> Tensor {
    let point = Tensor::line(bary.into().view().into_owned(), Variance::Contravariant);
    let blade = self.tensor.evaluate(COEFFICIENTS, &point);
    blade.pullback(&unit_difbarys(self.cell_dim()))
  }
}

impl std::ops::Add for PolyForm {
  type Output = Self;
  fn add(self, other: Self) -> Self {
    Self::from_tensor(self.tensor + other.tensor)
  }
}
impl std::ops::Mul<f64> for PolyForm {
  type Output = Self;
  fn mul(self, scalar: f64) -> Self {
    Self::from_tensor(self.tensor * scalar)
  }
}

/// The Whitney form $W_sigma$ of a subsimplex, as a polynomial form.
///
/// $W_sigma = k! space kappa (dif lambda_sigma)$. Expanding the Koszul deletion
/// gives the familiar
/// $W_sigma = k! sum_i (-1)^i lambda_(sigma_i) dif lambda_(sigma without sigma_i)$.
///
/// These are the basis of $P^-_1 Lambda^k$.
pub fn whitney(cell_dim: impl Into<Dim>, dof_simp: Combination) -> PolyForm {
  let cell_dim = cell_dim.into();
  let grade = dof_simp.card() - 1;
  let constant = PolyForm::monomial(cell_dim, MonoIndex::empty(Repetition::Allowed), dof_simp);
  constant.koszul() * factorial_f64(grade)
}

/// The dimension of $P_r Lambda^k$ on an $n$-simplex:
/// $binom(n+r, n) binom(n, k)$.
pub fn poly_dim(
  cell_dim: impl Into<Dim>,
  degree: impl Into<PolyDegree>,
  grade: impl Into<ExteriorGrade>,
) -> usize {
  let (n, r, k) = (cell_dim.into(), degree.into(), grade.into());
  if r.get() < 0 || !k.in_range(n) {
    return 0;
  }
  binomial(n.index() + r.index(), n.index()) * binomial(n.index(), k.index())
}

/// The dimension of the trimmed space $P^-_r Lambda^k$ on an $n$-simplex:
/// $binom(r+n, r+k) binom(r+k-1, k)$.
///
/// Between $P_(r-1) Lambda^k$ and $P_r Lambda^k$: the smallest space containing
/// the former that still closes the de Rham complex.
pub fn trimmed_dim(
  cell_dim: impl Into<Dim>,
  degree: impl Into<PolyDegree>,
  grade: impl Into<ExteriorGrade>,
) -> usize {
  let (n, r, k) = (cell_dim.into(), degree.into(), grade.into());
  if r.get() < 1 || !k.in_range(n) {
    return 0;
  }
  let (n, r, k) = (n.index(), r.index(), k.index());
  binomial(r + n, r + k) * binomial(r + k - 1, k)
}

/// The basis of $P^-_r Lambda^k (hat(K))$, each element paired with the
/// subsimplex it is attached to.
///
/// The basis is $lambda^alpha W_sigma$ over the pairs with $abs(alpha) = r-1$,
/// $sigma$ a $k$-subsimplex, and $alpha_i = 0$ for every $i$ below the smallest
/// vertex of $sigma$. Without that last condition the family spans but
/// over-counts, the $W_sigma$ satisfying relations once multiplied by
/// coordinates.
///
/// The subsimplex is the attachment of the geometric decomposition, which
/// [`GeometricDecomposition`](crate::decomposition::GeometricDecomposition)
/// indexes.
pub fn trimmed_basis(
  cell_dim: impl Into<Dim>,
  degree: impl Into<PolyDegree>,
  grade: impl Into<ExteriorGrade>,
) -> Vec<(Combination, PolyForm)> {
  let (cell_dim, degree, grade) = (cell_dim.into(), degree.into(), grade.into());
  if degree.get() < 1 || !grade.in_range(cell_dim) {
    return Vec::new();
  }
  let nvertices = (cell_dim + 1).index();

  let mut basis = Vec::new();
  for dof_simp in multiindex::combinations(nvertices, grade.index() + 1) {
    let lowest = dof_simp.iter().next().expect("a dof simplex has a vertex");
    let form = whitney(cell_dim, dof_simp);
    for coefficients in MonoIndex::all(Repetition::Allowed, nvertices, degree.index() - 1) {
      // Independence: no coordinate below the subsimplex's own lowest vertex.
      if coefficients.iter().any(|symbol| symbol < lowest) {
        continue;
      }
      let monomial = PolyForm::monomial(cell_dim, coefficients, Combination::empty());
      basis.push((
        dof_simp,
        PolyForm::from_tensor(monomial.tensor.product(&form.tensor)),
      ));
    }
  }
  basis
}

#[cfg(test)]
mod test {
  use super::*;
  use approx::assert_relative_eq;
  use simplicial::atlas::Bary;

  /// $dim P_r Lambda^k = binom(n+r, n) binom(n, k)$, against the tensor the
  /// space is, after the barycentric relation.
  #[test]
  fn the_polynomial_space_has_the_dimension_it_claims() {
    for n in 0..=4 {
      for r in 0..=3 {
        for k in 0..=n {
          // Homogeneous of degree r in n+1 barycentric coordinates is degree
          // at most r in n local ones.
          let coefficients = Factor::symmetric(r).multidim(n + 1);
          assert_eq!(coefficients, binomial(n + r, n));
          assert_eq!(poly_dim(n, r, k), coefficients * binomial(n, k));
        }
      }
    }
  }

  /// The trimmed space sits strictly between the two full spaces it is
  /// squeezed by, $P_(r-1) Lambda^k subset P^-_r Lambda^k subset P_r Lambda^k$,
  /// and coincides with them at the ends: $P_r$ at $k = 0$, $P_(r-1)$ at
  /// $k = n$. Lagrange and discontinuous elements as the two special cases.
  #[test]
  fn the_trimmed_space_is_squeezed_between_the_full_ones() {
    for n in 1..=4 {
      for r in 1..=4 {
        for k in 0..=n {
          let trimmed = trimmed_dim(n, r, k);
          assert!(poly_dim(n, r - 1, k) <= trimmed);
          assert!(trimmed <= poly_dim(n, r, k));
          if k == 0 {
            assert_eq!(trimmed, poly_dim(n, r, 0));
          }
          if k == n {
            assert_eq!(trimmed, poly_dim(n, r - 1, n));
          }
        }
      }
    }
  }

  /// The constructed basis of $P^-_r Lambda^k$ has the dimension the formula
  /// gives, and is linearly independent.
  ///
  /// The count alone passes on a family that repeats one element and omits
  /// another, the failure the lowest-vertex condition guards against, so
  /// independence is checked by rank.
  #[test]
  fn the_trimmed_basis_is_a_basis() {
    for n in 1..=3 {
      for r in 1..=3 {
        for k in 0..=n {
          let basis = trimmed_basis(n, r, k);
          assert_eq!(basis.len(), trimmed_dim(n, r, k), "n={n} r={r} k={k}");

          let ncomponents = basis[0].1.tensor().components().len();
          let matrix = multialgebra::Matrix::from_fn(ncomponents, basis.len(), |i, j| {
            basis[j].1.tensor().components()[i]
          });
          assert_eq!(
            matrix.rank(1e-9),
            basis.len(),
            "the trimmed basis is dependent at n={n} r={r} k={k}"
          );
        }
      }
    }
  }

  /// At $r = 1$ the trimmed basis *is* the Whitney basis: one element per
  /// $k$-subsimplex, and each equal to the Whitney form the existing
  /// first-order implementation builds, pointwise.
  ///
  /// Ties the general construction to independently tested code: a wrong sign
  /// or factorial in the transfer, the Koszul operator or the evaluation shows
  /// up here.
  #[test]
  fn the_first_order_trimmed_basis_is_the_whitney_basis() {
    use crate::interpolate::form::WhitneyLsf;

    for n in 1..=4 {
      for k in 0..=n {
        let basis = trimmed_basis(n, 1, k);
        let expected: Vec<_> = multiindex::combinations(n + 1, k + 1).collect();
        assert_eq!(
          basis.iter().map(|(simp, _)| *simp).collect::<Vec<_>>(),
          expected
        );

        for (dof_simp, form) in &basis {
          let reference = WhitneyLsf::unit(Dim::from(n), *dof_simp);
          for node in barycentric_probes(n) {
            assert_relative_eq!(
              form.at_bary(&node).components(),
              reference.at_bary(&node).components(),
              epsilon = 1e-12
            );
          }
        }
      }
    }
  }

  /// $dif$ maps the trimmed space into the *previous* full space,
  /// $dif P^-_r Lambda^k subset P_(r-1) Lambda^(k+1)$, which together with
  /// $P_(r-1) Lambda^k subset P^-_r Lambda^k$ is what closes the complex.
  #[test]
  fn the_trimmed_complex_closes() {
    for n in 1..=3 {
      for r in 1..=3 {
        for k in 0..n {
          for (_, form) in trimmed_basis(n, r, k) {
            let dif = form.dif();
            assert_eq!(dif.degree(), Degree::from(r) - 1);
            assert_eq!(dif.grade(), Degree::from(k) + 1);
            // And the complex is a complex.
            assert_relative_eq!(dif.dif().tensor().components().amax(), 0.0, epsilon = 1e-12);
          }
        }
      }
    }
  }

  /// Barycentric test points, none a vertex or the centroid, so a formula right
  /// only at the symmetric points cannot pass.
  fn barycentric_probes(cell_dim: usize) -> Vec<Bary> {
    (0..3)
      .map(|seed| {
        let weights: Vec<f64> = (0..=cell_dim)
          .map(|i| 1.0 + ((3 * i + 2 * seed) % 5) as f64)
          .collect();
        let total: f64 = weights.iter().sum();
        Bary::new(multialgebra::Vector::from_vec(
          weights.into_iter().map(|w| w / total).collect(),
        ))
      })
      .collect()
  }
}
