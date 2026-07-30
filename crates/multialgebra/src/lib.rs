#![doc = include_str!("../README.md")]

extern crate nalgebra as na;

pub mod tensor;
pub mod variance;

use multiindex::{
  Composition, MonoIndex, MultiIndex, MultiIndices, Repetition, Word, binomial, combinations,
};

pub use multiindex::{Degree, Dim};
pub use tensor::Tensor;
pub use variance::Variance;

pub type Vector<T = f64> = na::DVector<T>;
pub type Matrix<T = f64> = na::DMatrix<T>;

/// The symmetry a slot imposes on its positions: none, or one of the two
/// quotients by a character of $S_k$.
///
/// Two axes, not one. Either the $S_k$ action is quotiented or it is not, and
/// if it is, a character says how. A character lands in an abelian group, so it
/// factors through the abelianization of $S_k$, which is $ZZ\/2$ for $k >= 2$:
/// the sign gives [`Self::Alternating`] and the trivial character
/// [`Self::Symmetric`], and those are the whole list. [`Self::Free`] is the
/// unquotiented tensor power sitting above them both.
///
/// The representation follows the mathematics exactly. An alternating index is
/// a subset and a symmetric one a multiset, and the shift makes both a single
/// bitset with an alphabet-independent rank. A free index is a word: no
/// symmetry to exploit, so nothing to compress and no way to rank it without
/// knowing the alphabet. The cost of a family is its information content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Symmetry {
  /// No quotient: $V^(times.circle k)$, of dimension $n^k$.
  Free,
  /// Quotient by the sign character: $Lambda^k$, of dimension $binom(n, k)$.
  #[default]
  Alternating,
  /// Quotient by the trivial character: $"Sym"^k$, of dimension
  /// $binom(n+k-1, k)$.
  Symmetric,
}

impl Symmetry {
  /// The combinatorial reading, for the quotients: whether a basis multi-index
  /// may repeat a symbol.
  ///
  /// `None` on [`Self::Free`], whose basis is a word rather than a monotone
  /// index, so repetition is not the axis that describes it.
  pub fn repetition(self) -> Option<Repetition> {
    match self {
      Symmetry::Free => None,
      Symmetry::Alternating => Some(Repetition::Forbidden),
      Symmetry::Symmetric => Some(Repetition::Allowed),
    }
  }

  /// Whether this symmetry quotients the action at all.
  pub fn is_free(self) -> bool {
    self == Symmetry::Free
  }
}

/// One tensor factor: $Lambda^k$ or $"Sym"^k$, of whichever space it is
/// evaluated over.
///
/// The dimension of that space is deliberately not carried here: a factor is
/// the functor, not its value on a particular space. That is what lets one
/// factor describe both ends of a map between spaces of different dimensions,
/// as [`Factor::induced`] does on a rectangular one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Factor {
  symmetry: Symmetry,
  degree: Degree,
}

impl Factor {
  /// The factor of the given symmetry and degree.
  ///
  /// The symmetry is never normalized away, not even where the degree makes the
  /// two functors agree. $Lambda^0 = "Sym"^0 = RR$ and $Lambda^1 = "Sym"^1 = V$
  /// as spaces, but a factor is a position in a shape rather than the space
  /// it currently denotes, and operations move degree through that position:
  /// [`Tensor::transfer`] into a degree-zero symmetric factor must give
  /// $"Sym"^1$, not $Lambda^1$, and merging two degree-one factors gives
  /// $"Sym"^2$ or $Lambda^2$ by the symmetry. Collapsing it would lose which
  /// family the factor can grow back into.
  pub fn new(symmetry: Symmetry, degree: impl Into<Degree>) -> Self {
    Self {
      symmetry,
      degree: degree.into(),
    }
  }

  pub fn alternating(degree: impl Into<Degree>) -> Self {
    Self::new(Symmetry::Alternating, degree)
  }
  pub fn symmetric(degree: impl Into<Degree>) -> Self {
    Self::new(Symmetry::Symmetric, degree)
  }

  pub fn symmetry(&self) -> Symmetry {
    self.symmetry
  }
  pub fn degree(&self) -> Degree {
    self.degree
  }

  pub fn is_alternating(&self) -> bool {
    self.symmetry == Symmetry::Alternating
  }
  pub fn is_symmetric(&self) -> bool {
    self.symmetry == Symmetry::Symmetric
  }

  /// The same functor at another degree.
  pub fn with_degree(&self, degree: impl Into<Degree>) -> Self {
    Self::new(self.symmetry, degree)
  }

  /// $dim Lambda^k (RR^n) = binom(n, k)$, $dim "Sym"^k (RR^n) = binom(n+k-1, k)$
  /// and $dim V^(times.circle k) = n^k$.
  ///
  /// The two quotients are one binomial over the shifted alphabet, differing
  /// only in how wide the shift makes it. The free power is the count no
  /// symmetry reduces.
  ///
  /// Total at both trivial ends with no case of its own. A negative degree
  /// names the zero space. An alternating degree past the top gives
  /// $binom(n, k) = 0$ because the binomial already vanishes there, and a
  /// symmetric or free factor over the zero space gives $0$ in positive degree.
  pub fn multidim(&self, dim: impl Into<Dim>) -> usize {
    let dim = dim.into();
    if self.degree.get() < 0 {
      return 0;
    }
    let degree = self.degree.index();
    match self.symmetry.repetition() {
      Some(repetition) => binomial(repetition.shifted_nsymbols(dim.index(), degree), degree),
      None => Word::count(dim.index(), degree),
    }
  }

  /// The basis of $F(RR^n)$ in the family's own order: colex on the quotients,
  /// radix on the free power.
  ///
  /// Empty exactly where [`Self::multidim`] is zero, so the trivial ends need
  /// no case of their own.
  pub fn basis(&self, dim: impl Into<Dim>) -> MultiIndices {
    let dim = dim.into().index();
    // A trivial space enumerates nothing, expressed as a word of length one
    // over the empty alphabet, which no family can supply, rather than as a
    // case of its own. The family stays the one this factor names.
    let (dim, degree) = match self.multidim(dim) > 0 {
      true => (dim, self.degree.index()),
      false => (0, 1),
    };
    match self.symmetry.repetition() {
      Some(repetition) => MultiIndices::Mono(MonoIndex::all(repetition, dim, degree)),
      None => MultiIndices::Word(Word::all(dim, degree)),
    }
  }

  /// The bilinear form induced on $F(V)$ by one on $V$: the $k times k$ minors
  /// of its matrix, under $det$ when alternating, $"per"$ when symmetric, and
  /// the plain diagonal product when free.
  ///
  /// Takes a bare matrix and knows nothing of non-degeneracy or signature: this
  /// crate is the metric-free half of the algebra, and an induced form needs
  /// only a form. Wrapping it where those properties are wanted is the job of
  /// whoever has them.
  ///
  /// The same minors [`Self::induced`] takes, of the metric rather than of a
  /// map. Both are $sum_sigma "sign"(sigma) product_i A_(i sigma(i))$ under the
  /// family's own [`Repetition::sign_of`], but that shared form is not the
  /// implementation: it is $k!$ either way, where $det$ is cubic. The
  /// separation is real, $"per"$ being #P-hard, so the two stay separate here.
  ///
  /// The normalization is not free: both families take the inner product
  /// $V^(times.circle k)$ induces divided by $k!$, on unnormalized products
  /// ($v_1 wedge dots.c wedge v_k = sum_sigma "sgn"(sigma) v_(sigma(1))
  /// times.circle dots.c$, and the same unsigned for $v_1 dots.c v_k$). Both
  /// quotients are $k!$, leaving $det$ against $"per"$ with no further factor.
  ///
  /// So under a Euclidean metric the alternating basis is orthonormal and the
  /// symmetric one merely orthogonal, $norm(x^alpha)^2 = alpha!$, the
  /// multiplicity a repeated slot carries.
  pub fn induced_form(&self, single: &Matrix) -> Matrix {
    assert_eq!(single.nrows(), single.ncols(), "a bilinear form is square");
    let dim = single.nrows();
    let basis: Vec<MultiIndex> = self.basis(dim).collect();
    let entry = |row: &MultiIndex, col: &MultiIndex| {
      let minor = Matrix::from_fn(self.degree.index(), self.degree.index(), |i, j| {
        single[(row.symbol(i), col.symbol(j))]
      });
      match self.symmetry {
        Symmetry::Alternating => minor.determinant(),
        Symmetry::Symmetric => permanent(&minor),
        // The free power induces the tensor power of the inner product, so an
        // entry is the plain product down the diagonal: no sum over
        // permutations, because there is no permutation to sum over.
        Symmetry::Free => (0..self.degree.index()).map(|i| minor[(i, i)]).product(),
      }
    };
    Matrix::from_fn(basis.len(), basis.len(), |i, j| entry(&basis[i], &basis[j]))
  }

  /// The functor applied to a linear map: $Lambda^k A$ or $"Sym"^k A$.
  ///
  /// One construction read through the symmetry. The alternating entries are the
  /// $k times k$ minors under $det$, the symmetric ones the same minors under
  /// $"per"$, the signed and unsigned sums over the same permutations, which
  /// is the defining relation one level up.
  ///
  /// Functoriality $F(A B) = F(A) F(B)$ holds for both: Cauchy-Binet on the
  /// alternating side, its permanental counterpart on the symmetric one.
  pub fn induced(&self, map: &Matrix) -> Matrix {
    match self.symmetry {
      Symmetry::Alternating => exterior_power(map, self.degree),
      Symmetry::Symmetric => symmetric_power(map, self.degree),
      Symmetry::Free => tensor_power(map, self.degree),
    }
  }
}

/// One slot of a [`Tensor`]: a [`Factor`], the space it is evaluated over, and
/// the side it is built from.
///
/// The unit an operation names, and the unit an axis of a dense array would be:
/// a slot has an extent ([`Self::multidim`]) exactly as an axis has a length,
/// and additionally a symmetry and a variance the array has no room for.
///
/// A slot is simultaneously a value it holds and an argument it eats (of the
/// dual), those being the same thing under $V tilde.equals V^(**)$, so "value"
/// and "argument" are readings rather than structure. What is structure is
/// the [`Variance`], and the pattern of it across the slots.
///
/// The dimension lives here rather than on the tensor so that slots may be over
/// different spaces, which is what a rectangular map or a genuinely mixed
/// tensor needs. It does not live on [`Factor`], which is the functor and
/// must stay dimension-free for [`Factor::induced`] to describe both ends of a
/// map from one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Slot {
  pub factor: Factor,
  pub variance: Variance,
  pub dim: Dim,
}

impl Slot {
  pub fn new(factor: Factor, variance: Variance, dim: impl Into<Dim>) -> Self {
    Self {
      factor,
      variance,
      dim: dim.into(),
    }
  }
  /// A covariant slot: a factor of $V^*$.
  pub fn covariant(factor: Factor, dim: impl Into<Dim>) -> Self {
    Self::new(factor, Variance::Covariant, dim)
  }
  /// A contravariant slot: a factor of $V$.
  pub fn contravariant(factor: Factor, dim: impl Into<Dim>) -> Self {
    Self::new(factor, Variance::Contravariant, dim)
  }

  pub fn symmetry(&self) -> Symmetry {
    self.factor.symmetry()
  }
  pub fn degree(&self) -> Degree {
    self.factor.degree()
  }
  pub fn is_alternating(&self) -> bool {
    self.factor.is_alternating()
  }
  pub fn is_symmetric(&self) -> bool {
    self.factor.is_symmetric()
  }
  /// The extent of this slot: the dimension of $F(RR^n)$.
  pub fn multidim(&self) -> usize {
    self.factor.multidim(self.dim)
  }
  pub fn basis(&self) -> MultiIndices {
    self.factor.basis(self.dim)
  }
  /// The same slot with its variance flipped: what a musical isomorphism does.
  pub fn dual(&self) -> Self {
    Self::new(self.factor, self.variance.dual(), self.dim)
  }
  /// The same slot at another degree.
  pub fn with_degree(&self, degree: impl Into<Degree>) -> Self {
    Self::new(self.factor.with_degree(degree), self.variance, self.dim)
  }
  /// The same slot over another space.
  pub fn with_dim(&self, dim: impl Into<Dim>) -> Self {
    Self::new(self.factor, self.variance, dim)
  }
}

/// The grade of an exterior form: the [`Degree`] under its exterior-algebra
/// name.
pub type ExteriorGrade = Degree;

/// A basis blade $e_(i_1) wedge dots.c wedge e_(i_k)$: a strictly increasing
/// multi-index.
pub type Blade = multiindex::Combination;

/// $dim Lambda^k (RR^n) = binom(n, k)$, and $0$ off $[0, n]$ where the space is
/// trivial.
pub fn exterior_dim(dim: impl Into<Dim>, grade: impl Into<ExteriorGrade>) -> usize {
  Factor::alternating(grade).multidim(dim)
}

/// The basis blades of $Lambda^k (RR^n)$ in colexicographic order: the order of
/// the components of a single alternating slot.
pub fn exterior_bases(
  dim: impl Into<Dim>,
  grade: impl Into<ExteriorGrade>,
) -> impl Iterator<Item = Blade> {
  Factor::alternating(grade).basis(dim).map(|index| {
    index
      .as_mono()
      .expect("an alternating basis is monotone")
      .to_combination()
  })
}

/// The permanent: the determinant with every sign made positive,
/// $"per" A = sum_sigma product_i A_(i sigma(i))$.
///
/// Ryser's inclusion-exclusion, $2^n n$ rather than the $n!$ of the definition.
/// Still exponential, the permanent being #P-hard, so it is confined to the
/// $k times k$ minors of a Gramian and is not how [`symmetric_power`]
/// computes.
pub fn permanent(matrix: &Matrix) -> f64 {
  let n = matrix.nrows();
  assert_eq!(n, matrix.ncols(), "the permanent is of a square matrix");
  if n == 0 {
    return 1.0;
  }
  let mut total = 0.0;
  for subset in 1..(1u64 << n) {
    let selected = subset.count_ones() as usize;
    let product: f64 = (0..n)
      .map(|row| {
        (0..n)
          .filter(|col| subset & (1 << col) != 0)
          .map(|col| matrix[(row, col)])
          .sum::<f64>()
      })
      .product();
    total += if (n - selected).is_multiple_of(2) {
      product
    } else {
      -product
    };
  }
  total
}

/// The exterior power functor $Lambda^k$ applied to a linear map: the $k$-th
/// compound matrix, $(Lambda^k A)_(I J) = det A[I, J]$, on colex-ordered
/// subsets.
pub fn exterior_power(map: &Matrix, degree: impl Into<Degree>) -> Matrix {
  let degree = degree.into();
  let factor = Factor::alternating(degree);
  let (nrows, ncols) = (factor.multidim(map.nrows()), factor.multidim(map.ncols()));
  if degree.get() < 0 {
    return Matrix::zeros(nrows, ncols);
  }

  let mut power = Matrix::zeros(nrows, ncols);
  let mut minor = Matrix::zeros(degree.index(), degree.index());
  for (i, rows) in combinations(map.nrows(), degree.index()).enumerate() {
    for (j, cols) in combinations(map.ncols(), degree.index()).enumerate() {
      for (ii, row) in rows.iter().enumerate() {
        for (jj, col) in cols.iter().enumerate() {
          minor[(ii, jj)] = map[(row, col)];
        }
      }
      power[(i, j)] = minor.determinant();
    }
  }
  power
}

/// The tensor power functor $V^(times.circle k)$ applied to a linear map: the
/// $k$-fold Kronecker power $A^(times.circle k)$, on radix-ordered words.
///
/// The simplest of the three, and the one the other two are quotients of:
/// no minors, no signs, just the product down each word.
pub fn tensor_power(map: &Matrix, degree: impl Into<Degree>) -> Matrix {
  let degree = degree.into();
  if degree.get() < 0 {
    let factor = Factor::new(Symmetry::Free, degree);
    return Matrix::zeros(factor.multidim(map.nrows()), factor.multidim(map.ncols()));
  }
  (0..degree.index()).fold(Matrix::identity(1, 1), |power, _| power.kronecker(map))
}

/// The symmetric power functor $"Sym"^k$ applied to a linear map, on
/// colex-ordered monomials.
///
/// The entry is $"per" A[alpha, beta] \/ alpha!$, the permanent mirroring the
/// determinant of [`exterior_power`]. That is the formula. It is not the
/// algorithm, and it must not become one: the permanent is #P-hard, while
/// expanding the image of a basis monomial,
/// $product_j (sum_i A_(i j) f_i)^(beta_j)$, is polynomial and is what runs
/// here.
pub fn symmetric_power(map: &Matrix, degree: impl Into<Degree>) -> Matrix {
  let degree = degree.into();
  let factor = Factor::symmetric(degree);
  let (nrows, ncols) = (factor.multidim(map.nrows()), factor.multidim(map.ncols()));
  if degree.get() < 0 {
    return Matrix::zeros(nrows, ncols);
  }
  let degree = degree.index();

  let mut power = Matrix::zeros(nrows, ncols);
  for (j, source) in Composition::all(map.ncols(), degree).enumerate() {
    // The image of the basis monomial, built variable by variable: start at the
    // constant 1 and multiply in one column of `map` per unit of degree, so the
    // polynomial in hand has the degree of the units spent so far.
    let variables = (source.parts().iter().enumerate())
      .flat_map(|(col, &multiplicity)| std::iter::repeat_n(col, multiplicity));
    let poly = variables
      .enumerate()
      .fold(Vector::from_element(1, 1.0), |poly, (spent, col)| {
        multiply_by_linear(&poly, spent, &map.column(col).into_owned())
      });
    power.column_mut(j).copy_from(&poly);
  }
  power
}

/// Multiply a homogeneous polynomial of the given degree by the linear form
/// whose coefficients are `linear`, raising the degree by one.
///
/// The one place the symmetric side needs its own combinatorics: a monomial
/// times a variable is another monomial, so the product is a scatter over
/// [`Composition::rank`] with no sign, where the alternating side would carry
/// one and cancel repetitions.
fn multiply_by_linear(poly: &Vector, degree: usize, linear: &Vector) -> Vector {
  let nparts = linear.len();
  let mut product = Vector::zeros(Composition::count(nparts, degree + 1));
  for (coeff, monomial) in poly.iter().zip(Composition::all(nparts, degree)) {
    if *coeff == 0.0 {
      continue;
    }
    for (part, weight) in linear.iter().enumerate() {
      let mut parts = monomial.parts().to_vec();
      parts[part] += 1;
      product[Composition::new(parts).rank()] += coeff * weight;
    }
  }
  product
}

/// The multiplicity $alpha!$ of each basis element of a shape, in component
/// order: $alpha!$ on a symmetric slot and $1$ on an alternating or free one.
///
/// It is $norm(e_alpha)^2$ under the Euclidean form, so it is the diagonal of
/// [`Factor::induced_form`] of the identity, which is the law that ties the two
/// together. It is not computed that way: a permanent per entry of a square
/// matrix is exponential where the multiplicities are a product of factorials.
///
/// A quotient's basis element is an unnormalized (anti)symmetrization divided
/// by $k!$, in which each distinct word occurs $alpha!$ times, so its square
/// norm is $alpha!$. A free slot symmetrizes nothing, its basis element being a
/// single tensor, so it carries no multiplicity at all and the stabilizer of
/// its word does not enter.
///
/// Deliberately not public. It is the change of basis between the stored
/// multiplicative basis and its reciprocal, and the only ways to spend it are
/// [`Tensor::reciprocal`] and [`Tensor::from_reciprocal`], which say which
/// basis they mean.
pub(crate) fn basis_multiplicity(slots: &[Slot]) -> Vector {
  let per_slot: Vec<Matrix> = slots
    .iter()
    .map(|slot| {
      let weights = slot
        .basis()
        .map(|index| match slot.symmetry().repetition() {
          Some(_) => index.stabilizer() as f64,
          None => 1.0,
        });
      Matrix::from_iterator(slot.multidim(), 1, weights)
    })
    .collect();
  tensor::factorwise_kronecker(&per_slot).column(0).into()
}

#[cfg(test)]
mod test {
  use super::*;
  use approx::assert_relative_eq;

  /// A deterministic matrix with no symmetry, so a transposed index or a
  /// dropped factor cannot pass unnoticed.
  fn probe(nrows: usize, ncols: usize, seed: usize) -> Matrix {
    Matrix::from_fn(nrows, ncols, |i, j| {
      ((7 * i + 3 * j + 5 * seed + 1) % 11) as f64 - 5.0
    })
  }

  /// $dim Lambda^k (RR^n) = binom(n,k)$ and $dim "Sym"^k (RR^n) =
  /// binom(n+k-1,k)$, and the two agree exactly at degree $0$ and $1$, where
  /// $Lambda^1 = "Sym"^1 = V$ and there is no symmetry to impose.
  #[test]
  fn factor_dimensions_and_their_degenerate_agreement() {
    for n in 0..=4 {
      for k in 0..=4 {
        assert_eq!(Factor::alternating(k).multidim(n), binomial(n, k));
        assert_eq!(Factor::symmetric(k).multidim(n), Composition::count(n, k));
        if k <= 1 {
          assert_eq!(
            Factor::alternating(k).multidim(n),
            Factor::symmetric(k).multidim(n)
          );
        }
      }
      // Only the alternating side has a top degree.
      assert_eq!(Factor::alternating(n + 1).multidim(n), 0);
      assert!(Factor::symmetric(n + 1).multidim(n) > 0 || n == 0);
      // Both are trivial below zero.
      assert_eq!(Factor::alternating(-1).multidim(n), 0);
      assert_eq!(Factor::symmetric(-1).multidim(n), 0);
    }
  }

  /// $F(A B) = F(A) F(B)$ for a single factor of either symmetry: Cauchy-Binet
  /// on the alternating side and its permanental counterpart on the symmetric
  /// one.
  ///
  /// Swept over rectangular shapes, so the two dimensions of the map are not
  /// allowed to coincide and hide a transpose.
  #[test]
  fn each_factor_is_a_functor() {
    for degree in 0..=3 {
      for symmetry in [Symmetry::Alternating, Symmetry::Symmetric] {
        for &(p, q, r) in &[(2, 3, 2), (3, 2, 3), (4, 3, 2), (2, 2, 4)] {
          let factor = Factor {
            symmetry,
            degree: degree.into(),
          };
          let (a, b) = (probe(p, q, 1), probe(q, r, 2));
          assert_relative_eq!(
            factor.induced(&(&a * &b)),
            factor.induced(&a) * factor.induced(&b),
            epsilon = 1e-9
          );
        }
      }
    }
  }

  /// The induced map on a tensor product is the Kronecker product of the
  /// per-factor ones, and is itself a functor.
  ///
  /// This is the law the crate exists for: one composition rule covering
  /// $Lambda^k times.circle "Sym"^l$ with the symmetry consulted only per
  /// factor. Mixed symmetries and unequal degrees, so neither can stand in for
  /// the other.
  #[test]
  fn the_tensor_product_of_factors_is_a_functor() {
    let factor_lists: [Vec<Factor>; 5] = [
      vec![],
      vec![Factor::alternating(2)],
      vec![Factor::symmetric(1), Factor::alternating(2)],
      vec![Factor::symmetric(2), Factor::symmetric(1)],
      vec![
        Factor::symmetric(2),
        Factor::alternating(1),
        Factor::symmetric(1),
      ],
    ];
    for factors in &factor_lists {
      for &(p, q, r) in &[(3, 3, 3), (4, 3, 3), (3, 4, 2)] {
        // The slots name the domain; a transport reads both ends off the map.
        let slots = tensor::covariant_slots(factors.iter().copied(), q);
        let functor = |map: &Matrix| tensor::Transport::new(&slots, map).to_matrix();
        let (a, b) = (probe(p, q, 3), probe(q, r, 4));
        let composed = functor(&(&a * &b));
        let separate = functor(&a) * functor(&b);

        let expected_rows: usize = factors.iter().map(|f| f.multidim(p)).product();
        let expected_cols: usize = factors.iter().map(|f| f.multidim(r)).product();
        assert_eq!(
          (composed.nrows(), composed.ncols()),
          (expected_rows, expected_cols)
        );
        assert_relative_eq!(composed, separate, epsilon = 1e-6);
      }
    }
  }

  /// At degree one every factor is the space itself, so the functor is the map
  /// back again, whatever the symmetry. The base case that pins the conventions:
  /// it fails on a transposed index or a wrong basis order.
  #[test]
  fn degree_one_induces_the_map_itself() {
    for symmetry in [Symmetry::Alternating, Symmetry::Symmetric] {
      let map = probe(3, 4, 5);
      let factor = Factor {
        symmetry,
        degree: Degree::ONE,
      };
      assert_relative_eq!(factor.induced(&map), map);
    }
  }
}
