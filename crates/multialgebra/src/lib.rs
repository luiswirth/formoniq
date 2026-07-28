#![doc = include_str!("../README.md")]

extern crate nalgebra as na;

use multiindex::{Composition, binomial, combinations};

pub use multiindex::{Degree, Dim};

pub type Vector<T = f64> = na::DVector<T>;
pub type Matrix<T = f64> = na::DMatrix<T>;

/// The sign two indices of one factor pick up on swap: the single bit
/// separating $Lambda$ from $"Sym"$, since $Lambda(V) = "Sym"(V\[1\])$.
///
/// Everything else about the two constructions follows from it. Alternating
/// forbids repetition, so a basis element is a *subset* and the degree is
/// bounded by the dimension; symmetric permits it, so a basis element is a
/// *multiset* and the degree is unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity {
  Alternating,
  Symmetric,
}

/// One tensor factor: $Lambda^k$ or $"Sym"^k$, of whichever space it is
/// evaluated over.
///
/// The dimension of that space is deliberately *not* carried here: a factor is
/// the functor, not its value on a particular space. That is what lets one
/// factor describe both ends of a map between spaces of different dimensions,
/// as [`Factor::induced`] does on a rectangular one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Factor {
  pub parity: Parity,
  pub degree: Degree,
}

impl Factor {
  pub fn alternating(degree: impl Into<Degree>) -> Self {
    Self {
      parity: Parity::Alternating,
      degree: degree.into(),
    }
  }
  pub fn symmetric(degree: impl Into<Degree>) -> Self {
    Self {
      parity: Parity::Symmetric,
      degree: degree.into(),
    }
  }

  /// $dim Lambda^k (RR^n) = binom(n, k)$ and
  /// $dim "Sym"^k (RR^n) = binom(n + k - 1, k)$.
  ///
  /// Total at the trivial ends of both: a negative degree names the zero space
  /// either way, and only the alternating factor has a top degree beyond which
  /// it vanishes again. That asymmetry is the whole difference in the counting.
  pub fn multidim(&self, dim: impl Into<Dim>) -> usize {
    let dim = dim.into();
    match self.parity {
      Parity::Alternating => self
        .degree
        .index_in(dim)
        .map_or(0, |degree| binomial(dim.index(), degree)),
      Parity::Symmetric => {
        if self.degree.get() < 0 {
          0
        } else {
          Composition::count(dim.index(), self.degree.index())
        }
      }
    }
  }

  /// The functor applied to a linear map: $Lambda^k A$ or $"Sym"^k A$.
  ///
  /// One construction read through the parity. The alternating entries are the
  /// $k times k$ minors under $det$, the symmetric ones the same minors under
  /// $"per"$ -- the signed and unsigned sums over the same permutations, which
  /// is the defining relation one level up.
  ///
  /// Functoriality $F(A B) = F(A) F(B)$ holds for both: Cauchy-Binet on the
  /// alternating side, its permanental counterpart on the symmetric one.
  pub fn induced(&self, map: &Matrix) -> Matrix {
    match self.parity {
      Parity::Alternating => exterior_power(map, self.degree),
      Parity::Symmetric => symmetric_power(map, self.degree),
    }
  }
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

/// The symmetric power functor $"Sym"^k$ applied to a linear map, on
/// colex-ordered monomials.
///
/// The entry is $"per" A[alpha, beta] \/ alpha!$, the permanent mirroring the
/// determinant of [`exterior_power`]. That is the *formula*; it is not the
/// algorithm, and it must not become one -- the permanent is #P-hard, while
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
    // The image of the basis monomial, built factor by factor: start at the
    // constant 1 and multiply in one column of `map` per unit of degree.
    let mut poly = Vector::zeros(Factor::symmetric(0).multidim(map.nrows()));
    poly[0] = 1.0;
    for (col, &multiplicity) in source.parts().iter().enumerate() {
      for _ in 0..multiplicity {
        poly = multiply_by_linear(&poly, &map.column(col).into_owned(), map.nrows());
      }
    }
    power.column_mut(j).copy_from(&poly);
  }
  power
}

/// Multiply a homogeneous polynomial by the linear form whose coefficients are
/// `linear`, raising the degree by one.
///
/// The one place the symmetric side needs its own combinatorics: a monomial
/// times a variable is another monomial, so the product is a scatter over
/// [`Composition::rank`] with no sign, where the alternating side would carry
/// one and cancel repetitions.
fn multiply_by_linear(poly: &Vector, linear: &Vector, nparts: usize) -> Vector {
  let degree = degree_of(poly.len(), nparts);
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

/// The degree of a homogeneous polynomial in `nparts` variables from its
/// coefficient count. Inverse to [`Composition::count`] in the degree.
fn degree_of(ncoeffs: usize, nparts: usize) -> usize {
  (0..)
    .find(|&d| Composition::count(nparts, d) == ncoeffs)
    .expect("A coefficient count is a dimension of some degree.")
}

/// The map induced on $times.circle_i F_i$ by a linear map on the underlying
/// space: the Kronecker product of the per-factor induced maps.
///
/// This is the whole of functoriality on a tensor product, and it is uniform
/// over the factors -- the parity is consulted only inside
/// [`Factor::induced`], never here. An empty product of factors is the scalars,
/// on which every map induces the identity.
pub fn induced(factors: &[Factor], map: &Matrix) -> Matrix {
  factors
    .iter()
    .map(|factor| factor.induced(map))
    .reduce(|acc, factor| acc.kronecker(&factor))
    .unwrap_or_else(|| Matrix::identity(1, 1))
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

  /// $F(A B) = F(A) F(B)$ for a single factor of either parity: Cauchy-Binet
  /// on the alternating side and its permanental counterpart on the symmetric
  /// one.
  ///
  /// Swept over rectangular shapes, so the two dimensions of the map are not
  /// allowed to coincide and hide a transpose.
  #[test]
  fn each_factor_is_a_functor() {
    for degree in 0..=3 {
      for parity in [Parity::Alternating, Parity::Symmetric] {
        for &(p, q, r) in &[(2, 3, 2), (3, 2, 3), (4, 3, 2), (2, 2, 4)] {
          let factor = Factor {
            parity,
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
  /// $Lambda^k times.circle "Sym"^l$ with the parity consulted only per
  /// factor. Mixed parities and unequal degrees, so neither can stand in for
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
        let (a, b) = (probe(p, q, 3), probe(q, r, 4));
        let composed = induced(factors, &(&a * &b));
        let separate = induced(factors, &a) * induced(factors, &b);

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

  /// The factors sit in the index in the order given, the last running
  /// fastest: entry $(r_1 d_2 + r_2, c_1 e_2 + c_2)$ of the product is
  /// $F_1 (r_1, c_1) dot F_2 (r_2, c_2)$.
  ///
  /// Functoriality cannot see this. $(A times.circle B)(C times.circle D) =
  /// A C times.circle B D$ holds under any consistent ordering, so the law
  /// above passes just as well with the factors reversed, and the convention
  /// needs its own statement. It is load-bearing: it fixes which component of
  /// a tensor is which.
  #[test]
  fn the_last_factor_runs_fastest() {
    let (first, second) = (Factor::symmetric(2), Factor::alternating(1));
    let map = probe(3, 4, 6);
    let (a, b) = (first.induced(&map), second.induced(&map));
    let product = induced(&[first, second], &map);

    for r1 in 0..a.nrows() {
      for c1 in 0..a.ncols() {
        for r2 in 0..b.nrows() {
          for c2 in 0..b.ncols() {
            assert_relative_eq!(
              product[(r1 * b.nrows() + r2, c1 * b.ncols() + c2)],
              a[(r1, c1)] * b[(r2, c2)]
            );
          }
        }
      }
    }
  }

  /// At degree one every factor is the space itself, so the functor is the map
  /// back again, whatever the parity. The base case that pins the conventions:
  /// it fails on a transposed index or a wrong basis order.
  #[test]
  fn degree_one_induces_the_map_itself() {
    for parity in [Parity::Alternating, Parity::Symmetric] {
      let map = probe(3, 4, 5);
      let factor = Factor {
        parity,
        degree: Degree::ONE,
      };
      assert_relative_eq!(factor.induced(&map), map);
    }
  }
}
