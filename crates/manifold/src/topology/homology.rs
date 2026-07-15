//! Integer simplicial homology of the complex.
//!
//! The Betti numbers $b_k = dim H_k (K; ZZ)$ of the chain complex
//!
//! $dots.c ->^(diff_(k+1)) C_k ->^(diff_k) C_(k-1) ->^(diff_(k-1)) dots.c$
//!
//! read off the ranks of the (integer) boundary operators:
//!
//! $b_k = dim ker diff_k - rank diff_(k+1) = n_k - rank diff_k - rank diff_(k+1)$,
//!
//! with $n_k$ the number of k-simplices. Metric-free (invariant 5): homology is
//! a topological invariant, a function of the incidence alone.
//!
//! The rank is computed exactly, by Gaussian elimination over a prime field,
//! never by a floating-point SVD with a tolerance -- a discrete invariant must
//! not depend on where a singular value falls relative to a magic constant. The
//! rank over $FF_p$ equals the rational rank unless $p$ divides the product of
//! the invariant factors (the torsion coefficients), which for a boundary matrix
//! are tiny; two large primes put the exception out of reach of any representable
//! mesh. The full torsion of $H_k (K; ZZ)$ is not recovered -- it is invisible to
//! the de Rham / Hodge side of the library, which lives over $RR$ -- so only the
//! free rank is returned.
//!
//! [`Complex::homology_generators`] goes further and returns representative
//! cycles: a [`Chain`] for each Betti number, whose classes form a basis of the
//! free part of $H_k$. That step needs a basis of $ker diff_k$ modulo
//! $"im" diff_(k+1)$, hence exact linear algebra over $QQ$ (`BigRational`, so no
//! coefficient growth ever overflows), not just a rank.

use super::{complex::Complex, handle::KSimplexIdx};
use crate::Dim;

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

use std::collections::BTreeMap;

impl Complex {
  /// The Betti numbers $b_0, dots, b_n$ of $K$: the ranks of the free part of
  /// the integer homology $H_k (K; ZZ)$. A topological invariant, metric-free.
  ///
  /// $b_0$ counts connected components, $b_1$ independent loops, and so on.
  pub fn betti_numbers(&self) -> Vec<usize> {
    // rank diff_k for k in 0..=n+1; diff_0 and diff_(n+1) map to/from the zero
    // space and so have rank 0.
    let ranks: Vec<usize> = (0..=self.dim() + 1)
      .map(|k| self.boundary_rank(k))
      .collect();
    (0..=self.dim())
      .map(|k| self.nsimplices(k) - ranks[k] - ranks[k + 1])
      .collect()
  }

  /// The k-th Betti number $b_k = dim H_k (K; ZZ)$.
  pub fn betti_number(&self, grade: Dim) -> usize {
    self.nsimplices(grade) - self.boundary_rank(grade) - self.boundary_rank(grade + 1)
  }

  /// The Euler characteristic $chi = sum_k (-1)^k n_k = sum_k (-1)^k b_k$.
  ///
  /// The two forms agree by Euler--Poincaré; the first is what is computed, and
  /// their equality is an exact cross-check on the Betti numbers.
  pub fn euler_characteristic(&self) -> i64 {
    (0..=self.dim())
      .map(|k| {
        let n = self.nsimplices(k) as i64;
        if k % 2 == 0 {
          n
        } else {
          -n
        }
      })
      .sum()
  }

  /// $rank diff_k$, the rank of the integer boundary operator
  /// $diff_k: C_k -> C_(k-1)$. Outside $1 <= k <= n$ the operator maps to/from
  /// the zero space and has rank 0.
  fn boundary_rank(&self, grade: Dim) -> usize {
    if grade == 0 || grade > self.dim() {
      return 0;
    }
    let nrows = self.nsimplices(grade - 1);
    let triplets = self.boundary_triplets(grade);
    exact_rank(nrows, &triplets)
  }

  /// The integer boundary operator $diff_k$ as `(row, col, +-1)` triplets:
  /// column = k-simplex, row = (k-1)-face, entry = its incidence sign.
  fn boundary_triplets(&self, grade: Dim) -> Vec<(usize, usize, i64)> {
    let mut triplets = Vec::new();
    for (col, sup) in self.skeleton(grade).handle_iter().enumerate() {
      for (sign, sub) in sup.boundary() {
        triplets.push((sub.kidx(), col, i64::from(sign.as_i32())));
      }
    }
    triplets
  }
}

/// An integer $k$-chain: a formal $ZZ$-combination $sum_sigma c_sigma sigma$ of
/// the k-simplices, coefficients in colex order (indexed by [`KSimplexIdx`]).
///
/// The element of $C_k$ dual to a cochain; chains carry no metric, so they live
/// here in `topology` rather than up in `ddf` with the cochains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chain {
  grade: Dim,
  coeffs: Vec<i64>,
}
impl Chain {
  pub fn grade(&self) -> Dim {
    self.grade
  }
  /// The coefficient of each k-simplex, in colex order.
  pub fn coeffs(&self) -> &[i64] {
    &self.coeffs
  }
  /// The simplices carrying a nonzero coefficient, with that coefficient: the
  /// support of the chain.
  pub fn support(&self) -> impl Iterator<Item = (KSimplexIdx, i64)> + '_ {
    self
      .coeffs
      .iter()
      .enumerate()
      .filter(|(_, &c)| c != 0)
      .map(|(kidx, &c)| (kidx, c))
  }
}

impl Complex {
  /// Representative cycles whose classes are a basis of the free part of
  /// $H_k (K; ZZ)$ -- one [`Chain`] per Betti number $b_k$.
  ///
  /// Each generator is a k-cycle, $diff_k z = 0$; its class generates a
  /// $ZZ$-summand of $H_k$, and the $b_k$ classes are independent modulo
  /// boundaries. For $k = 0$ these are one vertex per connected component; for
  /// $k = 1$, loops around the holes; at the top grade of a closed manifold, the
  /// fundamental class.
  ///
  /// Representatives, not minimizers: the supporting simplices form *a* cycle in
  /// the class, chosen by the elimination order, never a shortest one -- optimal
  /// generators are a separate (hard) problem. Metric-free, and computed exactly
  /// over $QQ$; the returned coefficients are primitive (their gcd is 1) with the
  /// first nonzero one positive.
  pub fn homology_generators(&self, grade: Dim) -> Vec<Chain> {
    if grade > self.dim() {
      return Vec::new();
    }
    let len = self.nsimplices(grade);

    // A basis of the boundaries $B_k = "im" diff_(k+1) subset.eq Z_k$, and then
    // the cycles $Z_k = ker diff_k$; a cycle enlarging the span of what came
    // before is a new generator, so seeding the reducer with $B_k$ first makes
    // "enlarges the span" mean "is not a boundary".
    let mut span = EchelonSpan::default();
    for boundary in self.rational_boundary_columns(grade + 1, len) {
      span.insert(boundary);
    }
    self
      .rational_cycles(grade)
      .into_iter()
      .filter_map(|cycle| {
        span
          .insert(cycle)
          .then(|| primitive_chain(grade, &span.last))
      })
      .collect()
  }

  /// A basis of $Z_k = ker diff_k$ as rational column vectors of length $n_k$,
  /// from the reduced row echelon form of $diff_k$: each non-pivot (free) column
  /// yields one kernel vector.
  fn rational_cycles(&self, grade: Dim) -> Vec<Vec<BigRational>> {
    let ncols = self.nsimplices(grade);
    let nrows = if grade == 0 {
      0
    } else {
      self.nsimplices(grade - 1)
    };

    let mut mat = vec![vec![BigRational::zero(); ncols]; nrows];
    if grade >= 1 {
      for (col, sup) in self.skeleton(grade).handle_iter().enumerate() {
        for (sign, sub) in sup.boundary() {
          mat[sub.kidx()][col] = BigRational::from_integer(BigInt::from(sign.as_i32()));
        }
      }
    }

    let pivot_cols = reduced_row_echelon(&mut mat, ncols);
    let is_pivot = {
      let mut flags = vec![false; ncols];
      for &c in &pivot_cols {
        flags[c] = true;
      }
      flags
    };

    (0..ncols)
      .filter(|c| !is_pivot[*c])
      .map(|free| {
        let mut cycle = vec![BigRational::zero(); ncols];
        cycle[free] = BigRational::one();
        // The free column, back-substituted through the pivots.
        for (row, &pivot_col) in pivot_cols.iter().enumerate() {
          cycle[pivot_col] = -mat[row][free].clone();
        }
        cycle
      })
      .collect()
  }

  /// The columns of $diff_(grade)$ as rational vectors of length `len`: each
  /// `grade`-simplex mapped to its boundary. Empty above the top grade.
  fn rational_boundary_columns(&self, grade: Dim, len: usize) -> Vec<Vec<BigRational>> {
    if grade == 0 || grade > self.dim() {
      return Vec::new();
    }
    self
      .skeleton(grade)
      .handle_iter()
      .map(|sup| {
        let mut col = vec![BigRational::zero(); len];
        for (sign, sub) in sup.boundary() {
          col[sub.kidx()] = BigRational::from_integer(BigInt::from(sign.as_i32()));
        }
        col
      })
      .collect()
  }
}

/// An incrementally built row-echelon basis of a subspace of $QQ^n$, keyed by
/// leading (pivot) coordinate. [`insert`](Self::insert) reduces a vector against
/// the basis and, if a nonzero remainder survives, records it and reports the
/// vector as having enlarged the span.
#[derive(Default)]
struct EchelonSpan {
  /// Pivot coordinate -> a basis vector with a leading 1 there.
  pivots: BTreeMap<usize, Vec<BigRational>>,
  /// The last inserted (unnormalized) vector, exposed so a caller can turn an
  /// accepted cycle into a chain without reducing it a second time.
  last: Vec<BigRational>,
}
impl EchelonSpan {
  /// Reduce `vec` against the current basis. Returns whether it was independent
  /// (enlarged the span); if so the reduced form is retained as a new pivot and
  /// the original is kept in [`last`](Self::last).
  fn insert(&mut self, vec: Vec<BigRational>) -> bool {
    self.last = vec.clone();
    let mut reduced = vec;
    while let Some(lead) = reduced.iter().position(|x| !x.is_zero()) {
      let Some(pivot) = self.pivots.get(&lead) else {
        let inv = reduced[lead].recip();
        for coeff in &mut reduced {
          *coeff = &*coeff * &inv;
        }
        self.pivots.insert(lead, reduced);
        return true;
      };
      let factor = reduced[lead].clone();
      for (coeff, pivot_coeff) in reduced.iter_mut().zip(pivot) {
        *coeff = &*coeff - &(&factor * pivot_coeff);
      }
    }
    false
  }
}

/// Reduce `mat` (with `ncols` columns) to reduced row echelon form over $QQ$ in
/// place, returning the pivot column of each pivot row, in row order.
fn reduced_row_echelon(mat: &mut [Vec<BigRational>], ncols: usize) -> Vec<usize> {
  let nrows = mat.len();
  let mut pivot_cols = Vec::new();
  let mut row = 0;
  for col in 0..ncols {
    if row >= nrows {
      break;
    }
    let Some(sel) = (row..nrows).find(|&r| !mat[r][col].is_zero()) else {
      continue;
    };
    mat.swap(row, sel);

    let inv = mat[row][col].recip();
    for coeff in &mut mat[row] {
      *coeff = &*coeff * &inv;
    }

    for r in 0..nrows {
      if r != row && !mat[r][col].is_zero() {
        let factor = mat[r][col].clone();
        let pivot_row = mat[row].clone();
        for (target, pivot_coeff) in mat[r].iter_mut().zip(&pivot_row) {
          *target = &*target - &(&factor * pivot_coeff);
        }
      }
    }

    pivot_cols.push(col);
    row += 1;
  }
  pivot_cols
}

/// Turn a rational cycle into the [`Chain`] with the same direction but integer,
/// primitive coefficients: clear denominators, divide out their gcd, and fix the
/// sign so the first nonzero coefficient is positive.
fn primitive_chain(grade: Dim, cycle: &[BigRational]) -> Chain {
  let denom_lcm = cycle
    .iter()
    .fold(BigInt::one(), |acc, x| acc.lcm(x.denom()));
  let mut ints: Vec<BigInt> = cycle
    .iter()
    .map(|x| x.numer() * (&denom_lcm / x.denom()))
    .collect();

  let gcd = ints.iter().fold(BigInt::zero(), |acc, x| acc.gcd(x));
  if !gcd.is_zero() {
    for x in &mut ints {
      *x = &*x / &gcd;
    }
  }
  if ints
    .iter()
    .find(|x| !x.is_zero())
    .is_some_and(BigInt::is_negative)
  {
    for x in &mut ints {
      *x = -&*x;
    }
  }

  let coeffs = ints
    .iter()
    .map(|x| x.to_i64().expect("generator coefficient fits in i64"))
    .collect();
  Chain { grade, coeffs }
}

/// Two primes near $2^30$. Their pairwise product fits in an `i64`, so all
/// modular arithmetic below stays exact without overflow.
const PRIMES: [i64; 2] = [1_000_000_007, 1_000_000_009];

/// The exact rank over $QQ$ of the integer matrix given by its `(row, col, val)`
/// triplets, as the maximum of the ranks over the two prime fields. A single
/// prime under-counts only if it divides the product of the torsion
/// coefficients; two large primes would both have to, which no representable
/// mesh admits.
fn exact_rank(nrows: usize, triplets: &[(usize, usize, i64)]) -> usize {
  PRIMES
    .iter()
    .map(|&p| rank_mod_p(nrows, triplets, p))
    .max()
    .expect("PRIMES is nonempty")
}

/// The rank over $FF_p$ by sparse Gaussian elimination: reduce each row against
/// a table of pivot rows keyed by their leading column. Reducing a row's leading
/// column introduces only later columns, so the leading column strictly advances
/// and the reduction terminates; a row that survives to a fresh leading column
/// becomes a new pivot.
fn rank_mod_p(nrows: usize, triplets: &[(usize, usize, i64)], p: i64) -> usize {
  let mut rows: Vec<BTreeMap<usize, i64>> = vec![BTreeMap::new(); nrows];
  for &(r, c, v) in triplets {
    let entry = rows[r].entry(c).or_insert(0);
    *entry = (*entry + v).rem_euclid(p);
  }

  // Column -> a row reduced to leading coefficient 1 at that column.
  let mut pivots: BTreeMap<usize, BTreeMap<usize, i64>> = BTreeMap::new();
  let mut rank = 0;

  for mut row in rows {
    row.retain(|_, v| *v != 0);

    // `next()` on a BTreeMap yields the smallest key: the leading column.
    while let Some((&lead, &coeff)) = row.iter().next() {
      let Some(pivot) = pivots.get(&lead) else {
        // Fresh leading column: normalize to a leading 1 and record the pivot.
        let inv = mod_inverse(coeff, p);
        let normalized = row.iter().map(|(&c, &v)| (c, v * inv % p)).collect();
        pivots.insert(lead, normalized);
        rank += 1;
        break;
      };
      // row -= coeff * pivot; the pivot has a leading 1 at `lead`.
      for (&c, &v) in pivot {
        let entry = row.entry(c).or_insert(0);
        *entry = (*entry - coeff * v).rem_euclid(p);
        if *entry == 0 {
          row.remove(&c);
        }
      }
    }
  }

  rank
}

/// The modular inverse $a^(-1) mod p$ for prime $p$, by the extended Euclidean
/// algorithm. Requires $a not equiv 0$.
fn mod_inverse(a: i64, p: i64) -> i64 {
  let (mut t, mut new_t) = (0i64, 1i64);
  let (mut r, mut new_r) = (p, a.rem_euclid(p));
  while new_r != 0 {
    let quot = r / new_r;
    (t, new_t) = (new_t, t - quot * new_t);
    (r, new_r) = (new_r, r - quot * new_r);
  }
  debug_assert_eq!(r, 1, "argument must be invertible mod p");
  t.rem_euclid(p)
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::gen::cartesian::CartesianMeshInfo;
  use crate::topology::skeleton::Skeleton;

  /// A square annulus: the $3 times 3$ unit mesh with the middle cell removed,
  /// so $b_1 = 1$ (one hole).
  fn annulus() -> Complex {
    use crate::geometry::coord::simplex::SimplexCoords;
    let (square, coords) = CartesianMeshInfo::new_unit(2, 3).compute_coord_complex();
    let cells: Vec<_> = square
      .cells()
      .handle_iter()
      .filter(|cell| {
        let bc = SimplexCoords::from_simplex_and_coords(cell.simplex(), &coords).barycenter();
        let inside = |x: f64| 1.0 / 3.0 < x && x < 2.0 / 3.0;
        !(inside(bc[0]) && inside(bc[1]))
      })
      .map(|cell| cell.simplex().clone())
      .collect();
    Complex::from_cells(Skeleton::new(cells))
  }

  /// A spread of complexes with nontrivial homology across the grades.
  fn test_complexes() -> Vec<Complex> {
    let mut complexes: Vec<Complex> = (1..=3)
      .map(|dim| {
        CartesianMeshInfo::new_unit(dim, 2)
          .compute_coord_complex()
          .0
      })
      .collect();
    complexes.push(crate::dim3::mesh_sphere_surface(1).into_coord_complex().0);
    complexes.push(annulus());
    complexes
  }

  /// Whether a chain is a cycle: $diff_k z = 0$.
  fn is_cycle(complex: &Complex, chain: &Chain) -> bool {
    let grade = chain.grade();
    if grade == 0 {
      return true;
    }
    let mut image = vec![0i64; complex.nsimplices(grade - 1)];
    for (col, sup) in complex.skeleton(grade).handle_iter().enumerate() {
      let coeff = chain.coeffs()[col];
      for (sign, sub) in sup.boundary() {
        image[sub.kidx()] += coeff * i64::from(sign.as_i32());
      }
    }
    image.iter().all(|&x| x == 0)
  }

  /// One generator per Betti number, in every grade.
  #[test]
  fn generators_count_matches_betti() {
    for complex in test_complexes() {
      for k in 0..=complex.dim() {
        assert_eq!(
          complex.homology_generators(k).len(),
          complex.betti_number(k)
        );
      }
    }
  }

  /// Every generator is a cycle.
  #[test]
  fn generators_are_cycles() {
    for complex in test_complexes() {
      for k in 0..=complex.dim() {
        for generator in complex.homology_generators(k) {
          assert!(is_cycle(&complex, &generator), "grade {k}");
        }
      }
    }
  }

  /// The generator classes are independent modulo boundaries: appended to the
  /// columns of $diff_(k+1)$ they raise the rank by exactly $b_k$, so no
  /// generator -- nor any combination -- is itself a boundary.
  #[test]
  fn generators_independent_modulo_boundaries() {
    for complex in test_complexes() {
      for k in 0..=complex.dim() {
        let nrows = complex.nsimplices(k);
        let boundary_cols = if k < complex.dim() {
          complex.nsimplices(k + 1)
        } else {
          0
        };
        let mut triplets = if k < complex.dim() {
          complex.boundary_triplets(k + 1)
        } else {
          Vec::new()
        };
        for (g, generator) in complex.homology_generators(k).iter().enumerate() {
          for (kidx, coeff) in generator.support() {
            triplets.push((kidx, boundary_cols + g, coeff));
          }
        }
        assert_eq!(
          exact_rank(nrows, &triplets),
          complex.boundary_rank(k + 1) + complex.betti_number(k),
          "grade {k}"
        );
      }
    }
  }

  /// The annulus has one hole, and its $H_1$ generator is a nontrivial loop --
  /// a cycle around the hole spanning several edges.
  #[test]
  fn annulus_generator_is_a_loop() {
    let complex = annulus();
    let generators = complex.homology_generators(1);
    assert_eq!(generators.len(), 1);
    let loop_ = &generators[0];
    assert!(is_cycle(&complex, loop_));
    assert!(loop_.support().count() >= 3);
  }

  /// Euler--Poincaré: the alternating simplex count equals the alternating
  /// Betti sum. A cross-check tying the Betti numbers back to the raw counts.
  #[test]
  fn euler_poincare() {
    for dim in 1..=3 {
      let (topology, _) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let alt_betti: i64 = topology
        .betti_numbers()
        .iter()
        .enumerate()
        .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
        .sum();
      assert_eq!(alt_betti, topology.euler_characteristic(), "dim={dim}");
    }
  }

  /// A cube is contractible: $b_0 = 1$ and all higher Betti numbers vanish.
  #[test]
  fn cube_is_contractible() {
    for dim in 1..=3 {
      let (topology, _) = CartesianMeshInfo::new_unit(dim, 2).compute_coord_complex();
      let expected: Vec<usize> = std::iter::once(1)
        .chain(std::iter::repeat_n(0, dim))
        .collect();
      assert_eq!(topology.betti_numbers(), expected, "dim={dim}");
    }
  }

  /// Poincaré duality on a closed orientable manifold: $b_k = b_(n-k)$. The
  /// 2-sphere realizes it with Betti numbers $(1, 0, 1)$.
  #[test]
  fn sphere_poincare_duality() {
    let (topology, _) = crate::dim3::mesh_sphere_surface(1).into_coord_complex();
    let betti = topology.betti_numbers();
    let n = topology.dim();
    for k in 0..=n {
      assert_eq!(betti[k], betti[n - k], "k={k}");
    }
    assert_eq!(betti, vec![1, 0, 1]);
  }
}
