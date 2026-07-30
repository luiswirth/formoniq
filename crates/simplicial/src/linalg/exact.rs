//! Exact linear algebra on sparse integer matrices.
//!
//! What the discrete invariants are computed with. A Betti number and a
//! generator are integers and a lattice, not quantities near an integer, so
//! nothing here may depend on where a singular value falls relative to a
//! tolerance: the rank is a Gaussian elimination over a prime field and the
//! kernel is one over $QQ$ in `BigRational`, whose coefficient growth no mesh
//! can overflow.
//!
//! The homology and the cohomology of a complex are the same computation on
//! the same matrix, read in its two directions, so the computation lives here
//! and neither of them owns it. [`quotient_generators`] is that computation:
//! representatives of $ker A slash "im" B$ for a composable pair $A B = 0$,
//! which is what a homology and a cohomology class each are.

use super::Selection;

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

use std::collections::BTreeMap;

/// A sparse integer matrix, by its `(row, col, value)` triplets.
///
/// Repeated coordinates accumulate, so a caller may push an incidence more
/// than once.
#[derive(Clone, Debug)]
pub struct IntegerMatrix {
  nrows: usize,
  ncols: usize,
  triplets: Vec<(usize, usize, i64)>,
}

impl IntegerMatrix {
  pub fn new(nrows: usize, ncols: usize, triplets: Vec<(usize, usize, i64)>) -> Self {
    debug_assert!(
      triplets.iter().all(|&(r, c, _)| r < nrows && c < ncols),
      "a triplet lies outside the declared shape"
    );
    Self {
      nrows,
      ncols,
      triplets,
    }
  }

  pub fn nrows(&self) -> usize {
    self.nrows
  }
  pub fn ncols(&self) -> usize {
    self.ncols
  }
  /// The entries. Read back out only by the tests: a computation here consumes
  /// a matrix through its rank, its kernel or its columns.
  #[cfg(test)]
  pub fn triplets(&self) -> &[(usize, usize, i64)] {
    &self.triplets
  }

  /// The transpose, which turns a boundary operator into a coboundary one and
  /// so a homology computation into a cohomology one.
  pub fn transpose(&self) -> Self {
    Self::new(
      self.ncols,
      self.nrows,
      self.triplets.iter().map(|&(r, c, v)| (c, r, v)).collect(),
    )
  }

  /// The submatrix on the selected rows and columns, reindexed to their
  /// positions within the selections.
  ///
  /// The matrix of the same map between the subspaces the selections span,
  /// which is how a relative complex is formed: striking the simplices of a
  /// subcomplex from both the domain and the codomain.
  pub fn submatrix(&self, rows: &Selection, cols: &Selection) -> Self {
    Self::new(
      rows.len(),
      cols.len(),
      self
        .triplets
        .iter()
        .filter_map(|&(r, c, v)| Some((rows.position(r)?, cols.position(c)?, v)))
        .collect(),
    )
  }

  /// The rank over $QQ$, as the maximum of the ranks over two prime fields.
  ///
  /// The rank over $FF_p$ equals the rational rank unless $p$ divides the
  /// product of the invariant factors, which for an incidence matrix are tiny;
  /// two primes near $2^30$ put the exception out of reach of any
  /// representable mesh, and their pairwise product still fits in an `i64`, so
  /// the modular arithmetic stays exact without overflow.
  pub fn rank(&self) -> usize {
    const PRIMES: [i64; 2] = [1_000_000_007, 1_000_000_009];
    PRIMES
      .iter()
      .map(|&p| self.rank_mod(p))
      .max()
      .expect("PRIMES is nonempty")
  }

  /// The rank over $FF_p$ by sparse Gaussian elimination: reduce each row
  /// against a table of pivot rows keyed by their leading column. Reducing a
  /// row's leading column introduces only later columns, so the leading column
  /// strictly advances and the reduction terminates. A row that survives to a
  /// fresh leading column becomes a new pivot.
  fn rank_mod(&self, p: i64) -> usize {
    let mut rows: Vec<BTreeMap<usize, i64>> = vec![BTreeMap::new(); self.nrows];
    for &(r, c, v) in &self.triplets {
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
        // row -= coeff * pivot. The pivot has a leading 1 at `lead`.
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

  /// A basis of $ker$ as rational column vectors of length [`ncols`](Self::ncols),
  /// from the reduced row echelon form: each non-pivot (free) column yields one
  /// kernel vector.
  fn kernel(&self) -> Vec<Vec<BigRational>> {
    let mut mat = vec![vec![BigRational::zero(); self.ncols]; self.nrows];
    for &(r, c, v) in &self.triplets {
      mat[r][c] += BigRational::from_integer(BigInt::from(v));
    }

    let pivot_cols = reduced_row_echelon(&mut mat, self.ncols);
    let mut is_pivot = vec![false; self.ncols];
    for &c in &pivot_cols {
      is_pivot[c] = true;
    }

    (0..self.ncols)
      .filter(|c| !is_pivot[*c])
      .map(|free| {
        let mut kernel_vector = vec![BigRational::zero(); self.ncols];
        kernel_vector[free] = BigRational::one();
        // The free column, back-substituted through the pivots.
        for (row, &pivot_col) in pivot_cols.iter().enumerate() {
          kernel_vector[pivot_col] = -mat[row][free].clone();
        }
        kernel_vector
      })
      .collect()
  }

  /// The columns as rational vectors of length [`nrows`](Self::nrows): a
  /// spanning set of the image.
  fn columns(&self) -> Vec<Vec<BigRational>> {
    let mut columns = vec![vec![BigRational::zero(); self.nrows]; self.ncols];
    for &(r, c, v) in &self.triplets {
      columns[c][r] += BigRational::from_integer(BigInt::from(v));
    }
    columns
  }
}

/// Representatives whose classes are a basis of $ker A slash "im" B$ over $QQ$,
/// for a composable pair $A compose B = 0$ of integer matrices.
///
/// The subquotient every (co)homology class lives in, computed once: with $A =
/// diff_k$ and $B = diff_(k+1)$ it is $H_k$, and with the transposes it is
/// $H^k$. The two are one routine because they are one construction, the
/// incidence read in its two directions.
///
/// The boundaries seed an echelon basis first, so a cycle enlarging the span is
/// exactly one that is not a boundary, and the accepted cycles are independent
/// modulo boundaries by construction. Which cycles survive depends on the
/// elimination order: these are representatives, never minimizers, and optimal
/// generators are a separate and hard problem.
///
/// The classes span the free part over $QQ$; they need not generate the
/// integral lattice $ker A slash "im" B$ modulo torsion, which would take a
/// Smith or Hermite normal form. The coefficients returned are primitive (their
/// gcd is 1) with the first nonzero one positive.
///
/// # Panics
/// If the pair is not composable on the module in the middle.
pub fn quotient_generators(outgoing: &IntegerMatrix, incoming: &IntegerMatrix) -> Vec<Vec<i64>> {
  assert_eq!(
    outgoing.ncols(),
    incoming.nrows(),
    "the two maps must meet on one module"
  );

  let mut span = EchelonSpan::default();
  for boundary in incoming.columns() {
    span.insert(boundary);
  }
  outgoing
    .kernel()
    .into_iter()
    .filter_map(|cycle| span.insert(cycle).then(|| primitive(&span.last)))
    .collect()
}

/// An incrementally built row-echelon basis of a subspace of $QQ^n$, keyed by
/// leading (pivot) coordinate. [`insert`](Self::insert) reduces a vector against
/// the basis and, if a nonzero remainder survives, records it and reports the
/// vector as having enlarged the span.
#[derive(Default)]
struct EchelonSpan {
  /// Pivot coordinate -> a basis vector with a leading 1 there.
  pivots: BTreeMap<usize, Vec<BigRational>>,
  /// The last inserted (unnormalized) vector, exposed so a caller can use an
  /// accepted vector without reducing it a second time.
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

/// The integer vector in the same direction as a rational one, with primitive
/// coefficients: clear the denominators, divide out their gcd, and fix the sign
/// so the first nonzero coefficient is positive.
///
/// Primitivity is what makes the representative canonical along its own ray;
/// the sign convention is what makes it canonical up to none at all.
fn primitive(vector: &[BigRational]) -> Vec<i64> {
  let denom_lcm = vector
    .iter()
    .fold(BigInt::one(), |acc, x| acc.lcm(x.denom()));
  let mut ints: Vec<BigInt> = vector
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

  ints
    .iter()
    .map(|x| x.to_i64().expect("a primitive coefficient fits in i64"))
    .collect()
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
