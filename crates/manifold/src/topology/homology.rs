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

use super::complex::Complex;
use crate::Dim;

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
  use crate::gen::cartesian::CartesianMeshInfo;

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
