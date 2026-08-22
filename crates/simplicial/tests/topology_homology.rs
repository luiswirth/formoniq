//! Simplicial homology over $ZZ$: Euler-Poincaré, the Poincaré lemma on a
//! contractible complex, and the two duality theorems.

mod common;

use common::two_sphere;
use simplicial::Dim;
use simplicial::mesher::grid::CartesianTopology;

/// Euler--Poincaré: the alternating simplex count equals the alternating
/// Betti sum. A cross-check tying the Betti numbers back to the raw counts.
#[test]
fn euler_poincare() {
  for dim in (1..=3usize).map(Dim::from) {
    let topology = CartesianTopology::cube(dim, 2).triangulate();
    let alt_betti: i64 = topology
      .betti_numbers()
      .iter()
      .enumerate()
      .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
      .sum();
    assert_eq!(alt_betti, topology.euler_characteristic(), "dim={dim}");
  }
}

/// The Poincaré lemma, in its combinatorial form: on a contractible complex
/// every cocycle is a coboundary, so $b_k = delta_(k 0)$. The triangulated
/// cube is contractible in every dimension.
#[test]
fn cube_is_contractible() {
  for dim in (1..=3usize).map(Dim::from) {
    let topology = CartesianTopology::cube(dim, 2).triangulate();
    let expected: Vec<usize> = std::iter::once(1)
      .chain(std::iter::repeat_n(0, dim.index()))
      .collect();
    assert_eq!(topology.betti_numbers(), expected, "dim={dim}");
  }
}

/// Poincaré duality on a closed orientable manifold: $b_k = b_(n-k)$. The
/// 2-sphere realizes it with Betti numbers $(1, 0, 1)$.
#[test]
fn sphere_poincare_duality() {
  let topology = two_sphere();
  let betti = topology.betti_numbers();
  let n = topology.dim();
  for k in 0..=n.index() {
    assert_eq!(betti[k], betti[n.index() - k], "k={k}");
  }
  assert_eq!(betti, vec![1, 0, 1]);
}

/// Poincaré--Lefschetz duality on the box, an orientable manifold with
/// boundary: $b_k (K, diff K) = b_(n-k) (K)$. The box is contractible, so the
/// relative Betti numbers are $1$ at the top grade and $0$ below, the
/// harmonic dimension of the essential-BC Hodge-Laplace complex, dual to the
/// natural-BC one which is harmonic only at grade $0$. The relative
/// [`Complex::relative_betti_number`] is computed independently of duality
/// (from the incidence), so their agreement is a real cross-check.
#[test]
fn box_lefschetz_duality() {
  for dim in (1..=3usize).map(Dim::from) {
    let topology = CartesianTopology::cube(dim, 2).triangulate();
    for k in dim.range_inclusive() {
      assert_eq!(
        topology.relative_betti_number(k),
        topology.betti_number(dim - k),
        "dim={dim}, k={k}"
      );
      assert_eq!(
        topology.relative_betti_number(k),
        usize::from(k == dim),
        "dim={dim}, k={k}"
      );
    }
  }
}
