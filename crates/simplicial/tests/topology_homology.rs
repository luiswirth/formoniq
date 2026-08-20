//! Simplicial homology over $ZZ$: the generator count matches the Betti
//! number, every generator is a cycle independent modulo boundaries, and the
//! relative theory (Poincaré-Lefschetz duality) agrees with the absolute one.

mod common;

use common::{annulus, is_cycle, test_complexes, two_sphere};
use simplicial::Dim;
use simplicial::linalg::exact::IntegerMatrix;
use simplicial::mesher::grid::CartesianTopology;

/// One generator per Betti number, in every grade.
#[test]
fn generators_count_matches_betti() {
  for complex in test_complexes() {
    for k in complex.dim().range_inclusive() {
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
    for k in complex.dim().range_inclusive() {
      for generator in complex.homology_generators(k) {
        assert!(is_cycle(&complex, &generator), "grade {k}");
      }
    }
  }
}

/// The generator classes are independent modulo boundaries: appended to the
/// columns of $diff_(k+1)$ they raise the rank by exactly $b_k$, so no
/// generator, nor any combination, is itself a boundary.
#[test]
fn generators_independent_modulo_boundaries() {
  for complex in test_complexes() {
    for k in complex.dim().range_inclusive() {
      let boundaries = complex.integral_boundary(k + 1);
      let generators = complex.homology_generators(k);

      let mut triplets = boundaries.triplets().to_vec();
      for (g, generator) in generators.iter().enumerate() {
        for (kidx, &coeff) in generator.support() {
          triplets.push((kidx, boundaries.ncols() + g, coeff));
        }
      }
      let augmented = IntegerMatrix::new(
        boundaries.nrows(),
        boundaries.ncols() + generators.len(),
        triplets,
      );
      assert_eq!(
        augmented.rank(),
        boundaries.rank() + complex.betti_number(k),
        "grade {k}"
      );
    }
  }
}

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

/// A cube is contractible: $b_0 = 1$ and all higher Betti numbers vanish.
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

/// On a closed manifold ($diff K = nothing$) the relative complex is the full
/// complex, so relative and absolute Betti numbers coincide. The 2-sphere.
#[test]
fn closed_manifold_relative_equals_absolute() {
  let topology = two_sphere();
  for k in topology.dim().range_inclusive() {
    assert_eq!(
      topology.relative_betti_number(k),
      topology.betti_number(k),
      "k={k}"
    );
  }
}

/// The relative generators match the relative Betti numbers, are supported in
/// the interior, and are cycles *relative to* the boundary: $diff z$ is
/// carried by $diff K$ rather than vanishing.
#[test]
fn relative_generators_count_and_support() {
  for complex in test_complexes() {
    for k in complex.dim().range_inclusive() {
      let generators = complex.relative_homology_generators(k);
      assert_eq!(
        generators.len(),
        complex.relative_betti_number(k),
        "grade {k}"
      );
      let interior = complex.interior_selection(k);
      let interior_below = complex.interior_selection(k - 1);
      for generator in &generators {
        assert!(
          generator
            .support()
            .all(|(kidx, _)| interior.position(kidx).is_some()),
          "a relative cycle must vanish on the boundary, grade {k}"
        );
        assert!(
          generator
            .boundary(&complex)
            .support()
            .all(|(kidx, _)| interior_below.position(kidx).is_none()),
          "a relative cycle's boundary must lie in the boundary, grade {k}"
        );
      }
    }
  }
}

#[test]
fn annulus_generator_is_a_loop() {
  let complex = annulus();
  let generators = complex.homology_generators(1);
  assert_eq!(generators.len(), 1);
  let loop_ = &generators[0];
  assert!(is_cycle(&complex, loop_));
  assert!(loop_.support().count() >= 3);
}
