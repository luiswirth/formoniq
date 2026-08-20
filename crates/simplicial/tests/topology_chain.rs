//! [`Chain`]/[`Cochain`] as one incidence read both ways: boundary and
//! coboundary are adjoint under the [`pairing`], both nilpotent as one
//! statement, and agree with the assembled operator.

use simplicial::linalg::CsrMatrix;
use simplicial::mesher::grid::CartesianTopology;
use simplicial::topology::chain::{Chain, Cochain, pairing};
use simplicial::topology::complex::Complex;
use simplicial::topology::data::SkeletonData;

fn probe_complex(dim: usize) -> Complex {
  CartesianTopology::cube(dim, 2).triangulate()
}

/// A probe chain and cochain of a grade, with coefficients that are neither
/// constant nor symmetric, so a law cannot pass by accident.
fn probe_chain(topology: &Complex, grade: usize) -> Chain {
  let coeffs: Vec<i64> = (0..topology.nsimplices(grade))
    .map(|i| (i % 7) as i64 - 3)
    .collect();
  Chain::from_vec(grade, coeffs)
}
fn probe_cochain(topology: &Complex, grade: usize) -> Cochain {
  Cochain::from_function(|s| ((s.kidx() % 5) as f64) - 2.0, grade, topology)
}
/// The probe chain over $RR$, so it pairs with the probe cochain.
fn probe_real_chain(topology: &Complex, grade: usize) -> Chain<f64> {
  probe_chain(topology, grade).extend_scalars(|&c| c as f64)
}

/// The boundary and the coboundary are adjoint under the chain-cochain
/// pairing: $angle.l dif omega, c angle.r = angle.l omega, diff c angle.r$.
///
/// The statement that makes $C^k$ the dual complex of $C_k$ rather than
/// merely a module of the same rank, and the reason the coboundary is the
/// transpose of the boundary. It holds with no metric, no orientation and no
/// geometry.
#[test]
fn the_boundary_and_the_coboundary_are_adjoint() {
  for dim in 1..=3 {
    let topology = probe_complex(dim);
    for grade in 0..dim {
      let cochain = probe_cochain(&topology, grade);
      let chain = probe_real_chain(&topology, grade + 1);

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
      let chain = probe_real_chain(&topology, grade + 2);

      // One step must not already vanish, or both halves below hold for the
      // wrong reason. It is checked against a chain one grade up rather than
      // against the boundary: adjointness makes
      // $angle.l dif omega, diff c angle.r = angle.l dif dif omega, c angle.r$,
      // which is zero for the very reason being tested.
      assert!(
        pairing(
          &cochain.dif(&topology),
          &probe_real_chain(&topology, grade + 1)
        )
        .abs()
          > 1e-9,
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
  for dim in 1..=3 {
    let topology = probe_complex(dim);
    for grade in 0..=dim {
      let cochain = probe_cochain(&topology, grade);
      let assembled = CsrMatrix::from(&topology.coboundary_operator(grade)) * cochain.coeffs();
      assert_eq!(cochain.dif(&topology).coeffs(), &assembled);

      let chain = probe_chain(&topology, grade);
      let boundary = chain.boundary(&topology);
      let matrix = CsrMatrix::from(topology.boundary_operator(grade));
      let applied = matrix * probe_real_chain(&topology, grade).into_coeffs();
      for (kidx, &coefficient) in boundary.coeffs().iter().enumerate() {
        assert_eq!(coefficient as f64, applied[kidx]);
      }
    }
  }
}

/// Extension of scalars commutes with the differential, which is what makes
/// a ring map a map of complexes: an incidence coefficient is $plus.minus 1$,
/// and every ring map fixes those.
#[test]
fn extending_scalars_commutes_with_the_differential() {
  for dim in 1..=3 {
    let topology = probe_complex(dim);
    for grade in 0..=dim {
      let chain = probe_chain(&topology, grade);
      let cast_then_bounded = chain.extend_scalars(|&c| c as f64).boundary(&topology);
      let bounded_then_cast = chain.boundary(&topology).extend_scalars(|&c| c as f64);
      assert_eq!(cast_then_bounded.coeffs(), bounded_then_cast.coeffs());
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
      let chain = probe_real_chain(&topology, grade);

      let expected: f64 = chain
        .support()
        .map(|(kidx, multiplicity)| cochain.coeffs()[kidx] * multiplicity)
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
