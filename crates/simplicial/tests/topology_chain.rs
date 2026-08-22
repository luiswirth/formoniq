//! [\`Chain\`]/[\`Cochain\`] as one incidence read both ways: the boundary and
//! the coboundary are adjoint under the [\`pairing\`], and nilpotent as one
//! statement.

use simplicial::mesher::grid::CartesianTopology;
use simplicial::topology::chain::{Chain, Cochain, pairing};
use simplicial::topology::complex::Complex;

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
