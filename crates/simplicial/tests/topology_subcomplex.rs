//! Laws for the cochain trace $"tr": C^k (K) -> C^k (L)$ onto a
//! codimension-1 subcomplex: it restricts each coefficient to its parent's,
//! it is a cochain map, and it is total at the parent's top grade, where the
//! subcomplex carries no simplices at all.

use simplicial::topology::{chain::Cochain, complex::Complex};

/// The trace is the restriction of coefficients: each subcomplex simplex
/// carries exactly its parent's value. Stated on a cochain that distinguishes
/// every simplex, so an index permutation could not pass, and swept over
/// every dimension and every grade the subcomplex has.
#[test]
fn the_trace_restricts_each_coefficient_to_its_parent() {
  for dim in 1..=4 {
    let topology = Complex::unit(dim);
    let boundary = topology
      .boundary_complex()
      .expect("a simplex has a boundary");
    for grade in boundary.dim().range_inclusive() {
      let cochain = Cochain::from_function(|s| s.kidx() as f64, grade, &topology);
      let traced = boundary.trace(&cochain);

      let parent_kidxs = boundary.parent_kidxs(grade);
      assert_eq!(traced.len(), parent_kidxs.len());
      assert_eq!(traced.grade(), grade);
      for (kidx, &parent_kidx) in parent_kidxs.iter().enumerate() {
        assert_eq!(traced.coeffs()[kidx], parent_kidx as f64);
      }
    }
  }
}

/// $"tr" compose dif = dif compose "tr"$: the trace is a cochain map, which is
/// what makes the traced coefficients a discrete form on the subcomplex
/// rather than a resampling of one. Checked below the subcomplex's top grade,
/// where both sides have somewhere to land.
#[test]
fn the_trace_commutes_with_the_differential() {
  for dim in 2..=4 {
    let topology = Complex::unit(dim);
    let boundary = topology
      .boundary_complex()
      .expect("a simplex has a boundary");
    for grade in boundary.dim().range() {
      let cochain = Cochain::from_function(|s| (s.kidx() as f64).sin(), grade, &topology);
      let traced_then_dif = boundary.trace(&cochain).dif(boundary.complex());
      let dif_then_traced = boundary.trace(&cochain.dif(&topology));
      assert_eq!(traced_then_dif.coeffs(), dif_then_traced.coeffs());
    }
  }
}

/// The parent's top grade has no trace: a codimension-1 subcomplex carries no
/// $n$-simplices, so $C^n (L) = 0$ and the trace is the empty cochain. The
/// degenerate case runs on the same code and returns the trivial answer
/// rather than indexing one past the subcomplex's dimension.
#[test]
fn the_top_grade_traces_to_the_zero_group() {
  for dim in 1..=4 {
    let topology = Complex::unit(dim);
    let boundary = topology
      .boundary_complex()
      .expect("a simplex has a boundary");
    let top = Cochain::constant(1.0, topology.skeleton(dim));
    let traced = boundary.trace(&top);
    assert_eq!(traced.grade(), topology.dim());
    assert!(traced.is_empty());
  }
}
