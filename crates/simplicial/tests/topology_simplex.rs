//! The reference simplex: subsimplex enumeration, edge indexing, the word
//! orientation, and $diff compose diff = 0$.

use itertools::Itertools;
use multiindex::Combination;
use simplicial::Dim;
use simplicial::Sign;
use simplicial::topology::simplex::{
  Simplex, edge_index, nsubsimplices, unit_boundary_operator, unit_subsimps,
};

/// $diff compose diff = 0$ for the reference-cell boundary matrices, swept
/// over every dimension and grade including the ends, where the operator is
/// the zero map into or out of the zero module.
#[test]
fn unit_boundary_squares_to_zero() {
  for dim in (0..=5usize).map(Dim::from) {
    for grade in (dim + 1).range_inclusive() {
      let product = unit_boundary_operator(dim, grade - 1) * unit_boundary_operator(dim, grade);
      assert!(product.iter().all(|&v| v == 0.0), "dim {dim} grade {grade}");
    }
  }
}

#[test]
fn subsimps() {
  for dim in (0..=4usize).map(Dim::from) {
    let simp = Simplex::unit(dim);
    for sub_dim in dim.range_inclusive() {
      let subs = simp.subsimps(sub_dim).collect_vec();
      assert_eq!(subs.len(), nsubsimplices(dim, sub_dim));
      assert!(subs.iter().all(|sub| sub.is_subsimplex_of(&simp)));
      assert!(
        subs
          .iter()
          .all(|sub| sub.relative_to(&simp) == Combination::from_increasing(sub.iter()))
      );
    }
  }
}

/// The edge index is the position of the pair in the enumeration, in either
/// order of the endpoints, and every edge is hit exactly once.
#[test]
fn edge_indices_enumerate_the_edges() {
  for dim in (1..=4usize).map(Dim::from) {
    for (iedge, edge) in unit_subsimps(dim, Dim::ONE).enumerate() {
      let (vi, vj) = (edge.index_at(0), edge.index_at(1));
      assert_eq!(edge_index(vi, vj), iedge);
      assert_eq!(edge_index(vj, vi), iedge);
    }
  }
}

#[test]
fn from_word_orientation() {
  let (sign, simp) = Simplex::from_word(vec![2, 0, 1]);
  assert_eq!(sign, Sign::Pos);
  assert_eq!(simp, Simplex::from([0, 1, 2]));
  let (sign, _) = Simplex::from_word(vec![1, 0, 2]);
  assert_eq!(sign, Sign::Neg);
}

/// The boundary of the boundary is zero.
#[test]
fn boundary_of_boundary_cancels() {
  use std::collections::HashMap;
  let simp = Simplex::unit(Dim::new(3));
  let mut chain: HashMap<Simplex, i32> = HashMap::new();
  for (sign, face) in simp.boundary() {
    for (subsign, subface) in face.boundary() {
      *chain.entry(subface).or_default() += (sign * subsign).as_i32();
    }
  }
  assert!(chain.values().all(|&c| c == 0));
}
