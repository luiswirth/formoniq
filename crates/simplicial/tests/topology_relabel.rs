//! Vertex relabelling: closing the gaps of a used set while keeping order.

use simplicial::topology::relabel::VertexRelabelling;

/// Closing the gaps keeps the order of the vertices, so a word's own order,
/// and drops exactly the vertices no cell names.
#[test]
fn relabelling_is_monotone_and_onto() {
  let words = [vec![7, 2, 5], vec![2, 9, 7]];
  let relabelling = VertexRelabelling::of_used(words.iter().flatten().copied());

  assert_eq!(relabelling.used(), [2, 5, 7, 9]);
  assert_eq!(relabelling.nvertices(), 4);
  assert_eq!(relabelling.relabel_word(words[0].clone()), [2, 0, 1]);
  assert_eq!(relabelling.relabel_word(words[1].clone()), [0, 3, 2]);
}

/// A gapless list relabels to itself, so an import that needs nothing done to
/// it runs the same code and is left alone.
#[test]
fn a_gapless_list_is_the_identity() {
  let relabelling = VertexRelabelling::of_used([2, 0, 1, 1]);
  assert_eq!(relabelling.relabel_word([0, 1, 2]), [0, 1, 2]);
}
