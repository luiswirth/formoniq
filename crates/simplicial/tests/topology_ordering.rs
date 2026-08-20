//! [`CellOrdering`]: colex is always face-consistent, a generator's words
//! round-trip by vertex set, and a corrupted ordering is caught rather than
//! silently accepted.

use multiindex::Sign;
use simplicial::Dim;
use simplicial::mesher::grid::CartesianTopology;
use simplicial::topology::VertexIdx;
use simplicial::topology::ordering::CellOrdering;

/// The words of an existing ordering, by [`CellOrdering::word_by_kidx`]: the
/// only public route back to the full word list, and the one
/// [`CellOrdering::new`] round-trips against.
fn words_of(ordering: &CellOrdering) -> Vec<Vec<VertexIdx>> {
  (0..ordering.ncells())
    .map(|kidx| ordering.word_by_kidx(kidx).to_vec())
    .collect()
}

/// The colex ordering is face-consistent in every dimension: it restricts a
/// total order, so agreement on shared faces is automatic. The base case of
/// the whole structure.
#[test]
fn colex_ordering_is_face_consistent() {
  for dim in (0..=4usize).map(Dim::from) {
    for ncells_axis in 1..=2 {
      let complex = CartesianTopology::cube(dim, ncells_axis).triangulate();
      let ordering = CellOrdering::colex(&complex);
      assert_eq!(ordering.ncells(), complex.cells().len());
      assert!(ordering.is_face_consistent(&complex));
    }
  }
}

/// A generator's words are matched to cells by vertex set, not by position,
/// and round-trip: handing back the colex words in any order rebuilds the
/// colex ordering.
#[test]
fn words_are_matched_by_vertex_set() {
  for dim in (1..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();
    let colex = CellOrdering::colex(&complex);
    let mut words = words_of(&colex);
    words.reverse();
    assert_eq!(CellOrdering::new(&complex, words), colex);
  }
}

/// The Kuhn generator's own order is the colex one: a Kuhn chain ascends in
/// the grid's vertex numbering. This is why uniform refinement reproduces the
/// generator's family at the first level without any ordering being carried,
/// and why nothing noticed the datum was missing.
#[test]
fn the_kuhn_chain_ascends() {
  for dim in (1..=3usize).map(Dim::from) {
    let grid = CartesianTopology::cube(dim, 2);
    let skeleton = grid.cell_skeleton();
    for simplex in skeleton.iter() {
      assert!(simplex.vertices.windows(2).all(|w| w[0] < w[1]));
    }
  }
}

/// Face-consistency has teeth: transposing two vertices of a single cell
/// breaks agreement with its neighbors across a shared facet.
///
/// Guards against the check being vacuously true, the failure mode that
/// would let a non-conforming refinement through.
#[test]
fn a_transposed_cell_is_detected() {
  for dim in (2..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();
    let colex = CellOrdering::colex(&complex);
    // An interior facet, and one of the two cells meeting there. Transposing
    // two vertices of that facet in the cell's word is what the neighbor
    // must disagree with, swapping a vertex the facet omits changes nothing
    // it can see.
    let facet = complex
      .facets()
      .handle_iter()
      .find(|facet| facet.cells().count() == 2)
      .expect("an interior facet");
    let shared = (*facet).simplex().vertices.clone();
    let culprit = facet.cells().next().unwrap();
    let kidx = (*culprit).kidx();
    let positions: Vec<usize> = colex
      .word(culprit)
      .iter()
      .enumerate()
      .filter(|(_, v)| shared.contains(v))
      .map(|(i, _)| i)
      .collect();
    let mut words = words_of(&colex);
    words[kidx].swap(positions[0], positions[1]);
    let broken = CellOrdering::new(&complex, words);
    assert!(!broken.is_face_consistent(&complex));
  }
}

/// The colex ordering winds every cell positively, and that is coherent
/// exactly when the complex is orientable, so on an orientable mesh the
/// parity of the trivial ordering is the trivial orientation.
#[test]
fn the_colex_ordering_winds_positively() {
  for dim in (1..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();
    let ordering = CellOrdering::colex(&complex);
    let oriented = ordering.induced_orientation(&complex);
    // Colex gives every cell `Pos`. That is coherent only if no two adjacent
    // cells induce the same orientation on their shared facet, which a Kuhn
    // grid does not generally satisfy. Either way the answer is honest: a
    // witness or `None`, never a forged one.
    if let Some(orientation) = oriented {
      assert!(orientation.signs().iter().all(|&s| s == Sign::Pos));
      assert!(complex.is_orientable());
    }
  }
}

/// Ordering and winding are independent: reversing a cell's word flips its
/// parity while leaving the ordering just as much an ordering, and the
/// reversal is detected as a winding failure, not silently absorbed.
#[test]
fn winding_is_the_parity_and_nothing_more() {
  for dim in (2..=3usize).map(Dim::from) {
    let complex = CartesianTopology::cube(dim, 2).triangulate();
    let Some(coherent) = CellOrdering::colex(&complex).induced_orientation(&complex) else {
      continue;
    };
    let mut words = words_of(&CellOrdering::colex(&complex));
    words[0].swap(0, 1);
    let broken = CellOrdering::new(&complex, words);
    // A single transposition flips one cell's sign against its neighbors.
    assert_ne!(broken.induced_orientation(&complex), Some(coherent));
  }
}
