use super::VertexIdx;

/// The monotone relabelling of the vertices a set of cells uses onto the
/// contiguous range $0..m$.
///
/// A [`Complex`](super::complex::Complex) is built on vertices numbered $0..m$
/// with every one of them used by some cell, while a mesh file routinely names
/// its nodes with gaps: points no cell references, or a node block shared with
/// an object whose cells were not exported. This is the map that closes them.
///
/// It is order-preserving, so it leaves the vertex order within a cell alone,
/// and with it the winding a file means by that order. On a list that is
/// already gapless it is the identity.
///
/// The relabelling is one map applied to several things: the cells, the vertex
/// coordinates, and any per-cell vertex word kept alongside them. They stay in
/// one numbering because they are relabelled by the same object, rather than by
/// the same recipe followed twice.
#[derive(Debug, Clone)]
pub struct VertexRelabelling {
  /// The used vertices in increasing order, hence the new-to-old map.
  used: Vec<VertexIdx>,
}

impl VertexRelabelling {
  /// The relabelling closing the gaps of the vertices the cells name, given as
  /// every occurrence, in any order and with repetition.
  pub fn of_used(vertices: impl IntoIterator<Item = VertexIdx>) -> Self {
    let mut used: Vec<VertexIdx> = vertices.into_iter().collect();
    used.sort_unstable();
    used.dedup();
    Self { used }
  }

  /// How many vertices remain.
  pub fn nvertices(&self) -> usize {
    self.used.len()
  }
  /// The used vertices, in increasing order: the new index of each is its
  /// position here.
  pub fn used(&self) -> &[VertexIdx] {
    &self.used
  }
  /// The new label of a used vertex. Panics on one no cell names, which has no
  /// image.
  pub fn relabel(&self, vertex: VertexIdx) -> VertexIdx {
    self
      .used
      .binary_search(&vertex)
      .expect("an unused vertex has no relabelling")
  }
  /// A cell's vertex word under the relabelling, in the order it was given.
  pub fn relabel_word(&self, word: impl IntoIterator<Item = VertexIdx>) -> Vec<VertexIdx> {
    word.into_iter().map(|v| self.relabel(v)).collect()
  }
}

#[cfg(test)]
mod test {
  use super::*;

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
}
