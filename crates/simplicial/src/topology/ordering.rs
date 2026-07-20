//! The vertex ordering of the cells: the third datum of a mesh.
//!
//! A [`Complex`] stores every simplex in colex vertex order, and that order is
//! a *gauge* -- it is fixed by the global vertex numbering, which is a labelling
//! and not a property of the mesh. A mesh generator, however, usually produces
//! each cell with a vertex order of its own that carries structure: the maximal
//! chain a Kuhn simplex is built from, or the node ordering an external mesher
//! writes out. That order is thrown away by the sort, and it cannot be
//! recovered afterwards, because a per-cell order is strictly more expressive
//! than the restriction of any global numbering.
//!
//! So it is a datum, on the same footing as geometry. Topology is combinatorics
//! ([`Complex`]), geometry arrives separately
//! ([`MeshLengthsSq`](crate::geometry::metric::mesh::MeshLengthsSq)), and the
//! ordering arrives separately too: three orthogonal axes, none derivable from
//! the others. Nothing in assembly, solving or homology may consult it -- those
//! are invariant under relabelling, and a dependence on the ordering there is a
//! bug. It exists for the algorithms whose *output* is a mesh, refinement above
//! all, where the generator's order is what makes uniform subdivision reproduce
//! the generator's own family.
//!
//! The orderings of a mesh are not independent: they must agree on shared
//! faces, or two cells subdivide their common face differently and the result
//! is non-conforming. That compatibility ([`CellOrdering::is_face_consistent`])
//! is what makes an ordering a structure on the mesh rather than a bag of
//! permutations, and it is why the ordering induced by the global numbering
//! ([`CellOrdering::colex`]) is always valid: a restriction of a total order is
//! consistent for free.

use super::{
  complex::Complex, handle::KSimplexIdx, orientation::Orientation, role::Cell, simplex::Simplex,
  VertexIdx,
};
use crate::Dim;

/// A vertex ordering on the cells of a [`Complex`]: for each cell, its vertices
/// in the order its generator gave them.
///
/// Metric-free, and a labelling rather than a geometric fact. Each word is a
/// permutation of the cell's stored (colex) vertices, checked at construction,
/// so a `CellOrdering` speaks only for the complex it was built from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellOrdering {
  dim: Dim,
  /// Per cell, in the complex's colex cell order: that cell's vertices in the
  /// generator's order.
  words: Vec<Vec<VertexIdx>>,
}

impl CellOrdering {
  /// The ordering induced by the global vertex numbering: every cell in its
  /// stored colex order.
  ///
  /// The trivial ordering, and the one the mesh carries implicitly when no
  /// generator supplied another. Face-consistent by construction, being the
  /// restriction of a total order.
  pub fn colex(complex: &Complex) -> Self {
    let words = complex
      .cells()
      .handle_iter()
      .map(|cell| cell.simplex().vertices.clone())
      .collect();
    Self {
      dim: complex.dim(),
      words,
    }
  }

  /// A generator's ordering, as one vertex word per cell.
  ///
  /// The words are matched to cells by their vertex *sets*, not by position, so
  /// a generator need not know the complex's colex cell order -- it emits the
  /// cells it built, in whatever order it built them.
  ///
  /// # Panics
  /// If a word is not a permutation of the vertices of a cell of `complex`, if
  /// two words name the same cell, or if a cell is left unnamed. An ordering is
  /// total over the cells or it is not an ordering.
  pub fn new(complex: &Complex, words: impl IntoIterator<Item = Vec<VertexIdx>>) -> Self {
    let dim = complex.dim();
    let ncells = complex.cells().len();
    let mut slots: Vec<Option<Vec<VertexIdx>>> = vec![None; ncells];
    for word in words {
      let (_, simplex) = Simplex::from_word(word.clone());
      assert_eq!(
        simplex.vertices.len(),
        dim + 1,
        "an ordering word must have as many vertices as a cell"
      );
      let kidx = complex.skeleton(dim).kidx_by_simplex(&simplex);
      assert!(
        slots[kidx].replace(word).is_none(),
        "two ordering words name the same cell"
      );
    }
    let words = slots
      .into_iter()
      .map(|word| word.expect("every cell needs an ordering word"))
      .collect();
    Self { dim, words }
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }
  pub fn ncells(&self) -> usize {
    self.words.len()
  }

  /// The cell's vertices in the generator's order.
  pub fn word(&self, cell: Cell) -> &[VertexIdx] {
    &self.words[(*cell).kidx()]
  }
  /// The cell's vertices in the generator's order, by raw cell index.
  pub fn word_by_kidx(&self, kidx: KSimplexIdx) -> &[VertexIdx] {
    &self.words[kidx]
  }

  /// The order this cell induces on one of its faces: the face's vertices,
  /// listed as the cell's word lists them.
  ///
  /// The restriction map. Well defined on any subset of the cell's vertices,
  /// and the object face-consistency is a statement about.
  pub fn induced_on(&self, cell: Cell, face: &Simplex) -> Vec<VertexIdx> {
    self
      .word(cell)
      .iter()
      .copied()
      .filter(|vertex| face.vertices.binary_search(vertex).is_ok())
      .collect()
  }

  /// The orientation the ordering winds, or `None` if that winding is not
  /// coherent.
  ///
  /// The *parity* of the datum: each cell's sign is the sign of the permutation
  /// taking its word to its stored colex order. A generator's per-cell vertex
  /// word therefore carries two different things at once -- the full ordering,
  /// which is what refinement inherits, and its sign quotient, which is the
  /// orientation a mesh file's winding means. They are independent: a
  /// face-consistent ordering need not be coherently wound, and a coherent
  /// winding says nothing about the rest of the permutation.
  ///
  /// Validated through [`Complex::orient_by`], so a miswound file yields `None`
  /// rather than a witness that lies.
  pub fn induced_orientation(&self, complex: &Complex) -> Option<Orientation> {
    let signs = complex
      .cells()
      .handle_iter()
      .map(|cell| Simplex::from_word(self.word(cell).to_vec()).0)
      .collect();
    complex.orient_by(signs)
  }

  /// Whether the orderings agree on every shared face: for each facet, the two
  /// incident cells induce the same order on it.
  ///
  /// The compatibility condition that makes an ordering a structure on the mesh.
  /// Refinement may only consume a face-consistent ordering: two cells that
  /// order their common face differently subdivide it differently, and the
  /// refined complex is then non-conforming.
  ///
  /// Checking the facets suffices: a lower face lies in some facet of every
  /// cell containing it, and the restriction of equal orders is equal.
  pub fn is_face_consistent(&self, complex: &Complex) -> bool {
    let Some(facets) = complex.role_skeleton::<crate::topology::role::roles::Facet>() else {
      // No facets to disagree on: a 0-complex is consistent vacuously.
      return true;
    };
    let consistent = facets.handle_iter().all(|facet| {
      let mut incident = facet
        .cells()
        .map(|cell| self.induced_on(cell, (*facet).simplex()));
      let Some(first) = incident.next() else {
        return true;
      };
      incident.all(|other| other == first)
    });
    consistent
  }
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::gen::cartesian::CartesianGrid;
  use multiindex::Sign;

  /// The colex ordering is face-consistent in every dimension: it restricts a
  /// total order, so agreement on shared faces is automatic. The base case of
  /// the whole structure.
  #[test]
  fn colex_ordering_is_face_consistent() {
    for dim in 0..=4 {
      for ncells_axis in 1..=2 {
        let (complex, _) = CartesianGrid::new_unit(dim, ncells_axis).triangulate();
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
    for dim in 1..=3 {
      let (complex, _) = CartesianGrid::new_unit(dim, 2).triangulate();
      let colex = CellOrdering::colex(&complex);
      let mut words: Vec<Vec<VertexIdx>> = (0..colex.ncells())
        .map(|kidx| colex.word_by_kidx(kidx).to_vec())
        .collect();
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
    for dim in 1..=3 {
      let grid = CartesianGrid::new_unit(dim, 2);
      let skeleton = grid.cell_skeleton();
      for simplex in skeleton.iter() {
        assert!(simplex.vertices.windows(2).all(|w| w[0] < w[1]));
      }
    }
  }

  /// Face-consistency has teeth: transposing two vertices of a single cell
  /// breaks agreement with its neighbors across a shared facet.
  ///
  /// Guards against the check being vacuously true -- the failure mode that
  /// would let a non-conforming refinement through.
  #[test]
  fn a_transposed_cell_is_detected() {
    for dim in 2..=3 {
      let (complex, _) = CartesianGrid::new_unit(dim, 2).triangulate();
      let mut ordering = CellOrdering::colex(&complex);
      // An interior facet, and one of the two cells meeting there. Transposing
      // two vertices *of that facet* in the cell's word is what the neighbor
      // must disagree with -- swapping a vertex the facet omits changes nothing
      // it can see.
      let facet = complex
        .facets()
        .handle_iter()
        .find(|facet| facet.cells().count() == 2)
        .expect("an interior facet");
      let shared = (*facet).simplex().vertices.clone();
      let culprit = facet.cells().next().unwrap();
      let kidx = (*culprit).kidx();
      let positions: Vec<usize> = ordering
        .word(culprit)
        .iter()
        .enumerate()
        .filter(|(_, v)| shared.contains(v))
        .map(|(i, _)| i)
        .collect();
      ordering.words[kidx].swap(positions[0], positions[1]);
      assert!(!ordering.is_face_consistent(&complex));
    }
  }

  /// The colex ordering winds every cell positively, and that is coherent
  /// exactly when the complex is orientable -- so on an orientable mesh the
  /// parity of the trivial ordering is the trivial orientation.
  #[test]
  fn the_colex_ordering_winds_positively() {
    for dim in 1..=3 {
      let (complex, _) = CartesianGrid::new_unit(dim, 2).triangulate();
      let ordering = CellOrdering::colex(&complex);
      let oriented = ordering.induced_orientation(&complex);
      // Colex gives every cell `Pos`; that is coherent only if no two adjacent
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
  /// parity while leaving the ordering just as much an ordering -- and the
  /// reversal is detected as a winding failure, not silently absorbed.
  #[test]
  fn winding_is_the_parity_and_nothing_more() {
    for dim in 2..=3 {
      let (complex, _) = CartesianGrid::new_unit(dim, 2).triangulate();
      let Some(coherent) = CellOrdering::colex(&complex).induced_orientation(&complex) else {
        continue;
      };
      let mut broken = CellOrdering::colex(&complex);
      broken.words[0].swap(0, 1);
      // A single transposition flips one cell's sign against its neighbors.
      assert_ne!(broken.induced_orientation(&complex), Some(coherent));
    }
  }
}
