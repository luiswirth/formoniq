use super::complex::Complex;
use crate::{
  Dim,
  topology::{
    VertexIdx,
    role::{Cell, Edge, Roled, Vertex},
    simplex::Simplex,
    skeleton::Skeleton,
  },
};

use multiindex::Sign;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use std::collections::BTreeSet;

/// An index identifying a simplex in a skeleton.
pub type KSimplexIdx = usize;

/// An index identifying a simplex in the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SimplexIdx {
  pub dim: Dim,
  pub kidx: KSimplexIdx,
}
impl From<(Dim, KSimplexIdx)> for SimplexIdx {
  fn from((dim, kidx): (Dim, KSimplexIdx)) -> Self {
    Self { dim, kidx }
  }
}
impl SimplexIdx {
  pub fn new(dim: Dim, kidx: KSimplexIdx) -> Self {
    Self { dim, kidx }
  }
  pub fn dim(&self) -> Dim {
    self.dim
  }

  pub fn is_valid(self, complex: &Complex) -> bool {
    self.dim <= complex.dim() && self.kidx < complex.skeleton(self.dim).len()
  }
  pub fn assert_valid(self, mesh: &Complex) {
    assert!(self.is_valid(mesh), "Not a valid simplex index.");
  }
  pub fn handle(self, complex: &Complex) -> SimplexRef<'_> {
    SimplexRef::new(complex, self)
  }
}

/// A handle to a simplex in the mesh.
#[derive(Copy, Clone)]
pub struct SimplexRef<'c> {
  complex: &'c Complex,
  idx: SimplexIdx,
}
impl std::fmt::Debug for SimplexRef<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("SimplexRef")
      .field("idx", &self.idx)
      .field("vertices", &self.simplex().vertices)
      .finish()
  }
}

/// Identity and the escape hatch to the raw combinatorial atom.
impl<'m> SimplexRef<'m> {
  pub fn new(complex: &'m Complex, idx: SimplexIdx) -> Self {
    idx.assert_valid(complex);
    Self { complex, idx }
  }
  pub fn idx(&self) -> SimplexIdx {
    self.idx
  }
  pub fn kidx(&self) -> KSimplexIdx {
    self.idx.kidx
  }
  pub fn dim(&self) -> Dim {
    self.idx.dim
  }
  pub fn complex(&self) -> &'m Complex {
    self.complex
  }
  /// Whether this handle borrows `complex` -- the object, by identity, never
  /// a structural equal: a clone of a complex is a different complex, and
  /// handles and role proofs do not transfer to it.
  pub fn belongs_to(&self, complex: &Complex) -> bool {
    std::ptr::eq(self.complex, complex)
  }

  /// The underlying combinatorial simplex (its sorted vertex set): the escape
  /// hatch from ref-world down to the raw atom. Navigation lives on the ref;
  /// the pure combinatorics lives on [`Simplex`].
  pub fn simplex(&self) -> &'m Simplex {
    self
      .complex
      .skeleton_raw(self.idx.dim)
      .simplex_by_kidx(self.idx.kidx)
  }
  pub fn nvertices(&self) -> usize {
    self.simplex().nvertices()
  }
  pub fn contains(&self, vertex: VertexIdx) -> bool {
    self.simplex().contains(vertex)
  }

  pub fn skeleton(&self) -> SkeletonRef<'m> {
    self.complex.skeleton(self.idx.dim)
  }
  pub fn skeleton_raw(&self) -> &'m Skeleton {
    self.complex.skeleton_raw(self.idx.dim)
  }
}

/// Navigation: every method returns other [`SimplexRef`]s, so a ref reads like
/// a simplex you can walk the complex from. Down-incidence is pure
/// combinatorics on the vertex set; up-incidence is looked up through the
/// stored cell incidence.
impl<'m> SimplexRef<'m> {
  // --- down-incidence ---

  /// The vertices of this simplex, with their [`Vertex`] proofs by
  /// construction.
  pub fn vertices(self) -> impl Iterator<Item = Vertex<'m>> {
    let complex = self.complex;
    self
      .simplex()
      .iter()
      .map(move |v| Roled::trusted(SimplexIdx::new(0, v).handle(complex)))
  }
  /// The edges of this simplex, with their [`Edge`] proofs by construction.
  pub fn edges(self) -> impl Iterator<Item = Edge<'m>> {
    self.faces(1).map(Roled::trusted)
  }
  /// The `dim`-dimensional faces (subsimplices) of this simplex.
  pub fn faces(self, dim: Dim) -> impl Iterator<Item = SimplexRef<'m>> {
    let complex = self.complex;
    self
      .simplex()
      .subsimps(dim)
      .map(move |sub| complex.skeleton(dim).handle_by_simplex(&sub))
  }
  /// The facets: the codimension-1 faces. Empty for a vertex.
  pub fn facets(self) -> impl Iterator<Item = SimplexRef<'m>> {
    let below = self.idx.dim.checked_sub(1);
    below.into_iter().flat_map(move |d| self.faces(d))
  }
  /// The signed boundary $diff sigma$: each facet with its incidence sign.
  pub fn boundary(self) -> impl Iterator<Item = (Sign, SimplexRef<'m>)> {
    let complex = self.complex;
    let below = self.idx.dim.wrapping_sub(1);
    let has_boundary = self.idx.dim >= 1;
    self
      .simplex()
      .boundary()
      .filter(move |_| has_boundary)
      .map(move |signed| {
        (
          signed.sign,
          complex.skeleton(below).handle_by_simplex(&signed.simplex),
        )
      })
  }

  // --- up-incidence ---

  /// The cells (top-dimensional cofaces) containing this simplex, with their
  /// [`Cell`] proofs by construction: the intersection of the vertex-cell
  /// lists over its vertices.
  pub fn cells(self) -> impl Iterator<Item = Cell<'m>> {
    let complex = self.complex;
    let top = complex.dim();
    let mut vertices = self.simplex().iter();
    let cells: Vec<KSimplexIdx> = match vertices.next() {
      None => Vec::new(),
      Some(v0) => {
        let mut shared = complex.vertex_cells(v0).to_vec();
        for v in vertices {
          let others = complex.vertex_cells(v);
          shared.retain(|cell| others.binary_search(cell).is_ok());
        }
        shared
      }
    };
    cells
      .into_iter()
      .map(move |kidx| Roled::trusted(SimplexIdx::new(top, kidx).handle(complex)))
  }
  /// The `dim`-dimensional cofaces (supersimplices) of this simplex.
  pub fn cofaces(self, dim: Dim) -> impl Iterator<Item = SimplexRef<'m>> {
    let complex = self.complex;
    let simplex = self.simplex();
    let mut sups: Vec<Simplex> = self
      .cells()
      .flat_map(|cell| simplex.supersimps(dim, cell.simplex()).collect::<Vec<_>>())
      .collect();
    sups.sort_unstable();
    sups.dedup();
    sups
      .into_iter()
      .map(move |sup| complex.skeleton(dim).handle_by_simplex(&sup))
  }

  // --- neighborhoods ---

  /// The open star: every coface of every dimension (this simplex included).
  pub fn star(self) -> impl Iterator<Item = SimplexRef<'m>> {
    (self.idx.dim..=self.complex.dim()).flat_map(move |d| self.cofaces(d).collect::<Vec<_>>())
  }
  /// The link: the faces opposite this simplex across the cells containing it
  /// (the boundary of the star). For a vertex of a triangulation, the
  /// surrounding polygon.
  pub fn link(self) -> impl Iterator<Item = SimplexRef<'m>> {
    let complex = self.complex;
    let own: BTreeSet<VertexIdx> = self.simplex().iter().collect();
    let mut result: BTreeSet<SimplexIdx> = BTreeSet::new();
    for cell in self.cells() {
      let opposite: Vec<VertexIdx> = cell.simplex().iter().filter(|v| !own.contains(v)).collect();
      let opposite = Simplex::new(opposite);
      for d in 0..opposite.nvertices() {
        for face in opposite.subsimps(d) {
          result.insert(complex.skeleton(d).handle_by_simplex(&face).idx());
        }
      }
    }
    result.into_iter().map(move |idx| idx.handle(complex))
  }
}

impl PartialEq for SimplexRef<'_> {
  fn eq(&self, other: &Self) -> bool {
    std::ptr::eq(self.complex, other.complex) && self.idx == other.idx
  }
}
impl Eq for SimplexRef<'_> {}
impl std::hash::Hash for SimplexRef<'_> {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    std::ptr::from_ref::<Complex>(self.complex).hash(state);
    self.idx.hash(state);
  }
}

/// A handle to a skeleton in the mesh.
pub struct SkeletonRef<'m> {
  complex: &'m Complex,
  dim: Dim,
}
impl std::ops::Deref for SkeletonRef<'_> {
  type Target = Skeleton;
  fn deref(&self) -> &Self::Target {
    self.complex.skeleton_raw(self.dim)
  }
}

impl<'m> SkeletonRef<'m> {
  pub fn new(complex: &'m Complex, dim: Dim) -> Self {
    assert!(dim <= complex.dim());
    Self { complex, dim }
  }
  pub fn handle_by_kidx(&self, idx: KSimplexIdx) -> SimplexRef<'m> {
    SimplexIdx::new(self.dim, idx).handle(self.complex)
  }
  pub fn handle_by_simplex(&self, simp: &Simplex) -> SimplexRef<'m> {
    let idx = self.kidx_by_simplex(simp);
    SimplexIdx::new(self.dim, idx).handle(self.complex)
  }
  pub fn handle_iter(&self) -> impl ExactSizeIterator<Item = SimplexRef<'m>> {
    (0..self.len()).map(|idx| SimplexIdx::new(self.dim, idx).handle(self.complex))
  }
  pub fn handle_par_iter(&self) -> impl ParallelIterator<Item = SimplexRef<'m>> {
    (0..self.len())
      .into_par_iter()
      .map(|idx| SimplexIdx::new(self.dim, idx).handle(self.complex))
  }
}
