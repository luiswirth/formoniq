use multi_index::variants::CanonicalOrder;

use crate::{
  topology::simplex::{Simplex, SortedSimplex},
  Dim,
};

use super::{attribute::SparseSignChain, Complex, ComplexSkeleton, SimplexData};

pub type KSimplexIdx = usize;

impl Complex {
  pub fn skeletons(&self) -> impl Iterator<Item = SkeletonHandle> {
    (0..=self.dim()).map(|d| SkeletonHandle::new(self, d))
  }
  pub fn skeleton(&self, dim: Dim) -> SkeletonHandle {
    SkeletonHandle::new(self, dim)
  }
  pub fn nsimplicies(&self, dim: Dim) -> usize {
    self.skeleton(dim).len()
  }
  pub fn vertices(&self) -> SkeletonHandle {
    self.skeleton(0)
  }
  pub fn edges(&self) -> SkeletonHandle {
    self.skeleton(1)
  }
  pub fn facets(&self) -> SkeletonHandle {
    self.skeleton(self.dim() - 1)
  }
  pub fn cells(&self) -> SkeletonHandle {
    self.skeleton(self.dim())
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

  pub fn handle(self, complex: &Complex) -> SimplexHandle {
    SimplexHandle::new(complex, self)
  }
}

#[derive(Copy, Clone)]
pub struct SimplexHandle<'c> {
  complex: &'c Complex,
  idx: SimplexIdx,
}
impl<'m> std::fmt::Debug for SimplexHandle<'m> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("SimplexHandle")
      .field("idx", &self.idx)
      .field("complex", &(self.complex as *const Complex))
      .finish()
  }
}

impl<'m> SimplexHandle<'m> {
  pub fn new(complex: &'m Complex, idx: SimplexIdx) -> Self {
    idx.assert_valid(complex);
    Self { complex, idx }
  }

  pub fn idx(&self) -> SimplexIdx {
    self.idx
  }
  pub fn dim(&self) -> Dim {
    self.idx.dim()
  }
  pub fn kidx(&self) -> KSimplexIdx {
    self.idx.kidx
  }

  pub fn complex(&self) -> &'m Complex {
    self.complex
  }
  pub fn skeleton(&self) -> SkeletonHandle<'m> {
    self.complex.skeleton(self.dim())
  }
  pub fn simplex_data(&self) -> &SimplexData {
    self.complex.skeletons[self.dim()]
      .get_index(self.kidx())
      .unwrap()
      .1
  }
  pub fn nvertices(&self) -> usize {
    self.simplex_set().nvertices()
  }
  pub fn simplex_set(&self) -> &'m SortedSimplex {
    self.complex.skeletons[self.dim()]
      .get_index(self.kidx())
      .unwrap()
      .0
  }

  pub fn anti_boundary(&self) -> SparseSignChain {
    let mut idxs = Vec::new();
    let mut signs = Vec::new();
    for parent in self.cocells() {
      for sup in self.simplex_set().anti_boundary(parent.simplex_set()) {
        let sign = sup.sign;
        let idx = self
          .complex
          .skeleton(self.dim() + 1)
          .get_by_simplex(&sup.into_simplex())
          .kidx();

        idxs.push(idx);
        signs.push(sign);
      }
    }
    SparseSignChain::new(self.dim() + 1, idxs, signs)
  }

  pub fn boundary_chain(&self) -> SparseSignChain {
    let mut idxs = Vec::new();
    let mut signs = Vec::new();
    for sub in self.simplex_set().boundary() {
      let sign = sub.sign;
      let idx = self
        .complex
        .skeleton(self.dim() - 1)
        .get_by_simplex(&sub.into_simplex())
        .kidx();
      idxs.push(idx);
      signs.push(sign);
    }
    SparseSignChain::new(self.dim() - 1, idxs, signs)
  }

  pub fn cocells(&self) -> impl Iterator<Item = SimplexHandle> + '_ {
    self
      .simplex_data()
      .cocells
      .iter()
      .map(|&cell_idx| cell_idx.handle(self.complex))
  }

  pub fn edges(&self) -> impl Iterator<Item = SimplexHandle> {
    // TODO: optimize
    self.subsimps(1)
  }

  pub fn vertices(&self) -> impl Iterator<Item = SimplexHandle> {
    self
      .simplex_set()
      .vertices
      .iter()
      .map(|v| SimplexIdx::new(0, v).handle(self.complex))
  }

  /// The dim-subsimplicies of this simplex.
  ///
  /// These are ordered lexicographically w.r.t.
  /// the local vertex indices.
  /// e.g. tet.descendants(1) = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
  pub fn subsimps(&self, dim_sub: Dim) -> impl Iterator<Item = SimplexHandle> {
    self
      .simplex_set()
      .subsimps(dim_sub)
      .map(move |sub| self.complex.skeleton(dim_sub).get_by_simplex(&sub))
  }

  /// The dim-supersimplicies of this simplex.
  ///
  /// These are ordered first by cell index and then
  /// by lexicographically w.r.t. the local vertex indices.
  pub fn sups(&self, dim_sup: Dim) -> Vec<SimplexHandle> {
    self
      .cocells()
      .flat_map(|parent| {
        self
          .simplex_set()
          .supsimps(dim_sup, parent.simplex_set())
          .map(move |sup| self.complex.skeleton(dim_sup).get_by_simplex(&sup))
      })
      .collect()
  }
}

impl PartialEq for SimplexHandle<'_> {
  fn eq(&self, other: &Self) -> bool {
    std::ptr::eq(self.complex, other.complex) && self.idx == other.idx
  }
}
impl Eq for SimplexHandle<'_> {}

impl std::hash::Hash for SimplexHandle<'_> {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    (self.complex as *const Complex).hash(state);
    self.idx.hash(state);
  }
}

pub struct SkeletonHandle<'m> {
  complex: &'m Complex,
  dim: Dim,
}

impl<'m> SkeletonHandle<'m> {
  pub fn new(complex: &'m Complex, dim: Dim) -> Self {
    assert!(dim <= complex.dim());
    Self { complex, dim }
  }

  pub fn dim(&self) -> Dim {
    self.dim
  }

  pub fn raw(&self) -> &ComplexSkeleton {
    &self.complex.skeletons[self.dim()]
  }

  pub fn len(&self) -> usize {
    self.raw().len()
  }
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  pub fn get_by_kidx(&self, idx: KSimplexIdx) -> SimplexHandle<'m> {
    SimplexIdx::new(self.dim, idx).handle(self.complex)
  }
  pub fn get_by_simplex(&self, key: &SortedSimplex) -> SimplexHandle<'m> {
    let idx = self.raw().get_full(key).unwrap().0;
    SimplexIdx::new(self.dim, idx).handle(self.complex)
  }

  pub fn handle_iter(&self) -> impl ExactSizeIterator<Item = SimplexHandle<'m>> + '_ {
    (0..self.len()).map(|idx| SimplexIdx::new(self.dim, idx).handle(self.complex))
  }
  pub fn set_iter(&self) -> impl ExactSizeIterator<Item = &Simplex<CanonicalOrder>> + '_ {
    self.handle_iter().map(|s| s.simplex_set())
  }
}
