use crate::{
  simplex::{SimplexExt, SortedSimplex, SortedSimplexExt},
  Dim,
};

use super::{
  attribute::SparseSignChain,
  dim::{ConstCodim, ConstDim, DimInfoProvider},
  local::LocalComplex,
  ComplexSkeleton, ManifoldComplex, SimplexData,
};

pub type VertexDim = ConstDim<0>;
pub type EdgeDim = ConstDim<1>;
pub type FaceCodim = ConstCodim<1>;
pub type FacetCodim = ConstCodim<0>;

pub type VertexIdx = SimplexIdx<VertexDim>;
pub type EdgeIdx = SimplexIdx<EdgeDim>;
pub type FaceIdx = SimplexIdx<FaceCodim>;
pub type FacetIdx = SimplexIdx<FacetCodim>;

pub type VertexHandle<'c> = SimplexHandle<'c, VertexDim>;
pub type EdgeHandle<'c> = SimplexHandle<'c, EdgeDim>;
pub type FaceHandle<'c> = SimplexHandle<'c, FaceCodim>;
pub type FacetHandle<'c> = SimplexHandle<'c, FacetCodim>;

pub type VerticesHandle<'c> = SkeletonHandle<'c, VertexDim>;
pub type EdgesHandle<'c> = SkeletonHandle<'c, EdgeDim>;
pub type FacesHandle<'c> = SkeletonHandle<'c, FaceCodim>;
pub type FacetsHandle<'c> = SkeletonHandle<'c, FacetCodim>;

pub type KSimplexIdx = usize;

impl ManifoldComplex {
  pub fn skeletons(&self) -> impl Iterator<Item = SkeletonHandle<Dim>> {
    (0..=self.dim()).map(|d| SkeletonHandle::new(self, d))
  }
  pub fn skeleton<D: DimInfoProvider>(&self, dim: D) -> SkeletonHandle<D> {
    SkeletonHandle::new(self, dim)
  }
  pub fn nsimplicies<D: DimInfoProvider>(&self, dim: D) -> usize {
    self.skeleton(dim).len()
  }
  pub fn vertices(&self) -> VerticesHandle {
    self.skeleton(ConstDim)
  }
  pub fn edges(&self) -> EdgesHandle {
    self.skeleton(ConstDim)
  }
  pub fn faces(&self) -> FacesHandle {
    self.skeleton(ConstCodim)
  }
  pub fn facets(&self) -> FacetsHandle {
    self.skeleton(ConstCodim)
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimplexIdx<D: DimInfoProvider> {
  pub dim: D,
  pub kidx: KSimplexIdx,
}
impl<D: DimInfoProvider> From<(D, KSimplexIdx)> for SimplexIdx<D> {
  fn from((dim, kidx): (D, KSimplexIdx)) -> Self {
    Self { dim, kidx }
  }
}
impl<D: DimInfoProvider> SimplexIdx<D> {
  pub fn new(dim: D, kidx: KSimplexIdx) -> Self {
    Self { dim, kidx }
  }
  pub fn is_valid(self, complex: &ManifoldComplex) -> bool {
    self.dim.is_valid(complex.dim()) && self.kidx < complex.skeleton(self.dim).len()
  }
  pub fn assert_valid(self, mesh: &ManifoldComplex) {
    assert!(self.is_valid(mesh), "Not a valid simplex index.");
  }

  pub fn handle(self, complex: &ManifoldComplex) -> SimplexHandle<D> {
    SimplexHandle::new(complex, self)
  }
}

#[derive(Copy, Clone)]
pub struct SimplexHandle<'c, D: DimInfoProvider> {
  complex: &'c ManifoldComplex,
  idx: SimplexIdx<D>,
}
impl<'m, D: DimInfoProvider> std::fmt::Debug for SimplexHandle<'m, D> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("SimplexHandle")
      .field("idx", &self.idx)
      .field("complex", &(self.complex as *const ManifoldComplex))
      .finish()
  }
}

impl<'m, D: DimInfoProvider> SimplexHandle<'m, D> {
  pub fn new(complex: &'m ManifoldComplex, idx: SimplexIdx<D>) -> Self {
    idx.assert_valid(complex);
    Self { complex, idx }
  }

  pub fn idx(&self) -> SimplexIdx<D> {
    self.idx
  }
  pub fn dim_info(&self) -> D {
    self.idx.dim
  }
  pub fn dim(&self) -> Dim {
    self.idx.dim.dim(self.complex.dim())
  }
  pub fn kidx(&self) -> KSimplexIdx {
    self.idx.kidx
  }

  pub fn complex(&self) -> &'m ManifoldComplex {
    self.complex
  }
  pub fn skeleton(&self) -> SkeletonHandle<'m, D> {
    self.complex.skeleton(self.dim_info())
  }
  pub fn simplex_data(&self) -> &SimplexData {
    self.complex.skeletons[self.dim()]
      .get_index(self.kidx())
      .unwrap()
      .1
  }
  pub fn nvertices(&self) -> usize {
    self.simplex_set().len()
  }
  pub fn simplex_set(&self) -> &'m SortedSimplex {
    self.complex.skeletons[self.dim()]
      .get_index(self.kidx())
      .unwrap()
      .0
  }

  pub fn anti_boundary(&self) -> SparseSignChain<Dim> {
    let mut idxs = Vec::new();
    let mut signs = Vec::new();
    for parent in self.parent_facets() {
      for sup in self.simplex_set().anti_boundary(parent.simplex_set()) {
        let sign = sup.sign;
        let idx = self
          .complex
          .skeleton(self.dim() + 1)
          .get_by_simplex(&sup.set)
          .kidx();

        idxs.push(idx);
        signs.push(sign);
      }
    }
    SparseSignChain::new(self.dim() + 1, idxs, signs)
  }

  pub fn boundary_chain(&self) -> SparseSignChain<Dim> {
    let mut idxs = Vec::new();
    let mut signs = Vec::new();
    for sub in self.simplex_set().boundary() {
      let sign = sub.sign;
      let idx = self
        .complex
        .skeleton(self.dim() - 1)
        .get_by_simplex(&sub.set)
        .kidx();
      idxs.push(idx);
      signs.push(sign);
    }
    SparseSignChain::new(self.dim() - 1, idxs, signs)
  }

  pub fn parent_facets(&self) -> impl Iterator<Item = FacetHandle> + '_ {
    self
      .simplex_data()
      .parent_facets
      .iter()
      .map(|&cell_idx| SimplexIdx::new(ConstCodim, cell_idx).handle(self.complex))
  }

  pub fn edges(&self) -> impl Iterator<Item = EdgeHandle> {
    // TODO: optimize
    self.subsimps(ConstDim)
  }

  pub fn vertices(&self) -> impl Iterator<Item = VertexHandle> {
    self
      .simplex_set()
      .iter()
      .map(|v| VertexIdx::new(ConstDim, v).handle(self.complex))
  }

  /// The dim-subsimplicies of this simplex.
  ///
  /// These are ordered lexicographically w.r.t.
  /// the local vertex indices.
  /// e.g. tet.descendants(1) = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
  pub fn subsimps<DSub: DimInfoProvider>(
    &self,
    dim_sub: DSub,
  ) -> impl Iterator<Item = SimplexHandle<DSub>> {
    let dim_sub_dyn = dim_sub.dim(self.complex().dim());
    self
      .simplex_set()
      .subsimps(dim_sub_dyn)
      .map(move |sub| self.complex.skeleton(dim_sub).get_by_simplex(&sub))
  }

  /// The dim-supersimplicies of this simplex.
  ///
  /// These are ordered first by cell index and then
  /// by lexicographically w.r.t. the local vertex indices.
  pub fn sups<DSup: DimInfoProvider>(&self, dim_sup: DSup) -> Vec<SimplexHandle<DSup>> {
    let dim_sup_dyn = dim_sup.dim(self.complex().dim());
    self
      .parent_facets()
      .flat_map(|parent| {
        self
          .simplex_set()
          .supsimps(dim_sup_dyn, parent.simplex_set())
          .map(move |sup| self.complex.skeleton(dim_sup).get_by_simplex(&sup))
      })
      .collect()
  }
}

impl<D: DimInfoProvider> PartialEq for SimplexHandle<'_, D> {
  fn eq(&self, other: &Self) -> bool {
    std::ptr::eq(self.complex, other.complex) && self.idx == other.idx
  }
}
impl<D: DimInfoProvider> Eq for SimplexHandle<'_, D> {}

impl<D: DimInfoProvider> std::hash::Hash for SimplexHandle<'_, D> {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    (self.complex as *const ManifoldComplex).hash(state);
    self.idx.hash(state);
  }
}

impl<'m> FacetHandle<'m> {
  pub fn to_local_complex(&self) -> LocalComplex {
    LocalComplex::from_facet(*self)
  }
}

pub struct SkeletonHandle<'m, D: DimInfoProvider> {
  complex: &'m ManifoldComplex,
  dim: D,
}

impl<'m, D: DimInfoProvider> SkeletonHandle<'m, D> {
  pub fn new(complex: &'m ManifoldComplex, dim: D) -> Self {
    dim.assert_valid(complex.dim());
    Self { complex, dim }
  }

  pub fn dim_info(&self) -> D {
    self.dim
  }
  pub fn dim(&self) -> Dim {
    self.dim.dim(self.complex.dim())
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

  pub fn get_by_kidx(&self, idx: KSimplexIdx) -> SimplexHandle<'m, D> {
    SimplexIdx::new(self.dim, idx).handle(self.complex)
  }
  pub fn get_by_simplex(&self, key: &SortedSimplex) -> SimplexHandle<'m, D> {
    let idx = self.raw().get_full(key).unwrap().0;
    SimplexIdx::new(self.dim, idx).handle(self.complex)
  }

  pub fn iter(&self) -> impl ExactSizeIterator<Item = SimplexHandle<'m, D>> + '_ {
    (0..self.len()).map(|idx| SimplexIdx::new(self.dim, idx).handle(self.complex))
  }
}
