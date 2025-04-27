use super::complex::{Complex, ComplexSkeleton, SimplexComplexData};
use crate::{
  topology::{simplex::Simplex, skeleton::Skeleton},
  Dim,
};

use common::combo::Sign;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// An index identifying a simplex in a skeleton.
pub type KSimplexIdx = usize;

/// An index identifying a simplex in the mesh.
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

/// A handle to a simplex in the mesh.
#[derive(Copy, Clone)]
pub struct SimplexHandle<'c> {
  complex: &'c Complex,
  idx: SimplexIdx,
}
impl std::ops::Deref for SimplexHandle<'_> {
  type Target = Simplex;
  fn deref(&self) -> &Self::Target {
    self.skeleton_raw().simplex_by_kidx(self.kidx())
  }
}
impl std::fmt::Debug for SimplexHandle<'_> {
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
  pub fn kidx(&self) -> KSimplexIdx {
    self.idx.kidx
  }
  pub fn complex(&self) -> &'m Complex {
    self.complex
  }

  pub fn skeleton(&self) -> SkeletonHandle<'m> {
    self.complex.skeleton(self.dim())
  }
  pub fn mesh_skeleton_raw(&self) -> &ComplexSkeleton {
    self.complex.mesh_skeleton_raw(self.idx.dim())
  }
  pub fn skeleton_raw(&self) -> &Skeleton {
    self.mesh_skeleton_raw().skeleton()
  }
  pub fn mesh_data(&self) -> &SimplexComplexData {
    &self.mesh_skeleton_raw().complex_data()[self.kidx()]
  }
}

impl SimplexHandle<'_> {
  pub fn boundary_chain(&self) -> impl Iterator<Item = (Sign, SimplexHandle)> {
    self.boundary().map(move |sub| {
      let sign = sub.sign;
      let handle = self
        .complex
        .skeleton(self.dim() - 1)
        .handle_by_simplex(&sub.simplex);
      (sign, handle)
    })
  }

  pub fn cocells(&self) -> impl Iterator<Item = SimplexHandle> + '_ {
    self
      .mesh_data()
      .cocells
      .iter()
      .map(|&cell_idx| cell_idx.handle(self.complex))
  }

  /// The dim-subsimplices of this simplex.
  ///
  /// These are ordered lexicographically w.r.t.
  /// the local vertex indices.
  /// e.g. tet.descendants(1) = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
  pub fn mesh_subsimps(&self, dim_sub: Dim) -> impl Iterator<Item = SimplexHandle> {
    self
      .subsequences(dim_sub)
      .map(move |sub| self.complex.skeleton(dim_sub).handle_by_simplex(&sub))
  }

  /// The dim-supersimplices of this simplex.
  ///
  /// These are ordered first by cell index and then
  /// by lexicographically w.r.t. the local vertex indices.
  pub fn mesh_supersimps(&self, dim_super: Dim) -> impl Iterator<Item = SimplexHandle> {
    self.cocells().flat_map(move |parent| {
      self
        .supersequences(dim_super, &parent)
        .map(move |sup| self.complex.skeleton(dim_super).handle_by_simplex(&sup))
    })
  }

  pub fn mesh_vertices(&self) -> impl Iterator<Item = SimplexHandle> {
    self
      .iter()
      .map(|v| SimplexIdx::new(0, v).handle(self.complex))
  }
  pub fn mesh_edges(&self) -> impl Iterator<Item = SimplexHandle> {
    self.mesh_subsimps(1)
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

/// A handle to a skeleton in the mesh.
pub struct SkeletonHandle<'m> {
  complex: &'m Complex,
  dim: Dim,
}
impl std::ops::Deref for SkeletonHandle<'_> {
  type Target = Skeleton;
  fn deref(&self) -> &Self::Target {
    self.complex.mesh_skeleton_raw(self.dim).skeleton()
  }
}

impl<'m> SkeletonHandle<'m> {
  pub fn new(complex: &'m Complex, dim: Dim) -> Self {
    assert!(dim <= complex.dim());
    Self { complex, dim }
  }
  pub fn handle_by_kidx(&self, idx: KSimplexIdx) -> SimplexHandle<'m> {
    SimplexIdx::new(self.dim, idx).handle(self.complex)
  }
  pub fn handle_by_simplex(&self, simp: &Simplex) -> SimplexHandle<'m> {
    let idx = self.kidx_by_simplex(simp);
    SimplexIdx::new(self.dim, idx).handle(self.complex)
  }
  pub fn handle_iter(&self) -> impl ExactSizeIterator<Item = SimplexHandle<'m>> + '_ {
    (0..self.len()).map(|idx| SimplexIdx::new(self.dim, idx).handle(self.complex))
  }
  pub fn handle_par_iter(&self) -> impl ParallelIterator<Item = SimplexHandle<'m>> + '_ {
    (0..self.len())
      .into_par_iter()
      .map(|idx| SimplexIdx::new(self.dim, idx).handle(self.complex))
  }
}
