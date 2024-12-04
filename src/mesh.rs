//! Simplicial Manifold Datastructure for working with Topology and Geometry.
//!
//! - Container for mesh entities (Simplicies).
//! - Global numbering for unique identification of the entities.
//! - Entity Iteration
//! - Topological Information (Incidence)
//! - Geometrical information (Lengths, Volumes)

pub mod boundary;
pub mod coordinates;
pub mod dim3;
pub mod gmsh;
pub mod hyperbox;
pub mod raw;

use crate::{
  cell::{Length, StandaloneCell},
  combinatorics::{CanonicalVertplex, OrderedVertplex, OrientedVertplex, Sign},
  Dim, VertexIdx,
};

use indexmap::IndexMap;
use std::hash::Hash;

/// A simplicial manifold with both topological and geometric information.
#[derive(Debug)]
pub struct SimplicialManifold {
  cells: Vec<OrientedVertplex>,
  skeletons: Vec<Skeleton>,

  /// mapping [`EdgeIdx`] -> [`Length`]
  edge_lengths: Vec<Length>,
}

// getters
impl SimplicialManifold {
  pub fn dim(&self) -> Dim {
    self.skeletons.len() - 1
  }
  pub fn nnodes(&self) -> usize {
    self.nvertices()
  }

  pub fn skeleton(&self, dim: Dim) -> SkeletonHandle {
    SkeletonHandle::new(self, dim)
  }

  pub fn cells(&self) -> SkeletonHandle {
    self.skeleton(self.dim())
  }
  pub fn ncells(&self) -> usize {
    self.cells().len()
  }
  pub fn vertices(&self) -> SkeletonHandle {
    self.skeleton(0)
  }
  pub fn nvertices(&self) -> usize {
    self.vertices().len()
  }
  pub fn edges(&self) -> SkeletonHandle {
    self.skeleton(1)
  }
  pub fn nedges(&self) -> usize {
    self.edges().len()
  }
  pub fn facets(&self) -> SkeletonHandle {
    self.skeleton(self.dim() - 1)
  }
  pub fn nfacets(&self) -> usize {
    self.facets().len()
  }

  /// The mesh width $h$, which is the largest diameter of all cells.
  pub fn mesh_width(&self) -> f64 {
    self
      .edge_lengths
      .iter()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .copied()
      .unwrap()
  }

  /// The shape regularity measure $rho$ of the whole mesh, which is the largest
  /// shape regularity measure over all cells.
  pub fn shape_regularity_measure(&self) -> f64 {
    self
      .cells()
      .iter()
      .map(|cell| cell.as_standalone_cell().shape_reguarity_measure())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }
}

/// A container for topological simplicies of common dimension.
pub type Skeleton = IndexMap<CanonicalVertplex, SimplexData>;

/// Topological information of the simplex.
#[derive(Debug, Clone)]
pub struct SimplexData {
  /// The cells that this simplex is part of.
  /// This information is crucial for computing ancestor simplicies.
  /// Ordered increasing in [`CellIdx`].
  parent_cells: Vec<CellIdx>,
}

impl SimplexData {
  pub fn stub() -> Self {
    let parent_cells = Vec::new();
    Self { parent_cells }
  }
}

/// Fat pointer to simplex.
pub struct SimplexHandle<'m> {
  idx: SimplexIdx,
  mesh: &'m SimplicialManifold,
}
impl<'m> SimplexHandle<'m> {
  pub fn new(mesh: &'m SimplicialManifold, idx: impl Into<SimplexIdx>) -> Self {
    let idx = idx.into();
    idx.assert_valid(mesh);
    Self { mesh, idx }
  }

  pub fn idx(&self) -> SimplexIdx {
    self.idx
  }
  pub fn dim(&self) -> Dim {
    self.idx.dim
  }
  pub fn kidx(&self) -> KSimplexIdx {
    self.idx.kidx
  }

  pub fn mesh(&self) -> &'m SimplicialManifold {
    self.mesh
  }
  pub fn skeleton(&self) -> SkeletonHandle<'m> {
    self.mesh.skeleton(self.dim())
  }
  pub fn simplex_data(&self) -> &SimplexData {
    self.mesh.skeletons[self.dim()]
      .get_index(self.kidx())
      .unwrap()
      .1
  }

  pub fn is_cell(&self) -> bool {
    self.dim() == self.mesh.dim()
  }

  pub fn nvertices(&self) -> usize {
    self.canonical_vertplex().nvertices()
  }
  pub fn canonical_vertplex(&self) -> &'m CanonicalVertplex {
    self.mesh.skeletons[self.dim()]
      .get_index(self.kidx())
      .unwrap()
      .0
  }
  pub fn ordered_vertplex(&self) -> &'m OrderedVertplex {
    assert!(self.is_cell(), "Only Cells are ordered.");
    self.oriented_vertplex().as_ordered()
  }
  pub fn oriented_vertplex(&self) -> &'m OrientedVertplex {
    assert!(self.is_cell(), "Only Cells are oriented.");
    &self.mesh.cells[self.kidx()]
  }

  pub fn antiboundary(&self) -> SparseChain<'m> {
    let mut idxs = Vec::new();
    let mut coeffs = Vec::new();
    for (isup, sup) in self.sups(self.dim() + 1).enumerate() {
      idxs.push(sup.kidx());
      // TODO: check this orientation
      coeffs.push(Sign::from_parity(self.nvertices() - 1 - isup).as_i32());
    }
    SparseChain::new(self.mesh, self.dim() - 1, idxs, coeffs)
  }

  pub fn boundary(&self) -> SparseChain<'m> {
    let mut idxs = Vec::new();
    let mut coeffs = Vec::new();
    for (isub, sub) in self.subs(self.dim() - 1).enumerate() {
      idxs.push(sub.kidx());
      coeffs.push(Sign::from_parity(self.nvertices() - 1 - isub).as_i32());
    }
    SparseChain::new(self.mesh, self.dim() - 1, idxs, coeffs)
  }

  pub fn parent_cells(&self) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    self
      .simplex_data()
      .parent_cells
      .iter()
      .map(|&cell_kidx| SimplexHandle::new(self.mesh, (self.mesh.dim(), cell_kidx)))
  }
  pub fn edges(&self) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    self.subs(1)
  }
  fn edge_lengths(&self) -> Vec<f64> {
    self
      .edges()
      .map(|e| self.mesh.edge_lengths[e.idx.kidx])
      .collect()
  }
  pub fn as_standalone_cell(&self) -> StandaloneCell {
    assert!(self.is_cell(), "Simplex is not a cell.");

    let faces = (0..=self.dim())
      .map(|k| self.subs(k).map(|s| s.kidx()).collect())
      .collect();
    let orientation = self.oriented_vertplex().superimposed_orient();
    let edge_lengths = self.edge_lengths();
    StandaloneCell::new(faces, orientation, edge_lengths)
  }

  /// The dim-subsimplicies of this simplex.
  ///
  /// These are ordered lexicographically w.r.t.
  /// the local vertex indices.
  /// e.g. tet.descendants(1) = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
  pub fn subs(&self, dim: Dim) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    self
      .canonical_vertplex()
      .subs(dim)
      .into_iter()
      .map(move |sub| self.mesh.skeleton(dim).get_key(&sub))
  }

  /// The dim-supersimplicies of this simplex.
  ///
  /// These are ordered first by cell index and then
  /// by lexicographically w.r.t. the local vertex indices.
  pub fn sups(&self, dim: Dim) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    self
      .canonical_vertplex()
      .sups(
        dim,
        self.parent_cells().map(move |c| c.canonical_vertplex()),
      )
      .into_iter()
      .map(move |a| self.mesh.skeleton(dim).get_key(&a).idx)
      .map(move |a| Self::new(self.mesh, a))
  }
}

impl PartialEq for SimplexHandle<'_> {
  fn eq(&self, other: &Self) -> bool {
    std::ptr::eq(self.mesh, other.mesh) && self.idx == other.idx
  }
}
impl Eq for SimplexHandle<'_> {}

impl Hash for SimplexHandle<'_> {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    (self.mesh as *const SimplicialManifold).hash(state);
    self.idx.hash(state);
  }
}

pub struct SkeletonHandle<'m> {
  mesh: &'m SimplicialManifold,
  dim: Dim,
}

impl<'m> SkeletonHandle<'m> {
  pub fn new(mesh: &'m SimplicialManifold, dim: Dim) -> Self {
    assert!(dim <= mesh.dim(), "Invalid Skeleton Dimension");
    Self { mesh, dim }
  }

  pub fn raw(&self) -> &Skeleton {
    &self.mesh.skeletons[self.dim]
  }

  pub fn len(&self) -> usize {
    self.raw().len()
  }
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  pub fn get_kidx(&self, idx: KSimplexIdx) -> SimplexHandle<'m> {
    SimplexHandle::new(self.mesh, (self.dim, idx))
  }
  pub fn get_key(&self, key: &CanonicalVertplex) -> SimplexHandle<'m> {
    let idx = self.raw().get_full(key).unwrap().0;
    SimplexHandle::new(self.mesh, (self.dim, idx))
  }

  pub fn iter(&self) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    (0..self.len()).map(|idx| SimplexHandle::new(self.mesh, (self.dim, idx)))
  }
}

pub struct SparseChain<'m> {
  mesh: &'m SimplicialManifold,
  dim: Dim,
  idxs: Vec<KSimplexIdx>,
  coeffs: Vec<i32>,
}
impl<'m> SparseChain<'m> {
  fn new(mesh: &'m SimplicialManifold, dim: Dim, idxs: Vec<KSimplexIdx>, coeffs: Vec<i32>) -> Self {
    Self {
      mesh,
      dim,
      idxs,
      coeffs,
    }
  }
  pub fn iter(&self) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    self
      .idxs
      .iter()
      .map(|&idx| SimplexHandle::new(self.mesh, (self.dim, idx)))
  }

  pub fn coeffs(&self) -> &[i32] {
    &self.coeffs
  }
  pub fn len(&self) -> usize {
    self.idxs.len()
  }
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}

/// A simplicial k-cochain is a function assigning a number to each k-simplex of
/// a simplicial complex.
///
/// Whitney forms are isomorphic to simplicial cochains.
///
/// De Rham map: Differential k-form integrated over all k-simplicies gives
/// k-cochain, isomorphism on cohomology.
///
/// Whitney Interpolation: Inverse of de Rham map. k-cochain to differential k-form.
/// This induces a strong connection between FDM/DEC and FEEC.
#[allow(dead_code)]
pub struct SparseCochain<'m> {
  mesh: &'m SimplicialManifold,
  dim: Dim,
  idxs: Vec<KSimplexIdx>,
  coeffs: Vec<i32>,
}

pub type EdgeIdx = usize;
pub type CellIdx = usize;
pub type KSimplexIdx = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimplexIdx {
  dim: Dim,
  kidx: KSimplexIdx,
}
impl From<(Dim, KSimplexIdx)> for SimplexIdx {
  fn from((dim, kidx): (Dim, KSimplexIdx)) -> Self {
    Self { dim, kidx }
  }
}
impl SimplexIdx {
  pub fn is_valid(self, mesh: &SimplicialManifold) -> bool {
    self.dim <= mesh.dim() && self.kidx < mesh.skeleton(self.dim).len()
  }
  pub fn assert_valid(self, mesh: &SimplicialManifold) {
    assert!(self.is_valid(mesh), "Not a valid simplex index.");
  }
}

#[cfg(test)]
mod test {

  use crate::{
    cell::ReferenceCell,
    combinatorics::{nsubsimplicies, CanonicalVertplex},
  };

  #[test]
  fn incidence() {
    let dim = 3;
    let mesh = ReferenceCell::new(dim).to_singleton_mesh();

    let cell = mesh.cells().get_kidx(0);

    let cell_vertices = CanonicalVertplex::new_unchecked((0..(dim + 1)).collect());
    assert_eq!(cell.canonical_vertplex(), &cell_vertices);

    for dim_sub in 0..=dim {
      let skeleton = mesh.skeleton(dim_sub);
      for simp in skeleton.iter() {
        let simp_vertices = simp.canonical_vertplex();
        print!("{simp_vertices:?},");
      }
      println!();
    }

    for dim_sub in 0..=dim {
      let subs: Vec<_> = cell.subs(dim_sub).collect();
      assert_eq!(subs.len(), nsubsimplicies(dim, dim_sub));
      let subs_vertices: Vec<_> = cell_vertices.subs(dim_sub);
      assert_eq!(
        subs
          .iter()
          .map(|sub| sub.canonical_vertplex().clone())
          .collect::<Vec<_>>(),
        subs_vertices
      );

      for (isub, sub) in subs.iter().enumerate() {
        let sub_vertices = &subs_vertices[isub];
        for dim_sup in dim_sub..dim {
          let sups: Vec<_> = sub.sups(dim_sup).collect();
          let sups_vertices = sups
            .iter()
            .map(|sub| sub.canonical_vertplex().clone())
            .collect::<Vec<_>>();
          sups_vertices
            .iter()
            .all(|sup| sub_vertices <= sup && sup <= &cell_vertices);
        }
      }
    }
  }
}
