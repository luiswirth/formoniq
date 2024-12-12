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
  geometry::EdgeLengths,
  simplicial::{CellComplex, OrientedVertplex, SortedVertplex},
  Dim, VertexIdx,
};

use indexmap::IndexMap;
use std::hash::Hash;

/// A simplicial manifold with both topological and geometric information.
#[derive(Debug)]
pub struct SimplicialManifold {
  cells: Vec<OrientedVertplex>,
  skeletons: Vec<Skeleton>,
  edge_lengths: EdgeLengths,
}

/// A container for simplicies of common dimension.
pub type Skeleton = IndexMap<SortedVertplex, SimplexData>;

// getters
impl SimplicialManifold {
  pub fn dim(&self) -> Dim {
    self.skeletons.len() - 1
  }
  pub fn skeleton(&self, dim: Dim) -> SkeletonHandle {
    SkeletonHandle::new(self, dim)
  }

  pub fn cells(&self) -> impl Iterator<Item = CellHandle> {
    (0..self.ncells()).map(|icell| CellHandle::new(self, icell))
  }
  pub fn ncells(&self) -> usize {
    self.cells.len()
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
  pub fn faces(&self) -> SkeletonHandle {
    self.skeleton(self.dim() - 1)
  }
  pub fn nfaces(&self) -> usize {
    self.faces().len()
  }

  /// The mesh width $h_max$, equal to the largest diameter of all cells.
  pub fn mesh_width_max(&self) -> f64 {
    self
      .edge_lengths
      .iter()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .copied()
      .unwrap()
  }

  /// By convexity the smallest length of a line inside a simplex is the length
  /// one of the edges.
  pub fn mesh_width_min(&self) -> f64 {
    self
      .edge_lengths
      .iter()
      .min_by(|a, b| a.partial_cmp(b).unwrap())
      .copied()
      .unwrap()
  }

  /// The shape regularity measure $rho$ of the whole mesh, which is the largest
  /// shape regularity measure over all cells.
  pub fn shape_regularity_measure(&self) -> f64 {
    self
      .cells()
      .map(|cell| cell.as_cell_complex().shape_reguarity_measure())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }
}

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
pub struct CellHandle<'m> {
  idx: CellIdx,
  mesh: &'m SimplicialManifold,
}
impl<'m> CellHandle<'m> {
  pub fn new(mesh: &'m SimplicialManifold, idx: CellIdx) -> Self {
    Self { mesh, idx }
  }
  pub fn dim(&self) -> Dim {
    self.mesh.dim()
  }
  pub fn nvertices(&self) -> usize {
    self.dim() + 1
  }

  pub fn as_simplex(&self) -> SimplexHandle {
    SimplexHandle::new(self.mesh, (self.mesh.dim(), self.idx))
  }

  pub fn oriented_vertplex(&self) -> &'m OrientedVertplex {
    &self.mesh.cells[self.idx]
  }

  pub fn sorted_vertplex(&self) -> SortedVertplex {
    self.mesh.cells[self.idx]
      .clone()
      .into_sorted()
      .forget_sign()
  }

  pub fn as_cell_complex(&self) -> CellComplex {
    let combinatorial = self.oriented_vertplex();
    let simplex = self.as_simplex();
    let faces = (0..=self.dim())
      .map(|k| simplex.subs(k).map(|s| s.kidx()).collect())
      .collect();
    let sign = combinatorial.sign();
    let edge_lengths = simplex.edge_lengths();
    CellComplex::new(faces, sign, edge_lengths)
  }
}

/// Fat pointer to simplex.
pub struct SimplexHandle<'m> {
  idx: SimplexIdx,
  mesh: &'m SimplicialManifold,
}
impl<'m> std::fmt::Debug for SimplexHandle<'m> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("SimplexHandle")
      .field("idx", &self.idx)
      .field("mesh", &(self.mesh as *const SimplicialManifold))
      .finish()
  }
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

  pub fn try_as_cell(&self) -> Option<CellHandle> {
    if self.dim() == self.mesh.dim() {
      Some(CellHandle::new(self.mesh, self.kidx()))
    } else {
      None
    }
  }

  pub fn nvertices(&self) -> usize {
    self.sorted_vertplex().len()
  }
  pub fn sorted_vertplex(&self) -> &'m SortedVertplex {
    self.mesh.skeletons[self.dim()]
      .get_index(self.kidx())
      .unwrap()
      .0
  }

  pub fn anti_boundary(&self) -> SparseChain<'_> {
    let mut idxs = Vec::new();
    let mut coeffs = Vec::new();
    for parent_cell in self.parent_cells() {
      for sup in self
        .sorted_vertplex()
        .clone()
        .with_global_base(parent_cell.sorted_vertplex().into_global_base())
        .anti_boundary()
      {
        let coeff = sup.sign().as_i32();
        let idx = self
          .mesh
          .skeleton(self.dim() + 1)
          .get_by_vertplex(&sup.forget_sign().forget_base())
          .kidx();

        idxs.push(idx);
        coeffs.push(coeff);
      }
    }
    SparseChain::new(self.mesh, self.dim() + 1, idxs, coeffs)
  }

  pub fn boundary_chain(&self) -> SparseChain<'m> {
    let mut idxs = Vec::new();
    let mut coeffs = Vec::new();
    for sub in self.sorted_vertplex().boundary() {
      let coeff = sub.sign().as_i32();
      let idx = self
        .mesh
        .skeleton(self.dim() - 1)
        .get_by_vertplex(&sub.forget_sign())
        .kidx();
      idxs.push(idx);
      coeffs.push(coeff);
    }
    SparseChain::new(self.mesh, self.dim() - 1, idxs, coeffs)
  }

  pub fn parent_cells(&self) -> impl Iterator<Item = CellHandle<'m>> + '_ {
    self
      .simplex_data()
      .parent_cells
      .iter()
      .map(|&cell_idx| CellHandle::new(self.mesh, cell_idx))
  }
  pub fn edges(&self) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    self.subs(1)
  }
  fn edge_lengths(&self) -> Vec<f64> {
    self
      .edges()
      .map(|e| self.mesh.edge_lengths.length(e.kidx()))
      .collect()
  }

  /// The dim-subsimplicies of this simplex.
  ///
  /// These are ordered lexicographically w.r.t.
  /// the local vertex indices.
  /// e.g. tet.descendants(1) = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
  pub fn subs(&self, dim: Dim) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    self
      .sorted_vertplex()
      .subs(dim + 1)
      .map(move |sub| self.mesh.skeleton(dim).get_by_vertplex(&sub))
  }

  /// The dim-supersimplicies of this simplex.
  ///
  /// These are ordered first by cell index and then
  /// by lexicographically w.r.t. the local vertex indices.
  pub fn sups(&self, dim: Dim) -> Vec<SimplexHandle> {
    self
      .parent_cells()
      .flat_map(|parent_cell| {
        self
          .sorted_vertplex()
          .clone()
          .with_global_base(parent_cell.oriented_vertplex().clone().into_global_base())
          .sups(dim + 1)
          .map(move |a| {
            self
              .mesh
              .skeleton(dim)
              .get_by_vertplex(&a.forget_base())
              .idx
          })
          .map(move |a| Self::new(self.mesh, a))
      })
      .collect()
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

  pub fn get_by_kidx(&self, idx: KSimplexIdx) -> SimplexHandle<'m> {
    SimplexHandle::new(self.mesh, (self.dim, idx))
  }
  pub fn get_by_vertplex(&self, key: &SortedVertplex) -> SimplexHandle<'m> {
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
    combo::Sign,
    simplicial::{nsubsimplicies, ReferenceCell, Vertplex},
  };

  #[test]
  fn incidence() {
    let dim = 3;
    let mesh = ReferenceCell::new(dim).to_singleton_mesh();
    let cell = mesh.cells().next().unwrap();

    // print
    for dim_sub in 0..=dim {
      let skeleton = mesh.skeleton(dim_sub);
      for simp in skeleton.iter() {
        let simp_vertices = simp.sorted_vertplex();
        print!("{simp_vertices:?},");
      }
      println!();
    }

    let cell_vertplex = Vertplex::counting(dim + 1).with_sign(Sign::Pos);
    for dim_sub in 0..=dim {
      let subs: Vec<_> = cell.as_simplex().subs(dim_sub).collect();
      assert_eq!(subs.len(), nsubsimplicies(dim, dim_sub));
      let subs_vertices: Vec<_> = cell_vertplex.subs(dim_sub + 1).collect();
      assert_eq!(
        subs
          .iter()
          .map(|sub| sub.sorted_vertplex().clone())
          .collect::<Vec<_>>(),
        subs_vertices
      );

      for (isub, sub) in subs.iter().enumerate() {
        let sub_vertices = &subs_vertices[isub];
        for dim_sup in dim_sub..dim {
          let sups: Vec<_> = sub.sups(dim_sup);
          let sups_vertices = sups
            .iter()
            .map(|sub| sub.sorted_vertplex().clone())
            .collect::<Vec<_>>();
          sups_vertices
            .iter()
            .all(|sup| sub_vertices.is_sub_of(sup) && sup.is_sub_of(&cell_vertplex));
        }
      }
    }
  }
}
