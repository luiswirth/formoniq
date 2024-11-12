//! Simplicial Manifold Datastructure for working with Topology and Geometry.
//!
//! - Container for mesh entities (Simplicies).
//! - Global numbering for unique identification of the entities.
//! - Entity Iteration
//! - Topological Information (Incidence)
//! - Geometrical information (Lengths, Volumes)

pub mod boundary;
pub mod coordinates;
pub mod gmsh;
pub mod hyperbox;
pub mod raw;
pub mod util;

use crate::{
  combinatorics::SortedSimplex, geometry::GeometrySimplex, Dim, Length, Orientation, VertexIdx,
};

use indexmap::IndexMap;
use std::hash::Hash;

pub type EdgeIdx = usize;
pub type CellIdx = usize;
pub type KSimplexIdx = usize;
pub type SimplexIdx = (Dim, KSimplexIdx);

/// A simplicial manifold with both topological and geometric information.
#[derive(Debug)]
pub struct SimplicialManifold {
  topology: ManifoldTopology,
  geometry: ManifoldGeometry,
}

#[derive(Debug)]
pub struct ManifoldTopology {
  skeletons: Vec<SkeletonTopology>,
}

/// A container for topological simplicies of common dimension.
pub type SkeletonTopology = IndexMap<SortedSimplex, SimplexTopology>;

/// Topological information of the simplex.
#[derive(Debug, Clone)]
pub struct SimplexTopology {
  /// The vertices of the simplex.
  vertices: Vec<VertexIdx>,
  /// The cells that this simplex is part of.
  /// This information is crucial for computing ancestor simplicies.
  /// Ordered increasing in [`CellIdx`].
  cells: Vec<CellIdx>,
}

#[derive(Debug)]
pub struct ManifoldGeometry {
  /// mapping [`EdgeIdx`] -> [`Length`]
  edge_lengths: Vec<Length>,
}

impl ManifoldTopology {
  pub fn dim(&self) -> Dim {
    self.skeletons.len() - 1
  }
  pub fn nskeletons(&self) -> usize {
    self.skeletons().len()
  }
  pub fn skeletons(&self) -> &[SkeletonTopology] {
    &self.skeletons
  }
  pub fn skeleton(&self, d: Dim) -> &SkeletonTopology {
    &self.skeletons()[d]
  }
  pub fn simplex(&self, idx: SimplexIdx) -> &SimplexTopology {
    &self.skeleton(idx.0)[idx.1]
  }
  pub fn ncells(&self) -> usize {
    self.cells().len()
  }
  pub fn cells(&self) -> &SkeletonTopology {
    self.skeleton(self.dim())
  }
  pub fn cell(&self, idx: CellIdx) -> &SimplexTopology {
    &self.cells()[idx]
  }
  pub fn faces(&self) -> &SkeletonTopology {
    self.skeleton(self.dim() - 1)
  }
  pub fn face(&self, idx: KSimplexIdx) -> &SimplexTopology {
    &self.faces()[idx]
  }
  pub fn nedges(&self) -> usize {
    self.edges().len()
  }
  pub fn edges(&self) -> &SkeletonTopology {
    self.skeleton(1)
  }
  pub fn edge(&self, idx: EdgeIdx) -> &SimplexTopology {
    &self.edges()[idx]
  }
  pub fn nvertices(&self) -> usize {
    self.skeleton(0).len()
  }
}

// getters
impl SimplicialManifold {
  pub fn nvertices(&self) -> usize {
    self.skeleton(0).len()
  }
  pub fn dim(&self) -> Dim {
    self.topology.dim()
  }
  pub fn topology(&self) -> &ManifoldTopology {
    &self.topology
  }
  pub fn geometry(&self) -> &ManifoldGeometry {
    &self.geometry
  }
  pub fn skeleton(&self, dim: Dim) -> SkeletonHandle {
    SkeletonHandle::new(self, dim)
  }
  pub fn cells(&self) -> SkeletonHandle {
    self.skeleton(self.dim())
  }
  pub fn faces(&self) -> SkeletonHandle {
    self.skeleton(self.dim() - 1)
  }

  pub fn simplex(&self, idx: SimplexIdx) -> SimplexHandle {
    SimplexHandle::new(self, idx)
  }

  /// The mesh width $h$, which is the largest diameter of all cells.
  pub fn mesh_width(&self) -> f64 {
    self
      .cells()
      .iter()
      .map(|cell| cell.geometry().diameter())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  /// The shape regularity measure $rho$ of the whole mesh, which is the largest
  /// shape regularity measure over all cells.
  pub fn shape_regularity_measure(&self) -> f64 {
    self
      .cells()
      .iter()
      .map(|cell| cell.geometry().shape_reguarity_measure())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }
}

impl SimplexTopology {
  fn new(vertices: Vec<VertexIdx>, cells: Vec<CellIdx>) -> Self {
    Self { vertices, cells }
  }
}

/// Functionality methods.
impl SimplexTopology {
  pub fn dim(&self) -> Dim {
    self.vertices.len() - 1
  }
  pub fn nvertices(&self) -> usize {
    self.vertices.len()
  }
  pub fn vertices(&self) -> &[VertexIdx] {
    &self.vertices
  }
}
impl PartialEq for SimplexTopology {
  fn eq(&self, other: &Self) -> bool {
    self.vertices == other.vertices
  }
}
impl Eq for SimplexTopology {}

/// Fat pointer to simplex.
pub struct SimplexHandle<'m> {
  mesh: &'m SimplicialManifold,
  idx: SimplexIdx,
}
impl<'m> SimplexHandle<'m> {
  fn new(mesh: &'m SimplicialManifold, idx: SimplexIdx) -> Self {
    Self { mesh, idx }
  }
}

impl<'m> SimplexHandle<'m> {
  pub fn dim(&self) -> Dim {
    self.nvertices() - 1
  }
  pub fn mesh(&self) -> &'m SimplicialManifold {
    self.mesh
  }
  pub fn idx(&self) -> SimplexIdx {
    self.idx
  }
  pub fn kidx(&self) -> KSimplexIdx {
    self.idx.1
  }
  pub fn topology(&self) -> &'m SimplexTopology {
    &self.mesh.topology.skeletons[self.idx.0][self.idx.1]
  }
  pub fn vertices(&self) -> &'m [VertexIdx] {
    &self.topology().vertices
  }
  pub fn nvertices(&self) -> usize {
    self.vertices().len()
  }
  pub fn vertices_sorted(&self) -> &'m SortedSimplex {
    self.mesh.topology.skeletons[self.idx.0]
      .get_index(self.idx.1)
      .unwrap()
      .0
  }
  // TODO: check if this gives the right orientation
  pub fn antiboundary(&self) -> Chain<'m> {
    let mut idxs = Vec::new();
    let mut coeffs = Vec::new();
    for (isup, sup) in self.sups(self.dim() + 1).enumerate() {
      idxs.push(sup.kidx());
      coeffs.push(Orientation::from_permutation_parity(isup).as_i32());
    }
    Chain::new(self.mesh, self.dim() + 1, idxs, coeffs)
  }
  // TODO: check if this gives the right orientation
  pub fn boundary(&self) -> Chain<'m> {
    let mut idxs = Vec::new();
    let mut coeffs = Vec::new();
    for (isup, sup) in self.subs(self.dim() - 1).enumerate() {
      idxs.push(sup.kidx());
      coeffs.push(Orientation::from_permutation_parity(isup).as_i32());
    }
    Chain::new(self.mesh, self.dim() - 1, idxs, coeffs)
  }
  pub fn cells(&self) -> impl Iterator<Item = SimplexHandle<'m>> {
    self
      .topology()
      .cells
      .iter()
      .map(|&c| SimplexHandle::new(self.mesh, (self.mesh.dim(), c)))
  }
  pub fn edges(&self) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    self.subs(1)
  }
  fn edge_lengths(&self) -> Vec<f64> {
    self
      .edges()
      .map(|e| self.mesh.geometry.edge_lengths[e.idx.1])
      .collect()
  }
  pub fn geometry(&self) -> GeometrySimplex {
    // TODO: pass orientation correctly!!!
    GeometrySimplex::new(self.idx.0, self.edge_lengths(), Orientation::Pos)
  }

  /// The dim-subsimplicies of this simplex.
  ///
  /// These are ordered lexicographically w.r.t.
  /// the local vertex indices.
  /// e.g. tet.descendants(1) = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
  pub fn subs(&self, dim: Dim) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    self
      .vertices_sorted()
      .subs(dim)
      .map(move |sub| self.mesh.skeleton(dim).get_key(&sub))
  }

  /// The dim-supersimplicies of this simplex.
  ///
  /// These are ordered first by cell index and then
  /// by lexicographically w.r.t. the local vertex indices.
  pub fn sups(&self, dim: Dim) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    self
      .vertices_sorted()
      .sups(dim, self.cells().map(move |c| c.vertices_sorted()))
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
  fn new(mesh: &'m SimplicialManifold, dim: Dim) -> Self {
    Self { mesh, dim }
  }
}
impl SkeletonHandle<'_> {
  pub fn topology(&self) -> &SkeletonTopology {
    &self.mesh.topology.skeletons[self.dim]
  }
  pub fn len(&self) -> usize {
    self.topology().len()
  }
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}

impl<'m> SkeletonHandle<'m> {
  pub fn get_idx(&self, idx: KSimplexIdx) -> SimplexHandle<'m> {
    SimplexHandle::new(self.mesh, (self.dim, idx))
  }
  pub fn get_key(&self, key: &SortedSimplex) -> SimplexHandle<'m> {
    let idx = self.topology().get_full(key).unwrap().0;
    SimplexHandle::new(self.mesh, (self.dim, idx))
  }
}

impl<'m> SkeletonHandle<'m> {
  pub fn iter(&self) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    (0..self.mesh.topology.skeletons[self.dim].len())
      .map(|idx| SimplexHandle::new(self.mesh, (self.dim, idx)))
  }
}

pub struct Chain<'m> {
  mesh: &'m SimplicialManifold,
  dim: Dim,
  idxs: Vec<KSimplexIdx>,
  coeffs: Vec<i32>,
}
impl<'m> Chain<'m> {
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

#[cfg(test)]
mod test {

  use crate::{
    combinatorics::{nsubsimplicies, SortedSimplex},
    geometry::GeometrySimplex,
  };

  #[test]
  fn incidence() {
    let dim = 3;
    let mesh = GeometrySimplex::new_ref(dim).into_singleton_mesh();

    let cell = mesh.cells().get_idx(0);

    let cell_vertices = SortedSimplex::new_unchecked((0..(dim + 1)).collect());
    assert_eq!(cell.vertices_sorted(), &cell_vertices);

    for dim_sub in 0..=dim {
      let skeleton = mesh.skeleton(dim_sub);
      for simp in skeleton.iter() {
        let simp_vertices = simp.vertices();
        print!("{simp_vertices:?},");
      }
      println!();
    }

    for dim_sub in 0..=dim {
      let subs: Vec<_> = cell.subs(dim_sub).collect();
      assert_eq!(subs.len(), nsubsimplicies(dim, dim_sub));
      let subs_vertices: Vec<_> = cell_vertices.subs(dim_sub).collect();
      assert_eq!(
        subs
          .iter()
          .map(|sub| sub.vertices_sorted().clone())
          .collect::<Vec<_>>(),
        subs_vertices
      );

      for (isub, sub) in subs.iter().enumerate() {
        let sub_vertices = &subs_vertices[isub];
        for dim_sup in dim_sub..dim {
          let sups: Vec<_> = sub.sups(dim_sup).collect();
          let sups_vertices = sups
            .iter()
            .map(|sub| sub.vertices_sorted().clone())
            .collect::<Vec<_>>();
          sups_vertices
            .iter()
            .all(|sup| sub_vertices <= sup && sup <= &cell_vertices);
        }
      }
    }
  }
}
