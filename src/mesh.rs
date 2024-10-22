//! Simplicial Manifold Datastructure for working with Topology and Geometry.
//!
//! - Container for mesh entities (Simplicies).
//! - Global numbering for unique identification of the entities.
//! - Entity Iteration
//! - Topological Information (Incidence)
//! - Geometrical information (Lengths, Volumes)

pub mod boundary;
pub mod coordinates;
pub mod hyperbox;
pub mod raw;
pub mod util;

use crate::{geometry::GeometrySimplex, matrix::SparseMatrix, Dim, Length, Orientation};

use indexmap::IndexMap;
use std::hash::Hash;

pub type VertexIdx = usize;
pub type EdgeIdx = usize;
pub type CellIdx = usize;
pub type KSimplexIdx = usize;
pub type SimplexIdx = (Dim, KSimplexIdx);

/// A simplicial manifold with both topological and geometric information.
pub struct SimplicialManifold {
  topology: ManifoldTopology,
  geometry: ManifoldGeometry,
}

pub struct ManifoldTopology {
  skeletons: Vec<SkeletonTopology>,
  incidence_matrices: Vec<IncidenceMatrix>,
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

  pub fn incidence_matrices(&self) -> &[IncidenceMatrix] {
    &self.incidence_matrices
  }
}

/// A container for topological simplicies of common dimension.
pub type SkeletonTopology = IndexMap<SimplexBetweenVertices, SimplexTopology>;

pub struct ManifoldGeometry {
  /// mapping [`EdgeIdx`] -> [`Length`]
  edge_lengths: Vec<Length>,
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
      .map(|cell| cell.geometry_simplex().diameter())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  /// The shape regularity measure $rho$ of the whole mesh, which is the largest
  /// shape regularity measure over all cells.
  pub fn shape_regularity_measure(&self) -> f64 {
    self
      .cells()
      .iter()
      .map(|cell| cell.geometry_simplex().shape_reguarity_measure())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }
}

/// Topological information of the simplex.
#[derive(Debug, Clone)]
pub struct SimplexTopology {
  /// The vertices of the simplex.
  /// The ordering is arbitrary (?)
  vertices: Vec<VertexIdx>,
  /// The edges of the simplex.
  /// These are ordered lexicographically w.r.t.
  /// the local vertex indices.
  /// E.g. For 3-simplex: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
  edges: Vec<EdgeIdx>,
  /// The supersimplicies that are one layer above this simplex
  /// together with their relative orientation.
  supers: Vec<(KSimplexIdx, Orientation)>,
  /// The subsimplicies that are one layer below this simplex
  /// together with their relative orientation.
  subs: Vec<(KSimplexIdx, Orientation)>,
}

impl SimplexTopology {
  fn new(
    vertices: Vec<VertexIdx>,
    edges: Vec<EdgeIdx>,
    supers: Vec<(KSimplexIdx, Orientation)>,
    subs: Vec<(KSimplexIdx, Orientation)>,
  ) -> Self {
    Self {
      vertices,
      edges,
      supers,
      subs,
    }
  }
  fn stub(vertices: Vec<VertexIdx>) -> Self {
    Self::new(vertices, Vec::new(), Vec::new(), Vec::new())
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

/// Unoriented Edge
/// Unorientedness emphasized through word "Between".
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimplexBetweenVertices(Vec<VertexIdx>);
impl SimplexBetweenVertices {
  pub fn new(mut vertices: Vec<VertexIdx>) -> Self {
    vertices.sort();
    Self(vertices)
  }
  pub fn vertex(v: VertexIdx) -> SimplexBetweenVertices {
    Self(vec![v])
  }
  pub fn edge(a: VertexIdx, b: VertexIdx) -> Self {
    if a < b {
      Self(vec![a, b])
    } else {
      Self(vec![b, a])
    }
  }

  pub fn nvertices(&self) -> usize {
    self.0.len()
  }
  pub fn dim(&self) -> Dim {
    self.nvertices() - 1
  }
}

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
  pub fn edges(&self) -> impl Iterator<Item = SimplexHandle<'m>> {
    self
      .topology()
      .edges
      .iter()
      .map(|&idx| SimplexHandle::new(self.mesh, (1, idx)))
  }
  fn edge_lengths(&self) -> Vec<f64> {
    let edge_lengths = self
      .edges()
      .map(|e| self.mesh.geometry.edge_lengths[e.idx.1])
      .collect();
    edge_lengths
  }
  pub fn geometry_simplex(&self) -> GeometrySimplex {
    GeometrySimplex::new(self.idx.0, self.edge_lengths())
  }
  pub fn nsupers(&self) -> usize {
    self.topology().supers.len()
  }
  pub fn supers(&self) -> Chain<'m> {
    let mut idxs = Vec::new();
    let mut coeffs = Vec::new();
    for sup in &self.topology().supers {
      idxs.push(sup.0);
      coeffs.push(sup.1.as_i32());
    }
    Chain::new(self.mesh, self.dim() + 1, idxs, coeffs)
  }
  pub fn subs(&self) -> Chain<'m> {
    let mut idxs = Vec::new();
    let mut coeffs = Vec::new();
    for sub in &self.topology().subs {
      idxs.push(sub.0);
      coeffs.push(sub.1.as_i32());
    }
    Chain::new(self.mesh, self.dim() + 1, idxs, coeffs)
  }
  pub fn boundary(&self) -> Chain<'m> {
    self.subs()
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
  pub fn len(&self) -> usize {
    self.mesh.topology.skeletons[self.dim].len()
  }
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}

impl<'m> SkeletonHandle<'m> {
  pub fn get(&self, idx: KSimplexIdx) -> SimplexHandle<'m> {
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
}

/// entries are one of {0,-1,+1}
pub type IncidenceMatrix = SparseMatrix;
