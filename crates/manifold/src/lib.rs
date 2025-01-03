//! Simplicial Manifold Datastructure for working with Topology and Geometry.
//!
//! - Container for mesh entities (Simplicies).
//! - Global numbering for unique identification of the entities.
//! - Entity Iteration
//! - Topological Information (Incidence)
//! - Geometrical information (Lengths, Volumes)

extern crate nalgebra as na;
extern crate nalgebra_sparse as nas;

pub mod complex;
pub mod coords;
pub mod simplicial;

pub mod gen;

use common::{util, Dim};
use complex::{
  Complex, ComplexSkeleton, FacetIdx, KSimplexIdx, SimplexData, SimplexIdx, VertexIdx,
};
use geometry::regge::EdgeLengths;
use itertools::Itertools;
use simplicial::{LocalComplex, OrientedVertplex, SortedVertplex};

use std::hash::Hash;

pub struct RiemannianManifold {
  pub facets: Vec<OrientedVertplex>,
  pub lengths: EdgeLengths,
}
impl RiemannianManifold {
  pub fn into_complex(self) -> RiemannianComplex {
    let Self { facets, lengths } = self;
    let complex = Complex::new(facets.clone());
    RiemannianComplex::new(facets, complex, lengths)
  }
}

/// A simplicial manifold with both topological and geometric information.
#[derive(Debug)]
pub struct RiemannianComplex {
  facets: Vec<OrientedVertplex>,
  complex: Complex,
  edge_lengths: EdgeLengths,
}

impl RiemannianComplex {
  pub fn new(facets: Vec<OrientedVertplex>, complex: Complex, edge_lengths: EdgeLengths) -> Self {
    Self {
      facets,
      complex,
      edge_lengths,
    }
  }

  pub fn complex(&self) -> &Complex {
    &self.complex
  }
  pub fn edge_lengths(&self) -> &EdgeLengths {
    &self.edge_lengths
  }
  pub fn edge_lengths_mut(&mut self) -> &mut EdgeLengths {
    &mut self.edge_lengths
  }

  pub fn dim(&self) -> Dim {
    self.complex.dim()
  }
  pub fn skeleton(&self, dim: Dim) -> SkeletonHandle {
    SkeletonHandle::new(self, dim)
  }

  pub fn cells(&self) -> impl Iterator<Item = CellHandle> {
    (0..self.ncells()).map(|icell| CellHandle::new(self, icell))
  }
  pub fn ncells(&self) -> usize {
    self.facets.len()
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

  pub fn has_boundary(&self) -> bool {
    !self.boundary_faces().is_empty()
  }

  /// For a d-mesh computes the boundary, which consists of faces ((d-1)-subs).
  ///
  /// The boundary faces are characterized by the fact that they
  /// only have 1 cell as super entity.
  pub fn boundary_faces(&self) -> Vec<SimplexHandle> {
    self
      .faces()
      .iter()
      .filter(|f| f.anti_boundary().len() == 1)
      .collect()
  }

  /// The vertices that lie on the boundary of the mesh.
  /// No particular order of vertices.
  pub fn boundary_vertices(&self) -> Vec<VertexIdx> {
    self
      .boundary_faces()
      .into_iter()
      .flat_map(|face| face.sorted_vertplex().iter().copied())
      .unique()
      .collect()
  }

  pub fn flag_boundary_vertices(&self) -> Vec<bool> {
    util::indicies_to_flags(&self.boundary_vertices(), self.nvertices())
  }

  pub fn boundary_cells(&self) -> Vec<FacetIdx> {
    self
      .boundary_faces()
      .into_iter()
      // the boundary has only one parent cell by definition
      .map(|face| face.anti_boundary().idxs[0])
      .unique()
      .collect()
  }
}

/// Fat pointer to simplex.
pub struct CellHandle<'m> {
  idx: FacetIdx,
  mesh: &'m RiemannianComplex,
}
impl<'m> CellHandle<'m> {
  pub fn new(mesh: &'m RiemannianComplex, idx: FacetIdx) -> Self {
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
    &self.mesh.facets[self.idx]
  }

  pub fn sorted_vertplex(&self) -> SortedVertplex {
    self.mesh.facets[self.idx].clone().sort()
  }

  pub fn as_cell_complex(&self) -> LocalComplex {
    let combinatorial = self.oriented_vertplex();
    let simplex = self.as_simplex();
    let faces = (0..=self.dim())
      .map(|k| simplex.subs(k).map(|s| s.kidx()).collect())
      .collect();
    let sign = combinatorial.sign();
    let edge_lengths = simplex.edge_lengths();
    LocalComplex::new(faces, sign, edge_lengths)
  }
}

/// Fat pointer to simplex.
pub struct SimplexHandle<'m> {
  idx: SimplexIdx,
  mesh: &'m RiemannianComplex,
}
impl<'m> std::fmt::Debug for SimplexHandle<'m> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("SimplexHandle")
      .field("idx", &self.idx)
      .field("mesh", &(self.mesh as *const RiemannianComplex))
      .finish()
  }
}

impl<'m> SimplexHandle<'m> {
  pub fn new(mesh: &'m RiemannianComplex, idx: impl Into<SimplexIdx>) -> Self {
    let idx = idx.into();
    idx.assert_valid(&mesh.complex);
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

  pub fn mesh(&self) -> &'m RiemannianComplex {
    self.mesh
  }
  pub fn skeleton(&self) -> SkeletonHandle<'m> {
    self.mesh.skeleton(self.dim())
  }
  pub fn simplex_data(&self) -> &SimplexData {
    self
      .mesh
      .complex
      .skeleton(self.dim())
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
    self
      .mesh
      .complex
      .skeleton(self.dim())
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
        .anti_boundary(parent_cell.sorted_vertplex())
      {
        let coeff = sup.sign().as_i32();
        let idx = self
          .mesh
          .skeleton(self.dim() + 1)
          .get_by_vertplex(&sup.forget_sign())
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
      .parent_facets
      .iter()
      .map(|&cell_idx| CellHandle::new(self.mesh, cell_idx))
  }
  pub fn edges(&self) -> impl Iterator<Item = SimplexHandle<'m>> + '_ {
    self.subs(1)
  }
  fn edge_lengths(&self) -> EdgeLengths {
    let lengths: Vec<_> = self
      .edges()
      .map(|e| self.mesh.edge_lengths.length(e.kidx()))
      .collect();
    EdgeLengths::new(lengths.into())
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
          .sups(parent_cell.sorted_vertplex(), dim + 1)
          .map(move |a| self.mesh.skeleton(dim).get_by_vertplex(&a).idx)
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
    (self.mesh as *const RiemannianComplex).hash(state);
    self.idx.hash(state);
  }
}

pub struct SkeletonHandle<'m> {
  mesh: &'m RiemannianComplex,
  dim: Dim,
}

impl<'m> SkeletonHandle<'m> {
  pub fn new(mesh: &'m RiemannianComplex, dim: Dim) -> Self {
    assert!(dim <= mesh.dim(), "Invalid Skeleton Dimension");
    Self { mesh, dim }
  }

  pub fn raw(&self) -> &ComplexSkeleton {
    self.mesh.complex.skeleton(self.dim)
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
  mesh: &'m RiemannianComplex,
  dim: Dim,
  idxs: Vec<KSimplexIdx>,
  coeffs: Vec<i32>,
}
impl<'m> SparseChain<'m> {
  fn new(mesh: &'m RiemannianComplex, dim: Dim, idxs: Vec<KSimplexIdx>, coeffs: Vec<i32>) -> Self {
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
  mesh: &'m RiemannianComplex,
  dim: Dim,
  idxs: Vec<KSimplexIdx>,
  coeffs: Vec<i32>,
}

#[cfg(test)]
mod test {
  use index_algebra::sign::Sign;

  use super::simplicial::{ReferenceCell, Vertplex};
  use crate::simplicial::nsubsimplicies;

  #[test]
  fn incidence() {
    let dim = 3;
    let mesh = ReferenceCell::new(dim).into_singleton_mesh();
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

    let cell_vertplex = Vertplex::increasing(dim + 1).with_sign(Sign::Pos);
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
