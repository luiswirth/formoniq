//! A mesh plays the role of a container of mesh entities (in 3D: cells, faces, edges, vertices).
//! It provides a global numbering for unique identification  of the entities.
//! It allows for the traversal of the entities in a defined order.
//! It provides topology information (incidence).
//! It stores the mesh geometry (location information), but more of this can be
//! found in the geometry module.

pub mod boundary;
pub mod coordinates;
pub mod data;
pub mod hyperbox;

use crate::{
  combinatorics::sort_count_swaps, geometry::GeometrySimplex, orientation::Orientation, Dim,
};

use indexmap::{set::MutableValues, IndexSet};
use std::{
  collections::HashMap,
  hash::Hash,
  rc::{self, Rc},
};

pub type NodeId = usize;
pub type EdgeId = usize;
pub type CellId = usize;
pub type DSimplexId = usize;
pub type SimplexId = (Dim, DSimplexId);

pub type RawSimplex = Vec<NodeId>;

/// Helper struct that ensures that edges don't have an orientation.
/// Always use `Self::new` never construct tuple directly.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct EdgeBetweenVertices(NodeId, NodeId);
impl EdgeBetweenVertices {
  pub fn new(a: NodeId, b: NodeId) -> Self {
    if a < b {
      Self(a, b)
    } else {
      Self(b, a)
    }
  }
}

// A simplicial manifold is a piecewiese-linear manifold.
#[derive(Debug)]
pub struct SimplicialManifold {
  /// All simplicies of the mesh, from 0-simplicies (vertices) to d-simplicies (cells).
  nnodes: usize,
  /// topology
  simplicies: Vec<IndexSet<ManifoldSimplex>>,
  /// geometry
  edge_lengths: HashMap<EdgeBetweenVertices, f64>,
}

// getters
impl SimplicialManifold {
  pub fn dim(&self) -> Dim {
    self.simplicies.len() - 1
  }
  pub fn nnodes(&self) -> usize {
    self.nnodes
  }
  pub fn cells(&self) -> &IndexSet<ManifoldSimplex> {
    self.simplicies.last().unwrap()
  }
  pub fn ncells(&self) -> usize {
    self.cells().len()
  }
  pub fn cell(&self, id: CellId) -> &ManifoldSimplex {
    self.cells().get_index(id).unwrap()
  }
  pub fn simplicies(&self) -> &[IndexSet<ManifoldSimplex>] {
    &self.simplicies
  }
  pub fn dsimplicies(&self, d: Dim) -> &IndexSet<ManifoldSimplex> {
    &self.simplicies[d]
  }
  pub fn simplex(&self, id: SimplexId) -> &ManifoldSimplex {
    self.simplicies[id.0].get_index(id.1).unwrap()
  }
  pub fn facet(&self, id: DSimplexId) -> &ManifoldSimplex {
    self.simplicies[self.dim() - 1].get_index(id).unwrap()
  }

  /// The mesh width $h$, which is the largest diameter of all cells.
  pub fn mesh_width(&self) -> f64 {
    (0..self.cells().len())
      .map(|icell| self.cell(icell).geometry_simplex().diameter())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  /// The shape regularity measure $rho$ of the whole mesh, which is the largest
  /// shape regularity measure over all cells.
  pub fn shape_regularity_measure(&self) -> f64 {
    (0..self.cells().len())
      .map(|icell| {
        self
          .cell(icell)
          .geometry_simplex()
          .shape_reguarity_measure()
      })
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }
}

// constructors
impl SimplicialManifold {
  pub fn from_cells(
    nnodes: usize,
    cells: Vec<RawSimplex>,
    edge_lengths: HashMap<EdgeBetweenVertices, f64>,
  ) -> Rc<Self> {
    Rc::new_cyclic(|this| {
      let dim_intrinsic = cells[0].len() - 1;
      let mut simplicies = vec![IndexSet::new(); dim_intrinsic + 1];

      // add 0-simplicies (vertices)
      simplicies[0] = (0..nnodes)
        .map(|ivertex| ManifoldSimplex {
          vertices: vec![ivertex],
          sorted_vertices: vec![ivertex],
          sort_orientation: Orientation::Pos,
          subs: Vec::new(),
          supers: Vec::new(),
          id: ivertex,
          mesh: this.clone(),
        })
        .collect();

      // add d-simplicies (cells)
      simplicies[dim_intrinsic] = cells
        .into_iter()
        .enumerate()
        .map(|(icell, vertices)| {
          let mut sorted_vertices = vertices.clone();
          let nswaps = sort_count_swaps(&mut sorted_vertices);
          let orientation = Orientation::from_permutation_parity(nswaps);
          ManifoldSimplex {
            vertices,
            sorted_vertices,
            sort_orientation: orientation,
            subs: Vec::new(),
            supers: Vec::new(),
            id: icell,
            mesh: this.clone(),
          }
        })
        .collect();

      // add all other simplicies in between and record the incidence
      // relationship
      for super_dim in (1..=dim_intrinsic).rev() {
        let _sub_dim = super_dim - 1;

        let ([.., sub_simps], [super_simps, ..]) = simplicies.split_at_mut(super_dim) else {
          unreachable!()
        };

        for isuper_simp in 0..super_simps.len() {
          let super_simp = super_simps.get_index_mut2(isuper_simp).unwrap();
          for ivertex in 0..super_simp.vertices.len() {
            let super_orientation = Orientation::from_permutation_parity(ivertex);

            let mut vertices = super_simp.vertices.clone();
            vertices.remove(ivertex);
            let mut sorted_vertices = vertices.clone();
            let sort_orientation =
              Orientation::from_permutation_parity(sort_count_swaps(&mut sorted_vertices));

            // TODO: can we avoid constructing this, before checking that this simplex already exists?
            let sub_simp = ManifoldSimplex {
              vertices,
              sorted_vertices,
              sort_orientation,
              subs: Vec::new(),
              supers: vec![(isuper_simp, super_orientation)],
              id: sub_simps.len(),
              mesh: this.clone(),
            };

            let (isub_simp, new_insert) = sub_simps.insert_full(sub_simp);
            if !new_insert {
              let sub_simp = sub_simps.get_index_mut2(isub_simp).unwrap();
              sub_simp.supers.push((isuper_simp, super_orientation));
            }
            super_simp.subs.push((isub_simp, super_orientation));
          }
        }
      }

      Self {
        nnodes,
        simplicies,
        edge_lengths,
      }
    })
  }
}

/// A mesh entity of a simplicial mesh.
/// Defines the simplex based on its vertices and contains topological
/// information (incidence).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ManifoldSimplex {
  /// Verticies defining the simplex.
  vertices: Vec<NodeId>,
  /// The same as [`verticies`] but sorted. Used for comparing simplicies.
  sorted_vertices: Vec<NodeId>,
  /// The relative orientation between `vertices` and `sorted_vertices`.
  sort_orientation: Orientation,
  /// The sub entities of this simplex, which are all the direct (only 1
  /// dimension difference) faces of this simplex.
  subs: Vec<(DSimplexId, Orientation)>,
  /// The super entities of this simplex, which are all the simplicies that
  /// directly (only 1 dimension difference) contain this simplex as a face.
  supers: Vec<(DSimplexId, Orientation)>,
  /// ID identifiying this simplex in the d-th dimension of the mesh
  id: DSimplexId,
  /// The mesh this simplex lives in.
  mesh: rc::Weak<SimplicialManifold>,
}

/// Functionality methods.
impl ManifoldSimplex {
  pub fn dim(&self) -> Dim {
    self.vertices.len() - 1
  }
  pub fn nvertices(&self) -> usize {
    self.vertices.len()
  }
  pub fn vertices(&self) -> &[NodeId] {
    &self.vertices
  }
  pub fn dsimplex_id(&self) -> DSimplexId {
    self.id
  }
  pub fn simplex_id(&self) -> SimplexId {
    (self.dim(), self.id)
  }

  pub fn subs_with_orientation(&self) -> &[(DSimplexId, Orientation)] {
    &self.subs
  }
  pub fn subs(&self) -> Vec<DSimplexId> {
    self.subs.iter().map(|s| s.0).collect()
  }
  pub fn supers_with_orientation(&self) -> &[(DSimplexId, Orientation)] {
    &self.supers
  }
  pub fn supers(&self) -> Vec<DSimplexId> {
    self.supers.iter().map(|s| s.0).collect()
  }

  pub fn geometry_simplex(&self) -> GeometrySimplex {
    let mesh = &self.mesh.upgrade().unwrap();
    let mut edge_lengths = Vec::new();
    for &v0 in &self.vertices {
      for &v1 in &self.vertices {
        if v0 < v1 {
          edge_lengths.push(mesh.edge_lengths[&EdgeBetweenVertices::new(v0, v1)]);
        }
      }
    }
    GeometrySimplex::new(self.dim(), edge_lengths)
  }
}

/// Two simplicies are considered the same, if they are made out of the same
/// vertices. The only thing that might still be different is the orientation,
/// depending on the actual order of the vertices.
impl PartialEq for ManifoldSimplex {
  fn eq(&self, other: &Self) -> bool {
    self.sorted_vertices == other.sorted_vertices
  }
}
impl Eq for ManifoldSimplex {}
impl Hash for ManifoldSimplex {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.sorted_vertices.hash(state)
  }
}

#[cfg(test)]
mod test {
  use crate::{geometry::GeometrySimplex, orientation::Orientation as O};

  #[test]
  fn incidence_check_1d() {
    let mesh = GeometrySimplex::new_ref(2).into_singleton_mesh();
    let vertices = &mesh.simplicies[0];
    let edges = &mesh.simplicies[1];
    let cells = &mesh.simplicies[2];

    assert_eq!(vertices.len(), 3);
    assert_eq!(edges.len(), 3);
    assert_eq!(cells.len(), 1);

    for (i, v) in vertices.iter().enumerate() {
      assert!(v.id == i);
      assert!(v.subs.is_empty());
    }
    assert_eq!(vertices[0].supers, vec![(1, O::Neg), (2, O::Neg)]);
    assert_eq!(vertices[1].supers, vec![(0, O::Neg), (2, O::Pos)]);
    assert_eq!(vertices[2].supers, vec![(0, O::Pos), (1, O::Pos)]);

    for (i, e) in edges.iter().enumerate() {
      assert!(e.id == i);
    }
    assert_eq!(edges[0].subs, vec![(2, O::Pos), (1, O::Neg)]);
    assert_eq!(edges[1].subs, vec![(2, O::Pos), (0, O::Neg)]);
    assert_eq!(edges[2].subs, vec![(1, O::Pos), (0, O::Neg)]);
    assert_eq!(edges[0].supers, vec![(0, O::Pos)]);
    assert_eq!(edges[1].supers, vec![(0, O::Neg)]);
    assert_eq!(edges[2].supers, vec![(0, O::Pos)]);

    assert!(cells[0].id == 0);
    assert_eq!(cells[0].subs, vec![(0, O::Pos), (1, O::Neg), (2, O::Pos)]);
    assert!(cells[0].supers.is_empty());

    assert_eq!(mesh.boundary(), vec![(1, 0), (1, 1), (1, 2)]);

    let mut boundary_nodes = mesh.boundary_nodes();
    boundary_nodes.sort_unstable();
    assert_eq!(boundary_nodes, vec![0, 1, 2]);
  }
}
