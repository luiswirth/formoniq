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
use itertools::Itertools as _;
use std::{
  collections::{HashMap, HashSet},
  hash::Hash,
  ops::Deref,
  rc::{self, Rc},
};

pub type VertexIdx = usize;

/// Fat pointer to simplex.
pub struct SimplexHandle {
  mesh: Rc<SimplicialManifold>,
  id: usize,
}
impl Deref for SimplexHandle {
  type Target = ManifoldSimplex;

  fn deref(&self) -> &Self::Target {
    self.mesh.simplex(self.id)
  }
}

/// Contains the minimal amount of topological information required to define a simplex meaningfully
/// and potentially further precomputed topological properties.
pub struct SimplexTopology {
  /// Verticies defining the simplex.
  vertices: Vec<VertexIdx>,
  /// The same as [`verticies`] but sorted. Used for comparing simplicies.
  sorted_vertices: Vec<VertexIdx>,
  /// The relative orientation between `vertices` and `sorted_vertices`.
  sort_orientation: Orientation,
}

/// A enum for managing different levels of precomputed
/// topological qualities of a simplex.
/// At least [`MinTopology`] must always be given for topological properties
/// to be meaningfully defined. All others can be derived from them.
pub enum PrecomputedTopology {
  DirectTopology(DirectTopology),
  FullTopology(FullTopology),
}

/// Topology information only one level in both directions.
pub struct DirectTopology {
  /// The sub entities of this simplex, which are all the direct (only 1
  /// dimension difference) faces of this simplex.
  children: Vec<(DSimplexId, Orientation)>,
  /// The super entities of this simplex, which are all the simplicies that
  /// directly (only 1 dimension difference) contain this simplex as a face.
  parents: Vec<(DSimplexId, Orientation)>,
}

/// Full topological information for a simplex.
/// All descendants of the simplex are stored.
/// This is usually used for Cells.
pub struct FullTopology {
  /// The sub entities of this simplex, which are all the direct (only 1
  /// dimension difference) faces of this simplex.
  descendants: Vec<(DSimplexId, Orientation)>,
  /// The super entities of this simplex, which are all the simplicies that
  /// directly (only 1 dimension difference) contain this simplex as a face.
  ancestors: Vec<(DSimplexId, Orientation)>,
}

pub type Length = f64;

pub struct RawSimplexTopology {
  vertices: Vec<VertexIdx>,
}
pub struct RawSimplexGeometry {
  edge_lengths: Vec<Length>,
}

pub struct RawManifoldGeometry {
  edge_lengths: HashMap<EdgeBetweenVertices, Length>,
}

pub struct RawSimplicialManifold {
  cell_topologies: Vec<RawSimplexTopology>,
  geometry: RawManifoldGeometry,
}

/// Helper struct that ensures that edges don't have an orientation.
/// Always use `Self::new` never construct tuple directly.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct EdgeBetweenVertices(VertexIdx, VertexIdx);
impl EdgeBetweenVertices {
  pub fn new(a: VertexIdx, b: VertexIdx) -> Self {
    if a < b {
      Self(a, b)
    } else {
      Self(b, a)
    }
  }
}

/// A container for simplicies of common dimension.
pub struct Skeleton {
  simplicies: Vec<ManifoldSimplex>,
}

// A simplicial manifold with both topological and geometric information.
#[derive(Debug)]
pub struct SimplicialManifold {
  nnodes: usize,
  /// All simplicies of the mesh, from 0-simplicies (vertices) to d-simplicies (cells).
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
    self.simplicies[id.dim].get_index(id.idx).unwrap()
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
    cells: Vec<RawSimplexTopology>,
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
          idx: ivertex,
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
            idx: icell,
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
              idx: sub_simps.len(),
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
  vertices: Vec<VertexIdx>,
  /// The same as [`verticies`] but sorted. Used for comparing simplicies.
  sorted_vertices: Vec<VertexIdx>,
  /// The relative orientation between `vertices` and `sorted_vertices`.
  sort_orientation: Orientation,
  /// The sub entities of this simplex, which are all the direct (only 1
  /// dimension difference) faces of this simplex.
  subs: Vec<(DSimplexId, Orientation)>,
  /// The super entities of this simplex, which are all the simplicies that
  /// directly (only 1 dimension difference) contain this simplex as a face.
  supers: Vec<(DSimplexId, Orientation)>,
  /// ID identifiying this simplex in the d-th dimension of the mesh
  idx: DSimplexId,
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
  pub fn vertices(&self) -> &[VertexIdx] {
    &self.vertices
  }
  pub fn simplex_id(&self) -> SimplexId {
    SimplexId::new(self.dim(), self.idx)
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
  // TODO: optimize! We need a different data representation
  pub fn descendants(&self, dim_diff: Dim) -> Vec<DSimplexId> {
    if dim_diff == 0 {
      return vec![self.idx];
    }
    self
      .subs()
      .iter()
      .flat_map(|&s| {
        self
          .mesh
          .upgrade()
          .unwrap()
          .simplex(SimplexId::new(self.dim() - 1, s))
          .descendants(dim_diff - 1)
          .into_iter()
      })
      .collect::<HashSet<_>>()
      .into_iter()
      .sorted()
      .collect()
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
  use itertools::Itertools;

  use crate::{geometry::GeometrySimplex, orientation::Orientation as O};

  #[test]
  fn incidence_check_2d() {
    let mesh = GeometrySimplex::new_ref(2).into_singleton_mesh();
    let vertices = &mesh.simplicies[0];
    let edges = &mesh.simplicies[1];
    let cells = &mesh.simplicies[2];

    assert_eq!(vertices.len(), 3);
    assert_eq!(edges.len(), 3);
    assert_eq!(cells.len(), 1);

    for (i, v) in vertices.iter().enumerate() {
      assert!(v.idx == i);
      assert!(v.subs.is_empty());
    }
    assert_eq!(vertices[0].supers, vec![(1, O::Neg), (2, O::Neg)]);
    assert_eq!(vertices[1].supers, vec![(0, O::Neg), (2, O::Pos)]);
    assert_eq!(vertices[2].supers, vec![(0, O::Pos), (1, O::Pos)]);

    for (i, e) in edges.iter().enumerate() {
      assert!(e.idx == i);
    }
    assert_eq!(edges[0].subs, vec![(2, O::Pos), (1, O::Neg)]);
    assert_eq!(edges[1].subs, vec![(2, O::Pos), (0, O::Neg)]);
    assert_eq!(edges[2].subs, vec![(1, O::Pos), (0, O::Neg)]);
    assert_eq!(edges[0].supers, vec![(0, O::Pos)]);
    assert_eq!(edges[1].supers, vec![(0, O::Neg)]);
    assert_eq!(edges[2].supers, vec![(0, O::Pos)]);

    assert!(cells[0].idx == 0);
    assert_eq!(cells[0].subs, vec![(0, O::Pos), (1, O::Neg), (2, O::Pos)]);
    assert!(cells[0].supers.is_empty());

    assert_eq!(cells[0].descendants(0), vec![cells[0].idx]);
    assert_eq!(
      cells[0]
        .descendants(1)
        .into_iter()
        .sorted()
        .collect::<Vec<_>>(),
      vec![edges[0].idx, edges[1].idx, edges[2].idx]
    );
    assert_eq!(
      cells[0]
        .descendants(2)
        .into_iter()
        .sorted()
        .collect::<Vec<_>>(),
      vec![vertices[0].idx, vertices[1].idx, vertices[2].idx]
    );

    assert_eq!(
      mesh.boundary(),
      vec![(1, 0), (1, 1), (1, 2)]
        .into_iter()
        .map(From::from)
        .collect::<Vec<_>>()
    );

    let mut boundary_nodes = mesh.boundary_nodes();
    boundary_nodes.sort_unstable();
    assert_eq!(boundary_nodes, vec![0, 1, 2]);
  }
}
