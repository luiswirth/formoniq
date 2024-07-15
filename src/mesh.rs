//! A mesh plays the role of a container of mesh entities (in 3D: cells, faces, edges, vertices).
//! It provides a global numbering for unique identification  of the entities.
//! It allows for the traversal of the entities in a defined order.
//! It provides topology information (incidence).
//! It stores the mesh geometry (location information), but more of this can be
//! found in the geometry module.

pub mod gmsh;
pub mod hypercube;

use indexmap::{set::MutableValues, IndexSet};

use crate::{geometry::CoordSimplex, orientation::Orientation, util::sort_swap_count, Dim};

use std::{
  hash::Hash,
  rc::{self, Rc},
};

pub type NodeId = usize;
pub type CellId = usize;
pub type SimplexId = usize;

/// A mesh entity of a simplicial mesh.
/// Defines the simplex based on its vertices and contains topological
/// information (incidence).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MeshSimplex {
  /// Verticies defining the simplex.
  /// The verticies are stored as (column) indicies into the node mesh.
  vertices: Vec<NodeId>,
  /// The same as [`verticies`] but sorted. Used for comparing simplicies.
  sorted_vertices: Vec<NodeId>,
  /// The relative orientation between `vertices` and `sorted_vertices`.
  sort_orientation: Orientation,
  /// The sub entities of this simplex, which are all the direct (only 1
  /// dimension difference) faces of this simplex.
  subs: Vec<(SimplexId, Orientation)>,
  /// The super entities of this simplex, which are all the simplicies that
  /// directly (only 1 dimension difference) contain this simplex as a face.
  supers: Vec<(SimplexId, Orientation)>,
  /// ID identifiying this simplex in the mesh
  id: SimplexId,
  /// The mesh this simplex lives in.
  mesh: rc::Weak<SimplicialMesh>,
}

/// Functionality methods.
impl MeshSimplex {
  pub fn dim_intrinsic(&self) -> Dim {
    self.vertices.len() - 1
  }
  pub fn nvertices(&self) -> usize {
    self.vertices.len()
  }
  pub fn vertices(&self) -> &[NodeId] {
    &self.vertices
  }
}

/// Two simplicies are considered the same, if they are made out of the same
/// vertices. The only thing that might still be different is the orientation,
/// depending on the actual order of the vertices.
impl PartialEq for MeshSimplex {
  fn eq(&self, other: &Self) -> bool {
    self.sorted_vertices == other.sorted_vertices
  }
}
impl Eq for MeshSimplex {}
impl Hash for MeshSimplex {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.sorted_vertices.hash(state)
  }
}

/// A pure simplicial mesh also called a triangulation.
#[derive(Debug)]
pub struct SimplicialMesh {
  /// The coordinates of the nodes of this mesh in the matrix columns.
  node_coords: na::DMatrix<f64>,
  /// All simplicies of the mesh. From 0-simplicies (vertices) to d-simplicies (cells).
  simplicies: Vec<IndexSet<MeshSimplex>>,
}
impl SimplicialMesh {
  pub fn dim_intrinsic(&self) -> Dim {
    self.simplicies.len() - 1
  }
  pub fn dim_ambient(&self) -> Dim {
    self.node_coords.nrows()
  }
  pub fn nnodes(&self) -> usize {
    self.node_coords.ncols()
  }
  /// The dimension of highest dimensional [`Skeleton`]
  pub fn node_coords(&self) -> &na::DMatrix<f64> {
    &self.node_coords
  }

  pub fn cells(&self) -> &IndexSet<MeshSimplex> {
    self.simplicies.last().unwrap()
  }
  pub fn ncells(&self) -> usize {
    self.cells().len()
  }
  /// cells of the mesh
  pub fn cell(&self, id: CellId) -> &MeshSimplex {
    self.cells().get_index(id).unwrap()
  }

  pub fn coordinate_cell(&self, id: CellId) -> CoordSimplex {
    let cell = self.cell(id);
    let mut vertices = na::DMatrix::zeros(self.dim_ambient(), cell.nvertices());
    for (i, &v) in cell.vertices().iter().enumerate() {
      vertices
        .column_mut(i)
        .copy_from(&self.node_coords.column(v));
    }
    CoordSimplex::new(vertices)
  }

  /// The mesh width $h$, which is the largest diameter of all cells.
  pub fn mesh_width(&self) -> f64 {
    (0..self.cells().len())
      .map(|icell| self.coordinate_cell(icell).diameter())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  /// The shape regularity measure $rho$ of the whole mesh, which is the largest
  /// shape regularity measure over all cells.
  pub fn shape_regularity_measure(&self) -> f64 {
    (0..self.cells().len())
      .map(|icell| self.coordinate_cell(icell).shape_reguarity_measure())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }
}

pub type RawSimplex = Vec<usize>;

impl SimplicialMesh {
  pub fn from_cells(node_coords: na::DMatrix<f64>, cells: Vec<RawSimplex>) -> Rc<Self> {
    Rc::new_cyclic(|this| {
      let dim_intrinsic = cells[0].len() - 1;
      let mut simplicies = vec![IndexSet::new(); dim_intrinsic + 1];

      // add 0-simplicies (vertices)
      simplicies[0] = (0..node_coords.ncols())
        .map(|ivertex| MeshSimplex {
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
          let nswaps = sort_swap_count(&mut sorted_vertices);
          let orientation = Orientation::from_permutation_parity(nswaps);
          MeshSimplex {
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

      for sub_dim in (0..dim_intrinsic).rev() {
        let super_dim = sub_dim + 1;

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
              Orientation::from_permutation_parity(sort_swap_count(&mut sorted_vertices));

            // TODO: can we avoid constructing this, before checking that this simplex already exists?
            let sub_simp = MeshSimplex {
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
        node_coords,
        simplicies,
      }
    })
  }
}

#[cfg(test)]
mod test {
  use crate::{geometry::CoordSimplex, orientation::Orientation as O};

  #[test]
  fn incidence_check_1d() {
    let mesh = CoordSimplex::new_ref(2).into_singleton_mesh();
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
  }
}
