use super::{simplex::SimplexLengths, EdgeIdx};
use crate::{
  topology::{
    complex::Complex,
    data::SkeletonData,
    handle::{KSimplexIdx, SimplexRef, SkeletonRef},
  },
  Dim,
};

use crate::linalg::Vector;

use itertools::Itertools;
use rayon::iter::ParallelIterator;

use std::{io, path::Path};

/// The lengths of the edges of the mesh.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MeshLengths {
  vector: Vector,
}

/// Edge lengths are grade-1 data on the mesh: one scalar per edge.
impl SkeletonData for MeshLengths {
  type Item<'a> = &'a f64;
  fn grade(&self) -> Dim {
    1
  }
  fn len(&self) -> usize {
    self.vector.len()
  }
  fn at(&self, kidx: KSimplexIdx) -> &f64 {
    &self.vector[kidx]
  }
}
impl MeshLengths {
  pub fn new(vector: Vector, complex: &Complex) -> Self {
    let this = Self { vector };
    assert!(
      this.is_coordinate_realizable(complex.cells()),
      "Edge lengths are not coordinate realizable."
    );
    this
  }
  pub fn try_new(vector: Vector, complex: &Complex) -> Option<Self> {
    let this = Self { vector };
    this
      .is_coordinate_realizable(complex.cells())
      .then_some(this)
  }
  pub fn new_unchecked(vector: Vector) -> Self {
    Self { vector }
  }
  pub fn standard(dim: usize) -> MeshLengths {
    let vector = SimplexLengths::standard(dim).into_vector();
    Self::new_unchecked(vector)
  }

  pub fn nedges(&self) -> usize {
    self.vector.len()
  }
  pub fn length(&self, iedge: EdgeIdx) -> f64 {
    self[iedge]
  }
  pub fn vector(&self) -> &Vector {
    &self.vector
  }
  pub fn vector_mut(&mut self) -> &mut Vector {
    &mut self.vector
  }
  pub fn into_vector(self) -> Vector {
    self.vector
  }
  pub fn iter(
    &self,
  ) -> na::iter::MatrixIter<
    '_,
    f64,
    na::Dyn,
    na::Const<1>,
    na::VecStorage<f64, na::Dyn, na::Const<1>>,
  > {
    self.vector.iter()
  }

  /// The mesh width $h_max$, equal to the largest diameter of all cells.
  pub fn mesh_width_max(&self) -> f64 {
    self
      .iter()
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .copied()
      .unwrap()
  }

  /// By convexity the smallest length of a line inside a simplex is the length
  /// one of the edges.
  pub fn mesh_width_min(&self) -> f64 {
    self
      .iter()
      .min_by(|a, b| a.partial_cmp(b).unwrap())
      .copied()
      .unwrap()
  }

  /// The shape regularity measure $rho$ of the whole mesh, which is the largest
  /// shape regularity measure over all cells.
  pub fn shape_regularity_measure(&self, topology: &Complex) -> f64 {
    topology
      .cells()
      .handle_iter()
      .map(|cell| self.simplex_lengths(cell).shape_regularity_measure())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  pub fn simplex_lengths(&self, simplex: SimplexRef) -> SimplexLengths {
    let lengths = simplex
      .edges()
      .map(|edge| self.length(edge.kidx()))
      .collect_vec()
      .into();
    // SAFETY: Already checked realizability.
    SimplexLengths::new_unchecked(lengths, simplex.dim())
  }

  pub fn is_coordinate_realizable(&self, skeleton: SkeletonRef) -> bool {
    skeleton
      .handle_par_iter()
      .all(|simp| self.simplex_lengths(simp).is_coordinate_realizable())
  }

  /// Whether this could be the edge-length geometry of `topology`: one length
  /// per edge, nothing more.
  pub fn is_compatible_with(&self, topology: &Complex) -> bool {
    self.nedges() == topology.edges().len()
  }

  pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
    crate::io::cbor::save_cbor(self, path)
  }
  pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
    crate::io::cbor::load_cbor(path)
  }
}
impl std::ops::Index<EdgeIdx> for MeshLengths {
  type Output = f64;
  fn index(&self, iedge: EdgeIdx) -> &Self::Output {
    &self.vector[iedge]
  }
}

pub type MetricComplex = (Complex, MeshLengths);
pub fn standard_metric_complex(dim: Dim) -> MetricComplex {
  let topology = Complex::standard(dim);
  let lengths = MeshLengths::standard(dim);
  (topology, lengths)
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::gen::cartesian::CartesianMeshInfo;

  /// Coordinates and edge lengths read uniformly as data on simplices:
  /// coords (grade 0) return a column view, lengths (grade 1) a scalar ref.
  #[test]
  fn geometry_as_simplex_data() {
    let (topology, coords) = CartesianMeshInfo::new_unit(2, 2).compute_coord_complex();
    let lengths = coords.to_edge_lengths(&topology);

    assert_eq!(SkeletonData::grade(&coords), 0);
    assert_eq!(SkeletonData::grade(&lengths), 1);

    for vertex in topology.vertices().handle_iter() {
      assert_eq!(coords.at_ref(vertex), coords.coord(vertex.kidx()));
    }
    for edge in topology.edges().handle_iter() {
      let [vi, vj] = edge.simplex().clone().try_into().unwrap();
      let expected = (coords.coord(vj) - coords.coord(vi)).norm();
      assert_eq!(*lengths.at_ref(edge), expected);
    }
  }
}
