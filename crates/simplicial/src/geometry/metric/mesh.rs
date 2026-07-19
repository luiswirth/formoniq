use super::{simplex::SimplexLengthsSq, EdgeIdx};
use crate::{
  topology::{
    complex::Complex,
    data::SkeletonData,
    handle::{KSimplexIdx, SimplexRef, SkeletonRef},
    role::{Cell, Edge},
  },
  Dim,
};

use crate::linalg::Vector;
use gramian::{CausalType, Metric};

use itertools::Itertools;
use rayon::iter::ParallelIterator;

#[cfg(feature = "serde")]
use std::{io, path::Path};

/// The signed squared lengths of the edges of the mesh: the Regge geometry,
/// on any metric signature.
///
/// One scalar per edge is the whole geometry of the simplicial manifold --
/// Regge's "general relativity without coordinates" -- and the squared length
/// is the primitive that keeps it signature-blind: positive spacelike, zero
/// null, negative timelike, exactly the [`Gramian::norm_sq`] convention. A
/// Riemannian mesh is the all-positive, Euclidean-realizable corner; a
/// Lorentzian simplicial spacetime is the same data with causal signs.
///
/// [`Gramian::norm_sq`]: gramian::Gramian::norm_sq
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshLengthsSq {
  vector: Vector,
}

/// Squared edge lengths are grade-1 data on the mesh: one scalar per edge.
impl SkeletonData for MeshLengthsSq {
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
impl MeshLengthsSq {
  /// The invariant is per-cell non-degeneracy of the induced metric, checked
  /// over the cell skeleton; the signature is whatever the data describes.
  pub fn new(vector: Vector, complex: &Complex) -> Self {
    let this = Self { vector };
    assert!(
      this.is_nondegenerate(complex.cells().get()),
      "Squared edge lengths must induce non-degenerate cell metrics."
    );
    this
  }
  pub fn try_new(vector: Vector, complex: &Complex) -> Option<Self> {
    let this = Self { vector };
    this.is_nondegenerate(complex.cells().get()).then_some(this)
  }
  pub fn new_unchecked(vector: Vector) -> Self {
    Self { vector }
  }
  pub fn standard(dim: usize) -> MeshLengthsSq {
    let vector = SimplexLengthsSq::standard(dim).into_vector();
    Self::new_unchecked(vector)
  }

  pub fn nedges(&self) -> usize {
    self.vector.len()
  }
  /// The signed squared length of an edge: the Regge primitive, its sign the
  /// causal character.
  pub fn length_sq(&self, iedge: EdgeIdx) -> f64 {
    self[iedge]
  }
  /// The magnitude $sqrt(abs(s))$ of an edge. On an indefinite metric this is
  /// meaningful only together with [`Self::causal_type`]; it is never NaN.
  pub fn length(&self, iedge: EdgeIdx) -> f64 {
    self[iedge].abs().sqrt()
  }
  /// The causal character of an edge: the sign of its squared length.
  pub fn causal_type(&self, iedge: EdgeIdx) -> CausalType {
    CausalType::from_norm_sq(self[iedge])
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

  /// The mesh width $h_max$: the largest edge magnitude over the mesh, which
  /// on a Riemannian geometry is the largest cell diameter. On an indefinite
  /// one it is a mesh scale, not a distance.
  pub fn mesh_width_max(&self) -> f64 {
    self
      .iter()
      .map(|s| s.abs())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
      .sqrt()
  }

  /// The mesh width $h_min$: the smallest edge magnitude. On a Riemannian
  /// geometry, by convexity, the smallest distance inside any cell is along
  /// one of its edges.
  pub fn mesh_width_min(&self) -> f64 {
    self
      .iter()
      .map(|s| s.abs())
      .min_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
      .sqrt()
  }

  /// The shape regularity measure $rho$ of the whole mesh, which is the largest
  /// shape regularity measure over all cells.
  pub fn shape_regularity(&self, topology: &Complex) -> f64 {
    topology
      .cells()
      .handle_iter()
      .map(|cell| self.simplex_lengths_sq(cell.get()).shape_regularity())
      .max_by(|a, b| a.partial_cmp(b).unwrap())
      .unwrap()
  }

  pub fn simplex_lengths_sq(&self, simplex: SimplexRef) -> SimplexLengthsSq {
    let lengths_sq = simplex
      .edges()
      .map(|edge| edge.length_sq(self))
      .collect_vec()
      .into();
    // SAFETY: Non-degeneracy was checked at construction.
    SimplexLengthsSq::new_unchecked(lengths_sq, simplex.dim())
  }

  /// The intrinsic metric tensor of *any* simplex, of any grade: the Gramian
  /// of that simplex's own edges. Geometry is defined on the whole skeleton,
  /// not only the cells -- an edge has a length, a facet has an area, a hinge
  /// has a metric -- because every subsimplex's metric is the restriction of
  /// any containing cell's, equivalently the Gramian built from its edges. A
  /// containing cell need not be consulted: the edge lengths are shared, so
  /// every cell induces the same metric on a shared face, and this is well
  /// defined from the edge data alone.
  ///
  /// This is the metric, not the chart. Only a top-dimensional simplex carries
  /// a [`Chart`](crate::atlas::Chart) -- a frame in which to express a section
  /// -- but *every* simplex has a metric to measure it by.
  pub fn simplex_metric(&self, simplex: SimplexRef) -> Metric {
    self.simplex_lengths_sq(simplex).metric()
  }

  /// The flat metric tensor of a cell: [`Self::simplex_metric`] at top
  /// dimension, the form the assembly path consumes.
  pub fn cell_metric(&self, cell: Cell) -> Metric {
    self.simplex_metric(cell.get())
  }

  /// The volume of any simplex, of any grade and signature:
  /// $vol(hat(K)) sqrt(abs(det g))$ read off its own edge lengths. An edge's
  /// length, a facet's area, a cell's volume -- one formula, total over the
  /// skeleton.
  pub fn simplex_volume(&self, simplex: SimplexRef) -> f64 {
    self.simplex_lengths_sq(simplex).vol()
  }

  /// Whether every simplex of the skeleton has a non-degenerate induced
  /// metric: the constructor invariant, read at any grade.
  pub fn is_nondegenerate(&self, skeleton: SkeletonRef) -> bool {
    skeleton
      .handle_par_iter()
      .all(|simp| !self.simplex_lengths_sq(simp).is_degenerate())
  }

  /// Whether the mesh is realizable by a Euclidean point configuration cell
  /// by cell: the Riemannian ($q = 0$) corner of the signature range.
  pub fn is_coordinate_realizable(&self, skeleton: SkeletonRef) -> bool {
    skeleton
      .handle_par_iter()
      .all(|simp| self.simplex_lengths_sq(simp).is_coordinate_realizable())
  }

  /// Whether this could be the edge geometry of `topology`: one squared
  /// length per edge, nothing more.
  pub fn is_compatible_with(&self, topology: &Complex) -> bool {
    self.nedges() == topology.edges().len()
  }

  #[cfg(feature = "serde")]
  pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
    crate::io::cbor::save_cbor(self, path)
  }
  #[cfg(feature = "serde")]
  pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
    crate::io::cbor::load_cbor(path)
  }
}
impl std::ops::Index<EdgeIdx> for MeshLengthsSq {
  type Output = f64;
  fn index(&self, iedge: EdgeIdx) -> &Self::Output {
    &self.vector[iedge]
  }
}

/// Geometry read on a topology witness: the signed squared length an [`Edge`]
/// proof keys in the grade-1 Regge data, `edge.length_sq(&lengths_sq)`.
/// Reaches down from the metric side -- the topology never learns of metrics.
pub trait EdgeRefExt {
  /// The signed squared length: the Regge primitive.
  fn length_sq(self, lengths_sq: &MeshLengthsSq) -> f64;
  /// The magnitude $sqrt(abs(s))$; see [`MeshLengthsSq::length`].
  fn length(self, lengths_sq: &MeshLengthsSq) -> f64;
  /// The causal character of the edge.
  fn causal_type(self, lengths_sq: &MeshLengthsSq) -> CausalType;
}
impl EdgeRefExt for Edge<'_> {
  fn length_sq(self, lengths_sq: &MeshLengthsSq) -> f64 {
    lengths_sq.length_sq(self.kidx())
  }
  fn length(self, lengths_sq: &MeshLengthsSq) -> f64 {
    lengths_sq.length(self.kidx())
  }
  fn causal_type(self, lengths_sq: &MeshLengthsSq) -> CausalType {
    lengths_sq.causal_type(self.kidx())
  }
}

pub type MetricComplex = (Complex, MeshLengthsSq);
pub fn standard_metric_complex(dim: Dim) -> MetricComplex {
  let topology = Complex::standard(dim);
  let lengths_sq = MeshLengthsSq::standard(dim);
  (topology, lengths_sq)
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::gen::cartesian::CartesianGrid;

  /// Coordinates and squared edge lengths read uniformly as data on simplices:
  /// coords (grade 0) return a column view, squared lengths (grade 1) a
  /// scalar ref.
  #[test]
  fn geometry_as_simplex_data() {
    let (topology, coords) = CartesianGrid::new_unit(2, 2).triangulate();
    let lengths_sq = coords.to_edge_lengths_sq(&topology);

    assert_eq!(SkeletonData::grade(&coords), 0);
    assert_eq!(SkeletonData::grade(&lengths_sq), 1);

    for vertex in topology.vertices().handle_iter() {
      assert_eq!(coords.at_ref(vertex.get()), coords.coord(vertex.kidx()));
    }
    for edge in topology.edges().handle_iter() {
      let [vi, vj] = edge.simplex().clone().try_into().unwrap();
      let expected = (coords.coord(vj) - coords.coord(vi)).norm_squared();
      assert_eq!(*lengths_sq.at_ref(edge.get()), expected);
    }
  }
}
