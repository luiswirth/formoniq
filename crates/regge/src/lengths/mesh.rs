use super::{EdgeIdx, LengthsSq, simplex::SimplexLengthsSq};
use simplicial::{
  Dim,
  topology::{
    complex::Complex,
    data::SkeletonData,
    handle::{KSimplexIdx, SimplexRef, SkeletonRef},
    role::{Cell, Edge},
  },
};

use metric::{CausalType, Metric};
use simplicial::linalg::Vector;

use itertools::Itertools;
use rayon::iter::ParallelIterator;

#[cfg(feature = "serde")]
use std::{io, path::Path};

/// The signed squared lengths of the edges of the mesh: the Regge geometry,
/// on any metric signature.
///
/// One scalar per edge is the whole geometry of the simplicial manifold,
/// Regge's "general relativity without coordinates", and the squared length
/// is the primitive that keeps it signature-blind: positive spacelike, zero
/// null, negative timelike, exactly the [`Metric::norm_sq`] convention. A
/// Riemannian mesh is the all-positive, Euclidean-realizable corner; a
/// Lorentzian simplicial spacetime is the same data with causal signs.
///
/// [`Metric::norm_sq`]: metric::Metric::norm_sq
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshLengthsSq {
  vector: Vector,
}

/// Squared edge lengths are grade-1 data on the mesh: one scalar per edge.
impl SkeletonData for MeshLengthsSq {
  type Item<'a> = &'a f64;
  fn grade(&self) -> Dim {
    Dim::ONE
  }
  fn len(&self) -> usize {
    self.vector.len()
  }
  fn at(&self, kidx: KSimplexIdx) -> &f64 {
    &self.vector[kidx]
  }
}
impl LengthsSq for MeshLengthsSq {
  fn lengths_sq(&self) -> &Vector {
    &self.vector
  }
}
impl MeshLengthsSq {
  /// The geometry given by signed squared lengths on the 1-skeleton, one per
  /// edge in colex order.
  ///
  /// The hypothesis is per-cell non-degeneracy of the induced metric
  /// ([`Self::is_valid`]), and it is the caller's to hold. It usually holds
  /// upstream by construction: an embedding of a mesh whose cells are
  /// non-degenerate induces it, and a refinement or a subcomplex inherits it
  /// from the geometry it was cut from. Where it is genuinely in question, on a
  /// mesh file or on lengths a user supplied, [`Self::new_checked`] asks.
  ///
  /// Verifying costs a Cayley-Menger determinant per cell, a sweep over the
  /// whole mesh. Unchecked in every build profile, debug included: a
  /// constructor whose cost and whose panics depend on how the code was
  /// compiled is a worse trap than an unchecked one.
  ///
  /// The complex is not an argument, deliberately. The lengths are a vector
  /// over the 1-skeleton and mean nothing else; only the hypothesis is
  /// relational, which is why the complex enters at [`Self::is_valid`] and at
  /// the checked constructor.
  pub fn new(vector: Vector) -> Self {
    Self { vector }
  }
  /// The geometry, or `None` if some cell degenerates: the constructor that
  /// verifies what [`Self::new`] takes on contract.
  ///
  /// Holding the result *is* the proof that every cell metric is
  /// non-degenerate, so an import checks here once instead of every element
  /// computation re-asking.
  pub fn new_checked(vector: Vector, complex: &Complex) -> Option<Self> {
    let this = Self::new(vector);
    this.is_valid(complex).then_some(this)
  }
  /// Whether these lengths are a geometry on `complex`: every cell metric
  /// non-degenerate, which is exactly the contract [`Self::new`] takes on trust
  /// and [`Self::new_checked`] verifies.
  ///
  /// The signature is whatever the data describes; non-degeneracy is the
  /// hypothesis, definiteness is not.
  pub fn is_valid(&self, complex: &Complex) -> bool {
    self.is_nondegenerate(complex.cells().get())
  }
  /// The unit simplex as a one-cell mesh, non-degenerate by construction.
  pub fn unit(dim: impl Into<Dim>) -> MeshLengthsSq {
    let dim = dim.into();
    let vector = SimplexLengthsSq::unit(dim).into_vector();
    Self::new(vector)
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

  /// The mesh width $h_max$: the largest edge magnitude over the mesh, which
  /// on a Riemannian geometry is the largest cell diameter. On an indefinite
  /// one it is a mesh scale, not a distance.
  pub fn mesh_width_max(&self) -> f64 {
    self.max_length()
  }

  /// The mesh width $h_min$: the smallest edge magnitude. On a Riemannian
  /// geometry, by convexity, the smallest distance inside any cell is along
  /// one of its edges.
  pub fn mesh_width_min(&self) -> f64 {
    self.min_length()
  }

  /// The mean edge magnitude $h_"mean"$: the mesh's characteristic local
  /// length, as distinct from $h_max$, which is set by its single worst edge,
  /// and from the extent of an embedding, which is the object's global size.
  ///
  /// Zero on a mesh with no edges (a point cloud), where there is no local
  /// length to speak of.
  pub fn mesh_width_mean(&self) -> f64 {
    if self.nedges() == 0 {
      return 0.0;
    }
    self.iter().map(|s| s.abs().sqrt()).sum::<f64>() / self.nedges() as f64
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
    // A face of a non-degenerate cell is non-degenerate, and the cells were
    // established as such when this geometry was built.
    SimplexLengthsSq::new(lengths_sq, simplex.dim())
  }

  /// The intrinsic metric tensor of any simplex, of any grade: the Gramian
  /// of that simplex's own edges. Geometry is defined on the whole skeleton,
  /// not only the cells: an edge has a length, a facet has an area, a hinge
  /// has a metric, because every subsimplex's metric is the restriction of
  /// any containing cell's, equivalently the Gramian built from its edges. A
  /// containing cell need not be consulted: the edge lengths are shared, so
  /// every cell induces the same metric on a shared face, and this is well
  /// defined from the edge data alone.
  ///
  /// This is the metric, not the chart. Only a top-dimensional simplex carries
  /// a [`Chart`](simplicial::atlas::Chart), a frame in which to express a section
  ///, but every simplex has a metric to measure it by.
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
  /// length, a facet's area, a cell's volume, one formula, total over the
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

  /// The causal census of the edges: how many are timelike, null, spacelike
  /// under this (signed) Regge geometry.
  ///
  /// On a Riemannian mesh every edge is spacelike and the census is trivial.
  /// It carries content only on an indefinite signature, where it is the
  /// well-posedness diagnostic of spacetime FEEC: a null edge degenerates the
  /// indefinite $L^2$ pairing on Whitney 1-forms exactly (the mass rank-deficient
  /// by the null count), so a Lorentzian mesh is causally generic
  /// ([`Self::is_causally_generic`]) precisely when the null count is zero.
  pub fn causal_census(&self, topology: &Complex) -> CausalCensus {
    let mut census = CausalCensus::default();
    for edge in topology.edges().handle_iter() {
      *match edge.causal_type(self) {
        CausalType::Timelike => &mut census.timelike,
        CausalType::Null => &mut census.null,
        CausalType::Spacelike => &mut census.spacelike,
      } += 1;
    }
    census
  }

  /// Whether no edge is lightlike: the well-posedness condition of spacetime
  /// FEEC, since a null edge makes the indefinite Whitney 1-form mass singular.
  /// Always true on a Riemannian mesh.
  pub fn is_causally_generic(&self, topology: &Complex) -> bool {
    self.causal_census(topology).null == 0
  }

  /// Whether this could be the edge geometry of `topology`: one squared
  /// length per edge, nothing more.
  pub fn is_compatible_with(&self, topology: &Complex) -> bool {
    self.nedges() == topology.edges().len()
  }

  #[cfg(feature = "serde")]
  pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
    simplicial::io::cbor::save_cbor(self, path)
  }
  #[cfg(feature = "serde")]
  pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
    simplicial::io::cbor::load_cbor(path)
  }
}
impl std::ops::Index<EdgeIdx> for MeshLengthsSq {
  type Output = f64;
  fn index(&self, iedge: EdgeIdx) -> &Self::Output {
    &self.vector[iedge]
  }
}

/// The count of edges of each causal character in a mesh: the output of
/// [`MeshLengthsSq::causal_census`]. On a Lorentzian spacetime the `null` field
/// is the obstruction to well-posedness. On a Riemannian mesh every edge falls
/// in `spacelike`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CausalCensus {
  pub timelike: usize,
  pub null: usize,
  pub spacelike: usize,
}
impl std::fmt::Display for CausalCensus {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{} timelike, {} null, {} spacelike",
      self.timelike, self.null, self.spacelike
    )
  }
}

/// Geometry read on a topology witness: the signed squared length an [`Edge`]
/// proof keys in the grade-1 Regge data, `edge.length_sq(&lengths_sq)`.
/// Reaches down from the metric side, the topology never learns of metrics.
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
