use super::{Coord, CoordRef, simplex::SimplexRefExt};
use crate::lengths::{CellGramians, mesh::MeshLengthsSq};
use simplicial::{
  Dim,
  topology::{
    data::SkeletonData,
    handle::{KSimplexIdx, SimplexRef},
    role::{Cell, Vertex, roles},
    {VertexIdx, complex::Complex},
  },
};

use metric::Metric;
use simplicial::linalg::{Matrix, Vector};

use itertools::Itertools;

#[cfg(feature = "serde")]
use std::{io, path::Path};

/// The coordinates of the vertices of the mesh: an embedding into the flat
/// pseudo-Euclidean space $RR^(p, q)$, carried as the vertex columns together
/// with the ambient inner product. The Euclidean ambient ($q = 0$,
/// [`MeshCoords::new`]) is the default and one signature among all: a
/// spacetime mesh embeds into Minkowski space through
/// [`MeshCoords::with_ambient`], on the very same type.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshCoords {
  matrix: Matrix,
  /// The inner product of the ambient space the vertices live in.
  ambient: Metric,
}

impl MeshCoords {
  /// The metric an embedding induces on a cell: the pullback $J^top eta J$
  /// of the ambient inner product along the cell's spanning vectors, the
  /// first fundamental form, of whatever signature the ambient carries.
  ///
  /// This lives here, not in the metric layer, and that is the whole point:
  /// coordinates know about the metric they induce, the metric knows nothing
  /// of coordinates (invariant 2). An embedding reaches the intrinsic engine
  /// as a source, it converts to edge lengths ([`Self::to_edge_lengths_sq`])
  /// or per-cell metrics ([`Self::to_cell_gramians`]) at the boundary of the
  /// API. The core never asks an embedding for anything.
  pub fn cell_metric(&self, cell: Cell) -> Metric {
    self.simplex_metric(cell.get())
  }

  /// The metric of any simplex, an edge's length, a facet's area, the
  /// flat metric of a cell, as the Gramian of that simplex's own spanning
  /// vectors under the ambient inner product.
  ///
  /// The embedding counterpart of
  /// [`MeshLengthsSq::simplex_metric`](crate::MeshLengthsSq::simplex_metric),
  /// and total over every grade for the same reason (invariant 2): geometry is
  /// defined on every simplex, not only the cells, so the boundary trace and the
  /// metric of a subskeleton simplex are well defined from the shared
  /// coordinates with no containing cell consulted.
  pub fn simplex_metric(&self, simplex: SimplexRef) -> Metric {
    self
      .ambient
      .pullback(&simplex.coord_simplex(self).spanning_vectors())
  }

  /// Materialize the per-cell metrics this embedding induces.
  pub fn to_cell_gramians(&self, topology: &Complex) -> CellGramians {
    let metrics = topology
      .cells()
      .handle_iter()
      .map(|cell| self.cell_metric(cell))
      .collect();
    CellGramians::new(topology.dim(), metrics)
  }
}

impl MeshCoords {
  pub fn unit(ndim: impl Into<Dim>) -> Self {
    Self::new(simplicial::atlas::unit_vertices(ndim.into()))
  }
  /// Vertices of an embedding into Euclidean space: the ambient inner product
  /// is the standard one.
  pub fn new(matrix: Matrix) -> Self {
    let ambient = Metric::euclidean(matrix.nrows());
    Self::with_ambient(matrix, ambient)
  }
  /// Vertices of an embedding into the flat pseudo-Euclidean space the given
  /// ambient Gramian describes, e.g. [`Metric::minkowski`] for a mesh of a
  /// Lorentzian spacetime.
  pub fn with_ambient(matrix: Matrix, ambient: Metric) -> Self {
    assert_eq!(
      ambient.dim(),
      matrix.nrows(),
      "Ambient inner product must match the coordinate dimension."
    );
    Self { matrix, ambient }
  }

  pub fn matrix(&self) -> &Matrix {
    &self.matrix
  }
  /// The inner product of the ambient space.
  pub fn ambient(&self) -> &Metric {
    &self.ambient
  }
  pub fn matrix_mut(&mut self) -> &mut Matrix {
    &mut self.matrix
  }
  pub fn into_matrix(self) -> Matrix {
    self.matrix
  }

  pub fn swap_coords(&mut self, icol: usize, jcol: usize) {
    self.matrix.swap_columns(icol, jcol);
  }

  /// The coordinates of the named vertices, in the order named: the columns of
  /// this embedding at those indices, under the same ambient inner product.
  ///
  /// The geometric half of any renumbering of the vertices, the combinatorial
  /// half being whatever produced the indices, a
  /// [`VertexRelabelling`](simplicial::topology::relabel::VertexRelabelling)
  /// closing the gaps of an import or a subcomplex naming its parent's.
  pub fn select(&self, vertices: &[VertexIdx]) -> MeshCoords {
    let columns: Vec<_> = vertices
      .iter()
      .map(|&v| self.coord(v).into_view())
      .collect();
    Self::with_ambient(Matrix::from_columns(&columns), self.ambient.clone())
  }

  /// Whether this embedding could be the geometry of `topology`: one column
  /// per vertex, nothing more (the two carry no other shared invariant to
  /// check, since embedding and topology are otherwise fully independent
  /// inputs).
  pub fn is_compatible_with(&self, topology: &Complex) -> bool {
    self.nvertices() == topology.vertices().len()
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

/// Vertex coordinates are grade-0 data on the mesh, stored as the columns of a
/// matrix: `at` returns a column view, not an owned point.
impl SkeletonData for MeshCoords {
  type Item<'a> = CoordRef<'a>;
  fn grade(&self) -> Dim {
    Dim::ZERO
  }
  fn len(&self) -> usize {
    self.nvertices()
  }
  fn at(&self, kidx: KSimplexIdx) -> CoordRef<'_> {
    CoordRef::new(self.matrix.column(kidx))
  }
}

impl From<Matrix> for MeshCoords {
  fn from(matrix: Matrix) -> Self {
    Self::new(matrix)
  }
}

impl From<&[Coord]> for MeshCoords {
  fn from(vectors: &[Coord]) -> Self {
    let columns: Vec<_> = vectors.iter().map(Coord::vector).cloned().collect();
    Self::new(Matrix::from_columns(&columns))
  }
}

impl MeshCoords {
  pub fn dim(&self) -> Dim {
    self.matrix.nrows().into()
  }
  pub fn nvertices(&self) -> usize {
    self.matrix.ncols()
  }

  pub fn coord(&self, ivertex: VertexIdx) -> CoordRef<'_> {
    CoordRef::new(self.matrix.column(ivertex))
  }

  pub fn coord_iter(&self) -> impl ExactSizeIterator<Item = CoordRef<'_>> {
    self.matrix.column_iter().map(CoordRef::new)
  }

  /// The Regge geometry this embedding realizes: the signed squared length
  /// of each edge under the ambient inner product, of whatever signature the
  /// ambient carries, an embedding into Minkowski space yields Lorentzian
  /// Regge data, causal signs included.
  ///
  /// A 0-manifold is a discrete set of points: its 1-skeleton is empty, so its
  /// geometry is the empty vector, which the total accessor gives rather than
  /// a guard.
  ///
  /// Non-degeneracy is inherited from the embedding rather than established
  /// here: a coordinate realization with a flat or repeated cell yields a
  /// degenerate geometry, and this conversion faithfully reports it. A caller
  /// holding coordinates of unknown provenance, an imported mesh file above
  /// all, discharges [`MeshLengthsSq::is_valid`] on the result.
  pub fn to_edge_lengths_sq(&self, topology: &Complex) -> MeshLengthsSq {
    let edges = topology.skeleton(Dim::ONE);
    let mut edge_lengths_sq = Vector::zeros(edges.len());
    for (iedge, edge) in edges.handle_iter().enumerate() {
      let (vi, vj) = edge.role::<roles::Edge>().endpoints();
      edge_lengths_sq[iedge] = self.ambient.norm_sq(&(vj.coord(self) - vi.coord(self)));
    }
    MeshLengthsSq::new(edge_lengths_sq)
  }
}

impl MeshCoords {
  /// Pad the ambient space with additional Euclidean axes: the vertices gain
  /// zero coordinates, the ambient inner product an identity block.
  pub fn embed_euclidean(mut self, dim: impl Into<Dim>) -> MeshCoords {
    let dim = dim.into();
    let old_dim = self.matrix.nrows();
    let extra = (dim - old_dim).index();
    self.matrix = self.matrix.insert_rows(old_dim, extra, 0.0);
    let mut ambient = self
      .ambient
      .matrix()
      .clone()
      .insert_rows(old_dim, extra, 0.0)
      .insert_columns(old_dim, extra, 0.0);
    for i in old_dim..dim.index() {
      ambient[(i, i)] = 1.0;
    }
    self.ambient = Metric::new(self.ambient.variance(), ambient);
    self
  }
}

impl MeshCoords {
  pub fn find_cell_containing<'a>(
    &self,
    topology: &'a Complex,
    coord: CoordRef,
  ) -> Option<Cell<'a>> {
    topology
      .cells()
      .handle_iter()
      .find(|cell| cell.coord_simplex(self).is_global_inside(coord))
  }
}

/// Geometry read on a topology witness: the coordinate a [`Vertex`] proof
/// names in an embedding, `vertex.coord(&coords)`. Reaches down from the
/// coord side, the topology never learns of embeddings.
pub trait VertexRefExt {
  fn coord<'c>(self, coords: &'c MeshCoords) -> CoordRef<'c>;
}
impl VertexRefExt for Vertex<'_> {
  fn coord<'c>(self, coords: &'c MeshCoords) -> CoordRef<'c> {
    coords.coord(self.kidx())
  }
}

pub fn unit_coord_complex(dim: impl Into<Dim>) -> (Complex, MeshCoords) {
  let dim = dim.into();
  let topology = Complex::unit(dim);

  let coords = topology
    .vertices()
    .handle_iter()
    .map(|v| v.kidx())
    .map(|v| {
      let mut vec = Vector::zeros(dim.index());
      if v > 0 {
        vec[v - 1] = 1.0;
      }
      vec
    })
    .collect_vec();
  let coords = Matrix::from_columns(&coords);
  let coords = MeshCoords::new(coords);

  (topology, coords)
}
