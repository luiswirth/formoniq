use super::{
  simplex::{SimplexCoords, SimplexRefExt},
  Coord, CoordRef,
};
use crate::{
  geometry::metric::{mesh::MeshLengths, Geometry},
  topology::{
    data::SkeletonData,
    handle::{KSimplexIdx, SimplexRef},
    {complex::Complex, simplex::Simplex, VertexIdx},
  },
  Dim,
};

use crate::linalg::{Matrix, Vector};
use gramian::RiemannianMetric;

use itertools::Itertools;

#[cfg(feature = "serde")]
use std::{io, path::Path};

/// The coordinates of the vertices of the mesh.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshCoords {
  matrix: Matrix,
}

/// An embedding *induces* a metric: the Gramian of the cell's spanning vectors.
///
/// This impl lives here, not in the metric layer, and that is the whole point --
/// coordinates know about the metric they induce, the metric knows nothing of
/// coordinates (invariant 2).
impl Geometry for MeshCoords {
  fn cell_metric(&self, cell: SimplexRef) -> RiemannianMetric {
    RiemannianMetric::new(cell.coord_simplex(self).metric_tensor())
  }
}

impl MeshCoords {
  pub fn standard(ndim: Dim) -> Self {
    SimplexCoords::standard(ndim).vertices
  }
  pub fn new(matrix: Matrix) -> Self {
    Self { matrix }
  }

  pub fn matrix(&self) -> &Matrix {
    &self.matrix
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

  /// Whether this embedding could be the geometry of `topology`: one column
  /// per vertex, nothing more (the two carry no other shared invariant to
  /// check, since embedding and topology are otherwise fully independent
  /// inputs).
  pub fn is_compatible_with(&self, topology: &Complex) -> bool {
    self.nvertices() == topology.vertices().len()
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

/// Vertex coordinates are grade-0 data on the mesh, stored as the columns of a
/// matrix: `at` returns a column *view*, not an owned point.
impl SkeletonData for MeshCoords {
  type Item<'a> = CoordRef<'a>;
  fn grade(&self) -> Dim {
    0
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
    self.matrix.nrows()
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

  pub fn coord_iter_mut(
    &mut self,
  ) -> na::iter::ColumnIterMut<'_, f64, na::Dyn, na::Dyn, na::VecStorage<f64, na::Dyn, na::Dyn>> {
    self.matrix.column_iter_mut()
  }

  pub fn to_edge_lengths(&self, topology: &Complex) -> MeshLengths {
    // A 0-manifold is a discrete set of points: its 1-skeleton is empty, so the
    // edge-length representation of its (trivial, 0-dimensional) geometry is the
    // empty vector.
    if topology.dim() == 0 {
      return MeshLengths::new_unchecked(Vector::zeros(0));
    }
    let edges = topology.edges();
    let mut edge_lengths = Vector::zeros(edges.len());
    for (iedge, edge) in edges.handle_iter().enumerate() {
      let [vi, vj] = edge.simplex().clone().try_into().unwrap();
      let length = (self.coord(vj) - self.coord(vi)).norm();
      edge_lengths[iedge] = length;
    }
    // SAFETY: Edge Lengths come from a coordinate realizations.
    MeshLengths::new_unchecked(edge_lengths)
  }
}

impl MeshCoords {
  pub fn embed_euclidean(mut self, dim: Dim) -> MeshCoords {
    let old_dim = self.matrix.nrows();
    self.matrix = self.matrix.insert_rows(old_dim, dim - old_dim, 0.0);
    self
  }
}

impl MeshCoords {
  pub fn find_cell_containing<'a>(
    &self,
    topology: &'a Complex,
    coord: CoordRef,
  ) -> Option<SimplexRef<'a>> {
    topology
      .cells()
      .handle_iter()
      .find(|cell| cell.coord_simplex(self).is_global_inside(coord))
  }
}

/// Relabel vertices so they are contiguous and fully used: drop any vertex
/// absent from the cells and renumber the rest to $0..m$, closing gaps. The
/// renumbering is monotone (preserves increasing vertex order within cells) and
/// applied in lockstep to the coordinates. Idempotent on an already-gapless
/// mesh. Meant for external imports (e.g. Gmsh) before building a [`Complex`].
pub fn close_vertex_gaps(cells: Vec<Simplex>, coords: &MeshCoords) -> (Vec<Simplex>, MeshCoords) {
  let mut used: Vec<VertexIdx> = cells.iter().flat_map(|cell| cell.iter()).collect();
  used.sort_unstable();
  used.dedup();

  let relabel = |v: VertexIdx| used.binary_search(&v).expect("vertex is used");
  let cells = cells
    .into_iter()
    .map(|cell| Simplex::new(cell.iter().map(relabel).collect()))
    .collect();

  let columns: Vec<_> = used.iter().map(|&v| coords.coord(v).view()).collect();
  let coords = MeshCoords::new(Matrix::from_columns(&columns));
  (cells, coords)
}

pub fn standard_coord_complex(dim: Dim) -> (Complex, MeshCoords) {
  let topology = Complex::standard(dim);

  let coords = topology
    .vertices()
    .handle_iter()
    .map(|v| v.kidx())
    .map(|v| {
      let mut vec = Vector::zeros(dim);
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
