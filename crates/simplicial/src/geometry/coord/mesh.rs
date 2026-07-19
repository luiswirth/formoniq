use super::{simplex::SimplexRefExt, Coord, CoordRef};
use crate::{
  geometry::metric::{mesh::MeshLengthsSq, Geometry},
  topology::{
    data::SkeletonData,
    handle::KSimplexIdx,
    role::{Cell, Vertex},
    {complex::Complex, simplex::Simplex, VertexIdx},
  },
  Dim,
};

use crate::linalg::{Matrix, Vector};
use gramian::{Gramian, Metric};

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
  ambient: Gramian,
}

/// An embedding *induces* a metric: the pullback $J^top eta J$ of the ambient
/// inner product along the cell's spanning vectors -- the first fundamental
/// form, of whatever signature the ambient carries.
///
/// This impl lives here, not in the metric layer, and that is the whole point --
/// coordinates know about the metric they induce, the metric knows nothing of
/// coordinates (invariant 2).
impl Geometry for MeshCoords {
  fn cell_metric(&self, cell: Cell) -> Metric {
    Metric::new(
      self
        .ambient
        .pullback(&cell.coord_simplex(self).spanning_vectors()),
    )
  }
}

impl MeshCoords {
  pub fn standard(ndim: Dim) -> Self {
    Self::new(crate::atlas::ref_vertices(ndim))
  }
  /// Vertices of an embedding into Euclidean space: the ambient inner product
  /// is the standard one.
  pub fn new(matrix: Matrix) -> Self {
    let ambient = Gramian::standard(matrix.nrows());
    Self::with_ambient(matrix, ambient)
  }
  /// Vertices of an embedding into the flat pseudo-Euclidean space the given
  /// ambient Gramian describes -- e.g. [`Gramian::minkowski`] for a mesh of a
  /// Lorentzian spacetime.
  pub fn with_ambient(matrix: Matrix, ambient: Gramian) -> Self {
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
  pub fn ambient(&self) -> &Gramian {
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

  /// The Regge geometry this embedding realizes: the signed squared length
  /// of each edge under the ambient inner product, of whatever signature the
  /// ambient carries -- an embedding into Minkowski space yields Lorentzian
  /// Regge data, causal signs included.
  pub fn to_edge_lengths_sq(&self, topology: &Complex) -> MeshLengthsSq {
    // A 0-manifold is a discrete set of points: its 1-skeleton is empty, so the
    // edge-length representation of its (trivial, 0-dimensional) geometry is the
    // empty vector.
    if topology.dim() == 0 {
      return MeshLengthsSq::new_unchecked(Vector::zeros(0));
    }
    let edges = topology.edges();
    let mut edge_lengths_sq = Vector::zeros(edges.len());
    for (iedge, edge) in edges.handle_iter().enumerate() {
      let (vi, vj) = edge.endpoints();
      edge_lengths_sq[iedge] = self.ambient.norm_sq(&(vj.coord(self) - vi.coord(self)));
    }
    // SAFETY: Squared lengths come from a coordinate realization.
    MeshLengthsSq::new_unchecked(edge_lengths_sq)
  }
}

impl MeshCoords {
  /// Pad the ambient space with additional Euclidean axes: the vertices gain
  /// zero coordinates, the ambient inner product an identity block.
  pub fn embed_euclidean(mut self, dim: Dim) -> MeshCoords {
    let old_dim = self.matrix.nrows();
    let extra = dim - old_dim;
    self.matrix = self.matrix.insert_rows(old_dim, extra, 0.0);
    let mut ambient = self
      .ambient
      .matrix()
      .clone()
      .insert_rows(old_dim, extra, 0.0)
      .insert_columns(old_dim, extra, 0.0);
    for i in old_dim..dim {
      ambient[(i, i)] = 1.0;
    }
    self.ambient = Gramian::new_unchecked(ambient);
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
/// coord side -- the topology never learns of embeddings.
pub trait VertexRefExt {
  fn coord<'c>(self, coords: &'c MeshCoords) -> CoordRef<'c>;
}
impl VertexRefExt for Vertex<'_> {
  fn coord<'c>(self, coords: &'c MeshCoords) -> CoordRef<'c> {
    coords.coord(self.kidx())
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
  let coords = MeshCoords::with_ambient(Matrix::from_columns(&columns), coords.ambient().clone());
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

#[cfg(test)]
mod test {
  use super::*;
  use crate::{gen::cartesian::CartesianGrid, geometry::metric::mesh::EdgeRefExt};

  /// The witness reads cohere across the layers: an edge's Regge squared
  /// length is the squared distance of its endpoints' coordinates in the
  /// inducing embedding.
  #[test]
  fn edge_length_is_endpoint_distance() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let lengths_sq = coords.to_edge_lengths_sq(&topology);
      for edge in topology.edges().handle_iter() {
        let (vi, vj) = edge.endpoints();
        let displacement = vj.coord(&coords) - vi.coord(&coords);
        assert_eq!(edge.length_sq(&lengths_sq), displacement.norm_squared());
        assert_eq!(edge.length(&lengths_sq), displacement.norm());
      }
    }
  }

  /// A mesh embedded in Minkowski ambient space induces Lorentzian cell
  /// metrics: on a coordinate-aligned mesh the induced metric of every cell
  /// is congruent to $eta$ itself, so its signature is $(n - 1, 1)$ by
  /// Sylvester's law of inertia -- the same code path as the Euclidean
  /// ambient, one signature among all.
  #[test]
  fn minkowski_ambient_induces_lorentzian_cell_metrics() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
      let spacetime =
        MeshCoords::with_ambient(coords.matrix().clone(), gramian::Gramian::minkowski(dim));
      for cell in topology.cells().handle_iter() {
        let metric = spacetime.cell_metric(cell);
        assert_eq!(metric.signature(), (dim - 1, 1));
        assert!(!metric.is_riemannian());
      }
    }
  }

  /// A Minkowski embedding realizes Lorentzian Regge data: the signed
  /// squared edge lengths carry the causal character of every edge, and the
  /// per-cell metric reconstructed from them is the same Lorentzian metric
  /// the embedding induces -- Regge calculus doing exactly what it was
  /// invented for.
  #[test]
  fn lorentzian_ambient_realizes_lorentzian_regge_data() {
    use gramian::CausalType;
    let (topology, coords) = CartesianGrid::new_unit(2, 1).triangulate();
    let mut matrix = coords.matrix().clone();
    matrix.row_mut(0).scale_mut(0.7);
    let spacetime = MeshCoords::with_ambient(matrix, gramian::Gramian::minkowski(2));
    let regge = spacetime.to_edge_lengths_sq(&topology);

    let mut seen = std::collections::HashSet::new();
    for edge in topology.edges().handle_iter() {
      seen.insert(edge.causal_type(&regge) as u8);
      match edge.causal_type(&regge) {
        CausalType::Timelike => assert!(edge.length_sq(&regge) < 0.0),
        CausalType::Null => assert_eq!(edge.length_sq(&regge), 0.0),
        CausalType::Spacelike => assert!(edge.length_sq(&regge) > 0.0),
      }
    }
    // The time-scaled mesh has both timelike and spacelike edges.
    assert!(seen.len() >= 2);

    for cell in topology.cells().handle_iter() {
      let from_regge = regge.cell_metric(cell);
      let from_coords = spacetime.cell_metric(cell);
      approx::assert_relative_eq!(
        from_regge.vector_gramian().matrix(),
        from_coords.vector_gramian().matrix(),
        epsilon = 1e-12
      );
      assert_eq!(from_regge.signature(), (1, 1));
    }
  }
}
