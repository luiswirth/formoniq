pub mod local;
pub mod quadrature;

use itertools::Itertools;
use local::SimplexCoords;

use crate::{geometry::metric::MeshEdgeLengths, topology::complex::TopologyComplex, Dim};

pub type VertexIdx = usize;

pub type Coord<D = na::Dyn> = na::OVector<f64, D>;
pub type CoordRef<'a, D = na::Dyn> = na::VectorView<'a, f64, D>;

//type CoordVectorView<'a, D> = na::VectorView<'a, f64, D>;

pub type TangentVector = na::DVector<f64>;

pub fn standard_coord_complex(dim: Dim) -> (TopologyComplex, MeshVertexCoords) {
  let topology = TopologyComplex::standard(dim);

  let coords = topology
    .vertices()
    .handle_iter()
    .map(|v| v.kidx())
    .map(|v| {
      let mut vec = na::DVector::zeros(dim);
      if v > 0 {
        vec[v - 1] = 1.0;
      }
      vec
    })
    .collect_vec();
  let coords = na::DMatrix::from_columns(&coords);
  let coords = MeshVertexCoords::new(coords);

  (topology, coords)
}

pub type CoordMatrix<D> = na::OMatrix<f64, D, na::Dyn>;
#[derive(Debug, Clone)]
pub struct MeshVertexCoords<D: na::Dim = na::Dyn>
where
  na::DefaultAllocator: na::allocator::Allocator<D, na::Dyn>,
{
  coord_matrix: CoordMatrix<D>,
}

impl MeshVertexCoords<na::Dyn> {
  pub fn standard(ndim: Dim) -> Self {
    SimplexCoords::standard(ndim).vertices
  }
}

impl<D: na::Dim> MeshVertexCoords<D>
where
  na::DefaultAllocator: na::allocator::Allocator<D, na::Dyn>,
{
  pub fn new(coord_matrix: CoordMatrix<D>) -> Self {
    Self { coord_matrix }
  }

  pub fn matrix(&self) -> &CoordMatrix<D> {
    &self.coord_matrix
  }
  pub fn matrix_mut(&mut self) -> &mut CoordMatrix<D> {
    &mut self.coord_matrix
  }
  pub fn into_matrix(self) -> CoordMatrix<D> {
    self.coord_matrix
  }

  fn swap_coords(&mut self, icol: usize, jcol: usize) {
    self.coord_matrix.swap_columns(icol, jcol)
  }
}

impl<D: na::Dim> From<CoordMatrix<D>> for MeshVertexCoords<D>
where
  na::DefaultAllocator: na::allocator::Allocator<D, na::Dyn>,
{
  fn from(matrix: CoordMatrix<D>) -> Self {
    Self::new(matrix)
  }
}

impl<D: na::Dim> From<&[Coord<D>]> for MeshVertexCoords<D>
where
  na::DefaultAllocator: na::allocator::Allocator<D>,
  na::DefaultAllocator: na::allocator::Allocator<D, na::Dyn>,
{
  fn from(vectors: &[Coord<D>]) -> Self {
    let matrix = CoordMatrix::from_columns(vectors);
    Self::new(matrix)
  }
}

impl MeshVertexCoords<na::Dyn> {
  pub fn into_const_dim<D1: na::DimName>(self) -> MeshVertexCoords<D1> {
    let matrix: CoordMatrix<D1> = na::try_convert(self.coord_matrix).unwrap();
    MeshVertexCoords::new(matrix)
  }
}
impl<D: na::DimName> MeshVertexCoords<D> {
  pub fn into_dyn_dim(self) -> MeshVertexCoords<na::Dyn> {
    let matrix: CoordMatrix<na::Dyn> = na::try_convert(self.coord_matrix).unwrap();
    MeshVertexCoords::new(matrix)
  }
}

impl<D: na::Dim> MeshVertexCoords<D>
where
  na::DefaultAllocator: na::allocator::Allocator<D, na::Dyn>,
  na::DefaultAllocator: na::allocator::Allocator<D>,
{
  pub fn dim(&self) -> Dim {
    self.coord_matrix.nrows()
  }
  pub fn nvertices(&self) -> usize {
    self.coord_matrix.ncols()
  }

  // TODO: return view
  pub fn coord(&self, ivertex: VertexIdx) -> Coord<D> {
    self.coord_matrix.column(ivertex).into_owned()
  }

  pub fn coord_iter(
    &self,
  ) -> na::iter::ColumnIter<
    '_,
    f64,
    D,
    na::Dyn,
    <na::DefaultAllocator as na::allocator::Allocator<D, na::Dyn>>::Buffer<f64>,
  > {
    self.coord_matrix.column_iter()
  }

  pub fn coord_iter_mut(
    &mut self,
  ) -> na::iter::ColumnIterMut<
    '_,
    f64,
    D,
    na::Dyn,
    <na::DefaultAllocator as na::allocator::Allocator<D, na::Dyn>>::Buffer<f64>,
  > {
    self.coord_matrix.column_iter_mut()
  }

  pub fn to_edge_lengths(&self, topology: &TopologyComplex) -> MeshEdgeLengths {
    let edges = topology.edges();
    let mut edge_lengths = na::DVector::zeros(edges.len());
    for (iedge, edge) in edges.set_iter().enumerate() {
      let [vi, vj] = edge.vertices.clone().try_into().unwrap();
      let length = (self.coord(vj) - self.coord(vi)).norm();
      edge_lengths[iedge] = length;
    }
    MeshEdgeLengths::new(edge_lengths)
  }
}

impl MeshVertexCoords<na::Dyn> {
  pub fn embed_euclidean(mut self, dim: Dim) -> MeshVertexCoords {
    let old_dim = self.coord_matrix.nrows();
    self.coord_matrix = self.coord_matrix.insert_rows(old_dim, dim - old_dim, 0.0);
    self
  }
}

pub fn write_coords<W: std::io::Write>(
  mut writer: W,
  coords: &MeshVertexCoords,
) -> std::io::Result<()> {
  for coord in coords.coord_iter() {
    for &comp in coord {
      write!(writer, "{comp:.6} ")?;
    }
    writeln!(writer)?;
  }
  Ok(())
}
