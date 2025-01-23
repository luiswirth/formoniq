pub mod manifold;

use crate::metric::EdgeLengths;

use topology::Dim;

pub type VertexIdx = usize;

pub type Coord<D = na::Dyn> = na::OVector<f64, D>;
pub type CoordRef<'a, D = na::Dyn> = na::VectorView<'a, f64, D>;

pub type CoordMatrix<D> = na::OMatrix<f64, D, na::Dyn>;
//type CoordVectorView<'a, D> = na::VectorView<'a, f64, D>;

pub type TangentVector = na::DVector<f64>;

#[derive(Debug, Clone)]
pub struct VertexCoords<D: na::Dim = na::Dyn>
where
  na::DefaultAllocator: na::allocator::Allocator<D, na::Dyn>,
{
  matrix: CoordMatrix<D>,
}

impl<D: na::Dim> VertexCoords<D>
where
  na::DefaultAllocator: na::allocator::Allocator<D, na::Dyn>,
{
  pub fn new(matrix: CoordMatrix<D>) -> Self {
    Self { matrix }
  }
  pub fn matrix(&self) -> &CoordMatrix<D> {
    &self.matrix
  }
  pub fn matrix_mut(&mut self) -> &mut CoordMatrix<D> {
    &mut self.matrix
  }
  pub fn into_matrix(self) -> CoordMatrix<D> {
    self.matrix
  }

  fn swap_coords(&mut self, icol: usize, jcol: usize) {
    self.matrix.swap_columns(icol, jcol)
  }
}

impl<D: na::Dim> From<CoordMatrix<D>> for VertexCoords<D>
where
  na::DefaultAllocator: na::allocator::Allocator<D, na::Dyn>,
{
  fn from(matrix: CoordMatrix<D>) -> Self {
    Self::new(matrix)
  }
}

impl<D: na::Dim> From<&[Coord<D>]> for VertexCoords<D>
where
  na::DefaultAllocator: na::allocator::Allocator<D>,
  na::DefaultAllocator: na::allocator::Allocator<D, na::Dyn>,
{
  fn from(vectors: &[Coord<D>]) -> Self {
    let matrix = CoordMatrix::from_columns(vectors);
    Self::new(matrix)
  }
}

impl VertexCoords<na::Dyn> {
  pub fn into_const_dim<D1: na::DimName>(self) -> VertexCoords<D1> {
    let matrix: CoordMatrix<D1> = na::try_convert(self.matrix).unwrap();
    VertexCoords::new(matrix)
  }
}
impl<D: na::DimName> VertexCoords<D> {
  pub fn into_dyn_dim(self) -> VertexCoords<na::Dyn> {
    let matrix: CoordMatrix<na::Dyn> = na::try_convert(self.matrix).unwrap();
    VertexCoords::new(matrix)
  }
}

impl<D: na::Dim> VertexCoords<D>
where
  na::DefaultAllocator: na::allocator::Allocator<D, na::Dyn>,
  na::DefaultAllocator: na::allocator::Allocator<D>,
{
  pub fn dim(&self) -> Dim {
    self.matrix.nrows()
  }
  pub fn nvertices(&self) -> usize {
    self.matrix.ncols()
  }

  // TODO: return view
  pub fn coord(&self, ivertex: VertexIdx) -> Coord<D> {
    self.matrix.column(ivertex).into_owned()
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
    self.matrix.column_iter()
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
    self.matrix.column_iter_mut()
  }

  pub fn to_edge_lengths(
    &self,
    edges: impl ExactSizeIterator<Item = [VertexIdx; 2]>,
  ) -> EdgeLengths {
    let mut edge_lengths = na::DVector::zeros(edges.len());
    for (iedge, edge) in edges.enumerate() {
      let [vi, vj] = edge;
      let length = (self.coord(vj) - self.coord(vi)).norm();
      edge_lengths[iedge] = length;
    }
    EdgeLengths::new(edge_lengths)
  }
}

impl VertexCoords<na::Dyn> {
  pub fn embed_euclidean(mut self, dim: Dim) -> VertexCoords {
    let old_dim = self.matrix.nrows();
    self.matrix = self.matrix.insert_rows(old_dim, dim - old_dim, 0.0);
    self
  }
}

pub fn write_coords<W: std::io::Write>(
  mut writer: W,
  coords: &VertexCoords,
) -> std::io::Result<()> {
  for coord in coords.coord_iter() {
    for &comp in coord {
      write!(writer, "{comp:.6} ")?;
    }
    writeln!(writer)?;
  }
  Ok(())
}
