use crate::Dim;

pub struct Simplex {
  /// rows are vertex coordinates
  vertices: na::DMatrix<f64>,
}
impl Simplex {
  pub fn new(vertices: na::DMatrix<f64>) -> Self {
    Self { vertices }
  }

  /// Constructs a reference simplex in `dim` dimensions.
  /// The simplex has unit vertices plus the origin.
  /// The ambient and intrinsic dimension are the same.
  pub fn new_ref(dim: Dim) -> Self {
    let mut vertices = na::DMatrix::zeros(dim, dim + 1);
    for d in 0..dim {
      vertices[(d, d)] = 1.0;
    }
    // last col is already all zeros (origin)
    Self { vertices }
  }

  /// The vertices of the simplex, in the columns of the returned matrix.
  pub fn vertices(&self) -> &na::DMatrix<f64> {
    &self.vertices
  }

  /// Number of vertices
  pub fn nvertices(&self) -> usize {
    self.vertices.ncols()
  }

  /// Intrinsic dimension of simplex
  pub fn dim_intrinsic(&self) -> Dim {
    self.vertices.ncols() - 1
  }

  pub fn dim_ambient(&self) -> Dim {
    self.vertices.nrows()
  }

  /// an auxiliar matrix that aids the computation of some quantities
  pub fn auxiliar_matrix(&self) -> na::DMatrix<f64> {
    let n = self.nvertices();
    let mut mat = na::DMatrix::zeros(n, n);
    mat
      .column_mut(0)
      .copy_from(&na::DVector::from_element(n, 1.0));
    mat
      .view_range_mut(0..n, 1..n)
      .copy_from(&self.vertices.transpose());
    mat
  }

  /// The determinate (signed volume) of the simplex.
  pub fn det(&self) -> f64 {
    0.5 * self.auxiliar_matrix().determinant()
  }

  /// The (unsigned) volume of the simplex.
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }

  pub fn bary_coord_deriv(&self) -> na::DMatrix<f64> {
    let n = self.nvertices();
    let mat = self.auxiliar_matrix();
    mat
      .try_inverse()
      .unwrap()
      .view_range(1..n, 0..n)
      .clone_owned()
  }

  /// The element matrix for the laplacian in linear lagrangian fe.
  pub fn lin_laplacian_0form_elmat(&self) -> na::DMatrix<f64> {
    let m = self.bary_coord_deriv();
    self.vol() * m.transpose() * m
  }
}
