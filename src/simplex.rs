use crate::{
  util::{factorial, gram_det_sqrt},
  Dim,
};

#[derive(Debug, Clone)]
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

  pub fn span_matrix(&self) -> na::DMatrix<f64> {
    let n = self.nvertices() - 1;
    let p = self.vertices.column(n);

    let mut m = na::DMatrix::zeros(self.dim_ambient(), n);
    for (c, v) in self.vertices().column_iter().take(n).enumerate() {
      m.column_mut(c).copy_from(&(v - p));
    }
    m
  }

  /// an auxiliar matrix that aids the computation of some quantities
  pub fn auxiliary_matrix(&self) -> na::DMatrix<f64> {
    // TODO: think about this assert
    assert!(
      self.dim_ambient() == self.dim_intrinsic(),
      "auxiliary matrix only work when n-simplex is in R^n"
    );
    let n = self.vertices.nrows();
    self.vertices.clone().insert_row(n, 1.0)
  }

  /// The determinate (signed volume) of the simplex.
  pub fn det(&self) -> f64 {
    let mat = self.span_matrix();
    let det = if self.dim_ambient() == self.dim_intrinsic() {
      mat.determinant()
    } else {
      gram_det_sqrt(&mat)
    };
    (factorial(self.dim_intrinsic()) as f64).recip() * det
  }

  /// The (unsigned) volume of the simplex.
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }

  pub fn bary_coord_deriv(&self) -> na::DMatrix<f64> {
    let n = self.nvertices();
    let mat = self.auxiliary_matrix();
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

#[cfg(test)]
mod test {
  use super::*;

  fn ref_vol() {
    for d in 0..=8 {
      let simp = Simplex::new_ref(d);
      assert_eq!(simp.det(), (factorial(d) as f64).recip());
    }
  }
}
